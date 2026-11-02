"""condition_history 아카이브 대상 일별분봉(FHKST03010230) 1회성 소급 백필 CLI.

오래된 날짜부터 처리(롤오프 임박 우선)하며 날짜별 파티션 파일에 멱등 병합하므로
중단 후 재실행이 안전하다. 실행 시간이 길 수 있는 1회성 스크립트이다.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src import settings
from src.api.kis.client import KisApiClient, kis_data_client_kwargs
from src.backfill.intraday.collector import backfill_krx_aftermarket_bars, backfill_nxt_aftermarket_bars, backfill_regular_bars
from src.config.market_session import INTRADAY_SESSION_KRX_AFTERMARKET, INTRADAY_SESSION_NXT_AFTERMARKET, INTRADAY_SESSION_REGULAR
from src.daily import archive
from src.data.intraday_store import intraday_partition_path, write_intraday_partition

from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)


_LEGACY_CONDITION_HISTORY_PATH_NAME = "condition_history_cleaned.parquet"


def _load_condition_history_sources(trade_log_path: Path | None = None) -> list[pd.DataFrame]:
    """Load the backfill target sources: the condition archive, the legacy cleaned history and, when given, a trade log.

    - archive.parquet: condition candidates since archive.py went live (about 2026-08-04).
    - condition_history_cleaned.parquet: the legacy cleaned backfill for the earlier span.
    - trade_log_path: an explicit manual trade journal (매수날짜/종목코드), research-only and never read
      implicitly; dates beyond the KIS retention window are dropped by enumerate_backfill_targets' lookback.

    Raises:
        FileNotFoundError: trade_log_path is given but does not exist.
    """
    frames: list[pd.DataFrame] = []
    try:
        df = archive.fetch_archive_snapshot(all_rows=True)
        if df is not None and not df.empty and "스냅샷_날짜" in df.columns and "종목코드" in df.columns:
            frames.append(df[["스냅샷_날짜", "종목코드"]])
    except Exception as e:
        logger.warning("Archive snapshot fetch failed: %s", e)

    legacy_path = settings.DATA_DIR / "history" / _LEGACY_CONDITION_HISTORY_PATH_NAME
    if legacy_path.exists():
        try:
            legacy_df = pd.read_parquet(legacy_path)
            if not legacy_df.empty and "스냅샷_날짜" in legacy_df.columns and "종목코드" in legacy_df.columns:
                frames.append(legacy_df[["스냅샷_날짜", "종목코드"]])
        except Exception as e:
            logger.warning("Legacy condition history read failed %s: %s", legacy_path, e)

    if trade_log_path is not None:
        if not trade_log_path.exists():
            raise FileNotFoundError(f"trade log not found: {trade_log_path}")
        trade_df = pd.read_parquet(trade_log_path)
        if not trade_df.empty and "매수날짜" in trade_df.columns and "종목코드" in trade_df.columns:
            frames.append(trade_df[["매수날짜", "종목코드"]].rename(columns={"매수날짜": "스냅샷_날짜"}))

    return frames


def _load_price_history_trading_calendar() -> pd.DatetimeIndex | None:
    """price_history의 거래일 달력(정렬된 고유 date)을 컬럼-프루닝 로드로 반환한다."""
    try:
        dates = pd.read_parquet(settings.PRICE_HISTORY_PARQUET_PATH, columns=["date"])
    except Exception as exc:
        logger.warning("[DATA] stage=exit_day_targets status=no_calendar path=%s error=%s", settings.PRICE_HISTORY_PARQUET_PATH, exc)
        return None
    if dates is None or dates.empty or "date" not in dates.columns:
        return None
    cal = pd.DatetimeIndex(pd.to_datetime(dates["date"], errors="coerce").dropna().dt.normalize().unique())
    return cal.sort_values()


def enumerate_backfill_targets(
    as_of: str | None = None,
    lookback_days: int = 365,
    *,
    include_exit_day: bool = True,
    trading_calendar: pd.DatetimeIndex | None = None,
    trade_log_path: Path | None = None,
) -> list[tuple[str, str]]:
    """archive + 레거시 condition_history_cleaned.parquet의 (스냅샷_날짜, 종목코드) distinct 쌍을
    lookback 이내로 필터링해 날짜 오름차순으로 반환한다.

    include_exit_day가 True이면 각 (스냅샷_날짜, 종목코드)마다 price_history 거래일 달력의
    다음 거래일 (next_trading_day, code) 쌍을 추가로 방출해 exit-side 경로를 확보한다.
    """
    as_of_date = as_of or datetime.now().strftime("%Y-%m-%d")
    cutoff = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")
    frames = _load_condition_history_sources(trade_log_path=trade_log_path)
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    sub = df[["스냅샷_날짜", "종목코드"]].copy()
    sub["스냅샷_날짜"] = sub["스냅샷_날짜"].astype(str)
    sub["종목코드"] = sub["종목코드"].astype(str).str.zfill(6)
    sub = sub.dropna()
    sub = sub[(sub["스냅샷_날짜"] >= cutoff) & (sub["스냅샷_날짜"] <= as_of_date)]
    sub = sub.drop_duplicates()
    pairs = [(str(d), str(c)) for d, c in sub.itertuples(index=False)]

    if include_exit_day:
        cal = trading_calendar if trading_calendar is not None else _load_price_history_trading_calendar()
        resolved = 0
        skipped = 0
        exit_pairs: list[tuple[str, str]] = []
        if cal is not None and len(cal):
            cal_parsed = pd.DatetimeIndex(pd.to_datetime(cal, errors="coerce"))
            if cal_parsed.tz is not None:
                cal_parsed = cal_parsed.tz_convert("Asia/Seoul").tz_localize(None)
            cal_norm = cal_parsed.normalize()
            for snap_date, code in pairs:
                snap_ts = pd.to_datetime(snap_date, errors="coerce")
                if pd.isna(snap_ts):
                    skipped += 1
                    continue
                # 다음 거래일은 달력에서만 해결한다 -- snapshot_date + 1일이나
                # 다음 스냅샷 날짜를 추측값으로 쓰지 않는다.
                later = cal_norm[cal_norm > snap_ts.normalize()]
                if len(later) == 0:
                    skipped += 1
                    continue
                exit_date = later.min().strftime("%Y-%m-%d")
                # No cutoff re-check: exit_date > snapshot_date >= cutoff holds
                # by construction (entries are pre-filtered to [cutoff, as_of]).
                exit_pairs.append((exit_date, code))
                resolved += 1
        else:
            skipped = len(pairs)
        logger.info("[DATA] stage=exit_day_targets resolved=%d skipped=%d", resolved, skipped)
        pairs = sorted(set(pairs) | set(exit_pairs))
    else:
        pairs = sorted(set(pairs))
    return pairs


def _already_collected_codes(bar_interval_minutes: int, snapshot_date: str, session: str) -> set[str]:
    """해당 날짜/세션 파티션 파일에 이미 저장된 종목코드 집합을 반환한다. 없으면 빈 집합."""
    target = intraday_partition_path(bar_interval_minutes, snapshot_date, session)
    if not target.exists():
        return set()
    for col in ("symbol", "종목코드"):
        try:
            existing = pd.read_parquet(target, columns=[col])
        except Exception as e:
            logger.warning("Failed to read existing partition for skip-check %s: %s", target, e)
            return set()
        if not existing.empty and col in existing.columns:
            return set(existing[col].astype(str).str.zfill(6))
    return set()


def _merge_and_write_partition(df: pd.DataFrame, bar_interval_minutes: int, snapshot_date: str, session: str) -> int:
    """신규 df를 날짜 파티션에 키 기준 병합 저장한다. 빈 df는 0 반환 no-op."""
    if df is None or df.empty:
        return 0
    return write_intraday_partition(df, bar_interval_minutes, snapshot_date, session)


def run_minute_history_backfill(
    lookback_days: int = 365, bar_interval_minutes: int = 1, *, trade_log_path: Path | None = None
) -> dict[str, int]:
    """1회성 오케스트레이터: 날짜 오름차순으로 정규+NXT 백필 후 날짜별 파티션에 멱등 병합."""
    targets = enumerate_backfill_targets(lookback_days=lookback_days, trade_log_path=trade_log_path)
    if not targets:
        return {"dates": 0, "regular_rows": 0, "nxt_rows": 0}
    by_date: dict[str, list[str]] = {}
    for snap_date, code in targets:
        by_date.setdefault(snap_date, []).append(code)
    ordered_dates = sorted(by_date.keys())

    async def _run() -> dict[str, int]:
        client = KisApiClient(**kis_data_client_kwargs())
        dates = 0
        regular_rows = 0
        nxt_rows = 0
        krx_after_rows = 0
        async with client.create_session() as session:
            await client.ensure_token(session)
            for snap_date in ordered_dates:
                all_codes = sorted(set(by_date[snap_date]))
                # 재실행 시 이미 파티션에 저장된 종목은 다시 조회하지 않는다 (API 콜 낭비 방지, 멱등 재실행 가속).
                done_regular = _already_collected_codes(bar_interval_minutes, snap_date, INTRADAY_SESSION_REGULAR)
                done_nxt = _already_collected_codes(bar_interval_minutes, snap_date, INTRADAY_SESSION_NXT_AFTERMARKET)
                done_krx_after = _already_collected_codes(bar_interval_minutes, snap_date, INTRADAY_SESSION_KRX_AFTERMARKET)
                regular_codes = [c for c in all_codes if c not in done_regular]
                nxt_codes = [c for c in all_codes if c not in done_nxt]
                krx_after_codes = [c for c in all_codes if c not in done_krx_after]
                if not regular_codes and not nxt_codes and not krx_after_codes:
                    dates += 1
                    continue
                try:
                    regular_df, nxt_df, krx_after_df = await asyncio.gather(
                        backfill_regular_bars(client, session, regular_codes, snap_date, bar_interval_minutes),
                        backfill_nxt_aftermarket_bars(client, session, nxt_codes, snap_date, bar_interval_minutes),
                        backfill_krx_aftermarket_bars(client, session, krx_after_codes, snap_date, bar_interval_minutes),
                    )
                except Exception as e:
                    logger.warning("Backfill failed date=%s: %s", snap_date, e)
                    continue
                try:
                    regular_rows += _merge_and_write_partition(regular_df, bar_interval_minutes, snap_date, INTRADAY_SESSION_REGULAR)
                except Exception as e:
                    logger.warning("Regular partition write failed date=%s: %s", snap_date, e)
                try:
                    nxt_rows += _merge_and_write_partition(nxt_df, bar_interval_minutes, snap_date, INTRADAY_SESSION_NXT_AFTERMARKET)
                except Exception as e:
                    logger.warning("NXT partition write failed date=%s: %s", snap_date, e)
                krx_after_rows += _merge_and_write_partition(krx_after_df, bar_interval_minutes, snap_date, INTRADAY_SESSION_KRX_AFTERMARKET)
                dates += 1
        return {"dates": dates, "regular_rows": int(regular_rows), "nxt_rows": int(nxt_rows), "krx_after_rows": int(krx_after_rows)}

    return asyncio.run(_run())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="One-off minute-history backfill over archive/legacy condition targets")
    parser.add_argument(
        "--trade-log",
        type=Path,
        default=None,
        metavar="PATH",
        help="also backfill symbols bought in this trade-log parquet; research copy: "
        "gdrive:quant-lake/research/k-closing-alpha/manual_journal_v1/trade_log.parquet",
    )
    args = parser.parse_args(argv)
    configure_cli_logging()
    logger.info("1회성 분봉 히스토리 백필 시작 (다수 날짜 x 종목 조합으로 오래 걸릴 수 있음, store=%s)", settings.HISTORY_DIR)
    result = run_minute_history_backfill(trade_log_path=args.trade_log)
    logger.info(
        "[SUCCESS] 분봉 히스토리 백필 완료 (날짜수: %d, 정규: %d행, NXT 애프터: %d행)",
        result["dates"],
        result["regular_rows"],
        result["nxt_rows"],
    )


if __name__ == "__main__":
    main()
