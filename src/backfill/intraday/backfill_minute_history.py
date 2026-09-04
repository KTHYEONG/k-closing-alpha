"""condition_history 아카이브 대상 일별분봉(FHKST03010230) 1회성 소급 백필 CLI.

오래된 날짜부터 처리(롤오프 임박 우선)하며 날짜별 파티션 파일에 멱등 병합하므로
중단 후 재실행이 안전하다. 실행 시간이 길 수 있는 1회성 스크립트이다.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd

from src import settings
from src.api.kis_client import KisApiClient
from src.backfill.intraday.collector import backfill_nxt_aftermarket_bars, backfill_regular_bars
from src.config.market_session import INTRADAY_SESSION_NXT_AFTERMARKET, INTRADAY_SESSION_REGULAR
from src.daily import archive
from src.data.intraday_store import intraday_partition_path, write_intraday_partition

logger = logging.getLogger(__name__)

_TIME_FIELD_CANDIDATES = ("stck_cntg_hour", "cntg_hour", "stck_cntg_hour_tm", "bsop_hour", "hour")


def _resolve_time_field(columns) -> str | None:
    for key in _TIME_FIELD_CANDIDATES:
        if key in columns:
            return key
    return None


def enumerate_backfill_targets(as_of: str | None = None, lookback_days: int = 365) -> list[tuple[str, str]]:
    """archive의 (스냅샷_날짜, 종목코드) distinct 쌍을 lookback 이내로 필터링해 날짜 오름차순으로 반환."""
    as_of_date = as_of or datetime.now().strftime("%Y-%m-%d")
    cutoff = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")
    try:
        df = archive.fetch_archive_snapshot(all_rows=True)
    except Exception as e:
        logger.warning("Archive snapshot fetch failed: %s", e)
        return []
    if df is None or df.empty or "스냅샷_날짜" not in df.columns or "종목코드" not in df.columns:
        return []
    sub = df[["스냅샷_날짜", "종목코드"]].copy()
    sub["스냅샷_날짜"] = sub["스냅샷_날짜"].astype(str)
    sub["종목코드"] = sub["종목코드"].astype(str).str.zfill(6)
    sub = sub.dropna()
    sub = sub[(sub["스냅샷_날짜"] >= cutoff) & (sub["스냅샷_날짜"] <= as_of_date)]
    sub = sub.drop_duplicates()
    sub = sub.sort_values(["스냅샷_날짜", "종목코드"], kind="stable")
    return [(str(d), str(c)) for d, c in sub.itertuples(index=False)]


def _merge_and_write_partition(df: pd.DataFrame, bar_interval_minutes: int, snapshot_date: str, session: str) -> int:
    """기존 파티션 파일과 신규 df를 (종목코드, 시각 필드) 기준 병합 저장. 빈 df는 0 반환 no-op."""
    if df is None or df.empty:
        return 0
    target = intraday_partition_path(bar_interval_minutes, snapshot_date, session)
    if target.exists():
        try:
            existing = pd.read_parquet(target)
        except Exception as e:
            logger.warning("Failed to read existing partition %s: %s", target, e)
            existing = pd.DataFrame()
        if existing is not None and not existing.empty:
            merged = pd.concat([existing, df], ignore_index=True)
            time_field = _resolve_time_field(merged.columns) or _resolve_time_field(df.columns)
            if "종목코드" in merged.columns and time_field and time_field in merged.columns:
                merged = merged.drop_duplicates(subset=["종목코드", time_field], keep="last")
            else:
                merged = merged.drop_duplicates(keep="last")
            return write_intraday_partition(merged, bar_interval_minutes, snapshot_date, session)
    return write_intraday_partition(df, bar_interval_minutes, snapshot_date, session)


def run_minute_history_backfill(lookback_days: int = 365, bar_interval_minutes: int = 1) -> dict[str, int]:
    """1회성 오케스트레이터: 날짜 오름차순으로 정규+NXT 백필 후 날짜별 파티션에 멱등 병합."""
    targets = enumerate_backfill_targets(lookback_days=lookback_days)
    if not targets:
        return {"dates": 0, "regular_rows": 0, "nxt_rows": 0}
    by_date: dict[str, list[str]] = {}
    for snap_date, code in targets:
        by_date.setdefault(snap_date, []).append(code)
    ordered_dates = sorted(by_date.keys())

    async def _run() -> dict[str, int]:
        client = KisApiClient()
        dates = 0
        regular_rows = 0
        nxt_rows = 0
        async with client.create_session() as session:
            await client.ensure_token(session)
            for snap_date in ordered_dates:
                codes = sorted(set(by_date[snap_date]))
                try:
                    regular_df, nxt_df = await asyncio.gather(
                        backfill_regular_bars(client, session, codes, snap_date, bar_interval_minutes),
                        backfill_nxt_aftermarket_bars(client, session, codes, snap_date, bar_interval_minutes),
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
                dates += 1
        return {"dates": dates, "regular_rows": int(regular_rows), "nxt_rows": int(nxt_rows)}

    return asyncio.run(_run())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("1회성 분봉 히스토리 백필 시작 (다수 날짜 x 종목 조합으로 오래 걸릴 수 있음, store=%s)", settings.HISTORY_DIR)
    result = run_minute_history_backfill()
    logger.info(
        "[SUCCESS] 분봉 히스토리 백필 완료 (날짜수: %d, 정규: %d행, NXT 애프터: %d행)",
        result["dates"],
        result["regular_rows"],
        result["nxt_rows"],
    )


if __name__ == "__main__":
    main()
