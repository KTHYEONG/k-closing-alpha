"""Toss 기반 1회성 정규세션 1분봉 소급 백필 (2022-09~2025-09, KIS 보존기간 밖 청산일 공백).

KIS의 FHKST03010230는 ~1년 롤링 보존이라 이 구간은 이미 KIS에서 유실됐다
(scratch/probe_toss_1m_backfill_2022_2025.json 실측: 2022-09-01/2024-02-28/2025-09-01
샘플 전부 실거래 확인). Toss /api/v1/candles는 최소 이 구간 전체를 실측 커버하므로,
기존 enumerate_backfill_targets()/파티션 멱등병합 인프라를 그대로 재사용해 정규세션
(09:00-15:30)만 복구한다. NXT는 대상 기간 대부분 미존재라 범위에서 제외한다.
1회성 스크립트이며, 기존 롤링 KIS 백필 잡(backfill_minute_history.py)은 변경하지 않는다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
import pandas as pd

from src.backfill.intraday.backfill_minute_history import (
    _already_collected_codes,
    _merge_and_write_partition,
    enumerate_backfill_targets,
)
from src.config.market_session import INTRADAY_SESSION_REGULAR, KRX_REGULAR_HOUR_CEIL, KRX_REGULAR_HOUR_FLOOR
from src.data.intraday_schema import normalize_bar_frame

logger = logging.getLogger(__name__)


class TossCandleFetchError(RuntimeError):
    """Toss candles 페이지 응답이 에러 봉투({'error': {...}})인 경우.

    빈 결과({'result': {'candles': []}})와 벤더 오류를 구분하지 못하면 2페이지 중
    하나가 실패해도 나머지 절반만으로 세션이 조용히 절단된다(2023-07-05 14종목
    실측 재현 사례: 200/391봉, 오전 절반이 이 경로로 유실됐다).
    """


def _raise_if_toss_error(page: dict, code: str, snapshot_date: str, page_label: str) -> None:
    """page가 Toss 에러 봉투({'error': {...}})면 TossCandleFetchError를 raise한다."""
    if isinstance(page, dict) and "error" in page:
        err = page["error"]
        raise TossCandleFetchError(
            f"Toss candles {page_label} failed symbol={code} date={snapshot_date} "
            f"code={err.get('code')} msg={err.get('message', '')}"
        )

GAP_START_DATE: str = "2022-09-01"
GAP_END_DATE: str = "2025-09-01"


def _hms_colon(hms: str) -> str:
    """'153000' -> '15:30:00'."""
    return f"{hms[0:2]}:{hms[2:4]}:{hms[4:6]}"


async def _fetch_toss_regular_session_bars(toss: Any, session: Any, code: str, snapshot_date: str) -> pd.DataFrame:
    """한 종목의 과거 하루치 정규세션(09:00-15:30) 1분봉을 Toss로 2콜 페이징 수집+정규화한다.

    Toss의 before 커서는 inclusive라 두 페이지 경계에 중복 캔들이 1개 생긴다. 신규
    파티션(기존 파일이 아직 없는 첫 기록) 병합 시 merge_partition_frame은 new_df 내부의
    중복은 제거하지 않고 신규-vs-기존 비교만 dedupe하므로(실측 확인), 여기서 명시적으로
    (symbol, ts_hms) 기준 중복을 제거해야 한다 -- write 계층에 기대지 않는다.
    Toss 에러 봉투({'error': {...}})가 어느 페이지에 와도 TossCandleFetchError를
    raise하고 조용히 '캔들 없음'으로 취급하지 않는다.
    """
    before_close = f"{snapshot_date}T{_hms_colon(KRX_REGULAR_HOUR_CEIL)}.000+09:00"
    page1 = await toss.get_candles(session, code, interval="1m", count=200, before=before_close)
    _raise_if_toss_error(page1, code, snapshot_date, "page1")
    candles = list((page1.get("result") or {}).get("candles") or [])
    if candles:
        oldest_ts = candles[-1]["timestamp"]
        page2 = await toss.get_candles(session, code, interval="1m", count=200, before=oldest_ts)
        _raise_if_toss_error(page2, code, snapshot_date, "page2")
        candles += list((page2.get("result") or {}).get("candles") or [])
    df = normalize_bar_frame(pd.DataFrame(candles), "toss", snapshot_date, code)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["symbol", "ts_hms"], keep="last").reset_index(drop=True)
    floor, ceil_ = int(KRX_REGULAR_HOUR_FLOOR), int(KRX_REGULAR_HOUR_CEIL)
    return df[(df["ts_hms"] >= floor) & (df["ts_hms"] <= ceil_)].reset_index(drop=True)


def enumerate_toss_gap_targets(gap_start: str = GAP_START_DATE, gap_end: str = GAP_END_DATE) -> list[tuple[str, str]]:
    """enumerate_backfill_targets()의 결과를 [gap_start, gap_end] 구간으로만 좁힌다.

    이 구간은 KIS가 더 이상 보유하지 않는(롤링 보존기간 밖) 과거 구간이므로,
    같은 대상 산출 로직을 그대로 재사용하되 벤더만 Toss로 바꾼다.
    """
    pairs = enumerate_backfill_targets(as_of=gap_end, lookback_days=5000, include_exit_day=True)
    return sorted((d, c) for d, c in pairs if gap_start <= d <= gap_end)


def run_toss_1m_backfill(
    gap_start: str = GAP_START_DATE, gap_end: str = GAP_END_DATE, bar_interval_minutes: int = 1, toss: Any | None = None
) -> dict[str, int]:
    """1회성 오케스트레이터: Toss로 [gap_start, gap_end] 정규세션 1분봉을 복구해 기존 파티션에 멱등 병합."""
    targets = enumerate_toss_gap_targets(gap_start, gap_end)
    if not targets:
        return {"dates": 0, "rows": 0}
    by_date: dict[str, list[str]] = {}
    for snap_date, code in targets:
        by_date.setdefault(snap_date, []).append(code)
    ordered_dates = sorted(by_date.keys())

    async def _run() -> dict[str, int]:
        client = toss
        if client is None:
            from src.api.toss.client import TossApiClient

            client = TossApiClient()
        dates = 0
        rows = 0
        async with aiohttp.ClientSession() as session:
            for snap_date in ordered_dates:
                all_codes = sorted(set(by_date[snap_date]))
                done = _already_collected_codes(bar_interval_minutes, snap_date, INTRADAY_SESSION_REGULAR)
                codes = [c for c in all_codes if c not in done]
                if not codes:
                    dates += 1
                    continue
                results = await asyncio.gather(
                    *(_fetch_toss_regular_session_bars(client, session, c, snap_date) for c in codes),
                    return_exceptions=True,
                )
                ok_frames: list[pd.DataFrame] = []
                for code, result in zip(codes, results, strict=True):
                    if isinstance(result, BaseException):
                        logger.warning(
                            "[DATA] stage=toss_1m_backfill symbol=%s date=%s status=FAILED error=%s",
                            code, snap_date, result,
                        )
                        continue
                    if not result.empty:
                        ok_frames.append(result)
                if ok_frames:
                    combined = pd.concat(ok_frames, ignore_index=True)
                    rows += _merge_and_write_partition(combined, bar_interval_minutes, snap_date, INTRADAY_SESSION_REGULAR)
                dates += 1
        return {"dates": dates, "rows": int(rows)}

    return asyncio.run(_run())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(
        "[SYS] stage=toss_1m_backfill status=START gap=%s..%s (KIS 보존기간 밖 1회성 복구, 오래 걸릴 수 있음)",
        GAP_START_DATE, GAP_END_DATE,
    )
    result = run_toss_1m_backfill()
    logger.info(
        "[SYS] stage=toss_1m_backfill status=DONE dates=%d rows=%d",
        result["dates"], result["rows"],
    )


if __name__ == "__main__":
    main()
