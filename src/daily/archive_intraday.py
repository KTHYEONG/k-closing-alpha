"""저녁 1회 실행: 당일 워치리스트 정규세션+NXT 애프터마켓 1분봉 아카이브."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src import settings
from src.api.kis_client import KisApiClient
from src.api.ls.client import LsApiClient
from src.backfill.intraday.collector import (
    collect_intraday_bars,
    collect_intraday_trade_ticks,
    collect_nxt_aftermarket_bars,
)
from src.config.market_session import (
    DEFAULT_BAR_INTERVAL_MINUTES,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_REGULAR,
)
from src.daily import archive
from src.data.intraday_store import write_intraday_partition, write_tick_partition

logger = logging.getLogger(__name__)


def _today_watchlist_codes(snapshot_date: str) -> list[str]:
    try:
        df = archive.fetch_archive_snapshot(snapshot_date=snapshot_date)
    except Exception as e:
        logger.warning("Watchlist fetch failed date=%s: %s", snapshot_date, e)
        return []
    if df is None or df.empty or "종목코드" not in df.columns:
        return []
    return df["종목코드"].astype(str).str.zfill(6).dropna().unique().tolist()


def resolve_previous_archive_date(snapshot_date: str) -> str | None:
    """아카이브 날짜 인덱스에서 snapshot_date 직전 영업일을 찾는다."""
    try:
        df = archive.fetch_archive_snapshot(all_rows=True)
    except Exception as e:
        logger.warning("[DATA] Previous archive date lookup failed date=%s: %s", snapshot_date, e)
        return None
    if df is None or df.empty or "스냅샷_날짜" not in df.columns:
        return None
    dates = sorted({str(d) for d in df["스냅샷_날짜"].astype(str).tolist() if str(d) < str(snapshot_date)})
    return dates[-1] if dates else None


def _archive_target_codes(snapshot_date: str) -> list[str]:
    """당일 + 직전 아카이브 영업일 워치리스트의 중복 제거 합집합.

    _today_watchlist_codes/resolve_previous_archive_date는 자체적으로 조회 실패를
    흡수해 빈 결과를 반환하므로(never raise), 여기서는 별도 예외 처리가 필요 없다.
    """
    today = _today_watchlist_codes(snapshot_date)
    prev_date = resolve_previous_archive_date(snapshot_date)
    codes: list[str] = list(today)
    if prev_date is not None:
        for code in _today_watchlist_codes(prev_date):
            if code not in codes:
                codes.append(code)
    return codes


def run_intraday_archive(snapshot_date: str | None = None, bar_interval_minutes: int = DEFAULT_BAR_INTERVAL_MINUTES) -> tuple[int, int, int]:
    """당일 워치리스트 정규세션+NXT 애프터마켓 1분봉+정규세션 틱 체결을 세 파티션에 각각 저장. (정규행수, 애프터행수, 틱행수) 반환."""
    snap_date = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    codes = _archive_target_codes(snap_date)
    if not codes:
        return (0, 0, 0)

    async def _run() -> tuple[int, int, int]:
        client = KisApiClient()
        ls_client = LsApiClient() if getattr(settings, 'LS_APP_KEY', None) else None
        async with client.create_session() as session:
            await client.ensure_token(session)
            bars = await collect_intraday_bars(client, session, codes, snap_date, bar_interval_minutes, ls_client=ls_client)
            nxt = await collect_nxt_aftermarket_bars(client, session, codes, snap_date, bar_interval_minutes)
            n_bars = write_intraday_partition(bars, bar_interval_minutes, snap_date, INTRADAY_SESSION_REGULAR)
            n_nxt = write_intraday_partition(nxt, bar_interval_minutes, snap_date, INTRADAY_SESSION_NXT_AFTERMARKET)
            ticks = await collect_intraday_trade_ticks(client, session, codes, snap_date, ls_client=ls_client)
            n_ticks = write_tick_partition(ticks, snap_date, INTRADAY_SESSION_REGULAR)
            return (n_bars, n_nxt, n_ticks)

    return asyncio.run(_run())


def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    target_date = sys.argv[1] if len(sys.argv) > 1 else None
    logger.info("Intraday archive store root: %s (target_date=%s)", settings.HISTORY_DIR, target_date)
    bars_rows, nxt_rows, tick_rows = run_intraday_archive(snapshot_date=target_date)
    logger.info("[SUCCESS] intraday 아카이브 완료 (정규: %d행, NXT 애프터: %d행, 틱: %d행)", bars_rows, nxt_rows, tick_rows)


if __name__ == "__main__":
    main()
