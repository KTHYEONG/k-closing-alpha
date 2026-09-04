"""저녁 1회 실행: 당일 워치리스트 정규세션+NXT 애프터마켓 1분봉 아카이브."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from src import settings
from src.api.kis_client import KisApiClient
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


def run_intraday_archive(snapshot_date: str | None = None, bar_interval_minutes: int = DEFAULT_BAR_INTERVAL_MINUTES) -> tuple[int, int, int]:
    """당일 워치리스트 정규세션+NXT 애프터마켓 1분봉+정규세션 틱 체결을 세 파티션에 각각 저장. (정규행수, 애프터행수, 틱행수) 반환."""
    snap_date = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    codes = _today_watchlist_codes(snap_date)
    if not codes:
        return (0, 0, 0)

    async def _run() -> tuple[int, int, int]:
        client = KisApiClient()
        async with client.create_session() as session:
            await client.ensure_token(session)
            bars, nxt, ticks = await asyncio.gather(
                collect_intraday_bars(client, session, codes, snap_date, bar_interval_minutes),
                collect_nxt_aftermarket_bars(client, session, codes, snap_date, bar_interval_minutes),
                collect_intraday_trade_ticks(client, session, codes, snap_date),
            )
        n_bars = write_intraday_partition(bars, bar_interval_minutes, snap_date, INTRADAY_SESSION_REGULAR)
        n_nxt = write_intraday_partition(nxt, bar_interval_minutes, snap_date, INTRADAY_SESSION_NXT_AFTERMARKET)
        n_ticks = write_tick_partition(ticks, snap_date, INTRADAY_SESSION_REGULAR)
        return (n_bars, n_nxt, n_ticks)

    return asyncio.run(_run())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("Intraday archive store root: %s", settings.HISTORY_DIR)
    bars_rows, nxt_rows, tick_rows = run_intraday_archive()
    logger.info("[SUCCESS] intraday 아카이브 완료 (정규: %d행, NXT 애프터: %d행, 틱: %d행)", bars_rows, nxt_rows, tick_rows)


if __name__ == "__main__":
    main()
