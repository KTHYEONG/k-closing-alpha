"""장중 동시호가(15:20-15:30) 호가 스냅샷 폴러 (read-only CLI)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import logging
import time
from datetime import datetime
from typing import Any

from src.daily import archive
from src.data.orderbook_store import append_orderbook_snapshots, build_orderbook_rows

logger = logging.getLogger(__name__)


def _watchlist_codes(snapshot_date: str) -> list[str]:
    try:
        df = archive.fetch_archive_snapshot(snapshot_date=snapshot_date)
    except Exception as e:
        logger.warning("[DATA] Auction watchlist fetch failed date=%s: %s", snapshot_date, e)
        return []
    if df is None or df.empty or "종목코드" not in df.columns:
        return []
    return df["종목코드"].astype(str).str.zfill(6).dropna().unique().tolist()


def _call_orderbook_snapshot(client: Any, session: Any, code: str) -> dict:
    res = client.get_orderbook_snapshot(session, code, market_div_code="J")
    if inspect.isawaitable(res):
        res = asyncio.run(res)
    return res if isinstance(res, dict) else {}


def run_auction_capture(
    snapshot_date: str | None = None,
    interval_seconds: int = 10,
    start_hm: str = "1520",
    end_hm: str = "1530",
    client: Any | None = None,
    now_fn: Any | None = None,
) -> int:
    """동시호가 윈도우 동안 전 종목 호가를 interval 간격으로 스윕한다."""
    snap = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    codes = _watchlist_codes(snap)
    if not codes:
        return 0
    clock = now_fn or datetime.now
    owned_client = client
    session: Any = None
    if owned_client is None:
        from src.api.kis_client import KisApiClient

        owned_client = KisApiClient()
        session = owned_client.create_session()
        asyncio.run(owned_client.ensure_token(session))
    total = 0
    while True:
        now = clock()
        hm = now.strftime("%H%M")
        if not (str(start_hm) <= hm < str(end_hm)):
            break
        capture_ts = clock()
        sweep: list[dict] = []
        for code in codes:
            try:
                res = _call_orderbook_snapshot(owned_client, session, code)
            except Exception as e:
                logger.warning("[DATA] Auction sweep failed code=%s: %s", code, e)
                continue
            sweep.extend(build_orderbook_rows(res, code, "J", "auction", capture_ts))
        if sweep:
            try:
                total += append_orderbook_snapshots(sweep, snap)
            except Exception as e:
                logger.warning("[DATA] Auction snapshot persist failed: %s", e)
        if interval_seconds and interval_seconds > 0:
            time.sleep(interval_seconds)
    if session is not None:
        with contextlib.suppress(Exception):
            asyncio.run(session.close())
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Closing-auction orderbook capture (read-only)")
    parser.add_argument("--date", default=None)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--start", default="1520")
    parser.add_argument("--end", default="1530")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    total = run_auction_capture(
        snapshot_date=args.date,
        interval_seconds=args.interval,
        start_hm=args.start,
        end_hm=args.end,
    )
    logger.info("[SUCCESS] auction capture done rows=%d", total)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
