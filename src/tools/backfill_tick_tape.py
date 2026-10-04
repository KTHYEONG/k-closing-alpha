"""Manual CLI over tape_recovery."""

from __future__ import annotations

import argparse
import asyncio
import logging
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from src import settings
from src.backfill.intraday import tape_recovery as tr
from src.config.collection import CollectionSettings
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)

_MAX_BACKFILL_SPAN_DAYS = 31
_EST_PAGES_PER_DAY = 30
_EST_BYTES_PER_PAGE = 50_000
# Live Kiwoom units share this key's 5 req/s limit; a slowed 15:20 collect also delays auction-close, which needs its cohort.
# Windows cover collect/predict (15:20), regular archive (15:40-16:30), aftermarket archive (20:05-21:05), price-ingest, extended backfill.
_DEFAULT_BLACKOUTS = ("08:25-08:45", "11:25-11:45", "15:15-17:00", "20:00-21:10", "21:25-21:45", "23:00-23:20")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover missing tick days from the Kiwoom tapes.")
    parser.add_argument("--start", required=True, help="Inclusive backfill start (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Inclusive backfill end (YYYY-MM-DD).")
    parser.add_argument("--venue", default="all", help="Tape venue in {krx,nxt,all}.")
    parser.add_argument("--apply", action="store_true", help="Publish certified days through partition writers.")
    parser.add_argument("--deadline", default=None, help="Aware ISO timestamp after which no new walk starts.")
    parser.add_argument("--symbols-limit", type=int, default=None, help="Maximum distinct symbols to walk.")
    parser.add_argument("--blackout", action="append", default=[], help="Repeatable blackout window HH:MM-HH:MM (KST).")
    parser.add_argument("--ledger", default=None, help="Ledger JSONL path (default under the capture root).")
    parser.add_argument("--force", action="store_true", help="Re-recover certified days.")
    return parser.parse_args(argv)


def _parse_day(value: str, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Invalid {field} date: {value!r}") from None


def _parse_blackout(spec: str) -> tuple[int, int]:
    try:
        start_raw, end_raw = str(spec).split("-", 1)
        start_h, start_m = start_raw.split(":", 1)
        end_h, end_m = end_raw.split(":", 1)
        start = int(start_h) * 60 + int(start_m)
        end = int(end_h) * 60 + int(end_m)
    except ValueError:
        raise ValueError(f"Invalid --blackout window: {spec!r} (expected HH:MM-HH:MM)") from None
    if not (0 <= start < 24 * 60 and 0 <= end <= 24 * 60):
        raise ValueError(f"Invalid --blackout window: {spec!r} (expected HH:MM-HH:MM)")
    return start, end


def _now() -> datetime:
    from src.data.capture_contracts import SEOUL

    return datetime.now(SEOUL)


def _open_kiwoom() -> tuple[Any, Any]:
    import aiohttp

    from src.api.kiwoom.client import KiwoomApiClient

    if not settings.KIWOOM_APP_KEY:
        raise RuntimeError("Kiwoom credentials are not configured")
    return KiwoomApiClient(), aiohttp.ClientSession()


def _project_evidence(tasks: list[tr.WalkTask], profile: CollectionSettings) -> tuple[int, int]:
    guard = int(profile.COLLECTION_TICK_REPAIR_MAX_PAGES)
    pages = sum(min(guard, _EST_PAGES_PER_DAY * len(task.days)) for task in tasks)
    return pages, pages * _EST_BYTES_PER_PAGE


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def main(argv: list[str] | None = None) -> None:
    """Recover missing/incomplete tick days from the Kiwoom tapes.

    Arguments: --start/--end (inclusive YYYY-MM-DD closed days), --venue {krx,nxt,all}, --apply, --deadline (aware ISO),
    --symbols-limit, --blackout HH:MM-HH:MM (repeatable), --ledger PATH, --force (re-recover certified days).

    Raises:
        ValueError: Open-ended or oversized selection (span > 31 days), unknown venue, naive deadline.
    """
    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    args = _parse_args(argv)
    start = _parse_day(args.start, "--start")
    end = _parse_day(args.end, "--end")
    if end < start:
        raise ValueError(f"Invalid backfill range: {args.start!r}..{args.end!r}")
    span = (end - start).days + 1
    if span > _MAX_BACKFILL_SPAN_DAYS:
        raise ValueError(f"Unbounded backfill selection: span={span} exceeds {_MAX_BACKFILL_SPAN_DAYS} days")
    venue_arg = str(args.venue).lower()
    venues: list[Literal["KRX", "NXT"]]
    if venue_arg == "all":
        venues = ["KRX", "NXT"]
    elif venue_arg == "krx":
        venues = ["KRX"]
    elif venue_arg == "nxt":
        venues = ["NXT"]
    else:
        raise ValueError(f"Invalid --venue: {args.venue!r} (expected one of krx, nxt, all)")
    deadline = tr.parse_walk_deadline(args.deadline)
    blackouts = [_parse_blackout(spec) for spec in (args.blackout or [])] or [
        _parse_blackout(spec) for spec in _DEFAULT_BLACKOUTS
    ]
    if args.symbols_limit is not None and int(args.symbols_limit) <= 0:
        raise ValueError(f"Invalid --symbols-limit: {args.symbols_limit!r}")
    profile = CollectionSettings()
    store = CaptureStore(_capture_root(profile))
    ledger = tr.tape_ledger_path(args.ledger, profile)
    settled = {} if args.force else tr.read_settled_ledger(ledger)
    days = [(start + timedelta(days=offset)).isoformat() for offset in range(span)]
    needs = tr.collect_tape_needs(days, venues, store, settled, bool(args.force))
    tasks = tr.order_walk_tasks(needs)
    if args.symbols_limit is not None:
        kept: list[tr.WalkTask] = []
        seen_symbols: set[str] = set()
        for task in tasks:
            if task.symbol in seen_symbols or len(seen_symbols) < int(args.symbols_limit):
                kept.append(task)
                seen_symbols.add(task.symbol)
        tasks = kept
    projected_pages, projected_bytes = _project_evidence(tasks, profile)
    if not args.apply:
        logger.info(
            "[DATA] stage=tape_backfill_dry_run needs=%d tasks=%d expected_pages=%d expected_bytes=%d",
            len(needs),
            len(tasks),
            projected_pages,
            projected_bytes,
        )
    try:
        free = _free_bytes(_capture_root(profile))
    except OSError as exc:
        raise RuntimeError(f"Tape backfill storage check failed: {exc}") from exc
    reserve = int(profile.COLLECTION_TAPE_MIN_FREE_GIB) * 1024**3
    if projected_bytes > max(0, free - reserve):
        raise RuntimeError(
            f"Tape backfill storage budget exceeded: projected_bytes={projected_bytes} free={free} reserve={reserve}"
        )
    if not tasks:
        logger.info("[DATA] stage=tape_backfill status=NOOP needs=0")
        return

    async def _run() -> tr.TapeRunSummary:
        client, session_ctx = _open_kiwoom()
        async with session_ctx as http_session:
            return await tr.run_walk_tasks(
                tasks,
                client=client,
                http_session=http_session,
                store=store,
                profile=profile,
                apply=bool(args.apply),
                ledger=ledger,
                deadline=deadline,
                blackouts=blackouts,
                run_date=_now().date().isoformat(),
            )

    try:
        summary = asyncio.run(_run())
    except OSError as exc:
        raise RuntimeError(f"Tape backfill infrastructure failed: {exc}") from exc
    logger.info(
        "[DATA] stage=tape_backfill_report needs=%d pages=%d rows=%d unresolved=%d remaining=%s",
        len(needs),
        summary.pages,
        summary.rows,
        summary.unresolved,
        summary.remaining,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry, exercised via `python -m`
    main()
