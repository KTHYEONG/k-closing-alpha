"""Daily self-healing sweep of recent tick needs from the Kiwoom tapes."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from src.config import market_session as _session
from src.config.collection import CollectionSettings
from src.data.capture_contracts import SEOUL
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.tools import backfill_tick_tape as btt
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)

_TAPE_DEPTH_DAYS = 30
_EXPIRY_WARNING_DAYS = 3
_STAGING_DIRNAME = "tape_sweep"


@dataclass(frozen=True)
class TapeSweepReport:
    """Outcome of one daily tape sweep run."""

    days_checked: tuple[str, ...]
    needs: int
    recovered: tuple[str, ...]
    unresolved: tuple[str, ...]
    expired: tuple[str, ...]
    expiring: tuple[str, ...]
    remaining: tuple[str, ...]
    disk_guard: bool
    pages: int
    rows: int
    expiring_needs: int = 0


def _now() -> datetime:
    return datetime.now(SEOUL)


def _default_deadline() -> datetime | None:
    """No-new-walk cutoff ahead of the evening price-ingest slot; None once that time has passed."""
    now = _now()
    hh, mm = int(_session.TAPE_SWEEP_DEADLINE_HHMMSS[0:2]), int(_session.TAPE_SWEEP_DEADLINE_HHMMSS[2:4])
    candidate = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return candidate if candidate > now else None


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _open_kiwoom() -> tuple[Any, Any]:
    import aiohttp

    from src import settings
    from src.api.kiwoom.client import KiwoomApiClient

    if not settings.KIWOOM_APP_KEY:
        raise RuntimeError("Kiwoom credentials are not configured")
    return KiwoomApiClient(), aiohttp.ClientSession()


def _report_path(root: Path) -> Path:
    return root / "staging" / _STAGING_DIRNAME / "last_report.json"


def _window_days(lookback_days: int, today: date) -> list[str]:
    return [(today - timedelta(days=offset)).isoformat() for offset in range(lookback_days - 1, -1, -1)]


def _scan_days(today: date) -> list[str]:
    return [(today - timedelta(days=offset)).isoformat() for offset in range(_TAPE_DEPTH_DAYS - 1, -1, -1)]


def _need_key(symbol: str, day: str, session: str) -> str:
    return f"{symbol}/{day}/{session}"


def _check_deadline(deadline: datetime | None) -> None:
    if deadline is not None and (deadline.tzinfo is None or deadline.utcoffset() is None):
        raise ValueError(f"deadline must be timezone-aware: {deadline!r}")


def _write_report(root: Path, run_date: str, report: TapeSweepReport) -> None:
    payload = {
        "run_date": run_date,
        "days_checked": list(report.days_checked),
        "needs": report.needs,
        "recovered": list(report.recovered),
        "unresolved": list(report.unresolved),
        "expired": list(report.expired),
        "expiring": list(report.expiring),
        "remaining": list(report.remaining),
        "disk_guard": report.disk_guard,
        "pages": report.pages,
        "rows": report.rows,
        "expiring_needs": report.expiring_needs,
    }
    try:
        target = _report_path(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=True), encoding="utf-8")
    except OSError as exc:
        logger.warning("[DATA] stage=tape_sweep status=DEGRADED reason=report_write_failed error=%s", type(exc).__name__)


async def run_tick_tape_sweep(
    *,
    lookback_days: int,
    profile: CollectionSettings | None = None,
    deadline: datetime | None = None,
) -> TapeSweepReport:
    """Recover tick needs of the last `lookback_days` closed sessions from the Kiwoom tapes.

    Needs are computed with the same selection as backfill_tick_tape (volume-gap, missing, truncated, out-of-window);
    recovery uses the same walk/publisher modules; the report lists recovered, still-unresolved and expired
    (older than tape depth) symbol-days.

    Args:
        lookback_days: Re-check window in days; must not exceed COLLECTION_TAPE_LOOKBACK_DAYS.
        profile: Acquisition limits; defaults to CollectionSettings().
        deadline: Aware timestamp after which no new walk starts.

    Returns:
        TapeSweepReport with recovered/unresolved/expired symbol-days.

    Raises:
        ValueError: Invalid lookback window or naive deadline.
        RuntimeError: Infrastructure failure (storage check, evidence, publication).
    """
    resolved = profile if profile is not None else CollectionSettings()
    bound = int(resolved.COLLECTION_TAPE_LOOKBACK_DAYS)
    if int(lookback_days) <= 0 or int(lookback_days) > bound:
        raise ValueError(f"Invalid lookback_days: {lookback_days!r} (expected 1..{bound})")
    _check_deadline(deadline)
    today = _now().date()
    window = _window_days(int(lookback_days), today)
    scan = _scan_days(today)
    root = _capture_root(resolved)
    store = CaptureStore(root)
    ledger = btt._ledger_path(None, resolved)
    venues: list[Literal["KRX", "NXT"]] = ["KRX", "NXT"]
    found = btt._collect_needs(scan, venues, store, btt._read_settled(ledger), False)
    window_set = set(window)
    active = [item for item in found if item.day in window_set]
    expired = sorted({_need_key(item.symbol, item.day, item.session) for item in found if item.day not in window_set})
    active_keys = sorted({_need_key(item.symbol, item.day, item.session) for item in active})
    near_expiry = [item for item in active if (today - date.fromisoformat(item.day)).days >= _TAPE_DEPTH_DAYS - _EXPIRY_WARNING_DAYS]
    expiring = sorted({item.day for item in near_expiry})
    expiring_needs = len({(item.symbol, item.day) for item in near_expiry})
    if expiring:
        logger.warning("[DATA] stage=tape_sweep status=WARNING reason=expiring days=%s", expiring)
    try:
        free = _free_bytes(root)
    except OSError as exc:
        raise RuntimeError(f"Tape sweep storage check failed: {exc}") from exc
    if free < int(resolved.COLLECTION_TAPE_MIN_FREE_GIB) * 1024**3:
        logger.warning("[DATA] stage=tape_sweep status=DISK_GUARD free=%d", free)
        report = TapeSweepReport(
            days_checked=tuple(window), needs=len(active_keys), recovered=(), unresolved=tuple(active_keys),
            expired=tuple(expired), expiring=tuple(expiring), remaining=tuple(active_keys),
            disk_guard=True, pages=0, rows=0, expiring_needs=expiring_needs,
        )
        _write_report(root, today.isoformat(), report)
        return report
    tasks = btt._order_tasks(active)
    if not tasks:
        logger.info("[DATA] stage=tape_sweep status=NOOP needs=0")
        report = TapeSweepReport(
            days_checked=tuple(window), needs=0, recovered=(), unresolved=(), expired=tuple(expired),
            expiring=tuple(expiring), remaining=(), disk_guard=False, pages=0, rows=0, expiring_needs=expiring_needs,
        )
        _write_report(root, today.isoformat(), report)
        return report

    async def _run() -> dict[str, Any]:
        client, session_ctx = _open_kiwoom()
        async with session_ctx as http_session:
            return await btt._run_tasks(
                tasks, client=client, http_session=http_session, store=store, profile=resolved,
                apply=True, ledger=ledger, deadline=deadline, blackouts=[],
                run_date=today.isoformat(),
            )

    try:
        summary = await _run()
    except OSError as exc:
        raise RuntimeError(f"Tape sweep infrastructure failed: {exc}") from exc
    settled_now = btt._read_settled(ledger)
    recovered = sorted(
        {
            _need_key(item.symbol, item.day, item.session)
            for item in active
            if settled_now.get((item.symbol, item.day, item.session)) in ("COMPLETE", "NO_TRADES")
        }
    )
    still_keys = sorted(set(active_keys) - set(recovered))
    logger.info(
        "[DATA] stage=tape_sweep_report needs=%d pages=%d rows=%d recovered=%d unresolved=%d expired=%d remaining=%s",
        len(active_keys), summary["pages"], summary["rows"], len(recovered), len(still_keys),
        len(expired), summary["remaining"],
    )
    report = TapeSweepReport(
        days_checked=tuple(window), needs=len(active_keys), recovered=tuple(recovered),
        unresolved=tuple(still_keys), expired=tuple(expired), expiring=tuple(expiring),
        remaining=tuple(summary["remaining"]), disk_guard=summary.get("stopped_reason") == "disk_guard",
        pages=int(summary["pages"]), rows=int(summary["rows"]), expiring_needs=expiring_needs,
    )
    _write_report(root, today.isoformat(), report)
    return report


def main(argv: list[str] | None = None) -> None:
    """Run the daily tape sweep once (exit 0 when idle or disk-guarded).

    Raises:
        ValueError: Invalid lookback window or naive deadline.
        RuntimeError: Infrastructure failure; archive units are never affected.
    """
    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    parser = argparse.ArgumentParser(description="Daily Kiwoom tape sweep for recent tick needs.")
    parser.add_argument("--lookback-days", type=int, default=None, help="Re-check window in days.")
    parser.add_argument("--deadline", default=None, help="Aware ISO timestamp after which no new walk starts.")
    args = parser.parse_args(argv)
    profile = CollectionSettings()
    lookback = int(args.lookback_days) if args.lookback_days is not None else int(profile.COLLECTION_TAPE_LOOKBACK_DAYS)
    deadline = btt._parse_deadline(args.deadline) if args.deadline else _default_deadline()
    asyncio.run(run_tick_tape_sweep(lookback_days=lookback, profile=profile, deadline=deadline))


if __name__ == "__main__":  # pragma: no cover - CLI entry, exercised via run_tick_tape_sweep
    main()
