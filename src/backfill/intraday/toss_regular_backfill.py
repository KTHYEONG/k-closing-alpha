"""Toss regular-session 1m backfill for the pre-KIS-retention era and KIS-skipped symbol-days.

Toss serves raw-basis bars older than the KIS minute-history window, so dates the KIS regular
stream can no longer reach are replayed through the certified Toss route. Inside the KIS window
only symbol-days the KIS stream deliberately skips (adjusted or unknown price basis) are taken
here; no symbol-day is ever fetched by both streams. The run is resumable across invocations
through a dedicated Toss ledger that never shares keys with the KIS ledger.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src import settings
from src.backfill.intraday.blackout import blackout_end, parse_blackout_windows, wait_for_blackout
from src.backfill.intraday.extended_session_backfill import (
    ExtendedBackfillLedger,
    ExtendedBackfillTask,
    _stored_partition_symbols,
    regular_superset_symbols_by_day,
)
from src.backfill.intraday.price_basis import PriceReference
from src.backfill.intraday.toss_regular import acquire_toss_regular_bars, probe_toss_retention_floor
from src.config.collection import CollectionSettings
from src.config.market_session import INTRADAY_SESSION_REGULAR
from src.data.capture_contracts import (
    SEOUL,
    GOOD_ENTRY_STATES,
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.eod_superset import EodSupersetScreen
from src.data.intraday_store import write_intraday_partition
from src.data.panel_integrity import REQUIRED_SOURCE_COLUMNS, prepare_price_panel
from src.utils.cli_logging import configure_cli_logging
from src.utils.file_lock import exclusive_file_lock

logger = logging.getLogger(__name__)

TOSS_REGULAR_LEDGER_FILENAME: str = "toss_regular.parquet"

_STOCK_NOT_FOUND_REASON: str = "toss_stock_not_found"
_CACHED_NOT_FOUND_REASON: str = "toss_stock_not_found_cached"
_CONSOLIDATED_REASON: str = "toss_consolidated_tape"
_OUTAGE_REASON_PREFIXES: tuple[str, ...] = ("transport:", "vendor_failure:", "toss_empty_without_proof", "toss_malformed_page")
_TOSS_RAW_BASIS: str = "toss_raw"
_HEARTBEAT_EVERY_DATES: int = 25
_CALLS_PER_SYMBOL_DAY: int = 2


def default_toss_ledger_path() -> Path:
    """Dedicated Toss ledger location (one file under the history tree for the offsite loose path)."""
    return Path(settings.HISTORY_DIR) / "intraday" / "backfill_ledger" / TOSS_REGULAR_LEDGER_FILENAME


def toss_run_lock_path(ledger_path: Path | None = None) -> Path:
    """Single-instance lock beside the ledger (distinct from the ledger's own write sidecar)."""
    ledger = Path(ledger_path) if ledger_path is not None else default_toss_ledger_path()
    return ledger.parent / (TOSS_REGULAR_LEDGER_FILENAME + ".run.lock")


@dataclass(frozen=True)
class TossBackfillPlan:
    """Toss tasks plus the symbol-days deliberately left to the KIS stream or outside retention."""

    tasks: tuple[ExtendedBackfillTask, ...]
    retention_floor: str | None
    below_floor_symbol_days: int
    kis_window_symbol_days: int


def enumerate_toss_regular_tasks(
    *,
    as_of: date,
    kis_window_start: str,
    retention_floor: str | None,
    prepared_panel: pd.DataFrame,
    screen: EodSupersetScreen,
    price_reference: PriceReference,
) -> TossBackfillPlan:
    """Plan Toss regular-session tasks, ascending by date, for symbol-days the KIS regular stream will not or cannot serve. Dates before `kis_window_start` take the whole EOD fetch superset (Toss history is raw, so adjusted symbol-days are fetchable). Dates from `kis_window_start` take only superset symbols whose basis is adjusted or unknown, which the KIS stream skips by design. Dates before `retention_floor` are excluded and counted, never planned. Returns the plan with counts; `retention_floor=None` means the floor is unknown and nothing is excluded by it.

    Raises:
        ValueError: from the shared superset helper on missing panel columns.
    """
    symbols_by_day = regular_superset_symbols_by_day(
        as_of=as_of, earliest="", prepared_panel=prepared_panel, screen=screen
    )
    floor = str(retention_floor) if retention_floor is not None else None
    window_start = str(kis_window_start)
    tasks: list[ExtendedBackfillTask] = []
    below_floor = 0
    in_window = 0
    for day in sorted(symbols_by_day):
        symbols = symbols_by_day[day]
        if floor is not None and day < floor:
            below_floor += len(symbols)
            continue
        if day < window_start:
            picked = list(symbols)
        else:
            picked = [
                symbol
                for symbol in symbols
                if price_reference.is_adjusted(day, symbol) or not price_reference.is_known(day, symbol)
            ]
            in_window += len(symbols) - len(picked)
        if picked:
            tasks.append(
                ExtendedBackfillTask(
                    snapshot_date=str(day),
                    session=INTRADAY_SESSION_REGULAR,
                    symbols=tuple(sorted(set(picked))),
                )
            )
    return TossBackfillPlan(
        tasks=tuple(tasks),
        retention_floor=floor,
        below_floor_symbol_days=int(below_floor),
        kis_window_symbol_days=int(in_window),
    )


@dataclass(frozen=True)
class TossBackfillSummary:
    """Outcome counts of one bounded Toss backfill run."""

    tasks_done: int
    tasks_remaining: int
    complete: int
    consolidated: int
    not_listed: int
    failed: int
    exhausted: int
    stopped_by_deadline: bool
    outage_aborted: bool


def _delisted_symbols(ledger: ExtendedBackfillLedger) -> set[str]:
    """Symbols proven unavailable at symbol level by an earlier run (never re-requested)."""
    frame = ledger._read_all()
    if frame.empty or not {"status", "symbol", "reason"} <= set(frame.columns):
        return set()
    reasons = frame["reason"].astype(str)
    sub = frame[
        (frame["status"].astype(str) == CaptureStatus.NOT_APPLICABLE.value)
        & (reasons.isin({_STOCK_NOT_FOUND_REASON, _CACHED_NOT_FOUND_REASON}))
    ]
    if sub.empty:
        return set()
    return {str(item) for item in sub["symbol"].astype(str).tolist()}


@asynccontextmanager
async def _http_session(client: Any) -> AsyncIterator[Any]:
    """Yield a request session for any client shape (real, session-factory, or bare fake)."""
    factory = getattr(client, "create_session", None)
    if factory is None:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            yield session
        return
    produced = factory()
    if hasattr(produced, "__aenter__"):
        async with produced as entered:
            yield entered
    else:
        yield produced


def _is_outage_failure(entry_reason: str) -> bool:
    return str(entry_reason).startswith(_OUTAGE_REASON_PREFIXES)


async def run_toss_regular_backfill(
    *,
    as_of: date,
    stop_at: datetime | None,
    profile: CollectionSettings,
    client: Any,
    store: CaptureStore,
    ledger: ExtendedBackfillLedger,
    tasks: Sequence[ExtendedBackfillTask],
    eod_volumes: Mapping[tuple[str, str], float],
    now_fn: Callable[[], datetime] | None = None,
) -> TossBackfillSummary:
    """Execute Toss tasks oldest-first until done, the deadline, a blackout that outlasts the deadline, or an outage, resumable across runs through a durable ledger. Pending symbols of a task exclude ledger-terminal symbols, symbols already stored in the partition (live, KIS or earlier Toss output is never overwritten) and symbols proven unavailable at symbol level. Each date publishes its certified symbols, one manifest and the ledger rows before the next date starts, so an interruption loses at most the in-flight date.

    Side effects: partitions, manifest, raw page evidence and ledger rows; a date with no pending symbols performs no network call and writes nothing.

    Raises:
        ValueError: for an empty/naive-deadline contract violation; `OSError` when partition, manifest or ledger persistence fails (the date is not marked terminal), and once all readable dates are done when any stored partition was unreadable.
    """
    if client is None:
        raise ValueError("toss client must be provided")
    if stop_at is not None and (stop_at.tzinfo is None or stop_at.utcoffset() is None):
        raise ValueError("stop_at must be timezone-aware")
    clock = now_fn if now_fn is not None else (lambda: datetime.now(SEOUL))
    ordered = sorted(tasks, key=lambda task: (task.snapshot_date, task.session))
    windows = parse_blackout_windows(tuple(profile.COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS))
    concurrency = max(int(profile.COLLECTION_TOSS_BACKFILL_CONCURRENCY), 1)
    outage_share = float(profile.COLLECTION_TOSS_OUTAGE_FAILURE_SHARE)
    outage_min_sample = int(profile.COLLECTION_TOSS_OUTAGE_MIN_SAMPLE)
    dead = _delisted_symbols(ledger)
    done = 0
    complete = 0
    consolidated = 0
    not_listed = 0
    failed = 0
    exhausted = 0
    stopped = False
    outage_aborted = False
    unreadable: list[str] = []
    symbol_days_total = sum(len(task.symbols) for task in ordered)
    symbol_days_done = 0
    started_all = clock() if (stop_at is not None or windows) else None
    async with AsyncExitStack() as stack:
        http_session = await stack.enter_async_context(_http_session(client))
        for index, task in enumerate(ordered):
            if stop_at is not None and clock() >= stop_at:
                stopped = True
                break
            if windows:
                end = blackout_end(clock(), windows, weekdays_only=True)
                if end is not None:
                    if stop_at is not None and end > stop_at:
                        stopped = True
                        break
                    await wait_for_blackout(windows, now_fn=clock, weekdays_only=True)
                    if stop_at is not None and clock() >= stop_at:
                        stopped = True
                        break
            started = clock()
            terminal = ledger.terminal_symbols(task.snapshot_date, INTRADAY_SESSION_REGULAR)
            try:
                stored = _stored_partition_symbols(task.snapshot_date, INTRADAY_SESSION_REGULAR)
            except OSError:
                logger.error(
                    "[DATA] stage=toss_regular_backfill status=TASK_SKIPPED reason=unreadable_partition date=%s",
                    task.snapshot_date,
                )
                unreadable.append(f"{task.snapshot_date}/{INTRADAY_SESSION_REGULAR}")
                continue
            fetchable = [
                symbol
                for symbol in task.symbols
                if symbol not in terminal and symbol not in stored and symbol not in dead
            ]
            cached = [
                symbol
                for symbol in task.symbols
                if symbol not in terminal and symbol not in stored and symbol in dead
            ]
            if not fetchable and not cached:
                done += 1
                symbol_days_done += len(task.symbols)
                continue
            run_id = f"toss-regular-{task.snapshot_date}-{uuid.uuid4().hex[:8]}"
            semaphore = asyncio.Semaphore(concurrency)

            async def _one(symbol: str) -> tuple[pd.DataFrame, CoverageEntry]:
                async with semaphore:
                    return await acquire_toss_regular_bars(
                        client,
                        http_session,
                        symbol,
                        task.snapshot_date,
                        eod_volume=eod_volumes.get((task.snapshot_date, symbol)),
                        profile=profile,
                        capture_store=store,
                        run_id=run_id,
                    )

            fetched = await asyncio.gather(*(_one(symbol) for symbol in fetchable)) if fetchable else []
            frames: dict[str, pd.DataFrame] = {}
            attempted = []
            for symbol, (frame, entry) in zip(fetchable, fetched):
                attempted.append(entry)
                if (
                    entry.status == CaptureStatus.COMPLETE
                    and frame is not None
                    and not frame.empty
                ):
                    frames[str(symbol)] = frame
                if entry.reason == _STOCK_NOT_FOUND_REASON:
                    dead.add(str(symbol))
            is_outage = len(attempted) >= outage_min_sample and (
                sum(1 for entry in attempted if _is_outage_failure(entry.reason)) / len(attempted)
            ) >= outage_share
            kept = (
                [entry for entry in attempted if not _is_outage_failure(entry.reason)]
                if is_outage
                else list(attempted)
            )
            price_bases = {
                str(entry.symbol): _TOSS_RAW_BASIS
                for entry in kept
                if entry.status == CaptureStatus.COMPLETE and entry.symbol is not None
            }
            if frames:
                chunk = pd.concat([frames[name] for name in sorted(frames)], ignore_index=True)
                coverage = {
                    str(entry.symbol): entry
                    for entry in kept
                    if entry.symbol is not None and str(entry.symbol) in frames
                }
                write_intraday_partition(
                    chunk, 1, task.snapshot_date, INTRADAY_SESSION_REGULAR, coverage=coverage
                )
            if kept:
                manifest_status = (
                    CaptureStatus.COMPLETE
                    if all(entry.status in GOOD_ENTRY_STATES for entry in kept)
                    else CaptureStatus.PARTIAL
                )
                manifest = CaptureManifest(
                    schema_version=1,
                    context=CaptureContext(
                        trading_date=date.fromisoformat(task.snapshot_date),
                        run_id=run_id,
                        dataset=CaptureDataset.MINUTE_BARS,
                        vendor="toss",
                        endpoint="backfill-task",
                        symbol=None,
                        venue="owner-local",
                        session=task.session,
                        capture_reason="toss-regular-backfill",
                        cohort_id=None,
                        scheduled_at=None,
                    ),
                    cohort=None,
                    completed_at=clock(),
                    entries=tuple(kept),
                    artifacts=tuple(dict.fromkeys(ref for entry in kept for ref in entry.raw_refs)),
                    status=manifest_status,
                )
                store.publish_manifest(manifest)
                ledger.record(
                    task.snapshot_date,
                    INTRADAY_SESSION_REGULAR,
                    kept,
                    run_id=run_id,
                    attempted_at=clock(),
                    vendor="toss",
                    price_bases=price_bases,
                )
                latest = ledger._read_all()
                if not latest.empty:
                    sub = latest[
                        (latest["snapshot_date"].astype(str) == str(task.snapshot_date))
                        & (latest["session"].astype(str) == INTRADAY_SESSION_REGULAR)
                    ]
                    if not sub.empty:
                        final = sub.drop_duplicates(subset=["symbol"], keep="last")
                        exhausted_now = {
                            str(item)
                            for item in final.loc[
                                final["status"].astype(str) == "EXHAUSTED",
                                "symbol",
                            ].tolist()
                        }
                        exhausted += sum(
                            1
                            for entry in kept
                            if entry.status == CaptureStatus.FAILED
                            and entry.symbol is not None
                            and str(entry.symbol) in exhausted_now
                        )
            if cached:
                ledger.record_cached_absent(
                    task.snapshot_date,
                    INTRADAY_SESSION_REGULAR,
                    cached,
                    reason=_CACHED_NOT_FOUND_REASON,
                    run_id=run_id,
                    attempted_at=clock(),
                    vendor="toss",
                )
            complete += sum(1 for entry in kept if entry.status == CaptureStatus.COMPLETE)
            consolidated += sum(
                1 for entry in kept if entry.reason == _CONSOLIDATED_REASON
            )
            not_listed += sum(
                1 for entry in kept if entry.reason == _STOCK_NOT_FOUND_REASON
            ) + len(cached)
            failed += sum(
                1
                for entry in kept
                if entry.status in (CaptureStatus.FAILED, CaptureStatus.UNKNOWN)
            )
            if not is_outage:
                done += 1
                symbol_days_done += len(task.symbols)
            elapsed = (clock() - started).total_seconds()
            logger.info(
                "[DATA] stage=toss_regular_backfill date=%s pending=%d complete=%d consolidated=%d not_listed=%d failed=%d elapsed_s=%.1f",
                task.snapshot_date,
                len(fetchable),
                sum(1 for entry in kept if entry.status == CaptureStatus.COMPLETE),
                sum(1 for entry in kept if entry.reason == _CONSOLIDATED_REASON),
                sum(1 for entry in kept if entry.reason == _STOCK_NOT_FOUND_REASON) + len(cached),
                sum(
                    1
                    for entry in kept
                    if entry.status in (CaptureStatus.FAILED, CaptureStatus.UNKNOWN)
                ),
                elapsed,
            )
            if (index + 1) % _HEARTBEAT_EVERY_DATES == 0 or index + 1 == len(ordered):
                elapsed_all = (clock() - started_all).total_seconds() if started_all is not None else 0.0
                eta_min = (
                    elapsed_all / max(symbol_days_done, 1) * (symbol_days_total - symbol_days_done) / 60.0
                    if symbol_days_total > symbol_days_done
                    else 0.0
                )
                logger.info(
                    "[DATA] stage=toss_regular_backfill progress=dates %d/%d symbol_days=%d/%d eta_min=%.1f",
                    done,
                    len(ordered),
                    symbol_days_done,
                    symbol_days_total,
                    eta_min,
                )
            if is_outage:
                logger.error(
                    "[DATA] stage=toss_regular_backfill status=OUTAGE_ABORT date=%s outage_failures=%d attempted=%d",
                    task.snapshot_date,
                    sum(1 for entry in attempted if _is_outage_failure(entry.reason)),
                    len(attempted),
                )
                outage_aborted = True
                break
    if unreadable:
        raise OSError(f"Cannot read existing partition evidence for toss tasks {unreadable}")
    return TossBackfillSummary(
        tasks_done=done,
        tasks_remaining=len(ordered) - done,
        complete=complete,
        consolidated=consolidated,
        not_listed=not_listed,
        failed=failed,
        exhausted=exhausted,
        stopped_by_deadline=stopped,
        outage_aborted=outage_aborted,
    )


def _load_history_frame(as_of: date) -> pd.DataFrame:
    from src import settings as app_settings

    wide_columns = sorted(REQUIRED_SOURCE_COLUMNS | {"close_raw", "market"})
    history = pd.read_parquet(app_settings.PRICE_HISTORY_PARQUET_PATH, columns=wide_columns)
    days = pd.to_datetime(history["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return history[(days <= as_of.isoformat())].copy()


def _build_plan(
    *,
    as_of: date,
    retention_floor: str | None,
    retention_days: int,
    prepared_panel: pd.DataFrame,
    wide_history: pd.DataFrame,
    profile: CollectionSettings,
) -> tuple[TossBackfillPlan, dict[tuple[str, str], float]]:
    screen = EodSupersetScreen.from_profile(profile)
    price_reference = PriceReference.from_price_history(wide_history)
    kis_window_start = (as_of - timedelta(days=int(retention_days))).isoformat()
    plan = enumerate_toss_regular_tasks(
        as_of=as_of,
        kis_window_start=kis_window_start,
        retention_floor=retention_floor,
        prepared_panel=prepared_panel,
        screen=screen,
        price_reference=price_reference,
    )
    volumes = pd.to_numeric(prepared_panel["volume"], errors="coerce")
    days = pd.to_datetime(prepared_panel["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    symbols = prepared_panel["symbol"].astype(str).str.zfill(6)
    eod_volumes = {
        (str(day), str(symbol)): float(volume)
        for day, symbol, volume in zip(days.tolist(), symbols.tolist(), volumes.tolist())
    }
    return plan, eod_volumes


def _apply_plan_filters(
    plan: TossBackfillPlan,
    *,
    start: str | None,
    end: str | None,
    max_dates: int | None,
) -> TossBackfillPlan:
    tasks = list(plan.tasks)
    if start is not None:
        tasks = [task for task in tasks if task.snapshot_date >= str(start)]
    if end is not None:
        tasks = [task for task in tasks if task.snapshot_date <= str(end)]
    if max_dates is not None:
        if int(max_dates) < 1:
            raise ValueError(f"--max-dates must be >= 1, got {max_dates!r}")
        tasks = tasks[: int(max_dates)]
    return TossBackfillPlan(
        tasks=tuple(tasks),
        retention_floor=plan.retention_floor,
        below_floor_symbol_days=plan.below_floor_symbol_days,
        kis_window_symbol_days=plan.kis_window_symbol_days,
    )


def _print_plan_estimate(plan: TossBackfillPlan) -> None:
    from src.api.toss.client import TOSS_RATE_LIMIT_GROUPS

    symbol_days = sum(len(task.symbols) for task in plan.tasks)
    calls = _CALLS_PER_SYMBOL_DAY * symbol_days
    group_rate = float(TOSS_RATE_LIMIT_GROUPS["MARKET_DATA_CHART"])
    print(f"retention_floor={plan.retention_floor}")
    print(f"planned_dates={len(plan.tasks)}")
    print(f"planned_symbol_days={symbol_days}")
    print(f"below_floor_symbol_days={plan.below_floor_symbol_days}")
    print(f"kis_window_symbol_days={plan.kis_window_symbol_days}")
    print(f"estimated_calls={calls}")
    print(f"estimated_hours_at_group_rate={calls / group_rate / 3600:.2f}")
    print(f"estimated_hours_at_half_rate={calls / (group_rate / 2) / 3600:.2f}")


def _load_inputs(as_of: date) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    wide_history = _load_history_frame(as_of)
    prepared, _provenance = prepare_price_panel(wide_history)
    calendar = sorted(
        {str(item) for item in pd.to_datetime(wide_history["date"]).dt.strftime("%Y-%m-%d").tolist()}
    )
    return wide_history, prepared, calendar


def effective_floor(probed: str | None, usable_from: str) -> str | None:
    """Combine the vendor retention floor with the first date whose volumes are trusted.

    Args:
        probed: Earliest date Toss still serves, or None when the probe found the vendor empty (unknown).
        usable_from: ``COLLECTION_TOSS_USABLE_FROM_DATE``.

    Returns:
        The later of the two dates; None when the retention floor is unknown, so callers can fail closed.
    """
    if probed is None:
        return None
    return max(str(probed), str(usable_from))


async def _probe_floor(
    client: Any, session: Any, *, calendar: Sequence[str], reference_symbol: str, usable_from: str
) -> str | None:
    probed = await probe_toss_retention_floor(
        client, session, trading_days=list(calendar), reference_symbol=str(reference_symbol)
    )
    return effective_floor(probed, usable_from)


def _open_toss_client() -> Any:
    from src import settings as app_settings
    from src.api.toss.client import TossApiClient

    if not (app_settings.TOSS_APP_KEY and app_settings.TOSS_APP_SECRET):
        raise RuntimeError("Toss credentials are not configured")
    return TossApiClient()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Toss regular-session backfill (exit 0 on completion or deadline stop)."""
    import aiohttp

    parser = argparse.ArgumentParser(description="Toss regular-session 1m backfill (pre-KIS-retention era).")
    parser.add_argument("--as-of", default=None, help="Run date YYYY-MM-DD (default KST today).")
    parser.add_argument("--start", default=None, help="Inclusive plan start YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Inclusive plan end YYYY-MM-DD.")
    parser.add_argument("--stop-at", default=None, help="HHMMSS KST after which no new date starts (default none).")
    parser.add_argument("--max-dates", default=None, type=int, help="Oldest-first cap on planned dates.")
    parser.add_argument("--plan-only", action="store_true", help="Print the plan estimate without writing anything.")
    args = parser.parse_args(argv)
    profile = CollectionSettings()
    now = datetime.now(SEOUL)
    as_of = date.fromisoformat(str(args.as_of)) if args.as_of else now.date()
    if args.start is not None:
        date.fromisoformat(str(args.start))
    if args.end is not None:
        date.fromisoformat(str(args.end))
    if args.start is not None and args.end is not None and str(args.end) < str(args.start):
        raise ValueError(f"Invalid plan range: {args.start!r}..{args.end!r}")
    stop_at: datetime | None = None
    if args.stop_at is not None:
        stop_hhmmss = str(args.stop_at)
        if not re.fullmatch(r"\d{6}", stop_hhmmss):
            raise ValueError(f"Invalid --stop-at HHMMSS: {args.stop_at!r}")
        stop_at = datetime(
            now.year, now.month, now.day,
            int(stop_hhmmss[0:2]), int(stop_hhmmss[2:4]), int(stop_hhmmss[4:6]),
            tzinfo=SEOUL,
        )
        if stop_at <= now:
            stop_at += timedelta(days=1)
    retention_days = int(profile.COLLECTION_KIS_MINUTE_RETENTION_DAYS)

    async def _plan_only() -> int:
        wide_history, prepared, calendar = _load_inputs(as_of)
        client = _open_toss_client()
        async with aiohttp.ClientSession() as session:
            await client.ensure_token(session)
            floor = await _probe_floor(
                client, session, calendar=calendar,
                reference_symbol=profile.COLLECTION_TOSS_RETENTION_REFERENCE_SYMBOL,
                usable_from=profile.COLLECTION_TOSS_USABLE_FROM_DATE,
            )
        plan, _volumes = _build_plan(
            as_of=as_of, retention_floor=floor, retention_days=retention_days,
            prepared_panel=prepared, wide_history=wide_history, profile=profile,
        )
        plan = _apply_plan_filters(plan, start=args.start, end=args.end, max_dates=args.max_dates)
        _print_plan_estimate(plan)
        return 0

    if args.plan_only:
        return asyncio.run(_plan_only())

    async def _run_locked() -> int:
        wide_history, prepared, calendar = _load_inputs(as_of)
        client = _open_toss_client()
        async with aiohttp.ClientSession() as session:
            await client.ensure_token(session)
            floor = await _probe_floor(
                client, session, calendar=calendar,
                reference_symbol=profile.COLLECTION_TOSS_RETENTION_REFERENCE_SYMBOL,
                usable_from=profile.COLLECTION_TOSS_USABLE_FROM_DATE,
            )
            if floor is None:
                logger.error("[DATA] stage=toss_regular_backfill status=ABORT reason=retention_floor_unknown")
                return 1
            plan, eod_volumes = _build_plan(
                as_of=as_of, retention_floor=floor, retention_days=retention_days,
                prepared_panel=prepared, wide_history=wide_history, profile=profile,
            )
            plan = _apply_plan_filters(plan, start=args.start, end=args.end, max_dates=args.max_dates)
            store = CaptureStore(_capture_root(profile))
            ledger = ExtendedBackfillLedger(default_toss_ledger_path())
            summary = await run_toss_regular_backfill(
                as_of=as_of, stop_at=stop_at, profile=profile, client=client,
                store=store, ledger=ledger, tasks=list(plan.tasks), eod_volumes=eod_volumes,
            )
            logger.info(
                "[DATA] stage=toss_regular_backfill status=%s tasks_done=%d tasks_remaining=%d complete=%d consolidated=%d not_listed=%d failed=%d exhausted=%d stopped_by_deadline=%s outage_aborted=%s",
                "OUTAGE_ABORT" if summary.outage_aborted else "DONE",
                summary.tasks_done, summary.tasks_remaining, summary.complete,
                summary.consolidated, summary.not_listed, summary.failed,
                summary.exhausted, summary.stopped_by_deadline, summary.outage_aborted,
            )
            if summary.outage_aborted:
                return 1
            return 0

    ledger = ExtendedBackfillLedger(default_toss_ledger_path())
    try:
        with exclusive_file_lock(
            toss_run_lock_path(ledger.path), timeout_seconds=0.0, purpose="toss-backfill"
        ):
            return asyncio.run(_run_locked())
    except TimeoutError:
        logger.error(
            "[DATA] stage=toss_regular_backfill status=LOCKED reason=another_instance_holds_the_ledger_lock"
        )
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry; logic covered via runner scenarios
    configure_cli_logging()
    raise SystemExit(main())
