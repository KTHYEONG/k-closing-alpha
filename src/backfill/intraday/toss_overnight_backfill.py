"""Nightly Toss phase after the KIS phase for NXT overnight 1m backfill.

The KIS/Kiwoom minute window is perishable (a day of history is lost every night) while Toss keeps its
candles, so KIS work always runs first and Toss only fills entry days the primary vendors can no longer
serve. Toss tasks run newest entry day first because the NXT listing was broadest recently, keeping the
most valuable days when a night is interrupted. A calibration gate reproves Toss against stored
KIS/Kiwoom evenings before any write, because Toss is a consolidated tape. One manifest per
task-session is published (gdrive cost is object count, so per-chunk manifests are avoided).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src import settings as app_settings
from src.backfill.intraday.extended_session_backfill import (
    ExtendedBackfillLedger,
    ExtendedBackfillSummary,
    ExtendedBackfillTask,
    _entry_day_universes,
    _stored_partition_symbols,
    run_extended_session_backfill,
)
from src.backfill.intraday.toss_overnight import (
    fetch_toss_window,
    fetch_toss_overnight_pair,
    session_end_label_window,
    toss_session_frame,
)
from src.config.collection import CollectionSettings
from src.config.market_session import (
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_NXT_PREMARKET,
    KRX_AFTERMARKET_START_DATE,
)
from src.data.capture_contracts import (
    GOOD_ENTRY_STATES,
    SEOUL,
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.intraday_store import intraday_partition_path, write_intraday_partition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TossOvernightTask:
    entry_day: str
    next_day: str
    symbols: tuple[str, ...]


def enumerate_toss_overnight_tasks(
    *,
    as_of: date,
    kis_retention_days: int,
    nxt_start_date: str,
    krx_aftermarket_start_date: str,
    min_change_ratio: float,
    price_history: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
) -> list[TossOvernightTask]:
    """Enumerate point-in-time Toss overnight tasks for entry days older than the KIS/Kiwoom retention window.

    Entry days run from the NXT start date up to (exclusive) the earlier of the KIS window start and the KRX aftermarket
    start, so Toss never serves a day the primary vendors can still serve and never serves a consolidated-tape day. The
    next trading day comes from the price-history calendar only and must precede `as_of`. The universe of each day is the
    same as the KIS runner's, observable at that day's close.

    Args:
        as_of: KST run date.
        kis_retention_days: KIS window length in calendar days (window start = as_of - kis_retention_days).
        nxt_start_date: First NXT trading day (ISO).
        krx_aftermarket_start_date: First KRX evening-session day (ISO).
        min_change_ratio: Entry-day close-to-close screen threshold.
        price_history: Columns date, symbol, close, prev_close; its dates are the trading calendar.
        candidate_pairs: Columns snapshot_date, symbol.

    Returns:
        Tasks ordered newest entry day first; days without a universe or without a following trading day are omitted.
    """
    from datetime import timedelta

    as_of_str = as_of.isoformat()
    window_start = (as_of - timedelta(days=int(kis_retention_days))).isoformat()
    upper = min(str(window_start), str(krx_aftermarket_start_date))
    calendar, universes = _entry_day_universes(price_history, candidate_pairs, float(min_change_ratio))
    cal_index = {day: pos for pos, day in enumerate(calendar)}
    tasks: list[TossOvernightTask] = []
    for entry_day in sorted(universes):
        if not (str(nxt_start_date) <= str(entry_day) < upper):
            continue
        universe = universes[entry_day]
        pos = cal_index.get(entry_day)
        if pos is None or pos + 1 >= len(calendar):
            continue
        next_day = calendar[pos + 1]
        if not (str(next_day) < as_of_str):
            continue
        symbols = tuple(sorted({str(item).zfill(6) for item in universe}))
        tasks.append(TossOvernightTask(entry_day=str(entry_day), next_day=str(next_day), symbols=symbols))
    tasks.sort(key=lambda task: task.entry_day, reverse=True)
    return tasks


@dataclass(frozen=True)
class TossCalibrationVerdict:
    status: Literal["PASS", "FAIL", "NO_REFERENCE"]
    dates: tuple[str, ...]
    symbols: int
    traded_minutes: int
    exact_ratio: float | None


def list_stored_nxt_evening_dates() -> list[str]:
    """Return ascending ISO dates of stored `nxt_aftermarket` 1m partitions; missing tree -> empty list."""
    base = Path(app_settings.HISTORY_DIR) / "intraday" / "1m" / INTRADAY_SESSION_NXT_AFTERMARKET
    if not base.exists():
        return []
    found = [path.stem for path in base.glob("*/*.parquet")]
    return sorted(day for day in found if len(day) == 10 and day[:4].isdigit())


def _default_read_reference(day: str) -> pd.DataFrame | None:
    target = intraday_partition_path(1, str(day), INTRADAY_SESSION_NXT_AFTERMARKET)
    if not target.exists():
        return None
    try:
        return pd.read_parquet(target)
    except Exception:
        return None


def _evenly_spaced_indices(count: int, want: int) -> list[int]:
    if count <= want or want <= 1:
        return list(range(min(count, max(want, 1)))) if want > 1 else [0] if count else []
    return sorted({round(index * (count - 1) / (want - 1)) for index in range(want)})


def _reference_traded_minutes(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=["symbol", "ts_hms", "volume", "close"])
    work = frame
    if "vendor" in work.columns:
        work = work[work["vendor"].astype(str).isin(("kis", "kiwoom"))]
    if "has_trade" in work.columns:
        work = work[work["has_trade"].astype(bool)]
    else:
        work = work[pd.to_numeric(work["volume"], errors="coerce").fillna(0) > 0]
    if len(work) == 0:
        return pd.DataFrame(columns=["symbol", "ts_hms", "volume", "close"])
    return pd.DataFrame(
        {
            "symbol": work["symbol"].astype(str).str.zfill(6),
            "ts_hms": pd.to_numeric(work["ts_hms"], errors="coerce").astype("int64"),
            "volume": pd.to_numeric(work["volume"], errors="coerce").astype("int64"),
            "close": pd.to_numeric(work["close"], errors="coerce").astype("int64"),
        }
    ).dropna()


async def calibrate_toss_against_stored(
    toss: Any,
    http_session: Any,
    *,
    profile: CollectionSettings,
    window_start: str,
    krx_aftermarket_start_date: str,
    list_reference_dates: Callable[[], Sequence[str]] | None = None,
    read_reference: Callable[[str], pd.DataFrame | None] | None = None,
) -> TossCalibrationVerdict:
    """Prove that Toss still reproduces the primary vendors before any Toss data is written.

    The reference is what the KIS/Kiwoom archive already stored for NXT evenings inside [window_start, krx_aftermarket_start).
    A deterministic sample of dates and symbols is refetched from Toss and compared minute by minute on traded minutes
    (volume and close). The gate exists because Toss is a consolidated tape and a vendor-side change would otherwise poison
    the historical partitions silently.

    Args:
        toss: Client exposing `get_candles`.
        http_session: Open HTTP session.
        profile: Sample sizes and pass thresholds.
        window_start: First date of the reference window (ISO).
        krx_aftermarket_start_date: Exclusive end of the reference window (ISO).
        list_reference_dates: Stored `nxt_aftermarket` partition dates; None scans the partition tree.
        read_reference: date -> stored partition frame (symbol, ts_hms, volume, close, vendor); None reads parquet.

    Returns:
        PASS when at least `COLLECTION_TOSS_CALIBRATION_MIN_TRADED_MINUTES` traded minutes were compared with an exact
        ratio >= `COLLECTION_TOSS_CALIBRATION_MIN_EXACT_RATIO`; FAIL when compared but below the ratio; NO_REFERENCE when
        the stored reference is too thin to decide (fail-closed: callers write nothing).
    """
    lister = list_reference_dates if list_reference_dates is not None else list_stored_nxt_evening_dates
    reader = read_reference if read_reference is not None else _default_read_reference
    try:
        stored = sorted({str(day) for day in (lister() or [])})
    except Exception:
        stored = []
    eligible = [day for day in stored if str(window_start) <= day < str(krx_aftermarket_start_date)]

    per_date_symbols: dict[str, list[str]] = {}
    per_date_ref: dict[str, pd.DataFrame] = {}
    for day in eligible:
        try:
            frame = reader(day)
        except Exception:
            continue
        traded = _reference_traded_minutes(frame)
        if traded.empty:
            continue
        counts = traded.groupby("symbol").size().reset_index(name="n").sort_values(["n", "symbol"], kind="stable")
        per_date_symbols[day] = [str(item) for item in counts["symbol"].tolist()]
        per_date_ref[day] = traded

    if not per_date_symbols:
        logger.info(
            "[DATA] stage=toss_calibration status=NO_REFERENCE dates=[] symbols=0 minutes=0 ratio=None reason=thin_reference"
        )
        return TossCalibrationVerdict(status="NO_REFERENCE", dates=(), symbols=0, traded_minutes=0, exact_ratio=None)

    want_dates = int(profile.COLLECTION_TOSS_CALIBRATION_DATES)
    date_list = sorted(per_date_symbols)
    sampled_dates = [date_list[index] for index in _evenly_spaced_indices(len(date_list), want_dates)]

    want_symbols = int(profile.COLLECTION_TOSS_CALIBRATION_SYMBOLS_PER_DATE)
    samples: list[tuple[str, str]] = []
    for day in sampled_dates:
        symbols = per_date_symbols[day]
        samples.extend((day, symbols[index]) for index in _evenly_spaced_indices(len(symbols), want_symbols))

    evening_first, evening_last = session_end_label_window(INTRADAY_SESSION_NXT_AFTERMARKET)
    total_match = 0
    total_union = 0
    counted = 0
    for day, symbol in samples:
        window = await fetch_toss_window(
            toss, http_session, symbol,
            before=f"{day}T{evening_last}:00.000+09:00",
            stop_day=day, stop_label=evening_first,
        )
        if window.error is not None:
            continue
        tossed = toss_session_frame(window.candles, symbol, day, INTRADAY_SESSION_NXT_AFTERMARKET)
        ref = per_date_ref[day]
        ref_rows = ref[ref["symbol"] == str(symbol).zfill(6)]
        ref_map = {int(ts): (int(vol), int(close)) for ts, vol, close in zip(ref_rows["ts_hms"], ref_rows["volume"], ref_rows["close"])}
        toss_map = (
            {int(ts): (int(vol), int(close)) for ts, vol, close in zip(tossed["ts_hms"], tossed["volume"], tossed["close"])}
            if not tossed.empty
            else {}
        )
        union = set(ref_map) | set(toss_map)
        match = sum(1 for key in union if key in ref_map and key in toss_map and ref_map[key] == toss_map[key])
        total_match += match
        total_union += len(union)
        counted += 1

    if total_union < int(profile.COLLECTION_TOSS_CALIBRATION_MIN_TRADED_MINUTES):
        logger.info(
            "[DATA] stage=toss_calibration status=NO_REFERENCE dates=%s symbols=%d minutes=%d ratio=None reason=thin_reference",
            sorted(sampled_dates), counted, total_union,
        )
        return TossCalibrationVerdict(
            status="NO_REFERENCE", dates=tuple(sorted(sampled_dates)),
            symbols=counted, traded_minutes=total_union, exact_ratio=None,
        )
    ratio = total_match / total_union if total_union else 0.0
    status: Literal["PASS", "FAIL"] = (
        "PASS" if ratio >= float(profile.COLLECTION_TOSS_CALIBRATION_MIN_EXACT_RATIO) else "FAIL"
    )
    logger.info(
        "[DATA] stage=toss_calibration status=%s dates=%s symbols=%d minutes=%d ratio=%.6f",
        status, sorted(sampled_dates), counted, total_union, ratio,
    )
    return TossCalibrationVerdict(
        status=status, dates=tuple(sorted(sampled_dates)),
        symbols=counted, traded_minutes=total_union, exact_ratio=ratio,
    )


@dataclass(frozen=True)
class TossBackfillSummary:
    tasks_done: int
    tasks_remaining: int
    complete: int
    no_trades: int
    not_listed: int
    failed: int
    calls: int
    stopped_reason: Literal["done", "deadline", "date_cap", "circuit_open", "calibration_failed", "calibration_no_reference"]


def _seoul_now() -> datetime:
    return datetime.now(SEOUL)


def _session_pending(symbol: str, terminal: frozenset[str], stored: set[str]) -> bool:
    return symbol not in terminal and symbol not in stored


def _task_has_pending(task: TossOvernightTask, ledger: ExtendedBackfillLedger) -> bool:
    if str(task.entry_day) >= KRX_AFTERMARKET_START_DATE:
        return False
    evening_terminal = ledger.terminal_symbols(task.entry_day, INTRADAY_SESSION_NXT_AFTERMARKET)
    premarket_terminal = ledger.terminal_symbols(task.next_day, INTRADAY_SESSION_NXT_PREMARKET)
    evening_stored = _stored_partition_symbols(task.entry_day, INTRADAY_SESSION_NXT_AFTERMARKET)
    premarket_stored = _stored_partition_symbols(task.next_day, INTRADAY_SESSION_NXT_PREMARKET)
    for symbol in task.symbols:
        if _session_pending(symbol, evening_terminal, evening_stored):
            return True
        if _session_pending(symbol, premarket_terminal, premarket_stored):
            return True
    return False


async def run_toss_overnight_backfill(
    *,
    as_of: date,
    stop_at: datetime,
    profile: CollectionSettings,
    toss: Any,
    http_session: Any,
    store: CaptureStore,
    ledger: ExtendedBackfillLedger,
    tasks: Sequence[TossOvernightTask],
    window_start: str,
    now_fn: Callable[[], datetime] | None = None,
    list_reference_dates: Callable[[], Sequence[str]] | None = None,
    read_reference: Callable[[str], pd.DataFrame | None] | None = None,
) -> TossBackfillSummary:
    """Execute Toss overnight tasks newest-first, resumable across nights, behind a calibration gate.

    A symbol is pending for a task when either of its two sessions is neither ledger-terminal nor already stored; a pair
    is fetched once and each session is published only if that session was pending. Work is checkpointed per symbol chunk
    (partitions first, then ledger rows) so an interruption loses at most one chunk. Failed, partial and unknown outcomes
    stay non-terminal and are retried on a later night, never inside the same run.

    Args:
        as_of: KST run date.
        stop_at: Aware KST instant after which no new chunk starts.
        profile: Rate, concurrency, caps and calibration thresholds.
        toss: Client bound to the capped chart rate.
        http_session: Open HTTP session.
        store: Capture store for raw evidence and manifests.
        ledger: Durable outcome ledger shared with the KIS phase.
        tasks: Output of `enumerate_toss_overnight_tasks`.
        window_start: First date of the KIS/Kiwoom reference window (ISO).
        now_fn: Aware clock; None uses Asia/Seoul now.
        list_reference_dates: Calibration reference date lister (injectable).
        read_reference: Calibration reference reader (injectable).

    Returns:
        Counts, call total and the reason the run ended.

    Raises:
        ValueError: Naive `stop_at`.
        OSError: Partition, manifest or ledger persistence fails (fail loud; the chunk is not marked terminal).
    """
    if stop_at.tzinfo is None or stop_at.utcoffset() is None:
        raise ValueError("stop_at must be timezone-aware")
    clock = now_fn if now_fn is not None else _seoul_now
    ordered = list(tasks or [])

    verdict = await calibrate_toss_against_stored(
        toss, http_session, profile=profile, window_start=str(window_start),
        krx_aftermarket_start_date=KRX_AFTERMARKET_START_DATE,
        list_reference_dates=list_reference_dates, read_reference=read_reference,
    )
    if verdict.status != "PASS":
        reason: Literal["calibration_failed", "calibration_no_reference"] = (
            "calibration_failed" if verdict.status == "FAIL" else "calibration_no_reference"
        )
        pending_tasks = len(ordered)
        return TossBackfillSummary(
            tasks_done=0, tasks_remaining=pending_tasks, complete=0, no_trades=0,
            not_listed=0, failed=0, calls=0, stopped_reason=reason,
        )

    batch_size = int(profile.COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE)
    concurrency = int(profile.COLLECTION_TOSS_BACKFILL_CONCURRENCY)
    date_cap = int(profile.COLLECTION_TOSS_BACKFILL_MAX_DATES_PER_RUN)
    max_errors = int(profile.COLLECTION_TOSS_BACKFILL_MAX_CONSECUTIVE_ERRORS)

    complete = 0
    no_trades = 0
    not_listed = 0
    failed = 0
    calls = 0
    tasks_done = 0
    dates_used = 0
    consecutive_errors = 0
    stopped_reason: Literal["done", "deadline", "date_cap", "circuit_open"] = "done"
    remaining = 0

    for task_pos, task in enumerate(ordered):
        if str(task.entry_day) >= KRX_AFTERMARKET_START_DATE:
            continue
        evening_terminal = ledger.terminal_symbols(task.entry_day, INTRADAY_SESSION_NXT_AFTERMARKET)
        premarket_terminal = ledger.terminal_symbols(task.next_day, INTRADAY_SESSION_NXT_PREMARKET)
        evening_stored = _stored_partition_symbols(task.entry_day, INTRADAY_SESSION_NXT_AFTERMARKET)
        premarket_stored = _stored_partition_symbols(task.next_day, INTRADAY_SESSION_NXT_PREMARKET)

        def _pending(symbol: str) -> tuple[bool, bool]:
            eve = _session_pending(symbol, evening_terminal, evening_stored)
            pre = _session_pending(symbol, premarket_terminal, premarket_stored)
            return eve, pre

        pending_symbols = [s for s in task.symbols if any(_pending(s))]
        if not pending_symbols:
            continue
        if dates_used >= date_cap:
            remaining = sum(1 for later in ordered[task_pos:] if _task_has_pending(later, ledger))
            stopped_reason = "date_cap"
            break
        dates_used += 1
        task_started = clock()
        pair_run = f"toss-backfill-{task.entry_day}-pair-{uuid.uuid4().hex[:8]}"
        evening_processed: list[tuple[pd.DataFrame, CoverageEntry]] = []
        premarket_processed: list[tuple[pd.DataFrame, CoverageEntry]] = []
        task_calls = 0
        task_complete = 0
        task_no_trades = 0
        task_not_listed = 0
        task_failed = 0
        interrupted = False

        chunks = [pending_symbols[pos:pos + batch_size] for pos in range(0, len(pending_symbols), batch_size)]
        for chunk in chunks:
            if clock() >= stop_at:
                interrupted = True
                stopped_reason = "deadline"
                break
            sem = asyncio.Semaphore(max(1, concurrency))

            async def _fetch_one(symbol: str):
                async with sem:
                    result = await fetch_toss_overnight_pair(
                        toss, http_session, symbol, task.entry_day, task.next_day,
                        store=store, run_id=pair_run,
                    )
                    return symbol, result

            fetched = await asyncio.gather(*(_fetch_one(symbol) for symbol in chunk))
            task_calls += sum(result.calls for _, result in fetched)
            by_symbol = dict(fetched)

            evening_complete_batch: list[pd.DataFrame] = []
            evening_complete_cov: dict[str, CoverageEntry] = {}
            premarket_complete_batch: list[pd.DataFrame] = []
            premarket_complete_cov: dict[str, CoverageEntry] = {}
            evening_ledger_rows: list[CoverageEntry] = []
            premarket_ledger_rows: list[CoverageEntry] = []
            for symbol in chunk:
                result = by_symbol[symbol]
                eve_pending, pre_pending = _pending(symbol)
                evening_frame, evening_entry = result.evening
                premarket_frame, premarket_entry = result.premarket
                if eve_pending:
                    evening_processed.append((evening_frame, evening_entry))
                    if evening_entry.status == CaptureStatus.COMPLETE and not evening_frame.empty:
                        evening_complete_batch.append(evening_frame)
                        evening_complete_cov[str(evening_entry.symbol or symbol)] = evening_entry
                    evening_ledger_rows.append(evening_entry)
                if pre_pending:
                    premarket_processed.append((premarket_frame, premarket_entry))
                    if premarket_entry.status == CaptureStatus.COMPLETE and not premarket_frame.empty:
                        premarket_complete_batch.append(premarket_frame)
                        premarket_complete_cov[str(premarket_entry.symbol or symbol)] = premarket_entry
                    premarket_ledger_rows.append(premarket_entry)

            if evening_complete_batch:
                write_intraday_partition(
                    pd.concat(evening_complete_batch, ignore_index=True), 1,
                    task.entry_day, INTRADAY_SESSION_NXT_AFTERMARKET,
                    coverage=evening_complete_cov,
                )
            if premarket_complete_batch:
                write_intraday_partition(
                    pd.concat(premarket_complete_batch, ignore_index=True), 1,
                    task.next_day, INTRADAY_SESSION_NXT_PREMARKET,
                    coverage=premarket_complete_cov,
                )
            if evening_ledger_rows:
                ledger.record(
                    task.entry_day, INTRADAY_SESSION_NXT_AFTERMARKET, evening_ledger_rows,
                    run_id=f"toss-backfill-{task.entry_day}-{INTRADAY_SESSION_NXT_AFTERMARKET}-{pair_run[-8:]}",
                    attempted_at=clock(), vendor="toss",
                )
            if premarket_ledger_rows:
                ledger.record(
                    task.next_day, INTRADAY_SESSION_NXT_PREMARKET, premarket_ledger_rows,
                    run_id=f"toss-backfill-{task.entry_day}-{INTRADAY_SESSION_NXT_PREMARKET}-{pair_run[-8:]}",
                    attempted_at=clock(), vendor="toss",
                )
            for symbol in chunk:
                if by_symbol[symbol].failed:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0
            if consecutive_errors >= max_errors:
                interrupted = True
                stopped_reason = "circuit_open"
                break

        for session, session_date, processed in (
            (INTRADAY_SESSION_NXT_AFTERMARKET, task.entry_day, evening_processed),
            (INTRADAY_SESSION_NXT_PREMARKET, task.next_day, premarket_processed),
        ):
            if not processed:
                continue
            entries = tuple(entry for _, entry in processed)
            manifest_run = f"toss-backfill-{task.entry_day}-{session}-{uuid.uuid4().hex[:8]}"
            manifest = CaptureManifest(
                schema_version=1,
                context=CaptureContext(
                    trading_date=date.fromisoformat(session_date),
                    run_id=manifest_run,
                    dataset=CaptureDataset.MINUTE_BARS,
                    vendor="toss",
                    endpoint="backfill-task",
                    symbol=None,
                    venue="owner-local",
                    session=session,
                    capture_reason="extended-backfill",
                    cohort_id=None,
                    scheduled_at=None,
                ),
                cohort=None,
                completed_at=clock(),
                entries=entries,
                artifacts=tuple(dict.fromkeys(ref for entry in entries for ref in entry.raw_refs)),
                status=(
                    CaptureStatus.COMPLETE
                    if all(entry.status in GOOD_ENTRY_STATES for entry in entries)
                    else CaptureStatus.PARTIAL
                ),
            )
            store.publish_manifest(manifest)

        for _, entry in evening_processed:
            if entry.status == CaptureStatus.COMPLETE:
                task_complete += 1
            elif entry.status == CaptureStatus.NO_TRADES:
                task_no_trades += 1
            elif entry.status == CaptureStatus.NOT_APPLICABLE:
                task_not_listed += 1
            else:
                task_failed += 1
        for _, entry in premarket_processed:
            if entry.status == CaptureStatus.COMPLETE:
                task_complete += 1
            elif entry.status == CaptureStatus.NO_TRADES:
                task_no_trades += 1
            elif entry.status == CaptureStatus.NOT_APPLICABLE:
                task_not_listed += 1
            else:
                task_failed += 1
        complete += task_complete
        no_trades += task_no_trades
        not_listed += task_not_listed
        failed += task_failed
        calls += task_calls
        tasks_done += 1
        elapsed = (clock() - task_started).total_seconds()
        logger.info(
            "[DATA] stage=toss_backfill entry_day=%s next_day=%s pending=%d complete=%d no_trades=%d not_listed=%d failed=%d calls=%d elapsed_s=%.1f",
            task.entry_day, task.next_day, len(pending_symbols),
            task_complete, task_no_trades, task_not_listed, task_failed, task_calls, elapsed,
        )
        if interrupted:
            remaining = sum(1 for later in ordered[task_pos + 1:] if _task_has_pending(later, ledger))
            break

    if stopped_reason == "done":
        remaining = 0
    return TossBackfillSummary(
        tasks_done=tasks_done, tasks_remaining=remaining, complete=complete, no_trades=no_trades,
        not_listed=not_listed, failed=failed, calls=calls, stopped_reason=stopped_reason,
    )


async def run_overnight_backfill_phases(
    *,
    as_of: date,
    stop_at: datetime,
    profile: CollectionSettings,
    kis_clients: Sequence[Any],
    toss: Any | None,
    http_session: Any | None,
    store: CaptureStore,
    ledger: ExtendedBackfillLedger,
    kis_tasks: Sequence[ExtendedBackfillTask],
    toss_tasks: Sequence[TossOvernightTask],
    window_start: str,
    now_fn: Callable[[], datetime] | None = None,
) -> tuple[ExtendedBackfillSummary | None, TossBackfillSummary | None]:
    """Run the KIS phase and then the Toss phase inside one overnight window.

    The KIS window loses a day of history every night, so its tasks always run first. The Toss phase starts only when the
    KIS phase drained (or was not configured) and the window is still open; a KIS deadline stop therefore never lets Toss
    consume the remaining night.

    Returns:
        (KIS summary or None when no KIS credentials are configured, Toss summary or None when Toss is disabled or skipped).
    """
    clock = now_fn if now_fn is not None else _seoul_now
    kis_summary: ExtendedBackfillSummary | None = None
    if kis_clients:
        kis_summary = await run_extended_session_backfill(
            as_of=as_of, stop_at=stop_at, profile=profile, clients=kis_clients,
            store=store, ledger=ledger, tasks=kis_tasks, now_fn=now_fn,
        )
    if toss is None or http_session is None:
        return kis_summary, None
    if kis_summary is not None and (kis_summary.tasks_remaining > 0 or kis_summary.stopped_by_deadline):
        return kis_summary, None
    if clock() >= stop_at:
        return kis_summary, None
    toss_summary = await run_toss_overnight_backfill(
        as_of=as_of, stop_at=stop_at, profile=profile, toss=toss, http_session=http_session,
        store=store, ledger=ledger, tasks=toss_tasks, window_start=str(window_start), now_fn=now_fn,
    )
    return kis_summary, toss_summary
