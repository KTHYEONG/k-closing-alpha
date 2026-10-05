"""저녁 1회 실행: 당일 워치리스트 정규세션+NXT 애프터마켓 1분봉 아카이브."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src import settings
from src.api.kis.client import KisApiClient, kis_data_client_kwargs
from src.api.kiwoom.client import KiwoomApiClient
from src.api.ls.client import LsApiClient
from src.backfill.intraday.collector import (
    collect_aftermarket_trade_ticks,
    collect_intraday_bars,
    collect_intraday_trade_ticks,
    collect_krx_aftermarket_bars,
    collect_nxt_aftermarket_bars,
    collect_nxt_premarket_bars,
)
from src.config.collection import CollectionSettings
from src.config.market_session import (
    AFTERMARKET_TICKS_START_DATE,
    ARCHIVE_AFTERMARKET_READY_HHMMSS,
    ARCHIVE_REGULAR_READY_HHMMSS,
    DEFAULT_BAR_INTERVAL_MINUTES,
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_NXT_PREMARKET,
    INTRADAY_SESSION_REGULAR,
    KRX_AFTERMARKET_START_DATE,
)
from src.daily import archive
from src.data.capture_contracts import (
    GOOD_ENTRY_STATES,
    SEOUL,
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    Cohort,
    CoverageEntry,
    SymbolObserver,
)
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.intraday_store import write_intraday_partition, write_tick_partition
from src.data.session_calendar import SessionKind, resolve_session_day
from src.data.trading_calendar import is_kis_trading_day, resolve_prev_trading_day_kis
from src.execution.paper_broker import HeldRoster, load_held_roster
from src.tools.run_outcome import RUN_OUTCOME_DEGRADED, RUN_OUTCOME_SKIPPED, record_run_outcome
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)

_VALID_PHASES = ("regular", "aftermarket", "all")


def _validate_phase(phase: str) -> None:
    if phase not in _VALID_PHASES:
        raise ValueError(f"Invalid phase: {phase!r} (expected one of {', '.join(_VALID_PHASES)})")


def resolve_archive_target_date(now: datetime, phase: str) -> str:
    """Resolve which session date an archive run must cover.

    A run before the phase's ready time on day T is a catch-up of the previous
    weekday's missed run (Persistent timers fire late after downtime); it must
    never archive T, whose session has not finished.

    Args:
        now: Aware KST wall clock.
        phase: "regular", "aftermarket" or "all" ("all" uses the aftermarket ready time).

    Returns:
        ISO date: today when now is at/after the ready time, else the previous weekday.

    Raises:
        ValueError: naive now or unknown phase.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    _validate_phase(phase)
    current = now.astimezone(SEOUL)
    today = current.date()
    ready = ARCHIVE_AFTERMARKET_READY_HHMMSS if phase in ("aftermarket", "all") else ARCHIVE_REGULAR_READY_HHMMSS
    if current.strftime("%H%M%S") >= ready:
        return today.isoformat()
    prev_day = today - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)
    return prev_day.isoformat()


def archive_phase_complete(store: CaptureStore, target_date: str, phase: str) -> bool:
    """Return True when every session manifest of the phase is COMPLETE for the date.

    Args:
        store: Capture store holding evening-archive manifests.
        target_date: ISO date.
        phase: "regular" (regular MINUTE_BARS + TRADE_TICKS) or "aftermarket"
            (nxt_premarket, nxt_aftermarket and, from KRX_AFTERMARKET_START_DATE,
            krx_aftermarket MINUTE_BARS, plus, from AFTERMARKET_TICKS_START_DATE,
            krx_aftermarket and nxt_aftermarket TRADE_TICKS); "all" requires both.

    Returns:
        True only if, for each required (dataset, session), the latest
        evening-archive manifest is COMPLETE.
    """
    _validate_phase(phase)
    try:
        manifests = store.read_manifests(str(target_date))
    except (ValueError, OSError):
        return False
    phases = {"regular", "aftermarket"} if phase == "all" else {phase}
    required: list[tuple[CaptureDataset, str]] = [
        (s.dataset, s.session)
        for s in _STREAMS
        if s.phase in phases and (s.required_from is None or str(target_date) >= s.required_from)
    ]
    for dataset, session in required:
        candidates = [
            item
            for item in manifests
            if item.context.capture_reason == "evening-archive"
            and item.context.endpoint == "archive-task"
            and item.context.dataset == dataset
            and item.context.session == session
        ]
        if not candidates:
            return False
        latest = max(candidates, key=lambda item: (item.completed_at, item.context.run_id))
        if latest.status != CaptureStatus.COMPLETE:
            return False
    return True


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


async def _resolve_previous_trading_day(client: Any, session: Any, snapshot_date: str) -> str | None:
    """Resolve the actual previous KRX trading day through the KIS oracle (KRX fallback).

    Returns None instead of raising when both oracles fail or no trading day exists within the
    lookback bound, so a calendar outage degrades the archive to today's cohort (INCOMPLETE)
    rather than losing the whole day's intraday capture.

    Args:
        client: Authenticated KIS data client.
        session: Open HTTP session of that client.
        snapshot_date: KST archive date `YYYY-MM-DD`.

    Returns:
        Previous trading day `YYYY-MM-DD`, or None when unresolved.
    """
    try:
        prev = await resolve_prev_trading_day_kis(client, session, pd.Timestamp(snapshot_date))
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "[DATA] stage=cohort status=INCOMPLETE reason=previous_day_unresolved date=%s error=%s",
            snapshot_date,
            type(exc).__name__,
        )
        return None
    return str(prev.strftime("%Y-%m-%d"))


def _panel_listed_before(cohort_date: str) -> frozenset[str]:
    """Return symbols listed on the latest panel date strictly before cohort_date."""
    panel_path = Path(settings.PRICE_HISTORY_PARQUET_PATH)
    if not panel_path.exists():
        raise FileNotFoundError(f"price_history not found: {panel_path}")
    cohort_day = date.fromisoformat(str(cohort_date))
    # 코호트 적격성 규칙(collect)과 동일: 코호트일 직전 최신 패널일의 상장 종목. 공휴일 연휴를 덮는 14일 창.
    rows = pd.read_parquet(
        panel_path,
        columns=["date", "symbol"],
        filters=[("date", ">=", pd.Timestamp(cohort_day) - pd.Timedelta(days=14)), ("date", "<", pd.Timestamp(cohort_day))],
    )
    rows = rows.assign(_d=pd.to_datetime(rows["date"]).dt.normalize())
    if rows.empty:
        raise ValueError(f"stale price_history: no rows in 14d window before {cohort_day.isoformat()}")
    latest = rows["_d"].max()
    listed = rows.loc[rows["_d"] == latest, "symbol"].astype(str)
    return frozenset(listed.tolist())


def _verify_cohort_against_panel(cohort: Cohort) -> None:
    cohort_date = cohort.trading_date.isoformat()
    listed = _panel_listed_before(cohort_date)
    offending = [str(item) for item in cohort.eligible_symbols if str(item) not in listed]
    if offending:
        raise ValueError(
            f"cohort_contamination cohort_id={cohort.cohort_id} date={cohort_date} "
            f"n_offending={len(offending)} samples={offending[:5]}"
        )


def _resolve_cohort_codes(
    snapshot_date: str,
    profile: CollectionSettings,
    store: CaptureStore,
    *,
    previous_trading_day: str | None,
    held: HeldRoster,
) -> tuple[list[str], bool]:
    """Resolve the archive cohort as today's verified codes plus the prior session's carryover.

    `previous_trading_day` is the oracle-resolved prior session; None means it could not be
    resolved and the previous cohort is treated as missing.

    Args:
        snapshot_date: KST archive date `YYYY-MM-DD`.
        profile: Bounded acquisition profile (unused; kept for call-site symmetry).
        store: Capture store holding verified cohorts.
        previous_trading_day: Oracle-resolved prior session; None degrades to today's cohort.
        held: Open-lot roster read once by the caller; its symbols are appended and a
            not-ok roster makes the result incomplete.

    Returns:
        (codes, incomplete) with today's eligible codes plus the previous cohort's eligible
        codes and held symbols; incomplete True when any prior coverage is missing or the
        held roster is not ok.
    """
    now = datetime.now(SEOUL)
    today_cohort = store.read_cohort(str(snapshot_date), available_by=now)
    _verify_cohort_against_panel(today_cohort)
    codes: list[str] = [str(item) for item in today_cohort.eligible_symbols]
    if previous_trading_day is None:
        for item in held.symbols:
            if item not in codes:
                codes.append(item)
        if not held.ok:
            logger.warning(
                "[DATA] stage=cohort status=INCOMPLETE reason=held_roster_unavailable error=%s date=%s",
                held.failure_reason,
                snapshot_date,
            )
        logger.info(
            "[DATA] stage=cohort status=VERIFIED date=%s n_today=%d n_prev=%d",
            snapshot_date, len(today_cohort.eligible_symbols), 0,
        )
        return codes, True
    incomplete = False
    try:
        prev_cohort = store.read_cohort(previous_trading_day, available_by=now)
    except FileNotFoundError:
        logger.warning("[DATA] stage=cohort status=INCOMPLETE reason=missing_previous date=%s prev=%s", snapshot_date, previous_trading_day)
        prev_cohort = None
        incomplete = True
    if prev_cohort is not None:
        _verify_cohort_against_panel(prev_cohort)
        for item in prev_cohort.eligible_symbols:
            if str(item) not in codes:
                codes.append(str(item))
    for item in held.symbols:
        if item not in codes:
            codes.append(item)
    if not held.ok:
        logger.warning(
            "[DATA] stage=cohort status=INCOMPLETE reason=held_roster_unavailable error=%s date=%s",
            held.failure_reason,
            snapshot_date,
        )
        incomplete = True
    n_prev = len(prev_cohort.eligible_symbols) if prev_cohort is not None else 0
    logger.info(
        "[DATA] stage=cohort status=VERIFIED date=%s n_today=%d n_prev=%d",
        snapshot_date, len(today_cohort.eligible_symbols), n_prev,
    )
    return codes, incomplete


def _publish_task_manifest(
    store: CaptureStore,
    *,
    trading_day: date,
    run_id: str,
    dataset: CaptureDataset,
    vendor: str,
    session: str,
    entries: list[CoverageEntry],
    expected_symbols: Sequence[str],
    roster_incomplete: bool = False,
) -> CaptureManifest:
    """Publish one evening-archive task manifest for a (dataset, session) stream.

    A stream is certified only when the whole expected cohort is covered by GOOD entries and the held roster
    was fully known: entries alone cannot prove completeness because a stream that delivered nothing would
    otherwise be vacuously COMPLETE, and a cohort that silently skipped held lots would be certified.

    Args:
        store: Capture store receiving the manifest.
        trading_day: Archived session date.
        run_id: Attempt-unique task identity.
        dataset: MINUTE_BARS or TRADE_TICKS.
        vendor: Task-level vendor label.
        session: Intraday session tag.
        entries: Per-symbol coverage in delivery order.
        expected_symbols: Archive cohort codes this stream is accountable for; may be empty only when the
            archive cohort is empty.
        roster_incomplete: True when the held-lot roster could not be read (spec 06); forces PARTIAL.

    Returns:
        The published manifest. status is COMPLETE iff roster_incomplete is False, every entry status is in
        GOOD_ENTRY_STATES, and every expected symbol has at least one entry; otherwise PARTIAL. artifacts are the de-duplicated raw_refs
        of all entries in first-seen order (unchanged).

    Raises:
        ValueError: The store rejects the manifest (e.g. conflicting immutable identity).
        OSError: Manifest publication fails.
    """
    refs: list[Any] = []
    for entry in entries:
        for ref in entry.raw_refs:
            if ref not in refs:
                refs.append(ref)
    if roster_incomplete or not all(item.status in GOOD_ENTRY_STATES for item in entries):
        status = CaptureStatus.PARTIAL
    else:
        present = {item.symbol for item in entries if item.symbol is not None}
        missing = [code for code in expected_symbols if code not in present]
        if missing:
            logger.warning(
                "[DATA] stage=intraday_archive_manifest status=PARTIAL reason=missing_entries run_id=%s n_missing=%d",
                run_id,
                len(missing),
            )
            status = CaptureStatus.PARTIAL
        else:
            status = CaptureStatus.COMPLETE
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=trading_day,
            run_id=run_id,
            dataset=dataset,
            vendor=vendor,
            endpoint="archive-task",
            symbol=None,
            venue="owner-local",
            session=session,
            capture_reason="evening-archive",
            cohort_id=None,
            scheduled_at=None,
        ),
        cohort=None,
        completed_at=datetime.now(SEOUL),
        entries=tuple(entries),
        artifacts=tuple(refs),
        status=status,
    )
    store.publish_manifest(manifest)
    return manifest


def _admit_for_write(stream: str, symbol: str, frame: pd.DataFrame | None, entry: CoverageEntry) -> bool:
    """Decide whether one per-symbol collector result may enter the authoritative partition.

    The collector contract (src/backfill/intraday/collector.py) delivers rows only for COMPLETE results and
    stages every uncertified frame itself, recording the staged ref in entry.raw_refs so the task manifest
    references it. A non-certified result that still carries rows therefore means the contract was broken;
    discarding or re-staging those rows here would leave unreferenced evidence and hide the defect, so the
    archive run aborts instead (fail-closed: no task manifest of the run is published).

    Args:
        stream: Archive stream key ("bars", "ticks", "nxt_after", "nxt_pre", "krx_after",
            "krx_after_ticks", "nxt_after_ticks"); used only for the error message.
        symbol: Symbol the collector reported the result for.
        frame: Canonical rows delivered with the result; None is treated as zero rows.
        entry: Coverage entry delivered with the result.

    Returns:
        True when entry.status is COMPLETE or NO_TRADES (write-eligible); False when the status is anything
        else and the frame has zero rows.

    Raises:
        ValueError: entry.status is not COMPLETE/NO_TRADES and the frame has at least one row. The message
            starts with "collector_contract_violation" and names stream, symbol, status and row count.
    """
    if entry.status in (CaptureStatus.COMPLETE, CaptureStatus.NO_TRADES):
        return True
    n_rows = 0 if frame is None else int(len(frame))
    if n_rows > 0:
        status = entry.status.value if isinstance(entry.status, CaptureStatus) else str(entry.status)
        raise ValueError(
            f"collector_contract_violation stream={stream} symbol={symbol} status={status} rows={n_rows}"
        )
    return False


def _resolve_nxt_tick_targets(
    cohort_codes: Sequence[str], nxt_bar_entries: Sequence[CoverageEntry]
) -> tuple[list[str], list[CoverageEntry]]:
    """Split the archive cohort into NXT aftermarket tick targets and pre-resolved skip entries.

    The NXT tick tape (Kiwoom ka10079 "_NX") exists only for NXT-listed symbols; this run's NXT aftermarket
    bar results are the evidence of listing. Every cohort symbol must leave an entry in the NXT tick manifest
    so that an empty target list can never certify the stream: vendor-evidenced non-listing is a GOOD
    NOT_APPLICABLE entry, an unresolved or missing bar result is an UNKNOWN entry.

    Args:
        cohort_codes: Archive cohort in run order (today's verified cohort, previous-session carryover,
            paper-follow symbols); unique codes.
        nxt_bar_entries: Coverage entries delivered by this run's NXT aftermarket bars stream.

    Returns:
        (targets, skipped): targets are cohort codes whose NXT bar entry is COMPLETE or NO_TRADES at venue
        "NXT", sorted ascending; skipped holds one TRADE_TICKS / INTRADAY_SESSION_NXT_AFTERMARKET entry per
        remaining cohort code, in cohort order, with rows=0, scheduled_at/first/last event time None:
        - bar entry NOT_APPLICABLE -> status NOT_APPLICABLE, venue copied from the bar entry,
          reason "skipped:nxt_bars_not_applicable", raw_refs copied from the bar entry;
        - bar entry present otherwise -> status UNKNOWN, venue "UNKNOWN",
          reason "skipped:nxt_bars_unresolved", raw_refs copied from the bar entry;
        - no bar entry for the code -> status UNKNOWN, venue "UNKNOWN", reason "skipped:nxt_bars_missing",
          raw_refs empty.

    Raises:
        ValueError: two bar entries carry the same non-None symbol.
    """
    seen: set[str] = set()
    by_symbol: dict[str, CoverageEntry] = {}
    for bar_entry in nxt_bar_entries:
        symbol = bar_entry.symbol
        if symbol is None:
            continue
        if symbol in seen:
            raise ValueError(f"duplicate_nxt_bar_entry symbol={symbol}")
        seen.add(symbol)
        if symbol not in by_symbol:
            by_symbol[symbol] = bar_entry
    targets: list[str] = []
    skipped: list[CoverageEntry] = []
    for code in cohort_codes:
        found = by_symbol.get(code)
        if found is None:
            skipped.append(
                CoverageEntry(
                    symbol=code,
                    dataset=CaptureDataset.TRADE_TICKS,
                    venue="UNKNOWN",
                    session=INTRADAY_SESSION_NXT_AFTERMARKET,
                    scheduled_at=None,
                    status=CaptureStatus.UNKNOWN,
                    rows=0,
                    first_event_time=None,
                    last_event_time=None,
                    reason="skipped:nxt_bars_missing",
                    raw_refs=(),
                )
            )
            continue
        if found.status in (CaptureStatus.COMPLETE, CaptureStatus.NO_TRADES) and found.venue == "NXT":
            targets.append(code)
            continue
        if found.status == CaptureStatus.NOT_APPLICABLE:
            skipped.append(
                CoverageEntry(
                    symbol=code,
                    dataset=CaptureDataset.TRADE_TICKS,
                    venue=found.venue,
                    session=INTRADAY_SESSION_NXT_AFTERMARKET,
                    scheduled_at=None,
                    status=CaptureStatus.NOT_APPLICABLE,
                    rows=0,
                    first_event_time=None,
                    last_event_time=None,
                    reason="skipped:nxt_bars_not_applicable",
                    raw_refs=tuple(found.raw_refs),
                )
            )
            continue
        skipped.append(
            CoverageEntry(
                symbol=code,
                dataset=CaptureDataset.TRADE_TICKS,
                venue="UNKNOWN",
                session=INTRADAY_SESSION_NXT_AFTERMARKET,
                scheduled_at=None,
                status=CaptureStatus.UNKNOWN,
                rows=0,
                first_event_time=None,
                last_event_time=None,
                reason="skipped:nxt_bars_unresolved",
                raw_refs=tuple(found.raw_refs),
            )
        )
    targets.sort()
    return targets, skipped


class _BatchedPartitionPublisher:
    """Buffer certified per-symbol results and flush them as one partition write."""

    def __init__(
        self, write_fn: Callable[[pd.DataFrame, dict[str, CoverageEntry]], int], max_rows: int
    ) -> None:
        if max_rows <= 0:
            raise ValueError(f"Invalid max_rows: {max_rows!r}")
        self._write_fn = write_fn
        self._max_rows = max_rows
        self._frames: list[pd.DataFrame] = []
        self._coverage: dict[str, CoverageEntry] = {}
        self._buffered_rows = 0

    def add(self, symbol: str, frame: pd.DataFrame, entry: CoverageEntry) -> None:
        """Buffer one write-eligible symbol; flush once buffered rows reach max_rows."""
        if symbol in self._coverage:
            raise ValueError(f"Duplicate symbol buffered: {symbol!r}")
        self._coverage[symbol] = entry
        if frame is not None and len(frame) > 0:
            self._frames.append(frame)
            self._buffered_rows += int(len(frame))
        if self._buffered_rows >= self._max_rows:
            self.flush()

    def flush(self) -> int:
        """Write every buffered symbol in one call; no-op returning 0 when empty."""
        if not self._coverage:
            return 0
        if self._frames:
            combined = pd.concat(self._frames, ignore_index=True)
        else:
            combined = pd.DataFrame()
        written = self._write_fn(combined, dict(self._coverage))
        self._frames = []
        self._coverage = {}
        self._buffered_rows = 0
        return written


@dataclass(frozen=True)
class _ArchiveRun:
    """Per-attempt invocation context shared by every archive stream collector.

    Attributes:
        client: Authenticated KIS data client.
        session: Open HTTP session of that client.
        ls_client: LS client, or None without an LS key.
        kiwoom_client: Kiwoom client, or None without a Kiwoom key.
        snap_date: Archived session date, ISO `YYYY-MM-DD`.
        interval: Bar interval in minutes (> 0).
        profile: Validated acquisition profile.
        store: Capture store receiving raw evidence and manifests.
    """

    client: Any
    session: Any
    ls_client: Any | None
    kiwoom_client: Any | None
    snap_date: str
    interval: int
    profile: CollectionSettings
    store: CaptureStore


_StreamCollector = Callable[[_ArchiveRun, list[str], str, SymbolObserver], Awaitable[object]]
"""(run, target codes, run_id, on_symbol) -> collector result (ignored; rows arrive through on_symbol)."""

_TargetResolver = Callable[[Sequence[str], Sequence[CoverageEntry]], tuple[list[str], list[CoverageEntry]]]
"""(archive cohort, source stream entries) -> (collector targets, pre-resolved skip entries)."""


@dataclass(frozen=True)
class _CodesFrom:
    """Derives a stream's targets from an earlier stream of the same attempt.

    Attributes:
        source: Key of the stream whose entries prove eligibility; must precede the dependent stream in
            _STREAMS and belong to the same phase.
        resolve: Pure resolver returning (targets, skipped entries); skipped entries cover every cohort code
            that is not a target.
    """

    source: str
    resolve: _TargetResolver


@dataclass(frozen=True)
class _StreamSpec:
    """Declarative definition of one evening-archive stream (one task manifest per attempt).

    Attributes:
        key: Stream key used in logs and error messages.
        phase: "regular" or "aftermarket"; phase "all" runs both.
        dataset: MINUTE_BARS (written by write_intraday_partition with the run interval) or TRADE_TICKS
            (written by write_tick_partition).
        session: Intraday session tag of the partition and the manifest.
        collect: Adapter invoking the stream's collector with its exact historical call shape.
        manifest_vendor: Task-manifest vendor label.
        return_slot: Index of run_intraday_archive's return tuple this stream's admitted rows add to, or
            None when the stream is not reported.
        required_from: First ISO session date whose phase completeness requires this manifest; None means
            always required within its phase.
        codes_from: Target derivation from an earlier stream; None means the whole archive cohort.
    """

    key: str
    phase: str
    dataset: CaptureDataset
    session: str
    collect: _StreamCollector
    manifest_vendor: str
    return_slot: int | None
    required_from: str | None
    codes_from: _CodesFrom | None


async def _collect_regular_bars(
    run: _ArchiveRun, codes: list[str], run_id: str, on_symbol: SymbolObserver
) -> object:
    return await collect_intraday_bars(
        run.client,
        run.session,
        codes,
        run.snap_date,
        run.interval,
        ls_client=run.ls_client,
        profile=run.profile,
        capture_store=run.store,
        run_id=run_id,
        on_symbol=on_symbol,
    )


async def _collect_nxt_aftermarket_bars(
    run: _ArchiveRun, codes: list[str], run_id: str, on_symbol: SymbolObserver
) -> object:
    return await collect_nxt_aftermarket_bars(
        run.client,
        run.session,
        codes,
        run.snap_date,
        run.interval,
        kiwoom_client=run.kiwoom_client,
        profile=run.profile,
        capture_store=run.store,
        run_id=run_id,
        on_symbol=on_symbol,
    )


async def _collect_nxt_premarket_bars(
    run: _ArchiveRun, codes: list[str], run_id: str, on_symbol: SymbolObserver
) -> object:
    return await collect_nxt_premarket_bars(
        run.client,
        run.session,
        codes,
        run.snap_date,
        run.interval,
        kiwoom_client=run.kiwoom_client,
        profile=run.profile,
        capture_store=run.store,
        run_id=run_id,
        on_symbol=on_symbol,
    )


async def _collect_krx_aftermarket_bars(
    run: _ArchiveRun, codes: list[str], run_id: str, on_symbol: SymbolObserver
) -> object:
    return await collect_krx_aftermarket_bars(
        run.client,
        run.session,
        codes,
        run.snap_date,
        run.interval,
        profile=run.profile,
        capture_store=run.store,
        run_id=run_id,
        on_symbol=on_symbol,
    )


async def _collect_krx_aftermarket_ticks(
    run: _ArchiveRun, codes: list[str], run_id: str, on_symbol: SymbolObserver
) -> object:
    return await collect_aftermarket_trade_ticks(
        run.kiwoom_client,
        run.session,
        codes,
        run.snap_date,
        venue="KRX",
        profile=run.profile,
        capture_store=run.store,
        run_id=run_id,
        on_symbol=on_symbol,
    )


async def _collect_nxt_aftermarket_ticks(
    run: _ArchiveRun, codes: list[str], run_id: str, on_symbol: SymbolObserver
) -> object:
    return await collect_aftermarket_trade_ticks(
        run.kiwoom_client,
        run.session,
        codes,
        run.snap_date,
        venue="NXT",
        profile=run.profile,
        capture_store=run.store,
        run_id=run_id,
        on_symbol=on_symbol,
    )


async def _collect_regular_ticks(
    run: _ArchiveRun, codes: list[str], run_id: str, on_symbol: SymbolObserver
) -> object:
    return await collect_intraday_trade_ticks(
        run.client,
        run.session,
        codes,
        run.snap_date,
        ls_client=run.ls_client,
        kiwoom_client=run.kiwoom_client,
        profile=run.profile,
        capture_store=run.store,
        run_id=run_id,
        on_symbol=on_symbol,
    )


_STREAMS: tuple[_StreamSpec, ...] = (
    _StreamSpec(
        key="bars",
        phase="regular",
        dataset=CaptureDataset.MINUTE_BARS,
        session=INTRADAY_SESSION_REGULAR,
        collect=_collect_regular_bars,
        manifest_vendor="owner-local",
        return_slot=0,
        required_from=None,
        codes_from=None,
    ),
    _StreamSpec(
        key="nxt_after",
        phase="aftermarket",
        dataset=CaptureDataset.MINUTE_BARS,
        session=INTRADAY_SESSION_NXT_AFTERMARKET,
        collect=_collect_nxt_aftermarket_bars,
        manifest_vendor="owner-local",
        return_slot=1,
        required_from=None,
        codes_from=None,
    ),
    _StreamSpec(
        key="nxt_pre",
        phase="aftermarket",
        dataset=CaptureDataset.MINUTE_BARS,
        session=INTRADAY_SESSION_NXT_PREMARKET,
        collect=_collect_nxt_premarket_bars,
        manifest_vendor="owner-local",
        return_slot=1,
        required_from=None,
        codes_from=None,
    ),
    _StreamSpec(
        key="krx_after",
        phase="aftermarket",
        dataset=CaptureDataset.MINUTE_BARS,
        session=INTRADAY_SESSION_KRX_AFTERMARKET,
        collect=_collect_krx_aftermarket_bars,
        manifest_vendor="owner-local",
        return_slot=None,
        required_from=KRX_AFTERMARKET_START_DATE,
        codes_from=None,
    ),
    _StreamSpec(
        key="krx_after_ticks",
        phase="aftermarket",
        dataset=CaptureDataset.TRADE_TICKS,
        session=INTRADAY_SESSION_KRX_AFTERMARKET,
        collect=_collect_krx_aftermarket_ticks,
        manifest_vendor="owner-local",
        return_slot=None,
        required_from=AFTERMARKET_TICKS_START_DATE,
        codes_from=None,
    ),
    _StreamSpec(
        key="nxt_after_ticks",
        phase="aftermarket",
        dataset=CaptureDataset.TRADE_TICKS,
        session=INTRADAY_SESSION_NXT_AFTERMARKET,
        collect=_collect_nxt_aftermarket_ticks,
        manifest_vendor="owner-local",
        return_slot=None,
        required_from=AFTERMARKET_TICKS_START_DATE,
        codes_from=_CodesFrom("nxt_after", _resolve_nxt_tick_targets),
    ),
    _StreamSpec(
        key="ticks",
        phase="regular",
        dataset=CaptureDataset.TRADE_TICKS,
        session=INTRADAY_SESSION_REGULAR,
        collect=_collect_regular_ticks,
        manifest_vendor="owner-local",
        return_slot=2,
        required_from=None,
        codes_from=None,
    ),
)
"""Evening-archive streams in execution and manifest-publication order.

Order is load-bearing (D7): with phase "all" regular ticks run after the aftermarket block, and the NXT tick
stream must follow the NXT aftermarket bars stream it derives targets from.
"""


def _stream_run_id(snap_date: str, spec: _StreamSpec, attempt: str) -> str:
    """Build the attempt-unique task run_id of one stream.

    The slug is derived as "<session with '_' -> '-'>-<bars|ticks>" so that, followed by "-<attempt>", no
    run_id of one attempt is a string prefix of another (the previous hand-written slugs made
    "nxt-aftermarket" a prefix of "nxt-aftermarket-ticks").

    Returns:
        "archive-<snap_date>-<slug>-<attempt>".
    """
    suffix = "bars" if spec.dataset is CaptureDataset.MINUTE_BARS else "ticks"
    slug = f"{str(spec.session).replace('_', '-')}-{suffix}"
    return f"archive-{snap_date}-{slug}-{attempt}"


def run_intraday_archive(snapshot_date: str | None = None, bar_interval_minutes: int = DEFAULT_BAR_INTERVAL_MINUTES, *, profile: CollectionSettings | None = None, phase: str = "all") -> tuple[int, int, int]:
    """Archive the project's dated candidate cohort independently of other collectors.

    Every stream in _STREAMS whose phase is selected runs in table order: its collector delivers per-symbol
    results, certified rows are buffered and flushed to the stream's partition right after the collector
    returns, and after all selected streams finished one evening-archive task manifest per stream is
    published in table order. A manifest is COMPLETE only when the whole archive cohort is covered by GOOD
    entries.

    Args:
        snapshot_date: Exact trading date, default current Asia/Seoul date.
        bar_interval_minutes: Bar interval in minutes (> 0).
        profile: Validated bounded acquisition profile; None loads CollectionSettings() from the environment.
        phase: "regular" archives regular-session 1m bars and trade ticks (settled at the 15:30 close).
            "aftermarket" archives NXT premarket/aftermarket bars, KRX aftermarket bars, and KRX/NXT
            aftermarket trade ticks (sessions close at 20:00; aftermarket tick tapes are current-day only).
            "all" runs both, regular bars first and regular ticks last.

    Returns:
        (regular-bar rows, NXT premarket+aftermarket bar rows, regular-tick rows) admitted for publication by
        this call, i.e. summed frame lengths of COMPLETE/NO_TRADES results. Partition totals may differ
        because writers merge with existing partitions. KRX aftermarket bars and aftermarket ticks are
        archived but not counted. A count is 0 for every stream the phase did not run.

    Raises:
        FileNotFoundError: Expected owner-local cohort evidence is absent.
        ValueError: Invalid date, certification, profile, unrecognized phase, or a collector delivered rows for
            a result that is not COMPLETE/NO_TRADES (no task manifest of the run is published).
        OSError: Acquisition evidence or verified publication fails.
    """
    prof = profile if profile is not None else CollectionSettings()
    snap_date = snapshot_date or datetime.now(SEOUL).date().isoformat()
    try:
        trading_day = date.fromisoformat(str(snap_date))
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snap_date!r}") from None
    if int(bar_interval_minutes) <= 0:
        raise ValueError(f"Invalid bar_interval_minutes: {bar_interval_minutes!r}")
    _validate_phase(phase)
    session_day = resolve_session_day(trading_day)
    if session_day.kind is SessionKind.CLOSED:
        record_run_outcome(
            "archive_intraday",
            RUN_OUTCOME_SKIPPED,
            run_date=str(snap_date),
            reason="non_trading_day",
            metrics={"session": "CLOSED"},
        )
        logger.info("[DATA] stage=intraday_archive status=SKIP reason=non_trading_day date=%s", snap_date)
        return (0, 0, 0)
    store = CaptureStore(_capture_root(prof))
    active_phases = {"regular", "aftermarket"} if phase == "all" else {phase}

    async def _run() -> tuple[int, int, int]:
        client = KisApiClient(**kis_data_client_kwargs())
        ls_client = LsApiClient() if settings.LS_APP_KEY else None
        kiwoom_client = KiwoomApiClient() if settings.KIWOOM_APP_KEY else None
        async with client.create_session() as session:
            await client.ensure_token(session)
            if not await is_kis_trading_day(client, session, str(snap_date)):
                record_run_outcome(
                    "archive_intraday",
                    RUN_OUTCOME_SKIPPED,
                    run_date=str(snap_date),
                    reason="non_trading_day",
                    metrics={"session": "CLOSED"},
                )
                logger.info("[DATA] stage=intraday_archive status=SKIP reason=non_trading_day date=%s", snap_date)
                return (0, 0, 0)
            # 휴장일엔 collect가 코호트를 발행하지 않으므로, 코호트 조회는 거래일 판정 뒤에 해야 오탐 실패가 없다.
            prev_trading_day = await _resolve_previous_trading_day(client, session, str(snap_date))
            held = load_held_roster()
            codes, prev_incomplete = _resolve_cohort_codes(str(snap_date), prof, store, previous_trading_day=prev_trading_day, held=held)
            interval = int(bar_interval_minutes)
            batch_rows = int(prof.COLLECTION_ARROW_BATCH_ROWS)
            max_rows = int(prof.COLLECTION_ARCHIVE_PUBLISH_ROWS)
            archive_run = _ArchiveRun(
                client=client,
                session=session,
                ls_client=ls_client,
                kiwoom_client=kiwoom_client,
                snap_date=str(snap_date),
                interval=interval,
                profile=prof,
                store=store,
            )
            # Same-day retry must not collide with prior attempt's immutable manifests.
            attempt = uuid.uuid4().hex[:8]
            active_specs = [spec for spec in _STREAMS if spec.phase in active_phases]
            entries_by_key: dict[str, list[CoverageEntry]] = {}
            publishers: dict[str, _BatchedPartitionPublisher] = {}
            counts: dict[str, int] = {}
            run_ids: dict[str, str] = {}
            targets_by_key: dict[str, list[str]] = {}
            for spec in active_specs:
                stream_run_id = _stream_run_id(str(snap_date), spec, attempt)
                run_ids[spec.key] = stream_run_id
                if spec.codes_from is None:
                    targets = list(codes)
                    entries: list[CoverageEntry] = []
                else:
                    source_entries = entries_by_key[spec.codes_from.source]
                    targets, skipped = spec.codes_from.resolve(codes, source_entries)
                    entries = list(skipped)
                entries_by_key[spec.key] = entries
                targets_by_key[spec.key] = list(targets)
                counts[spec.key] = 0
                if spec.dataset is CaptureDataset.MINUTE_BARS:
                    def _write_bars(df: pd.DataFrame, coverage: dict[str, CoverageEntry], _interval: int = interval, _date: str = str(snap_date), _session: str = spec.session, _batch: int = batch_rows) -> int:
                        return write_intraday_partition(
                            df, _interval, _date, _session, coverage=coverage, batch_rows=_batch,
                        )

                    publishers[spec.key] = _BatchedPartitionPublisher(
                        _write_bars,
                        max_rows,
                    )
                else:
                    def _write_ticks(df: pd.DataFrame, coverage: dict[str, CoverageEntry], _date: str = str(snap_date), _session: str = spec.session, _batch: int = batch_rows) -> int:
                        return write_tick_partition(
                            df, _date, _session, coverage=coverage, batch_rows=_batch,
                        )

                    publishers[spec.key] = _BatchedPartitionPublisher(
                        _write_ticks,
                        max_rows,
                    )

                def _make_observer(
                    _key: str = spec.key,
                    _entries: list[CoverageEntry] = entries,
                    _publisher_key: str = spec.key,
                ) -> Callable[[str, pd.DataFrame, CoverageEntry], None]:
                    def _on_symbol(symbol: str, frame: pd.DataFrame, entry: CoverageEntry) -> None:
                        _entries.append(entry)
                        if _admit_for_write(_key, symbol, frame, entry):
                            publishers[_publisher_key].add(symbol, frame, entry)
                            counts[_publisher_key] += 0 if frame is None else len(frame)

                    return _on_symbol

                on_symbol = _make_observer()
                await spec.collect(archive_run, list(targets), stream_run_id, on_symbol)
                publishers[spec.key].flush()
            for spec in active_specs:
                manifest = _publish_task_manifest(
                    store,
                    trading_day=trading_day,
                    run_id=run_ids[spec.key],
                    dataset=spec.dataset,
                    vendor=spec.manifest_vendor,
                    session=spec.session,
                    entries=entries_by_key[spec.key],
                    expected_symbols=codes,
                    roster_incomplete=not held.ok,
                )
                status = manifest.status.value if isinstance(manifest.status, CaptureStatus) else str(manifest.status)
                logger.info(
                    "[DATA] stage=intraday_archive_stream stream=%s date=%s status=%s targets=%d entries=%d rows=%d",
                    spec.key,
                    snap_date,
                    status,
                    len(targets_by_key[spec.key]),
                    len(entries_by_key[spec.key]),
                    counts[spec.key],
                )
            collected_entries: list[CoverageEntry] = []
            for spec in active_specs:
                collected_entries.extend(entries_by_key[spec.key])
            incomplete = prev_incomplete or any(
                item.status not in GOOD_ENTRY_STATES
                for item in collected_entries
            )
            if incomplete:
                logger.warning("[DATA] stage=intraday_archive status=DEGRADED date=%s", snap_date)
            slots = [0, 0, 0]
            for spec in active_specs:
                if spec.return_slot is not None:
                    slots[spec.return_slot] += counts[spec.key]
            return (slots[0], slots[1], slots[2])

    n_bars, n_nxt, n_ticks = asyncio.run(_run())
    if session_day.kind is SessionKind.SHIFTED:
        record_run_outcome(
            "archive_intraday", RUN_OUTCOME_DEGRADED, run_date=str(snap_date), reason="shifted_session_standard_window"
        )
    return (n_bars, n_nxt, n_ticks)


def main() -> None:
    import argparse

    from src.utils.display import Colors

    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    parser = argparse.ArgumentParser(description="Intraday archive session split")
    parser.add_argument("--phase", choices=["regular", "aftermarket", "all"], default="all")
    parser.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default today)")
    args = parser.parse_args()
    profile = CollectionSettings()
    target_date = args.date or resolve_archive_target_date(datetime.now(SEOUL), args.phase)
    today_str = datetime.now(SEOUL).date().isoformat()
    catchup = target_date != today_str
    logger.info("[DATA] stage=intraday_archive target_date=%s phase=%s catchup=%s", target_date, args.phase, catchup)
    store = CaptureStore(_capture_root(profile))
    if archive_phase_complete(store, target_date, args.phase):
        logger.info("[DATA] stage=intraday_archive status=SKIP reason=already_archived date=%s phase=%s", target_date, args.phase)
        return
    effective_phase = args.phase
    if args.phase in ("aftermarket", "all") and target_date < today_str:
        record_run_outcome("archive_intraday", RUN_OUTCOME_DEGRADED, run_date=target_date, reason="aftermarket_not_replayable")
        if args.phase == "aftermarket":
            logger.info(
                "🚀 [Intraday 아카이브 시작] 대상일: %s, 저장소: %s, phase=%s",
                target_date,
                settings.HISTORY_DIR,
                args.phase,
            )
            logger.info("[DATA] stage=intraday_archive status=SKIP reason=aftermarket_not_replayable date=%s", target_date)
            return
        if archive_phase_complete(store, target_date, "regular"):
            logger.info("[DATA] stage=intraday_archive status=SKIP reason=already_archived date=%s phase=%s", target_date, "regular")
            return
        effective_phase = "regular"
    logger.info(
        "🚀 [Intraday 아카이브 시작] 대상일: %s, 저장소: %s, phase=%s",
        target_date,
        settings.HISTORY_DIR,
        args.phase,
    )
    try:
        bars_rows, nxt_rows, tick_rows = run_intraday_archive(snapshot_date=target_date, profile=profile, phase=effective_phase)
    except ValueError as e:
        logger.error("[DATA] stage=intraday_archive status=ERROR reason=%s", e)
        raise SystemExit(2) from e
    except OSError as e:
        logger.error("[DATA] stage=intraday_archive status=ERROR reason=%s", e)
        raise SystemExit(1) from e

    box_top = "━" * 60
    divider = "─" * 60
    logger.info(f"\n{Colors.BOLD}{box_top}{Colors.RESET}")
    logger.info(f" {Colors.GREEN}{Colors.BOLD}📦 [Intraday 분봉/틱 아카이브 완료]{Colors.RESET} (기준일: {target_date}, phase={args.phase})")
    logger.info(f"{Colors.BOLD}{divider}{Colors.RESET}")
    logger.info(f"   • 정규 세션   : {Colors.GREEN}{bars_rows:>5,}{Colors.RESET} 행 (1분봉)")
    logger.info(f"   • NXT 세션    : {Colors.GREEN}{nxt_rows:>5,}{Colors.RESET} 행 (프리/애프터마켓)")
    logger.info(f"   • 체결 틱     : {Colors.GREEN}{tick_rows:>5,}{Colors.RESET} 행 (정규장 틱 데이터)")
    logger.info(f"   • 저장 경로   : {settings.HISTORY_DIR}")
    logger.info(f"{Colors.BOLD}{box_top}{Colors.RESET}")


if __name__ == "__main__":
    main()
