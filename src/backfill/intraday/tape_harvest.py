"""Certified Kiwoom tape-day publication into session tick partitions."""

from __future__ import annotations

import hashlib
import uuid
import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from src.backfill.intraday.collector import (
    _call_with_transport_retry,
    _safe_normalize_ticks,
    _split_session_window,
    _stage_fragment,
)
from src.config.collection import CollectionSettings
from src.config.market_session import (
    ARCHIVE_AFTERMARKET_READY_HHMMSS,
    ARCHIVE_REGULAR_READY_HHMMSS,
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_REGULAR,
    KRX_AFTERMARKET_HOUR_CEIL,
    KRX_AFTERMARKET_HOUR_FLOOR,
    KRX_REGULAR_HOUR_CEIL,
    KRX_REGULAR_HOUR_FLOOR,
    NXT_AFTERMARKET_HOUR_CEIL,
    NXT_AFTERMARKET_HOUR_FLOOR,
)
from src.data.capture_contracts import (
    GOOD_ENTRY_STATES,
    ArtifactRef,
    BrokerPayload,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureManifest,
    CaptureStatus,
    ChartBudget,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.intraday_store import tick_partition_path, write_tick_partition

logger = logging.getLogger(__name__)

_SEOUL = ZoneInfo("Asia/Seoul")


@dataclass(frozen=True)
class TapeSession:
    """Stored tick session carved out of one Kiwoom tape."""

    session: str
    venue: Literal["KRX", "NXT"]
    floor: str
    ceil: str


TAPE_SESSIONS: tuple[TapeSession, ...] = (
    TapeSession(session=INTRADAY_SESSION_REGULAR, venue="KRX", floor=KRX_REGULAR_HOUR_FLOOR, ceil=KRX_REGULAR_HOUR_CEIL),
    TapeSession(session=INTRADAY_SESSION_KRX_AFTERMARKET, venue="KRX", floor=KRX_AFTERMARKET_HOUR_FLOOR, ceil=KRX_AFTERMARKET_HOUR_CEIL),
    TapeSession(session=INTRADAY_SESSION_NXT_AFTERMARKET, venue="NXT", floor=NXT_AFTERMARKET_HOUR_FLOOR, ceil=NXT_AFTERMARKET_HOUR_CEIL),
)


@dataclass(frozen=True)
class TapeDayResult:
    """Settled per-symbol result for one (day, session)."""

    symbol: str
    day: str
    session: str
    frame: pd.DataFrame
    entry: CoverageEntry


@dataclass(frozen=True)
class TapeWalkOutcome:
    """Outcome of one symbol tape walk."""

    termination_reason: str
    pages_fetched: int
    unresolved_days: tuple[str, ...]
    skipped_unclosed: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class FlushReport:
    """Outcome of one publisher flush."""

    partitions_written: int
    rows_written: int
    entries_by_status: dict[str, int]


def _tape_endpoint(venue: str) -> str:
    return "ka10079" if venue == "KRX" else "ka10079-nx"


def _verified_venue(venue: str, profile: CollectionSettings) -> str:
    routes = profile.COLLECTION_VERIFIED_CHART_ROUTES or {}
    return str(routes.get(f"kiwoom:{_tape_endpoint(venue)}", "UNKNOWN"))


def _parse_days(days: Collection[str]) -> list[str]:
    unique = sorted({str(item) for item in days})
    if not unique:
        raise ValueError("days must be nonempty")
    for item in unique:
        try:
            date.fromisoformat(item)
        except ValueError:
            raise ValueError(f"invalid day: {item!r}") from None
    return unique


def is_session_closed(day: str, session: str, now: datetime) -> bool:
    """True when a (day, session) is certifiably closed at `now`.

    Regular closes at ARCHIVE_REGULAR_READY_HHMMSS on the day itself;
    any other session closes at ARCHIVE_AFTERMARKET_READY_HHMMSS on the day itself.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    kst = now.astimezone(_SEOUL)
    today = kst.date().isoformat()
    if day > today:
        return False
    if session == INTRADAY_SESSION_REGULAR:
        return day < today or (day == today and kst.strftime("%H%M%S") >= ARCHIVE_REGULAR_READY_HHMMSS)
    return day < today or (day == today and kst.strftime("%H%M%S") >= ARCHIVE_AFTERMARKET_READY_HHMMSS)


def _reject_unclosed_day(day: str, session: str, symbol: str = "") -> None:
    now = datetime.now(_SEOUL)
    if is_session_closed(day, session, now):
        return
    if session == INTRADAY_SESSION_REGULAR:
        raise ValueError(f"regular session not closed for day: {day!r} symbol={symbol!r} session={session!r}")
    raise ValueError(f"aftermarket session not closed for day: {day!r} symbol={symbol!r} session={session!r}")


def _settle_certified_day(
    *,
    code: str,
    day: str,
    rows: list[dict[str, Any]],
    vendor_total: int | None,
    basis: str,
    sessions: Sequence[TapeSession],
    venue: str,
    store: CaptureStore,
    run_id: str,
    refs: list[Any],
    on_result: Callable[[TapeDayResult], None],
) -> None:
    ymd = day.replace("-", "")
    if basis == "vendor_total":
        proof = "tape_complete"
    elif basis == "tape_empty":
        proof = "tape_empty"
    else:
        proof = "tape_bracketed"
    in_any: set[int] = set()
    windows: dict[str, list[dict[str, Any]]] = {}
    for spec in sessions:
        kept, _ = _split_session_window(rows, ymd, spec.floor, spec.ceil, "kiwoom")
        windows[spec.session] = kept
        in_any.update(id(r) for r in kept)
    stray = [r for r in rows if id(r) not in in_any]
    day_refs = list(refs)
    if stray:
        frag = _stage_fragment(
            store,
            run_id=run_id,
            trading_day=date.fromisoformat(day),
            dataset=CaptureDataset.TRADE_TICKS,
            symbol=code,
            session=sessions[0].session,
            frame=_safe_normalize_ticks("kiwoom", stray, day, code, truncated=False),
            reason="out_of_window",
        )
        day_refs.append(frag)
    for spec in sessions:
        kept = windows[spec.session]
        frame = _safe_normalize_ticks("kiwoom", kept, day, code, truncated=False)
        if venue == "UNKNOWN":
            entry = CoverageEntry(
                symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue=venue, session=spec.session,
                scheduled_at=None, status=CaptureStatus.UNKNOWN, rows=0,
                first_event_time=None, last_event_time=None,
                reason="uncertified_venue", raw_refs=tuple(day_refs),
            )
        elif len(frame) > 0:
            entry = CoverageEntry(
                symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue=venue, session=spec.session,
                scheduled_at=None, status=CaptureStatus.COMPLETE, rows=len(frame),
                first_event_time=None, last_event_time=None,
                reason=f"{proof}:{spec.session}={len(kept)}:vendor_total={vendor_total}",
                raw_refs=tuple(day_refs),
            )
        else:
            entry = CoverageEntry(
                symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue=venue, session=spec.session,
                scheduled_at=None, status=CaptureStatus.NO_TRADES, rows=0,
                first_event_time=None, last_event_time=None,
                reason=f"{proof}:{spec.session}=0:vendor_total={vendor_total}",
                raw_refs=tuple(day_refs),
            )
        on_result(TapeDayResult(symbol=code, day=day, session=spec.session, frame=frame, entry=entry))


def _settle_uncertified_day(
    *,
    code: str,
    day: str,
    status: CaptureStatus,
    reason: str,
    sessions: Sequence[TapeSession],
    venue: str,
    refs: list[Any],
    on_result: Callable[[TapeDayResult], None],
) -> None:
    from src.data.intraday_schema import CANONICAL_TICK_COLUMNS

    for spec in sessions:
        frame = pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_TICK_COLUMNS})
        entry = CoverageEntry(
            symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue=venue, session=spec.session,
            scheduled_at=None, status=status, rows=0,
            first_event_time=None, last_event_time=None,
            reason=reason, raw_refs=tuple(refs),
        )
        on_result(TapeDayResult(symbol=code, day=day, session=spec.session, frame=frame, entry=entry))


async def harvest_symbol_tape(
    client: Any,
    http_session: Any,
    code: str,
    days: Collection[str],
    *,
    venue: Literal["KRX", "NXT"] = "KRX",
    sessions: Sequence[TapeSession],
    store: CaptureStore,
    run_id: str,
    profile: CollectionSettings,
    on_result: Callable[[TapeDayResult], None],
    walk_deadline: datetime | None = None,
    needed: Collection[tuple[str, str]] | None = None,
) -> TapeWalkOutcome:
    """Walk one tape for one symbol and emit a settled result per needed (day, session).

    Args:
        client: Kiwoom client exposing walk_tick_tape.
        http_session: Existing HTTP session.
        code: Symbol.
        days: Market dates that must be settled (YYYY-MM-DD); the walk stops once the oldest is passed.
            When `needed` is given, the wanted days are derived from it.
        venue: Tape to walk; sessions of the other venue are ignored.
        sessions: Stored sessions to carve (subset of TAPE_SESSIONS matching venue).
        store: Evidence store; every page is persisted via the shared page observer under distinct attempt slots.
        run_id: Acquisition identity (`tape-<run date>-<seq>`).
        profile: Page guard (COLLECTION_TAPE_MAX_PAGES), routes, flush bounds.
        on_result: Receives each settled result as its date completes (newest first).
        walk_deadline: Optional aware instant after which the walk stops requesting pages (used to keep a long walk
            out of reserved vendor slots); days not reached are reported PARTIAL `walk_incomplete`, never not-on-tape.
        needed: Exact (day, session) keys to settle; None keeps the old
            days-x-sessions semantics restricted to CLOSED pairs. A needed pair
            that is not closed is never emitted, settled or ledgered; it is
            reported in `skipped_unclosed`.

    Returns:
        TapeWalkOutcome(termination_reason, pages_fetched, unresolved_days, skipped_unclosed);
        `unresolved_days` covers closed pairs only.
    """
    if venue not in ("KRX", "NXT"):
        raise ValueError(f"unknown tape venue: {venue!r}")
    if not str(run_id).strip():
        raise ValueError("run_id must be nonempty")
    matched = [s for s in sessions if s.venue == venue]
    closed_now = datetime.now(_SEOUL)
    matched_names = {s.session for s in matched}
    if needed is None:
        wanted = _parse_days(days)
        full = {(day, spec.session) for day in wanted for spec in matched}
        needed_set = {(day, session) for day, session in full if is_session_closed(day, session, closed_now)}
        skipped_unclosed = tuple(sorted(full - needed_set))
    else:
        normalized = sorted({(str(day), str(session)) for day, session in needed})
        for day, _session in normalized:
            try:
                date.fromisoformat(day)
            except ValueError:
                raise ValueError(f"invalid needed day: {day!r}") from None
        matched_needed = [(day, session) for day, session in normalized if session in matched_names]
        needed_set = {
            (day, session) for day, session in matched_needed if is_session_closed(day, session, closed_now)
        }
        skipped_unclosed = tuple(sorted(set(matched_needed) - needed_set))
        wanted = sorted({day for day, _session in normalized})
    if not needed_set:
        return TapeWalkOutcome(
            termination_reason="tape_end", pages_fetched=0, unresolved_days=(),
            skipped_unclosed=skipped_unclosed,
        )
    oldest = min(wanted)
    endpoint = _tape_endpoint(venue)
    resolved_venue = _verified_venue(venue, profile)
    context_day = date.fromisoformat(oldest)
    budget = ChartBudget(
        max_pages=int(profile.COLLECTION_TAPE_MAX_PAGES),
        deadline=walk_deadline,
        request_timeout_seconds=float(profile.COLLECTION_REQUEST_TIMEOUT_SECONDS),
    )
    wanted_set = set(wanted)
    page_refs: list[tuple[Any, frozenset[str]]] = []

    def _observer_for(slot: int) -> Any:
        from src.data.capture_contracts import CaptureStatus as _Status

        def _on_page(
            payload: BrokerPayload | None,
            metadata: Mapping[str, str],
            started: datetime,
            received: datetime,
            page_index: int,
            _retry: int,
        ) -> None:
            page_days: frozenset[str] = frozenset()
            if payload is not None:
                stamps = (str(r.get("cntr_tm", ""))[:8] for r in (payload.get("stk_tic_chart_qry") or []) if isinstance(r, Mapping))
                page_days = frozenset(f"{x[:4]}-{x[4:6]}-{x[6:8]}" for x in stamps if len(x) == 8 and x.isdigit())
                if not page_days & wanted_set and not (page_index == 0 and not page_days):
                    return
            context = CaptureContext(
                trading_date=context_day, run_id=str(run_id), dataset=CaptureDataset.TRADE_TICKS,
                vendor="kiwoom", endpoint=endpoint, symbol=str(code),
                venue=resolved_venue, session=matched[0].session,
                capture_reason="intraday-trade_ticks", cohort_id=None, scheduled_at=None,
            )
            response = CapturedResponse(
                context=context, request_started_at=started, received_at=received,
                payload=dict(payload) if payload is not None else None,
                status=_Status.COMPLETE if payload is not None else _Status.FAILED,
                source_timestamp=None, source_published_at=None,
                page_index=int(page_index), attempt_index=int(slot),
                continuation={str(k): str(v) for k, v in dict(metadata).items()},
                error_type=None,
            )
            try:
                ref = store.append_response(response)
            except ValueError:
                existing_rel = CaptureStore._raw_rel(response)
                data = (store.root / existing_rel).read_bytes()
                ref = ArtifactRef(
                    path=existing_rel, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), rows=None,
                )
            page_refs.append((ref, page_days))

        return _on_page

    attempt_buffer: dict[str, list[tuple[str, list[dict[str, Any]], Any]]] = {}

    async def _invoke(slot: int) -> Any:
        buffered: list[tuple[str, list[dict[str, Any]], Any]] = []

        def _on_day(day: str, rows: list[dict[str, Any]], cert: Any) -> None:
            if day in wanted_set:
                buffered.append((day, [dict(r) for r in rows], cert))

        payload = await client.walk_tick_tape(
            http_session, str(code), venue=venue, stop_before_day=oldest,
            budget=budget, on_page=_observer_for(slot), on_day_complete=_on_day,
        )
        attempt_buffer["latest"] = buffered
        return payload

    def _refs_for(day: str) -> list[Any]:
        return [ref for ref, days in page_refs if day in days]

    payload, _ = await _call_with_transport_retry(_invoke, first_attempt=0, profile=profile, code=str(code))
    cert_by_day: dict[str, Any] = {str(c.day): c for c in (payload.get("certificates") or []) if hasattr(c, "day")}
    settled: set[str] = set()

    def _day_sessions(day: str) -> list[TapeSession]:
        return [spec for spec in matched if (day, spec.session) in needed_set]

    for day, rows, cert in attempt_buffer.get("latest", []):
        day_sessions = _day_sessions(day)
        if not day_sessions:
            continue
        _settle_certified_day(
            code=str(code), day=day, rows=rows, vendor_total=getattr(cert, "vendor_total", None),
            basis=str(getattr(cert, "basis", "vendor_total")), sessions=day_sessions, venue=resolved_venue,
            store=store, run_id=str(run_id), refs=_refs_for(day), on_result=on_result,
        )
        settled.add(day)
    empty_refs = [ref for ref, days in page_refs if not days]
    today = datetime.now(_SEOUL).date()
    lookback = int(profile.COLLECTION_TAPE_LOOKBACK_DAYS)
    for day in sorted(wanted_set - settled, reverse=True):
        day_sessions = _day_sessions(day)
        if not day_sessions:
            continue
        cert = cert_by_day.get(day)
        if cert is not None:
            _settle_uncertified_day(
                code=str(code), day=day, status=CaptureStatus.PARTIAL,
                reason=f"tape_total_mismatch:received={cert.received}:total={cert.vendor_total}",
                sessions=day_sessions, venue=resolved_venue, refs=_refs_for(day), on_result=on_result,
            )
        elif str(payload.get("termination_reason", "")) == "tape_empty":
            if day == today.isoformat() and any(spec.session != INTRADAY_SESSION_REGULAR for spec in day_sessions):
                _settle_uncertified_day(
                    code=str(code), day=day, status=CaptureStatus.PARTIAL,
                    reason="tape_total_mismatch:received=0:total=None",
                    sessions=[spec for spec in day_sessions if spec.session != INTRADAY_SESSION_REGULAR],
                    venue=resolved_venue, refs=empty_refs, on_result=on_result,
                )
                regular_sessions = [spec for spec in day_sessions if spec.session == INTRADAY_SESSION_REGULAR]
                if regular_sessions:
                    _settle_certified_day(
                        code=str(code), day=day, rows=[], vendor_total=None, basis="tape_empty",
                        sessions=regular_sessions, venue=resolved_venue, store=store, run_id=str(run_id),
                        refs=empty_refs, on_result=on_result,
                    )
            elif resolved_venue == "UNKNOWN":
                _settle_certified_day(
                    code=str(code), day=day, rows=[], vendor_total=None, basis="tape_empty",
                    sessions=day_sessions, venue=resolved_venue, store=store, run_id=str(run_id),
                    refs=[*_refs_for(day), *[r for r in empty_refs if r not in _refs_for(day)]],
                    on_result=on_result,
                )
            elif (today - date.fromisoformat(day)).days <= lookback:
                _settle_certified_day(
                    code=str(code), day=day, rows=[], vendor_total=None, basis="tape_empty",
                    sessions=day_sessions, venue=resolved_venue, store=store, run_id=str(run_id),
                    refs=[*_refs_for(day), *[r for r in empty_refs if r not in _refs_for(day)]],
                    on_result=on_result,
                )
                settled.add(day)
            else:
                _settle_uncertified_day(
                    code=str(code), day=day, status=CaptureStatus.UNKNOWN, reason="day_not_on_tape",
                    sessions=day_sessions, venue=resolved_venue, refs=[], on_result=on_result,
                )
        elif str(payload.get("termination_reason", "")) in ("tape_end", "crossed_stop_day"):
            _settle_uncertified_day(
                code=str(code), day=day, status=CaptureStatus.UNKNOWN, reason="day_not_on_tape",
                sessions=day_sessions, venue=resolved_venue, refs=[], on_result=on_result,
            )
        else:
            _settle_uncertified_day(
                code=str(code), day=day, status=CaptureStatus.PARTIAL,
                reason=f"walk_incomplete:{payload.get('termination_reason', '')}",
                sessions=day_sessions, venue=resolved_venue, refs=[], on_result=on_result,
            )
    unresolved = tuple(sorted(d for d in wanted if d not in settled and _day_sessions(d)))
    return TapeWalkOutcome(
        termination_reason=str(payload.get("termination_reason", "")),
        pages_fetched=int(payload.get("pages_fetched", 0)),
        unresolved_days=unresolved,
        skipped_unclosed=skipped_unclosed,
    )


class TickTapePublisher:
    """Buffers settled results and publishes them in row-bounded batches.

    Publishing writes one partition per (day, session) per flush through write_tick_partition with the certified
    entries of that flush, then one manifest per (day, session, flush).

    Group manifests are immutable per run_id, and the flush sequence restarts at zero in every process, so each
    publisher carries a unique namespace in the run_id; otherwise a restarted backfill collides with the manifests
    an earlier process already published for the same (day, session, seq) and its evidence is silently dropped.
    """

    def __init__(
        self, *, store: CaptureStore, profile: CollectionSettings, flush_rows: int, auto_flush: bool = True,
        publish_namespace: str | None = None,
    ) -> None:
        """Bind the evidence store, profile, and row bound.

        Args:
            store: Evidence store owning manifests.
            profile: Supplies Arrow batch bounds.
            flush_rows: Buffered-row bound triggering an automatic flush.
            auto_flush: False lets the caller own flush timing (ledger commits and disk checks around each flush).
            publish_namespace: Token distinguishing this publisher's manifests from other processes'; defaults to a
                fresh UTC-timestamp plus random suffix.

        Raises:
            ValueError: Nonpositive flush bound or empty namespace.
        """
        if int(flush_rows) <= 0:
            raise ValueError(f"invalid flush_rows: {flush_rows!r}")
        self._store = store
        self._profile = profile
        self._flush_rows = int(flush_rows)
        self._auto_flush = bool(auto_flush)
        self._buffer: list[TapeDayResult] = []
        self._buffered_rows = 0
        self._flush_seq = 0
        namespace = (
            publish_namespace
            if publish_namespace is not None
            else f"{datetime.now(_SEOUL):%Y%m%dT%H%M%S}{uuid.uuid4().hex[:6]}"
        )
        if not str(namespace).strip():
            raise ValueError("publish_namespace must be nonempty")
        self._namespace = str(namespace)

    def _guard(self, result: TapeDayResult) -> None:
        _reject_unclosed_day(result.day, result.session, result.symbol)

    def add(self, result: TapeDayResult) -> None:
        """Buffer one settled result, auto-flushing once the row bound fills."""
        self._guard(result)
        self._buffer.append(result)
        self._buffered_rows += len(result.frame)
        if self._auto_flush and self.should_flush():
            self.flush()

    def should_flush(self) -> bool:
        """True when the buffered rows reached the flush bound."""
        return self._buffered_rows >= self._flush_rows

    def flush(self) -> FlushReport:
        """Write every buffered (day, session) group and its manifest."""
        groups: dict[tuple[str, str], list[TapeDayResult]] = {}
        for result in self._buffer:
            groups.setdefault((result.day, result.session), []).append(result)
        partitions_written = 0
        rows_written = 0
        entries_by_status: dict[str, int] = {}
        seq = self._flush_seq
        self._flush_seq += 1
        for (day, session), items in groups.items():
            for item in items:
                self._guard(item)
                entries_by_status[item.entry.status.value] = entries_by_status.get(item.entry.status.value, 0) + 1
            certified = [r for r in items if r.entry.status in (CaptureStatus.COMPLETE, CaptureStatus.NO_TRADES)]
            if certified:
                coverage = {r.entry.symbol: r.entry for r in certified if r.entry.symbol}
                frames = [r.frame for r in certified if r.entry.status == CaptureStatus.COMPLETE and len(r.frame) > 0]
                if frames:
                    combined = pd.concat(frames, ignore_index=True)
                else:
                    combined = certified[0].frame.iloc[0:0].copy()
                stored = self._stored_counts(day, session)
                for r in certified:
                    self._log_replace(day, session, r, stored)
                write_tick_partition(
                    combined, day, session, coverage=coverage,
                    batch_rows=int(self._profile.COLLECTION_ARROW_BATCH_ROWS),
                )
                partitions_written += 1
                rows_written += sum(len(r.frame) for r in certified if r.entry.status == CaptureStatus.COMPLETE)
            self._publish_group_manifest(day, session, seq, [r.entry for r in items])
        self._buffer = []
        self._buffered_rows = 0
        return FlushReport(
            partitions_written=partitions_written, rows_written=rows_written,
            entries_by_status=entries_by_status,
        )

    @staticmethod
    def _stored_counts(day: str, session: str) -> dict[str, int]:
        target = tick_partition_path(day, session)
        try:
            if not target.exists():
                return {}
            stored = pd.read_parquet(target, columns=["symbol"])
        except (OSError, ValueError):
            return {}
        return {str(k): int(v) for k, v in stored["symbol"].astype(str).value_counts().items()}

    @staticmethod
    def _log_replace(day: str, session: str, result: TapeDayResult, stored: Mapping[str, int]) -> None:
        symbol = str(result.entry.symbol)
        after = int(len(result.frame)) if result.entry.status == CaptureStatus.COMPLETE else 0
        logger.info(
            "[DATA] stage=tape_replace symbol=%s day=%s session=%s before=%d after=%d",
            symbol, day, session, int(stored.get(symbol, 0)), after,
        )

    def _publish_group_manifest(self, day: str, session: str, seq: int, entries: list[CoverageEntry]) -> None:
        refs: list[Any] = []
        for entry in entries:
            for ref in entry.raw_refs:
                if ref not in refs:
                    refs.append(ref)
        status = (
            CaptureStatus.COMPLETE
            if all(item.status in GOOD_ENTRY_STATES for item in entries)
            else CaptureStatus.PARTIAL
        )
        manifest = CaptureManifest(
            schema_version=1,
            context=CaptureContext(
                trading_date=date.fromisoformat(day), run_id=f"tape-{day}-{session}-publish-{self._namespace}-{seq}",
                dataset=CaptureDataset.TRADE_TICKS, vendor="kiwoom", endpoint="tape-publish",
                symbol=None, venue="owner-local", session=session,
                capture_reason="tape-publish", cohort_id=None, scheduled_at=None,
            ),
            cohort=None, completed_at=datetime.now(_SEOUL),
            entries=tuple(entries), artifacts=tuple(refs), status=status,
        )
        try:
            self._store.publish_manifest(manifest)
        except ValueError as exc:
            logger.warning(
                "[DATA] stage=tape_publish status=DEGRADED reason=manifest_conflict day=%s session=%s detail=%s",
                day, session, exc,
            )
