"""KCA-owned closing/opening auction observation sweeps."""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Literal

import pandas as pd

from src import settings
from src.api.kis.key_pool import resolve_research_credentials, token_cache_path
from src.config.collection import CollectionSettings
from src.config.market_session import KRX_CLOSE_MARKET_DIV_CODE
from src.data.capture_contracts import (
    GOOD_ENTRY_STATES,
    SEOUL,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
    SessionClock,
)
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.session_calendar import SessionKind, resolve_session_day
from src.data.trading_calendar import is_kis_trading_day, resolve_prev_trading_day_kis
from src.execution.paper_broker import load_held_roster
from src.tools.run_outcome import RUN_OUTCOME_SKIPPED, record_run_outcome
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

_OPEN_OFFSET_MINUTES: tuple[int, ...] = (-20, -10, -5, -2, -1)
_PROGRAM_OFFSET_MINUTES: tuple[int, ...] = (-8, -4)
_OPEN_TRUST_FLOOR_SECONDS: int = 30


def _previous_trading_day(snapshot_date: str) -> str:
    try:
        day = date.fromisoformat(snapshot_date)
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.isoformat()


def close_rounds(clock: SessionClock, interval_seconds: int) -> list[datetime]:
    """Closing-auction orderbook sweep instants: every `interval_seconds` from close-9min up to (excluding) close.

    Shared with the daily audit, which must expect exactly the rounds the capture job schedules.
    """
    start = clock.close_at - timedelta(minutes=9)
    rounds: list[datetime] = []
    moment = start
    while moment < clock.close_at:
        rounds.append(moment)
        moment += timedelta(seconds=int(interval_seconds))
    return rounds


def _open_rounds(clock: SessionClock) -> list[datetime]:
    return [clock.open_at + timedelta(minutes=offset) for offset in _OPEN_OFFSET_MINUTES]


def program_rounds(clock: SessionClock) -> list[datetime]:
    """Program-trading snapshot instants at the configured minute offsets from the session close.

    Shared with the daily audit for the same expected-coverage reason.
    """
    return [clock.close_at + timedelta(minutes=offset) for offset in _PROGRAM_OFFSET_MINUTES]


def _resolve_roster(
    snapshot_date: str,
    phase: str,
    store: CaptureStore,
    now: datetime,
    previous_trading_day: str | None = None,
) -> tuple[list[str], str | None, bool]:
    if phase == "close":
        try:
            cohort = store.read_cohort(snapshot_date, available_by=now)
        except FileNotFoundError as exc:
            raise RuntimeError(f"auction cohort cannot be verified: {snapshot_date!r}") from exc
        return [str(item) for item in cohort.eligible_symbols], cohort.cohort_id, False
    prev_day = previous_trading_day or _previous_trading_day(snapshot_date)
    held = load_held_roster()
    try:
        prev_cohort = store.read_cohort(prev_day, available_by=now)
    except FileNotFoundError:
        logger.warning("[DATA] stage=auction_cohort status=INCOMPLETE reason=missing_previous date=%s prev=%s", snapshot_date, prev_day)
        if not held.ok:
            logger.warning(
                "[DATA] stage=auction_cohort status=INCOMPLETE reason=held_roster_unavailable error=%s date=%s",
                held.failure_reason,
                snapshot_date,
            )
        return list(held.symbols), None, True
    codes: list[str] = [str(item) for item in prev_cohort.eligible_symbols]
    for item in held.symbols:
        if item not in codes:
            codes.append(item)
    if not held.ok:
        logger.warning(
            "[DATA] stage=auction_cohort status=INCOMPLETE reason=held_roster_unavailable error=%s date=%s",
            held.failure_reason,
            snapshot_date,
        )
    return codes, prev_cohort.cohort_id, not held.ok


def _extract_open_price(payload: dict[str, Any] | None) -> int:
    if not isinstance(payload, dict):
        return 0
    output = payload.get("output")
    if not isinstance(output, dict):
        return 0
    try:
        return int(float(str(output.get("stck_oprc", "0") or "0").replace(",", "")))
    except (ValueError, TypeError):
        return 0


def _fragment_frame(symbol: str, scheduled_at: datetime, received_at: datetime, payload: dict[str, Any] | None) -> pd.DataFrame:
    import json

    def _encode(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)

    output1: str | None = _encode(payload.get("output1")) if isinstance(payload, dict) and "output1" in payload else None
    output2: str | None = _encode(payload.get("output2")) if isinstance(payload, dict) and "output2" in payload else None
    return pd.DataFrame([{"symbol": symbol, "scheduled_at": scheduled_at, "received_at": received_at, "output1": output1, "output2": output2}])


async def _wait_until_scheduled(
    scheduled_at: datetime,
    *,
    now_fn: Callable[[], datetime],
    sleeper: Callable[[float], Awaitable[None]],
    injected_clock: bool,
) -> None:
    """Wait for a scheduled observation, leaving injected clocks under test control."""
    wait_seconds = (scheduled_at - now_fn()).total_seconds()
    if wait_seconds <= 0 or (injected_clock and sleeper is asyncio.sleep):
        return
    await sleeper(wait_seconds)


async def _observe_roster(
    *,
    roster: Sequence[str],
    clients: Sequence[Any],
    semaphore: asyncio.Semaphore,
    deadline: datetime,
    now_fn: Callable[[], datetime],
    observe: Callable[[int, str, Any], Awaitable[CoverageEntry]],
    on_deadline: Callable[[int, str], CoverageEntry],
) -> list[CoverageEntry]:
    """Observe every roster symbol for one scheduled round with bounded concurrency.

    A serial loop makes round duration proportional to roster size times round-trip
    latency; with host-wide admission pacing the account, concurrency up to the per-key
    bound lets a round finish at the account's documented rate instead.

    Args:
        roster: Ordered symbols; position selects the client (position % len(clients)).
        clients: Research-slot data clients.
        semaphore: Shared bound of COLLECTION_CONCURRENCY_PER_KEY * len(clients).
        deadline: Dispatch cutoff for this round (next round start or phase end).
        observe: Performs one symbol's request and persistence, returning its entry.
        on_deadline: Builds the PARTIAL(reason="deadline") entry for an undispatched symbol.

    Returns:
        One entry per roster symbol, ordered by roster position.
    """

    async def _one(position: int, symbol: str) -> CoverageEntry:
        async with semaphore:
            if now_fn() >= deadline:
                return on_deadline(position, symbol)
            return await observe(position, symbol, clients[position % len(clients)])

    return list(
        await asyncio.gather(*(_one(position, symbol) for position, symbol in enumerate(roster)))
    )


def _program_deadline_entry(position: int, symbol: str, *, scheduled_at: datetime) -> CoverageEntry:
    """PARTIAL deadline entry for a program symbol never dispatched in its round."""
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.PROGRAM,
        venue="KRX",
        session="regular",
        scheduled_at=scheduled_at,
        status=CaptureStatus.PARTIAL,
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason="deadline",
        raw_refs=(),
    )


async def _observe_program_symbol(
    position: int,
    symbol: str,
    client: Any,
    *,
    session: Any,
    scheduled_at: datetime,
    prog_index: int,
    store: CaptureStore,
    trading_day: date,
    run_id: str,
    cohort_id: str | None,
    reason_tag: str,
    now_fn: Callable[[], datetime],
) -> CoverageEntry:
    """Request and persist one symbol's program-trading snapshot for a scheduled round."""
    started = now_fn()
    try:
        payload = await client.get_program_net_buy(session, symbol, market_div_code=KRX_CLOSE_MARKET_DIV_CODE)
    except Exception as exc:
        received = now_fn()
        return CoverageEntry(
            symbol=symbol,
            dataset=CaptureDataset.PROGRAM,
            venue="KRX",
            session="regular",
            scheduled_at=scheduled_at,
            status=CaptureStatus.FAILED,
            rows=0,
            first_event_time=started,
            last_event_time=received,
            reason=type(exc).__name__,
            raw_refs=(),
        )
    received = now_fn()
    body = dict(payload) if isinstance(payload, dict) else None
    ok = isinstance(body, dict) and body.get("rt_cd") == "0"
    context = CaptureContext(
        trading_date=trading_day,
        run_id=run_id,
        dataset=CaptureDataset.PROGRAM,
        vendor="kis",
        endpoint="program-trade-by-stock",
        symbol=symbol,
        venue="KRX",
        session="regular",
        capture_reason=reason_tag,
        cohort_id=cohort_id,
        scheduled_at=scheduled_at,
    )
    try:
        ref = store.append_response(
            CapturedResponse(
                context=context,
                request_started_at=started,
                received_at=received,
                payload=body,
                status=CaptureStatus.COMPLETE if ok else CaptureStatus.FAILED,
                source_timestamp=None,
                source_published_at=None,
                page_index=1000 + prog_index,
                attempt_index=0,
                continuation={},
                error_type=None if ok else "vendor_failure",
            )
        )
    except OSError:
        return CoverageEntry(
            symbol=symbol,
            dataset=CaptureDataset.PROGRAM,
            venue="KRX",
            session="regular",
            scheduled_at=scheduled_at,
            status=CaptureStatus.FAILED,
            rows=0,
            first_event_time=started,
            last_event_time=received,
            reason="persistence",
            raw_refs=(),
        )
    frame_status = CaptureStatus.COMPLETE if ok else CaptureStatus.FAILED
    try:
        frame_context = context.model_copy(update={"run_id": f"{run_id}-g{prog_index}-{position}"})
        store.publish_frame(_fragment_frame(symbol, scheduled_at, received, body), context=frame_context)
    except OSError:
        frame_status = CaptureStatus.PARTIAL
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.PROGRAM,
        venue="KRX",
        session="regular",
        scheduled_at=scheduled_at,
        status=frame_status,
        rows=1 if ok else 0,
        first_event_time=started,
        last_event_time=received,
        reason=reason_tag if frame_status == CaptureStatus.COMPLETE else ("persistence" if ok else "vendor_failure"),
        raw_refs=(ref,),
    )


async def _capture_program_rounds(
    *,
    session: Any,
    rounds: Sequence[datetime],
    roster: Sequence[str],
    clients: Sequence[Any],
    semaphore: asyncio.Semaphore,
    store: CaptureStore,
    trading_day: date,
    run_id: str,
    cohort_id: str | None,
    reason_tag: str,
    close_at: datetime,
    now_fn: Callable[[], datetime],
    sleeper: Callable[[float], Awaitable[None]],
    injected_clock: bool,
) -> list[CoverageEntry]:
    """Capture program observations at their declared pre-close timestamps."""
    entries: list[CoverageEntry] = []
    for prog_index, scheduled_at in enumerate(rounds):
        await _wait_until_scheduled(
            scheduled_at, now_fn=now_fn, sleeper=sleeper, injected_clock=injected_clock
        )
        deadline = rounds[prog_index + 1] if prog_index + 1 < len(rounds) else close_at
        entries.extend(
            await _observe_roster(
                roster=roster,
                clients=clients,
                semaphore=semaphore,
                deadline=deadline,
                now_fn=now_fn,
                observe=functools.partial(
                    _observe_program_symbol,
                    session=session,
                    scheduled_at=scheduled_at,
                    prog_index=prog_index,
                    store=store,
                    trading_day=trading_day,
                    run_id=run_id,
                    cohort_id=cohort_id,
                    reason_tag=reason_tag,
                    now_fn=now_fn,
                ),
                on_deadline=functools.partial(_program_deadline_entry, scheduled_at=scheduled_at),
            )
        )
    return entries


def _orderbook_deadline_entry(position: int, symbol: str, *, scheduled_at: datetime) -> CoverageEntry:
    """PARTIAL deadline entry for an orderbook symbol never dispatched in its round."""
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.ORDERBOOK,
        venue="KRX",
        session="regular",
        scheduled_at=scheduled_at,
        status=CaptureStatus.PARTIAL,
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason="deadline",
        raw_refs=(),
    )


async def _observe_orderbook_symbol(
    position: int,
    symbol: str,
    client: Any,
    *,
    session: Any,
    scheduled_at: datetime,
    page_index: int,
    deadline: datetime,
    store: CaptureStore,
    trading_day: date,
    run_id: str,
    cohort_id: str | None,
    reason_tag: str,
    phase: str,
    now_fn: Callable[[], datetime],
    degraded: list[str],
) -> CoverageEntry:
    """Request and persist one symbol's orderbook snapshot for a scheduled round."""
    started = now_fn()
    try:
        payload = await client.get_orderbook_snapshot(session, symbol, market_div_code=KRX_CLOSE_MARKET_DIV_CODE)
    except Exception as exc:
        received = now_fn()
        degraded.append(symbol)
        return CoverageEntry(
            symbol=symbol,
            dataset=CaptureDataset.ORDERBOOK,
            venue="KRX",
            session="regular",
            scheduled_at=scheduled_at,
            status=CaptureStatus.FAILED,
            rows=0,
            first_event_time=started,
            last_event_time=received,
            reason=type(exc).__name__,
            raw_refs=(),
        )
    received = now_fn()
    body = dict(payload) if isinstance(payload, dict) else None
    ok = isinstance(body, dict) and body.get("rt_cd") == "0"
    context = CaptureContext(
        trading_date=trading_day,
        run_id=run_id,
        dataset=CaptureDataset.ORDERBOOK,
        vendor="kis",
        endpoint="inquire-asking-price-exp-ccn",
        symbol=symbol,
        venue="KRX",
        session="regular",
        capture_reason=reason_tag,
        cohort_id=cohort_id,
        scheduled_at=scheduled_at,
    )
    try:
        ref = store.append_response(
            CapturedResponse(
                context=context,
                request_started_at=started,
                received_at=received,
                payload=body,
                status=CaptureStatus.COMPLETE if ok else CaptureStatus.FAILED,
                source_timestamp=None,
                source_published_at=None,
                page_index=page_index,
                attempt_index=0,
                continuation={},
                error_type=None if ok else "vendor_failure",
            )
        )
    except OSError:
        degraded.append(symbol)
        return CoverageEntry(
            symbol=symbol,
            dataset=CaptureDataset.ORDERBOOK,
            venue="KRX",
            session="regular",
            scheduled_at=scheduled_at,
            status=CaptureStatus.FAILED,
            rows=0,
            first_event_time=started,
            last_event_time=received,
            reason="persistence",
            raw_refs=(),
        )
    late = received > deadline
    if late:
        degraded.append(symbol)
        logger.warning(
            "[DATA] stage=auction_capture status=LATE phase=%s symbol=%s scheduled=%s received=%s",
            phase,
            symbol,
            scheduled_at.isoformat(),
            received.isoformat(),
        )
    try:
        frame_context = context.model_copy(update={"run_id": f"{run_id}-r{page_index}-{position}"})
        store.publish_frame(_fragment_frame(symbol, scheduled_at, received, body), context=frame_context)
    except OSError:
        degraded.append(symbol)
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.ORDERBOOK,
        venue="KRX",
        session="regular",
        scheduled_at=scheduled_at,
        status=CaptureStatus.COMPLETE if ok and not late else CaptureStatus.PARTIAL,
        rows=1 if ok else 0,
        first_event_time=started,
        last_event_time=received,
        reason=reason_tag if ok and not late else ("deadline" if late else "vendor_failure"),
        raw_refs=(ref,),
    )


async def _observe_open_symbol(
    position: int,
    symbol: str,
    client: Any,
    *,
    session: Any,
    floor: datetime,
    store: CaptureStore,
    trading_day: date,
    run_id: str,
    cohort_id: str | None,
    reason_tag: str,
    now_fn: Callable[[], datetime],
    attempt: int,
    poll_started: dict[str, datetime],
    last_seen: dict[str, tuple[Any, datetime]],
    settled: set[int],
    degraded: list[str],
) -> CoverageEntry:
    """Poll one symbol's opening price; unresolved symbols return a provisional entry for re-pass."""
    started_at = poll_started.setdefault(symbol, now_fn())
    payload = await client.get_current_price(session, symbol, market_div_code=KRX_CLOSE_MARKET_DIV_CODE)
    received = now_fn()
    body = dict(payload) if isinstance(payload, dict) else None
    price = _extract_open_price(body)
    context = CaptureContext(
        trading_date=trading_day,
        run_id=run_id,
        dataset=CaptureDataset.PRICE,
        vendor="kis",
        endpoint="inquire-price",
        symbol=symbol,
        venue="KRX",
        session="regular",
        capture_reason=reason_tag,
        cohort_id=cohort_id,
        scheduled_at=floor,
    )
    try:
        ref = store.append_response(
            CapturedResponse(
                context=context,
                request_started_at=started_at,
                received_at=received,
                payload=body,
                status=CaptureStatus.COMPLETE if price > 0 else CaptureStatus.FAILED,
                source_timestamp=None,
                source_published_at=None,
                page_index=position,
                attempt_index=attempt,
                continuation={},
                error_type=None if price > 0 else "open_unresolved",
            )
        )
    except OSError:
        degraded.append(symbol)
        settled.add(position)
        return CoverageEntry(
            symbol=symbol,
            dataset=CaptureDataset.PRICE,
            venue="KRX",
            session="regular",
            scheduled_at=floor,
            status=CaptureStatus.FAILED,
            rows=0,
            first_event_time=started_at,
            last_event_time=received,
            reason="persistence",
            raw_refs=(),
        )
    try:
        frame_context = context.model_copy(update={"run_id": f"{run_id}-o{position}-{attempt}"})
        store.publish_frame(_fragment_frame(symbol, floor, received, body), context=frame_context)
    except OSError:
        degraded.append(symbol)
    if price > 0:
        settled.add(position)
        return CoverageEntry(
            symbol=symbol,
            dataset=CaptureDataset.PRICE,
            venue="KRX",
            session="regular",
            scheduled_at=floor,
            status=CaptureStatus.COMPLETE,
            rows=1,
            first_event_time=started_at,
            last_event_time=received,
            reason=reason_tag,
            raw_refs=(ref,),
        )
    last_seen[symbol] = (ref, received)
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.PRICE,
        venue="KRX",
        session="regular",
        scheduled_at=floor,
        status=CaptureStatus.PARTIAL,
        rows=0,
        first_event_time=started_at,
        last_event_time=received,
        reason="open_unresolved",
        raw_refs=(ref,),
    )


def _open_deadline_entry(
    position: int,
    symbol: str,
    *,
    floor: datetime,
    now_fn: Callable[[], datetime],
    poll_started: dict[str, datetime],
    last_seen: dict[str, tuple[Any, datetime]],
    settled: set[int],
) -> CoverageEntry:
    """Final PARTIAL entry for an open symbol never dispatched before confirmation end."""
    settled.add(position)
    moment = last_seen.get(symbol)
    seen_at = moment[1] if moment is not None else now_fn()
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.PRICE,
        venue="KRX",
        session="regular",
        scheduled_at=floor,
        status=CaptureStatus.PARTIAL,
        rows=0,
        first_event_time=poll_started.get(symbol, seen_at),
        last_event_time=seen_at,
        reason="open_unresolved",
        raw_refs=(moment[0],) if moment is not None else (),
    )


async def run_auction_capture(
    snapshot_date: str,
    *,
    phase: Literal["close", "open"],
    profile: CollectionSettings,
    store: CaptureStore,
    clients: Sequence[Any],
    session_clock: SessionClock,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    previous_trading_day: str | None = None,
) -> CaptureManifest:
    """Capture the complete project cohort through independently timed auctions.

    Sweeps are observations across time, not simultaneous market cross sections.
    Opening follow-up uses the previous actual trading day's population even
    when a stock is absent from today's candidates or another collector's feed.

    Args:
        snapshot_date: Actual date of the observed session.
        phase: Closing or opening observation phase.
        profile: Explicit key/call/deadline configuration.
        store: First-party raw and task evidence storage.
        clients: Exclusively qualified, prewarmed data clients.
        session_clock: Verified standard or exceptional dated market clock.
        now_fn: Actual aware clock; None uses Asia/Seoul now.
        sleep_fn: Bounded scheduler wait; None uses asyncio.sleep.

    Returns:
        Manifest retaining all expected entries and observed task states.

    Raises:
        ValueError: Wrong date/phase, clock, or profile.
        OSError: Required raw/coverage persistence fails.
        RuntimeError: Calendar or acquisition infrastructure cannot be verified.
    """
    now_clock = now_fn or (lambda: datetime.now(SEOUL))
    sleeper = sleep_fn or asyncio.sleep
    injected_clock = now_fn is not None
    try:
        trading_day = date.fromisoformat(snapshot_date)
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    if phase not in ("close", "open"):
        raise ValueError(f"unknown phase {phase!r}")
    if session_clock.trading_date != trading_day:
        raise ValueError("session clock trading_date must equal snapshot_date")
    if not profile.COLLECTION_AUCTION_ENABLED:
        raise ValueError("auction capture requires enabled auction collection")
    if not profile.COLLECTION_RESEARCH_SLOTS:
        raise ValueError("auction capture requires declared research slots")
    if not clients:
        raise ValueError("auction capture requires prewarmed data clients")
    now = now_clock()
    roster, cohort_id, cohort_incomplete = _resolve_roster(snapshot_date, phase, store, now, previous_trading_day)
    if phase == "close":
        rounds = close_rounds(session_clock, int(profile.COLLECTION_AUCTION_INTERVAL_SECONDS))
        phase_end = session_clock.close_at
    else:
        rounds = _open_rounds(session_clock)
        phase_end = session_clock.open_at + timedelta(seconds=int(profile.COLLECTION_OPEN_CONFIRM_SECONDS))
    run_id = f"auction-{phase}-{snapshot_date}-{uuid.uuid4().hex[:8]}"
    reason_tag = f"auction-{phase}"
    sem = asyncio.Semaphore(int(profile.COLLECTION_CONCURRENCY_PER_KEY) * len(clients))
    entries: list[CoverageEntry] = []
    incomplete = bool(cohort_incomplete)
    first_client: Any = clients[0]

    async with first_client.create_session() as broker_session:
        program_task: asyncio.Task[list[CoverageEntry]] | None = None
        if phase == "close":
            program_task = asyncio.create_task(
                _capture_program_rounds(
                    session=broker_session,
                    rounds=program_rounds(session_clock),
                    roster=roster,
                    clients=clients,
                    semaphore=sem,
                    store=store,
                    trading_day=trading_day,
                    run_id=run_id,
                    cohort_id=cohort_id,
                    reason_tag=reason_tag,
                    close_at=session_clock.close_at,
                    now_fn=now_clock,
                    sleeper=sleeper,
                    injected_clock=injected_clock,
                )
            )
        for index, scheduled_at in enumerate(rounds):
            await _wait_until_scheduled(
                scheduled_at, now_fn=now_clock, sleeper=sleeper, injected_clock=injected_clock
            )
            deadline = rounds[index + 1] if index + 1 < len(rounds) else phase_end
            if now_clock() >= deadline:
                logger.warning(
                    "[DATA] stage=auction_capture status=DEADLINE_EXCEEDED phase=%s round=%s n_missing=%d",
                    phase,
                    scheduled_at.isoformat(),
                    len(roster),
                )
                incomplete = True
                entries.extend(
                    CoverageEntry(
                        symbol=symbol,
                        dataset=CaptureDataset.ORDERBOOK,
                        venue="KRX",
                        session="regular",
                        scheduled_at=scheduled_at,
                        status=CaptureStatus.PARTIAL,
                        rows=0,
                        first_event_time=None,
                        last_event_time=None,
                        reason="deadline",
                        raw_refs=(),
                    )
                    for symbol in roster
                )
                continue
            degraded: list[str] = []
            round_entries = await _observe_roster(
                roster=roster,
                clients=clients,
                semaphore=sem,
                deadline=deadline,
                now_fn=now_clock,
                observe=functools.partial(
                    _observe_orderbook_symbol,
                    session=broker_session,
                    scheduled_at=scheduled_at,
                    page_index=index,
                    deadline=deadline,
                    store=store,
                    trading_day=trading_day,
                    run_id=run_id,
                    cohort_id=cohort_id,
                    reason_tag=reason_tag,
                    phase=phase,
                    now_fn=now_clock,
                    degraded=degraded,
                ),
                on_deadline=functools.partial(_orderbook_deadline_entry, scheduled_at=scheduled_at),
            )
            entries.extend(round_entries)
            if degraded or any(entry.status is not CaptureStatus.COMPLETE for entry in round_entries):
                incomplete = True
        if phase == "close":
            if program_task is not None:
                program_entries = await program_task
                entries.extend(program_entries)
                if any(entry.status not in GOOD_ENTRY_STATES for entry in program_entries):
                    incomplete = True
        else:
            floor = session_clock.open_at + timedelta(seconds=_OPEN_TRUST_FLOOR_SECONDS)
            confirm_end = session_clock.open_at + timedelta(seconds=int(profile.COLLECTION_OPEN_CONFIRM_SECONDS))
            # 종목별 순차 폴링은 시초가가 늦게 형성되는 종목 하나가 확인 예산을 독점해 뒤 종목이 시도조차
            # 못 하게 한다(실측 2026-09-29: 0010S0 131.7초 → 542종목 중 494개 미시도). 전 종목을 한 패스씩
            # 돌고 미형성 종목만 예산 안에서 재패스해 종목 간 간섭을 없앤다.
            wait = (floor - now_clock()).total_seconds()
            if wait > 0:
                await sleeper(wait)
            pending = list(enumerate(roster))
            poll_started: dict[str, datetime] = {}
            last_seen: dict[str, tuple[Any, datetime]] = {}
            attempt = 0
            while pending and now_clock() <= confirm_end:
                pass_symbols = [symbol for _, symbol in pending]
                settled: set[int] = set()
                pass_degraded: list[str] = []
                round_entries = await _observe_roster(
                    roster=pass_symbols,
                    clients=clients,
                    semaphore=sem,
                    deadline=confirm_end,
                    now_fn=now_clock,
                    observe=functools.partial(
                        _observe_open_symbol,
                        session=broker_session,
                        floor=floor,
                        store=store,
                        trading_day=trading_day,
                        run_id=run_id,
                        cohort_id=cohort_id,
                        reason_tag=reason_tag,
                        now_fn=now_clock,
                        attempt=attempt,
                        poll_started=poll_started,
                        last_seen=last_seen,
                        settled=settled,
                        degraded=pass_degraded,
                    ),
                    on_deadline=functools.partial(
                        _open_deadline_entry,
                        floor=floor,
                        now_fn=now_clock,
                        poll_started=poll_started,
                        last_seen=last_seen,
                        settled=settled,
                    ),
                )
                next_pending: list[tuple[int, str]] = []
                for pass_index, ((orig_position, symbol), entry) in enumerate(zip(pending, round_entries)):
                    if pass_index in settled:
                        entries.append(entry)
                    else:
                        next_pending.append((orig_position, symbol))
                if pass_degraded or any(
                    entry.status is not CaptureStatus.COMPLETE
                    for pass_index, entry in enumerate(round_entries)
                    if pass_index in settled
                ):
                    incomplete = True
                pending = next_pending
                if pending:
                    attempt += 1
                    await sleeper(1.0)
            for _, symbol in pending:
                incomplete = True
                seen = last_seen.get(symbol)
                moment = seen[1] if seen is not None else now_clock()
                entries.append(
                    CoverageEntry(
                        symbol=symbol,
                        dataset=CaptureDataset.PRICE,
                        venue="KRX",
                        session="regular",
                        scheduled_at=floor,
                        status=CaptureStatus.PARTIAL,
                        rows=0,
                        first_event_time=poll_started.get(symbol, moment),
                        last_event_time=moment,
                        reason="open_unresolved",
                        raw_refs=(seen[0],) if seen is not None else (),
                    )
                )
    status = CaptureStatus.COMPLETE
    for item in entries:
        if item.status not in GOOD_ENTRY_STATES:
            status = CaptureStatus.PARTIAL
            break
    if incomplete and status == CaptureStatus.COMPLETE:
        status = CaptureStatus.PARTIAL
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=trading_day,
            run_id=run_id,
            dataset=CaptureDataset.ORDERBOOK if phase == "close" else CaptureDataset.PRICE,
            vendor="kis",
            endpoint="auction-capture",
            symbol=None,
            venue="KRX",
            session="regular",
            capture_reason=reason_tag,
            cohort_id=cohort_id,
            scheduled_at=None,
        ),
        cohort=None,
        completed_at=now_clock(),
        entries=tuple(entries),
        artifacts=tuple(ref for item in entries for ref in item.raw_refs),
        status=status,
    )
    try:
        store.publish_manifest(manifest)
    except OSError as exc:
        raise OSError("required auction coverage persistence fails") from exc
    return manifest


async def _run_async(snapshot_date: str, phase: str, profile: CollectionSettings) -> CaptureManifest | None:
    """Run one capture phase; ``None`` when KIS reports ``snapshot_date`` as a market holiday."""
    trading_day = date.fromisoformat(snapshot_date)
    session_day = resolve_session_day(trading_day, overrides=profile.COLLECTION_SESSION_OVERRIDES)
    if session_day.kind is SessionKind.CLOSED:
        record_run_outcome(
            "auction_capture",
            RUN_OUTCOME_SKIPPED,
            run_date=snapshot_date,
            reason="non_trading_day",
            metrics={"session": "CLOSED"},
        )
        logger.info("[DATA] stage=auction_capture status=SKIP reason=non_trading_day date=%s", snapshot_date)
        return None
    import os

    from src.api.kis.client import KisApiClient

    env = dict(os.environ)
    creds = resolve_research_credentials(env, slots=tuple(profile.COLLECTION_RESEARCH_SLOTS))
    clients = [
        KisApiClient(
            cred.app_key,
            cred.app_secret,
            "",
            cred.hts_id,
            token_file=str(token_cache_path(cred.app_key, settings.KIS_TOKEN_CACHE_DIR)),
        )
        for cred in creds
    ]
    store = CaptureStore(_capture_root(profile))
    trading_day = date.fromisoformat(snapshot_date)
    clock = session_day.clock if session_day.clock is not None else SessionClock.standard(trading_day)
    previous_trading_day: str | None = None
    async with clients[0].create_session() as broker_session:
        for client in clients:
            await client.ensure_token(broker_session)
        # 평일 공휴일(명절 등)은 주말 판정으로 걸러지지 않는다. collect 와 같은 KIS 거래일 오라클로 판정한다.
        if not await is_kis_trading_day(clients[0], broker_session, snapshot_date):
            return None
        if phase == "open":
            # 연휴 직후 개장의 모집단은 직전 '실제' 거래일 코호트여야 한다(주말만 건너뛰면 휴장일을 가리킨다).
            prev = await resolve_prev_trading_day_kis(clients[0], broker_session, pd.Timestamp(snapshot_date))
            previous_trading_day = prev.strftime("%Y-%m-%d")
    return await run_auction_capture(
        snapshot_date,
        phase=phase,  # type: ignore[arg-type]
        profile=profile,
        store=store,
        clients=clients,
        session_clock=clock,
        previous_trading_day=previous_trading_day,
    )


def main(argv: list[str] | None = None) -> None:
    """Run an explicitly configured research phase without transmitting orders.

    Args:
        argv: Optional --phase {close,open} and --date ISO-date arguments.

    Raises:
        ValueError: Invalid or uncertified configuration.
        RuntimeError: Required calendar, cohort, or evidence cannot be verified.
    """
    parser = argparse.ArgumentParser(description="KCA auction capture (research, no orders)")
    parser.add_argument("--phase", choices=["close", "open"], required=True)
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)
    profile = CollectionSettings()
    if not profile.COLLECTION_AUCTION_ENABLED:
        logger.info("[DATA] stage=auction_capture status=SKIP reason=disabled")
        return
    snapshot_date = args.date or datetime.now(SEOUL).date().isoformat()
    try:
        trading_day = date.fromisoformat(snapshot_date)
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    if trading_day.weekday() >= 5:
        logger.info("[DATA] stage=auction_capture status=SKIP reason=holiday date=%s", snapshot_date)
        return
    manifest = asyncio.run(_run_async(snapshot_date, args.phase, profile))
    if manifest is None:
        # 휴장일은 장애가 아니다: 정상 종료해 OnFailure 오탐 알림과 무의미한 캡처를 막는다.
        record_run_outcome(
            "auction_capture",
            RUN_OUTCOME_SKIPPED,
            run_date=snapshot_date,
            reason="non_trading_day",
            metrics={"session": "CLOSED"},
        )
        logger.info("[DATA] stage=auction_capture status=SKIP reason=non_trading_day date=%s", snapshot_date)
        return
    logger.info(
        "[DATA] stage=auction_capture status=%s phase=%s date=%s entries=%d",
        manifest.status.value,
        args.phase,
        snapshot_date,
        len(manifest.entries),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    # 실측: __main__ 가드가 없어 `python -m`으로 실행해도 main()이 전혀 호출되지 않고
    # 조용히 성공 종료(exit 0)했다 -- kca-auction-open/kca-auction-close가 매번
    # "성공"으로 보이면서 실제로는 아무 것도 수집하지 않았다(2026-09-21 daily-audit이
    # collection:auction_open:1:missing_manifest / auction_close:*:missing_entries로 포착).
    configure_cli_logging()
    main()
