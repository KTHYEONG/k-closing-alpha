"""KCA-owned closing/opening auction observation sweeps."""

from __future__ import annotations

import argparse
import asyncio
import collections
import functools
import logging
import math
import uuid
from collections.abc import Awaitable, Callable, Sequence
from collections.abc import Set as AbstractSet
from datetime import date, datetime, timedelta
from typing import Any, Literal

import pandas as pd

from src import settings
from src.api.kis.key_pool import (
    KisCredential,
    prewarm_healthy_clients,
    resolve_handoff_credentials,
    resolve_research_credentials,
    token_cache_path,
)
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
from src.data.research_roster import research_extras, research_manifest, split_research_entries
from src.data.session_calendar import SessionKind, resolve_session_day
from src.data.trading_calendar import is_kis_trading_day, resolve_prev_trading_day_kis
from src.execution.paper_broker import load_held_roster
from src.tools.run_outcome import RUN_OUTCOME_DEGRADED, RUN_OUTCOME_OK, RUN_OUTCOME_SKIPPED, record_run_outcome
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

_OPEN_OFFSET_MINUTES: tuple[int, ...] = (-20, -10, -5, -2, -1)
_PROGRAM_OFFSET_MINUTES: tuple[int, ...] = (-8, -4)
_OPEN_TRUST_FLOOR_SECONDS: int = 30
# KIS gateway asks the caller to simply re-query for this code (observed 2026-11-04 15:25, one symbol lost).
_TRANSIENT_VENDOR_CODES: frozenset[str] = frozenset({"EGW00316"})
_TRANSIENT_VENDOR_RETRIES: int = 1


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


def estimate_round_capacity(n_clients: int, interval_seconds: int, *, utilization: float, tps_per_key: float) -> int:
    """Requests one round of `interval_seconds` can plan for across `n_clients` rate-limited keys.

    Raises:
        ValueError: Non-positive interval, utilization outside (0, 1], or non-positive/non-finite rate.
    """
    if interval_seconds <= 0:
        raise ValueError(f"interval_seconds must be positive: {interval_seconds!r}")
    if not 0 < utilization <= 1:
        raise ValueError(f"utilization must be in (0, 1]: {utilization!r}")
    if not math.isfinite(tps_per_key) or tps_per_key <= 0:
        raise ValueError(f"tps_per_key must be positive and finite: {tps_per_key!r}")
    if n_clients <= 0:
        return 0
    return math.floor(n_clients * tps_per_key * interval_seconds * utilization)


def _rotate_tier(tier: list[str], round_index: int, round_count: int) -> list[str]:
    if not tier:
        return []
    offset = ((round_index * len(tier)) // round_count) % len(tier)
    return tier[offset:] + tier[:offset]


def plan_round_order(
    dispatch: Sequence[str],
    *,
    priority: AbstractSet[str],
    low_priority: AbstractSet[str],
    round_index: int,
    round_count: int,
) -> list[str]:
    """Deterministic request order for one orderbook round.

    Tier 0 (`priority`) keeps dispatch order; tier 1 (the rest) and tier 2 (`low_priority`
    not in `priority`, always last) rotate left by `round_index * len(tier) // round_count`,
    so the symbols a saturated round cuts off differ from round to round.

    Raises:
        ValueError: `round_count < 1`, `round_index` outside `[0, round_count)`, or duplicate symbols.
    """
    if round_count < 1:
        raise ValueError(f"round_count must be >= 1: {round_count!r}")
    if not 0 <= round_index < round_count:
        raise ValueError(f"round_index {round_index!r} outside [0, {round_count})")
    if len(set(dispatch)) != len(dispatch):
        raise ValueError("dispatch symbols must be unique")
    head = [symbol for symbol in dispatch if symbol in priority]
    body = [symbol for symbol in dispatch if symbol not in priority and symbol not in low_priority]
    tail = [symbol for symbol in dispatch if symbol not in priority and symbol in low_priority]
    return [*head, *_rotate_tier(body, round_index, round_count), *_rotate_tier(tail, round_index, round_count)]


class _OrderbookPriorityGate:
    """Idle except between `begin_round` and `end_round` of an orderbook round."""

    def __init__(self) -> None:
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def is_idle(self) -> bool:
        return self._idle.is_set()

    def begin_round(self) -> None:
        self._idle.clear()

    def end_round(self) -> None:
        self._idle.set()

    async def wait_idle(self) -> None:
        await self._idle.wait()


class _ClientPool:
    """Research keys always serve; each handoff key joins once its single activation attempt succeeds.

    Every key owns its semaphore, so the per-key in-flight bound holds while the active set grows.
    """

    def __init__(self, base: Sequence[Any], handoff: Sequence[Any], handoff_labels: Sequence[str], *, per_key: int) -> None:
        self._base: list[tuple[Any, asyncio.Semaphore]] = [(client, asyncio.Semaphore(per_key)) for client in base]
        self._handoff: list[tuple[Any, asyncio.Semaphore]] = [(client, asyncio.Semaphore(per_key)) for client in handoff]
        self.handoff_labels: tuple[str, ...] = tuple(handoff_labels)
        self.decision_published = False
        self._attempted: set[int] = set()
        self._activated: set[int] = set()

    @property
    def handoff_configured(self) -> bool:
        return bool(self._handoff)

    @property
    def handoff_clients(self) -> list[Any]:
        return [client for client, _ in self._handoff]

    @property
    def unattempted_handoff(self) -> list[int]:
        return [index for index in range(len(self._handoff)) if index not in self._attempted]

    def mark_attempted(self, index: int) -> None:
        self._attempted.add(index)

    def activate_handoff(self, index: int = 0) -> None:
        self._activated.add(index)

    def active_count(self) -> int:
        return len(self._active())

    def pick(self, position: int) -> tuple[Any, asyncio.Semaphore]:
        active = self._active()
        return active[position % len(active)]

    def _active(self) -> list[tuple[Any, asyncio.Semaphore]]:
        return self._base + [self._handoff[index] for index in sorted(self._activated)]


def _decision_manifest_published(store: CaptureStore, snapshot_date: str) -> bool:
    """A decision manifest is published only after its collect, including every decision-shard request, has returned."""
    return any(
        manifest.cohort is not None and manifest.context.run_id.startswith("decision-")
        for manifest in store.read_manifests(snapshot_date)
    )


async def _maybe_activate_handoff(
    pool: _ClientPool,
    store: CaptureStore,
    snapshot_date: str,
    *,
    round_at: datetime,
) -> bool:
    """Adopt each handoff slot's cached token once, only after the day's decision collect finished; never issues, never raises."""
    if not pool.unattempted_handoff:
        return False
    if not pool.decision_published:
        try:
            pool.decision_published = _decision_manifest_published(store, snapshot_date)
        except (OSError, ValueError) as exc:
            logger.warning("[DATA] stage=auction_capture status=HANDOFF_WAIT reason=%s", type(exc).__name__)
            return False
        if not pool.decision_published:
            return False
    activated: list[str] = []
    for index in pool.unattempted_handoff:
        pool.mark_attempted(index)
        label = pool.handoff_labels[index]
        try:
            adopted = pool.handoff_clients[index].adopt_cached_token()
        except Exception as exc:
            logger.warning(
                "[DATA] stage=auction_capture status=HANDOFF_FAILED slot=%s reason=%s", label, type(exc).__name__
            )
            continue
        if adopted is None:
            logger.warning("[DATA] stage=auction_capture status=HANDOFF_FAILED slot=%s reason=no_cached_token", label)
            continue
        pool.activate_handoff(index)
        activated.append(label)
    if activated:
        logger.info(
            "[DATA] stage=auction_capture status=HANDOFF_ACTIVE phase=close round=%s slots=%s active_keys=%d",
            round_at.isoformat(),
            ",".join(activated),
            pool.active_count(),
        )
    return bool(activated)


async def _observe_roster(
    *,
    roster: Sequence[str],
    clients: Sequence[Any] = (),
    semaphore: asyncio.Semaphore | None = None,
    pool: _ClientPool | None = None,
    deadline: datetime,
    now_fn: Callable[[], datetime],
    observe: Callable[[int, str, Any], Awaitable[CoverageEntry]],
    on_deadline: Callable[[int, str], CoverageEntry],
    gate: _OrderbookPriorityGate | None = None,
) -> list[CoverageEntry]:
    """Observe every roster symbol for one scheduled round with bounded concurrency.

    A serial loop makes round duration proportional to roster size times round-trip
    latency; with host-wide admission pacing the account, concurrency up to the per-key
    bound lets a round finish at the account's documented rate instead.

    Args:
        roster: Ordered symbols; position selects the client (position % len(clients)).
        clients: Research-slot data clients, used with ``semaphore`` when ``pool`` is None.
        semaphore: Shared bound of COLLECTION_CONCURRENCY_PER_KEY * len(clients) when ``pool`` is None.
        pool: Per-key lanes; takes precedence over ``clients`` and ``semaphore``.
        deadline: Dispatch cutoff for this round (next round start or phase end).
        observe: Performs one symbol's request and persistence, returning its entry.
        on_deadline: Builds the PARTIAL(reason="deadline") entry for an undispatched symbol.
        gate: Program lane only; requests wait while an orderbook round is active and never hold a slot then.

    Returns:
        One entry per roster symbol, ordered by roster position.
    """

    lane: Callable[[int], tuple[Any, asyncio.Semaphore]]
    if pool is not None:
        lane = pool.pick
    else:
        if not clients or semaphore is None:
            raise ValueError("observe roster requires a client pool, or clients with a semaphore")
        shared = semaphore
        lane = lambda position: (clients[position % len(clients)], shared)  # noqa: E731

    async def _one(position: int, symbol: str) -> CoverageEntry:
        if gate is None:
            client, lane_sem = lane(position)
            async with lane_sem:
                if now_fn() >= deadline:
                    return on_deadline(position, symbol)
                return await observe(position, symbol, client)
        while True:
            if now_fn() >= deadline:
                return on_deadline(position, symbol)
            await gate.wait_idle()
            client, lane_sem = lane(position)
            async with lane_sem:
                if not gate.is_idle:
                    continue
                if now_fn() >= deadline:
                    return on_deadline(position, symbol)
                return await observe(position, symbol, client)

    return list(
        await asyncio.gather(*(_one(position, symbol) for position, symbol in enumerate(roster)))
    )


def _log_round_line(
    dataset: CaptureDataset,
    *,
    scheduled_at: datetime,
    index: int,
    entries: Sequence[CoverageEntry],
    started: datetime,
    ended: datetime,
    capacity_per_round: int,
    active_keys: int | None = None,
) -> float:
    n_complete = sum(1 for entry in entries if entry.status is CaptureStatus.COMPLETE)
    n_deadline = sum(1 for entry in entries if entry.reason == "deadline")
    n_late = sum(1 for entry in entries if entry.reason == "deadline" and entry.first_event_time is not None)
    duration_s = max((ended - max(scheduled_at, started)).total_seconds(), 0.0)
    logger.info(
        "[DATA] stage=auction_capture_round phase=close dataset=%s round=%s index=%d n_roster=%d n_requested=%d "
        "n_complete=%d n_deadline=%d n_late=%d n_failed=%d duration_s=%.1f capacity_per_round=%d%s",
        dataset.value,
        scheduled_at.isoformat(),
        index,
        len(entries),
        sum(1 for entry in entries if entry.first_event_time is not None),
        n_complete,
        n_deadline,
        n_late,
        len(entries) - n_complete - n_deadline,
        duration_s,
        capacity_per_round,
        "" if active_keys is None else f" active_keys={active_keys}",
    )
    return duration_s


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
    pool: _ClientPool,
    store: CaptureStore,
    trading_day: date,
    run_id: str,
    cohort_id: str | None,
    reason_tag: str,
    close_at: datetime,
    now_fn: Callable[[], datetime],
    sleeper: Callable[[float], Awaitable[None]],
    injected_clock: bool,
    capacity_per_round: int,
    gate: _OrderbookPriorityGate | None = None,
) -> list[CoverageEntry]:
    """Capture program observations at their declared pre-close timestamps."""
    entries: list[CoverageEntry] = []
    for prog_index, scheduled_at in enumerate(rounds):
        await _wait_until_scheduled(
            scheduled_at, now_fn=now_fn, sleeper=sleeper, injected_clock=injected_clock
        )
        await _maybe_activate_handoff(pool, store, trading_day.isoformat(), round_at=scheduled_at)
        deadline = rounds[prog_index + 1] if prog_index + 1 < len(rounds) else close_at
        started = now_fn()
        round_entries = await _observe_roster(
            roster=roster,
            pool=pool,
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
            gate=gate,
        )
        entries.extend(round_entries)
        _log_round_line(
            CaptureDataset.PROGRAM,
            scheduled_at=scheduled_at,
            index=prog_index,
            entries=round_entries,
            started=started,
            ended=now_fn(),
            capacity_per_round=capacity_per_round,
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
    retries = 0
    while True:
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
        if ok:
            break
        msg_cd = str(body.get("msg_cd")) if body else "non_dict_payload"
        logger.warning(
            "[DATA] stage=auction_capture status=VENDOR_ERROR phase=%s symbol=%s scheduled=%s retry=%d rt_cd=%s msg_cd=%s msg=%s",
            phase,
            symbol,
            scheduled_at.isoformat(),
            retries,
            body.get("rt_cd") if body else "-",
            msg_cd,
            str(body.get("msg1", ""))[:80] if body else "-",
        )
        if retries >= _TRANSIENT_VENDOR_RETRIES or msg_cd not in _TRANSIENT_VENDOR_CODES or received >= deadline:
            break
        retries += 1
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
    confirmed_empty: set[str] | None = None,
) -> CoverageEntry:
    """Poll one symbol's opening price; unresolved symbols return a provisional entry for re-pass.

    `confirmed_empty` tracks symbols whose latest response was a valid vendor reply without an open price.
    """
    started_at = poll_started.setdefault(symbol, now_fn())
    payload = await client.get_current_price(session, symbol, market_div_code=KRX_CLOSE_MARKET_DIV_CODE)
    received = now_fn()
    body = dict(payload) if isinstance(payload, dict) else None
    price = _extract_open_price(body)
    if confirmed_empty is not None:
        if price <= 0 and body is not None and str(body.get("rt_cd")) == "0":
            confirmed_empty.add(symbol)
        else:
            confirmed_empty.discard(symbol)
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


def _unresolved_open_entry(
    symbol: str,
    *,
    floor: datetime,
    first_seen: datetime,
    last_seen_at: datetime,
    ref: Any,
    no_open_trade: bool,
) -> CoverageEntry:
    """Final entry of a symbol without an open price: NO_TRADES when the vendor kept answering that none printed, else PARTIAL."""
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.PRICE,
        venue="KRX",
        session="regular",
        scheduled_at=floor,
        status=CaptureStatus.NO_TRADES if no_open_trade else CaptureStatus.PARTIAL,
        rows=0,
        first_event_time=first_seen,
        last_event_time=last_seen_at,
        reason="no_open_trade" if no_open_trade else "open_unresolved",
        raw_refs=(ref,) if ref is not None else (),
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
    confirmed_empty: AbstractSet[str] = frozenset(),
) -> CoverageEntry:
    """Final entry for an open symbol never dispatched before confirmation end."""
    settled.add(position)
    moment = last_seen.get(symbol)
    seen_at = moment[1] if moment is not None else now_fn()
    return _unresolved_open_entry(
        symbol,
        floor=floor,
        first_seen=poll_started.get(symbol, seen_at),
        last_seen_at=seen_at,
        ref=moment[0] if moment is not None else None,
        no_open_trade=symbol in confirmed_empty,
    )


def _log_capture_plan(
    *,
    n_roster: int,
    n_extras: int,
    n_priority: int,
    n_clients: int,
    rounds: Sequence[datetime],
    program_instants: Sequence[datetime],
    interval_seconds: int,
    capacity_per_round: int,
    start_lag_seconds: float,
    n_handoff: int = 0,
    capacity_with_handoff: int = 0,
) -> dict[str, float | str]:
    """Log whether the closing plan fits the key budget and return its numbers; program demand is averaged over the rounds that can carry it."""
    demand_orderbook = n_roster + n_extras
    rounds_after_first_program = sum(1 for moment in rounds if program_instants and moment >= program_instants[0])
    demand_program = math.ceil(demand_orderbook * len(program_instants) / max(rounds_after_first_program, 1))
    overcommit = demand_orderbook + demand_program > capacity_per_round
    status = "OVERCOMMIT" if overcommit else "OK"
    (logger.warning if overcommit else logger.info)(
        "[DATA] stage=auction_capture_plan phase=close status=%s n_roster=%d n_extras=%d n_priority=%d n_clients=%d "
        "rounds=%d interval_s=%d capacity_per_round=%d demand_orderbook=%d demand_program_per_round=%d start_lag_s=%.1f "
        "handoff_slots=%d capacity_with_handoff=%d",
        status,
        n_roster,
        n_extras,
        n_priority,
        n_clients,
        len(rounds),
        interval_seconds,
        capacity_per_round,
        demand_orderbook,
        demand_program,
        start_lag_seconds,
        n_handoff,
        capacity_with_handoff,
    )
    return {
        "roster": n_roster,
        "extras": n_extras,
        "clients": n_clients,
        "capacity_per_round": capacity_per_round,
        "demand_per_round": demand_orderbook + demand_program,
        "status": status,
        "start_lag_s": start_lag_seconds,
    }


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
    degraded_slots: Sequence[str] = (),
    stats_sink: dict[str, float | str] | None = None,
    handoff_clients: Sequence[Any] = (),
    handoff_labels: Sequence[str] = (),
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
        previous_trading_day: Previous actual trading day for the opening population.
        degraded_slots: Slots whose token issuance failed before capture; any makes the manifest PARTIAL.
        stats_sink: Optional close-phase plan numbers and orderbook round duration, filled in place.
        handoff_clients: Decision-shard clients lent to research after the decision manifest; not prewarmed.
        handoff_labels: Log-safe slot labels aligned with ``handoff_clients``.

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
    if len(handoff_clients) != len(handoff_labels):
        raise ValueError("handoff clients and labels must align")
    if handoff_clients and not profile.COLLECTION_AUCTION_HANDOFF_SLOTS:
        raise ValueError("handoff clients require declared handoff slots")
    now = now_clock()
    roster, cohort_id, cohort_incomplete = _resolve_roster(snapshot_date, phase, store, now, previous_trading_day)
    roster_date = snapshot_date if phase == "close" else (previous_trading_day or _previous_trading_day(snapshot_date))
    extras = research_extras(
        store,
        roster_date,
        available_by=now,
        exclude=set(roster),
        enabled=profile.COLLECTION_RESEARCH_ROSTER_ENABLED,
    )
    dispatch = [*roster, *extras]
    research_set = frozenset(extras)
    priority: frozenset[str] = frozenset()
    capacity_per_round = 0
    capacity_with_handoff = 0
    if phase == "close":
        from src.api.kis.client import KIS_REST_TPS_PER_APP_KEY

        held = load_held_roster()
        if held.ok:
            priority = frozenset(held.symbols) & frozenset(dispatch)
        else:
            logger.warning(
                "[DATA] stage=auction_capture status=HELD_ROSTER_UNAVAILABLE reason=%s", held.failure_reason
            )
        capacity_per_round = estimate_round_capacity(
            len(clients),
            int(profile.COLLECTION_AUCTION_INTERVAL_SECONDS),
            utilization=profile.COLLECTION_AUCTION_CAPACITY_UTILIZATION,
            tps_per_key=KIS_REST_TPS_PER_APP_KEY,
        )
        capacity_with_handoff = estimate_round_capacity(
            len(clients) + len(handoff_clients),
            int(profile.COLLECTION_AUCTION_INTERVAL_SECONDS),
            utilization=profile.COLLECTION_AUCTION_CAPACITY_UTILIZATION,
            tps_per_key=KIS_REST_TPS_PER_APP_KEY,
        )
        rounds = close_rounds(session_clock, int(profile.COLLECTION_AUCTION_INTERVAL_SECONDS))
        phase_end = session_clock.close_at
    else:
        rounds = _open_rounds(session_clock)
        phase_end = session_clock.open_at + timedelta(seconds=int(profile.COLLECTION_OPEN_CONFIRM_SECONDS))
    run_id = f"auction-{phase}-{snapshot_date}-{uuid.uuid4().hex[:8]}"
    reason_tag = f"auction-{phase}"
    pool = _ClientPool(
        clients, handoff_clients, handoff_labels, per_key=int(profile.COLLECTION_CONCURRENCY_PER_KEY)
    )
    entries: list[CoverageEntry] = []
    incomplete = bool(cohort_incomplete) or bool(degraded_slots)
    if degraded_slots:
        logger.warning(
            "[DATA] stage=auction_capture status=DEGRADED reason=slot_token_failed slots=%s healthy=%d",
            ",".join(degraded_slots), len(clients),
        )
    first_client: Any = clients[0]
    max_round_duration_s = 0.0
    if phase == "close":
        plan = _log_capture_plan(
            n_roster=len(roster),
            n_extras=len(extras),
            n_priority=len(priority),
            n_clients=len(clients),
            rounds=rounds,
            program_instants=program_rounds(session_clock),
            interval_seconds=int(profile.COLLECTION_AUCTION_INTERVAL_SECONDS),
            capacity_per_round=capacity_per_round,
            start_lag_seconds=(now_clock() - rounds[0]).total_seconds(),
            n_handoff=len(handoff_clients),
            capacity_with_handoff=capacity_with_handoff,
        )
        if stats_sink is not None:
            stats_sink.update(plan)

    async with first_client.create_session() as broker_session:
        program_task: asyncio.Task[list[CoverageEntry]] | None = None
        gate = _OrderbookPriorityGate()
        if phase == "close":
            program_task = asyncio.create_task(
                _capture_program_rounds(
                    session=broker_session,
                    rounds=program_rounds(session_clock),
                    roster=dispatch,
                    pool=pool,
                    store=store,
                    trading_day=trading_day,
                    run_id=run_id,
                    cohort_id=cohort_id,
                    reason_tag=reason_tag,
                    close_at=session_clock.close_at,
                    now_fn=now_clock,
                    sleeper=sleeper,
                    injected_clock=injected_clock,
                    capacity_per_round=capacity_per_round,
                    gate=gate,
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
                    len(dispatch),
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
                    for symbol in dispatch
                )
                continue
            degraded: list[str] = []
            if phase == "close":
                await _maybe_activate_handoff(pool, store, snapshot_date, round_at=scheduled_at)
                ordered = plan_round_order(
                    dispatch,
                    priority=priority,
                    low_priority=research_set,
                    round_index=index,
                    round_count=len(rounds),
                )
                gate.begin_round()
            else:
                ordered = dispatch
            round_started = now_clock()
            try:
                round_entries = await _observe_roster(
                    roster=ordered,
                    pool=pool,
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
            finally:
                gate.end_round()
            entries.extend(round_entries)
            if phase == "close":
                max_round_duration_s = max(
                    max_round_duration_s,
                    _log_round_line(
                        CaptureDataset.ORDERBOOK,
                        scheduled_at=scheduled_at,
                        index=index,
                        entries=round_entries,
                        started=round_started,
                        ended=now_clock(),
                        capacity_per_round=capacity_per_round,
                        active_keys=pool.active_count() if pool.handoff_configured else None,
                    ),
                )
            if any(symbol not in research_set for symbol in degraded) or any(
                entry.status is not CaptureStatus.COMPLETE
                for entry in round_entries
                if entry.symbol not in research_set
            ):
                incomplete = True
        if phase == "close":
            if stats_sink is not None:
                stats_sink["max_round_duration_s"] = max_round_duration_s
            if program_task is not None:
                program_entries = await program_task
                entries.extend(program_entries)
                if any(
                    entry.status not in GOOD_ENTRY_STATES
                    for entry in program_entries
                    if entry.symbol not in research_set
                ):
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
            pending = list(enumerate(dispatch))
            poll_started: dict[str, datetime] = {}
            last_seen: dict[str, tuple[Any, datetime]] = {}
            confirmed_empty: set[str] = set()
            attempt = 0
            while pending and now_clock() <= confirm_end:
                pass_symbols = [symbol for _, symbol in pending]
                settled: set[int] = set()
                pass_degraded: list[str] = []
                round_entries = await _observe_roster(
                    roster=pass_symbols,
                    pool=pool,
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
                        confirmed_empty=confirmed_empty,
                    ),
                    on_deadline=functools.partial(
                        _open_deadline_entry,
                        floor=floor,
                        now_fn=now_clock,
                        poll_started=poll_started,
                        last_seen=last_seen,
                        settled=settled,
                        confirmed_empty=confirmed_empty,
                    ),
                )
                next_pending: list[tuple[int, str]] = []
                for pass_index, ((orig_position, symbol), entry) in enumerate(zip(pending, round_entries)):
                    if pass_index in settled:
                        entries.append(entry)
                    else:
                        next_pending.append((orig_position, symbol))
                if any(symbol not in research_set for symbol in pass_degraded) or any(
                    entry.status not in GOOD_ENTRY_STATES
                    for pass_index, entry in enumerate(round_entries)
                    if pass_index in settled and entry.symbol not in research_set
                ):
                    incomplete = True
                pending = next_pending
                if pending:
                    attempt += 1
                    await sleeper(1.0)
            for _, symbol in pending:
                seen = last_seen.get(symbol)
                moment = seen[1] if seen is not None else now_clock()
                entry = _unresolved_open_entry(
                    symbol,
                    floor=floor,
                    first_seen=poll_started.get(symbol, moment),
                    last_seen_at=moment,
                    ref=seen[0] if seen is not None else None,
                    no_open_trade=symbol in confirmed_empty,
                )
                if symbol not in research_set and entry.status not in GOOD_ENTRY_STATES:
                    incomplete = True
                entries.append(entry)
    primary, research = split_research_entries(entries, research_set)
    status = CaptureStatus.COMPLETE
    for item in primary:
        if item.status not in GOOD_ENTRY_STATES:
            status = CaptureStatus.PARTIAL
            break
    if incomplete and status == CaptureStatus.COMPLETE:
        status = CaptureStatus.PARTIAL
    completed_at = now_clock()
    context = CaptureContext(
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
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=context,
        cohort=None,
        completed_at=completed_at,
        entries=tuple(primary),
        artifacts=tuple(ref for item in primary for ref in item.raw_refs),
        status=status,
    )
    try:
        store.publish_manifest(manifest)
        if research:
            store.publish_manifest(
                research_manifest(
                    context,
                    research,
                    completed_at=completed_at,
                    force_partial=bool(degraded_slots),
                )
            )
    except OSError as exc:
        raise OSError("required auction coverage persistence fails") from exc
    return manifest


async def _run_async(
    snapshot_date: str,
    phase: str,
    profile: CollectionSettings,
    stats_sink: dict[str, float | str] | None = None,
) -> CaptureManifest | None:
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

    def _client(cred: KisCredential) -> Any:
        return KisApiClient(
            cred.app_key,
            cred.app_secret,
            "",
            cred.hts_id,
            token_file=str(token_cache_path(cred.app_key, settings.KIS_TOKEN_CACHE_DIR)),
        )

    env = dict(os.environ)
    creds = resolve_research_credentials(env, slots=tuple(profile.COLLECTION_RESEARCH_SLOTS))
    clients = [_client(cred) for cred in creds]
    handoff_creds: tuple[KisCredential, ...] = ()
    if profile.COLLECTION_AUCTION_HANDOFF_SLOTS:
        try:
            handoff_creds = resolve_handoff_credentials(
                env,
                slots=tuple(profile.COLLECTION_AUCTION_HANDOFF_SLOTS),
                research_slots=tuple(profile.COLLECTION_RESEARCH_SLOTS),
            )
        except ValueError:
            logger.warning("[DATA] stage=auction_capture status=HANDOFF_DISABLED reason=config")
    handoff_clients = [_client(cred) for cred in handoff_creds]
    handoff_labels = tuple(cred.slot for cred in handoff_creds)
    store = CaptureStore(_capture_root(profile))
    trading_day = date.fromisoformat(snapshot_date)
    clock = session_day.clock if session_day.clock is not None else SessionClock.standard(trading_day)
    previous_trading_day: str | None = None
    async with clients[0].create_session() as broker_session:
        clients, degraded_slots = await prewarm_healthy_clients(clients, creds, broker_session)
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
        degraded_slots=degraded_slots,
        stats_sink=stats_sink,
        handoff_clients=handoff_clients,
        handoff_labels=handoff_labels,
    )


def _close_entry_counts(entries: Sequence[CoverageEntry]) -> dict[str, int]:
    deadline = [entry for entry in entries if entry.reason == "deadline"]
    rounds_missed: dict[str | None, set[datetime | None]] = collections.defaultdict(set)
    for entry in deadline:
        rounds_missed[entry.symbol].add(entry.scheduled_at)
    return {
        "n_deadline": len(deadline),
        "n_late": sum(1 for entry in deadline if entry.first_event_time is not None),
        "n_failed": sum(1 for entry in entries if entry.status is not CaptureStatus.COMPLETE and entry.reason != "deadline"),
        "rounds_with_deadline": len({entry.scheduled_at for entry in deadline}),
        "symbols_missing_any": len(rounds_missed),
        "max_missed_rounds_per_symbol": max((len(rounds) for rounds in rounds_missed.values()), default=0),
    }


def _close_degraded_reason(counts: dict[str, int]) -> str:
    if counts["n_deadline"] and counts["n_failed"]:
        return "mixed"
    if counts["n_deadline"]:
        return "deadline"
    if counts["n_failed"]:
        return "vendor_failure"
    return ""


def _record_close_outcome(manifest: CaptureManifest, stats: dict[str, float | str], snapshot_date: str) -> None:
    """Append the close verdict and its numbers to the run event log; a recording fault never fails the capture."""
    try:
        complete = manifest.status is CaptureStatus.COMPLETE
        counts = _close_entry_counts(manifest.entries)
        record_run_outcome(
            "auction_close",
            RUN_OUTCOME_OK if complete else RUN_OUTCOME_DEGRADED,
            run_date=snapshot_date,
            reason="" if complete else _close_degraded_reason(counts),
            metrics={**stats, **counts},
        )
    except Exception:
        logger.warning(
            "[DATA] stage=auction_capture status=OUTCOME_RECORD_FAILED phase=close date=%s",
            snapshot_date,
            exc_info=True,
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
    stats: dict[str, float | str] = {}
    manifest = asyncio.run(_run_async(snapshot_date, args.phase, profile, stats_sink=stats))
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
    not_complete = [entry for entry in manifest.entries if entry.status is not CaptureStatus.COMPLETE]
    if not_complete:
        reasons = collections.Counter(f"{entry.dataset.value}:{entry.status.value}:{entry.reason}" for entry in not_complete)
        sample = ",".join(f"{entry.symbol}@{entry.scheduled_at:%H:%M:%S}" for entry in not_complete[:5])
        logger.warning(
            "[DATA] stage=auction_capture status=NOT_COMPLETE_DETAIL phase=%s date=%s n=%d reasons=%s sample=%s",
            args.phase,
            snapshot_date,
            len(not_complete),
            dict(reasons),
            sample,
        )
    if args.phase == "close":
        _record_close_outcome(manifest, stats, snapshot_date)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    # 실측: __main__ 가드가 없어 `python -m`으로 실행해도 main()이 전혀 호출되지 않고
    # 조용히 성공 종료(exit 0)했다 -- kca-auction-open/kca-auction-close가 매번
    # "성공"으로 보이면서 실제로는 아무 것도 수집하지 않았다(2026-09-21 daily-audit이
    # collection:auction_open:1:missing_manifest / auction_close:*:missing_entries로 포착).
    configure_cli_logging()
    main()
