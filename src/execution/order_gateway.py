"""Idempotent gateway turning OrderIntents into at most one broker order."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from src.api.kis.trading import BrokerReadError
from src.execution.order_contract import (
    BrokerOrder,
    BrokerPosition,
    OrderIntent,
    OrderPhase,
    OrderSide,
    SubmitResult,
    SubmitVerdict,
    normalize_order_no,
)
from src.execution.order_journal import (
    IntentState,
    IntentStatus,
    JournalCorruptError,
    JournalEvent,
    JournalEventKind,
    OrderJournal,
    derive_intent_state,
    intent_to_payload,
)

__all__ = [
    "BrokerPort",
    "GatewayLimits",
    "OrderGateway",
    "PlannedSkip",
    "ReconcileReport",
]

logger = logging.getLogger(__name__)


class BrokerPort(Protocol):
    """Broker operations the gateway needs (implemented by KisTradingClient)."""

    async def submit_order(self, session: Any, intent: OrderIntent) -> SubmitResult: ...
    async def cancel_order(self, session: Any, *, org_no: str, order_no: str) -> SubmitResult: ...
    async def fetch_daily_orders(self, session: Any, trading_date: str) -> list[BrokerOrder]: ...
    async def fetch_positions(self, session: Any) -> list[BrokerPosition]: ...


@dataclass(frozen=True)
class GatewayLimits:
    """Risk and retry limits of one gateway run."""

    seed_capital_krw: int
    max_notional_per_order_krw: int
    max_sends_per_run: int
    send_max_attempts: int
    unknown_resolve_polls: int
    unknown_resolve_interval_seconds: float
    exit_resend_on_ambiguous: bool


@dataclass(frozen=True)
class PlannedSkip:
    """A pick that will not be ordered, recorded so every pick leaves exactly one terminal trace."""

    intent_id: str
    symbol: str
    trading_date: str
    phase: OrderPhase
    reason: str


@dataclass(frozen=True)
class ReconcileReport:
    """Summary of one reconciliation pass."""

    trading_date: str
    n_intents: int
    n_filled: int
    n_partial: int
    n_unfilled: int
    n_working: int
    n_unknown: int
    n_missing_at_broker: int
    n_foreign_orders: int
    position_shortfalls: tuple[tuple[str, int, int], ...]
    filled_amount_krw: int
    n_unknown_unresolved_total: int = 0
    unknown_exposures: tuple[tuple[str, int, int], ...] = ()


_SENDABLE: frozenset[IntentStatus] = frozenset({IntentStatus.RECORDED, IntentStatus.NOT_SENT})


def _notional(intent: OrderIntent) -> int:
    price = intent.limit_price if intent.limit_price is not None else (intent.decision_price or 0)
    return int(intent.qty) * int(price or 0)


class OrderGateway:
    """Durable gateway binding one broker, one journal and one run id."""

    def __init__(
        self,
        broker: BrokerPort,
        journal: OrderJournal,
        limits: GatewayLimits,
        *,
        run_id: str,
        now_fn: Callable[[], datetime],
        sleep_fn: Callable[[float], Awaitable[None]],
    ) -> None:
        """Bind a gateway to one broker, one journal and one run id."""
        self._broker = broker
        self._journal = journal
        self._limits = limits
        self._run_id = run_id
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn
        self._sends_made = 0

    def _require_lock(self) -> None:
        if not self._journal.locked:
            raise RuntimeError("OrderGateway requires journal exclusive()")

    async def _read_baseline(self, session: Any, trading_date: str) -> list[str] | None:
        """Return normalized same-day broker order numbers, or None when the read fails."""
        try:
            snapshot = await self._broker.fetch_daily_orders(session, trading_date)
        except Exception:
            logger.exception(
                "[EXEC] stage=submit trading_date=%s run_id=%s baseline_read_failed",
                trading_date, self._run_id,
            )
            return None
        out: list[str] = []
        for order in snapshot:
            normalized = normalize_order_no(order.order_no)
            if normalized:
                out.append(normalized)
        return out

    def plan(self, intents: Sequence[OrderIntent], skips: Sequence[PlannedSkip]) -> None:
        """Durably record every intent and skip of a run before any send."""
        self._require_lock()
        recorded: dict[str, OrderIntent | None] = {
            key: state.intent for key, state in self._journal.states().items()
        }
        for intent in intents:
            if intent.intent_id in recorded:
                stored = recorded[intent.intent_id]
                if stored is not None and stored != intent:
                    logger.warning(
                        "[RISK] stage=plan intent_id=%s run_id=%s stored_intent_wins offered_qty=%s",
                        intent.intent_id, self._run_id, intent.qty,
                    )
                continue
            self._journal.append(JournalEventKind.INTENT, intent.intent_id,
                                 intent_to_payload(intent), run_id=self._run_id)
            logger.info("[EXEC] stage=plan intent_id=%s run_id=%s kind=intent", intent.intent_id, self._run_id)
            recorded[intent.intent_id] = intent
        for skip in skips:
            if skip.intent_id in recorded:
                continue
            self._journal.append(
                JournalEventKind.SKIP, skip.intent_id,
                {"reason": skip.reason, "symbol": skip.symbol,
                 "trading_date": skip.trading_date, "phase": skip.phase.value},
                run_id=self._run_id,
            )
            logger.info("[EXEC] stage=plan intent_id=%s run_id=%s kind=skip", skip.intent_id, self._run_id)
            recorded[skip.intent_id] = None

    async def submit(self, session: Any, intent_id: str, *, deadline: datetime) -> IntentState:
        """Drive one recorded intent to at most one broker order and return its state."""
        self._require_lock()
        states = self._journal.states()
        if intent_id not in states:
            raise KeyError(intent_id)
        state = states[intent_id]
        if state.status not in _SENDABLE:
            return state
        assert state.intent is not None
        intent = state.intent
        if self._journal.is_halted():
            self._journal.append(JournalEventKind.SKIP, intent_id, self._skip_payload(intent, "halted"),
                                 run_id=self._run_id)
            logger.warning("[RISK] stage=submit intent_id=%s run_id=%s refusal=halted", intent_id, self._run_id)
            return self._journal.states()[intent_id]
        if state.n_sends == 0:
            refusal = self._risk_refusal(intent)
            if refusal is not None:
                self._journal.append(JournalEventKind.SKIP, intent_id, self._skip_payload(intent, refusal),
                                     run_id=self._run_id)
                logger.warning("[RISK] stage=submit intent_id=%s run_id=%s refusal=%s",
                               intent_id, self._run_id, refusal)
                return self._journal.states()[intent_id]
        baseline: list[str] | None = None
        baseline_pending = False
        if state.n_sends == 0:
            baseline = await self._read_baseline(session, intent.trading_date)
            baseline_pending = baseline is not None
            if self._journal.is_halted():
                self._journal.append(JournalEventKind.SKIP, intent_id, self._skip_payload(intent, "halted"),
                                     run_id=self._run_id)
                logger.warning("[RISK] stage=submit intent_id=%s run_id=%s refusal=halted",
                               intent_id, self._run_id)
                return self._journal.states()[intent_id]
        for _ in range(max(1, self._limits.send_max_attempts)):
            if self._now_fn() >= deadline:
                fresh = self._journal.states()[intent_id]
                if fresh.n_sends == 0:
                    self._journal.append(JournalEventKind.SKIP, intent_id,
                                         self._skip_payload(intent, "deadline_passed"), run_id=self._run_id)
                    logger.warning("[RISK] stage=submit intent_id=%s run_id=%s refusal=deadline_passed",
                                   intent_id, self._run_id)
                    return self._journal.states()[intent_id]
                return fresh
            attempt = self._journal.states()[intent_id].n_sends + 1
            send_payload: dict[str, Any] = {"attempt": attempt}
            if baseline_pending:
                assert baseline is not None
                send_payload["baseline_order_nos"] = list(baseline)
                baseline_pending = False
            self._journal.append(JournalEventKind.SEND, intent_id, send_payload, run_id=self._run_id)
            logger.info("[EXEC] stage=submit intent_id=%s run_id=%s kind=send attempt=%s",
                        intent_id, self._run_id, attempt)
            result = await self._safe_submit(session, intent)
            self._sends_made += 1
            self._journal.append(
                JournalEventKind.RESULT, intent_id,
                {"attempt": attempt, "verdict": result.verdict.value,
                 "broker_order_no": result.broker_order_no, "broker_org_no": result.broker_org_no,
                 "http_status": result.http_status, "msg_cd": result.msg_cd,
                 "msg": result.msg, "latency_ms": result.latency_ms},
                run_id=self._run_id,
            )
            logger.info("[EXEC] stage=submit intent_id=%s run_id=%s kind=result verdict=%s",
                        intent_id, self._run_id, result.verdict.value)
            if result.verdict in (SubmitVerdict.ACCEPTED, SubmitVerdict.REJECTED):
                return self._journal.states()[intent_id]
            if result.verdict in (SubmitVerdict.THROTTLED, SubmitVerdict.AUTH_REJECTED, SubmitVerdict.NOT_SENT):
                continue
            if self._was_not_found(intent_id):
                return self._journal.states()[intent_id]
            return await self._resolve_unknown(session, intent, deadline)
        return self._journal.states()[intent_id]

    async def reconcile(self, session: Any, trading_date: str, *, finalize: bool) -> ReconcileReport:
        """Bind broker order state to journal intents and return the summary."""
        self._require_lock()
        snapshot = await self._broker.fetch_daily_orders(session, trading_date)
        states = self._journal.states()
        day_ids = [key for key, st in states.items() if self._state_date(st) == trading_date and st.intent is not None]
        bound: set[str] = set()
        for st in states.values():
            if st.broker_order_no:
                bound.add(normalize_order_no(st.broker_order_no))
        for intent_id in day_ids:
            st = self._journal.states()[intent_id]
            if st.status is not IntentStatus.UNKNOWN or st.intent is None:
                continue
            candidate = self._candidates(st.intent, snapshot, bound, baseline=st.baseline_order_nos)
            if len(candidate) == 1:
                order = candidate[0]
                self._journal.append(JournalEventKind.ADOPT, intent_id,
                                     {"broker_order_no": order.order_no, "broker_org_no": order.org_no},
                                     run_id=self._run_id)
                logger.info("[EXEC] stage=reconcile intent_id=%s run_id=%s kind=adopt order_no=%s",
                            intent_id, self._run_id, order.order_no)
                bound.add(normalize_order_no(order.order_no))
            elif len(candidate) > 1:
                self._journal.append(JournalEventKind.RESOLVE_CONFLICT, intent_id,
                                     {"n_candidates": len(candidate)}, run_id=self._run_id)
                logger.info("[EXEC] stage=reconcile intent_id=%s run_id=%s kind=resolve_conflict n=%s",
                            intent_id, self._run_id, len(candidate))
        states = self._journal.states()
        history: dict[str, list[JournalEvent]] = {}
        for event in self._journal.load():
            history.setdefault(event.intent_id, []).append(event)
        contradictions: list[str] = []
        for intent_id in day_ids:
            st = states[intent_id]
            if st.broker_order_no is None or st.final:
                continue
            target = normalize_order_no(st.broker_order_no)
            found: BrokerOrder | None = None
            for order in snapshot:
                if normalize_order_no(order.order_no) == target and target:
                    found = order
                    break
            if found is None:
                continue
            final = bool(found.remaining_qty == 0 or found.cancelled
                         or found.rejected_qty == found.order_qty or finalize)
            payload = {"order_qty": found.order_qty, "filled_qty": found.filled_qty,
                       "filled_amount_krw": found.filled_amount_krw, "remaining_qty": found.remaining_qty,
                       "cancelled": found.cancelled, "rejected_qty": found.rejected_qty, "final": final}
            events = history[intent_id]
            prior = next((dict(e.payload) for e in reversed(events)
                          if e.kind is JournalEventKind.BROKER_STATE), None)
            if prior == payload:
                continue
            # A broker row violating the fill invariants would make every later load raise; keep it out.
            probe = JournalEvent(seq=events[-1].seq + 1, ts="", kind=JournalEventKind.BROKER_STATE,
                                 intent_id=intent_id, run_id=self._run_id, payload=payload)
            try:
                derive_intent_state([*events, probe])
            except JournalCorruptError as exc:
                logger.error("[RISK] stage=reconcile intent_id=%s run_id=%s refusal=broker_state_contradiction "
                             "order_no=%s detail=%s", intent_id, self._run_id, found.order_no, exc)
                contradictions.append(intent_id)
                continue
            self._journal.append(JournalEventKind.BROKER_STATE, intent_id, payload, run_id=self._run_id)
            logger.info("[EXEC] stage=reconcile intent_id=%s run_id=%s kind=broker_state filled=%s final=%s",
                        intent_id, self._run_id, found.filled_qty, final)
        if contradictions:
            raise BrokerReadError(f"broker order state contradicts journal intents: {sorted(contradictions)}")
        states = self._journal.states()
        bound_final: set[str] = set()
        for st in states.values():
            if st.broker_order_no:
                bound_final.add(normalize_order_no(st.broker_order_no))
        n_foreign = sum(1 for o in snapshot if normalize_order_no(o.order_no) not in bound_final)
        positions = await self._broker.fetch_positions(session)
        holding: dict[str, int] = {p.symbol: p.holding_qty for p in positions}
        shortfalls: list[tuple[str, int, int]] = []
        journal_qty: dict[str, int] = {}
        for lot in self._journal.open_lots():
            journal_qty[lot.symbol] = journal_qty.get(lot.symbol, 0) + lot.qty
        for symbol, qty in sorted(journal_qty.items()):
            broker_qty = holding.get(symbol, 0)
            if broker_qty < qty:
                shortfalls.append((symbol, qty, broker_qty))
        states = self._journal.states()
        filled_amount = sum(st.filled_amount_krw for key, st in states.items()
                            if self._state_date(st) == trading_date)
        n_unknown_unresolved_total = sum(
            1 for st in states.values() if st.intent is not None and st.status is IntentStatus.UNKNOWN
        )
        unknown_buy: dict[str, int] = {}
        for st in states.values():
            if st.intent is None or st.status is not IntentStatus.UNKNOWN:
                continue
            if st.intent.side is not OrderSide.BUY:
                continue
            unknown_buy[st.intent.symbol] = unknown_buy.get(st.intent.symbol, 0) + st.intent.qty
        exposures: list[tuple[str, int, int]] = []
        for symbol in sorted(unknown_buy):
            excess = holding.get(symbol, 0) - journal_qty.get(symbol, 0)
            if excess > 0:
                exposures.append((symbol, unknown_buy[symbol], excess))
        day_states = [states[k] for k in day_ids]
        n_missing = 0
        broker_nos = {normalize_order_no(o.order_no) for o in snapshot}
        for st in day_states:
            if st.broker_order_no and not st.final and normalize_order_no(st.broker_order_no) not in broker_nos:
                n_missing += 1
        return ReconcileReport(
            trading_date=trading_date,
            n_intents=len(day_ids),
            n_filled=sum(1 for s in day_states if s.status is IntentStatus.FILLED),
            n_partial=sum(1 for s in day_states if s.status is IntentStatus.PARTIAL),
            n_unfilled=sum(1 for s in day_states if s.status is IntentStatus.UNFILLED),
            n_working=sum(1 for s in day_states if s.status is IntentStatus.ACKED and not s.final),
            n_unknown=sum(1 for s in day_states if s.status is IntentStatus.UNKNOWN),
            n_missing_at_broker=n_missing,
            n_foreign_orders=n_foreign,
            position_shortfalls=tuple(shortfalls),
            filled_amount_krw=filled_amount,
            n_unknown_unresolved_total=n_unknown_unresolved_total,
            unknown_exposures=tuple(exposures),
        )

    async def cancel_working(self, session: Any, trading_date: str) -> int:
        """Cancel every acked, non-final order of trading_date."""
        self._require_lock()
        states = self._journal.states()
        targets = [key for key, st in states.items()
                   if self._state_date(st) == trading_date
                   and st.status is IntentStatus.ACKED and not st.final
                   and st.broker_order_no]
        prior_sends: dict[str, int] = {}
        for event in self._journal.load():
            if event.kind is JournalEventKind.CANCEL_SEND:
                prior_sends[event.intent_id] = prior_sends.get(event.intent_id, 0) + 1
        accepted = 0
        for intent_id in targets:
            st = states[intent_id]
            assert st.broker_order_no is not None
            attempt = prior_sends.get(intent_id, 0) + 1
            self._journal.append(JournalEventKind.CANCEL_SEND, intent_id, {"attempt": attempt},
                                 run_id=self._run_id)
            logger.info("[EXEC] stage=cancel intent_id=%s run_id=%s kind=cancel_send", intent_id, self._run_id)
            try:
                result = await self._broker.cancel_order(
                    session, org_no=st.broker_org_no or "", order_no=st.broker_order_no)
            except Exception:
                logger.exception("[EXEC] stage=cancel intent_id=%s run_id=%s cancel_raised",
                                 intent_id, self._run_id)
                result = SubmitResult(verdict=SubmitVerdict.AMBIGUOUS, broker_order_no=None,
                                      broker_org_no=None, http_status=None,
                                      msg_cd="", msg="cancel raised", latency_ms=0.0)
            self._journal.append(
                JournalEventKind.CANCEL_RESULT, intent_id,
                {"attempt": attempt, "verdict": result.verdict.value,
                 "msg_cd": result.msg_cd, "http_status": result.http_status},
                run_id=self._run_id,
            )
            logger.info("[EXEC] stage=cancel intent_id=%s run_id=%s kind=cancel_result verdict=%s",
                        intent_id, self._run_id, result.verdict.value)
            if result.verdict is SubmitVerdict.ACCEPTED:
                accepted += 1
        return accepted

    def _skip_payload(self, intent: OrderIntent, reason: str) -> dict[str, Any]:
        return {"reason": reason, "symbol": intent.symbol,
                "trading_date": intent.trading_date, "phase": intent.phase.value}

    def _risk_refusal(self, intent: OrderIntent) -> str | None:
        if intent.side is OrderSide.BUY:
            if _notional(intent) > self._limits.max_notional_per_order_krw:
                return "notional_cap"
            states = self._journal.states()
            total = _notional(intent)
            for st in states.values():
                other = st.intent
                if other is None or other.side is not OrderSide.BUY:
                    continue
                if other.trading_date != intent.trading_date or other.intent_id == intent.intent_id:
                    continue
                if st.status in (IntentStatus.ACKED, IntentStatus.UNKNOWN,
                                 IntentStatus.FILLED, IntentStatus.PARTIAL):
                    total += _notional(other)
            if total > self._limits.seed_capital_krw:
                return "seed_cap"
        else:
            parent = str(intent.parent_intent_id or "")
            lot_qty = 0
            for lot in self._journal.open_lots():
                if lot.entry_intent_id == parent:
                    lot_qty += lot.qty
            if intent.qty > lot_qty:
                return "exceeds_open_lot"
        if self._sends_made >= self._limits.max_sends_per_run:
            return "run_send_cap"
        return None

    async def _safe_submit(self, session: Any, intent: OrderIntent) -> SubmitResult:
        try:
            return await self._broker.submit_order(session, intent)
        except Exception:
            logger.exception("[EXEC] stage=submit intent_id=%s run_id=%s submit_raised",
                             intent.intent_id, self._run_id)
            return SubmitResult(verdict=SubmitVerdict.AMBIGUOUS, broker_order_no=None,
                                broker_org_no=None, http_status=None,
                                msg_cd="", msg="submit raised", latency_ms=0.0)

    def _was_not_found(self, intent_id: str) -> bool:
        return any(e.intent_id == intent_id and e.kind is JournalEventKind.NOT_FOUND
                   for e in self._journal.load())

    def _candidates(
        self, intent: OrderIntent, snapshot: Sequence[BrokerOrder], bound: set[str],
        *, baseline: frozenset[str] | None = None,
    ) -> list[BrokerOrder]:
        out: list[BrokerOrder] = []
        for order in snapshot:
            normalized_no = normalize_order_no(order.order_no)
            if baseline is not None and (not normalized_no or normalized_no in baseline):
                continue
            if normalized_no in bound:
                continue
            if order.symbol != intent.symbol or order.side != intent.side:
                continue
            if order.order_qty != intent.qty:
                continue
            if (
                intent.order_type.value == "limit"
                and intent.limit_price is not None
                and order.order_price != intent.limit_price
            ):
                continue
            out.append(order)
        return out

    async def _resolve_unknown(self, session: Any, intent: OrderIntent, deadline: datetime) -> IntentState:
        polls = self._limits.unknown_resolve_polls
        interval = max(0.0, self._limits.unknown_resolve_interval_seconds)
        states = self._journal.states()
        if polls <= 0:
            # Zero polls prove nothing about absence, so NOT_FOUND (and the exit resend it unlocks) is off the table.
            return states[intent.intent_id]
        bound: set[str] = set()
        for st in states.values():
            if st.broker_order_no:
                bound.add(normalize_order_no(st.broker_order_no))
        read_failed = False
        for index in range(polls):
            try:
                snapshot = await self._broker.fetch_daily_orders(session, intent.trading_date)
            except Exception:
                logger.exception("[EXEC] stage=submit intent_id=%s run_id=%s resolve_read_failed",
                                 intent.intent_id, self._run_id)
                read_failed = True
                break
            candidates = self._candidates(
                intent, snapshot, bound,
                baseline=self._journal.states()[intent.intent_id].baseline_order_nos,
            )
            if len(candidates) == 1:
                order = candidates[0]
                self._journal.append(JournalEventKind.ADOPT, intent.intent_id,
                                     {"broker_order_no": order.order_no, "broker_org_no": order.org_no},
                                     run_id=self._run_id)
                logger.info("[EXEC] stage=submit intent_id=%s run_id=%s kind=adopt order_no=%s",
                            intent.intent_id, self._run_id, order.order_no)
                return self._journal.states()[intent.intent_id]
            if len(candidates) > 1:
                self._journal.append(JournalEventKind.RESOLVE_CONFLICT, intent.intent_id,
                                     {"n_candidates": len(candidates)}, run_id=self._run_id)
                logger.info("[EXEC] stage=submit intent_id=%s run_id=%s kind=resolve_conflict n=%s",
                            intent.intent_id, self._run_id, len(candidates))
                return self._journal.states()[intent.intent_id]
            if index < polls - 1:
                await self._sleep_fn(interval)
        if read_failed:
            return self._journal.states()[intent.intent_id]
        span = float(max(0, polls - 1)) * float(interval)
        self._journal.append(JournalEventKind.NOT_FOUND, intent.intent_id,
                             {"n_polls": polls, "span_seconds": span}, run_id=self._run_id)
        logger.info("[EXEC] stage=submit intent_id=%s run_id=%s kind=not_found polls=%s",
                    intent.intent_id, self._run_id, polls)
        after = self._journal.states()[intent.intent_id]
        if intent.phase is OrderPhase.EXIT and self._limits.exit_resend_on_ambiguous:
            if self._journal.is_halted():
                logger.warning("[RISK] stage=submit intent_id=%s run_id=%s refusal=halted resend=exit",
                               intent.intent_id, self._run_id)
                return after
            try:
                positions = await self._broker.fetch_positions(session)
            except Exception:
                logger.exception("[EXEC] stage=submit intent_id=%s run_id=%s resend_positions_read_failed",
                                 intent.intent_id, self._run_id)
                return after
            sellable = 0
            for pos in positions:
                if pos.symbol == intent.symbol:
                    sellable = pos.sellable_qty
                    break
            if sellable >= intent.qty and self._now_fn() < deadline:
                attempt = self._journal.states()[intent.intent_id].n_sends + 1
                self._journal.append(JournalEventKind.SEND, intent.intent_id,
                                     {"attempt": attempt}, run_id=self._run_id)
                logger.info("[EXEC] stage=submit intent_id=%s run_id=%s kind=send attempt=%s resend=exit",
                            intent.intent_id, self._run_id, attempt)
                result = await self._safe_submit(session, intent)
                self._sends_made += 1
                self._journal.append(
                    JournalEventKind.RESULT, intent.intent_id,
                    {"attempt": attempt, "verdict": result.verdict.value,
                     "broker_order_no": result.broker_order_no, "broker_org_no": result.broker_org_no,
                     "http_status": result.http_status, "msg_cd": result.msg_cd,
                     "msg": result.msg, "latency_ms": result.latency_ms},
                    run_id=self._run_id,
                )
                logger.info("[EXEC] stage=submit intent_id=%s run_id=%s kind=result verdict=%s resend=exit",
                            intent.intent_id, self._run_id, result.verdict.value)
        return self._journal.states()[intent.intent_id]

    def _state_date(self, state: IntentState) -> str | None:
        if state.intent is not None:
            return state.intent.trading_date
        return None
