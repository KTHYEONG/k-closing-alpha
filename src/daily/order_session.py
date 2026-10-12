"""Gated real-order session (entry/exit/reconcile/cancel) over the durable order journal."""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import math
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from src.api.kis.trading import (
    BrokerReadError,
    KisTradingClient,
    TradingDisabledError,
    resolve_trading_binding,
)
from src.config import settings
from src.config.market_session import (
    DECISION_WINDOW_END_HHMMSS,
    ORDER_ENTRY_DECISION_WAIT_DEADLINE_HHMMSS,
    ORDER_ENTRY_EARLIEST_HHMMSS,
    ORDER_ENTRY_RECONCILE_DEADLINE_HHMMSS,
    ORDER_ENTRY_RECONCILE_EARLIEST_HHMMSS,
    ORDER_ENTRY_SUBMIT_DEADLINE_HHMMSS,
    ORDER_EXIT_EARLIEST_HHMMSS,
    ORDER_EXIT_RECONCILE_DEADLINE_HHMMSS,
    ORDER_EXIT_RECONCILE_EARLIEST_HHMMSS,
    ORDER_EXIT_SUBMIT_DEADLINE_HHMMSS,
)
from src.daily.predict import load_topk_decision
from src.data.session_calendar import SessionDay, resolve_session_day, trading_session_gate
from src.execution.order_contract import (
    EntryPick,
    OrderIntent,
    OrderPhase,
    OrderSide,
    OrderType,
    PriceLimits,
    entry_intent_id,
    entry_limit_price,
    exit_intent_id,
    size_entry_orders,
)
from src.execution.order_gateway import BrokerPort, GatewayLimits, OrderGateway, PlannedSkip, ReconcileReport
from src.execution.order_journal import IntentStatus, OpenLot, OrderJournal
from src.tools.run_outcome import (
    RUN_OUTCOME_DEGRADED,
    RUN_OUTCOME_OK,
    RUN_OUTCOME_SKIPPED,
    record_run_outcome,
)
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

ORDER_JOB_ENTRY: str = "order_entry"
ORDER_JOB_EXIT: str = "order_exit"
ORDER_JOB_RECONCILE_ENTRY: str = "order_reconcile_entry"
ORDER_JOB_RECONCILE_EXIT: str = "order_reconcile_exit"
ORDER_JOB_CANCEL: str = "order_cancel"
ORDER_PHASES: tuple[str, ...] = ("entry", "exit", "reconcile-entry", "reconcile-exit", "cancel")
ORDER_DECISION_POLL_SECONDS: float = 2.0
ORDER_RECONCILE_POLL_SECONDS: float = 10.0
ORDER_MAX_PRESTART_WAIT_SECONDS: float = 120.0

_JOB_BY_PHASE: dict[str, str] = {
    "entry": ORDER_JOB_ENTRY,
    "exit": ORDER_JOB_EXIT,
    "reconcile-entry": ORDER_JOB_RECONCILE_ENTRY,
    "reconcile-exit": ORDER_JOB_RECONCILE_EXIT,
    "cancel": ORDER_JOB_CANCEL,
}

_KST = ZoneInfo("Asia/Seoul")
_SYMBOL_RE = re.compile(r"^\d{6}$")
_OK_SEND_STATUSES: frozenset[IntentStatus] = frozenset({IntentStatus.ACKED, IntentStatus.FILLED})


class SessionBrokerPort(BrokerPort, Protocol):
    """BrokerPort plus the pre-trade reads the session needs (implemented by KisTradingClient)."""

    async def ensure_token(self, session: Any) -> None: ...
    async def fetch_orderable_cash(self, session: Any, *, symbol: str, limit_price: int) -> int: ...
    async def fetch_price_limits(self, session: Any, symbol: str) -> PriceLimits: ...


@dataclass(frozen=True)
class OrderSessionReport:
    """Outcome of one order-session phase run."""

    phase: str
    trading_date: str
    outcome: str
    reason: str
    metrics: Mapping[str, Any]


def _hhmmss(moment: pd.Timestamp) -> str:
    return str(moment.tz_convert("Asia/Seoul").strftime("%H%M%S"))


def _kst_deadline(date_str: str, hhmmss: str) -> datetime:
    return datetime.strptime(f"{date_str} {hhmmss}", "%Y-%m-%d %H%M%S").replace(tzinfo=_KST)


def _gateway_now(now_fn: Callable[[], pd.Timestamp]) -> Callable[[], datetime]:
    return lambda: now_fn().to_pydatetime()


def _limits_from_settings() -> GatewayLimits:
    return GatewayLimits(
        seed_capital_krw=settings.ORDER_SEED_CAPITAL,
        max_notional_per_order_krw=settings.ORDER_MAX_NOTIONAL_PER_ORDER_KRW,
        max_sends_per_run=settings.ORDER_MAX_SENDS_PER_RUN,
        send_max_attempts=settings.ORDER_SEND_MAX_ATTEMPTS,
        unknown_resolve_polls=settings.ORDER_UNKNOWN_RESOLVE_POLLS,
        unknown_resolve_interval_seconds=settings.ORDER_UNKNOWN_RESOLVE_INTERVAL_SECONDS,
        exit_resend_on_ambiguous=settings.ORDER_EXIT_RESEND_ON_AMBIGUOUS,
    )


def _pred_value(raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return math.nan


def _rank_key(pred: float, symbol: str) -> tuple[int, float, str]:
    if isinstance(pred, float) and math.isnan(pred):
        return (1, 0.0, symbol)
    return (0, -float(pred), symbol)


def _validate_decision(
    frame: pd.DataFrame, *, date_str: str, now: pd.Timestamp, max_sends: int
) -> tuple[list[dict[str, Any]] | None, str | None]:
    for column in ("symbol", "allocation", "close", "decided_at"):
        if column not in frame.columns:
            return None, f"invalid_decision:missing_column_{column}"
    if len(frame) == 0:
        return None, "invalid_decision:empty"
    if len(frame) > max_sends:
        return None, "invalid_decision:too_many_rows"
    symbols = [str(v) for v in frame["symbol"].tolist()]
    if any(not _SYMBOL_RE.fullmatch(s) for s in symbols):
        return None, "invalid_decision:bad_symbol"
    if len(set(symbols)) != len(symbols):
        return None, "invalid_decision:duplicate_symbol"
    allocations: list[float] = []
    for raw in frame["allocation"].tolist():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return None, "invalid_decision:bad_allocation"
        if not math.isfinite(value) or value <= 0:
            return None, "invalid_decision:bad_allocation"
        allocations.append(value)
    if sum(allocations) > 1 + 1e-9:
        return None, "invalid_decision:over_allocated"
    picks: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        symbol = str(row["symbol"])
        close_raw = row["close"]
        if isinstance(close_raw, bool):
            return None, "invalid_decision:bad_close"
        if isinstance(close_raw, int):
            close = close_raw
        elif isinstance(close_raw, float) and math.isfinite(close_raw) and close_raw.is_integer():
            close = int(close_raw)
        else:
            return None, "invalid_decision:bad_close"
        if close <= 0:
            return None, "invalid_decision:bad_close"
        decided = row["decided_at"]
        try:
            stamp = pd.Timestamp(decided)
        except (TypeError, ValueError):
            return None, "invalid_decision:bad_decided_at"
        if stamp.tzinfo is None:
            return None, "invalid_decision:naive_decided_at"
        kst = stamp.tz_convert("Asia/Seoul")
        if kst.strftime("%Y-%m-%d") != date_str:
            return None, "invalid_decision:decided_at_wrong_day"
        hhmmss = kst.strftime("%H%M%S")
        if hhmmss < ORDER_ENTRY_EARLIEST_HHMMSS:
            return None, "invalid_decision:decided_before_window"
        if hhmmss >= DECISION_WINDOW_END_HHMMSS:
            return None, "invalid_decision:decided_after_close"
        if stamp >= now:
            return None, "invalid_decision:decided_in_future"
        picks.append(
            {
                "symbol": symbol,
                "allocation": float(row["allocation"]),
                "pred": _pred_value(row["pred"] if "pred" in frame.columns else math.nan),
                "close": close,
            }
        )
    picks.sort(key=lambda p: _rank_key(p["pred"], p["symbol"]))
    return picks, None


def _max_latency_ms(journal: OrderJournal, run_id: str) -> float:
    peak = 0.0
    for event in journal.load():
        if event.run_id != run_id or event.kind.value != "result":
            continue
        raw = event.payload.get("latency_ms", 0.0)
        try:
            peak = max(peak, float(raw))
        except (TypeError, ValueError):
            continue
    return peak


def _entry_counts(journal: OrderJournal, intent_ids: list[str]) -> dict[str, int]:
    states = journal.states()
    counts = {"n_acked": 0, "n_rejected": 0, "n_unknown": 0, "n_not_sent": 0, "n_skipped": 0}
    for intent_id in intent_ids:
        state = states.get(intent_id)
        if state is None:
            counts["n_not_sent"] += 1
        elif state.status in _OK_SEND_STATUSES:
            counts["n_acked"] += 1
        elif state.status is IntentStatus.REJECTED:
            counts["n_rejected"] += 1
        elif state.status is IntentStatus.UNKNOWN:
            counts["n_unknown"] += 1
        elif state.status is IntentStatus.SKIPPED:
            counts["n_skipped"] += 1
        else:
            counts["n_not_sent"] += 1
    return counts


def _stored_entry_intents(journal: OrderJournal, date_str: str) -> list[OrderIntent]:
    ordered: list[str] = []
    for event in journal.load():
        if event.kind.value != "intent":
            continue
        payload = event.payload
        if payload.get("trading_date") != date_str or payload.get("phase") != OrderPhase.ENTRY.value:
            continue
        if event.intent_id not in ordered:
            ordered.append(event.intent_id)
    states = journal.states()
    out: list[OrderIntent] = []
    for intent_id in ordered:
        state = states.get(intent_id)
        if state is not None and state.intent is not None:
            out.append(state.intent)
    return out


def _has_entry_history(journal: OrderJournal, date_str: str) -> bool:
    for event in journal.load():
        if event.kind.value not in ("intent", "skip"):
            continue
        if event.payload.get("trading_date") == date_str and event.payload.get("phase") == OrderPhase.ENTRY.value:
            return True
    return False


def _planned_skip_count(journal: OrderJournal, date_str: str, phase: OrderPhase) -> int:
    states = journal.states()
    skipped: set[str] = set()
    for event in journal.load():
        if event.kind.value != "skip":
            continue
        if event.payload.get("trading_date") != date_str or event.payload.get("phase") != phase.value:
            continue
        if states[event.intent_id].intent is None:
            skipped.add(event.intent_id)
    return len(skipped)


async def _run_entry(
    *,
    date_str: str,
    now_fn: Callable[[], pd.Timestamp],
    sleep_fn: Callable[[float], Awaitable[None]],
    decision_fn: Callable[[pd.Timestamp], pd.DataFrame],
    trading_date: pd.Timestamp,
    broker: SessionBrokerPort,
    journal: OrderJournal,
    sess: Any,
    env: str,
    run_id: str,
) -> OrderSessionReport:
    now = now_fn()
    if _hhmmss(now) < ORDER_ENTRY_EARLIEST_HHMMSS:
        return OrderSessionReport("entry", date_str, RUN_OUTCOME_SKIPPED, "before_window", {"env": env})
    if _hhmmss(now) >= ORDER_ENTRY_SUBMIT_DEADLINE_HHMMSS:
        return OrderSessionReport("entry", date_str, RUN_OUTCOME_DEGRADED, "entry_window_expired", {"env": env})
    gateway = OrderGateway(broker, journal, _limits_from_settings(), run_id=run_id,
                           now_fn=_gateway_now(now_fn), sleep_fn=sleep_fn)
    with journal.exclusive():
        if _has_entry_history(journal, date_str):
            return await _submit_stored_entry(
                stored=_stored_entry_intents(journal, date_str), date_str=date_str, now_fn=now_fn,
                gateway=gateway, journal=journal, sess=sess, env=env, run_id=run_id,
            )
    frame = decision_fn(trading_date)
    while frame.empty:
        now = now_fn()
        if _hhmmss(now) >= ORDER_ENTRY_DECISION_WAIT_DEADLINE_HHMMSS:
            return OrderSessionReport(
                "entry", date_str, RUN_OUTCOME_SKIPPED, "no_persisted_decision", {"env": env}
            )
        await sleep_fn(ORDER_DECISION_POLL_SECONDS)
        frame = decision_fn(trading_date)
    ready_at = now_fn()
    picks, rule = _validate_decision(frame, date_str=date_str, now=now_fn(), max_sends=settings.ORDER_MAX_SENDS_PER_RUN)
    if picks is None:
        assert rule is not None
        return OrderSessionReport("entry", date_str, RUN_OUTCOME_DEGRADED, rule, {"env": env, "n_picks": len(frame)})
    top = picks[0]
    deadline = _kst_deadline(date_str, ORDER_ENTRY_SUBMIT_DEADLINE_HHMMSS)
    with journal.exclusive():
        if _has_entry_history(journal, date_str):
            return await _submit_stored_entry(
                stored=_stored_entry_intents(journal, date_str), date_str=date_str, now_fn=now_fn,
                gateway=gateway, journal=journal, sess=sess, env=env, run_id=run_id,
            )
        if journal.is_halted():
            halted = [PlannedSkip(
                intent_id=entry_intent_id(date_str, pick["symbol"]), symbol=pick["symbol"],
                trading_date=date_str, phase=OrderPhase.ENTRY, reason="halted") for pick in picks]
            gateway.plan([], halted)
            moment = now_fn()
            return _finish_entry_report(
                journal=journal, intent_ids=[], n_picks=len(picks), n_skipped=len(halted),
                capital=0, date_str=date_str, env=env, ready_at=ready_at,
                t_first=moment, t_last=moment, deadline=deadline, run_id=run_id, post_send=None)
        try:
            await broker.ensure_token(sess)
        except BrokerReadError:
            return OrderSessionReport(
                "entry", date_str, RUN_OUTCOME_DEGRADED, "orderable_cash_unreadable",
                {"env": env, "n_picks": len(picks)},
            )
        try:
            cash = await broker.fetch_orderable_cash(sess, symbol=top["symbol"], limit_price=top["close"])
        except BrokerReadError:
            return OrderSessionReport(
                "entry", date_str, RUN_OUTCOME_DEGRADED, "orderable_cash_unreadable",
                {"env": env, "n_picks": len(picks)},
            )
        capital = min(settings.ORDER_SEED_CAPITAL, cash)
        priced: list[EntryPick] = []
        skips: list[PlannedSkip] = []
        for pick in picks:
            try:
                limits = await broker.fetch_price_limits(sess, pick["symbol"])
                price = entry_limit_price(pick["close"], settings.ORDER_ENTRY_LIMIT_BUFFER_BP, limits.upper)
            except (BrokerReadError, ValueError):
                skips.append(PlannedSkip(
                    intent_id=entry_intent_id(date_str, pick["symbol"]), symbol=pick["symbol"],
                    trading_date=date_str, phase=OrderPhase.ENTRY, reason="no_price_limits"))
                continue
            priced.append(EntryPick(symbol=pick["symbol"], allocation=pick["allocation"],
                                    pred=pick["pred"], decision_price=pick["close"], limit_price=price))
        quantities = size_entry_orders(priced, capital, settings.ORDER_MAX_NOTIONAL_PER_ORDER_KRW)
        intents: list[OrderIntent] = []
        for candidate in priced:
            qty = quantities.get(candidate.symbol, 0)
            if qty <= 0:
                skips.append(PlannedSkip(
                    intent_id=entry_intent_id(date_str, candidate.symbol), symbol=candidate.symbol,
                    trading_date=date_str, phase=OrderPhase.ENTRY, reason="zero_qty"))
                continue
            intents.append(OrderIntent(
                intent_id=entry_intent_id(date_str, candidate.symbol), trading_date=date_str,
                phase=OrderPhase.ENTRY, symbol=candidate.symbol, side=OrderSide.BUY,
                order_type=OrderType.LIMIT, qty=qty, limit_price=candidate.limit_price,
                decision_price=candidate.decision_price, parent_intent_id=None))
        gateway.plan(intents, skips)
        t_first = now_fn()
        for intent in intents:
            await gateway.submit(sess, intent.intent_id, deadline=deadline)
        t_last = now_fn()
        try:
            await gateway.reconcile(sess, date_str, finalize=False)
            post_send: str | None = None
        except BrokerReadError:
            post_send = "unreadable"
    return _finish_entry_report(
        journal=journal, intent_ids=[i.intent_id for i in intents], n_picks=len(picks),
        n_skipped=len(skips), capital=capital, date_str=date_str, env=env,
        ready_at=ready_at, t_first=t_first, t_last=t_last, deadline=deadline,
        run_id=run_id, post_send=post_send,
    )


async def _submit_stored_entry(
    *,
    stored: list[OrderIntent],
    date_str: str,
    now_fn: Callable[[], pd.Timestamp],
    gateway: OrderGateway,
    journal: OrderJournal,
    sess: Any,
    env: str,
    run_id: str,
) -> OrderSessionReport:
    deadline = _kst_deadline(date_str, ORDER_ENTRY_SUBMIT_DEADLINE_HHMMSS)
    ready_at = now_fn()
    t_first = now_fn()
    for intent in stored:
        await gateway.submit(sess, intent.intent_id, deadline=deadline)
    t_last = now_fn()
    post_send: str | None = None
    if stored:
        try:
            await gateway.reconcile(sess, date_str, finalize=False)
        except BrokerReadError:
            post_send = "unreadable"
    notional = sum(intent.qty * intent.limit_price for intent in stored if intent.limit_price is not None)
    n_plan_skips = _planned_skip_count(journal, date_str, OrderPhase.ENTRY)
    return _finish_entry_report(
        journal=journal, intent_ids=[i.intent_id for i in stored], n_picks=len(stored) + n_plan_skips,
        n_skipped=n_plan_skips, capital=notional, date_str=date_str, env=env,
        ready_at=ready_at, t_first=t_first, t_last=t_last, deadline=deadline,
        run_id=run_id, post_send=post_send,
    )


def _finish_entry_report(
    *,
    journal: OrderJournal,
    intent_ids: list[str],
    n_picks: int,
    n_skipped: int,
    capital: int,
    date_str: str,
    env: str,
    ready_at: pd.Timestamp,
    t_first: pd.Timestamp,
    t_last: pd.Timestamp,
    deadline: datetime,
    run_id: str,
    post_send: str | None,
) -> OrderSessionReport:
    states = journal.states()
    counts = _entry_counts(journal, intent_ids)
    ok = (len(intent_ids) > 0 and n_skipped == 0 and all(
        states.get(i) is not None and states[i].status in _OK_SEND_STATUSES for i in intent_ids
    ))
    notional = 0
    for intent_id in intent_ids:
        state = states.get(intent_id)
        if state is not None and state.intent is not None and state.intent.limit_price is not None:
            if state.status is not IntentStatus.SKIPPED:
                notional += state.intent.qty * state.intent.limit_price
    metrics: dict[str, Any] = {
        "env": env,
        "n_picks": n_picks,
        "n_acked": counts["n_acked"],
        "n_rejected": counts["n_rejected"],
        "n_unknown": counts["n_unknown"],
        "n_not_sent": counts["n_not_sent"],
        "n_skipped": n_skipped + counts["n_skipped"],
        "capital_krw": capital,
        "planned_notional_krw": notional,
        "decided_to_first_send_s": (t_first - ready_at).total_seconds(),
        "last_send_seconds_before_deadline": (deadline - t_last.to_pydatetime()).total_seconds(),
        "max_latency_ms": _max_latency_ms(journal, run_id),
    }
    if post_send is not None:
        metrics["post_send_reconcile"] = post_send
    if ok:
        return OrderSessionReport("entry", date_str, RUN_OUTCOME_OK, "entry_complete", metrics)
    return OrderSessionReport("entry", date_str, RUN_OUTCOME_DEGRADED, "entry_incomplete", metrics)


async def _run_exit(
    *,
    date_str: str,
    now_fn: Callable[[], pd.Timestamp],
    sleep_fn: Callable[[float], Awaitable[None]],
    broker: SessionBrokerPort,
    journal: OrderJournal,
    sess: Any,
    env: str,
    run_id: str,
) -> OrderSessionReport:
    lots = [lot for lot in journal.open_lots() if lot.trading_date < date_str]
    if not lots:
        return OrderSessionReport("exit", date_str, RUN_OUTCOME_SKIPPED, "no_open_positions", {"env": env})
    now = now_fn()
    if _hhmmss(now) < ORDER_EXIT_EARLIEST_HHMMSS:
        gap = (_kst_deadline(date_str, ORDER_EXIT_EARLIEST_HHMMSS) - now.to_pydatetime()).total_seconds()
        if gap > ORDER_MAX_PRESTART_WAIT_SECONDS:
            return OrderSessionReport(
                "exit", date_str, RUN_OUTCOME_SKIPPED, "before_window", {"env": env, "n_lots": len(lots)})
        await sleep_fn(max(0.0, gap))
        now = now_fn()
        lots = [lot for lot in journal.open_lots() if lot.trading_date < date_str]
        if not lots:
            return OrderSessionReport("exit", date_str, RUN_OUTCOME_SKIPPED, "no_open_positions", {"env": env})
    if _hhmmss(now) >= ORDER_EXIT_SUBMIT_DEADLINE_HHMMSS:
        return OrderSessionReport(
            "exit", date_str, RUN_OUTCOME_DEGRADED, "exit_window_expired",
            {"env": env, "n_lots": len(lots)})
    gateway = OrderGateway(broker, journal, _limits_from_settings(), run_id=run_id, now_fn=_gateway_now(now_fn),
                           sleep_fn=sleep_fn)
    deadline = _kst_deadline(date_str, ORDER_EXIT_SUBMIT_DEADLINE_HHMMSS)
    with journal.exclusive():
        fresh_lots = sorted(
            (lot for lot in journal.open_lots() if lot.trading_date < date_str),
            key=lambda lot: (lot.trading_date, lot.entry_intent_id),
        )
        states = journal.states()
        existing = [
            st.intent for st in states.values()
            if st.intent is not None and st.intent.phase is OrderPhase.EXIT and st.intent.trading_date == date_str
        ]
        if existing:
            stored_ids = [intent.intent_id for intent in existing]
            for intent_id in stored_ids:
                await gateway.submit(sess, intent_id, deadline=deadline)
            t_last = now_fn()
            # open_lots() is already net of today's sell fills; add them back to recover the planned lot size.
            filled_today: dict[str, int] = {}
            for intent in existing:
                parent = str(intent.parent_intent_id)
                filled_today[parent] = filled_today.get(parent, 0) + states[intent.intent_id].filled_qty
            planned_lots = [replace(lot, qty=lot.qty + filled_today.get(lot.entry_intent_id, 0))
                            for lot in fresh_lots]
            return _finish_exit_report(
                journal=journal, lots=planned_lots, exit_ids=stored_ids,
                n_skipped=_planned_skip_count(journal, date_str, OrderPhase.EXIT),
                date_str=date_str, env=env, t_last=t_last, deadline=deadline, run_id=run_id)
        if journal.is_halted():
            halted = [PlannedSkip(
                intent_id=exit_intent_id(lot.entry_intent_id, date_str), symbol=lot.symbol,
                trading_date=date_str, phase=OrderPhase.EXIT, reason="halted") for lot in fresh_lots]
            gateway.plan([], halted)
            t_last = now_fn()
            return _finish_exit_report(
                journal=journal, lots=fresh_lots, exit_ids=[], n_skipped=len(halted),
                date_str=date_str, env=env, t_last=t_last, deadline=deadline, run_id=run_id)
        try:
            await broker.ensure_token(sess)
        except BrokerReadError:
            return OrderSessionReport(
                "exit", date_str, RUN_OUTCOME_DEGRADED, "positions_unreadable",
                {"env": env, "n_lots": len(fresh_lots)})
        try:
            positions = await broker.fetch_positions(sess)
        except BrokerReadError:
            return OrderSessionReport(
                "exit", date_str, RUN_OUTCOME_DEGRADED, "positions_unreadable",
                {"env": env, "n_lots": len(fresh_lots)})
        sellable: dict[str, int] = {}
        for pos in positions:
            sellable[pos.symbol] = pos.sellable_qty
        intents: list[OrderIntent] = []
        skips: list[PlannedSkip] = []
        for lot in fresh_lots:
            available = sellable.get(lot.symbol, 0)
            qty = min(lot.qty, available)
            sellable[lot.symbol] = available - qty
            if qty <= 0:
                skips.append(PlannedSkip(
                    intent_id=exit_intent_id(lot.entry_intent_id, date_str), symbol=lot.symbol,
                    trading_date=date_str, phase=OrderPhase.EXIT, reason="not_sellable"))
                continue
            intents.append(OrderIntent(
                intent_id=exit_intent_id(lot.entry_intent_id, date_str), trading_date=date_str,
                phase=OrderPhase.EXIT, symbol=lot.symbol, side=OrderSide.SELL,
                order_type=OrderType.MARKET, qty=qty, limit_price=None,
                decision_price=None, parent_intent_id=lot.entry_intent_id))
        gateway.plan(intents, skips)
        for intent in intents:
            await gateway.submit(sess, intent.intent_id, deadline=deadline)
        t_last = now_fn()
        try:
            await gateway.reconcile(sess, date_str, finalize=False)
            post_send: str | None = None
        except BrokerReadError:
            post_send = "unreadable"
        report = _finish_exit_report(
            journal=journal, lots=fresh_lots, exit_ids=[i.intent_id for i in intents],
            n_skipped=len(skips), date_str=date_str, env=env, t_last=t_last,
            deadline=deadline, run_id=run_id)
        if post_send is not None:
            metrics = dict(report.metrics)
            metrics["post_send_reconcile"] = post_send
            report = OrderSessionReport(report.phase, report.trading_date, report.outcome, report.reason, metrics)
        return report


def _finish_exit_report(
    *,
    journal: OrderJournal,
    lots: list[OpenLot],
    exit_ids: list[str],
    n_skipped: int,
    date_str: str,
    env: str,
    t_last: pd.Timestamp,
    deadline: datetime,
    run_id: str,
) -> OrderSessionReport:
    states = journal.states()
    sold_by_parent: dict[str, int] = {}
    for intent_id in exit_ids:
        state = states.get(intent_id)
        if state is None or state.intent is None or state.intent.parent_intent_id is None:
            continue
        sold = state.intent.qty if state.status in _OK_SEND_STATUSES else state.filled_qty
        sold_by_parent[state.intent.parent_intent_id] = sold
    shortfall = 0
    for lot in lots:
        shortfall += lot.qty - sold_by_parent.get(lot.entry_intent_id, 0)
    counts = _entry_counts(journal, exit_ids)
    metrics: dict[str, Any] = {
        "env": env,
        "n_lots": len(lots),
        "n_acked": counts["n_acked"],
        "n_rejected": counts["n_rejected"],
        "n_unknown": counts["n_unknown"],
        "n_skipped": n_skipped,
        "shortfall_qty": shortfall,
        "max_latency_ms": _max_latency_ms(journal, run_id),
        "last_send_seconds_before_deadline": (deadline - t_last.to_pydatetime()).total_seconds(),
    }
    if shortfall == 0 and counts["n_rejected"] == 0 and counts["n_unknown"] == 0 and counts["n_not_sent"] == 0:
        return OrderSessionReport("exit", date_str, RUN_OUTCOME_OK, "exit_complete", metrics)
    return OrderSessionReport("exit", date_str, RUN_OUTCOME_DEGRADED, "exit_incomplete", metrics)


def _scope_intents(journal: OrderJournal, date_str: str, scope: OrderPhase) -> list[str]:
    states = journal.states()
    return [key for key, st in states.items()
            if st.intent is not None and st.intent.phase is scope and st.intent.trading_date == date_str]


def _all_final(journal: OrderJournal, date_str: str, scope: OrderPhase) -> bool:
    states = journal.states()
    relevant = [st for st in states.values()
                if st.intent is not None and st.intent.phase is scope and st.intent.trading_date == date_str]
    if not relevant:
        return not _has_working_orders(journal, date_str)
    return all(st.final for st in relevant)


def _has_working_orders(journal: OrderJournal, date_str: str) -> bool:
    return any(st.intent is not None and st.intent.trading_date == date_str
               and st.broker_order_no is not None and not st.final
               for st in journal.states().values())


async def _run_reconcile(
    *,
    phase: str,
    date_str: str,
    now_fn: Callable[[], pd.Timestamp],
    sleep_fn: Callable[[float], Awaitable[None]],
    broker: SessionBrokerPort,
    journal: OrderJournal,
    sess: Any,
    env: str,
    run_id: str,
) -> OrderSessionReport:
    scope = OrderPhase.ENTRY if phase == "reconcile-entry" else OrderPhase.EXIT
    if phase == "reconcile-entry":
        earliest, deadline_hhmmss = ORDER_ENTRY_RECONCILE_EARLIEST_HHMMSS, ORDER_ENTRY_RECONCILE_DEADLINE_HHMMSS
    else:
        earliest, deadline_hhmmss = ORDER_EXIT_RECONCILE_EARLIEST_HHMMSS, ORDER_EXIT_RECONCILE_DEADLINE_HHMMSS
    today_str = now_fn().tz_convert("Asia/Seoul").strftime("%Y-%m-%d")
    backdated = date_str != today_str
    gateway = OrderGateway(broker, journal, _limits_from_settings(), run_id=run_id, now_fn=_gateway_now(now_fn),
                           sleep_fn=sleep_fn)
    if not backdated:
        now = now_fn()
        if _hhmmss(now) < earliest:
            gap = (_kst_deadline(date_str, earliest) - now.to_pydatetime()).total_seconds()
            if gap > ORDER_MAX_PRESTART_WAIT_SECONDS:
                return OrderSessionReport(phase, date_str, RUN_OUTCOME_SKIPPED, "before_window", {"env": env})
            await sleep_fn(max(0.0, gap))
    # The entry-scope pass is the only one that finalizes a same-day sell remainder, so it must also run on a
    # day without entry orders while such a sell is still working.
    sweeps_sells = scope is OrderPhase.ENTRY and _has_working_orders(journal, date_str)
    if not _scope_intents(journal, date_str, scope) and not sweeps_sells:
        return OrderSessionReport(phase, date_str, RUN_OUTCOME_SKIPPED, "no_orders",
                                  {"env": env, "scope": scope.value})
    last_report: ReconcileReport | None = None
    last_failed = False
    while True:
        now = now_fn()
        past_deadline = backdated or _hhmmss(now) >= deadline_hhmmss
        finalize = backdated or (scope is OrderPhase.ENTRY and past_deadline)
        try:
            with journal.exclusive():
                last_report = await gateway.reconcile(sess, date_str, finalize=finalize)
            last_failed = False
        except BrokerReadError:
            logger.warning("[EXEC] stage=reconcile env=%s phase=%s run_id=%s status=read_failed",
                           env, phase, run_id)
            last_failed = True
        if not last_failed and last_report is not None and _all_final(journal, date_str, scope):
            break
        if past_deadline:
            break
        await sleep_fn(ORDER_RECONCILE_POLL_SECONDS)
    if last_failed or last_report is None:
        return OrderSessionReport(phase, date_str, RUN_OUTCOME_DEGRADED, "broker_unreadable",
                                  {"env": env, "scope": scope.value})
    states = journal.states()
    relevant = [st for st in states.values()
                if st.intent is not None and st.intent.phase is scope and st.intent.trading_date == date_str]
    metrics: dict[str, Any] = {
        "env": env,
        "scope": scope.value,
        "n_intents": last_report.n_intents,
        "n_filled": last_report.n_filled,
        "n_partial": last_report.n_partial,
        "n_unfilled": last_report.n_unfilled,
        "n_working": last_report.n_working,
        "n_unknown": last_report.n_unknown,
        "n_missing_at_broker": last_report.n_missing_at_broker,
        "n_foreign_orders": last_report.n_foreign_orders,
        "position_shortfalls": [list(t) for t in last_report.position_shortfalls],
        "filled_amount_krw": last_report.filled_amount_krw,
    }
    if scope is OrderPhase.ENTRY:
        filled_qty_total = sum(st.filled_qty for st in relevant)
        if filled_qty_total > 0:
            decision_basis = sum(
                st.filled_qty * (st.intent.decision_price or 0) for st in relevant if st.intent is not None)
            if decision_basis > 0:
                metrics["entry_fill_vs_decision_bp"] = (
                    sum(st.filled_amount_krw for st in relevant) / decision_basis - 1) * 1e4
    complete = (all(st.status is IntentStatus.FILLED for st in relevant) and last_report.n_unknown == 0
                and last_report.n_missing_at_broker == 0 and not last_report.position_shortfalls)
    if complete:
        return OrderSessionReport(phase, date_str, RUN_OUTCOME_OK, "reconcile_complete", metrics)
    if last_report.n_unknown > 0:
        reason = "unknown_orders"
    elif last_report.n_missing_at_broker > 0:
        reason = "missing_at_broker"
    elif last_report.position_shortfalls:
        reason = "position_shortfall"
    elif scope is OrderPhase.EXIT and last_report.n_working > 0:
        reason = "exit_working"
    else:
        reason = "partial_or_unfilled"
    return OrderSessionReport(phase, date_str, RUN_OUTCOME_DEGRADED, reason, metrics)


async def _run_cancel(
    *,
    date_str: str,
    broker: SessionBrokerPort,
    journal: OrderJournal,
    sess: Any,
    env: str,
    run_id: str,
    now_fn: Callable[[], pd.Timestamp],
    sleep_fn: Callable[[float], Awaitable[None]],
) -> OrderSessionReport:
    gateway = OrderGateway(broker, journal, _limits_from_settings(), run_id=run_id, now_fn=_gateway_now(now_fn),
                           sleep_fn=sleep_fn)
    with journal.exclusive():
        accepted = await gateway.cancel_working(sess, date_str)
        report = await gateway.reconcile(sess, date_str, finalize=False)
    metrics: dict[str, Any] = {
        "env": env,
        "n_cancel_accepted": accepted,
        "n_intents": report.n_intents,
        "n_filled": report.n_filled,
        "n_unknown": report.n_unknown,
        "n_missing_at_broker": report.n_missing_at_broker,
    }
    return OrderSessionReport("cancel", date_str, RUN_OUTCOME_OK, "cancel_complete", metrics)


async def run_order_session(
    phase: str,
    trading_date: pd.Timestamp,
    *,
    broker: SessionBrokerPort | None = None,
    journal: OrderJournal | None = None,
    session: Any | None = None,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    record_fn: Callable[..., Any] | None = None,
    session_day_fn: Callable[[date], SessionDay] | None = None,
    decision_fn: Callable[[pd.Timestamp], pd.DataFrame] | None = None,
) -> OrderSessionReport:
    """Run one order-session phase and return its report; raises ValueError on an unknown phase or a backdated order phase."""
    if phase not in ORDER_PHASES:
        raise ValueError(f"unknown order phase {phase!r}")
    clock = now_fn or (lambda: pd.Timestamp.now(tz="Asia/Seoul"))
    sleeper = sleep_fn or asyncio.sleep
    date_str = pd.Timestamp(trading_date).strftime("%Y-%m-%d")
    try:
        binding = resolve_trading_binding()
    except TradingDisabledError:
        report = OrderSessionReport(phase, date_str, RUN_OUTCOME_SKIPPED, "trading_disabled", {"env": "off"})
        if record_fn is not None:
            record_fn(report.outcome, run_date=date_str, reason=report.reason, metrics=dict(report.metrics))
        logger.info("[EXEC] stage=order_session env=off phase=%s run_id=none outcome=%s reason=%s",
                    phase, report.outcome, report.reason)
        return report
    active_broker: SessionBrokerPort = broker if broker is not None else KisTradingClient(
        binding, http_timeout_seconds=(
            settings.ORDER_MOCK_HTTP_TIMEOUT_SECONDS if binding.env == "mock" else settings.ORDER_HTTP_TIMEOUT_SECONDS))
    active_journal = journal if journal is not None else OrderJournal(
        settings.ORDER_JOURNAL_DIR, binding.env,
        lock_timeout_seconds=settings.ORDER_JOURNAL_LOCK_TIMEOUT_SECONDS)
    if phase in ("entry", "exit"):
        today = clock().tz_convert("Asia/Seoul").date()
        if today != pd.Timestamp(trading_date).date():
            raise ValueError(f"{phase} cannot run for another day: {date_str}")
    resolve_day = session_day_fn or resolve_session_day
    gate = trading_session_gate(resolve_day(pd.Timestamp(trading_date).date()))
    if gate is not None:
        if phase == "exit" and gate in ("session_shifted", "calendar_unverified"):
            lots = [lot for lot in active_journal.open_lots() if lot.trading_date < date_str]
            if lots:
                hold_reason = {"session_shifted": "shifted_session_hold",
                               "calendar_unverified": "calendar_unverified_hold"}[gate]
                report = OrderSessionReport(
                    phase, date_str, RUN_OUTCOME_DEGRADED, hold_reason,
                    {"env": binding.env, "n_lots": len(lots)})
                if record_fn is not None:
                    record_fn(report.outcome, run_date=date_str, reason=report.reason,
                              metrics=dict(report.metrics))
                logger.warning("[RISK] stage=order_session env=%s phase=%s run_id=none outcome=%s reason=%s",
                               binding.env, phase, report.outcome, report.reason)
                return report
        report = OrderSessionReport(phase, date_str, RUN_OUTCOME_SKIPPED, gate, {"env": binding.env})
        if record_fn is not None:
            record_fn(report.outcome, run_date=date_str, reason=report.reason, metrics=dict(report.metrics))
        logger.info("[EXEC] stage=order_session env=%s phase=%s run_id=none outcome=%s reason=%s",
                    binding.env, phase, report.outcome, report.reason)
        return report
    run_id = f"{phase}:{date_str}:{clock().strftime('%H%M%S')}"
    decide = decision_fn or load_topk_decision

    async def _execute(sess: Any) -> OrderSessionReport:
        if phase == "entry":
            return await _run_entry(date_str=date_str, now_fn=clock, sleep_fn=sleeper, decision_fn=decide,
                                    trading_date=trading_date, broker=active_broker, journal=active_journal,
                                    sess=sess, env=binding.env, run_id=run_id)
        if phase == "exit":
            return await _run_exit(date_str=date_str, now_fn=clock, sleep_fn=sleeper, broker=active_broker,
                                   journal=active_journal, sess=sess, env=binding.env, run_id=run_id)
        if phase in ("reconcile-entry", "reconcile-exit"):
            return await _run_reconcile(phase=phase, date_str=date_str, now_fn=clock, sleep_fn=sleeper,
                                        broker=active_broker, journal=active_journal, sess=sess,
                                        env=binding.env, run_id=run_id)
        return await _run_cancel(date_str=date_str, broker=active_broker, journal=active_journal, sess=sess,
                                 env=binding.env, run_id=run_id, now_fn=clock, sleep_fn=sleeper)

    if session is not None:
        report = await _execute(session)
    elif hasattr(active_broker, "create_session"):
        async with active_broker.create_session() as sess:
            report = await _execute(sess)
    else:
        report = await _execute(None)
    if record_fn is not None:
        record_fn(report.outcome, run_date=date_str, reason=report.reason, metrics=dict(report.metrics))
    level = logging.INFO if report.outcome == RUN_OUTCOME_OK else logging.WARNING
    logger.log(level, "[EXEC] stage=order_session env=%s phase=%s run_id=%s outcome=%s reason=%s",
               binding.env, phase, run_id, report.outcome, report.reason)
    logger.info("[PORTFOLIO] stage=order_session env=%s phase=%s run_id=%s metrics=%s",
                binding.env, phase, run_id, dict(report.metrics))
    return report


def main() -> None:
    """CLI entry: --phase {entry,exit,reconcile-entry,reconcile-exit,cancel} [--date YYYY-MM-DD]."""
    parser = argparse.ArgumentParser(description="Gated real-order session")
    parser.add_argument("--phase", required=True, choices=ORDER_PHASES)
    parser.add_argument("--date", default=None, help="Trading date YYYY-MM-DD (reconcile/cancel only)")
    args = parser.parse_args()
    if args.date is not None and args.phase in ("entry", "exit"):
        parser.error("--date is not accepted for entry/exit; they always run for today")
    configure_cli_logging()
    trading_date = pd.Timestamp(args.date) if args.date else pd.Timestamp.now(tz="Asia/Seoul").normalize()
    report = asyncio.run(run_order_session(
        args.phase, trading_date,
        record_fn=functools.partial(record_run_outcome, _JOB_BY_PHASE[args.phase])))
    if report.outcome not in (RUN_OUTCOME_OK, RUN_OUTCOME_SKIPPED):
        logger.warning("[RISK] stage=order_session_cli phase=%s outcome=%s reason=%s",
                       args.phase, report.outcome, report.reason)


if __name__ == "__main__":
    main()
