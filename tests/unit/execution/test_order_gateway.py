"""Order gateway invariant guards."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src.api.kis.trading import BrokerReadError
from src.execution.order_contract import (
    BrokerOrder,
    BrokerPosition,
    OrderIntent,
    OrderPhase,
    OrderSide,
    OrderType,
    SubmitResult,
    SubmitVerdict,
)
from src.execution.order_gateway import GatewayLimits, OrderGateway, PlannedSkip
from src.execution.order_journal import IntentStatus, JournalEventKind, OrderJournal, intent_to_payload

_KST = timezone(timedelta(hours=9))


def _entry(intent_id: str = "2026-10-12:005930:entry", qty: int = 10,
           limit: int = 10000, date: str = "2026-10-12") -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id, trading_date=date, phase=OrderPhase.ENTRY,
        symbol="005930", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=qty, limit_price=limit, decision_price=limit, parent_intent_id=None,
    )


def _exit(parent: str, qty: int = 4, intent_id: str = "e-exit",
          date: str = "2026-10-13") -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id, trading_date=date, phase=OrderPhase.EXIT,
        symbol="005930", side=OrderSide.SELL, order_type=OrderType.LIMIT,
        qty=qty, limit_price=10000, decision_price=None, parent_intent_id=parent,
    )


def _result(verdict: SubmitVerdict, order_no: str | None = None) -> SubmitResult:
    return SubmitResult(verdict=verdict, broker_order_no=order_no, broker_org_no="001" if order_no else None,
                        http_status=200 if verdict is not SubmitVerdict.NOT_SENT else None,
                        msg_cd="0", msg="m", latency_ms=1.0)


def _broker_order(order_no: str = "12345", symbol: str = "005930", side: OrderSide = OrderSide.BUY,
                  qty: int = 10, price: int = 10000, filled: int = 0) -> BrokerOrder:
    return BrokerOrder(order_no=order_no, orig_order_no="", org_no="001", symbol=symbol, side=side,
                       order_qty=qty, order_price=price, filled_qty=filled,
                       filled_amount_krw=filled * price, remaining_qty=qty - filled,
                       cancelled=False, rejected_qty=0)


class FakeBroker:
    def __init__(self) -> None:
        self.submit_script: list[SubmitResult] = []
        self.submit_calls: list[str] = []
        self.daily_script: list[list[BrokerOrder] | Exception] = []
        self.daily_calls = 0
        self.positions: list[BrokerPosition] = []
        self.positions_error: Exception | None = None
        self.cancel_calls: list[str] = []
        self.cancel_result = _result(SubmitVerdict.ACCEPTED, "1")
        self.seen_send_at_call: list[bool] = []

    async def submit_order(self, session: Any, intent: OrderIntent) -> SubmitResult:
        self.submit_calls.append(intent.intent_id)
        if not self.submit_script:
            return _result(SubmitVerdict.ACCEPTED, "99999")
        return self.submit_script.pop(0)

    async def cancel_order(self, session: Any, *, org_no: str, order_no: str) -> SubmitResult:
        self.cancel_calls.append(order_no)
        return self.cancel_result

    async def fetch_daily_orders(self, session: Any, trading_date: str) -> list[BrokerOrder]:
        self.daily_calls += 1
        if not self.daily_script:
            return []
        item = self.daily_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def fetch_positions(self, session: Any) -> list[BrokerPosition]:
        if self.positions_error is not None:
            raise self.positions_error
        return list(self.positions)


def _limits(**over: Any) -> GatewayLimits:
    base: dict[str, Any] = {"seed_capital_krw": 10_000_000_000, "max_notional_per_order_krw": 10_000_000_000,
                            "max_sends_per_run": 100, "send_max_attempts": 3, "unknown_resolve_polls": 2,
                            "unknown_resolve_interval_seconds": 0.0, "exit_resend_on_ambiguous": False}
    base.update(over)
    return GatewayLimits(**base)


def _gateway(tmp_path: Path, broker: FakeBroker, limits: GatewayLimits,
             now: datetime, journal: OrderJournal | None = None) -> tuple[OrderGateway, OrderJournal]:
    journal = journal or OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
    sleeps: list[float] = []

    async def _sleep(delay: float) -> None:
        sleeps.append(delay)

    gateway = OrderGateway(broker, journal, limits, run_id="run1",
                           now_fn=lambda: now, sleep_fn=_sleep)
    return gateway, journal


def _now() -> datetime:
    return datetime(2026, 10, 12, 10, 0, tzinfo=_KST)


def test_requires_lock(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, _ = _gateway(tmp_path, broker, _limits(), _now())
    with pytest.raises(RuntimeError):
        gateway.plan([_entry()], [])
    with pytest.raises(RuntimeError):
        asyncio.run(gateway.submit(object(), "x", deadline=_now()))
    with pytest.raises(RuntimeError):
        asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))


def test_accepted_once_and_replay(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.ACCEPTED, "12345")]
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.ACKED
        state2 = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                            deadline=_now() + timedelta(hours=1)))
        assert state2.status is IntentStatus.ACKED
    assert broker.submit_calls == [_entry().intent_id]
    with journal.exclusive():
        kinds = [e.kind for e in journal.load() if e.intent_id == _entry().intent_id]
    assert kinds == [JournalEventKind.INTENT, JournalEventKind.SEND, JournalEventKind.RESULT]


def test_send_durable_before_post(tmp_path: Path) -> None:
    journal = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)

    class _Inspect(FakeBroker):
        async def submit_order(self, session: Any, intent: OrderIntent) -> SubmitResult:
            events = journal.load()
            sends = [e for e in events if e.intent_id == intent.intent_id and e.kind is JournalEventKind.SEND]
            assert len(sends) == 1
            assert sends[0].payload["attempt"] == 1
            return _result(SubmitVerdict.ACCEPTED, "77")

    broker = _Inspect()
    gateway, _ = _gateway(tmp_path, broker, _limits(), _now(), journal=journal)
    with journal.exclusive():
        gateway.plan([_entry()], [])
        asyncio.run(gateway.submit(object(), _entry().intent_id, deadline=_now() + timedelta(hours=1)))


def test_crash_between_send_and_result_never_resent(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        intent = _entry()
        journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="r0")
        journal.append(JournalEventKind.SEND, intent.intent_id, {"attempt": 1}, run_id="r0")
        state = asyncio.run(gateway.submit(object(), intent.intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.UNKNOWN
    assert broker.submit_calls == []


def test_second_trigger_no_resize(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        gateway.plan([_entry(qty=10)], [])
        gateway.plan([_entry(qty=3)], [])
        assert journal.states()[_entry().intent_id].intent is not None
        assert journal.states()[_entry().intent_id].intent.qty == 10  # type: ignore[union-attr]


def test_throttle_retried_and_exhaustion(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.THROTTLED), _result(SubmitVerdict.THROTTLED),
                            _result(SubmitVerdict.ACCEPTED, "1")]
    gateway, journal = _gateway(tmp_path, broker, _limits(send_max_attempts=3), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.ACKED
    assert len(broker.submit_calls) == 3


def test_throttle_exhaustion_sendable(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.THROTTLED)] * 3
    gateway, journal = _gateway(tmp_path, broker, _limits(send_max_attempts=3), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.NOT_SENT
        broker.submit_script = [_result(SubmitVerdict.ACCEPTED, "5")]
        state2 = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                            deadline=_now() + timedelta(hours=1)))
        assert state2.status is IntentStatus.ACKED


def test_not_sent_retried_and_rejected_terminal(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.NOT_SENT), _result(SubmitVerdict.ACCEPTED, "9")]
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.ACKED
    assert len(broker.submit_calls) == 2
    broker2 = FakeBroker()
    broker2.submit_script = [_result(SubmitVerdict.REJECTED)]
    gateway2, journal2 = _gateway(tmp_path / "rej-b", broker2, _limits(), _now())
    with journal2.exclusive():
        gateway2.plan([_entry("2026-10-12:000660:entry", qty=1, limit=10000)], [])
        iid = "2026-10-12:000660:entry"
        first = asyncio.run(gateway2.submit(object(), iid, deadline=_now() + timedelta(hours=1)))
        assert first.status is IntentStatus.REJECTED
        asyncio.run(gateway2.submit(object(), iid, deadline=_now() + timedelta(hours=1)))
    assert broker2.submit_calls == [iid]


def test_ambiguous_adopt_and_not_found(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.AMBIGUOUS)]
    broker.daily_script = [[_broker_order("12345", qty=10, price=10000)]]
    gateway, journal = _gateway(tmp_path, broker, _limits(unknown_resolve_polls=1), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.ACKED
        assert state.broker_order_no == "12345"
    assert len(broker.submit_calls) == 1
    broker2 = FakeBroker()
    broker2.submit_script = [_result(SubmitVerdict.AMBIGUOUS)]
    broker2.daily_script = [[], []]
    gateway2, journal2 = _gateway(tmp_path / "amb-b", broker2, _limits(unknown_resolve_polls=2), _now())
    iid = "2026-10-12:000660:entry"
    with journal2.exclusive():
        gateway2.plan([_entry(iid, qty=1)], [])
        state = asyncio.run(gateway2.submit(object(), iid, deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.NOT_FOUND
        later = asyncio.run(gateway2.submit(object(), iid, deadline=_now() + timedelta(hours=1)))
        assert later.status is IntentStatus.NOT_FOUND
    assert len(broker2.submit_calls) == 1


def test_conflict_and_bound_excluded(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.AMBIGUOUS)]
    broker.daily_script = [[_broker_order("1", qty=10, price=10000), _broker_order("2", qty=10, price=10000)]]
    gateway, journal = _gateway(tmp_path, broker, _limits(unknown_resolve_polls=1), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.UNKNOWN
    broker2 = FakeBroker()
    broker2.submit_script = [_result(SubmitVerdict.ACCEPTED, " Bound ")]
    gateway2, journal2 = _gateway(tmp_path / "conflict-b", broker2, _limits(), _now())
    with journal2.exclusive():
        entry_fresh = _entry()
        other = _entry("2026-10-12:000660:entry", qty=1)
        gateway2.plan([entry_fresh, other], [])
        asyncio.run(gateway2.submit(object(), other.intent_id, deadline=_now() + timedelta(hours=1)))
        broker2.submit_script = [_result(SubmitVerdict.AMBIGUOUS)]
        broker2.daily_script = [[_broker_order("0000Bound", qty=10, price=10000)]]
        gateway2b, _ = _gateway(tmp_path / "conflict-b", broker2, _limits(unknown_resolve_polls=1), _now(), journal=journal2)
        state = asyncio.run(gateway2b.submit(object(), entry_fresh.intent_id,
                                             deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.NOT_FOUND


def test_read_error_leaves_unknown(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.AMBIGUOUS)]
    broker.daily_script = [BrokerReadError("down")]
    gateway, journal = _gateway(tmp_path, broker, _limits(unknown_resolve_polls=2), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.UNKNOWN
        assert broker.daily_calls == 1


def test_exit_resend_variants(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.AMBIGUOUS)]
    broker.daily_script = [[], []]
    gateway, journal = _gateway(
        tmp_path, broker, _limits(unknown_resolve_polls=2, exit_resend_on_ambiguous=False), _now())
    entry = _entry()
    ex = _exit(entry.intent_id, qty=4, intent_id="x-exit")
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, entry.intent_id, intent_to_payload(entry), run_id="r")
        journal.append(JournalEventKind.SEND, entry.intent_id, {"attempt": 1}, run_id="r")
        journal.append(JournalEventKind.RESULT, entry.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "e1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r")
        journal.append(JournalEventKind.BROKER_STATE, entry.intent_id,
                       {"order_qty": 10, "filled_qty": 10, "filled_amount_krw": 100,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="r")
        journal.append(JournalEventKind.INTENT, ex.intent_id, intent_to_payload(ex), run_id="r")
        state = asyncio.run(gateway.submit(object(), ex.intent_id, deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.NOT_FOUND
    assert len(broker.submit_calls) == 1


def test_exit_resend_once_when_sellable(tmp_path: Path) -> None:
    for sellable, expect_two in ((10, True), (1, False)):
        broker = FakeBroker()
        broker.submit_script = [_result(SubmitVerdict.AMBIGUOUS), _result(SubmitVerdict.ACCEPTED, "s2")]
        broker.daily_script = [[], []]
        broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=sellable)]
        gateway, journal = _gateway(
            tmp_path, broker, _limits(unknown_resolve_polls=2, exit_resend_on_ambiguous=True), _now())
        entry = _entry(intent_id=f"2026-10-12:005930:entry-{sellable}")
        ex = _exit(entry.intent_id, qty=4, intent_id=f"exit-{sellable}")
        with journal.exclusive():
            journal.append(JournalEventKind.INTENT, entry.intent_id, intent_to_payload(entry), run_id="r")
            journal.append(JournalEventKind.SEND, entry.intent_id, {"attempt": 1}, run_id="r")
            journal.append(JournalEventKind.RESULT, entry.intent_id,
                           {"attempt": 1, "verdict": "accepted", "broker_order_no": f"e{sellable}",
                            "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                            "msg": "", "latency_ms": 1.0}, run_id="r")
            journal.append(JournalEventKind.BROKER_STATE, entry.intent_id,
                           {"order_qty": 10, "filled_qty": 10, "filled_amount_krw": 100,
                            "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                           run_id="r")
            journal.append(JournalEventKind.INTENT, ex.intent_id, intent_to_payload(ex), run_id="r")
            state = asyncio.run(gateway.submit(object(), ex.intent_id,
                                               deadline=_now() + timedelta(hours=1)))
            if expect_two:
                assert state.status is IntentStatus.ACKED
                assert len(broker.submit_calls) == 2
            else:
                assert state.status is IntentStatus.NOT_FOUND
                assert len(broker.submit_calls) == 1


def test_halt_and_deadline_and_caps(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        (journal._dir / "HALT").touch()
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.SKIPPED
    assert broker.submit_calls == []
    (journal._dir / "HALT").unlink()
    broker2 = FakeBroker()
    gateway2, journal2 = _gateway(tmp_path / "halt-b", broker2, _limits(), _now())
    with journal2.exclusive():
        iid = "2026-10-12:000661:entry"
        gateway2.plan([_entry(iid, qty=1)], [])
        state = asyncio.run(gateway2.submit(object(), iid, deadline=_now()))
        assert state.status is IntentStatus.SKIPPED
    assert broker2.submit_calls == []
    broker3 = FakeBroker()
    gateway3, journal3 = _gateway(
        tmp_path / "halt-c", broker3, _limits(max_notional_per_order_krw=1000), _now())
    with journal3.exclusive():
        iid = "2026-10-12:000662:entry"
        gateway3.plan([_entry(iid, qty=10, limit=10000)], [])
        state = asyncio.run(gateway3.submit(object(), iid, deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.SKIPPED
    assert broker3.submit_calls == []


def test_seed_cap_and_open_lot_and_run_cap(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.ACCEPTED, "a1"), _result(SubmitVerdict.AMBIGUOUS)]
    broker.daily_script = [[], []]
    limits = _limits(seed_capital_krw=20000, max_sends_per_run=100,
                     unknown_resolve_polls=2, send_max_attempts=1)
    gateway, journal = _gateway(tmp_path, broker, limits, _now())
    with journal.exclusive():
        gateway.plan([_entry("2026-10-12:005930:entry", qty=1, limit=10000),
                      _entry("2026-10-12:000660:entry", qty=1, limit=10000)], [])
        asyncio.run(gateway.submit(object(), "2026-10-12:005930:entry",
                                   deadline=_now() + timedelta(hours=1)))
        unknown_intent = _entry("2026-10-12:000660:entry", qty=1, limit=10000)
        journal.append(JournalEventKind.SEND, unknown_intent.intent_id, {"attempt": 1}, run_id="r")
        journal.append(JournalEventKind.RESULT, unknown_intent.intent_id,
                       {"attempt": 1, "verdict": "ambiguous", "broker_order_no": None,
                        "broker_org_no": None, "http_status": None, "msg_cd": "",
                        "msg": "", "latency_ms": 1.0}, run_id="r")
        third = _entry("2026-10-12:035420:entry", qty=1, limit=10000)
        gateway.plan([third], [])
        state = asyncio.run(gateway.submit(object(), third.intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.SKIPPED
    broker4 = FakeBroker()
    gateway4, journal4 = _gateway(tmp_path, broker4, _limits(), _now())
    with journal4.exclusive():
        entry = _entry(intent_id="p-entry")
        journal4.append(JournalEventKind.INTENT, entry.intent_id, intent_to_payload(entry), run_id="r")
        journal4.append(JournalEventKind.SEND, entry.intent_id, {"attempt": 1}, run_id="r")
        journal4.append(JournalEventKind.RESULT, entry.intent_id,
                        {"attempt": 1, "verdict": "accepted", "broker_order_no": "pe",
                         "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                         "msg": "", "latency_ms": 1.0}, run_id="r")
        journal4.append(JournalEventKind.BROKER_STATE, entry.intent_id,
                        {"order_qty": 10, "filled_qty": 5, "filled_amount_krw": 500,
                         "remaining_qty": 5, "cancelled": False, "rejected_qty": 0, "final": False},
                        run_id="r")
        big_sell = _exit(entry.intent_id, qty=6, intent_id="big-sell")
        journal4.append(JournalEventKind.INTENT, big_sell.intent_id, intent_to_payload(big_sell), run_id="r")
        state = asyncio.run(gateway4.submit(object(), big_sell.intent_id,
                                            deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.SKIPPED
    broker5 = FakeBroker()
    gateway5, journal5 = _gateway(tmp_path, broker5, _limits(max_sends_per_run=2), _now())
    with journal5.exclusive():
        intents = [_entry(f"2026-10-12:00000{i}:entry", qty=1, limit=100) for i in range(3)]
        gateway5.plan(intents, [])
        for intent in intents:
            broker5.submit_script.append(_result(SubmitVerdict.ACCEPTED, f"n{intent.intent_id[-2:]}"))
        for intent in intents[:2]:
            asyncio.run(gateway5.submit(object(), intent.intent_id, deadline=_now() + timedelta(hours=1)))
        last = asyncio.run(gateway5.submit(object(), intents[2].intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert last.status is IntentStatus.SKIPPED


def test_deadline_stops_retry(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.THROTTLED)]
    times = [_now(), _now() + timedelta(hours=2)]

    async def _sleep(delay: float) -> None:
        return None

    journal = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
    gateway = OrderGateway(broker, journal, _limits(send_max_attempts=3), run_id="r",
                           now_fn=lambda: times.pop(0) if times else _now() + timedelta(hours=2),
                           sleep_fn=_sleep)
    with journal.exclusive():
        gateway.plan([_entry()], [])
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.NOT_SENT
    assert len(broker.submit_calls) == 1


def test_reconcile_fills_once_and_partial(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        journal.append(JournalEventKind.SEND, _entry().intent_id, {"attempt": 1}, run_id="r")
        journal.append(JournalEventKind.RESULT, _entry().intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "12345",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r")
        broker.daily_script = [[_broker_order("12345", qty=10, price=10000, filled=10)]]
        broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
        report = asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))
        assert report.n_filled == 1
        assert report.filled_amount_krw == 100000
        broker.daily_script = [[_broker_order("12345", qty=10, price=10000, filled=10)]]
        asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))
        assert sum(1 for e in journal.load() if e.kind is JournalEventKind.BROKER_STATE) == 1
    broker2 = FakeBroker()
    gateway2, journal2 = _gateway(tmp_path / "rec-partial", broker2, _limits(), _now())
    with journal2.exclusive():
        iid = "2026-10-12:000660:entry"
        gateway2.plan([_entry(iid, qty=10)], [])
        journal2.append(JournalEventKind.SEND, iid, {"attempt": 1}, run_id="r")
        journal2.append(JournalEventKind.RESULT, iid,
                        {"attempt": 1, "verdict": "accepted", "broker_order_no": "p1",
                         "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                         "msg": "", "latency_ms": 1.0}, run_id="r")
        partial = _broker_order("p1", symbol="005930", qty=10, price=10000, filled=4)
        partial = BrokerOrder(order_no="p1", orig_order_no="", org_no="001", symbol="005930",
                              side=partial.side, order_qty=10, order_price=10000, filled_qty=4,
                              filled_amount_krw=40000, remaining_qty=6,
                              cancelled=False, rejected_qty=0)
        broker2.daily_script = [[partial]]
        broker2.positions = []
        report = asyncio.run(gateway2.reconcile(object(), "2026-10-12", finalize=False))
        assert report.n_working == 1
        broker2.daily_script = [[partial]]
        report2 = asyncio.run(gateway2.reconcile(object(), "2026-10-12", finalize=True))
        assert report2.n_partial == 1


def test_reconcile_adopt_crash_and_foreign_missing(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        intent = _entry()
        journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="r")
        journal.append(JournalEventKind.SEND, intent.intent_id, {"attempt": 1}, run_id="r")
        broker.daily_script = [[_broker_order("777", qty=10, price=10000)]]
        broker.positions = []
        report = asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))
        assert journal.states()[intent.intent_id].status is IntentStatus.ACKED
        assert report.n_foreign_orders == 0
    broker2 = FakeBroker()
    gateway2, journal2 = _gateway(tmp_path / "rec-foreign", broker2, _limits(), _now())
    with journal2.exclusive():
        iid = "2026-10-12:000661:entry"
        gateway2.plan([_entry(iid, qty=1)], [])
        journal2.append(JournalEventKind.SEND, iid, {"attempt": 1}, run_id="r")
        journal2.append(JournalEventKind.RESULT, iid,
                        {"attempt": 1, "verdict": "accepted", "broker_order_no": "miss",
                         "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                         "msg": "", "latency_ms": 1.0}, run_id="r")
        broker2.daily_script = [[_broker_order("foreign-1", qty=1, price=100)]]
        broker2.positions = []
        report = asyncio.run(gateway2.reconcile(object(), "2026-10-12", finalize=False))
        assert report.n_foreign_orders == 1
        assert report.n_missing_at_broker == 1


def test_position_shortfall_and_surplus(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        entry = _entry()
        journal.append(JournalEventKind.INTENT, entry.intent_id, intent_to_payload(entry), run_id="r")
        journal.append(JournalEventKind.SEND, entry.intent_id, {"attempt": 1}, run_id="r")
        journal.append(JournalEventKind.RESULT, entry.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "e1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r")
        journal.append(JournalEventKind.BROKER_STATE, entry.intent_id,
                       {"order_qty": 10, "filled_qty": 10, "filled_amount_krw": 100,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": False},
                       run_id="r")
        broker.daily_script = [[]]
        broker.positions = [BrokerPosition(symbol="005930", holding_qty=7, sellable_qty=7)]
        report = asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))
        assert report.position_shortfalls == (("005930", 10, 7),)
        broker.daily_script = [[]]
        broker.positions = [BrokerPosition(symbol="005930", holding_qty=15, sellable_qty=15)]
        report2 = asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))
        assert report2.position_shortfalls == ()


def test_reconcile_read_failure_writes_nothing(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.daily_script = [BrokerReadError("down")]
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        before = len(journal.load())
        with pytest.raises(BrokerReadError):
            asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))
        assert len(journal.load()) == before


def test_cancel_working_only_and_halted(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        acked = _entry("2026-10-12:005930:entry", qty=10)
        gateway.plan([acked,
                      _entry("2026-10-12:000660:entry", qty=1),
                      _entry("2026-10-12:035420:entry", qty=1)], [])
        for iid, verdict in (("2026-10-12:005930:entry", "accepted"),
                             ("2026-10-12:000660:entry", "accepted"),
                             ("2026-10-12:035420:entry", "rejected")):
            journal.append(JournalEventKind.SEND, iid, {"attempt": 1}, run_id="r")
            journal.append(JournalEventKind.RESULT, iid,
                           {"attempt": 1, "verdict": verdict, "broker_order_no": f"no-{iid[-2:]}",
                            "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                            "msg": "", "latency_ms": 1.0}, run_id="r")
        journal.append(JournalEventKind.BROKER_STATE, "2026-10-12:000660:entry",
                       {"order_qty": 1, "filled_qty": 1, "filled_amount_krw": 100,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="r")
        (journal._dir / "HALT").touch()
        n = asyncio.run(gateway.cancel_working(object(), "2026-10-12"))
        assert n == 1
        assert broker.cancel_calls == ["no-ry"]
        kinds = [e.kind for e in journal.load() if e.intent_id == "2026-10-12:005930:entry"]
        assert JournalEventKind.CANCEL_SEND in kinds
        assert JournalEventKind.CANCEL_RESULT in kinds
        asyncio.run(gateway.cancel_working(object(), "2026-10-12"))
        attempts = [e.payload["attempt"] for e in journal.load() if e.kind is JournalEventKind.CANCEL_SEND]
        assert attempts == [1, 2]


def test_unknown_intent_keyerror_and_plan_skip(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        with pytest.raises(KeyError):
            asyncio.run(gateway.submit(object(), "missing", deadline=_now() + timedelta(hours=1)))
        gateway.plan([], [PlannedSkip(intent_id="skip1", symbol="005930", trading_date="2026-10-12",
                                      phase=OrderPhase.ENTRY, reason="filtered")])
        assert journal.states()["skip1"].status is IntentStatus.SKIPPED
        gateway.plan([], [PlannedSkip(intent_id="skip1", symbol="005930", trading_date="2026-10-12",
                                      phase=OrderPhase.ENTRY, reason="filtered")])
        assert len([e for e in journal.load() if e.intent_id == "skip1"]) == 1


def test_buy_send_bounded_across_restarts(tmp_path: Path) -> None:
    for verdicts in ([SubmitVerdict.THROTTLED, SubmitVerdict.AMBIGUOUS],
                     [SubmitVerdict.NOT_SENT, SubmitVerdict.THROTTLED]):
        broker = FakeBroker()
        broker.submit_script = [_result(v) for v in verdicts]
        broker.daily_script = [[], []]
        journal = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
        iid = f"2026-10-12:00593{len(verdicts)}:entry"
        with journal.exclusive():
            gateway = OrderGateway(broker, journal, _limits(unknown_resolve_polls=2, send_max_attempts=1),
                                   run_id="a", now_fn=_now,
                                   sleep_fn=lambda d: asyncio.sleep(0))
            gateway.plan([_entry(iid, qty=1)], [])
            asyncio.run(gateway.submit(object(), iid, deadline=_now() + timedelta(hours=1)))
        with journal.exclusive():
            gateway2 = OrderGateway(broker, journal, _limits(unknown_resolve_polls=2, send_max_attempts=1),
                                    run_id="b", now_fn=_now,
                                    sleep_fn=lambda d: asyncio.sleep(0))
            asyncio.run(gateway2.submit(object(), iid, deadline=_now() + timedelta(hours=1)))
        sends = [e for e in journal.load() if e.intent_id == iid and e.kind is JournalEventKind.SEND]
        risky = 0
        for send in sends:
            attempt = send.payload["attempt"]
            res = [e for e in journal.load() if e.intent_id == iid and e.kind is JournalEventKind.RESULT
                   and e.payload.get("attempt") == attempt]
            if not res or res[0].payload.get("verdict") in ("accepted", "ambiguous"):
                risky += 1
        assert risky <= 1


def test_submit_exception_maps_ambiguous(tmp_path: Path) -> None:
    class _Boom(FakeBroker):
        async def submit_order(self, session: Any, intent: OrderIntent) -> SubmitResult:
            raise ConnectionError("boom")

    broker = _Boom()
    broker.daily_script = [[]]
    gateway, journal = _gateway(tmp_path, broker, _limits(unknown_resolve_polls=1), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.NOT_FOUND


def test_reconcile_conflict_and_mismatch_filters(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        intent = _entry()
        journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="r")
        journal.append(JournalEventKind.SEND, intent.intent_id, {"attempt": 1}, run_id="r")
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "ambiguous", "broker_order_no": None,
                        "broker_org_no": None, "http_status": None, "msg_cd": "",
                        "msg": "", "latency_ms": 1.0}, run_id="r")
        gateway.plan([], [PlannedSkip(intent_id="skip-1", symbol="005930",
                                      trading_date="2026-10-12", phase=OrderPhase.ENTRY,
                                      reason="filtered")])
        broker.daily_script = [[_broker_order("c1", qty=10, price=10000),
                                _broker_order("c2", qty=10, price=10000),
                                _broker_order("other-sym", symbol="000660", qty=10, price=10000),
                                _broker_order("wrong-qty", qty=1, price=10000),
                                _broker_order("wrong-price", qty=10, price=9999)]]
        broker.positions = []
        report = asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))
        assert journal.states()[intent.intent_id].status is IntentStatus.UNKNOWN
        assert report.n_unknown == 1


def test_resolve_generic_error_and_positions_error(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.AMBIGUOUS)]
    broker.daily_script = [RuntimeError("boom")]
    gateway, journal = _gateway(tmp_path / "gen-err", broker, _limits(unknown_resolve_polls=2), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        state = asyncio.run(gateway.submit(object(), _entry().intent_id,
                                           deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.UNKNOWN
    broker2 = FakeBroker()
    broker2.submit_script = [_result(SubmitVerdict.AMBIGUOUS)]
    broker2.daily_script = [[], []]
    broker2.positions_error = RuntimeError("pos down")
    gateway2, journal2 = _gateway(tmp_path / "pos-err", broker2,
                                  _limits(unknown_resolve_polls=2,
                                          exit_resend_on_ambiguous=True), _now())
    with journal2.exclusive():
        entry = _entry(intent_id="2026-10-12:005930:entry-pos")
        ex = _exit(entry.intent_id, qty=4, intent_id="exit-pos")
        journal2.append(JournalEventKind.INTENT, entry.intent_id, intent_to_payload(entry), run_id="r")
        journal2.append(JournalEventKind.SEND, entry.intent_id, {"attempt": 1}, run_id="r")
        journal2.append(JournalEventKind.RESULT, entry.intent_id,
                        {"attempt": 1, "verdict": "accepted", "broker_order_no": "e9",
                         "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                         "msg": "", "latency_ms": 1.0}, run_id="r")
        journal2.append(JournalEventKind.BROKER_STATE, entry.intent_id,
                        {"order_qty": 10, "filled_qty": 10, "filled_amount_krw": 100,
                         "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                        run_id="r")
        journal2.append(JournalEventKind.INTENT, ex.intent_id, intent_to_payload(ex), run_id="r")
        state = asyncio.run(gateway2.submit(object(), ex.intent_id,
                                            deadline=_now() + timedelta(hours=1)))
        assert state.status is IntentStatus.NOT_FOUND


def test_cancel_exception_counts_nothing(tmp_path: Path) -> None:
    class _CancelBoom(FakeBroker):
        async def cancel_order(self, session: Any, *, org_no: str, order_no: str) -> SubmitResult:
            raise ConnectionError("boom")

    broker = _CancelBoom()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        journal.append(JournalEventKind.SEND, _entry().intent_id, {"attempt": 1}, run_id="r")
        journal.append(JournalEventKind.RESULT, _entry().intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "k1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r")
        assert asyncio.run(gateway.cancel_working(object(), "2026-10-12")) == 0


def test_reconcile_identical_nonfinal_is_noop(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    with journal.exclusive():
        gateway.plan([_entry()], [])
        journal.append(JournalEventKind.SEND, _entry().intent_id, {"attempt": 1}, run_id="r")
        journal.append(JournalEventKind.RESULT, _entry().intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "dup",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r")
        snap = [_broker_order("dup", qty=10, price=10000, filled=4)]
        snap[0] = BrokerOrder(order_no="dup", orig_order_no="", org_no="001", symbol="005930",
                              side=snap[0].side, order_qty=10, order_price=10000, filled_qty=4,
                              filled_amount_krw=40000, remaining_qty=6,
                              cancelled=False, rejected_qty=0)
        broker.daily_script = [list(snap)]
        broker.positions = []
        asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))
        before = len([e for e in journal.load() if e.kind is JournalEventKind.BROKER_STATE])
        broker.daily_script = [list(snap)]
        asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))
        after = len([e for e in journal.load() if e.kind is JournalEventKind.BROKER_STATE])
        assert before == 1
        assert after == 1


def _filled_entry_with_exit(journal: OrderJournal, tag: str) -> OrderIntent:
    entry = _entry(intent_id=f"2026-10-12:005930:entry-{tag}")
    ex = _exit(entry.intent_id, qty=4, intent_id=f"exit-{tag}")
    journal.append(JournalEventKind.INTENT, entry.intent_id, intent_to_payload(entry), run_id="r")
    journal.append(JournalEventKind.SEND, entry.intent_id, {"attempt": 1}, run_id="r")
    journal.append(JournalEventKind.RESULT, entry.intent_id,
                   {"attempt": 1, "verdict": "accepted", "broker_order_no": f"e-{tag}",
                    "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                    "msg": "", "latency_ms": 1.0}, run_id="r")
    journal.append(JournalEventKind.BROKER_STATE, entry.intent_id,
                   {"order_qty": 10, "filled_qty": 10, "filled_amount_krw": 100,
                    "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                   run_id="r")
    journal.append(JournalEventKind.INTENT, ex.intent_id, intent_to_payload(ex), run_id="r")
    return ex


def test_sell_ambiguous_sends_bounded_after_throttled_resend(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.AMBIGUOUS), _result(SubmitVerdict.THROTTLED),
                            _result(SubmitVerdict.AMBIGUOUS), _result(SubmitVerdict.AMBIGUOUS)]
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
    limits = _limits(unknown_resolve_polls=2, exit_resend_on_ambiguous=True, send_max_attempts=1)
    gateway, journal = _gateway(tmp_path, broker, limits, _now())
    deadline = _now() + timedelta(hours=1)
    with journal.exclusive():
        ex = _filled_entry_with_exit(journal, "bound")
        first = asyncio.run(gateway.submit(object(), ex.intent_id, deadline=deadline))
        assert first.status is IntentStatus.NOT_SENT
        second = asyncio.run(gateway.submit(object(), ex.intent_id, deadline=deadline))
        assert second.status is IntentStatus.UNKNOWN
        asyncio.run(gateway.submit(object(), ex.intent_id, deadline=deadline))
        events = [e for e in journal.load() if e.intent_id == ex.intent_id]
    assert len(broker.submit_calls) == 3
    assert sum(1 for e in events if e.kind is JournalEventKind.NOT_FOUND) == 1
    ambiguous = [e for e in events if e.kind is JournalEventKind.RESULT and e.payload["verdict"] == "ambiguous"]
    assert len(ambiguous) == 2


def test_market_buy_is_bounded_by_notional_cap(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(max_notional_per_order_krw=1000), _now())
    intent = OrderIntent(
        intent_id="2026-10-12:005930:entry", trading_date="2026-10-12", phase=OrderPhase.ENTRY,
        symbol="005930", side=OrderSide.BUY, order_type=OrderType.MARKET,
        qty=10, limit_price=None, decision_price=10000, parent_intent_id=None,
    )
    with journal.exclusive():
        gateway.plan([intent], [])
        state = asyncio.run(gateway.submit(object(), intent.intent_id, deadline=_now() + timedelta(hours=1)))
    assert state.status is IntentStatus.SKIPPED
    assert state.reason == "notional_cap"
    assert broker.submit_calls == []


def test_zero_resolve_polls_never_claims_not_found(tmp_path: Path) -> None:
    broker = FakeBroker()
    broker.submit_script = [_result(SubmitVerdict.AMBIGUOUS)]
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
    limits = _limits(unknown_resolve_polls=0, exit_resend_on_ambiguous=True)
    gateway, journal = _gateway(tmp_path, broker, limits, _now())
    with journal.exclusive():
        ex = _filled_entry_with_exit(journal, "zero")
        state = asyncio.run(gateway.submit(object(), ex.intent_id, deadline=_now() + timedelta(hours=1)))
        kinds = [e.kind for e in journal.load() if e.intent_id == ex.intent_id]
    assert state.status is IntentStatus.UNKNOWN
    assert JournalEventKind.NOT_FOUND not in kinds
    assert broker.daily_calls == 0
    assert len(broker.submit_calls) == 1


def test_halt_during_resolution_blocks_exit_resend(tmp_path: Path) -> None:
    class HaltingBroker(FakeBroker):
        halt_path: Path

        async def fetch_daily_orders(self, session: Any, trading_date: str) -> list[BrokerOrder]:
            self.halt_path.touch()
            return await super().fetch_daily_orders(session, trading_date)

    broker = HaltingBroker()
    broker.submit_script = [_result(SubmitVerdict.AMBIGUOUS)]
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
    limits = _limits(unknown_resolve_polls=1, exit_resend_on_ambiguous=True)
    gateway, journal = _gateway(tmp_path, broker, limits, _now())
    broker.halt_path = journal.journal_path.parent / "HALT"
    with journal.exclusive():
        ex = _filled_entry_with_exit(journal, "halt")
        state = asyncio.run(gateway.submit(object(), ex.intent_id, deadline=_now() + timedelta(hours=1)))
    assert state.status is IntentStatus.NOT_FOUND
    assert len(broker.submit_calls) == 1


def test_reconcile_refuses_broker_state_that_would_corrupt_journal(tmp_path: Path) -> None:
    broker = FakeBroker()
    gateway, journal = _gateway(tmp_path, broker, _limits(), _now())
    other = _entry("2026-10-12:000660:entry", qty=5)
    with journal.exclusive():
        gateway.plan([_entry(), other], [])
        broker.submit_script = [_result(SubmitVerdict.ACCEPTED, "over"), _result(SubmitVerdict.ACCEPTED, "ok")]
        for intent in (_entry(), other):
            asyncio.run(gateway.submit(object(), intent.intent_id, deadline=_now() + timedelta(hours=1)))
        overfilled = BrokerOrder(order_no="over", orig_order_no="", org_no="001", symbol="005930",
                                 side=OrderSide.BUY, order_qty=10, order_price=10000, filled_qty=11,
                                 filled_amount_krw=110000, remaining_qty=0, cancelled=False, rejected_qty=0)
        broker.daily_script = [[overfilled, _broker_order("ok", qty=5, filled=5)]]
        with pytest.raises(BrokerReadError, match="contradicts"):
            asyncio.run(gateway.reconcile(object(), "2026-10-12", finalize=False))
        states = journal.states()
    assert states[_entry().intent_id].status is IntentStatus.ACKED
    assert states[_entry().intent_id].filled_qty == 0
    assert states[other.intent_id].status is IntentStatus.FILLED
