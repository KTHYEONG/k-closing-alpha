"""Operator resolve invariant guards."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.api.kis.trading import BrokerReadError, TradingDisabledError
from src.daily import order_resolve
from src.daily.order_resolve import ResolveReport, resolve_unknown_intent
from src.execution.order_contract import (
    BrokerOrder,
    OrderIntent,
    OrderPhase,
    OrderSide,
    OrderType,
)
from src.execution.order_journal import IntentStatus, JournalEventKind, OrderJournal, intent_to_payload


def _entry(intent_id: str = "2026-10-12:005930:entry", qty: int = 10, limit: int = 70000) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id, trading_date="2026-10-12", phase=OrderPhase.ENTRY,
        symbol="005930", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=qty, limit_price=limit, decision_price=limit, parent_intent_id=None,
    )


def _order(order_no: str = "12345", qty: int = 10, price: int = 70000,
           symbol: str = "005930", side: OrderSide = OrderSide.BUY) -> BrokerOrder:
    return BrokerOrder(order_no=order_no, orig_order_no="", org_no="001", symbol=symbol, side=side,
                       order_qty=qty, order_price=price, filled_qty=0,
                       filled_amount_krw=0, remaining_qty=qty, cancelled=False, rejected_qty=0)


def _journal_with_unknown(tmp_path: Path, intent: OrderIntent, baseline: list[str] | None = None) -> OrderJournal:
    journal = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="r1")
        payload: dict[str, Any] = {"attempt": 1}
        if baseline is not None:
            payload["baseline_order_nos"] = list(baseline)
        journal.append(JournalEventKind.SEND, intent.intent_id, payload, run_id="r1")
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "ambiguous", "broker_order_no": None,
                        "broker_org_no": None, "http_status": None, "msg_cd": "",
                        "msg": "", "latency_ms": 1.0}, run_id="r1")
    return journal


class FakeBroker:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls = 0
        self.ensure_calls = 0

    async def ensure_token(self, session: Any) -> None:
        self.ensure_calls += 1

    async def fetch_daily_orders(self, session: Any, trading_date: str) -> list[BrokerOrder]:
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _mock_binding(monkeypatch: pytest.MonkeyPatch, env: str = "mock") -> None:
    monkeypatch.setattr(order_resolve, "resolve_trading_binding",
                        lambda: SimpleNamespace(env=env))


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    before = journal.load()
    broker = FakeBroker([[_order()]])
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="verified", apply=False,
                                        broker=broker, journal=journal))
    assert report.action == "adopt" and report.applied is False and report.reason_code == "ok"
    assert [e.kind for e in journal.load()] == [e.kind for e in before]
    assert journal.states()[intent.intent_id].status is IntentStatus.UNKNOWN


def test_adopt_binds_stale_baseline_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent, baseline=["12345"])
    broker = FakeBroker([[_order("12345")]])
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="verified", apply=True,
                                        broker=broker, journal=journal))
    assert report.action == "adopt" and report.applied is True
    assert journal.states()[intent.intent_id].status is IntentStatus.ACKED


def test_reject_mismatched_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry(qty=10)
    journal = _journal_with_unknown(tmp_path, intent)
    broker = FakeBroker([[_order("12345", qty=9)]])
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="verified", apply=True,
                                        broker=broker, journal=journal))
    assert report.action == "refused" and report.reason_code == "order_mismatch"
    assert all(e.kind is not JournalEventKind.OPERATOR_RESOLVE for e in journal.load())
    priced = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="verified", apply=True,
                                        broker=FakeBroker([[_order("12345", price=69000)]]),
                                        journal=journal))
    assert priced.reason_code == "order_mismatch"


def test_reject_already_bound_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    other = _entry("2026-10-12:000660:entry")
    journal = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
    with journal.exclusive():
        for one in (intent, other):
            journal.append(JournalEventKind.INTENT, one.intent_id, intent_to_payload(one), run_id="r1")
            journal.append(JournalEventKind.SEND, one.intent_id, {"attempt": 1}, run_id="r1")
            journal.append(JournalEventKind.RESULT, one.intent_id,
                           {"attempt": 1, "verdict": "ambiguous", "broker_order_no": None,
                            "broker_org_no": None, "http_status": None, "msg_cd": "",
                            "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.OPERATOR_RESOLVE, other.intent_id,
                       {"resolution": "adopt", "operator": "op", "reason": "r",
                        "broker_order_no": "12345", "broker_org_no": "001"}, run_id="r1")
    broker = FakeBroker([[_order("12345")]])
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="verified", apply=True,
                                        broker=broker, journal=journal))
    assert report.action == "refused" and report.reason_code == "order_mismatch"


def test_not_found_refused_when_candidate_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    broker = FakeBroker([[_order("777")]])
    report = _run(resolve_unknown_intent(intent.intent_id, order_no=None, not_found=True,
                                        operator="op", reason="checked", apply=True,
                                        broker=broker, journal=journal))
    assert report.action == "refused" and report.reason_code == "candidate_exists"
    assert report.candidates == ("777",)


def test_not_found_accepted_when_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    broker = FakeBroker([[_order("999", symbol="000660")]])
    report = _run(resolve_unknown_intent(intent.intent_id, order_no=None, not_found=True,
                                        operator="op", reason="checked", apply=True,
                                        broker=broker, journal=journal))
    assert report.action == "not_found" and report.applied is True
    assert journal.states()[intent.intent_id].status is IntentStatus.NOT_FOUND


def test_cannot_override_filled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="r1")
        journal.append(JournalEventKind.SEND, intent.intent_id, {"attempt": 1}, run_id="r1")
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r1")
    broker = FakeBroker([[_order()]])
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="verified", apply=True,
                                        broker=broker, journal=journal))
    assert report.action == "refused" and report.reason_code == "not_unknown"
    assert broker.calls == 0


def test_read_failure_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    broker = FakeBroker([BrokerReadError("down")])
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="verified", apply=True,
                                        broker=broker, journal=journal))
    assert report.action == "refused" and report.reason_code == "broker_unreadable"
    assert all(e.kind is not JournalEventKind.OPERATOR_RESOLVE for e in journal.load())


def test_trading_disabled_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _off() -> Any:
        raise TradingDisabledError("off")
    monkeypatch.setattr(order_resolve, "resolve_trading_binding", _off)
    journal = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
    broker = FakeBroker([[]])
    report = _run(resolve_unknown_intent("x", order_no="1", not_found=False,
                                        operator="op", reason="r", apply=True,
                                        broker=broker, journal=journal))
    assert report.action == "refused" and report.reason_code == "trading_disabled"
    assert broker.calls == 0


def test_live_needs_confirmation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch, env="live")
    argv = ["prog", "--intent-id", "x", "--order-no", "1", "--operator", "op", "--reason", "r"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        order_resolve.main()
    assert exc.value.code == 2


def test_blank_operator_reason_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    broker = FakeBroker([[_order()]])
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="  ", reason="r", apply=True,
                                        broker=broker, journal=journal))
    assert report.action == "refused" and report.reason_code == "bad_arguments"
    assert isinstance(report, ResolveReport)
    blank_reason = _run(resolve_unknown_intent(intent.intent_id, order_no=None, not_found=True,
                                              operator="op", reason="  ", apply=False,
                                              broker=FakeBroker([[]]), journal=journal))
    assert blank_reason.reason_code == "bad_arguments"


def test_missing_intent_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    journal = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
    report = _run(resolve_unknown_intent("no-such", order_no=None, not_found=True,
                                        operator="op", reason="r", apply=False,
                                        broker=FakeBroker([[]]), journal=journal))
    assert report.action == "refused" and report.reason_code == "not_unknown"


def test_zero_normalized_order_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="000", not_found=False,
                                        operator="op", reason="r", apply=False,
                                        broker=FakeBroker([[]]), journal=journal))
    assert report.reason_code == "bad_arguments"


def test_blank_org_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    bad = _order("12345")
    object.__setattr__(bad, "org_no", "  ")
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="r", apply=True,
                                        broker=FakeBroker([[bad]]), journal=journal))
    assert report.reason_code == "order_mismatch"


def test_not_found_read_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    report = _run(resolve_unknown_intent(intent.intent_id, order_no=None, not_found=True,
                                        operator="op", reason="r", apply=False,
                                        broker=FakeBroker([BrokerReadError("down")]), journal=journal))
    assert report.reason_code == "broker_unreadable"


def test_not_found_dry_run_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    before = len(journal.load())
    report = _run(resolve_unknown_intent(intent.intent_id, order_no=None, not_found=True,
                                        operator="op", reason="r", apply=False,
                                        broker=FakeBroker([[_order("999", symbol="000660")]]),
                                        journal=journal))
    assert report.action == "not_found" and report.applied is False
    assert len(journal.load()) == before


def test_not_found_ignores_baseline_and_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    other = _entry("2026-10-12:000660:entry")
    journal = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
    with journal.exclusive():
        for one in (intent, other):
            journal.append(JournalEventKind.INTENT, one.intent_id, intent_to_payload(one), run_id="r1")
            journal.append(JournalEventKind.SEND, one.intent_id,
                           {"attempt": 1, "baseline_order_nos": ["111"]}, run_id="r1")
            journal.append(JournalEventKind.RESULT, one.intent_id,
                           {"attempt": 1, "verdict": "ambiguous", "broker_order_no": None,
                            "broker_org_no": None, "http_status": None, "msg_cd": "",
                            "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.OPERATOR_RESOLVE, other.intent_id,
                       {"resolution": "adopt", "operator": "op", "reason": "r",
                        "broker_order_no": "222", "broker_org_no": "001"}, run_id="r1")
    orders = [_order("111"), _order("222"), _order("999", symbol="000660")]
    report = _run(resolve_unknown_intent(intent.intent_id, order_no=None, not_found=True,
                                        operator="op", reason="r", apply=True,
                                        broker=FakeBroker([orders]), journal=journal))
    assert report.action == "not_found" and report.applied is True


def test_bad_arguments_both_or_neither(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    journal = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
    broker = FakeBroker([[]])
    both = _run(resolve_unknown_intent("x", order_no="1", not_found=True,
                                      operator="op", reason="r", apply=False,
                                      broker=broker, journal=journal))
    neither = _run(resolve_unknown_intent("x", order_no=None, not_found=False,
                                         operator="op", reason="r", apply=False,
                                         broker=broker, journal=journal))
    assert both.reason_code == "bad_arguments" and neither.reason_code == "bad_arguments"


def test_idempotent_apply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    first = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                       operator="op", reason="verified", apply=True,
                                       broker=FakeBroker([[_order("12345")]]), journal=journal))
    assert first.applied is True
    second = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="verified", apply=True,
                                        broker=FakeBroker([[_order("12345")]]), journal=journal))
    assert second.action == "refused" and second.reason_code == "not_unknown"


def test_default_broker_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    monkeypatch.setattr(order_resolve, "KisTradingClient",
                        lambda binding, http_timeout_seconds: FakeBroker([[_order("12345")]]))
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="r", apply=False,
                                        broker=None, journal=journal))
    assert report.action == "adopt"
    _mock_binding(monkeypatch, env="live")
    journal2 = _journal_with_unknown(tmp_path / "live", intent)
    monkeypatch.setattr(order_resolve, "KisTradingClient",
                        lambda binding, http_timeout_seconds: FakeBroker([[_order("12345")]]))
    live_report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                             operator="op", reason="r", apply=False,
                                             broker=None, journal=journal2))
    assert live_report.action == "adopt"


class _SessionCtx:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: Any) -> None:
        return None


class SessionBroker(FakeBroker):
    def create_session(self) -> _SessionCtx:
        return _SessionCtx()


def test_session_broker_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    broker = SessionBroker([[_order("12345")]])
    report = _run(resolve_unknown_intent(intent.intent_id, order_no="12345", not_found=False,
                                        operator="op", reason="r", apply=False,
                                        broker=broker, journal=journal))
    assert report.action == "adopt"


def test_main_trading_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def _off() -> Any:
        raise TradingDisabledError("off")
    monkeypatch.setattr(order_resolve, "resolve_trading_binding", _off)
    monkeypatch.setattr(sys, "argv", ["prog", "--intent-id", "x", "--order-no", "1",
                                     "--operator", "op", "--reason", "r"])
    with pytest.raises(SystemExit) as exc:
        order_resolve.main()
    assert exc.value.code == 2


def test_main_accepted_and_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)

    async def _ok(*args: Any, **kwargs: Any) -> ResolveReport:
        return ResolveReport(intent_id="x", action="adopt", applied=False,
                             reason_code="ok", candidates=("1",))

    async def _no(*args: Any, **kwargs: Any) -> ResolveReport:
        return ResolveReport(intent_id="x", action="refused", applied=False,
                             reason_code="order_mismatch", candidates=())

    monkeypatch.setattr(order_resolve, "resolve_unknown_intent", _ok)
    monkeypatch.setattr(sys, "argv", ["prog", "--intent-id", "x", "--order-no", "1",
                                     "--operator", "op", "--reason", "r"])
    order_resolve.main()
    monkeypatch.setattr(order_resolve, "resolve_unknown_intent", _no)
    with pytest.raises(SystemExit) as exc:
        order_resolve.main()
    assert exc.value.code == 2


def test_not_found_apply_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_binding(monkeypatch)
    intent = _entry()
    journal = _journal_with_unknown(tmp_path, intent)
    first = _run(resolve_unknown_intent(intent.intent_id, order_no=None, not_found=True,
                                       operator="op", reason="r", apply=True,
                                       broker=FakeBroker([[]]), journal=journal))
    assert first.applied is True
    assert journal.states()[intent.intent_id].status is IntentStatus.NOT_FOUND
    before = len(journal.load())
    broker = FakeBroker([[]])
    second = _run(resolve_unknown_intent(intent.intent_id, order_no=None, not_found=True,
                                        operator="op", reason="r", apply=True,
                                        broker=broker, journal=journal))
    assert (second.action, second.reason_code) == ("refused", "not_unknown")
    assert len(journal.load()) == before and broker.calls == 0
