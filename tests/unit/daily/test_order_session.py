"""Order session invariant guards (entry/exit/reconcile/cancel)."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.api.kis.trading import BrokerReadError, TradingBinding, TradingDisabledError
from src.daily import order_session as sut
from src.data.session_calendar import SessionDay, SessionKind
from src.execution.order_contract import (
    BrokerOrder,
    BrokerPosition,
    OrderIntent,
    OrderPhase,
    OrderSide,
    OrderType,
    PriceLimits,
    SubmitResult,
    SubmitVerdict,
    entry_intent_id,
    entry_limit_price,
)
from src.execution.order_journal import JournalEventKind, OrderJournal, intent_to_payload

DAY = "2026-10-13"
YESTERDAY = "2026-10-12"


def ts(hhmmss: str, day: str = DAY) -> pd.Timestamp:
    return pd.Timestamp(f"{day} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]}", tz="Asia/Seoul")


def _accepted(order_no: str = "11111") -> SubmitResult:
    return SubmitResult(verdict=SubmitVerdict.ACCEPTED, broker_order_no=order_no, broker_org_no="001",
                        http_status=200, msg_cd="0", msg="ok", latency_ms=2.5)


def _rejected() -> SubmitResult:
    return SubmitResult(verdict=SubmitVerdict.REJECTED, broker_order_no=None, broker_org_no=None,
                        http_status=200, msg_cd="EGW02006", msg="no", latency_ms=1.0)


class FakeBroker:
    def __init__(self) -> None:
        self.ensure_calls = 0
        self.ensure_error: Exception | None = None
        self.cash_calls = 0
        self.cash_value = 9_950_000
        self.cash_error: Exception | None = None
        self.price_calls: list[str] = []
        self.price_errors: dict[str, Exception] = {}
        self.upper = 1_000_000
        self.submit_calls: list[str] = []
        self.submit_script: list[SubmitResult] = []
        self.daily_script: list[list[BrokerOrder] | Exception] = []
        self.daily_calls = 0
        self.positions: list[BrokerPosition] = []
        self.positions_calls = 0
        self.positions_error: Exception | None = None
        self.cancel_calls: list[str] = []
        self.journal_probe: Any | None = None

    async def ensure_token(self, session: Any) -> None:
        self.ensure_calls += 1
        if self.ensure_error is not None:
            raise self.ensure_error

    async def fetch_orderable_cash(self, session: Any, *, symbol: str, limit_price: int) -> int:
        self.cash_calls += 1
        if self.cash_error is not None:
            raise self.cash_error
        return self.cash_value

    async def fetch_price_limits(self, session: Any, symbol: str) -> PriceLimits:
        self.price_calls.append(symbol)
        if symbol in self.price_errors:
            raise self.price_errors[symbol]
        return PriceLimits(symbol=symbol, last=10_000, upper=self.upper, lower=100)

    async def submit_order(self, session: Any, intent: OrderIntent) -> SubmitResult:
        self.submit_calls.append(intent.intent_id)
        if self.journal_probe is not None:
            self.journal_probe()
        if self.submit_script:
            return self.submit_script.pop(0)
        return _accepted(f"od-{intent.symbol}")

    async def cancel_order(self, session: Any, *, org_no: str, order_no: str) -> SubmitResult:
        self.cancel_calls.append(order_no)
        return _accepted(order_no)

    async def fetch_daily_orders(self, session: Any, trading_date: str) -> list[BrokerOrder]:
        self.daily_calls += 1
        if not self.daily_script:
            return []
        item = self.daily_script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def fetch_positions(self, session: Any) -> list[BrokerPosition]:
        self.positions_calls += 1
        if self.positions_error is not None:
            raise self.positions_error
        return list(self.positions)

    def assert_quiet(self) -> None:
        assert self.ensure_calls == 0
        assert self.cash_calls == 0
        assert self.price_calls == []
        assert self.submit_calls == []
        assert self.daily_calls == 0
        assert self.positions_calls == 0
        assert self.cancel_calls == []


def _binding(env: str = "mock") -> TradingBinding:
    host = "openapivts.koreainvestment.com" if env == "mock" else "openapi.koreainvestment.com"
    port = "29443" if env == "mock" else "9443"
    return TradingBinding(env=env, base_url=f"https://{host}:{port}", app_key=f"{env}-key",
                          app_secret="s", account_no="12345678", product_code="01", hts_id="h", rest_tps=1.0)


@pytest.fixture
def mock_binding(monkeypatch) -> TradingBinding:
    binding = _binding("mock")
    monkeypatch.setattr(sut, "resolve_trading_binding", lambda: binding)
    return binding


def _standard_day(day: date) -> SessionDay:
    return SessionDay(trading_date=day, kind=SessionKind.STANDARD, clock=None, provenance="test")


def decision(rows: list[tuple[str, float, int, float]], *, day: str = DAY,
             decided: str = "152500") -> pd.DataFrame:
    return pd.DataFrame([
        {"symbol": s, "allocation": a, "close": c, "pred": p,
         "decided_at": pd.Timestamp(f"{day} {decided[:2]}:{decided[2:4]}:{decided[4:]}", tz="Asia/Seoul")}
        for s, a, c, p in rows
    ])


def three_picks() -> pd.DataFrame:
    return decision([("005930", 1 / 3, 263_000, 0.9), ("000660", 1 / 3, 50_000, 0.5), ("005380", 1 / 3, 9_990, 0.1)])


def make_clock(times: list[pd.Timestamp]) -> tuple[Any, list[pd.Timestamp]]:
    pending = list(times)
    log: list[pd.Timestamp] = []

    def _now() -> pd.Timestamp:
        current = pending.pop(0) if pending else log[-1]
        log.append(current)
        return current

    return _now, log


class Runner:
    def __init__(self, tmp_path: Path, monkeypatch: Any = None) -> None:
        self.tmp_path = tmp_path
        self.sleeps: list[float] = []
        self.recorded: list[dict[str, Any]] = []
        self.clock_log: list[pd.Timestamp] = []

    async def _sleep(self, delay: float) -> None:
        self.sleeps.append(delay)

    def _record(self, outcome: str, *, run_date: str, reason: str, metrics: Any) -> None:
        self.recorded.append({"outcome": outcome, "run_date": run_date, "reason": reason,
                              "metrics": dict(metrics)})

    def journal(self, env: str = "mock") -> OrderJournal:
        return OrderJournal(self.tmp_path, env, lock_timeout_seconds=5.0)

    def run(self, phase: str, day: str, *, broker: FakeBroker, journal: OrderJournal,
            clock: list[pd.Timestamp], decisions: list[pd.DataFrame] | None = None,
            session_day_fn: Any = None) -> Any:
        now_fn, self.clock_log = make_clock(list(clock))
        queue = list(decisions or [])
        before = len(self.recorded)

        def _decision(_when: pd.Timestamp) -> pd.DataFrame:
            return queue.pop(0) if queue else pd.DataFrame()

        report = asyncio.run(sut.run_order_session(
            phase, pd.Timestamp(day), broker=broker, journal=journal, session=object(),
            now_fn=now_fn, sleep_fn=self._sleep, record_fn=self._record,
            session_day_fn=session_day_fn or _standard_day, decision_fn=_decision))
        assert len(self.recorded) == before + 1
        assert self.recorded[-1]["outcome"] == report.outcome
        assert self.recorded[-1]["reason"] == report.reason
        assert self.recorded[-1]["metrics"] == dict(report.metrics)
        return report


@pytest.fixture
def runner(tmp_path, monkeypatch) -> Runner:
    return Runner(tmp_path)


def _add_filled_entry(journal: OrderJournal, intent_id: str, symbol: str, date_str: str,
                      qty: int, price: int) -> None:
    intent = OrderIntent(intent_id=intent_id, trading_date=date_str, phase=OrderPhase.ENTRY,
                         symbol=symbol, side=OrderSide.BUY, order_type=OrderType.LIMIT,
                         qty=qty, limit_price=price, decision_price=price, parent_intent_id=None)
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, intent_id, intent_to_payload(intent), run_id="setup")
        journal.append(JournalEventKind.BROKER_STATE, intent_id,
                       {"order_qty": qty, "filled_qty": qty, "filled_amount_krw": qty * price,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="setup")


def _add_acked(journal: OrderJournal, intent: OrderIntent, order_no: str) -> None:
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="setup")
        journal.append(JournalEventKind.SEND, intent.intent_id, {"attempt": 1}, run_id="setup")
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": order_no,
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0", "msg": "",
                        "latency_ms": 1.0}, run_id="setup")


def _entry_intent(symbol: str, date_str: str, qty: int = 10, price: int = 10_000) -> OrderIntent:
    return OrderIntent(intent_id=entry_intent_id(date_str, symbol), trading_date=date_str,
                       phase=OrderPhase.ENTRY, symbol=symbol, side=OrderSide.BUY,
                       order_type=OrderType.LIMIT, qty=qty, limit_price=price,
                       decision_price=price, parent_intent_id=None)


def _broker_row(order_no: str, symbol: str, side: OrderSide, qty: int, price: int,
                filled: int, amount: int) -> BrokerOrder:
    return BrokerOrder(order_no=order_no, orig_order_no="", org_no="001", symbol=symbol, side=side,
                       order_qty=qty, order_price=price, filled_qty=filled, filled_amount_krw=amount,
                       remaining_qty=qty - filled, cancelled=False, rejected_qty=0)


def test_disabled_trading_is_inert(tmp_path, monkeypatch, runner) -> None:
    from src.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(sut, "resolve_trading_binding",
                        lambda: (_ for _ in ()).throw(TradingDisabledError("off")))
    broker, journal = FakeBroker(), runner.journal()
    report = runner.run("entry", DAY, broker=broker, journal=journal, clock=[ts("152115")],
                        decisions=[three_picks()])
    assert (report.outcome, report.reason) == ("SKIPPED", "trading_disabled")
    assert not (tmp_path / "data" / "orders").exists()
    broker.assert_quiet()


def test_unknown_phase_is_refused(mock_binding, runner) -> None:
    with pytest.raises(ValueError, match="unknown order phase"):
        asyncio.run(sut.run_order_session("buy", pd.Timestamp(DAY), broker=FakeBroker(),
                                          journal=runner.journal(), session=object()))


def test_backdated_order_phase_is_refused(mock_binding, runner) -> None:
    for phase in ("entry", "exit"):
        broker = FakeBroker()
        with pytest.raises(ValueError, match="cannot run for another day"):
            asyncio.run(sut.run_order_session(
                phase, pd.Timestamp(YESTERDAY), broker=broker, journal=runner.journal(),
                session=object(), now_fn=lambda: ts("152115"),
                session_day_fn=_standard_day))
        broker.assert_quiet()


def test_closed_day_sends_nothing(mock_binding, runner) -> None:
    def _closed(day: date) -> SessionDay:
        return SessionDay(trading_date=day, kind=SessionKind.CLOSED, clock=None, provenance="test")

    broker, journal = FakeBroker(), runner.journal()
    report = runner.run("entry", DAY, broker=broker, journal=journal, clock=[ts("152115")] * 4,
                        decisions=[three_picks()], session_day_fn=_closed)
    assert (report.outcome, report.reason) == ("SKIPPED", "non_trading_day")
    broker.assert_quiet()


def test_shifted_session_holds_open_lots(mock_binding, runner) -> None:
    def _shifted(day: date) -> SessionDay:
        return SessionDay(trading_date=day, kind=SessionKind.SHIFTED, clock=None, provenance="test")

    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084010")] * 3,
                        session_day_fn=_shifted)
    assert (report.outcome, report.reason) == ("DEGRADED", "shifted_session_hold")
    broker.assert_quiet()
    assert len(journal.open_lots()) == 1


def test_entry_before_window_is_skipped(mock_binding, runner) -> None:
    report = runner.run("entry", DAY, broker=FakeBroker(), journal=runner.journal(),
                        clock=[ts("151959")] * 3, decisions=[three_picks()])
    assert (report.outcome, report.reason) == ("SKIPPED", "before_window")


def test_entry_after_submit_deadline_sends_nothing(mock_binding, runner) -> None:
    broker = FakeBroker()
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152900")] * 3, decisions=[three_picks()])
    assert (report.outcome, report.reason) == ("DEGRADED", "entry_window_expired")
    assert broker.submit_calls == []


def test_entry_waits_for_decision(mock_binding, runner) -> None:
    broker = FakeBroker()
    early = decision([("005930", 1 / 3, 263_000, 0.9), ("000660", 1 / 3, 50_000, 0.5),
                      ("005380", 1 / 3, 9_990, 0.1)], decided="152000")
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152115")] * 30,
                        decisions=[pd.DataFrame(), pd.DataFrame(), early])
    assert runner.sleeps == [sut.ORDER_DECISION_POLL_SECONDS] * 2
    assert report.outcome == "OK"
    assert len(broker.submit_calls) == 3


def test_missing_decision_at_wait_deadline_is_skip(mock_binding, runner) -> None:
    broker = FakeBroker()
    clock = [pd.Timestamp(f"{DAY} 15:21:15", tz="Asia/Seoul") + pd.Timedelta(seconds=60 * i)
             for i in range(12)]
    report = runner.run("entry", DAY, broker=broker, journal=journal_holder(runner),
                        clock=clock, decisions=[pd.DataFrame()])
    assert (report.outcome, report.reason) == ("SKIPPED", "no_persisted_decision")
    broker.assert_quiet()
    assert max(runner.clock_log) >= ts("152800")


def journal_holder(runner: Runner) -> OrderJournal:
    return runner.journal()


def test_decision_after_auction_close_is_refused(mock_binding, runner) -> None:
    broker = FakeBroker()
    frame = decision([("005930", 1.0, 70_000, 0.5)], decided="153001")
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152700")] * 6, decisions=[frame])
    assert report.outcome == "DEGRADED" and report.reason.startswith("invalid_decision")
    broker.assert_quiet()


def test_decision_from_another_day_is_refused(mock_binding, runner) -> None:
    broker = FakeBroker()
    frame = decision([("005930", 1.0, 70_000, 0.5)], day=YESTERDAY)
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152700")] * 6, decisions=[frame])
    assert report.outcome == "DEGRADED" and report.reason.startswith("invalid_decision")
    broker.assert_quiet()


def test_decision_in_future_is_refused(mock_binding, runner) -> None:
    broker = FakeBroker()
    frame = decision([("005930", 1.0, 70_000, 0.5)], decided="152800")
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152700")] * 6, decisions=[frame])
    assert report.outcome == "DEGRADED" and report.reason.startswith("invalid_decision")
    broker.assert_quiet()


def test_naive_decision_timestamp_is_refused(mock_binding, runner) -> None:
    broker = FakeBroker()
    frame = pd.DataFrame([{"symbol": "005930", "allocation": 1.0, "close": 70_000, "pred": 0.5,
                           "decided_at": pd.Timestamp(f"{DAY} 15:25:00")}])
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152700")] * 6, decisions=[frame])
    assert report.outcome == "DEGRADED" and report.reason.startswith("invalid_decision")
    broker.assert_quiet()


def test_over_allocated_decision_is_refused(mock_binding, runner) -> None:
    broker = FakeBroker()
    frame = decision([("005930", 0.5, 70_000, 0.5), ("000660", 0.5, 50_000, 0.4), ("005380", 0.2, 9_990, 0.3)])
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152700")] * 6, decisions=[frame])
    assert report.outcome == "DEGRADED" and report.reason.startswith("invalid_decision")
    broker.assert_quiet()


def test_entry_sends_buffered_limit_buys_on_broker_cash(mock_binding, runner) -> None:
    broker = FakeBroker()
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert report.outcome == "OK"
    assert [i.split(":")[1] for i in broker.submit_calls] == ["005930", "000660", "005380"]
    states = runner_states(runner, DAY, "mock")
    for symbol, close in (("005930", 263_000), ("000660", 50_000), ("005380", 9_990)):
        state = states[entry_intent_id(DAY, symbol)]
        assert state.intent is not None and state.intent.limit_price == entry_limit_price(close, 300.0, 1_000_000)
    assert report.metrics["planned_notional_krw"] <= 9_950_000
    assert report.metrics["n_acked"] == 3


def runner_states(runner: Runner, day: str, env: str) -> Any:
    return OrderJournal(runner.tmp_path, env, lock_timeout_seconds=5.0).states()


def test_seed_caps_capital_below_broker_cash(mock_binding, runner, monkeypatch) -> None:
    from src.config import settings

    monkeypatch.setattr(settings, "ORDER_SEED_CAPITAL", 10_000_000)
    broker = FakeBroker()
    broker.cash_value = 50_000_000
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert report.outcome == "OK"
    assert report.metrics["planned_notional_krw"] <= 10_000_000


def test_unreadable_cash_plans_nothing(mock_binding, runner) -> None:
    broker = FakeBroker()
    broker.cash_error = BrokerReadError("no cash")
    journal = runner.journal()
    report = runner.run("entry", DAY, broker=broker, journal=journal,
                        clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert (report.outcome, report.reason) == ("DEGRADED", "orderable_cash_unreadable")
    assert broker.submit_calls == []
    assert [e for e in journal.load() if e.payload.get("trading_date") == DAY] == []


def test_unpriceable_pick_is_skipped(mock_binding, runner) -> None:
    broker = FakeBroker()
    broker.price_errors = {"000660": BrokerReadError("no band")}
    journal = runner.journal()
    report = runner.run("entry", DAY, broker=broker, journal=journal,
                        clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert (report.outcome, report.reason) == ("DEGRADED", "entry_incomplete")
    assert len(broker.submit_calls) == 2
    skips = [e for e in journal.load() if e.kind is JournalEventKind.SKIP]
    assert any(e.intent_id == entry_intent_id(DAY, "000660") and e.payload["reason"] == "no_price_limits"
               for e in skips)


def test_second_trigger_reuses_stored_plan(mock_binding, runner) -> None:
    broker = FakeBroker()
    journal = runner.journal()
    first = runner.run("entry", DAY, broker=broker, journal=journal,
                       clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert first.outcome == "OK"
    before = [(e.kind, e.intent_id) for e in journal.load()]
    broker.cash_calls = 0
    broker.price_calls.clear()
    broker.submit_calls.clear()
    broker.cash_value = 1_000
    runner.sleeps.clear()
    second = runner.run("entry", DAY, broker=broker, journal=journal,
                        clock=[ts("152710")] * 30, decisions=[three_picks()])
    assert second.outcome == "OK"
    assert broker.cash_calls == 0 and broker.price_calls == [] and broker.submit_calls == []
    assert [(e.kind, e.intent_id) for e in journal.load()] == before


def test_crash_after_planning_resumes_without_resizing(mock_binding, runner) -> None:
    broker = FakeBroker()
    journal = runner.journal()
    stored = [_entry_intent("005930", DAY, 7, 70_000), _entry_intent("000660", DAY, 11, 50_000),
              _entry_intent("005380", DAY, 13, 9_990)]
    from src.execution.order_gateway import GatewayLimits, OrderGateway
    gateway = OrderGateway(broker, journal, GatewayLimits(10**12, 10**12, 100, 3, 1, 0.0, False),
                           run_id="setup", now_fn=lambda: __import__("datetime").datetime.now(),
                           sleep_fn=runner._sleep)
    with journal.exclusive():
        gateway.plan(stored, [])
    broker.cash_value = 1_000
    report = runner.run("entry", DAY, broker=broker, journal=journal,
                        clock=[ts("152710")] * 30, decisions=[three_picks()])
    assert report.outcome == "OK"
    assert broker.cash_calls == 0 and broker.price_calls == []
    assert broker.submit_calls == [i.intent_id for i in stored]


def test_plan_is_durable_before_first_send(mock_binding, runner) -> None:
    broker = FakeBroker()
    journal = runner.journal()

    def _probe() -> None:
        intents = [e for e in journal.load() if e.kind is JournalEventKind.INTENT]
        assert len(intents) == 3

    broker.journal_probe = _probe
    report = runner.run("entry", DAY, broker=broker, journal=journal,
                        clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert report.outcome == "OK"


def test_rejected_order_degrades_run(mock_binding, runner) -> None:
    broker = FakeBroker()
    broker.submit_script = [_accepted("a1"), _rejected(), _accepted("a3")]
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert (report.outcome, report.reason) == ("DEGRADED", "entry_incomplete")
    assert report.metrics["n_rejected"] == 1 and report.metrics["n_acked"] == 2


def test_entry_metrics_expose_timing(mock_binding, runner) -> None:
    broker = FakeBroker()
    moment = ts("152700")
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[moment] * 40, decisions=[three_picks()])
    assert report.outcome == "OK"
    assert report.metrics["decided_to_first_send_s"] == 0.0
    assert report.metrics["last_send_seconds_before_deadline"] == (
        pd.Timestamp(f"{DAY} 15:29:00", tz="Asia/Seoul") - moment).total_seconds() > 0


def test_exit_without_open_lots_is_skip(mock_binding, runner) -> None:
    broker = FakeBroker()
    report = runner.run("exit", DAY, broker=broker, journal=runner.journal(), clock=[ts("084010")] * 3)
    assert (report.outcome, report.reason) == ("SKIPPED", "no_open_positions")
    broker.assert_quiet()


def test_exit_ignores_lots_entered_today(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(DAY, "005930"), "005930", DAY, 10, 70_000)
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084010")] * 3)
    assert (report.outcome, report.reason) == ("SKIPPED", "no_open_positions")


def test_exit_sells_each_lot_at_market(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "000660"), "000660", YESTERDAY, 4, 50_000)
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10),
                        BrokerPosition(symbol="000660", holding_qty=4, sellable_qty=4)]
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084010")] * 30)
    assert report.outcome == "OK"
    assert len(broker.submit_calls) == 2
    states = OrderJournal(runner.tmp_path, "mock", lock_timeout_seconds=5.0).states()
    for parent in (entry_intent_id(YESTERDAY, "005930"), entry_intent_id(YESTERDAY, "000660")):
        sito = [st for st in states.values()
                if st.intent is not None and st.intent.parent_intent_id == parent]
        assert len(sito) == 1 and sito[0].intent is not None
        assert sito[0].intent.order_type.value == "market" and sito[0].intent.limit_price is None


def test_exit_never_sells_more_than_journal_lot(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=25, sellable_qty=25)]
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084010")] * 30)
    assert report.outcome == "OK"
    states = OrderJournal(runner.tmp_path, "mock", lock_timeout_seconds=5.0).states()
    qtys = [st.intent.qty for st in states.values()
            if st.intent is not None and st.intent.phase is OrderPhase.EXIT]
    assert qtys == [10]


def test_exit_reports_shortfall(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=6)]
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084010")] * 30)
    assert (report.outcome, report.reason) == ("DEGRADED", "exit_incomplete")
    assert report.metrics["shortfall_qty"] == 4


def test_unsellable_lot_is_skipped(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=0)]
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084010")] * 30)
    assert report.outcome == "DEGRADED"
    assert broker.submit_calls == []
    skips = [e for e in journal.load() if e.kind is JournalEventKind.SKIP]
    assert any(e.payload["reason"] == "not_sellable" for e in skips)


def test_exit_after_deadline_sends_nothing(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("085900")] * 4)
    assert (report.outcome, report.reason) == ("DEGRADED", "exit_window_expired")
    assert broker.submit_calls == []


def test_exit_second_trigger_does_not_resend(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
    first = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084010")] * 30)
    assert first.outcome == "OK"
    broker.submit_calls.clear()
    broker.positions_calls = 0
    runner.sleeps.clear()
    second = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084020")] * 30)
    assert second.outcome == "OK"
    assert broker.submit_calls == [] and broker.positions_calls == 0


def test_reconcile_without_orders_is_skip(mock_binding, runner) -> None:
    broker = FakeBroker()
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("153100")] * 5)
    assert (report.outcome, report.reason) == ("SKIPPED", "no_orders")
    broker.assert_quiet()


def test_entry_reconcile_stops_when_final(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    for symbol, qty, price, odno in (("005930", 10, 70_000, "n1"), ("000660", 4, 50_000, "n2"),
                                     ("005380", 7, 9_990, "n3")):
        _add_acked(journal, _entry_intent(symbol, DAY, qty, price), odno)
    broker.positions = [BrokerPosition(symbol=s, holding_qty=100, sellable_qty=100)
                        for s in ("005930", "000660", "005380")]
    broker.daily_script = [[
        _broker_row("n1", "005930", OrderSide.BUY, 10, 70_000, 10, 701_000),
        _broker_row("n2", "000660", OrderSide.BUY, 4, 50_000, 4, 200_000),
        _broker_row("n3", "005380", OrderSide.BUY, 7, 9_990, 7, 69_930),
    ]]
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=journal, clock=[ts("153100")] * 10)
    assert broker.daily_calls == 1
    assert report.outcome == "OK"
    assert report.metrics["entry_fill_vs_decision_bp"] == pytest.approx(
        ((701_000 + 200_000 + 69_930) / (10 * 70_000 + 4 * 50_000 + 7 * 9_990) - 1) * 1e4)


def test_entry_reconcile_finalizes_dead_remainder(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_acked(journal, _entry_intent("005930", DAY, 10, 70_000), "n1")
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
    broker.daily_script = [
        [_broker_row("n1", "005930", OrderSide.BUY, 10, 70_000, 4, 280_000)] for _ in range(12)]
    clock = [pd.Timestamp(f"{DAY} 15:31:00", tz="Asia/Seoul") + pd.Timedelta(seconds=60 * i)
             for i in range(14)]
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=journal, clock=clock)
    assert runner.sleeps and all(s == sut.ORDER_RECONCILE_POLL_SECONDS for s in runner.sleeps)
    states = OrderJournal(runner.tmp_path, "mock", lock_timeout_seconds=5.0).states()
    assert states[entry_intent_id(DAY, "005930")].status.value == "partial"
    assert (report.outcome, report.reason) == ("DEGRADED", "partial_or_unfilled")


def test_exit_reconcile_never_finalizes_working_sell(mock_binding, runner) -> None:
    from src.execution.order_contract import exit_intent_id as _exit_id

    broker, journal = FakeBroker(), runner.journal()
    parent = entry_intent_id(YESTERDAY, "005930")
    intent = OrderIntent(intent_id=_exit_id(parent, DAY), trading_date=DAY, phase=OrderPhase.EXIT,
                         symbol="005930", side=OrderSide.SELL, order_type=OrderType.MARKET,
                         qty=10, limit_price=None, decision_price=None, parent_intent_id=parent)
    _add_acked(journal, intent, "s1")
    broker.daily_script = [
        [_broker_row("s1", "005930", OrderSide.SELL, 10, 70_000, 4, 280_000)] for _ in range(30)]
    clock = [pd.Timestamp(f"{DAY} 09:01:00", tz="Asia/Seoul") + pd.Timedelta(seconds=60 * i)
             for i in range(32)]
    report = runner.run("reconcile-exit", DAY, broker=broker, journal=journal, clock=clock)
    assert (report.outcome, report.reason) == ("DEGRADED", "exit_working")
    states = OrderJournal(runner.tmp_path, "mock", lock_timeout_seconds=5.0).states()
    assert states[intent.intent_id].final is False
    assert all(e.payload.get("final") is False
               for e in journal.load() if e.kind.value == "broker_state")


def test_reconcile_lock_not_held_across_sleeps(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_acked(journal, _entry_intent("005930", DAY, 10, 70_000), "n1")
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
    broker.daily_script = [
        [_broker_row("n1", "005930", OrderSide.BUY, 10, 70_000, 4, 280_000)] for _ in range(12)]

    async def _guarded_sleep(delay: float) -> None:
        assert journal.locked is False
        runner.sleeps.append(delay)

    now_fn, _ = make_clock(
        [pd.Timestamp(f"{DAY} 15:31:00", tz="Asia/Seoul") + pd.Timedelta(seconds=60 * i) for i in range(14)])
    recorded: list[dict[str, Any]] = []
    report = asyncio.run(sut.run_order_session(
        "reconcile-entry", pd.Timestamp(DAY), broker=broker, journal=journal, session=object(),
        now_fn=now_fn, sleep_fn=_guarded_sleep,
        record_fn=lambda o, **k: recorded.append({"outcome": o, **k}),
        session_day_fn=_standard_day, decision_fn=lambda _d: pd.DataFrame()))
    assert report.outcome == "DEGRADED"


def test_reconcile_survives_transient_read_failure(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_acked(journal, _entry_intent("005930", DAY, 10, 70_000), "n1")
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
    broker.daily_script = [BrokerReadError("boom"),
                           [_broker_row("n1", "005930", OrderSide.BUY, 10, 70_000, 10, 700_000)]]
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=journal, clock=[ts("153100")] * 10)
    assert report.outcome == "OK"


def test_reconcile_reports_unreadable_broker(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_acked(journal, _entry_intent("005930", DAY, 10, 70_000), "n1")
    broker.daily_script = [BrokerReadError("down")] * 12
    before = [(e.kind, e.intent_id, dict(e.payload)) for e in journal.load()]
    clock = [pd.Timestamp(f"{DAY} 15:31:00", tz="Asia/Seoul") + pd.Timedelta(seconds=60 * i)
             for i in range(14)]
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=journal, clock=clock)
    assert (report.outcome, report.reason) == ("DEGRADED", "broker_unreadable")
    assert [(e.kind, e.intent_id, dict(e.payload)) for e in journal.load()] == before


def test_backdated_reconcile_runs_one_finalizing_pass(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_acked(journal, _entry_intent("005930", YESTERDAY, 10, 70_000), "n1")
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
    broker.daily_script = [[_broker_row("n1", "005930", OrderSide.BUY, 10, 70_000, 10, 700_000)]]
    report = runner.run("reconcile-entry", YESTERDAY, broker=broker, journal=journal,
                        clock=[ts("152700")] * 6)
    assert broker.daily_calls == 1 and runner.sleeps == []
    assert report.outcome == "OK"
    finals = [e.payload.get("final") for e in journal.load() if e.kind.value == "broker_state"]
    assert finals and all(finals)


def _halted_journal(runner: Runner, name: str) -> OrderJournal:
    root = runner.tmp_path / name
    journal = OrderJournal(root, "mock", lock_timeout_seconds=5.0)
    (root / "mock" / "HALT").write_text("halt\n", encoding="utf-8")
    return journal


def test_halt_blocks_entry_and_exit_but_not_cancel(mock_binding, runner) -> None:
    broker = FakeBroker()
    entry_journal = _halted_journal(runner, "h1")
    entry = runner.run("entry", DAY, broker=broker, journal=entry_journal,
                       clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert entry.outcome == "DEGRADED"
    assert broker.submit_calls == []
    assert any(e.kind is JournalEventKind.SKIP and e.payload.get("reason") == "halted"
               for e in entry_journal.load())

    exit_journal = _halted_journal(runner, "h2")
    _add_filled_entry(exit_journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    exited = runner.run("exit", DAY, broker=broker, journal=exit_journal, clock=[ts("084010")] * 30)
    assert exited.outcome == "DEGRADED"
    assert broker.submit_calls == []
    assert any(e.kind is JournalEventKind.SKIP and e.payload.get("reason") == "halted"
               for e in exit_journal.load())

    cancel_journal = _halted_journal(runner, "h3")
    _add_acked(cancel_journal, _entry_intent("000660", DAY, 4, 50_000), "work-1")
    cancelled = runner.run("cancel", DAY, broker=broker, journal=cancel_journal, clock=[ts("120000")] * 6)
    assert cancelled.outcome == "OK"
    assert broker.cancel_calls == ["work-1"]


def test_mock_and_live_journals_never_mix(tmp_path, monkeypatch, runner) -> None:
    from src.config import settings

    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sut, "resolve_trading_binding", lambda: _binding("mock"))
    broker = FakeBroker()
    journal = OrderJournal(tmp_path / "orders", "mock", lock_timeout_seconds=5.0)
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    monkeypatch.setattr(sut, "resolve_trading_binding", lambda: _binding("live"))
    now_fn, _ = make_clock([ts("084010")] * 4)
    recorded: list[dict[str, Any]] = []
    report = asyncio.run(sut.run_order_session(
        "exit", pd.Timestamp(DAY), broker=broker, journal=None, session=object(), now_fn=now_fn,
        sleep_fn=runner._sleep, record_fn=lambda o, **k: recorded.append({"outcome": o, **k}),
        session_day_fn=_standard_day, decision_fn=lambda _d: pd.DataFrame()))
    assert (report.outcome, report.reason) == ("SKIPPED", "no_open_positions")


def _add_unknown(journal: OrderJournal, intent: OrderIntent) -> None:
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="setup")
        journal.append(JournalEventKind.SEND, intent.intent_id, {"attempt": 1}, run_id="setup")
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "ambiguous", "broker_order_no": None,
                        "broker_org_no": None, "http_status": None, "msg_cd": "",
                        "msg": "timeout", "latency_ms": 1.0}, run_id="setup")


def test_validate_decision_rule_branches() -> None:
    now = ts("152700")
    base = {"symbol": "005930", "allocation": 1.0, "close": 70_000, "pred": 0.5,
            "decided_at": ts("152500")}

    def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    import math as _math
    assert _math.isnan(sut._pred_value(None))
    assert _math.isnan(sut._pred_value("n/a"))
    assert sut._rank_key(float("nan"), "005930")[0] == 1

    picks, rule = sut._validate_decision(_frame([{k: v for k, v in base.items() if k != "close"}]),
                                         date_str=DAY, now=now, max_sends=10)
    assert picks is None and rule == "invalid_decision:missing_column_close"
    picks, rule = sut._validate_decision(
        pd.DataFrame(columns=["symbol", "allocation", "close", "decided_at"]),
        date_str=DAY, now=now, max_sends=10)
    assert rule == "invalid_decision:empty"
    many = [{**base, "symbol": f"{i:06d}"} for i in range(11)]
    picks, rule = sut._validate_decision(_frame(many), date_str=DAY, now=now, max_sends=10)
    assert rule == "invalid_decision:too_many_rows"
    picks, rule = sut._validate_decision(_frame([{**base, "symbol": "ABC"}]), date_str=DAY,
                                         now=now, max_sends=10)
    assert rule == "invalid_decision:bad_symbol"
    picks, rule = sut._validate_decision(
        _frame([{**base}, {**base, "pred": 0.1}]), date_str=DAY, now=now, max_sends=10)
    assert rule == "invalid_decision:duplicate_symbol"
    picks, rule = sut._validate_decision(_frame([{**base, "allocation": "lots"}]), date_str=DAY,
                                         now=now, max_sends=10)
    assert rule == "invalid_decision:bad_allocation"
    picks, rule = sut._validate_decision(_frame([{**base, "allocation": float("inf")}]),
                                         date_str=DAY, now=now, max_sends=10)
    assert rule == "invalid_decision:bad_allocation"
    picks, rule = sut._validate_decision(_frame([{**base, "close": True}]), date_str=DAY,
                                         now=now, max_sends=10)
    assert rule == "invalid_decision:bad_close"
    picks, rule = sut._validate_decision(_frame([{**base, "close": 70_000.5}]), date_str=DAY,
                                         now=now, max_sends=10)
    assert rule == "invalid_decision:bad_close"
    picks, rule = sut._validate_decision(_frame([{**base, "close": 0}]), date_str=DAY,
                                         now=now, max_sends=10)
    assert rule == "invalid_decision:bad_close"
    picks, rule = sut._validate_decision(_frame([{**base, "decided_at": "not-a-time"}]),
                                         date_str=DAY, now=now, max_sends=10)
    assert rule == "invalid_decision:bad_decided_at"
    picks, rule = sut._validate_decision(_frame([{**base, "decided_at": ts("151900")}]),
                                         date_str=DAY, now=now, max_sends=10)
    assert rule == "invalid_decision:decided_before_window"
    good, rule = sut._validate_decision(_frame([{**base, "close": 70_000.0, "pred": float("nan")}]),
                                        date_str=DAY, now=now, max_sends=10)
    assert rule is None and good is not None and good[0]["close"] == 70_000


def test_entry_counts_and_latency_helpers(runner) -> None:
    journal = runner.journal()
    rec = _entry_intent("005930", DAY, 10, 70_000)
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, rec.intent_id, intent_to_payload(rec), run_id="r")
        journal.append(JournalEventKind.SKIP, "skip-1",
                       {"reason": "x", "symbol": "0", "trading_date": DAY, "phase": "entry"}, run_id="r")
        journal.append(JournalEventKind.RESULT, rec.intent_id,
                       {"attempt": 1, "verdict": "ambiguous", "latency_ms": "bad",
                        "broker_order_no": None, "broker_org_no": None,
                        "http_status": None, "msg_cd": "", "msg": ""}, run_id="r")
    _add_unknown(journal, _entry_intent("000660", DAY, 4, 50_000))
    recorded_only = _entry_intent("005380", DAY, 7, 9_990)
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, recorded_only.intent_id,
                       intent_to_payload(recorded_only), run_id="r")
    counts = sut._entry_counts(journal, [rec.intent_id, "ghost", _entry_intent("000660", DAY).intent_id,
                                         "skip-1", recorded_only.intent_id])
    assert counts == {"n_acked": 0, "n_rejected": 0, "n_unknown": 2, "n_not_sent": 2, "n_skipped": 1}
    assert sut._max_latency_ms(journal, "r") == 0.0
    assert sut._has_entry_history(journal, YESTERDAY) is False
    assert sut._stored_entry_intents(journal, YESTERDAY) == []


def test_entry_token_failure_plans_nothing(mock_binding, runner) -> None:
    broker = FakeBroker()
    broker.ensure_error = BrokerReadError("token down")
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert (report.outcome, report.reason) == ("DEGRADED", "orderable_cash_unreadable")
    assert broker.submit_calls == [] and broker.cash_calls == 0


def test_zero_qty_pick_is_skipped(mock_binding, runner) -> None:
    broker = FakeBroker()
    frame = decision([("005930", 0.5, 70_000, 0.9), ("000660", 0.499999, 50_000, 0.5),
                      ("999999", 0.000001, 100_000, 0.1)])
    journal = runner.journal()
    report = runner.run("entry", DAY, broker=broker, journal=journal,
                        clock=[ts("152700")] * 30, decisions=[frame])
    assert (report.outcome, report.reason) == ("DEGRADED", "entry_incomplete")
    assert len(broker.submit_calls) == 2
    assert any(e.kind is JournalEventKind.SKIP and e.payload["reason"] == "zero_qty"
               for e in journal.load())


def test_entry_post_send_reconcile_failure_is_recorded(mock_binding, runner) -> None:
    broker = FakeBroker()
    broker.daily_script = [BrokerReadError("unreadable")]
    report = runner.run("entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert report.outcome == "OK"
    assert report.metrics["post_send_reconcile"] == "unreadable"


def test_stored_entry_post_send_failure_is_recorded(mock_binding, runner) -> None:
    broker = FakeBroker()
    journal = runner.journal()
    first = runner.run("entry", DAY, broker=broker, journal=journal,
                       clock=[ts("152700")] * 30, decisions=[three_picks()])
    assert first.outcome == "OK"
    broker.daily_script = [BrokerReadError("unreadable")]
    second = runner.run("entry", DAY, broker=broker, journal=journal,
                        clock=[ts("152710")] * 30, decisions=[three_picks()])
    assert second.outcome == "OK"
    assert second.metrics["post_send_reconcile"] == "unreadable"


def test_planted_history_between_checks_is_authoritative(mock_binding, runner) -> None:
    broker = FakeBroker()
    journal = runner.journal()
    planted = [_entry_intent("005930", DAY, 7, 70_000), _entry_intent("000660", DAY, 11, 50_000)]

    def _planted_decision(_when: pd.Timestamp) -> pd.DataFrame:
        from src.execution.order_gateway import GatewayLimits, OrderGateway

        gateway = OrderGateway(broker, journal, GatewayLimits(10**12, 10**12, 100, 3, 1, 0.0, False),
                               run_id="plant", now_fn=lambda: ts("152700").to_pydatetime(),
                               sleep_fn=runner._sleep)
        with journal.exclusive():
            gateway.plan(planted, [])
        return three_picks()

    now_fn, _ = make_clock([ts("152700")] * 30)
    recorded: list[dict[str, Any]] = []
    report = asyncio.run(sut.run_order_session(
        "entry", pd.Timestamp(DAY), broker=broker, journal=journal, session=object(),
        now_fn=now_fn, sleep_fn=runner._sleep,
        record_fn=lambda o, **k: recorded.append({"outcome": o, **k}),
        session_day_fn=_standard_day, decision_fn=_planted_decision))
    assert report.outcome == "OK"
    assert broker.cash_calls == 0 and broker.price_calls == []
    assert broker.submit_calls == [i.intent_id for i in planted]


def test_exit_wait_aborts_when_lots_vanish(mock_binding, runner) -> None:
    from src.execution.order_contract import exit_intent_id as _exit_id

    broker, journal = FakeBroker(), runner.journal()
    parent = entry_intent_id(YESTERDAY, "005930")
    _add_filled_entry(journal, parent, "005930", YESTERDAY, 10, 70_000)

    async def _sleep_and_sell(delay: float) -> None:
        runner.sleeps.append(delay)
        sold = OrderIntent(intent_id=_exit_id(parent, DAY), trading_date=DAY, phase=OrderPhase.EXIT,
                           symbol="005930", side=OrderSide.SELL, order_type=OrderType.MARKET,
                           qty=10, limit_price=None, decision_price=None, parent_intent_id=parent)
        with journal.exclusive():
            journal.append(JournalEventKind.INTENT, sold.intent_id, intent_to_payload(sold), run_id="ext")
            journal.append(JournalEventKind.BROKER_STATE, sold.intent_id,
                           {"order_qty": 10, "filled_qty": 10, "filled_amount_krw": 700_000,
                            "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                           run_id="ext")

    now_fn, _ = make_clock([ts("083900")] * 6)
    recorded: list[dict[str, Any]] = []
    report = asyncio.run(sut.run_order_session(
        "exit", pd.Timestamp(DAY), broker=broker, journal=journal, session=object(),
        now_fn=now_fn, sleep_fn=_sleep_and_sell,
        record_fn=lambda o, **k: recorded.append({"outcome": o, **k}),
        session_day_fn=_standard_day, decision_fn=lambda _d: pd.DataFrame()))
    assert (report.outcome, report.reason) == ("SKIPPED", "no_open_positions")
    assert runner.sleeps == [60.0]
    assert broker.submit_calls == []


def test_finish_exit_report_ignores_ghost_ids(runner) -> None:
    report = sut._finish_exit_report(
        journal=runner.journal(), lots=[], exit_ids=["ghost"], n_skipped=0,
        date_str=DAY, env="mock", t_last=ts("084010"),
        deadline=ts("085900").to_pydatetime(), run_id="r")
    assert report.metrics["shortfall_qty"] == 0


def test_exit_waits_then_proceeds_when_lots_remain(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("083900")] * 10)
    assert runner.sleeps == [60.0]
    assert (report.outcome, report.reason) == ("DEGRADED", "exit_incomplete")


def test_exit_before_window_is_skipped(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("083000")] * 4)
    assert (report.outcome, report.reason) == ("SKIPPED", "before_window")
    broker.assert_quiet()


def test_exit_token_and_position_failures_plan_nothing(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    broker.ensure_error = BrokerReadError("token down")
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084010")] * 10)
    assert (report.outcome, report.reason) == ("DEGRADED", "positions_unreadable")
    broker.ensure_error = None
    broker.positions_error = BrokerReadError("balance down")
    runner.recorded.clear()
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084010")] * 10)
    assert (report.outcome, report.reason) == ("DEGRADED", "positions_unreadable")
    assert broker.submit_calls == []


def test_exit_post_send_reconcile_failure_is_recorded(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_filled_entry(journal, entry_intent_id(YESTERDAY, "005930"), "005930", YESTERDAY, 10, 70_000)
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
    broker.daily_script = [BrokerReadError("unreadable")]
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084010")] * 30)
    assert report.outcome == "OK"
    assert report.metrics["post_send_reconcile"] == "unreadable"


def test_reconcile_entry_waits_for_window(mock_binding, runner) -> None:
    broker = FakeBroker()
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("152900")] * 6)
    assert (report.outcome, report.reason) == ("SKIPPED", "no_orders")
    assert runner.sleeps == [90.0]


def test_reconcile_before_window_is_skipped(mock_binding, runner) -> None:
    broker = FakeBroker()
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=runner.journal(),
                        clock=[ts("150000")] * 4)
    assert (report.outcome, report.reason) == ("SKIPPED", "before_window")
    broker.assert_quiet()


def test_reconcile_reports_unknown_orders(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_unknown(journal, _entry_intent("005930", DAY, 10, 70_000))
    broker.daily_script = [[] for _ in range(14)]
    clock = [pd.Timestamp(f"{DAY} 15:31:00", tz="Asia/Seoul") + pd.Timedelta(seconds=60 * i)
             for i in range(16)]
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=journal, clock=clock)
    assert (report.outcome, report.reason) == ("DEGRADED", "unknown_orders")


def test_reconcile_reports_missing_at_broker(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_acked(journal, _entry_intent("005930", DAY, 10, 70_000), "gone-1")
    broker.daily_script = [[] for _ in range(14)]
    clock = [pd.Timestamp(f"{DAY} 15:31:00", tz="Asia/Seoul") + pd.Timedelta(seconds=60 * i)
             for i in range(16)]
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=journal, clock=clock)
    assert (report.outcome, report.reason) == ("DEGRADED", "missing_at_broker")


def test_reconcile_reports_position_shortfall(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    _add_acked(journal, _entry_intent("005930", DAY, 10, 70_000), "n1")
    broker.daily_script = [[_broker_row("n1", "005930", OrderSide.BUY, 10, 70_000, 10, 700_000)]]
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=journal, clock=[ts("153100")] * 10)
    assert (report.outcome, report.reason) == ("DEGRADED", "position_shortfall")


def test_session_is_created_when_not_injected(mock_binding, runner) -> None:
    seen: list[Any] = []

    class SessionBroker(FakeBroker):
        def create_session(self) -> Any:
            class _Ctx:
                async def __aenter__(self) -> str:
                    return "live-sess"

                async def __aexit__(self, *args: Any) -> bool:
                    return False

            return _Ctx()

        async def submit_order(self, session: Any, intent: OrderIntent) -> SubmitResult:
            seen.append(session)
            return await super().submit_order(session, intent)

    broker = SessionBroker()
    now_fn, _ = make_clock([ts("152700")] * 40)
    recorded: list[dict[str, Any]] = []
    report = asyncio.run(sut.run_order_session(
        "entry", pd.Timestamp(DAY), broker=broker, journal=runner.journal(), session=None,
        now_fn=now_fn, sleep_fn=runner._sleep,
        record_fn=lambda o, **k: recorded.append({"outcome": o, **k}),
        session_day_fn=_standard_day, decision_fn=lambda _d: three_picks()))
    assert report.outcome == "OK"
    assert seen and all(s == "live-sess" for s in seen)


def test_none_session_without_factory_is_passed_through(mock_binding, runner) -> None:
    broker = FakeBroker()
    now_fn, _ = make_clock([ts("084010")] * 6)
    recorded: list[dict[str, Any]] = []
    report = asyncio.run(sut.run_order_session(
        "exit", pd.Timestamp(DAY), broker=broker, journal=runner.journal(), session=None,
        now_fn=now_fn, sleep_fn=runner._sleep,
        record_fn=lambda o, **k: recorded.append({"outcome": o, **k}),
        session_day_fn=_standard_day, decision_fn=lambda _d: pd.DataFrame()))
    assert (report.outcome, report.reason) == ("SKIPPED", "no_open_positions")


def test_main_rejects_date_for_order_phases(monkeypatch) -> None:
    import sys

    import pytest

    monkeypatch.setattr(sys, "argv", ["order_session", "--phase", "entry", "--date", DAY])
    with pytest.raises(SystemExit):
        sut.main()


def test_main_runs_cancel_for_past_date(monkeypatch, runner) -> None:
    import sys

    async def _fake_run(phase: str, trading_date: pd.Timestamp, **kwargs: Any) -> Any:
        assert phase == "cancel"
        assert trading_date.strftime("%Y-%m-%d") == DAY
        assert kwargs["record_fn"] is not None
        return sut.OrderSessionReport("cancel", DAY, "OK", "cancel_complete", {"env": "mock"})

    monkeypatch.setattr(sys, "argv", ["order_session", "--phase", "cancel", "--date", DAY])
    monkeypatch.setattr(sut, "run_order_session", _fake_run)
    monkeypatch.setattr(sut, "record_run_outcome", lambda *a, **k: {})
    sut.main()


def test_main_warns_on_degraded_outcome(monkeypatch, caplog) -> None:
    import logging
    import sys

    async def _fake_run(phase: str, trading_date: pd.Timestamp, **kwargs: Any) -> Any:
        return sut.OrderSessionReport("cancel", DAY, "DEGRADED", "broker_unreadable", {"env": "mock"})

    monkeypatch.setattr(sys, "argv", ["order_session", "--phase", "cancel", "--date", DAY])
    monkeypatch.setattr(sut, "run_order_session", _fake_run)
    monkeypatch.setattr(sut, "record_run_outcome", lambda *a, **k: {})
    with caplog.at_level(logging.WARNING):
        sut.main()
    assert "order_session_cli" in caplog.text


def test_module_main_entrypoint(monkeypatch, runner) -> None:
    import runpy
    import sys
    import warnings

    async def _fake_run(phase: str, trading_date: pd.Timestamp, **kwargs: Any) -> Any:
        return sut.OrderSessionReport(phase, DAY, "OK", "done", {"env": "mock"})

    monkeypatch.setattr(sys, "argv", ["order_session", "--phase", "cancel", "--date", DAY])
    monkeypatch.setattr(sut, "run_order_session", _fake_run)
    monkeypatch.setattr(sut, "record_run_outcome", lambda *a, **k: {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("src.daily.order_session", run_name="__main__")


def _exit_intent(parent: str, qty: int = 10) -> OrderIntent:
    return OrderIntent(intent_id=sut.exit_intent_id(parent, DAY), trading_date=DAY, phase=OrderPhase.EXIT,
                       symbol="005930", side=OrderSide.SELL, order_type=OrderType.MARKET,
                       qty=qty, limit_price=None, decision_price=None, parent_intent_id=parent)


def test_skip_only_entry_plan_is_authoritative(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), _halted_journal(runner, "skip-only")
    runner.run("entry", DAY, broker=broker, journal=journal, clock=[ts("152700")] * 30,
               decisions=[three_picks()])
    (runner.tmp_path / "skip-only" / "mock" / "HALT").unlink()
    before = journal.load()
    second = runner.run("entry", DAY, broker=broker, journal=journal, clock=[ts("152710")] * 30,
                        decisions=[three_picks()])
    assert (second.outcome, second.reason) == ("DEGRADED", "entry_incomplete")
    assert second.metrics["n_skipped"] == 3 and second.metrics["n_picks"] == 3
    assert journal.load() == before
    broker.assert_quiet()


def test_second_trigger_keeps_planned_skip_degraded(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    broker.price_errors = {"000660": BrokerReadError("no band")}
    with journal.exclusive():
        journal.append(JournalEventKind.SKIP, entry_intent_id(YESTERDAY, "000660"),
                       {"reason": "zero_qty", "symbol": "000660", "trading_date": YESTERDAY, "phase": "entry"},
                       run_id="setup")
    runner.run("entry", DAY, broker=broker, journal=journal, clock=[ts("152700")] * 30,
               decisions=[three_picks()])
    second = runner.run("entry", DAY, broker=broker, journal=journal, clock=[ts("152710")] * 30)
    assert (second.outcome, second.reason) == ("DEGRADED", "entry_incomplete")
    assert second.metrics["n_skipped"] == 1 and second.metrics["n_picks"] == 3
    assert len(broker.submit_calls) == 2


def test_exit_second_trigger_after_partial_fill_reports_no_shortfall(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    parent = entry_intent_id(YESTERDAY, "005930")
    _add_filled_entry(journal, parent, "005930", YESTERDAY, 10, 70_000)
    _add_acked(journal, _exit_intent(parent), "s1")
    with journal.exclusive():
        journal.append(JournalEventKind.BROKER_STATE, sut.exit_intent_id(parent, DAY),
                       {"order_qty": 10, "filled_qty": 4, "filled_amount_krw": 280_000,
                        "remaining_qty": 6, "cancelled": False, "rejected_qty": 0, "final": False},
                       run_id="setup")
    report = runner.run("exit", DAY, broker=broker, journal=journal, clock=[ts("084020")] * 30)
    assert report.outcome == "OK"
    assert report.metrics["shortfall_qty"] == 0
    assert broker.submit_calls == [] and broker.positions_calls == 0


def test_entry_reconcile_finalizes_working_sell_on_day_without_entries(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    parent = entry_intent_id(YESTERDAY, "005930")
    _add_filled_entry(journal, parent, "005930", YESTERDAY, 10, 70_000)
    _add_acked(journal, _exit_intent(parent), "s1")
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=6, sellable_qty=6)]
    broker.daily_script = [[_broker_row("s1", "005930", OrderSide.SELL, 10, 70_000, 4, 280_000)]]
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=journal, clock=[ts("154000")] * 10)
    assert report.reason != "no_orders"
    state = journal.states()[sut.exit_intent_id(parent, DAY)]
    assert state.final and state.status.value == "partial"
    assert [lot.qty for lot in journal.open_lots()] == [6]


def test_backdated_exit_reconcile_finalizes(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    parent = entry_intent_id("2026-10-08", "005930")
    _add_filled_entry(journal, parent, "005930", "2026-10-08", 10, 70_000)
    intent = OrderIntent(intent_id=sut.exit_intent_id(parent, YESTERDAY), trading_date=YESTERDAY,
                         phase=OrderPhase.EXIT, symbol="005930", side=OrderSide.SELL,
                         order_type=OrderType.MARKET, qty=10, limit_price=None, decision_price=None,
                         parent_intent_id=parent)
    _add_acked(journal, intent, "s1")
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=6, sellable_qty=6)]
    broker.daily_script = [[_broker_row("s1", "005930", OrderSide.SELL, 10, 70_000, 4, 280_000)]]
    runner.run("reconcile-exit", YESTERDAY, broker=broker, journal=journal, clock=[ts("120000")] * 6)
    assert broker.daily_calls == 1 and runner.sleeps == []
    assert journal.states()[intent.intent_id].final


def test_entry_fill_metric_excludes_same_day_sell_proceeds(mock_binding, runner) -> None:
    broker, journal = FakeBroker(), runner.journal()
    parent = entry_intent_id(YESTERDAY, "000660")
    _add_filled_entry(journal, parent, "000660", YESTERDAY, 10, 50_000)
    sell = OrderIntent(intent_id=sut.exit_intent_id(parent, DAY), trading_date=DAY, phase=OrderPhase.EXIT,
                       symbol="000660", side=OrderSide.SELL, order_type=OrderType.MARKET,
                       qty=10, limit_price=None, decision_price=None, parent_intent_id=parent)
    _add_acked(journal, sell, "s1")
    _add_acked(journal, _entry_intent("005930", DAY, 10, 70_000), "n1")
    broker.positions = [BrokerPosition(symbol="005930", holding_qty=10, sellable_qty=10)]
    broker.daily_script = [[
        _broker_row("s1", "000660", OrderSide.SELL, 10, 50_000, 10, 510_000),
        _broker_row("n1", "005930", OrderSide.BUY, 10, 70_000, 10, 701_000),
    ]]
    report = runner.run("reconcile-entry", DAY, broker=broker, journal=journal, clock=[ts("153100")] * 10)
    assert report.metrics["entry_fill_vs_decision_bp"] == pytest.approx((701_000 / 700_000 - 1) * 1e4)
