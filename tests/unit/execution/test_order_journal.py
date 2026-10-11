"""Order journal invariant guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.execution.order_contract import OrderIntent, OrderPhase, OrderSide, OrderType
from src.execution.order_journal import (
    HALT_FILENAME,
    IntentStatus,
    JournalCorruptError,
    JournalEvent,
    JournalEventKind,
    OrderJournal,
    derive_intent_state,
    intent_to_payload,
)


def _entry(intent_id: str = "2026-10-12:005930:entry", qty: int = 10) -> OrderIntent:
    return OrderIntent(
        intent_id=intent_id, trading_date="2026-10-12", phase=OrderPhase.ENTRY,
        symbol="005930", side=OrderSide.BUY, order_type=OrderType.LIMIT,
        qty=qty, limit_price=70000, decision_price=70000, parent_intent_id=None,
    )


def _journal(tmp_path: Path, timeout: float = 5.0) -> OrderJournal:
    return OrderJournal(tmp_path, "mock", lock_timeout_seconds=timeout)


def _append_intent_send(journal: OrderJournal, intent: OrderIntent, run: str = "r1") -> None:
    journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id=run)
    journal.append(JournalEventKind.SEND, intent.intent_id, {"attempt": 1}, run_id=run)


def test_append_requires_lock(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    before = journal.journal_path.exists()
    with pytest.raises(RuntimeError):
        journal.append(JournalEventKind.SKIP, "a", {"reason": "x", "symbol": "005930",
                                                   "trading_date": "2026-10-12", "phase": "entry"}, run_id="r1")
    assert journal.journal_path.exists() == before


def test_events_durable_and_ordered(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry()
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="r1")
        journal.append(JournalEventKind.SEND, intent.intent_id, {"attempt": 1}, run_id="r1")
        journal.append(JournalEventKind.SKIP, "s1", {"reason": "nope", "symbol": "005930",
                                                     "trading_date": "2026-10-12", "phase": "entry"}, run_id="r1")
    other = _journal(tmp_path)
    events = other.load()
    assert [e.seq for e in events] == [1, 2, 3]
    assert events[0].payload["qty"] == 10


def test_environments_isolated(tmp_path: Path) -> None:
    mock = OrderJournal(tmp_path, "mock", lock_timeout_seconds=5.0)
    live = OrderJournal(tmp_path, "live", lock_timeout_seconds=5.0)
    with mock.exclusive():
        mock.append(JournalEventKind.INTENT, _entry().intent_id, intent_to_payload(_entry()), run_id="r1")
    assert live.load() == []


def test_unknown_env_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="env"):
        OrderJournal(tmp_path, "prod", lock_timeout_seconds=5.0)


def test_duplicate_intent_refused(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry()
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="r1")
        with pytest.raises(ValueError, match="duplicate"):
            journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="r1")


def test_corrupt_line_fails_closed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    with journal.exclusive():
        journal.append(JournalEventKind.SKIP, "s1", {"reason": "x", "symbol": "005930",
                                                     "trading_date": "2026-10-12", "phase": "entry"}, run_id="r1")
    with open(journal.journal_path, "a", encoding="utf-8") as handle:
        handle.write('{"seq": 2, "ts": "broken\n')
    with pytest.raises(JournalCorruptError, match="line 2"):
        journal.load()
    with pytest.raises(JournalCorruptError):
        journal.states()


def test_event_without_intent_fails_closed(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal._dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"seq": 1, "ts": "2026-10-12T00:00:00.000+09:00", "kind": "send",
                       "intent_id": "ghost", "run_id": "r1", "payload": {"attempt": 1}})
    journal.journal_path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(JournalCorruptError, match="line 1"):
        journal.load()


def test_second_process_serialized(tmp_path: Path) -> None:
    import fcntl

    from src.utils.file_lock import sidecar_lock_path

    journal = _journal(tmp_path)
    journal._dir.mkdir(parents=True, exist_ok=True)
    journal.journal_path.touch()
    lock_path = sidecar_lock_path(journal.journal_path)
    lock_path.touch()
    with open(lock_path, "w") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            other = OrderJournal(tmp_path, "mock", lock_timeout_seconds=0.2)
            with pytest.raises(TimeoutError), other.exclusive():
                pass
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


def test_send_without_result_unknown(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    with journal.exclusive():
        _append_intent_send(journal, _entry())
        state = journal.states()[_entry().intent_id]
    assert state.status is IntentStatus.UNKNOWN
    assert state.n_sends == 1
    assert state.final is False


def test_ambiguous_result_unknown(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry()
    with journal.exclusive():
        _append_intent_send(journal, intent)
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "ambiguous", "broker_order_no": None,
                        "broker_org_no": None, "http_status": 500, "msg_cd": "X",
                        "msg": "m", "latency_ms": 1.0}, run_id="r1")
        assert journal.states()[intent.intent_id].status is IntentStatus.UNKNOWN


def test_throttled_stays_sendable(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry()
    with journal.exclusive():
        _append_intent_send(journal, intent)
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "throttled", "broker_order_no": None,
                        "broker_org_no": None, "http_status": 500, "msg_cd": "EGW00201",
                        "msg": "m", "latency_ms": 1.0}, run_id="r1")
        assert journal.states()[intent.intent_id].status is IntentStatus.NOT_SENT


def test_accepted_is_acked(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry()
    with journal.exclusive():
        _append_intent_send(journal, intent)
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "12345",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "ok", "latency_ms": 2.0}, run_id="r1")
        state = journal.states()[intent.intent_id]
    assert state.status is IntentStatus.ACKED
    assert state.broker_order_no == "12345"


def test_adopt_resolves_unknown(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry()
    with journal.exclusive():
        _append_intent_send(journal, intent)
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "ambiguous", "broker_order_no": None,
                        "broker_org_no": None, "http_status": None, "msg_cd": "",
                        "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.ADOPT, intent.intent_id,
                       {"broker_order_no": "999", "broker_org_no": "001"}, run_id="r1")
        assert journal.states()[intent.intent_id].status is IntentStatus.ACKED


def test_not_found_after_ambiguous(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry()
    with journal.exclusive():
        _append_intent_send(journal, intent)
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "ambiguous", "broker_order_no": None,
                        "broker_org_no": None, "http_status": None, "msg_cd": "",
                        "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.NOT_FOUND, intent.intent_id,
                       {"n_polls": 2, "span_seconds": 1.0}, run_id="r1")
        assert journal.states()[intent.intent_id].status is IntentStatus.NOT_FOUND


def _acked_with_state(tmp_path: Path, filled: int, order_qty: int = 10, final: bool = True,
                      cancelled: bool = False) -> IntentStatus:
    root = tmp_path / f"j-{filled}-{cancelled}-{final}-{order_qty}"
    journal = OrderJournal(root, "mock", lock_timeout_seconds=5.0)
    intent = _entry(qty=order_qty)
    with journal.exclusive():
        _append_intent_send(journal, intent)
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.BROKER_STATE, intent.intent_id,
                       {"order_qty": order_qty, "filled_qty": filled, "filled_amount_krw": filled * 100,
                        "remaining_qty": order_qty - filled, "cancelled": cancelled,
                        "rejected_qty": 0, "final": final}, run_id="r1")
        return journal.states()[intent.intent_id].status


def test_final_broker_state_dominates(tmp_path: Path) -> None:
    assert _acked_with_state(tmp_path, 10) is IntentStatus.FILLED


def test_partial_final(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry()
    with journal.exclusive():
        _append_intent_send(journal, intent)
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.BROKER_STATE, intent.intent_id,
                       {"order_qty": 10, "filled_qty": 4, "filled_amount_krw": 400,
                        "remaining_qty": 6, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="r1")
        state = journal.states()[intent.intent_id]
    assert state.status is IntentStatus.PARTIAL
    assert state.filled_qty == 4
    assert state.final is True


def test_cancelled_and_unfilled_final(tmp_path: Path) -> None:
    assert _acked_with_state(tmp_path, 0, cancelled=True) is IntentStatus.CANCELLED
    assert _acked_with_state(tmp_path, 0, cancelled=False) is IntentStatus.UNFILLED


def test_overfill_corrupt(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry(qty=10)
    with journal.exclusive():
        _append_intent_send(journal, intent)
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.BROKER_STATE, intent.intent_id,
                       {"order_qty": 10, "filled_qty": 11, "filled_amount_krw": 1100,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="r1")
        with pytest.raises(JournalCorruptError):
            journal.states()


def test_fill_decrease_corrupt(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry()
    with journal.exclusive():
        _append_intent_send(journal, intent)
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.BROKER_STATE, intent.intent_id,
                       {"order_qty": 10, "filled_qty": 5, "filled_amount_krw": 500,
                        "remaining_qty": 5, "cancelled": False, "rejected_qty": 0, "final": False},
                       run_id="r1")
        journal.append(JournalEventKind.BROKER_STATE, intent.intent_id,
                       {"order_qty": 10, "filled_qty": 3, "filled_amount_krw": 300,
                        "remaining_qty": 7, "cancelled": False, "rejected_qty": 0, "final": False},
                       run_id="r1")
        with pytest.raises(JournalCorruptError):
            journal.states()


def _exit_intent(entry_id: str, qty: int = 4) -> OrderIntent:
    return OrderIntent(
        intent_id=f"{entry_id}:exit:2026-10-13", trading_date="2026-10-13", phase=OrderPhase.EXIT,
        symbol="005930", side=OrderSide.SELL, order_type=OrderType.LIMIT,
        qty=qty, limit_price=70000, decision_price=None, parent_intent_id=entry_id,
    )


def test_open_lot_minus_exits(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    entry = _entry()
    exit1 = _exit_intent(entry.intent_id, qty=4)
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, entry.intent_id, intent_to_payload(entry), run_id="r1")
        journal.append(JournalEventKind.INTENT, exit1.intent_id, intent_to_payload(exit1), run_id="r1")
        for iid in (entry.intent_id, exit1.intent_id):
            journal.append(JournalEventKind.SEND, iid, {"attempt": 1}, run_id="r1")
            journal.append(JournalEventKind.RESULT, iid,
                           {"attempt": 1, "verdict": "accepted", "broker_order_no": f"no-{iid[-4:]}",
                            "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                            "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.BROKER_STATE, entry.intent_id,
                       {"order_qty": 10, "filled_qty": 10, "filled_amount_krw": 1000,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="r1")
        journal.append(JournalEventKind.BROKER_STATE, exit1.intent_id,
                       {"order_qty": 4, "filled_qty": 4, "filled_amount_krw": 400,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="r1")
        lots = journal.open_lots()
        assert len(lots) == 1
        assert lots[0].qty == 6
        exit2 = _exit_intent(entry.intent_id, qty=6)
        exit2 = OrderIntent(intent_id="2026-10-12:005930:entry:exit:2026-10-14", trading_date="2026-10-14",
                            phase=exit2.phase, symbol=exit2.symbol, side=exit2.side,
                            order_type=exit2.order_type, qty=6, limit_price=70000,
                            decision_price=None, parent_intent_id=entry.intent_id)
        journal.append(JournalEventKind.INTENT, exit2.intent_id, intent_to_payload(exit2), run_id="r1")
        journal.append(JournalEventKind.SEND, exit2.intent_id, {"attempt": 1}, run_id="r1")
        journal.append(JournalEventKind.RESULT, exit2.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "zz",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.BROKER_STATE, exit2.intent_id,
                       {"order_qty": 6, "filled_qty": 6, "filled_amount_krw": 600,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="r1")
        assert journal.open_lots() == []


def test_unfilled_entry_no_lot(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    entry = _entry()
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, entry.intent_id, intent_to_payload(entry), run_id="r1")
        journal.append(JournalEventKind.SEND, entry.intent_id, {"attempt": 1}, run_id="r1")
        journal.append(JournalEventKind.RESULT, entry.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r1")
        journal.append(JournalEventKind.BROKER_STATE, entry.intent_id,
                       {"order_qty": 10, "filled_qty": 0, "filled_amount_krw": 0,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="r1")
        assert journal.open_lots() == []


def test_halt_flag(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    assert journal.is_halted() is False
    (journal._dir / HALT_FILENAME).touch()
    assert journal.is_halted() is True


def test_rejected_terminal_and_skipped(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    intent = _entry()
    with journal.exclusive():
        _append_intent_send(journal, intent)
        journal.append(JournalEventKind.RESULT, intent.intent_id,
                       {"attempt": 1, "verdict": "rejected", "broker_order_no": None,
                        "broker_org_no": None, "http_status": 200, "msg_cd": "40100000",
                        "msg": "no", "latency_ms": 1.0}, run_id="r1")
        state = journal.states()[intent.intent_id]
        assert state.status is IntentStatus.REJECTED
        assert state.final is True
        journal.append(JournalEventKind.SKIP, "only-skip",
                       {"reason": "filtered", "symbol": "005930",
                        "trading_date": "2026-10-12", "phase": "entry"}, run_id="r1")
        assert journal.states()["only-skip"].status is IntentStatus.SKIPPED


def test_derive_empty_raises() -> None:
    with pytest.raises(JournalCorruptError):
        derive_intent_state([])


def test_journal_rejects_bad_payload_and_reentrant(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    with journal.exclusive():
        with pytest.raises(ValueError, match="JSON scalar"):
            journal.append(JournalEventKind.SEND, "x", {"attempt": [1]}, run_id="r1")
        with pytest.raises(ValueError, match="non-blank reason"):
            journal.append(JournalEventKind.SKIP, "x", {"reason": "  ", "symbol": "s",
                                                        "trading_date": "d", "phase": "entry"}, run_id="r1")
        with pytest.raises(RuntimeError, match="re-entrant"), journal.exclusive():
            pass
    with pytest.raises(ValueError, match="lock_timeout_seconds"):
        OrderJournal(tmp_path, "mock", lock_timeout_seconds=0)


def test_load_rejects_unknown_kind_and_nonincreasing_seq(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    journal._dir.mkdir(parents=True, exist_ok=True)
    bad_kind = json.dumps({"seq": 1, "ts": "t", "kind": "nope", "intent_id": "a",
                           "run_id": "r", "payload": {}})
    journal.journal_path.write_text(bad_kind + "\n", encoding="utf-8")
    with pytest.raises(JournalCorruptError, match="unknown kind"):
        journal.load()
    journal.journal_path.write_text("", encoding="utf-8")
    journal2 = _journal(tmp_path)
    with journal2.exclusive():
        journal2.append(JournalEventKind.SKIP, "a", {"reason": "x", "symbol": "005930",
                                                     "trading_date": "2026-10-12", "phase": "entry"}, run_id="r")
        raw = journal2.journal_path.read_text(encoding="utf-8").strip()
        obj = json.loads(raw)
        obj["seq"] = 1
        journal2.journal_path.write_text(json.dumps(obj) + "\n" + json.dumps({**obj, "seq": 1}) + "\n",
                                         encoding="utf-8")
    with pytest.raises(JournalCorruptError, match="non-increasing"):
        journal2.load()


def test_duplicate_intent_line_corrupt_and_negative_exit(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    entry = _entry()
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, entry.intent_id, intent_to_payload(entry), run_id="r1")
        raw = journal.journal_path.read_text(encoding="utf-8").strip()
        obj = json.loads(raw)
        obj["seq"] = 2
        journal.journal_path.write_text(raw + "\n" + json.dumps(obj) + "\n", encoding="utf-8")
    with pytest.raises(JournalCorruptError, match="duplicate INTENT"):
        journal.load()


def test_journal_event_dataclass_shape() -> None:
    ev = JournalEvent(seq=1, ts="t", kind=JournalEventKind.SEND, intent_id="a", run_id="r", payload={})
    assert ev.seq == 1
    assert ev.kind is JournalEventKind.SEND


def _ev(seq: int, kind: JournalEventKind, intent_id: str, payload: dict) -> JournalEvent:
    return JournalEvent(seq=seq, ts="2026-10-12T10:00:00.000+09:00", kind=kind,
                        intent_id=intent_id, run_id="r", payload=payload)


def test_derive_covers_corrupt_branches() -> None:
    intent = _entry()
    base = intent_to_payload(intent)
    with pytest.raises(JournalCorruptError):
        derive_intent_state([_ev(1, JournalEventKind.INTENT, intent.intent_id, base),
                             _ev(1, JournalEventKind.SEND, intent.intent_id, {"attempt": 1})])
    with pytest.raises(JournalCorruptError):
        derive_intent_state([_ev(1, JournalEventKind.INTENT, intent.intent_id, base),
                             _ev(2, JournalEventKind.INTENT, intent.intent_id, base)])
    with pytest.raises(JournalCorruptError):
        derive_intent_state([_ev(1, JournalEventKind.INTENT, intent.intent_id, {"qty": 1}),
                             _ev(2, JournalEventKind.SEND, intent.intent_id, {"attempt": 1})])
    with pytest.raises(JournalCorruptError):
        derive_intent_state([_ev(1, JournalEventKind.INTENT, intent.intent_id, base),
                             _ev(2, JournalEventKind.SEND, intent.intent_id, {"attempt": 1}),
                             _ev(3, JournalEventKind.RESULT, intent.intent_id,
                                  {"attempt": 1, "verdict": "bogus"})])
    skip_only = [JournalEvent(seq=1, ts="t", kind=JournalEventKind.SKIP, intent_id="s",
                              run_id="r", payload={"reason": "x"}),
                 JournalEvent(seq=2, ts="t", kind=JournalEventKind.BROKER_STATE, intent_id="s",
                              run_id="r", payload={"order_qty": 5, "filled_qty": 5,
                                                   "filled_amount_krw": 5, "remaining_qty": 0,
                                                   "cancelled": False, "rejected_qty": 0, "final": True})]
    assert derive_intent_state(skip_only).status is IntentStatus.FILLED
    with pytest.raises(JournalCorruptError):
        derive_intent_state([_ev(1, JournalEventKind.INTENT, intent.intent_id, base),
                             _ev(2, JournalEventKind.SEND, intent.intent_id, {"attempt": 1}),
                             _ev(3, JournalEventKind.RESULT, intent.intent_id,
                                  {"attempt": 1, "verdict": "accepted", "broker_order_no": "1"}),
                             _ev(4, JournalEventKind.BROKER_STATE, intent.intent_id,
                                  {"order_qty": 10, "filled_qty": True, "filled_amount_krw": 0,
                                   "remaining_qty": 0, "cancelled": False, "rejected_qty": 0,
                                   "final": False})])
    with pytest.raises(JournalCorruptError):
        derive_intent_state([_ev(1, JournalEventKind.INTENT, intent.intent_id, base),
                             _ev(2, JournalEventKind.SEND, intent.intent_id, {"attempt": 1}),
                             _ev(3, JournalEventKind.RESULT, intent.intent_id,
                                  {"attempt": 1, "verdict": "accepted", "broker_order_no": "1"}),
                             _ev(4, JournalEventKind.BROKER_STATE, intent.intent_id,
                                  {"order_qty": 10, "filled_qty": -1, "filled_amount_krw": 0,
                                   "remaining_qty": 0, "cancelled": False, "rejected_qty": 0,
                                   "final": False})])
    with pytest.raises(JournalCorruptError):
        derive_intent_state([_ev(1, JournalEventKind.INTENT, intent.intent_id, base),
                             _ev(2, JournalEventKind.SEND, intent.intent_id, {"attempt": 1}),
                             _ev(3, JournalEventKind.RESULT, intent.intent_id,
                                  {"attempt": 1, "verdict": "accepted", "broker_order_no": "1"}),
                             _ev(4, JournalEventKind.BROKER_STATE, intent.intent_id,
                                  {"order_qty": 10, "filled_qty": 6, "filled_amount_krw": 0,
                                   "remaining_qty": 5, "cancelled": False, "rejected_qty": 0,
                                   "final": False})])


def test_load_covers_malformed_lines(tmp_path: Path) -> None:
    def _write(lines: list[str]) -> OrderJournal:
        journal = OrderJournal(tmp_path / "mal", "mock", lock_timeout_seconds=5.0)
        journal._dir.mkdir(parents=True, exist_ok=True)
        journal.journal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return journal

    good_skip = {"seq": 1, "ts": "t", "kind": "skip", "intent_id": "a",
                 "run_id": "r", "payload": {"reason": "x"}}
    cases = [
        [""],
        ["[1,2]"],
        [json.dumps({**good_skip, "seq": 0})],
        [json.dumps({**good_skip, "ts": ""})],
        [json.dumps({**good_skip, "intent_id": ""})],
        [json.dumps({**good_skip, "run_id": ""})],
        [json.dumps({**good_skip, "payload": []})],
        [json.dumps({**good_skip, "payload": {"k": {"nested": 1}}})],
        [json.dumps({"seq": 1, "ts": "t", "kind": "skip"})],
        [json.dumps({**good_skip, "kind": "skip", "payload": {"reason": "x", "k": float("nan")}})],
    ]
    for lines in cases:
        with pytest.raises(JournalCorruptError):
            _write(lines).load()
    journal = _write([json.dumps(good_skip)])
    assert len(journal.load()) == 1
    assert OrderJournal(tmp_path / "fresh", "mock", lock_timeout_seconds=5.0).load() == []
    journal_locked = OrderJournal(tmp_path / "argcheck", "mock", lock_timeout_seconds=5.0)
    with journal_locked.exclusive():
        with pytest.raises(ValueError, match="non-blank"):
            journal_locked.append(JournalEventKind.SEND, "  ", {"attempt": 1}, run_id="r")
        with pytest.raises(ValueError, match="kind"):
            journal_locked.append("send", "x", {"attempt": 1}, run_id="r")  # type: ignore[arg-type]


def test_open_lot_negative_and_sort(tmp_path: Path) -> None:
    journal = _journal(tmp_path)
    entry = OrderIntent(intent_id="2026-10-12:000660:entry", trading_date="2026-10-12",
                        phase=OrderPhase.ENTRY, symbol="000660", side=OrderSide.BUY,
                        order_type=OrderType.LIMIT, qty=5, limit_price=70000,
                        decision_price=70000, parent_intent_id=None)
    with journal.exclusive():
        journal.append(JournalEventKind.INTENT, entry.intent_id, intent_to_payload(entry), run_id="r")
        journal.append(JournalEventKind.SEND, entry.intent_id, {"attempt": 1}, run_id="r")
        journal.append(JournalEventKind.RESULT, entry.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "e1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r")
        journal.append(JournalEventKind.BROKER_STATE, entry.intent_id,
                       {"order_qty": 5, "filled_qty": 5, "filled_amount_krw": 5,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="r")
        big = OrderIntent(intent_id="2026-10-13:000660:exit", trading_date="2026-10-13",
                          phase=OrderPhase.EXIT, symbol="000660", side=OrderSide.SELL,
                          order_type=OrderType.LIMIT, qty=9, limit_price=70000,
                          decision_price=None, parent_intent_id=entry.intent_id)
        journal.append(JournalEventKind.INTENT, big.intent_id, intent_to_payload(big), run_id="r")
        journal.append(JournalEventKind.SEND, big.intent_id, {"attempt": 1}, run_id="r")
        journal.append(JournalEventKind.RESULT, big.intent_id,
                       {"attempt": 1, "verdict": "accepted", "broker_order_no": "x1",
                        "broker_org_no": "001", "http_status": 200, "msg_cd": "0",
                        "msg": "", "latency_ms": 1.0}, run_id="r")
        journal.append(JournalEventKind.BROKER_STATE, big.intent_id,
                       {"order_qty": 9, "filled_qty": 9, "filled_amount_krw": 9,
                        "remaining_qty": 0, "cancelled": False, "rejected_qty": 0, "final": True},
                       run_id="r")
        with pytest.raises(JournalCorruptError, match="exceed"):
            journal.open_lots()


def test_result_without_send_maps_directly() -> None:
    intent = _entry()
    base = intent_to_payload(intent)
    throttled = [_ev(1, JournalEventKind.INTENT, intent.intent_id, base),
                 _ev(2, JournalEventKind.RESULT, intent.intent_id, {"attempt": 1, "verdict": "throttled"})]
    assert derive_intent_state(throttled).status is IntentStatus.NOT_SENT
    ambiguous = [_ev(1, JournalEventKind.INTENT, intent.intent_id, base),
                 _ev(2, JournalEventKind.RESULT, intent.intent_id, {"attempt": 1, "verdict": "ambiguous"})]
    assert derive_intent_state(ambiguous).status is IntentStatus.UNKNOWN


def test_append_validates_args_and_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        OrderJournal(tmp_path, "mock", lock_timeout_seconds=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        OrderJournal(tmp_path, "mock", lock_timeout_seconds=float("inf"))
    journal = OrderJournal(tmp_path / "args", "mock", lock_timeout_seconds=5.0)
    with journal.exclusive():
        with pytest.raises(ValueError, match="run_id"):
            journal.append(JournalEventKind.SEND, "x", {"attempt": 1}, run_id="  ")
        with pytest.raises(ValueError, match="mapping"):
            journal.append(JournalEventKind.SEND, "x", ["nope"], run_id="r")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-empty strings"):
            journal.append(JournalEventKind.SEND, "x", {"": 1}, run_id="r")


def test_duplicate_refused_on_corrupt_journal(tmp_path: Path) -> None:
    journal = OrderJournal(tmp_path / "corr", "mock", lock_timeout_seconds=5.0)
    with journal.exclusive():
        intent = _entry()
        journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="r")
        with open(journal.journal_path, "a", encoding="utf-8") as handle:
            handle.write('{"seq": 99, "broken\n')
        with pytest.raises(ValueError, match="duplicate"):
            journal.append(JournalEventKind.INTENT, intent.intent_id, intent_to_payload(intent), run_id="r")
