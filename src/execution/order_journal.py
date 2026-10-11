"""Durable write-ahead order journal (idempotency authority)."""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.execution.order_contract import OrderIntent, OrderPhase, OrderSide, OrderType, SubmitVerdict
from src.utils.file_lock import exclusive_file_lock, sidecar_lock_path

__all__ = [
    "HALT_FILENAME",
    "JOURNAL_FILENAME",
    "IntentState",
    "IntentStatus",
    "JournalCorruptError",
    "JournalEvent",
    "JournalEventKind",
    "OpenLot",
    "OrderJournal",
    "derive_intent_state",
]

JOURNAL_FILENAME: str = "journal.jsonl"
HALT_FILENAME: str = "HALT"

_KST = timezone(timedelta(hours=9))
_VALID_ENVS: frozenset[str] = frozenset({"mock", "live"})


class JournalCorruptError(RuntimeError):
    """The journal contains a line that cannot be decoded or violates the event schema."""


class JournalEventKind(StrEnum):
    INTENT = "intent"
    SKIP = "skip"
    SEND = "send"
    RESULT = "result"
    ADOPT = "adopt"
    NOT_FOUND = "not_found"
    RESOLVE_CONFLICT = "resolve_conflict"
    BROKER_STATE = "broker_state"
    CANCEL_SEND = "cancel_send"
    CANCEL_RESULT = "cancel_result"


class IntentStatus(StrEnum):
    RECORDED = "recorded"
    SKIPPED = "skipped"
    NOT_SENT = "not_sent"
    UNKNOWN = "unknown"
    NOT_FOUND = "not_found"
    REJECTED = "rejected"
    ACKED = "acked"
    FILLED = "filled"
    PARTIAL = "partial"
    UNFILLED = "unfilled"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class JournalEvent:
    """One immutable journal line."""

    seq: int
    ts: str
    kind: JournalEventKind
    intent_id: str
    run_id: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class IntentState:
    """State of one intent derived from its events."""

    intent: OrderIntent | None
    status: IntentStatus
    n_sends: int
    broker_order_no: str | None
    broker_org_no: str | None
    filled_qty: int
    filled_amount_krw: int
    remaining_qty: int
    final: bool
    reason: str


@dataclass(frozen=True)
class OpenLot:
    """Shares bought by one entry intent and not yet sold by its exit intents."""

    entry_intent_id: str
    symbol: str
    trading_date: str
    qty: int


def _now_kst_iso() -> str:
    return datetime.now(_KST).isoformat(timespec="milliseconds")


def _is_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return False


def intent_to_payload(intent: OrderIntent) -> dict[str, Any]:
    return {
        "intent_id": intent.intent_id,
        "trading_date": intent.trading_date,
        "phase": intent.phase.value,
        "symbol": intent.symbol,
        "side": intent.side.value,
        "order_type": intent.order_type.value,
        "qty": intent.qty,
        "limit_price": intent.limit_price,
        "decision_price": intent.decision_price,
        "parent_intent_id": intent.parent_intent_id,
    }


def intent_from_payload(intent_id: str, payload: Mapping[str, Any]) -> OrderIntent:
    try:
        return OrderIntent(
            intent_id=intent_id,
            trading_date=str(payload["trading_date"]),
            phase=OrderPhase(str(payload["phase"])),
            symbol=str(payload["symbol"]),
            side=OrderSide(str(payload["side"])),
            order_type=OrderType(str(payload["order_type"])),
            qty=int(payload["qty"]),
            limit_price=payload["limit_price"],
            decision_price=payload["decision_price"],
            parent_intent_id=payload["parent_intent_id"],
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise JournalCorruptError(f"intent {intent_id!r}: bad INTENT payload ({exc})") from None


def _payload_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if type(value) is not int or isinstance(value, bool):
        raise JournalCorruptError(f"BROKER_STATE: {key} must be int")
    return value


def derive_intent_state(events: Sequence[JournalEvent]) -> IntentState:
    """Fold one intent's events into its state."""
    if len(events) == 0:
        raise JournalCorruptError("empty event sequence")
    ordered = sorted(events, key=lambda e: e.seq)
    for prev, cur in itertools.pairwise(ordered):
        if cur.seq <= prev.seq:
            raise JournalCorruptError("non-increasing seq")
    intent_id = ordered[0].intent_id
    kinds = [e.kind for e in ordered]
    if kinds.count(JournalEventKind.INTENT) > 1:
        raise JournalCorruptError(f"intent {intent_id!r}: duplicate INTENT")
    intent: OrderIntent | None = None
    for ev in ordered:
        if ev.kind is JournalEventKind.INTENT:
            intent = intent_from_payload(ev.intent_id, ev.payload)
    reason = ""
    for ev in ordered:
        if ev.kind is JournalEventKind.SKIP:
            raw = ev.payload.get("reason", "")
            reason = str(raw) if isinstance(raw, str) else ""
    n_sends = sum(1 for e in ordered if e.kind is JournalEventKind.SEND)
    broker_order_no: str | None = None
    broker_org_no: str | None = None
    for ev in ordered:
        if ev.kind is JournalEventKind.RESULT:
            try:
                verdict = SubmitVerdict(str(ev.payload.get("verdict", "")))
            except ValueError:
                raise JournalCorruptError(f"intent {intent_id!r}: bad RESULT verdict") from None
            if verdict is SubmitVerdict.ACCEPTED:
                raw_no = ev.payload.get("broker_order_no")
                raw_org = ev.payload.get("broker_org_no")
                broker_order_no = str(raw_no) if raw_no is not None else None
                broker_org_no = str(raw_org) if raw_org is not None else None
        elif ev.kind is JournalEventKind.ADOPT:
            raw_no = ev.payload.get("broker_order_no")
            raw_org = ev.payload.get("broker_org_no")
            broker_order_no = str(raw_no) if raw_no is not None else None
            broker_org_no = str(raw_org) if raw_org is not None else None
    states_payloads = [e.payload for e in ordered if e.kind is JournalEventKind.BROKER_STATE]
    filled_qty = 0
    filled_amount = 0
    remaining_qty = 0
    if states_payloads:
        prev_filled: int | None = None
        for payload in states_payloads:
            fq = _payload_int(payload, "filled_qty")
            fa = _payload_int(payload, "filled_amount_krw")
            rq = _payload_int(payload, "remaining_qty")
            _payload_int(payload, "order_qty")
            if intent is not None and fq > intent.qty:
                raise JournalCorruptError(f"intent {intent_id!r}: overfill")
            if intent is not None and fq + rq > intent.qty:
                raise JournalCorruptError(f"intent {intent_id!r}: fill+remaining exceeds qty")
            if fq < 0 or rq < 0 or fa < 0:
                raise JournalCorruptError(f"intent {intent_id!r}: negative fill field")
            if prev_filled is not None and fq < prev_filled:
                raise JournalCorruptError(f"intent {intent_id!r}: fill decreased")
            prev_filled = fq
        last = states_payloads[-1]
        filled_qty = int(last["filled_qty"])
        filled_amount = int(last["filled_amount_krw"])
        remaining_qty = int(last["remaining_qty"])
    finals = [e for e in ordered if e.kind is JournalEventKind.BROKER_STATE and bool(e.payload.get("final"))]
    if finals:
        last_final = finals[-1].payload
        fq = int(last_final["filled_qty"])
        cancelled = bool(last_final.get("cancelled"))
        ref_qty = intent.qty if intent is not None else int(last_final["order_qty"])
        if fq == ref_qty and ref_qty > 0:
            status = IntentStatus.FILLED
        elif 0 < fq < ref_qty:
            status = IntentStatus.PARTIAL
        elif fq == 0 and cancelled:
            status = IntentStatus.CANCELLED
        else:
            status = IntentStatus.UNFILLED
        return IntentState(intent, status, n_sends, broker_order_no, broker_org_no,
                           filled_qty, filled_amount, remaining_qty, True, reason)
    has_accepted = False
    has_adopt = any(e.kind is JournalEventKind.ADOPT for e in ordered)
    for e in ordered:
        if e.kind is JournalEventKind.RESULT and SubmitVerdict(str(e.payload.get("verdict", ""))) is SubmitVerdict.ACCEPTED:
            has_accepted = True
    if has_accepted or has_adopt:
        return IntentState(intent, IntentStatus.ACKED, n_sends, broker_order_no, broker_org_no,
                           filled_qty, filled_amount, remaining_qty, False, reason)
    for e in ordered:
        if e.kind is JournalEventKind.RESULT and SubmitVerdict(str(e.payload.get("verdict", ""))) is SubmitVerdict.REJECTED:
            return IntentState(intent, IntentStatus.REJECTED, n_sends, broker_order_no, broker_org_no,
                               filled_qty, filled_amount, remaining_qty, True, reason)
    sends = [e for e in ordered if e.kind is JournalEventKind.SEND]
    results = [e for e in ordered if e.kind is JournalEventKind.RESULT]
    not_founds = [e for e in ordered if e.kind is JournalEventKind.NOT_FOUND]
    conflicts = [e for e in ordered if e.kind is JournalEventKind.RESOLVE_CONFLICT]
    if sends and not_founds:
        last_send_seq = max(e.seq for e in sends)
        if max(e.seq for e in not_founds) > last_send_seq:
            return IntentState(intent, IntentStatus.NOT_FOUND, n_sends, broker_order_no, broker_org_no,
                               filled_qty, filled_amount, remaining_qty, False, reason)
    if sends:
        last_send_seq = max(e.seq for e in sends)
        later_results = [e for e in results if e.seq > last_send_seq]
        later_conflicts = [e for e in conflicts if e.seq > last_send_seq]
        if not later_results or later_conflicts:
            return IntentState(intent, IntentStatus.UNKNOWN, n_sends, broker_order_no, broker_org_no,
                               filled_qty, filled_amount, remaining_qty, False, reason)
        last_verdict = SubmitVerdict(str(later_results[-1].payload.get("verdict", "")))
        if last_verdict is SubmitVerdict.AMBIGUOUS:
            return IntentState(intent, IntentStatus.UNKNOWN, n_sends, broker_order_no, broker_org_no,
                               filled_qty, filled_amount, remaining_qty, False, reason)
        if last_verdict in (SubmitVerdict.THROTTLED, SubmitVerdict.AUTH_REJECTED, SubmitVerdict.NOT_SENT):
            return IntentState(intent, IntentStatus.NOT_SENT, n_sends, broker_order_no, broker_org_no,
                               filled_qty, filled_amount, remaining_qty, False, reason)
    if results:
        last_verdict = SubmitVerdict(str(results[-1].payload.get("verdict", "")))
        if last_verdict in (SubmitVerdict.THROTTLED, SubmitVerdict.AUTH_REJECTED, SubmitVerdict.NOT_SENT):
            return IntentState(intent, IntentStatus.NOT_SENT, n_sends, broker_order_no, broker_org_no,
                               filled_qty, filled_amount, remaining_qty, False, reason)
        if last_verdict is SubmitVerdict.AMBIGUOUS:
            return IntentState(intent, IntentStatus.UNKNOWN, n_sends, broker_order_no, broker_org_no,
                               filled_qty, filled_amount, remaining_qty, False, reason)
    if any(e.kind is JournalEventKind.SKIP for e in ordered):
        return IntentState(intent, IntentStatus.SKIPPED, n_sends, broker_order_no, broker_org_no,
                           filled_qty, filled_amount, remaining_qty, True, reason)
    return IntentState(intent, IntentStatus.RECORDED, n_sends, broker_order_no, broker_org_no,
                       filled_qty, filled_amount, remaining_qty, False, reason)


class OrderJournal:
    """Append-only journal of order events with an exclusive host lock."""

    def __init__(self, root: Path, env: str, *, lock_timeout_seconds: float) -> None:
        """Bind to <root>/<env>/journal.jsonl; env must be "mock" or "live"."""
        if env not in _VALID_ENVS:
            raise ValueError(f"env must be one of {sorted(_VALID_ENVS)}, got {env!r}")
        if not isinstance(lock_timeout_seconds, (int, float)) or not math.isfinite(float(lock_timeout_seconds)):
            raise ValueError(f"lock_timeout_seconds must be finite positive, got {lock_timeout_seconds!r}")
        if float(lock_timeout_seconds) <= 0:
            raise ValueError(f"lock_timeout_seconds must be positive, got {lock_timeout_seconds!r}")
        self._root = Path(root)
        self._env = env
        self._timeout = float(lock_timeout_seconds)
        self._dir = self._root / env
        self._path = self._dir / JOURNAL_FILENAME
        self._dir.mkdir(parents=True, exist_ok=True)
        self._locked = False

    @property
    def locked(self) -> bool:
        """True while this instance holds the exclusive lock."""
        return self._locked

    @property
    def journal_path(self) -> Path:
        return self._path

    def exclusive(self) -> AbstractContextManager[None]:
        """Hold the host-wide journal lock; raises TimeoutError when not acquired in time."""
        @contextlib.contextmanager
        def _ctx() -> Iterator[None]:
            if self._locked:
                raise RuntimeError("order-journal lock is not re-entrant")
            with exclusive_file_lock(
                sidecar_lock_path(self._path), timeout_seconds=self._timeout, purpose="order-journal"
            ):
                self._locked = True
                try:
                    yield
                finally:
                    self._locked = False
        return _ctx()

    def append(
        self, kind: JournalEventKind, intent_id: str, payload: Mapping[str, Any], *, run_id: str
    ) -> JournalEvent:
        """Durably append one event and return it; raises RuntimeError when the lock is not held."""
        if not self._locked:
            raise RuntimeError("OrderJournal.append requires exclusive()")
        if not isinstance(kind, JournalEventKind):
            raise ValueError(f"kind must be JournalEventKind, got {kind!r}")
        if not isinstance(intent_id, str) or not intent_id.strip():
            raise ValueError("intent_id must be non-blank")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be non-blank")
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping")
        clean: dict[str, Any] = dict(payload)
        for key, value in clean.items():
            if not isinstance(key, str) or not key:
                raise ValueError("payload keys must be non-empty strings")
            if not _is_scalar(value):
                raise ValueError(f"payload[{key!r}] must be a JSON scalar")
        if kind is JournalEventKind.SKIP:
            reason = clean.get("reason", "")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("SKIP requires a non-blank reason")
        if kind is JournalEventKind.INTENT:
            existing = self._intent_ids_with_intent()
            if intent_id in existing:
                raise ValueError(f"duplicate INTENT for {intent_id!r}")
        events = self._load_unchecked()
        seq = (events[-1].seq + 1) if events else 1
        event = JournalEvent(seq=seq, ts=_now_kst_iso(), kind=kind,
                             intent_id=intent_id, run_id=run_id, payload=clean)
        line = json.dumps(
            {"seq": event.seq, "ts": event.ts, "kind": event.kind.value,
             "intent_id": event.intent_id, "run_id": event.run_id, "payload": clean},
            ensure_ascii=False,
        )
        created = not self._path.exists()
        fd = os.open(str(self._path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if created:
            # A new file's directory entry is not durable until the directory itself is synced.
            dir_fd = os.open(str(self._dir), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        return event

    def load(self) -> list[JournalEvent]:
        """Return all events in file order; raises JournalCorruptError."""
        return self._load_unchecked()

    def states(self) -> dict[str, IntentState]:
        """Return intent_id -> derived state for every intent in the journal."""
        events = self.load()
        grouped: dict[str, list[JournalEvent]] = {}
        for ev in events:
            grouped.setdefault(ev.intent_id, []).append(ev)
        return {key: derive_intent_state(sorted(val, key=lambda e: e.seq)) for key, val in grouped.items()}

    def open_lots(self) -> list[OpenLot]:
        """Return entry lots with a positive unsold quantity."""
        states = self.states()
        exits_by_parent: dict[str, int] = {}
        for state in states.values():
            if state.intent is not None and state.intent.phase is OrderPhase.EXIT:
                parent = str(state.intent.parent_intent_id)
                exits_by_parent[parent] = exits_by_parent.get(parent, 0) + state.filled_qty
        lots: list[OpenLot] = []
        for intent_id, state in states.items():
            if state.intent is None or state.intent.phase is not OrderPhase.ENTRY:
                continue
            qty = state.filled_qty - exits_by_parent.get(intent_id, 0)
            if qty < 0:
                raise JournalCorruptError(f"entry {intent_id!r}: exits exceed entry fills")
            if qty > 0:
                lots.append(OpenLot(entry_intent_id=intent_id, symbol=state.intent.symbol,
                                    trading_date=state.intent.trading_date, qty=qty))
        lots.sort(key=lambda lot: (lot.trading_date, lot.symbol))
        return lots

    def is_halted(self) -> bool:
        """True when the HALT file exists in the journal directory."""
        return (self._dir / HALT_FILENAME).exists()

    def _intent_ids_with_intent(self) -> set[str]:
        try:
            events = self._load_unchecked()
        except JournalCorruptError:
            found: set[str] = set()
            with open(self._path, encoding="utf-8") as handle:
                for raw in handle:
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and obj.get("kind") == JournalEventKind.INTENT.value:
                        intent_id = obj.get("intent_id")
                        if isinstance(intent_id, str):
                            found.add(intent_id)
            return found
        return {e.intent_id for e in events if e.kind is JournalEventKind.INTENT}

    def _load_unchecked(self) -> list[JournalEvent]:
        if not self._path.exists():
            return []
        events: list[JournalEvent] = []
        seen: set[str] = set()
        seen_intent: set[str] = set()
        prev_seq: int | None = None
        with open(self._path, encoding="utf-8") as handle:
            for lineno, raw in enumerate(handle, start=1):
                text = raw.rstrip("\n")
                if not text.strip():
                    raise JournalCorruptError(f"line {lineno}: empty line")
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise JournalCorruptError(f"line {lineno}: undecodable JSON ({exc})") from None
                if not isinstance(obj, dict):
                    raise JournalCorruptError(f"line {lineno}: event must be an object")
                try:
                    seq = obj["seq"]
                    ts = obj["ts"]
                    kind_raw = obj["kind"]
                    intent_id = obj["intent_id"]
                    run_id = obj["run_id"]
                    payload = obj["payload"]
                except KeyError as exc:
                    raise JournalCorruptError(f"line {lineno}: missing field {exc}") from None
                if type(seq) is not int or isinstance(seq, bool) or seq <= 0:
                    raise JournalCorruptError(f"line {lineno}: bad seq")
                if prev_seq is not None and seq <= prev_seq:
                    raise JournalCorruptError(f"line {lineno}: non-increasing seq")
                prev_seq = seq
                if not isinstance(ts, str) or not ts.strip():
                    raise JournalCorruptError(f"line {lineno}: bad ts")
                if not isinstance(intent_id, str) or not intent_id:
                    raise JournalCorruptError(f"line {lineno}: bad intent_id")
                if not isinstance(run_id, str) or not run_id:
                    raise JournalCorruptError(f"line {lineno}: bad run_id")
                if not isinstance(payload, dict):
                    raise JournalCorruptError(f"line {lineno}: bad payload")
                try:
                    kind = JournalEventKind(str(kind_raw))
                except ValueError:
                    raise JournalCorruptError(f"line {lineno}: unknown kind {kind_raw!r}") from None
                for value in payload.values():
                    if not _is_scalar(value):
                        raise JournalCorruptError(f"line {lineno}: non-scalar payload value")
                if kind is JournalEventKind.INTENT:
                    if intent_id in seen_intent:
                        raise JournalCorruptError(f"line {lineno}: duplicate INTENT for {intent_id!r}")
                    seen_intent.add(intent_id)
                    seen.add(intent_id)
                elif kind is JournalEventKind.SKIP:
                    seen.add(intent_id)
                else:
                    if intent_id not in seen:
                        raise JournalCorruptError(f"line {lineno}: event without preceding INTENT/SKIP")
                events.append(JournalEvent(seq=seq, ts=ts, kind=kind,
                                           intent_id=intent_id, run_id=run_id, payload=dict(payload)))
        return events
