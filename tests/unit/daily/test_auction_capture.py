"""Auction capture invariant guards."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from src.execution.paper_broker import HeldRoster


@pytest.fixture(autouse=True)
def _hermetic_held_roster(monkeypatch: pytest.MonkeyPatch) -> None:
    """Close-phase capture reads the held roster; never touch the real paper ledger."""
    from src.daily import auction_capture

    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))


def _seoul(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> dt.datetime:
    from src.data.capture_contracts import SEOUL

    return dt.datetime(year, month, day, hour, minute, second, tzinfo=SEOUL)


def _clock(snapshot_date: str, open_hm=(9, 0), close_hm=(15, 30)):
    from src.data.capture_contracts import SessionClock

    day = dt.date.fromisoformat(snapshot_date)
    return SessionClock(
        trading_date=day,
        open_at=_seoul(day.year, day.month, day.day, *open_hm),
        close_at=_seoul(day.year, day.month, day.day, *close_hm),
        close_confirmation_deadline=_seoul(day.year, day.month, day.day, close_hm[0], close_hm[1], 0) + dt.timedelta(minutes=3),
        provenance="test",
    )


def _profile(tmp_path: Path, **overrides: Any):
    from src.config.collection import CollectionSettings

    base: dict[str, Any] = {
        "COLLECTION_ROOT": tmp_path / "capture",
        "COLLECTION_RESEARCH_SLOTS": ("5",),
        "COLLECTION_AUCTION_INTERVAL_SECONDS": 60,
        "COLLECTION_CONCURRENCY_PER_KEY": 8,
        "COLLECTION_OPEN_CONFIRM_SECONDS": 60,
    }
    base.update(overrides)
    return CollectionSettings(**base)


def _store(tmp_path: Path):
    from src.data.capture_store import CaptureStore

    return CaptureStore(tmp_path / "capture")


def _publish_cohort(store: Any, snapshot_date: str, eligible: list[str], *, run_id: str = "decision-1") -> Any:
    from src.data.capture_contracts import (
        SEOUL,
        CaptureContext,
        CaptureDataset,
        CaptureManifest,
        CaptureStatus,
        Cohort,
    )

    day = dt.date.fromisoformat(snapshot_date)
    cohort = Cohort(
        trading_date=day,
        cohort_id=f"c-{snapshot_date}",
        eligible_symbols=tuple(eligible),
        scanned_symbols=tuple(eligible),
        eligibility_rule_version="v1",
        rejections={},
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=day,
            run_id=run_id,
            dataset=CaptureDataset.SCAN,
            vendor="owner-local",
            endpoint="decision-input",
            symbol=None,
            venue="KRX",
            session="regular",
            capture_reason="test",
            cohort_id=cohort.cohort_id,
            scheduled_at=None,
        ),
        cohort=cohort,
        completed_at=dt.datetime(2026, 9, 1, 9, 0, tzinfo=SEOUL),
        entries=(),
        artifacts=(),
        status=CaptureStatus.COMPLETE,
    )
    store.publish_manifest(manifest)
    return cohort


class _Session:
    async def __aenter__(self) -> Any:
        return object()

    async def __aexit__(self, *args: Any) -> bool:
        return False


class _FakeClient:
    def __init__(
        self,
        orderbook: dict[str, Any] | None = None,
        price_seq: dict[str, list[int]] | None = None,
        program: dict[str, Any] | None = None,
        calls: list[tuple[str, str]] | None = None,
        advance: Any = None,
    ) -> None:
        self._orderbook = orderbook or {"rt_cd": "0", "output1": {"a": "1"}, "output2": [{"b": "2"}]}
        self._price_seq = {k: list(v) for k, v in (price_seq or {}).items()}
        self._program = program or {"rt_cd": "0", "output1": {}, "output2": []}
        self.calls: list[tuple[str, str]] = calls if calls is not None else []
        self._advance = advance

    def create_session(self) -> _Session:
        return _Session()

    async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
        assert market_div_code == "J"
        self.calls.append(("orderbook", code))
        if self._advance is not None:
            self._advance()
        return dict(self._orderbook)

    async def get_program_net_buy(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
        assert market_div_code == "J"
        self.calls.append(("program", code))
        return dict(self._program)

    async def get_current_price(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
        self.calls.append(("price", code))
        seq = self._price_seq.get(code, [100])
        price = seq.pop(0) if len(seq) > 1 else seq[0]
        self._price_seq[code] = seq if seq else [price]
        return {"rt_cd": "0", "output": {"stck_oprc": str(price)}}


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_closing_baseline_includes_non_admitted(tmp_path) -> None:
    """Every eligible symbol has an expected entry each round."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001", "000002"])
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17")
    now = _seoul(2026, 9, 17, 15, 0)
    client = _FakeClient(calls=[])
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: now,
        )
    )
    rounds = auction_capture.close_rounds(clock, 60)
    orderbook_entries = [e for e in manifest.entries if e.dataset == CaptureDataset.ORDERBOOK]
    assert len(orderbook_entries) == len(rounds) * 2
    assert {e.symbol for e in orderbook_entries} == {"000001", "000002"}


def test_close_and_program_requests_follow_scheduled_rounds(tmp_path) -> None:
    """Close and attribution calls occur no earlier than their declared timestamps."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 15, 20)}
    observed: list[tuple[str, dt.datetime]] = []

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    class _TimedClient(_FakeClient):
        async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None):
            observed.append(("orderbook", state["now"]))
            return await super().get_orderbook_snapshot(session, code, market_div_code)

        async def get_program_net_buy(self, session: Any, code: str, market_div_code: str | None = None):
            observed.append(("program", state["now"]))
            return await super().get_program_net_buy(session, code, market_div_code)

    client = _TimedClient(calls=[])
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: state["now"],
            sleep_fn=_sleep,
        )
    )
    expected = {
        ("orderbook", round_at)
        for round_at in auction_capture.close_rounds(clock, 60)
    } | {
        ("program", round_at)
        for round_at in auction_capture.program_rounds(clock)
    }
    for kind, scheduled_at in expected:
        assert any(observed_kind == kind and observed_at >= scheduled_at for observed_kind, observed_at in observed)
    for entry in manifest.entries:
        if entry.dataset in (CaptureDataset.ORDERBOOK, CaptureDataset.PROGRAM) and entry.first_event_time is not None:
            assert entry.scheduled_at is not None
            assert entry.first_event_time >= entry.scheduled_at


def test_opening_follows_previous_cohort(tmp_path, monkeypatch) -> None:
    """Prior candidate absent from today stays in opening roster."""
    from src.daily import auction_capture

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001", "000002"])
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=600)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    def _advance() -> None:
        state["now"] += dt.timedelta(seconds=20)

    client = _FakeClient(price_seq={"000001": [100], "000002": [200]}, calls=[], advance=_advance)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="open",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: state["now"],
            sleep_fn=_sleep,
        )
    )
    assert "000002" in {e.symbol for e in manifest.entries}


def test_absent_prior_cohort_keeps_known_position(tmp_path, monkeypatch) -> None:
    """Missing previous cohort still observes positions with PARTIAL."""
    import pandas as pd

    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    store = _store(tmp_path)
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster(("000003",), True, None))
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17")
    now = _seoul(2026, 9, 17, 9, 5)
    client = _FakeClient(price_seq={"000003": [50]}, calls=[])
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="open",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: now,
            sleep_fn=lambda s: asyncio.sleep(0),
        )
    )
    assert "000003" in {e.symbol for e in manifest.entries}
    assert manifest.status == CaptureStatus.PARTIAL
    _ = pd.DataFrame([{"a": 1}])


def test_exceptional_hours_shift_schedule(tmp_path) -> None:
    """All instants follow the provided session clock."""
    from src.daily import auction_capture

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17", open_hm=(10, 0), close_hm=(14, 0))
    now = _seoul(2026, 9, 17, 13, 0)
    client = _FakeClient(calls=[])
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: now,
        )
    )
    rounds = auction_capture.close_rounds(clock, 60)
    assert rounds[0] == clock.close_at - dt.timedelta(minutes=9)
    assert all(r < clock.close_at for r in rounds)
    assert manifest.entries[0].scheduled_at == rounds[0]


def test_expired_rounds_do_not_burst(tmp_path) -> None:
    """Slow responses mark missing entries without backlog burst."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001", "000002"])
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 15, 20)}

    def advance() -> None:
        state["now"] += dt.timedelta(seconds=120)

    client = _FakeClient(calls=[], advance=advance)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: state["now"],
        )
    )
    orderbook_calls = [c for c in client.calls if c[0] == "orderbook"]
    assert len(orderbook_calls) < 2 * len(auction_capture.close_rounds(clock, 60))
    partial_deadlines = [e for e in manifest.entries if e.status == CaptureStatus.PARTIAL and e.reason == "deadline"]
    assert partial_deadlines


def test_program_observations_are_attribution_only(tmp_path) -> None:
    """Program evidence arrives after decision with its own schedule."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17")
    now = _seoul(2026, 9, 17, 15, 0)
    client = _FakeClient(calls=[])
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: now,
        )
    )
    program_entries = [e for e in manifest.entries if e.dataset == CaptureDataset.PROGRAM]
    assert len(program_entries) == 2
    expected = {clock.close_at - dt.timedelta(minutes=8), clock.close_at - dt.timedelta(minutes=4)}
    assert {e.scheduled_at for e in program_entries} == expected


def test_other_project_outage_has_no_effect(tmp_path, monkeypatch) -> None:
    """First-party acquisition is independent of krx-alpha."""
    import sys

    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    monkeypatch.delitem(sys.modules, "krx_alpha", raising=False)
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17")
    now = _seoul(2026, 9, 17, 15, 0)
    client = _FakeClient(calls=[])
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: now,
        )
    )
    assert manifest.status == CaptureStatus.COMPLETE


def test_delayed_open_resolves_within_deadline(tmp_path, monkeypatch) -> None:
    """Absent open then appearing keeps actual observation times."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001"])
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=600)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    def _advance() -> None:
        state["now"] += dt.timedelta(seconds=20)

    class _SeqClient(_FakeClient):
        def __init__(self) -> None:
            super().__init__(calls=[], advance=_advance)
            self.n = 0

        async def get_current_price(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
            self.calls.append(("price", code))
            self.n += 1
            price = "0" if self.n == 1 else "72000"
            return {"rt_cd": "0", "output": {"stck_oprc": price}}

    client = _SeqClient()
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="open",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: state["now"],
            sleep_fn=_sleep,
        )
    )
    price_entries = [e for e in manifest.entries if e.dataset == CaptureDataset.PRICE]
    assert any(e.status == CaptureStatus.COMPLETE and (e.rows or 0) > 0 for e in price_entries)
    assert client.n >= 2


def test_late_open_symbol_does_not_starve_later_symbols(tmp_path, monkeypatch) -> None:
    """A never-forming open on one symbol must not block polling of the rest of the roster."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001", "000002", "000003"])
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=180)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    class _StuckFirstClient(_FakeClient):
        async def get_current_price(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
            self.calls.append(("price", code))
            state["now"] += dt.timedelta(seconds=0.1)
            price = "0" if code == "000001" else "72000"
            return {"rt_cd": "0", "output": {"stck_oprc": price}}

    client = _StuckFirstClient(calls=[])
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17", phase="open", profile=profile, store=store, clients=[client],
            session_clock=clock, now_fn=lambda: state["now"], sleep_fn=_sleep,
        )
    )
    by_symbol = {e.symbol: e for e in manifest.entries if e.dataset == CaptureDataset.PRICE}
    assert by_symbol["000001"].status == CaptureStatus.NO_TRADES and by_symbol["000001"].reason == "no_open_trade"
    assert by_symbol["000002"].status == CaptureStatus.COMPLETE and by_symbol["000003"].status == CaptureStatus.COMPLETE
    assert manifest.status == CaptureStatus.COMPLETE


def test_valid_reply_without_open_price_is_no_trades(tmp_path, monkeypatch) -> None:
    """A symbol the vendor keeps answering with no open price is a terminal NO_TRADES, not a collection gap."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001"])
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=31)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    def _advance() -> None:
        state["now"] += dt.timedelta(seconds=20)

    class _ZeroClient(_FakeClient):
        async def get_current_price(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
            self.calls.append(("price", code))
            state["now"] += dt.timedelta(seconds=40)
            return {"rt_cd": "0", "output": {"stck_oprc": "0"}}

    client = _ZeroClient(calls=[], advance=_advance)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="open",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: state["now"],
            sleep_fn=_sleep,
        )
    )
    price_entries = [e for e in manifest.entries if e.dataset == CaptureDataset.PRICE]
    assert price_entries and all(e.status == CaptureStatus.NO_TRADES for e in price_entries)
    assert all(e.reason == "no_open_trade" for e in price_entries)
    assert manifest.status == CaptureStatus.COMPLETE


def test_error_reply_without_open_price_stays_partial(tmp_path, monkeypatch) -> None:
    """A vendor error reply is not evidence that nothing traded: the symbol stays an explicit unresolved gap."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001"])
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=31)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    def _advance() -> None:
        state["now"] += dt.timedelta(seconds=20)

    class _ErrorClient(_FakeClient):
        async def get_current_price(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
            self.calls.append(("price", code))
            state["now"] += dt.timedelta(seconds=40)
            return {"rt_cd": "1", "output": {"stck_oprc": "0"}}

    client = _ErrorClient(calls=[], advance=_advance)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="open",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: state["now"],
            sleep_fn=_sleep,
        )
    )
    price_entries = [e for e in manifest.entries if e.dataset == CaptureDataset.PRICE]
    assert price_entries and all(e.status == CaptureStatus.PARTIAL for e in price_entries)
    assert all(e.reason == "open_unresolved" for e in price_entries)


def test_validation_rejects_bad_inputs(tmp_path) -> None:
    """Wrong date/phase/clock/profile fail closed."""
    import pytest

    from src.daily import auction_capture

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17")
    now = _seoul(2026, 9, 17, 15, 0)
    client = _FakeClient(calls=[])
    with pytest.raises(ValueError, match="snapshot_date"):
        _run(auction_capture.run_auction_capture("not-a-date", phase="close", profile=profile, store=store, clients=[client], session_clock=clock, now_fn=lambda: now))
    with pytest.raises(ValueError, match="phase"):
        _run(auction_capture.run_auction_capture("2026-09-17", phase="dawn", profile=profile, store=store, clients=[client], session_clock=clock, now_fn=lambda: now))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="trading_date"):
        _run(auction_capture.run_auction_capture("2026-09-17", phase="close", profile=profile, store=store, clients=[client], session_clock=_clock("2026-09-16"), now_fn=lambda: now))
    bare = _profile(tmp_path).model_copy(update={"COLLECTION_RESEARCH_SLOTS": ()})
    with pytest.raises(ValueError, match="research slots"):
        _run(auction_capture.run_auction_capture("2026-09-17", phase="close", profile=bare, store=store, clients=[client], session_clock=clock, now_fn=lambda: now))
    with pytest.raises(ValueError, match="prewarmed"):
        _run(auction_capture.run_auction_capture("2026-09-17", phase="close", profile=profile, store=store, clients=[], session_clock=clock, now_fn=lambda: now))
    with pytest.raises(RuntimeError, match="cohort"):
        _run(auction_capture.run_auction_capture("2026-09-18", phase="close", profile=profile, store=store, clients=[client], session_clock=_clock("2026-09-18"), now_fn=lambda: now))
    with pytest.raises(ValueError, match="snapshot_date"):
        auction_capture._previous_trading_day("bad-date")
    assert auction_capture._previous_trading_day("2026-09-21") == "2026-09-18"


def test_failures_prevent_complete(tmp_path, monkeypatch) -> None:
    """Transport and persistence faults never certify COMPLETE."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17")
    now = _seoul(2026, 9, 17, 15, 0)

    class _BoomClient(_FakeClient):
        async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
            raise RuntimeError("transport down")

        async def get_program_net_buy(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
            raise RuntimeError("transport down")

    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17", phase="close", profile=profile, store=store, clients=[_BoomClient(calls=[])], session_clock=clock, now_fn=lambda: now
        )
    )
    assert manifest.status == CaptureStatus.PARTIAL

    monkeypatch.setattr(store, "append_response", lambda response: (_ for _ in ()).throw(OSError("disk full")))
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17", phase="close", profile=profile, store=store, clients=[_FakeClient(calls=[])], session_clock=clock, now_fn=lambda: now
        )
    )
    assert manifest.status == CaptureStatus.PARTIAL

    import pandas as pd

    monkeypatch.setattr(store, "append_response", store.__class__.append_response.__get__(store, store.__class__))

    def _boom_frame(*args: Any, **kwargs: Any) -> Any:
        raise OSError("frame full")

    monkeypatch.setattr(store, "publish_frame", _boom_frame)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17", phase="close", profile=profile, store=store, clients=[_FakeClient(calls=[])], session_clock=clock, now_fn=lambda: now
        )
    )
    assert manifest.status == CaptureStatus.PARTIAL
    assert pd.DataFrame([{"x": 1}]).shape == (1, 1)

    def _boom_manifest(*args: Any, **kwargs: Any) -> Any:
        raise OSError("manifest full")

    monkeypatch.setattr(store, "publish_frame", lambda *a, **k: None)
    monkeypatch.setattr(store, "publish_manifest", _boom_manifest)
    try:
        _run(
            auction_capture.run_auction_capture(
                "2026-09-17", phase="close", profile=profile, store=store, clients=[_FakeClient(calls=[])], session_clock=clock, now_fn=lambda: now
            )
        )
        raise AssertionError("expected OSError")
    except OSError:
        pass


def test_open_roster_merges_positions_with_previous(tmp_path, monkeypatch) -> None:
    """Previous cohort plus outstanding positions form the opening roster."""
    from src.daily import auction_capture

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001"])
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster(("000002",), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=600)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    def _advance() -> None:
        state["now"] += dt.timedelta(seconds=20)

    client = _FakeClient(price_seq={"000001": [100], "000002": [200]}, calls=[], advance=_advance)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="open",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: state["now"],
            sleep_fn=_sleep,
        )
    )
    assert {"000001", "000002"} <= {e.symbol for e in manifest.entries}


class _FlakyOrderbookClient(_FakeClient):
    def __init__(self, failures: list[dict[str, Any]]) -> None:
        super().__init__(calls=[])
        self._failures = list(failures)

    async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
        self.calls.append(("orderbook", code))
        if self._failures:
            return dict(self._failures.pop(0))
        return {"rt_cd": "0", "output1": {"a": "1"}, "output2": [{"b": "2"}]}


def _close_with_orderbook_failures(tmp_path, failures):
    from src.daily import auction_capture

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    client = _FlakyOrderbookClient(failures)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_profile(tmp_path),
            store=store,
            clients=[client],
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: _seoul(2026, 9, 17, 15, 0),
        )
    )
    return manifest, client


_TRANSIENT = {"rt_cd": "1", "msg_cd": "EGW00316", "msg1": "retry please"}


def test_transient_gateway_error_is_requeried_once_and_completes(tmp_path, caplog) -> None:
    """EGW00316 on the first call no longer costs the round's snapshot."""
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    caplog.set_level("WARNING")
    manifest, client = _close_with_orderbook_failures(tmp_path, [_TRANSIENT])
    book = [e for e in manifest.entries if e.dataset == CaptureDataset.ORDERBOOK]
    assert book and all(e.status == CaptureStatus.COMPLETE for e in book)
    assert "status=VENDOR_ERROR" in caplog.text and "msg_cd=EGW00316" in caplog.text
    first_round_calls = [c for c in client.calls if c == ("orderbook", "000001")]
    assert len(first_round_calls) == len(book) + 1


def test_persistent_gateway_error_stays_partial_with_cause_logged(tmp_path, caplog) -> None:
    """A second consecutive failure is bounded: PARTIAL and the vendor code is in the journal."""
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    caplog.set_level("WARNING")
    manifest, _ = _close_with_orderbook_failures(tmp_path, [_TRANSIENT, _TRANSIENT])
    book = [e for e in manifest.entries if e.dataset == CaptureDataset.ORDERBOOK]
    assert book[0].status == CaptureStatus.PARTIAL and book[0].reason == "vendor_failure"
    assert manifest.status == CaptureStatus.PARTIAL
    assert caplog.text.count("status=VENDOR_ERROR") == 2


def test_non_transient_vendor_error_is_not_retried(tmp_path) -> None:
    manifest, client = _close_with_orderbook_failures(tmp_path, [{"rt_cd": "1", "msg_cd": "EGW00123", "msg1": "bad"}])
    from src.data.capture_contracts import CaptureDataset

    book = [e for e in manifest.entries if e.dataset == CaptureDataset.ORDERBOOK]
    assert len([c for c in client.calls if c[0] == "orderbook"]) == len(book)


def test_program_vendor_failure_stays_partial(tmp_path) -> None:
    """Failed program attribution never certifies COMPLETE."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17")
    now = _seoul(2026, 9, 17, 15, 0)
    client = _FakeClient(program={"rt_cd": "9", "msg1": "busy"}, calls=[])
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: now,
        )
    )
    program_entries = [e for e in manifest.entries if e.dataset == CaptureDataset.PROGRAM]
    assert program_entries and all(e.status == CaptureStatus.FAILED for e in program_entries)
    assert manifest.status == CaptureStatus.PARTIAL


def test_open_persistence_faults_stay_partial(tmp_path, monkeypatch) -> None:
    """Open-phase raw/frame faults keep explicit PARTIAL."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001"])
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=600)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    def _advance() -> None:
        state["now"] += dt.timedelta(seconds=20)

    monkeypatch.setattr(store, "append_response", lambda response: (_ for _ in ()).throw(OSError("disk full")))
    client = _FakeClient(price_seq={"000001": [10]}, calls=[], advance=_advance)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17", phase="open", profile=profile, store=store, clients=[client],
            session_clock=clock, now_fn=lambda: state["now"], sleep_fn=_sleep,
        )
    )
    assert manifest.status == CaptureStatus.PARTIAL


def test_open_frame_fault_keeps_partial(tmp_path, monkeypatch) -> None:
    """Open-phase frame faults keep explicit PARTIAL."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001"])
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=600)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    def _advance() -> None:
        state["now"] += dt.timedelta(seconds=20)

    def _boom_frame(*args: Any, **kwargs: Any) -> Any:
        raise OSError("frame full")

    monkeypatch.setattr(store, "publish_frame", _boom_frame)
    client = _FakeClient(price_seq={"000001": [10]}, calls=[], advance=_advance)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17", phase="open", profile=profile, store=store, clients=[client],
            session_clock=clock, now_fn=lambda: state["now"], sleep_fn=_sleep,
        )
    )
    assert manifest.status == CaptureStatus.PARTIAL


def test_capture_root_and_positions_helpers(tmp_path, monkeypatch) -> None:
    """Root override and capture helpers behave."""
    from src import settings as app_settings
    from src.daily import auction_capture

    explicit = _profile(tmp_path, COLLECTION_ROOT=tmp_path / "explicit")
    assert auction_capture._capture_root(explicit) == tmp_path / "explicit"
    monkeypatch.setattr(app_settings, "HISTORY_DIR", tmp_path, raising=False)
    implicit = _profile(tmp_path, COLLECTION_ROOT=None)
    assert auction_capture._capture_root(implicit) == tmp_path / "capture"
    assert auction_capture._extract_open_price(None) == 0
    assert auction_capture._extract_open_price({"output": None}) == 0
    assert auction_capture._extract_open_price({"output": {"stck_oprc": "bad"}}) == 0
    assert auction_capture._extract_open_price({"output": {"stck_oprc": "71,000"}}) == 71000
    assert auction_capture._extract_open_price({"output1": None}) == 0
    frame = auction_capture._fragment_frame("000001", now := _seoul(2026, 9, 17, 9, 0), now, {"output1": {"x": 1}, "output2": []})
    assert list(frame["symbol"]) == ["000001"]
    null_frame = auction_capture._fragment_frame("000001", now, now, {"output1": None, "output2": None})
    assert null_frame["output1"].iloc[0] is None
    missing_frame = auction_capture._fragment_frame("000001", now, now, {"rt_cd": "0"})
    assert missing_frame["output1"].iloc[0] is None
    assert auction_capture._open_rounds(_clock("2026-09-17"))[0] < _clock("2026-09-17").open_at
    assert len(auction_capture.program_rounds(_clock("2026-09-17"))) == 2


def test_main_skip_and_success(tmp_path, monkeypatch, caplog) -> None:
    """CLI skips holidays and runs configured phases."""
    import logging

    from src.daily import auction_capture

    monkeypatch.setattr(auction_capture, "CollectionSettings", lambda: _profile(tmp_path))
    with caplog.at_level(logging.INFO, logger=auction_capture.logger.name):
        auction_capture.main(["--phase", "close", "--date", "2026-09-20"])
    assert any("SKIP" in r.message for r in caplog.records)

    async def _fake_run(*args: Any, **kwargs: Any) -> Any:
        from src.data.capture_contracts import CaptureStatus

        class _Manifest:
            status = CaptureStatus.COMPLETE
            entries: tuple[Any, ...] = ()

        return _Manifest()

    async def _fake_async(snapshot_date: str, phase: str, profile: Any, stats_sink: Any = None) -> Any:
        return await _fake_run()

    monkeypatch.setattr(auction_capture, "CollectionSettings", lambda: _profile(tmp_path))
    monkeypatch.setattr(auction_capture, "_run_async", _fake_async)
    with caplog.at_level(logging.INFO, logger=auction_capture.logger.name):
        auction_capture.main(["--phase", "open", "--date", "2026-09-17"])
    assert any("auction_capture" in r.message for r in caplog.records)
    try:
        auction_capture.main(["--phase", "close", "--date", "bad-date"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_main_logs_not_complete_detail_for_partial_manifest(tmp_path, monkeypatch, caplog) -> None:
    """A PARTIAL run names the symbol, round and reason in the journal without opening the manifest."""
    import logging

    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry

    entry = CoverageEntry(
        symbol="029460",
        dataset=CaptureDataset.ORDERBOOK,
        venue="KRX",
        session="regular",
        scheduled_at=_seoul(2026, 9, 17, 15, 25),
        status=CaptureStatus.PARTIAL,
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason="vendor_failure",
        raw_refs=(),
    )

    class _Manifest:
        status = CaptureStatus.PARTIAL
        entries = (entry,)

    async def _fake_async(snapshot_date: str, phase: str, profile: Any, stats_sink: Any = None) -> Any:
        return _Manifest()

    monkeypatch.setattr(auction_capture, "CollectionSettings", lambda: _profile(tmp_path))
    monkeypatch.setattr(auction_capture, "_run_async", _fake_async)
    with caplog.at_level(logging.INFO, logger=auction_capture.logger.name):
        auction_capture.main(["--phase", "close", "--date", "2026-09-17"])
    detail = [r.message for r in caplog.records if "NOT_COMPLETE_DETAIL" in r.message]
    assert detail and "ORDERBOOK:PARTIAL:vendor_failure" in detail[0] and "029460@15:25:00" in detail[0]


def test_module_execution_actually_invokes_main(monkeypatch) -> None:
    """Running the module as __main__ must reach main(), so a missing __main__ guard fails here."""
    import runpy
    import sys

    import pytest

    monkeypatch.setattr(sys, "argv", ["auction_capture", "--phase", "open", "--date", "bad-date"])
    with pytest.raises(ValueError, match="Invalid snapshot_date"):
        runpy.run_module("src.daily.auction_capture", run_name="__main__")


def test_run_async_builds_clients(tmp_path, monkeypatch) -> None:
    """Research clients use token cache and host pacing contracts."""
    import os

    from src.daily import auction_capture

    profile = _profile(tmp_path)
    seen: dict[str, Any] = {}

    class _Cred:
        slot = "DATA_5"
        app_key = "key5"
        app_secret = "sec5"
        hts_id = "hts5"

    monkeypatch.setattr(auction_capture, "resolve_research_credentials", lambda env, *, slots: (_Cred(),))
    created: list[Any] = []

    class _Client:
        def __init__(self, app_key: str, app_secret: str, account_id: str, hts_id: str, token_file: str | None = None) -> None:
            created.append((app_key, token_file))
            self._token_file = token_file

        def create_session(self) -> _Session:
            return _Session()

        async def ensure_token(self, session: Any) -> None:
            seen["warmed"] = True

    monkeypatch.setattr("src.api.kis.client.KisApiClient", _Client)
    monkeypatch.setattr(auction_capture, "KisApiClient", _Client, raising=False)
    monkeypatch.setenv("KIS_DATA_SLOTS", "5")

    async def _fake_capture(*args: Any, **kwargs: Any) -> str:
        return "manifest-ok"

    monkeypatch.setattr(auction_capture, "run_auction_capture", _fake_capture)

    async def _open(client: Any, session: Any, date: Any) -> bool:
        return True

    monkeypatch.setattr(auction_capture, "is_kis_trading_day", _open)
    out = asyncio.run(auction_capture._run_async("2026-09-17", "close", profile))
    assert out == "manifest-ok"
    assert seen["warmed"] is True
    assert created and "token_" in str(created[0][1])
    assert os.environ["KIS_DATA_SLOTS"] == "5"


def _stub_research_clients(monkeypatch: Any) -> None:
    from src.daily import auction_capture

    class _Cred:
        slot = "DATA_5"
        app_key = "key5"
        app_secret = "sec5"
        hts_id = "hts5"

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def create_session(self) -> _Session:
            return _Session()

        async def ensure_token(self, session: Any) -> None:
            return None

    monkeypatch.setattr(auction_capture, "resolve_research_credentials", lambda env, *, slots: (_Cred(),))
    monkeypatch.setattr("src.api.kis.client.KisApiClient", _Client)


def test_run_async_skips_weekday_market_holiday(tmp_path, monkeypatch) -> None:
    """Weekday holidays (e.g. Chuseok) return None without capturing."""
    from src.daily import auction_capture

    _stub_research_clients(monkeypatch)

    async def _closed(client: Any, session: Any, date: Any) -> bool:
        return False

    async def _must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("capture must not run on a holiday")

    monkeypatch.setattr(auction_capture, "is_kis_trading_day", _closed)
    monkeypatch.setattr(auction_capture, "run_auction_capture", _must_not_run)
    assert asyncio.run(auction_capture._run_async("2026-09-24", "close", _profile(tmp_path))) is None


def test_run_async_open_uses_actual_previous_trading_day(tmp_path, monkeypatch) -> None:
    """Opening after a holiday run uses the real previous trading day, not the previous weekday."""
    import pandas as pd

    from src.daily import auction_capture

    _stub_research_clients(monkeypatch)

    async def _open(client: Any, session: Any, date: Any) -> bool:
        return True

    async def _prev(client: Any, session: Any, decision_date: Any, **kwargs: Any) -> Any:
        return pd.Timestamp("2026-09-23")

    seen: dict[str, Any] = {}

    async def _capture(*args: Any, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(auction_capture, "is_kis_trading_day", _open)
    monkeypatch.setattr(auction_capture, "resolve_prev_trading_day_kis", _prev)
    monkeypatch.setattr(auction_capture, "run_auction_capture", _capture)
    assert asyncio.run(auction_capture._run_async("2026-09-28", "open", _profile(tmp_path))) == "ok"
    assert seen["previous_trading_day"] == "2026-09-23"


def test_resolve_roster_prefers_supplied_previous_trading_day(tmp_path, monkeypatch) -> None:
    """The weekday walk would pick 2026-09-25 (holiday); the supplied day wins."""
    from src.daily import auction_capture

    requested: list[str] = []

    class _Store:
        def read_cohort(self, day: str, *, available_by: Any) -> Any:
            requested.append(day)
            raise FileNotFoundError(day)

    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))
    from datetime import datetime

    auction_capture._resolve_roster("2026-09-28", "open", _Store(), datetime(2026, 9, 28, 8, 0), "2026-09-23")  # type: ignore[arg-type]
    assert requested == ["2026-09-23"]


def test_main_logs_non_trading_day_skip(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from src.daily import auction_capture

    async def _holiday(snapshot_date: str, phase: str, profile: Any, stats_sink: Any = None) -> None:
        return None

    monkeypatch.setattr(auction_capture, "CollectionSettings", lambda: _profile(tmp_path))
    monkeypatch.setattr(auction_capture, "_run_async", _holiday)
    recorded: list[tuple] = []
    monkeypatch.setattr(
        auction_capture, "record_run_outcome", lambda job, outcome, **kw: recorded.append((job, outcome, kw))
    )
    with caplog.at_level(logging.INFO, logger=auction_capture.logger.name):
        auction_capture.main(["--phase", "close", "--date", "2026-09-24"])
    assert any("reason=non_trading_day" in r.message for r in caplog.records)
    assert recorded == [
        (
            "auction_capture",
            "SKIPPED",
            {"run_date": "2026-09-24", "reason": "non_trading_day", "metrics": {"session": "CLOSED"}},
        )
    ]


def _shifted_close_clock(trading_day):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.data.capture_contracts import SessionClock

    seoul = ZoneInfo("Asia/Seoul")

    def _at(hour: int, minute: int) -> datetime:
        return datetime(trading_day.year, trading_day.month, trading_day.day, hour, minute, 0, tzinfo=seoul)

    return SessionClock(
        trading_date=trading_day,
        open_at=_at(10, 0),
        close_at=_at(16, 30),
        close_confirmation_deadline=_at(16, 33),
        provenance="csat_delayed_open",
    )


def test_run_async_uses_calendar_clock_for_shifted_session(tmp_path, monkeypatch) -> None:
    import asyncio
    from datetime import date

    from src.daily import auction_capture
    from src.data.session_calendar import SessionDay, SessionKind

    _stub_research_clients(monkeypatch)
    target = date(2026, 11, 19)
    clock = _shifted_close_clock(target)
    monkeypatch.setattr(
        auction_capture, "resolve_session_day",
        lambda _d, **_k: SessionDay(trading_date=target, kind=SessionKind.SHIFTED, clock=clock, provenance="krx_calendar"),
    )

    async def _open(client: Any, session: Any, date: Any) -> bool:
        return True

    monkeypatch.setattr(auction_capture, "is_kis_trading_day", _open)
    seen: dict[str, Any] = {}

    async def _capture_clock(*args: Any, **kwargs: Any) -> str:
        seen["clock"] = kwargs["session_clock"]
        return "manifest-ok"

    monkeypatch.setattr(auction_capture, "run_auction_capture", _capture_clock)
    assert asyncio.run(auction_capture._run_async("2026-11-19", "close", _profile(tmp_path))) == "manifest-ok"
    assert seen["clock"].close_at.strftime("%H:%M") == "16:30"


def test_run_async_skips_closed_day_before_broker_session(tmp_path, monkeypatch) -> None:
    import asyncio
    from datetime import date

    from src.daily import auction_capture
    from src.data.session_calendar import SessionDay, SessionKind

    _stub_research_clients(monkeypatch)
    monkeypatch.setattr(
        auction_capture, "resolve_session_day",
        lambda _d, **_k: SessionDay(trading_date=date(2026, 10, 9), kind=SessionKind.CLOSED, clock=None, provenance="krx_calendar"),
    )

    class _RaisingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("closed day must not construct broker clients")

    monkeypatch.setattr("src.api.kis.client.KisApiClient", _RaisingClient)

    async def _must_not_run(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("capture must not run on a closed day")

    monkeypatch.setattr(auction_capture, "run_auction_capture", _must_not_run)
    assert asyncio.run(auction_capture._run_async("2026-10-09", "close", _profile(tmp_path))) is None


def test_audit_and_capture_schedule_identical_rounds() -> None:
    import datetime as dt

    from src.daily import auction_capture

    clock = _clock("2026-09-17")
    close_rounds = auction_capture.close_rounds(clock, 60)
    assert len(close_rounds) == 9
    assert close_rounds[0] == clock.close_at - dt.timedelta(minutes=9)
    assert close_rounds[-1] == clock.close_at - dt.timedelta(minutes=1)
    assert all(
        (close_rounds[index + 1] - close_rounds[index]).total_seconds() == 60
        for index in range(len(close_rounds) - 1)
    )
    program_rounds = auction_capture.program_rounds(clock)
    assert len(program_rounds) == 2


def test_daily_audit_imports_no_private_round_names() -> None:
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/tools/daily_audit.py").read_text(encoding="utf-8"))
    private: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "src.daily.auction_capture":
            private.extend(alias.name for alias in node.names if alias.name.startswith("_"))
    assert private == []


def _surge_entry(symbol: str, status: Any, scheduled_at: Any) -> Any:
    from src.data.capture_contracts import CaptureDataset, CoverageEntry

    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.ORDERBOOK,
        venue="KRX",
        session="regular",
        scheduled_at=scheduled_at,
        status=status,
        rows=1,
        first_event_time=scheduled_at,
        last_event_time=scheduled_at,
        reason="auction-close",
        raw_refs=(),
    )


def test_observe_roster_completes_round_at_concurrent_throughput() -> None:
    """400-symbol round with 75 ms legs and 8-per-key concurrency finishes in ~25 waves."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    state = {"now": _seoul(2026, 9, 17, 15, 21, 0)}
    roster = [f"{i:06d}" for i in range(400)]

    class _LagClient:
        def __init__(self) -> None:
            self.calls = 0

        async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
            self.calls += 1
            start = state["now"]
            await asyncio.sleep(0)
            target = start + dt.timedelta(seconds=0.075)
            if state["now"] < target:
                state["now"] = target
            return {"rt_cd": "0"}

    clients = [_LagClient(), _LagClient()]
    t0 = state["now"]

    async def _observe(position: int, symbol: str, client: Any) -> Any:
        await client.get_orderbook_snapshot(None, symbol)
        return _surge_entry(symbol, CaptureStatus.COMPLETE, t0)

    def _on_deadline(position: int, symbol: str) -> Any:
        return _surge_entry(symbol, CaptureStatus.PARTIAL, t0)

    entries = _run(
        auction_capture._observe_roster(
            roster=roster,
            clients=clients,
            semaphore=asyncio.Semaphore(8 * 2),
            deadline=t0 + dt.timedelta(hours=1),
            now_fn=lambda: state["now"],
            observe=_observe,
            on_deadline=_on_deadline,
        )
    )

    # Then: every symbol completes at the account rate, not the serial latency rate
    assert len(entries) == 400
    assert all(e.status == CaptureStatus.COMPLETE for e in entries)
    assert [e.symbol for e in entries] == roster
    assert clients[0].calls + clients[1].calls == 400
    elapsed = (state["now"] - t0).total_seconds()
    assert elapsed <= 400 * 0.075 / 16 + 0.075


def test_observe_roster_keeps_roster_order_under_random_delays() -> None:
    """Completion order never leaks into the deterministic manifest order."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    now = _seoul(2026, 9, 17, 15, 21, 0)
    roster = [f"{i:06d}" for i in range(50)]
    delays = [((i * 37) % 11) * 0.0005 for i in range(len(roster))]

    async def _observe(position: int, symbol: str, client: Any) -> Any:
        await asyncio.sleep(delays[position])
        return _surge_entry(symbol, CaptureStatus.COMPLETE, now)

    def _on_deadline(position: int, symbol: str) -> Any:
        return _surge_entry(symbol, CaptureStatus.PARTIAL, now)

    entries = _run(
        auction_capture._observe_roster(
            roster=roster,
            clients=[object()],
            semaphore=asyncio.Semaphore(8),
            deadline=now + dt.timedelta(hours=1),
            now_fn=lambda: now,
            observe=_observe,
            on_deadline=_on_deadline,
        )
    )

    assert [e.symbol for e in entries] == roster


def test_observe_roster_marks_undispatched_tail_deadline_without_requests() -> None:
    """Symbols dispatched at or after the cutoff never hit the vendor."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    state = {"now": _seoul(2026, 9, 17, 15, 21, 0)}
    t0 = state["now"]
    roster = [f"{i:06d}" for i in range(10)]
    sent: list[str] = []

    async def _observe(position: int, symbol: str, client: Any) -> Any:
        sent.append(symbol)
        state["now"] += dt.timedelta(seconds=0.1)
        return _surge_entry(symbol, CaptureStatus.COMPLETE, t0)

    def _on_deadline(position: int, symbol: str) -> Any:
        entry = _surge_entry(symbol, CaptureStatus.PARTIAL, t0)
        return entry.model_copy(
            update={"reason": "deadline", "rows": 0, "first_event_time": None, "last_event_time": None}
        )

    entries = _run(
        auction_capture._observe_roster(
            roster=roster,
            clients=[object()],
            semaphore=asyncio.Semaphore(16),
            deadline=t0 + dt.timedelta(seconds=0.25),
            now_fn=lambda: state["now"],
            observe=_observe,
            on_deadline=_on_deadline,
        )
    )

    # Then: three dispatches fit before the cutoff; the tail is PARTIAL without a request
    assert sent == roster[:3]
    assert len(entries) == len(roster)
    assert [e.symbol for e in entries] == roster
    tail = entries[3:]
    assert all(e.status == CaptureStatus.PARTIAL and e.reason == "deadline" for e in tail)
    assert all(e.first_event_time is None for e in tail)


def test_program_deadline_entry_marks_undispatched_tail() -> None:
    """확인 마감에 못 든 프로그램 심볼은 요청 없이 PARTIAL(deadline)이다."""
    import functools

    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    state = {"now": _seoul(2026, 9, 17, 15, 22, 0)}
    t0 = state["now"]
    roster = [f"{i:06d}" for i in range(6)]
    sent: list[str] = []

    async def _observe(position: int, symbol: str, client: Any) -> Any:
        sent.append(symbol)
        state["now"] += dt.timedelta(seconds=60)
        return _surge_entry(symbol, CaptureStatus.COMPLETE, t0)

    entries = _run(
        auction_capture._observe_roster(
            roster=roster,
            clients=[object()],
            semaphore=asyncio.Semaphore(16),
            deadline=t0 + dt.timedelta(seconds=90),
            now_fn=lambda: state["now"],
            observe=_observe,
            on_deadline=functools.partial(auction_capture._program_deadline_entry, scheduled_at=t0),
        )
    )

    # Then: 두 디스패치만 마감 안에 들고 나머지는 요청 없이 PARTIAL이다
    assert sent == roster[:2]
    assert [e.symbol for e in entries] == roster
    tail = entries[2:]
    assert all(e.status == CaptureStatus.PARTIAL and e.reason == "deadline" for e in tail)
    assert all(e.dataset == CaptureDataset.PROGRAM and e.first_event_time is None for e in tail)


def test_open_deadline_entry_keeps_last_seen_evidence() -> None:
    """확인 종료 후 디스패치는 최종 PARTIAL(open_unresolved)로 확정된다."""
    import functools

    from src.daily import auction_capture
    from src.data.capture_contracts import ArtifactRef, CaptureDataset, CaptureStatus

    now = _seoul(2026, 9, 17, 9, 1, 0)
    floor = _seoul(2026, 9, 17, 9, 0, 30)
    ref = ArtifactRef(path="r/0.json.gz", sha256="ab" * 32, bytes=10)
    poll_started = {"000002": _seoul(2026, 9, 17, 9, 0, 35)}
    last_seen = {"000002": (ref, _seoul(2026, 9, 17, 9, 0, 50))}
    settled: set[int] = set()

    async def _never(_position: int, _symbol: str, _client: Any) -> Any:
        raise AssertionError("must not dispatch past confirmation end")

    entries = _run(
        auction_capture._observe_roster(
            roster=["000001", "000002"],
            clients=[object()],
            semaphore=asyncio.Semaphore(8),
            deadline=now,
            now_fn=lambda: now,
            observe=_never,
            on_deadline=functools.partial(
                auction_capture._open_deadline_entry,
                floor=floor,
                now_fn=lambda: now,
                poll_started=poll_started,
                last_seen=last_seen,
                settled=settled,
            ),
        )
    )

    # Then: 두 심볼 모두 최종 확정되고 마지막 목격 증거가 보존된다
    assert settled == {0, 1}
    by_symbol = {e.symbol: e for e in entries}
    assert by_symbol["000001"].reason == "open_unresolved"
    assert by_symbol["000001"].first_event_time == now
    assert by_symbol["000001"].raw_refs == ()
    assert by_symbol["000002"].first_event_time == poll_started["000002"]
    assert by_symbol["000002"].raw_refs == (ref,)
    assert all(e.dataset == CaptureDataset.PRICE and e.status == CaptureStatus.PARTIAL for e in entries)


def test_program_and_orderbook_rounds_share_one_bound(tmp_path) -> None:
    """Concurrent program attribution never pushes in-flight requests past the semaphore."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", [f"{i:06d}" for i in range(6)])
    profile = _profile(tmp_path)
    clock = _clock("2026-09-17")
    now = _seoul(2026, 9, 17, 15, 0)
    state = {"inflight": 0, "max": 0}
    orderbook_seen: list[str] = []
    program_seen: list[str] = []

    class _BoundedClient(_FakeClient):
        async def _tracked(self, kind: str, code: str) -> dict[str, Any]:
            state["inflight"] += 1
            state["max"] = max(state["max"], state["inflight"])
            await asyncio.sleep(0.01)
            state["inflight"] -= 1
            return {"rt_cd": "0", "output1": {}, "output2": []}

        async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None):
            orderbook_seen.append(code)
            return await self._tracked("orderbook", code)

        async def get_program_net_buy(self, session: Any, code: str, market_div_code: str | None = None):
            program_seen.append(code)
            return await self._tracked("program", code)

    clients = [_BoundedClient(calls=[]), _BoundedClient(calls=[])]
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=profile,
            store=store,
            clients=clients,
            session_clock=clock,
            now_fn=lambda: now,
        )
    )

    # Then: both sweeps ran against the vendor under the shared 8-per-key bound
    assert orderbook_seen and program_seen
    assert state["max"] <= 8 * len(clients)
    orderbook_entries = [e for e in manifest.entries if e.dataset == CaptureDataset.ORDERBOOK]
    assert len(orderbook_entries) == len(auction_capture.close_rounds(clock, 60)) * 6


def test_open_repass_observes_only_unresolved_symbols(tmp_path, monkeypatch) -> None:
    """Pass 2 polls exactly the symbols still without an opening price."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    store = _store(tmp_path)
    roster = ["000001", "000002", "000003", "000004", "000005"]
    _publish_cohort(store, "2026-09-16", roster)
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=60)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    client = _FakeClient(
        calls=[],
        price_seq={
            "000001": [72000],
            "000002": [72000],
            "000003": [0, 72000],
            "000004": [0, 72000],
            "000005": [0, 72000],
        },
    )
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17", phase="open", profile=profile, store=store, clients=[client],
            session_clock=clock, now_fn=lambda: state["now"], sleep_fn=_sleep,
        )
    )

    # Then: pass 2 observes exactly the 3 symbols unresolved after pass 1, each once more
    price_calls = [code for kind, code in client.calls if kind == "price"]
    assert price_calls.count("000001") == 1
    assert price_calls.count("000002") == 1
    assert price_calls.count("000003") == 2
    assert price_calls.count("000004") == 2
    assert price_calls.count("000005") == 2
    price_entries = [e for e in manifest.entries if e.dataset == CaptureDataset.PRICE]
    assert len(price_entries) == 5
    assert all(e.status == CaptureStatus.COMPLETE for e in price_entries)
    assert manifest.status == CaptureStatus.COMPLETE


def test_open_roster_ledger_failure_marks_manifest_partial(tmp_path, monkeypatch, caplog) -> None:
    """A not-ok held roster forces PARTIAL even when every observed entry is COMPLETE."""
    import logging

    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001", "000002"])
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), False, "OSError"))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=600)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    def _advance() -> None:
        state["now"] += dt.timedelta(seconds=20)

    client = _FakeClient(price_seq={"000001": [100], "000002": [200]}, calls=[], advance=_advance)
    with caplog.at_level(logging.WARNING, logger=auction_capture.logger.name):
        manifest = _run(
            auction_capture.run_auction_capture(
                "2026-09-17",
                phase="open",
                profile=profile,
                store=store,
                clients=[client],
                session_clock=clock,
                now_fn=lambda: state["now"],
                sleep_fn=_sleep,
            )
        )
    assert manifest.status == CaptureStatus.PARTIAL
    assert any("reason=held_roster_unavailable error=OSError" in r.message for r in caplog.records)


def test_open_roster_happy_path_stays_complete(tmp_path, monkeypatch) -> None:
    """Held symbols join the roster after cohort codes without forcing PARTIAL."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001"])
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster(("000009",), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=600)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    def _advance() -> None:
        state["now"] += dt.timedelta(seconds=20)

    client = _FakeClient(price_seq={"000001": [100], "000009": [200]}, calls=[], advance=_advance)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="open",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: state["now"],
            sleep_fn=_sleep,
        )
    )
    symbols = [e.symbol for e in manifest.entries]
    assert symbols.index("000001") < symbols.index("000009")
    assert manifest.status == CaptureStatus.COMPLETE


def test_open_roster_missing_previous_uses_held_symbols(tmp_path, monkeypatch) -> None:
    """No previous cohort: the roster is exactly the held symbols, still incomplete."""
    from src.daily import auction_capture

    store = _store(tmp_path)
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster(("000003",), True, None))
    roster, cohort_id, incomplete = auction_capture._resolve_roster(  # type: ignore[arg-type]
        "2026-09-17", "open", store, _seoul(2026, 9, 17, 9, 5), "2026-09-16",
    )
    assert (roster, cohort_id, incomplete) == (["000003"], None, True)


def test_close_roster_never_reads_ledger(tmp_path, monkeypatch) -> None:
    """Close phase resolves without touching the ledger."""
    from src.daily import auction_capture

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])

    def _boom() -> HeldRoster:
        raise AssertionError("ledger must not be read")

    monkeypatch.setattr(auction_capture, "load_held_roster", _boom)
    roster, _, incomplete = auction_capture._resolve_roster(  # type: ignore[arg-type]
        "2026-09-17", "close", store, _seoul(2026, 9, 17, 15, 0),
    )
    assert roster == ["000001"]
    assert incomplete is False


def test_open_roster_missing_previous_and_ledger_failure_warns_both(tmp_path, monkeypatch, caplog) -> None:
    """No previous cohort plus a not-ok roster: empty roster, still incomplete, both warnings."""
    import logging

    from src.daily import auction_capture

    store = _store(tmp_path)
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), False, "OSError"))
    with caplog.at_level(logging.WARNING, logger=auction_capture.logger.name):
        roster, cohort_id, incomplete = auction_capture._resolve_roster(  # type: ignore[arg-type]
            "2026-09-17", "open", store, _seoul(2026, 9, 17, 9, 5), "2026-09-16",
        )
    assert (roster, cohort_id, incomplete) == ([], None, True)
    assert any("reason=missing_previous" in r.message for r in caplog.records)
    assert any("reason=held_roster_unavailable error=OSError" in r.message for r in caplog.records)


def test_run_async_continues_with_healthy_slots_when_one_token_fails(tmp_path, monkeypatch) -> None:
    from src.daily import auction_capture

    class _Cred:
        def __init__(self, slot: str) -> None:
            self.slot, self.app_key, self.app_secret, self.hts_id = slot, f"key-{slot}", "sec", "hts"

    monkeypatch.setattr(
        auction_capture, "resolve_research_credentials", lambda env, *, slots: (_Cred("DATA_1"), _Cred("DATA_2"))
    )

    class _Client:
        def __init__(self, app_key: str, *args: Any, **kwargs: Any) -> None:
            self.app_key = app_key

        def create_session(self) -> _Session:
            return _Session()

        async def ensure_token(self, session: Any) -> None:
            if self.app_key == "key-DATA_1":
                raise RuntimeError("token issuance failed")

    monkeypatch.setattr("src.api.kis.client.KisApiClient", _Client)
    seen: dict[str, Any] = {}

    async def _capture(*args: Any, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "manifest-ok"

    async def _open(client: Any, session: Any, date: Any) -> bool:
        seen["probe_client"] = client.app_key
        return True

    monkeypatch.setattr(auction_capture, "run_auction_capture", _capture)
    monkeypatch.setattr(auction_capture, "is_kis_trading_day", _open)

    out = asyncio.run(auction_capture._run_async("2026-09-17", "close", _profile(tmp_path)))

    assert out == "manifest-ok"
    assert [c.app_key for c in seen["clients"]] == ["key-DATA_2"]
    assert seen["degraded_slots"] == ("DATA_1",)
    assert seen["probe_client"] == "key-DATA_2"


def test_degraded_slot_marks_manifest_partial(tmp_path) -> None:
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    now = _seoul(2026, 9, 17, 15, 0)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_profile(tmp_path),
            store=store,
            clients=[_FakeClient(calls=[])],
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: now,
            degraded_slots=("DATA_1",),
        )
    )
    assert manifest.status == CaptureStatus.PARTIAL


def _publish_research_roster(store: Any, trading_date: str, members: list[str], *, completed_at: dt.datetime) -> None:
    from src.data.research_roster import ResearchRoster, RosterCandidate, build_roster_frame, publish_research_roster

    day = dt.date.fromisoformat(trading_date)
    roster = ResearchRoster(
        trading_date=day,
        roster_id=f"research-roster-{trading_date}",
        rule_version="liquid_any_direction@v1",
        observed_at=completed_at - dt.timedelta(seconds=5),
        as_of_panel_date=day - dt.timedelta(days=1),
        members=tuple(members),
        dropped={},
        truncated=False,
    )
    candidates = [
        RosterCandidate(symbol=member, rank=rank, trade_value_100m=500.0, change_pct=0.0, last_price=10_000)
        for rank, member in enumerate(members, start=1)
    ]
    frame = build_roster_frame(candidates, roster.members, {}, dict.fromkeys(roster.members, 10_000.0))
    publish_research_roster(store, roster, frame, raw_refs=(), completed_at=completed_at)


def _close_run(root: Path, client: Any, *, state: dict[str, dt.datetime] | None = None) -> tuple[Any, Any]:
    from src.daily import auction_capture

    store = _store(root)
    _publish_cohort(store, "2026-09-17", ["000001", "000002"])
    _publish_research_roster(store, "2026-09-17", ["000003"], completed_at=_seoul(2026, 9, 17, 15, 0))
    profile = _profile(root)
    clock_state = state if state is not None else {"now": _seoul(2026, 9, 17, 15, 20)}
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: clock_state["now"],
        )
    )
    return store, manifest


def test_close_primary_manifest_with_extras(tmp_path) -> None:
    """Extras never alter the primary manifest and get their own sibling manifest."""
    from src.daily.auction_capture import close_rounds, program_rounds
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    on_store, on = _close_run(tmp_path / "on", _FakeClient(calls=[]))

    assert {e.symbol for e in on.entries} == {"000001", "000002"}
    assert on.status == CaptureStatus.COMPLETE
    research = [m for m in on_store.read_manifests("2026-09-17") if m.context.capture_reason == "auction-close-research"]
    assert len(research) == 1
    assert {e.symbol for e in research[0].entries} == {"000003"}
    clock = _clock("2026-09-17")
    expected = {(CaptureDataset.ORDERBOOK, slot) for slot in close_rounds(clock, 60)} | {
        (CaptureDataset.PROGRAM, slot) for slot in program_rounds(clock)
    }
    assert {(e.dataset, e.scheduled_at) for e in research[0].entries} == expected


def test_close_extras_dispatched_after_cohort(tmp_path) -> None:
    """Every round requests the whole cohort before any extra; rotation never changes the multiset."""
    from src.daily.auction_capture import close_rounds, program_rounds

    client = _FakeClient(calls=[])
    _close_run(tmp_path, client)

    rounds = len(close_rounds(_clock("2026-09-17"), 60))
    orderbook = [code for kind, code in client.calls if kind == "orderbook"]
    assert len(orderbook) == 3 * rounds
    for start in range(0, len(orderbook), 3):
        round_codes = orderbook[start : start + 3]
        assert set(round_codes[:2]) == {"000001", "000002"}
        assert round_codes[2] == "000003"
    assert sorted(orderbook) == sorted(["000001", "000002", "000003"] * rounds)
    assert [code for kind, code in client.calls if kind == "program"] == ["000001", "000002", "000003"] * len(
        program_rounds(_clock("2026-09-17"))
    )


def test_close_deadline_hits_extras_first(tmp_path, monkeypatch) -> None:
    """At the round deadline the cohort is complete while the extra is dispatched as a deadline miss."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    monkeypatch.setattr(auction_capture, "program_rounds", lambda clock: [])
    state = {"now": _seoul(2026, 9, 17, 15, 20)}
    round_start = _seoul(2026, 9, 17, 15, 21)
    round_deadline = _seoul(2026, 9, 17, 15, 22)

    class _DeadlineClient(_FakeClient):
        async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
            if code == "000002" and state["now"] < round_deadline:
                state["now"] = round_deadline
            return await super().get_orderbook_snapshot(session, code, market_div_code)

    store, manifest = _close_run(tmp_path, _DeadlineClient(calls=[]), state=state)

    assert manifest.status == CaptureStatus.COMPLETE
    first_round = [e for e in manifest.entries if e.scheduled_at == round_start]
    assert {e.symbol: e.status for e in first_round} == {"000001": CaptureStatus.COMPLETE, "000002": CaptureStatus.COMPLETE}
    research = [m for m in store.read_manifests("2026-09-17") if m.context.capture_reason == "auction-close-research"]
    assert len(research) == 1
    assert research[0].status == CaptureStatus.PARTIAL
    (extra,) = [e for e in research[0].entries if e.scheduled_at == round_start]
    assert extra.symbol == "000003"
    assert extra.dataset == CaptureDataset.ORDERBOOK
    assert extra.status == CaptureStatus.PARTIAL
    assert extra.reason == "deadline"


def test_open_uses_previous_day_roster(tmp_path, monkeypatch) -> None:
    """Opening extras come from the previous trading day's roster, never today's."""
    from src.daily import auction_capture

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001"])
    _publish_research_roster(store, "2026-09-16", ["000009"], completed_at=_seoul(2026, 9, 16, 15, 20))
    _publish_research_roster(store, "2026-09-17", ["000008"], completed_at=_seoul(2026, 9, 17, 8, 0))
    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster((), True, None))
    profile = _profile(tmp_path, COLLECTION_OPEN_CONFIRM_SECONDS=600)
    clock = _clock("2026-09-17")
    state = {"now": _seoul(2026, 9, 17, 8, 30)}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    client = _FakeClient(price_seq={"000001": [100], "000008": [100], "000009": [100]}, calls=[])
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="open",
            profile=profile,
            store=store,
            clients=[client],
            session_clock=clock,
            now_fn=lambda: state["now"],
            sleep_fn=_sleep,
        )
    )

    assert {e.symbol for e in manifest.entries} == {"000001"}
    research = [m for m in store.read_manifests("2026-09-17") if m.context.capture_reason == "auction-open-research"]
    assert len(research) == 1
    assert {e.symbol for e in research[0].entries} == {"000009"}
    assert ("price", "000008") not in client.calls


def test_estimate_capacity_two_keys() -> None:
    from src.daily.auction_capture import estimate_round_capacity

    assert estimate_round_capacity(2, 60, utilization=0.85, tps_per_key=18.0) == 1836
    assert estimate_round_capacity(0, 60, utilization=0.85, tps_per_key=18.0) == 0
    for bad, pattern in (
        ({"interval_seconds": 0}, "interval_seconds"),
        ({"utilization": 0.0}, "utilization"),
        ({"utilization": 1.01}, "utilization"),
        ({"utilization": float("nan")}, "utilization"),
        ({"tps_per_key": 0.0}, "tps_per_key"),
        ({"tps_per_key": float("inf")}, "tps_per_key"),
        ({"tps_per_key": float("nan")}, "tps_per_key"),
    ):
        kwargs: dict[str, Any] = {"interval_seconds": 60, "utilization": 0.85, "tps_per_key": 18.0, **bad}
        with pytest.raises(ValueError, match=pattern):
            estimate_round_capacity(2, **kwargs)


def test_plan_order_is_permutation_and_tiered() -> None:
    from src.daily.auction_capture import plan_round_order

    symbols = [f"{i:06d}" for i in range(10)]
    priority = {symbols[7], symbols[3]}
    low = {symbols[5], symbols[3]}
    for round_index in range(9):
        order = plan_round_order(symbols, priority=priority, low_priority=low, round_index=round_index, round_count=9)
        assert sorted(order) == symbols
        assert order[:2] == [symbols[3], symbols[7]]
        assert order[-1] == symbols[5]
        assert set(order[2:-1]) == set(symbols) - priority - {symbols[5]}


def test_plan_order_rotation_spreads_tail() -> None:
    from src.daily.auction_capture import plan_round_order

    n, rounds, capacity = 1125, 9, 700
    symbols = [f"{i:06d}" for i in range(n)]
    tail_rounds = dict.fromkeys(symbols, 0)
    naive_tail_rounds = dict.fromkeys(symbols, 0)
    for round_index in range(rounds):
        order = plan_round_order(symbols, priority=frozenset(), low_priority=frozenset(), round_index=round_index, round_count=rounds)
        for symbol in order[capacity:]:
            tail_rounds[symbol] += 1
        for symbol in symbols[capacity:]:
            naive_tail_rounds[symbol] += 1

    assert sum(tail_rounds.values()) == rounds * (n - capacity)
    assert set(tail_rounds.values()) <= {3, 4}
    assert max(tail_rounds.values()) - min(tail_rounds.values()) <= 1
    assert sorted(set(naive_tail_rounds.values())) == [0, rounds]
    assert sum(1 for count in naive_tail_rounds.values() if count == rounds) == n - capacity


def test_plan_order_deterministic_and_validates() -> None:
    from src.daily.auction_capture import plan_round_order

    symbols = [f"{i:06d}" for i in range(20)]
    kwargs: dict[str, Any] = {"priority": {symbols[1]}, "low_priority": {symbols[2]}, "round_index": 4, "round_count": 9}
    assert plan_round_order(symbols, **kwargs) == plan_round_order(symbols, **kwargs)
    assert plan_round_order([], priority=set(), low_priority=set(), round_index=0, round_count=1) == []
    with pytest.raises(ValueError, match="round_count"):
        plan_round_order(symbols, **{**kwargs, "round_count": 0})
    with pytest.raises(ValueError, match="round_index"):
        plan_round_order(symbols, **{**kwargs, "round_index": 9})
    with pytest.raises(ValueError, match="round_index"):
        plan_round_order(symbols, **{**kwargs, "round_index": -1})
    with pytest.raises(ValueError, match="unique"):
        plan_round_order([*symbols, symbols[0]], **kwargs)


def _gate_entry(dataset: Any, symbol: str, scheduled_at: dt.datetime, status: Any, *, reason: str = "auction-close") -> Any:
    from src.data.capture_contracts import CaptureStatus, CoverageEntry

    requested = reason != "deadline" or status is CaptureStatus.COMPLETE
    return CoverageEntry(
        symbol=symbol,
        dataset=dataset,
        venue="KRX",
        session="regular",
        scheduled_at=scheduled_at,
        status=status,
        rows=1 if status is CaptureStatus.COMPLETE else 0,
        first_event_time=scheduled_at if requested else None,
        last_event_time=scheduled_at if requested else None,
        reason=reason,
        raw_refs=(),
    )


class _VirtualLane:
    """Serial virtual-time harness: each request advances the clock one second."""

    def __init__(self, start: dt.datetime) -> None:
        from src.data.capture_contracts import CaptureDataset

        self.dataset = CaptureDataset
        self.state = {"now": start}
        self.starts: list[tuple[str, dt.datetime]] = []
        self.hold = asyncio.Event()

    def now(self) -> dt.datetime:
        return self.state["now"]

    def observe(self, kind: str, dataset: Any, scheduled_at: dt.datetime, *, hold_first: bool = False) -> Any:
        from src.data.capture_contracts import CaptureStatus

        async def _observe(position: int, symbol: str, client: Any) -> Any:
            self.starts.append((kind, self.state["now"]))
            if hold_first and position == 0:
                await self.hold.wait()
            began = self.state["now"]
            await asyncio.sleep(0)
            self.state["now"] = max(self.state["now"], began) + dt.timedelta(seconds=1)
            return _gate_entry(dataset, symbol, scheduled_at, CaptureStatus.COMPLETE)

        return _observe

    def on_deadline(self, dataset: Any, scheduled_at: dt.datetime) -> Any:
        from src.data.capture_contracts import CaptureStatus

        return lambda position, symbol: _gate_entry(dataset, symbol, scheduled_at, CaptureStatus.PARTIAL, reason="deadline")


def test_program_lane_yields_to_orderbook() -> None:
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    t0 = _seoul(2026, 11, 10, 15, 22)
    lane = _VirtualLane(t0)
    roster = [f"{i:06d}" for i in range(12)]
    semaphore = asyncio.Semaphore(1)

    async def scenario() -> tuple[list[Any], list[Any], dt.datetime, dt.datetime]:
        gate = auction_capture._OrderbookPriorityGate()
        program = asyncio.create_task(
            auction_capture._observe_roster(
                roster=roster,
                clients=[object()],
                semaphore=semaphore,
                deadline=t0 + dt.timedelta(minutes=4),
                now_fn=lane.now,
                observe=lane.observe("program", CaptureDataset.PROGRAM, t0, hold_first=True),
                on_deadline=lane.on_deadline(CaptureDataset.PROGRAM, t0),
                gate=gate,
            )
        )
        for _ in range(20):
            await asyncio.sleep(0)
        assert [kind for kind, _ in lane.starts] == ["program"]  # one request in flight, the rest parked
        gate.begin_round()
        orderbook_started = lane.now()
        orderbook = asyncio.create_task(
            auction_capture._observe_roster(
                roster=roster,
                clients=[object()],
                semaphore=semaphore,
                deadline=t0 + dt.timedelta(seconds=20),
                now_fn=lane.now,
                observe=lane.observe("orderbook", CaptureDataset.ORDERBOOK, t0),
                on_deadline=lane.on_deadline(CaptureDataset.ORDERBOOK, t0),
            )
        )
        await asyncio.sleep(0)
        lane.hold.set()
        try:
            orderbook_entries = await orderbook
        finally:
            gate.end_round()
        orderbook_ended = lane.now()
        assert orderbook_started == t0
        return orderbook_entries, await program, orderbook_started, orderbook_ended

    orderbook_entries, program_entries, _, orderbook_ended = _run(scenario())

    assert [e.status for e in orderbook_entries] == [CaptureStatus.COMPLETE] * len(roster)
    program_starts = [moment for kind, moment in lane.starts if kind == "program"]
    assert len(program_starts) == len(roster)
    assert program_starts[0] == t0
    assert all(moment >= orderbook_ended for moment in program_starts[1:])
    assert [e.status for e in program_entries] == [CaptureStatus.COMPLETE] * len(roster)


def test_program_lane_starves_only_when_orderbook_saturates() -> None:
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    t0 = _seoul(2026, 11, 10, 15, 22)
    lane = _VirtualLane(t0)
    roster = [f"{i:06d}" for i in range(5)]

    async def scenario() -> tuple[list[Any], list[Any]]:
        gate = auction_capture._OrderbookPriorityGate()
        gate.begin_round()
        program = asyncio.create_task(
            auction_capture._observe_roster(
                roster=roster,
                clients=[object()],
                semaphore=asyncio.Semaphore(2),
                deadline=t0 + dt.timedelta(seconds=10),
                now_fn=lane.now,
                observe=lane.observe("program", CaptureDataset.PROGRAM, t0),
                on_deadline=lane.on_deadline(CaptureDataset.PROGRAM, t0),
                gate=gate,
            )
        )
        try:
            orderbook_entries = await auction_capture._observe_roster(
                roster=roster,
                clients=[object()],
                semaphore=asyncio.Semaphore(2),
                deadline=t0 + dt.timedelta(minutes=1),
                now_fn=lane.now,
                observe=lane.observe("orderbook", CaptureDataset.ORDERBOOK, t0),
                on_deadline=lane.on_deadline(CaptureDataset.ORDERBOOK, t0),
            )
            lane.state["now"] = t0 + dt.timedelta(seconds=20)
        finally:
            gate.end_round()
        return orderbook_entries, await program

    orderbook_entries, program_entries = _run(scenario())

    assert all(e.status is CaptureStatus.COMPLETE for e in orderbook_entries)
    assert [kind for kind, _ in lane.starts] == ["orderbook"] * len(roster)
    assert all(e.status is CaptureStatus.PARTIAL and e.reason == "deadline" and e.first_event_time is None for e in program_entries)
    assert [e.symbol for e in program_entries] == roster


def test_gate_none_behavior_unchanged() -> None:
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset

    t0 = _seoul(2026, 11, 10, 15, 22)
    roster = [f"{i:06d}" for i in range(6)]

    def _entries(**extra: Any) -> tuple[list[Any], list[tuple[str, dt.datetime]]]:
        lane = _VirtualLane(t0)
        entries = _run(
            auction_capture._observe_roster(
                roster=roster,
                clients=[object()],
                semaphore=asyncio.Semaphore(1),
                deadline=t0 + dt.timedelta(seconds=3),
                now_fn=lane.now,
                observe=lane.observe("orderbook", CaptureDataset.ORDERBOOK, t0),
                on_deadline=lane.on_deadline(CaptureDataset.ORDERBOOK, t0),
                **extra,
            )
        )
        return entries, lane.starts

    default_entries, default_starts = _entries()
    explicit_entries, explicit_starts = _entries(gate=None)

    assert default_entries == explicit_entries
    assert default_starts == explicit_starts
    assert [e.symbol for e in default_entries] == roster
    assert [e.reason for e in default_entries] == ["auction-close"] * 3 + ["deadline"] * 3


def test_gate_released_on_exception(tmp_path, monkeypatch) -> None:
    from src.daily import auction_capture

    gates: list[Any] = []

    class _RecordingGate(auction_capture._OrderbookPriorityGate):
        def __init__(self) -> None:
            super().__init__()
            gates.append(self)

    real = auction_capture._observe_roster

    async def _boom_for_orderbook(**kwargs: Any) -> Any:
        if kwargs.get("gate") is None:
            assert not gates[0].is_idle
            raise RuntimeError("round blew up")
        return await real(**kwargs)

    monkeypatch.setattr(auction_capture, "_OrderbookPriorityGate", _RecordingGate)
    monkeypatch.setattr(auction_capture, "_observe_roster", _boom_for_orderbook)
    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])

    with pytest.raises(RuntimeError, match="round blew up"):
        _run(
            auction_capture.run_auction_capture(
                "2026-09-17",
                phase="close",
                profile=_profile(tmp_path),
                store=store,
                clients=[_FakeClient()],
                session_clock=_clock("2026-09-17"),
                now_fn=lambda: _seoul(2026, 9, 17, 15, 20),
            )
        )

    assert len(gates) == 1
    assert gates[0].is_idle


def _stubbed_close_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbols: list[str],
    *,
    n_clients: int = 2,
    status_for: Any = None,
    stats_sink: dict[str, float | str] | None = None,
) -> tuple[Any, list[list[str]]]:
    """Close run whose request layer is replaced by a stub; returns the manifest and each orderbook round's roster."""
    from src.daily import auction_capture

    rosters: list[list[str]] = []
    monkeypatch.setattr(auction_capture, "_observe_roster", _roster_stub(rosters, status_for))
    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", symbols)
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_profile(tmp_path),
            store=store,
            clients=[_FakeClient() for _ in range(n_clients)],
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: _seoul(2026, 9, 17, 15, 20),
            stats_sink=stats_sink,
        )
    )
    return manifest, rosters


def _roster_stub(rosters: list[list[str]], status_for: Any = None) -> Any:
    from src.data.capture_contracts import CaptureStatus

    async def _stub(*, roster: Any, on_deadline: Any, gate: Any = None, **_: Any) -> list[Any]:
        if gate is None:
            rosters.append(list(roster))
        entries = []
        for position, symbol in enumerate(roster):
            base = on_deadline(position, symbol)
            if status_for is None:
                entries.append(base.model_copy(update={"status": CaptureStatus.COMPLETE, "rows": 1, "reason": "auction-close", "first_event_time": base.scheduled_at, "last_event_time": base.scheduled_at}))
            else:
                entries.append(status_for(position, base))
        return entries

    return _stub


def _main_close_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symbols: list[str],
    *,
    status_for: Any = None,
    degraded_slots: tuple[str, ...] = (),
) -> Any:
    """Drive the real CLI `main` for a close run with the request layer stubbed; returns the manifest it produced."""
    from src.daily import auction_capture

    produced: list[Any] = []
    monkeypatch.setattr(auction_capture, "_observe_roster", _roster_stub([], status_for))
    _publish_cohort(_store(tmp_path), "2026-09-17", symbols)
    monkeypatch.setattr(auction_capture, "CollectionSettings", lambda: _profile(tmp_path))

    async def _run_phase(snapshot_date: str, phase: str, profile: Any, stats_sink: Any = None) -> Any:
        manifest = await auction_capture.run_auction_capture(
            snapshot_date,
            phase="close",
            profile=profile,
            store=_store(tmp_path),
            clients=[_FakeClient(), _FakeClient()],
            session_clock=_clock(snapshot_date),
            now_fn=lambda: _seoul(2026, 9, 17, 15, 20),
            stats_sink=stats_sink,
            degraded_slots=degraded_slots,
        )
        produced.append(manifest)
        return manifest

    monkeypatch.setattr(auction_capture, "_run_async", _run_phase)
    auction_capture.main(["--phase", "close", "--date", "2026-09-17"])
    return produced[0]


def _auction_close_events(run_date: str) -> list[dict[str, Any]]:
    import json

    from src.tools.run_outcome import run_events_path

    path = run_events_path(run_date)
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [row for row in rows if row.get("job") == "auction_close"]


def _complete(base: Any) -> Any:
    from src.data.capture_contracts import CaptureStatus

    return base.model_copy(update={"status": CaptureStatus.COMPLETE, "rows": 1, "reason": "auction-close", "first_event_time": base.scheduled_at, "last_event_time": base.scheduled_at})


def _deadline_miss(base: Any) -> Any:
    from src.data.capture_contracts import CaptureStatus

    return base.model_copy(update={"status": CaptureStatus.PARTIAL, "reason": "deadline"})


def _vendor_miss(base: Any) -> Any:
    from src.data.capture_contracts import CaptureStatus

    return base.model_copy(update={"status": CaptureStatus.FAILED, "reason": "TimeoutError", "first_event_time": base.scheduled_at, "last_event_time": base.scheduled_at})


def _head_deadline(position: int, base: Any) -> Any:
    return _deadline_miss(base) if position < 3 else _complete(base)


def _head_vendor_failure(position: int, base: Any) -> Any:
    return _vendor_miss(base) if position < 3 else _complete(base)


def _head_mixed(position: int, base: Any) -> Any:
    if position == 0:
        return _vendor_miss(base)
    return _deadline_miss(base) if position < 3 else _complete(base)


def test_close_run_records_ok_outcome(tmp_path, monkeypatch) -> None:
    from src.data.capture_contracts import CaptureStatus

    manifest = _main_close_run(tmp_path, monkeypatch, [f"{i:06d}" for i in range(5)])

    assert manifest.status is CaptureStatus.COMPLETE
    (event,) = _auction_close_events("2026-09-17")
    assert event["outcome"] == "OK"
    assert event["reason"] == ""
    metrics = event["metrics"]
    assert {
        "roster", "extras", "clients", "capacity_per_round", "demand_per_round", "status", "start_lag_s",
        "n_deadline", "n_late", "n_failed", "max_round_duration_s", "rounds_with_deadline",
        "symbols_missing_any", "max_missed_rounds_per_symbol",
    } <= set(metrics)
    assert metrics["n_deadline"] == 0
    assert metrics["rounds_with_deadline"] == 0
    assert metrics["status"] == "OK"
    assert metrics["roster"] == 5 and metrics["clients"] == 2


def test_close_run_records_degraded_with_counts(tmp_path, monkeypatch) -> None:
    manifest = _main_close_run(tmp_path, monkeypatch, [f"{i:06d}" for i in range(10)], status_for=_head_deadline)

    deadline = [entry for entry in manifest.entries if entry.reason == "deadline"]
    assert deadline
    missed_rounds: dict[str, set[Any]] = {}
    for entry in deadline:
        missed_rounds.setdefault(entry.symbol, set()).add(entry.scheduled_at)
    (event,) = _auction_close_events("2026-09-17")
    assert event["outcome"] == "DEGRADED"
    assert event["reason"] == "deadline"
    metrics = event["metrics"]
    assert metrics["n_deadline"] == len(deadline)
    assert metrics["n_late"] == sum(entry.first_event_time is not None for entry in deadline)
    assert metrics["n_failed"] == 0
    assert metrics["rounds_with_deadline"] == len({entry.scheduled_at for entry in deadline})
    assert metrics["symbols_missing_any"] == len({entry.symbol for entry in deadline})
    assert metrics["max_missed_rounds_per_symbol"] == max(len(rounds) for rounds in missed_rounds.values())


@pytest.mark.parametrize(
    ("status_for", "reason"),
    [(_head_deadline, "deadline"), (_head_vendor_failure, "vendor_failure"), (_head_mixed, "mixed")],
)
def test_close_outcome_reason_names_dominant_cause(tmp_path, monkeypatch, status_for, reason) -> None:
    _main_close_run(tmp_path, monkeypatch, [f"{i:06d}" for i in range(10)], status_for=status_for)

    (event,) = _auction_close_events("2026-09-17")
    assert event["outcome"] == "DEGRADED"
    assert event["reason"] == reason


def test_close_outcome_degraded_without_entry_failures_has_no_reason(tmp_path, monkeypatch) -> None:
    from src.data.capture_contracts import CaptureStatus

    manifest = _main_close_run(tmp_path, monkeypatch, [f"{i:06d}" for i in range(5)], degraded_slots=("5",))

    assert manifest.status is CaptureStatus.PARTIAL
    (event,) = _auction_close_events("2026-09-17")
    assert event["outcome"] == "DEGRADED"
    assert event["reason"] == ""


def test_stats_sink_filled_and_optional(tmp_path, monkeypatch) -> None:
    symbols = [f"{i:06d}" for i in range(5)]
    sink: dict[str, float | str] = {}
    with_sink, _ = _stubbed_close_run(tmp_path / "with", monkeypatch, symbols, stats_sink=sink)
    without, _ = _stubbed_close_run(tmp_path / "without", monkeypatch, symbols)

    assert {"roster", "extras", "clients", "capacity_per_round", "demand_per_round", "status", "start_lag_s", "max_round_duration_s"} <= set(sink)
    assert sink["roster"] == 5 and sink["clients"] == 2 and sink["status"] == "OK"
    assert with_sink.status is without.status
    assert [(e.symbol, e.dataset, e.status) for e in with_sink.entries] == [(e.symbol, e.dataset, e.status) for e in without.entries]


def test_open_phase_records_nothing(tmp_path, monkeypatch) -> None:
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    class _Manifest:
        status = CaptureStatus.COMPLETE
        entries: tuple[Any, ...] = ()

    async def _open_run(snapshot_date: str, phase: str, profile: Any, stats_sink: Any = None) -> Any:
        return _Manifest()

    monkeypatch.setattr(auction_capture, "CollectionSettings", lambda: _profile(tmp_path))
    monkeypatch.setattr(auction_capture, "_run_async", _open_run)
    auction_capture.main(["--phase", "open", "--date", "2026-09-17"])

    assert _auction_close_events("2026-09-17") == []


def test_outcome_record_failure_is_non_fatal(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise OSError("events volume read-only")

    monkeypatch.setattr(auction_capture, "record_run_outcome", _raise)
    with caplog.at_level(logging.WARNING, logger=auction_capture.logger.name):
        manifest = _main_close_run(tmp_path, monkeypatch, [f"{i:06d}" for i in range(5)])

    assert manifest.status is CaptureStatus.COMPLETE
    warnings = [r for r in caplog.records if "OUTCOME_RECORD_FAILED" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING


def test_auction_close_outcome_is_silent(tmp_path) -> None:
    from src.tools.alert_policy import classify_run_outcome

    assert classify_run_outcome("auction_close", "DEGRADED", "deadline") is None
    assert classify_run_outcome("auction_close", "DEGRADED", "mixed") is None
    assert classify_run_outcome("auction_close", "OK", "") is None


def _plan_records(caplog: pytest.LogCaptureFixture) -> list[Any]:
    return [r for r in caplog.records if "stage=auction_capture_plan" in r.getMessage()]


def test_overcommit_logged_at_start(tmp_path, monkeypatch, caplog) -> None:
    import logging

    cases = [(449, logging.INFO, "OK", 449, 113), (1125, logging.INFO, "OK", 1125, 282), (1600, logging.WARNING, "OVERCOMMIT", 1600, 400)]
    for n, level, status, demand_orderbook, demand_program in cases:
        caplog.clear()
        with caplog.at_level(logging.INFO, logger="src.daily.auction_capture"):
            _stubbed_close_run(tmp_path / f"n{n}", monkeypatch, [f"{i:06d}" for i in range(n)])
        (record,) = _plan_records(caplog)
        message = record.getMessage()
        assert record.levelno == level
        assert f"status={status} " in message
        assert "capacity_per_round=1836" in message
        assert f"demand_orderbook={demand_orderbook} " in message
        assert f"demand_program_per_round={demand_program} " in message
        assert "start_lag_s=-60.0" in message  # 15:20 now, first round 15:21
        assert [r for r in caplog.records if r.levelno >= logging.WARNING and "OVERCOMMIT" in r.getMessage()] == (
            [record] if status == "OVERCOMMIT" else []
        )


def test_round_line_counts_match_entries(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from src.data.capture_contracts import CaptureStatus

    def _status_for(position: int, base: Any) -> Any:
        if position < 3:
            return base.model_copy(update={"status": CaptureStatus.PARTIAL, "reason": "deadline"})
        if position < 5:
            return base.model_copy(update={"status": CaptureStatus.FAILED, "reason": "TimeoutError", "first_event_time": base.scheduled_at, "last_event_time": base.scheduled_at})
        return base.model_copy(update={"status": CaptureStatus.COMPLETE, "rows": 1, "reason": "auction-close", "first_event_time": base.scheduled_at, "last_event_time": base.scheduled_at})

    with caplog.at_level(logging.INFO, logger="src.daily.auction_capture"):
        _stubbed_close_run(tmp_path, monkeypatch, [f"{i:06d}" for i in range(10)], status_for=_status_for)

    lines = [r.getMessage() for r in caplog.records if "stage=auction_capture_round" in r.getMessage()]
    orderbook = [line for line in lines if "dataset=ORDERBOOK" in line]
    program = [line for line in lines if "dataset=PROGRAM" in line]
    assert len(orderbook) == 9
    assert len(program) == 2
    for line in [*orderbook, *program]:
        fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
        assert fields["n_roster"] == "10"
        assert (fields["n_requested"], fields["n_complete"], fields["n_deadline"], fields["n_late"], fields["n_failed"]) == ("7", "5", "3", "0", "2")
        assert int(fields["n_complete"]) + int(fields["n_deadline"]) + int(fields["n_failed"]) == int(fields["n_roster"])
        assert fields["capacity_per_round"] == "1836"
        assert "n_deferred" not in fields
    assert [dict(p.split("=", 1) for p in line.split() if "=" in p)["index"] for line in orderbook] == [str(i) for i in range(9)]


def test_held_roster_failure_is_non_fatal(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster(("000003",), False, "OSError"))
    symbols = [f"{i:06d}" for i in range(1, 6)]
    with caplog.at_level(logging.INFO, logger="src.daily.auction_capture"):
        manifest, rosters = _stubbed_close_run(tmp_path, monkeypatch, symbols)

    assert manifest.status == CaptureStatus.COMPLETE
    warnings = [r for r in caplog.records if "HELD_ROSTER_UNAVAILABLE" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "reason=OSError" in warnings[0].getMessage()
    (plan,) = _plan_records(caplog)
    assert "n_priority=0 " in plan.getMessage()
    assert len(rosters) == 9


def test_held_roster_leads_every_round(tmp_path, monkeypatch) -> None:
    from src.daily import auction_capture

    monkeypatch.setattr(auction_capture, "load_held_roster", lambda: HeldRoster(("000004", "999999"), True, None))
    symbols = [f"{i:06d}" for i in range(1, 8)]
    _, rosters = _stubbed_close_run(tmp_path, monkeypatch, symbols)

    assert len(rosters) == 9
    assert all(order[0] == "000004" and sorted(order) == symbols for order in rosters)
    assert len({tuple(order) for order in rosters}) > 1


def test_audit_expectation_unchanged(tmp_path, monkeypatch) -> None:
    from src.tools.daily_audit import _audit_auction_sweeps

    symbols = [f"{i:06d}" for i in range(449)]
    manifest, rosters = _stubbed_close_run(tmp_path, monkeypatch, symbols)
    clock = _clock("2026-09-17")

    issues = _audit_auction_sweeps((manifest,), tuple(symbols), _profile(tmp_path), clock, clock.close_at)

    assert not [issue for issue in issues if "missing_entries" in issue]
    assert len(manifest.entries) == 449 * (len(rosters) + 2)


def _handoff_profile(tmp_path: Path, **overrides: Any):
    return _profile(tmp_path, COLLECTION_RESEARCH_SLOTS=("3", "4"), COLLECTION_AUCTION_HANDOFF_SLOTS=("5",), **overrides)


class _KeyedClient(_FakeClient):
    """Fake key that logs token issuance and orderbook requests with the virtual clock reading."""

    def __init__(
        self,
        key: str,
        state: dict[str, Any],
        log: list[tuple[Any, ...]],
        *,
        token_error: Exception | None = None,
        cached: bool = True,
    ) -> None:
        super().__init__(calls=[])
        self.key = key
        self._state = state
        self._log = log
        self._token_error = token_error
        self._cached = cached

    def adopt_cached_token(self) -> str | None:
        self._log.append(("token", self.key, self._state["now"]))
        if self._token_error is not None:
            raise self._token_error
        return "cached-token" if self._cached else None

    async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
        self._log.append(("orderbook", self.key, self._state["now"]))
        return await super().get_orderbook_snapshot(session, code, market_div_code)


def _publishing_sleeper(state: dict[str, Any], publish_at: dt.datetime | None, publish: Any):
    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)
        if publish_at is not None and not state.get("published") and state["now"] >= publish_at:
            publish()
            state["published"] = True

    return _sleep


_HANDOFF_SYMBOLS = [f"{i:06d}" for i in range(6)]


def test_pool_pick_uses_active_set_only() -> None:
    from src.daily.auction_capture import _ClientPool

    pool = _ClientPool(["b1", "b2"], ["h1"], ["DATA_5"], per_key=8)
    assert [pool.pick(i)[0] for i in range(4)] == ["b1", "b2", "b1", "b2"]

    pool.activate_handoff(0)
    pool.activate_handoff(0)
    assert pool.active_count() == 3
    assert [pool.pick(i)[0] for i in range(4)] == ["b1", "b2", "h1", "b1"]
    assert pool.pick(0)[1] is not pool.pick(1)[1]
    assert pool.pick(0)[1] is not pool.pick(2)[1]


def test_per_key_concurrency_bound_holds() -> None:
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    state = {"now": _seoul(2026, 9, 17, 15, 21, 0)}
    inflight: dict[str, int] = {"a": 0, "b": 0, "c": 0}
    peak: dict[str, int] = {"a": 0, "b": 0, "c": 0}

    class _Slow:
        def __init__(self, key: str, delay: float) -> None:
            self.key = key
            self.delay = delay

        async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
            inflight[self.key] += 1
            peak[self.key] = max(peak[self.key], inflight[self.key])
            await asyncio.sleep(self.delay)
            inflight[self.key] -= 1
            return {"rt_cd": "0"}

    pool = auction_capture._ClientPool(
        [_Slow("a", 0.02), _Slow("b", 0.001)], [_Slow("c", 0.001)], ["DATA_5"], per_key=8
    )
    pool.activate_handoff(0)
    roster = [f"{i:06d}" for i in range(100)]
    t0 = state["now"]

    async def _observe(position: int, symbol: str, client: Any) -> Any:
        await client.get_orderbook_snapshot(None, symbol)
        return _surge_entry(symbol, CaptureStatus.COMPLETE, t0)

    entries = _run(
        auction_capture._observe_roster(
            roster=roster,
            pool=pool,
            deadline=t0 + dt.timedelta(hours=1),
            now_fn=lambda: state["now"],
            observe=_observe,
            on_deadline=lambda position, symbol: _surge_entry(symbol, CaptureStatus.PARTIAL, t0),
        )
    )

    assert [e.symbol for e in entries] == roster
    assert all(e.status == CaptureStatus.COMPLETE for e in entries)
    assert peak == {"a": 8, "b": 8, "c": 8}


def test_handoff_not_activated_before_decision_manifest(tmp_path, caplog, monkeypatch) -> None:
    import logging

    from src.daily import auction_capture

    # Shared fake clock: a program sleep would advance time under the orderbook round; program lane is covered by its own sweep tests.
    monkeypatch.setattr(auction_capture, "program_rounds", lambda clock: [])
    caplog.set_level(logging.INFO, logger=auction_capture.logger.name)
    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", _HANDOFF_SYMBOLS, run_id="scan-1")
    state = {"now": _seoul(2026, 9, 17, 15, 20)}
    log: list[tuple[Any, ...]] = []
    base = [_KeyedClient("base0", state, log), _KeyedClient("base1", state, log)]
    handoff = [_KeyedClient("DATA_5", state, log)]

    _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_handoff_profile(tmp_path),
            store=store,
            clients=base,
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: state["now"],
            sleep_fn=_publishing_sleeper(state, None, None),
            handoff_clients=handoff,
            handoff_labels=("DATA_5",),
        )
    )

    assert not [event for event in log if event[0] == "token"]
    assert not [event for event in log if event[1] == "DATA_5"]
    assert "HANDOFF_ACTIVE" not in caplog.text


def test_handoff_activates_after_decision_manifest(tmp_path, caplog, monkeypatch) -> None:
    import logging

    from src.daily import auction_capture

    # Shared fake clock: a program sleep would advance time under the orderbook round; program lane is covered by its own sweep tests.
    monkeypatch.setattr(auction_capture, "program_rounds", lambda clock: [])
    caplog.set_level(logging.INFO, logger=auction_capture.logger.name)
    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", _HANDOFF_SYMBOLS, run_id="scan-1")
    clock = _clock("2026-09-17")
    publish_at = auction_capture.close_rounds(clock, 60)[2]
    state = {"now": _seoul(2026, 9, 17, 15, 20)}
    log: list[tuple[Any, ...]] = []
    base = [_KeyedClient("base0", state, log), _KeyedClient("base1", state, log)]
    handoff = [_KeyedClient("DATA_5", state, log)]

    _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_handoff_profile(tmp_path),
            store=store,
            clients=base,
            session_clock=clock,
            now_fn=lambda: state["now"],
            sleep_fn=_publishing_sleeper(
                state, publish_at, lambda: _publish_cohort(store, "2026-09-17", _HANDOFF_SYMBOLS, run_id="decision-1")
            ),
            handoff_clients=handoff,
            handoff_labels=("DATA_5",),
        )
    )

    tokens = [event for event in log if event[0] == "token"]
    handoff_requests = [event for event in log if event[0] == "orderbook" and event[1] == "DATA_5"]
    early_base = [event for event in log if event[0] == "orderbook" and event[2] < publish_at]
    assert [event[2] for event in tokens] == [publish_at]
    assert handoff_requests and all(event[2] >= publish_at for event in handoff_requests)
    assert early_base and all(event[1].startswith("base") for event in early_base)
    activations = [r.getMessage() for r in caplog.records if "HANDOFF_ACTIVE" in r.getMessage()]
    assert len(activations) == 1
    assert "active_keys=3" in activations[0]
    assert "slots=DATA_5" in activations[0]


def test_handoff_token_failure_is_non_fatal(tmp_path, caplog, monkeypatch) -> None:
    import logging

    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus
    # Shared fake clock: a program sleep would advance time under the orderbook round; program lane is covered by its own sweep tests.
    monkeypatch.setattr(auction_capture, "program_rounds", lambda clock: [])

    caplog.set_level(logging.INFO, logger=auction_capture.logger.name)
    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", _HANDOFF_SYMBOLS)
    state = {"now": _seoul(2026, 9, 17, 15, 20)}
    log: list[tuple[Any, ...]] = []
    base = [_KeyedClient("base0", state, log), _KeyedClient("base1", state, log)]
    handoff = [_KeyedClient("DATA_5", state, log, token_error=RuntimeError("upstream secret detail"))]

    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_handoff_profile(tmp_path),
            store=store,
            clients=base,
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: state["now"],
            sleep_fn=_publishing_sleeper(state, None, None),
            handoff_clients=handoff,
            handoff_labels=("DATA_5",),
        )
    )

    warnings = [r.getMessage() for r in caplog.records if "HANDOFF_FAILED" in r.getMessage()]
    assert warnings == ["[DATA] stage=auction_capture status=HANDOFF_FAILED slot=DATA_5 reason=RuntimeError"]
    assert "secret detail" not in caplog.text
    assert len([event for event in log if event[0] == "token"]) == 1
    assert not [event for event in log if event[0] == "orderbook" and event[1] == "DATA_5"]
    assert manifest.status is CaptureStatus.COMPLETE


def _run_async_stubs(monkeypatch, created: list[Any], captured: dict[str, Any]) -> None:
    from src.daily import auction_capture

    class _Cred:
        slot = "DATA_3"
        app_key = "key-research"
        app_secret = "sec"
        hts_id = "hts"

    class _Client:
        def __init__(self, app_key: str, app_secret: str, account_id: str, hts_id: str, token_file: str | None = None) -> None:
            created.append(app_key)
            self.app_key = app_key

        def create_session(self) -> _Session:
            return _Session()

        async def ensure_token(self, session: Any) -> None:
            return None

    async def _capture(*args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "manifest-ok"

    async def _open(client: Any, session: Any, date: Any) -> bool:
        return True

    monkeypatch.setattr(auction_capture, "resolve_research_credentials", lambda env, *, slots: (_Cred(),))
    monkeypatch.setattr("src.api.kis.client.KisApiClient", _Client)
    monkeypatch.setattr(auction_capture, "run_auction_capture", _capture)
    monkeypatch.setattr(auction_capture, "is_kis_trading_day", _open)


def test_handoff_config_error_is_non_fatal(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from src.daily import auction_capture

    caplog.set_level(logging.WARNING, logger=auction_capture.logger.name)
    created: list[str] = []
    captured: dict[str, Any] = {}
    _run_async_stubs(monkeypatch, created, captured)
    monkeypatch.delenv("KIS_DATA_SLOTS", raising=False)

    out = asyncio.run(
        auction_capture._run_async("2026-09-17", "close", _handoff_profile(tmp_path))
    )

    assert out == "manifest-ok"
    assert [c.app_key for c in captured["clients"]] == ["key-research"]
    assert captured["handoff_clients"] == []
    warnings = [r.getMessage() for r in caplog.records if "HANDOFF_DISABLED" in r.getMessage()]
    assert warnings == ["[DATA] stage=auction_capture status=HANDOFF_DISABLED reason=config"]


def test_handoff_wiring_builds_borrowed_clients(tmp_path, monkeypatch) -> None:
    from src.daily import auction_capture

    created: list[str] = []
    captured: dict[str, Any] = {}
    _run_async_stubs(monkeypatch, created, captured)
    monkeypatch.setenv("KIS_DATA_SLOTS", "1,2,3,4,5")
    monkeypatch.setenv("KIS_DECISION_SHARD_SLOTS", "1,5")
    monkeypatch.setenv("KIS_DATA_1_APP_KEY", "key-1")
    monkeypatch.setenv("KIS_DATA_1_APP_SECRET", "sec-1")
    monkeypatch.setenv("KIS_DATA_5_APP_KEY", "key-5")
    monkeypatch.setenv("KIS_DATA_5_APP_SECRET", "sec-5")

    asyncio.run(auction_capture._run_async("2026-09-17", "close", _handoff_profile(tmp_path)))

    assert [c.app_key for c in captured["handoff_clients"]] == ["key-5"]
    assert captured["handoff_labels"] == ("DATA_5",)
    assert created == ["key-research", "key-5"]


def test_handoff_disabled_by_default(tmp_path, monkeypatch) -> None:
    from src.daily import auction_capture

    created: list[str] = []
    captured: dict[str, Any] = {}
    _run_async_stubs(monkeypatch, created, captured)
    monkeypatch.setenv("KIS_DATA_SLOTS", "1,2,3,4,5")
    monkeypatch.setenv("KIS_DECISION_SHARD_SLOTS", "1,5")
    monkeypatch.setenv("KIS_DATA_5_APP_KEY", "key-5")
    monkeypatch.setenv("KIS_DATA_5_APP_SECRET", "sec-5")

    asyncio.run(auction_capture._run_async("2026-09-17", "close", _profile(tmp_path)))

    assert created == ["key-research"]
    assert captured["handoff_clients"] == []
    assert captured["handoff_labels"] == ()


def test_capacity_gain_visible(tmp_path, caplog, monkeypatch) -> None:
    import logging

    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    caplog.set_level(logging.INFO, logger=auction_capture.logger.name)
    # The fake clock is shared by both lanes, so a program sleep would move time under round 0; this test isolates the orderbook lane.
    monkeypatch.setattr(auction_capture, "program_rounds", lambda clock: [])
    symbols = [f"{i:06d}" for i in range(1600)]
    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", symbols, run_id="scan-1")
    clock = _clock("2026-09-17")
    rounds = auction_capture.close_rounds(clock, 60)
    state = {"now": _seoul(2026, 9, 17, 15, 20)}
    free: dict[str, dt.datetime] = {}
    served: list[tuple[str, dt.datetime, dt.datetime]] = []
    service = dt.timedelta(seconds=1 / 18)

    class _TimedKey(_KeyedClient):
        async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
            issued = state["now"]
            start = max(issued, free.get(self.key, issued))
            free[self.key] = start + service
            served.append((self.key, issued, free[self.key]))
            return await super().get_orderbook_snapshot(session, code, market_div_code)

    log: list[tuple[Any, ...]] = []
    base = [_TimedKey("base0", state, log), _TimedKey("base1", state, log)]
    handoff = [_TimedKey("DATA_5", state, log)]
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_handoff_profile(tmp_path),
            store=store,
            clients=base,
            session_clock=clock,
            now_fn=lambda: state["now"],
            sleep_fn=_publishing_sleeper(
                state, rounds[1], lambda: _publish_cohort(store, "2026-09-17", symbols, run_id="decision-1")
            ),
            handoff_clients=handoff,
            handoff_labels=("DATA_5",),
        )
    )

    assert "capacity_with_handoff=2754" in caplog.text
    orderbook = [e for e in manifest.entries if e.dataset == CaptureDataset.ORDERBOOK]
    for moment in rounds:
        done = [e for e in orderbook if e.scheduled_at == moment and e.status is CaptureStatus.COMPLETE]
        assert len(done) == len(symbols)
    by_round = {moment: [(key, finish) for key, issued, finish in served if issued == moment] for moment in rounds}
    assert {key for key, _ in by_round[rounds[0]]} == {"base0", "base1"}
    assert {key for key, _ in by_round[rounds[1]]} == {"base0", "base1", "DATA_5"}
    durations = [max(f for _, f in by_round[moment]) - moment for moment in rounds[:2]]
    assert durations[1] < durations[0]
    assert all(
        max(f for _, f in by_round[moment]) - moment <= dt.timedelta(seconds=60) for moment in rounds[1:]
    )


def test_handoff_waits_while_manifest_store_unreadable(caplog) -> None:
    import logging

    from src.daily import auction_capture

    caplog.set_level(logging.WARNING, logger=auction_capture.logger.name)

    class _UnreadableStore:
        def read_manifests(self, snapshot_date: str) -> tuple[Any, ...]:
            raise OSError("evidence store offline")

    calls: list[Any] = []

    class _Handoff:
        def adopt_cached_token(self) -> str | None:
            calls.append("adopt")
            return "cached-token"

    pool = auction_capture._ClientPool([], [_Handoff()], ["DATA_5"], per_key=8)
    activated = _run(
        auction_capture._maybe_activate_handoff(
            pool, _UnreadableStore(), "2026-09-17", round_at=_seoul(2026, 9, 17, 15, 21)
        )
    )

    assert activated is False
    assert calls == []
    assert "HANDOFF_WAIT reason=OSError" in caplog.text


def test_handoff_waits_without_decision_manifest() -> None:
    from src.daily import auction_capture

    class _EmptyStore:
        def read_manifests(self, snapshot_date: str) -> tuple[Any, ...]:
            return ()

    calls: list[Any] = []

    class _Handoff:
        def adopt_cached_token(self) -> str | None:
            calls.append("adopt")
            return "cached-token"

    pool = auction_capture._ClientPool([], [_Handoff()], ["DATA_5"], per_key=8)
    activated = _run(
        auction_capture._maybe_activate_handoff(
            pool, _EmptyStore(), "2026-09-17", round_at=_seoul(2026, 9, 17, 15, 21)
        )
    )

    assert activated is False
    assert calls == []
    assert pool.decision_published is False


def test_handoff_never_issues_token(tmp_path, caplog, monkeypatch) -> None:
    import logging

    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureStatus

    caplog.set_level(logging.INFO, logger=auction_capture.logger.name)
    monkeypatch.setattr(auction_capture, "program_rounds", lambda clock: [])
    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", _HANDOFF_SYMBOLS)
    state = {"now": _seoul(2026, 9, 17, 15, 20)}
    log: list[tuple[Any, ...]] = []
    issued: list[str] = []

    class _EmptyCacheKey(_KeyedClient):
        async def ensure_token(self, session: Any) -> None:
            issued.append(self.key)

    base = [_KeyedClient("base0", state, log), _KeyedClient("base1", state, log)]
    handoff = [_EmptyCacheKey("DATA_5", state, log, cached=False)]
    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_handoff_profile(tmp_path),
            store=store,
            clients=base,
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: state["now"],
            sleep_fn=_publishing_sleeper(state, None, None),
            handoff_clients=handoff,
            handoff_labels=("DATA_5",),
        )
    )

    warnings = [r.getMessage() for r in caplog.records if "HANDOFF_FAILED" in r.getMessage()]
    assert warnings == ["[DATA] stage=auction_capture status=HANDOFF_FAILED slot=DATA_5 reason=no_cached_token"]
    assert issued == []
    assert not [event for event in log if event[0] == "orderbook" and event[1] == "DATA_5"]
    assert "HANDOFF_ACTIVE" not in caplog.text
    assert manifest.status is CaptureStatus.COMPLETE


def test_observe_roster_requires_pool_or_clients() -> None:
    from src.daily import auction_capture

    with pytest.raises(ValueError, match="client pool"):
        _run(
            auction_capture._observe_roster(
                roster=["000001"],
                deadline=_seoul(2026, 9, 17, 15, 21),
                now_fn=lambda: _seoul(2026, 9, 17, 15, 20),
                observe=lambda position, symbol, client: None,
                on_deadline=lambda position, symbol: None,
            )
        )


def test_run_auction_capture_rejects_misaligned_or_undeclared_handoff(tmp_path) -> None:
    from src.daily import auction_capture

    common = {
        "phase": "close",
        "store": _store(tmp_path),
        "clients": [_FakeClient()],
        "session_clock": _clock("2026-09-17"),
        "now_fn": lambda: _seoul(2026, 9, 17, 15, 20),
    }
    with pytest.raises(ValueError, match="must align"):
        _run(
            auction_capture.run_auction_capture(
                "2026-09-17", profile=_handoff_profile(tmp_path), handoff_clients=[_FakeClient()], handoff_labels=(), **common
            )
        )
    with pytest.raises(ValueError, match="declared handoff slots"):
        _run(
            auction_capture.run_auction_capture(
                "2026-09-17", profile=_profile(tmp_path), handoff_clients=[_FakeClient()], handoff_labels=("DATA_5",), **common
            )
        )



def _declare_cohort(store: Any, snapshot_date: str, eligible: list[str], *, declared_at: dt.datetime | None = None) -> Any:
    from src.data.capture_contracts import build_cohort

    cohort = build_cohort(
        dt.date.fromisoformat(snapshot_date), eligible, eligible, {}, eligibility_rule_version="v1"
    )
    store.publish_cohort_declaration(
        cohort, declared_at=declared_at or _seoul(2026, 9, 17, 15, 20, 10), retain_days=40
    )
    return cohort


def _virtual_clock(start: dt.datetime) -> tuple[dict[str, Any], Any]:
    state: dict[str, Any] = {"now": start, "sleeps": []}

    async def _sleep(seconds: float) -> None:
        state["sleeps"].append(seconds)
        state["now"] += dt.timedelta(seconds=seconds)

    return state, _sleep


def test_start_before_collect_end_uses_declaration(tmp_path) -> None:
    """Only the early declaration exists: the capture proceeds on the declared population and stamps its id."""
    from src.daily import auction_capture

    store = _store(tmp_path)
    cohort = _declare_cohort(store, "2026-09-17", ["000001", "000002"])
    state, sleep = _virtual_clock(_seoul(2026, 9, 17, 15, 20, 20))
    client = _FakeClient(calls=[])
    stats: dict[str, float | str] = {}

    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_profile(tmp_path),
            store=store,
            clients=[client],
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: state["now"],
            sleep_fn=sleep,
            stats_sink=stats,
        )
    )

    assert stats["cohort_source"] == "declared"
    assert manifest.context.cohort_id == cohort.cohort_id
    assert {code for kind, code in client.calls if kind == "orderbook"} == {"000001", "000002"}


def test_decision_manifest_preferred_and_declaration_unread(tmp_path, monkeypatch) -> None:
    from src.daily import auction_capture

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    _declare_cohort(store, "2026-09-17", ["000001", "000009"])

    def _must_not_read(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("declaration read although a decision manifest exists")

    monkeypatch.setattr(store, "read_declared_cohort", _must_not_read)
    stats: dict[str, float | str] = {}
    _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_profile(tmp_path),
            store=store,
            clients=[_FakeClient(calls=[])],
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: _seoul(2026, 9, 17, 15, 20, 20),
            stats_sink=stats,
        )
    )

    assert stats["cohort_source"] == "decision"


def test_declaration_alone_does_not_activate_handoff(tmp_path) -> None:
    from src.daily import auction_capture

    store = _store(tmp_path)
    _declare_cohort(store, "2026-09-17", ["000001"])

    assert auction_capture._decision_manifest_published(store, "2026-09-17") is False


def test_input_wait_is_bounded_by_deadline(tmp_path) -> None:
    from src.daily import auction_capture

    store = _store(tmp_path)
    state, sleep = _virtual_clock(_seoul(2026, 9, 17, 15, 20, 10))

    with pytest.raises(RuntimeError, match="cannot be verified"):
        _run(
            auction_capture.run_auction_capture(
                "2026-09-17",
                phase="close",
                profile=_profile(tmp_path),
                store=store,
                clients=[_FakeClient(calls=[])],
                session_clock=_clock("2026-09-17"),
                now_fn=lambda: state["now"],
                sleep_fn=sleep,
            )
        )

    assert state["now"] == _seoul(2026, 9, 17, 15, 20, 45)  # start + COLLECTION_AUCTION_INPUT_WAIT_SECONDS (35 s)
    assert all(seconds <= 0.5 for seconds in state["sleeps"])


def test_input_wait_returns_early_and_rounds_keep_schedule(tmp_path) -> None:
    from src.daily import auction_capture

    store = _store(tmp_path)
    state, base_sleep = _virtual_clock(_seoul(2026, 9, 17, 15, 20, 20))
    _publish_research_roster(store, "2026-09-17", ["000003"], completed_at=_seoul(2026, 9, 17, 15, 0))
    polls = {"n": 0}

    async def _sleep(seconds: float) -> None:
        await base_sleep(seconds)
        polls["n"] += 1
        if polls["n"] == 3:
            _declare_cohort(store, "2026-09-17", ["000001"], declared_at=state["now"])

    observed: list[dt.datetime] = []

    class _TimedClient(_FakeClient):
        async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None):
            observed.append(state["now"])
            return await super().get_orderbook_snapshot(session, code, market_div_code)

    stats: dict[str, float | str] = {}
    _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_profile(tmp_path),
            store=store,
            clients=[_TimedClient(calls=[])],
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: state["now"],
            sleep_fn=_sleep,
            stats_sink=stats,
        )
    )

    assert stats["inputs_wait_s"] == pytest.approx(1.5)
    assert min(observed) >= _seoul(2026, 9, 17, 15, 21, 0)


def test_research_roster_gates_the_input_wait(tmp_path) -> None:
    from src.daily import auction_capture

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    state, sleep = _virtual_clock(_seoul(2026, 9, 17, 15, 20, 20))

    asyncio.run(
        auction_capture._await_auction_inputs(
            store,
            "2026-09-17",
            deadline=_seoul(2026, 9, 17, 15, 20, 30),
            now_fn=lambda: state["now"],
            sleeper=sleep,
        )
    )

    assert state["now"] == _seoul(2026, 9, 17, 15, 20, 30)


def test_round0_full_window_when_started_early(tmp_path, monkeypatch) -> None:
    """Start at 15:20:40 with 1,125 symbols: round 0 dispatches everything inside its 60 s window.

    The program lane is suppressed because its own sleep would share the single virtual clock and
    advance it past round 0; the lane's scheduling is covered by its own tests.
    """
    from src.api.kis.client import KIS_REST_TPS_PER_APP_KEY
    from src.daily import auction_capture
    from src.data.capture_contracts import CoverageEntry

    monkeypatch.setattr(auction_capture, "program_rounds", lambda clock: [])
    symbols = [f"{i:06d}" for i in range(1, 1126)]
    store = _store(tmp_path)
    _declare_cohort(store, "2026-09-17", symbols)
    state, sleep = _virtual_clock(_seoul(2026, 9, 17, 15, 20, 40))
    clients = [_FakeClient(calls=[]), _FakeClient(calls=[])]
    assert auction_capture.estimate_round_capacity(
        len(clients), 60, utilization=0.85, tps_per_key=KIS_REST_TPS_PER_APP_KEY
    ) >= len(symbols)

    manifest = _run(
        auction_capture.run_auction_capture(
            "2026-09-17",
            phase="close",
            profile=_profile(tmp_path),
            store=store,
            clients=clients,
            session_clock=_clock("2026-09-17"),
            now_fn=lambda: state["now"],
            sleep_fn=sleep,
        )
    )

    round0 = _seoul(2026, 9, 17, 15, 21, 0)
    round0_entries = [e for e in manifest.entries if isinstance(e, CoverageEntry) and e.scheduled_at == round0]
    assert round0_entries
    assert all(e.reason != "deadline" for e in round0_entries)
    assert sum(1 for kind, _ in clients[0].calls + clients[1].calls if kind == "orderbook") >= len(symbols)


def test_declared_and_decision_cohort_ids_identical(tmp_path) -> None:
    """Same scan inputs yield the same content-hash identity, so the later decision manifest cannot diverge."""
    from src.data.capture_contracts import CaptureContext, CaptureDataset, CaptureManifest, CaptureStatus, build_cohort

    store = _store(tmp_path)
    eligible = ["000001", "000002"]
    declared = _declare_cohort(store, "2026-09-17", eligible)
    decision = build_cohort(dt.date(2026, 9, 17), eligible, eligible, {}, eligibility_rule_version="v1")
    store.publish_manifest(
        CaptureManifest(
            schema_version=1,
            context=CaptureContext(
                trading_date=dt.date(2026, 9, 17),
                run_id="decision-2026-09-17-abcd1234",
                dataset=CaptureDataset.SCAN,
                vendor="owner-local",
                endpoint="decision-input",
                symbol=None,
                venue="KRX",
                session="regular",
                capture_reason="decision-input",
                cohort_id=decision.cohort_id,
                scheduled_at=None,
            ),
            cohort=decision,
            completed_at=_seoul(2026, 9, 17, 15, 21, 0),
            entries=(),
            artifacts=(),
            status=CaptureStatus.COMPLETE,
        )
    )

    assert decision.cohort_id == declared.cohort_id
    assert store.read_cohort("2026-09-17", available_by=_seoul(2026, 9, 17, 16, 0)).cohort_id == declared.cohort_id


def test_research_gate_opens_when_roster_is_readable(tmp_path, monkeypatch) -> None:
    from src.daily import auction_capture

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-17", ["000001"])
    state, sleep = _virtual_clock(_seoul(2026, 9, 17, 15, 20, 20))
    monkeypatch.setattr(auction_capture, "read_research_roster", lambda *a, **k: object())

    asyncio.run(
        auction_capture._await_auction_inputs(
            store,
            "2026-09-17",
            deadline=_seoul(2026, 9, 17, 15, 20, 30),
            now_fn=lambda: state["now"],
            sleeper=sleep,
        )
    )

    assert state["sleeps"] == []
