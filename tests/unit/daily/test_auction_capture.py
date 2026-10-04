"""Auction capture invariant guards."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

from src.execution.paper_broker import HeldRoster


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
        "COLLECTION_AUCTION_ENABLED": True,
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


def _publish_cohort(store: Any, snapshot_date: str, eligible: list[str]) -> Any:
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
            run_id="decision-1",
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
    assert by_symbol["000001"].status == CaptureStatus.PARTIAL and by_symbol["000001"].reason == "open_unresolved"
    assert by_symbol["000002"].status == CaptureStatus.COMPLETE and by_symbol["000003"].status == CaptureStatus.COMPLETE
    assert manifest.status == CaptureStatus.PARTIAL


def test_unresolved_open_stays_partial(tmp_path, monkeypatch) -> None:
    """Never-appearing open retains explicit unresolved state."""
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
    disabled = _profile(tmp_path).model_copy(update={"COLLECTION_AUCTION_ENABLED": False})
    with pytest.raises(ValueError, match="enabled"):
        _run(auction_capture.run_auction_capture("2026-09-17", phase="close", profile=disabled, store=store, clients=[client], session_clock=clock, now_fn=lambda: now))
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
    """CLI skips disabled/holiday and runs configured phases."""
    import logging

    from src.daily import auction_capture

    monkeypatch.setattr(auction_capture, "CollectionSettings", lambda: _profile(tmp_path, COLLECTION_AUCTION_ENABLED=False))
    with caplog.at_level(logging.INFO, logger=auction_capture.logger.name):
        auction_capture.main(["--phase", "close"])
    assert any("SKIP" in r.message for r in caplog.records)
    caplog.clear()
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

    async def _fake_async(snapshot_date: str, phase: str, profile: Any) -> Any:
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


def test_module_execution_actually_invokes_main(monkeypatch, caplog) -> None:
    """실측: 2026-09-21 __main__ 가드 자체가 파일에 없어 `python -m ...`로 실행해도
    main()이 전혀 호출되지 않고 조용히 성공 종료했다(kca-auction-open/close가 매번
    "성공"으로 보이면서 실제로는 아무 것도 수집하지 않음). main()을 직접 호출하는
    테스트만으로는 이 가드 누락을 잡지 못하므로, 모듈을 __main__으로 실행해 확인한다.
    conftest의 autouse 픽스처가 COLLECTION_AUCTION_ENABLED를 지우므로 main()이 실제로
    호출됐다면 SKIP 로그가 결정론적으로 찍힌다.
    """
    import logging
    import runpy
    import sys

    monkeypatch.setattr(sys, "argv", ["auction_capture", "--phase", "open"])

    # __main__으로 재실행되는 코드는 __name__="__main__"이라 별도 logger 인스턴스를
    # 얻으므로, 특정 logger name이 아니라 루트 레벨로 캡처해야 한다.
    with caplog.at_level(logging.INFO):
        runpy.run_module("src.daily.auction_capture", run_name="__main__")

    assert any("SKIP" in r.message and "disabled" in r.message for r in caplog.records)


def test_run_async_builds_clients(tmp_path, monkeypatch) -> None:
    """Research clients use token cache and host pacing contracts."""
    import os

    from src.daily import auction_capture

    profile = _profile(tmp_path)
    seen: dict[str, Any] = {}

    class _Cred:
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

    async def _holiday(snapshot_date: str, phase: str, profile: Any) -> None:
        return None

    monkeypatch.setattr(auction_capture, "CollectionSettings", lambda: _profile(tmp_path))
    monkeypatch.setattr(auction_capture, "_run_async", _holiday)
    with caplog.at_level(logging.INFO, logger=auction_capture.logger.name):
        auction_capture.main(["--phase", "close", "--date", "2026-09-24"])
    assert any("reason=non_trading_day" in r.message for r in caplog.records)


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


def test_run_auction_capture_rejects_disabled_auction(tmp_path) -> None:
    """A profile without the auction opt-in fails closed before any vendor call."""
    import pytest

    from src.daily import auction_capture

    store = _store(tmp_path)
    profile = _profile(tmp_path, COLLECTION_AUCTION_ENABLED=False)
    clock = _clock("2026-09-17")
    with pytest.raises(ValueError, match="enabled auction collection"):
        _run(
            auction_capture.run_auction_capture(
                "2026-09-17",
                phase="close",
                profile=profile,
                store=store,
                clients=[],
                session_clock=clock,
            )
        )


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
