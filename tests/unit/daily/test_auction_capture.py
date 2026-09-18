"""Auction capture invariant guards."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any


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
        "COLLECTION_RAW_ENABLED": True,
        "COLLECTION_AUCTION_ENABLED": True,
        "COLLECTION_RESEARCH_SLOTS": ("5",),
        "COLLECTION_KEY_OWNERSHIP_PATH": tmp_path / "ownership.json",
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
    rounds = auction_capture._close_rounds(clock, 60)
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
        for round_at in auction_capture._close_rounds(clock, 60)
    } | {
        ("program", round_at)
        for round_at in auction_capture._program_rounds(clock)
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
    monkeypatch.setattr(auction_capture, "_open_position_symbols", lambda: [])
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
    monkeypatch.setattr(auction_capture, "_open_position_symbols", lambda: ["000003"])
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
    rounds = auction_capture._close_rounds(clock, 60)
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
    assert len(orderbook_calls) < 2 * len(auction_capture._close_rounds(clock, 60))
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
    monkeypatch.setattr(auction_capture, "_open_position_symbols", lambda: [])
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


def test_unresolved_open_stays_partial(tmp_path, monkeypatch) -> None:
    """Never-appearing open retains explicit unresolved state."""
    from src.daily import auction_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-16", ["000001"])
    monkeypatch.setattr(auction_capture, "_open_position_symbols", lambda: [])
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
    monkeypatch.setattr(auction_capture, "_open_position_symbols", lambda: ["000002"])
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
    monkeypatch.setattr(auction_capture, "_open_position_symbols", lambda: [])
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
    monkeypatch.setattr(auction_capture, "_open_position_symbols", lambda: [])
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
    """Root override and position reader fallbacks behave."""
    import pandas as pd

    from src import settings as app_settings
    from src.daily import auction_capture

    explicit = _profile(tmp_path, COLLECTION_ROOT=tmp_path / "explicit")
    assert auction_capture._capture_root(explicit) == tmp_path / "explicit"
    monkeypatch.setattr(app_settings, "HISTORY_DIR", tmp_path, raising=False)
    implicit = _profile(tmp_path, COLLECTION_ROOT=None)
    assert auction_capture._capture_root(implicit) == tmp_path / "capture"
    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert auction_capture._open_position_symbols() == []

    class _Ledger:
        def load_open_positions(self) -> Any:
            return pd.DataFrame({"symbol": ["000001", " ", "000001"]})

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", lambda *a, **k: _Ledger())
    assert auction_capture._open_position_symbols() == ["000001"]

    class _EmptyLedger:
        def load_open_positions(self) -> Any:
            return pd.DataFrame()

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", lambda *a, **k: _EmptyLedger())
    assert auction_capture._open_position_symbols() == []
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
    assert len(auction_capture._program_rounds(_clock("2026-09-17"))) == 2


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


def test_run_async_builds_clients(tmp_path, monkeypatch) -> None:
    """Research clients use token cache and host pacing contracts."""
    import os

    from src.daily import auction_capture

    ownership = tmp_path / "ownership.json"
    ownership.write_text("{}", encoding="utf-8")
    profile = _profile(tmp_path, COLLECTION_KEY_OWNERSHIP_PATH=ownership)
    seen: dict[str, Any] = {}

    class _Cred:
        app_key = "key5"
        app_secret = "sec5"
        hts_id = "hts5"

    monkeypatch.setattr(auction_capture, "resolve_research_credentials", lambda env, *, slots, ownership_path: (_Cred(),))
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
    out = asyncio.run(auction_capture._run_async("2026-09-17", "close", profile))
    assert out == "manifest-ok"
    assert seen["warmed"] is True
    assert created and "token_" in str(created[0][1])
    assert os.environ["KIS_DATA_SLOTS"] == "5"

    bad_profile = _profile(tmp_path, COLLECTION_AUCTION_ENABLED=False, COLLECTION_KEY_OWNERSHIP_PATH=None)
    try:
        asyncio.run(auction_capture._run_async("2026-09-17", "close", bad_profile))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
