"""Invariant guards for certified tape-day publication."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.api.kiwoom.client import TapeDayCertificate
from src.backfill.intraday import tape_harvest as th
from src.backfill.intraday.tape_harvest import (
    TAPE_SESSIONS,
    TapeDayResult,
    TickTapePublisher,
    harvest_symbol_tape,
)
from src.config.collection import CollectionSettings
from src.data.capture_contracts import CaptureStatus
from src.data.capture_store import CaptureStore

_SEOUL = ZoneInfo("Asia/Seoul")
_DAY = "2020-01-02"
_DAY2 = "2020-01-03"


def _tick(hms: str, price: str = "10000", qty: str = "10") -> dict:
    return {"cntr_tm": f"{_DAY.replace('-', '')}{hms}", "cur_prc": price, "trde_qty": qty}


def _tick_on(day: str, hms: str) -> dict:
    return {"cntr_tm": f"{day.replace('-', '')}{hms}", "cur_prc": "10000", "trde_qty": "10"}


def _cert(day: str, received: int, total: int | None, complete: bool) -> TapeDayCertificate:
    return TapeDayCertificate(day=day, received=received, vendor_total=total, complete=complete)


def _profile(tmp_path, routes=None, **overrides: Any) -> CollectionSettings:
    base = {"COLLECTION_ROOT": tmp_path / "capture"}
    base["COLLECTION_VERIFIED_CHART_ROUTES"] = dict(
        routes if routes is not None else {"kiwoom:ka10079": "KRX", "kiwoom:ka10079-nx": "NXT"}
    )
    base.update(overrides)
    return CollectionSettings(**base)


class _StubTape:
    def __init__(
        self,
        pages: list[tuple[list[dict], dict]],
        delivered: list[tuple[str, list[dict], TapeDayCertificate]],
        certificates: list[TapeDayCertificate],
        termination: str = "crossed_stop_day",
    ) -> None:
        self.pages = pages
        self.delivered = delivered
        self.certificates = certificates
        self.termination = termination
        self.calls = 0
        self.seen: dict[str, Any] = {}

    async def walk_tick_tape(self, session, code, **kwargs: Any) -> dict:
        self.calls += 1
        self.seen = dict(kwargs)
        now = datetime.now(_SEOUL)
        on_page = kwargs.get("on_page")
        if on_page is not None:
            for i, (rows, meta) in enumerate(self.pages):
                on_page({"return_code": 0, "stk_tic_chart_qry": rows}, meta, now, now, i, 0)
        on_day = kwargs.get("on_day_complete")
        if on_day is not None:
            for day, rows, cert in self.delivered:
                on_day(day, rows, cert)
        return {
            "rt_cd": "0", "termination_reason": self.termination,
            "pages_fetched": len(self.pages), "certificates": list(self.certificates),
        }


def _krx_sessions() -> list:
    return [s for s in TAPE_SESSIONS if s.venue == "KRX"]


def test_certified_day_yields_complete_per_session(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    rows = [_tick("093000"), _tick("170000")]
    stub = _StubTape(
        [([[dict(r) for r in rows], {"cont-yn": "Y", "next-key": "k"}])][0:1],
        [(_DAY, rows, _cert(_DAY, 2, 3, True))],
        [_cert(_DAY, 2, 3, True)],
    )
    out: list[TapeDayResult] = []
    res = asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-001", profile=profile, on_result=out.append,
        )
    )
    assert res.unresolved_days == ()
    assert len(out) == 2
    by_session = {r.session: r for r in out}
    assert by_session["regular"].entry.status == CaptureStatus.COMPLETE
    assert by_session["krx_aftermarket"].entry.status == CaptureStatus.COMPLETE
    assert len(by_session["regular"].frame) == 1
    assert len(by_session["krx_aftermarket"].frame) == 1
    for r in out:
        assert len(r.entry.raw_refs) >= 1
        assert "vendor_total=3" in r.entry.reason
    assert stub.seen["venue"] == "KRX"


def test_out_of_window_rows_staged_not_written(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    rows = [_tick("085000"), _tick("153500")]
    stub = _StubTape(
        [(rows, {"cont-yn": "Y", "next-key": "k"})],
        [(_DAY, rows, _cert(_DAY, 2, 3, True))],
        [_cert(_DAY, 2, 3, True)],
    )
    out: list[TapeDayResult] = []
    asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-002", profile=profile, on_result=out.append,
        )
    )
    for r in out:
        assert r.frame.empty
        assert len(r.entry.raw_refs) >= 2


def test_uncertified_day_is_partial_and_writes_nothing(tmp_path, monkeypatch) -> None:
    from src import settings as _settings

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    stub = _StubTape(
        [([_tick("093000")], {"cont-yn": "Y", "next-key": "k"})],
        [], [_cert(_DAY, 1, 10, False)],
    )
    out: list[TapeDayResult] = []
    res = asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-003", profile=profile, on_result=out.append,
        )
    )
    assert res.unresolved_days == (_DAY,)
    assert all(r.entry.status == CaptureStatus.PARTIAL for r in out)
    assert all(r.frame.empty for r in out)
    assert "tape_total_mismatch:received=1:total=10" in out[0].entry.reason
    pub = TickTapePublisher(store=store, profile=profile, flush_rows=10**9)
    for r in out:
        pub.add(r)
    report = pub.flush()
    assert report.partitions_written == 0 and report.rows_written == 0


def test_day_not_on_tape_is_unknown(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    stub = _StubTape([([], {"cont-yn": "N", "next-key": ""})], [], [])
    out: list[TapeDayResult] = []
    res = asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-004", profile=profile, on_result=out.append,
        )
    )
    assert res.unresolved_days == (_DAY,)
    assert all(r.entry.status == CaptureStatus.UNKNOWN for r in out)
    assert all(r.entry.reason == "day_not_on_tape" for r in out)


def test_certified_empty_window_is_no_trades(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    rows = [_tick("093000")]
    stub = _StubTape(
        [(rows, {"cont-yn": "Y", "next-key": "k"})],
        [(_DAY, rows, _cert(_DAY, 1, 2, True))],
        [_cert(_DAY, 1, 2, True)],
    )
    out: list[TapeDayResult] = []
    asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-005", profile=profile, on_result=out.append,
        )
    )
    by_session = {r.session: r for r in out}
    assert by_session["regular"].entry.status == CaptureStatus.COMPLETE
    assert by_session["krx_aftermarket"].entry.status == CaptureStatus.NO_TRADES


def test_nxt_tape_maps_to_nxt_aftermarket(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    rows = [_tick("160000")]
    stub = _StubTape(
        [(rows, {"cont-yn": "Y", "next-key": "k"})],
        [(_DAY, rows, _cert(_DAY, 1, 2, True))],
        [_cert(_DAY, 1, 2, True)],
    )
    out: list[TapeDayResult] = []
    asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="NXT", sessions=list(TAPE_SESSIONS),
            store=store, run_id="tape-2020-01-04-006", profile=profile, on_result=out.append,
        )
    )
    assert len(out) == 1 and out[0].session == "nxt_aftermarket"
    assert out[0].entry.venue == "NXT"
    assert stub.seen["venue"] == "NXT"


def test_batched_publication_rewrites_per_flush(tmp_path, monkeypatch) -> None:
    from src import settings as _settings

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    sessions = [s for s in TAPE_SESSIONS if s.session == "regular"]
    pub = TickTapePublisher(store=store, profile=profile, flush_rows=5)
    calls = {"n": 0}
    real_write = th.write_tick_partition
    flushed: list = []
    real_flush = pub.flush

    def _tracking() -> Any:
        report = real_flush()
        flushed.append(report)
        return report

    def _counting(df: pd.DataFrame, day: str, session: str, **kwargs: Any) -> int:
        calls["n"] += 1
        return real_write(df, day, session, **kwargs)

    monkeypatch.setattr(th, "write_tick_partition", _counting)
    monkeypatch.setattr(pub, "flush", _tracking)
    for symbol in ("005930", "000660", "035420"):
        rows = [_tick_on(_DAY, "093000"), _tick_on(_DAY, "093100"), _tick_on(_DAY2, "093000"), _tick_on(_DAY2, "093100")]
        stub = _StubTape(
            [(rows, {"cont-yn": "Y", "next-key": "k"})],
            [(_DAY2, rows[2:], _cert(_DAY2, 2, 3, True)), (_DAY, rows[:2], _cert(_DAY, 2, 3, True))],
            [_cert(_DAY2, 2, 3, True), _cert(_DAY, 2, 3, True)],
        )
        asyncio.run(
            harvest_symbol_tape(
                stub, object(), symbol, [_DAY, _DAY2], venue="KRX", sessions=sessions,
                store=store, run_id=f"tape-2020-01-04-{symbol}", profile=profile, on_result=pub.add,
            )
        )
    report = pub.flush()
    total_rows = sum(r.rows_written for r in flushed) + report.rows_written
    total_parts = sum(r.partitions_written for r in flushed) + report.partitions_written
    assert calls["n"] < 6 and total_parts >= 2
    assert total_rows == 12
    for day in (_DAY, _DAY2):
        part = pd.read_parquet(th.tick_partition_path(day, "regular"))
        assert set(part["symbol"].astype(str)) == {"005930", "000660", "035420"}
        assert len(part) == 6


def test_replacement_is_certified_only(tmp_path, monkeypatch) -> None:
    from src import settings as _settings

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    sessions = [s for s in TAPE_SESSIONS if s.session == "regular"]
    rows = [_tick("093000"), _tick("093100")]
    stub = _StubTape(
        [(rows, {"cont-yn": "Y", "next-key": "k"})],
        [(_DAY, rows, _cert(_DAY, 2, 3, True))],
        [_cert(_DAY, 2, 3, True)],
    )
    pub = TickTapePublisher(store=store, profile=profile, flush_rows=10**9)
    asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=sessions,
            store=store, run_id="tape-2020-01-04-010", profile=profile, on_result=pub.add,
        )
    )
    pub.flush()
    before = pd.read_parquet(th.tick_partition_path(_DAY, "regular"))
    assert len(before) == 2
    partial = _StubTape(
        [(rows[:1], {"cont-yn": "Y", "next-key": "k"})], [], [_cert(_DAY, 1, 9, False)],
    )
    pub2 = TickTapePublisher(store=store, profile=profile, flush_rows=10**9)
    asyncio.run(
        harvest_symbol_tape(
            partial, object(), "005930", [_DAY], venue="KRX", sessions=sessions,
            store=store, run_id="tape-2020-01-04-011", profile=profile, on_result=pub2.add,
        )
    )
    pub2.flush()
    after = pd.read_parquet(th.tick_partition_path(_DAY, "regular"))
    assert len(after) == 2
    assert after["ts_hms"].tolist() == before["ts_hms"].tolist()


def test_replay_same_run_id_is_idempotent(tmp_path, monkeypatch) -> None:
    from src import settings as _settings

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    sessions = [s for s in TAPE_SESSIONS if s.session == "regular"]
    rows = [_tick("093000")]

    def _run() -> None:
        stub = _StubTape(
            [(rows, {"cont-yn": "Y", "next-key": "k"})],
            [(_DAY, rows, _cert(_DAY, 1, 2, True))],
            [_cert(_DAY, 1, 2, True)],
        )
        pub = TickTapePublisher(store=store, profile=profile, flush_rows=10**9)
        asyncio.run(
            harvest_symbol_tape(
                stub, object(), "005930", [_DAY], venue="KRX", sessions=sessions,
                store=store, run_id="tape-2020-01-04-replay", profile=profile, on_result=pub.add,
            )
        )
        pub.flush()

    _run()
    first = pd.read_parquet(th.tick_partition_path(_DAY, "regular"))
    _run()
    second = pd.read_parquet(th.tick_partition_path(_DAY, "regular"))
    assert first[["symbol", "ts_hms", "price", "volume"]].reset_index(drop=True).equals(
        second[["symbol", "ts_hms", "price", "volume"]].reset_index(drop=True)
    )


def test_closed_day_guard_rejects_open_aftermarket(tmp_path) -> None:
    today = datetime.now(_SEOUL).date().isoformat()
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    pub = TickTapePublisher(store=store, profile=profile, flush_rows=10**9)
    from src.data.capture_contracts import CoverageEntry
    from src.data.capture_contracts import ArtifactRef
    from src.data.capture_contracts import CaptureDataset

    bad = TapeDayResult(
        symbol="005930", day=today, session="krx_aftermarket",
        frame=pd.DataFrame(),
        entry=CoverageEntry(
            symbol="005930", dataset=CaptureDataset.TRADE_TICKS, venue="KRX",
            session="krx_aftermarket", scheduled_at=None, status=CaptureStatus.NO_TRADES,
            rows=0, first_event_time=None, last_event_time=None,
            reason="tape_complete:krx_aftermarket=0:vendor_total=1",
            raw_refs=(ArtifactRef(path="raw/ref", sha256="abc", bytes=1),),
        ),
    )
    try:
        pub.add(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unclosed aftermarket day")


def test_memory_bound_flushes_by_rows(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    pub = TickTapePublisher(store=store, profile=profile, flush_rows=10)
    sessions = [s for s in TAPE_SESSIONS if s.session == "regular"]
    peak = 0
    for i in range(4):
        day = f"2020-01-{10 + i:02d}"
        rows = [_tick_on(day, f"0930{i:02d}") for _ in range(6)]
        frame = th._safe_normalize_ticks("kiwoom", rows, day, "005930", truncated=False)
        from src.data.capture_contracts import CoverageEntry
        from src.data.capture_contracts import CaptureDataset

        entry = CoverageEntry(
            symbol="005930", dataset=CaptureDataset.TRADE_TICKS, venue="KRX",
            session="regular", scheduled_at=None, status=CaptureStatus.COMPLETE,
            rows=len(frame), first_event_time=None, last_event_time=None,
            reason="tape_complete:regular=6:vendor_total=7", raw_refs=(),
        )
        pub.add(TapeDayResult(symbol="005930", day=day, session="regular", frame=frame, entry=entry))
        peak = max(peak, pub._buffered_rows)
    assert peak <= 10 + 6
    assert sessions


def test_transient_transport_error_retries_next_slot(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    rows = [_tick("093000")]
    good = _StubTape(
        [(rows, {"cont-yn": "Y", "next-key": "k"})],
        [(_DAY, rows, _cert(_DAY, 1, 2, True))],
        [_cert(_DAY, 1, 2, True)],
    )
    attempts = {"n": 0}

    class _Flaky:
        async def walk_tick_tape(self, *args: Any, **kwargs: Any) -> Any:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("transport down")
            return await good.walk_tick_tape(*args, **kwargs)

    out: list[TapeDayResult] = []
    res = asyncio.run(
        harvest_symbol_tape(
            _Flaky(), object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-020", profile=profile, on_result=out.append,
        )
    )
    assert attempts["n"] == 2
    assert res.unresolved_days == ()
    assert len(out) == 2


def test_unrequested_certified_day_is_not_settled_or_written(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    other = [_tick_on(_DAY2, "093000")]
    wanted = [_tick("093000")]
    stub = _StubTape(
        [(other, {"cont-yn": "Y", "next-key": "a"}), (wanted, {"cont-yn": "Y", "next-key": "b"})],
        [(_DAY2, other, _cert(_DAY2, 1, 2, True)), (_DAY, wanted, _cert(_DAY, 1, 2, True))],
        [_cert(_DAY2, 1, 2, True), _cert(_DAY, 1, 2, True)],
    )
    out: list[TapeDayResult] = []
    asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-010", profile=_profile(tmp_path), on_result=out.append,
        )
    )
    assert {r.day for r in out} == {_DAY}


def test_evidence_is_stored_only_for_pages_holding_requested_days(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    other = [_tick_on(_DAY2, "093000")]
    wanted = [_tick("093000")]
    stub = _StubTape(
        [(other, {"cont-yn": "Y", "next-key": "a"}), (other, {"cont-yn": "Y", "next-key": "b"}), (wanted, {"cont-yn": "Y", "next-key": "c"})],
        [(_DAY, wanted, _cert(_DAY, 1, 2, True))],
        [_cert(_DAY, 1, 2, True)],
    )
    out: list[TapeDayResult] = []
    asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-011", profile=_profile(tmp_path), on_result=out.append,
        )
    )
    raw_refs = {ref.path for r in out for ref in r.entry.raw_refs if ref.path.startswith("raw/")}
    assert len(raw_refs) == 1 and raw_refs.pop().endswith("-p0002-a00.json.gz")


def test_bracketed_day_reason_records_the_proof_used(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    rows = [_tick("093000")]
    bracketed = TapeDayCertificate(day=_DAY, received=1, vendor_total=None, complete=True, basis="bracketed")
    stub = _StubTape([(rows, {"cont-yn": "Y", "next-key": "a"})], [(_DAY, rows, bracketed)], [bracketed])
    out: list[TapeDayResult] = []
    asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-012", profile=_profile(tmp_path), on_result=out.append,
        )
    )
    regular = next(r for r in out if r.session == "regular")
    assert regular.entry.status == CaptureStatus.COMPLETE
    assert regular.entry.reason.startswith("tape_bracketed:regular=1")


def test_walk_stopped_by_deadline_reports_walk_incomplete_not_day_not_on_tape(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    stub = _StubTape([], [], [], termination="deadline")
    out: list[TapeDayResult] = []
    res = asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-020", profile=_profile(tmp_path), on_result=out.append,
        )
    )
    assert res.unresolved_days == (_DAY,)
    assert {r.entry.status for r in out} == {CaptureStatus.PARTIAL}
    assert all(r.entry.reason == "walk_incomplete:deadline" for r in out)
    assert stub.seen["budget"].deadline is None


def test_walk_deadline_is_passed_to_the_client_budget(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    stub = _StubTape([], [], [], termination="crossed_stop_day")
    limit = datetime(2030, 1, 1, tzinfo=_SEOUL)
    asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-021", profile=_profile(tmp_path), on_result=lambda r: None,
            walk_deadline=limit,
        )
    )
    assert stub.seen["budget"].deadline == limit
