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
    from datetime import timedelta

    tomorrow = (datetime.now(_SEOUL).date() + timedelta(days=1)).isoformat()
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    pub = TickTapePublisher(store=store, profile=profile, flush_rows=10**9)
    from src.data.capture_contracts import CoverageEntry
    from src.data.capture_contracts import ArtifactRef
    from src.data.capture_contracts import CaptureDataset

    bad = TapeDayResult(
        symbol="005930", day=tomorrow, session="krx_aftermarket",
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


def _recent_day(days_ago: int) -> str:
    from datetime import timedelta

    return (datetime.now(_SEOUL).date() - timedelta(days=days_ago)).isoformat()


_EMPTY_PAGE = ([{"cntr_tm": "", "cur_prc": "", "trde_qty": ""}], {"cont-yn": "N", "next-key": ""})


def test_tape_walk_uses_the_tape_page_guard(tmp_path) -> None:
    stub = _StubTape([], [], [], termination="crossed_stop_day")
    profile = _profile(tmp_path, COLLECTION_TAPE_MAX_PAGES=5000, COLLECTION_TICK_REPAIR_MAX_PAGES=1000)
    asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY], venue="KRX", sessions=_krx_sessions(),
            store=CaptureStore(tmp_path / "capture"), run_id="tape-2020-01-04-030", profile=profile, on_result=lambda r: None,
        )
    )
    assert stub.seen["budget"].max_pages == 5000


def test_empty_tape_settles_recent_days_as_no_trades_with_evidence(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    day = _recent_day(2)
    stub = _StubTape([_EMPTY_PAGE], [], [], termination="tape_empty")
    out: list[TapeDayResult] = []
    res = asyncio.run(
        harvest_symbol_tape(
            stub, object(), "031980", [day], venue="NXT", sessions=list(TAPE_SESSIONS),
            store=store, run_id="tape-2020-01-04-031", profile=_profile(tmp_path), on_result=out.append,
        )
    )
    assert res.unresolved_days == ()
    assert [r.session for r in out] == ["nxt_aftermarket"]
    entry = out[0].entry
    assert entry.status == CaptureStatus.NO_TRADES
    assert entry.reason.startswith("tape_empty:")
    assert entry.raw_refs and all((store.root / ref.path).exists() for ref in entry.raw_refs)


def test_empty_tape_cannot_prove_days_beyond_its_depth(tmp_path) -> None:
    profile = _profile(tmp_path)
    old = _recent_day(int(profile.COLLECTION_TAPE_LOOKBACK_DAYS) + 1)
    stub = _StubTape([_EMPTY_PAGE], [], [], termination="tape_empty")
    out: list[TapeDayResult] = []
    res = asyncio.run(
        harvest_symbol_tape(
            stub, object(), "031980", [old], venue="NXT", sessions=list(TAPE_SESSIONS),
            store=CaptureStore(tmp_path / "capture"), run_id="tape-2020-01-04-032", profile=profile, on_result=out.append,
        )
    )
    assert res.unresolved_days == (old,)
    assert [(r.entry.status, r.entry.reason) for r in out] == [(CaptureStatus.UNKNOWN, "day_not_on_tape")]


def test_empty_tape_with_uncertified_venue_stays_unknown(tmp_path) -> None:
    profile = _profile(tmp_path, routes={})
    stub = _StubTape([_EMPTY_PAGE], [], [], termination="tape_empty")
    out: list[TapeDayResult] = []
    asyncio.run(
        harvest_symbol_tape(
            stub, object(), "031980", [_recent_day(2)], venue="NXT", sessions=list(TAPE_SESSIONS),
            store=CaptureStore(tmp_path / "capture"), run_id="tape-2020-01-04-033", profile=profile, on_result=out.append,
        )
    )
    assert out and all(r.entry.status == CaptureStatus.UNKNOWN for r in out)
    assert all(r.entry.reason == "uncertified_venue" for r in out)


def _flush_one_certified_day(store, profile, namespace=None) -> None:
    rows = [_tick("093000")]
    pub = TickTapePublisher(store=store, profile=profile, flush_rows=10**9, publish_namespace=namespace)
    pub.add(
        TapeDayResult(
            symbol="005930", day=_DAY, session="regular",
            frame=th._safe_normalize_ticks("kiwoom", rows, _DAY, "005930", truncated=False),
            entry=th.CoverageEntry(
                symbol="005930", dataset=th.CaptureDataset.TRADE_TICKS, venue="KRX", session="regular",
                scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1, first_event_time=None,
                last_event_time=None, reason="tape_complete:regular=1:vendor_total=2", raw_refs=(),
            ),
        )
    )
    pub.flush()


def test_restarted_publisher_does_not_collide_with_earlier_group_manifests(tmp_path, monkeypatch, caplog) -> None:
    from src import settings as _settings

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    with caplog.at_level("WARNING"):
        _flush_one_certified_day(store, profile)
        _flush_one_certified_day(store, profile)
    assert "manifest_conflict" not in caplog.text
    runs = [p.name for p in (store.root / "manifests" / _DAY).iterdir() if "publish" in p.name]
    assert len(runs) == 2 and len(set(runs)) == 2


def test_publish_namespace_is_part_of_the_manifest_run_id_and_must_be_nonempty(tmp_path, monkeypatch) -> None:
    import pytest

    from src import settings as _settings

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    _flush_one_certified_day(store, profile, namespace="nsA")
    names = [p.name for p in (store.root / "manifests" / _DAY).iterdir()]
    assert f"tape-{_DAY}-regular-publish-nsA-0" in names
    with pytest.raises(ValueError, match="publish_namespace"):
        TickTapePublisher(store=store, profile=profile, flush_rows=1, publish_namespace=" ")


def test_needed_settles_only_requested_pairs_without_cross_product(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    old, today = "2020-01-02", "2020-01-03"
    rows_old = [_tick_on(old, "170000")]
    rows_today = [_tick_on(today, "093000")]
    stub = _StubTape(
        [(rows_old, {"cont-yn": "Y", "next-key": "a"}), (rows_today, {"cont-yn": "Y", "next-key": "b"})],
        [(today, rows_today, _cert(today, 1, 2, True)), (old, rows_old, _cert(old, 1, 2, True))],
        [_cert(today, 1, 2, True), _cert(old, 1, 2, True)],
    )
    out: list[TapeDayResult] = []
    res = asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [old, today], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-pair", profile=profile, on_result=out.append,
            needed=[(today, "regular"), (old, "krx_aftermarket")],
        )
    )
    assert res.skipped_unclosed == ()
    assert res.unresolved_days == ()
    assert {(r.day, r.session) for r in out} == {(today, "regular"), (old, "krx_aftermarket")}


def test_unclosed_needed_pair_is_skipped_not_fatal(tmp_path) -> None:
    from datetime import timedelta

    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    tomorrow = (datetime.now(_SEOUL).date() + timedelta(days=1)).isoformat()
    old = (datetime.now(_SEOUL).date() - timedelta(days=30)).isoformat()
    rows_old = [_tick_on(old, "093000")]
    stub = _StubTape(
        [(rows_old, {"cont-yn": "Y", "next-key": "a"})],
        [(old, rows_old, _cert(old, 1, 2, True))],
        [_cert(old, 1, 2, True)],
    )
    out: list[TapeDayResult] = []
    res = asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [old, tomorrow], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-skip", profile=profile, on_result=out.append,
            needed=[(old, "regular"), (tomorrow, "krx_aftermarket")],
        )
    )
    assert (tomorrow, "krx_aftermarket") in res.skipped_unclosed
    assert {(r.day, r.session) for r in out} == {(old, "regular")}
    assert tomorrow not in res.unresolved_days


def test_normal_day_single_session_matches_legacy_shape(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    rows = [_tick("093000")]
    regular_only = [s for s in _krx_sessions() if s.session == "regular"]

    def _run(**kwargs: Any) -> tuple[list[TapeDayResult], Any]:
        stub = _StubTape(
            [(rows, {"cont-yn": "Y", "next-key": "k"})],
            [(_DAY, rows, _cert(_DAY, 1, 2, True))],
            [_cert(_DAY, 1, 2, True)],
        )
        out: list[TapeDayResult] = []
        res = asyncio.run(
            harvest_symbol_tape(
                stub, object(), "005930", [_DAY], venue="KRX", sessions=regular_only,
                store=store, run_id="tape-2020-01-04-legacy", profile=profile,
                on_result=out.append, **kwargs,
            )
        )
        return out, res

    legacy_out, _ = _run()
    needed_out, needed_res = _run(needed=[(_DAY, "regular")])
    assert {(r.day, r.session) for r in needed_out} == {(r.day, r.session) for r in legacy_out}
    assert len(needed_out) == 1 and needed_out[0].entry.status == CaptureStatus.COMPLETE
    assert needed_res.skipped_unclosed == () and needed_res.unresolved_days == ()


def test_regular_guard_rejects_future_day(tmp_path) -> None:
    import pytest
    from datetime import timedelta

    from src.data.capture_contracts import ArtifactRef, CaptureDataset, CoverageEntry

    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    tomorrow = (datetime.now(_SEOUL).date() + timedelta(days=1)).isoformat()
    bad = TapeDayResult(
        symbol="005930", day=tomorrow, session="regular", frame=pd.DataFrame(),
        entry=CoverageEntry(
            symbol="005930", dataset=CaptureDataset.TRADE_TICKS, venue="KRX", session="regular",
            scheduled_at=None, status=CaptureStatus.NO_TRADES, rows=0, first_event_time=None,
            last_event_time=None, reason="tape_complete:regular=0:vendor_total=1",
            raw_refs=(ArtifactRef(path="raw/seed", sha256="abc", bytes=1),),
        ),
    )
    pub = TickTapePublisher(store=store, profile=profile, flush_rows=10**9)
    with pytest.raises(ValueError, match="regular session not closed"):
        pub.add(bad)


def test_needed_rejects_invalid_day(tmp_path) -> None:
    import pytest

    store = CaptureStore(tmp_path / "capture")
    with pytest.raises(ValueError, match="invalid needed day"):
        asyncio.run(
            harvest_symbol_tape(
                _StubTape([], [], []), object(), "005930", [_DAY], venue="KRX",
                sessions=_krx_sessions(), store=store, run_id="tape-2020-01-04-bad",
                profile=_profile(tmp_path), on_result=lambda r: None,
                needed=[("not-a-date", "regular")],
            )
        )


def test_empty_needed_settles_nothing(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    res = asyncio.run(
        harvest_symbol_tape(
            _StubTape([], [], []), object(), "005930", [_DAY], venue="KRX",
            sessions=_krx_sessions(), store=store, run_id="tape-2020-01-04-empty",
            profile=_profile(tmp_path), on_result=lambda r: None, needed=[],
        )
    )
    assert res.unresolved_days == () and res.skipped_unclosed == ()


def test_certified_day_without_needed_session_is_ignored(tmp_path) -> None:
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    other = [_tick_on("2020-01-03", "170000")]
    wanted_rows = [_tick("093000")]
    stub = _StubTape(
        [(other, {"cont-yn": "Y", "next-key": "a"}), (wanted_rows, {"cont-yn": "Y", "next-key": "b"})],
        [("2020-01-03", other, _cert("2020-01-03", 1, 2, True)), (_DAY, wanted_rows, _cert(_DAY, 1, 2, True))],
        [_cert("2020-01-03", 1, 2, True), _cert(_DAY, 1, 2, True)],
    )
    out: list[TapeDayResult] = []
    res = asyncio.run(
        harvest_symbol_tape(
            stub, object(), "005930", [_DAY, "2020-01-03"], venue="KRX", sessions=_krx_sessions(),
            store=store, run_id="tape-2020-01-04-ignore", profile=profile, on_result=out.append,
            needed=[(_DAY, "regular"), ("2020-01-03", "nxt_aftermarket")],
        )
    )
    assert {(r.day, r.session) for r in out} == {(_DAY, "regular")}
    assert res.unresolved_days == ()


def test_explicit_needed_does_not_require_days(tmp_path) -> None:
    stub = _StubTape([], [], [], termination="deadline")
    out: list[TapeDayResult] = []
    result = asyncio.run(harvest_symbol_tape(
        stub, object(), "005930", [], sessions=_krx_sessions(),
        store=CaptureStore(tmp_path / "capture"), run_id="needed-only",
        profile=_profile(tmp_path), on_result=out.append, needed=[(_DAY, "regular")],
    ))
    assert result.unresolved_days == (_DAY,)
    assert [(r.day, r.session) for r in out] == [(_DAY, "regular")]


def test_empty_or_unclosed_request_never_calls_vendor(tmp_path) -> None:
    from datetime import timedelta

    tomorrow = (datetime.now(_SEOUL).date() + timedelta(days=1)).isoformat()

    class Unavailable:
        async def walk_tick_tape(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("no closable request must not call vendor")

    for sessions, needed, skipped in (
        ([], [], ()),
        (_krx_sessions(), [(tomorrow, "krx_aftermarket")], ((tomorrow, "krx_aftermarket"),)),
        ([], [(_DAY, "regular")], ()),
    ):
        out: list[TapeDayResult] = []
        result = asyncio.run(harvest_symbol_tape(
            Unavailable(), object(), "005930", [], sessions=sessions,
            store=CaptureStore(tmp_path / "capture"), run_id="no-query",
            profile=_profile(tmp_path), on_result=out.append, needed=needed,
        ))
        assert result.pages_fetched == 0 and result.unresolved_days == ()
        assert result.skipped_unclosed == skipped
        assert out == []


def test_pair_selection_is_preserved_through_publication_and_ledger(tmp_path, monkeypatch) -> None:
    from src import settings
    from src.backfill.intraday import tape_recovery as tr

    monkeypatch.setattr(settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(tr, "_disk_ok", lambda root, profile: True)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    old_rows = [_tick_on(_DAY, "170000")]
    new_rows = [_tick_on(_DAY2, "093000")]
    stub = _StubTape(
        [(new_rows + old_rows, {"cont-yn": "N", "next-key": ""})],
        [(_DAY2, new_rows, _cert(_DAY2, 1, 2, True)), (_DAY, old_rows, _cert(_DAY, 1, 2, True))],
        [_cert(_DAY2, 1, 2, True), _cert(_DAY, 1, 2, True)],
    )
    tasks = tr.order_walk_tasks([
        tr.Need("005930", _DAY2, "regular", "KRX"),
        tr.Need("005930", _DAY, "krx_aftermarket", "KRX"),
    ])
    ledger = tmp_path / "ledger.jsonl"
    summary = asyncio.run(tr.run_walk_tasks(
        tasks, client=stub, http_session=object(), store=store, profile=profile,
        apply=True, ledger=ledger, deadline=None, blackouts=[], run_date="2026-10-05",
    ))
    assert summary.unresolved == 0
    assert set(tr.read_settled_ledger(ledger)) == {
        ("005930", _DAY2, "regular"), ("005930", _DAY, "krx_aftermarket"),
    }
    for day, session in ((_DAY2, "regular"), (_DAY, "krx_aftermarket")):
        assert len(pd.read_parquet(th.tick_partition_path(day, session))) == 1
        assert store.read_manifests(day)
    assert not th.tick_partition_path(_DAY, "regular").exists()
    assert not th.tick_partition_path(_DAY2, "krx_aftermarket").exists()
