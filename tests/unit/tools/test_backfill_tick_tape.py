"""Invariant guards for the one-shot tick tape backfill tool."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import src.tools.backfill_tick_tape as btt
from src.backfill.intraday.tape_harvest import TapeDayResult
from src.backfill.intraday.tape_harvest import TapeWalkOutcome
from src.config.collection import CollectionSettings
from src.data.capture_contracts import (
    ArtifactRef,
    CaptureDataset,
    CaptureStatus,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.intraday_schema import normalize_bar_frame, normalize_tick_frame
from src.data.intraday_store import tick_partition_path, write_intraday_partition, write_tick_partition

_SEOUL = ZoneInfo("Asia/Seoul")
_DAY = "2026-09-02"
_YMD = "20260902"


def _profile(tmp_path) -> CollectionSettings:
    return CollectionSettings(COLLECTION_ROOT=tmp_path / "capture")


def _patch_roots(tmp_path, monkeypatch) -> None:
    from src import settings as _settings

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    monkeypatch.setattr(btt, "_capture_root", lambda profile: tmp_path / "capture")
    monkeypatch.setattr(btt, "_free_bytes", lambda path: 100 * 1024**3)


def _tick_rows(hms_list: list[str], qty: str = "100", day: str = _YMD) -> list[dict]:
    return [{"cntr_tm": f"{day}{hms}", "cur_prc": "10000", "trde_qty": qty} for hms in hms_list]


def _bar_rows(hms_list: list[str], qty: str = "100", day: str = _YMD) -> list[dict]:
    return [
        {
            "cntr_tm": f"{day}{hms}", "cur_prc": "10000", "open_pric": "9900",
            "high_pric": "10100", "low_pric": "9800", "trde_qty": qty,
        }
        for hms in hms_list
    ]


def _complete_entry(symbol: str, session: str, rows: int) -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol, dataset=CaptureDataset.TRADE_TICKS, venue="KRX", session=session,
        scheduled_at=None, status=CaptureStatus.COMPLETE, rows=rows,
        first_event_time=None, last_event_time=None, reason="seeded",
        raw_refs=(ArtifactRef(path="raw/seed", sha256="abc", bytes=1),),
    )


def _seed_tick_day(day: str, session: str, symbol: str, tick_rows: list[dict], bar_rows: list[dict] | None) -> None:
    ymd = day.replace("-", "")
    for row in tick_rows:
        row["cntr_tm"] = f"{ymd}{row['cntr_tm'][-6:]}"
    frame = normalize_tick_frame(pd.DataFrame(tick_rows), "kiwoom", day, symbol)
    write_tick_partition(frame, day, session, coverage={symbol: _complete_entry(symbol, session, len(frame))})
    if bar_rows is not None:
        for row in bar_rows:
            row["cntr_tm"] = f"{ymd}{row['cntr_tm'][-6:]}"
        bars = normalize_bar_frame(pd.DataFrame(bar_rows), "kiwoom", day, symbol)
        write_intraday_partition(bars, 1, day, session, coverage={symbol: _complete_entry(symbol, session, len(bars))})


class _SessionCtx:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: Any) -> bool:
        return False


class _StubKiwoom:
    def __init__(self, outcome: Any = None) -> None:
        self.walks: list[dict[str, Any]] = []
        self.outcome = outcome

    async def walk_tick_tape(self, session: Any, code: str, **kwargs: Any) -> Any:
        self.walks.append({"symbol": code, **kwargs})
        return self.outcome


def _fake_harvest_factory(calls: list[dict[str, Any]], outcome: Any = None, emit: Any = None) -> Any:
    async def _fake(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        calls.append({"symbol": symbol, "venue": kwargs.get("venue"), "days": list(days)})
        if emit is not None:
            for result in emit(symbol, list(days)):
                kwargs["on_result"](result)
        return outcome or TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    return _fake


def test_needs_exclude_healthy_symbol_days(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(btt, "_day_universe", lambda day, store: ["005930", "000660"])
    _seed_tick_day(_DAY, "regular", "005930", _tick_rows(["090000", "090100"], "100"), _bar_rows(["090000", "090100"], "100"))
    _seed_tick_day(_DAY, "regular", "000660", _tick_rows(["090000"], "10"), _bar_rows(["090000", "090100"], "1000"))
    needs = btt._collect_needs([_DAY], ["KRX"], store, {}, False)
    selected = {(n.symbol, n.session) for n in needs if n.day == _DAY}
    assert ("000660", "regular") in selected
    assert ("005930", "regular") not in selected


def test_missing_day_selected(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(btt, "_day_universe", lambda day, store: ["005930"])
    needs = btt._collect_needs([_DAY], ["KRX"], store, {}, False)
    assert {(n.symbol, n.day, n.session) for n in needs} == {
        ("005930", _DAY, "regular"), ("005930", _DAY, "krx_aftermarket"),
    }


def _run_main(tmp_path, monkeypatch, argv: list[str], StubCls: Any = _StubKiwoom) -> Any:
    _patch_roots(tmp_path, monkeypatch)
    stub = StubCls()
    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (stub, _SessionCtx()))
    btt.main(argv)
    return stub


def test_oldest_need_ordering_and_single_walk(tmp_path, monkeypatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(btt, "harvest_symbol_tape", _fake_harvest_factory(calls))
    _patch_roots(tmp_path, monkeypatch)
    canned = [
        btt.Need(symbol="BBB", day="2026-09-02", session="regular", venue="KRX"),
        btt.Need(symbol="AAA", day="2026-09-01", session="regular", venue="KRX"),
        btt.Need(symbol="CCC", day="2026-09-15", session="regular", venue="KRX"),
        btt.Need(symbol="BBB", day="2026-09-03", session="regular", venue="KRX"),
        btt.Need(symbol="BBB", day="2026-09-04", session="regular", venue="KRX"),
    ]
    monkeypatch.setattr(btt, "_collect_needs", lambda *a, **k: list(canned))
    btt.main(["--start", "2026-09-01", "--end", "2026-09-15", "--venue", "krx", "--apply", "--ledger", str(tmp_path / "ledger.jsonl")])
    assert [c["symbol"] for c in calls] == ["AAA", "BBB", "CCC"]
    bbb = next(c for c in calls if c["symbol"] == "BBB")
    assert sum(1 for c in calls if c["symbol"] == "BBB") == 1
    assert bbb["days"] == ["2026-09-02", "2026-09-03", "2026-09-04"]


def test_blackout_waits_before_walk(tmp_path, monkeypatch) -> None:
    moments = {"n": 0}
    base = datetime(2026, 9, 10, 20, 10, tzinfo=_SEOUL)
    after = datetime(2026, 9, 10, 20, 31, tzinfo=_SEOUL)

    def _fake_now() -> datetime:
        moments["n"] += 1
        return base if moments["n"] <= 3 else after

    slept: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(btt, "_now", _fake_now)
    monkeypatch.setattr(btt, "_sleep", _fake_sleep)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(btt, "harvest_symbol_tape", _fake_harvest_factory(calls))
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.setattr(
        btt, "_collect_needs",
        lambda *a, **k: [btt.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--apply", "--ledger", str(tmp_path / "l.jsonl")])
    assert len(slept) == 1 and slept[0] > 0
    assert len(calls) == 1


def test_deadline_stops_new_walks(tmp_path, monkeypatch, caplog) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(btt, "harvest_symbol_tape", _fake_harvest_factory(calls))
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.setattr(
        btt, "_collect_needs",
        lambda *a, **k: [btt.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    with caplog.at_level("INFO"):
        btt.main([
            "--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--apply",
            "--deadline", "2020-01-01T00:00:00+09:00", "--ledger", str(tmp_path / "l.jsonl"),
        ])
    assert calls == []
    assert "remaining" in caplog.text


def test_resume_skips_settled(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    ledger = tmp_path / "l.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "".join(
            json.dumps({"symbol": "A", "day": "2026-09-02", "session": s, "status": "COMPLETE", "run_id": "r"})
            + "\n"
            for s in ("regular", "krx_aftermarket")
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(btt, "_day_universe", lambda day, store: ["A", "B"])
    store = CaptureStore(tmp_path / "capture")
    settled = btt._read_settled(ledger)
    needs = btt._collect_needs(["2026-09-02"], ["KRX"], store, settled, False)
    assert needs and all(n.symbol == "B" for n in needs)
    partial_ledger = tmp_path / "p.jsonl"
    partial_ledger.write_text(
        json.dumps({"symbol": "B", "day": "2026-09-02", "session": "regular", "status": "PARTIAL", "run_id": "r"}) + "\n",
        encoding="utf-8",
    )
    needs2 = btt._collect_needs(["2026-09-02"], ["KRX"], store, btt._read_settled(partial_ledger), False)
    assert any(n.symbol == "B" and n.session == "regular" for n in needs2)


def test_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    from src.data.capture_store import CaptureStore as _Store

    _patch_roots(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []

    async def _dry_harvest(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        calls.append({"symbol": symbol})
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=2, unresolved_days=())

    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(btt, "harvest_symbol_tape", _dry_harvest)
    monkeypatch.setattr(
        btt, "_collect_needs",
        lambda *a, **k: [btt.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--ledger", str(tmp_path / "l.jsonl")])
    assert len(calls) == 1
    assert not tick_partition_path("2026-09-02", "regular").exists()
    assert not (tmp_path / "capture" / "manifests").exists()
    assert not (tmp_path / "l.jsonl").exists()
    assert _Store


def test_oversized_span_rejected(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="span"):
        btt.main(["--start", "2026-08-01", "--end", "2026-09-09", "--venue", "krx"])


def test_guard_stop_reports_remaining(tmp_path, monkeypatch, caplog) -> None:
    _patch_roots(tmp_path, monkeypatch)

    def _emit(symbol: str, days: list[str]) -> list[Any]:
        from src.backfill.intraday.tape_harvest import TapeDayResult

        return [
            TapeDayResult(
                symbol=symbol, day=day, session="regular", frame=pd.DataFrame(),
                entry=CoverageEntry(
                    symbol=symbol, dataset=CaptureDataset.TRADE_TICKS, venue="KRX",
                    session="regular", scheduled_at=None, status=CaptureStatus.PARTIAL,
                    rows=0, first_event_time=None, last_event_time=None,
                    reason="tape_total_mismatch:received=1:total=9", raw_refs=(),
                ),
            )
            for day in days
        ]

    async def _guarded(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        for result in _emit(symbol, list(days)):
            kwargs["on_result"](result)
        return TapeWalkOutcome(termination_reason="page_budget", pages_fetched=1000, unresolved_days=tuple(days))

    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(btt, "harvest_symbol_tape", _guarded)
    monkeypatch.setattr(
        btt, "_collect_needs",
        lambda *a, **k: [btt.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    with caplog.at_level("INFO"):
        btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--apply", "--ledger", str(tmp_path / "l.jsonl")])
    assert "GUARD_STOP" in caplog.text
    assert not tick_partition_path("2026-09-02", "regular").exists()


def test_kiwoom_only(tmp_path, monkeypatch) -> None:
    for name in ("src.api.kis.client", "src.api.ls.client"):
        mod = types.ModuleType(name)

        def _boom(*args: Any, _name: str = name, **kwargs: Any) -> Any:
            raise AssertionError(f"forbidden client use: {_name}")

        mod.__getattr__ = _boom  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, mod)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(btt, "harvest_symbol_tape", _fake_harvest_factory(calls))
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.setattr(
        btt, "_collect_needs",
        lambda *a, **k: [btt.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--apply", "--ledger", str(tmp_path / "l.jsonl")])
    assert len(calls) == 1


def test_storage_refusal(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.setattr(btt, "_free_bytes", lambda path: 0)
    monkeypatch.setattr(
        btt, "_collect_needs",
        lambda *a, **k: [btt.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    with pytest.raises(RuntimeError, match="storage budget"):
        btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--apply", "--ledger", str(tmp_path / "l.jsonl")])


def test_invalid_venue_and_naive_deadline(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="venue"):
        btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "bogus"])
    with pytest.raises(ValueError, match="timezone-aware"):
        btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--deadline", "2026-09-20T10:00:00"])


def test_parse_helpers_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="--start"):
        btt._parse_day("not-a-date", "--start")
    with pytest.raises(ValueError, match="--deadline timestamp"):
        btt._parse_deadline("not-a-time")
    assert btt._parse_deadline(None) is None
    assert btt._parse_blackout("09:00-10:00") == (540, 600)
    with pytest.raises(ValueError, match="blackout"):
        btt._parse_blackout("bogus")
    with pytest.raises(ValueError, match="blackout"):
        btt._parse_blackout("25:00-26:00")
    assert btt._in_blackout(30, (1380, 60)) is True
    assert btt._in_blackout(120, (1380, 60)) is False


def test_overnight_blackout_end() -> None:
    night = datetime(2026, 9, 10, 23, 30, tzinfo=_SEOUL)
    end = btt._blackout_end(night, [(1380, 60)])
    assert end is not None and (end.day, end.hour, end.minute) == (11, 1, 0)
    assert btt._blackout_end(night, [(540, 600)]) is None
    midnight = btt._blackout_end(night, [(1380, 1440)])
    assert midnight is not None and (midnight.day, midnight.hour) == (11, 0)
    early = datetime(2026, 9, 11, 0, 30, tzinfo=_SEOUL)
    same_day = btt._blackout_end(early, [(1380, 60)])
    assert same_day is not None and (same_day.day, same_day.hour, same_day.minute) == (11, 1, 0)
    asyncio.run(btt._sleep(0))


def test_open_kiwoom_guards_and_builds(tmp_path, monkeypatch) -> None:
    from src import settings as _settings

    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.setattr(_settings, "KIWOOM_APP_KEY", "", raising=False)
    with pytest.raises(RuntimeError, match="Kiwoom credentials"):
        btt._open_kiwoom()
    monkeypatch.setattr(_settings, "KIWOOM_APP_KEY", "k", raising=False)

    async def _build() -> Any:
        return btt._open_kiwoom()

    client, ctx = asyncio.run(_build())
    assert client is not None

    async def _close() -> None:
        async with ctx:
            pass

    asyncio.run(_close())


def test_read_symbols_branches(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    assert btt._read_symbols(tmp_path / "absent.parquet") == []
    nosym = tmp_path / "nosym.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(nosym)
    assert btt._read_symbols(nosym) == []
    monkeypatch.setattr(btt.pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    try:
        garbage = tmp_path / "garbage.parquet"
        garbage.write_bytes(b"not a parquet file")
        assert btt._read_symbols(garbage) == []
    finally:
        monkeypatch.undo()


def test_read_cohort_and_universe(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")

    def _fake_read(day: str, available_by: Any = None) -> Any:
        if day == "2026-09-03":
            return SimpleNamespace(eligible_symbols=("A",))
        if day == "2026-09-01":
            return SimpleNamespace(eligible_symbols=("B",))
        raise FileNotFoundError(day)

    monkeypatch.setattr(store, "read_cohort", _fake_read)
    assert btt._read_cohort_symbols(store, "2026-09-05") == []
    _seed_tick_day("2026-09-03", "regular", "000003", _tick_rows(["090000"]), None)
    assert btt._day_universe("2026-09-03", store) == ["A", "B", "000003"]
    assert btt._day_universe("2026-09-05", store) == ["A"]
    close_rows = [
        {
            "cntr_tm": "20260920160000", "cur_prc": "10000", "open_pric": "9900",
            "high_pric": "10100", "low_pric": "9800", "trde_qty": "100",
        }
    ]
    bars = normalize_bar_frame(pd.DataFrame(close_rows), "kiwoom", "2026-09-20", "000004")
    write_intraday_partition(bars, 1, "2026-09-20", "nxt_aftermarket", coverage={"000004": _complete_entry("000004", "nxt_aftermarket", len(bars))})
    assert btt._day_universe("2026-09-20", store) == ["000004"]


def test_bar_volumes_index_branches(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    index = btt.PartitionIndex()
    assert index.bar_volumes(_DAY, "regular") is None
    _seed_tick_day(_DAY, "regular", "005930", _tick_rows(["090000"]), _bar_rows(["090000"]))
    fresh = btt.PartitionIndex()
    assert "OTHER" not in (fresh.bar_volumes(_DAY, "regular") or {})
    only_close_rows = [
        {
            "cntr_tm": "20260906153000", "cur_prc": "10000", "open_pric": "9900",
            "high_pric": "10100", "low_pric": "9800", "trde_qty": "100",
        }
    ]
    only_close = normalize_bar_frame(pd.DataFrame(only_close_rows), "kiwoom", "2026-09-06", "005930")
    write_intraday_partition(only_close, 1, "2026-09-06", "regular", coverage={"005930": _complete_entry("005930", "regular", len(only_close))})
    assert (fresh.bar_volumes("2026-09-06", "regular") or {}).get("005930") is None
    assert fresh.bar_volumes(_DAY, "regular")["005930"] == pytest.approx(100.0)


def test_partition_index_reads_each_partition_once(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    _seed_tick_day(_DAY, "regular", "005930", _tick_rows(["090000"]), _bar_rows(["090000"]))
    regular = next(s for s in btt.TAPE_SESSIONS if s.session == "regular")
    reads = {"n": 0}
    real = btt.pd.read_parquet

    def _counting(*args: Any, **kwargs: Any) -> Any:
        reads["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(btt.pd, "read_parquet", _counting)
    index = btt.PartitionIndex()
    for symbol in ("005930", "000660", "035420", "051910"):
        btt._session_need(symbol, _DAY, regular, index)
    assert reads["n"] == 2  # one tick partition + one bar partition, regardless of symbol count


def test_session_need_flags(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    regular = next(s for s in btt.TAPE_SESSIONS if s.session == "regular")
    frame = normalize_tick_frame(pd.DataFrame(_tick_rows(["090000"])), "kiwoom", _DAY, "005930")
    frame["truncated"] = True
    write_tick_partition(frame, _DAY, "regular", coverage={"005930": _complete_entry("005930", "regular", len(frame))})
    assert btt._session_need("005930", _DAY, regular, btt.PartitionIndex()) is True
    frame2 = normalize_tick_frame(pd.DataFrame(_tick_rows(["085000"], day="20260907")), "kiwoom", "2026-09-07", "005930")
    write_tick_partition(frame2, "2026-09-07", "regular", coverage={"005930": _complete_entry("005930", "regular", len(frame2))})
    assert btt._session_need("005930", "2026-09-07", regular, btt.PartitionIndex()) is True
    frame3 = normalize_tick_frame(pd.DataFrame(_tick_rows(["090000"], "10", day="20260908")), "kiwoom", "2026-09-08", "005930")
    frame3["vendor"] = "ls"
    write_tick_partition(frame3, "2026-09-08", "regular", coverage={"005930": _complete_entry("005930", "regular", len(frame3))})
    bars = normalize_bar_frame(pd.DataFrame(_bar_rows(["090000"], "1000", day="20260908")), "kiwoom", "2026-09-08", "005930")
    write_intraday_partition(bars, 1, "2026-09-08", "regular", coverage={"005930": _complete_entry("005930", "regular", len(bars))})
    assert btt._session_need("005930", "2026-09-08", regular, btt.PartitionIndex()) is True


def test_collect_skips_unclosed_and_session_closed_units(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(btt, "_day_universe", lambda day, store: ["005930"])
    assert btt._collect_needs(["2999-01-01"], ["KRX", "NXT"], store, {}, False) == []
    assert btt._collect_needs(["2999-01-01"], ["KRX"], store, {}, True) == []
    morning = datetime(2026, 9, 10, 10, 0, tzinfo=_SEOUL)
    evening = datetime(2026, 9, 10, 16, 0, tzinfo=_SEOUL)
    assert btt._session_closed("2026-09-10", "regular", morning) is False
    assert btt._session_closed("2026-09-10", "regular", evening) is True
    assert btt._session_closed("2026-09-10", "krx_aftermarket", evening) is False
    assert btt._session_closed("2026-09-09", "krx_aftermarket", morning) is True


def test_ledger_helpers(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    profile = _profile(tmp_path)
    assert btt._ledger_path(None, profile) == tmp_path / "capture" / "staging" / "tape_backfill" / "ledger.jsonl"
    assert btt._ledger_path(str(tmp_path / "x.jsonl"), profile) == tmp_path / "x.jsonl"
    assert btt._read_settled(tmp_path / "absent.jsonl") == {}
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(
        "\nnot-json\n" + json.dumps({"symbol": "A", "day": "2026-09-02", "session": "regular", "status": "COMPLETE"}) + "\n",
        encoding="utf-8",
    )
    assert btt._read_settled(ledger) == {("A", "2026-09-02", "regular"): "COMPLETE"}
    with pytest.raises(RuntimeError, match="unreadable"):
        btt._read_settled(tmp_path)
    btt._append_ledger(tmp_path / "noop.jsonl", [])
    assert not (tmp_path / "noop.jsonl").exists()
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ledger write failed"):
        btt._append_ledger(blocker / "l.jsonl", [{"symbol": "A"}])


def test_free_bytes_real(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.undo()
    assert btt._free_bytes(tmp_path) > 0


def test_verify_groups(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    _seed_tick_day(_DAY, "regular", "005930", _tick_rows(["090000"]), None)
    btt._verify_groups({(_DAY, "regular"): 1})
    with pytest.raises(RuntimeError, match="verification failed"):
        btt._verify_groups({(_DAY, "regular"): 99})
    corrupt = tick_partition_path("2026-09-09", "regular")
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="publication failed"):
        btt._verify_groups({("2026-09-09", "regular"): 0})


def test_session_need_missing_symbol_in_partition(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    regular = next(s for s in btt.TAPE_SESSIONS if s.session == "regular")
    _seed_tick_day(_DAY, "regular", "005930", _tick_rows(["090000"]), None)
    assert btt._session_need("000660", _DAY, regular, btt.PartitionIndex()) is True


def test_apply_publishes_complete_results(tmp_path, monkeypatch) -> None:
    from src.backfill.intraday.tape_harvest import TapeDayResult

    _patch_roots(tmp_path, monkeypatch)
    rows = _tick_rows(["090000", "090100"])

    def _emit(symbol: str, days: list[str]) -> list[Any]:
        out = []
        for day in days:
            ymd = day.replace("-", "")
            day_rows = [{"cntr_tm": f"{ymd}093000", "cur_prc": "10000", "trde_qty": "10"}]
            frame = normalize_tick_frame(pd.DataFrame(day_rows), "kiwoom", day, symbol)
            ref = ArtifactRef(path="raw/seed", sha256="abc", bytes=1)
            entry = CoverageEntry(
                symbol=symbol, dataset=CaptureDataset.TRADE_TICKS, venue="KRX",
                session="regular", scheduled_at=None, status=CaptureStatus.COMPLETE,
                rows=len(frame), first_event_time=None, last_event_time=None,
                reason="tape_complete:regular=1:vendor_total=2", raw_refs=(ref,),
            )
            out.append(TapeDayResult(symbol=symbol, day=day, session="regular", frame=frame, entry=entry))
        return out

    async def _emitting(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        for result in _emit(symbol, list(days)):
            kwargs["on_result"](result)
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(btt, "harvest_symbol_tape", _emitting)
    monkeypatch.setattr(
        btt, "_collect_needs",
        lambda *a, **k: [btt.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    ledger = tmp_path / "l.jsonl"
    btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--apply", "--ledger", str(ledger)])
    assert rows
    part = pd.read_parquet(tick_partition_path("2026-09-02", "regular"))
    assert len(part) == 1 and part["symbol"].astype(str).tolist() == ["005930"]
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert records[0]["status"] == "COMPLETE"


def test_run_tasks_evidence_error(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)

    async def _boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("evidence down")

    monkeypatch.setattr(btt, "harvest_symbol_tape", _boom)
    task = btt.WalkTask(symbol="005930", venue="KRX", days=(_DAY,), sessions=tuple(s for s in btt.TAPE_SESSIONS if s.venue == "KRX"))
    with pytest.raises(RuntimeError, match="evidence failed"):
        asyncio.run(
            btt._run_tasks(
                [task], client=object(), http_session=object(), store=store, profile=profile,
                apply=True, ledger=tmp_path / "l.jsonl", deadline=None, blackouts=[], run_date="2026-09-20",
            )
        )


def test_run_tasks_flushes_by_row_bound_and_commits_ledger_only_after_flush(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path).model_copy(update={"COLLECTION_TAPE_FLUSH_ROWS": 2})
    events: list[str] = []
    ledger = tmp_path / "l.jsonl"

    class _FakePublisher:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.rows = 0
            assert kwargs.get("auto_flush") is False

        def add(self, result: Any) -> None:
            self.rows += len(result.frame)
            events.append("add")

        def should_flush(self) -> bool:
            return self.rows >= 2

        def flush(self) -> Any:
            events.append(f"flush:ledger_exists={ledger.exists()}")
            self.rows = 0
            return None

    async def _emit(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        frame = normalize_tick_frame(pd.DataFrame(_tick_rows(["090000"])), "kiwoom", _DAY, symbol)
        entry = _complete_entry(symbol, "regular", len(frame))
        kwargs["on_result"](TapeDayResult(symbol=symbol, day=_DAY, session="regular", frame=frame, entry=entry))
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(btt, "TickTapePublisher", _FakePublisher)
    monkeypatch.setattr(btt, "harvest_symbol_tape", _emit)
    monkeypatch.setattr(btt, "_verify_groups", lambda expected: None)
    sessions = tuple(s for s in btt.TAPE_SESSIONS if s.venue == "KRX")
    tasks = [btt.WalkTask(symbol=f"{i:06d}", venue="KRX", days=(_DAY,), sessions=sessions) for i in range(3)]
    summary = asyncio.run(
        btt._run_tasks(
            tasks, client=object(), http_session=object(), store=store, profile=profile,
            apply=True, ledger=ledger, deadline=None, blackouts=[], run_date="2026-09-20",
        )
    )
    assert summary["pages"] == 3
    # rows reach the bound after 2 symbols (flush 1: no ledger yet); the 3rd symbol flushes at the end, by which time
    # only the first flush's records exist, so a record is never written before its own flush succeeded
    assert [e for e in events if e.startswith("flush")] == ["flush:ledger_exists=False", "flush:ledger_exists=True"]
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3


def test_run_tasks_does_not_record_ledger_when_flush_fails(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    ledger = tmp_path / "l.jsonl"

    async def _emit(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        frame = normalize_tick_frame(pd.DataFrame(_tick_rows(["090000"])), "kiwoom", _DAY, symbol)
        kwargs["on_result"](TapeDayResult(symbol=symbol, day=_DAY, session="regular", frame=frame, entry=_complete_entry(symbol, "regular", len(frame))))
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    def _boom(self: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(btt, "harvest_symbol_tape", _emit)
    monkeypatch.setattr(btt.TickTapePublisher, "flush", _boom)
    sessions = tuple(s for s in btt.TAPE_SESSIONS if s.venue == "KRX")
    task = btt.WalkTask(symbol="005930", venue="KRX", days=(_DAY,), sessions=sessions)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(
            btt._run_tasks(
                [task], client=object(), http_session=object(), store=store, profile=profile,
                apply=True, ledger=ledger, deadline=None, blackouts=[], run_date="2026-09-20",
            )
        )
    assert not ledger.exists()


def test_main_branches(tmp_path, monkeypatch, caplog) -> None:
    _patch_roots(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Invalid backfill range"):
        btt.main(["--start", "2026-09-05", "--end", "2026-09-02", "--venue", "krx"])
    with pytest.raises(ValueError, match="symbols-limit"):
        btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--symbols-limit", "0"])
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(btt, "harvest_symbol_tape", _fake_harvest_factory(calls))
    canned = [
        btt.Need(symbol="AAA", day="2026-09-02", session="regular", venue="KRX"),
        btt.Need(symbol="BBB", day="2026-09-02", session="regular", venue="KRX"),
        btt.Need(symbol="CCC", day="2026-09-02", session="regular", venue="KRX"),
    ]
    monkeypatch.setattr(btt, "_collect_needs", lambda *a, **k: list(canned))
    nxt_only = [btt.Need(symbol="AAA", day="2026-09-02", session="nxt_aftermarket", venue="NXT")]
    monkeypatch.setattr(btt, "_collect_needs", lambda *a, **k: list(nxt_only))
    btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "nxt", "--apply", "--ledger", str(tmp_path / "l.jsonl")])
    assert [c["venue"] for c in calls] == ["NXT"]
    calls.clear()
    monkeypatch.setattr(btt, "_collect_needs", lambda *a, **k: list(canned))
    btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--apply", "--symbols-limit", "2", "--ledger", str(tmp_path / "l2.jsonl")])
    assert [c["symbol"] for c in calls] == ["AAA", "BBB"]
    monkeypatch.setattr(btt, "_collect_needs", lambda *a, **k: [])
    with caplog.at_level("INFO"):
        btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--ledger", str(tmp_path / "l3.jsonl")])
    assert "NOOP" in caplog.text
    monkeypatch.setattr(btt, "_collect_needs", lambda *a, **k: list(canned)[:1])
    monkeypatch.setattr(btt, "_free_bytes", lambda path: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(RuntimeError, match="storage check failed"):
        btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--ledger", str(tmp_path / "l4.jsonl")])
    monkeypatch.setattr(btt, "_free_bytes", lambda path: 100 * 1024**3)

    def _raise_open() -> Any:
        raise OSError("auth down")

    monkeypatch.setattr(btt, "_open_kiwoom", _raise_open)
    with pytest.raises(RuntimeError, match="infrastructure failed"):
        btt.main(["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--apply", "--ledger", str(tmp_path / "l5.jsonl")])


def test_run_tasks_stops_cleanly_on_disk_guard(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    walked: list[str] = []

    async def _walk(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        walked.append(symbol)
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(btt, "harvest_symbol_tape", _walk)
    monkeypatch.setattr(btt, "_free_bytes", lambda path: 0)
    purged: list[Any] = []
    import src.tools.backup_prune as prune

    monkeypatch.setattr(prune, "prune_local_intraday_backups", lambda **kw: purged.append(kw) or [])
    sessions = tuple(s for s in btt.TAPE_SESSIONS if s.venue == "KRX")
    tasks = [btt.WalkTask(symbol="005930", venue="KRX", days=(_DAY,), sessions=sessions)]
    summary = asyncio.run(
        btt._run_tasks(
            tasks, client=object(), http_session=object(), store=store, profile=profile,
            apply=True, ledger=tmp_path / "l.jsonl", deadline=None, blackouts=[], run_date="2026-09-20",
        )
    )
    assert walked == []
    assert summary["stopped_reason"] == "disk_guard" and summary["remaining"] == ["005930/KRX"]
    assert len(purged) == 1  # exactly one expired-snapshot prune attempt before stopping


def test_run_tasks_bounds_each_walk_by_the_next_blackout(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    seen: dict[str, Any] = {}
    fixed = datetime(2026, 9, 20, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    monkeypatch.setattr(btt, "_now", lambda: fixed)

    async def _walk(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        seen["walk_deadline"] = kwargs.get("walk_deadline")
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(btt, "harvest_symbol_tape", _walk)
    sessions = tuple(s for s in btt.TAPE_SESSIONS if s.venue == "KRX")
    tasks = [btt.WalkTask(symbol="005930", venue="KRX", days=(_DAY,), sessions=sessions)]
    asyncio.run(
        btt._run_tasks(
            tasks, client=object(), http_session=object(), store=store, profile=profile,
            apply=False, ledger=tmp_path / "l.jsonl", deadline=None,
            blackouts=[btt._parse_blackout("15:35-15:55")], run_date="2026-09-20",
        )
    )
    assert seen["walk_deadline"] == fixed.replace(hour=15, minute=35)


def test_next_blackout_start_and_session_need_unreadable_partition(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 9, 20, 16, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    windows = [btt._parse_blackout("15:35-15:55"), btt._parse_blackout("20:00-20:30")]
    assert btt._next_blackout_start(now, windows) == now.replace(hour=20, minute=0)
    assert btt._next_blackout_start(now.replace(hour=21), windows) is None


def test_partition_index_degrades_on_unreadable_or_incomplete_partitions(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    regular = next(s for s in btt.TAPE_SESSIONS if s.session == "regular")
    tick_path = tick_partition_path(_DAY, "regular")
    tick_path.parent.mkdir(parents=True, exist_ok=True)
    tick_path.write_bytes(b"not a parquet file")
    assert btt.PartitionIndex().tick_stats(_DAY, regular) is None
    pd.DataFrame({"price": [1]}).to_parquet(tick_path)
    assert btt.PartitionIndex().tick_stats(_DAY, regular) is None

    from src.data.intraday_store import intraday_partition_path

    bar_path = intraday_partition_path(1, _DAY, "regular")
    bar_path.parent.mkdir(parents=True, exist_ok=True)
    bar_path.write_bytes(b"not a parquet file")
    assert btt.PartitionIndex().bar_volumes(_DAY, "regular") is None
    pd.DataFrame({"symbol": ["005930"]}).to_parquet(bar_path)
    assert btt.PartitionIndex().bar_volumes(_DAY, "regular") is None


def test_run_tasks_waits_out_a_blackout_that_starts_within_the_margin(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    clock = {"now": datetime(2026, 9, 20, 15, 33, tzinfo=ZoneInfo("Asia/Seoul"))}

    async def _advance(seconds: float) -> None:
        clock["now"] = clock["now"] + __import__("datetime").timedelta(seconds=seconds)

    started_at: list[datetime] = []

    async def _walk(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        started_at.append(clock["now"])
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(btt, "_now", lambda: clock["now"])
    monkeypatch.setattr(btt, "_sleep", _advance)
    monkeypatch.setattr(btt, "harvest_symbol_tape", _walk)
    sessions = tuple(s for s in btt.TAPE_SESSIONS if s.venue == "KRX")
    tasks = [btt.WalkTask(symbol="005930", venue="KRX", days=(_DAY,), sessions=sessions)]
    asyncio.run(
        btt._run_tasks(
            tasks, client=object(), http_session=object(), store=store, profile=profile,
            apply=False, ledger=tmp_path / "l.jsonl", deadline=None,
            blackouts=[btt._parse_blackout("15:35-15:55")], run_date="2026-09-20",
        )
    )
    assert started_at and started_at[0] >= datetime(2026, 9, 20, 15, 55, tzinfo=ZoneInfo("Asia/Seoul"))


def test_run_tasks_emits_a_heartbeat_every_fifty_walks(tmp_path, monkeypatch, caplog) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)

    async def _walk(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(btt, "harvest_symbol_tape", _walk)
    sessions = tuple(s for s in btt.TAPE_SESSIONS if s.venue == "KRX")
    tasks = [btt.WalkTask(symbol=f"{i:06d}", venue="KRX", days=(_DAY,), sessions=sessions) for i in range(50)]
    with caplog.at_level("INFO", logger="src.tools.backfill_tick_tape"):
        asyncio.run(
            btt._run_tasks(
                tasks, client=object(), http_session=object(), store=store, profile=profile,
                apply=False, ledger=tmp_path / "l.jsonl", deadline=None, blackouts=[], run_date="2026-09-20",
            )
        )
    beats = [rec.message for rec in caplog.records if "stage=tape_backfill done=" in rec.message]
    assert any("done=50/50" in beat for beat in beats) and len(beats) == 2
