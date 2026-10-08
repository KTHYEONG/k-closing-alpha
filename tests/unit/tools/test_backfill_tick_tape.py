"""Invariant guards for the one-shot tick tape backfill tool."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import src.backfill.intraday.tape_recovery as tr
import src.tools.backfill_tick_tape as btt
from src.backfill.intraday.tape_harvest import TAPE_SESSIONS, TapeDayResult, TapeWalkOutcome
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
_DAY = "2026-10-02"
_YMD = "20261002"


_PRODUCTION_BLACKOUTS = btt._DEFAULT_BLACKOUTS


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tr, "_price_history_path", lambda: tmp_path / "price_history.parquet")
    monkeypatch.setattr(btt, "_DEFAULT_BLACKOUTS", ())
    monkeypatch.setattr(tr, "_history_archive_path", lambda: tmp_path / "archive.parquet")


def _profile(tmp_path) -> CollectionSettings:
    return CollectionSettings(COLLECTION_ROOT=tmp_path / "capture")


def _patch_roots(tmp_path, monkeypatch) -> None:
    from src import settings as _settings

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    monkeypatch.setattr(btt, "_capture_root", lambda profile: tmp_path / "capture")
    monkeypatch.setattr(tr, "_capture_root", lambda profile: tmp_path / "capture")
    monkeypatch.setattr(btt, "_free_bytes", lambda path: 100 * 1024**3)
    monkeypatch.setattr(tr, "_free_bytes", lambda path: 100 * 1024**3)


def _tick_rows(hms_list: list[str], qty: str = "100", day: str = _YMD) -> list[dict]:
    return [{"cntr_tm": f"{day}{hms}", "cur_prc": "10000", "trde_qty": qty} for hms in hms_list]


def _bar_rows(hms_list: list[str], qty: str = "100", day: str = _YMD) -> list[dict]:
    return [
        {
            "cntr_tm": f"{day}{hms}",
            "cur_prc": "10000",
            "open_pric": "9900",
            "high_pric": "10100",
            "low_pric": "9800",
            "trde_qty": qty,
        }
        for hms in hms_list
    ]


def _complete_entry(symbol: str, session: str, rows: int) -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.TRADE_TICKS,
        venue="KRX",
        session=session,
        scheduled_at=None,
        status=CaptureStatus.COMPLETE,
        rows=rows,
        first_event_time=None,
        last_event_time=None,
        reason="seeded",
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
    monkeypatch.setattr(tr, "_day_universe", lambda day, store: ["005930", "000660"])
    _seed_tick_day(
        _DAY, "regular", "005930", _tick_rows(["090000", "090100"], "100"), _bar_rows(["090000", "090100"], "100")
    )
    _seed_tick_day(_DAY, "regular", "000660", _tick_rows(["090000"], "10"), _bar_rows(["090000", "090100"], "1000"))
    needs = tr.collect_tape_needs([_DAY], ["KRX"], store, {}, False)
    selected = {(n.symbol, n.session) for n in needs if n.day == _DAY}
    assert ("000660", "regular") in selected
    assert ("005930", "regular") not in selected


def test_missing_day_selected(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(tr, "_day_universe", lambda day, store: ["005930"])
    needs = tr.collect_tape_needs([_DAY], ["KRX"], store, {}, False)
    assert {(n.symbol, n.day, n.session) for n in needs} == {
        ("005930", _DAY, "regular"),
        ("005930", _DAY, "krx_aftermarket"),
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
    monkeypatch.setattr(tr, "harvest_symbol_tape", _fake_harvest_factory(calls))
    _patch_roots(tmp_path, monkeypatch)
    canned = [
        tr.Need(symbol="BBB", day="2026-09-02", session="regular", venue="KRX"),
        tr.Need(symbol="AAA", day="2026-09-01", session="regular", venue="KRX"),
        tr.Need(symbol="CCC", day="2026-09-15", session="regular", venue="KRX"),
        tr.Need(symbol="BBB", day="2026-09-03", session="regular", venue="KRX"),
        tr.Need(symbol="BBB", day="2026-09-04", session="regular", venue="KRX"),
    ]
    monkeypatch.setattr(tr, "collect_tape_needs", lambda *a, **k: list(canned))
    btt.main(
        [
            "--start",
            "2026-09-01",
            "--end",
            "2026-09-15",
            "--venue",
            "krx",
            "--apply",
            "--ledger",
            str(tmp_path / "ledger.jsonl"),
        ]
    )
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

    monkeypatch.setattr(tr, "_now", _fake_now)
    monkeypatch.setattr(btt, "_now", _fake_now)
    monkeypatch.setattr(tr, "_sleep", _fake_sleep)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(tr, "harvest_symbol_tape", _fake_harvest_factory(calls))
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.setattr(
        tr,
        "collect_tape_needs",
        lambda *a, **k: [tr.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    btt.main(
        [
            "--start",
            "2026-09-02",
            "--end",
            "2026-09-02",
            "--venue",
            "krx",
            "--apply",
            "--blackout",
            "20:00-20:30",
            "--ledger",
            str(tmp_path / "l.jsonl"),
        ]
    )
    assert len(slept) == 1 and slept[0] > 0
    assert len(calls) == 1


def test_deadline_stops_new_walks(tmp_path, monkeypatch, caplog) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(tr, "harvest_symbol_tape", _fake_harvest_factory(calls))
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.setattr(
        tr,
        "collect_tape_needs",
        lambda *a, **k: [tr.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    with caplog.at_level("INFO"):
        btt.main(
            [
                "--start",
                "2026-09-02",
                "--end",
                "2026-09-02",
                "--venue",
                "krx",
                "--apply",
                "--deadline",
                "2020-01-01T00:00:00+09:00",
                "--ledger",
                str(tmp_path / "l.jsonl"),
            ]
        )
    assert calls == []
    assert "remaining" in caplog.text


def test_resume_skips_settled(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    ledger = tmp_path / "l.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "".join(
            json.dumps({"symbol": "A", "day": "2026-10-02", "session": s, "status": "COMPLETE", "run_id": "r"}) + "\n"
            for s in ("regular", "krx_aftermarket")
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(tr, "_day_universe", lambda day, store: ["A", "B"])
    store = CaptureStore(tmp_path / "capture")
    settled = tr.read_settled_ledger(ledger)
    needs = tr.collect_tape_needs(["2026-10-02"], ["KRX"], store, settled, False)
    assert needs and all(n.symbol == "B" for n in needs)
    partial_ledger = tmp_path / "p.jsonl"
    partial_ledger.write_text(
        json.dumps({"symbol": "B", "day": "2026-10-02", "session": "regular", "status": "PARTIAL", "run_id": "r"})
        + "\n",
        encoding="utf-8",
    )
    needs2 = tr.collect_tape_needs(["2026-10-02"], ["KRX"], store, tr.read_settled_ledger(partial_ledger), False)
    assert any(n.symbol == "B" and n.session == "regular" for n in needs2)


def test_dry_run_writes_nothing(tmp_path, monkeypatch) -> None:
    from src.data.capture_store import CaptureStore as _Store

    _patch_roots(tmp_path, monkeypatch)
    calls: list[dict[str, Any]] = []

    async def _dry_harvest(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        calls.append({"symbol": symbol})
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=2, unresolved_days=())

    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(tr, "harvest_symbol_tape", _dry_harvest)
    monkeypatch.setattr(
        tr,
        "collect_tape_needs",
        lambda *a, **k: [tr.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
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
                symbol=symbol,
                day=day,
                session="regular",
                frame=pd.DataFrame(),
                entry=CoverageEntry(
                    symbol=symbol,
                    dataset=CaptureDataset.TRADE_TICKS,
                    venue="KRX",
                    session="regular",
                    scheduled_at=None,
                    status=CaptureStatus.PARTIAL,
                    rows=0,
                    first_event_time=None,
                    last_event_time=None,
                    reason="tape_total_mismatch:received=1:total=9",
                    raw_refs=(),
                ),
            )
            for day in days
        ]

    async def _guarded(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        for result in _emit(symbol, list(days)):
            kwargs["on_result"](result)
        return TapeWalkOutcome(termination_reason="page_budget", pages_fetched=1000, unresolved_days=tuple(days))

    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(tr, "harvest_symbol_tape", _guarded)
    monkeypatch.setattr(
        tr,
        "collect_tape_needs",
        lambda *a, **k: [tr.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    with caplog.at_level("INFO"):
        btt.main(
            [
                "--start",
                "2026-09-02",
                "--end",
                "2026-09-02",
                "--venue",
                "krx",
                "--apply",
                "--ledger",
                str(tmp_path / "l.jsonl"),
            ]
        )
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
    monkeypatch.setattr(tr, "harvest_symbol_tape", _fake_harvest_factory(calls))
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.setattr(
        tr,
        "collect_tape_needs",
        lambda *a, **k: [tr.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    btt.main(
        [
            "--start",
            "2026-09-02",
            "--end",
            "2026-09-02",
            "--venue",
            "krx",
            "--apply",
            "--ledger",
            str(tmp_path / "l.jsonl"),
        ]
    )
    assert len(calls) == 1


def test_storage_refusal(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.setattr(btt, "_free_bytes", lambda path: 0)
    monkeypatch.setattr(tr, "_free_bytes", lambda path: 0)
    monkeypatch.setattr(
        tr,
        "collect_tape_needs",
        lambda *a, **k: [tr.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
    )
    with pytest.raises(RuntimeError, match="storage budget"):
        btt.main(
            [
                "--start",
                "2026-09-02",
                "--end",
                "2026-09-02",
                "--venue",
                "krx",
                "--apply",
                "--ledger",
                str(tmp_path / "l.jsonl"),
            ]
        )


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
        tr.parse_walk_deadline("not-a-time")
    assert tr.parse_walk_deadline(None) is None
    assert btt._parse_blackout("09:00-10:00") == (540, 600)
    with pytest.raises(ValueError, match="blackout"):
        btt._parse_blackout("bogus")
    with pytest.raises(ValueError, match="blackout"):
        btt._parse_blackout("25:00-26:00")
    assert tr._in_blackout(30, (1380, 60)) is True
    assert tr._in_blackout(120, (1380, 60)) is False


def test_overnight_blackout_end() -> None:
    night = datetime(2026, 9, 10, 23, 30, tzinfo=_SEOUL)
    end = tr._blackout_end(night, [(1380, 60)])
    assert end is not None and (end.day, end.hour, end.minute) == (11, 1, 0)
    assert tr._blackout_end(night, [(540, 600)]) is None
    midnight = tr._blackout_end(night, [(1380, 1440)])
    assert midnight is not None and (midnight.day, midnight.hour) == (11, 0)
    early = datetime(2026, 9, 11, 0, 30, tzinfo=_SEOUL)
    same_day = tr._blackout_end(early, [(1380, 60)])
    assert same_day is not None and (same_day.day, same_day.hour, same_day.minute) == (11, 1, 0)
    asyncio.run(tr._sleep(0))


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
    assert tr._read_symbols(tmp_path / "absent.parquet") == []
    nosym = tmp_path / "nosym.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(nosym)
    assert tr._read_symbols(nosym) == []
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    try:
        garbage = tmp_path / "garbage.parquet"
        garbage.write_bytes(b"not a parquet file")
        assert tr._read_symbols(garbage) == []
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
    assert tr._read_cohort_symbols(store, "2026-09-05") == []
    _seed_tick_day("2026-09-03", "regular", "000003", _tick_rows(["090000"]), None)
    assert tr._day_universe("2026-09-03", store) == ["A", "B", "000003"]
    assert tr._day_universe("2026-09-05", store) == ["A"]
    close_rows = [
        {
            "cntr_tm": "20260920160000",
            "cur_prc": "10000",
            "open_pric": "9900",
            "high_pric": "10100",
            "low_pric": "9800",
            "trde_qty": "100",
        }
    ]
    bars = normalize_bar_frame(pd.DataFrame(close_rows), "kiwoom", "2026-09-20", "000004")
    write_intraday_partition(
        bars,
        1,
        "2026-09-20",
        "nxt_aftermarket",
        coverage={"000004": _complete_entry("000004", "nxt_aftermarket", len(bars))},
    )
    assert tr._day_universe("2026-09-20", store) == ["000004"]


def test_bar_volumes_index_branches(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    index = tr.PartitionIndex()
    assert index.bar_volumes(_DAY, "regular") is None
    _seed_tick_day(_DAY, "regular", "005930", _tick_rows(["090000"]), _bar_rows(["090000"]))
    fresh = tr.PartitionIndex()
    assert "OTHER" not in (fresh.bar_volumes(_DAY, "regular") or {})
    only_close_rows = [
        {
            "cntr_tm": "20260906153000",
            "cur_prc": "10000",
            "open_pric": "9900",
            "high_pric": "10100",
            "low_pric": "9800",
            "trde_qty": "100",
        }
    ]
    only_close = normalize_bar_frame(pd.DataFrame(only_close_rows), "kiwoom", "2026-09-06", "005930")
    write_intraday_partition(
        only_close,
        1,
        "2026-09-06",
        "regular",
        coverage={"005930": _complete_entry("005930", "regular", len(only_close))},
    )
    assert (fresh.bar_volumes("2026-09-06", "regular") or {}).get("005930") is None
    assert fresh.bar_volumes(_DAY, "regular")["005930"] == pytest.approx(100.0)


def test_partition_index_reads_each_partition_once(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    _seed_tick_day(_DAY, "regular", "005930", _tick_rows(["090000"]), _bar_rows(["090000"]))
    regular = next(s for s in TAPE_SESSIONS if s.session == "regular")
    reads = {"n": 0}
    real = pd.read_parquet

    def _counting(*args: Any, **kwargs: Any) -> Any:
        reads["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", _counting)
    index = tr.PartitionIndex()
    for symbol in ("005930", "000660", "035420", "051910"):
        tr._session_need(symbol, _DAY, regular, index)
    assert reads["n"] == 2  # one tick partition + one bar partition, regardless of symbol count


def test_session_need_flags(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    regular = next(s for s in TAPE_SESSIONS if s.session == "regular")
    frame = normalize_tick_frame(pd.DataFrame(_tick_rows(["090000"])), "kiwoom", _DAY, "005930")
    frame["truncated"] = True
    write_tick_partition(frame, _DAY, "regular", coverage={"005930": _complete_entry("005930", "regular", len(frame))})
    assert tr._session_need("005930", _DAY, regular, tr.PartitionIndex()) is True
    frame2 = normalize_tick_frame(
        pd.DataFrame(_tick_rows(["085000"], day="20260907")), "kiwoom", "2026-09-07", "005930"
    )
    write_tick_partition(
        frame2, "2026-09-07", "regular", coverage={"005930": _complete_entry("005930", "regular", len(frame2))}
    )
    assert tr._session_need("005930", "2026-09-07", regular, tr.PartitionIndex()) is True
    frame3 = normalize_tick_frame(
        pd.DataFrame(_tick_rows(["090000"], "10", day="20260908")), "kiwoom", "2026-09-08", "005930"
    )
    frame3["vendor"] = "ls"
    write_tick_partition(
        frame3, "2026-09-08", "regular", coverage={"005930": _complete_entry("005930", "regular", len(frame3))}
    )
    bars = normalize_bar_frame(
        pd.DataFrame(_bar_rows(["090000"], "1000", day="20260908")), "kiwoom", "2026-09-08", "005930"
    )
    write_intraday_partition(
        bars, 1, "2026-09-08", "regular", coverage={"005930": _complete_entry("005930", "regular", len(bars))}
    )
    assert tr._session_need("005930", "2026-09-08", regular, tr.PartitionIndex()) is True


def test_collect_skips_unclosed_and_session_closed_units(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(tr, "_day_universe", lambda day, store: ["005930"])
    assert tr.collect_tape_needs(["2999-01-01"], ["KRX", "NXT"], store, {}, False) == []
    assert tr.collect_tape_needs(["2999-01-01"], ["KRX"], store, {}, True) == []
    morning = datetime(2026, 9, 10, 10, 0, tzinfo=_SEOUL)
    evening = datetime(2026, 9, 10, 16, 0, tzinfo=_SEOUL)
    assert tr._session_closed("2026-09-10", "regular", morning) is False
    assert tr._session_closed("2026-09-10", "regular", evening) is True
    assert tr._session_closed("2026-09-10", "krx_aftermarket", evening) is False
    assert tr._session_closed("2026-09-09", "krx_aftermarket", morning) is True
    assert tr._session_closed("2026-09-10", "krx_aftermarket", datetime(2026, 9, 10, 20, 4, 59, tzinfo=_SEOUL)) is False
    assert tr._session_closed("2026-09-10", "krx_aftermarket", datetime(2026, 9, 10, 20, 5, 0, tzinfo=_SEOUL)) is True
    assert tr._session_closed("2026-09-10", "nxt_aftermarket", datetime(2026, 9, 10, 20, 5, 0, tzinfo=_SEOUL)) is True


def test_ledger_helpers(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    profile = _profile(tmp_path)
    assert tr.tape_ledger_path(None, profile) == tmp_path / "capture" / "staging" / "tape_backfill" / "ledger.jsonl"
    assert tr.tape_ledger_path(str(tmp_path / "x.jsonl"), profile) == tmp_path / "x.jsonl"
    assert tr.read_settled_ledger(tmp_path / "absent.jsonl") == {}
    ledger = tmp_path / "l.jsonl"
    ledger.write_text(
        "\nnot-json\n"
        + json.dumps({"symbol": "A", "day": "2026-09-02", "session": "regular", "status": "COMPLETE"})
        + "\n",
        encoding="utf-8",
    )
    assert tr.read_settled_ledger(ledger) == {("A", "2026-09-02", "regular"): "COMPLETE"}
    with pytest.raises(RuntimeError, match="unreadable"):
        tr.read_settled_ledger(tmp_path)
    tr._append_ledger(tmp_path / "noop.jsonl", [])
    assert not (tmp_path / "noop.jsonl").exists()
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ledger write failed"):
        tr._append_ledger(blocker / "l.jsonl", [{"symbol": "A"}])


def test_free_bytes_real(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.undo()
    assert btt._free_bytes(tmp_path) > 0


def test_verify_groups(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    _seed_tick_day(_DAY, "regular", "005930", _tick_rows(["090000"]), None)
    tr._verify_groups({(_DAY, "regular"): 1})
    with pytest.raises(RuntimeError, match="verification failed"):
        tr._verify_groups({(_DAY, "regular"): 99})
    corrupt = tick_partition_path("2026-09-09", "regular")
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"garbage")
    with pytest.raises(RuntimeError, match="publication failed"):
        tr._verify_groups({("2026-09-09", "regular"): 0})


def test_session_need_missing_symbol_in_partition(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    regular = next(s for s in TAPE_SESSIONS if s.session == "regular")
    _seed_tick_day(_DAY, "regular", "005930", _tick_rows(["090000"]), None)
    assert tr._session_need("000660", _DAY, regular, tr.PartitionIndex()) is True


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
                symbol=symbol,
                dataset=CaptureDataset.TRADE_TICKS,
                venue="KRX",
                session="regular",
                scheduled_at=None,
                status=CaptureStatus.COMPLETE,
                rows=len(frame),
                first_event_time=None,
                last_event_time=None,
                reason="tape_complete:regular=1:vendor_total=2",
                raw_refs=(ref,),
            )
            out.append(TapeDayResult(symbol=symbol, day=day, session="regular", frame=frame, entry=entry))
        return out

    async def _emitting(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        for result in _emit(symbol, list(days)):
            kwargs["on_result"](result)
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(btt, "_open_kiwoom", lambda: (object(), _SessionCtx()))
    monkeypatch.setattr(tr, "harvest_symbol_tape", _emitting)
    monkeypatch.setattr(
        tr,
        "collect_tape_needs",
        lambda *a, **k: [tr.Need(symbol="005930", day="2026-09-02", session="regular", venue="KRX")],
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

    monkeypatch.setattr(tr, "harvest_symbol_tape", _boom)
    task = tr.WalkTask(
        symbol="005930", venue="KRX", days=(_DAY,), sessions=tuple(s for s in TAPE_SESSIONS if s.venue == "KRX")
    )
    with pytest.raises(RuntimeError, match="evidence failed"):
        asyncio.run(
            tr.run_walk_tasks(
                [task],
                client=object(),
                http_session=object(),
                store=store,
                profile=profile,
                apply=True,
                ledger=tmp_path / "l.jsonl",
                deadline=None,
                blackouts=[],
                run_date="2026-09-20",
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

    monkeypatch.setattr(tr, "TickTapePublisher", _FakePublisher)
    monkeypatch.setattr(tr, "harvest_symbol_tape", _emit)
    monkeypatch.setattr(tr, "_verify_groups", lambda expected: None)
    sessions = tuple(s for s in TAPE_SESSIONS if s.venue == "KRX")
    tasks = [tr.WalkTask(symbol=f"{i:06d}", venue="KRX", days=(_DAY,), sessions=sessions) for i in range(3)]
    summary = asyncio.run(
        tr.run_walk_tasks(
            tasks,
            client=object(),
            http_session=object(),
            store=store,
            profile=profile,
            apply=True,
            ledger=ledger,
            deadline=None,
            blackouts=[],
            run_date="2026-09-20",
        )
    )
    assert summary.pages == 3
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
        kwargs["on_result"](
            TapeDayResult(
                symbol=symbol,
                day=_DAY,
                session="regular",
                frame=frame,
                entry=_complete_entry(symbol, "regular", len(frame)),
            )
        )
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    def _boom(self: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(tr, "harvest_symbol_tape", _emit)
    monkeypatch.setattr(tr.TickTapePublisher, "flush", _boom)
    sessions = tuple(s for s in TAPE_SESSIONS if s.venue == "KRX")
    task = tr.WalkTask(symbol="005930", venue="KRX", days=(_DAY,), sessions=sessions)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(
            tr.run_walk_tasks(
                [task],
                client=object(),
                http_session=object(),
                store=store,
                profile=profile,
                apply=True,
                ledger=ledger,
                deadline=None,
                blackouts=[],
                run_date="2026-09-20",
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
    monkeypatch.setattr(tr, "harvest_symbol_tape", _fake_harvest_factory(calls))
    canned = [
        tr.Need(symbol="AAA", day="2026-09-02", session="regular", venue="KRX"),
        tr.Need(symbol="BBB", day="2026-09-02", session="regular", venue="KRX"),
        tr.Need(symbol="CCC", day="2026-09-02", session="regular", venue="KRX"),
    ]
    monkeypatch.setattr(tr, "collect_tape_needs", lambda *a, **k: list(canned))
    nxt_only = [tr.Need(symbol="AAA", day="2026-09-02", session="nxt_aftermarket", venue="NXT")]
    monkeypatch.setattr(tr, "collect_tape_needs", lambda *a, **k: list(nxt_only))
    btt.main(
        [
            "--start",
            "2026-09-02",
            "--end",
            "2026-09-02",
            "--venue",
            "nxt",
            "--apply",
            "--ledger",
            str(tmp_path / "l.jsonl"),
        ]
    )
    assert [c["venue"] for c in calls] == ["NXT"]
    calls.clear()
    monkeypatch.setattr(tr, "collect_tape_needs", lambda *a, **k: list(canned))
    btt.main(
        [
            "--start",
            "2026-09-02",
            "--end",
            "2026-09-02",
            "--venue",
            "krx",
            "--apply",
            "--symbols-limit",
            "2",
            "--ledger",
            str(tmp_path / "l2.jsonl"),
        ]
    )
    assert [c["symbol"] for c in calls] == ["AAA", "BBB"]
    monkeypatch.setattr(tr, "collect_tape_needs", lambda *a, **k: [])
    with caplog.at_level("INFO"):
        btt.main(
            ["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--ledger", str(tmp_path / "l3.jsonl")]
        )
    assert "NOOP" in caplog.text
    monkeypatch.setattr(tr, "collect_tape_needs", lambda *a, **k: list(canned)[:1])
    monkeypatch.setattr(btt, "_free_bytes", lambda path: (_ for _ in ()).throw(OSError("disk")))
    monkeypatch.setattr(tr, "_free_bytes", lambda path: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(RuntimeError, match="storage check failed"):
        btt.main(
            ["--start", "2026-09-02", "--end", "2026-09-02", "--venue", "krx", "--ledger", str(tmp_path / "l4.jsonl")]
        )
    monkeypatch.setattr(btt, "_free_bytes", lambda path: 100 * 1024**3)

    def _raise_open() -> Any:
        raise OSError("auth down")

    monkeypatch.setattr(btt, "_open_kiwoom", _raise_open)
    with pytest.raises(RuntimeError, match="infrastructure failed"):
        btt.main(
            [
                "--start",
                "2026-09-02",
                "--end",
                "2026-09-02",
                "--venue",
                "krx",
                "--apply",
                "--ledger",
                str(tmp_path / "l5.jsonl"),
            ]
        )


def test_run_tasks_stops_cleanly_on_disk_guard(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    walked: list[str] = []

    async def _walk(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        walked.append(symbol)
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(tr, "harvest_symbol_tape", _walk)
    monkeypatch.setattr(btt, "_free_bytes", lambda path: 0)
    monkeypatch.setattr(tr, "_free_bytes", lambda path: 0)
    purged: list[Any] = []
    import src.tools.backup_prune as prune

    monkeypatch.setattr(prune, "prune_local_intraday_backups", lambda **kw: purged.append(kw) or [])
    sessions = tuple(s for s in TAPE_SESSIONS if s.venue == "KRX")
    tasks = [tr.WalkTask(symbol="005930", venue="KRX", days=(_DAY,), sessions=sessions)]
    summary = asyncio.run(
        tr.run_walk_tasks(
            tasks,
            client=object(),
            http_session=object(),
            store=store,
            profile=profile,
            apply=True,
            ledger=tmp_path / "l.jsonl",
            deadline=None,
            blackouts=[],
            run_date="2026-09-20",
        )
    )
    assert walked == []
    assert summary.stopped_reason == "disk_guard" and summary.remaining == ("005930/KRX",)
    assert len(purged) == 1  # exactly one expired-snapshot prune attempt before stopping


def test_run_tasks_bounds_each_walk_by_the_next_blackout(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    seen: dict[str, Any] = {}
    fixed = datetime(2026, 9, 20, 10, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    monkeypatch.setattr(tr, "_now", lambda: fixed)

    async def _walk(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        seen["walk_deadline"] = kwargs.get("walk_deadline")
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(tr, "harvest_symbol_tape", _walk)
    sessions = tuple(s for s in TAPE_SESSIONS if s.venue == "KRX")
    tasks = [tr.WalkTask(symbol="005930", venue="KRX", days=(_DAY,), sessions=sessions)]
    asyncio.run(
        tr.run_walk_tasks(
            tasks,
            client=object(),
            http_session=object(),
            store=store,
            profile=profile,
            apply=False,
            ledger=tmp_path / "l.jsonl",
            deadline=None,
            blackouts=[btt._parse_blackout("15:35-15:55")],
            run_date="2026-09-20",
        )
    )
    assert seen["walk_deadline"] == fixed.replace(hour=15, minute=35)


def test_next_blackout_start_and_session_need_unreadable_partition(tmp_path, monkeypatch) -> None:
    now = datetime(2026, 9, 20, 16, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    windows = [btt._parse_blackout("15:35-15:55"), btt._parse_blackout("20:00-20:30")]
    assert tr._next_blackout_start(now, windows) == now.replace(hour=20, minute=0)
    assert tr._next_blackout_start(now.replace(hour=21), windows) is None


def test_partition_index_degrades_on_unreadable_or_incomplete_partitions(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    regular = next(s for s in TAPE_SESSIONS if s.session == "regular")
    tick_path = tick_partition_path(_DAY, "regular")
    tick_path.parent.mkdir(parents=True, exist_ok=True)
    tick_path.write_bytes(b"not a parquet file")
    assert tr.PartitionIndex().tick_stats(_DAY, regular) is None
    pd.DataFrame({"price": [1]}).to_parquet(tick_path)
    assert tr.PartitionIndex().tick_stats(_DAY, regular) is None

    from src.data.intraday_store import intraday_partition_path

    bar_path = intraday_partition_path(1, _DAY, "regular")
    bar_path.parent.mkdir(parents=True, exist_ok=True)
    bar_path.write_bytes(b"not a parquet file")
    assert tr.PartitionIndex().bar_volumes(_DAY, "regular") is None
    pd.DataFrame({"symbol": ["005930"]}).to_parquet(bar_path)
    assert tr.PartitionIndex().bar_volumes(_DAY, "regular") is None


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

    monkeypatch.setattr(tr, "_now", lambda: clock["now"])
    monkeypatch.setattr(tr, "_sleep", _advance)
    monkeypatch.setattr(tr, "harvest_symbol_tape", _walk)
    sessions = tuple(s for s in TAPE_SESSIONS if s.venue == "KRX")
    tasks = [tr.WalkTask(symbol="005930", venue="KRX", days=(_DAY,), sessions=sessions)]
    asyncio.run(
        tr.run_walk_tasks(
            tasks,
            client=object(),
            http_session=object(),
            store=store,
            profile=profile,
            apply=False,
            ledger=tmp_path / "l.jsonl",
            deadline=None,
            blackouts=[btt._parse_blackout("15:35-15:55")],
            run_date="2026-09-20",
        )
    )
    assert started_at and started_at[0] >= datetime(2026, 9, 20, 15, 55, tzinfo=ZoneInfo("Asia/Seoul"))


def test_run_tasks_emits_a_heartbeat_every_fifty_walks(tmp_path, monkeypatch, caplog) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)

    async def _walk(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(tr, "harvest_symbol_tape", _walk)
    sessions = tuple(s for s in TAPE_SESSIONS if s.venue == "KRX")
    tasks = [tr.WalkTask(symbol=f"{i:06d}", venue="KRX", days=(_DAY,), sessions=sessions) for i in range(50)]
    with caplog.at_level("INFO", logger="src.backfill.intraday.tape_recovery"):
        asyncio.run(
            tr.run_walk_tasks(
                tasks,
                client=object(),
                http_session=object(),
                store=store,
                profile=profile,
                apply=False,
                ledger=tmp_path / "l.jsonl",
                deadline=None,
                blackouts=[],
                run_date="2026-09-20",
            )
        )
    beats = [rec.message for rec in caplog.records if "stage=tape_backfill done=" in rec.message]
    assert any("done=50/50" in beat for beat in beats) and len(beats) == 2


def test_collect_skips_weekends_and_ingested_holidays(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(tr, "_day_universe", lambda day, store: ["005930"])
    ingested = pd.DataFrame({"date": pd.to_datetime(["2026-09-23", "2026-09-28"]), "symbol": "005930"})
    ingested.to_parquet(tmp_path / "price_history.parquet")
    days = ["2026-09-23", "2026-09-24", "2026-09-26", "2026-09-28", "2026-09-30"]
    needs = tr.collect_tape_needs(days, ["KRX"], store, {}, False)
    assert {n.day for n in needs} == {"2026-09-23", "2026-09-28", "2026-09-30"}


def test_trading_day_falls_back_to_weekday_without_price_history() -> None:
    assert tr._is_trading_day("2026-10-02", set()) is True
    assert tr._is_trading_day("2026-09-26", set()) is False
    assert tr._is_trading_day("2026-09-24", set()) is False


def test_ingested_trading_days_tolerates_unreadable_file(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    assert tr._ingested_trading_days() == set()
    (tmp_path / "price_history.parquet").write_bytes(b"not parquet")
    assert tr._ingested_trading_days() == set()


def test_price_history_path_points_at_configured_parquet(monkeypatch) -> None:
    monkeypatch.undo()
    from src import settings as _settings

    assert tr._price_history_path() == Path(_settings.PRICE_HISTORY_PARQUET_PATH)


class _TokenClient:
    def __init__(self) -> None:
        self.resets = 0

    def reset_token(self) -> None:
        self.resets += 1


def _failure_tasks(count: int) -> list[Any]:
    sessions = tuple(s for s in TAPE_SESSIONS if s.venue == "KRX")
    return [tr.WalkTask(symbol=f"00000{i}", venue="KRX", days=(_DAY,), sessions=sessions) for i in range(count)]


def test_vendor_failure_resets_token_and_a_single_failure_does_not_stop(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    outcomes = iter(
        [
            TapeWalkOutcome(termination_reason="vendor_failure", pages_fetched=1, unresolved_days=(_DAY,)),
            TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=2, unresolved_days=()),
        ]
    )

    async def _fake(*args: Any, **kwargs: Any) -> Any:
        return next(outcomes)

    monkeypatch.setattr(tr, "harvest_symbol_tape", _fake)
    client = _TokenClient()
    summary = asyncio.run(
        tr.run_walk_tasks(
            _failure_tasks(2),
            client=client,
            http_session=object(),
            store=CaptureStore(tmp_path / "capture"),
            profile=_profile(tmp_path),
            apply=False,
            ledger=tmp_path / "l.jsonl",
            deadline=None,
            blackouts=[],
            run_date="2026-09-20",
        )
    )
    assert client.resets == 1
    assert summary.stopped_reason == ""
    assert summary.remaining == ()


def test_vendor_failure_streak_stops_run(tmp_path, monkeypatch, caplog) -> None:
    _patch_roots(tmp_path, monkeypatch)

    async def _fake(*args: Any, **kwargs: Any) -> Any:
        return TapeWalkOutcome(termination_reason="vendor_failure", pages_fetched=1, unresolved_days=(_DAY,))

    monkeypatch.setattr(tr, "harvest_symbol_tape", _fake)
    client = _TokenClient()
    with caplog.at_level("WARNING"):
        summary = asyncio.run(
            tr.run_walk_tasks(
                _failure_tasks(6),
                client=client,
                http_session=object(),
                store=CaptureStore(tmp_path / "capture"),
                profile=_profile(tmp_path),
                apply=False,
                ledger=tmp_path / "l.jsonl",
                deadline=None,
                blackouts=[],
                run_date="2026-09-20",
            )
        )
    assert client.resets == tr._MAX_VENDOR_FAILURE_STREAK
    assert summary.stopped_reason == "vendor_failure"
    assert len(summary.remaining) == 6 - tr._MAX_VENDOR_FAILURE_STREAK
    assert "VENDOR_FAILURE_STREAK" in caplog.text


def test_default_blackouts_cover_kiwoom_live_windows() -> None:
    windows = [btt._parse_blackout(spec) for spec in _PRODUCTION_BLACKOUTS]

    def covered(hhmm: str) -> bool:
        minute = int(hhmm[:2]) * 60 + int(hhmm[3:])
        return any(tr._in_blackout(minute, window) for window in windows)

    for live in ("08:30", "11:30", "15:20", "15:21", "15:40", "16:25", "20:05", "21:02", "21:30", "23:05"):
        assert covered(live), live
    assert not covered("13:00")


def _write_archive(tmp_path, rows: dict[str, list[str]]) -> None:
    frame = pd.DataFrame(
        [{"스냅샷_날짜": pd.Timestamp(day), "종목코드": code} for day, codes in rows.items() for code in codes]
    )
    frame.to_parquet(tmp_path / "archive.parquet", index=False)


def _write_prices(tmp_path, day: str, rows: list[tuple[str, float, float, float]]) -> None:
    frame = pd.DataFrame(
        [
            {"date": pd.Timestamp(day), "symbol": sym, "close": close, "prev_close": prev, "trade_value_100m": tv}
            for sym, close, prev, tv in rows
        ]
    )
    frame.to_parquet(tmp_path / "price_history.parquet", index=False)


def test_pool_symbols_prefers_archive_pool_from_the_universe_start(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    _write_archive(tmp_path, {"2026-09-11": ["5930", "000660"], "2026-09-03": ["111111"]})
    assert tr._pool_symbols(store, "2026-09-11") == (["005930", "000660"], "archive_pool")
    assert tr._pool_symbols(store, "2026-09-03") == ([], "none")


def test_pool_symbols_reconstructs_missing_day_from_band_and_top_trade_value(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    monkeypatch.setattr(tr, "RECONSTRUCTED_TOP_TRADE_VALUE", 1)
    store = CaptureStore(tmp_path / "capture")
    _write_prices(
        tmp_path,
        "2026-09-14",
        [
            ("000001", 103.0, 100.0, 1.0),
            ("000002", 100.0, 100.0, 50.0),
            ("000003", 120.0, 100.0, 2.0),
            ("000004", 99.0, 100.0, 3.0),
        ],
    )
    symbols, source = tr._pool_symbols(store, "2026-09-14")
    assert source == "reconstructed"
    assert symbols == ["000001", "000002"]


def test_pool_symbols_does_not_reconstruct_before_universe_start_or_without_prices(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    _write_prices(tmp_path, "2026-09-14", [("000001", 103.0, 100.0, 1.0)])
    assert tr._pool_symbols(store, "2026-09-10") == ([], "none")
    assert tr._pool_symbols(store, "2026-09-15") == ([], "none")


def test_day_universe_merges_own_and_previous_pool_with_provenance(tmp_path, monkeypatch, caplog) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    _write_archive(tmp_path, {"2026-09-11": ["000001"], "2026-09-15": ["000009"]})
    _write_prices(tmp_path, "2026-09-14", [("000005", 103.0, 100.0, 1.0)])
    with caplog.at_level("INFO"):
        assert tr._day_universe("2026-09-14", store) == ["000005", "000001"]
        assert tr._day_universe("2026-09-15", store) == ["000009", "000005"]
        assert tr._day_universe("2026-09-11", store) == ["000001"]
    assert "own=reconstructed:1 prev=2026-09-11:archive_pool" in caplog.text
    assert "own=archive_pool:1 prev=2026-09-14:reconstructed" in caplog.text


def test_unreadable_archive_and_price_history_degrade_to_no_pool(tmp_path, monkeypatch, caplog) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    (tmp_path / "archive.parquet").write_bytes(b"not parquet")
    (tmp_path / "price_history.parquet").write_bytes(b"not parquet")
    with caplog.at_level("WARNING"):
        assert tr._pool_symbols(store, "2026-09-14") == ([], "none")
    assert "archive_unreadable" in caplog.text and "price_history_unreadable" in caplog.text


def test_history_archive_path_points_at_configured_archive(monkeypatch) -> None:
    monkeypatch.undo()
    from src import settings as _settings

    assert tr._history_archive_path() == Path(_settings.HISTORY_PARQUET_PATH)


def _volume_need(tmp_path, monkeypatch, session: str, bar_volume: float, tick_volume: float) -> bool:
    import pandas as pd

    from src.data.intraday_schema import normalize_bar_frame, normalize_tick_frame
    from src.data.intraday_store import write_intraday_partition, write_tick_partition

    day = "2026-09-30"
    ymd = day.replace("-", "")
    symbol = "005930"
    spec = next(s for s in TAPE_SESSIONS if s.session == session)
    hms = "090000" if session == "regular" else "160000"
    tick_rows = pd.DataFrame([{"cntr_tm": f"{ymd}{hms}", "cur_prc": "10000", "trde_qty": str(int(tick_volume))}])
    bar_rows = pd.DataFrame(
        [
            {
                "cntr_tm": f"{ymd}{hms}",
                "cur_prc": "10000",
                "open_pric": "9900",
                "high_pric": "10100",
                "low_pric": "9800",
                "trde_qty": str(int(bar_volume)),
            }
        ]
    )
    tick_frame = normalize_tick_frame(tick_rows, "kiwoom", day, symbol)
    bar_frame = normalize_bar_frame(bar_rows, "kiwoom", day, symbol)
    write_tick_partition(tick_frame, day, session, coverage={symbol: _complete_entry(symbol, session, len(tick_frame))})
    write_intraday_partition(
        bar_frame, 1, day, session, coverage={symbol: _complete_entry(symbol, session, len(bar_frame))}
    )
    return tr._session_need(symbol, day, spec, tr.PartitionIndex())


def test_aftermarket_need_on_any_shortfall(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    assert _volume_need(tmp_path, monkeypatch, "krx_aftermarket", 1_000, 999) is True
    assert _volume_need(tmp_path, monkeypatch, "nxt_aftermarket", 1_000, 999) is True


def test_aftermarket_surplus_is_not_a_need(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    assert _volume_need(tmp_path, monkeypatch, "krx_aftermarket", 1_000, 1_002) is False


def test_regular_need_threshold_unchanged(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    assert _volume_need(tmp_path, monkeypatch, "regular", 10_000, 9_950) is False
    assert _volume_need(tmp_path, monkeypatch, "regular", 10_000, 9_850) is True


def test_session_need_existing_causes_still_hold(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    regular = next(s for s in TAPE_SESSIONS if s.session == "regular")
    # Missing tick partition is a need.
    assert tr._session_need("005930", "2026-09-30", regular, tr.PartitionIndex()) is True
    # Zero rows are a need regardless of volume.
    index = tr.PartitionIndex()
    index._ticks[("2026-09-30", "regular")] = {
        "005930": tr._TickStats(rows=0, truncated=False, out_of_window=False, volume=10000.0)
    }
    index._bars[("2026-09-30", "regular")] = {"005930": 10000.0}
    assert tr._session_need("005930", "2026-09-30", regular, index) is True


def test_cli_exposes_no_engine_aliases() -> None:
    for name in (
        "_collect_needs",
        "_order_tasks",
        "_run_tasks",
        "_read_settled",
        "_ledger_path",
        "_parse_deadline",
        "PartitionIndex",
        "_CLOSE_AUCTION_TS",
        "_REGULAR_READY_HHMMSS",
        "_TICK_SESSIONS",
    ):
        assert hasattr(btt, name) is False, name


def _write_ingested_days(tmp_path, days: list[str]) -> None:
    pd.DataFrame({"date": pd.to_datetime(days), "symbol": "005930"}).to_parquet(
        tmp_path / "price_history.parquet"
    )


def test_holiday_after_newest_ingested_is_not_trading(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    _write_ingested_days(tmp_path, ["2026-09-30", "2026-10-01", "2026-10-02"])
    ingested = tr._ingested_trading_days()
    assert max(ingested) == "2026-10-02"
    assert tr._is_trading_day("2026-10-05", ingested) is False
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(tr, "_day_universe", lambda day, store: ["005930"])
    assert tr.collect_tape_needs(["2026-10-05"], ["KRX", "NXT"], store, {}, False) == []


def test_weekday_after_newest_ingested_is_trading_when_standard(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    _write_ingested_days(tmp_path, ["2026-09-30", "2026-10-01", "2026-10-02"])
    ingested = tr._ingested_trading_days()
    assert tr._is_trading_day("2026-10-06", ingested) is True


def test_calendar_unknown_requires_ingested_evidence() -> None:
    assert tr._is_trading_day("2026-09-24", set()) is False
    assert tr._is_trading_day("2026-09-24", {"2026-09-24"}) is True


def test_price_history_still_excludes_in_range_holidays() -> None:
    ingested = {"2026-10-01", "2026-10-06"}
    assert tr._is_trading_day("2026-10-02", ingested) is False


def test_calendar_closed_beats_price_history() -> None:
    assert tr._is_trading_day("2026-10-05", {"2026-10-02", "2026-10-05"}) is False
    with pytest.raises(ValueError, match="isoformat"):
        tr._is_trading_day("not-a-date", set())


def test_task_pairs_partition_the_needs() -> None:
    needs = [
        tr.Need(symbol="A", day="2026-10-01", session="regular", venue="KRX"),
        tr.Need(symbol="A", day="2026-09-30", session="krx_aftermarket", venue="KRX"),
        tr.Need(symbol="A", day="2026-10-01", session="regular", venue="KRX"),
        tr.Need(symbol="B", day="2026-10-01", session="nxt_aftermarket", venue="NXT"),
        tr.Need(symbol="B", day="2026-10-01", session="regular", venue="KRX"),
    ]
    tasks = tr.order_walk_tasks(needs)
    union: set[tuple[str, str, str, str]] = set()
    for task in tasks:
        assert task.pairs == tuple(sorted(set(task.pairs)))
        assert len(task.pairs) == len(set(task.pairs))
        for pair in task.pairs:
            key = (task.symbol, task.venue, *pair)
            assert key not in union
            union.add(key)
    assert union == {(n.symbol, n.venue, n.day, n.session) for n in needs}
    by_key = {(t.symbol, t.venue): t for t in tasks}
    assert by_key[("A", "KRX")].pairs == (("2026-09-30", "krx_aftermarket"), ("2026-10-01", "regular"))
    assert by_key[("A", "KRX")].days == ("2026-09-30", "2026-10-01")


def test_publisher_guard_names_symbol_day_session(tmp_path, monkeypatch) -> None:
    from src.backfill.intraday.tape_harvest import TickTapePublisher

    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    from datetime import timedelta

    tomorrow = (datetime.now(_SEOUL).date() + timedelta(days=1)).isoformat()
    bad = TapeDayResult(
        symbol="005930",
        day=tomorrow,
        session="krx_aftermarket",
        frame=pd.DataFrame(),
        entry=CoverageEntry(
            symbol="005930",
            dataset=CaptureDataset.TRADE_TICKS,
            venue="KRX",
            session="krx_aftermarket",
            scheduled_at=None,
            status=CaptureStatus.NO_TRADES,
            rows=0,
            first_event_time=None,
            last_event_time=None,
            reason="tape_complete:krx_aftermarket=0:vendor_total=1",
            raw_refs=(ArtifactRef(path="raw/seed", sha256="abc", bytes=1),),
        ),
    )
    pub = TickTapePublisher(store=store, profile=profile, flush_rows=10**9)
    with pytest.raises(ValueError, match="not closed") as excinfo:
        pub.add(bad)
    message = str(excinfo.value)
    assert "005930" in message and tomorrow in message and "krx_aftermarket" in message


def test_run_walk_tasks_passes_pairs_and_survives_holiday_shape(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    holiday = datetime(2026, 10, 5, 21, 0, tzinfo=_SEOUL)
    monkeypatch.setattr(tr, "_now", lambda: holiday)
    _write_ingested_days(tmp_path, ["2026-09-30", "2026-10-01", "2026-10-02"])
    monkeypatch.setattr(tr, "_day_universe", lambda day, store: ["005930"] if day == "2026-09-30" else [])
    needs = tr.collect_tape_needs(["2026-09-30", "2026-10-05"], ["KRX"], store, {}, False)
    assert needs and all(n.day == "2026-09-30" for n in needs)
    tasks = tr.order_walk_tasks(needs)
    assert tasks and all(t.pairs for t in tasks)
    seen: list[dict[str, Any]] = []

    async def _fake(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        seen.append({"needed": kwargs.get("needed")})
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=0, unresolved_days=())

    monkeypatch.setattr(tr, "harvest_symbol_tape", _fake)
    summary = asyncio.run(
        tr.run_walk_tasks(
            tasks,
            client=object(),
            http_session=object(),
            store=store,
            profile=profile,
            apply=False,
            ledger=tmp_path / "l.jsonl",
            deadline=None,
            blackouts=[],
            run_date="2026-10-05",
        )
    )
    assert summary.stopped is False
    assert seen and all(item["needed"] for item in seen)
    flat = {pair for item in seen for pair in item["needed"]}
    assert ("2026-10-05", "krx_aftermarket") not in flat
    assert ("2026-10-05", "regular") not in flat


def test_run_walk_tasks_wraps_unclosed_publication_with_task_index(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    from datetime import timedelta

    tomorrow = (datetime.now(_SEOUL).date() + timedelta(days=1)).isoformat()

    async def _emit_unclosed(client: Any, session: Any, symbol: str, days: Any, **kwargs: Any) -> Any:
        frame = normalize_tick_frame(
            pd.DataFrame(_tick_rows(["090000"])), "kiwoom", tomorrow, symbol
        )
        kwargs["on_result"](
            TapeDayResult(
                symbol=symbol,
                day=tomorrow,
                session="krx_aftermarket",
                frame=frame,
                entry=_complete_entry(symbol, "krx_aftermarket", len(frame)),
            )
        )
        return TapeWalkOutcome(termination_reason="crossed_stop_day", pages_fetched=1, unresolved_days=())

    monkeypatch.setattr(tr, "harvest_symbol_tape", _emit_unclosed)
    task = tr.WalkTask(
        symbol="005930",
        venue="KRX",
        days=(tomorrow,),
        sessions=tuple(s for s in TAPE_SESSIONS if s.venue == "KRX"),
        pairs=((tomorrow, "krx_aftermarket"),),
    )
    with pytest.raises(ValueError, match="task=0"):
        asyncio.run(
            tr.run_walk_tasks(
                [task],
                client=object(),
                http_session=object(),
                store=store,
                profile=profile,
                apply=True,
                ledger=tmp_path / "l.jsonl",
                deadline=None,
                blackouts=[],
                run_date="2026-10-05",
            )
        )


def test_collect_includes_today_aftermarket_only_after_ready(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    day = "2026-10-02"
    _seed_tick_day(day, "krx_aftermarket", "000660", _tick_rows(["170000"]), None)
    monkeypatch.setattr(tr, "_day_universe", lambda d, s: ["005930", "000660"] if d == day else [])
    monkeypatch.setattr(tr, "_is_trading_day", lambda d, ing: True)
    monkeypatch.setattr(tr, "_now", lambda: datetime(2026, 10, 2, 20, 0, tzinfo=_SEOUL))
    assert not [n for n in tr.collect_tape_needs([day], ["KRX"], store, {}, False) if n.session == "krx_aftermarket"]
    monkeypatch.setattr(tr, "_now", lambda: datetime(2026, 10, 2, 20, 36, tzinfo=_SEOUL))
    assert ("005930", day, "krx_aftermarket") in {(n.symbol, n.day, n.session) for n in tr.collect_tape_needs([day], ["KRX"], store, {}, False)}
    assert not any(n.symbol == "000660" and n.session == "krx_aftermarket" for n in tr.collect_tape_needs([day], ["KRX"], store, {}, False))


@pytest.mark.parametrize("explicit_needed", [False, True])
def test_harvest_accepts_today_aftermarket_only_after_ready(tmp_path, monkeypatch, explicit_needed) -> None:
    import datetime as _dt

    import src.backfill.intraday.tape_harvest as th

    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    day = "2026-10-02"
    rows = [{"cntr_tm": f"{day.replace('-', '')}170000", "cur_prc": "10000", "trde_qty": "10"}]
    real_dt = _dt.datetime

    def _run_at(hour: int, minute: int) -> Any:
        fixed = real_dt(2026, 10, 2, hour, minute, tzinfo=_SEOUL)

        class _FakeDT(real_dt):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                return fixed.astimezone(tz) if tz is not None else fixed.replace(tzinfo=None)

        monkeypatch.setattr(th, "datetime", _FakeDT)

        class _Stub:
            async def walk_tick_tape(self, *a: Any, **k: Any) -> Any:
                for cb in (k.get("on_day_complete"),):
                    if cb is not None:
                        from src.api.kiwoom.client import TapeDayCertificate

                        cb(day, rows, TapeDayCertificate(day=day, received=1, vendor_total=2, complete=True, basis="vendor_total"))
                return {"termination_reason": "crossed_stop_day", "pages_fetched": 1, "certificates": []}

        out: list[TapeDayResult] = []
        res = asyncio.run(
            th.harvest_symbol_tape(
                _Stub(), object(), "005930", [day], venue="KRX",
                sessions=tuple(s for s in th.TAPE_SESSIONS if s.session == "krx_aftermarket"),
                store=store, run_id="tape-test", profile=profile, on_result=out.append,
                needed=[(day, "krx_aftermarket")] if explicit_needed else None,
            )
        )
        return out, res

    out_early, res_early = _run_at(19, 59)
    assert ((day, "krx_aftermarket") in res_early.skipped_unclosed) and not out_early
    out_late, res_late = _run_at(20, 36)
    assert res_late.skipped_unclosed == () and {(r.day, r.session) for r in out_late} == {(day, "krx_aftermarket")}
    assert out_late[0].entry.reason.startswith("tape_complete:")


@pytest.mark.parametrize(("empty", "include_regular"), [(False, False), (True, False), (True, True)])
def test_incomplete_same_day_tape_stays_partial(tmp_path, monkeypatch, empty, include_regular) -> None:
    import datetime as _dt

    import src.backfill.intraday.tape_harvest as th

    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    day = "2026-10-02"
    _patch_roots(tmp_path, monkeypatch)
    real_dt = _dt.datetime
    fixed = real_dt(2026, 10, 2, 20, 36, tzinfo=_SEOUL)

    class _FakeDT(real_dt):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return fixed.astimezone(tz) if tz is not None else fixed.replace(tzinfo=None)

    monkeypatch.setattr(th, "datetime", _FakeDT)

    from src.api.kiwoom.client import TapeDayCertificate

    class _MismatchStub:
        async def walk_tick_tape(self, *a: Any, **k: Any) -> Any:
            k["on_page"]({"stk_tic_chart_qry": []}, {}, fixed, fixed, 0, 0)
            return {
                "termination_reason": "tape_empty" if empty else "crossed_stop_day",
                "pages_fetched": 1,
                "certificates": [] if empty else [TapeDayCertificate(day=day, received=1, vendor_total=5, complete=False)],
            }

    out: list[TapeDayResult] = []
    res = asyncio.run(
        th.harvest_symbol_tape(
            _MismatchStub(), object(), "005930", [day], venue="KRX",
            sessions=tuple(s for s in th.TAPE_SESSIONS if s.venue == "KRX"),
            store=store, run_id="tape-test", profile=profile, on_result=out.append,
            needed=[(day, "krx_aftermarket"), (day, "regular")] if include_regular else [(day, "krx_aftermarket")],
        )
    )
    assert res.skipped_unclosed == ()
    assert out[0].session == "krx_aftermarket" and out[0].entry.status.value == "PARTIAL"
    if include_regular:
        assert len(out) == 2 and out[1].entry.status == CaptureStatus.NO_TRADES
    else:
        assert len(out) == 1
    assert "tape_total_mismatch" in out[0].entry.reason
    assert res.unresolved_days == (day,)
    publisher = th.TickTapePublisher(store=store, profile=profile, flush_rows=100)
    publisher.add(out[0])
    assert publisher.flush().partitions_written == 0
    assert not tick_partition_path(day, "krx_aftermarket").exists()
    ledger = tmp_path / "ledger.jsonl"
    tr._append_ledger(ledger, [{
        "symbol": "005930", "day": day, "session": "krx_aftermarket",
        "status": out[0].entry.status.value, "reason": out[0].entry.reason, "run_date": day,
    }])
    monkeypatch.setattr(tr, "_now", lambda: fixed.replace(day=3))
    monkeypatch.setattr(tr, "_day_universe", lambda d, s: ["005930"])
    monkeypatch.setattr(tr, "_is_trading_day", lambda d, ing: True)
    needs = tr.collect_tape_needs([day], ["KRX"], store, tr.read_settled_ledger(ledger), False)
    assert any(n.symbol == "005930" and n.session == "krx_aftermarket" for n in needs)
