"""Invariant guards for the tape-recovery engine (ADR-008 contract owner)."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backfill.intraday import tape_recovery as tr
from src.backfill.intraday.tape_harvest import TAPE_SESSIONS, TapeWalkOutcome
from src.config.collection import CollectionSettings
from src.config.market_session import ARCHIVE_REGULAR_READY_HHMMSS
from src.data.capture_store import CaptureStore
from src.data.intraday_schema import normalize_bar_frame, normalize_tick_frame
from src.data.intraday_store import write_intraday_partition, write_tick_partition
from src.data.capture_contracts import ArtifactRef, CaptureDataset, CaptureStatus, CoverageEntry

_SEOUL = ZoneInfo("Asia/Seoul")
_DAY = "2026-09-02"


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tr, "_price_history_path", lambda: tmp_path / "price_history.parquet")
    monkeypatch.setattr(tr, "_history_archive_path", lambda: tmp_path / "archive.parquet")


def _profile(tmp_path) -> CollectionSettings:
    return CollectionSettings(COLLECTION_ROOT=tmp_path / "capture")


def _patch_roots(tmp_path, monkeypatch) -> None:
    from src import settings as _settings

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    monkeypatch.setattr(tr, "_capture_root", lambda profile: tmp_path / "capture")
    monkeypatch.setattr(tr, "_free_bytes", lambda path: 100 * 1024**3)


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


def _raw_ticks(day: str, pairs: list[tuple[str, str]]) -> list[dict]:
    ymd = day.replace("-", "")
    return [{"cntr_tm": f"{ymd}{hms}", "cur_prc": "10000", "trde_qty": qty} for hms, qty in pairs]


def _raw_bars(day: str, pairs: list[tuple[str, str]]) -> list[dict]:
    ymd = day.replace("-", "")
    return [
        {
            "cntr_tm": f"{ymd}{hms}",
            "cur_prc": "10000",
            "open_pric": "9900",
            "high_pric": "10100",
            "low_pric": "9800",
            "trde_qty": qty,
        }
        for hms, qty in pairs
    ]


def _seed(
    day: str, session: str, symbol: str, ticks: list[tuple[str, str]], bars: list[tuple[str, str]] | None
) -> None:
    if ticks:
        frame = normalize_tick_frame(pd.DataFrame(_raw_ticks(day, ticks)), "kiwoom", day, symbol)
        write_tick_partition(frame, day, session, coverage={symbol: _complete_entry(symbol, session, len(frame))})
    if bars is not None:
        bframe = normalize_bar_frame(pd.DataFrame(_raw_bars(day, bars)), "kiwoom", day, symbol)
        write_intraday_partition(
            bframe, 1, day, session, coverage={symbol: _complete_entry(symbol, session, len(bframe))}
        )


def _audit_frames(
    bars: list[tuple[str, int]], ticks: list[tuple[str, int]], symbol: str = "005930"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bar_frame = pd.DataFrame([{"symbol": symbol, "ts_hms": h, "volume": v} for h, v in bars])
    tick_frame = pd.DataFrame([{"symbol": symbol, "ts_hms": h, "volume": v} for h, v in ticks])
    return bar_frame, tick_frame


def test_audit_and_sweep_agree_on_krx_aftermarket_ceiling_bar(tmp_path, monkeypatch) -> None:
    from src.tools import daily_audit

    _patch_roots(tmp_path, monkeypatch)
    day = "2026-09-10"
    _seed(
        day,
        "krx_aftermarket",
        "005930",
        [("160000", "100"), ("161000", "50")],
        [("160000", "100"), ("161000", "50"), ("200000", "7")],
    )
    bars, ticks = _audit_frames([(160000, 100), (161000, 50), (200000, 7)], [(160000, 100), (161000, 50)])
    issues = daily_audit.audit_aftermarket_ticks(
        date.fromisoformat(day),
        read_ticks=lambda session: (
            ticks if session == "krx_aftermarket" else pd.DataFrame({"symbol": [], "volume": []})
        ),
        read_bars=lambda session: bars if session == "krx_aftermarket" else None,
    )
    assert issues == ()
    spec = next(s for s in TAPE_SESSIONS if s.session == "krx_aftermarket")
    assert tr._session_need("005930", day, spec, tr.PartitionIndex()) is False


def test_agreement_holds_for_real_aftermarket_shortfall(tmp_path, monkeypatch) -> None:
    from src.tools import daily_audit

    _patch_roots(tmp_path, monkeypatch)
    day = "2026-09-10"
    _seed(
        day,
        "krx_aftermarket",
        "005930",
        [("160000", "90"), ("161000", "50")],
        [("160000", "100"), ("161000", "50"), ("200000", "7")],
    )
    bars, ticks = _audit_frames([(160000, 100), (161000, 50), (200000, 7)], [(160000, 90), (161000, 50)])
    issues = daily_audit.audit_aftermarket_ticks(
        date.fromisoformat(day),
        read_ticks=lambda session: (
            ticks if session == "krx_aftermarket" else pd.DataFrame({"symbol": [], "volume": []})
        ),
        read_bars=lambda session: bars if session == "krx_aftermarket" else None,
    )
    assert issues == ("intraday:krx_aftermarket_ticks:1:volume_mismatch",)
    spec = next(s for s in TAPE_SESSIONS if s.session == "krx_aftermarket")
    assert tr._session_need("005930", day, spec, tr.PartitionIndex()) is True


def test_regular_closing_auction_exclusion_unchanged(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    day = "2026-09-10"
    _seed(day, "regular", "005930", [("090000", "100")], [("152000", "100"), ("153000", "900")])
    spec = next(s for s in TAPE_SESSIONS if s.session == "regular")
    assert tr._session_need("005930", day, spec, tr.PartitionIndex()) is False
    assert (tr.PartitionIndex().bar_volumes(day, "regular") or {}).get("005930") == pytest.approx(100.0)


def test_nxt_aftermarket_ceiling_bar_still_counted(tmp_path, monkeypatch) -> None:
    from src.tools import daily_audit

    _patch_roots(tmp_path, monkeypatch)
    day = "2026-09-10"
    _seed(day, "nxt_aftermarket", "005930", [("160000", "50")], [("195900", "50"), ("200000", "50")])
    spec = next(s for s in TAPE_SESSIONS if s.session == "nxt_aftermarket")
    assert tr._session_need("005930", day, spec, tr.PartitionIndex()) is True
    bars, ticks = _audit_frames([(195900, 50), (200000, 50)], [(160000, 50)])
    issues = daily_audit.audit_aftermarket_ticks(
        date.fromisoformat(day),
        read_ticks=lambda session: (
            ticks if session == "nxt_aftermarket" else pd.DataFrame({"symbol": [], "volume": []})
        ),
        read_bars=lambda session: bars if session == "nxt_aftermarket" else None,
    )
    assert issues == ("intraday:nxt_aftermarket_ticks:1:volume_mismatch",)


def test_tick_surplus_is_never_a_need(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    day = "2026-09-10"
    _seed(day, "krx_aftermarket", "005930", [("160000", "160")], [("160000", "100"), ("161000", "50")])
    spec = next(s for s in TAPE_SESSIONS if s.session == "krx_aftermarket")
    assert tr._session_need("005930", day, spec, tr.PartitionIndex()) is False


def test_regular_readiness_follows_market_session() -> None:
    today = date(2026, 9, 10).isoformat()
    hh, mm, ss = ARCHIVE_REGULAR_READY_HHMMSS[0:2], ARCHIVE_REGULAR_READY_HHMMSS[2:4], ARCHIVE_REGULAR_READY_HHMMSS[4:6]
    base = datetime(2026, 9, 10, int(hh), int(mm), int(ss), tzinfo=_SEOUL)
    before = base - pd.Timedelta(seconds=1).to_pytimedelta()
    assert tr._session_closed(today, "regular", before) is False
    assert tr._session_closed(today, "regular", base) is True


def test_settled_statuses_end_recovery(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    day = "2026-09-08"
    _seed(day, "regular", "000001", [("090000", "10")], [("090000", "1000")])
    _seed(day, "regular", "000002", [("090000", "10")], [("090000", "1000")])
    _seed(day, "regular", "000003", [("090000", "10")], [("090000", "1000")])
    monkeypatch.setattr(tr, "_day_universe", lambda d, s: ["000001", "000002", "000003"] if d == day else [])
    monkeypatch.setattr(tr, "_session_closed", lambda d, s, n: True)
    settled = {
        (s, day, "regular"): st for s, st in (("000001", "COMPLETE"), ("000002", "NO_TRADES"), ("000003", "PARTIAL"))
    }
    needs = tr.collect_tape_needs([day], ["KRX"], store, settled, False)
    regular_needs = {(n.symbol, n.session) for n in needs if n.session == "regular"}
    assert regular_needs == {("000003", "regular")}
    forced = tr.collect_tape_needs([day], ["KRX"], store, settled, True)
    assert {n.symbol for n in forced if n.session == "regular"} >= {"000001", "000002", "000003"}


def test_run_summary_is_typed_and_complete(tmp_path, monkeypatch) -> None:
    _patch_roots(tmp_path, monkeypatch)
    store = CaptureStore(tmp_path / "capture")
    profile = _profile(tmp_path)
    sessions = tuple(s for s in TAPE_SESSIONS if s.venue == "KRX")
    tasks = [
        tr.WalkTask(symbol="A", venue="KRX", days=(_DAY,), sessions=sessions),
        tr.WalkTask(symbol="B", venue="KRX", days=(_DAY,), sessions=sessions),
    ]
    past = datetime(2020, 1, 1, tzinfo=_SEOUL)
    summary = asyncio.run(
        tr.run_walk_tasks(
            tasks,
            client=object(),
            http_session=object(),
            store=store,
            profile=profile,
            apply=False,
            ledger=tmp_path / "l.jsonl",
            deadline=past,
            blackouts=[],
            run_date="2026-09-20",
        )
    )
    assert isinstance(summary, tr.TapeRunSummary)
    assert summary.stopped is True and summary.stopped_reason == "deadline"
    assert summary.remaining == ("A/KRX", "B/KRX") and summary.pages == 0


def test_deadline_parsing_fails_closed() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        tr.parse_walk_deadline("2026-10-02T21:15:00")
    with pytest.raises(ValueError, match="deadline"):
        tr.parse_walk_deadline("bad")
    assert tr.parse_walk_deadline(None) is None
    aware = tr.parse_walk_deadline("2026-10-02T21:15:00+09:00")
    assert aware is not None and aware.tzinfo is not None


def test_engine_has_no_cli_or_daily_dependency() -> None:
    code = (
        "import src.backfill.intraday.tape_recovery as m, sys; "
        "print('TOOLS' if 'src.tools.backfill_tick_tape' in sys.modules else 'no-tools'); "
        "print('DAILY' if 'src.daily.tick_tape_sweep' in sys.modules else 'no-daily')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=".")  # noqa: S603
    assert out.returncode == 0, out.stderr
    assert "no-tools" in out.stdout and "no-daily" in out.stdout
