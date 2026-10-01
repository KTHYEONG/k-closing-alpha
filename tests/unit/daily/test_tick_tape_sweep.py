"""Invariant guards for the daily tick tape sweep (self-healing steady state)."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.backfill.intraday.tape_harvest import TapeDayResult, TapeWalkOutcome
from src.config.collection import CollectionSettings
from src.daily import tick_tape_sweep as sweep
from src.data.capture_contracts import (
    ArtifactRef,
    CaptureDataset,
    CaptureStatus,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.intraday_schema import normalize_bar_frame, normalize_tick_frame
from src.data.intraday_store import tick_partition_path, write_intraday_partition
from src.tools import backfill_tick_tape as btt

_SEOUL = ZoneInfo("Asia/Seoul")
_FIXED_NOW = datetime(2026, 10, 1, 20, 40, tzinfo=_SEOUL)

@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(btt, "_price_history_path", lambda: tmp_path / "price_history.parquet")
    monkeypatch.setattr(btt, "_DEFAULT_BLACKOUTS", ())
    monkeypatch.setattr(btt, "_history_archive_path", lambda: tmp_path / "archive.parquet")

_DAY = "2026-09-30"
_OLD_DAY = "2026-09-03"
_BIG_FREE = 100 * 1024**3


def _profile(tmp_path, **overrides: Any) -> CollectionSettings:
    return CollectionSettings(COLLECTION_ROOT=tmp_path / "capture", **overrides)


def _patch_sweep(monkeypatch: pytest.MonkeyPatch, tmp_path, *, now: datetime = _FIXED_NOW, free: int = _BIG_FREE) -> None:
    monkeypatch.setattr(sweep, "_capture_root", lambda profile: tmp_path / "capture")
    monkeypatch.setattr(sweep, "_free_bytes", lambda path: free)
    monkeypatch.setattr(btt, "_free_bytes", lambda path: free)
    monkeypatch.setattr(sweep, "_now", lambda: now)


def _patch_universe(monkeypatch: pytest.MonkeyPatch, days: set[str], symbols: list[str] | None = None) -> None:
    members = symbols if symbols is not None else ["005930"]
    monkeypatch.setattr(btt, "_day_universe", lambda day, store: list(members) if day in days else [])


class _SessionCtx:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *args: Any) -> bool:
        return False


def _refuse_kiwoom() -> Any:
    def _open() -> Any:
        raise AssertionError("Kiwoom must not be called")

    return _open


def _stub_kiwoom() -> Any:
    return (object(), _SessionCtx())


def _tick_rows(day: str, hms_list: list[str], qty: str = "100") -> list[dict]:
    ymd = day.replace("-", "")
    return [{"cntr_tm": f"{ymd}{hms}", "cur_prc": "10000", "trde_qty": qty} for hms in hms_list]


def _bar_rows(day: str, hms_list: list[str], qty: str = "100") -> list[dict]:
    ymd = day.replace("-", "")
    return [
        {
            "cntr_tm": f"{ymd}{hms}", "cur_prc": "10000", "open_pric": "9900",
            "high_pric": "10100", "low_pric": "9800", "trde_qty": qty,
        }
        for hms in hms_list
    ]


def _complete_entry(symbol: str, session: str, venue: str, rows: int) -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol, dataset=CaptureDataset.TRADE_TICKS, venue=venue, session=session,
        scheduled_at=None, status=CaptureStatus.COMPLETE, rows=rows,
        first_event_time=None, last_event_time=None, reason="sweep-test",
        raw_refs=(ArtifactRef(path="raw/seed", sha256="abc", bytes=1),),
    )


def _seed_bars(day: str, session: str, symbol: str, hms_list: list[str], qty: str = "100") -> None:
    bars = normalize_bar_frame(pd.DataFrame(_bar_rows(day, hms_list, qty)), "kiwoom", day, symbol)
    write_intraday_partition(bars, 1, day, session, coverage={symbol: _complete_entry(symbol, session, "KRX", len(bars))})


def _seed_ticks(day: str, session: str, symbol: str, hms_list: list[str], qty: str = "100") -> None:
    from src.data.intraday_store import write_tick_partition

    frame = normalize_tick_frame(pd.DataFrame(_tick_rows(day, hms_list, qty)), "kiwoom", day, symbol)
    write_tick_partition(frame, day, session, coverage={symbol: _complete_entry(symbol, session, "KRX", len(frame))})


def _fake_harvest_ok(calls: list[dict[str, Any]]) -> Any:
    async def _fake(client: Any, http_session: Any, code: str, days: Any, **kwargs: Any) -> Any:
        calls.append({"symbol": code, "venue": kwargs.get("venue"), "days": sorted(days)})
        for day in sorted(days):
            for spec in kwargs.get("sessions", ()):
                hms = "090000" if spec.session == "regular" else "160000"
                qty = "100" if spec.session == "regular" else "10"
                frame = normalize_tick_frame(pd.DataFrame(_tick_rows(day, [hms], qty)), "kiwoom", day, code)
                kwargs["on_result"](
                    TapeDayResult(symbol=code, day=day, session=spec.session, frame=frame,
                                  entry=_complete_entry(code, spec.session, kwargs.get("venue", "KRX"), len(frame)))
                )
        return TapeWalkOutcome(termination_reason="tape_end", pages_fetched=2, unresolved_days=())

    return _fake


def _run(profile: CollectionSettings, **kwargs: Any) -> Any:
    params: dict[str, Any] = {"lookback_days": 5, "profile": profile}
    params.update(kwargs)
    return asyncio.run(sweep.run_tick_tape_sweep(**params))


def test_nothing_needed_is_noop(tmp_path, monkeypatch) -> None:
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, {_DAY})
    monkeypatch.setattr(sweep, "_open_kiwoom", _refuse_kiwoom())
    monkeypatch.setattr(btt, "harvest_symbol_tape", _refuse_kiwoom())
    _seed_ticks(_DAY, "regular", "005930", ["090000"], "100")
    _seed_bars(_DAY, "regular", "005930", ["090000"], "100")
    _seed_ticks(_DAY, "krx_aftermarket", "005930", ["160000"], "10")
    _seed_ticks(_DAY, "nxt_aftermarket", "005930", ["160000"], "10")

    report = _run(_profile(tmp_path))

    assert report.needs == 0
    assert report.recovered == ()
    assert report.unresolved == ()
    assert report.expired == ()
    assert report.remaining == ()
    assert report.disk_guard is False
    assert report.days_checked == ("2026-09-27", "2026-09-28", "2026-09-29", "2026-09-30", "2026-10-01")


def test_missed_day_is_recovered(tmp_path, monkeypatch) -> None:
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, {_DAY})
    monkeypatch.setattr(sweep, "_open_kiwoom", lambda: _stub_kiwoom())
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(btt, "harvest_symbol_tape", _fake_harvest_ok(calls))
    _seed_bars(_DAY, "regular", "005930", ["090000"], "100")

    report = _run(_profile(tmp_path))

    assert calls, "expected at least one tape walk"
    assert f"005930/{_DAY}/regular" in report.recovered
    assert report.unresolved == ()
    assert report.expired == ()
    assert report.remaining == ()
    stored = pd.read_parquet(tick_partition_path(_DAY, "regular"), columns=["symbol", "volume"])
    assert (stored["symbol"].astype(str) == "005930").any()
    assert int(stored["volume"].sum()) == 100


def test_lookback_bound_reports_expired_without_walks(tmp_path, monkeypatch) -> None:
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, {"2026-09-18"})
    monkeypatch.setattr(sweep, "_open_kiwoom", _refuse_kiwoom())
    monkeypatch.setattr(btt, "harvest_symbol_tape", _refuse_kiwoom())
    _seed_bars("2026-09-18", "regular", "005930", ["090000"], "100")

    report = _run(_profile(tmp_path))

    assert report.expired == (
        "005930/2026-09-18/krx_aftermarket",
        "005930/2026-09-18/nxt_aftermarket",
        "005930/2026-09-18/regular",
    )
    assert report.needs == 0
    assert report.recovered == ()
    assert report.unresolved == ()


def test_expiry_warning_and_audit_issue(tmp_path, monkeypatch, caplog) -> None:
    profile = _profile(tmp_path, COLLECTION_TAPE_LOOKBACK_DAYS=30)
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, {_OLD_DAY})
    monkeypatch.setattr(sweep, "_open_kiwoom", lambda: _stub_kiwoom())
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(btt, "harvest_symbol_tape", _fake_harvest_ok(calls))
    _seed_bars(_OLD_DAY, "regular", "005930", ["090000"], "100")

    with caplog.at_level("WARNING", logger="src.daily.tick_tape_sweep"):
        report = _run(profile, lookback_days=30)

    assert report.expiring == (_OLD_DAY,)
    assert f"005930/{_OLD_DAY}/regular" in report.recovered
    assert any("reason=expiring" in rec.message and _OLD_DAY in rec.message for rec in caplog.records)

    from src.tools import daily_audit

    # The sweep report is the audit's only input: one distinct symbol-day was within the expiry window.
    monkeypatch.setattr(daily_audit, "_capture_root", lambda profile: tmp_path / "capture")
    issues = daily_audit.audit_tape_sweep(date(2026, 10, 1), profile=profile)
    assert issues == ("intraday:tape_expiring:1:expiring_need",)


def test_deadline_respected(tmp_path, monkeypatch) -> None:
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, {_DAY})
    monkeypatch.setattr(sweep, "_open_kiwoom", lambda: _stub_kiwoom())
    calls: list[dict[str, Any]] = []

    async def _counting(client: Any, session: Any, code: str, days: Any, **kwargs: Any) -> Any:
        calls.append({"symbol": code})
        return TapeWalkOutcome(termination_reason="tape_end", pages_fetched=0, unresolved_days=())

    monkeypatch.setattr(btt, "harvest_symbol_tape", _counting)
    _seed_bars(_DAY, "regular", "005930", ["090000"], "100")

    report = _run(_profile(tmp_path), deadline=btt._now() - timedelta(hours=1))

    assert calls == []
    assert report.remaining != ()
    assert report.recovered == ()
    assert f"005930/{_DAY}/regular" in report.unresolved


def test_failure_isolation(tmp_path, monkeypatch) -> None:
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, {_DAY})
    monkeypatch.setattr(sweep, "_open_kiwoom", lambda: _stub_kiwoom())

    async def _auth_error(client: Any, session: Any, code: str, days: Any, **kwargs: Any) -> Any:
        raise RuntimeError("auth failed: token rejected")

    monkeypatch.setattr(btt, "harvest_symbol_tape", _auth_error)
    _seed_bars(_DAY, "regular", "005930", ["090000"], "100")
    store = CaptureStore(tmp_path / "capture")

    with pytest.raises(RuntimeError, match="auth failed"):
        _run(_profile(tmp_path))

    assert not tick_partition_path(_DAY, "regular").exists()
    assert store.read_manifests(_DAY) == ()


def test_transport_error_wrapped_as_infrastructure(tmp_path, monkeypatch) -> None:
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, {_DAY})
    monkeypatch.setattr(sweep, "_open_kiwoom", lambda: _stub_kiwoom())

    async def _broken(client: Any, session: Any, code: str, days: Any, **kwargs: Any) -> Any:
        raise OSError("connection reset")

    monkeypatch.setattr(btt, "harvest_symbol_tape", _broken)
    _seed_bars(_DAY, "regular", "005930", ["090000"], "100")

    with pytest.raises(RuntimeError, match="evidence failed"):
        _run(_profile(tmp_path))


def test_disk_guard_reports_and_skips_walks(tmp_path, monkeypatch) -> None:
    _patch_sweep(monkeypatch, tmp_path, free=0)
    _patch_universe(monkeypatch, {_DAY})
    monkeypatch.setattr(sweep, "_open_kiwoom", _refuse_kiwoom())
    monkeypatch.setattr(btt, "harvest_symbol_tape", _refuse_kiwoom())
    _seed_bars(_DAY, "regular", "005930", ["090000"], "100")

    report = _run(_profile(tmp_path))

    assert report.disk_guard is True
    assert f"005930/{_DAY}/regular" in report.unresolved
    assert report.recovered == ()
    assert report.remaining == report.unresolved


def test_lookback_and_deadline_validated(tmp_path) -> None:
    profile = _profile(tmp_path)
    with pytest.raises(ValueError, match="lookback_days"):
        asyncio.run(sweep.run_tick_tape_sweep(lookback_days=0, profile=profile))
    with pytest.raises(ValueError, match="lookback_days"):
        asyncio.run(sweep.run_tick_tape_sweep(lookback_days=99, profile=profile))
    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(sweep.run_tick_tape_sweep(lookback_days=5, profile=profile, deadline=datetime(2026, 10, 1, 19, 0)))


def test_storage_check_failure_is_infrastructure(tmp_path, monkeypatch) -> None:
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, set())
    monkeypatch.setattr(sweep, "_free_bytes", lambda path: (_ for _ in ()).throw(OSError("disk gone")))

    with pytest.raises(RuntimeError, match="storage check"):
        _run(_profile(tmp_path))


def test_report_write_failure_degrades_only(tmp_path, monkeypatch, caplog) -> None:
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, {_DAY})
    _seed_ticks(_DAY, "regular", "005930", ["090000"], "100")
    _seed_bars(_DAY, "regular", "005930", ["090000"], "100")
    _seed_ticks(_DAY, "krx_aftermarket", "005930", ["160000"], "10")
    _seed_ticks(_DAY, "nxt_aftermarket", "005930", ["160000"], "10")
    staging = tmp_path / "capture" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "tape_sweep").write_text("blocking file", encoding="utf-8")

    with caplog.at_level("WARNING", logger="src.daily.tick_tape_sweep"):
        report = _run(_profile(tmp_path))

    assert report.needs == 0
    assert any("report_write_failed" in rec.message for rec in caplog.records)


def test_helpers_and_open_kiwoom_branches(tmp_path, monkeypatch) -> None:
    assert sweep._now().tzinfo is not None
    assert sweep._free_bytes(tmp_path) > 0
    assert btt._ledger_path(None, _profile(tmp_path)).name == "ledger.jsonl"
    assert sweep._report_path(tmp_path).name == "last_report.json"
    assert sweep._need_key("A", "2026-09-30", "regular") == "A/2026-09-30/regular"
    assert sweep._window_days(2, date(2026, 10, 1)) == ["2026-09-30", "2026-10-01"]
    assert sweep._scan_days(date(2026, 10, 1))[0] == "2026-09-02"
    assert len(sweep._scan_days(date(2026, 10, 1))) == 30

    from src import settings as live_settings

    monkeypatch.setattr(live_settings, "KIWOOM_APP_KEY", "", raising=False)
    with pytest.raises(RuntimeError, match="credentials"):
        sweep._open_kiwoom()

    monkeypatch.setattr(live_settings, "KIWOOM_APP_KEY", "test-key", raising=False)
    monkeypatch.setattr("src.api.kiwoom.client.KiwoomApiClient", lambda: object())
    import aiohttp

    monkeypatch.setattr(aiohttp, "ClientSession", lambda: object())
    client, session = sweep._open_kiwoom()
    assert client is not None and session is not None


def test_default_profile_used_when_omitted(tmp_path, monkeypatch) -> None:
    profile = _profile(tmp_path)
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, set())
    monkeypatch.setattr(sweep, "CollectionSettings", lambda: profile)

    report = asyncio.run(sweep.run_tick_tape_sweep(lookback_days=5))

    assert report.needs == 0


def test_main_parses_args(tmp_path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_run(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(sweep, "run_tick_tape_sweep", _fake_run)
    sweep.main(["--lookback-days", "7", "--deadline", "2026-10-01T19:00:00+09:00"])
    assert captured["lookback_days"] == 7
    assert captured["deadline"] == datetime(2026, 10, 1, 19, 0, tzinfo=_SEOUL)

    captured.clear()
    monkeypatch.setattr(sweep, "CollectionSettings", lambda: _profile(tmp_path))
    sweep.main([])
    assert captured["lookback_days"] == 25
    assert captured["deadline"] is None


def test_audit_tape_sweep_reads_expiring_count_from_report() -> None:
    from src.tools import daily_audit

    report = {"run_date": "2026-10-01", "expiring_needs": 3, "disk_guard": False}
    assert daily_audit.audit_tape_sweep(date(2026, 10, 1), report=report) == ("intraday:tape_expiring:3:expiring_need",)


def test_audit_tape_sweep_disk_guard_and_clean() -> None:
    from src.tools import daily_audit

    guard = {"run_date": "2026-10-01", "expiring_needs": 0, "disk_guard": True}
    clean = {"run_date": "2026-10-01", "expiring_needs": 0, "disk_guard": False}
    assert daily_audit.audit_tape_sweep(date(2026, 10, 1), report=guard) == ("intraday:tape_sweep:1:disk_guard",)
    assert daily_audit.audit_tape_sweep(date(2026, 10, 1), report=clean) == ()


def test_audit_tape_sweep_ignores_stale_future_or_malformed_reports() -> None:
    from src.tools import daily_audit

    stale = {"run_date": "2026-09-18", "expiring_needs": 5, "disk_guard": True}
    future = {"run_date": "2026-10-09", "expiring_needs": 5, "disk_guard": True}
    assert daily_audit.audit_tape_sweep(date(2026, 10, 1), report=stale) == ()
    assert daily_audit.audit_tape_sweep(date(2026, 10, 1), report=future) == ()
    assert daily_audit.audit_tape_sweep(date(2026, 10, 1), report={"expiring_needs": 2}) == ()
    assert daily_audit.audit_tape_sweep(date(2026, 10, 1), report={"run_date": "bad", "expiring_needs": 2}) == ()


def test_audit_tape_sweep_reads_report_file_and_never_recomputes_needs(tmp_path, monkeypatch) -> None:
    import json

    from src.tools import daily_audit

    profile = _profile(tmp_path)
    monkeypatch.setattr(daily_audit, "_capture_root", lambda profile: tmp_path / "capture")
    monkeypatch.setattr(btt, "_collect_needs", lambda *a, **k: (_ for _ in ()).throw(AssertionError("audit must not rescan partitions")))

    assert daily_audit.audit_tape_sweep(date(2026, 10, 1), profile=profile) == ()

    report_dir = tmp_path / "capture" / "staging" / "tape_sweep"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "last_report.json").write_text("not json", encoding="utf-8")
    assert daily_audit.audit_tape_sweep(date(2026, 10, 1), profile=profile) == ()

    (report_dir / "last_report.json").write_text(json.dumps({"run_date": "2026-10-01", "disk_guard": True}), encoding="utf-8")
    assert daily_audit.audit_tape_sweep(date(2026, 10, 1), profile=profile) == ("intraday:tape_sweep:1:disk_guard",)


def test_settled_no_trades_is_not_walked_again(tmp_path, monkeypatch) -> None:
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, {_DAY})
    monkeypatch.setattr(sweep, "_open_kiwoom", _refuse_kiwoom())
    monkeypatch.setattr(btt, "harvest_symbol_tape", _refuse_kiwoom())
    _seed_ticks(_DAY, "regular", "005930", ["090000"], "100")
    _seed_bars(_DAY, "regular", "005930", ["090000"], "100")
    profile = _profile(tmp_path)
    ledger = btt._ledger_path(None, profile)
    btt._append_ledger(ledger, [
        {"symbol": "005930", "day": _DAY, "session": s, "status": "NO_TRADES", "run_id": "tape-x"}
        for s in ("krx_aftermarket", "nxt_aftermarket")
    ])

    report = _run(profile)

    assert report.needs == 0 and report.remaining == ()


def test_sweep_report_feeds_audit_digest(tmp_path, monkeypatch) -> None:
    from src.tools import daily_audit

    _patch_sweep(monkeypatch, tmp_path, free=0)
    _patch_universe(monkeypatch, {_DAY})
    monkeypatch.setattr(sweep, "_open_kiwoom", _refuse_kiwoom())
    monkeypatch.setattr(btt, "harvest_symbol_tape", _refuse_kiwoom())
    _seed_bars(_DAY, "regular", "005930", ["090000"], "100")

    report = _run(_profile(tmp_path))

    assert report.disk_guard is True
    monkeypatch.setattr(daily_audit, "_capture_root", lambda profile: tmp_path / "capture")
    issues = daily_audit.audit_tape_sweep(date(2026, 10, 1), profile=_profile(tmp_path))
    assert issues == ("intraday:tape_sweep:1:disk_guard",)


def test_open_kiwoom_oserror_wrapped_as_infrastructure(tmp_path, monkeypatch) -> None:
    _patch_sweep(monkeypatch, tmp_path)
    _patch_universe(monkeypatch, {_DAY})
    _seed_bars(_DAY, "regular", "005930", ["090000"], "100")

    def _broken_open() -> Any:
        raise OSError("kiwoom socket unavailable")

    monkeypatch.setattr(sweep, "_open_kiwoom", _broken_open)

    with pytest.raises(RuntimeError, match="infrastructure"):
        _run(_profile(tmp_path))
