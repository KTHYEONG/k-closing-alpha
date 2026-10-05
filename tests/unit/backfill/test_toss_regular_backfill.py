"""Invariant guards for the Toss regular-session backfill planner, runner, ledger and CLI."""

from __future__ import annotations

import asyncio
import itertools
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from src.backfill.intraday import toss_regular_backfill as trb
from src.backfill.intraday.extended_session_backfill import (
    ExtendedBackfillLedger,
    ExtendedBackfillTask,
    regular_superset_symbols_by_day,
)
from src.backfill.intraday.price_basis import PriceReference
from src.backfill.intraday.toss_regular import acquire_toss_regular_bars
from src.config.collection import CollectionSettings
from src.data.capture_contracts import SEOUL, CaptureDataset, CaptureStatus, CoverageEntry
from src.data.capture_store import CaptureStore
from src.data.eod_superset import EodSupersetScreen

_DAY1 = "2026-03-02"
_DAY2 = "2026-03-03"
_AS_OF = date(2026, 3, 10)
_FIXED_NOW = datetime(2026, 9, 29, 3, 0, tzinfo=SEOUL)
_FAR_FUTURE = datetime(2099, 1, 1, tzinfo=SEOUL)


def _hhmmss(minute_of_day: int) -> str:
    return f"{minute_of_day // 60:02d}{minute_of_day % 60:02d}00"


def _candle(day: str, hhmmss: str, volume: int = 100) -> dict:
    return {
        "timestamp": f"{day}T{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}.000+09:00",
        "openPrice": "70000",
        "highPrice": "70000",
        "lowPrice": "70000",
        "closePrice": "70000",
        "volume": str(volume),
        "currency": "KRW",
    }


def _regular_grid(day: str, volume: int = 100) -> list[dict]:
    out = [_candle(day, _hhmmss(m), volume=volume) for m in range(9 * 60 + 1, 15 * 60 + 31)]
    assert len(out) == 390
    return out


class _FakeToss:
    """Scripted per-symbol Toss stand-in serving newest-first grids with call tracking."""

    def __init__(
        self,
        grids: dict[str, list[dict]] | None = None,
        *,
        errors: dict[str, dict] | None = None,
        raises: set[str] | None = None,
        latency: float = 0.0,
    ) -> None:
        self._grids = dict(grids or {})
        self._errors = dict(errors or {})
        self._raises = set(raises or set())
        self._latency = latency
        self.calls: list[str] = []
        self.kwargs: list[dict] = []
        self.inflight = 0
        self.max_inflight = 0

    async def get_candles(
        self, session, symbol: str, *, interval: str = "1m", count: int = 200,
        before: str | None = None, adjusted: bool | None = None,
    ) -> dict:
        self.calls.append(str(symbol))
        self.kwargs.append({"count": count, "before": before, "adjusted": adjusted})
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self._latency:
                await asyncio.sleep(self._latency)
            if str(symbol) in self._raises:
                raise ConnectionError("boom")
            if str(symbol) in self._errors:
                return dict(self._errors[str(symbol)])
            grid = self._grids.get(str(symbol), [])
            bound = datetime.fromisoformat(str(before)) if before else None
            out = []
            for candle in grid:
                current = datetime.fromisoformat(str(candle.get("timestamp", "")))
                if bound is None or current <= bound:
                    out.append(candle)
                    if len(out) >= int(count):
                        break
            return {"result": {"candles": out}}
        finally:
            self.inflight -= 1


def _complete_client(symbols: list[str], day: str, volume: int = 100, **kw) -> _FakeToss:
    return _FakeToss({s: list(reversed(_regular_grid(day, volume))) for s in symbols}, **kw)


def _prepared(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "symbol", "chg_ratio", "tv_clean", "mc_clean", "volume"])


def _screen() -> EodSupersetScreen:
    return EodSupersetScreen(
        min_change_ratio=0.01, max_change_ratio=0.12,
        min_trade_value_100m=100.0, min_market_cap_100m=495.0, common_stock_only=False,
    )


def _ref(rows: list[tuple]) -> PriceReference:
    return PriceReference.from_price_history(
        pd.DataFrame(rows, columns=["date", "symbol", "close", "close_raw"])
    )


def _eod(days: list[str], symbols: list[str], volume: float = 39000.0) -> dict:
    return {(day, symbol): float(volume) for day in days for symbol in symbols}


@pytest.fixture
def env(tmp_path, monkeypatch):
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    profile = CollectionSettings(
        COLLECTION_ROOT=tmp_path / "capture",
        COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS=(),
        COLLECTION_TOSS_OUTAGE_MIN_SAMPLE=1,
        _env_file=None,
    )
    store = CaptureStore(tmp_path / "capture")
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    return profile, store, ledger


def _run(profile, store, ledger, client, tasks, eod_volumes, now_fn=None, stop_at=None):
    return asyncio.run(
        trb.run_toss_regular_backfill(
            as_of=_AS_OF, stop_at=stop_at if stop_at is not None else _FAR_FUTURE,
            profile=profile, client=client, store=store, ledger=ledger,
            tasks=tasks, eod_volumes=eod_volumes,
            now_fn=now_fn if now_fn is not None else (lambda: _FIXED_NOW),
        )
    )


def _task(day: str, symbols: tuple[str, ...]) -> ExtendedBackfillTask:
    return ExtendedBackfillTask(snapshot_date=day, session="regular", symbols=symbols)


def _failed_entry(symbol: str, reason: str = "transport:boom") -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol, dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="regular",
        scheduled_at=None, status=CaptureStatus.FAILED, rows=0,
        first_event_time=None, last_event_time=None, reason=reason, raw_refs=(),
    )


# ---------------------------------------------------------------- planner


def test_planner_is_disjoint_from_the_kis_stream() -> None:
    """Before the KIS window every superset symbol is planned; inside it only adjusted/unknown ones."""
    panel = _prepared([
        (_DAY1, "000001", 0.05, 500.0, 900.0, 100.0),
        (_DAY1, "000002", 0.05, 500.0, 900.0, 100.0),
        (_DAY1, "000003", 0.05, 500.0, 900.0, 100.0),
        (_DAY2, "000001", 0.05, 500.0, 900.0, 100.0),
        (_DAY2, "000002", 0.05, 500.0, 900.0, 100.0),
        (_DAY2, "000003", 0.05, 500.0, 900.0, 100.0),
    ])
    ref = _ref([
        (_DAY1, "000001", 1000.0, 1000.0), (_DAY1, "000002", 900.0, 1000.0),
        (_DAY2, "000001", 1000.0, 1000.0), (_DAY2, "000002", 900.0, 1000.0),
    ])
    plan = trb.enumerate_toss_regular_tasks(
        as_of=_AS_OF, kis_window_start=_DAY2, retention_floor=None,
        prepared_panel=panel, screen=_screen(), price_reference=ref,
    )
    by_day = {task.snapshot_date: task.symbols for task in plan.tasks}
    assert by_day[_DAY1] == ("000001", "000002", "000003")
    assert by_day[_DAY2] == ("000002", "000003")
    assert plan.kis_window_symbol_days == 1


def test_floor_is_inclusive_and_counted() -> None:
    """The floor date itself is planned; earlier dates are excluded and counted."""
    days = ["2026-03-02", "2026-03-03", "2026-03-04"]
    panel = _prepared([(d, "000001", 0.05, 500.0, 900.0, 100.0) for d in days])
    ref = _ref([(d, "000001", 1000.0, 1000.0) for d in days])
    plan = trb.enumerate_toss_regular_tasks(
        as_of=_AS_OF, kis_window_start="2027-01-01", retention_floor="2026-03-03",
        prepared_panel=panel, screen=_screen(), price_reference=ref,
    )
    assert [t.snapshot_date for t in plan.tasks] == ["2026-03-03", "2026-03-04"]
    assert plan.below_floor_symbol_days == 1
    assert plan.retention_floor == "2026-03-03"
    open_plan = trb.enumerate_toss_regular_tasks(
        as_of=_AS_OF, kis_window_start="2027-01-01", retention_floor=None,
        prepared_panel=panel, screen=_screen(), price_reference=ref,
    )
    assert [t.snapshot_date for t in open_plan.tasks] == days
    assert open_plan.below_floor_symbol_days == 0


def test_plan_is_pure_deterministic_and_padded() -> None:
    """The same panel twice gives identical ascending tasks with zero-padded unique symbols."""
    panel = _prepared([
        (_DAY2, "1", 0.05, 500.0, 900.0, 100.0),
        (_DAY2, "000001", 0.05, 500.0, 900.0, 100.0),
        (_DAY1, "2", 0.05, 500.0, 900.0, 100.0),
    ])
    ref = _ref([(_DAY1, "000002", 1000.0, 1000.0), (_DAY2, "000001", 1000.0, 1000.0)])
    kw = {"as_of": _AS_OF, "kis_window_start": "2027-01-01", "retention_floor": None,
          "prepared_panel": panel, "screen": _screen(), "price_reference": ref}
    first = trb.enumerate_toss_regular_tasks(**kw)
    second = trb.enumerate_toss_regular_tasks(**kw)
    assert first == second
    assert [t.snapshot_date for t in first.tasks] == [_DAY1, _DAY2]
    assert first.tasks[0].symbols == ("000002",)
    assert first.tasks[1].symbols == ("000001",)


def test_superset_helper_matches_kis_enumeration() -> None:
    """The extracted helper returns the same per-day symbol sets the KIS planner consumes."""
    from src.backfill.intraday.extended_session_backfill import enumerate_regular_session_tasks

    panel = _prepared([
        (_DAY1, "000001", 0.05, 500.0, 900.0, 100.0),
        (_DAY1, "000002", 0.05, 500.0, 900.0, 100.0),
        (_DAY2, "000003", 0.05, 500.0, 900.0, 100.0),
    ])
    ref = _ref([
        (_DAY1, "000001", 1000.0, 1000.0), (_DAY1, "000002", 1000.0, 1000.0),
        (_DAY2, "000003", 1000.0, 1000.0),
    ])
    helper = regular_superset_symbols_by_day(
        as_of=date(2026, 3, 4), earliest="2020-01-01", prepared_panel=panel, screen=_screen()
    )
    assert helper == {_DAY1: ("000001", "000002"), _DAY2: ("000003",)}
    plan = enumerate_regular_session_tasks(
        as_of=date(2026, 3, 4), retention_days=365, prepared_panel=panel,
        screen=_screen(), price_reference=ref,
    )
    assert [t.symbols for t in plan.tasks] == [("000001", "000002"), ("000003",)]
    assert regular_superset_symbols_by_day(
        as_of=date(2026, 3, 3), earliest="2020-01-01",
        prepared_panel=_prepared([]), screen=_screen(),
    ) == {}
    with pytest.raises(ValueError, match="missing required columns"):
        regular_superset_symbols_by_day(
            as_of=date(2026, 3, 3), earliest="2020-01-01",
            prepared_panel=pd.DataFrame([{"date": _DAY1, "symbol": "000001"}]), screen=_screen(),
        )
    assert regular_superset_symbols_by_day(
        as_of=date(2026, 3, 3), earliest="2027-01-01", prepared_panel=panel, screen=_screen()
    ) == {}


# ---------------------------------------------------------------- runner


def test_stored_symbols_are_never_refetched(env) -> None:
    """A partition already holding a symbol triggers no request and stays byte-identical."""
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    client = _complete_client(["000001", "000002"], _DAY1)
    frame, entry = asyncio.run(
        acquire_toss_regular_bars(
            client, None, "000001", _DAY1, eod_volume=39000.0,
            profile=profile, capture_store=store, run_id="seed",
        )
    )
    from src.data.intraday_store import write_intraday_partition

    write_intraday_partition(frame, 1, _DAY1, "regular", coverage={"000001": entry})
    before = pd.read_parquet(intraday_partition_path(1, _DAY1, "regular"))
    client.calls.clear()
    summary = _run(profile, store, ledger, client, [_task(_DAY1, ("000001", "000002"))],
                   _eod([_DAY1], ["000001", "000002"]))
    assert set(client.calls) == {"000002"}
    assert summary.complete == 1
    assert all(kw["adjusted"] is False for kw in client.kwargs)
    after = pd.read_parquet(intraday_partition_path(1, _DAY1, "regular"))
    assert after[after["symbol"] == "000001"].reset_index(drop=True).equals(
        before[before["symbol"] == "000001"].reset_index(drop=True)
    )


def test_ledger_terminal_skip_and_retry_cap(env) -> None:
    """NOT_APPLICABLE is skipped; the third consecutive failure becomes EXHAUSTED."""
    profile, store, ledger = env
    ledger.record_cached_absent(
        _DAY1, "regular", ["000009"], reason="toss_stock_not_found_cached",
        run_id="seed", attempted_at=_FIXED_NOW, vendor="toss",
    )
    for _ in range(2):
        ledger.record(_DAY1, "regular", [_failed_entry("000008")], run_id="seed",
                      attempted_at=_FIXED_NOW, vendor="toss")
    client = _complete_client(["000006", "000010"], _DAY1)
    client._raises = {"000008"}
    summary = _run(profile, store, ledger, client,
                   [_task(_DAY1, ("000006", "000008", "000009", "000010"))],
                   _eod([_DAY1], ["000006", "000008", "000009", "000010"]))
    assert set(client.calls) == {"000006", "000008", "000010"}
    assert summary.exhausted == 1 and summary.failed == 1 and summary.complete == 2
    assert ledger.terminal_symbols(_DAY1, "regular") == frozenset({"000006", "000008", "000009", "000010"})
    frame = ledger._read_all()
    latest = frame[frame["symbol"] == "000008"].iloc[-1]
    assert latest["status"] == "EXHAUSTED"


def test_delisted_symbol_resolved_once(env) -> None:
    """One stock-not-found request, cached rows for the rest, and silence on later runs."""
    profile, store, ledger = env
    days = [(date(2026, 1, 5) + timedelta(days=i)).isoformat() for i in range(50)]
    tasks = [_task(day, ("000007",)) for day in days]
    client = _FakeToss(errors={"000007": {"error": {"code": "stock-not-found"}}})
    summary = _run(profile, store, ledger, client, tasks, _eod(days, ["000007"]))
    assert client.calls == ["000007"]
    assert summary.not_listed == 50
    frame = ledger._read_all()
    mine = frame[frame["symbol"] == "000007"]
    assert len(mine) == 50
    assert (mine["reason"] == "toss_stock_not_found").sum() == 1
    assert (mine["reason"] == "toss_stock_not_found_cached").sum() == 49
    client.calls.clear()
    rerun = _run(profile, store, ledger, client, tasks, _eod(days, ["000007"]))
    assert client.calls == []
    assert (rerun.tasks_done, rerun.tasks_remaining) == (50, 0)


def test_consolidated_tape_is_terminal(env) -> None:
    """NXT-style volume ratios are rejected terminally with no partition rows and no retry."""
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    client = _complete_client(["000005"], _DAY1, volume=150)
    summary = _run(profile, store, ledger, client, [_task(_DAY1, ("000005",))],
                   _eod([_DAY1], ["000005"], volume=39000.0))
    assert summary.consolidated == 1 and summary.failed == 0
    target = intraday_partition_path(1, _DAY1, "regular")
    assert not target.exists()
    client.calls.clear()
    _run(profile, store, ledger, client, [_task(_DAY1, ("000005",))],
         _eod([_DAY1], ["000005"], volume=39000.0))
    assert client.calls == []


def test_volume_shortfall_and_unverifiable_are_retryable(env) -> None:
    """Shortfall and missing-EOD failures stay retryable and never burn the cap in one run."""
    profile, store, ledger = env
    short = _complete_client(["000006"], _DAY1, volume=50)
    summary = _run(profile, store, ledger, short, [_task(_DAY1, ("000006",))],
                   _eod([_DAY1], ["000006"], volume=39000.0))
    assert summary.failed == 1 and summary.exhausted == 0
    assert "000006" not in ledger.terminal_symbols(_DAY1, "regular")
    missing = _complete_client(["000010"], _DAY1)
    summary = _run(profile, store, ledger, missing, [_task(_DAY1, ("000010",))], {})
    assert summary.failed == 1
    frame = ledger._read_all()
    assert frame[frame["symbol"] == "000010"].iloc[-1]["reason"] == "toss_basis_unverifiable"


def test_outage_does_not_burn_the_cap(env) -> None:
    """All-transport date records nothing and the next run retries with attempts still 0."""
    profile, store, ledger = env
    tasks = [_task(_DAY1, ("000011", "000012", "000013")), _task(_DAY2, ("000014",))]
    client = _FakeToss(raises={"000011", "000012", "000013"})
    summary = _run(profile, store, ledger, client, tasks, _eod([_DAY1, _DAY2], ["000011", "000012", "000013", "000014"]))
    assert summary.outage_aborted is True
    assert client.calls == ["000011", "000012", "000013"]
    assert (summary.tasks_done, summary.tasks_remaining) == (0, 2)  # an outage-aborted date is not done
    assert ledger._read_all().empty
    client.calls.clear()
    again = _run(profile, store, ledger, client, tasks, _eod([_DAY1, _DAY2], ["000011", "000012", "000013", "000014"]))
    assert again.outage_aborted is True and client.calls == ["000011", "000012", "000013"]


def test_partial_outage_keeps_good_symbols(env) -> None:
    """Certified symbols commit while outage failures go unrecorded and the run stops."""
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    symbols = ["000021", "000022", "000023", "000024", "000025"]
    client = _FakeToss(
        {s: list(reversed(_regular_grid(_DAY1))) for s in ("000024", "000025")},
        raises={"000021", "000022", "000023"},
    )
    summary = _run(profile, store, ledger, client, [_task(_DAY1, tuple(symbols))],
                   _eod([_DAY1], symbols))
    assert summary.outage_aborted is True and summary.complete == 2 and summary.failed == 0
    stored = pd.read_parquet(intraday_partition_path(1, _DAY1, "regular"))
    assert sorted(stored["symbol"].unique().tolist()) == ["000024", "000025"]
    frame = ledger._read_all()
    assert sorted(frame["symbol"].unique().tolist()) == ["000024", "000025"]
    assert (frame["vendor"] == "toss").all()
    assert (frame[frame["status"] == "COMPLETE"]["price_basis"] == "toss_raw").all()


def test_empty_pages_count_toward_outage(env) -> None:
    """A 200 answer with no candles for every symbol is an outage, not three burnt attempts per symbol-day."""
    profile, store, ledger = env
    symbols = ["000041", "000042", "000043"]
    client = _FakeToss({})
    summary = _run(profile, store, ledger, client, [_task(_DAY1, tuple(symbols))], _eod([_DAY1], symbols))
    assert summary.outage_aborted is True and summary.failed == 0
    assert ledger._read_all().empty


def test_small_sample_below_min_is_recorded_not_aborted(tmp_path, monkeypatch) -> None:
    """Fewer attempted symbols than the minimum sample never trigger the outage abort (no permanent wedge)."""
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    profile = CollectionSettings(
        COLLECTION_ROOT=tmp_path / "capture", COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS=(),
        COLLECTION_TOSS_OUTAGE_MIN_SAMPLE=5, _env_file=None,
    )
    store = CaptureStore(tmp_path / "capture")
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    client = _FakeToss(errors={"000051": {"error": {"code": "overloaded"}}})
    summary = _run(profile, store, ledger, client, [_task(_DAY1, ("000051",))], _eod([_DAY1], ["000051"]))
    assert summary.outage_aborted is False and summary.failed == 1
    assert ledger._read_all()["status"].tolist() == ["FAILED"]


def test_effective_floor_is_the_later_date_and_unknown_stays_unknown() -> None:
    """The usable-from date can only raise the floor; an unknown vendor floor is never replaced by it."""
    assert trb.effective_floor("2021-12-20", "2023-01-02") == "2023-01-02"
    assert trb.effective_floor("2024-05-01", "2023-01-02") == "2024-05-01"
    assert trb.effective_floor(None, "2023-01-02") is None


def test_vendor_failures_count_toward_outage(env) -> None:
    """vendor_failure reasons join the outage share, not the retryable failure count."""
    profile, store, ledger = env
    client = _FakeToss(errors={"000031": {"error": {"code": "overloaded"}}})
    summary = _run(profile, store, ledger, client, [_task(_DAY1, ("000031",))],
                   _eod([_DAY1], ["000031"]))
    assert summary.outage_aborted is True and summary.failed == 0
    assert ledger._read_all().empty


def test_commit_order_keeps_symbol_pending_on_ledger_failure(env, monkeypatch) -> None:
    """When the ledger append fails the partition exists but the symbol stays non-terminal."""
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    client = _complete_client(["000041"], _DAY1)

    def _boom(*args, **kwargs) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(ledger, "record", _boom)
    with pytest.raises(OSError, match="disk gone"):
        _run(profile, store, ledger, client, [_task(_DAY1, ("000041",))],
             _eod([_DAY1], ["000041"]))
    stored = pd.read_parquet(intraday_partition_path(1, _DAY1, "regular"))
    assert "000041" in stored["symbol"].tolist()
    assert ledger.terminal_symbols(_DAY1, "regular") == frozenset()


def test_unreadable_partition_fails_loud_last(env) -> None:
    """Readable dates complete first; then OSError names the corrupt date."""
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    corrupt = intraday_partition_path(1, _DAY1, "regular")
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"not a parquet file")
    client = _complete_client(["000042"], _DAY2)
    with pytest.raises(OSError, match="2026-03-02"):
        _run(profile, store, ledger, client,
             [_task(_DAY1, ("000001",)), _task(_DAY2, ("000042",))],
             _eod([_DAY1, _DAY2], ["000001", "000042"]))
    assert set(client.calls) == {"000042"}
    assert ledger.terminal_symbols(_DAY2, "regular") == frozenset({"000042"})


def test_empty_date_writes_nothing(env) -> None:
    """A date with no pending symbols performs no network call and writes nothing."""
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    ledger.record_cached_absent(
        _DAY1, "regular", ["000051"], reason="toss_stock_not_found_cached",
        run_id="seed", attempted_at=_FIXED_NOW, vendor="toss",
    )
    client = _complete_client(["000051"], _DAY1)
    summary = _run(profile, store, ledger, client, [_task(_DAY1, ("000051",))],
                   _eod([_DAY1], ["000051"]))
    assert client.calls == []
    assert (summary.tasks_done, summary.tasks_remaining) == (1, 0)
    assert not intraday_partition_path(1, _DAY1, "regular").exists()


def test_deadline_stops_new_dates_but_finishes_started_ones(env) -> None:
    """After stop_at no new date starts; a started date still commits fully."""
    profile, store, ledger = env
    stop = datetime(2026, 9, 29, 6, 0, tzinfo=SEOUL)
    client = _complete_client(["000061", "000062"], _DAY1)
    ticks = itertools.chain([_FIXED_NOW] * 6, itertools.repeat(stop))
    summary = _run(
        profile, store, ledger, client,
        [_task(_DAY1, ("000061",)), _task(_DAY2, ("000062",))],
        _eod([_DAY1, _DAY2], ["000061", "000062"]),
        now_fn=lambda: next(ticks), stop_at=stop,
    )
    assert set(client.calls) == {"000061"}
    assert summary.stopped_by_deadline is True
    assert (summary.tasks_done, summary.tasks_remaining) == (1, 1)
    started_client = _complete_client(["000063"], _DAY1)
    started_ticks = itertools.chain([_FIXED_NOW, _FIXED_NOW], itertools.repeat(stop))
    started = _run(
        profile, store, ledger, started_client, [_task(_DAY1, ("000063",))],
        _eod([_DAY1], ["000063"]), now_fn=lambda: next(started_ticks), stop_at=stop,
    )
    assert started.complete == 1 and started.stopped_by_deadline is False


def test_blackout_stops_when_it_outlasts_the_deadline(env) -> None:
    """A blackout ending after stop_at stops the run with no new date started."""
    from src.data.intraday_store import intraday_partition_path

    profile = CollectionSettings(
        COLLECTION_ROOT=env[0].COLLECTION_ROOT,
        COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS=("0900-1000",),
        _env_file=None,
    )
    _, store, ledger = env
    morning = datetime(2026, 9, 10, 9, 30, tzinfo=SEOUL)
    stop = datetime(2026, 9, 10, 9, 45, tzinfo=SEOUL)
    client = _complete_client(["000071"], _DAY1)
    summary = _run(profile, store, ledger, client, [_task(_DAY1, ("000071",))],
                   _eod([_DAY1], ["000071"]), now_fn=lambda: morning, stop_at=stop)
    assert client.calls == []
    assert summary.stopped_by_deadline is True
    assert not intraday_partition_path(1, _DAY1, "regular").exists()


def test_blackout_wait_then_proceeds(env, monkeypatch) -> None:
    """Inside a blackout the runner waits (without its own sleeps) and then fetches."""
    profile = CollectionSettings(
        COLLECTION_ROOT=env[0].COLLECTION_ROOT,
        COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS=("0900-1000",),
        _env_file=None,
    )
    _, store, ledger = env
    waited: list = []

    async def _fake_wait(windows, **kwargs) -> None:
        waited.append(tuple(windows))

    monkeypatch.setattr(trb, "wait_for_blackout", _fake_wait)
    morning = datetime(2026, 9, 10, 9, 30, tzinfo=SEOUL)
    client = _complete_client(["000072"], _DAY1)
    summary = _run(profile, store, ledger, client, [_task(_DAY1, ("000072",))],
                   _eod([_DAY1], ["000072"]), now_fn=lambda: morning)
    assert len(waited) == 1
    assert summary.complete == 1


def test_blackout_wait_then_deadline(env, monkeypatch) -> None:
    """Time passing during a blackout wait can push the run past the deadline."""
    profile = CollectionSettings(
        COLLECTION_ROOT=env[0].COLLECTION_ROOT,
        COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS=("0900-0932",),
        _env_file=None,
    )
    _, store, ledger = env
    state = {"now": datetime(2026, 9, 10, 9, 30, tzinfo=SEOUL)}
    stop = datetime(2026, 9, 10, 9, 35, tzinfo=SEOUL)

    async def _advancing_wait(windows, **kwargs) -> None:
        state["now"] = stop + timedelta(minutes=1)

    monkeypatch.setattr(trb, "wait_for_blackout", _advancing_wait)
    client = _complete_client(["000073"], _DAY1)
    summary = _run(profile, store, ledger, client, [_task(_DAY1, ("000073",))],
                   _eod([_DAY1], ["000073"]), now_fn=lambda: state["now"], stop_at=stop)
    assert client.calls == []
    assert summary.stopped_by_deadline is True


def test_bounded_concurrency(env) -> None:
    """In-flight symbol-days never exceed the configured concurrency."""
    profile, store, ledger = env
    profile = CollectionSettings(
        COLLECTION_ROOT=profile.COLLECTION_ROOT,
        COLLECTION_TOSS_BACKFILL_CONCURRENCY=3,
        COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS=(),
        _env_file=None,
    )
    symbols = [f"{i:06d}" for i in range(1, 10)]
    client = _complete_client(symbols, _DAY1, latency=0.01)
    summary = _run(profile, store, ledger, client, [_task(_DAY1, tuple(symbols))],
                   _eod([_DAY1], symbols))
    assert summary.complete == 9
    assert client.max_inflight <= 3


def test_session_factory_shapes(env) -> None:
    """Session-factory clients (context and plain) are both supported."""

    class _Ctx:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *args):
            return False

    class _CtxClient(_FakeToss):
        def create_session(self):
            return _Ctx()

    class _PlainClient(_FakeToss):
        def create_session(self):
            return object()

    profile, store, ledger = env
    first = _run(profile, store, ledger, _CtxClient({"000081": list(reversed(_regular_grid(_DAY1)))}),
                 [_task(_DAY1, ("000081",))], _eod([_DAY1], ["000081"]))
    assert first.complete == 1
    second = _run(profile, store, ledger, _PlainClient({"000082": list(reversed(_regular_grid(_DAY2)))}),
                  [_task(_DAY2, ("000082",))], _eod([_DAY2], ["000082"]))
    assert second.complete == 1


def test_runner_rejects_bad_contracts(env) -> None:
    """A missing client or a naive deadline raises immediately."""
    profile, store, ledger = env
    with pytest.raises(ValueError, match="client"):
        _run(profile, store, ledger, None, [_task(_DAY1, ("000001",))], {})
    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(
            trb.run_toss_regular_backfill(
                as_of=_AS_OF, stop_at=datetime(2026, 9, 29, 6, 0), profile=profile,
                client=_FakeToss(), store=store, ledger=ledger,
                tasks=[_task(_DAY1, ("000001",))], eod_volumes={},
            )
        )


# ---------------------------------------------------------------- CLI


def _wide_history(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["date", "symbol", "open", "high", "low", "close", "prev_close", "volume",
                 "market_cap_100m", "trade_value_100m", "close_raw", "market"],
    )


def _cli_rows() -> list[tuple]:
    return [
        (day, symbol, 1000.0, 1050.0, 990.0, 1050.0, 1000.0, 39000.0, 900.0, 500.0, 1000.0, "KOSPI")
        for day in (_DAY1, _DAY2)
        for symbol in ("000001", "000002")
    ]


class _CliTossClient:
    """Module-level fake behind the TossApiClient patch point, scripted per test via class attrs."""

    grids: dict[str, list[dict]] = {}
    errors: dict[str, dict] = {}
    raises: set[str] = frozenset()

    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[str] = []

    async def ensure_token(self, session) -> str:
        return "token"

    async def get_candles(self, session, symbol: str, *, interval: str = "1m",
                          count: int = 200, before: str | None = None,
                          adjusted: bool | None = None) -> dict:
        type(self).calls.append(str(symbol))
        if str(symbol) in type(self).raises:
            raise ConnectionError("boom")
        if str(symbol) in type(self).errors:
            return dict(type(self).errors[str(symbol)])
        grid = type(self).grids.get(str(symbol), [])
        bound = datetime.fromisoformat(str(before)) if before else None
        out = []
        for candle in grid:
            current = datetime.fromisoformat(str(candle.get("timestamp", "")))
            if bound is None or current <= bound:
                out.append(candle)
                if len(out) >= int(count):
                    break
        return {"result": {"candles": out}}


_CliTossClient.grids = {}
_CliTossClient.errors = {}
_CliTossClient.raises = frozenset()


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    from src.data import intraday_store
    from src import settings as app_settings

    history_dir = tmp_path / "history"
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", history_dir, raising=False)
    hist_path = tmp_path / "price_history.parquet"
    _wide_history(_cli_rows()).to_parquet(hist_path)
    monkeypatch.setattr(app_settings, "PRICE_HISTORY_PARQUET_PATH", hist_path, raising=False)
    monkeypatch.setattr(app_settings, "TOSS_APP_KEY", "dummy", raising=False)
    monkeypatch.setattr(app_settings, "TOSS_APP_SECRET", "dummy", raising=False)
    monkeypatch.setenv("COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS", "")
    monkeypatch.setenv("COLLECTION_TOSS_OUTAGE_MIN_SAMPLE", "1")
    monkeypatch.setattr("src.api.toss.client.TossApiClient", _CliTossClient)
    _CliTossClient.grids = {
        symbol: [c for day in sorted((_DAY1, _DAY2), reverse=True) for c in reversed(_regular_grid(day))]
        for symbol in ("000001", "000002")
    }
    _CliTossClient.errors = {}
    _CliTossClient.raises = frozenset()
    _CliTossClient.calls = []

    async def _floor(client, session, *, trading_days, reference_symbol):
        assert list(trading_days) == sorted(trading_days)
        return "2026-01-01"

    monkeypatch.setattr(trb, "probe_toss_retention_floor", _floor)
    return tmp_path, history_dir


def test_plan_only_writes_nothing(cli_env, capsys) -> None:
    """--plan-only prints the estimate and creates no partition, ledger, manifest or lock file."""
    _, history_dir = cli_env
    assert trb.main(["--as-of", "2026-03-10", "--plan-only"]) == 0
    out = capsys.readouterr().out
    assert "planned_dates=2" in out
    assert "planned_symbol_days=4" in out
    assert "estimated_calls=8" in out
    assert (history_dir.exists() and any(history_dir.rglob("*"))) is False


def test_cli_run_completes_and_writes_ledger(cli_env) -> None:
    """A full CLI run certifies every planned symbol-day under vendor toss."""
    _, history_dir = cli_env
    assert trb.main(["--as-of", "2026-03-10", "--start", "2026-03-01", "--end", "2026-03-10",
                     "--max-dates", "5", "--stop-at", "060000"]) == 0
    ledger_path = history_dir / "intraday" / "backfill_ledger" / "toss_regular.parquet"
    frame = pd.read_parquet(ledger_path)
    assert len(frame) == 4
    assert (frame["vendor"] == "toss").all()
    assert (frame[frame["status"] == "COMPLETE"]["price_basis"] == "toss_raw").all()
    assert not list(history_dir.rglob("*.lock"))


def test_cli_outage_returns_nonzero_and_keeps_retryable(cli_env) -> None:
    """An outage abort exits non-zero without burning the attempt cap."""
    _, history_dir = cli_env
    _CliTossClient.grids = {}
    _CliTossClient.raises = frozenset({"000001", "000002"})
    assert trb.main(["--as-of", "2026-03-10"]) == 1
    ledger_path = history_dir / "intraday" / "backfill_ledger" / "toss_regular.parquet"
    assert not ledger_path.exists()


def test_cli_unknown_retention_floor_aborts_without_planning(cli_env, monkeypatch) -> None:
    """An empty vendor (floor unknown) must not plan every pre-retention date and burn attempts."""
    _, history_dir = cli_env

    async def _unknown(client, session, *, trading_days, reference_symbol):
        return None

    monkeypatch.setattr(trb, "probe_toss_retention_floor", _unknown)
    assert trb.main(["--as-of", "2026-03-10"]) == 1
    assert not (history_dir / "intraday" / "backfill_ledger" / "toss_regular.parquet").exists()
    assert _CliTossClient.calls == []


def test_cli_single_instance(cli_env) -> None:
    """A held ledger lock exits non-zero immediately without touching the ledger."""
    from src.utils.file_lock import exclusive_file_lock

    _, history_dir = cli_env
    ledger_path = history_dir / "intraday" / "backfill_ledger" / "toss_regular.parquet"
    with exclusive_file_lock(
        trb.toss_run_lock_path(ledger_path), timeout_seconds=30, purpose="test"
    ):
        assert trb.main(["--as-of", "2026-03-10"]) == 2
    assert not ledger_path.exists()


def test_cli_rejects_bad_args_and_missing_creds(cli_env, monkeypatch) -> None:
    """Bad ranges, bad stop-at, bad max-dates and missing credentials fail loudly."""
    from src import settings as app_settings

    with pytest.raises(ValueError, match="plan range"):
        trb.main(["--as-of", "2026-03-10", "--start", "2026-03-05", "--end", "2026-03-01"])
    with pytest.raises(ValueError, match="stop-at"):
        trb.main(["--as-of", "2026-03-10", "--stop-at", "nope"])
    with pytest.raises(ValueError, match="max-dates"):
        trb.main(["--as-of", "2026-03-10", "--max-dates", "0"])
    monkeypatch.setattr(app_settings, "TOSS_APP_KEY", "", raising=False)
    with pytest.raises(RuntimeError, match="credentials"):
        trb.main(["--as-of", "2026-03-10", "--plan-only"])


# ---------------------------------------------------------------- consolidated


def _consolidated_task(day: str, symbols: tuple[str, ...]) -> ExtendedBackfillTask:
    return ExtendedBackfillTask(snapshot_date=day, session="regular_consolidated", symbols=symbols)


def _run_consolidated(profile, store, ledger, client, tasks, eod_volumes, now_fn=None, stop_at=None):
    return asyncio.run(
        trb.run_toss_consolidated_backfill(
            as_of=_AS_OF, stop_at=stop_at if stop_at is not None else _FAR_FUTURE,
            profile=profile, client=client, store=store, ledger=ledger,
            tasks=tasks, eod_volumes=eod_volumes,
            now_fn=now_fn if now_fn is not None else (lambda: _FIXED_NOW),
        )
    )


def _seed_regular_consolidated_ledger(ledger, day: str, symbols: tuple[str, ...]) -> None:
    from src.data.capture_contracts import ArtifactRef, CoverageEntry

    entries = [
        CoverageEntry(
            symbol=symbol, dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="regular",
            scheduled_at=None, status=CaptureStatus.NOT_APPLICABLE, rows=390,
            first_event_time=None, last_event_time=None, reason="toss_consolidated_tape",
            raw_refs=(ArtifactRef(path=f"raw/{symbol}.json.gz", sha256="b" * 64, bytes=8, rows=0),),
        )
        for symbol in symbols
    ]
    ledger.record(day, "regular", entries, run_id="seed", attempted_at=_FIXED_NOW, vendor="toss")


def test_consolidated_day_lands_in_its_own_session(env) -> None:
    """A consolidated day is stored only in regular_consolidated with a same-session ledger row."""
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    _seed_regular_consolidated_ledger(ledger, _DAY1, ("000005",))
    plan = trb.enumerate_toss_consolidated_tasks(ledger=ledger)
    assert [t.snapshot_date for t in plan.tasks] == [_DAY1]
    assert plan.tasks[0].session == "regular_consolidated"
    client = _complete_client(["000005"], _DAY1, volume=150)
    summary = _run_consolidated(
        profile, store, ledger, client, list(plan.tasks), _eod([_DAY1], ["000005"], volume=39000.0)
    )
    assert summary.consolidated == 1 and summary.failed == 0
    target = intraday_partition_path(1, _DAY1, "regular_consolidated")
    stored = pd.read_parquet(target)
    assert sorted(stored["symbol"].unique().tolist()) == ["000005"]
    assert (stored["vendor"] == "toss").all()
    assert not intraday_partition_path(1, _DAY1, "regular").exists()
    frame = ledger._read_all()
    cons_rows = frame[frame["session"] == "regular_consolidated"]
    assert len(cons_rows) == 1 and cons_rows.iloc[0]["reason"] == "toss_consolidated_tape"


def test_consolidated_reopen_from_existing_ledger(env) -> None:
    """Planning reads the regular ledger verdicts; a rerun after completion plans zero tasks."""
    profile, store, ledger = env
    _seed_regular_consolidated_ledger(ledger, _DAY1, ("000005", "000006"))
    _seed_regular_consolidated_ledger(ledger, _DAY2, ("000007",))
    plan = trb.enumerate_toss_consolidated_tasks(ledger=ledger)
    by_day = {task.snapshot_date: task.symbols for task in plan.tasks}
    assert by_day == {_DAY1: ("000005", "000006"), _DAY2: ("000007",)}
    grids = {s: list(reversed(_regular_grid(_DAY1 if s != "000007" else _DAY2, 150))) for s in ("000005", "000006", "000007")}
    client = _FakeToss(grids)
    summary = _run_consolidated(
        profile, store, ledger, client, list(plan.tasks),
        _eod([_DAY1, _DAY2], ["000005", "000006", "000007"], volume=39000.0),
    )
    assert summary.consolidated == 3
    assert trb.enumerate_toss_consolidated_tasks(ledger=ledger).tasks == ()


def test_consolidated_no_cross_contamination(env) -> None:
    """An accepted day and a consolidated day on one date each land only in their own partition."""
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    accepted_client = _complete_client(["000001"], _DAY1)
    accepted = _run(profile, store, ledger, accepted_client, [_task(_DAY1, ("000001",))], _eod([_DAY1], ["000001"]))
    assert accepted.complete == 1
    _seed_regular_consolidated_ledger(ledger, _DAY1, ("000005",))
    cons_client = _complete_client(["000005"], _DAY1, volume=150)
    summary = _run_consolidated(
        profile, store, ledger, cons_client,
        [_consolidated_task(_DAY1, ("000005",))], _eod([_DAY1], ["000005"], volume=39000.0),
    )
    assert summary.consolidated == 1
    regular_rows = pd.read_parquet(intraday_partition_path(1, _DAY1, "regular"))
    assert sorted(regular_rows["symbol"].unique().tolist()) == ["000001"]
    cons_rows = pd.read_parquet(intraday_partition_path(1, _DAY1, "regular_consolidated"))
    assert sorted(cons_rows["symbol"].unique().tolist()) == ["000005"]


def test_consolidated_outage_guard_applies(env) -> None:
    """Transport failures record nothing and abort the consolidated run like the regular runner."""
    profile, store, ledger = env
    _seed_regular_consolidated_ledger(ledger, _DAY1, ("000011", "000012", "000013"))
    client = _FakeToss(raises={"000011", "000012", "000013"})
    summary = _run_consolidated(
        profile, store, ledger, client,
        [_consolidated_task(_DAY1, ("000011", "000012", "000013"))],
        _eod([_DAY1], ["000011", "000012", "000013"]),
    )
    assert summary.outage_aborted is True
    assert ledger.terminal_symbols(_DAY1, "regular_consolidated") == frozenset()


def test_consolidated_plan_only_counts_and_estimates(cli_env, capsys) -> None:
    """--consolidated --plan-only prints the ledger-derived count without any vendor call."""
    from src.backfill.intraday.extended_session_backfill import ExtendedBackfillLedger

    _, history_dir = cli_env
    ledger = ExtendedBackfillLedger(history_dir / "intraday" / "backfill_ledger" / "toss_regular.parquet")
    _seed_regular_consolidated_ledger(ledger, _DAY1, ("000001", "000002"))
    assert trb.main(["--as-of", "2026-03-10", "--consolidated", "--plan-only"]) == 0
    out = capsys.readouterr().out
    assert "planned_symbol_days=2" in out
    assert "estimated_calls=4" in out
    assert _CliTossClient.calls == []
    _CliTossClient.grids = {
        symbol: list(reversed(_regular_grid(_DAY1, 150))) for symbol in ("000001", "000002")
    }
    assert trb.main(["--as-of", "2026-03-10", "--consolidated"]) == 0
    assert trb.main(["--as-of", "2026-03-10", "--consolidated", "--plan-only"]) == 0
    out = capsys.readouterr().out
    assert "planned_symbol_days=0" in out


def test_consolidated_replan_retries_failed_rows_but_not_terminal_ones(env) -> None:
    """A non-terminal FAILED consolidated row is replanned; a terminal row is never replanned."""
    _profile, _store, ledger = env
    _seed_regular_consolidated_ledger(ledger, _DAY1, ("000005", "000006"))
    ledger.record(
        _DAY1, "regular_consolidated",
        [_failed_entry("000005", "transport:boom")], run_id="r1", attempted_at=_FIXED_NOW, vendor="toss",
    )
    from src.data.capture_contracts import ArtifactRef, CoverageEntry

    done = CoverageEntry(
        symbol="000006", dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="regular_consolidated",
        scheduled_at=None, status=CaptureStatus.NOT_APPLICABLE, rows=0,
        first_event_time=None, last_event_time=None, reason="toss_consolidated_tape",
        raw_refs=(ArtifactRef(path="raw/000006.json.gz", sha256="c" * 64, bytes=8, rows=0),),
    )
    ledger.record(_DAY1, "regular_consolidated", [done], run_id="r2", attempted_at=_FIXED_NOW, vendor="toss")
    plan = trb.enumerate_toss_consolidated_tasks(ledger=ledger)
    assert {t.snapshot_date: t.symbols for t in plan.tasks} == {_DAY1: ("000005",)}
