"""Invariant scenarios for the nightly extended-session 1m backfill."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from src.backfill.intraday.extended_session_backfill import (
    ExtendedBackfillLedger,
    ExtendedBackfillTask,
    enumerate_extended_session_tasks,
    run_extended_session_backfill,
)
from src.config.collection import CollectionSettings
from src.config.market_session import KRX_AFTERMARKET_START_DATE
from src.data.capture_contracts import SEOUL, CaptureDataset, CaptureStatus, CoverageEntry
from src.data.capture_store import CaptureStore

AS_OF = date(2026, 9, 28)


def _ph(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["date", "symbol", "close", "prev_close"])


def _pairs(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["snapshot_date", "symbol"])


def _enum(ph: pd.DataFrame, pairs: pd.DataFrame | None = None, **kw) -> list[ExtendedBackfillTask]:
    return enumerate_extended_session_tasks(
        as_of=kw.get("as_of", AS_OF),
        retention_days=kw.get("retention_days", 365),
        min_change_ratio=kw.get("min_change_ratio", 0.02),
        price_history=ph,
        candidate_pairs=pairs if pairs is not None else _pairs([]),
    )


def _by_key(tasks: list[ExtendedBackfillTask]) -> dict[tuple[str, str], tuple[str, ...]]:
    return {(t.snapshot_date, t.session): t.symbols for t in tasks}


def test_entry_day_universe_unions_candidates_and_change_screen() -> None:
    ph = _ph([
        ("2026-03-02", "000001", 103.0, 100.0),
        ("2026-03-02", "000002", 101.0, 100.0),
        ("2026-03-02", "000003", 100.0, 100.0),
        ("2026-03-03", "000001", 100.0, 100.0),
    ])
    tasks = _by_key(_enum(ph, _pairs([("2026-03-02", "000002")])))
    assert tasks[("2026-03-02", "nxt_aftermarket")] == ("000001", "000002")


def test_premarket_task_targets_next_trading_day_from_calendar() -> None:
    ph = _ph([
        ("2026-03-06", "000001", 110.0, 100.0),  # 금요일
        ("2026-03-09", "000001", 100.0, 110.0),  # 다음 거래일(월)
    ])
    tasks = _by_key(_enum(ph))
    assert tasks[("2026-03-09", "nxt_premarket")] == ("000001",)
    assert ("2026-03-07", "nxt_premarket") not in tasks


def test_retention_and_as_of_bounds_are_exclusive() -> None:
    old = (AS_OF - timedelta(days=400)).isoformat()
    ph = _ph([
        (old, "000001", 110.0, 100.0),
        (AS_OF.isoformat(), "000001", 110.0, 100.0),
        ("2026-03-02", "000001", 110.0, 100.0),
    ])
    days = {t.snapshot_date for t in _enum(ph)}
    assert old not in days
    assert AS_OF.isoformat() not in days
    assert "2026-03-02" in days


def test_krx_aftermarket_tasks_start_at_krx_start_date() -> None:
    start = date.fromisoformat(KRX_AFTERMARKET_START_DATE)
    before = (start - timedelta(days=3)).isoformat()
    after = (start + timedelta(days=1)).isoformat()
    ph = _ph([(before, "000001", 110.0, 100.0), (after, "000001", 110.0, 100.0)])
    tasks = _by_key(_enum(ph))
    assert (after, "krx_aftermarket") in tasks
    assert (before, "krx_aftermarket") not in tasks


def test_tasks_are_ordered_oldest_first_and_deterministic() -> None:
    rows = [
        ("2026-05-04", "000002", 110.0, 100.0),
        ("2026-03-02", "000001", 110.0, 100.0),
        ("2026-04-01", "000003", 110.0, 100.0),
    ]
    first = _enum(_ph(rows))
    second = _enum(_ph(list(reversed(rows))))
    assert first == second
    keys = [(t.snapshot_date, t.session) for t in first]
    assert keys == sorted(keys)


def test_no_lookahead_in_the_screen() -> None:
    base = [("2026-03-02", "000001", 110.0, 100.0), ("2026-03-03", "000002", 100.0, 100.0)]
    changed = [("2026-03-02", "000001", 110.0, 100.0), ("2026-03-03", "000002", 150.0, 100.0)]
    t_base = _by_key(_enum(_ph(base)))[("2026-03-02", "nxt_aftermarket")]
    t_changed = _by_key(_enum(_ph(changed)))[("2026-03-02", "nxt_aftermarket")]
    assert t_base == t_changed == ("000001",)


def test_nonpositive_prev_close_is_excluded() -> None:
    tasks = _enum(_ph([("2026-03-02", "000001", 110.0, 0.0)]))
    assert tasks == []


# ---------------------------------------------------------------- runner


def _row(h: str, close: str, vol: str, cum: str, day: str) -> dict:
    return {
        "stck_bsop_date": day.replace("-", ""), "stck_cntg_hour": h, "stck_oprc": close,
        "stck_hgpr": close, "stck_lwpr": close, "stck_prpr": close, "cntg_vol": vol, "acml_tr_pbmn": cum,
    }


class _FakeKis:
    """Historical-only KIS stand-in keyed by symbol."""

    def __init__(
        self,
        day: str,
        traded: set[str],
        fail: set[str] | None = None,
        bars: dict[str, tuple[str, str, str]] | None = None,
        hms: str = "160000",
    ) -> None:
        self._day = day
        self._hms = hms
        self._traded = traded
        self._fail = fail or set()
        self._bars = dict(bars) if bars else {}
        self.requested: list[str] = []

    async def get_historical_minute_chart(self, session, code, target_date, **kwargs):
        self.requested.append(code)
        if code in self._fail:
            raise RuntimeError("down")
        if code in self._traded:
            close, vol, cum = self._bars.get(code, ("1000", "10", "10000"))
            return {"rt_cd": "0", "output2": [_row(self._hms, close, vol, cum, self._day)]}
        return {"rt_cd": "0", "output2": []}


def _kw_row(day: str, hms: str, price: str, vol: str) -> dict:
    return {"cntr_tm": day.replace("-", "") + hms, "cur_prc": price, "open_pric": price, "high_pric": price,
            "low_pric": price, "trde_qty": vol}


class _FakeKiwoom:
    """Kiwoom raw-basis stand-in: per-symbol raw bars, or empty (symbol dropped from NXT)."""

    def __init__(self, day: str, raw: dict[str, str], error: set[str] | None = None) -> None:
        self._day = day
        self._raw = dict(raw)
        self._error = set(error or set())
        self.calls: list[tuple[str, str]] = []

    async def _chart(self, method: str, session, code: str, target_date: str) -> dict:
        self.calls.append((method, code))
        if code in self._error:
            raise RuntimeError("kiwoom down")
        if code not in self._raw:
            return {"rt_cd": "0", "output2": [], "vendor": "kiwoom"}
        hms = "160000" if method == "get_nxt_minute_chart" else "080500"
        return {"rt_cd": "0", "output2": [_kw_row(self._day, hms, self._raw[code], "770")], "vendor": "kiwoom"}

    async def get_nxt_minute_chart(self, session, code, target_date):
        return await self._chart("get_nxt_minute_chart", session, code, target_date)

    async def get_nxt_premarket_chart(self, session, code, target_date):
        return await self._chart("get_nxt_premarket_chart", session, code, target_date)


def _ref(rows: list[tuple[str, str, float, float]]):
    from src.backfill.intraday.price_basis import PriceReference

    return PriceReference.from_price_history(
        pd.DataFrame(rows, columns=["date", "symbol", "close", "close_raw"])
    )


def _raw_ref(day: str, symbols: set[str] | list[str], raw: int = 1000) -> object:
    return _ref([(day, symbol, float(raw), float(raw)) for symbol in symbols])


@pytest.fixture
def env(tmp_path, monkeypatch):
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    profile = CollectionSettings(COLLECTION_ROOT=tmp_path / "capture")
    store = CaptureStore(tmp_path / "capture")
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    return profile, store, ledger


def _run(profile, store, ledger, clients, tasks, price_reference, now_fn=None, stop_at=None, raw_client=None):
    stop = stop_at or datetime(2099, 1, 1, tzinfo=SEOUL)
    return asyncio.run(run_extended_session_backfill(
        as_of=AS_OF, stop_at=stop, profile=profile, clients=clients, store=store,
        ledger=ledger, tasks=tasks, price_reference=price_reference, now_fn=now_fn, raw_client=raw_client,
    ))


def _entry(symbol: str, status: CaptureStatus, session: str = "nxt_aftermarket") -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol, dataset=CaptureDataset.MINUTE_BARS, venue="NXT", session=session,
        scheduled_at=None, status=status, rows=0, first_event_time=None, last_event_time=None,
        reason="seed", raw_refs=(),
    )


def test_ledger_terminal_and_stored_symbols_are_not_refetched(env) -> None:
    from src.data.intraday_schema import normalize_bar_frame
    from src.data.intraday_store import write_intraday_partition

    profile, store, ledger = env
    day = "2026-03-02"
    ledger.record(day, "nxt_aftermarket", [_entry("000001", CaptureStatus.COMPLETE)],
                  run_id="seed", attempted_at=datetime.now(SEOUL))
    stored = normalize_bar_frame(pd.DataFrame([_row("160000", "1000", "10", "10000", day)]), "kis", day, "000002")
    write_intraday_partition(stored, 1, day, "nxt_aftermarket")
    kis = _FakeKis(day, traded={"000003"})
    summary = _run(profile, store, ledger, [kis],
                   [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001", "000002", "000003"))],
                   _raw_ref(day, {"000003"}))
    assert kis.requested == ["000003"]
    assert summary.complete == 1 and summary.tasks_done == 1


def _manifest_files(store) -> list[str]:
    root = store.root / "manifests"
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()) if root.exists() else []


def test_noop_task_publishes_no_manifest_and_leaves_store_untouched(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    ledger.record(day, "nxt_aftermarket", [_entry("000001", CaptureStatus.COMPLETE)],
                  run_id="seed", attempted_at=datetime.now(SEOUL))
    ledger_before = ledger._read_all().copy()
    kis = _FakeKis(day, traded=set())
    task = ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))

    first = _run(profile, store, ledger, [kis], [task], _raw_ref(day, {"000001"}))
    second = _run(profile, store, ledger, [kis], [task], _raw_ref(day, {"000001"}))

    assert kis.requested == []
    assert _manifest_files(store) == []
    assert ledger._read_all().equals(ledger_before)
    assert first.tasks_done == 1 and second.tasks_done == 1
    assert first.complete == 0 and first.failed == 0 and first.exhausted == 0


def test_pending_task_publishes_exactly_one_manifest_even_when_all_fail(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    kis = _FakeKis(day, traded={"000003"}, fail={"000003"})

    summary = _run(profile, store, ledger, [kis], [ExtendedBackfillTask(day, "nxt_aftermarket", ("000003",))],
                   _raw_ref(day, {"000003"}))

    run_dirs = {name.rsplit("/", 1)[0] for name in _manifest_files(store)}
    assert summary.failed == 1
    assert len(run_dirs) == 1
    assert any(name.endswith("manifest-partial.json") for name in _manifest_files(store))


def test_failed_symbols_are_retried_on_next_run(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    ledger.record(day, "nxt_aftermarket", [_entry("000003", CaptureStatus.FAILED)],
                  run_id="seed", attempted_at=datetime.now(SEOUL))
    kis = _FakeKis(day, traded={"000003"})
    _run(profile, store, ledger, [kis], [ExtendedBackfillTask(day, "nxt_aftermarket", ("000003",))],
         _raw_ref(day, {"000003"}))
    assert kis.requested == ["000003"]
    assert "000003" in ledger.terminal_symbols(day, "nxt_aftermarket")


def test_deadline_stops_before_next_task(env) -> None:
    profile, store, ledger = env
    stop = datetime(2026, 9, 29, 6, 50, tzinfo=SEOUL)
    ticks = iter([stop - timedelta(minutes=1)] * 3 + [stop] * 20)
    kis = _FakeKis("2026-03-02", traded={"000001"})
    tasks = [
        ExtendedBackfillTask("2026-03-02", "nxt_aftermarket", ("000001",)),
        ExtendedBackfillTask("2026-03-03", "nxt_aftermarket", ("000001",)),
    ]
    summary = _run(profile, store, ledger, [kis], tasks, _ref([
        ("2026-03-02", "000001", 1000.0, 1000.0),
        ("2026-03-03", "000001", 1000.0, 1000.0),
    ]), now_fn=lambda: next(ticks), stop_at=stop)
    assert summary.tasks_done == 1
    assert summary.tasks_remaining == 1
    assert summary.stopped_by_deadline is True
    assert kis.requested == ["000001"]


def test_unlisted_nxt_symbols_are_terminal_without_partition_rows(env) -> None:
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    day = "2026-03-02"
    summary = _run(profile, store, ledger, [_FakeKis(day, traded=set())],
                   [ExtendedBackfillTask(day, "nxt_aftermarket", ("000009",))], _ref([]))
    assert summary.not_listed == 1
    assert "000009" in ledger.terminal_symbols(day, "nxt_aftermarket")
    assert not intraday_partition_path(1, day, "nxt_aftermarket").exists()
    manifests = [m for m in store.read_manifests(day) if m.context.capture_reason == "extended-backfill"]
    assert manifests and manifests[0].status is CaptureStatus.COMPLETE


def test_symbols_spread_round_robin_over_clients(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    a, b = _FakeKis(day, traded={"000001", "000002"}), _FakeKis(day, traded={"000001", "000002"})
    _run(profile, store, ledger, [a, b], [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001", "000002"))],
         _raw_ref(day, {"000001", "000002"}))
    assert a.requested == ["000001"] and b.requested == ["000002"]


def test_persistence_failure_is_loud_and_non_terminal(env, monkeypatch) -> None:
    from src.backfill.intraday import extended_session_backfill as mod

    profile, store, ledger = env
    day = "2026-03-02"

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mod, "write_intraday_partition", _boom)
    with pytest.raises(OSError, match="disk full"):
        _run(profile, store, ledger, [_FakeKis(day, traded={"000001"})],
             [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))],
             _raw_ref(day, {"000001"}))
    assert "000001" not in ledger.terminal_symbols(day, "nxt_aftermarket")


def test_run_ids_are_unique_per_task_run(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    task = ExtendedBackfillTask(day, "nxt_aftermarket", ("000009",))
    _run(profile, store, ledger, [_FakeKis(day, traded=set())], [task], _ref([]))
    ledger_path = ledger.path
    ledger_path.unlink()
    _run(profile, store, ledger, [_FakeKis(day, traded=set())], [task], _ref([]))
    run_ids = {m.context.run_id for m in store.read_manifests(day) if m.context.capture_reason == "extended-backfill"}
    assert len(run_ids) == 2


def test_runner_rejects_empty_clients_and_naive_stop(env) -> None:
    profile, store, ledger = env
    with pytest.raises(ValueError, match="clients must be nonempty"):
        _run(profile, store, ledger, [], [], _ref([]))
    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(run_extended_session_backfill(
            as_of=AS_OF, stop_at=datetime(2099, 1, 1), profile=profile, clients=[object()],
            store=store, ledger=ledger, tasks=[], price_reference=_ref([]),
        ))


def test_default_ledger_path_lives_in_history_tree(tmp_path, monkeypatch) -> None:
    from src.backfill.intraday import extended_session_backfill as mod

    monkeypatch.setattr(mod.settings, "HISTORY_DIR", tmp_path, raising=False)
    assert ExtendedBackfillLedger().path == tmp_path / "intraday" / "backfill_ledger" / "extended_sessions.parquet"


def test_candidates_alone_emit_tasks_without_price_history() -> None:
    tasks = _by_key(_enum(_ph([]), _pairs([("2026-03-02", "7")])))
    # 달력이 없으면 다음 거래일을 추측하지 않으므로 프리마켓 과제는 없다
    assert tasks == {("2026-03-02", "nxt_aftermarket"): ("000007",)}


def test_ledger_scopes_by_date_and_ignores_empty_records(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    ledger.record("2026-03-02", "nxt_aftermarket", [], run_id="r", attempted_at=datetime.now(SEOUL))
    assert not ledger.path.exists()
    ledger.record("2026-03-02", "nxt_aftermarket", [_entry("000001", CaptureStatus.COMPLETE)],
                  run_id="r", attempted_at=datetime.now(SEOUL))
    assert ledger.terminal_symbols("2026-03-03", "nxt_aftermarket") == frozenset()
    assert ledger.terminal_symbols("2026-03-02", "nxt_aftermarket") == frozenset({"000001"})


def test_unreadable_stored_partition_fails_loud(env) -> None:
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    day = "2026-03-02"
    target = intraday_partition_path(1, day, "nxt_aftermarket")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not parquet")
    with pytest.raises(OSError, match="existing partition evidence"):
        _run(profile, store, ledger, [_FakeKis(day, traded=set())],
             [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))], _ref([]))


class _SessionClient(_FakeKis):
    def __init__(self, day: str, produced) -> None:
        super().__init__(day, traded={"000001"})
        self._produced = produced
        self.seen_sessions: list = []

    def create_session(self):
        return self._produced

    async def get_historical_minute_chart(self, session, code, target_date, **kwargs):
        self.seen_sessions.append(session)
        return await super().get_historical_minute_chart(session, code, target_date, **kwargs)


class _AsyncCM:
    async def __aenter__(self):
        return "entered-session"

    async def __aexit__(self, *exc):
        return False


def test_client_sessions_are_opened_from_async_and_plain_factories(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    async_client = _SessionClient(day, _AsyncCM())
    plain_client = _SessionClient(day, "plain-session")
    _run(profile, store, ledger, [async_client, plain_client],
         [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001", "000002"))],
         _raw_ref(day, {"000001"}))
    assert set(async_client.seen_sessions) == {"entered-session"}
    assert async_client.seen_sessions.count("entered-session") == 1
    assert plain_client.seen_sessions == ["plain-session"]


def test_empty_stored_partition_does_not_block_fetch(env) -> None:
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    day = "2026-03-02"
    target = intraday_partition_path(1, day, "nxt_aftermarket")
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": pd.Series([], dtype="string")}).to_parquet(target, index=False)
    kis = _FakeKis(day, traded=set())
    _run(profile, store, ledger, [kis], [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))], _ref([]))
    assert kis.requested == ["000001"]


def test_universe_helper_preserves_kis_enumeration() -> None:
    from src.backfill.intraday.extended_session_backfill import _entry_day_universes

    ph = _ph([("2026-03-02", "000001", 103.0, 100.0), ("2026-03-03", "000002", 100.0, 100.0)])
    calendar, universes = _entry_day_universes(ph, _pairs([("2026-03-03", "7")]), 0.02)
    assert calendar == ["2026-03-02", "2026-03-03"]
    assert universes == {"2026-03-02": {"000001"}, "2026-03-03": {"000007"}}
    tasks = _by_key(_enum(ph, _pairs([("2026-03-03", "7")])))
    assert tasks[("2026-03-02", "nxt_aftermarket")] == ("000001",)
    assert tasks[("2026-03-03", "nxt_aftermarket")] == ("000007",)


def test_legacy_ledger_without_vendor_still_works(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    pd.DataFrame([{
        "snapshot_date": "2026-03-02", "session": "nxt_aftermarket", "symbol": "000001", "status": "COMPLETE",
        "rows": 1, "reason": "seed", "run_id": "old", "attempted_at": "2026-03-02T23:00:00+09:00",
    }]).to_parquet(path, index=False)
    ledger = ExtendedBackfillLedger(path)
    ledger.record("2026-03-02", "nxt_aftermarket", [_entry("000002", CaptureStatus.COMPLETE)],
                  run_id="new", attempted_at=datetime.now(SEOUL))
    rows = pd.read_parquet(path)
    assert set(rows["symbol"]) == {"000001", "000002"}
    assert rows.set_index("symbol")["vendor"].to_dict() == {"000001": "kis", "000002": "kis"}
    assert ledger.terminal_symbols("2026-03-02", "nxt_aftermarket") == frozenset({"000001", "000002"})


def test_ledger_vendor_is_recorded_but_not_part_of_the_key(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    ledger.record("2026-03-02", "nxt_aftermarket", [_entry("000001", CaptureStatus.FAILED)],
                  run_id="a", attempted_at=datetime.now(SEOUL))
    assert ledger.terminal_symbols("2026-03-02", "nxt_aftermarket") == frozenset()
    ledger.record("2026-03-02", "nxt_aftermarket", [_entry("000001", CaptureStatus.COMPLETE)],
                  run_id="b", attempted_at=datetime.now(SEOUL), vendor="kis")
    rows = pd.read_parquet(ledger.path)
    assert len(rows) == 1 and rows.iloc[0]["vendor"] == "kis"
    assert ledger.terminal_symbols("2026-03-02", "nxt_aftermarket") == frozenset({"000001"})


def _adj_ref(day: str, symbol: str, adjusted: float = 358565.0, raw: float = 466000.0) -> object:
    return _ref([(day, symbol, adjusted, raw)])


def _ledger_row(ledger: ExtendedBackfillLedger, symbol: str) -> pd.Series:
    rows = pd.read_parquet(ledger.path)
    return rows[rows["symbol"] == symbol].iloc[0]


def test_raw_panel_day_keeps_kis_bars_without_raw_source_call(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    kiwoom = _FakeKiwoom(day, raw={"000001": "999"})
    summary = _run(profile, store, ledger, [_FakeKis(day, traded={"000001"})],
                   [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))],
                   _raw_ref(day, {"000001"}), raw_client=kiwoom)
    assert summary.complete == 1
    assert kiwoom.calls == []
    assert _ledger_row(ledger, "000001")["price_basis"] == "kis_raw"


def test_adjusted_day_is_stored_from_kiwoom_raw_bars(env) -> None:
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    day = "2025-09-29"
    kis = _FakeKis(day, traded={"196170"}, bars={"196170": ("358565", "1000", "358565000")})
    kiwoom = _FakeKiwoom(day, raw={"196170": "466000"})
    summary = _run(profile, store, ledger, [kis], [ExtendedBackfillTask(day, "nxt_aftermarket", ("196170",))],
                   _adj_ref(day, "196170"), raw_client=kiwoom)
    assert summary.complete == 1
    stored = pd.read_parquet(intraday_partition_path(1, day, "nxt_aftermarket"))
    assert stored["close"].tolist() == [466000]
    assert stored["vendor"].tolist() == ["kiwoom"]
    assert kiwoom.calls == [("get_nxt_minute_chart", "196170")]
    row = _ledger_row(ledger, "196170")
    assert row["status"] == "COMPLETE" and row["price_basis"] == "kiwoom_raw"


def test_adjusted_premarket_day_uses_the_premarket_raw_chart(env) -> None:
    profile, store, ledger = env
    day = "2025-09-30"
    kis = _FakeKis(day, traded={"196170"}, hms="080500")
    kiwoom = _FakeKiwoom(day, raw={"196170": "466000"})
    summary = _run(profile, store, ledger, [kis], [ExtendedBackfillTask(day, "nxt_premarket", ("196170",))],
                   _adj_ref(day, "196170"), raw_client=kiwoom)
    assert summary.complete == 1
    assert kiwoom.calls == [("get_nxt_premarket_chart", "196170")]


def test_adjusted_day_without_kiwoom_bars_fails_closed(env) -> None:
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    day = "2025-09-29"
    summary = _run(profile, store, ledger, [_FakeKis(day, traded={"067310"})],
                   [ExtendedBackfillTask(day, "nxt_aftermarket", ("067310",))],
                   _adj_ref(day, "067310"), raw_client=_FakeKiwoom(day, raw={}))
    assert summary.failed == 1 and summary.complete == 0
    assert not intraday_partition_path(1, day, "nxt_aftermarket").exists()
    row = _ledger_row(ledger, "067310")
    assert row["reason"] == "price_basis_raw_basis_unavailable"
    assert "067310" not in ledger.terminal_symbols(day, "nxt_aftermarket")


def test_kiwoom_transport_error_fails_closed_with_reason(env) -> None:
    profile, store, ledger = env
    day = "2025-09-29"
    _run(profile, store, ledger, [_FakeKis(day, traded={"196170"})],
         [ExtendedBackfillTask(day, "nxt_aftermarket", ("196170",))],
         _adj_ref(day, "196170"), raw_client=_FakeKiwoom(day, raw={}, error={"196170"}))
    assert _ledger_row(ledger, "196170")["reason"].startswith("price_basis_raw_basis_transport:")


def test_adjusted_day_without_raw_source_fails_closed(env) -> None:
    profile, store, ledger = env
    krx_day = "2026-09-20"
    summary = _run(profile, store, ledger, [_FakeKis(krx_day, traded={"000001"})],
                   [ExtendedBackfillTask(krx_day, "krx_aftermarket", ("000001",))],
                   _adj_ref(krx_day, "000001", 5300.0, 1060.0), raw_client=_FakeKiwoom(krx_day, raw={"000001": "1060"}))
    assert summary.failed == 1
    assert _ledger_row(ledger, "000001")["reason"] == "price_basis_no_raw_source"
    nxt_day = "2026-03-02"
    _run(profile, store, ledger, [_FakeKis(nxt_day, traded={"000002"})],
         [ExtendedBackfillTask(nxt_day, "nxt_aftermarket", ("000002",))],
         _adj_ref(nxt_day, "000002", 5300.0, 1060.0), raw_client=None)
    assert _ledger_row(ledger, "000002")["reason"] == "price_basis_no_raw_source"


def test_unknown_panel_day_fails_closed(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    kiwoom = _FakeKiwoom(day, raw={"000001": "1000"})
    summary = _run(profile, store, ledger, [_FakeKis(day, traded={"000001"})],
                   [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))], _ref([]), raw_client=kiwoom)
    assert summary.failed == 1 and kiwoom.calls == []
    assert _ledger_row(ledger, "000001")["reason"] == "price_basis_unknown"


def test_not_listed_and_no_trade_symbols_make_no_raw_source_call(env) -> None:
    profile, store, ledger = env
    krx_day = "2026-09-20"
    kiwoom = _FakeKiwoom(krx_day, raw={"000009": "1"})
    summary = _run(profile, store, ledger, [_FakeKis(krx_day, traded=set())],
                   [ExtendedBackfillTask(krx_day, "krx_aftermarket", ("000009",))], _ref([]), raw_client=kiwoom)
    assert summary.no_trades == 1
    nxt_day = "2026-03-02"
    nxt_summary = _run(profile, store, ledger, [_FakeKis(nxt_day, traded=set())],
                       [ExtendedBackfillTask(nxt_day, "nxt_aftermarket", ("000007",))], _ref([]), raw_client=kiwoom)
    assert nxt_summary.not_listed == 1
    assert kiwoom.calls == []


def _seed_legacy_adjusted(ledger: ExtendedBackfillLedger, day: str, symbol: str) -> None:
    from src.data.intraday_schema import normalize_bar_frame
    from src.data.intraday_store import write_intraday_partition

    adjusted = normalize_bar_frame(
        pd.DataFrame([_row("160000", "358565", "1000", "358565000", day)]), "kis", day, symbol,
    )
    write_intraday_partition(adjusted, 1, day, "nxt_aftermarket")
    ledger.record(day, "nxt_aftermarket", [_entry(symbol, CaptureStatus.COMPLETE)],
                  run_id="old", attempted_at=datetime.now(SEOUL))


def test_legacy_adjusted_rows_are_replaced_by_raw_bars(env) -> None:
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    day = "2025-09-29"
    _seed_legacy_adjusted(ledger, day, "196170")
    assert ledger.unverified_complete_symbols(day, "nxt_aftermarket") == frozenset({"196170"})
    kis = _FakeKis(day, traded={"196170"}, bars={"196170": ("358565", "1000", "358565000")})
    _run(profile, store, ledger, [kis], [ExtendedBackfillTask(day, "nxt_aftermarket", ("196170",))],
         _adj_ref(day, "196170"), raw_client=_FakeKiwoom(day, raw={"196170": "466000"}))
    assert kis.requested == ["196170"]
    stored = pd.read_parquet(intraday_partition_path(1, day, "nxt_aftermarket"))
    assert stored["close"].tolist() == [466000]
    assert ledger.unverified_complete_symbols(day, "nxt_aftermarket") == frozenset()


def test_failed_repair_removes_adjusted_rows_and_stays_pending(env) -> None:
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    day = "2025-09-29"
    _seed_legacy_adjusted(ledger, day, "067310")
    task = [ExtendedBackfillTask(day, "nxt_aftermarket", ("067310",))]
    _run(profile, store, ledger, [_FakeKis(day, traded={"067310"})], task,
         _adj_ref(day, "067310"), raw_client=_FakeKiwoom(day, raw={}))
    assert not intraday_partition_path(1, day, "nxt_aftermarket").exists()
    assert "067310" not in ledger.terminal_symbols(day, "nxt_aftermarket")
    retry = _FakeKis(day, traded={"067310"})
    _run(profile, store, ledger, [retry], task, _adj_ref(day, "067310"), raw_client=_FakeKiwoom(day, raw={}))
    assert retry.requested == ["067310"]


def test_legacy_raw_day_rows_are_not_refetched(env) -> None:
    from src.data.intraday_schema import normalize_bar_frame
    from src.data.intraday_store import write_intraday_partition

    profile, store, ledger = env
    day = "2026-03-02"
    stored = normalize_bar_frame(pd.DataFrame([_row("160000", "1000", "10", "10000", day)]), "kis", day, "000001")
    write_intraday_partition(stored, 1, day, "nxt_aftermarket")
    ledger.record(day, "nxt_aftermarket", [_entry("000001", CaptureStatus.COMPLETE)],
                  run_id="old", attempted_at=datetime.now(SEOUL))
    kis = _FakeKis(day, traded={"000001"})
    _run(profile, store, ledger, [kis], [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))],
         _raw_ref(day, {"000001"}))
    assert kis.requested == []


def test_legacy_ledger_without_basis_column_loads(tmp_path) -> None:
    assert ExtendedBackfillLedger(tmp_path / "empty.parquet").unverified_complete_symbols(
        "2026-03-02", "nxt_aftermarket") == frozenset()
    path = tmp_path / "ledger.parquet"
    pd.DataFrame([{
        "snapshot_date": "2026-03-02", "session": "nxt_aftermarket", "symbol": "000001", "status": "COMPLETE",
        "rows": 1, "reason": "seed", "run_id": "old", "attempted_at": "2026-03-02T23:00:00+09:00",
    }]).to_parquet(path, index=False)
    ledger = ExtendedBackfillLedger(path)
    assert ledger.unverified_complete_symbols("2026-03-02", "nxt_aftermarket") == frozenset({"000001"})
    ledger.record("2026-03-02", "nxt_aftermarket", [_entry("000002", CaptureStatus.COMPLETE)],
                  run_id="new", attempted_at=datetime.now(SEOUL), price_bases={"000002": "kiwoom_raw"})
    rows = pd.read_parquet(path).set_index("symbol")
    assert rows.loc["000001", "price_basis"] == "" and rows.loc["000002", "price_basis"] == "kiwoom_raw"
    assert ledger.unverified_complete_symbols("2026-03-02", "nxt_aftermarket") == frozenset({"000001"})
    assert ledger.unverified_complete_symbols("2026-03-03", "nxt_aftermarket") == frozenset()


def test_kiwoom_raw_helper_rejects_sessions_without_raw_source(env) -> None:
    from src.backfill.intraday.collector import backfill_kiwoom_raw_bars

    profile, store, _ledger = env
    with pytest.raises(ValueError, match="No Kiwoom raw source"):
        asyncio.run(backfill_kiwoom_raw_bars(
            _FakeKiwoom("2026-09-20", raw={}), None, "000001", "2026-09-20",
            session_tag="krx_aftermarket", profile=profile, capture_store=store, run_id="r",
        ))


def test_kiwoom_raw_helper_classifies_vendor_errors(env) -> None:
    from src.backfill.intraday.collector import backfill_kiwoom_raw_bars

    profile, store, _ledger = env

    class _ErrKiwoom(_FakeKiwoom):
        async def get_nxt_minute_chart(self, session, code, target_date):
            return {"rt_cd": "1", "msg1": "x", "output2": []}

    frame, entry = asyncio.run(backfill_kiwoom_raw_bars(
        _ErrKiwoom("2025-09-29", raw={}), None, "000001", "2025-09-29",
        session_tag="nxt_aftermarket", profile=profile, capture_store=store, run_id="r",
    ))
    assert frame.empty and entry.status is CaptureStatus.FAILED and entry.reason == "raw_basis_vendor_error"
    assert len(entry.raw_refs) == 1


def test_remove_intraday_symbols_keeps_other_symbols(env) -> None:
    from src.data.intraday_schema import normalize_bar_frame
    from src.data.intraday_store import intraday_partition_path, remove_intraday_symbols, write_intraday_partition

    day = "2026-03-02"
    frames = [normalize_bar_frame(pd.DataFrame([_row("160000", "1000", "10", "10000", day)]), "kis", day, s)
              for s in ("000001", "000002")]
    write_intraday_partition(pd.concat(frames, ignore_index=True), 1, day, "nxt_aftermarket")
    assert remove_intraday_symbols(1, day, "nxt_aftermarket", set()) == 2
    assert remove_intraday_symbols(1, day, "nxt_aftermarket", {"000001"}) == 1
    assert pd.read_parquet(intraday_partition_path(1, day, "nxt_aftermarket"))["symbol"].tolist() == ["000002"]
    assert remove_intraday_symbols(1, "2026-03-03", "nxt_aftermarket", {"000001"}) == 0


def test_kiwoom_raw_helper_refuses_uncertified_venue_and_malformed_rows(env) -> None:
    from src.backfill.intraday.collector import backfill_kiwoom_raw_bars

    profile, store, _ledger = env
    day = "2025-09-29"
    uncertified = CollectionSettings(COLLECTION_ROOT=profile.COLLECTION_ROOT, COLLECTION_VERIFIED_CHART_ROUTES={"kiwoom:ka10080": "UNKNOWN"})
    _frame, entry = asyncio.run(backfill_kiwoom_raw_bars(
        _FakeKiwoom(day, raw={"000001": "1000"}), None, "000001", day,
        session_tag="nxt_aftermarket", profile=uncertified, capture_store=store, run_id="r",
    ))
    assert entry.status is CaptureStatus.FAILED and entry.reason == "raw_basis_uncertified_venue"
    _frame, entry = asyncio.run(backfill_kiwoom_raw_bars(
        _FakeKiwoom(day, raw={"000001": "not-a-price"}), None, "000001", day,
        session_tag="nxt_aftermarket", profile=profile, capture_store=store, run_id="r2",
    ))
    assert entry.status is CaptureStatus.FAILED and entry.reason.startswith("raw_basis_normalize:")


def _record(ledger: ExtendedBackfillLedger, symbol: str, status: CaptureStatus, day: str = "2026-03-02") -> None:
    ledger.record(day, "nxt_aftermarket", [_entry(symbol, status)], run_id="r", attempted_at=datetime.now(SEOUL))


def test_third_consecutive_failure_becomes_terminal(tmp_path) -> None:
    from src.backfill.intraday.extended_session_backfill import EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS

    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    for _ in range(EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS - 1):
        _record(ledger, "000001", CaptureStatus.FAILED)
    assert "000001" not in ledger.terminal_symbols("2026-03-02", "nxt_aftermarket")
    _record(ledger, "000001", CaptureStatus.FAILED)
    row = _ledger_row(ledger, "000001")
    assert row["status"] == "EXHAUSTED"
    assert int(row["attempts"]) == EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS
    assert row["reason"].startswith("exhausted:")
    assert "000001" in ledger.terminal_symbols("2026-03-02", "nxt_aftermarket")


def test_success_resets_the_attempt_count(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    _record(ledger, "000001", CaptureStatus.FAILED)
    _record(ledger, "000001", CaptureStatus.FAILED)
    _record(ledger, "000001", CaptureStatus.COMPLETE)
    row = _ledger_row(ledger, "000001")
    assert (row["status"], int(row["attempts"])) == ("COMPLETE", 0)
    _record(ledger, "000001", CaptureStatus.FAILED)
    row = _ledger_row(ledger, "000001")
    assert (row["status"], int(row["attempts"])) == ("FAILED", 1)


def test_attempts_are_scoped_per_date_and_session(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    _record(ledger, "000001", CaptureStatus.FAILED, day="2026-03-02")
    _record(ledger, "000001", CaptureStatus.FAILED, day="2026-03-02")
    _record(ledger, "000001", CaptureStatus.FAILED, day="2026-03-03")
    rows = pd.read_parquet(ledger.path).set_index("snapshot_date")
    assert int(rows.loc["2026-03-03", "attempts"]) == 1
    assert rows.loc["2026-03-03", "status"] == "FAILED"


def test_legacy_ledger_without_attempts_is_upgraded(tmp_path) -> None:
    path = tmp_path / "ledger.parquet"
    pd.DataFrame([{
        "snapshot_date": "2026-03-02", "session": "nxt_aftermarket", "symbol": "000001", "status": "FAILED",
        "rows": 0, "reason": "seed", "run_id": "old", "attempted_at": "2026-03-02T23:00:00+09:00",
        "vendor": "kis", "price_basis": "",
    }]).to_parquet(path, index=False)
    ledger = ExtendedBackfillLedger(path)
    _record(ledger, "000001", CaptureStatus.FAILED)
    row = _ledger_row(ledger, "000001")
    assert (row["status"], int(row["attempts"])) == ("FAILED", 2)


def test_exhausted_keys_are_not_refetched_and_counted(env) -> None:
    from src.backfill.intraday.extended_session_backfill import EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS

    profile, store, ledger = env
    day = "2025-09-29"
    tasks = [ExtendedBackfillTask(day, "nxt_aftermarket", ("067310",))]
    summaries = [
        _run(profile, store, ledger, [_FakeKis(day, traded={"067310"})], tasks,
             _adj_ref(day, "067310"), raw_client=_FakeKiwoom(day, raw={}))
        for _ in range(EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS)
    ]
    assert [s.exhausted for s in summaries] == [0] * (EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS - 1) + [1]
    assert "067310" in ledger.terminal_symbols(day, "nxt_aftermarket")
    kis = _FakeKis(day, traded={"067310"})
    after = _run(profile, store, ledger, [kis], tasks, _adj_ref(day, "067310"), raw_client=_FakeKiwoom(day, raw={}))
    assert after.failed == 0 and after.exhausted == 0
