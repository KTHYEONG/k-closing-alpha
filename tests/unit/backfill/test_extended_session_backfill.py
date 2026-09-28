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

    def __init__(self, day: str, traded: set[str], fail: set[str] | None = None) -> None:
        self._day = day
        self._traded = traded
        self._fail = fail or set()
        self.requested: list[str] = []

    async def get_historical_minute_chart(self, session, code, target_date, **kwargs):
        self.requested.append(code)
        if code in self._fail:
            raise RuntimeError("down")
        if code in self._traded:
            return {"rt_cd": "0", "output2": [_row("160000", "1000", "10", "10000", self._day)]}
        return {"rt_cd": "0", "output2": []}


@pytest.fixture
def env(tmp_path, monkeypatch):
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    profile = CollectionSettings(COLLECTION_ROOT=tmp_path / "capture")
    store = CaptureStore(tmp_path / "capture")
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    return profile, store, ledger


def _run(profile, store, ledger, clients, tasks, now_fn=None, stop_at=None):
    stop = stop_at or datetime(2099, 1, 1, tzinfo=SEOUL)
    return asyncio.run(run_extended_session_backfill(
        as_of=AS_OF, stop_at=stop, profile=profile, clients=clients, store=store,
        ledger=ledger, tasks=tasks, now_fn=now_fn,
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
                   [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001", "000002", "000003"))])
    assert kis.requested == ["000003"]
    assert summary.complete == 1 and summary.tasks_done == 1


def test_failed_symbols_are_retried_on_next_run(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    ledger.record(day, "nxt_aftermarket", [_entry("000003", CaptureStatus.FAILED)],
                  run_id="seed", attempted_at=datetime.now(SEOUL))
    kis = _FakeKis(day, traded={"000003"})
    _run(profile, store, ledger, [kis], [ExtendedBackfillTask(day, "nxt_aftermarket", ("000003",))])
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
    summary = _run(profile, store, ledger, [kis], tasks, now_fn=lambda: next(ticks), stop_at=stop)
    assert summary.tasks_done == 1
    assert summary.tasks_remaining == 1
    assert summary.stopped_by_deadline is True
    assert kis.requested == ["000001"]


def test_unlisted_nxt_symbols_are_terminal_without_partition_rows(env) -> None:
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    day = "2026-03-02"
    summary = _run(profile, store, ledger, [_FakeKis(day, traded=set())],
                   [ExtendedBackfillTask(day, "nxt_aftermarket", ("000009",))])
    assert summary.not_listed == 1
    assert "000009" in ledger.terminal_symbols(day, "nxt_aftermarket")
    assert not intraday_partition_path(1, day, "nxt_aftermarket").exists()
    manifests = [m for m in store.read_manifests(day) if m.context.capture_reason == "extended-backfill"]
    assert manifests and manifests[0].status is CaptureStatus.COMPLETE


def test_symbols_spread_round_robin_over_clients(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    a, b = _FakeKis(day, traded={"000001", "000002"}), _FakeKis(day, traded={"000001", "000002"})
    _run(profile, store, ledger, [a, b], [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001", "000002"))])
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
             [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))])
    assert "000001" not in ledger.terminal_symbols(day, "nxt_aftermarket")


def test_run_ids_are_unique_per_task_run(env) -> None:
    profile, store, ledger = env
    day = "2026-03-02"
    task = ExtendedBackfillTask(day, "nxt_aftermarket", ("000009",))
    _run(profile, store, ledger, [_FakeKis(day, traded=set())], [task])
    ledger_path = ledger.path
    ledger_path.unlink()
    _run(profile, store, ledger, [_FakeKis(day, traded=set())], [task])
    run_ids = {m.context.run_id for m in store.read_manifests(day) if m.context.capture_reason == "extended-backfill"}
    assert len(run_ids) == 2


def test_runner_rejects_empty_clients_and_naive_stop(env) -> None:
    profile, store, ledger = env
    with pytest.raises(ValueError, match="clients must be nonempty"):
        _run(profile, store, ledger, [], [])
    with pytest.raises(ValueError, match="timezone-aware"):
        asyncio.run(run_extended_session_backfill(
            as_of=AS_OF, stop_at=datetime(2099, 1, 1), profile=profile, clients=[object()],
            store=store, ledger=ledger, tasks=[],
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
             [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))])


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
         [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001", "000002"))])
    assert async_client.seen_sessions == ["entered-session"]
    assert plain_client.seen_sessions == ["plain-session"]


def test_empty_stored_partition_does_not_block_fetch(env) -> None:
    from src.data.intraday_store import intraday_partition_path

    profile, store, ledger = env
    day = "2026-03-02"
    target = intraday_partition_path(1, day, "nxt_aftermarket")
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": pd.Series([], dtype="string")}).to_parquet(target, index=False)
    kis = _FakeKis(day, traded=set())
    _run(profile, store, ledger, [kis], [ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))])
    assert kis.requested == ["000001"]
