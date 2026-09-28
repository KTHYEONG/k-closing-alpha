"""Invariant scenarios for the Toss NXT overnight backfill runner."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from src.backfill.intraday import toss_overnight_backfill as mod
from src.backfill.intraday.extended_session_backfill import ExtendedBackfillLedger, ExtendedBackfillTask
from src.backfill.intraday.toss_overnight_backfill import (
    TossOvernightTask,
    calibrate_toss_against_stored,
    enumerate_toss_overnight_tasks,
    list_stored_nxt_evening_dates,
    run_overnight_backfill_phases,
    run_toss_overnight_backfill,
)
from src.config.collection import CollectionSettings
from src.config.market_session import KRX_AFTERMARKET_START_DATE, NXT_START_DATE
from src.data import intraday_store
from src.data.capture_contracts import SEOUL, CaptureDataset, CaptureStatus, CoverageEntry
from src.data.capture_store import CaptureStore
from tests.unit.backfill.test_extended_session_backfill import _FakeKis
from tests.unit.backfill.toss_fakes import FakeToss, always, day_series, reference_frame

AS_OF = date(2026, 9, 28)
WINDOW_START = "2025-09-28"
REF_DAY = "2025-11-03"
REF_SYM = "000100"
REF_FN = lambda label: 3 if int(label[3:]) % 2 == 0 else 0  # noqa: E731
STOP_FAR = datetime(2099, 1, 1, tzinfo=SEOUL)


def _ph(rows):
    return pd.DataFrame(rows, columns=["date", "symbol", "close", "prev_close"])


def _pairs(rows):
    return pd.DataFrame(rows, columns=["snapshot_date", "symbol"])


def _enum(ph, pairs=None, **kw):
    return enumerate_toss_overnight_tasks(
        as_of=kw.get("as_of", AS_OF), kis_retention_days=kw.get("retention", 365),
        nxt_start_date=NXT_START_DATE, krx_aftermarket_start_date=KRX_AFTERMARKET_START_DATE,
        min_change_ratio=0.02, price_history=ph, candidate_pairs=pairs if pairs is not None else _pairs([]),
    )


# ------------------------------------------------------------------ enumeration


def test_toss_range_stops_at_kis_window_and_krx_start() -> None:
    days = ["2025-03-03", "2025-03-04", "2025-06-12", "2025-09-26", "2025-09-29", "2026-01-05", "2026-09-15"]
    ph = _ph([*((d, "000001", 103.0, 100.0) for d in days), ("2026-09-16", "000001", 100.0, 100.0)])
    entry_days = {t.entry_day for t in _enum(ph)}
    assert entry_days == {"2025-03-04", "2025-06-12", "2025-09-26"}


def test_next_trading_day_comes_from_the_calendar() -> None:
    ph = _ph([
        ("2025-06-13", "000001", 103.0, 100.0),  # 금요일
        ("2025-06-16", "000001", 100.0, 103.0),  # 다음 거래일(월)
        ("2025-09-26", "000002", 103.0, 100.0),  # 다음 행이 as_of 이후 -> 제외
        ("2026-09-30", "000002", 100.0, 100.0),
        ("2025-06-20", "000003", 103.0, 100.0),  # 마지막 행이라 다음 거래일 없음이 아님(아래 행이 있음)
        ("2025-06-23", "000003", 100.0, 100.0),
    ])
    tasks = {t.entry_day: t.next_day for t in _enum(ph)}
    assert tasks["2025-06-13"] == "2025-06-16"
    assert "2025-09-26" not in tasks
    assert tasks["2025-06-20"] == "2025-06-23"


def test_last_calendar_row_has_no_next_day() -> None:
    ph = _ph([("2025-06-13", "000001", 103.0, 100.0)])
    assert _enum(ph) == []


def test_tasks_are_newest_entry_day_first() -> None:
    rows = []
    for d in ["2025-05-02", "2025-06-02", "2025-04-01"]:
        rows += [(d, "000001", 103.0, 100.0), (str(pd.Timestamp(d) + pd.Timedelta(days=1))[:10], "000001", 100.0, 100.0)]
    days = [t.entry_day for t in _enum(_ph(rows))]
    assert days == sorted(days, reverse=True) and len(days) == 3


def test_universe_matches_the_screen_and_has_no_lookahead() -> None:
    base = [
        ("2025-06-12", "000001", 103.0, 100.0), ("2025-06-12", "000002", 101.0, 100.0),
        ("2025-06-13", "000002", 100.0, 100.0),
    ]
    changed = [*base[:2], ("2025-06-13", "000002", 200.0, 100.0)]
    pairs = _pairs([("2025-06-12", "9")])
    first = {t.entry_day: t.symbols for t in _enum(_ph(base), pairs)}["2025-06-12"]
    second = {t.entry_day: t.symbols for t in _enum(_ph(changed), pairs)}["2025-06-12"]
    assert first == second == ("000001", "000009")


def test_days_without_a_universe_are_omitted() -> None:
    ph = _ph([("2025-06-12", "000001", 100.0, 100.0), ("2025-06-13", "000001", 100.0, 100.0)])
    assert _enum(ph) == []


# ------------------------------------------------------------------ calibration


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    profile = CollectionSettings(
        COLLECTION_ROOT=tmp_path / "capture", COLLECTION_TOSS_CALIBRATION_DATES=1,
        COLLECTION_TOSS_CALIBRATION_SYMBOLS_PER_DATE=1, COLLECTION_TOSS_CALIBRATION_MIN_TRADED_MINUTES=5,
        COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE=2, COLLECTION_TOSS_BACKFILL_CONCURRENCY=2,
    )
    return {"tmp": tmp_path, "profile": profile, "store": CaptureStore(tmp_path / "capture"),
            "ledger": ExtendedBackfillLedger(tmp_path / "ledger.parquet")}


def _profile(env, **kw) -> CollectionSettings:
    return CollectionSettings(
        COLLECTION_ROOT=env["tmp"] / "capture", **{
            "COLLECTION_TOSS_CALIBRATION_DATES": 1, "COLLECTION_TOSS_CALIBRATION_SYMBOLS_PER_DATE": 1,
            "COLLECTION_TOSS_CALIBRATION_MIN_TRADED_MINUTES": 5, "COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE": 2,
            "COLLECTION_TOSS_BACKFILL_CONCURRENCY": 2, **kw},
    )


def _toss_with_reference(symbols=(), *, task_days=("2025-06-12", "2025-06-13"), ref_fn=REF_FN) -> FakeToss:
    series = {REF_SYM: day_series(REF_DAY, evening=ref_fn)}
    for sym in symbols:
        series[sym] = day_series(task_days[0], evening=always(10)) + day_series(task_days[1], premarket=always(7))
    return FakeToss(series)


def _cal(env, toss, *, ref_frames=None, dates=None, profile=None, window_start=WINDOW_START):
    frames = ref_frames if ref_frames is not None else {REF_DAY: reference_frame(REF_SYM, REF_DAY, REF_FN)}
    listed = list(frames) if dates is None else dates
    return asyncio.run(calibrate_toss_against_stored(
        toss, None, profile=profile or env["profile"], window_start=window_start,
        krx_aftermarket_start_date=KRX_AFTERMARKET_START_DATE,
        list_reference_dates=lambda: listed, read_reference=lambda d: frames.get(d),
    ))


def test_calibration_passes_on_matching_data(env) -> None:
    verdict = _cal(env, _toss_with_reference())
    assert verdict.status == "PASS" and verdict.exact_ratio == 1.0
    assert verdict.dates == (REF_DAY,) and verdict.symbols == 1 and verdict.traded_minutes >= 5


def test_calibration_fails_on_divergence(env) -> None:
    toss = _toss_with_reference(ref_fn=lambda label: 6 if int(label[3:]) % 2 == 0 else 0)
    verdict = _cal(env, toss)
    assert verdict.status == "FAIL" and verdict.exact_ratio is not None and verdict.exact_ratio < 0.999


def test_calibration_without_reference_is_fail_closed(env) -> None:
    toss = _toss_with_reference()
    verdict = _cal(env, toss, ref_frames={}, dates=[])
    assert verdict.status == "NO_REFERENCE" and verdict.exact_ratio is None
    assert toss.calls == []


def test_toss_written_rows_never_calibrate_toss(env) -> None:
    frames = {REF_DAY: reference_frame(REF_SYM, REF_DAY, REF_FN, vendor="toss")}
    assert _cal(env, _toss_with_reference(), ref_frames=frames).status == "NO_REFERENCE"


def test_calibration_is_deterministic(env) -> None:
    profile = _profile(env, COLLECTION_TOSS_CALIBRATION_DATES=2, COLLECTION_TOSS_CALIBRATION_SYMBOLS_PER_DATE=2)
    frames = {}
    series = {}
    for day in ("2025-10-01", "2025-11-03", "2025-12-01"):
        parts = []
        for sym in ("000100", "000101", "000102"):
            parts.append(reference_frame(sym, day, REF_FN))
            series.setdefault(sym, []).extend(day_series(day, evening=REF_FN))
        frames[day] = pd.concat(parts, ignore_index=True)
    first = _cal(env, FakeToss(series), ref_frames=frames, profile=profile)
    second = _cal(env, FakeToss(series), ref_frames=frames, profile=profile)
    assert first == second and first.status == "PASS" and len(first.dates) == 2


def test_calibration_skips_windows_with_errors(env) -> None:
    profile = _profile(env, COLLECTION_TOSS_CALIBRATION_SYMBOLS_PER_DATE=2)
    frames = {REF_DAY: pd.concat([reference_frame("000100", REF_DAY, REF_FN), reference_frame("000101", REF_DAY, REF_FN)])}
    toss = FakeToss({s: day_series(REF_DAY, evening=REF_FN) for s in ("000100", "000101")})
    toss.script[0] = {"error": {"code": "rate-limit"}}
    verdict = _cal(env, toss, ref_frames=frames, profile=profile)
    assert verdict.status == "PASS" and verdict.symbols == 1


def test_reference_outside_the_window_is_ignored(env) -> None:
    toss = _toss_with_reference()
    assert _cal(env, toss, dates=["2025-09-01"]).status == "NO_REFERENCE"
    assert _cal(env, toss, dates=[KRX_AFTERMARKET_START_DATE]).status == "NO_REFERENCE"


def test_unreadable_reference_is_tolerated(env) -> None:
    toss = _toss_with_reference()

    def boom() -> list[str]:
        raise OSError("gone")

    assert asyncio.run(calibrate_toss_against_stored(
        toss, None, profile=env["profile"], window_start=WINDOW_START,
        krx_aftermarket_start_date=KRX_AFTERMARKET_START_DATE, list_reference_dates=boom,
    )).status == "NO_REFERENCE"

    def bad_reader(day: str):
        raise ValueError("corrupt")

    assert asyncio.run(calibrate_toss_against_stored(
        toss, None, profile=env["profile"], window_start=WINDOW_START,
        krx_aftermarket_start_date=KRX_AFTERMARKET_START_DATE,
        list_reference_dates=lambda: [REF_DAY], read_reference=bad_reader,
    )).status == "NO_REFERENCE"


def test_reference_without_has_trade_column_or_with_empty_frames(env) -> None:
    frame = reference_frame(REF_SYM, REF_DAY, REF_FN).drop(columns=["has_trade"])
    assert _cal(env, _toss_with_reference(), ref_frames={REF_DAY: frame}).status == "PASS"
    empty = reference_frame(REF_SYM, REF_DAY, lambda label: 0)
    assert _cal(env, _toss_with_reference(), ref_frames={REF_DAY: empty}).status == "NO_REFERENCE"
    only_toss = reference_frame(REF_SYM, REF_DAY, REF_FN, vendor="toss")
    assert _cal(env, _toss_with_reference(), ref_frames={REF_DAY: only_toss.iloc[0:0]}).status == "NO_REFERENCE"


def test_too_few_compared_minutes_is_no_reference(env) -> None:
    profile = _profile(env, COLLECTION_TOSS_CALIBRATION_MIN_TRADED_MINUTES=100000)
    verdict = _cal(env, _toss_with_reference(), profile=profile)
    assert verdict.status == "NO_REFERENCE" and verdict.exact_ratio is None and verdict.traded_minutes > 0


def test_evenly_spaced_indices() -> None:
    assert mod._evenly_spaced_indices(3, 1) == [0]
    assert mod._evenly_spaced_indices(0, 1) == []
    assert mod._evenly_spaced_indices(2, 6) == [0, 1]
    assert mod._evenly_spaced_indices(10, 3) == [0, 4, 9]


def test_stored_date_lister_and_default_reader(env) -> None:
    assert list_stored_nxt_evening_dates() == []
    frame = reference_frame(REF_SYM, REF_DAY, REF_FN)
    for day in ("2025-11-03", "2025-10-01", "2025-12-15"):
        target = intraday_store.intraday_partition_path(1, day, "nxt_aftermarket")
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.assign(snapshot_date=day).to_parquet(target, index=False)
    assert list_stored_nxt_evening_dates() == ["2025-10-01", "2025-11-03", "2025-12-15"]
    toss = FakeToss({REF_SYM: [c for day in ("2025-10-01", "2025-11-03", "2025-12-15") for c in day_series(day, evening=REF_FN)]})
    verdict = asyncio.run(calibrate_toss_against_stored(
        toss, None, profile=env["profile"], window_start=WINDOW_START,
        krx_aftermarket_start_date=KRX_AFTERMARKET_START_DATE,
    ))
    assert verdict.status == "PASS"
    corrupt = intraday_store.intraday_partition_path(1, "2025-11-03", "nxt_aftermarket")
    corrupt.write_bytes(b"not parquet")
    assert mod._default_read_reference("2025-11-03") is None


# ------------------------------------------------------------------ runner


T, T1 = "2025-06-12", "2025-06-13"


def _run(env, toss, tasks, *, profile=None, now_fn=None, stop_at=STOP_FAR, frames=None):
    ref = frames if frames is not None else {REF_DAY: reference_frame(REF_SYM, REF_DAY, REF_FN)}
    return asyncio.run(run_toss_overnight_backfill(
        as_of=AS_OF, stop_at=stop_at, profile=profile or env["profile"], toss=toss, http_session=None,
        store=env["store"], ledger=env["ledger"], tasks=tasks, window_start=WINDOW_START, now_fn=now_fn,
        list_reference_dates=lambda: list(ref), read_reference=lambda d: ref.get(d),
    ))


def _task(symbols, entry=T, nxt=T1):
    return TossOvernightTask(entry_day=entry, next_day=nxt, symbols=tuple(symbols))


def _ledger_rows(env) -> pd.DataFrame:
    return env["ledger"]._read_all()


def _manifests(env, day):
    return [m for m in env["store"].read_manifests(day) if m.context.capture_reason == "extended-backfill"]


def test_pair_result_is_published_per_session(env) -> None:
    toss = _toss_with_reference(("000001", "000002"))
    summary = _run(env, toss, [_task(["000001", "000002"])])
    assert summary.stopped_reason == "done" and summary.complete == 4 and summary.tasks_done == 1
    evening = pd.read_parquet(intraday_store.intraday_partition_path(1, T, "nxt_aftermarket"))
    premarket = pd.read_parquet(intraday_store.intraday_partition_path(1, T1, "nxt_premarket"))
    assert set(evening["symbol"]) == {"000001", "000002"} and set(premarket["symbol"]) == {"000001", "000002"}
    assert set(evening["vendor"]) == {"toss"}
    rows = _ledger_rows(env)
    assert set(rows["vendor"]) == {"toss"} and set(rows["session"]) == {"nxt_aftermarket", "nxt_premarket"}
    assert len(rows) == 4


def test_unlisted_symbols_are_terminal_without_rows(env) -> None:
    toss = _toss_with_reference()
    toss.series["000001"] = day_series(T) + day_series(T1)
    summary = _run(env, toss, [_task(["000001"])])
    assert summary.not_listed == 2 and summary.complete == 0
    assert not intraday_store.intraday_partition_path(1, T, "nxt_aftermarket").exists()
    assert set(_ledger_rows(env)["status"]) == {"NOT_APPLICABLE"}
    assert all(m.status is CaptureStatus.COMPLETE for m in _manifests(env, T) + _manifests(env, T1))


def test_no_trades_symbols_are_counted_and_terminal(env) -> None:
    toss = _toss_with_reference()
    toss.series["000001"] = day_series(T, evening=always(0)) + day_series(T1, premarket=always(0))
    summary = _run(env, toss, [_task(["000001"])])
    assert summary.no_trades == 2
    assert env["ledger"].terminal_symbols(T, "nxt_aftermarket") == frozenset({"000001"})


def test_terminal_and_stored_sessions_are_not_rewritten(env) -> None:
    from src.data.intraday_schema import normalize_bar_frame

    toss = _toss_with_reference(("000001",))
    raw = pd.DataFrame([{"cntr_tm": f"{T.replace('-', '')}160000", "cur_prc": "+500", "open_pric": "+500",
                         "high_pric": "+500", "low_pric": "+500", "trde_qty": "9"}])
    stored = normalize_bar_frame(raw, "kiwoom", T, "000001")
    intraday_store.write_intraday_partition(stored, 1, T, "nxt_aftermarket")
    before_calls = len(toss.calls)
    summary = _run(env, toss, [_task(["000001"])])
    evening = pd.read_parquet(intraday_store.intraday_partition_path(1, T, "nxt_aftermarket"))
    assert evening["vendor"].tolist() == ["kiwoom"] and len(evening) == 1
    assert summary.complete == 1
    pair_calls = [c for c in toss.calls[before_calls:] if c["symbol"] == "000001"]
    assert len(pair_calls) == 2
    assert set(_ledger_rows(env)["session"]) == {"nxt_premarket"}


def test_failed_symbols_retry_next_night_not_in_the_same_run(env) -> None:
    toss = _toss_with_reference(("000001",))
    toss.fail_symbols = {"000001"}
    first = _run(env, toss, [_task(["000001"])])
    assert first.failed == 2 and env["ledger"].terminal_symbols(T, "nxt_aftermarket") == frozenset()
    assert len([c for c in toss.calls if c["symbol"] == "000001"]) == 1
    toss.fail_symbols = set()
    second = _run(env, toss, [_task(["000001"])])
    assert second.complete == 2 and env["ledger"].terminal_symbols(T, "nxt_aftermarket") == frozenset({"000001"})


def test_date_cap_counts_only_tasks_with_pending_work(env) -> None:
    profile = _profile(env, COLLECTION_TOSS_BACKFILL_MAX_DATES_PER_RUN=2)
    days = [("2025-06-02", "2025-06-03"), ("2025-06-03", "2025-06-04"), ("2025-06-04", "2025-06-05"),
            ("2025-06-05", "2025-06-06"), ("2025-06-06", "2025-06-09"), ("2025-06-09", "2025-06-10")]
    toss = _toss_with_reference()
    tasks = []
    for index, (entry, nxt) in enumerate(days):
        sym = f"{index + 1:06d}"
        toss.series[sym] = day_series(entry, evening=always(4)) + day_series(nxt, premarket=always(4))
        tasks.append(_task([sym], entry, nxt))
    for task in tasks[:3]:  # 앞의 3개는 이미 종료 상태
        for session, day in (("nxt_aftermarket", task.entry_day), ("nxt_premarket", task.next_day)):
            done = CoverageEntry(symbol=task.symbols[0], dataset=CaptureDataset.MINUTE_BARS, venue="NXT", session=session,
                                 scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1, first_event_time=None,
                                 last_event_time=None, reason="seed", raw_refs=())
            env["ledger"].record(day, session, [done], run_id="seed", attempted_at=datetime.now(SEOUL))
    summary = _run(env, toss, tasks, profile=profile)
    assert summary.tasks_done == 2 and summary.stopped_reason == "date_cap" and summary.tasks_remaining == 1


def test_deadline_stops_before_the_next_chunk_and_still_publishes_evidence(env) -> None:
    toss = _toss_with_reference(("000001", "000002", "000003"))
    stop_at = datetime(2030, 1, 1, tzinfo=SEOUL)
    marker = {"late": False}
    toss.on_call = lambda fake: marker.update(late=len([c for c in fake.calls if c["symbol"] != REF_SYM]) >= 4)
    summary = _run(env, toss, [_task(["000001", "000002", "000003"])], stop_at=stop_at,
                   now_fn=lambda: stop_at + timedelta(seconds=1) if marker["late"] else stop_at - timedelta(hours=1))
    assert summary.stopped_reason == "deadline" and summary.tasks_done == 1
    assert env["ledger"].terminal_symbols(T, "nxt_aftermarket") == frozenset({"000001", "000002"})
    manifests = _manifests(env, T)
    assert len(manifests) == 1 and len(manifests[0].entries) == 2


def test_circuit_breaker_stops_after_consecutive_errors_and_resets_on_success(env) -> None:
    profile = _profile(env, COLLECTION_TOSS_BACKFILL_MAX_CONSECUTIVE_ERRORS=2, COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE=1)
    toss = _toss_with_reference(("000001", "000002", "000003", "000004"))
    toss.fail_symbols = {"000001", "000003"}
    ok = _run(env, toss, [_task(["000001", "000002", "000003", "000004"])], profile=profile)
    assert ok.stopped_reason == "done" and ok.failed == 4  # 성공이 사이에 끼면 연속 카운트가 리셋된다
    toss2 = _toss_with_reference(("000001", "000003", "000005"))
    toss2.fail_symbols = {"000001", "000003", "000005"}
    tripped = _run(env, toss2, [_task(["000001", "000003", "000005"])], profile=profile)
    assert tripped.stopped_reason == "circuit_open"
    assert len([c for c in toss2.calls if c["symbol"] == "000003"]) == 1
    assert len([c for c in toss2.calls if c["symbol"] == "000005"]) == 0


def test_persistence_failure_is_loud_and_non_terminal(env, monkeypatch) -> None:
    toss = _toss_with_reference(("000001",))

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(mod, "write_intraday_partition", boom)
    with pytest.raises(OSError, match="disk full"):
        _run(env, toss, [_task(["000001"])])
    assert env["ledger"].terminal_symbols(T, "nxt_aftermarket") == frozenset()


def test_one_manifest_per_task_session(env) -> None:
    symbols = [f"{n:06d}" for n in range(1, 6)]
    toss = _toss_with_reference(symbols)
    _run(env, toss, [_task(symbols)])
    manifests = _manifests(env, T) + _manifests(env, T1)
    assert len(manifests) == 2
    run_ids = {m.context.run_id for m in manifests}
    assert len(run_ids) == 2 and all(r.startswith("toss-backfill-") for r in run_ids)
    assert {m.context.vendor for m in manifests} == {"toss"}


def test_calibration_failure_and_no_reference_write_nothing(env) -> None:
    toss = _toss_with_reference(("000001",), ref_fn=lambda label: 6 if int(label[3:]) % 2 == 0 else 0)
    failed = _run(env, toss, [_task(["000001"])])
    assert failed.stopped_reason == "calibration_failed" and failed.tasks_remaining == 1
    no_ref = _run(env, toss, [_task(["000001"])], frames={})
    assert no_ref.stopped_reason == "calibration_no_reference"
    assert not env["ledger"].path.exists()
    assert _manifests(env, T) == []
    assert not any(c["symbol"] == "000001" for c in toss.calls)


def test_consolidated_regime_entry_days_and_pending_free_tasks_are_skipped(env) -> None:
    toss = _toss_with_reference(("000001",))
    task = _task(["000001"], KRX_AFTERMARKET_START_DATE, "2026-09-15")
    summary = _run(env, toss, [task])
    assert summary.tasks_done == 0 and summary.stopped_reason == "done"
    assert not any(c["symbol"] == "000001" for c in toss.calls)
    assert mod._task_has_pending(task, env["ledger"]) is False


def test_naive_stop_at_is_refused(env) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _run(env, _toss_with_reference(), [], stop_at=datetime(2099, 1, 1))


# ------------------------------------------------------------------ phases


def _kis_task_day():
    return "2026-03-02"


def _phase_env(env):
    frame = reference_frame(REF_SYM, REF_DAY, REF_FN)
    target = intraday_store.intraday_partition_path(1, REF_DAY, "nxt_aftermarket")
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)


def _phases(env, *, kis_clients, toss, http, kis_tasks=(), toss_tasks=(), now_fn=None, stop_at=STOP_FAR):
    return asyncio.run(run_overnight_backfill_phases(
        as_of=AS_OF, stop_at=stop_at, profile=env["profile"], kis_clients=kis_clients, toss=toss,
        http_session=http, store=env["store"], ledger=env["ledger"], kis_tasks=list(kis_tasks),
        toss_tasks=list(toss_tasks), window_start=WINDOW_START, now_fn=now_fn,
    ))


class _OrderedKis(_FakeKis):
    def __init__(self, day, events):
        super().__init__(day, traded={"000001"})
        self.events = events

    async def get_historical_minute_chart(self, session, code, target_date, **kwargs):
        self.events.append("kis")
        return await super().get_historical_minute_chart(session, code, target_date, **kwargs)


def test_kis_runs_first_and_toss_only_after_it_drains(env) -> None:
    _phase_env(env)
    events: list[str] = []
    toss = _toss_with_reference(("000001",))
    toss.on_call = lambda fake: events.append("toss")
    day = _kis_task_day()
    kis, toss_summary = _phases(
        env, kis_clients=[_OrderedKis(day, events)], toss=toss, http=object(),
        kis_tasks=[ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))], toss_tasks=[_task(["000001"])],
    )
    assert kis is not None and toss_summary is not None and toss_summary.complete == 2
    assert events.index("toss") > max(i for i, e in enumerate(events) if e == "kis")


def test_kis_deadline_stop_suppresses_toss(env) -> None:
    _phase_env(env)
    day = _kis_task_day()
    stop_at = datetime(2030, 1, 1, tzinfo=SEOUL)
    toss = _toss_with_reference(("000001",))
    kis, toss_summary = _phases(
        env, kis_clients=[_FakeKis(day, {"000001"})], toss=toss, http=object(),
        kis_tasks=[ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))], toss_tasks=[_task(["000001"])],
        now_fn=lambda: stop_at + timedelta(seconds=1), stop_at=stop_at,
    )
    assert kis is not None and kis.stopped_by_deadline and toss_summary is None and toss.calls == []


def test_closed_window_suppresses_toss(env) -> None:
    _phase_env(env)
    stop_at = datetime(2030, 1, 1, tzinfo=SEOUL)
    toss = _toss_with_reference(("000001",))
    kis, toss_summary = _phases(
        env, kis_clients=[_FakeKis(_kis_task_day(), set())], toss=toss, http=object(), toss_tasks=[_task(["000001"])],
        now_fn=lambda: stop_at + timedelta(seconds=1), stop_at=stop_at,
    )
    assert kis is not None and kis.tasks_remaining == 0 and toss_summary is None


def test_toss_only_night(env) -> None:
    _phase_env(env)
    toss = _toss_with_reference(("000001",))
    kis, toss_summary = _phases(env, kis_clients=[], toss=toss, http=object(), toss_tasks=[_task(["000001"])])
    assert kis is None and toss_summary is not None and toss_summary.complete == 2


def test_toss_disabled_runs_only_the_kis_phase(env) -> None:
    day = _kis_task_day()
    kis, toss_summary = _phases(
        env, kis_clients=[_FakeKis(day, {"000001"})], toss=None, http=None,
        kis_tasks=[ExtendedBackfillTask(day, "nxt_aftermarket", ("000001",))],
    )
    assert kis is not None and kis.complete == 1 and toss_summary is None
    kis2, toss2 = _phases(env, kis_clients=[], toss=object(), http=None)
    assert kis2 is None and toss2 is None


def test_default_reader_returns_none_for_missing_partition(env) -> None:
    assert mod._default_read_reference("2030-01-01") is None


def test_lister_dates_without_a_stored_frame_are_skipped(env) -> None:
    frames = {REF_DAY: reference_frame(REF_SYM, REF_DAY, REF_FN)}
    verdict = _cal(env, _toss_with_reference(), ref_frames=frames, dates=["2025-10-01", REF_DAY])
    assert verdict.status == "PASS" and verdict.dates == (REF_DAY,)


def test_date_cap_remaining_counts_premarket_only_work(env) -> None:
    profile = _profile(env, COLLECTION_TOSS_BACKFILL_MAX_DATES_PER_RUN=1)
    toss = _toss_with_reference()
    days = [("2025-06-02", "2025-06-03"), ("2025-06-03", "2025-06-04"), ("2025-06-04", "2025-06-05")]
    tasks = []
    for index, (entry, nxt) in enumerate(days):
        sym = f"{index + 1:06d}"
        toss.series[sym] = day_series(entry, evening=always(4)) + day_series(nxt, premarket=always(4))
        tasks.append(_task([sym], entry, nxt))

    def _seed(task, sessions):
        for session, day in sessions:
            done = CoverageEntry(symbol=task.symbols[0], dataset=CaptureDataset.MINUTE_BARS, venue="NXT", session=session,
                                 scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1, first_event_time=None,
                                 last_event_time=None, reason="seed", raw_refs=())
            env["ledger"].record(day, session, [done], run_id="seed", attempted_at=datetime.now(SEOUL))

    _seed(tasks[1], [("nxt_aftermarket", tasks[1].entry_day)])  # 저녁만 종료, 프리마켓 대기
    _seed(tasks[2], [("nxt_aftermarket", tasks[2].entry_day), ("nxt_premarket", tasks[2].next_day)])  # 전부 종료
    summary = _run(env, toss, tasks, profile=profile)
    assert summary.tasks_done == 1 and summary.stopped_reason == "date_cap" and summary.tasks_remaining == 1
