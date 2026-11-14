"""Invariant guards for the post-close trade-strength and investor-estimate snapshot."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src import settings
from src.config.collection import CollectionSettings
from src.daily import postclose_snapshot
from src.daily.postclose_snapshot import (
    INVESTOR_DATASET,
    STRENGTH_DATASET,
    investor_rows,
    run_postclose_snapshot,
    summarize_strength,
)
from src.data.capture_contracts import (
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    SessionClock,
    build_cohort,
)
from src.data.capture_store import CaptureStore
from src.data.research_roster import (
    ResearchRoster,
    RosterCandidate,
    build_roster_frame,
    publish_research_roster,
)
from src.data.research_store import (
    read_research_partition,
    research_partition_path,
    write_research_partition,
)
from src.data.session_calendar import SessionDay, SessionKind
from src.tools.run_outcome import RUN_OUTCOME_DEGRADED, RUN_OUTCOME_OK, RUN_OUTCOME_SKIPPED

SEOUL = ZoneInfo("Asia/Seoul")
DATE = "2026-11-04"
READY_NOW = datetime(2026, 11, 4, 15, 31, 30, tzinfo=SEOUL)


def _seoul(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 11, 4, hour, minute, second, tzinfo=SEOUL)


def _tick(hhmmss: str, acml: str, rltv: str) -> dict[str, str]:
    return {"stck_cntg_hour": hhmmss, "acml_vol": acml, "tday_rltv": rltv}


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class _NullSession:
    async def __aenter__(self) -> _NullSession:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _FakeKis:
    def __init__(
        self,
        *,
        ticks: dict[str, dict[str, Any]],
        investors: dict[str, dict[str, Any]] | None = None,
        on_tick: Any = None,
    ) -> None:
        self._ticks = ticks
        self._investors = investors or {}
        self._on_tick = on_tick
        self.tick_calls: list[tuple[str, dict[str, Any]]] = []
        self.investor_calls: list[str] = []
        self.token_ensured = 0

    def create_session(self) -> Any:
        return _NullSession()

    async def ensure_token(self, session: Any) -> bool:
        self.token_ensured += 1
        return True

    async def get_intraday_trade_ticks(self, session: Any, code: str, **kwargs: Any) -> dict[str, Any]:
        self.tick_calls.append((code, kwargs))
        if self._on_tick is not None:
            self._on_tick(code)
        return self._ticks[code]

    async def get_investor_trend_estimate(self, session: Any, code: str) -> dict[str, Any]:
        self.investor_calls.append(code)
        return self._investors[code]


@pytest.fixture(autouse=True)
def _history(monkeypatch: Any, tmp_path: Path) -> Path:
    history = tmp_path / "history"
    monkeypatch.setattr(settings, "HISTORY_DIR", history)
    return history


def _profile(**overrides: Any) -> CollectionSettings:
    values: dict[str, Any] = {}
    values.update(overrides)
    return CollectionSettings(_env_file=None, **values)


def _seed_cohort(store: CaptureStore, symbols: list[str]) -> None:
    day = date.fromisoformat(DATE)
    cohort = build_cohort(day, scanned_symbols=symbols, eligible_symbols=symbols, rejections={}, eligibility_rule_version="v1")
    store.publish_manifest(
        CaptureManifest(
            schema_version=1,
            context=CaptureContext(
                trading_date=day,
                run_id="seed-scan",
                dataset=CaptureDataset.SCAN,
                vendor="kis",
                endpoint="seed",
                symbol=None,
                venue="KRX",
                session="regular",
                capture_reason="seed",
                cohort_id=cohort.cohort_id,
                scheduled_at=None,
            ),
            cohort=cohort,
            completed_at=_seoul(15, 20),
            entries=(),
            artifacts=(),
            status=CaptureStatus.COMPLETE,
        )
    )


def _seed_roster(store: CaptureStore, members: list[str]) -> None:
    day = date.fromisoformat(DATE)
    completed = _seoul(15, 20, 5)
    roster = ResearchRoster(
        trading_date=day,
        roster_id=f"research-roster-{DATE}",
        rule_version="liquid_any_direction@v1",
        observed_at=completed,
        as_of_panel_date=date(2026, 11, 3),
        members=tuple(members),
        dropped={},
        truncated=False,
    )
    candidates = [
        RosterCandidate(symbol=member, rank=rank, trade_value_100m=500.0, change_pct=0.0, last_price=10_000)
        for rank, member in enumerate(members, start=1)
    ]
    frame = build_roster_frame(candidates, roster.members, {}, dict.fromkeys(roster.members, 10_000.0))
    publish_research_roster(store, roster, frame, raw_refs=(), completed_at=completed)


def _session_clock() -> SessionClock:
    return SessionClock.standard(date.fromisoformat(DATE))


def test_strength_pre_and_final_from_one_page() -> None:
    rows = [
        _tick("155958", "100", "95.1"),
        _tick("151959", "110", "96.0"),
        _tick("153001", "500", "101.2"),
    ]

    summary = summarize_strength(rows)

    assert summary["strength_pre_auction"] == pytest.approx(96.0)
    assert summary["pre_auction_hour"] == "151959"
    assert summary["strength_final"] == pytest.approx(101.2)
    assert summary["final_hour"] == "153001"
    assert summary["n_ticks"] == 3


def test_strength_no_pre_auction_row() -> None:
    summary = summarize_strength([_tick("153001", "500", "101.2")])

    assert summary["strength_pre_auction"] is None
    assert summary["pre_auction_hour"] is None
    assert summary["strength_final"] == pytest.approx(101.2)


def test_strength_garbage_never_raises() -> None:
    rows = [
        {"stck_cntg_hour": "", "acml_vol": "abc", "tday_rltv": "nan"},
        {"acml_vol": None},
        {"stck_cntg_hour": None, "acml_vol": "5", "tday_rltv": "--"},
    ]

    summary = summarize_strength(rows)

    assert summary["strength_pre_auction"] is None
    assert summary["pre_auction_hour"] is None
    assert summary["strength_final"] is None
    assert summary["final_hour"] is None
    assert summary["n_ticks"] == 3


def test_investor_rows_parse() -> None:
    payload = {
        "rt_cd": "0",
        "output2": [
            {"bsop_hour_gb": "1", "frgn_fake_ntby_qty": "-100", "orgn_fake_ntby_qty": "", "sum_fake_ntby_qty": "-90"},
            {"bsop_hour_gb": "2", "frgn_fake_ntby_qty": "5", "orgn_fake_ntby_qty": "x", "sum_fake_ntby_qty": "7"},
        ],
    }

    rows = investor_rows("000660", payload)

    assert rows == [
        {"symbol": "000660", "bsop_hour_gb": "1", "frgn_fake_ntby_qty": -100, "orgn_fake_ntby_qty": None, "sum_fake_ntby_qty": -90},
        {"symbol": "000660", "bsop_hour_gb": "2", "frgn_fake_ntby_qty": 5, "orgn_fake_ntby_qty": None, "sum_fake_ntby_qty": 7},
    ]
    assert all(isinstance(row["frgn_fake_ntby_qty"], int) for row in rows)


def test_run_rejects_before_close_plus_one_minute(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    client = _FakeKis(ticks={})

    with pytest.raises(ValueError, match="too_early"):
        asyncio.run(
            run_postclose_snapshot(
                DATE,
                profile=_profile(),
                store=store,
                client=client,
                session_clock=_session_clock(),
                now_fn=_Clock(_seoul(15, 30, 30)),
            )
        )
    assert client.tick_calls == []


def test_run_population_cohort_then_extras(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    _seed_cohort(store, ["000001", "000002"])
    _seed_roster(store, ["000001", "000002", "000003"])
    ticks = {
        "000001": {"rt_cd": "0", "output2": [_tick("151959", "110", "96.0"), _tick("153001", "500", "101.2")]},
        "000002": {"rt_cd": "0", "output2": [_tick("153001", "300", "99.0")]},
        "000003": {"rt_cd": "0", "output2": [_tick("153001", "200", "80.0")]},
    }
    investors = {"000003": {"rt_cd": "0", "output2": [{"bsop_hour_gb": "1", "frgn_fake_ntby_qty": "10", "orgn_fake_ntby_qty": "-3", "sum_fake_ntby_qty": "7"}]}}
    client = _FakeKis(ticks=ticks, investors=investors)

    manifest = asyncio.run(
        run_postclose_snapshot(
            DATE,
            profile=_profile(),
            store=store,
            client=client,
            session_clock=_session_clock(),
            now_fn=_Clock(READY_NOW),
        )
    )

    assert [code for code, _ in client.tick_calls] == ["000001", "000002", "000003"]
    assert all(kwargs["max_pages"] == 1 for _, kwargs in client.tick_calls)
    assert client.investor_calls == ["000003"]

    strength = read_research_partition(STRENGTH_DATASET, DATE).set_index("symbol")
    assert strength.loc["000001", "population"] == "cohort"
    assert strength.loc["000002", "population"] == "cohort"
    assert strength.loc["000003", "population"] == "research"
    assert strength.loc["000001", "strength_pre_auction"] == pytest.approx(96.0)
    assert strength.loc["000001", "strength_final"] == pytest.approx(101.2)

    investor = read_research_partition(INVESTOR_DATASET, DATE)
    assert investor["symbol"].tolist() == ["000003"]
    assert investor["frgn_fake_ntby_qty"].tolist() == [10]

    assert manifest.status == CaptureStatus.COMPLETE
    assert manifest.cohort is None
    assert manifest.context.capture_reason == postclose_snapshot.POSTCLOSE_REASON
    assert {(e.symbol, e.dataset) for e in manifest.entries} == {
        ("000001", CaptureDataset.TRADE_TICKS),
        ("000002", CaptureDataset.TRADE_TICKS),
        ("000003", CaptureDataset.TRADE_TICKS),
        ("000003", CaptureDataset.INVESTOR_ESTIMATE),
    }


def test_run_deadline_marks_undispatched_partial(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    _seed_cohort(store, ["000001", "000002"])
    _seed_roster(store, ["000001", "000002", "000003"])
    clock = _Clock(READY_NOW)
    ticks = {
        "000001": {"rt_cd": "0", "output2": [_tick("153001", "500", "101.2")]},
        "000002": {"rt_cd": "0", "output2": [_tick("153001", "300", "99.0")]},
        "000003": {"rt_cd": "0", "output2": [_tick("153001", "200", "80.0")]},
    }
    client = _FakeKis(ticks=ticks, investors={}, on_tick=lambda _code: setattr(clock, "now", _seoul(15, 39, 0)))

    manifest = asyncio.run(
        run_postclose_snapshot(
            DATE,
            profile=_profile(COLLECTION_POSTCLOSE_SNAPSHOT_CONCURRENCY=1),
            store=store,
            client=client,
            session_clock=_session_clock(),
            now_fn=clock,
        )
    )

    assert [code for code, _ in client.tick_calls] == ["000001"]
    assert client.investor_calls == []
    deadline_entries = {(e.symbol, e.dataset): e for e in manifest.entries if e.reason == "deadline"}
    assert set(deadline_entries) == {
        ("000002", CaptureDataset.TRADE_TICKS),
        ("000003", CaptureDataset.TRADE_TICKS),
        ("000003", CaptureDataset.INVESTOR_ESTIMATE),
    }
    assert all(e.status == CaptureStatus.PARTIAL for e in deadline_entries.values())
    assert manifest.status == CaptureStatus.PARTIAL
    strength = read_research_partition(STRENGTH_DATASET, DATE).set_index("symbol")
    assert set(strength.index) == {"000001", "000002", "000003"}


def test_run_vendor_failure_kept_as_row(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    _seed_cohort(store, ["000001", "000002"])
    ticks = {
        "000001": {"rt_cd": "0", "output2": [_tick("153001", "500", "101.2")]},
        "000002": {"rt_cd": "1", "msg1": "error", "output2": []},
    }
    client = _FakeKis(ticks=ticks)

    manifest = asyncio.run(
        run_postclose_snapshot(
            DATE,
            profile=_profile(),
            store=store,
            client=client,
            session_clock=_session_clock(),
            now_fn=_Clock(READY_NOW),
        )
    )

    strength = read_research_partition(STRENGTH_DATASET, DATE).set_index("symbol")
    assert set(strength.index) == {"000001", "000002"}
    assert pd.isna(strength.loc["000002", "strength_final"])
    assert pd.isna(strength.loc["000002", "strength_pre_auction"])
    failed = [e for e in manifest.entries if e.symbol == "000002"]
    assert [(e.status, e.reason) for e in failed] == [(CaptureStatus.FAILED, "vendor_failure")]
    assert manifest.status == CaptureStatus.PARTIAL


def test_run_empty_ticks_is_no_trades_with_evidence(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    _seed_cohort(store, ["000001"])
    client = _FakeKis(ticks={"000001": {"rt_cd": "0", "output2": []}})

    manifest = asyncio.run(
        run_postclose_snapshot(
            DATE,
            profile=_profile(),
            store=store,
            client=client,
            session_clock=_session_clock(),
            now_fn=_Clock(READY_NOW),
        )
    )

    (entry,) = manifest.entries
    assert (entry.status, entry.reason) == (CaptureStatus.NO_TRADES, "no_ticks")
    assert len(entry.raw_refs) == 1
    assert manifest.status == CaptureStatus.COMPLETE


def test_run_refuses_second_capture_before_any_request(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    _seed_cohort(store, ["000001"])
    write_research_partition(pd.DataFrame({"symbol": ["000001"]}), STRENGTH_DATASET, DATE, overwrite=False)
    client = _FakeKis(ticks={"000001": {"rt_cd": "0", "output2": []}})

    with pytest.raises(FileExistsError):
        asyncio.run(
            run_postclose_snapshot(
                DATE,
                profile=_profile(),
                store=store,
                client=client,
                session_clock=_session_clock(),
                now_fn=_Clock(READY_NOW),
            )
        )
    assert client.tick_calls == []


def test_run_guards_clock_date_mismatch(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    client = _FakeKis(ticks={})

    with pytest.raises(ValueError, match="session clock date must equal snapshot_date"):
        asyncio.run(
            run_postclose_snapshot(
                "2026-11-05",
                profile=_profile(),
                store=store,
                client=client,
                session_clock=_session_clock(),
                now_fn=_Clock(READY_NOW),
            )
        )
    assert client.tick_calls == []


def test_run_without_cohort_records_no_entries_and_no_partition(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")

    manifest = asyncio.run(
        run_postclose_snapshot(
            DATE,
            profile=_profile(),
            store=store,
            client=_FakeKis(ticks={}),
            session_clock=_session_clock(),
            now_fn=_Clock(READY_NOW),
        )
    )

    assert manifest.entries == ()
    assert manifest.status is CaptureStatus.PARTIAL
    assert not research_partition_path(STRENGTH_DATASET, DATE).exists()


def test_run_failure_modes_for_ticks_investors_and_empty_estimates(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    _seed_cohort(store, ["000001"])
    _seed_roster(store, ["000001", "000002", "000003", "000004"])

    class _Flaky(_FakeKis):
        async def get_intraday_trade_ticks(self, session: Any, code: str, **kwargs: Any) -> dict[str, Any]:
            if code == "000001":
                raise RuntimeError("transport down")
            return await super().get_intraday_trade_ticks(session, code, **kwargs)

        async def get_investor_trend_estimate(self, session: Any, code: str) -> dict[str, Any]:
            self.investor_calls.append(code)
            if code == "000002":
                raise ValueError("bad payload")
            return self._investors[code]

    ok_ticks = {"rt_cd": "0", "output2": [_tick("153001", "500", "101.2")]}
    client = _Flaky(
        ticks={"000001": {}, "000002": ok_ticks, "000003": ok_ticks, "000004": ok_ticks},
        investors={"000003": {"rt_cd": "1"}, "000004": {"rt_cd": "0", "output2": []}},
    )

    manifest = asyncio.run(
        run_postclose_snapshot(
            DATE,
            profile=_profile(),
            store=store,
            client=client,
            session_clock=_session_clock(),
            now_fn=_Clock(READY_NOW),
        )
    )

    outcomes = {(e.symbol, e.dataset): (e.status, e.reason) for e in manifest.entries}
    assert outcomes[("000001", CaptureDataset.TRADE_TICKS)] == (CaptureStatus.FAILED, "RuntimeError")
    assert outcomes[("000002", CaptureDataset.INVESTOR_ESTIMATE)] == (CaptureStatus.FAILED, "ValueError")
    assert outcomes[("000003", CaptureDataset.INVESTOR_ESTIMATE)] == (CaptureStatus.FAILED, "vendor_failure")
    assert outcomes[("000004", CaptureDataset.INVESTOR_ESTIMATE)] == (CaptureStatus.NO_TRADES, "no_ticks")
    assert manifest.status == CaptureStatus.PARTIAL
    strength = read_research_partition(STRENGTH_DATASET, DATE).set_index("symbol")
    assert pd.isna(strength.loc["000001", "strength_final"])


def _main_setup(
    monkeypatch: Any,
    *,
    session_kind: SessionKind = SessionKind.STANDARD,
    trading: bool = True,
    client: _FakeKis | None = None,
    oracle_error: Exception | None = None,
) -> list[tuple[str, str, dict[str, Any]]]:
    recorded: list[tuple[str, str, dict[str, Any]]] = []
    fake = client or _FakeKis(ticks={"000001": {"rt_cd": "0", "output2": [_tick("153001", "500", "101.2")]}})

    async def _is_trading_day(_client: Any, _session: Any, _day: Any) -> bool:
        if oracle_error is not None:
            raise oracle_error
        return trading

    monkeypatch.setattr(postclose_snapshot, "_now", lambda: READY_NOW)
    monkeypatch.setattr(postclose_snapshot, "CollectionSettings", lambda: _profile())
    monkeypatch.setattr(
        postclose_snapshot,
        "resolve_session_day",
        lambda day, overrides=None: SessionDay(
            trading_date=day,
            kind=session_kind,
            clock=_session_clock() if session_kind is SessionKind.STANDARD else None,
            provenance="test",
        ),
    )
    monkeypatch.setattr(postclose_snapshot, "kis_data_client_kwargs", lambda: {})
    monkeypatch.setattr(postclose_snapshot, "KisApiClient", lambda **_kwargs: fake)
    monkeypatch.setattr(postclose_snapshot, "is_kis_trading_day", _is_trading_day)
    monkeypatch.setattr(
        postclose_snapshot,
        "record_run_outcome",
        lambda job, outcome, **kwargs: recorded.append((job, outcome, kwargs)),
    )
    return recorded


def test_main_completed_snapshot_records_ok(monkeypatch: Any) -> None:
    _seed_cohort(CaptureStore(Path(settings.HISTORY_DIR) / "capture"), ["000001"])
    recorded = _main_setup(monkeypatch)

    postclose_snapshot.main(["--date", DATE])

    assert [(job, outcome) for job, outcome, _ in recorded] == [("postclose_snapshot", RUN_OUTCOME_OK)]
    assert read_research_partition(STRENGTH_DATASET, DATE)["strength_final"].tolist() == [pytest.approx(101.2)]
    assert postclose_snapshot.KisApiClient().token_ensured == 1


def test_main_partial_snapshot_records_degraded(monkeypatch: Any) -> None:
    _seed_cohort(CaptureStore(Path(settings.HISTORY_DIR) / "capture"), ["000001"])
    recorded = _main_setup(
        monkeypatch,
        client=_FakeKis(ticks={"000001": {"rt_cd": "1", "output2": []}}),
    )

    postclose_snapshot.main(["--date", DATE])

    assert [(outcome, kwargs.get("reason")) for _, outcome, kwargs in recorded] == [(RUN_OUTCOME_DEGRADED, "partial")]


@pytest.mark.parametrize(
    ("session_kind", "trading", "reason"),
    [
        (SessionKind.CLOSED, True, "non_trading_day"),
        (SessionKind.SHIFTED, True, "session_shifted"),
        (SessionKind.UNKNOWN, True, "session_unknown"),
        (SessionKind.STANDARD, False, "non_trading_day"),
    ],
)
def test_main_skips_closed_unknown_and_holiday(monkeypatch: Any, session_kind: SessionKind, trading: bool, reason: str) -> None:
    recorded = _main_setup(monkeypatch, session_kind=session_kind, trading=trading)

    postclose_snapshot.main(["--date", DATE])

    assert recorded == [("postclose_snapshot", RUN_OUTCOME_SKIPPED, {"run_date": DATE, "reason": reason})]


def test_main_existing_partition_skips_as_already_captured(monkeypatch: Any) -> None:
    write_research_partition(pd.DataFrame({"symbol": ["000001"]}), STRENGTH_DATASET, DATE, overwrite=False)
    recorded = _main_setup(monkeypatch)

    postclose_snapshot.main(["--date", DATE])

    assert recorded == [("postclose_snapshot", RUN_OUTCOME_SKIPPED, {"run_date": DATE, "reason": "already_captured"})]


def test_main_oracle_failure_records_degraded(monkeypatch: Any) -> None:
    recorded = _main_setup(monkeypatch, oracle_error=RuntimeError("KIS trading-day oracle failed rt_cd=1"))

    postclose_snapshot.main(["--date", DATE])

    assert [outcome for _, outcome, _ in recorded] == [RUN_OUTCOME_DEGRADED]


