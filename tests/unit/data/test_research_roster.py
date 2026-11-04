"""Research roster invariant guards."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.capture_contracts import (
    ArtifactRef,
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    CapturedResponse,
    CoverageEntry,
    build_cohort,
)
from src.data.capture_store import CaptureStore
from src.data.research_roster import (
    RESEARCH_ROSTER_REASON,
    ResearchRoster,
    RosterCandidate,
    build_roster_frame,
    parse_trade_value_rows,
    publish_research_roster,
    read_research_roster,
    select_research_roster,
)

SEOUL = ZoneInfo("Asia/Seoul")
DAY = date(2026, 11, 4)
DECISION_AT = datetime(2026, 11, 4, 15, 20, 28, tzinfo=SEOUL)
OBSERVED = datetime(2026, 11, 4, 15, 20, 30, tzinfo=SEOUL)
COMPLETED = datetime(2026, 11, 4, 15, 20, 35, tzinfo=SEOUL)
ROSTER_ID = "research-roster-2026-11-04-abcdef12"
LARGE_SHARES = 1e8


def _candidate(
    symbol: str, *, tv: float = 500.0, last: int = 10_000, rank: int = 1, chg: float = 0.0
) -> RosterCandidate:
    return RosterCandidate(symbol=symbol, rank=rank, trade_value_100m=tv, change_pct=chg, last_price=last)


def _select(
    candidates: list[RosterCandidate],
    *,
    listed: frozenset[str] | None = None,
    screenable: frozenset[str] | None = None,
    shares: dict[str, float] | None = None,
    max_members: int = 400,
) -> tuple[tuple[str, ...], dict[str, str], dict[str, float]]:
    return select_research_roster(
        candidates,
        listed=frozenset(c.symbol for c in candidates) if listed is None else listed,
        screenable=screenable,
        shares_by_symbol={c.symbol: LARGE_SHARES for c in candidates} if shares is None else shares,
        min_trade_value_100m=100.0,
        min_market_cap_100m=500.0,
        max_members=max_members,
    )


def _roster(
    members: tuple[str, ...],
    dropped: dict[str, str],
    *,
    truncated: bool = False,
) -> ResearchRoster:
    return ResearchRoster(
        trading_date=DAY,
        roster_id=ROSTER_ID,
        rule_version="liquid_any_direction@v1",
        observed_at=OBSERVED,
        as_of_panel_date=date(2026, 11, 3),
        members=members,
        dropped=dropped,
        truncated=truncated,
    )


def _raw_ref(store: CaptureStore, roster_id: str = ROSTER_ID) -> ArtifactRef:
    context = CaptureContext(
        trading_date=DAY,
        run_id=roster_id,
        dataset=CaptureDataset.SCAN,
        vendor="kiwoom",
        endpoint="trade-value-ranking",
        symbol=None,
        venue="KRX",
        session="regular",
        capture_reason=RESEARCH_ROSTER_REASON,
        cohort_id=None,
        scheduled_at=None,
    )
    response = CapturedResponse(
        context=context,
        request_started_at=OBSERVED,
        received_at=OBSERVED + timedelta(seconds=1),
        payload={"return_code": 0, "trde_prica_upper": []},
        status=CaptureStatus.COMPLETE,
        source_timestamp=None,
        source_published_at=None,
        page_index=0,
        attempt_index=0,
        continuation={},
        error_type=None,
    )
    return store.append_response(response)


def _publish_roster(
    store: CaptureStore,
    roster: ResearchRoster,
    *,
    completed_at: datetime = COMPLETED,
) -> None:
    candidates = [_candidate(symbol) for symbol in (*roster.members, *roster.dropped)]
    est_mc = dict.fromkeys(roster.members, 10000.0)
    frame = build_roster_frame(candidates, roster.members, roster.dropped, est_mc)
    publish_research_roster(
        store,
        roster,
        frame,
        raw_refs=[_raw_ref(store, roster.roster_id)],
        completed_at=completed_at,
    )


def _decision_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["005930", "000660"],
            "admitted": [True, False],
            "close": [72000.0, float("nan")],
            "snapshot_timestamp": [DECISION_AT, DECISION_AT],
            "feature_available_timestamp": [DECISION_AT, DECISION_AT],
        }
    )


def _publish_decision(store: CaptureStore) -> str:
    run_id = "decision-1"
    cohort = build_cohort(
        DAY,
        ["005930", "000660", "035420"],
        ["005930", "000660"],
        {"035420": "suspended"},
        eligibility_rule_version="v1",
    )
    context = CaptureContext(
        trading_date=DAY,
        run_id=run_id,
        dataset=CaptureDataset.PRICE,
        vendor="kis",
        endpoint="price",
        symbol="005930",
        venue="KRX",
        session="regular",
        capture_reason="closing-sweep",
        cohort_id=None,
        scheduled_at=None,
    )
    raw = store.append_response(
        CapturedResponse(
            context=context,
            request_started_at=DECISION_AT,
            received_at=DECISION_AT + timedelta(seconds=1),
            payload={"output": {"code": "005930", "price": "72000"}},
            status=CaptureStatus.COMPLETE,
            source_timestamp=None,
            source_published_at=None,
            page_index=0,
            attempt_index=0,
            continuation={},
            error_type=None,
        )
    )
    entry = CoverageEntry(
        symbol="005930",
        dataset=CaptureDataset.PRICE,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=CaptureStatus.COMPLETE,
        rows=1,
        first_event_time=None,
        last_event_time=None,
        reason="sweep done",
        raw_refs=(raw,),
    )
    store.publish_decision(_decision_frame(), cohort=cohort, run_id=run_id, completed_at=DECISION_AT, entries=[entry])
    return run_id


def test_select_orders_by_trade_value() -> None:
    candidates = [_candidate("000200", tv=300.0), _candidate("000100", tv=300.0), _candidate("000300", tv=900.0)]

    members, dropped, _ = _select(candidates)

    assert members == ("000300", "000100", "000200")
    assert dropped == {}


@pytest.mark.parametrize(
    ("expected", "candidate", "listed", "screenable", "shares"),
    [
        ("below_trade_value", _candidate("000001", tv=50.0), False, False, None),
        ("not_listed_in_panel", _candidate("000002", tv=500.0), False, False, None),
        ("non_screenable_class", _candidate("000003", tv=500.0), True, False, LARGE_SHARES),
        ("market_cap_unknown", _candidate("000004", tv=500.0), True, True, None),
        ("below_market_cap", _candidate("000005", tv=500.0, last=10_000), True, True, 1e6),
    ],
)
def test_select_drop_reasons_first_failure(
    expected: str,
    candidate: RosterCandidate,
    listed: bool,
    screenable: bool,
    shares: float | None,
) -> None:
    symbol = candidate.symbol
    members, dropped, _ = _select(
        [candidate],
        listed=frozenset({symbol}) if listed else frozenset(),
        screenable=frozenset({symbol}) if screenable else frozenset(),
        shares={} if shares is None else {symbol: shares},
    )

    assert members == ()
    assert dropped == {symbol: expected}


def test_select_over_capacity() -> None:
    candidates = [_candidate(f"00010{i}", tv=1000.0 - i) for i in range(5)]

    members, dropped, _ = _select(candidates, max_members=3)

    assert members == ("000100", "000101", "000102")
    assert dropped == {"000103": "over_capacity", "000104": "over_capacity"}


def test_select_unknown_shares() -> None:
    candidates = [_candidate("005930")]

    members, dropped, est_mc = _select(candidates, shares={})

    assert members == ()
    assert dropped == {"005930": "market_cap_unknown"}
    assert "005930" not in est_mc


def test_select_direction_free() -> None:
    candidates = [_candidate("000010", chg=-15.0), _candidate("000020", chg=0.0), _candidate("000030", chg=25.0)]

    members, dropped, _ = _select(candidates)

    assert set(members) == {"000010", "000020", "000030"}
    assert dropped == {}


def test_parse_rows_normalizes_symbol_and_units() -> None:
    rows = [
        {"stk_cd": "0147B0", "now_rank": "1", "trde_prica": "12345", "cur_prc": "+10000", "flu_rt": "+1.50"},
        {"stk_cd": "005930_AL", "now_rank": "2", "trde_prica": "100", "cur_prc": "-80000", "flu_rt": "-15.00"},
    ]

    candidates, dropped = parse_trade_value_rows(rows)

    assert [c.symbol for c in candidates] == ["0147B0", "005930"]
    assert candidates[0].trade_value_100m == pytest.approx(123.45, rel=1e-12, abs=1e-12)
    assert candidates[1].last_price == 80000
    assert candidates[1].change_pct == pytest.approx(-15.0)
    assert dropped == {}


def test_parse_rows_unparseable_recorded() -> None:
    rows = [
        {"stk_cd": "005930", "now_rank": "1", "trde_prica": "abc", "cur_prc": "+1000", "flu_rt": "+1.00"},
        {"stk_cd": "000660", "now_rank": "2", "trde_prica": "500", "cur_prc": "+1000", "flu_rt": "+1.00"},
        {"stk_cd": "000660", "now_rank": "3", "trde_prica": "900", "cur_prc": "+1000", "flu_rt": "+1.00"},
    ]

    candidates, dropped = parse_trade_value_rows(rows)

    assert dropped == {"005930": "unparseable"}
    assert [c.symbol for c in candidates] == ["000660"]
    assert candidates[0].rank == 2


def test_publish_then_read_round_trip(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    roster = _roster(("005930", "000660"), {"035420": "below_market_cap"})
    _publish_roster(store, roster)

    read = read_research_roster(store, "2026-11-04", available_by=COMPLETED + timedelta(minutes=1))

    assert read.members == roster.members
    assert read.dropped == roster.dropped
    assert read.observed_at == roster.observed_at
    assert read.roster_id == roster.roster_id
    assert read.truncated is False


def test_read_respects_available_by(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    _publish_roster(store, _roster(("005930",), {}))

    with pytest.raises(FileNotFoundError):
        read_research_roster(store, "2026-11-04", available_by=COMPLETED - timedelta(seconds=1))


def test_roster_never_hijacks_cohort(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    _publish_decision(store)
    _publish_roster(store, _roster(("005930",), {}))

    cohort = store.read_cohort("2026-11-04", available_by=COMPLETED + timedelta(minutes=1))

    assert cohort.eligible_symbols == ("000660", "005930")
    assert cohort.cohort_id.startswith("cohort-2026-11-04-")


def test_roster_never_hijacks_decision_read(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    run_id = _publish_decision(store)
    _publish_roster(store, _roster(("005930", "000660"), {}))

    decision = store.read_decision("2026-11-04", available_by=datetime(2026, 11, 4, 15, 21, tzinfo=SEOUL))

    assert decision["capture_run_id"].tolist() == [run_id, run_id]
    assert decision["symbol"].tolist() == ["005930", "000660"]


def test_publish_rejects_naive_or_early_completed_at(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    roster = _roster(("005930",), {})
    frame = build_roster_frame([_candidate("005930")], roster.members, roster.dropped, {"005930": 10_000.0})

    with pytest.raises(ValueError, match="timezone-aware"):
        publish_research_roster(
            store,
            roster,
            frame,
            raw_refs=[_raw_ref(store)],
            completed_at=datetime(2026, 11, 4, 15, 20, 35),
        )
    with pytest.raises(ValueError, match="precedes observed_at"):
        publish_research_roster(
            store,
            roster,
            frame,
            raw_refs=[_raw_ref(store)],
            completed_at=OBSERVED - timedelta(seconds=1),
        )
    assert store.read_manifests("2026-11-04") == ()


def test_roster_manifest_truncated_is_partial(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    roster = _roster(("005930", "000660"), {}, truncated=True)
    _publish_roster(store, roster)

    (manifest,) = store.read_manifests("2026-11-04")

    assert manifest.status == CaptureStatus.PARTIAL
    assert manifest.entries[0].status == CaptureStatus.PARTIAL
    assert manifest.entries[0].reason == "truncated"
    assert manifest.cohort is None
    assert read_research_roster(store, "2026-11-04", available_by=COMPLETED + timedelta(minutes=1)).truncated is True


def test_roster_rejects_member_also_dropped() -> None:
    with pytest.raises(ValidationError):
        _roster(("005930",), {"005930": "over_capacity"})


def test_roster_rejects_naive_observed_at() -> None:
    with pytest.raises(ValidationError):
        ResearchRoster(
            trading_date=DAY,
            roster_id=ROSTER_ID,
            rule_version="liquid_any_direction@v1",
            observed_at=datetime(2026, 11, 4, 15, 20, 30),
            as_of_panel_date=date(2026, 11, 3),
            members=(),
            dropped={},
            truncated=False,
        )


@pytest.mark.parametrize(
    ("override"),
    [
        {"roster_id": "research/roster"},
        {"members": ("005930", "005930")},
        {"members": ("00-5930",)},
    ],
)
def test_roster_rejects_unsafe_identity_or_members(override: dict[str, Any]) -> None:
    fields: dict[str, Any] = {
        "trading_date": DAY,
        "roster_id": ROSTER_ID,
        "rule_version": "liquid_any_direction@v1",
        "observed_at": OBSERVED,
        "as_of_panel_date": date(2026, 11, 3),
        "members": ("005930",),
        "dropped": {},
        "truncated": False,
    }
    fields.update(override)
    with pytest.raises(ValidationError):
        ResearchRoster(**fields)


def test_parse_rows_missing_symbol_and_nonfinite_change_are_unparseable() -> None:
    rows = [
        {"now_rank": "1", "trde_prica": "50000", "cur_prc": "+1000", "flu_rt": "+1.00"},
        {"stk_cd": "000660", "now_rank": "2", "trde_prica": "50000", "cur_prc": "+1000", "flu_rt": "nan"},
    ]

    candidates, dropped = parse_trade_value_rows(rows)

    assert candidates == []
    assert dropped == {"<row-0>": "unparseable", "000660": "unparseable"}


def test_publish_rejects_empty_frame_and_member_mismatch(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    roster = _roster(("005930",), {})
    empty = build_roster_frame([], (), {}, {})
    mismatched = build_roster_frame([_candidate("000660")], ("000660",), {}, {"000660": 10_000.0})

    with pytest.raises(ValueError, match="empty"):
        publish_research_roster(store, roster, empty, raw_refs=[_raw_ref(store)], completed_at=COMPLETED)
    with pytest.raises(ValueError, match="do not match"):
        publish_research_roster(store, roster, mismatched, raw_refs=[_raw_ref(store)], completed_at=COMPLETED)
    assert store.read_manifests("2026-11-04") == ()


def test_read_rejects_naive_available_by(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    _publish_roster(store, _roster(("005930",), {}))

    with pytest.raises(ValueError, match="aware cutoff"):
        read_research_roster(store, "2026-11-04", available_by=datetime(2026, 11, 4, 15, 21))


def _enriched(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["observed_at"] = OBSERVED.isoformat()
    frame["as_of_panel_date"] = "2026-11-03"
    frame["rule_version"] = "liquid_any_direction@v1"
    return frame


def _valid_frame() -> pd.DataFrame:
    return _enriched(
        build_roster_frame(
            [_candidate("005930"), _candidate("035420")],
            ("005930",),
            {"035420": "below_market_cap"},
            {"005930": 10_000.0},
        )
    )


def _publish_evidence(store: CaptureStore, frames: list[pd.DataFrame]) -> None:
    context = CaptureContext(
        trading_date=DAY,
        run_id=ROSTER_ID,
        dataset=CaptureDataset.SCAN,
        vendor="kiwoom",
        endpoint="trade-value-ranking",
        symbol=None,
        venue="KRX",
        session="regular",
        capture_reason=RESEARCH_ROSTER_REASON,
        cohort_id=None,
        scheduled_at=None,
    )
    refs = [
        store.publish_frame(frame, context=context.model_copy(update={"run_id": f"{ROSTER_ID}-{index}"}))
        for index, frame in enumerate(frames)
    ]
    raw = _raw_ref(store)
    entry = CoverageEntry(
        symbol=None,
        dataset=CaptureDataset.SCAN,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=CaptureStatus.COMPLETE,
        rows=1,
        first_event_time=None,
        last_event_time=None,
        reason=RESEARCH_ROSTER_REASON,
        raw_refs=(raw,),
    )
    store.publish_manifest(
        CaptureManifest(
            schema_version=1,
            context=context,
            cohort=None,
            completed_at=COMPLETED,
            entries=(entry,),
            artifacts=(*refs, raw),
            status=CaptureStatus.COMPLETE,
        )
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: [frame, frame.copy()], "exactly one frame artifact"),
        (lambda frame: [frame.drop(columns=["member_order"])], "incomplete"),
        (
            lambda frame: [frame.assign(observed_at=["2026-11-04T15:20:30+09:00", "2026-11-04T15:20:31+09:00"])],
            "inconsistent 'observed_at'",
        ),
        (lambda frame: [frame.assign(reason=[None, None])], "drop reason"),
    ],
)
def test_read_rejects_inconsistent_evidence(tmp_path: Path, mutate: Any, message: str) -> None:
    store = CaptureStore(tmp_path / "capture")
    _publish_evidence(store, mutate(_valid_frame()))

    with pytest.raises(ValueError, match=message):
        read_research_roster(store, "2026-11-04", available_by=COMPLETED + timedelta(minutes=1))


def test_roster_without_members_round_trips(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path / "capture")
    roster = _roster((), {"005930": "below_trade_value"})
    frame = build_roster_frame([_candidate("005930", tv=50.0)], (), roster.dropped, {})
    publish_research_roster(store, roster, frame, raw_refs=[_raw_ref(store)], completed_at=COMPLETED)

    read = read_research_roster(store, "2026-11-04", available_by=COMPLETED + timedelta(minutes=1))

    assert read.members == ()
    assert read.dropped == {"005930": "below_trade_value"}
