"""Capture contract invariant guards for cohorts, clocks, and certification."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.data.capture_contracts import (
    ArtifactRef,
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    CapturedResponse,
    ChartBudget,
    Cohort,
    CoverageEntry,
    RawCaptureError,
    SessionClock,
    build_cohort,
)

SEOUL = ZoneInfo("Asia/Seoul")


def _context(**overrides: object) -> CaptureContext:
    base: dict[str, object] = {
        "trading_date": date(2026, 9, 17),
        "run_id": "run-1",
        "dataset": CaptureDataset.PRICE,
        "vendor": "kis",
        "endpoint": "price",
        "symbol": "005930",
        "venue": "KRX",
        "session": "regular",
        "capture_reason": "closing-sweep",
        "cohort_id": None,
        "scheduled_at": None,
    }
    base.update(overrides)
    return CaptureContext(**base)  # type: ignore[arg-type]


def _response(**overrides: object) -> CapturedResponse:
    start = datetime(2026, 9, 17, 15, 30, tzinfo=SEOUL)
    base: dict[str, object] = {
        "context": _context(),
        "request_started_at": start,
        "received_at": start + timedelta(seconds=1),
        "payload": {"output": {"code": "005930"}},
        "status": CaptureStatus.COMPLETE,
        "source_timestamp": None,
        "source_published_at": None,
        "page_index": 0,
        "attempt_index": 0,
        "continuation": {},
        "error_type": None,
    }
    base.update(overrides)
    return CapturedResponse(**base)  # type: ignore[arg-type]


def _artifact() -> ArtifactRef:
    return ArtifactRef(path="raw-2026-09-17.parquet", sha256="a" * 64, bytes=128, rows=10)


def test_cohort_includes_admission_failures() -> None:
    """Cohort includes admission failures."""
    cohort = build_cohort(
        date(2026, 9, 17),
        ["005930", "000660", "035420"],
        ["005930", "000660"],
        {"035420": "suspended"},
        eligibility_rule_version="v1",
    )
    assert set(cohort.eligible_symbols) == {"005930", "000660"}
    assert set(cohort.scanned_symbols) == {"005930", "000660", "035420"}


def test_cohort_identity_is_deterministic() -> None:
    """Cohort identity is deterministic."""
    first = build_cohort(
        date(2026, 9, 17), ["005930", "000660"], ["005930"], {"000660": "halted"}, eligibility_rule_version="v1"
    )
    second = build_cohort(
        date(2026, 9, 17), ["000660", "005930"], ["005930"], {"000660": "halted"}, eligibility_rule_version="v1"
    )
    assert first.cohort_id == second.cohort_id
    changed = build_cohort(
        date(2026, 9, 17), ["005930", "000660"], ["005930", "000660"], {}, eligibility_rule_version="v1"
    )
    assert changed.cohort_id != first.cohort_id


def test_unknown_clocks_remain_unknown() -> None:
    """Unknown clocks remain unknown."""
    response = _response(source_published_at=None, source_timestamp=None)
    assert response.source_published_at is None
    assert response.source_timestamp is None
    clock = SessionClock.standard(date(2026, 9, 17))
    assert clock.provenance == "standard_profile"
    assert clock.open_at.tzinfo is not None


def test_invalid_observation_chronology_fails() -> None:
    """Invalid observation chronology fails."""
    start = datetime(2026, 9, 17, 15, 30, tzinfo=SEOUL)
    with pytest.raises(ValueError, match="must not precede"):
        _response(request_started_at=start, received_at=start - timedelta(seconds=1))
    with pytest.raises(ValueError, match="timezone-aware"):
        _response(request_started_at=datetime(2026, 9, 17, 15, 30), received_at=start)
    with pytest.raises(ValueError, match="timezone-aware"):
        _response(source_timestamp=datetime(2026, 9, 17, 15, 30))
    with pytest.raises(ValueError, match="open_at < close_at"):
        SessionClock(
            trading_date=date(2026, 9, 17),
            open_at=datetime(2026, 9, 17, 15, 30, tzinfo=SEOUL),
            close_at=datetime(2026, 9, 17, 9, 0, tzinfo=SEOUL),
            close_confirmation_deadline=datetime(2026, 9, 17, 15, 33, tzinfo=SEOUL),
            provenance="standard_profile",
        )
    with pytest.raises(ValueError, match="must share trading_date"):
        SessionClock(
            trading_date=date(2026, 9, 18),
            open_at=datetime(2026, 9, 17, 9, 0, tzinfo=SEOUL),
            close_at=datetime(2026, 9, 17, 15, 30, tzinfo=SEOUL),
            close_confirmation_deadline=datetime(2026, 9, 17, 15, 33, tzinfo=SEOUL),
            provenance="standard_profile",
        )


def test_empty_does_not_certify_no_trade() -> None:
    """Empty does not certify no trade."""
    entry = CoverageEntry(
        symbol="005930",
        dataset=CaptureDataset.PRICE,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=CaptureStatus.COMPLETE,
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason="empty page retained",
        raw_refs=(_artifact(),),
    )
    assert entry.rows == 0
    with pytest.raises(ValueError, match="explicit evidence"):
        CoverageEntry(
            symbol="005930",
            dataset=CaptureDataset.PRICE,
            venue="KRX",
            session="regular",
            scheduled_at=None,
            status=CaptureStatus.NO_TRADES,
            rows=0,
            first_event_time=None,
            last_event_time=None,
            reason="",
            raw_refs=(),
        )
    with pytest.raises(ValueError, match="explicit evidence"):
        CoverageEntry(
            symbol="005930",
            dataset=CaptureDataset.PRICE,
            venue="KRX",
            session="regular",
            scheduled_at=None,
            status=CaptureStatus.NOT_APPLICABLE,
            rows=0,
            first_event_time=None,
            last_event_time=None,
            reason="no proof",
            raw_refs=(),
        )
    with pytest.raises(ValueError, match="UNKNOWN venue"):
        CoverageEntry(
            symbol="005930",
            dataset=CaptureDataset.PRICE,
            venue="UNKNOWN",
            session="regular",
            scheduled_at=None,
            status=CaptureStatus.COMPLETE,
            rows=1,
            first_event_time=None,
            last_event_time=None,
            reason="sweep done",
            raw_refs=(_artifact(),),
        )


def test_rejected_and_eligible_branches_reconcile() -> None:
    """Rejected and eligible branches reconcile."""
    with pytest.raises(ValueError, match="one consistent branch"):
        build_cohort(
            date(2026, 9, 17),
            ["005930", "000660"],
            ["005930"],
            {},
            eligibility_rule_version="v1",
        )
    with pytest.raises(ValueError, match="must be disjoint"):
        build_cohort(
            date(2026, 9, 17),
            ["005930"],
            ["005930"],
            {"005930": "halted"},
            eligibility_rule_version="v1",
        )
    with pytest.raises(ValueError, match="one consistent branch"):
        build_cohort(
            date(2026, 9, 17),
            ["005930"],
            ["005930", "000660"],
            {},
            eligibility_rule_version="v1",
        )
    with pytest.raises(ValueError, match="nonempty"):
        build_cohort(
            date(2026, 9, 17),
            ["005930", "000660"],
            ["005930"],
            {"000660": ""},
            eligibility_rule_version="v1",
        )
    with pytest.raises(ValueError, match="one consistent branch"):
        Cohort(
            trading_date=date(2026, 9, 17),
            cohort_id="cohort-x",
            eligible_symbols=("005930",),
            scanned_symbols=("005930", "000660"),
            eligibility_rule_version="v1",
            rejections={},
        )


def test_response_guards_payload_and_continuation() -> None:
    payload = {"output": {"code": "005930"}}
    response = _response(payload=payload)
    payload["output"]["code"] = "MUTATED"
    assert response.payload == {"output": {"code": "005930"}}
    with pytest.raises(ValueError, match="nonfinite"):
        _response(payload={"v": float("inf")})
    with pytest.raises(ValueError, match="nonfinite"):
        _response(payload={"v": float("nan")})
    with pytest.raises(ValueError, match="only JSON"):
        _response(payload={"v": datetime(2026, 9, 17)})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="must not carry credentials"):
        _response(continuation={"Authorization": "secret"})
    with pytest.raises(ValueError, match="must not carry credentials"):
        _response(continuation={"x-appkey": "1"})
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        _response(page_index=-1)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        _response(attempt_index=-1)
    with pytest.raises(ValueError, match="security identifier"):
        _context(symbol="005930!")
    with pytest.raises(ValueError, match="must be nonempty"):
        _context(vendor="")
    with pytest.raises(ValueError, match="path-safe"):
        _context(vendor="a/b")


def test_budgets_artifacts_and_manifest_validate() -> None:
    budget = ChartBudget(max_pages=3, deadline=None, request_timeout_seconds=2.5)
    assert budget.max_pages == 3
    aware_deadline = datetime(2026, 9, 17, 15, 33, tzinfo=SEOUL)
    assert ChartBudget(max_pages=1, deadline=aware_deadline, request_timeout_seconds=1.0).deadline == aware_deadline
    with pytest.raises(ValueError, match="greater than 0"):
        ChartBudget(max_pages=0, deadline=None, request_timeout_seconds=1.0)
    with pytest.raises(ValueError, match=r"finite|greater than 0"):
        ChartBudget(max_pages=1, deadline=None, request_timeout_seconds=float("inf"))
    with pytest.raises(ValueError, match="timezone-aware"):
        ChartBudget(max_pages=1, deadline=datetime(2026, 9, 17, 15, 33), request_timeout_seconds=1.0)
    with pytest.raises(ValueError, match=r"must be nonempty|path-safe"):
        ArtifactRef(path="", sha256="a" * 64, bytes=1)
    with pytest.raises(ValueError, match="path-safe"):
        ArtifactRef(path="/absolute/path.parquet", sha256="a" * 64, bytes=1)
    with pytest.raises(ValueError, match="path-safe"):
        ArtifactRef(path="raw/./dot.parquet", sha256="a" * 64, bytes=1)
    nested = ArtifactRef(path="raw/2026-09-17/kis/PRICE/run-1/a.json.gz", sha256="a" * 64, bytes=1)
    assert nested.path.startswith("raw/")
    with pytest.raises(ValueError, match="nonempty"):
        ArtifactRef(path="p", sha256="  ", bytes=1)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        ArtifactRef(path="p", sha256="a" * 64, bytes=-1)
    manifest = CaptureManifest(
        schema_version=1,
        context=_context(),
        cohort=None,
        completed_at=datetime(2026, 9, 17, 15, 34, tzinfo=SEOUL),
        entries=(),
        artifacts=(),
        status=CaptureStatus.COMPLETE,
    )
    assert manifest.schema_version == 1
    with pytest.raises(ValueError, match="timezone-aware"):
        CaptureManifest(
            schema_version=1,
            context=_context(),
            cohort=None,
            completed_at=datetime(2026, 9, 17, 15, 34),
            entries=(),
            artifacts=(),
            status=CaptureStatus.COMPLETE,
        )
    assert issubclass(RawCaptureError, OSError)


def test_capture_boundaries_cover_optional_branches() -> None:
    start = datetime(2026, 9, 17, 15, 30, tzinfo=SEOUL)
    context = _context(symbol=None, cohort_id="cohort-x", scheduled_at=start)
    assert context.symbol is None
    assert context.cohort_id == "cohort-x"
    assert context.scheduled_at == start
    assert _response(payload=None).payload is None
    assert _response(error_type="timeout").error_type == "timeout"
    mixed = _response(payload={"n": 3, "ratio": 1.5, "tags": ["a", 1], "nested": {"k": True}})
    assert mixed.payload == {"n": 3, "ratio": 1.5, "tags": ["a", 1], "nested": {"k": True}}
    with pytest.raises(ValueError, match="keys must be strings"):
        _response(payload={1: "x"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="cursor names"):
        _response(continuation={"": "next"})
    with pytest.raises(ValueError, match="path-safe"):
        _context(vendor="..")
    tick = CoverageEntry(
        symbol=None,
        dataset=CaptureDataset.TRADE_TICKS,
        venue="KRX",
        session="regular",
        scheduled_at=start,
        status=CaptureStatus.PARTIAL,
        rows=4,
        first_event_time=start,
        last_event_time=start + timedelta(seconds=2),
        reason="partial page",
        raw_refs=(_artifact(),),
    )
    assert tick.first_event_time == start
    with pytest.raises(ValueError, match="nonempty"):
        Cohort(
            trading_date=date(2026, 9, 17),
            cohort_id="cohort-x",
            eligible_symbols=("005930",),
            scanned_symbols=("005930", "000660"),
            eligibility_rule_version="v1",
            rejections={"000660": " "},
        )
    with pytest.raises(ValueError, match="must be disjoint"):
        Cohort(
            trading_date=date(2026, 9, 17),
            cohort_id="cohort-x",
            eligible_symbols=("005930",),
            scanned_symbols=("005930",),
            eligibility_rule_version="v1",
            rejections={"005930": "halted"},
        )
