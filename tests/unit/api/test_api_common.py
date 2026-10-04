"""Invariant guards for the vendor-neutral broker helpers in src.api._common."""

from __future__ import annotations


def test_parse_expires_in_numeric_types() -> None:
    from src.api._common import parse_expires_in

    assert parse_expires_in(86400) == 86400.0
    assert parse_expires_in(86400.5) == 86400.5
    assert parse_expires_in(-5) == -5.0
    assert isinstance(parse_expires_in(86400), float)


def test_parse_expires_in_rejects_booleans() -> None:
    from src.api._common import parse_expires_in

    assert parse_expires_in(True) is None
    assert parse_expires_in(False) is None


def test_parse_expires_in_numeric_strings() -> None:
    from src.api._common import parse_expires_in

    assert parse_expires_in("86400") == 86400.0
    assert parse_expires_in(" 86400 ") == 86400.0
    assert parse_expires_in("1.5") == 1.5
    assert parse_expires_in("+60") == 60.0
    assert parse_expires_in("-60") == -60.0


def test_parse_expires_in_rejects_malformed_strings() -> None:
    from src.api._common import parse_expires_in

    for raw in ("abc", "", "1.2.3", "1e5", "+-5"):
        assert parse_expires_in(raw) is None, raw


def test_parse_expires_in_rejects_other_types() -> None:
    from src.api._common import parse_expires_in

    assert parse_expires_in(None) is None
    assert parse_expires_in([]) is None
    assert parse_expires_in({}) is None


def test_validate_target_ymd_normalizes() -> None:
    from src.api._common import validate_target_ymd

    assert validate_target_ymd("2026-09-04") == "20260904"
    assert validate_target_ymd("20260904") == "20260904"


def test_validate_target_ymd_rejects_invalid_dates() -> None:
    import pytest

    from src.api._common import validate_target_ymd

    for raw in ("2026-13-01", "abc"):
        with pytest.raises(ValueError, match="invalid target date") as exc_info:
            validate_target_ymd(raw)
        assert repr(raw) in str(exc_info.value)
        assert exc_info.value.__cause__ is None


def test_resolve_chart_budget_defaults() -> None:
    from src.api._common import resolve_chart_budget

    assert resolve_chart_budget(None, 0) == (1, None)
    assert resolve_chart_budget(None, 7) == (7, None)


def test_resolve_chart_budget_explicit() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.api._common import resolve_chart_budget
    from src.data.capture_contracts import ChartBudget

    deadline = datetime(2026, 9, 4, 15, 20, tzinfo=ZoneInfo("Asia/Seoul"))
    budget = ChartBudget(max_pages=3, deadline=deadline, request_timeout_seconds=5.0)
    assert resolve_chart_budget(budget, 7) == (3, deadline)


def test_deadline_remaining_sign() -> None:
    from datetime import timedelta

    from src.api._common import deadline_remaining, now_seoul

    assert deadline_remaining(None) is None
    ahead = deadline_remaining(now_seoul() + timedelta(seconds=60))
    assert ahead is not None and 0 < ahead <= 60
    behind = deadline_remaining(now_seoul() - timedelta(seconds=60))
    assert behind is not None and behind < 0


def test_now_seoul_is_aware_kst() -> None:
    from datetime import timedelta

    from src.api._common import now_seoul

    now = now_seoul()
    assert now.utcoffset() == timedelta(hours=9)
    assert now.tzinfo is not None and now.tzinfo.key == "Asia/Seoul"  # type: ignore[attr-defined]
