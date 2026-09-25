"""Session calendar resolver invariant guards."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from src.config.krx_calendar import KRX_CALENDAR
from src.data.capture_contracts import SessionClock
from src.data.session_calendar import SessionKind, resolve_session_day, trading_session_gate

_SEOUL = ZoneInfo("Asia/Seoul")


def _shifted_clock(trading_day: date) -> SessionClock:
    return SessionClock(
        trading_date=trading_day,
        open_at=datetime(trading_day.year, trading_day.month, trading_day.day, 10, 0, 0, tzinfo=_SEOUL),
        close_at=datetime(trading_day.year, trading_day.month, trading_day.day, 16, 30, 0, tzinfo=_SEOUL),
        close_confirmation_deadline=datetime(trading_day.year, trading_day.month, trading_day.day, 16, 33, 0, tzinfo=_SEOUL),
        provenance="operator_override",
    )


def test_resolve_session_day_marks_weekend_closed_beyond_coverage() -> None:
    day = resolve_session_day(date(2027, 6, 5), overrides={})

    assert day.kind is SessionKind.CLOSED
    assert day.clock is None
    assert day.provenance == "weekend"
    assert trading_session_gate(day) == "non_trading_day"


def test_resolve_session_day_marks_declared_holiday_closed() -> None:
    day = resolve_session_day(date(2026, 10, 9), overrides={})

    assert day.kind is SessionKind.CLOSED
    assert day.clock is None
    assert trading_session_gate(day) == "non_trading_day"


def test_resolve_session_day_marks_ordinary_weekday_standard() -> None:
    target = date(2026, 10, 6)
    day = resolve_session_day(target, overrides={})

    assert day.kind is SessionKind.STANDARD
    assert day.clock == SessionClock.standard(target)
    assert day.clock is not None and day.clock.trading_date == target
    assert trading_session_gate(day) is None


def test_resolve_session_day_marks_beyond_verified_range_unknown() -> None:
    target = KRX_CALENDAR.verified_through + timedelta(days=1)
    assert target.weekday() < 5
    day = resolve_session_day(target, overrides={})

    assert day.kind is SessionKind.UNKNOWN
    assert day.clock is None
    assert day.provenance == "outside_verified_range"
    assert trading_session_gate(day) == "calendar_unverified"


def test_resolve_session_day_marks_before_verified_range_unknown() -> None:
    day = resolve_session_day(date(2026, 9, 22), overrides={})

    assert day.kind is SessionKind.UNKNOWN
    assert day.clock is None
    assert trading_session_gate(day) == "calendar_unverified"


def test_resolve_session_day_applies_operator_override_as_shifted() -> None:
    target = date(2026, 10, 6)
    clock = _shifted_clock(target)
    day = resolve_session_day(target, overrides={target.isoformat(): clock})

    assert day.kind is SessionKind.SHIFTED
    assert day.provenance == "operator_override"
    assert day.clock is clock
    assert trading_session_gate(day) == "session_shifted"


def test_resolve_session_day_reads_operator_override_from_settings(monkeypatch) -> None:
    from src import settings as _settings

    target = date(2026, 10, 6)
    clock = _shifted_clock(target)
    monkeypatch.setattr(_settings, "COLLECTION_SESSION_OVERRIDES", {target.isoformat(): clock})
    day = resolve_session_day(target)

    assert day.kind is SessionKind.SHIFTED
    assert day.provenance == "operator_override"
    assert day.clock is clock


def test_resolve_session_day_rejects_override_on_closed_date() -> None:
    import pytest

    target = date(2026, 10, 9)
    with pytest.raises(ValueError, match="closed date"):
        resolve_session_day(target, overrides={target.isoformat(): _shifted_clock(target)})


def test_resolve_session_day_rejects_override_with_mismatched_clock() -> None:
    import pytest

    target = date(2026, 10, 6)
    with pytest.raises(ValueError, match="differs from its key"):
        resolve_session_day(target, overrides={target.isoformat(): _shifted_clock(date(2026, 10, 7))})


def test_resolve_session_day_applies_calendar_shifted_session() -> None:
    from src.config.krx_calendar import KrxCalendar

    target = date(2026, 10, 20)
    clock = _shifted_clock(target)
    calendar = KrxCalendar(
        verified_from=date(2026, 10, 1),
        verified_through=date(2026, 10, 31),
        closed_weekdays=frozenset({date(2026, 10, 9)}),
        shifted_sessions={target: clock},
        provenance="test",
    )
    day = resolve_session_day(target, overrides={}, calendar=calendar)

    assert day.kind is SessionKind.SHIFTED
    assert day.clock is clock
    assert trading_session_gate(day) == "session_shifted"
