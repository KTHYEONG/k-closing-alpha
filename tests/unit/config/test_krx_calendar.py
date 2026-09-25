"""Shipped KRX calendar consistency guards."""

from __future__ import annotations

from datetime import date

import pytest

from src.config.krx_calendar import KRX_CALENDAR, KrxCalendar


def test_shipped_calendar_is_internally_consistent() -> None:
    assert KRX_CALENDAR.verified_from == date(2026, 9, 25)
    assert KRX_CALENDAR.verified_through == date(2026, 12, 31)
    for required in (date(2026, 10, 5), date(2026, 10, 9), date(2026, 12, 25), date(2026, 12, 31)):
        assert required in KRX_CALENDAR.closed_weekdays
    for closed_day in KRX_CALENDAR.closed_weekdays:
        assert closed_day.weekday() < 5
        assert KRX_CALENDAR.verified_from <= closed_day <= KRX_CALENDAR.verified_through
    for key, clock in KRX_CALENDAR.shifted_sessions.items():
        assert key.weekday() < 5
        assert KRX_CALENDAR.verified_from <= key <= KRX_CALENDAR.verified_through
        assert clock.trading_date == key
    assert KRX_CALENDAR.provenance.strip()


def _base_kwargs(**overrides):
    kwargs: dict = {
        "verified_from": date(2026, 10, 1),
        "verified_through": date(2026, 10, 31),
        "closed_weekdays": frozenset(),
        "shifted_sessions": {},
        "provenance": "test",
    }
    kwargs.update(overrides)
    return kwargs


def test_calendar_rejects_closed_weekend_date() -> None:
    with pytest.raises(ValueError, match="weekend"):
        KrxCalendar(**_base_kwargs(closed_weekdays=frozenset({date(2026, 10, 10)})))


def test_calendar_rejects_shifted_weekend_date() -> None:
    from tests.unit.data.test_session_calendar import _shifted_clock

    with pytest.raises(ValueError, match="weekend"):
        KrxCalendar(**_base_kwargs(shifted_sessions={date(2026, 10, 10): _shifted_clock(date(2026, 10, 10))}))


def test_calendar_rejects_date_in_both_sets() -> None:
    from tests.unit.data.test_session_calendar import _shifted_clock

    target = date(2026, 10, 20)
    with pytest.raises(ValueError, match="both closed and shifted"):
        KrxCalendar(
            **_base_kwargs(
                closed_weekdays=frozenset({target}),
                shifted_sessions={target: _shifted_clock(target)},
            )
        )


def test_calendar_rejects_shifted_clock_keyed_to_another_date() -> None:
    from tests.unit.data.test_session_calendar import _shifted_clock

    with pytest.raises(ValueError, match="differs from its key"):
        KrxCalendar(**_base_kwargs(shifted_sessions={date(2026, 10, 20): _shifted_clock(date(2026, 10, 21))}))


def test_calendar_rejects_date_outside_verified_range() -> None:
    with pytest.raises(ValueError, match="outside verified range"):
        KrxCalendar(**_base_kwargs(closed_weekdays=frozenset({date(2026, 11, 2)})))


def test_calendar_rejects_shifted_date_outside_verified_range() -> None:
    from tests.unit.data.test_session_calendar import _shifted_clock

    target = date(2026, 11, 2)
    with pytest.raises(ValueError, match="outside verified range"):
        KrxCalendar(**_base_kwargs(shifted_sessions={target: _shifted_clock(target)}))
