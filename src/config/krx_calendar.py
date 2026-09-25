"""Verified KRX session calendar for a bounded date range."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.data.capture_contracts import SessionClock

_SEOUL = ZoneInfo("Asia/Seoul")


def _at(trading_day: date, hhmmss: str) -> datetime:
    return datetime(
        trading_day.year,
        trading_day.month,
        trading_day.day,
        int(hhmmss[0:2]),
        int(hhmmss[2:4]),
        int(hhmmss[4:6]),
        tzinfo=_SEOUL,
    )


@dataclass(frozen=True)
class KrxCalendar:
    """Verified KRX session calendar for a bounded date range.

    Attributes:
        verified_from: First date whose status is verified.
        verified_through: Last date whose status is verified; later dates are UNKNOWN.
        closed_weekdays: Weekday market closures (public holidays, year-end close).
        shifted_sessions: Verified non-standard session clocks keyed by date.
        provenance: Source reference of the data (KRX notice identifiers/dates).

    Raises:
        ValueError: Any closed or shifted date is a weekend, lies outside the
            verified range, appears in both sets, or a shifted clock's
            trading_date differs from its key.
    """

    verified_from: date
    verified_through: date
    closed_weekdays: frozenset[date] = field(default_factory=frozenset)
    shifted_sessions: Mapping[date, SessionClock] = field(default_factory=dict)
    provenance: str = ""

    def __post_init__(self) -> None:
        closed = frozenset(self.closed_weekdays)
        object.__setattr__(self, "closed_weekdays", closed)
        shifted = dict(self.shifted_sessions)
        object.__setattr__(self, "shifted_sessions", shifted)
        for closed_day in closed:
            if closed_day.weekday() >= 5:
                raise ValueError(f"closed date is a weekend: {closed_day.isoformat()}")
            if not (self.verified_from <= closed_day <= self.verified_through):
                raise ValueError(f"closed date outside verified range: {closed_day.isoformat()}")
            if closed_day in shifted:
                raise ValueError(f"date is both closed and shifted: {closed_day.isoformat()}")
        for key, clock in shifted.items():
            if key.weekday() >= 5:
                raise ValueError(f"shifted date is a weekend: {key.isoformat()}")
            if not (self.verified_from <= key <= self.verified_through):
                raise ValueError(f"shifted date outside verified range: {key.isoformat()}")
            if clock.trading_date != key:
                raise ValueError(f"shifted clock trading_date differs from its key: {key.isoformat()}")


# 매년 12월 KRX 휴장일/개장시간 공지를 반영해 다음 해를 연장해야 하며, 연장하지 않으면 해당 일자는 UNKNOWN(매매 중단)으로 해석된다.
KRX_CALENDAR = KrxCalendar(
    verified_from=date(2026, 9, 25),
    verified_through=date(2026, 12, 31),
    closed_weekdays=frozenset(
        {
            date(2026, 9, 25),
            date(2026, 10, 5),
            date(2026, 10, 9),
            date(2026, 12, 25),
            date(2026, 12, 31),
        }
    ),
    shifted_sessions={
        date(2026, 11, 19): SessionClock(
            trading_date=date(2026, 11, 19),
            open_at=_at(date(2026, 11, 19), "100000"),
            close_at=_at(date(2026, 11, 19), "163000"),
            close_confirmation_deadline=_at(date(2026, 11, 19), "163300"),
            provenance="csat_delayed_open",
        ),
    },
    provenance="KRX 2026 market-closure and CSAT-day delayed-open notices, owner-verified 2026-09-24",
)
