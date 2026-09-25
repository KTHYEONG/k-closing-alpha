"""Per-date KRX session resolver over the verified calendar."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from src.config.krx_calendar import KRX_CALENDAR, KrxCalendar
from src.data.capture_contracts import SessionClock


class SessionKind(StrEnum):
    """Resolved market-session status of one KST date."""

    STANDARD = "STANDARD"
    SHIFTED = "SHIFTED"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SessionDay:
    """Session status and verified clock for one date.

    Attributes:
        trading_date: KST date.
        kind: Resolved status.
        clock: Verified session clock for STANDARD/SHIFTED; None for CLOSED/UNKNOWN.
        provenance: "weekend", "krx_calendar", "operator_override", "standard" or
            "outside_verified_range".
    """

    trading_date: date
    kind: SessionKind
    clock: SessionClock | None
    provenance: str


def resolve_session_day(
    trading_date: date,
    *,
    overrides: Mapping[str, SessionClock] | None = None,
    calendar: KrxCalendar | None = None,
) -> SessionDay:
    """Resolve one date's KRX session from the verified calendar.

    Resolution order: weekend -> CLOSED; outside the verified range -> UNKNOWN;
    closed weekday -> CLOSED; operator override -> SHIFTED; calendar shifted
    session -> SHIFTED; otherwise STANDARD with SessionClock.standard. The
    resolver never calls a network oracle, so it answers before the open.

    Args:
        trading_date: KST date to resolve.
        overrides: Operator emergency clocks keyed by ISO date; None reads
            settings.COLLECTION_SESSION_OVERRIDES.
        calendar: Calendar data; None uses KRX_CALENDAR.

    Returns:
        SessionDay for trading_date.

    Raises:
        ValueError: An override targets a date the calendar declares closed, or
            an override clock's trading_date differs from its key.
    """
    if trading_date.weekday() >= 5:
        return SessionDay(trading_date=trading_date, kind=SessionKind.CLOSED, clock=None, provenance="weekend")
    resolved_calendar = calendar if calendar is not None else KRX_CALENDAR
    if overrides is None:
        from src import settings as _settings

        effective_overrides: Mapping[str, SessionClock] = _settings.COLLECTION_SESSION_OVERRIDES
    else:
        effective_overrides = overrides
    key = trading_date.isoformat()
    if key in effective_overrides:
        clock = effective_overrides[key]
        if trading_date in resolved_calendar.closed_weekdays:
            raise ValueError(f"override targets a closed date: {key}")
        if clock.trading_date != trading_date:
            raise ValueError(f"override clock trading_date differs from its key: {key}")
        return SessionDay(trading_date=trading_date, kind=SessionKind.SHIFTED, clock=clock, provenance="operator_override")
    if not (resolved_calendar.verified_from <= trading_date <= resolved_calendar.verified_through):
        return SessionDay(trading_date=trading_date, kind=SessionKind.UNKNOWN, clock=None, provenance="outside_verified_range")
    if trading_date in resolved_calendar.closed_weekdays:
        return SessionDay(trading_date=trading_date, kind=SessionKind.CLOSED, clock=None, provenance="krx_calendar")
    if trading_date in resolved_calendar.shifted_sessions:
        return SessionDay(
            trading_date=trading_date,
            kind=SessionKind.SHIFTED,
            clock=resolved_calendar.shifted_sessions[trading_date],
            provenance="krx_calendar",
        )
    return SessionDay(
        trading_date=trading_date,
        kind=SessionKind.STANDARD,
        clock=SessionClock.standard(trading_date),
        provenance="standard",
    )


def trading_session_gate(day: SessionDay) -> str | None:
    """Return the NO_TRADE reason for a date, or None when trading is allowed.

    Only a STANDARD session matches every scheduled job and the model's label
    geometry (15:20 decision, 15:30 close, next-day 09:00 open), so any other
    status is a reason not to trade.

    Returns:
        None for STANDARD; "non_trading_day" for CLOSED; "session_shifted" for
        SHIFTED; "calendar_unverified" for UNKNOWN.
    """
    if day.kind is SessionKind.STANDARD:
        return None
    if day.kind is SessionKind.CLOSED:
        return "non_trading_day"
    if day.kind is SessionKind.SHIFTED:
        return "session_shifted"
    return "calendar_unverified"
