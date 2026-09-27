"""Dated operational dependencies approaching expiry (credentials + calendar horizon)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from src.config.credential_expiry import (
    CREDENTIAL_EXPIRIES,
    EXPIRY_NOTICE_DAYS,
    EXPIRY_WARNING_DAYS,
    CredentialExpiry,
)
from src.config.krx_calendar import KRX_CALENDAR, KrxCalendar

CALENDAR_EXPIRY_NAME: str = "krx_calendar"
CALENDAR_RENEW_HINT: str = "KRX 다음 해 휴장일·개장시간 공지 반영(연초 첫 거래일 10:00 개장 SHIFTED 포함)"


@dataclass(frozen=True)
class ExpiryReport:
    """Dated items approaching expiry, split by urgency.

    Attributes:
        notices: Items with warning_days < days_left <= notice_days, as
            "<name>:d<days_left>:<expires_on>".
        warnings: Items with days_left <= warning_days (including negative =
            already expired), same format.
    """

    notices: tuple[str, ...]
    warnings: tuple[str, ...]


def evaluate_expiries(
    today: date,
    *,
    credentials: Sequence[CredentialExpiry] = CREDENTIAL_EXPIRIES,
    calendar: KrxCalendar = KRX_CALENDAR,
    notice_days: int = EXPIRY_NOTICE_DAYS,
    warning_days: int = EXPIRY_WARNING_DAYS,
) -> ExpiryReport:
    """Classify every dated operational dependency against today.

    The verified KRX calendar horizon is treated as one more expiring item
    named "krx_calendar" (expires_on = calendar.verified_through), because an
    unextended calendar halts decisions exactly like an expired key.

    Args:
        today: KST date of the audit.
        credentials: Registry entries.
        calendar: Verified session calendar.
        notice_days: Notice horizon in calendar days.
        warning_days: Escalation horizon in calendar days.

    Returns:
        ExpiryReport sorted by (days_left, name).
    """
    dated: list[tuple[str, date]] = [(entry.name, entry.expires_on) for entry in credentials]
    dated.append((CALENDAR_EXPIRY_NAME, calendar.verified_through))
    flagged: list[tuple[int, str, str]] = []
    for name, expires_on in dated:
        days_left = (expires_on - today).days
        if days_left > notice_days:
            continue
        flagged.append((days_left, name, f"{name}:d{days_left}:{expires_on.isoformat()}"))
    flagged.sort()
    notices = tuple(tag for days_left, _name, tag in flagged if days_left > warning_days)
    warnings = tuple(tag for days_left, _name, tag in flagged if days_left <= warning_days)
    return ExpiryReport(notices=notices, warnings=warnings)
