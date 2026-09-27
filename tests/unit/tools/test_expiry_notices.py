"""Expiry classification guards (credentials + KRX calendar horizon)."""

from __future__ import annotations

from datetime import date

from src.config.credential_expiry import CredentialExpiry
from src.config.krx_calendar import KrxCalendar


def _credential(name: str, expires_on: date) -> CredentialExpiry:
    return CredentialExpiry(name=name, expires_on=expires_on)


def _calendar(through: date) -> KrxCalendar:
    return KrxCalendar(
        verified_from=date(2026, 9, 25),
        verified_through=through,
        closed_weekdays=frozenset(),
        shifted_sessions={},
        provenance="test",
    )


def test_far_items_omitted() -> None:
    from src.tools.expiry_notices import evaluate_expiries

    today = date(2026, 9, 29)
    report = evaluate_expiries(today, credentials=(_credential("KIS_DATA_1", date(2026, 11, 28)),))

    assert report.notices == () and report.warnings == ()


def test_notice_window_inclusive_at_30_days() -> None:
    from src.tools.expiry_notices import evaluate_expiries

    today = date(2026, 9, 29)
    report = evaluate_expiries(
        today,
        credentials=(_credential("KIS_DATA_1", date(2026, 10, 29)),),
        calendar=_calendar(date(2027, 9, 29)),
    )

    assert report.notices == ("KIS_DATA_1:d30:2026-10-29",)
    assert report.warnings == ()


def test_warning_window_inclusive_at_7_days() -> None:
    from src.tools.expiry_notices import evaluate_expiries

    today = date(2026, 9, 29)
    report = evaluate_expiries(
        today,
        credentials=(_credential("KIS_DATA_1", date(2026, 10, 6)),),
        calendar=_calendar(date(2027, 9, 29)),
    )

    assert report.warnings == ("KIS_DATA_1:d7:2026-10-06",)
    assert report.notices == ()


def test_expired_item_is_warning_with_negative_days() -> None:
    from src.tools.expiry_notices import evaluate_expiries

    today = date(2026, 9, 29)
    report = evaluate_expiries(
        today,
        credentials=(_credential("KIS_DATA_1", date(2026, 9, 26)),),
        calendar=_calendar(date(2027, 9, 29)),
    )

    assert report.warnings == ("KIS_DATA_1:d-3:2026-09-26",)
    assert report.notices == ()


def test_calendar_horizon_treated_as_expiry() -> None:
    from src.config.krx_calendar import KRX_CALENDAR
    from src.tools.expiry_notices import evaluate_expiries

    today = date(2026, 12, 11)
    assert (KRX_CALENDAR.verified_through - today).days == 20
    report = evaluate_expiries(today, credentials=())

    assert "krx_calendar:d20:2026-12-31" in report.notices
    assert report.warnings == ()


def test_sorted_by_urgency_then_name() -> None:
    from src.tools.expiry_notices import evaluate_expiries

    today = date(2026, 9, 29)
    report = evaluate_expiries(
        today,
        credentials=(
            _credential("KIS_B", date(2026, 10, 9)),
            _credential("KIS_A", date(2026, 10, 9)),
            _credential("KIS_C", date(2026, 10, 4)),
        ),
        calendar=_calendar(date(2027, 9, 29)),
    )

    assert report.warnings == ("KIS_C:d5:2026-10-04",)
    assert report.notices == ("KIS_A:d10:2026-10-09", "KIS_B:d10:2026-10-09")


def test_empty_registry_only_calendar() -> None:
    from src.tools.expiry_notices import evaluate_expiries

    report = evaluate_expiries(date(2026, 9, 29), credentials=(), calendar=_calendar(date(2027, 9, 29)))

    assert report.notices == () and report.warnings == ()
