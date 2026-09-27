"""Operator-maintained registry of dated credentials (metadata only, no secrets)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

EXPIRY_NOTICE_DAYS: int = 30
EXPIRY_WARNING_DAYS: int = 7


@dataclass(frozen=True)
class CredentialExpiry:
    """One dated credential the operator must renew by hand.

    Only non-secret metadata lives here (slot name and date), never key
    material, so the registry is safe to commit and review.

    Attributes:
        name: Stable slot label shown in mail (e.g. "KIS_DATA_1", "KIS_APP_KEY", "KRX_OPENAPI_KEY").
        expires_on: Last valid KST date as printed by the issuer.
        renew_hint: Short Korean hint of where to renew (e.g. "KIS Developers 서비스 연장").
    """

    name: str
    expires_on: date
    renew_hint: str = ""


def validate_credential_expiries(entries: Sequence[CredentialExpiry]) -> None:
    """Reject an expiry registry with blank/duplicate names or an inverted horizon.

    Args:
        entries: Registry entries to check.

    Raises:
        ValueError: A name is empty, a name repeats, or EXPIRY_WARNING_DAYS
            is not below EXPIRY_NOTICE_DAYS.
    """
    names = [entry.name for entry in entries]
    if any(not name for name in names):
        raise ValueError("credential expiry names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate credential expiry names: {sorted(names)}")
    if not EXPIRY_WARNING_DAYS < EXPIRY_NOTICE_DAYS:
        raise ValueError(
            f"EXPIRY_WARNING_DAYS ({EXPIRY_WARNING_DAYS}) must be below "
            f"EXPIRY_NOTICE_DAYS ({EXPIRY_NOTICE_DAYS})"
        )


# 갱신할 때마다 이 목록을 업데이트한다. 날짜는 발급기관이 통보한 마지막 유효일이다.
# KIS 앱키는 전부 발급일 기준 1년 유효(사용자 확인 2026-09-27).
_KIS_RENEW_HINT = "KIS Developers 서비스 연장"
CREDENTIAL_EXPIRIES: tuple[CredentialExpiry, ...] = (
    CredentialExpiry(name="KIS_APP_KEY", expires_on=date(2027, 3, 11), renew_hint=_KIS_RENEW_HINT),
    CredentialExpiry(name="KIS_DATA_1", expires_on=date(2026, 12, 4), renew_hint=_KIS_RENEW_HINT),
    CredentialExpiry(name="KIS_DATA_2", expires_on=date(2027, 9, 15), renew_hint=_KIS_RENEW_HINT),
    CredentialExpiry(name="KIS_DATA_3", expires_on=date(2027, 9, 15), renew_hint=_KIS_RENEW_HINT),
    CredentialExpiry(name="KIS_DATA_4", expires_on=date(2027, 9, 15), renew_hint=_KIS_RENEW_HINT),
    CredentialExpiry(name="KIS_DATA_5", expires_on=date(2027, 9, 15), renew_hint=_KIS_RENEW_HINT),
    CredentialExpiry(name="KRX_OPENAPI_KEY", expires_on=date(2027, 1, 18), renew_hint="공공데이터포털 KRX Open API 활용신청 연장"),
)

validate_credential_expiries(CREDENTIAL_EXPIRIES)
