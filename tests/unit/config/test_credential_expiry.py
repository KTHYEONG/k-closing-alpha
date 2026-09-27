"""Shipped credential-expiry registry guards."""

from __future__ import annotations

from datetime import date

import pytest

from src.config.credential_expiry import (
    CREDENTIAL_EXPIRIES,
    CredentialExpiry,
    validate_credential_expiries,
)


def test_registry_rejects_duplicate_names() -> None:
    entries = (
        CredentialExpiry(name="KIS_DATA_1", expires_on=date(2027, 1, 1)),
        CredentialExpiry(name="KIS_DATA_1", expires_on=date(2027, 2, 1)),
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_credential_expiries(entries)


def test_registry_rejects_blank_names() -> None:
    entries = (CredentialExpiry(name="", expires_on=date(2027, 1, 1)),)

    with pytest.raises(ValueError, match="non-empty"):
        validate_credential_expiries(entries)


def test_shipped_registry_is_valid() -> None:
    validate_credential_expiries(CREDENTIAL_EXPIRIES)

    assert len({entry.name for entry in CREDENTIAL_EXPIRIES}) == len(CREDENTIAL_EXPIRIES)


def test_registry_rejects_inverted_horizon(monkeypatch) -> None:
    import src.config.credential_expiry as registry

    monkeypatch.setattr(registry, "EXPIRY_WARNING_DAYS", 30)
    monkeypatch.setattr(registry, "EXPIRY_NOTICE_DAYS", 30)

    with pytest.raises(ValueError, match="must be below"):
        registry.validate_credential_expiries(())
