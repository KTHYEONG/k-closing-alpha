"""Credential-expiry registry guards (validator and shared-registry reader)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import src.config.credential_expiry as registry
from src.config.credential_expiry import (
    CredentialExpiry,
    CredentialRegistryError,
    load_credential_expiries,
    validate_credential_expiries,
)

_HEADER = 'schema = "credential-registry.v1"\n'


def _write(tmp_path: Path, body: str, name: str = "credential-registry.toml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


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


def test_registry_rejects_inverted_horizon(monkeypatch) -> None:
    monkeypatch.setattr(registry, "EXPIRY_WARNING_DAYS", 30)
    monkeypatch.setattr(registry, "EXPIRY_NOTICE_DAYS", 30)

    with pytest.raises(ValueError, match="must be below"):
        registry.validate_credential_expiries(())


def test_registry_loader_keeps_only_active_kca_dated_entries(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _HEADER
        + """
[[credential]]
name = "KIS_DATA_1"
status = "active"
confidence = "confirmed"
consumers = ["kca", "kse"]
expires_on = 2027-01-02
renew_hint = "hint-kis"

[[credential]]
name = "KRX_OPENAPI_KEY"
status = "active"
confidence = "confirmed"
consumers = ["kca"]
expires_on = 2027-01-18

[[credential]]
name = "KSE_ONLY"
status = "active"
confidence = "confirmed"
consumers = ["kse"]
expires_on = 2027-01-03

[[credential]]
name = "RETIRING_KCA"
status = "retiring"
confidence = "confirmed"
consumers = ["kca"]
expires_on = 2027-01-04

[[credential]]
name = "NO_EXPIRY_KCA"
status = "no_expiry"
confidence = "confirmed"
consumers = ["kca"]

[[credential]]
name = "UNDATED_KCA"
status = "active"
confidence = "unknown"
consumers = ["kca"]
""",
    )

    entries = load_credential_expiries(path)

    assert entries == (
        CredentialExpiry(name="KIS_DATA_1", expires_on=date(2027, 1, 2), renew_hint="hint-kis"),
        CredentialExpiry(name="KRX_OPENAPI_KEY", expires_on=date(2027, 1, 18), renew_hint=""),
    )


def test_wrong_schema_or_malformed_entry_raises_without_contents(tmp_path: Path) -> None:
    marker = "MARKER_SECRET_VALUE"
    wrong_schema = _write(tmp_path, 'schema = "credential-registry.v2"\n', "wrong.toml")
    missing_schema = _write(tmp_path, '[[credential]]\nname = "KIS_DATA_1"\n', "missing.toml")
    malformed = _write(
        tmp_path,
        _HEADER
        + f"""
[[credential]]
name = "KIS_DATA_1"
status = "active"
consumers = "{marker}"
expires_on = 2027-01-02
""",
        "malformed.toml",
    )

    for path in (wrong_schema, missing_schema, malformed):
        with pytest.raises(CredentialRegistryError) as excinfo:
            load_credential_expiries(path)
        assert marker not in str(excinfo.value)


@pytest.mark.parametrize(
    "entries_body",
    [
        'credential = "not-a-list"\n',
        'credential = ["bare-string"]\n',
        '[[credential]]\nname = ""\nstatus = "active"\nconsumers = ["kca"]\n',
        '[[credential]]\nname = "KIS_DATA_1"\nstatus = "paused"\nconsumers = ["kca"]\n',
        '[[credential]]\nname = "KIS_DATA_1"\nstatus = "active"\nconsumers = [1]\n',
        '[[credential]]\nname = "KIS_DATA_1"\nstatus = "active"\nconsumers = ["kca"]\nexpires_on = 2027-01-02T00:00:00\n',
        '[[credential]]\nname = "KIS_DATA_1"\nstatus = "active"\nconsumers = ["kca"]\nexpires_on = 2027-01-02\nrenew_hint = 7\n',
    ],
)
def test_malformed_registry_shapes_raise_registry_error(tmp_path: Path, entries_body: str) -> None:
    path = _write(tmp_path, _HEADER + entries_body)

    with pytest.raises(CredentialRegistryError):
        load_credential_expiries(path)


def test_undecodable_registry_raises_registry_error(tmp_path: Path) -> None:
    path = tmp_path / "credential-registry.toml"
    path.write_bytes(b'schema = "credential-registry.v1"\nname = "\xff"\n')

    with pytest.raises(CredentialRegistryError, match="not valid TOML"):
        load_credential_expiries(path)


def test_duplicate_kca_names_raise_registry_error(tmp_path: Path) -> None:
    entry = """
[[credential]]
name = "KIS_DATA_1"
status = "active"
consumers = ["kca"]
expires_on = 2027-01-02
"""
    path = _write(tmp_path, _HEADER + entry + entry)

    with pytest.raises(CredentialRegistryError, match="failed validation"):
        load_credential_expiries(path)


def test_missing_file_raises_registry_error(tmp_path: Path) -> None:
    with pytest.raises(CredentialRegistryError):
        load_credential_expiries(tmp_path / "absent.toml")


def test_no_dates_live_in_source() -> None:
    assert not hasattr(registry, "CREDENTIAL_EXPIRIES")
