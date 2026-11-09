"""Operator-maintained registry of dated credentials (metadata only, no secrets)."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

EXPIRY_NOTICE_DAYS: int = 30
EXPIRY_WARNING_DAYS: int = 7

REGISTRY_SCHEMA: Final[str] = "credential-registry.v1"
REGISTRY_CONSUMER: Final[str] = "kca"
_REGISTRY_STATUSES: Final[frozenset[str]] = frozenset({"active", "retiring", "no_expiry"})


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


class CredentialRegistryError(ValueError):
    """The shared credential registry is missing, unreadable, or malformed."""


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


def load_credential_expiries(path: Path) -> tuple[CredentialExpiry, ...]:
    """Read the dated credentials kca must watch from the shared registry.

    Keeps only active entries that list kca as a consumer and carry an expiry date.

    Args:
        path: Registry TOML written by the api-manager credential tooling.

    Returns:
        Validated entries in registry order.

    Raises:
        CredentialRegistryError: The file is unreadable, not TOML, has a foreign schema,
            contains a malformed entry, or repeats a name. Messages carry no file contents.
    """
    try:
        with path.open("rb") as handle:
            document = tomllib.load(handle)
    except OSError as exc:
        raise CredentialRegistryError("registry file unreadable") from exc
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise CredentialRegistryError("registry is not valid TOML") from exc
    if document.get("schema") != REGISTRY_SCHEMA:
        raise CredentialRegistryError(f"registry schema is not {REGISTRY_SCHEMA}")
    raw_entries = document.get("credential", [])
    if not isinstance(raw_entries, list):
        raise CredentialRegistryError("registry credential list is malformed")
    parsed = [_parse_entry(raw) for raw in raw_entries]
    selected = tuple(
        CredentialExpiry(name=name, expires_on=expires_on, renew_hint=hint)
        for name, status, consumers, expires_on, hint in parsed
        if status == "active" and REGISTRY_CONSUMER in consumers and expires_on is not None
    )
    try:
        validate_credential_expiries(selected)
    except ValueError as exc:
        raise CredentialRegistryError("registry entries failed validation") from exc
    return selected


def _parse_entry(raw: Any) -> tuple[str, str, tuple[str, ...], date | None, str]:
    if not isinstance(raw, dict):
        raise CredentialRegistryError("registry entry is malformed")
    name, status, consumers = raw.get("name"), raw.get("status"), raw.get("consumers")
    expires_on, hint = raw.get("expires_on"), raw.get("renew_hint", "")
    if not isinstance(name, str) or not name:
        raise CredentialRegistryError("registry entry name is malformed")
    if status not in _REGISTRY_STATUSES:
        raise CredentialRegistryError("registry entry status is malformed")
    if not isinstance(consumers, list) or not all(isinstance(item, str) for item in consumers):
        raise CredentialRegistryError("registry entry consumers are malformed")
    if expires_on is not None and (not isinstance(expires_on, date) or isinstance(expires_on, datetime)):
        raise CredentialRegistryError("registry entry expires_on is malformed")
    if not isinstance(hint, str):
        raise CredentialRegistryError("registry entry renew_hint is malformed")
    return name, status, tuple(consumers), expires_on, hint
