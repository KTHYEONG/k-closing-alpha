"""Single-issuer OAuth token cache shared by every process on the host."""

from __future__ import annotations

from pathlib import Path

from src.api.host_broker_contract import (
    AsyncTokenStore,
    IssuedToken,
    TokenIssueThrottled,
    TokenRecord,
    TokenStoreLockTimeout,
    token_record_path,
)
from src.api.kis.rate_limit import resolve_admission_dir

SharedTokenStore = AsyncTokenStore


def shared_token_path(vendor: str, credential: str) -> Path:
    """Return token_<vendor>_<sha12>.json under resolve_admission_dir(); raise ValueError on bad vendor or empty credential."""
    return token_record_path(resolve_admission_dir(), vendor, credential)


__all__ = [
    "IssuedToken",
    "SharedTokenStore",
    "TokenIssueThrottled",
    "TokenRecord",
    "TokenStoreLockTimeout",
    "shared_token_path",
]
