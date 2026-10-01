"""Rotate the shared Kiwoom token phase so its expiry avoids the decision window."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

import aiohttp

from src.api.kiwoom.client import KiwoomApiClient
from src.config import settings
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)

_SEOUL = ZoneInfo("Asia/Seoul")

RotationStatus = Literal["UNCHANGED", "ROTATED", "DRY_RUN"]


@dataclass(frozen=True)
class RotationOutcome:
    """Result of one token-phase rotation run."""

    status: RotationStatus
    previous_expiry: datetime
    new_expiry: datetime


def _window_to_seconds(spec: str) -> tuple[int, int]:
    start_raw, end_raw = str(spec).split("-", 1)
    start_h, start_m = start_raw.split(":", 1)
    end_h, end_m = end_raw.split(":", 1)
    return (int(start_h) * 3600 + int(start_m) * 60, int(end_h) * 3600 + int(end_m) * 60)


def expiry_in_protected_window(expires_at: datetime, windows: Sequence[str]) -> bool:
    """True when expires_at (KST wall time) falls inside any protected HH:MM-HH:MM window."""
    local = expires_at.astimezone(_SEOUL) if expires_at.tzinfo is not None else expires_at.replace(tzinfo=_SEOUL)
    second = local.hour * 3600 + local.minute * 60 + local.second
    for spec in windows:
        start, end = _window_to_seconds(spec)
        if start <= second < end:
            return True
    return False


async def rotate_kiwoom_token(
    client: KiwoomApiClient, session: Any, *, windows: Sequence[str], dry_run: bool
) -> RotationOutcome:
    """Move the shared Kiwoom token's expiry out of protected windows.

    Issues (receives) the live token; if its expiry is already outside every protected
    window, does nothing. Otherwise revokes it and issues a fresh one, then re-checks.

    Returns:
        RotationOutcome(status in {"UNCHANGED", "ROTATED", "DRY_RUN"}, previous_expiry,
        new_expiry).

    Raises:
        RuntimeError: Revocation or reissuance failed, or the reissued token's expiry is
            still inside a protected window. The CLI exits non-zero so OnFailure alerts.
    """
    issued = await client.issue_token(session)
    previous = issued.expires_at
    if not expiry_in_protected_window(previous, windows):
        outcome = RotationOutcome(status="UNCHANGED", previous_expiry=previous, new_expiry=previous)
        logger.info(
            "[SYS] stage=kiwoom_token_rotate status=%s prev_expiry=%s new_expiry=%s",
            outcome.status,
            outcome.previous_expiry.isoformat(),
            outcome.new_expiry.isoformat(),
        )
        return outcome
    if dry_run:
        outcome = RotationOutcome(status="DRY_RUN", previous_expiry=previous, new_expiry=previous)
        logger.info(
            "[SYS] stage=kiwoom_token_rotate status=%s prev_expiry=%s new_expiry=%s",
            outcome.status,
            outcome.previous_expiry.isoformat(),
            outcome.new_expiry.isoformat(),
        )
        return outcome
    await client.revoke_token(session, issued.token)
    client.reset_token()
    reissued = await client.issue_token(session)
    if expiry_in_protected_window(reissued.expires_at, windows):
        raise RuntimeError("Kiwoom reissued token expiry still inside a protected window")
    outcome = RotationOutcome(status="ROTATED", previous_expiry=previous, new_expiry=reissued.expires_at)
    logger.info(
        "[SYS] stage=kiwoom_token_rotate status=%s prev_expiry=%s new_expiry=%s",
        outcome.status,
        outcome.previous_expiry.isoformat(),
        outcome.new_expiry.isoformat(),
    )
    return outcome


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"invalid boolean value: {raw!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the systemd rotation timer."""
    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    parser = argparse.ArgumentParser(description="Rotate the shared Kiwoom token phase out of protected windows.")
    parser.add_argument("--dry-run", dest="dry_run", nargs="?", const=True, default=False, type=_parse_bool)
    args = parser.parse_args(list(argv) if argv is not None else None)

    async def _run() -> RotationOutcome:
        timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            client = KiwoomApiClient()
            return await rotate_kiwoom_token(
                client, session, windows=tuple(settings.KIWOOM_TOKEN_PROTECTED_WINDOWS), dry_run=bool(args.dry_run)
            )

    try:
        asyncio.run(_run())
    except RuntimeError as exc:
        logger.error("[SYS] stage=kiwoom_token_rotate status=FAILED reason=%s", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
