"""Shared offsite (Google Drive via rclone) constants and helpers for backup tooling."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

OFFSITE_REMOTE_BASE: str = "gdrive:quant-lake/live/k-closing-alpha"
"""rclone root for this project's live offsite tree; every backup remote is derived from it."""

DATED_DIR_RE: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}$")
"""Matches a `YYYY-MM-DD` directory name exactly (retention and restore key on dated dirs)."""

OPS_LOG_DAY_RELDIRS: tuple[str, ...] = ("logs/audit_digests", "logs/alerts/digest", "logs/alerts/silent")
"""DATA_DIR-relative dirs of per-day `<YYYY-MM-DD>.<ext>` ops diagnostics: never offsite, pruned locally."""

OPS_LOG_MONTHLY_RELDIRS: tuple[str, ...] = ("logs/events",)
"""DATA_DIR-relative dirs of `<YYYY-MM>/<YYYY-MM-DD>.<ext>` ops diagnostics: never offsite, pruned locally."""

OPS_LOG_LOCAL_STATE_RELDIRS: tuple[str, ...] = ("logs/alerts/outbox",)
"""DATA_DIR-relative local-only state: a rebuilt host must not resend stale undelivered alerts."""

OPS_LOG_ORPHAN_RELFILES: tuple[str, ...] = ("logs/alerts/failure_sent.json",)
"""DATA_DIR-relative files left by removed alert code; no reader exists, so they never travel and are removed locally."""

_CHUNK = 1024 * 1024


def resolve_rclone_bin() -> str:
    """Resolve the rclone executable.

    systemd user units run with a PATH that omits ~/.local/bin, where rclone is installed on
    the host, so PATH lookup falls back to that fixed location.

    Returns:
        PATH-resolved rclone, else `~/.local/bin/rclone` (not checked for existence).
    """
    return shutil.which("rclone") or str(Path.home() / ".local" / "bin" / "rclone")


def sha256_file(path: Path) -> str:
    """Stream a file's SHA-256 hex digest in 1 MiB chunks (bounded memory for multi-GB segments).

    Args:
        path: Regular file to hash.

    Returns:
        Lowercase hex digest.

    Raises:
        OSError: The file cannot be read.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    """Stream a file's MD5 hex digest (the checksum Drive exposes via `rclone md5sum`; not a security hash).

    Raises:
        OSError: The file cannot be read.
    """
    digest = hashlib.md5()  # noqa: S324 - content checksum, not security
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
