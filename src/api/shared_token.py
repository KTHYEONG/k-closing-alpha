"""Single-issuer OAuth token cache shared by every process on the host."""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from src.api.kis.rate_limit import _NAME_RE, resolve_admission_dir

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1
_POLL_INTERVAL_SECONDS = 0.01


@dataclass(frozen=True)
class IssuedToken:
    access_token: str
    expires_in_seconds: float | None


@dataclass(frozen=True)
class TokenRecord:
    access_token: str
    issued_at: datetime
    expires_at: datetime | None
    generation: int


class TokenStoreLockTimeout(RuntimeError):  # noqa: N818 - spec-mandated protocol name
    """The shared token lock was not obtained within the configured bound."""


def shared_token_path(vendor: str, credential: str) -> Path:
    """Return token_<vendor>_<sha12>.json under resolve_admission_dir()."""
    if not _NAME_RE.match(vendor):
        raise ValueError(f"invalid vendor: {vendor!r}")
    if not credential:
        raise ValueError("credential must be non-empty")
    digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()[:12]
    return resolve_admission_dir() / f"token_{vendor}_{digest}.json"


def _lock_path_for(path: Path) -> Path:
    return Path(str(path) + ".lock")


def _parse_record(payload: object) -> TokenRecord | None:
    if not isinstance(payload, dict):
        return None
    if set(payload) != {"schema_version", "access_token", "issued_at", "expires_at", "generation"}:
        return None
    try:
        if payload.get("schema_version") != _SCHEMA_VERSION:
            return None
        access_token = payload.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            return None
        issued_raw = payload.get("issued_at")
        expires_raw = payload.get("expires_at")
        generation = payload.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
            return None
        if not isinstance(issued_raw, str):
            return None
        issued_at = datetime.fromisoformat(issued_raw)
        if issued_at.tzinfo is None:
            return None
        expires_at: datetime | None = None
        if expires_raw is not None:
            if not isinstance(expires_raw, str):
                return None
            expires_at = datetime.fromisoformat(expires_raw)
            if expires_at.tzinfo is None:
                return None
        return TokenRecord(
            access_token=access_token,
            issued_at=issued_at,
            expires_at=expires_at,
            generation=generation,
        )
    except (ValueError, TypeError):
        return None


class SharedTokenStore:
    """Single-issuer OAuth token cache shared by every process on the host.

    Some vendors (Toss) invalidate the previous token the moment a new one is issued,
    and others (LS) are unverified. Independent issuers would therefore revoke each
    other's live tokens. All issuance goes through one file lock and is published with
    a monotonically increasing generation, so a process that sees a rejection can tell
    whether someone else already replaced the token.
    """

    def __init__(
        self,
        path: Path,
        *,
        lock_timeout_seconds: float,
        expiry_margin_seconds: float,
        clock: Callable[[], datetime],
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if lock_timeout_seconds <= 0:
            raise ValueError("lock_timeout_seconds must be positive")
        if expiry_margin_seconds < 0:
            raise ValueError("expiry_margin_seconds must be non-negative")
        self._path = Path(path)
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._expiry_margin_seconds = float(expiry_margin_seconds)
        self._clock = clock
        self._sleep = sleep

    def _is_usable(self, record: TokenRecord, now: datetime) -> bool:
        if record.expires_at is None:
            return True
        return now < record.expires_at - timedelta(seconds=self._expiry_margin_seconds)

    def read(self) -> TokenRecord | None:
        """Return the stored record, or None when absent or schema-invalid (logged, never raised)."""
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning(
                "[SYS] stage=shared_token status=UNREADABLE path=%s reason=%s",
                self._path.name,
                type(exc).__name__,
            )
            return None
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("[SYS] stage=shared_token status=SCHEMA_INVALID path=%s", self._path.name)
            return None
        record = _parse_record(payload)
        if record is None:
            logger.warning("[SYS] stage=shared_token status=SCHEMA_INVALID path=%s", self._path.name)
            return None
        return record

    def _publish(self, record: TokenRecord) -> None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "access_token": record.access_token,
            "issued_at": record.issued_at.isoformat(),
            "expires_at": record.expires_at.isoformat() if record.expires_at is not None else None,
            "generation": record.generation,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self._path.parent), prefix=".token_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self._path)
            os.chmod(self._path, 0o600)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
        logger.info(
            "[SYS] stage=shared_token status=PUBLISHED path=%s generation=%d",
            self._path.name,
            record.generation,
        )

    def _build_record(self, issued: IssuedToken, previous: TokenRecord | None) -> TokenRecord:
        now = self._clock()
        expires_at: datetime | None = None
        if issued.expires_in_seconds is not None:
            expires_at = now + timedelta(seconds=float(issued.expires_in_seconds))
        generation = previous.generation + 1 if previous is not None else 1
        return TokenRecord(
            access_token=issued.access_token,
            issued_at=now,
            expires_at=expires_at,
            generation=generation,
        )

    async def _hold_lock(self) -> int:
        lock_path = _lock_path_for(self._path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.monotonic() + self._lock_timeout_seconds
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return fd
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TokenStoreLockTimeout(f"token lock not acquired: {lock_path}") from None
                    await self._sleep(_POLL_INTERVAL_SECONDS)
        except BaseException:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            raise

    async def get_or_issue(self, issue: Callable[[], Awaitable[IssuedToken]]) -> TokenRecord:
        """Return a usable stored token, issuing exactly once host-wide when none is usable.

        Raises:
            TokenStoreLockTimeout: Lock not acquired within lock_timeout_seconds.
            Exception: Any error from `issue` propagates; the store is left unchanged.
        """
        now = self._clock()
        cached = self.read()
        if cached is not None and self._is_usable(cached, now):
            return cached
        fd = await self._hold_lock()
        try:
            now = self._clock()
            cached = self.read()
            if cached is not None and self._is_usable(cached, now):
                return cached
            issued = await issue()
            record = self._build_record(issued, cached)
            self._publish(record)
            return record
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    async def replace_rejected(
        self, rejected_token: str, issue: Callable[[], Awaitable[IssuedToken]]
    ) -> TokenRecord:
        """Recover from a vendor auth rejection without refresh ping-pong.

        Under the lock, if the stored token differs from rejected_token and is still
        usable it is returned unchanged (another process already rotated it); otherwise
        a new token is issued and stored with generation + 1.

        Raises:
            TokenStoreLockTimeout: Lock not acquired within the bound.
        """
        fd = await self._hold_lock()
        try:
            cached = self.read()
            if (
                cached is not None
                and cached.access_token != rejected_token
                and self._is_usable(cached, self._clock())
            ):
                return cached
            issued = await issue()
            record = self._build_record(issued, cached)
            self._publish(record)
            return record
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


__all__ = [
    "IssuedToken",
    "SharedTokenStore",
    "TokenRecord",
    "TokenStoreLockTimeout",
    "shared_token_path",
]
