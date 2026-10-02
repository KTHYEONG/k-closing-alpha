"""한국투자증권 REST API 요청용 비동기 레이트 리미터 + 호스트 admission 페이싱."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class AsyncRateLimiter:
    """한국투자증권 REST API 요청용 비동기 레이트 리미터 (슬라이딩 윈도우)."""

    def __init__(self, max_rate: float = 18.0, time_period: float = 1.0):
        self.max_rate = max_rate
        self.time_period = time_period
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Rate limit 초과 시 대기 후 권한 획득."""
        import time
        while True:
            async with self._lock:
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < self.time_period]
                if len(self._timestamps) < self.max_rate:
                    self._timestamps.append(now)
                    return
                sleep_time = self._timestamps[0] + self.time_period - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)


_SHARED_RATE_LIMITERS: dict[tuple[str, str, float, float], AsyncRateLimiter] = {}


def get_shared_rate_limiter(vendor: str, credential_key: str, max_rate: float, time_period: float = 1.0) -> AsyncRateLimiter:
    """프로세스 전역 단일 버킷을 (vendor, credential, rate, period) 키로 반환한다."""
    key = (vendor, credential_key, max_rate, time_period)
    limiter = _SHARED_RATE_LIMITERS.get(key)
    if limiter is None:
        limiter = AsyncRateLimiter(max_rate=max_rate, time_period=time_period)
        _SHARED_RATE_LIMITERS[key] = limiter
    return limiter


class AdmissionClass(StrEnum):
    CRITICAL = "critical"
    STANDARD = "standard"
    BULK = "bulk"


HOST_ADMISSION_MARKER: str = ".host-admission"

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class AdmissionDirNotSharedError(RuntimeError):
    """The admission directory is not the host-shared one, so pacing would be process-private."""


def _in_container() -> bool:
    return Path("/.dockerenv").exists()


def resolve_admission_dir() -> Path:
    """Return the host admission directory and enforce that it is host-shared when required.

    Returns:
        settings.BROKER_ADMISSION_DIR, or settings.KIS_TOKEN_CACHE_DIR when unset.

    Raises:
        AdmissionDirNotSharedError: Enforcement is active ("always", or "auto" inside a
            container) and the marker file is absent. A container without the shared
            mount would otherwise pace against a private directory and silently defeat
            host-wide limits; refusing to start is the fail-closed outcome.
    """
    from src import settings as _settings

    configured = _settings.BROKER_ADMISSION_DIR
    directory = Path(configured) if configured is not None else Path(_settings.KIS_TOKEN_CACHE_DIR)
    mode = str(_settings.BROKER_ADMISSION_REQUIRE_SHARED)
    enforced = mode == "always" or (mode == "auto" and _in_container())
    if enforced and not (directory / HOST_ADMISSION_MARKER).is_file():
        raise AdmissionDirNotSharedError(f"admission dir is not host-shared: {directory}")
    return directory


def host_admission_state_path(vendor: str, credential: str, scope: str | None = None) -> Path:
    """Return the protocol path admission_<vendor>_<sha12>[_<scope>].state under resolve_admission_dir().

    Raises:
        ValueError: vendor or scope violates [A-Za-z0-9_-]+, or credential is empty.
    """
    if not _NAME_RE.match(vendor):
        raise ValueError(f"invalid vendor: {vendor!r}")
    if not credential:
        raise ValueError("credential must be non-empty")
    if scope is not None and not _NAME_RE.match(scope):
        raise ValueError(f"invalid scope: {scope!r}")
    digest = hashlib.sha256(credential.encode("utf-8")).hexdigest()[:12]
    name = f"admission_{vendor}_{digest}.state" if scope is None else f"admission_{vendor}_{digest}_{scope}.state"
    return resolve_admission_dir() / name


def max_lead_for(admission_class: AdmissionClass) -> float | None:
    """Map a class to its configured reservation lead; CRITICAL maps to None (unbounded)."""
    from src import settings as _settings

    if admission_class == AdmissionClass.CRITICAL:
        return None
    if admission_class == AdmissionClass.STANDARD:
        return float(_settings.BROKER_ADMISSION_STANDARD_MAX_LEAD_SECONDS)
    return float(_settings.BROKER_ADMISSION_BULK_MAX_LEAD_SECONDS)


def _resolve_class(admission_class: AdmissionClass | str | None) -> AdmissionClass:
    if admission_class is None:
        from src import settings as _settings

        return AdmissionClass(str(_settings.BROKER_ADMISSION_CLASS))
    return admission_class if isinstance(admission_class, AdmissionClass) else AdmissionClass(str(admission_class))


class HostPacedRateLimiter:
    """호스트 공유 파일 버킷으로 app_key당 합산 TPS를 제한한다."""

    def __init__(
        self,
        state_path: Path,
        max_rate: float,
        time_period: float = 1.0,
        *,
        max_lead_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_rate <= 0 or time_period <= 0:
            raise ValueError("max_rate and time_period must be positive")
        if max_lead_seconds is not None and max_lead_seconds <= 0:
            raise ValueError("max_lead_seconds must be positive")
        self.max_rate = float(max_rate)
        self.time_period = float(time_period)
        self._state_path = Path(state_path)
        self._max_rate = float(max_rate)
        self._time_period = float(time_period)
        self._interval = self._time_period / self._max_rate
        self._max_lead_seconds = max_lead_seconds
        self._clock = clock
        self._sleep = sleep
        self._lock: asyncio.Lock | None = None
        self._lock_loop: asyncio.AbstractEventLoop | None = None

    def _loop_lock(self) -> asyncio.Lock:
        # The limiter is a process singleton but callers may run several asyncio.run() loops in sequence;
        # a contended asyncio.Lock stays bound to the loop it first waited on.
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    def _read_next_free_locked(self, fd: int) -> float:
        raw = os.read(fd, 64).decode("ascii", errors="replace").strip()
        if not raw:
            return 0.0
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                "[SYS] stage=kis_rate_limit status=STATE_RESET path=%s",
                self._state_path,
            )
            return 0.0

    def _book_unconditional(self) -> float:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._state_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                next_free = self._read_next_free_locked(fd)
                now = self._clock()
                slot = max(now, next_free)
                os.lseek(fd, 0, os.SEEK_SET)
                os.ftruncate(fd, 0)
                os.write(fd, repr(slot + self._interval).encode("ascii"))
                return slot - now
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _try_book_or_wait(self) -> tuple[bool, float]:
        """Single file poll: book when within lead, else report lead-wait without writing."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._state_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                next_free = self._read_next_free_locked(fd)
                now = self._clock()
                lead = self._max_lead_seconds
                assert lead is not None
                slot = max(now, next_free)
                if slot - now <= lead:
                    os.lseek(fd, 0, os.SEEK_SET)
                    os.ftruncate(fd, 0)
                    os.write(fd, repr(slot + self._interval).encode("ascii"))
                    return True, slot - now
                return False, max(slot - now - lead, self._interval)
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _reserve_slot(self) -> float:
        return self._book_unconditional()

    async def _acquire_bounded(self) -> None:
        assert self._max_lead_seconds is not None
        async with self._loop_lock():
            while True:
                booked, wait = self._try_book_or_wait()
                if booked:
                    if wait > 0:
                        await self._sleep(wait)
                    return
                await self._sleep(wait)

    async def acquire(self) -> None:
        """Block until this process owns the next host-wide slot for the scope."""
        if self._max_lead_seconds is not None:
            await self._acquire_bounded()
            return
        async with self._loop_lock():
            delay = self._book_unconditional()
        if delay > 0:
            await self._sleep(delay)


_HOST_RATE_LIMITERS: dict[tuple[str, float, float, str], HostPacedRateLimiter] = {}

_ADMISSION_DIR_VERIFIED = False


def _ensure_admission_dir_shared() -> None:
    global _ADMISSION_DIR_VERIFIED
    if not _ADMISSION_DIR_VERIFIED:
        resolve_admission_dir()
        _ADMISSION_DIR_VERIFIED = True


def get_host_rate_limiter(
    state_path: Path, max_rate: float, time_period: float = 1.0, *, admission_class: AdmissionClass | str | None = None
) -> HostPacedRateLimiter:
    """Return the process-singleton limiter for (path, rate, class); class None means settings.BROKER_ADMISSION_CLASS."""
    _ensure_admission_dir_shared()
    resolved = _resolve_class(admission_class)
    key = (str(Path(state_path)), float(max_rate), float(time_period), str(resolved.value))
    limiter = _HOST_RATE_LIMITERS.get(key)
    if limiter is None:
        limiter = HostPacedRateLimiter(
            Path(state_path),
            max_rate,
            time_period,
            max_lead_seconds=max_lead_for(resolved),
        )
        _HOST_RATE_LIMITERS[key] = limiter
    return limiter
