"""한국투자증권 REST API 요청용 비동기 레이트 리미터."""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import time
from collections.abc import Awaitable, Callable
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


class HostPacedRateLimiter:
    """호스트 공유 파일 버킷으로 app_key당 합산 TPS를 제한한다."""

    def __init__(
        self,
        state_path: Path,
        max_rate: float,
        time_period: float = 1.0,
        *,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if max_rate <= 0 or time_period <= 0:
            raise ValueError("max_rate and time_period must be positive")
        self.max_rate = float(max_rate)
        self.time_period = float(time_period)
        self._state_path = Path(state_path)
        self._max_rate = float(max_rate)
        self._time_period = float(time_period)
        self._interval = self._time_period / self._max_rate
        self._clock = clock
        self._sleep = sleep
        self._lock: asyncio.Lock | None = None

    def _reserve_slot(self) -> float:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._state_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                raw = os.read(fd, 64).decode("ascii", errors="replace").strip()
                if not raw:
                    next_free = 0.0
                else:
                    try:
                        next_free = float(raw)
                    except ValueError:
                        logger.warning(
                            "[SYS] stage=kis_rate_limit status=STATE_RESET path=%s",
                            self._state_path,
                        )
                        next_free = 0.0
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

    async def acquire(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            delay = self._reserve_slot()
        if delay > 0:
            await self._sleep(delay)


_HOST_RATE_LIMITERS: dict[tuple[str, float], HostPacedRateLimiter] = {}


def get_host_rate_limiter(state_path: Path, max_rate: float) -> HostPacedRateLimiter:
    """경로·레이트별 프로세스 단일 인스턴스를 반환한다."""
    key = (str(Path(state_path)), float(max_rate))
    limiter = _HOST_RATE_LIMITERS.get(key)
    if limiter is None:
        limiter = HostPacedRateLimiter(Path(state_path), max_rate)
        _HOST_RATE_LIMITERS[key] = limiter
    return limiter
