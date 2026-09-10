"""한국투자증권 REST API 요청용 비동기 레이트 리미터."""

from __future__ import annotations

import asyncio


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
