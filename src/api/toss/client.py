"""토스증권 OpenAPI 클라이언트 (그룹별 레이트리밋 GET 요청/토큰 발급).

docs/architecture/broker_toss.md 1.4 레이트리밋 매트릭스를 그대로 옮긴 표이며,
현재는 STOCK_TRADING_TREND(프로그램/투자자/신용/대차 동향)만 실사용한다. 나머지
그룹은 향후 확장 시 매직넘버 없이 바로 쓰도록 남겨둔 것으로, 미사용 자체가
결함은 아니다(단순 데이터 테이블, 분기 로직 없음).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os

from src import settings
from src.api.kis.rate_limit import AsyncRateLimiter, get_shared_rate_limiter

logger = logging.getLogger(__name__)

_OAUTH_PATH = "/oauth2/token"

TOSS_RATE_LIMIT_GROUPS: dict[str, float] = {
    "AUTH": 5.0,
    "ACCOUNT": 1.0,
    "ASSET": 5.0,
    "STOCK": 5.0,
    "STOCK_ALL": 1.0,
    "STOCK_TRADING_TREND": 10.0,
    "MARKET_INFO": 3.0,
    "MARKET_DATA": 15.0,
    "MARKET_DATA_CHART": 20.0,
    "RANKING": 5.0,
    "MARKET_INDICATOR": 10.0,
    "MARKET_INDICATOR_CHART": 5.0,
    "ORDER": 10.0,
    "ORDER_HISTORY": 5.0,
    "ORDER_INFO": 6.0,
    "CONDITIONAL_ORDER": 5.0,
    "CONDITIONAL_ORDER_HISTORY": 10.0,
}


class TossResponseError(RuntimeError):
    """Toss가 `{"error": {...}}` 봉투로 응답한 business-level 실패."""


class TossApiClient:
    def __init__(self, app_key: str | None = None, app_secret: str | None = None, base_url: str | None = None) -> None:
        self.app_key = app_key or getattr(settings, "TOSS_APP_KEY", "") or os.getenv("TOSS_APP_KEY", "")
        self.app_secret = app_secret or getattr(settings, "TOSS_APP_SECRET", "") or os.getenv("TOSS_APP_SECRET", "")
        self.base_url = base_url or getattr(settings, "TOSS_BASE_URL", "") or "https://openapi.tossinvest.com"
        self.token: str | None = None
        self._token_lock: asyncio.Lock | None = None

    def _limiter_for(self, group: str) -> AsyncRateLimiter:
        rate = TOSS_RATE_LIMIT_GROUPS.get(group)
        if rate is None:
            raise ValueError(f"unknown Toss rate limit group: {group}")
        return get_shared_rate_limiter("toss", f"{self.app_key}:{group}", rate)

    async def ensure_token(self, session) -> str:
        """OAuth2 client_credentials 토큰 발급 (단일비행, 메모리 캐시, 24h 유효)."""
        if self.token:
            return self.token
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()
        async with self._token_lock:
            if self.token:
                return self.token
            payload = {
                "grant_type": "client_credentials",
                "client_id": self.app_key,
                "client_secret": self.app_secret,
            }
            raw = session.post(f"{self.base_url}{_OAUTH_PATH}", data=payload)
            if inspect.isawaitable(raw):
                raw = await raw
            async with raw as resp:
                body = await resp.json()
            token = str(body.get("access_token", ""))
            if not token:
                raise RuntimeError(f"Toss token issuance failed: {body}")
            self.token = token
            return token

    async def _get(self, session, path: str, group: str, params: dict | None = None, max_retries: int = 3) -> dict:
        if not self.token:
            await self.ensure_token(session)
        limiter = self._limiter_for(group)

        data: dict = {}
        for attempt in range(max_retries):
            await limiter.acquire()
            headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
            raw = session.get(f"{self.base_url}{path}", headers=headers, params=params or {})
            if inspect.isawaitable(raw):
                raw = await raw
            async with raw as resp:
                data = await resp.json()
                status = resp.status
            if status == 429 and attempt < max_retries - 1:
                logger.warning(
                    "Toss rate limit hit (429) group=%s path=%s. Retrying in 1.2s... (attempt %d/%d)",
                    group, path, attempt + 1, max_retries,
                )
                await asyncio.sleep(1.2)
                continue
            return data
        return data

    async def get_program_trades(self, session, symbol: str, count: int = 100, until: str | None = None) -> dict:
        """일별 프로그램 매매동향 (`GET /api/v1/stocks/{symbol}/program-trades`)."""
        params: dict[str, str | int] = {"count": int(count)}
        if until:
            params["until"] = until
        return await self._get(session, f"/api/v1/stocks/{symbol}/program-trades", "STOCK_TRADING_TREND", params=params)

    async def get_rankings(
        self,
        session,
        *,
        ranking_type: str,
        market_country: str = "KR",
        duration: str = "1d",
        count: int = 100,
        exclude_investment_caution: bool | None = None,
    ) -> dict:
        """시장 랭킹 조회 (`GET /api/v1/rankings`)."""
        params: dict[str, str | int] = {
            "type": ranking_type,
            "marketCountry": market_country,
            "duration": duration,
            "count": int(count),
        }
        if exclude_investment_caution is not None:
            params["excludeInvestmentCaution"] = str(bool(exclude_investment_caution)).lower()
        return await self._get(session, "/api/v1/rankings", "RANKING", params=params)
