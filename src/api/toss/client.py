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
from datetime import UTC, datetime

from src.api.kis.rate_limit import HostPacedRateLimiter, get_host_rate_limiter, host_admission_state_path
from src.api.shared_token import IssuedToken, SharedTokenStore, shared_token_path
from src.config import settings

logger = logging.getLogger(__name__)

_OAUTH_PATH = "/oauth2/token"

TOSS_INVALID_TOKEN_CODES: frozenset[str] = frozenset({"invalid-token"})

_TOSS_TOKEN_EXPIRY_MARGIN_SECONDS: float = 300.0

TOSS_RETRY_AFTER_MAX_SECONDS: float = 5.0
"""Ceiling on a server-sent Retry-After wait; collect calls Toss inside the 15:20 decision window,
so an unbounded vendor hint could stall the decision snapshot."""

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
        """Create a Toss OpenAPI client.

        Args:
            app_key: OAuth client id; defaults to settings.
            app_secret: OAuth client secret; defaults to settings.
            base_url: API base URL; defaults to settings.
        """
        self.app_key = app_key or settings.TOSS_APP_KEY
        self.app_secret = app_secret or settings.TOSS_APP_SECRET
        self.base_url = base_url or settings.TOSS_BASE_URL
        self.token: str | None = None

    def _limiter_for(self, group: str) -> HostPacedRateLimiter:
        rate = TOSS_RATE_LIMIT_GROUPS.get(group)
        if rate is None:
            raise ValueError(f"unknown Toss rate limit group: {group}")
        return get_host_rate_limiter(host_admission_state_path("toss", self.app_key or "", group), rate)

    def _token_store(self) -> SharedTokenStore:
        return SharedTokenStore(
            shared_token_path("toss", self.app_key or ""),
            lock_timeout_seconds=float(settings.BROKER_ADMISSION_LOCK_TIMEOUT_SECONDS),
            expiry_margin_seconds=_TOSS_TOKEN_EXPIRY_MARGIN_SECONDS,
            clock=lambda: datetime.now(UTC),
        )

    async def _issue_token(self, session) -> IssuedToken:
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
            raise RuntimeError("Toss token issuance failed")
        expires_raw = body.get("expires_in")
        expires_in: float | None = None
        if isinstance(expires_raw, bool):
            expires_in = None
        elif isinstance(expires_raw, (int, float)):
            expires_in = float(expires_raw)
        elif isinstance(expires_raw, str) and expires_raw.strip().lstrip("+-").replace(".", "", 1).isdigit():
            try:
                expires_in = float(expires_raw.strip())
            except ValueError:
                expires_in = None
        return IssuedToken(access_token=token, expires_in_seconds=expires_in)

    async def ensure_token(self, session) -> str:
        record = await self._token_store().get_or_issue(lambda: self._issue_token(session))
        self.token = record.access_token
        return self.token

    @staticmethod
    def _is_invalid_token(status: int, data: dict) -> bool:
        if status == 401:
            return True
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("code", "")) in TOSS_INVALID_TOKEN_CODES
        return False

    @staticmethod
    def _retry_after_seconds(resp_headers: dict) -> float | None:
        for key, value in resp_headers.items():
            if str(key).lower() == "retry-after":
                try:
                    parsed = float(str(value).strip().split(",")[0])
                except (ValueError, TypeError):
                    return None
                if parsed > 0 and parsed != float("inf"):
                    return min(parsed, TOSS_RETRY_AFTER_MAX_SECONDS)
                return None
        return None

    async def _get(self, session, path: str, group: str, params: dict | None = None, max_retries: int = 3) -> dict:
        if not self.token:
            await self.ensure_token(session)
        limiter = self._limiter_for(group)

        async def _single_get() -> tuple[dict, int, dict]:
            await limiter.acquire()
            headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
            raw = session.get(f"{self.base_url}{path}", headers=headers, params=params or {})
            if inspect.isawaitable(raw):
                raw = await raw
            async with raw as resp:
                data = await resp.json()
                status = int(getattr(resp, "status", 200) or 200)
                headers_raw = getattr(resp, "headers", None)
                if isinstance(headers_raw, dict):
                    resp_headers = dict(headers_raw)
                elif hasattr(headers_raw, "items") and not type(headers_raw).__name__.endswith("Mock"):
                    try:
                        resp_headers = dict(headers_raw)
                    except Exception:
                        resp_headers = {}
                else:
                    resp_headers = {}
            return data, status, resp_headers

        refreshed = False
        data: dict = {}
        for attempt in range(max_retries):
            data, status, resp_headers = await _single_get()
            if self._is_invalid_token(status, data) and not refreshed:
                refreshed = True
                record = await self._token_store().replace_rejected(
                    self.token or "", lambda: self._issue_token(session)
                )
                self.token = record.access_token
                data, status, resp_headers = await _single_get()
                if self._is_invalid_token(status, data):
                    return data
                if status == 429 and attempt < max_retries - 1:
                    wait = self._retry_after_seconds(resp_headers) or 1.2
                    logger.warning(
                        "Toss rate limit hit (429) group=%s path=%s. Retrying in %.1fs... (attempt %d/%d)",
                        group, path, wait, attempt + 1, max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                return data
            if status == 429 and attempt < max_retries - 1:
                wait = self._retry_after_seconds(resp_headers) or 1.2
                logger.warning(
                    "Toss rate limit hit (429) group=%s path=%s. Retrying in %.1fs... (attempt %d/%d)",
                    group, path, wait, attempt + 1, max_retries,
                )
                await asyncio.sleep(wait)
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

    async def get_candles(
        self,
        session,
        symbol: str,
        *,
        interval: str = "1m",
        count: int = 200,
        before: str | None = None,
        adjusted: bool | None = None,
    ) -> dict:
        """캔들 차트 조회 (`GET /api/v1/candles`)."""
        params: dict[str, str | int] = {"symbol": symbol, "interval": interval, "count": int(count)}
        if before:
            params["before"] = before
        if adjusted is not None:
            params["adjusted"] = str(bool(adjusted)).lower()
        return await self._get(session, "/api/v1/candles", "MARKET_DATA_CHART", params=params)
