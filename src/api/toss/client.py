"""토스증권 OpenAPI 클라이언트 (그룹별 레이트리밋 GET 요청/토큰 발급).

docs/architecture/broker_toss.md 1.4 레이트리밋 매트릭스를 그대로 옮긴 표이며,
현재는 STOCK_TRADING_TREND(프로그램/투자자/신용/대차 동향)만 실사용한다. 나머지
그룹은 향후 확장 시 매직넘버 없이 바로 쓰도록 남겨둔 것으로, 미사용 자체가
결함은 아니다(단순 데이터 테이블, 분기 로직 없음).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from src.api._common import parse_expires_in
from src.api._transport import RetryPolicy, send_with_auth_retry
from src.api.kis.rate_limit import HostPacedRateLimiter, get_host_rate_limiter, host_admission_state_path
from src.api.shared_token import IssuedToken, SharedTokenStore, shared_token_path
from src.config import settings

if TYPE_CHECKING:
    import aiohttp

    from src.data.capture_contracts import BrokerPayload

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
        self._rate_limit_max_retries: int = int(settings.TOSS_RATE_LIMIT_MAX_RETRIES)
        self._rate_limit_backoff: float = float(settings.TOSS_RATE_LIMIT_BACKOFF_SECONDS)

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

    async def _issue_token(self, session: aiohttp.ClientSession) -> IssuedToken:
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.app_key,
            "client_secret": self.app_secret,
        }
        async with session.post(f"{self.base_url}{_OAUTH_PATH}", data=payload) as resp:
            body = await resp.json()
        token = str(body.get("access_token", ""))
        if not token:
            raise RuntimeError("Toss token issuance failed")
        expires_in = parse_expires_in(body.get("expires_in"))
        return IssuedToken(access_token=token, expires_in_seconds=expires_in)

    async def ensure_token(self, session: aiohttp.ClientSession) -> str:
        record = await self._token_store().get_or_issue(lambda: self._issue_token(session))
        self.token = record.access_token
        return self.token

    @staticmethod
    def _is_invalid_token(status: int, data: dict[str, Any]) -> bool:
        if status == 401:
            return True
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("code", "")) in TOSS_INVALID_TOKEN_CODES
        return False

    @staticmethod
    def _retry_after_seconds(resp_headers: dict[str, str]) -> float | None:
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

    async def _refresh_rejected_token(self, session: aiohttp.ClientSession, rejected_token: str) -> None:
        """Replace the rejected token through the host-shared store (compare-and-swap on the sent token).

        Toss invalidates the previous token the moment a new one is issued. A redundant issuance would therefore
        revoke the token a peer just obtained, so the token that was actually sent decides whether to adopt or
        issue.

        Raises:
            TokenStoreLockTimeout: Store lock not acquired within the bound.
            RuntimeError: Issuance returned no token.
        """
        record = await self._token_store().replace_rejected(rejected_token, lambda: self._issue_token(session))
        self.token = record.access_token

    async def _get(
        self,
        session: aiohttp.ClientSession,
        path: str,
        group: str,
        params: dict[str, Any] | None = None,
        max_retries: int | None = None,
    ) -> BrokerPayload:
        """GET one Toss endpoint and return the decoded body.

        Throttling is HTTP 429. The wait honours ``Retry-After`` (capped at ``TOSS_RETRY_AFTER_MAX_SECONDS``
        because the 15:20 decision window cannot absorb an unbounded vendor hint), and falls back to
        ``TOSS_RATE_LIMIT_BACKOFF_SECONDS``. An auth rejection (HTTP 401 or ``error.code`` in
        ``TOSS_INVALID_TOKEN_CODES``) triggers one compare-and-swap refresh and an identical replay that does not
        consume an attempt. Every send takes a slot from the group's host bucket.

        Args:
            session: Open HTTP session.
            path: Endpoint path appended to ``base_url``.
            group: Rate-limit group in ``TOSS_RATE_LIMIT_GROUPS``.
            params: Query parameters (``{}`` when None), sent identically on retries and the replay.
            max_retries: Rate-limit attempts; None means ``TOSS_RATE_LIMIT_MAX_RETRIES`` captured at construction.

        Returns:
            The body of the last HTTP call. A second auth rejection, or an exhausted 429, is returned as-is.

        Raises:
            ValueError: Unknown ``group`` (raised after ``ensure_token``, before any data send), or ``max_retries < 1``.
            TokenStoreLockTimeout, RuntimeError: Refresh failed.
            aiohttp.ClientError: Transport or non-JSON reply.
        """
        if not self.token:
            await self.ensure_token(session)
        limiter = self._limiter_for(group)
        attempts = int(max_retries) if max_retries is not None else self._rate_limit_max_retries
        backoff = self._rate_limit_backoff

        def open_request(token: str) -> Any:
            return session.get(
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                params=params or {},
            )

        policy = RetryPolicy(
            max_attempts=attempts,
            rate_limit_wait=lambda attempt, response: self._retry_after_seconds(response.headers) or backoff
            if response.status == 429
            else None,
        )
        response = await send_with_auth_retry(open_request, acquire=limiter.acquire, current_token=lambda: self.token or "", is_auth_rejected=lambda r: self._is_invalid_token(r.status, r.body), refresh=lambda sent: self._refresh_rejected_token(session, sent), policy=policy, log_stage="toss_get", log_context=f"group={group} path={path}")
        return cast("BrokerPayload", response.body)

    async def get_program_trades(self, session: aiohttp.ClientSession, symbol: str, count: int = 100, until: str | None = None) -> BrokerPayload:
        """일별 프로그램 매매동향 (`GET /api/v1/stocks/{symbol}/program-trades`)."""
        params: dict[str, str | int] = {"count": int(count)}
        if until:
            params["until"] = until
        return await self._get(session, f"/api/v1/stocks/{symbol}/program-trades", "STOCK_TRADING_TREND", params=params)

    async def get_rankings(
        self,
        session: aiohttp.ClientSession,
        *,
        ranking_type: str,
        market_country: str = "KR",
        duration: str = "1d",
        count: int = 100,
        exclude_investment_caution: bool | None = None,
    ) -> BrokerPayload:
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
        session: aiohttp.ClientSession,
        symbol: str,
        *,
        interval: str = "1m",
        count: int = 200,
        before: str | None = None,
        adjusted: bool | None = None,
    ) -> BrokerPayload:
        """캔들 차트 조회 (`GET /api/v1/candles`)."""
        params: dict[str, str | int] = {"symbol": symbol, "interval": interval, "count": int(count)}
        if before:
            params["before"] = before
        if adjusted is not None:
            params["adjusted"] = str(bool(adjusted)).lower()
        return await self._get(session, "/api/v1/candles", "MARKET_DATA_CHART", params=params)
