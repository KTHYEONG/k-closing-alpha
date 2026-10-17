"""토스증권 OpenAPI 클라이언트 (그룹별 레이트리밋 GET 요청/토큰 발급).

docs/architecture/broker_toss.md 1.4 레이트리밋 매트릭스를 그대로 옮긴 표이며,
현재는 STOCK_TRADING_TREND(프로그램/투자자/신용/대차 동향)만 실사용한다. 나머지
그룹은 향후 확장 시 매직넘버 없이 바로 쓰도록 남겨둔 것으로, 미사용 자체가
결함은 아니다(단순 데이터 테이블, 분기 로직 없음).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

from src.api._common import parse_expires_in
from src.api._token_client import SharedTokenClientMixin
from src.api._transport import RetryPolicy, send_with_auth_retry
from src.api.auth_rejection import AUTH_REJECTION_CODES
from src.api.host_broker_contract import DEFAULT_RATES, TokenIssueThrottled
from src.api.kis.rate_limit import HostPacedRateLimiter, vendor_limiter
from src.api.shared_token import IssuedToken
from src.config import settings

if TYPE_CHECKING:
    import aiohttp

    from src.data.capture_contracts import BrokerPayload

logger = logging.getLogger(__name__)

_OAUTH_PATH = "/oauth2/token"

TOSS_INVALID_TOKEN_CODES: frozenset[str] = AUTH_REJECTION_CODES["toss"]

TOSS_RETRY_AFTER_MAX_SECONDS: float = 5.0
"""Ceiling on a server-sent Retry-After wait; collect calls Toss inside the 15:20 decision window,
so an unbounded vendor hint could stall the decision snapshot."""

TOSS_RATE_LIMIT_GROUPS: Mapping[str, float] = MappingProxyType(
    {scope: rate for (vendor, scope), rate in DEFAULT_RATES.items() if vendor == "toss" and scope is not None}
)


class TossApiClient(SharedTokenClientMixin):
    _TOKEN_VENDOR = "toss"  # noqa: S105 - vendor id, not a secret
    _TOKEN_EXPIRY_MARGIN_SECONDS = 300.0
    _TOKEN_ISSUE_THROTTLE_POLICY = "raise"  # noqa: S105 - policy label, not a secret
    _TOKEN_THROTTLED_ERROR = TokenIssueThrottled

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
        if group not in TOSS_RATE_LIMIT_GROUPS:
            raise ValueError(f"unknown Toss rate limit group: {group}")
        return vendor_limiter("toss", self.app_key or "", group)

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
            rate_limit_wait=lambda attempt, response: (
                self._retry_after_seconds(response.headers) or backoff if response.status == 429 else None
            ),
        )
        response = await send_with_auth_retry(
            open_request,
            acquire=limiter.acquire,
            current_token=lambda: self.token or "",
            is_auth_rejected=self._auth_rejected,
            refresh=lambda sent: self._refresh_rejected_token(session, sent),
            policy=policy,
            log_stage="toss_get",
            log_context=f"group={group} path={path}",
        )
        return cast("BrokerPayload", response.body)

    async def get_program_trades(
        self, session: aiohttp.ClientSession, symbol: str, count: int = 100, until: str | None = None
    ) -> BrokerPayload:
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
