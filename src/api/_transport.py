"""Shared request loop for the Kiwoom, LS and Toss REST clients.

The loop applies one auth refresh, rate-limit backoff and host admission to every send. Vendors supply the
contract-specific parts: how a rejection is recognized, where a rate limit is signalled and how long to wait,
and how the result is shaped. KIS is deliberately not a client of this module because its callers depend on
synthesized ``rt_cd == "9"`` failure dicts instead of raised transport errors.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class BrokerResponse(Protocol):
    """Structural subset of ``aiohttp.ClientResponse`` that the broker clients read."""

    @property
    def status(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    async def json(self) -> Any: ...


@dataclass(frozen=True)
class TransportResponse:
    """One decoded HTTP exchange.

    Attributes:
        body: Decoded JSON exactly as returned. Its shape is not validated; vendor classifiers interpret it.
        headers: Plain ``dict`` copy of the response headers (``{}`` when absent or not convertible).
        status: HTTP status; a missing, ``None`` or ``0`` status reads as 200.
    """

    body: Any
    headers: dict[str, str]
    status: int


RateLimitWait = Callable[[int, TransportResponse], float | None]


@dataclass(frozen=True)
class RetryPolicy:
    """Rate-limit retry budget for one logical request.

    Attributes:
        max_attempts: Number of rate-limit attempts. An auth-refresh replay does not consume one, so the HTTP
            call cap is ``max_attempts + 1``.
        rate_limit_wait: ``(attempt_index, response) -> seconds | None``. Returns None when ``response`` is not
            rate limited. Otherwise returns the wait before the next attempt. ``attempt_index`` is 0-based, and
            a replay is evaluated with the index of the attempt whose rejection triggered it.

    Raises:
        ValueError: ``max_attempts < 1``. A zero budget would return an empty response without contacting the
            vendor, which would turn a configuration error into silent missing data.
    """

    max_attempts: int
    rate_limit_wait: RateLimitWait

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1: {self.max_attempts!r}")


async def read_response(request: AbstractAsyncContextManager[BrokerResponse]) -> TransportResponse:
    """Enter ``request``, decode the JSON body, then capture status and headers.

    The body is decoded first, on purpose. A non-JSON reply (for example an HTML 429 from a WAF) raises
    ``aiohttp.ContentTypeError``, an ``aiohttp.ClientError``. That keeps it a transport failure, which the
    collector retries with a fresh evidence attempt slot, and stops it from being treated as a vendor rate
    limit with a reused slot.

    Raises:
        Exception: Anything raised while entering the context, decoding or exiting propagates unchanged.
    """
    async with request as resp:
        body = await resp.json()
        status = int(getattr(resp, "status", 200) or 200)
        headers_raw: Any = getattr(resp, "headers", None)
        if isinstance(headers_raw, dict):
            headers = dict(headers_raw)
        elif hasattr(headers_raw, "items"):
            try:
                headers = dict(headers_raw)
            except Exception:
                headers = {}
        else:
            headers = {}
    return TransportResponse(body=body, headers=headers, status=status)


async def send_with_auth_retry(
    open_request: Callable[[str], AbstractAsyncContextManager[BrokerResponse]],
    *,
    acquire: Callable[[], Awaitable[None]],
    current_token: Callable[[], str],
    is_auth_rejected: Callable[[TransportResponse], bool],
    refresh: Callable[[str], Awaitable[None]],
    policy: RetryPolicy,
    log_stage: str,
    log_context: str,
) -> TransportResponse:
    """Send one logical request with host admission, a single CAS auth refresh and rate-limit backoff.

    Args:
        open_request: Builds the HTTP request for a bearer token, e.g. ``session.post(url, headers=..., json=...)``.
            It is called once per send with the token read immediately after admission.
        acquire: Host admission (``HostPacedRateLimiter.acquire``). It is awaited before every send, including
            retries and the refresh replay, so no send bypasses the host-wide vendor quota.
        current_token: Returns the client's current bearer token.
        is_auth_rejected: The vendor's auth-rejection classifier.
        refresh: Called with the exact token the rejected request carried. It must adopt a token a peer already
            rotated, or issue a new one, and update the client so that ``current_token()`` returns it.
        policy: Rate-limit budget and wait function.
        log_stage: ``stage=`` value for log records (``kiwoom_tr``, ``ls_tr``, ``toss_get``).
        log_context: Pre-formatted ``key=value`` correlation fields (for example ``api_id=ka10079``). It must
            never contain tokens, app keys or secrets.

    Returns:
        The response of the last HTTP call. That is either the first non-rate-limited response, a second auth
        rejection returned as-is, or the last rate-limited response once the budget is exhausted.

    Raises:
        Exception: Errors from ``acquire``, ``open_request``, ``read_response`` or ``refresh`` propagate on the
            first occurrence, with no retry and no sleep. Transport retries belong to the caller.
    """
    refreshed = False
    for attempt in range(policy.max_attempts):
        await acquire()
        sent_token = current_token()
        response = await read_response(open_request(sent_token))
        if is_auth_rejected(response) and not refreshed:
            refreshed = True
            await refresh(sent_token)
            logger.warning("[EXEC] stage=%s status=TOKEN_REPLACED %s", log_stage, log_context)
            await acquire()
            response = await read_response(open_request(current_token()))
            if is_auth_rejected(response):
                logger.warning("[EXEC] stage=%s status=AUTH_REJECTED_AFTER_REFRESH %s", log_stage, log_context)
                return response
        wait = policy.rate_limit_wait(attempt, response)
        if wait is None:
            return response
        if attempt < policy.max_attempts - 1:
            logger.warning(
                "[EXEC] stage=%s status=RATE_LIMIT_RETRY wait_s=%.3f attempt=%d/%d %s",
                log_stage,
                wait,
                attempt + 1,
                policy.max_attempts,
                log_context,
            )
            await asyncio.sleep(wait)
            continue
        logger.warning(
            "[EXEC] stage=%s status=RATE_LIMITED attempts=%d %s", log_stage, policy.max_attempts, log_context
        )
        return response
    raise AssertionError("unreachable: the rate-limit loop always returns")
