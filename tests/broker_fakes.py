"""Typed fake broker sessions for transport tests (test-only, no I/O)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

from src.api._transport import BrokerResponse


@dataclass(frozen=True)
class RecordedRequest:
    """One captured request: method ("GET"/"POST"), url, and the headers/json/data/params kwargs as sent."""

    method: str
    url: str
    headers: dict[str, str]
    json: Any
    data: Any
    params: Any


class FakeBrokerResponse(AbstractAsyncContextManager["BrokerResponse"]):
    """Scripted reply usable as both the async context manager and the response it yields.

    Args:
        body: Value returned by ``json()``.
        status: HTTP status.
        headers: Response headers (any Mapping, e.g. ``CIMultiDictProxy``) or None.
        json_error: Raised by ``json()`` instead of returning ``body`` (e.g. ``aiohttp.ContentTypeError``).
        enter_error: Raised on context entry to simulate a transport failure.
        enter_gate: Awaited on context entry before returning, to order concurrent scenarios deterministically.
        enter_delay: Seconds to sleep on entry (holds a request in flight).
    """

    def __init__(
        self,
        body: Any = None,
        status: int = 200,
        headers: Any = None,
        *,
        json_error: BaseException | None = None,
        enter_error: BaseException | None = None,
        enter_gate: asyncio.Event | None = None,
        enter_delay: float = 0.0,
    ) -> None:
        self._body = body
        self.status = status
        self.headers = headers
        self._json_error = json_error
        self._enter_error = enter_error
        self._enter_gate = enter_gate
        self._enter_delay = enter_delay

    async def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._body

    async def __aenter__(self) -> BrokerResponse:
        if self._enter_gate is not None:
            await self._enter_gate.wait()
        if self._enter_delay:
            await asyncio.sleep(self._enter_delay)
        if self._enter_error is not None:
            raise self._enter_error
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class FakeBrokerSession:
    """Synchronous ``post``/``get`` that record every request and return ``responder(request)``.

    Args:
        responder: Maps a ``RecordedRequest`` to the reply. It raises ``AssertionError`` for an unexpected extra
            request.

    Attributes:
        requests: All recorded requests in send order.
    """

    def __init__(self, responder: Callable[[RecordedRequest], FakeBrokerResponse]) -> None:
        self._responder = responder
        self.requests: list[RecordedRequest] = []

    def _record(self, method: str, url: str, kwargs: dict[str, Any]) -> FakeBrokerResponse:
        recorded = RecordedRequest(
            method=method,
            url=url,
            headers=dict(kwargs.get("headers") or {}),
            json=kwargs.get("json"),
            data=kwargs.get("data"),
            params=kwargs.get("params"),
        )
        self.requests.append(recorded)
        return self._responder(recorded)

    def post(self, url: str, **kwargs: Any) -> FakeBrokerResponse:
        return self._record("POST", url, kwargs)

    def get(self, url: str, **kwargs: Any) -> FakeBrokerResponse:
        return self._record("GET", url, kwargs)

    def requests_to(self, suffix: str) -> list[RecordedRequest]:
        """Requests whose URL ends with ``suffix`` (e.g. token vs data endpoints)."""
        return [request for request in self.requests if request.url.endswith(suffix)]


def scripted_session(
    data_replies: Sequence[Any],
    *,
    token_bodies: Sequence[Any] = (),
    token_path: str = "/oauth2/token",  # noqa: S107 - URL path suffix, not a credential
) -> FakeBrokerSession:
    """Route ``token_path`` requests to ``token_bodies`` in order (the last one repeats) and everything else to
    ``data_replies`` in order. An exhausted ``data_replies`` fails the test."""

    def _as_response(reply: Any) -> FakeBrokerResponse:
        if isinstance(reply, FakeBrokerResponse):
            return reply
        return FakeBrokerResponse(body=reply, status=200)

    pending = [_as_response(reply) for reply in data_replies]
    tokens = list(token_bodies)

    def _respond(request: RecordedRequest) -> FakeBrokerResponse:
        if request.url.endswith(token_path):
            if not tokens:
                raise AssertionError(f"unexpected token request: {request.url}")
            body = tokens.pop(0) if len(tokens) > 1 else tokens[0]
            return _as_response(body)
        if not pending:
            raise AssertionError(f"unexpected extra data request: {request.url}")
        return pending.pop(0)

    return FakeBrokerSession(_respond)
