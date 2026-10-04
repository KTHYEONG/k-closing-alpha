"""KisApiClient._handle_request retry and rate-limit slot behavior."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import aiohttp

from src.api.kis.client import KisApiClient


class _FakeResponse:
    """_handle_request용 aiohttp 응답 대역 (async context manager)."""

    def __init__(self, status: int, data: dict) -> None:
        self.status = status
        self._data = data

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def json(self) -> dict:
        return self._data


def _client() -> KisApiClient:
    return KisApiClient(
        app_key="test-key",
        account_id="test-account",
        hts_id="test-hts",
        base_url="http://localhost",
    )


def _run_handle_request(session_method) -> dict:
    async def _runner() -> dict:
        client = _client()
        return await client._handle_request(session_method, "http://localhost/quote")

    return asyncio.run(_runner())


def test_handle_request_success_acquires_rate_limiter() -> None:
    """정상 응답 시 rate_limiter.acquire()를 경유해 요청이 전달된다."""
    session_method = Mock(return_value=_FakeResponse(200, {"rt_cd": "0", "msg1": "ok"}))

    result = _run_handle_request(session_method)

    assert result["rt_cd"] == "0"
    assert session_method.call_count == 1


def test_handle_request_retries_on_client_error() -> None:
    """네트워크 에러 발생 시 지수 백오프 재시도 후 성공한다."""
    session_method = Mock(
        side_effect=[
            aiohttp.ClientConnectionError("boom"),
            _FakeResponse(200, {"rt_cd": "0", "msg1": "recovered"}),
        ]
    )

    result = _run_handle_request(session_method)

    assert result["rt_cd"] == "0"
    assert session_method.call_count == 2


def test_handle_request_retries_on_tps_message() -> None:
    """KIS '초당 거래건수' TPS 초과 메시지 응답 시 재시도 후 성공한다."""
    session_method = Mock(
        side_effect=[
            _FakeResponse(200, {"rt_cd": "9", "msg1": "초당 거래건수를 초과하였습니다."}),
            _FakeResponse(200, {"rt_cd": "0", "msg1": "ok"}),
        ]
    )

    result = _run_handle_request(session_method)

    assert result["rt_cd"] == "0"
    assert session_method.call_count == 2
