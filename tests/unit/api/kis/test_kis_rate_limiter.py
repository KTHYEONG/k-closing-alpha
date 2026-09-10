"""Unit test for KisApiClient AsyncRateLimiter (SCENARIO_KIS_RATE_LIMITER_01)."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import Mock

import aiohttp

from src.api.kis.client import KisApiClient
from src.api.kis.rate_limit import AsyncRateLimiter


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


def test_scenario_kis_rate_limiter_01() -> None:
    """[SCENARIO_KIS_RATE_LIMITER_01] Verify rate limiter throttles calls correctly within max TPS limit."""
    async def _runner() -> None:
        limiter = AsyncRateLimiter(max_rate=5.0, time_period=1.0)
        start_time = time.monotonic()

        # 6개 요청 획득시 max_rate가 5이므로 6번째 요청은 최소 1초 후 처리되어야 함
        for _ in range(6):
            await limiter.acquire()

        elapsed = time.monotonic() - start_time
        assert elapsed >= 0.9, f"Elapsed time should be at least ~1.0s, got {elapsed:.3f}s"

    asyncio.run(_runner())


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


def test_get_shared_rate_limiter_returns_one_bucket_per_vendor_credential() -> None:
    from src.api.kis.rate_limit import AsyncRateLimiter, get_shared_rate_limiter

    # Given / When: 동일 (vendor, credential, rate, period)
    a = get_shared_rate_limiter("kis", "APPKEY_A", 18.0)
    b = get_shared_rate_limiter("kis", "APPKEY_A", 18.0)

    # Then: 프로세스 전역 단일 인스턴스
    assert a is b
    assert isinstance(a, AsyncRateLimiter)
    assert a.max_rate == 18.0

    # And: 자격증명이 다르면 별도 버킷
    assert get_shared_rate_limiter("kis", "APPKEY_B", 18.0) is not a
    # And: 벤더가 다르면 별도 버킷
    assert get_shared_rate_limiter("kiwoom", "APPKEY_A", 18.0) is not a
    # And: TR별 버킷도 분리된다
    kw1 = get_shared_rate_limiter("kiwoom", "K:ka10027", 5.0)
    kw2 = get_shared_rate_limiter("kiwoom", "K:ka10080", 5.0)
    assert kw1 is not kw2
    assert kw1.max_rate == 5.0
