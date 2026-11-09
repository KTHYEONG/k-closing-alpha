"""KisApiClient._handle_request retry and rate-limit slot behavior."""

from __future__ import annotations

import asyncio
import logging
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


_DISTINCT_HEADERS = {"appKey": "APPKEY-VALUE-111", "appSecret": "SECRET-VALUE-222", "tr_id": "FHKST01010100"}


async def _no_sleep_async(_seconds: float) -> None:
    return None


def _run_with_headers(session_method) -> dict:
    async def _runner() -> dict:
        client = _client()
        return await client._handle_request(session_method, "http://localhost/quote", headers=dict(_DISTINCT_HEADERS))

    return asyncio.run(_runner())


def _client_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "src.api.kis.client" and r.levelno >= logging.WARNING]


def test_network_retry_line_has_identifiers(monkeypatch, caplog) -> None:
    from src.api.kis.key_pool import kis_key_id

    monkeypatch.setattr(asyncio, "sleep", _no_sleep_async)
    session_method = Mock(
        side_effect=[aiohttp.ClientConnectionError("boom"), _FakeResponse(200, {"rt_cd": "0", "msg1": "ok"})]
    )

    with caplog.at_level(logging.WARNING, logger="src.api.kis.client"):
        result = _run_with_headers(session_method)

    assert result["rt_cd"] == "0"
    retries = [m for m in _client_warnings(caplog) if "status=NETWORK_RETRY" in m]
    assert len(retries) == 1
    assert "tr_id=FHKST01010100" in retries[0]
    assert f"key_id={kis_key_id('test-key')}" in retries[0]
    assert "error=ClientConnectionError" in retries[0]
    assert "attempt=1/5" in retries[0]
    joined = "\n".join(_client_warnings(caplog))
    for forbidden in ("APPKEY-VALUE-111", "SECRET-VALUE-222", "http://localhost"):
        assert forbidden not in joined


def test_network_exhausted_logs_once_and_returns_same_dict(monkeypatch, caplog) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep_async)
    session_method = Mock(side_effect=[aiohttp.ClientConnectionError("boom")] * 5)

    with caplog.at_level(logging.WARNING, logger="src.api.kis.client"):
        result = _run_with_headers(session_method)

    assert result == {"rt_cd": "9", "msg1": "네트워크 연결 실패: boom"}
    messages = _client_warnings(caplog)
    assert sum("status=NETWORK_RETRY" in m for m in messages) == 4
    assert sum("status=NETWORK_FAILED" in m for m in messages) == 1
    assert not any("status=RETRY_EXHAUSTED" in m for m in messages)


def test_rate_limit_exhaustion_logs_one_summary(monkeypatch, caplog) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep_async)
    session_method = Mock(side_effect=[_FakeResponse(429, {})] * 5)

    with caplog.at_level(logging.WARNING, logger="src.api.kis.client"):
        result = _run_with_headers(session_method)

    assert result == {"rt_cd": "9", "msg1": "최대 재시도 횟수 초과 (TPS 제한)"}
    messages = _client_warnings(caplog)
    assert len(messages) == 1
    assert "status=RETRY_EXHAUSTED" in messages[0]
    assert "http_429=5 tps_limit=0" in messages[0]


def test_tps_body_exhaustion_logs_msg_cd(monkeypatch, caplog) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep_async)
    tps_body = {"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "초당 거래건수를 초과하였습니다."}
    session_method = Mock(side_effect=[_FakeResponse(200, tps_body)] * 5)

    with caplog.at_level(logging.WARNING, logger="src.api.kis.client"):
        result = _run_with_headers(session_method)

    assert result["rt_cd"] == "9"
    messages = _client_warnings(caplog)
    assert len(messages) == 1
    assert "tps_limit=5 auth_rejected=0 last_msg_cd=EGW00201" in messages[0]


def test_success_after_one_429_is_silent(monkeypatch, caplog) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep_async)
    session_method = Mock(side_effect=[_FakeResponse(429, {}), _FakeResponse(200, {"rt_cd": "0", "msg1": "ok"})])

    with caplog.at_level(logging.WARNING, logger="src.api.kis.client"):
        result = _run_with_headers(session_method)

    assert result["rt_cd"] == "0"
    assert _client_warnings(caplog) == []
