"""KIS task-local market-response observation guards."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.api.kis.client import KisApiClient
from src.data.capture_contracts import RawCaptureError


def _kis_client(monkeypatch: Any) -> KisApiClient:
    client = KisApiClient(app_key="k", app_secret="s")
    client.token = "T"

    async def _free_acquire() -> None:
        return None

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client.rate_limiter, "acquire", _free_acquire)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    return client


class _Resp:
    def __init__(self, status: int, body: Any, *, json_error: Any = None):
        self.status = status
        self._body = body
        self._json_error = json_error

    async def json(self) -> Any:
        if self._json_error is not None:
            raise self._json_error
        return self._body


class _Ctx:
    def __init__(self, action: Any):
        self._action = action

    async def __aenter__(self) -> Any:
        action = self._action
        if isinstance(action, BaseException):
            raise action
        return action

    async def __aexit__(self, *_a: Any) -> bool:
        return False


class _Session:
    def __init__(self, actions: list[Any]):
        self._actions = actions
        self.calls = 0

    def get(self, _url: str, **_kw: Any) -> Any:
        action = self._actions[min(self.calls, len(self._actions) - 1)]
        self.calls += 1
        return _Ctx(action)


def test_failed_retry_response_is_retained(monkeypatch: Any) -> None:
    client = _kis_client(monkeypatch)
    client.ensure_token = AsyncMock(return_value="T2")  # type: ignore[method-assign]
    session = _Session(
        [
            _Resp(200, {"rt_cd": "1", "msg_cd": "EGW00121", "msg1": "token expired"}),
            _Resp(200, {"rt_cd": "0", "output": {"stck_prpr": "1000"}}),
        ]
    )
    seen: list[tuple[Any, ...]] = []
    with client.observe_market_responses(lambda *a: seen.append(a)):
        out = asyncio.run(
            client._handle_request(session.get, "http://x", headers={"tr_id": "FHKST01010100"}),
        )
    assert out["rt_cd"] == "0"
    assert session.calls == 2
    assert len(seen) == 2
    assert [call[5] for call in seen] == [0, 1]
    assert seen[0][0]["msg_cd"] == "EGW00121"
    assert seen[1][0]["rt_cd"] == "0"
    assert seen[0][1]["tr_id"] == "FHKST01010100"
    for call in seen:
        assert call[2].tzinfo is not None and call[3] >= call[2]


def test_concurrent_contexts_remain_isolated(monkeypatch: Any) -> None:
    client = _kis_client(monkeypatch)

    async def _main() -> tuple[list[Any], list[Any]]:
        seen_a: list[Any] = []
        seen_b: list[Any] = []

        async def _call(code: str, tr_id: str, seen: list[Any]) -> None:
            session = _Session([_Resp(200, {"rt_cd": "0", "output": {"code": code}})])
            with client.observe_market_responses(lambda *a: seen.append(a)):
                await asyncio.sleep(0)
                await client._handle_request(session.get, "http://x", headers={"tr_id": tr_id})

        await asyncio.gather(
            _call("005930", "TR-A", seen_a),
            _call("000660", "TR-B", seen_b),
        )
        return seen_a, seen_b

    seen_a, seen_b = asyncio.run(_main())
    assert len(seen_a) == 1 and len(seen_b) == 1
    assert seen_a[0][0]["output"] == {"code": "005930"}
    assert seen_b[0][0]["output"] == {"code": "000660"}
    assert seen_a[0][1]["tr_id"] == "TR-A"
    assert seen_b[0][1]["tr_id"] == "TR-B"


def test_transport_failure_stays_explicit(monkeypatch: Any) -> None:
    client = _kis_client(monkeypatch)
    session = _Session([TimeoutError("slow"), _Resp(200, {"rt_cd": "0", "output": {}})])
    seen: list[tuple[Any, ...]] = []
    with client.observe_market_responses(lambda *a: seen.append(a)):
        out = asyncio.run(client._handle_request(session.get, "http://x", headers={"tr_id": "T"}))
    assert out["rt_cd"] == "0"
    assert len(seen) == 2
    assert seen[0][0] is None
    assert seen[0][1]["error_type"] == "TimeoutError"
    assert seen[0][3] >= seen[0][2]
    assert seen[1][0]["rt_cd"] == "0"


def test_observer_failure_escapes_retry(monkeypatch: Any) -> None:
    client = _kis_client(monkeypatch)
    session = _Session([_Resp(200, {"rt_cd": "0", "output": {}})])

    def _boom(*args: Any) -> None:
        raise RawCaptureError("durable store down")

    with client.observe_market_responses(_boom), pytest.raises(RawCaptureError, match="durable store down"):
        asyncio.run(client._handle_request(session.get, "http://x", headers={}))
    assert session.calls == 1


def test_token_payload_is_excluded(monkeypatch: Any) -> None:
    client = _kis_client(monkeypatch)
    issued = {"n": 0}

    async def _fake_issue(session: Any, force_refresh: bool = False) -> str:
        issued["n"] += 1
        client.token = "T2"
        return "T2"

    client.ensure_token = _fake_issue  # type: ignore[method-assign]
    session = _Session(
        [
            _Resp(200, {"rt_cd": "1", "msg_cd": "EGW00121", "msg1": "token expired"}),
            _Resp(200, {"rt_cd": "0", "output": {"stck_prpr": "1000"}}),
        ]
    )
    seen: list[tuple[Any, ...]] = []
    with client.observe_market_responses(lambda *a: seen.append(a)):
        out = asyncio.run(client._handle_request(session.get, "http://x", headers={"tr_id": "T"}))
    assert out["rt_cd"] == "0"
    assert issued["n"] == 1
    assert len(seen) == 2
    for payload, metadata, *_rest in seen:
        assert "access_token" not in (payload or {})
        assert set(metadata) <= {"tr_id", "rt_cd", "msg_cd", "error_type", "http_status"}
        assert "authorization" not in metadata.values()


def test_legacy_transport_remains_compatible(monkeypatch: Any) -> None:
    client = _kis_client(monkeypatch)
    session = _Session([_Resp(200, {"rt_cd": "0", "output": {"stck_prpr": "1000"}})])
    out = asyncio.run(client._handle_request(session.get, "http://x", headers={"tr_id": "T"}))
    assert out == {"rt_cd": "0", "output": {"stck_prpr": "1000"}}


def test_rate_limited_undecodable_evidence_uses_none_payload(monkeypatch: Any) -> None:
    client = _kis_client(monkeypatch)
    session = _Session(
        [
            _Resp(429, {}, json_error=ValueError("no json")),
            _Resp(200, {"rt_cd": "0", "output": {}}),
        ]
    )
    seen: list[tuple[Any, ...]] = []
    with client.observe_market_responses(lambda *a: seen.append(a)):
        out = asyncio.run(client._handle_request(session.get, "http://x", headers={"tr_id": "T"}))
    assert out["rt_cd"] == "0"
    assert seen[0][0] is None
    assert seen[0][1]["http_status"] == "429"


def test_decode_error_reports_none_payload_and_raises(monkeypatch: Any) -> None:
    client = _kis_client(monkeypatch)
    session = _Session([_Resp(200, {}, json_error=ValueError("cut"))])
    seen: list[tuple[Any, ...]] = []
    with client.observe_market_responses(lambda *a: seen.append(a)), pytest.raises(ValueError, match="cut"):
        asyncio.run(client._handle_request(session.get, "http://x", headers={"tr_id": "T"}))
    assert len(seen) == 1
    assert seen[0][0] is None
    assert seen[0][1]["error_type"] == "ValueError"


def test_scope_restores_after_exception(monkeypatch: Any) -> None:
    client = _kis_client(monkeypatch)
    session = _Session([_Resp(200, {"rt_cd": "0"})])
    seen: list[Any] = []
    with pytest.raises(RuntimeError, match="inside"), client.observe_market_responses(lambda *a: seen.append(a)):
        raise RuntimeError("inside")
    out = asyncio.run(client._handle_request(session.get, "http://x", headers={}))
    assert out == {"rt_cd": "0"}
    assert seen == []
