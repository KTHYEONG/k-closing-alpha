"""Invariant guards for the shared broker request loop in src.api._transport."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import Mock

import pytest

from src.api._transport import RetryPolicy, TransportResponse, read_response, send_with_auth_retry
from tests.broker_fakes import FakeBrokerResponse, FakeBrokerSession, scripted_session


def _recorder(monkeypatch: Any) -> list[float]:
    sleeps: list[float] = []

    async def _record(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _record)
    return sleeps


def test_read_response_decodes_body_before_status() -> None:
    import aiohttp

    status_reads = {"n": 0}

    class _Html429:
        headers: dict = {}

        async def json(self) -> Any:
            raise aiohttp.ContentTypeError(Mock(), (), message="no json")

        @property
        def status(self) -> int:
            status_reads["n"] += 1
            return 429

        async def __aenter__(self) -> _Html429:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    with pytest.raises(aiohttp.ContentTypeError):
        asyncio.run(read_response(_Html429()))  # type: ignore[arg-type]
    assert status_reads["n"] == 0


def test_read_response_coerces_missing_status() -> None:
    class _NoStatusAttr:
        def __init__(self, body: Any) -> None:
            self._body = body
            self.headers: dict = {}

        async def json(self) -> Any:
            return self._body

        async def __aenter__(self) -> _NoStatusAttr:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    async def _main() -> list[TransportResponse]:
        return [
            await read_response(FakeBrokerResponse(body="a", status=None)),  # type: ignore[arg-type]
            await read_response(FakeBrokerResponse(body="b", status=0)),
            await read_response(_NoStatusAttr("c")),  # type: ignore[arg-type]
        ]

    for response in asyncio.run(_main()):
        assert response.status == 200


def test_read_response_copies_headers() -> None:
    source = {"cont-yn": "Y", "next-key": "k1"}

    async def _main() -> TransportResponse:
        return await read_response(FakeBrokerResponse(body={}, headers=source))

    response = asyncio.run(_main())
    assert response.headers == source
    assert response.headers is not source
    response.headers["cont-yn"] = "MUT"
    assert source == {"cont-yn": "Y", "next-key": "k1"}


def test_read_response_converts_mapping_headers() -> None:
    from multidict import CIMultiDict, CIMultiDictProxy

    async def _main() -> TransportResponse:
        return await read_response(
            FakeBrokerResponse(body={}, headers=CIMultiDictProxy(CIMultiDict({"cont-yn": "Y"})))
        )

    response = asyncio.run(_main())
    assert type(response.headers) is dict
    assert response.headers["cont-yn"] == "Y"


def test_read_response_tolerates_unconvertible_headers() -> None:
    class _BadHeaders:
        def items(self) -> list:
            return [("a", "1")]

        def __iter__(self) -> Any:
            raise TypeError("not iterable")

    async def _main() -> list[TransportResponse]:
        return [
            await read_response(FakeBrokerResponse(body="a")),
            await read_response(FakeBrokerResponse(body="b", headers=object())),
            await read_response(FakeBrokerResponse(body="c", headers=_BadHeaders())),
        ]

    for body, response in zip(("a", "b", "c"), asyncio.run(_main()), strict=True):
        assert response.body == body
        assert response.headers == {}


def test_retry_policy_rejects_empty_budget() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0, rate_limit_wait=lambda attempt, response: None)
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=-1, rate_limit_wait=lambda attempt, response: None)


def _retry_harness(
    replies: list[FakeBrokerResponse],
    *,
    max_attempts: int,
    wait: Any,
    rejected_body: Any = "REJ",
) -> dict[str, Any]:
    events: list[str] = []
    sent_tokens: list[str] = []
    refresh_calls: list[str] = []
    tokens = {"current": "A"}
    session = scripted_session(replies)

    async def _acquire() -> None:
        events.append("acquire")

    def _open(token: str) -> Any:
        events.append("send")
        sent_tokens.append(token)
        return session.post("http://x/data", headers={"authorization": f"Bearer {token}"})

    async def _refresh(sent: str) -> None:
        events.append("refresh")
        refresh_calls.append(sent)
        tokens["current"] = "C"

    async def _main() -> TransportResponse:
        return await send_with_auth_retry(
            _open,
            acquire=_acquire,
            current_token=lambda: tokens["current"],
            is_auth_rejected=lambda r: r.body == rejected_body,
            refresh=_refresh,
            policy=RetryPolicy(max_attempts=max_attempts, rate_limit_wait=wait),
            log_stage="test_stage",
            log_context="k=v",
        )

    return {
        "events": events,
        "sent_tokens": sent_tokens,
        "refresh_calls": refresh_calls,
        "tokens": tokens,
        "session": session,
        "response": asyncio.run(_main()),
    }


def test_acquire_precedes_every_send_including_replay(monkeypatch: Any) -> None:
    _recorder(monkeypatch)
    state = _retry_harness(
        [FakeBrokerResponse(body="REJ", status=401), FakeBrokerResponse(body="OK")],
        max_attempts=3,
        wait=lambda attempt, response: None,
    )
    assert state["events"] == ["acquire", "send", "refresh", "acquire", "send"]
    assert state["response"].body == "OK"


def test_refresh_receives_sent_token_and_replay_uses_refreshed_token(monkeypatch: Any) -> None:
    _recorder(monkeypatch)
    events: list[str] = []
    sent_tokens: list[str] = []
    refresh_calls: list[str] = []
    tokens = {"current": "A"}
    session = scripted_session([FakeBrokerResponse(body="REJ", status=401), FakeBrokerResponse(body="OK")])

    async def _acquire() -> None:
        events.append("acquire")

    def _open(token: str) -> Any:
        sent_tokens.append(token)
        if len(sent_tokens) == 1:
            tokens["current"] = "B"
        return session.post("http://x/data", headers={"authorization": f"Bearer {token}"})

    async def _refresh(sent: str) -> None:
        refresh_calls.append(sent)
        tokens["current"] = "C"

    async def _main() -> TransportResponse:
        return await send_with_auth_retry(
            _open,
            acquire=_acquire,
            current_token=lambda: tokens["current"],
            is_auth_rejected=lambda r: r.body == "REJ",
            refresh=_refresh,
            policy=RetryPolicy(max_attempts=3, rate_limit_wait=lambda attempt, response: None),
            log_stage="test_stage",
            log_context="k=v",
        )

    assert asyncio.run(_main()).body == "OK"
    assert refresh_calls == ["A"]
    assert sent_tokens == ["A", "C"]


def test_second_rejection_returned_without_further_refresh(monkeypatch: Any, caplog: Any) -> None:
    import logging

    sleeps = _recorder(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="src.api._transport"):
        state = _retry_harness(
            [FakeBrokerResponse(body="REJ"), FakeBrokerResponse(body="REJ"), FakeBrokerResponse(body="OK")],
            max_attempts=3,
            wait=lambda attempt, response: None,
        )
    assert state["response"].body == "REJ"
    assert len(state["session"].requests) == 2
    assert len(state["refresh_calls"]) == 1
    assert sleeps == []
    assert any("AUTH_REJECTED_AFTER_REFRESH" in r.getMessage() for r in caplog.records)


def test_rate_limited_exhaustion_returns_last_response(monkeypatch: Any, caplog: Any) -> None:
    import logging

    sleeps = _recorder(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="src.api._transport"):
        state = _retry_harness(
            [FakeBrokerResponse(body=f"rl-{n}", status=429) for n in range(3)],
            max_attempts=3,
            wait=lambda attempt, response: 0.5 * (attempt + 1),
            rejected_body="NEVER",
        )
    assert len(state["session"].requests) == 3
    assert sleeps == [0.5, 1.0]
    assert state["response"].body == "rl-2"
    assert sum("RATE_LIMITED attempts=3" in r.getMessage() for r in caplog.records) == 1


def test_replay_does_not_consume_an_attempt(monkeypatch: Any) -> None:
    sleeps = _recorder(monkeypatch)
    state = _retry_harness(
        [FakeBrokerResponse(body="RL"), FakeBrokerResponse(body="REJ"), FakeBrokerResponse(body="RL")],
        max_attempts=2,
        wait=lambda attempt, response: 0.5 * (attempt + 1) if response.body == "RL" else None,
    )
    assert len(state["session"].requests) == 3
    assert sleeps == [0.5]
    assert len(state["refresh_calls"]) == 1
    assert state["response"].body == "RL"


def test_replay_wait_uses_triggering_attempt_index(monkeypatch: Any) -> None:
    sleeps = _recorder(monkeypatch)
    state = _retry_harness(
        [FakeBrokerResponse(body="REJ"), FakeBrokerResponse(body="RL"), FakeBrokerResponse(body="OK")],
        max_attempts=3,
        wait=lambda attempt, response: float(2**attempt) if response.body == "RL" else None,
    )
    assert len(state["session"].requests) == 3
    assert sleeps == [1.0]
    assert state["response"].body == "OK"


def test_transport_error_propagates_without_retry(monkeypatch: Any) -> None:
    import aiohttp

    sleeps = _recorder(monkeypatch)
    acquires = {"n": 0}

    async def _acquire() -> None:
        acquires["n"] += 1

    session = scripted_session([FakeBrokerResponse(enter_error=aiohttp.ClientConnectionError("boom"))])
    refresh_mock = Mock(side_effect=AssertionError("must not refresh"))

    async def _main() -> TransportResponse:
        return await send_with_auth_retry(
            lambda token: session.post("http://x/data"),
            acquire=_acquire,
            current_token=lambda: "tok",
            is_auth_rejected=lambda r: False,
            refresh=refresh_mock,
            policy=RetryPolicy(max_attempts=3, rate_limit_wait=lambda attempt, response: 1.0),
            log_stage="test_stage",
            log_context="k=v",
        )

    with pytest.raises(aiohttp.ClientConnectionError):
        asyncio.run(_main())
    refresh_mock.assert_not_called()
    assert acquires["n"] == 1
    assert len(session.requests) == 1
    assert sleeps == []


def test_refresh_failure_propagates_without_replay(monkeypatch: Any) -> None:
    _recorder(monkeypatch)
    session = scripted_session(
        [FakeBrokerResponse(body="REJ", status=401), FakeBrokerResponse(body="OK")]
    )

    async def _failing_refresh(sent: str) -> None:
        raise RuntimeError("issuance down")

    async def _main() -> TransportResponse:
        return await send_with_auth_retry(
            lambda token: session.post("http://x/data"),
            acquire=lambda: asyncio.sleep(0),
            current_token=lambda: "tok",
            is_auth_rejected=lambda r: r.body == "REJ",
            refresh=_failing_refresh,
            policy=RetryPolicy(max_attempts=3, rate_limit_wait=lambda attempt, response: None),
            log_stage="test_stage",
            log_context="k=v",
        )

    with pytest.raises(RuntimeError, match="issuance down"):
        asyncio.run(_main())
    assert len(session.requests) == 1


def test_logs_never_contain_the_token(monkeypatch: Any, caplog: Any) -> None:
    import logging

    _recorder(monkeypatch)
    token = "SECRET-TOKEN-XYZ"
    session = scripted_session(
        [FakeBrokerResponse(body="REJ", status=401), FakeBrokerResponse(body="RL"), FakeBrokerResponse(body="RL")]
    )

    async def _main() -> TransportResponse:
        return await send_with_auth_retry(
            lambda sent: session.post("http://x/data", headers={"authorization": f"Bearer {sent}"}),
            acquire=lambda: asyncio.sleep(0),
            current_token=lambda: token,
            is_auth_rejected=lambda r: r.body == "REJ",
            refresh=lambda sent: asyncio.sleep(0),
            policy=RetryPolicy(
                max_attempts=2, rate_limit_wait=lambda attempt, response: 0.5 if response.body == "RL" else None
            ),
            log_stage="test_stage",
            log_context="k=v",
        )

    assert asyncio.run(_main()).body == "RL"
    records = [r for r in caplog.records if r.name == "src.api._transport"]
    assert len(records) == 3
    for record in records:
        message = record.getMessage()
        assert token not in message
        assert message.startswith("[EXEC] stage=")


def test_read_response_works_with_real_aiohttp_session() -> None:
    import aiohttp
    from aiohttp import web

    async def _main() -> None:
        async def _ok(request: Any) -> Any:
            return web.json_response({"hello": "world"}, headers={"cont-yn": "Y"})

        async def _html429(request: Any) -> Any:
            return web.Response(status=429, text="<html>busy</html>", content_type="text/html")

        app = web.Application()
        app.router.add_post("/ok", _ok)
        app.router.add_post("/html429", _html429)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            async with aiohttp.ClientSession() as session:
                ok = await read_response(session.post(f"http://127.0.0.1:{port}/ok"))
                assert ok.body == {"hello": "world"}
                assert ok.status == 200
                assert ok.headers["cont-yn"] == "Y"
                with pytest.raises(aiohttp.ContentTypeError) as exc_info:
                    await read_response(session.post(f"http://127.0.0.1:{port}/html429"))
                assert isinstance(exc_info.value, aiohttp.ClientError)
        finally:
            await runner.cleanup()

    asyncio.run(_main())


def test_fake_session_records_sync_post_without_await() -> None:
    session = FakeBrokerSession(lambda request: FakeBrokerResponse(body={"ok": True}))

    async def _main() -> TransportResponse:
        return await read_response(session.post("http://x/data", headers={"a": "b"}, json={"k": 1}))

    assert asyncio.run(_main()).body == {"ok": True}
    assert session.requests[0].method == "POST"
    assert session.requests[0].headers == {"a": "b"}
