from __future__ import annotations


def test_toss_client_ensure_token() -> None:
    import asyncio

    from src.api.toss.client import TossApiClient
    from tests.broker_fakes import scripted_session

    client = TossApiClient(app_key="k", app_secret="s")
    session = scripted_session(
        [], token_bodies=[{"access_token": "mock_tok", "token_type": "Bearer", "expires_in": 86400}]
    )

    token = asyncio.run(client.ensure_token(session))

    assert token == "mock_tok"
    assert client.token == "mock_tok"


def test_toss_client_ensure_token_concurrent_calls_lock_and_request_once() -> None:
    import asyncio

    from src.api.toss.client import TossApiClient
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    client = TossApiClient(app_key="k", app_secret="s")
    session = scripted_session(
        [],
        token_bodies=[
            FakeBrokerResponse(
                body={"access_token": "tok_123", "token_type": "Bearer", "expires_in": 86400},
                enter_delay=0.01,
            )
        ],
    )

    async def runner():
        tokens = await asyncio.gather(*[client.ensure_token(session) for _ in range(10)])
        assert all(t == "tok_123" for t in tokens)

    asyncio.run(runner())
    assert len(session.requests_to("/oauth2/token")) == 1


def test_toss_client_ensure_token_raises_on_empty_token() -> None:
    import asyncio

    import pytest

    from src.api.toss.client import TossApiClient
    from tests.broker_fakes import scripted_session

    client = TossApiClient(app_key="k", app_secret="s")
    session = scripted_session(
        [], token_bodies=[{"error": {"code": "invalid-client", "message": "bad credentials"}}]
    )

    with pytest.raises(RuntimeError, match="Toss token issuance failed"):
        asyncio.run(client.ensure_token(session))


def test_toss_client_get_program_trades_parses_result_envelope() -> None:
    import asyncio

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    session = scripted_session(
        [FakeBrokerResponse(body={"result": {"records": [
            {"date": "2026-09-10", "arbitrage": {"netBuyVolume": "3"}, "nonArbitrage": {"netBuyVolume": "4"}}
        ]}})]
    )

    data = asyncio.run(client.get_program_trades(session, "005930", count=30, until="2026-09-10"))

    assert data["result"]["records"][0]["arbitrage"]["netBuyVolume"] == "3"
    (request,) = session.requests
    assert request.url == "https://openapi.tossinvest.com/api/v1/stocks/005930/program-trades"
    assert request.params == {"count": 30, "until": "2026-09-10"}


def test_toss_client_get_program_trades_retries_on_429_then_succeeds(monkeypatch) -> None:
    import asyncio

    import src.api.toss.client as toss_client_mod
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    client = toss_client_mod.TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    _real_sleep = asyncio.sleep

    async def _fast_sleep(seconds) -> None:
        await _real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    session = scripted_session(
        [
            FakeBrokerResponse(body={"error": {"code": "rate-limit-exceeded", "message": "too many requests"}}, status=429),
            FakeBrokerResponse(body={"result": {"records": []}}),
        ]
    )

    data = asyncio.run(client.get_program_trades(session, "005930"))

    assert data == {"result": {"records": []}}
    assert len(session.requests) == 2


def test_toss_client_unknown_rate_limit_group_raises() -> None:
    import pytest

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")

    with pytest.raises(ValueError, match="NOT_A_GROUP"):
        client._limiter_for("NOT_A_GROUP")


def test_toss_client_get_rankings_parses_result_envelope() -> None:
    import asyncio

    from src.api.toss.client import TossApiClient
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    session = scripted_session(
        [FakeBrokerResponse(body={"result": {"rankedAt": "2026-09-11T19:59:46.466+09:00", "rankings": [
            {"rank": 1, "symbol": "000660", "price": {"lastPrice": "1832000", "changeRate": "-0.0113"}, "tradingVolume": "79878", "tradingAmount": "145941219000"}
        ]}})]
    )

    data = asyncio.run(client.get_rankings(session, ranking_type="TOP_GAINERS", market_country="KR", duration="1d", count=100))

    assert data["result"]["rankings"][0]["symbol"] == "000660"
    (request,) = session.requests
    assert request.url == "https://openapi.tossinvest.com/api/v1/rankings"
    assert request.params == {"type": "TOP_GAINERS", "marketCountry": "KR", "duration": "1d", "count": 100}


def test_toss_client_get_rankings_forwards_exclude_investment_caution() -> None:
    import asyncio

    from src.api.toss.client import TossApiClient
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    session = scripted_session(
        [FakeBrokerResponse(body={"result": {"rankedAt": "2026-09-11T19:59:46.466+09:00", "rankings": []}})]
    )

    asyncio.run(client.get_rankings(session, ranking_type="TOP_GAINERS", exclude_investment_caution=True))

    (request,) = session.requests
    assert request.params["excludeInvestmentCaution"] == "true"


def test_toss_client_get_candles_parses_result_envelope() -> None:
    import asyncio

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    session = scripted_session(
        [FakeBrokerResponse(body={"result": {"candles": [
            {"timestamp": "2024-02-28T15:30:00.000+09:00", "openPrice": "70000", "highPrice": "70100", "lowPrice": "69900", "closePrice": "70000", "volume": "10", "currency": "KRW"}
        ]}})]
    )

    data = asyncio.run(client.get_candles(session, "000250", interval="1m", count=200, before="2024-02-28T15:30:00.000+09:00"))

    assert data["result"]["candles"][0]["closePrice"] == "70000"
    (request,) = session.requests
    assert request.url == "https://openapi.tossinvest.com/api/v1/candles"
    assert request.params == {"symbol": "000250", "interval": "1m", "count": 200, "before": "2024-02-28T15:30:00.000+09:00"}


def test_toss_client_get_candles_forwards_adjusted_flag_and_omits_before_when_absent() -> None:
    import asyncio

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    session = scripted_session([FakeBrokerResponse(body={"result": {"candles": []}})])

    asyncio.run(client.get_candles(session, "005930", adjusted=False))

    (request,) = session.requests
    assert request.params["adjusted"] == "false"
    assert "before" not in request.params


def test_toss_client_explicit_credentials_win_over_instance(monkeypatch) -> None:
    from src.api.toss.client import TossApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "TOSS_APP_KEY", "inst")
    monkeypatch.setattr(settings_instance, "TOSS_BASE_URL", "https://toss.example")

    assert TossApiClient().app_key == "inst"
    assert TossApiClient().base_url == "https://toss.example"
    assert TossApiClient(app_key="arg").app_key == "arg"
    assert TossApiClient(base_url="https://arg.example").base_url == "https://arg.example"


def test_toss_client_documented_chart_rate_is_kept() -> None:
    from src.api.toss.client import TossApiClient

    assert TossApiClient(app_key="k", app_secret="s")._limiter_for("MARKET_DATA_CHART").max_rate == 20.0


def test_toss_invalid_token_recovers_once(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.toss.client import TossApiClient
    from src.config import settings as settings_instance
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")

    client = TossApiClient(app_key="k", app_secret="s")
    session = scripted_session(
        [
            FakeBrokerResponse(body={"error": {"code": "invalid-token"}}),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ],
        token_bodies=[
            {"access_token": "tok-A", "expires_in": 86400},
            {"access_token": "tok-B", "expires_in": 86400},
        ],
    )

    async def _seed():
        await client.ensure_token(session)

    asyncio.run(_seed())
    from src.api.shared_token import SharedTokenStore

    before = client._token_store().read()
    assert before is not None and before.generation == 1

    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))
    assert data == {"result": {"ok": True}}
    assert len(session.requests_to("/oauth2/token")) == 2
    after = client._token_store().read()
    assert after is not None and after.generation == before.generation + 1
    auths = [request.headers.get("Authorization") for request in session.requests if request.method == "GET"]
    assert auths[-1] == f"Bearer {after.access_token}"


def test_toss_rotation_by_peer_is_adopted(tmp_path, monkeypatch) -> None:
    import asyncio
    from datetime import UTC, datetime

    from src.api.shared_token import IssuedToken, SharedTokenStore, shared_token_path
    from src.api.toss.client import TossApiClient
    from src.config import settings as settings_instance
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")

    async def _seed_store():
        store = SharedTokenStore(
            shared_token_path("toss", "k"),
            lock_timeout_seconds=5.0,
            expiry_margin_seconds=0.0,
            clock=lambda: datetime.now(UTC),
        )
        await store.get_or_issue(lambda: asyncio.sleep(0, result=IssuedToken(access_token="tok-A", expires_in_seconds=86400)))
        await store.replace_rejected("tok-A", lambda: asyncio.sleep(0, result=IssuedToken(access_token="tok-B", expires_in_seconds=86400)))

    asyncio.run(_seed_store())

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok-A"
    session = scripted_session(
        [
            FakeBrokerResponse(body={"error": {"code": "invalid-token"}}),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ]
    )
    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))
    assert data == {"result": {"ok": True}}
    assert session.requests_to("/oauth2/token") == []
    auths = [request.headers.get("Authorization") for request in session.requests if request.method == "GET"]
    assert auths == ["Bearer tok-A", "Bearer tok-B"]


def test_toss_second_rejection_surfaces(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.toss.client import TossApiClient
    from src.config import settings as settings_instance
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")

    client = TossApiClient(app_key="k", app_secret="s")
    session = scripted_session(
        [
            FakeBrokerResponse(body={"error": {"code": "invalid-token"}}),
            FakeBrokerResponse(body={"error": {"code": "invalid-token"}}),
        ],
        token_bodies=[{"access_token": "tok-B", "expires_in": 86400}],
    )
    asyncio.run(client.ensure_token(session))
    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))
    assert data == {"error": {"code": "invalid-token"}}
    assert len(session.requests_to("/oauth2/token")) == 2


def test_toss_retry_after_honoured(tmp_path, monkeypatch) -> None:
    import asyncio

    import src.api.toss.client as toss_client_mod
    from src.api.toss.client import TossApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"
    sleeps: list[float] = []

    async def _fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    async def _fake_acquire(self):
        return None

    monkeypatch.setattr(toss_client_mod.HostPacedRateLimiter, "acquire", _fake_acquire)

    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    session = scripted_session(
        [
            FakeBrokerResponse(body={"error": {"code": "rate-limit-exceeded"}}, status=429, headers={"Retry-After": "2"}),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ]
    )

    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))
    assert data == {"result": {"ok": True}}
    assert sleeps == [2.0]


def test_toss_retry_after_capped_for_decision_window() -> None:
    from src.api.toss.client import TOSS_RETRY_AFTER_MAX_SECONDS, TossApiClient

    assert TossApiClient._retry_after_seconds({"Retry-After": "600"}) == TOSS_RETRY_AFTER_MAX_SECONDS
    assert TossApiClient._retry_after_seconds({"retry-after": "1.5"}) == 1.5


def test_toss_get_reads_retry_after_from_mapping_headers(monkeypatch) -> None:
    import asyncio
    from typing import Any

    from multidict import CIMultiDict, CIMultiDictProxy

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    async def _no_acquire(self) -> None:
        return None

    monkeypatch.setattr("src.api.toss.client.HostPacedRateLimiter.acquire", _no_acquire)

    sleeps: list[float] = []

    async def _record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _record_sleep)

    class _Resp:
        def __init__(self, body: dict, status: int, headers: Any) -> None:
            self._body = body
            self.status = status
            self.headers = headers

        async def json(self) -> dict:
            return dict(self._body)

        async def __aenter__(self) -> _Resp:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    responses = [
        _Resp({"error": {"code": "rate-limit-exceeded"}}, 429, CIMultiDictProxy(CIMultiDict({"Retry-After": "2"}))),
        _Resp({"result": {"records": []}}, 200, CIMultiDictProxy(CIMultiDict({}))),
    ]

    class _Session:
        def get(self, url: str, headers: dict | None = None, params: dict | None = None) -> _Resp:
            return responses.pop(0)

    data = asyncio.run(client.get_program_trades(_Session(), "005930"))

    assert data == {"result": {"records": []}}
    assert sleeps == [2.0]


def _toss_tmp_client(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from src.api.toss.client import TossApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")
    return TossApiClient(app_key="k", app_secret="s")


def _toss_get_session(get_replies, token_bodies=()):  # type: ignore[no-untyped-def]
    from tests.broker_fakes import scripted_session

    return scripted_session(list(get_replies), token_bodies=list(token_bodies))


def _toss_record_sleep(monkeypatch):  # type: ignore[no-untyped-def]
    import asyncio

    sleeps: list[float] = []

    async def _record(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _record)
    return sleeps


def _toss_count_acquires(monkeypatch):  # type: ignore[no-untyped-def]
    import src.api.toss.client as toss_client_mod

    calls = {"n": 0}

    async def _count(self) -> None:
        calls["n"] += 1

    monkeypatch.setattr(toss_client_mod.HostPacedRateLimiter, "acquire", _count)
    return calls


def _toss_get_auths(session):  # type: ignore[no-untyped-def]
    return [
        request.headers.get("Authorization", "")
        for request in session.requests
        if request.method == "GET"
    ]


def test_toss_get_429_without_retry_after_uses_fallback(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _toss_tmp_client(tmp_path, monkeypatch)
    client.token = "tok"
    session = _toss_get_session(
        [
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ]
    )
    sleeps = _toss_record_sleep(monkeypatch)

    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))

    assert data == {"result": {"ok": True}}
    assert len(session.requests_to("/api/v1/candles")) == 2
    assert sleeps == [1.2]


def test_toss_get_retry_after_honoured_and_capped(tmp_path, monkeypatch, caplog) -> None:
    import asyncio
    import logging

    from tests.broker_fakes import FakeBrokerResponse

    client = _toss_tmp_client(tmp_path, monkeypatch)
    client.token = "tok"
    session = _toss_get_session(
        [
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429, headers={"Retry-After": "600"}),
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429, headers={"Retry-After": "0"}),
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429),
        ]
    )
    sleeps = _toss_record_sleep(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="src.api._transport"):
        data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))

    assert data == {"error": {"code": "rate-limit"}}
    assert len(session.requests_to("/api/v1/candles")) == 3
    assert sleeps == [5.0, 1.2]
    assert sum("RATE_LIMITED attempts=3" in r.getMessage() for r in caplog.records) == 1


def test_toss_get_invalid_token_code_refreshes_once(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _toss_tmp_client(tmp_path, monkeypatch)
    session = _toss_get_session(
        [
            FakeBrokerResponse(body={"error": {"code": "invalid-token"}}),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ],
        token_bodies=[
            {"access_token": "tok-A", "expires_in": 86400},
            {"access_token": "tok-B", "expires_in": 86400},
        ],
    )
    asyncio.run(client.ensure_token(session))

    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))

    assert data == {"result": {"ok": True}}
    assert len(session.requests_to("/oauth2/token")) == 2
    assert _toss_get_auths(session) == ["Bearer tok-A", "Bearer tok-B"]


def test_toss_get_http_401_refreshes_once(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _toss_tmp_client(tmp_path, monkeypatch)
    session = _toss_get_session(
        [
            FakeBrokerResponse(body={}, status=401),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ],
        token_bodies=[
            {"access_token": "tok-A", "expires_in": 86400},
            {"access_token": "tok-B", "expires_in": 86400},
        ],
    )
    asyncio.run(client.ensure_token(session))

    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))

    assert data == {"result": {"ok": True}}
    assert len(session.requests_to("/oauth2/token")) == 2
    assert _toss_get_auths(session) == ["Bearer tok-A", "Bearer tok-B"]


def test_toss_get_refresh_does_not_consume_attempts(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _toss_tmp_client(tmp_path, monkeypatch)
    session = _toss_get_session(
        [
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429),
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429),
            FakeBrokerResponse(body={}, status=401),
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429),
        ],
        token_bodies=[
            {"access_token": "tok-A", "expires_in": 86400},
            {"access_token": "tok-B", "expires_in": 86400},
        ],
    )
    sleeps = _toss_record_sleep(monkeypatch)
    asyncio.run(client.ensure_token(session))

    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART", max_retries=3))

    assert data == {"error": {"code": "rate-limit"}}
    assert len(session.requests_to("/api/v1/candles")) == 4
    assert sleeps == [1.2, 1.2]
    assert len(session.requests_to("/oauth2/token")) == 2


def test_toss_get_replay_rate_limit_honours_retry_after(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _toss_tmp_client(tmp_path, monkeypatch)
    session = _toss_get_session(
        [
            FakeBrokerResponse(body={}, status=401),
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429, headers={"Retry-After": "3"}),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ],
        token_bodies=[
            {"access_token": "tok-A", "expires_in": 86400},
            {"access_token": "tok-B", "expires_in": 86400},
        ],
    )
    sleeps = _toss_record_sleep(monkeypatch)
    asyncio.run(client.ensure_token(session))

    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))

    assert data == {"result": {"ok": True}}
    assert len(session.requests_to("/api/v1/candles")) == 3
    assert sleeps == [3.0]


def test_toss_get_acquires_before_every_send(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _toss_tmp_client(tmp_path, monkeypatch)
    session = _toss_get_session(
        [
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429),
            FakeBrokerResponse(body={}, status=401),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ],
        token_bodies=[
            {"access_token": "tok-A", "expires_in": 86400},
            {"access_token": "tok-B", "expires_in": 86400},
        ],
    )
    acquires = _toss_count_acquires(monkeypatch)
    _toss_record_sleep(monkeypatch)
    asyncio.run(client.ensure_token(session))

    asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))

    assert acquires["n"] == len(session.requests_to("/api/v1/candles")) == 3


def test_toss_get_concurrent_rejections_issue_once(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.shared_token import SharedTokenStore
    from tests.broker_fakes import FakeBrokerResponse

    client = _toss_tmp_client(tmp_path, monkeypatch)
    _toss_count_acquires(monkeypatch)
    session = _toss_get_session(
        [
            FakeBrokerResponse(body={"error": {"code": "invalid-token"}}, enter_delay=0.01),
            FakeBrokerResponse(body={"error": {"code": "invalid-token"}}, enter_delay=0.01),
            FakeBrokerResponse(body={"result": {"ok": True}}),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ],
        token_bodies=[
            {"access_token": "tok-A", "expires_in": 86400},
            {"access_token": "tok-B", "expires_in": 86400},
            {"access_token": "tok-C", "expires_in": 86400},
        ],
    )
    asyncio.run(client.ensure_token(session))
    store: SharedTokenStore = client._token_store()
    assert store.read() is not None and store.read().generation == 1  # type: ignore[union-attr]

    async def _main():  # type: ignore[no-untyped-def]
        return await asyncio.gather(
            client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"),
            client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"),
        )

    data_a, data_b = asyncio.run(_main())

    assert data_a == {"result": {"ok": True}}
    assert data_b == {"result": {"ok": True}}
    assert len(session.requests_to("/oauth2/token")) == 2
    assert store.read() is not None and store.read().generation == 2  # type: ignore[union-attr]
    assert sorted(_toss_get_auths(session)) == ["Bearer tok-A", "Bearer tok-A", "Bearer tok-B", "Bearer tok-B"]


def test_toss_get_transport_error_propagates(tmp_path, monkeypatch) -> None:
    import asyncio

    import aiohttp
    import pytest

    from tests.broker_fakes import FakeBrokerResponse

    client = _toss_tmp_client(tmp_path, monkeypatch)
    client.token = "tok"
    session = _toss_get_session([FakeBrokerResponse(enter_error=aiohttp.ClientConnectionError("boom"))])
    sleeps = _toss_record_sleep(monkeypatch)

    with pytest.raises(aiohttp.ClientConnectionError):
        asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))

    assert len(session.requests_to("/api/v1/candles")) == 1
    assert sleeps == []


def test_toss_get_retry_settings_honored(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.config import settings as settings_instance
    from tests.broker_fakes import FakeBrokerResponse

    monkeypatch.setattr(settings_instance, "TOSS_RATE_LIMIT_MAX_RETRIES", 2)
    monkeypatch.setattr(settings_instance, "TOSS_RATE_LIMIT_BACKOFF_SECONDS", 0.5)
    client = _toss_tmp_client(tmp_path, monkeypatch)
    client.token = "tok"
    session = _toss_get_session(
        [
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429),
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429),
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429),
        ]
    )
    sleeps = _toss_record_sleep(monkeypatch)

    asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))

    assert len(session.requests_to("/api/v1/candles")) == 2
    assert sleeps == [0.5]

    retry_after_session = _toss_get_session(
        [
            FakeBrokerResponse(body={"error": {"code": "rate-limit"}}, status=429, headers={"Retry-After": "2"}),
            FakeBrokerResponse(body={"result": {"ok": True}}),
        ]
    )
    sleeps.clear()
    data = asyncio.run(client._get(retry_after_session, "/api/v1/candles", "MARKET_DATA_CHART"))

    assert data == {"result": {"ok": True}}
    assert sleeps == [2.0]


def test_toss_get_unknown_group_raises_before_send(tmp_path, monkeypatch) -> None:
    import asyncio

    import pytest

    from tests.broker_fakes import FakeBrokerResponse

    client = _toss_tmp_client(tmp_path, monkeypatch)
    client.token = "tok"
    session = _toss_get_session([FakeBrokerResponse(body={"result": {"ok": True}})])

    with pytest.raises(ValueError, match="NOT_A_GROUP"):
        asyncio.run(client._get(session, "/api/v1/candles", "NOT_A_GROUP"))

    assert session.requests == []
