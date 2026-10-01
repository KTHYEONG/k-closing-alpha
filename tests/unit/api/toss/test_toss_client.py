from __future__ import annotations


def test_toss_client_ensure_token() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"access_token": "mock_tok", "token_type": "Bearer", "expires_in": 86400})
    session = AsyncMock()
    session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.post.return_value.__aexit__ = AsyncMock(return_value=False)

    token = asyncio.run(client.ensure_token(session))

    assert token == "mock_tok"
    assert client.token == "mock_tok"


def test_toss_client_ensure_token_concurrent_calls_lock_and_request_once() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    call_count = {"n": 0}

    async def fake_post(*args, **kwargs):
        call_count["n"] += 1
        await asyncio.sleep(0.01)
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"access_token": "tok_123", "token_type": "Bearer", "expires_in": 86400})
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = AsyncMock()
    session.post = fake_post

    async def runner():
        tokens = await asyncio.gather(*[client.ensure_token(session) for _ in range(10)])
        assert all(t == "tok_123" for t in tokens)

    asyncio.run(runner())
    assert call_count["n"] == 1


def test_toss_client_ensure_token_raises_on_empty_token() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import pytest

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"error": {"code": "invalid-client", "message": "bad credentials"}})
    session = AsyncMock()
    session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.post.return_value.__aexit__ = AsyncMock(return_value=False)

    with pytest.raises(RuntimeError, match="Toss token issuance failed"):
        asyncio.run(client.ensure_token(session))


def test_toss_client_get_program_trades_parses_result_envelope() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"result": {"records": [
        {"date": "2026-09-10", "arbitrage": {"netBuyVolume": "3"}, "nonArbitrage": {"netBuyVolume": "4"}}
    ]}})
    session = AsyncMock()
    session.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.get.return_value.__aexit__ = AsyncMock(return_value=False)

    data = asyncio.run(client.get_program_trades(session, "005930", count=30, until="2026-09-10"))

    assert data["result"]["records"][0]["arbitrage"]["netBuyVolume"] == "3"
    called_args, called_kwargs = session.get.call_args
    assert called_args[0] == "https://openapi.tossinvest.com/api/v1/stocks/005930/program-trades"
    assert called_kwargs["params"] == {"count": 30, "until": "2026-09-10"}


def test_toss_client_get_program_trades_retries_on_429_then_succeeds(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import src.api.toss.client as toss_client_mod
    from src.api.kis.rate_limit import AsyncRateLimiter

    client = toss_client_mod.TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    async def _fast_acquire(self) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(AsyncRateLimiter, "acquire", _fast_acquire)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(seconds) -> None:
        await _real_sleep(0)

    monkeypatch.setattr(toss_client_mod.asyncio, "sleep", _fast_sleep)

    mock_resp_429 = AsyncMock()
    mock_resp_429.status = 429
    mock_resp_429.json = AsyncMock(return_value={"error": {"code": "rate-limit-exceeded", "message": "too many requests"}})
    mock_resp_200 = AsyncMock()
    mock_resp_200.status = 200
    mock_resp_200.json = AsyncMock(return_value={"result": {"records": []}})

    session = AsyncMock()
    session.get.return_value.__aenter__ = AsyncMock(side_effect=[mock_resp_429, mock_resp_200])
    session.get.return_value.__aexit__ = AsyncMock(return_value=False)

    data = asyncio.run(client.get_program_trades(session, "005930"))

    assert data == {"result": {"records": []}}
    assert session.get.call_count == 2


def test_toss_client_unknown_rate_limit_group_raises() -> None:
    import pytest

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")

    with pytest.raises(ValueError, match="NOT_A_GROUP"):
        client._limiter_for("NOT_A_GROUP")


def test_toss_client_get_rankings_parses_result_envelope() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"result": {"rankedAt": "2026-09-11T19:59:46.466+09:00", "rankings": [
        {"rank": 1, "symbol": "000660", "price": {"lastPrice": "1832000", "changeRate": "-0.0113"}, "tradingVolume": "79878", "tradingAmount": "145941219000"}
    ]}})
    session = AsyncMock()
    session.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.get.return_value.__aexit__ = AsyncMock(return_value=False)

    data = asyncio.run(client.get_rankings(session, ranking_type="TOP_GAINERS", market_country="KR", duration="1d", count=100))

    assert data["result"]["rankings"][0]["symbol"] == "000660"
    called_args, called_kwargs = session.get.call_args
    assert called_args[0] == "https://openapi.tossinvest.com/api/v1/rankings"
    assert called_kwargs["params"] == {"type": "TOP_GAINERS", "marketCountry": "KR", "duration": "1d", "count": 100}


def test_toss_client_get_rankings_forwards_exclude_investment_caution() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"result": {"rankedAt": "2026-09-11T19:59:46.466+09:00", "rankings": []}})
    session = AsyncMock()
    session.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.get.return_value.__aexit__ = AsyncMock(return_value=False)

    asyncio.run(client.get_rankings(session, ranking_type="TOP_GAINERS", exclude_investment_caution=True))

    _, called_kwargs = session.get.call_args
    assert called_kwargs["params"]["excludeInvestmentCaution"] == "true"


def test_toss_client_get_candles_parses_result_envelope() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"result": {"candles": [
        {"timestamp": "2024-02-28T15:30:00.000+09:00", "openPrice": "70000", "highPrice": "70100", "lowPrice": "69900", "closePrice": "70000", "volume": "10", "currency": "KRW"}
    ]}})
    session = AsyncMock()
    session.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.get.return_value.__aexit__ = AsyncMock(return_value=False)

    data = asyncio.run(client.get_candles(session, "000250", interval="1m", count=200, before="2024-02-28T15:30:00.000+09:00"))

    assert data["result"]["candles"][0]["closePrice"] == "70000"
    called_args, called_kwargs = session.get.call_args
    assert called_args[0] == "https://openapi.tossinvest.com/api/v1/candles"
    assert called_kwargs["params"] == {"symbol": "000250", "interval": "1m", "count": 200, "before": "2024-02-28T15:30:00.000+09:00"}


def test_toss_client_get_candles_forwards_adjusted_flag_and_omits_before_when_absent() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.api.toss.client import TossApiClient

    client = TossApiClient(app_key="k", app_secret="s")
    client.token = "tok"

    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"result": {"candles": []}})
    session = AsyncMock()
    session.get.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.get.return_value.__aexit__ = AsyncMock(return_value=False)

    asyncio.run(client.get_candles(session, "005930", adjusted=False))

    _, called_kwargs = session.get.call_args
    assert called_kwargs["params"]["adjusted"] == "false"
    assert "before" not in called_kwargs["params"]


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


def _toss_session(get_bodies, token_bodies=None, headers_list=None):
    from unittest.mock import AsyncMock

    state = {"gets": 0, "posts": 0, "auths": []}

    def _resp(body, status=200, headers=None):
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=body)
        mock_resp.headers = dict(headers or {})
        return mock_resp

    async def _fake_get(url, headers=None, params=None):
        idx = state["gets"]
        state["gets"] += 1
        state["auths"].append((headers or {}).get("Authorization"))
        bodies = get_bodies[idx]
        if len(bodies) == 3:
            body, status, headers = bodies
        else:
            body, status = bodies
            headers = (headers_list[idx] if headers_list and idx < len(headers_list) else {})
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=_resp(body, status, headers))
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    async def _fake_post(url, data=None, **kw):
        state["posts"] += 1
        fallbacks = token_bodies or [{"access_token": "new-tok", "expires_in": 86400}]
        payload = fallbacks[min(state["posts"] - 1, len(fallbacks) - 1)]
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=_resp(payload, 200))
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    from unittest.mock import AsyncMock as _AM

    session = _AM()
    session.get.side_effect = _fake_get
    session.post.side_effect = _fake_post
    return session, state


def test_toss_invalid_token_recovers_once(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.toss.client import TossApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")

    client = TossApiClient(app_key="k", app_secret="s")
    session, state = _toss_session(
        [({"error": {"code": "invalid-token"}}, 200), ({"result": {"ok": True}}, 200)],
        [{"access_token": "tok-B", "expires_in": 86400}],
    )

    async def _seed():
        await client.ensure_token(session)

    asyncio.run(_seed())
    from src.api.shared_token import SharedTokenStore

    before = client._token_store().read()
    assert before is not None and before.generation == 1

    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))
    assert data == {"result": {"ok": True}}
    assert state["posts"] == 2
    after = client._token_store().read()
    assert after is not None and after.generation == before.generation + 1
    assert state["auths"][-1] == f"Bearer {after.access_token}"


def test_toss_rotation_by_peer_is_adopted(tmp_path, monkeypatch) -> None:
    import asyncio
    from datetime import UTC, datetime

    from src.api.shared_token import IssuedToken, SharedTokenStore, shared_token_path
    from src.api.toss.client import TossApiClient
    from src.config import settings as settings_instance

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
    session, state = _toss_session([({"error": {"code": "invalid-token"}}, 200), ({"result": {"ok": True}}, 200)])
    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))
    assert data == {"result": {"ok": True}}
    assert state["posts"] == 0
    assert state["auths"] == ["Bearer tok-A", "Bearer tok-B"]


def test_toss_second_rejection_surfaces(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.toss.client import TossApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")

    client = TossApiClient(app_key="k", app_secret="s")
    session, state = _toss_session(
        [({"error": {"code": "invalid-token"}}, 200), ({"error": {"code": "invalid-token"}}, 200)],
        [{"access_token": "tok-B", "expires_in": 86400}],
    )
    asyncio.run(client.ensure_token(session))
    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))
    assert data == {"error": {"code": "invalid-token"}}
    assert state["posts"] == 2


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

    monkeypatch.setattr(toss_client_mod.asyncio, "sleep", _fake_sleep)

    async def _fake_acquire(self):
        return None

    monkeypatch.setattr(toss_client_mod.HostPacedRateLimiter, "acquire", _fake_acquire)

    from unittest.mock import AsyncMock

    def _resp(body, status=200, headers=None):
        mock_resp = AsyncMock()
        mock_resp.status = status
        mock_resp.json = AsyncMock(return_value=body)
        mock_resp.headers = dict(headers or {})
        return mock_resp

    r429 = _resp({"error": {"code": "rate-limit-exceeded"}}, 429, {"Retry-After": "2"})
    r200 = _resp({"result": {"ok": True}}, 200)
    session = AsyncMock()
    session.get.return_value.__aenter__ = AsyncMock(side_effect=[r429, r200])
    session.get.return_value.__aexit__ = AsyncMock(return_value=False)

    data = asyncio.run(client._get(session, "/api/v1/candles", "MARKET_DATA_CHART"))
    assert data == {"result": {"ok": True}}
    assert sleeps == [2.0]


def test_toss_retry_after_capped_for_decision_window() -> None:
    from src.api.toss.client import TOSS_RETRY_AFTER_MAX_SECONDS, TossApiClient

    assert TossApiClient._retry_after_seconds({"Retry-After": "600"}) == TOSS_RETRY_AFTER_MAX_SECONDS
    assert TossApiClient._retry_after_seconds({"retry-after": "1.5"}) == 1.5
