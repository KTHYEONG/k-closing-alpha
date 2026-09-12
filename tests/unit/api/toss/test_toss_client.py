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
