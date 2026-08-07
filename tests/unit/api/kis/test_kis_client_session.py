from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from src.api.kis_client import KisApiClient


def test_create_session_uses_bounded_timeout() -> None:
    async def run() -> None:
        session = KisApiClient().create_session()
        try:
            assert session.timeout.total == 60
            assert session.timeout.connect == 10
            assert session.timeout.sock_read == 30
        finally:
            await session.close()

    asyncio.run(run())


def test_client_quote_helpers_reuse_common_request_path() -> None:
    async def run() -> None:
        client = KisApiClient(app_key="key", app_secret="secret")
        client.token = "token"
        client._handle_request = AsyncMock(return_value={"rt_cd": "0", "output": []})
        session = type("Session", (), {"get": object()})()
        assert (await client.get_current_price(session, "005930"))["rt_cd"] == "0"
        assert (await client.get_program_net_buy(session, "005930"))["rt_cd"] == "0"
        assert (await client.get_trade_strength(session, "005930"))["rt_cd"] == "0"
        assert (await client.get_market_index_rate(session, "0001"))["rt_cd"] == "0"
        assert (await client.get_market_index_history(session, "0001", "20200101", "20200102"))["rt_cd"] == "0"
        assert (await client.get_investor_trend_estimate(session, "005930"))["rt_cd"] == "0"
        headers = client._get_headers("TEST")
        assert headers["authorization"] == "Bearer token"
        assert client._market_div_candidates("bad") == ["UN", "J", "NX"]

    asyncio.run(run())
