"""src.api.kis.client 모듈 직접 참조 테스트 (호환 파사드 src.api.kis_client 우회).

FID_FAKE_TICK_INCU_YN 누락으로 FHKST03010230 전체 호출이 실패했던 회귀
방지, 그리고 market_div_code 명시 요구(fail-closed) 계약 검증.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.api.kis.client import KisApiClient


class _FakeSession:
    def get(self, *args, **kwargs):
        raise AssertionError("직접 patch된 _handle_request만 사용되어야 한다")


def test_get_historical_minute_chart_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_historical_minute_chart(_FakeSession(), "005930", "20260901")

    asyncio.run(_runner())


def test_get_intraday_minute_chart_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_intraday_minute_chart(_FakeSession(), "005930")

    asyncio.run(_runner())


def test_get_historical_minute_chart_includes_fake_tick_field() -> None:
    """FID_FAKE_TICK_INCU_YN 필드 키 누락 시 KIS가 OPSQ2001로 전체 거부하던 회귀 방지."""
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    handle_request = AsyncMock(return_value={"rt_cd": "0", "output2": []})

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_historical_minute_chart(
                _FakeSession(), "005930", "20260901", market_div_code="J",
            )

    asyncio.run(_runner())

    params = handle_request.await_args.kwargs.get("params", {})
    assert "FID_FAKE_TICK_INCU_YN" in params
