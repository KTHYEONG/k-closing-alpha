from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from src.api.kis_client import KisApiClient


class _FakeSession:
    def get(self, *args, **kwargs):
        raise AssertionError("세션의 실제 HTTP 메서드는 호출되지 않아야 한다 (직접 patch됨)")


def test_get_intraday_minute_chart_paginates_within_floor_and_ceil() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    page1 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "200000"}, {"stck_cntg_hour": "154500"}]}
    page2 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "154000"}]}
    handle_request = AsyncMock(side_effect=[page1, page2])

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_intraday_minute_chart(
                _FakeSession(), "005930",
                end_hour="200000", floor_hour="154000", market_div_code="NX",
            )

    result = asyncio.run(_runner())

    assert result["rt_cd"] == "0"
    assert len(result["output2"]) == 3
    for call in handle_request.await_args_list:
        params = call.kwargs.get("params", {})
        assert params.get("FID_COND_MRKT_DIV_CODE") == "NX"

def test_get_historical_minute_chart_paginates_with_target_date() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    page1 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "153000"}, {"stck_cntg_hour": "152900"}]}
    page2 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "090000"}]}
    handle_request = AsyncMock(side_effect=[page1, page2])

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_historical_minute_chart(
                _FakeSession(), "005930", "20250815",
                end_hour="153000", floor_hour="090000", market_div_code="J",
            )

    result = asyncio.run(_runner())

    assert result["rt_cd"] == "0"
    assert len(result["output2"]) == 3
    for call in handle_request.await_args_list:
        params = call.kwargs.get("params", {})
        assert params.get("FID_INPUT_DATE_1") == "20250815"
        assert params.get("FID_COND_MRKT_DIV_CODE") == "J"


def test_get_historical_minute_chart_returns_raw_failure_outside_retention() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    failure = {"rt_cd": "9", "msg1": "조회 가능 기간을 초과하였습니다"}
    handle_request = AsyncMock(return_value=failure)

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_historical_minute_chart(
                _FakeSession(), "005930", "20200101", market_div_code="J",
            )

    result = asyncio.run(_runner())

    assert result["rt_cd"] == "9"
    handle_request.assert_awaited_once()
