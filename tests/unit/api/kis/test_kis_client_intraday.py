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


def test_decrement_hour_one_second() -> None:
    from src.api.kis.client import KisApiClient
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    assert client._decrement_hour_one_second("151906") == "151905"
    assert client._decrement_hour_one_second("150000") == "145959"
    assert client._decrement_hour_one_second("090000") == "085959"


def test_get_intraday_trade_ticks_handles_cursor_stall_with_second_decrement() -> None:
    import asyncio
    from unittest.mock import AsyncMock, patch
    from src.api.kis.client import KisApiClient
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    page1 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "151906", "acml_vol": "200", "stck_prpr": "1000"}, {"stck_cntg_hour": "151906", "acml_vol": "190", "stck_prpr": "1000"}]}
    page2 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "151906", "acml_vol": "200", "stck_prpr": "1000"}]}
    page3 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "151905", "acml_vol": "180", "stck_prpr": "990"}]}
    page4 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "090000", "acml_vol": "10", "stck_prpr": "950"}]}
    handle_request = AsyncMock(side_effect=[page1, page2, page3, page4])
    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_intraday_trade_ticks(
                None, "005930", floor_hour="090000", end_hour="153000", market_div_code="J", max_pages=10
            )
    res = asyncio.run(_runner())
    assert res["rt_cd"] == "0"
    hours = [r["stck_cntg_hour"] for r in res["output2"]]
    assert "151906" in hours
    assert "151905" in hours
    assert "090000" in hours


def test_get_intraday_trade_ticks_includes_closing_auction_tick() -> None:
    import asyncio
    from unittest.mock import AsyncMock, patch
    from src.api.kis.client import KisApiClient
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    page1 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "153000", "acml_vol": "5000", "stck_prpr": "10000"}, {"stck_cntg_hour": "151959", "acml_vol": "4900", "stck_prpr": "9980"}]}
    page2 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "090000", "acml_vol": "100", "stck_prpr": "9500"}]}
    handle_request = AsyncMock(side_effect=[page1, page2])
    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_intraday_trade_ticks(
                None, "005930", floor_hour="090000", end_hour="153000", market_div_code="J", max_pages=10
            )
    res = asyncio.run(_runner())
    assert res["rt_cd"] == "0"
    hours = [r["stck_cntg_hour"] for r in res["output2"]]
    assert "153000" in hours
    first_call_params = handle_request.await_args_list[0].kwargs["params"]
    assert first_call_params["FID_INPUT_HOUR_1"] in ("", "153001")


def test_decrement_hour_one_second_returns_input_on_parse_failure() -> None:
    from src.api.kis.client import KisApiClient
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    assert client._decrement_hour_one_second("not-a-time") == "not-a-time"
    assert client._decrement_hour_one_second("") == ""


def test_get_intraday_trade_ticks_stops_without_decrement_when_base_at_floor() -> None:
    import asyncio
    from unittest.mock import AsyncMock, patch
    from src.api.kis.client import KisApiClient
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    # acml_vol이 없는 행은 vol_key가 빈 문자열이라 new_rows에서 걸러져 빈 배치가 되고,
    # cursor_hour(=end_hour="090000")가 이미 floor_hour와 같아 감산 없이 즉시 종료된다.
    page1 = {"rt_cd": "0", "output2": [{"stck_cntg_hour": "090000", "stck_prpr": "9500"}]}
    handle_request = AsyncMock(side_effect=[page1])
    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_intraday_trade_ticks(
                None, "005930", floor_hour="090000", end_hour="090000", market_div_code="J", max_pages=10
            )
    res = asyncio.run(_runner())
    assert res["rt_cd"] == "0"
    assert res["output2"] == []
    assert handle_request.await_count == 1
