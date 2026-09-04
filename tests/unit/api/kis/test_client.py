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


def test_get_orderbook_snapshot_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_orderbook_snapshot(_FakeSession(), "005930")

    asyncio.run(_runner())


def test_get_orderbook_snapshot_returns_raw_output() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    handle_request = AsyncMock(return_value={
        "rt_cd": "0",
        "output1": {"askp1": "70100", "bidp1": "70000", "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500"},
        "output2": {},
    })

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_orderbook_snapshot(_FakeSession(), "005930", market_div_code="J")

    res = asyncio.run(_runner())
    assert res["output1"]["askp1"] == "70100"
    params = handle_request.await_args.kwargs.get("params", {})
    assert params.get("FID_COND_MRKT_DIV_CODE") == "J"
    assert params.get("FID_INPUT_ISCD") == "005930"


def test_get_intraday_trade_ticks_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_intraday_trade_ticks(_FakeSession(), "005930")

    asyncio.run(_runner())


def test_get_intraday_trade_ticks_dedupes_by_acml_vol_not_hour() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    page1 = {"rt_cd": "0", "output2": [
        {"stck_cntg_hour": "093000", "acml_vol": "1000", "stck_prpr": "70000"},
        {"stck_cntg_hour": "093000", "acml_vol": "990", "stck_prpr": "69900"},
    ]}
    page2 = {"rt_cd": "0", "output2": [
        {"stck_cntg_hour": "093000", "acml_vol": "990", "stck_prpr": "69900"},
        {"stck_cntg_hour": "090000", "acml_vol": "100", "stck_prpr": "69000"},
    ]}
    handle_request = AsyncMock(side_effect=[page1, page2])

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_intraday_trade_ticks(
                _FakeSession(), "005930", floor_hour="090000", end_hour="153000", market_div_code="J",
            )

    res = asyncio.run(_runner())
    acml_vols = {row["acml_vol"] for row in res["output2"]}
    assert len(res["output2"]) == 3
    assert acml_vols == {"1000", "990", "100"}


def test_get_daily_short_sale_history_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_daily_short_sale_history(_FakeSession(), "005930", "20240101", "20240110")

    asyncio.run(_runner())


def test_get_daily_short_sale_history_returns_raw_output_with_date_range_params() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    handle_request = AsyncMock(return_value={"rt_cd": "0", "output2": [
        {"stck_bsop_date": "20240103", "ssts_cntg_qty": "100"},
        {"stck_bsop_date": "20240102", "ssts_cntg_qty": "90"},
    ]})

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_daily_short_sale_history(
                _FakeSession(), "005930", "20240101", "20240110", market_div_code="J",
            )

    res = asyncio.run(_runner())
    dates = [row["stck_bsop_date"] for row in res["output2"]]
    assert dates == ["20240102", "20240103"]
    params = handle_request.await_args.kwargs.get("params", {})
    assert params.get("FID_INPUT_DATE_1") == "20240101"
