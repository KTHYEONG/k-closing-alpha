"""일일 수집(collect) 서비스 단위 테스트: 스칼라 파싱·검증·시나리오 선택·집계."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.daily import collect


def test_safe_float_converts_values() -> None:
    assert collect.safe_float(None) == 0.0
    assert collect.safe_float("1,234.5") == 1234.5
    assert collect.safe_float("abc") == 0.0
    assert collect.safe_float(7) == 7.0


def test_parse_market_index_rate_returns_zero_on_missing() -> None:
    assert collect.parse_market_index_rate(None) == 0.0
    assert collect.parse_market_index_rate({"rt_cd": "1"}) == 0.0
    assert collect.parse_market_index_rate({"rt_cd": "0", "output1": None}) == 0.0


def test_parse_market_index_rate_uses_rate_and_fallback() -> None:
    assert (
        collect.parse_market_index_rate(
            {"rt_cd": "0", "output1": {"bstp_nmix_prdy_ctrt": "1.25"}}
        )
        == 1.25
    )
    fallback = collect.parse_market_index_rate(
        {"rt_cd": "0", "output1": {"bstp_nmix_prpr": "100", "bstp_nmix_prdy_vrss": "2"}}
    )
    assert fallback == pytest.approx(2.04)


def test_validate_hts_id_raises_on_placeholder() -> None:
    with (
        patch.object(collect, "HTS_ID", "여기에 HTS ID를 입력"),
        pytest.raises(RuntimeError),
    ):
        collect._validate_hts_id()
    with patch.object(collect, "HTS_ID", "real-hts"):
        collect._validate_hts_id()  # should not raise


def _fake_client() -> SimpleNamespace:
    return SimpleNamespace(
        get_current_price=AsyncMock(
            return_value={
                "rt_cd": "0",
                "output": {
                    "stck_prpr": "10500",
                    "stck_oprc": "10000",
                    "stck_hgpr": "11000",
                    "stck_lwpr": "9000",
                    "acml_vol": "100000",
                    "prdy_ctrt": "5.00",
                    "lstn_stcn": "1000000",
                    "rprs_mrkt_kor_name": "KOSPI",
                    "hts_avls": "100000",
                    "acml_tr_pbmn": "12000000000",
                },
            }
        ),
        get_trade_strength=AsyncMock(
            return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]}
        ),
        get_investor_trend_estimate=AsyncMock(
            return_value={
                "rt_cd": "0",
                "output2": [{"frgn_fake_ntby_qty": "10000", "orgn_fake_ntby_qty": "5000"}],
            }
        ),
        get_program_net_buy=AsyncMock(
            return_value={
                "rt_cd": "0",
                "output": [{"whol_smtn_ntby_tr_pbmn": "500000000"}],
            }
        ),
    )


async def _run_fetch_single_stock(client, **scenario_sets) -> tuple[dict, list[str]]:
    sem = asyncio.Semaphore(2)
    stock = {"code": "005930", "name": "삼성전자", "price": "10000", "chgrate": "1.0"}
    with patch(
        "src.api.kis_client.calculate_all_moving_averages",
        new=AsyncMock(
            return_value=(
                {5: 10000, 10: 10000, 20: 10000},
                (10000.0, True, 300),
                (10000.0, True),
                (10000.0, True),
            )
        ),
    ):
        return await collect.fetch_single_stock(
            0, stock, 1, sem, client, None, **scenario_sets
        )


def test_fetch_single_stock_sangdda_scenario() -> None:
    result, failed = asyncio.run(
        _run_fetch_single_stock(_fake_client(), upper_limit_stock_codes={"005930"})
    )
    assert failed == []
    assert result["시나리오"] == "상따"
    assert result["종목코드"] == "005930"


def test_fetch_single_stock_default_scenario_volume_surge() -> None:
    result, failed = asyncio.run(_run_fetch_single_stock(_fake_client()))
    assert failed == []
    assert result["시나리오"] == "거래량 폭증"


def test_fetch_all_stock_data_aggregates_and_reports_failures() -> None:
    async def _fake_single_stock(*args, **kwargs) -> tuple[dict, list[str]]:
        return ({"종목명": "AAA", "종목코드": "005930"}, ["체결강도"])

    with patch.object(collect, "fetch_single_stock", side_effect=_fake_single_stock):
        results, failed = asyncio.run(
            collect.fetch_all_stock_data(
                [{"code": "005930", "name": "AAA"}],
                None,
                None,
                set(),
                set(),
                set(),
                set(),
                set(),
            )
        )
    assert len(results) == 1
    assert results[0]["종목명"] == "AAA"
    assert failed == [("AAA", "005930", ["체결강도"])]
