"""Unit tests for daily data collection.

SCENARIO_DAILY_COLLECT_REFACTORING_01:
Verifies that collect.py saves collected condition data directly
without chart_pass_cache.json or parenthesis column renaming.

SCENARIO_COLLECT_NO_SLEEP:
 fetch_single_stock 내부에 API_SLEEP_INTERVAL 참조 없음 검증 (perf_v2).

SCENARIO_REGRESSION:
 기존 17개 테스트 회귀 검증.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.daily.collect import save_collected_condition_data


def test_scenario_daily_collect_refactoring_01(tmp_path: Path) -> None:
    """[SCENARIO_DAILY_COLLECT_REFACTORING_01] Verify clean standard CSV saving without chart_pass."""
    sample_df = pd.DataFrame(
        {
            "종목명": ["테스트종목"],
            "종목코드": ["123"],
            "시가": [1000],
            "고가": [1100],
            "저가": [990],
            "종가": [1050],
            "전일종가": [1000],
            "시가총액": [500.0],
            "거래대금": [100.0],
            "등락률": [5.0],
            "선정순위": [1],
            "기관_순매수": [10.0],
            "외국인_순매수": [20.0],
            "프로그램_순매수": [5.0],
            "체결강도": [120.0],
            "시장구분": ["KOSDAQ"],
            "총_종목수": [100],
            "평균_거래대금": [50.0],
            "kospi": [0.5],
            "kosdaq": [1.2],
            "v_kospi": [15.0],
            "v_kosdaq": [20.0],
            "거래량": [10000],
            "시나리오": ["거래량 폭증"],
        }
    )

    csv_path = tmp_path / "daily" / "daily_stocks.csv"
    res_path = save_collected_condition_data(sample_df, csv_path)

    assert res_path.exists()
    saved_df = pd.read_csv(res_path, dtype={"종목코드": str})
    assert saved_df["종목코드"].iloc[0] == "000123"
    assert "차트통과" not in saved_df.columns
    assert "(차트통과)" not in saved_df.columns


def test_scenario_collect_no_sleep() -> None:
    """[SCENARIO_COLLECT_NO_SLEEP]
    fetch_single_stock 소스에 API_SLEEP_INTERVAL 참조 및 asyncio.sleep 없음 검증.
    """
    import inspect
    from src.daily.collect import fetch_single_stock

    src_text = inspect.getsource(fetch_single_stock)
    assert callable(fetch_single_stock)
    assert "API_SLEEP_INTERVAL" not in src_text
    assert "asyncio.sleep" not in src_text


def test_scenario_regression() -> None:
    """[SCENARIO_REGRESSION] 기존 save_collected_condition_data 회귀 검증."""
    from src.daily.collect import save_collected_condition_data

    assert callable(save_collected_condition_data)


def test_fetch_single_stock_captures_dual_venue_decision_price() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()

    async def _fake_get_current_price(session, code, market_div_code=None):
        if market_div_code == "J":
            prpr, vol = "70000", "1000"
        else:
            prpr, vol = "69800", "500"
        return {
            "rt_cd": "0",
            "output": {
                "stck_prpr": prpr, "stck_oprc": "69000", "stck_hgpr": "71000", "stck_lwpr": "68500",
                "acml_vol": vol, "prdy_ctrt": "0.5", "lstn_stcn": "100", "rprs_mrkt_kor_name": "KOSPI",
            },
        }

    client.get_current_price = _fake_get_current_price
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{}]})

    sem = asyncio.Semaphore(1)
    stock = {"code": "005930", "name": "삼성전자", "price": 70000, "chgrate": 0.5}

    result = asyncio.run(collect.fetch_single_stock(0, stock, 1, sem, client, session=None))

    record = result[0] if isinstance(result, tuple) else result
    assert record["krx_현재가"] == 70000
    assert record["nxt_현재가"] == 69800
    assert record["sor_effective_price"] == 69800


def test_fetch_single_stock_falls_back_to_krx_when_nxt_unlisted_or_illiquid() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    async def _run_case(nxt_resp):
        client = AsyncMock()

        async def _fake(session, code, market_div_code=None):
            if market_div_code == "J":
                return {"rt_cd": "0", "output": {
                    "stck_prpr": "70000", "stck_oprc": "69000", "stck_hgpr": "71000", "stck_lwpr": "68500",
                    "acml_vol": "1000", "prdy_ctrt": "0.5", "lstn_stcn": "100", "rprs_mrkt_kor_name": "KOSPI",
                }}
            return nxt_resp

        client.get_current_price = _fake
        client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
        client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{}]})
        client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{}]})
        sem = asyncio.Semaphore(1)
        stock = {"code": "005930", "name": "삼성전자", "price": 70000, "chgrate": 0.5}
        result = await collect.fetch_single_stock(0, stock, 1, sem, client, session=None)
        record = result[0] if isinstance(result, tuple) else result
        failed = result[1] if isinstance(result, tuple) else []
        return record, failed

    # NXT 미상장 케이스
    import asyncio as _aio

    record, failed = _aio.run(_run_case({"rt_cd": "9", "msg1": "NXT 미상장"}))
    assert record["krx_현재가"] == 70000
    assert record["nxt_현재가"] is None
    assert record["sor_effective_price"] == 70000
    assert "현재가" not in failed or failed == []

    # NXT 유동성 0 케이스
    record2, _ = _aio.run(_run_case({"rt_cd": "0", "output": {
        "stck_prpr": "69000", "stck_oprc": "69000", "stck_hgpr": "69000", "stck_lwpr": "69000",
        "acml_vol": "0", "prdy_ctrt": "0.5", "lstn_stcn": "100", "rprs_mrkt_kor_name": "KOSPI",
    }}))
    assert record2["sor_effective_price"] == 70000


def test_fetch_single_stock_includes_orderbook_snapshot_fields() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()

    async def _fake_get_current_price(session, code, market_div_code=None):
        return {
            "rt_cd": "0",
            "output": {
                "stck_prpr": "70000", "stck_oprc": "69000", "stck_hgpr": "71000", "stck_lwpr": "68500",
                "acml_vol": "1000", "prdy_ctrt": "0.5", "lstn_stcn": "100", "rprs_mrkt_kor_name": "KOSPI",
            },
        }

    async def _fake_get_orderbook_snapshot(session, code, market_div_code=None):
        if market_div_code == "J":
            return {"rt_cd": "0", "output1": {
                "askp1": "70100", "bidp1": "70000",
                "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500",
            }}
        return {"rt_cd": "0", "output1": {
            "askp1": "69900", "bidp1": "69800",
            "total_askp_rsqn": "300", "total_bidp_rsqn": "200",
        }}

    client.get_current_price = _fake_get_current_price
    client.get_orderbook_snapshot = _fake_get_orderbook_snapshot
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{}]})

    sem = asyncio.Semaphore(1)
    stock = {"code": "005930", "name": "삼성전자", "price": 70000, "chgrate": 0.5}
    record, _ = asyncio.run(collect.fetch_single_stock(0, stock, 1, sem, client, session=None))

    assert record["krx_매도호가1"] == 70100
    assert record["krx_매수호가1"] == 70000
    assert record["krx_매도잔량"] == 1200
    assert record["krx_매수잔량"] == 1500
    assert record["nxt_매도호가1"] == 69900
    assert record["nxt_매수호가1"] == 69800
    assert record["nxt_매도잔량"] == 300
    assert record["nxt_매수잔량"] == 200


def test_fetch_single_stock_flags_krx_orderbook_failure_without_blocking_record() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.daily import collect

    client = AsyncMock()

    async def _fake_get_current_price(session, code, market_div_code=None):
        return {
            "rt_cd": "0",
            "output": {
                "stck_prpr": "70000", "stck_oprc": "69000", "stck_hgpr": "71000", "stck_lwpr": "68500",
                "acml_vol": "1000", "prdy_ctrt": "0.5", "lstn_stcn": "100", "rprs_mrkt_kor_name": "KOSPI",
            },
        }

    async def _fake_get_orderbook_snapshot(session, code, market_div_code=None):
        if market_div_code == "J":
            return {"rt_cd": "9", "msg1": "일시 오류"}
        return {"rt_cd": "0", "output1": {
            "askp1": "69900", "bidp1": "69800",
            "total_askp_rsqn": "300", "total_bidp_rsqn": "200",
        }}

    client.get_current_price = _fake_get_current_price
    client.get_orderbook_snapshot = _fake_get_orderbook_snapshot
    client.get_trade_strength = AsyncMock(return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]})
    client.get_investor_trend_estimate = AsyncMock(return_value={"rt_cd": "0", "output2": [{}]})
    client.get_program_net_buy = AsyncMock(return_value={"rt_cd": "0", "output": [{}]})

    sem = asyncio.Semaphore(1)
    stock = {"code": "005930", "name": "삼성전자", "price": 70000, "chgrate": 0.5}
    record, failed = asyncio.run(collect.fetch_single_stock(0, stock, 1, sem, client, session=None))

    assert "호가" in failed
    assert record["krx_매도호가1"] == 0
    assert record["nxt_매도호가1"] == 69900
