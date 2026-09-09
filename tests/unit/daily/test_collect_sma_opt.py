"""Unit tests for daily collect SMA120 prefetch optimization.

Scenarios:
- T03: Phase A 분류 정확성 (1차 시나리오 확정 종목은 sma_needed_codes에서 제외)
- T05: ohlcv_cache miss 시 fetch_single_stock에서 예외 없이 API fallback 실행
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from src.daily.collect import fetch_all_stock_data, fetch_single_stock


class _FakeSession:
    """네트워크 접속 없는 가짜 aiohttp 세션."""


class _FakeClient:
    """4대 실시간 API를 모킹한 최소 KisApiClient 대역."""

    def __init__(self, responses=None) -> None:
        responses = responses or _base_responses()
        self.get_current_price = AsyncMock(return_value=responses["detail"])
        self.get_trade_strength = AsyncMock(return_value=responses["strength"])
        self.get_investor_trend_estimate = AsyncMock(
            return_value=responses["investor"]
        )
        self.get_program_net_buy = AsyncMock(return_value=responses["program"])


def _detail(close: int = 10500, open_: int = 10000, rate: float = 5.0) -> dict:
    return {
        "rt_cd": "0",
        "output": {
            "stck_prpr": str(close),
            "stck_oprc": str(open_),
            "stck_hgpr": str(close + 100),
            "stck_lwpr": str(open_ - 100),
            "acml_vol": "10000",
            "prdy_ctrt": str(rate),
            "lstn_stcn": "1000000",
            "rprs_mrkt_kor_name": "KOSPI",
            "hts_avls": "100000",
            "acml_tr_pbmn": "5000000000",
        },
    }


def _base_responses() -> dict:
    return {
        "detail": _detail(),
        "strength": {"rt_cd": "0", "output": [{"tday_rltv": "120.0"}]},
        "investor": {
            "rt_cd": "0",
            "output2": [{"frgn_fake_ntby_qty": "1000", "orgn_fake_ntby_qty": "2000"}],
        },
        "program": {
            "rt_cd": "0",
            "output": [{"whol_smtn_ntby_tr_pbmn": "100000000"}],
        },
    }


def _stock(code: str) -> dict:
    return {"code": code, "name": f"종목{code}", "price": "10500", "chgrate": "5.0"}


def _run_fetch(client, *, scenario_sets=None, ohlcv_cache=None):
    scenario_sets = scenario_sets or {}

    async def _runner():
        stock = _stock("005930")
        sem = asyncio.Semaphore(1)
        return await fetch_single_stock(
            0,
            stock,
            1,
            sem,
            client,
            None,
            overheated_stock_codes=set(),
            new_high_stock_codes=scenario_sets.get("new_high", set()),
            near_new_high_stock_codes=scenario_sets.get("near_new_high", set()),
            upper_limit_next_day_stock_codes=scenario_sets.get("upper_next", set()),
            upper_limit_stock_codes=scenario_sets.get("upper", set()),
            ohlcv_cache=ohlcv_cache,
        )

    return asyncio.run(_runner())


def _run_fetch_all(stock_list, client, **scenario_sets):
    async def _runner():
        return await fetch_all_stock_data(
            stock_list,
            client,
            _FakeSession(),
            overheated_stock_codes=set(),
            new_high_stock_codes=scenario_sets.get("new_high", set()),
            near_new_high_stock_codes=scenario_sets.get("near_new_high", set()),
            upper_limit_next_day_stock_codes=scenario_sets.get("upper_next", set()),
            upper_limit_stock_codes=scenario_sets.get("upper", set()),
        )

    return asyncio.run(_runner())


# ---------------------------------------------------------
# T03: Phase A 분류 정확성
# ---------------------------------------------------------




# ---------------------------------------------------------
# T05: ohlcv_cache hit/miss
# ---------------------------------------------------------


