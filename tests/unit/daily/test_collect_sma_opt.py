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
def test_phase_a_excludes_primary_matched_codes() -> None:
    """[T03] 1차 시나리오 확정 종목은 sma_needed_codes에서 제외된다."""
    stock_list = [_stock(c) for c in ("000001", "000002", "000003", "000004", "000005")]
    client = _FakeClient()
    prefetch_mock = AsyncMock(return_value={})
    single_mock = AsyncMock(return_value=({}, []))

    with (
        patch("src.daily.collect.prefetch_ohlcv_for_sma120", new=prefetch_mock),
        patch("src.daily.collect.fetch_single_stock", new=single_mock),
    ):
        _run_fetch_all(
            stock_list,
            client,
            new_high={"000001", "000002"},
            upper_next={"000003"},
        )

    prefetch_mock.assert_awaited_once()
    called = prefetch_mock.await_args.args[0]
    assert "000001" not in called
    assert "000002" not in called
    assert "000003" not in called
    assert set(called) == {"000004", "000005"}


def test_phase_a_all_primary_skips_prefetch() -> None:
    """[T03] 모든 종목이 1차 매칭이면 prefetch 호출 없이 빈 cache 전달."""
    stock_list = [_stock(c) for c in ("000001", "000002")]
    client = _FakeClient()
    prefetch_mock = AsyncMock(return_value={})
    single_mock = AsyncMock(return_value=({}, []))

    with (
        patch("src.daily.collect.prefetch_ohlcv_for_sma120", new=prefetch_mock),
        patch("src.daily.collect.fetch_single_stock", new=single_mock),
    ):
        _run_fetch_all(stock_list, client, upper={"000001", "000002"})

    prefetch_mock.assert_not_called()
    call_args = single_mock.await_args
    assert call_args.args[-1] == {}


# ---------------------------------------------------------
# T05: ohlcv_cache hit/miss
# ---------------------------------------------------------
def test_cache_miss_falls_back_to_api() -> None:
    """[T05] ohlcv_cache miss 시 예외 없이 prefetched_records=None으로 fallback."""
    client = _FakeClient(_base_responses())
    ma_calc = AsyncMock(return_value=({}, (0.0, False, 0), (0.0, False), (10200.0, True)))

    with patch("src.api.kis_client.calculate_all_moving_averages", new=ma_calc):
        row, failed = _run_fetch(client, scenario_sets={}, ohlcv_cache={})

    assert row["시나리오"] == "120 돌파"
    assert failed == []
    assert ma_calc.await_count == 1
    assert ma_calc.await_args.kwargs["prefetched_records"] is None


def test_cache_hit_passes_prefetched_records() -> None:
    """[T05] ohlcv_cache 히트 시 prefetched_records가 계산 함수로 전달된다."""
    client = _FakeClient(_base_responses())
    records = [{"date": f"2025{i:04d}", "close": "50000"} for i in range(1, 151)]
    ma_calc = AsyncMock(return_value=({}, (0.0, False, 150), (0.0, False), (10200.0, True)))

    with patch("src.api.kis_client.calculate_all_moving_averages", new=ma_calc):
        row, failed = _run_fetch(client, scenario_sets={}, ohlcv_cache={"005930": records})

    assert row["시나리오"] == "120 돌파"
    assert failed == []
    assert ma_calc.await_count == 1
    assert ma_calc.await_args.kwargs["prefetched_records"] == records
