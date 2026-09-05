"""Unit test for daily collection lazy evaluation (SCENARIO_COLLECT_LAZY_EVALUATION_01)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from src.daily.collect import fetch_single_stock


class _FakeClient:
    """4대 실시간 API를 모킹한 최소 KisApiClient 대역."""

    def __init__(self, responses: dict) -> None:
        self.get_current_price = AsyncMock(return_value=responses["detail"])
        self.get_trade_strength = AsyncMock(return_value=responses["strength"])
        self.get_investor_trend_estimate = AsyncMock(
            return_value=responses["investor"]
        )
        self.get_program_net_buy = AsyncMock(return_value=responses["program"])


def _detail(close: int, open_: int, rate: float) -> dict:
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


def _base_responses(close: int = 10500, open_: int = 10000, rate: float = 5.0) -> dict:
    return {
        "detail": _detail(close, open_, rate),
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


def _run_fetch(client: _FakeClient, *, scenario_sets: dict) -> tuple[dict, list[str], list[dict]]:
    async def _runner() -> tuple[dict, list[str], list[dict]]:
        stock = {"code": "005930", "name": "테스트", "price": "10500", "chgrate": "5.0"}
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
        )

    return asyncio.run(_runner())


def test_primary_scenario_skips_moving_average() -> None:
    """[SCENARIO_COLLECT_LAZY_EVALUATION_01] 신고가 1차 조건 충족 시 일봉 차트 API 미호출."""
    client = _FakeClient(_base_responses())
    ma_calc = AsyncMock(return_value=({}, (0.0, False, 0), (0.0, False), (0.0, False)))

    with patch("src.api.kis_client.calculate_all_moving_averages", new=ma_calc):
        row, failed, _ = _run_fetch(client, scenario_sets={"new_high": {"005930"}})

    assert row["시나리오"] == "신고가"
    assert failed == []
    assert ma_calc.await_count == 0


def test_upper_limit_scenario_skips_moving_average() -> None:
    """[SCENARIO_COLLECT_LAZY_EVALUATION_01] 상따 1차 조건 충족 시 일봉 차트 API 미호출."""
    client = _FakeClient(_base_responses())
    ma_calc = AsyncMock(return_value=({}, (0.0, False, 0), (0.0, False), (0.0, False)))

    with patch("src.api.kis_client.calculate_all_moving_averages", new=ma_calc):
        row, _, _ = _run_fetch(client, scenario_sets={"upper": {"005930"}})

    assert row["시나리오"] == "상따"
    assert ma_calc.await_count == 0


def test_non_primary_scenario_calls_moving_average() -> None:
    """[SCENARIO_COLLECT_LAZY_EVALUATION_01] 1차 조건 미충족 시 일봉 차트 API 호출로 120 돌파 판별."""
    client = _FakeClient(_base_responses())
    # close=10500, rate=5.0 → prev_close=10000, sma 10200이 10000 < 10200 <= 10500 충족
    ma_calc = AsyncMock(return_value=({}, (0.0, False, 0), (0.0, False), (10200.0, True)))

    with patch("src.api.kis_client.calculate_all_moving_averages", new=ma_calc):
        row, failed, _ = _run_fetch(client, scenario_sets={})

    assert row["시나리오"] == "120 돌파"
    assert failed == []
    assert ma_calc.await_count == 1


def test_moving_average_exception_falls_back_to_default() -> None:
    """[SCENARIO_COLLECT_LAZY_EVALUATION_01] 일봉 차트 API 예외 시 거래량 폭증 기본값 할당."""
    client = _FakeClient(_base_responses())
    ma_calc = AsyncMock(side_effect=RuntimeError("api down"))

    with patch("src.api.kis_client.calculate_all_moving_averages", new=ma_calc):
        row, _, _ = _run_fetch(client, scenario_sets={})

    assert row["시나리오"] == "거래량 폭증"
    assert ma_calc.await_count == 1
