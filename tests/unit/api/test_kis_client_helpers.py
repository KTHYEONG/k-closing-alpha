"""Unit tests for KisApiClient OHLCV/volatility helper functions.

Covers success and error paths of the moving-average and volatility helpers
whose diagnostic logging was migrated from print() to logger.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pandas as pd

from src.api.kis_client import (
    calculate_all_moving_averages,
    calculate_multiple_emas,
    calculate_stock_ema,
    calculate_stock_sma,
    fetch_index_and_calculate_volatility,
)


class _FakeSession:
    """네트워크 접속 없는 가짜 aiohttp 세션."""


def _ohlcv_response(rows: int) -> dict:
    base = pd.Timestamp("2024-01-01")
    items = []
    for i in range(rows):
        date = (base + pd.Timedelta(days=i)).strftime("%Y%m%d")
        items.append({"stck_bsop_date": date, "stck_clpr": str(10_000 + i)})
    return {"rt_cd": "0", "output2": items}


def _index_response(rows: int) -> dict:
    base = pd.Timestamp("2024-01-01")
    items = []
    for i in range(rows):
        date = (base + pd.Timedelta(days=i)).strftime("%Y%m%d")
        items.append({"stck_bsop_date": date, "bstp_nmix_prpr": str(3_000 + i)})
    return {"rt_cd": "0", "output2": items}


def _run(coro):
    return asyncio.run(coro)


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(return_value=_ohlcv_response(200)),
)
def test_calculate_stock_sma_success() -> None:
    sma_value, ok = _run(calculate_stock_sma("005930", session=_FakeSession()))
    assert ok is True
    assert sma_value > 0


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(return_value={"rt_cd": "9", "msg1": "조회 실패"}),
)
def test_calculate_stock_sma_first_chunk_failure() -> None:
    sma_value, ok = _run(calculate_stock_sma("005930", session=_FakeSession()))
    assert ok is False
    assert sma_value == 0.0


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(side_effect=RuntimeError("api down")),
)
def test_calculate_stock_sma_exception() -> None:
    sma_value, ok = _run(calculate_stock_sma("005930", session=_FakeSession()))
    assert ok is False
    assert sma_value == 0.0


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(return_value=_ohlcv_response(60)),
)
def test_calculate_stock_ema_success() -> None:
    ema_value, ok, count = _run(calculate_stock_ema("005930", session=_FakeSession()))
    assert ok is True
    assert ema_value > 0
    assert count >= 20


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(return_value={"rt_cd": "9", "msg1": "조회 실패"}),
)
def test_calculate_stock_ema_failure() -> None:
    ema_value, ok, count = _run(calculate_stock_ema("005930", session=_FakeSession()))
    assert ok is False
    assert ema_value == 0.0
    assert count == 0


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(return_value={"rt_cd": "0", "output2": []}),
)
def test_calculate_stock_ema_empty_items() -> None:
    ema_value, ok, count = _run(calculate_stock_ema("005930", session=_FakeSession()))
    assert ok is False
    assert ema_value == 0.0
    assert count == 0


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(side_effect=RuntimeError("api down")),
)
def test_calculate_stock_ema_exception() -> None:
    ema_value, ok, count = _run(calculate_stock_ema("005930", session=_FakeSession()))
    assert ok is False
    assert ema_value == 0.0
    assert count == 0


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(return_value=_ohlcv_response(60)),
)
def test_calculate_multiple_emas_success() -> None:
    results = _run(calculate_multiple_emas("005930", session=_FakeSession()))
    assert set(results) == {5, 10, 20}
    assert all(v > 0 for v in results.values())


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(return_value={"rt_cd": "9", "msg1": "조회 실패"}),
)
def test_calculate_multiple_emas_failure() -> None:
    results = _run(calculate_multiple_emas("005930", session=_FakeSession()))
    assert results == {}


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(return_value=_ohlcv_response(200)),
)
def test_calculate_all_moving_averages_success() -> None:
    ema_res, (ema20, ema_ok, _), (sma60, sma60_ok), (sma120, sma120_ok) = _run(
        calculate_all_moving_averages("005930", session=_FakeSession())
    )
    assert set(ema_res) == {5, 10, 20}
    assert ema_ok is True
    assert ema20 > 0
    assert sma60_ok is True
    assert sma60 > 0
    assert sma120_ok is True
    assert sma120 > 0


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(return_value={"rt_cd": "9", "msg1": "조회 실패"}),
)
def test_calculate_all_moving_averages_first_chunk_failure() -> None:
    result = _run(calculate_all_moving_averages("005930", session=_FakeSession()))
    assert result == ({}, (0.0, False, 0), (0.0, False), (0.0, False))


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
    new=AsyncMock(side_effect=RuntimeError("api down")),
)
def test_calculate_all_moving_averages_exception() -> None:
    result = _run(calculate_all_moving_averages("005930", session=_FakeSession()))
    assert result == ({}, (0.0, False, 0), (0.0, False), (0.0, False))


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_market_index_history",
    new=AsyncMock(return_value=_index_response(30)),
)
def test_fetch_index_volatility_success() -> None:
    hv_today, hv_change = _run(fetch_index_and_calculate_volatility(session=_FakeSession()))
    assert hv_today >= 0.0
    assert isinstance(hv_change, float)


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_market_index_history",
    new=AsyncMock(return_value={"rt_cd": "9", "msg1": "조회 실패"}),
)
def test_fetch_index_volatility_failure() -> None:
    hv_today, hv_change = _run(fetch_index_and_calculate_volatility(session=_FakeSession()))
    assert hv_today == 0.0
    assert hv_change == 0.0


@patch("src.api.kis_client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis_client.KisApiClient.get_market_index_history",
    new=AsyncMock(return_value={"rt_cd": "0", "output2": []}),
)
def test_fetch_index_volatility_insufficient_data() -> None:
    hv_today, hv_change = _run(fetch_index_and_calculate_volatility(session=_FakeSession()))
    assert hv_today == 0.0
    assert hv_change == 0.0


def test_fetch_index_volatility_date_range() -> None:
    """기본 index_code 파라미터가 그대로 get_market_index_history로 전달된다."""
    mock_history = AsyncMock(return_value={"rt_cd": "0", "output2": []})

    async def _runner() -> None:
        with (
            patch(
                "src.api.kis_client.KisApiClient.ensure_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "src.api.kis_client.KisApiClient.get_market_index_history",
                new=mock_history,
            ),
        ):
            await fetch_index_and_calculate_volatility("1028", session=_FakeSession())

    _run(_runner())

    session_arg, index_code, start_date, end_date = mock_history.await_args.args
    assert index_code == "1028"
    assert start_date <= datetime.now().strftime("%Y%m%d")
    assert end_date <= datetime.now().strftime("%Y%m%d")
    assert start_date < end_date or start_date == end_date
