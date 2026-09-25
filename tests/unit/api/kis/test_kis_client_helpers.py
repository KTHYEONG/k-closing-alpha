"""Unit tests for KisApiClient OHLCV/volatility helper functions.

Covers success and error paths of the moving-average and volatility helpers
whose diagnostic logging was migrated from console output to logger.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pandas as pd

from src.api.kis.indicators import fetch_index_and_calculate_volatility


class _FakeSession:
    """네트워크 접속 없는 가짜 aiohttp 세션."""


def _index_response(rows: int) -> dict:
    base = pd.Timestamp("2024-01-01")
    items = []
    for i in range(rows):
        date = (base + pd.Timedelta(days=i)).strftime("%Y%m%d")
        items.append({"stck_bsop_date": date, "bstp_nmix_prpr": str(3_000 + i)})
    return {"rt_cd": "0", "output2": items}


def _run(coro):
    return asyncio.run(coro)


@patch("src.api.kis.client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis.client.KisApiClient.get_market_index_history",
    new=AsyncMock(return_value=_index_response(30)),
)
def test_fetch_index_volatility_success() -> None:
    hv_today, hv_change = _run(fetch_index_and_calculate_volatility(session=_FakeSession()))
    assert hv_today >= 0.0
    assert isinstance(hv_change, float)


@patch("src.api.kis.client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis.client.KisApiClient.get_market_index_history",
    new=AsyncMock(return_value={"rt_cd": "9", "msg1": "조회 실패"}),
)
def test_fetch_index_volatility_failure() -> None:
    hv_today, hv_change = _run(fetch_index_and_calculate_volatility(session=_FakeSession()))
    assert hv_today == 0.0
    assert hv_change == 0.0


@patch("src.api.kis.client.KisApiClient.ensure_token", new=AsyncMock(return_value="tok"))
@patch(
    "src.api.kis.client.KisApiClient.get_market_index_history",
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
                "src.api.kis.client.KisApiClient.ensure_token",
                new=AsyncMock(return_value="tok"),
            ),
            patch(
                "src.api.kis.client.KisApiClient.get_market_index_history",
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
def test_indicators_kis_clients_use_data_key_kwargs() -> None:
    import ast
    import inspect

    from src.api.kis import indicators

    tree = ast.parse(inspect.getsource(indicators))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "KisApiClient"
    ]

    assert len(calls) == 1
    for call in calls:
        assert call.args == []
        assert len(call.keywords) == 1
        assert call.keywords[0].arg is None
        inner = call.keywords[0].value
        assert isinstance(inner, ast.Call)
        assert isinstance(inner.func, ast.Name)
        assert inner.func.id == "kis_data_client_kwargs"
        assert inner.args == [] and inner.keywords == []

