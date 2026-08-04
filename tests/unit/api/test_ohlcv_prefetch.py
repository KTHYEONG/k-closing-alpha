"""Unit tests for SMA120 OHLCV prefetch (Prefetch-then-Compute).

Scenarios:
- T01: prefetched_records 주어지면 get_stock_ohlcv_history 호출 횟수 == 0
- T02: prefetched_records=None 시 기존 chunk loop 실행 (하위 호환)
- T04: prefetch_ohlcv_for_sma120은 codes 수만큼 단일 청크 호출
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from src.api.kis_client import (
    KisApiClient,
    calculate_all_moving_averages,
    prefetch_ohlcv_for_sma120,
)


class _FakeSession:
    """네트워크 접속 없는 가짜 aiohttp 세션."""


def _ohlcv_response(rows: int, offset_days: int = 0) -> dict:
    import pandas as pd

    base = pd.Timestamp("2024-01-01") + pd.Timedelta(days=offset_days)
    items = []
    for i in range(rows):
        date = (base + pd.Timedelta(days=i)).strftime("%Y%m%d")
        items.append({"stck_bsop_date": date, "stck_clpr": str(10_000 + i)})
    return {"rt_cd": "0", "output2": items}


def _prefetched_records(count: int = 150) -> list[dict[str, str]]:
    return [{"date": f"2025{i:04d}", "close": "50000"} for i in range(1, count + 1)]


def _client() -> KisApiClient:
    return KisApiClient(
        app_key="test-key", account_id="test-account", hts_id="test-hts"
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------
# T01 / T02: calculate_all_moving_averages prefetched_records
# ---------------------------------------------------------
def test_prefetched_records_skips_api_call() -> None:
    """[T01] prefetched_records 주어지면 get_stock_ohlcv_history 호출 횟수 == 0."""
    mock_history = AsyncMock(return_value=_ohlcv_response(200))
    with patch(
        "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
        new=mock_history,
    ):
        result = _run(
            calculate_all_moving_averages(
                "005930", session=_FakeSession(), prefetched_records=_prefetched_records()
            )
        )
    mock_history.assert_not_called()
    assert result[3][1] is True
    assert result[1][2] == 150


def test_prefetched_records_none_uses_chunk_loop() -> None:
    """[T02] prefetched_records=None 시 기존 chunk loop 동작 유지."""
    mock_history = AsyncMock(return_value=_ohlcv_response(200))
    with (
        patch(
            "src.api.kis_client.KisApiClient.ensure_token",
            new=AsyncMock(return_value="tok"),
        ),
        patch(
            "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
            new=mock_history,
        ),
    ):
        result = _run(
            calculate_all_moving_averages("005930", session=_FakeSession())
        )
    mock_history.assert_awaited()
    assert result[3][1] is True


def test_prefetched_records_none_with_local_session() -> None:
    """[T02] session 미지원 호출 시 내부 로컬 세션 생성 후 chunk loop 실행."""
    mock_history = AsyncMock(return_value=_ohlcv_response(200))
    client = _client()
    with patch.object(client, "get_stock_ohlcv_history", mock_history):
        result = _run(
            calculate_all_moving_averages("005930", session=None, client=client)
        )
    mock_history.assert_awaited()
    assert result[3][1] is True


def test_prefetched_records_invalid_items_skipped() -> None:
    """[T01] 빈 날짜/무효 종가 레코드는 계산에서 제외된다."""
    records: list[dict[str, str]] = [
        {"date": "", "close": "50000"},
        {"date": "20250102", "close": "not-a-number"},
        {"date": "20250103", "close": "0"},
        {"date": "20250104", "close": "50000"},
    ]
    result = _run(
        calculate_all_moving_averages(
            "005930", session=_FakeSession(), prefetched_records=records
        )
    )
    assert result[1][2] == 1


def test_prefetched_records_all_invalid_returns_empty() -> None:
    """[T01] prefetched_records 전부 무효 시 기본값 tuple 반환."""
    records: list[dict[str, str]] = [
        {"date": "", "close": "50000"},
        {"date": "20250102", "close": "abc"},
    ]
    result = _run(
        calculate_all_moving_averages(
            "005930", session=_FakeSession(), prefetched_records=records
        )
    )
    assert result == ({}, (0.0, False, 0), (0.0, False), (0.0, False))


def test_chunk_loop_second_chunk_failure_breaks() -> None:
    """[T02] 1차 chunk 미충분 시 2차 chunk 실패는 break 후 부분 데이터로 계산."""
    mock_history = AsyncMock(
        side_effect=[_ohlcv_response(60), {"rt_cd": "9", "msg1": "2차 조회 실패"}]
    )
    with (
        patch(
            "src.api.kis_client.KisApiClient.ensure_token",
            new=AsyncMock(return_value="tok"),
        ),
        patch(
            "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
            new=mock_history,
        ),
    ):
        result = _run(calculate_all_moving_averages("005930", session=_FakeSession()))
    assert mock_history.await_count == 2
    assert result[2][1] is True
    assert result[3][1] is False


def test_chunk_loop_sleeps_between_chunks() -> None:
    """[T02] 1차 chunk 150건 미만 시 sleep 후 2차 chunk로 데이터 보강."""
    mock_history = AsyncMock(
        side_effect=[_ohlcv_response(100), _ohlcv_response(100, offset_days=200)]
    )
    with (
        patch(
            "src.api.kis_client.KisApiClient.ensure_token",
            new=AsyncMock(return_value="tok"),
        ),
        patch(
            "src.api.kis_client.KisApiClient.get_stock_ohlcv_history",
            new=mock_history,
        ),
    ):
        result = _run(calculate_all_moving_averages("005930", session=_FakeSession()))
    assert mock_history.await_count == 2
    assert result[1][2] >= 150
    assert result[3][1] is True


# ---------------------------------------------------------
# T04: prefetch_ohlcv_for_sma120
# ---------------------------------------------------------
def test_prefetch_single_chunk_call_per_code() -> None:
    """[T04] prefetch_ohlcv_for_sma120은 codes 수만큼 단일 청크 호출."""
    codes = ["005930", "000660", "035420"]
    mock_history = AsyncMock(return_value=_ohlcv_response(200))
    client = _client()
    with patch.object(client, "get_stock_ohlcv_history", mock_history):
        result = _run(prefetch_ohlcv_for_sma120(codes, session=_FakeSession(), client=client))
    assert mock_history.await_count == len(codes)
    assert set(result) == set(codes)
    assert all(len(records) >= 120 for records in result.values())


def test_prefetch_empty_codes_returns_empty() -> None:
    """[T01] 빈 codes면 API 호출 없이 빈 dict 반환."""
    mock_history = AsyncMock(return_value=_ohlcv_response(200))
    client = _client()
    with patch.object(client, "get_stock_ohlcv_history", mock_history):
        result = _run(prefetch_ohlcv_for_sma120([], session=_FakeSession(), client=client))
    assert result == {}
    mock_history.assert_not_called()


def test_prefetch_excludes_failed_codes() -> None:
    """[T01] 실패한 종목은 반환 dict에서 key로 제외된다."""
    codes = ["005930", "000660"]

    async def _side_effect(session, code, start_date, end_date):
        if code == "000660":
            return {"rt_cd": "9", "msg1": "조회 실패"}
        return _ohlcv_response(200)

    mock_history = AsyncMock(side_effect=_side_effect)
    client = _client()
    with patch.object(client, "get_stock_ohlcv_history", mock_history):
        result = _run(prefetch_ohlcv_for_sma120(codes, session=_FakeSession(), client=client))
    assert "000660" not in result
    assert "005930" in result


def test_prefetch_empty_output_excluded() -> None:
    """[T01] output2가 비어 있으면 해당 종목은 제외된다."""
    mock_history = AsyncMock(return_value={"rt_cd": "0", "output2": []})
    client = _client()
    with patch.object(client, "get_stock_ohlcv_history", mock_history):
        result = _run(prefetch_ohlcv_for_sma120(["005930"], session=_FakeSession(), client=client))
    assert result == {}


def test_prefetch_survives_exception() -> None:
    """[T01] API 예외 발생 시 예외 전파 없이 해당 종목 제외."""
    mock_history = AsyncMock(side_effect=RuntimeError("api down"))
    client = _client()
    with patch.object(client, "get_stock_ohlcv_history", mock_history):
        result = _run(prefetch_ohlcv_for_sma120(["005930"], session=_FakeSession(), client=client))
    assert result == {}
