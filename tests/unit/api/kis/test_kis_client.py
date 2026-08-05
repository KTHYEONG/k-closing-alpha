"""Unit tests for KisApiClient — perf_v2 scenario tests.

SCENARIO_RATE_LIMITER_NO_LOCK_WHILE_SLEEP:
  acquire() 호출 시 Lock 외부에서 sleep 수행 — lock-while-sleeping 버그 수정 검증.

SCENARIO_MA_CLIENT_PARAM:
  calculate_all_moving_averages에 client 파라미터 주입 시 ensure_token 미호출 검증.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from unittest.mock import AsyncMock, patch

from src.api.kis_client import AsyncRateLimiter, KisApiClient, calculate_all_moving_averages


class _FakeSession:
    """네트워크 접속 없는 가짜 aiohttp 세션."""


def _ohlcv_response(rows: int) -> dict:
    import pandas as pd

    base = pd.Timestamp("2024-01-01")
    items = []
    for i in range(rows):
        date = (base + pd.Timedelta(days=i)).strftime("%Y%m%d")
        items.append({"stck_bsop_date": date, "stck_clpr": str(10_000 + i)})
    return {"rt_cd": "0", "output2": items}


def test_scenario_rate_limiter_no_lock_while_sleep() -> None:
    """[SCENARIO_RATE_LIMITER_NO_LOCK_WHILE_SLEEP]
    lock을 보유한 채 sleep하지 않아야 하므로, 동시 대기자가 sleep 동안 함께 진행된다.
    window가 가득 찬 상태에서 2개의 동시 대기자는 lock 외부 sleep 시 ~1 window 내 함께
    허용되고, lock-while-sleeping 버그(직렬화) 시에는 두 배의 시간이 걸린다.
    """
    async def _runner() -> None:
        limiter = AsyncRateLimiter(max_rate=2.0, time_period=0.4)
        await limiter.acquire()
        await limiter.acquire()

        async def _acquire() -> float:
            start = time.monotonic()
            await limiter.acquire()
            return time.monotonic() - start

        wait_b, wait_c = await asyncio.gather(_acquire(), _acquire())
        # 직렬화 시 두 번째 대기자는 ~0.8s, lock 외부 sleep 시 ~0.4s 내 동시 완료
        assert wait_b < 0.7
        assert wait_c < 0.7

    asyncio.run(_runner())


def test_scenario_ma_client_param() -> None:
    """[SCENARIO_MA_CLIENT_PARAM]
    calculate_all_moving_averages(code, session, client=existing_client) 호출 시
    ensure_token이 호출되지 않아야 한다.
    """
    sig = inspect.signature(calculate_all_moving_averages)
    assert "stock_code" in sig.parameters
    assert "client" in sig.parameters

    client = KisApiClient(app_key="test-key", account_id="test-account", hts_id="test-hts")
    ensure_token = AsyncMock(return_value="tok")
    get_ohlcv = AsyncMock(return_value=_ohlcv_response(200))

    async def _runner() -> None:
        with (
            patch.object(client, "ensure_token", ensure_token),
            patch.object(client, "get_stock_ohlcv_history", get_ohlcv),
        ):
            await calculate_all_moving_averages(
                "005930", session=_FakeSession(), client=client
            )

    asyncio.run(_runner())

    ensure_token.assert_not_called()
    get_ohlcv.assert_awaited()
