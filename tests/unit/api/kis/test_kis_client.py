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

from src.api.kis.client import KisApiClient
from src.api.kis.indicators import calculate_all_moving_averages
from src.api.kis.rate_limit import AsyncRateLimiter


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


def test_kis_client_instances_share_process_global_rate_limiter() -> None:
    from src.api.kis.client import KisApiClient

    # Given: 동일 app_key 로 만든 두 클라이언트 (프로덕션 17개 생성 지점 재현)
    a = KisApiClient(app_key="SAME", app_secret="s")
    b = KisApiClient(app_key="SAME", app_secret="s")
    other = KisApiClient(app_key="OTHER", app_secret="s")

    # Then: 리미터는 프로세스 전역 공유 -> 합산 TPS 가 서버 한도를 넘지 않는다
    assert a.rate_limiter is b.rate_limiter
    assert a.rate_limiter is not other.rate_limiter
    assert a.rate_limiter.max_rate == 18.0

    # And: 사용되지 않던 세마포어 데드코드는 제거되었다
    assert not hasattr(a, "semaphore")


def test_handle_request_reacquires_rate_limit_slot_on_every_retry(monkeypatch) -> None:
    import asyncio

    from src.api.kis.client import KisApiClient

    client = KisApiClient(app_key="k", app_secret="s")
    client.token = "T"

    acquires = {"n": 0}

    async def _counting_acquire() -> None:
        acquires["n"] += 1

    monkeypatch.setattr(client.rate_limiter, "acquire", _counting_acquire)

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    # Given: 429 -> "초당 거래건수" -> 정상 의 3회 응답 시퀀스
    responses = [
        {"status": 429, "body": {}},
        {"status": 200, "body": {"rt_cd": "1", "msg1": "초당 거래건수를 초과하였습니다"}},
        {"status": 200, "body": {"rt_cd": "0", "output": {"stck_prpr": "1000"}}},
    ]

    class _Resp:
        def __init__(self, spec):
            self.status = spec["status"]
            self._body = spec["body"]

        async def json(self):
            return self._body

    class _Ctx:
        def __init__(self, spec):
            self._spec = spec

        async def __aenter__(self):
            return _Resp(self._spec)

        async def __aexit__(self, *_a):
            return False

    calls = {"n": 0}

    def _session_get(_url, **_kw):
        spec = responses[calls["n"]]
        calls["n"] += 1
        return _Ctx(spec)

    # When
    out = asyncio.run(client._handle_request(_session_get, "http://x", headers={"authorization": "Bearer T"}))

    # Then: 3회 발사 = 3회 슬롯 획득 (재시도가 리미터를 우회하지 않는다)
    assert out["rt_cd"] == "0"
    assert calls["n"] == 3
    assert acquires["n"] == 3


def test_handle_request_retries_request_timeout_instead_of_raising(monkeypatch) -> None:
    import asyncio

    from src.api.kis.client import KisApiClient

    client = KisApiClient(app_key="k-timeout", app_secret="s")
    client.token = "T"

    async def _free_acquire() -> None:
        return None

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client.rate_limiter, "acquire", _free_acquire)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    class _Ctx:
        def __init__(self, fail: bool):
            self._fail = fail

        async def __aenter__(self):
            if self._fail:
                raise TimeoutError
            resp = type("_Resp", (), {"status": 200})()

            async def _json():
                return {"rt_cd": "0"}

            resp.json = _json
            return resp

        async def __aexit__(self, *_a):
            return False

    # Given: 첫 요청은 세션 total 타임아웃, 재시도는 정상
    calls = {"n": 0}

    def _flaky_get(_url, **_kw):
        calls["n"] += 1
        return _Ctx(fail=calls["n"] == 1)

    # When / Then: 타임아웃은 재시도되어 정상 응답을 반환한다
    assert asyncio.run(client._handle_request(_flaky_get, "http://x", headers={}))["rt_cd"] == "0"
    assert calls["n"] == 2

    # Given: 모든 시도가 타임아웃
    def _always_timeout(_url, **_kw):
        return _Ctx(fail=True)

    # When / Then: 예외 대신 실패 envelope를 반환해 호출자의 degraded 판정 경로로 흐른다
    assert asyncio.run(client._handle_request(_always_timeout, "http://x", headers={}))["rt_cd"] == "9"


def test_get_current_price_without_fallback_queries_only_requested_market(monkeypatch) -> None:
    import asyncio

    import pytest

    from src.api.kis.client import KisApiClient

    client = KisApiClient(app_key="k-venue", app_secret="s")
    client.token = "T"
    calls = []

    async def _fake_handle(_session_method, _url, **kwargs):
        calls.append(dict(kwargs["params"]))
        return {"rt_cd": "1", "msg1": "fail"}

    monkeypatch.setattr(client, "_handle_request", _fake_handle)

    class _Session:
        get = object()

    # When: 폴백 금지
    res = asyncio.run(client.get_current_price(_Session(), "005930", market_div_code="J", allow_market_div_fallback=False))

    # Then: J 한 번만 조회
    assert res["rt_cd"] == "1"
    assert calls == [{"fid_cond_mrkt_div_code": "J", "fid_input_iscd": "005930"}]

    # And: 기본값은 기존 J->UN->NX 폴백 유지
    calls.clear()
    asyncio.run(client.get_current_price(_Session(), "005930", market_div_code="J"))
    assert [c["fid_cond_mrkt_div_code"] for c in calls] == ["J", "UN", "NX"]

    # And: 폴백 금지인데 시장 미지정이면 거부
    with pytest.raises(ValueError, match="market_div_code"):
        asyncio.run(client.get_current_price(_Session(), "005930", allow_market_div_fallback=False))


def test_kis_data_client_kwargs_reads_data_account_settings(monkeypatch) -> None:
    from src import settings
    from src.api.kis.client import kis_data_client_kwargs
    from src.api.kis.key_pool import token_cache_path

    # Given: 체결 계좌와 분리된 데이터 슬롯 풀
    monkeypatch.setattr(settings, "KIS_API_CONFIG", {
        "app_key": "exec-key", "app_secret": "exec-secret", "account_id": "exec-acct", "hts_id": "exec-hts",
    })
    env = {
        "KIS_DATA_SLOTS": "1", "KIS_HOST_DATA_SLOTS": "1",
        "KIS_DATA_1_APP_KEY": "data-key", "KIS_DATA_1_APP_SECRET": "data-secret", "KIS_DATA_1_HTS_ID": "data-hts",
    }

    # When
    out = kis_data_client_kwargs(env)

    # Then: 데이터 슬롯 값만 반영, 체결 계좌 값은 섞이지 않는다
    assert out == {
        "app_key": "data-key", "app_secret": "data-secret", "account_id": "", "hts_id": "data-hts",
        "token_file": str(token_cache_path("data-key", settings.KIS_TOKEN_CACHE_DIR)),
    }


def test_kis_data_client_gets_isolated_rate_limiter_from_execution_client(monkeypatch) -> None:
    from src import settings
    from src.api.kis.client import KisApiClient, kis_data_client_kwargs

    # Given: 서로 다른 앱키를 가진 데이터 슬롯/체결 계좌
    monkeypatch.setattr(settings, "KIS_API_CONFIG", {
        "app_key": "exec-key-iso", "app_secret": "s", "account_id": "a", "hts_id": "h",
    })
    env = {
        "KIS_DATA_SLOTS": "1", "KIS_HOST_DATA_SLOTS": "1",
        "KIS_DATA_1_APP_KEY": "data-key-iso", "KIS_DATA_1_APP_SECRET": "s", "KIS_DATA_1_HTS_ID": "h",
    }

    # When
    data_client = KisApiClient(**kis_data_client_kwargs(env))
    exec_client = KisApiClient()

    # Then: 프로세스 전역 공유 리미터가 앱키별로 분리된 버킷을 갖는다
    assert data_client.rate_limiter is not exec_client.rate_limiter
    assert data_client.app_key == "data-key-iso"
    assert exec_client.app_key == "exec-key-iso"

