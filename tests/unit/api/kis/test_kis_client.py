"""Unit tests for KisApiClient — perf_v2 scenario tests.

SCENARIO_RATE_LIMITER_NO_LOCK_WHILE_SLEEP:
  acquire() 호출 시 Lock 외부에서 sleep 수행 — lock-while-sleeping 버그 수정 검증.
"""

from __future__ import annotations

import asyncio
import time

from src.api.kis.client import KisApiClient
from src.api.kis.rate_limit import AsyncRateLimiter


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


def test_kis_decision_shard_client_kwargs_returns_single_entry_when_unset() -> None:
    from src.api.kis.client import kis_decision_shard_client_kwargs

    env = {
        "KIS_DATA_SLOTS": "1,4",
        "KIS_HOST_DATA_SLOTS": "1,4",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_4_APP_KEY": "key4", "KIS_DATA_4_APP_SECRET": "sec4", "KIS_DATA_4_HTS_ID": "hts4",
    }
    result = kis_decision_shard_client_kwargs(env)
    assert len(result) == 1
    assert result[0]["app_key"] == "key1"
    assert result[0]["account_id"] == ""


def test_kis_decision_shard_client_kwargs_returns_two_entries_when_configured() -> None:
    from src.api.kis.client import kis_decision_shard_client_kwargs

    env = {
        "KIS_DATA_SLOTS": "1,2,3,4,5",
        "KIS_HOST_DATA_SLOTS": "1,2,3,4",
        "KIS_DECISION_SHARD_SLOTS": "1,5",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_2_APP_KEY": "key2", "KIS_DATA_2_APP_SECRET": "sec2", "KIS_DATA_2_HTS_ID": "hts2",
        "KIS_DATA_3_APP_KEY": "key3", "KIS_DATA_3_APP_SECRET": "sec3", "KIS_DATA_3_HTS_ID": "hts3",
        "KIS_DATA_4_APP_KEY": "key4", "KIS_DATA_4_APP_SECRET": "sec4", "KIS_DATA_4_HTS_ID": "hts4",
        "KIS_DATA_5_APP_KEY": "key5", "KIS_DATA_5_APP_SECRET": "sec5", "KIS_DATA_5_HTS_ID": "hts5",
    }
    result = kis_decision_shard_client_kwargs(env)
    assert [kw["app_key"] for kw in result] == ["key1", "key5"]
    assert result[1]["token_file"].endswith(".json")


def _skip_if_root() -> None:
    import os

    import pytest

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root bypasses file permission checks")


def _lock_client(tmp_path, app_key: str = "KEY123456"):
    from src.api.kis.client import KisApiClient

    return KisApiClient(app_key=app_key, app_secret="SECRET999", token_file=str(tmp_path / "token.json"))


class _TokenIssuedResponse:
    async def json(self):
        return {"access_token": "TOKEN-ABC", "expires_in": 86400}


class _TokenIssuedContext:
    async def __aenter__(self):
        return _TokenIssuedResponse()

    async def __aexit__(self, *_args):
        return False


class _TokenIssueSession:
    """Stub aiohttp session serving one successful token issuance."""

    def __init__(self) -> None:
        self.posts = 0

    def post(self, _url, **_kwargs):
        self.posts += 1
        return _TokenIssuedContext()


class _BoomSession:
    def post(self, _url, **_kwargs):
        raise AssertionError("no token request must be sent")


def test_host_token_lock_falls_back_to_readonly_descriptor(tmp_path, caplog) -> None:
    import asyncio
    import json
    import logging
    import os
    import stat

    _skip_if_root()
    lock_path = tmp_path / "token.json.lock"
    lock_path.write_bytes(b"")
    lock_path.chmod(0o444)
    client = _lock_client(tmp_path)
    session = _TokenIssueSession()

    with caplog.at_level(logging.INFO, logger="src.api.kis.client"):
        token = asyncio.run(client.ensure_token(session))  # type: ignore[arg-type]

    assert token == "TOKEN-ABC"
    assert session.posts == 1
    assert json.loads((tmp_path / "token.json").read_text(encoding="utf-8"))["access_token"] == "TOKEN-ABC"
    records = [rec for rec in caplog.records if rec.name == "src.api.kis.client"]
    assert any("READONLY_FALLBACK" in rec.getMessage() for rec in records)
    assert all("KEY123456" not in rec.getMessage() and "SECRET999" not in rec.getMessage() for rec in records)
    assert stat.S_IMODE(os.stat(lock_path).st_mode) == 0o444


def test_host_token_lock_readonly_descriptor_still_excludes(tmp_path) -> None:
    import asyncio
    import time

    _skip_if_root()
    lock_path = tmp_path / "token.json.lock"
    lock_path.write_bytes(b"")
    lock_path.chmod(0o444)
    first = _lock_client(tmp_path)
    second = _lock_client(tmp_path)
    spans: dict[str, list[float]] = {}

    async def _worker(client, tag: str) -> None:
        async with client._host_token_lock():
            spans[tag] = [time.monotonic()]
            await asyncio.sleep(0.05)
            spans[tag].append(time.monotonic())

    async def _main() -> None:
        await asyncio.gather(_worker(first, "a"), _worker(second, "b"))

    asyncio.run(_main())

    assert set(spans) == {"a", "b"}
    assert spans["a"][1] <= spans["b"][0] or spans["b"][1] <= spans["a"][0]


def test_host_token_lock_unopenable_directory_raises_with_log(tmp_path, caplog) -> None:
    import asyncio
    import logging

    import pytest

    (tmp_path / "token.json.lock").mkdir()
    client = _lock_client(tmp_path)

    with caplog.at_level(logging.INFO, logger="src.api.kis.client"), pytest.raises(IsADirectoryError):
        asyncio.run(client.ensure_token(_BoomSession()))  # type: ignore[arg-type]
    records = [rec for rec in caplog.records if rec.name == "src.api.kis.client"]
    assert any("UNOPENABLE" in rec.getMessage() for rec in records)


def test_host_token_lock_unopenable_mode_raises_with_log(tmp_path, caplog) -> None:
    import asyncio
    import logging

    import pytest

    _skip_if_root()
    lock_path = tmp_path / "token.json.lock"
    lock_path.write_bytes(b"")
    lock_path.chmod(0o000)
    client = _lock_client(tmp_path)

    with caplog.at_level(logging.INFO, logger="src.api.kis.client"), pytest.raises(PermissionError):
        asyncio.run(client.ensure_token(_BoomSession()))  # type: ignore[arg-type]
    records = [rec for rec in caplog.records if rec.name == "src.api.kis.client"]
    assert any("UNOPENABLE" in rec.getMessage() for rec in records)


def test_host_token_lock_uncreatable_lock_raises_with_log(tmp_path, caplog) -> None:
    import asyncio
    import logging
    import os

    import pytest

    _skip_if_root()
    from src.api.kis.client import KisApiClient

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    os.chmod(cache_dir, 0o555)  # noqa: S103 - fixture needs a directory where file creation fails
    try:
        client = KisApiClient(app_key="KEY123456", app_secret="SECRET999", token_file=str(cache_dir / "token.json"))
        with caplog.at_level(logging.INFO, logger="src.api.kis.client"), pytest.raises(PermissionError):
            asyncio.run(client.ensure_token(_BoomSession()))  # type: ignore[arg-type]
    finally:
        os.chmod(cache_dir, 0o755)  # noqa: S103 - restore test fixture permissions
    records = [rec for rec in caplog.records if rec.name == "src.api.kis.client"]
    assert any("UNOPENABLE" in rec.getMessage() for rec in records)


def test_host_token_lock_normal_path_unchanged(tmp_path, caplog) -> None:
    import asyncio
    import logging
    import os
    import stat

    client = _lock_client(tmp_path)
    session = _TokenIssueSession()

    with caplog.at_level(logging.INFO, logger="src.api.kis.client"):
        token = asyncio.run(client.ensure_token(session))  # type: ignore[arg-type]

    assert token == "TOKEN-ABC"
    assert stat.S_IMODE(os.stat(tmp_path / "token.json.lock").st_mode) == 0o600
    records = [rec for rec in caplog.records if rec.name == "src.api.kis.client"]
    assert not any("READONLY_FALLBACK" in rec.getMessage() or "UNOPENABLE" in rec.getMessage() for rec in records)

