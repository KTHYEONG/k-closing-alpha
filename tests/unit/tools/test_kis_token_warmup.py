"""Auto-generated from contract: kis_key_pool."""

from __future__ import annotations

def test_warmup_host_tokens_issues_each_host_slot_once_per_day(monkeypatch, tmp_path) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src import settings
    from src.api.kis import client as client_mod
    from src.api.kis.key_pool import token_cache_path
    from src.tools import kis_token_warmup

    monkeypatch.setattr(settings, "KIS_TOKEN_CACHE_DIR", tmp_path)
    monkeypatch.setattr(client_mod, "_now_kst", lambda: datetime(2026, 9, 16, 7, 5, 0, tzinfo=ZoneInfo("Asia/Seoul")))
    env = {
        "KIS_DATA_SLOTS": "1,2,3", "KIS_HOST_DATA_SLOTS": "1,2",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1",
        "KIS_DATA_2_APP_KEY": "key2", "KIS_DATA_2_APP_SECRET": "sec2",
        "KIS_APP_KEY": "primary", "KIS_APP_SECRET": "psec", "KIS_HTS_ID": "phts",
    }
    posted: list[str] = []

    class _Resp:
        def __init__(self, key: str) -> None:
            self.key = key

        async def json(self):
            return {"access_token": f"TOK-{self.key}", "expires_in": 86400}

    class _Ctx:
        def __init__(self, key: str) -> None:
            self.key = key

        async def __aenter__(self):
            return _Resp(self.key)

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, _url, **kw):
            posted.append(kw["json"]["appkey"])
            return _Ctx(kw["json"]["appkey"])

    # When
    first = asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today="2026-09-16"))
    second = asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today="2026-09-16"))

    # Then: 배정 슬롯과 선언된 비풀 키(PRIMARY)가 모두 1회씩 발급된다
    assert first == {"DATA_1": True, "DATA_2": True, "PRIMARY": True}
    assert second == {"DATA_1": False, "DATA_2": False, "PRIMARY": False}
    assert posted == ["key1", "key2", "primary"]
    assert token_cache_path("key1", tmp_path).exists()
    assert token_cache_path("key2", tmp_path).exists()
    assert token_cache_path("primary", tmp_path).exists()


def test_warmup_host_tokens_fails_closed_without_host_slots() -> None:
    import asyncio

    import pytest

    from src.tools import kis_token_warmup

    class _Session:
        def post(self, _url, **_kw):
            raise AssertionError("must not call network")

    with pytest.raises(ValueError, match="KIS host data slots are not configured"):
        asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), {}))

def test_warmup_main_runs_host_warmup_with_project_env_and_bounded_session(monkeypatch, tmp_path) -> None:
    import aiohttp

    from src import settings
    from src.tools import kis_token_warmup

    calls: dict[str, object] = {}

    def _fake_load(env_file):
        calls["env_file"] = env_file
        return {"KIS_DATA_SLOTS": "1"}

    async def _fake_warmup(session, env):
        calls["session_type"] = type(session)
        calls["timeout_total"] = session.timeout.total
        calls["env"] = env
        return {"DATA_1": True}

    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(kis_token_warmup, "load_kis_env", _fake_load)
    monkeypatch.setattr(kis_token_warmup, "warmup_host_tokens", _fake_warmup)

    # When
    kis_token_warmup.main()

    # Then
    assert calls["env_file"] == tmp_path / ".env"
    assert calls["env"] == {"KIS_DATA_SLOTS": "1"}
    assert calls["session_type"] is aiohttp.ClientSession
    assert calls["timeout_total"] == 60


def test_warmup_module_entrypoint_invokes_main(monkeypatch, tmp_path) -> None:
    import runpy
    import warnings

    from src import settings
    from src.api.kis import key_pool

    seen: dict[str, object] = {}

    def _fake_load(env_file):
        seen["env_file"] = env_file
        return {}

    def _fake_resolve(env):
        seen["resolved_env"] = env
        return ()

    # Given: runpy가 모듈을 새로 실행하므로 key_pool 원본 심볼을 패치
    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(key_pool, "load_kis_env", _fake_load)
    monkeypatch.setattr(key_pool, "resolve_host_issued_credentials", _fake_resolve)

    # When
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        runpy.run_module("src.tools.kis_token_warmup", run_name="__main__")

    # Then
    assert seen["env_file"] == tmp_path / ".env"
    assert seen["resolved_env"] == {}



def test_warmup_host_tokens_fails_when_cache_not_refreshed(monkeypatch, tmp_path) -> None:
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pytest

    from src import settings
    from src.api.kis import client as client_mod
    from src.tools import kis_token_warmup

    # Given: 발급 호출은 성공하지만 캐시에는 기준일과 다른 날짜가 기록되는 상태
    monkeypatch.setattr(settings, "KIS_TOKEN_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        client_mod, "_now_kst", lambda: datetime(2026, 9, 15, 7, 5, 0, tzinfo=ZoneInfo("Asia/Seoul"))
    )
    env = {
        "KIS_DATA_SLOTS": "1", "KIS_HOST_DATA_SLOTS": "1",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1",
        "KIS_APP_KEY": "primary", "KIS_APP_SECRET": "psec", "KIS_HTS_ID": "phts",
    }

    class _Resp:
        async def json(self):
            return {"access_token": "TOK", "expires_in": 86400}

    class _Ctx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, _url, **_kw):
            return _Ctx()

    # When/Then: 캐시 반영 확인 실패는 침묵하지 않고 실패로 드러난다
    with pytest.raises(RuntimeError, match="token cache not refreshed"):
        asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today="2026-09-16"))

