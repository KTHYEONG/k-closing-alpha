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
    monkeypatch.setattr(kis_token_warmup, "should_skip_warmup", lambda _today: False)

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
    from src.data.session_calendar import SessionDay, SessionKind

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

    # And: 모듈 재실행 네임스페이스가 참조하는 실제 달력 해석을 표준일로 고정
    import src.data.session_calendar as session_calendar_mod

    monkeypatch.setattr(
        session_calendar_mod,
        "resolve_session_day",
        lambda day, **_k: SessionDay(trading_date=day, kind=SessionKind.STANDARD, clock=None, provenance="test"),
    )

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
    with pytest.raises(RuntimeError, match="failed slots"):
        asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today="2026-09-16"))


def _warmup_env_three_slots() -> dict[str, str]:
    return {
        "KIS_DATA_SLOTS": "1,2", "KIS_HOST_DATA_SLOTS": "1,2",
        "KIS_DATA_1_APP_KEY": "warm-key-1", "KIS_DATA_1_APP_SECRET": "warm-sec-1",
        "KIS_DATA_2_APP_KEY": "warm-key-2", "KIS_DATA_2_APP_SECRET": "warm-sec-2",
        "KIS_APP_KEY": "warm-primary", "KIS_APP_SECRET": "warm-psec", "KIS_HTS_ID": "warm-hts",
    }


def _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, *, day, fail_keys=(), stale_keys=()):
    """Fake KisApiClient honoring the same-day guard so re-runs skip issued slots."""
    import json
    from pathlib import Path

    from src import settings

    monkeypatch.setattr(settings, "KIS_TOKEN_CACHE_DIR", tmp_path)
    calls: list[str] = []

    def _factory(*, app_key, app_secret, hts_id, token_file):
        class _FakeClient:
            async def issue_daily_token(self, session):
                from src.api.kis.key_pool import read_token_issued_date

                path = Path(token_file)
                if read_token_issued_date(path) == day:
                    return False
                calls.append(app_key)
                if app_key in fail_keys:
                    raise RuntimeError("토큰 발급 실패: {'msg_cd': 'EGW00103', 'msg1': 'invalid'}")
                stamped = day if app_key not in stale_keys else "2026-09-01"
                path.parent.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 - test fake cache write
                path.write_text(json.dumps({  # noqa: ASYNC240 - test fake cache write
                    "access_token": "tok", "expired_at": f"{stamped}T23:59:59+09:00",
                    "app_key": app_key, "issued_at": f"{stamped}T07:05:00+09:00",
                }), encoding="utf-8")
                return True

        return _FakeClient()

    monkeypatch.setattr(kis_token_warmup, "KisApiClient", _factory)
    return calls


def test_warmup_host_tokens_isolates_slot_failure(monkeypatch, tmp_path) -> None:
    import asyncio

    import pytest

    from src.api.kis.key_pool import kis_key_id, read_token_issued_date, token_cache_path
    from src.tools import kis_token_warmup

    day = "2026-09-16"
    env = _warmup_env_three_slots()
    _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day, fail_keys={"warm-key-2"})

    class _Session:
        pass

    with pytest.raises(RuntimeError, match="DATA_2") as exc:
        asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today=day))

    assert "warm-key-2" not in str(exc.value)
    assert kis_key_id("warm-key-2") in str(exc.value)
    assert read_token_issued_date(token_cache_path("warm-key-1", tmp_path)) == day
    assert read_token_issued_date(token_cache_path("warm-primary", tmp_path)) == day


def test_warmup_host_tokens_rerun_issues_only_missing_slots(monkeypatch, tmp_path) -> None:
    import asyncio

    import pytest

    from src.tools import kis_token_warmup

    day = "2026-09-16"
    env = _warmup_env_three_slots()
    calls = _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day, fail_keys={"warm-key-2"})

    class _Session:
        pass

    with pytest.raises(RuntimeError, match="DATA_2"):
        asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today=day))
    assert sorted(calls) == ["warm-key-1", "warm-key-2", "warm-primary"]

    calls.clear()
    rerun_calls = _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day)
    result = asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today=day))

    assert result == {"DATA_1": False, "DATA_2": True, "PRIMARY": False}
    assert rerun_calls == ["warm-key-2"]


def test_warmup_host_tokens_stale_cache_joins_failure_list(monkeypatch, tmp_path) -> None:
    import asyncio

    import pytest

    from src.tools import kis_token_warmup

    day = "2026-09-16"
    env = _warmup_env_three_slots()
    calls = _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day, stale_keys={"warm-key-1"})

    class _Session:
        pass

    with pytest.raises(RuntimeError, match="DATA_1"):
        asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today=day))

    assert sorted(calls) == ["warm-key-1", "warm-key-2", "warm-primary"]


def test_warmup_failure_hides_credentials(monkeypatch, tmp_path, caplog) -> None:
    import asyncio
    import logging

    import pytest

    from src.api.kis.key_pool import kis_key_id
    from src.tools import kis_token_warmup

    day = "2026-09-16"
    env = _warmup_env_three_slots()
    _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day, fail_keys={"warm-key-2"})

    class _Session:
        pass

    with caplog.at_level(logging.ERROR, logger=kis_token_warmup.logger.name), pytest.raises(RuntimeError) as exc:
        asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today=day))

    message = str(exc.value)
    assert kis_key_id("warm-key-2") in message
    assert "warm-key-2" not in message
    assert "warm-sec-2" not in message
    assert any(
        kis_key_id("warm-key-2") in rec.message and "EGW00103" in rec.message for rec in caplog.records
    )
    assert not any("warm-sec-2" in rec.message for rec in caplog.records)


def test_warmup_main_skips_on_verified_closure(monkeypatch, tmp_path, caplog) -> None:
    import logging

    from src import settings
    from src.tools import kis_token_warmup

    monkeypatch.setattr(settings, "BASE_DIR", tmp_path)
    monkeypatch.setattr(kis_token_warmup, "should_skip_warmup", lambda _today: True)
    recorded: list[tuple] = []
    monkeypatch.setattr(
        kis_token_warmup, "record_run_outcome", lambda job, outcome, **kw: recorded.append((job, outcome, kw))
    )

    def _boom(*_a, **_k):
        raise AssertionError("no client on closed day")

    monkeypatch.setattr(kis_token_warmup, "KisApiClient", _boom)

    with caplog.at_level(logging.INFO, logger=kis_token_warmup.logger.name):
        assert kis_token_warmup.main() is None

    assert any("status=SKIP" in rec.message for rec in caplog.records)
    assert len(recorded) == 1
    job, outcome, kw = recorded[0]
    assert job == "kis_token_warmup"
    assert outcome == "SKIPPED"
    assert kw["reason"] == "non_trading_day"
    assert kw["metrics"] == {"session": "CLOSED"}
    assert len(kw["run_date"]) == 10 and kw["run_date"][4] == "-" and kw["run_date"][7] == "-"


def test_should_skip_warmup_only_on_closed() -> None:
    from datetime import date

    from src.data.session_calendar import SessionDay, SessionKind
    from src.tools.kis_token_warmup import should_skip_warmup

    day = date(2026, 9, 16)
    assert should_skip_warmup(day, session_day_fn=lambda _d: SessionDay(trading_date=day, kind=SessionKind.CLOSED, clock=None, provenance="test")) is True
    assert should_skip_warmup(day, session_day_fn=lambda _d: SessionDay(trading_date=day, kind=SessionKind.UNKNOWN, clock=None, provenance="test")) is False
    assert should_skip_warmup(day, session_day_fn=lambda _d: SessionDay(trading_date=day, kind=SessionKind.SHIFTED, clock=None, provenance="test")) is False


def _trading_binding(*, env, app_key, base_url):
    from src.api.kis.trading import TradingBinding

    return TradingBinding(env=env, base_url=base_url, app_key=app_key, app_secret="trade-secret",
                          account_no="12345678", product_code="01", hts_id="hts", rest_tps=1.0)


def _install_trading_client_fake(monkeypatch, kis_token_warmup, tmp_path, *, day, urls):
    import json
    from pathlib import Path

    from src import settings

    monkeypatch.setattr(settings, "KIS_TOKEN_CACHE_DIR", tmp_path)

    def _factory(binding, *, http_timeout_seconds):
        class _FakeApi:
            async def issue_daily_token(self, session):
                from src.api.kis.key_pool import token_cache_path

                urls.append(binding.base_url)
                path = Path(token_cache_path(binding.app_key, tmp_path))
                path.parent.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 - test fake cache write
                path.write_text(json.dumps({  # noqa: ASYNC240 - test fake cache write
                    "access_token": "tok", "expired_at": f"{day}T23:59:59+09:00",
                    "app_key": binding.app_key, "issued_at": f"{day}T07:05:00+09:00",
                }), encoding="utf-8")
                return True

        class _FakeClient:
            def __init__(self):
                self.api_client = _FakeApi()

        return _FakeClient()

    monkeypatch.setattr(kis_token_warmup, "KisTradingClient", _factory)


def test_warmup_issues_mock_trading_token_on_mock_domain(monkeypatch, tmp_path) -> None:
    import asyncio

    from src.api.kis.trading import KIS_MOCK_HOST
    from src.tools import kis_token_warmup

    day = "2026-10-12"
    env = _warmup_env_three_slots()
    data_calls = _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day)
    urls: list[str] = []
    _install_trading_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day, urls=urls)
    binding = _trading_binding(env="mock", app_key="mock-trade-key",
                               base_url=f"https://{KIS_MOCK_HOST}:29443")
    monkeypatch.setattr(kis_token_warmup, "resolve_trading_binding", lambda: binding)

    class _Session:
        pass

    result = asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today=day))

    assert "TRADING_MOCK" in result
    assert result["TRADING_MOCK"] is True
    assert urls == [f"https://{KIS_MOCK_HOST}:29443"]
    assert KIS_MOCK_HOST in urls[0]
    assert sorted(data_calls) == ["warm-key-1", "warm-key-2", "warm-primary"]


def test_warmup_does_not_double_issue_live_trading_key(monkeypatch, tmp_path) -> None:
    import asyncio

    from src.tools import kis_token_warmup

    day = "2026-10-12"
    env = _warmup_env_three_slots()
    data_calls = _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day)

    def _boom(binding, **_kw):
        raise AssertionError("live trading key must reuse the PRIMARY issuance")

    monkeypatch.setattr(kis_token_warmup, "KisTradingClient", _boom)
    binding = _trading_binding(env="live", app_key="warm-primary", base_url="https://openapi.koreainvestment.com:9443")
    monkeypatch.setattr(kis_token_warmup, "resolve_trading_binding", lambda: binding)

    class _Session:
        pass

    result = asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today=day))

    assert "TRADING" not in result and "TRADING_MOCK" not in result
    assert data_calls.count("warm-primary") == 1


def test_warmup_unchanged_when_trading_is_off(monkeypatch, tmp_path) -> None:
    import asyncio

    from src.api.kis.trading import TradingDisabledError
    from src.tools import kis_token_warmup

    day = "2026-10-12"
    env = _warmup_env_three_slots()
    _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day)

    def _disabled():
        raise TradingDisabledError("KIS_TRADING_ENV is off")

    monkeypatch.setattr(kis_token_warmup, "resolve_trading_binding", _disabled)
    monkeypatch.setattr(kis_token_warmup, "KisTradingClient",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no trading client when off")))

    class _Session:
        pass

    assert asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today=day)) == {
        "DATA_1": True, "DATA_2": True, "PRIMARY": True}


def test_warmup_misconfigured_trading_fails_without_leaking(monkeypatch, tmp_path) -> None:
    import asyncio

    import pytest

    from src.api.kis.trading import TradingConfigError
    from src.tools import kis_token_warmup

    day = "2026-10-12"
    env = _warmup_env_three_slots()
    _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day)

    def _misconfigured():
        raise TradingConfigError("mock: KIS_MOCK_APP_SECRET must be non-blank")

    monkeypatch.setattr(kis_token_warmup, "resolve_trading_binding", _misconfigured)

    class _Session:
        pass

    with pytest.raises(RuntimeError, match="TRADING") as exc:
        asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today=day))
    assert "TradingConfigError" in str(exc.value)
    assert "warm-sec-1" not in str(exc.value)
    assert "warm-key-1" not in str(exc.value)


def test_warmup_stale_trading_cache_joins_failure_list(monkeypatch, tmp_path) -> None:
    import asyncio
    import json
    from pathlib import Path

    import pytest

    from src import settings
    from src.api.kis.key_pool import kis_key_id
    from src.api.kis.trading import KIS_MOCK_HOST
    from src.tools import kis_token_warmup

    day = "2026-10-12"
    env = _warmup_env_three_slots()
    _install_warmup_client_fake(monkeypatch, kis_token_warmup, tmp_path, day=day)
    monkeypatch.setattr(settings, "KIS_TOKEN_CACHE_DIR", tmp_path)

    def _factory(binding, *, http_timeout_seconds):
        from src.api.kis.key_pool import token_cache_path

        class _FakeApi:
            async def issue_daily_token(self, session):
                path = Path(token_cache_path(binding.app_key, tmp_path))
                path.parent.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 - test fake cache write
                path.write_text(json.dumps({  # noqa: ASYNC240 - test fake cache write
                    "access_token": "tok", "expired_at": "2026-09-01T23:59:59+09:00",
                    "app_key": binding.app_key, "issued_at": "2026-09-01T07:05:00+09:00",
                }), encoding="utf-8")
                return True

        class _FakeClient:
            def __init__(self):
                self.api_client = _FakeApi()

        return _FakeClient()

    monkeypatch.setattr(kis_token_warmup, "KisTradingClient", _factory)
    binding = _trading_binding(env="mock", app_key="mock-trade-key",
                               base_url=f"https://{KIS_MOCK_HOST}:29443")
    monkeypatch.setattr(kis_token_warmup, "resolve_trading_binding", lambda: binding)

    class _Session:
        pass

    with pytest.raises(RuntimeError, match="TRADING_MOCK") as exc:
        asyncio.run(kis_token_warmup.warmup_host_tokens(_Session(), env, today=day))
    assert kis_key_id("mock-trade-key") in str(exc.value)
    assert "mock-trade-key" not in str(exc.value)

