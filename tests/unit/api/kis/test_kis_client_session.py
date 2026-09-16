from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from src.api.kis.client import KisApiClient


def test_create_session_uses_bounded_timeout() -> None:
    async def run() -> None:
        session = KisApiClient().create_session()
        try:
            assert session.timeout.total == 60
            assert session.timeout.connect == 10
            assert session.timeout.sock_read == 30
        finally:
            await session.close()

    asyncio.run(run())


def test_client_quote_helpers_reuse_common_request_path() -> None:
    async def run() -> None:
        client = KisApiClient(app_key="key", app_secret="secret")
        client.token = "token"
        client._handle_request = AsyncMock(return_value={"rt_cd": "0", "output": []})
        session = type("Session", (), {"get": object()})()
        assert (await client.get_current_price(session, "005930"))["rt_cd"] == "0"
        assert (await client.get_program_net_buy(session, "005930"))["rt_cd"] == "0"
        assert (await client.get_trade_strength(session, "005930"))["rt_cd"] == "0"
        assert (await client.get_market_index_rate(session, "0001"))["rt_cd"] == "0"
        assert (await client.get_market_index_history(session, "0001", "20200101", "20200102"))["rt_cd"] == "0"
        assert (await client.get_investor_trend_estimate(session, "005930"))["rt_cd"] == "0"
        headers = client._get_headers("TEST")
        assert headers["authorization"] == "Bearer token"
        assert client._market_div_candidates("bad") == ["UN", "J", "NX"]

    asyncio.run(run())


def test_kis_ensure_token_single_flight_issues_once_under_concurrency(tmp_path) -> None:
    import asyncio

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "kis_token.json"
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    posts = {"n": 0}

    class _Resp:
        async def json(self):
            return {"access_token": "TOK", "expires_in": 86400}

    class _Ctx:
        async def __aenter__(self):
            posts["n"] += 1
            await asyncio.sleep(0.01)
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, _url, **_kw):
            return _Ctx()

    async def _run():
        session = _Session()
        return await asyncio.gather(*[client.ensure_token(session) for _ in range(10)])

    # When: 10개 태스크가 동시에 첫 토큰을 요구
    tokens = asyncio.run(_run())

    # Then: 발급은 정확히 1회
    assert posts["n"] == 1
    assert tokens == ["TOK"] * 10
    assert client.token == "TOK"


def test_kis_ensure_token_writes_token_file_atomically_with_owner_only_mode(tmp_path, monkeypatch) -> None:
    import asyncio
    import json
    import os
    import stat

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "kis_token.json"
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    replaced = {"n": 0}
    real_replace = os.replace

    def _counting_replace(src, dst):
        replaced["n"] += 1
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _counting_replace)

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

    # When
    asyncio.run(client.ensure_token(_Session()))

    # Then: os.replace 로 원자적 교체 + 소유자 전용 권한
    assert replaced["n"] == 1
    assert token_file.exists()
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    saved = json.loads(token_file.read_text(encoding="utf-8"))
    assert saved["access_token"] == "TOK"
    assert saved["app_key"] == "k"
    # And: 임시 파일 잔여물이 없다
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([token_file.name, token_file.name + ".lock"])


def test_kis_ensure_token_creates_missing_parent_directory_before_writing(tmp_path) -> None:
    """설정 디렉터리가 (재배포/마이그레이션 등으로) 아직 없어도 토큰 캐시 쓰기가 자가치유되어야 한다."""
    import asyncio

    from src.api.kis.client import KisApiClient

    configs_dir = tmp_path / "configs"
    token_file = configs_dir / "kis_token_cache.json"
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

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

    # Given: configs_dir 자체가 아직 존재하지 않음
    assert not configs_dir.exists()

    # When
    asyncio.run(client.ensure_token(_Session()))

    # Then: 디렉터리가 자동 생성되고 토큰이 정상 기록됨
    assert configs_dir.is_dir()
    assert token_file.exists()


def test_kis_ensure_token_falls_back_to_cached_token_when_issuance_throttled(tmp_path) -> None:
    import asyncio
    import json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    import pytest

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "kis_token.json"
    # Given: 10분 미만 남아 조기갱신 대상이지만 아직 만료되지 않은 캐시 토큰
    near_expiry = (datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(minutes=3)).isoformat(timespec="seconds")
    token_file.write_text(
        json.dumps({"access_token": "CACHED", "expired_at": near_expiry, "app_key": "k"}),
        encoding="utf-8",
    )
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    class _Resp:
        async def json(self):
            return {"msg_cd": "EGW00133", "msg1": "접근토큰 발급 잠시 후 다시 시도하세요"}

    class _Ctx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, _url, **_kw):
            return _Ctx()

    # When: 발급이 스로틀링됨
    token = asyncio.run(client.ensure_token(_Session(), force_refresh=True))

    # Then: 만료 전 캐시 토큰을 재사용한다 (결정창 인증 실패 방지)
    assert token == "CACHED"
    assert client.token == "CACHED"

    # And: 쓸 수 있는 캐시가 없으면 침묵하지 않고 raise
    expired = (datetime.now(ZoneInfo("Asia/Seoul")) - timedelta(minutes=1)).isoformat(timespec="seconds")
    token_file.write_text(
        json.dumps({"access_token": "OLD", "expired_at": expired, "app_key": "k"}),
        encoding="utf-8",
    )
    cold = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    with pytest.raises(RuntimeError):
        asyncio.run(cold.ensure_token(_Session(), force_refresh=True))



def test_kis_read_cached_token_rejects_foreign_app_key(tmp_path) -> None:
    import json

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / 'kis_token.json'
    token_file.write_text(
        json.dumps({'access_token': 'OTHER', 'expired_at': '2099-01-01 00:00:00', 'app_key': 'OTHER'}),
        encoding='utf-8',
    )
    client = KisApiClient(app_key='k', app_secret='s', token_file=str(token_file))

    assert client._read_cached_token() is None


def test_kis_ensure_token_reissues_when_cached_file_is_unreadable(tmp_path) -> None:
    import asyncio
    import json

    from src.api.kis.client import KisApiClient

    # Given: 구버전 비원자 쓰기가 남길 수 있는 절단된 캐시 파일
    token_file = tmp_path / "kis_token.json"
    token_file.write_text('{"access_token": "T", "expi', encoding="utf-8")
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    class _Resp:
        async def json(self):
            return {"access_token": "NEW", "expires_in": 86400}

    class _Ctx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, _url, **_kw):
            return _Ctx()

    # When: 판독 불가 캐시로 토큰을 요구
    token = asyncio.run(client.ensure_token(_Session()))

    # Then: 예외로 죽지 않고 재발급 경로로 복구하며 캐시를 정상 파일로 치유한다
    assert token == "NEW"
    assert json.loads(token_file.read_text(encoding="utf-8"))["access_token"] == "NEW"


def test_kis_ensure_token_retries_transient_network_error_then_issues(tmp_path, monkeypatch) -> None:
    import asyncio

    import aiohttp

    from src.api.kis import client as client_mod
    from src.api.kis.client import KisApiClient

    sleeps = []

    async def _no_sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr(client_mod.asyncio, "sleep", _no_sleep)
    client = KisApiClient(app_key="k-retry", app_secret="s", token_file=str(tmp_path / "tok.json"))
    posts = {"n": 0}

    class _Resp:
        async def json(self):
            return {"access_token": "TOK", "expires_in": 86400}

    class _Ctx:
        async def __aenter__(self):
            posts["n"] += 1
            if posts["n"] == 1:
                raise aiohttp.ClientConnectionError("reset")
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, _url, **_kw):
            return _Ctx()

    # When
    token = asyncio.run(client.ensure_token(_Session()))

    # Then
    assert token == "TOK"
    assert posts["n"] == 2
    assert sleeps == [client_mod.KIS_TOKEN_ISSUE_BACKOFF_SEC]


def test_kis_ensure_token_raises_after_retries_and_never_retries_business_error(tmp_path, monkeypatch) -> None:
    import asyncio

    import aiohttp
    import pytest

    from src.api.kis import client as client_mod
    from src.api.kis.client import KisApiClient

    async def _no_sleep(_sec):
        return None

    monkeypatch.setattr(client_mod.asyncio, "sleep", _no_sleep)
    posts = {"n": 0}

    class _DownCtx:
        async def __aenter__(self):
            posts["n"] += 1
            raise aiohttp.ClientConnectionError("down")

        async def __aexit__(self, *_a):
            return False

    class _DownSession:
        def post(self, _url, **_kw):
            return _DownCtx()

    down = KisApiClient(app_key="k-down", app_secret="s", token_file=str(tmp_path / "down.json"))

    # When / Then: 전송 오류 지속
    with pytest.raises(aiohttp.ClientConnectionError):
        asyncio.run(down.ensure_token(_DownSession()))
    assert posts["n"] == client_mod.KIS_TOKEN_ISSUE_ATTEMPTS

    posts["n"] = 0

    class _Resp:
        async def json(self):
            return {"msg_cd": "EGW00103", "msg1": "invalid appkey"}

    class _BizCtx:
        async def __aenter__(self):
            posts["n"] += 1
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _BizSession:
        def post(self, _url, **_kw):
            return _BizCtx()

    biz = KisApiClient(app_key="k-biz", app_secret="s", token_file=str(tmp_path / "biz.json"))

    # When / Then: 업무 오류는 재시도하지 않음
    with pytest.raises(RuntimeError, match="토큰 발급 실패"):
        asyncio.run(biz.ensure_token(_BizSession()))
    assert posts["n"] == 1
