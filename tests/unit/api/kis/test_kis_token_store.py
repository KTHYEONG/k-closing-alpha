"""Auto-generated from contract: kis_key_pool."""

from __future__ import annotations

def test_kis_data_client_kwargs_selects_role_slot_and_shared_cache_path(monkeypatch, tmp_path) -> None:
    from src import settings
    from src.api.kis.client import kis_data_client_kwargs
    from src.api.kis.key_pool import token_cache_path

    env = {
        "KIS_DATA_SLOTS": "1,2,3", "KIS_HOST_DATA_SLOTS": "1,2",
        "KIS_DATA_1_APP_KEY": "key1", "KIS_DATA_1_APP_SECRET": "sec1", "KIS_DATA_1_HTS_ID": "hts1",
        "KIS_DATA_2_APP_KEY": "key2", "KIS_DATA_2_APP_SECRET": "sec2", "KIS_DATA_2_HTS_ID": "hts2",
    }
    monkeypatch.setattr(settings, "KIS_TOKEN_CACHE_DIR", tmp_path)
    monkeypatch.setattr(settings, "KIS_DATA_ROLE", "decision")

    out = kis_data_client_kwargs(env)

    assert out == {
        "app_key": "key1", "app_secret": "sec1", "account_id": "", "hts_id": "hts1",
        "token_file": str(token_cache_path("key1", tmp_path)),
    }
    monkeypatch.setattr(settings, "KIS_DATA_ROLE", "batch")
    batch = kis_data_client_kwargs(env)
    assert batch["app_key"] == "key2"
    assert batch["token_file"] == str(token_cache_path("key2", tmp_path))


def test_kis_data_client_kwargs_fails_closed_without_host_slots() -> None:
    import pytest

    from src.api.kis.client import kis_data_client_kwargs

    with pytest.raises(ValueError, match="KIS_HOST_DATA_SLOTS"):
        kis_data_client_kwargs({"KIS_DATA_SLOTS": "1", "KIS_DATA_1_APP_KEY": "a", "KIS_DATA_1_APP_SECRET": "b"})


def test_kis_client_default_token_file_uses_shared_cache_dir(monkeypatch, tmp_path) -> None:
    from src import settings
    from src.api.kis.client import KisApiClient
    from src.api.kis.key_pool import token_cache_path

    monkeypatch.setattr(settings, "KIS_TOKEN_CACHE_DIR", tmp_path)

    client = KisApiClient(app_key="k-default", app_secret="s")

    assert client.token_file == str(token_cache_path("k-default", tmp_path))


def test_ensure_token_force_refresh_adopts_token_refreshed_by_other_process(tmp_path) -> None:
    import asyncio
    import json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "tok.json"
    valid = (datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(hours=20)).isoformat(timespec="seconds")
    token_file.write_text(
        json.dumps({"access_token": "FRESH", "expired_at": valid, "app_key": "k", "issued_at": "2026-09-15T07:05:00+09:00"}),
        encoding="utf-8",
    )
    posts = {"n": 0}

    class _Resp:
        async def json(self):
            return {"access_token": "NEW", "expires_in": 86400}

    class _Ctx:
        async def __aenter__(self):
            posts["n"] += 1
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, _url, **_kw):
            return _Ctx()

    # Given: 다른 프로세스가 이미 FRESH로 갱신, 이 클라이언트는 STALE로 실패
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    client.token = "STALE"

    # When
    adopted = asyncio.run(client.ensure_token(_Session(), force_refresh=True))

    # Then: 발급 없이 채택
    assert adopted == "FRESH"
    assert posts["n"] == 0

    # And: 캐시 토큰 자체가 거부된 경우(내 토큰 == 캐시)는 실제 발급
    renewed = asyncio.run(client.ensure_token(_Session(), force_refresh=True))
    assert renewed == "NEW"
    assert posts["n"] == 1
    assert json.loads(token_file.read_text(encoding="utf-8"))["access_token"] == "NEW"


def test_ensure_token_records_issued_at_and_logs_repeat_issue(tmp_path, monkeypatch, caplog) -> None:
    import asyncio
    import json
    import logging
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from src.api.kis import client as client_mod
    from src.api.kis.client import KisApiClient

    monkeypatch.setattr(client_mod, "_now_kst", lambda: datetime(2026, 9, 15, 10, 0, 0, tzinfo=ZoneInfo("Asia/Seoul")))
    token_file = tmp_path / "tok.json"
    expired = datetime(2026, 9, 15, 9, 59, 0, tzinfo=ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    token_file.write_text(
        json.dumps({"access_token": "OLD", "expired_at": expired, "app_key": "k", "issued_at": "2026-09-15T07:05:00+09:00"}),
        encoding="utf-8",
    )

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

    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    with caplog.at_level(logging.INFO, logger="src.api.kis.client"):
        token = asyncio.run(client.ensure_token(_Session()))

    assert token == "NEW"
    saved = json.loads(token_file.read_text(encoding="utf-8"))
    assert saved["issued_at"] == "2026-09-15T10:00:00+09:00"
    messages = [r.getMessage() for r in caplog.records]
    assert any("stage=kis_token status=ISSUED" in m for m in messages)
    repeat = [r for r in caplog.records if "status=REPEAT_ISSUE" in r.getMessage()]
    assert len(repeat) == 1
    assert repeat[0].levelno == logging.ERROR
    assert all("NEW" not in m for m in messages)


def test_ensure_token_logs_decision_window_issue(tmp_path, monkeypatch, caplog) -> None:
    import asyncio
    import logging
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.api.kis import client as client_mod
    from src.api.kis.client import KisApiClient

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

    kst = ZoneInfo("Asia/Seoul")

    # Given/When: 결정창 안 발급
    monkeypatch.setattr(client_mod, "_now_kst", lambda: datetime(2026, 9, 15, 15, 20, 0, tzinfo=kst))
    inside = KisApiClient(app_key="k-in", app_secret="s", token_file=str(tmp_path / "in.json"))
    with caplog.at_level(logging.INFO, logger="src.api.kis.client"):
        asyncio.run(inside.ensure_token(_Session()))
    # Then
    hits = [r for r in caplog.records if "status=DECISION_WINDOW_ISSUE" in r.getMessage()]
    assert len(hits) == 1
    assert hits[0].levelno == logging.ERROR

    # And: 창 경계 밖(15:35)은 경보 없음
    caplog.clear()
    monkeypatch.setattr(client_mod, "_now_kst", lambda: datetime(2026, 9, 15, 15, 35, 0, tzinfo=kst))
    outside = KisApiClient(app_key="k-out", app_secret="s", token_file=str(tmp_path / "out.json"))
    with caplog.at_level(logging.INFO, logger="src.api.kis.client"):
        asyncio.run(outside.ensure_token(_Session()))
    assert not [r for r in caplog.records if "status=DECISION_WINDOW_ISSUE" in r.getMessage()]


def test_issue_daily_token_skips_when_issued_today_and_reissues_stale(tmp_path, monkeypatch) -> None:
    import asyncio
    import json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    from src.api.kis import client as client_mod
    from src.api.kis.client import KisApiClient

    monkeypatch.setattr(client_mod, "_now_kst", lambda: datetime(2026, 9, 16, 7, 5, 0, tzinfo=ZoneInfo("Asia/Seoul")))
    token_file = tmp_path / "tok.json"
    valid = (datetime(2026, 9, 16, 7, 5, 0, tzinfo=ZoneInfo("Asia/Seoul")) + timedelta(hours=20)).isoformat(timespec="seconds")
    posts = {"n": 0}
    reply = {"body": {"access_token": "NEW", "expires_in": 86400}}

    class _Resp:
        async def json(self):
            return reply["body"]

    class _Ctx:
        async def __aenter__(self):
            posts["n"] += 1
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, _url, **_kw):
            return _Ctx()

    def _seed(issued_at: str) -> None:
        token_file.write_text(
            json.dumps({"access_token": "CACHED", "expired_at": valid, "app_key": "k", "issued_at": issued_at}),
            encoding="utf-8",
        )

    # Case A: 오늘 이미 발급 → 스킵
    _seed("2026-09-16T07:05:01+09:00")
    client_a = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    assert asyncio.run(client_a.issue_daily_token(_Session())) is False
    assert client_a.token == "CACHED"
    assert posts["n"] == 0

    # Case B: 전일 발급(아직 유효) → 오늘 1회 발급으로 07:05 재고정
    _seed("2026-09-15T15:20:02+09:00")
    client_b = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    assert asyncio.run(client_b.issue_daily_token(_Session())) is True
    assert client_b.token == "NEW"
    assert posts["n"] == 1
    assert json.loads(token_file.read_text(encoding="utf-8"))["issued_at"].startswith("2026-09-16")

    # Case C: issued_at 없는 구버전 캐시 + 발급 스로틀 → 캐시 채택, 발급 아님
    token_file.write_text(json.dumps({"access_token": "LEGACY", "expired_at": valid, "app_key": "k"}), encoding="utf-8")
    reply["body"] = {"msg_cd": "EGW00133", "msg1": "throttled"}
    client_c = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    assert asyncio.run(client_c.issue_daily_token(_Session())) is False
    assert client_c.token == "LEGACY"


def test_ensure_token_cross_instance_file_lock_serializes_issuance(tmp_path) -> None:
    import asyncio

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "shared.json"
    posts = {"n": 0}

    class _Resp:
        async def json(self):
            return {"access_token": "TOK", "expires_in": 86400}

    class _Ctx:
        async def __aenter__(self):
            posts["n"] += 1
            await asyncio.sleep(0.05)
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, _url, **_kw):
            return _Ctx()

    # Given: 프로세스 경계를 모사하는 두 독립 인스턴스
    first = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    second = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    async def _run():
        session = _Session()
        return await asyncio.gather(first.ensure_token(session), second.ensure_token(session))

    # When
    tokens = asyncio.run(_run())

    # Then
    assert tokens == ["TOK", "TOK"]
    assert posts["n"] == 1

def test_read_cache_payload_rejects_non_dict_payload(tmp_path) -> None:
    import json

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "tok.json"
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    for payload in (["access_token", "TOK"], "TOK", 42):
        token_file.write_text(json.dumps(payload), encoding="utf-8")
        assert client._read_cache_payload() is None
        assert client._read_cached_token() is None


def _seed_valid_cache(token_file, access_token: str) -> None:
    import json
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    valid = (datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(hours=20)).isoformat(timespec="seconds")
    token_file.write_text(
        json.dumps({"access_token": access_token, "expired_at": valid, "app_key": "k", "issued_at": "2026-09-15T07:05:00+09:00"}),
        encoding="utf-8",
    )


class _IssueCountingSession:
    """Fake session whose POST issues one fixed token and counts issuance."""

    def __init__(self, issued: str = "NEW"):
        self.posts = 0
        self._issued = issued

    def post(self, _url, **_kw):
        session = self

        class _Resp:
            async def json(self):
                return {"access_token": session._issued, "expires_in": 86400}

        class _Ctx:
            async def __aenter__(self):
                session.posts += 1
                return _Resp()

            async def __aexit__(self, *_a):
                return False

        return _Ctx()


def test_handle_request_concurrent_rejects_issue_single_token(tmp_path, monkeypatch) -> None:
    import asyncio
    import json

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "tok.json"
    _seed_valid_cache(token_file, "T1")
    real_sleep = asyncio.sleep

    class _Resp:
        def __init__(self, body):
            self.status = 200
            self._body = body

        async def json(self):
            return self._body

    class _Ctx:
        def __init__(self, resp):
            self._resp = resp

        async def __aenter__(self):
            # Yield at the network boundary so every request is rejected with T1 before any refresh runs.
            await real_sleep(0)
            return self._resp

        async def __aexit__(self, *_a):
            return False

    class _Session(_IssueCountingSession):
        def __init__(self):
            super().__init__(issued="T2")

        def get(self, _url, **kw):
            if (kw.get("headers") or {}).get("authorization", "") == "Bearer T1":
                return _Ctx(_Resp({"rt_cd": "1", "msg_cd": "EGW00121", "msg1": "token expired"}))
            return _Ctx(_Resp({"rt_cd": "0", "output": {"ok": True}}))

    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    client.token = "T1"

    async def _free_acquire() -> None:
        return None

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client.rate_limiter, "acquire", _free_acquire)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    async def _main():
        session = _Session()
        results = await asyncio.gather(*[
            client._handle_request(session.get, "http://x", headers={"tr_id": "T"}) for _ in range(4)
        ])
        return session, results

    session, results = asyncio.run(_main())

    assert session.posts == 1
    assert [r["rt_cd"] for r in results] == ["0"] * 4
    assert json.loads(token_file.read_text(encoding="utf-8"))["access_token"] == "T2"


def test_ensure_token_rejected_token_adopts_rotated_cache(tmp_path, caplog) -> None:
    import asyncio
    import logging

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "tok.json"
    _seed_valid_cache(token_file, "T2")
    session = _IssueCountingSession()
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    client.token = "T1"

    with caplog.at_level(logging.INFO, logger="src.api.kis.client"):
        out = asyncio.run(client.ensure_token(session, rejected_token="T1"))

    assert out == "T2"
    assert client.token == "T2"
    assert session.posts == 0
    assert any("status=ADOPTED_ROTATED" in r.getMessage() for r in caplog.records)


def test_ensure_token_rejected_token_issues_and_keeps_cache_schema(tmp_path) -> None:
    import asyncio
    import json
    from datetime import datetime

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "tok.json"
    _seed_valid_cache(token_file, "T1")
    session = _IssueCountingSession()
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    client.token = "T1"

    out = asyncio.run(client.ensure_token(session, rejected_token="T1"))

    assert out == "NEW"
    assert session.posts == 1
    saved = json.loads(token_file.read_text(encoding="utf-8"))
    assert set(saved) == {"access_token", "expired_at", "app_key", "issued_at"}
    assert saved["access_token"] == "NEW"
    assert datetime.fromisoformat(saved["expired_at"]).tzinfo is not None
    assert datetime.fromisoformat(saved["issued_at"]).tzinfo is not None


def test_handle_request_passes_sent_token_as_rejected(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.kis.client import KisApiClient

    client = KisApiClient(app_key="k", app_secret="s", token_file=str(tmp_path / "tok.json"))
    client.token = "T1"

    async def _free_acquire() -> None:
        return None

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client.rate_limiter, "acquire", _free_acquire)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    seen: dict = {}

    async def _spy(session, *args, **kwargs):
        seen.update(kwargs)
        return "T2"

    client.ensure_token = _spy  # type: ignore[method-assign]
    calls = {"n": 0}

    class _Resp:
        status = 200

        async def json(self):
            calls["n"] += 1
            if calls["n"] == 1:
                client.token = "T2"
                return {"rt_cd": "1", "msg_cd": "EGW00121", "msg1": "expired"}
            return {"rt_cd": "0", "output": {}}

    class _Ctx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def get(self, _url, **_kw):
            return _Ctx()

    out = asyncio.run(client._handle_request(_Session().get, "http://x", headers={"tr_id": "T"}))

    assert out["rt_cd"] == "0"
    assert seen == {"rejected_token": "T1"}


def test_handle_request_without_token_adopts_cache(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "tok.json"
    _seed_valid_cache(token_file, "T2")
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    client.token = None

    async def _free_acquire() -> None:
        return None

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client.rate_limiter, "acquire", _free_acquire)
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    seen: dict = {}
    real_ensure = client.ensure_token

    async def _spy(wrapped_session, *args, **kwargs):
        seen.update(kwargs)
        return await real_ensure(wrapped_session, *args, **kwargs)

    client.ensure_token = _spy  # type: ignore[method-assign]
    calls = {"n": 0}

    class _Resp:
        status = 200

        async def json(self):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"rt_cd": "1", "msg_cd": "EGW00121", "msg1": "token expired"}
            return {"rt_cd": "0", "output": {}}

    class _Ctx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session(_IssueCountingSession):
        def get(self, _url, **_kw):
            return _Ctx()

    get_session = _Session()
    out = asyncio.run(client._handle_request(get_session.get, "http://x", headers={"tr_id": "T"}))

    assert out["rt_cd"] == "0"
    assert seen == {"rejected_token": ""}
    assert get_session.posts == 0
    assert client.token == "T2"

