from __future__ import annotations

import asyncio

from src.sync.fetcher_program import get_program_history_async


class _Client:
    base_url = "https://example.test"

    async def _handle_request(self, *args, **kwargs):
        return {"rt_cd": "0", "output": [{"stck_bsop_date": "20200102", "whol_smtn_ntby_tr_pbmn": "1234"}]}

    def _get_headers(self, tr_id):
        return {}


class _Session:
    def get(self, *args, **kwargs):
        raise AssertionError("fake request method should not be invoked directly")


class _FailingClient(_Client):
    async def _handle_request(self, *args, **kwargs):
        return {"rt_cd": "1", "msg1": "temporary failure"}


class _RaisingClient(_Client):
    async def _handle_request(self, *args, **kwargs):
        raise RuntimeError("connection reset")


def test_program_async_uses_rate_slot() -> None:
    calls = 0

    async def slot() -> None:
        nonlocal calls
        calls += 1

    out = asyncio.run(
        get_program_history_async(
            _Session(), _Client(), "005930", "20200102", "20200102", request_slot=slot
        )
    )
    assert calls == 1
    assert out == {"20200102": 1234.0}


def test_program_async_stops_after_consecutive_failures() -> None:
    calls = 0

    async def slot() -> None:
        nonlocal calls
        calls += 1

    out = asyncio.run(
        get_program_history_async(
            _Session(), _FailingClient(), "005930", "20200102", "20200110",
            request_slot=slot, max_consecutive_failures=2,
        )
    )
    assert out == {}
    assert calls == 2


def test_program_async_stops_after_request_errors() -> None:
    out = asyncio.run(
        get_program_history_async(
            _Session(), _RaisingClient(), "005930", "20200102", "20200110",
            max_consecutive_failures=2,
        )
    )
    assert out == {}


def test_get_program_history_builds_client_from_data_account(monkeypatch) -> None:
    from src.sync import fetcher_program

    captured: dict = {}

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc_info):
            return False

    class _FakeKisClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def create_session(self):
            return _FakeSession()

        async def ensure_token(self, session):
            return "tok"

    async def _fake_worker(*args, **kwargs):
        return {}

    monkeypatch.setattr(fetcher_program, "KisApiClient", _FakeKisClient)
    monkeypatch.setattr(fetcher_program, "get_program_history_async", _fake_worker)
    monkeypatch.setattr(
        fetcher_program,
        "kis_data_client_kwargs",
        lambda: {"app_key": "data-key", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "x.json"},
    )

    out = fetcher_program.get_program_history("005930", "20200102", "20200102")

    assert captured["app_key"] == "data-key"
    assert out == {}
