from __future__ import annotations

import asyncio

from src.sync.fetcher_program import get_program_history_async


class _Client:
    async def get_program_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
        return {"rt_cd": "0", "output": [{"stck_bsop_date": "20200102", "whol_smtn_ntby_tr_pbmn": "1234"}]}


class _Session:
    def get(self, *args, **kwargs):
        raise AssertionError("fake request method should not be invoked directly")


class _FailingClient(_Client):
    async def get_program_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
        return {"rt_cd": "1", "msg1": "temporary failure"}


class _RaisingClient(_Client):
    async def get_program_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
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


def test_program_async_page_arguments_pinned() -> None:
    seen: list[tuple] = []

    class _Recording(_Client):
        async def get_program_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
            seen.append((code, cursor, market_div_code))
            return await super().get_program_trade_daily_page(session, code, cursor, market_div_code=market_div_code)

    asyncio.run(
        get_program_history_async(
            _Session(), _Recording(), "5930", "20200102", "20200102",
        )
    )
    assert seen[0] == ("005930", "20200102", "J")


def test_program_async_slot_precedes_page() -> None:
    events: list[str] = []

    async def slot() -> None:
        events.append("slot")

    class _Recording(_Client):
        async def get_program_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
            events.append("page")
            return await super().get_program_trade_daily_page(session, code, cursor, market_div_code=market_div_code)

    asyncio.run(
        get_program_history_async(
            _Session(), _Recording(), "005930", "20200102", "20200102", request_slot=slot
        )
    )
    assert events == ["slot", "page"]


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


def test_program_async_consecutive_failures_use_descending_cursors() -> None:
    cursors: list[str] = []

    class _RecordingFail(_FailingClient):
        async def get_program_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
            cursors.append(cursor)
            return await super().get_program_trade_daily_page(session, code, cursor, market_div_code=market_div_code)

    asyncio.run(
        get_program_history_async(
            _Session(), _RecordingFail(), "005930", "20200102", "20200110",
            max_consecutive_failures=2,
        )
    )
    assert cursors == ["20200110", "20200109"]


def test_program_async_empty_page_jumps_30_days() -> None:
    cursors: list[str] = []

    class _Empty(_Client):
        async def get_program_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
            cursors.append(cursor)
            return {"rt_cd": "0", "output": []}

    out = asyncio.run(
        get_program_history_async(
            _Session(), _Empty(), "005930", "20200101", "20200301",
        )
    )
    assert out == {}
    assert cursors == ["20200301", "20200131", "20200101"]


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
