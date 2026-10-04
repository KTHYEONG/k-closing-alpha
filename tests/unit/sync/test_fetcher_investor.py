from __future__ import annotations

import asyncio
import logging

from src.sync.fetcher_investor import _prev_day_ymd, get_investor_trade_daily_async


class _Client:
    async def get_investor_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
        return {
            "rt_cd": "0",
            "output2": [
                {
                    "stck_bsop_date": "20200102",
                    "frgn_ntby_tr_pbmn": "1000",
                    "orgn_ntby_tr_pbmn": "-2000",
                    "frgn_ntby_qty": "999999",
                    "orgn_ntby_qty": "999999",
                }
            ],
        }


class _Session:
    def get(self, *args, **kwargs):
        raise AssertionError("fake request method should not be invoked directly")


class _FailingClient(_Client):
    async def get_investor_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
        return {"rt_cd": "1", "msg1": "temporary failure"}


class _RaisingClient(_Client):
    async def get_investor_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
        raise RuntimeError("connection reset")


def test_investor_async_uses_amount_and_rate_slot() -> None:
    calls = 0

    async def slot() -> None:
        nonlocal calls
        calls += 1

    out = asyncio.run(
        get_investor_trade_daily_async(
            _Session(), _Client(), "005930", "20200102", "20200102", request_slot=slot
        )
    )
    assert calls == 1
    assert out.loc[0, "foreign_netbuy"] == 1000.0
    assert out.loc[0, "inst_netbuy"] == -2000.0


def test_investor_async_page_arguments_pinned() -> None:
    seen: list[tuple] = []

    class _Recording(_Client):
        async def get_investor_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
            seen.append((code, cursor, market_div_code))
            return await super().get_investor_trade_daily_page(session, code, cursor, market_div_code=market_div_code)

    asyncio.run(
        get_investor_trade_daily_async(
            _Session(), _Recording(), "5930", "20200102", "20200102",
        )
    )
    assert seen[0] == ("005930", "20200102", "J")


def test_investor_async_slot_precedes_page() -> None:
    events: list[str] = []

    async def slot() -> None:
        events.append("slot")

    class _Recording(_Client):
        async def get_investor_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
            events.append("page")
            return await super().get_investor_trade_daily_page(session, code, cursor, market_div_code=market_div_code)

    asyncio.run(
        get_investor_trade_daily_async(
            _Session(), _Recording(), "005930", "20200102", "20200102", request_slot=slot
        )
    )
    assert events == ["slot", "page"]


def test_investor_async_stops_after_consecutive_failures() -> None:
    calls = 0

    async def slot() -> None:
        nonlocal calls
        calls += 1

    out = asyncio.run(
        get_investor_trade_daily_async(
            _Session(), _FailingClient(), "005930", "20200102", "20200110",
            request_slot=slot, max_consecutive_failures=2,
        )
    )
    assert out.empty
    assert calls == 2


def test_investor_async_consecutive_failures_use_descending_cursors() -> None:
    cursors: list[str] = []

    class _RecordingFail(_FailingClient):
        async def get_investor_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
            cursors.append(cursor)
            return await super().get_investor_trade_daily_page(session, code, cursor, market_div_code=market_div_code)

    asyncio.run(
        get_investor_trade_daily_async(
            _Session(), _RecordingFail(), "005930", "20200102", "20200110",
            max_consecutive_failures=2,
        )
    )
    assert cursors == ["20200110", "20200109"]


def test_investor_async_rt_cd_9_counted_as_api_failure(caplog) -> None:
    pages = 0

    class _Rt9:
        async def get_investor_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
            nonlocal pages
            pages += 1
            return {"rt_cd": "9", "msg1": "최대 재시도 횟수 초과 (TPS 제한)"}

    with caplog.at_level(logging.WARNING):
        out = asyncio.run(
            get_investor_trade_daily_async(
                _Session(), _Rt9(), "005930", "20200102", "20200110",
                max_consecutive_failures=1,
            )
        )
    assert out.empty
    assert pages == 1
    assert any("status=API_FAIL" in r.message for r in caplog.records)


def test_investor_async_stops_after_request_errors() -> None:
    out = asyncio.run(
        get_investor_trade_daily_async(
            _Session(), _RaisingClient(), "005930", "20200102", "20200110",
            max_consecutive_failures=2,
        )
    )
    assert out.empty


def test_investor_async_slot_failure_is_request_failure(caplog) -> None:
    called = False

    class _NeverCalled(_Client):
        async def get_investor_trade_daily_page(self, session, code, cursor, *, market_div_code="J"):
            nonlocal called
            called = True
            raise AssertionError("page must not be called")

    async def bad_slot() -> None:
        raise RuntimeError("slot boom")

    with caplog.at_level(logging.WARNING):
        out = asyncio.run(
            get_investor_trade_daily_async(
                _Session(), _NeverCalled(), "005930", "20200102", "20200102",
                request_slot=bad_slot, max_consecutive_failures=1,
            )
        )
    assert out.empty
    assert called is False
    assert any("status=REQUEST_FAIL" in r.message for r in caplog.records)


def test_get_investor_trade_daily_builds_client_from_data_account(monkeypatch) -> None:
    import pandas as pd

    from src.sync import fetcher_investor

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
        return pd.DataFrame()

    monkeypatch.setattr(fetcher_investor, "KisApiClient", _FakeKisClient)
    monkeypatch.setattr(fetcher_investor, "get_investor_trade_daily_async", _fake_worker)
    monkeypatch.setattr(
        fetcher_investor,
        "kis_data_client_kwargs",
        lambda: {"app_key": "data-key", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": "x.json"},
    )

    out = fetcher_investor.get_investor_trade_daily("005930", "20200102", "20200102")

    assert captured["app_key"] == "data-key"
    assert out.empty


def test_prev_day_ymd_returns_string_and_passes_through_invalid_input() -> None:
    result = _prev_day_ymd("20260105", days=1)
    assert result == "20260104"
    assert isinstance(result, str)
    assert _prev_day_ymd("2026XX05") == "2026XX05"
