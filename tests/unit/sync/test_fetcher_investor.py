from __future__ import annotations

import asyncio

from src.sync.fetcher_investor import get_investor_trade_daily_async


class _Client:
    base_url = "https://example.test"

    async def _handle_request(self, *args, **kwargs):
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


def test_investor_async_stops_after_request_errors() -> None:
    out = asyncio.run(
        get_investor_trade_daily_async(
            _Session(), _RaisingClient(), "005930", "20200102", "20200110",
            max_consecutive_failures=2,
        )
    )
    assert out.empty


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
