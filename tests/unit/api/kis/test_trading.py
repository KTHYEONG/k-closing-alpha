"""Invariant guard tests for the KIS order/account transport."""

from __future__ import annotations

import asyncio
import logging

import aiohttp
import pytest

from src.api.kis.client import KisApiClient
from src.api.kis.trading import (
    ORDER_EXCHANGE_CODE,
    TR_SUFFIX_BALANCE,
    TR_SUFFIX_BUY,
    TR_SUFFIX_BUYING_POWER,
    TR_SUFFIX_DAILY_ORDERS,
    TR_SUFFIX_REVISE_CANCEL,
    TR_SUFFIX_SELL,
    BrokerReadError,
    KisTradingClient,
    TradingBinding,
    TradingConfigError,
    TradingDisabledError,
    resolve_trading_binding,
)
from src.config import Settings
from src.execution.order_contract import OrderIntent, OrderPhase, OrderSide, OrderType, SubmitVerdict
from tests.broker_fakes import FakeBrokerResponse, FakeBrokerSession, scripted_session


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    import os
    import src.config as config

    for name in tuple(os.environ):
        if name.startswith(("KIS_", "ORDER_")):
            monkeypatch.delenv(name)
    monkeypatch.setattr(config, "settings", _mock_settings(ORDER_SEND_MAX_ATTEMPTS=3))


def _mock_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "KIS_TRADING_ENV": "mock",
        "KIS_MOCK_APP_KEY": "MOCKKEY1",
        "KIS_MOCK_APP_SECRET": "MOCKSECRET1",
        "KIS_MOCK_ACCOUNT_NO": "12345678",
        "KIS_MOCK_ACCOUNT_PRODUCT_CODE": "01",
        "KIS_APP_KEY": "LIVEKEY111",
        "KIS_APP_SECRET": "LIVESECRET1",
        "KIS_ACCOUNT_ID": "87654321",
        "KIS_HTS_ID": "hts1",
        "KIS_MOCK_REST_TPS": 1000.0,
        "KIS_MOCK_BASE_URL": "https://openapivts.koreainvestment.com:29443",
        "KIS_BASE_URL": "https://openapi.koreainvestment.com:9443",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _live_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "KIS_TRADING_ENV": "live",
        "KIS_APP_KEY": "LIVEKEY111",
        "KIS_APP_SECRET": "LIVESECRET1",
        "KIS_ACCOUNT_ID": "12345678",
        "KIS_ACCOUNT_PRODUCT_CODE": "01",
        "KIS_HTS_ID": "hts1",
        "KIS_MOCK_APP_KEY": "MOCKKEY111",
        "KIS_LIVE_TRADING_CONFIRM": "LIVE-5678",
        "KIS_BASE_URL": "https://openapi.koreainvestment.com:9443",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def _mock_binding(**overrides: object) -> TradingBinding:
    return resolve_trading_binding(_mock_settings(**overrides))


class _CountingLimiter:
    def __init__(self) -> None:
        self.n = 0

    async def acquire(self) -> None:
        self.n += 1


def _trading_client(binding: TradingBinding, tmp_path: object, bearer: str | None = "T") -> tuple:
    api = KisApiClient(
        app_key=binding.app_key,
        app_secret=binding.app_secret,
        base_url=binding.base_url,
        token_file=str(tmp_path / f"tok_{binding.env}.json"),  # type: ignore[operator]
        rest_tps=binding.rest_tps,
    )
    api.token = bearer
    limiter = _CountingLimiter()
    api.rate_limiter = limiter  # type: ignore[assignment]
    return KisTradingClient(binding, http_timeout_seconds=5.0, api_client=api), api, limiter


def _limit_buy() -> OrderIntent:
    return OrderIntent(
        intent_id="2026-01-02:005930:entry",
        trading_date="2026-01-02",
        phase=OrderPhase.ENTRY,
        symbol="005930",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        qty=10,
        limit_price=70000,
        decision_price=69800,
        parent_intent_id=None,
    )


def _market_sell() -> OrderIntent:
    return OrderIntent(
        intent_id="2026-01-02:005930:entry:exit:2026-01-05",
        trading_date="2026-01-05",
        phase=OrderPhase.EXIT,
        symbol="005930",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        qty=5,
        limit_price=None,
        decision_price=None,
        parent_intent_id="2026-01-02:005930:entry",
    )


def _accept_body() -> dict:
    return {
        "rt_cd": "0",
        "msg_cd": "00000000",
        "msg1": "OK",
        "output": {"ODNO": "00012345", "KRX_FWDG_ORD_ORGNO": "00123"},
    }


def _buy_row(**overrides: object) -> dict:
    row: dict[str, object] = {
        "odno": "00012345",
        "orgn_odno": "00012344",
        "ord_gno_brno": "00123",
        "pdno": "005930",
        "sll_buy_dvsn_cd": "02",
        "ord_qty": "10",
        "ord_unpr": "70000",
        "tot_ccld_qty": "10",
        "tot_ccld_amt": "700000",
        "rmn_qty": "0",
        "rjct_qty": "0",
        "cncl_yn": "N",
    }
    row.update(overrides)
    return row


def _sell_row(**overrides: object) -> dict:
    row = _buy_row(**overrides)
    row.update({"odno": "00012346", "orgn_odno": "00012346", "sll_buy_dvsn_cd": "01", "cncl_yn": "Y"})
    return row


def _daily_page(rows: list, **overrides: object) -> FakeBrokerResponse:
    body: dict[str, object] = {
        "rt_cd": "0",
        "msg_cd": "00000000",
        "msg1": "OK",
        "output1": rows,
        "ctx_area_fk100": "",
        "ctx_area_nk100": "",
    }
    body.update(overrides)
    return FakeBrokerResponse(body=body, status=200, headers={"tr_cont": "D"})


def test_disabled_environment_has_no_binding() -> None:
    with pytest.raises(TradingDisabledError):
        resolve_trading_binding(_mock_settings(KIS_TRADING_ENV="off"))


def test_default_binding_reads_live_settings(monkeypatch) -> None:
    import src.config as config

    monkeypatch.setattr(config, "settings", _mock_settings(KIS_TRADING_ENV="off"))
    with pytest.raises(TradingDisabledError):
        resolve_trading_binding()
    monkeypatch.setattr(config, "settings", _mock_settings())
    assert resolve_trading_binding().env == "mock"


def test_rate_limiter_property_delegates(tmp_path) -> None:
    client, api, _ = _trading_client(_mock_binding(), tmp_path)
    assert client.rate_limiter is api.rate_limiter
    replacement = _CountingLimiter()
    client.rate_limiter = replacement  # type: ignore[assignment]
    assert client.rate_limiter is replacement


def test_unknown_env_is_refused() -> None:
    cfg = _mock_settings()
    cfg.KIS_TRADING_ENV = "weird"  # type: ignore[assignment]
    with pytest.raises(TradingConfigError):
        resolve_trading_binding(cfg)


def test_mock_binding_resolves_with_v_prefix() -> None:
    binding = _mock_binding()
    assert binding.env == "mock"
    assert binding.base_url == "https://openapivts.koreainvestment.com:29443"
    assert binding.tr_id(TR_SUFFIX_BUY) == "V" + TR_SUFFIX_BUY
    assert binding.rest_tps == 1000.0


def test_mock_env_on_live_host_is_refused() -> None:
    with pytest.raises(TradingConfigError, match="mock"):
        _mock_binding(KIS_MOCK_BASE_URL="https://openapi.koreainvestment.com:9443")


def test_mock_key_equal_to_primary_is_refused() -> None:
    with pytest.raises(TradingConfigError, match="mock"):
        _mock_binding(KIS_MOCK_APP_KEY="LIVEKEY111")


def test_mock_account_equal_to_live_is_refused() -> None:
    with pytest.raises(TradingConfigError, match="mock"):
        _mock_binding(KIS_MOCK_ACCOUNT_NO="87654321")


def test_mock_blank_credentials_refused() -> None:
    with pytest.raises(TradingConfigError, match="mock"):
        _mock_binding(KIS_MOCK_APP_KEY="  ")


def test_live_binding_requires_confirm_phrase() -> None:
    for bad in ("", "LIVE", "LIVE-0000"):
        with pytest.raises(TradingConfigError, match="live"):
            resolve_trading_binding(_live_settings(KIS_LIVE_TRADING_CONFIRM=bad))
    binding = resolve_trading_binding(_live_settings())
    assert binding.env == "live"
    assert binding.tr_id(TR_SUFFIX_SELL) == "T" + TR_SUFFIX_SELL


def test_live_env_on_mock_host_is_refused() -> None:
    with pytest.raises(TradingConfigError, match="live"):
        resolve_trading_binding(_live_settings(KIS_BASE_URL="https://openapivts.koreainvestment.com:29443"))


def test_live_blank_credentials_refused() -> None:
    with pytest.raises(TradingConfigError, match="live"):
        resolve_trading_binding(_live_settings(KIS_APP_KEY=""))


def test_live_key_colliding_with_mock_is_refused() -> None:
    with pytest.raises(TradingConfigError, match="live"):
        resolve_trading_binding(_live_settings(KIS_APP_KEY="MOCKKEY111"))


def test_malformed_account_or_product_refused() -> None:
    with pytest.raises(TradingConfigError):
        _mock_binding(KIS_MOCK_ACCOUNT_NO="1234-5678")
    with pytest.raises(TradingConfigError):
        _mock_binding(KIS_MOCK_ACCOUNT_NO="1234567")
    with pytest.raises(TradingConfigError):
        _mock_binding(KIS_MOCK_ACCOUNT_PRODUCT_CODE="1")
    with pytest.raises(TradingConfigError):
        resolve_trading_binding(_live_settings(KIS_ACCOUNT_ID="1234", KIS_LIVE_TRADING_CONFIRM="LIVE-1234"))
    with pytest.raises(TradingConfigError):
        resolve_trading_binding(_live_settings(KIS_ACCOUNT_PRODUCT_CODE="1"))


def test_config_errors_never_leak_credentials() -> None:
    sentinels = ("SENTKEY9", "SENTSEC9", "99887766", "LIVE-9999")
    failing = [
        lambda: _mock_binding(KIS_MOCK_BASE_URL="http://x"),
        lambda: _mock_binding(KIS_MOCK_APP_KEY="SENTKEY9", KIS_APP_KEY="SENTKEY9"),
        lambda: _mock_binding(KIS_MOCK_ACCOUNT_NO="99887766", KIS_ACCOUNT_ID="99887766"),
        lambda: resolve_trading_binding(_live_settings(KIS_LIVE_TRADING_CONFIRM="LIVE-9999")),
    ]
    for build in failing:
        with pytest.raises(TradingConfigError) as exc_info:
            build()
        assert all(s not in str(exc_info.value) for s in sentinels)
    binding = _mock_binding(
        KIS_MOCK_APP_KEY="SENTKEY9A",
        KIS_MOCK_APP_SECRET="SENTSEC9B",
        KIS_MOCK_ACCOUNT_NO="99887755",
    )
    assert all(s not in repr(binding) for s in ("SENTKEY9A", "SENTSEC9B", "99887755"))


def test_unknown_tr_suffix_refused() -> None:
    with pytest.raises(ValueError, match="suffix"):
        _mock_binding().tr_id("TTC0802U")


def test_default_client_owns_keyed_api_client(tmp_path) -> None:
    binding = _mock_binding()
    client = KisTradingClient(binding, http_timeout_seconds=5.0)
    assert client.api_client.app_key == binding.app_key
    assert client.api_client.base_url == binding.base_url
    assert client.api_client.rate_limiter.max_rate == binding.rest_tps
    live = KisTradingClient(resolve_trading_binding(_live_settings()), http_timeout_seconds=5.0)
    assert live.api_client.token_file != client.api_client.token_file


def test_create_session_uses_configured_timeout(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)

    async def _run() -> None:
        session = client.create_session()
        try:
            assert session.timeout.total == 5.0
        finally:
            await session.close()

    asyncio.run(_run())


def test_ensure_token_loads_and_rotates(tmp_path) -> None:
    client, api, _ = _trading_client(_mock_binding(), tmp_path, bearer=None)

    class _Resp:
        status = 200

        def __init__(self, payload: dict) -> None:
            self._payload = payload

        async def json(self) -> dict:
            return self._payload

    class _Ctx:
        def __init__(self, resp: _Resp) -> None:
            self._resp = resp

        async def __aenter__(self) -> _Resp:
            return self._resp

        async def __aexit__(self, *_a: object) -> bool:
            return False

    class _Session:
        def __init__(self) -> None:
            self.posts = 0

        def post(self, _url: str, **_kw: object) -> _Ctx:
            self.posts += 1
            return _Ctx(_Resp({"access_token": "TOK", "expires_in": 86400}))

    async def _run() -> None:
        session = _Session()
        await client.ensure_token(session)  # type: ignore[arg-type]
        assert api.token == "TOK"
        assert session.posts == 1
        await client.ensure_token(session, rejected_token="OTHER")  # type: ignore[arg-type]
        assert api.token == "TOK"
        assert session.posts == 1

    asyncio.run(_run())


def test_limit_buy_sends_documented_body_once(tmp_path) -> None:
    client, _, limiter = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([FakeBrokerResponse(body=_accept_body(), status=200)])

    async def _run() -> object:
        return await client.submit_order(session, _limit_buy())  # type: ignore[arg-type]

    result = asyncio.run(_run())
    assert result.verdict is SubmitVerdict.ACCEPTED
    assert result.broker_order_no == "12345"
    assert result.broker_org_no == "00123"
    assert len(session.requests) == 1
    assert limiter.n == 1
    req = session.requests[0]
    assert req.method == "POST"
    assert req.url.endswith("/uapi/domestic-stock/v1/trading/order-cash")
    assert req.headers["tr_id"] == "V" + TR_SUFFIX_BUY
    assert req.json == {
        "CANO": "12345678",
        "ACNT_PRDT_CD": "01",
        "PDNO": "005930",
        "ORD_DVSN": "00",
        "ORD_QTY": "10",
        "ORD_UNPR": "70000",
        "EXCG_ID_DVSN_CD": ORDER_EXCHANGE_CODE,
    }
    assert "SLL_TYPE" not in req.json


def test_market_sell_sends_price_zero_and_sell_type(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([FakeBrokerResponse(body=_accept_body(), status=200)])
    asyncio.run(client.submit_order(session, _market_sell()))  # type: ignore[arg-type]
    req = session.requests[0]
    assert req.headers["tr_id"] == "V" + TR_SUFFIX_SELL
    assert req.json["ORD_DVSN"] == "01"
    assert req.json["ORD_UNPR"] == "0"
    assert req.json["SLL_TYPE"] == "01"


def test_accept_without_org_number_keeps_order_number(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    body = _accept_body()
    body["output"] = {"ODNO": "00012345"}
    session = scripted_session([FakeBrokerResponse(body=body, status=200)])
    result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.ACCEPTED
    assert result.broker_order_no == "12345"
    assert result.broker_org_no is None


def test_timeout_is_ambiguous_and_never_resent(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([FakeBrokerResponse(enter_error=TimeoutError("slow"), status=200)])
    result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.AMBIGUOUS
    assert len(session.requests) == 1


def test_disconnect_after_send_is_ambiguous(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([FakeBrokerResponse(enter_error=aiohttp.ServerDisconnectedError(), status=200)])
    result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.AMBIGUOUS
    assert len(session.requests) == 1


def test_connection_failure_is_not_sent(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session(
        [FakeBrokerResponse(enter_error=aiohttp.ClientConnectorError(None, OSError("down")), status=200)]
    )
    result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.NOT_SENT
    assert len(session.requests) == 1


def test_decode_failure_is_ambiguous_with_status(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([FakeBrokerResponse(body={"rt_cd": "0"}, status=200, json_error=ValueError("bad"))])
    result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.AMBIGUOUS
    assert result.http_status == 200


def test_non_dict_body_is_ambiguous(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([FakeBrokerResponse(body=None, status=200)])
    result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.AMBIGUOUS


def test_throttle_is_returned_not_retried(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session(
        [FakeBrokerResponse(body={"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "throttled"}, status=500)]
    )
    result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.THROTTLED
    assert result.msg_cd == "EGW00201"
    assert len(session.requests) == 1


def test_auth_rejection_refreshes_token_then_returns(tmp_path) -> None:
    client, api, _ = _trading_client(_mock_binding(), tmp_path, bearer="OLD")
    session = scripted_session(
        [FakeBrokerResponse(body={"rt_cd": "1", "msg_cd": "EGW00123", "msg1": "expired"}, status=200)],
        token_bodies=[{"access_token": "NEW", "expires_in": 86400}],
        token_path="/oauth2/tokenP",
    )
    result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.AUTH_REJECTED
    assert api.token == "NEW"
    assert len(session.requests_to("/uapi/domestic-stock/v1/trading/order-cash")) == 1


def test_order_send_never_replays(tmp_path, monkeypatch) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path, bearer="OLD")
    seen: list[str | None] = []
    original = client.ensure_token

    async def _spy(session, *, rejected_token=None):
        seen.append(rejected_token)
        await original(session, rejected_token=rejected_token)

    monkeypatch.setattr(client, "ensure_token", _spy)
    session = scripted_session(
        [FakeBrokerResponse(body={"rt_cd": "1", "msg_cd": "EGW00123", "msg1": "expired"}, status=200)],
        token_bodies=[{"access_token": "NEW", "expires_in": 86400}],
        token_path="/oauth2/tokenP",
    )
    result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.AUTH_REJECTED
    assert len(session.requests_to("/uapi/domestic-stock/v1/trading/order-cash")) == 1
    assert seen == ["OLD"]


def test_auth_rejection_with_failed_refresh_still_returns(tmp_path, caplog) -> None:
    client, api, _ = _trading_client(_mock_binding(), tmp_path, bearer="OLD")
    session = scripted_session(
        [FakeBrokerResponse(body={"rt_cd": "1", "msg_cd": "EGW00123", "msg1": "expired"}, status=200)],
        token_bodies=[{"rt_cd": "1", "msg_cd": "XXXX"}],
        token_path="/oauth2/tokenP",
    )
    with caplog.at_level(logging.WARNING):
        result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.AUTH_REJECTED
    assert api.token == "OLD"
    assert any("TOKEN_REFRESH_FAILED" in rec.message for rec in caplog.records)


def test_non_business_day_rejection_is_definite(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session(
        [FakeBrokerResponse(body={"rt_cd": "1", "msg_cd": "40100000", "msg1": "holiday"}, status=200)]
    )
    result = asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.REJECTED
    assert result.broker_order_no is None


def test_every_send_passes_host_admission(tmp_path) -> None:
    client, _, limiter = _trading_client(_mock_binding(), tmp_path)
    order_session = scripted_session([FakeBrokerResponse(body=_accept_body(), status=200)])
    read_session = scripted_session(
        [FakeBrokerResponse(body={"rt_cd": "0", "output1": [], "ctx_area_fk100": "", "ctx_area_nk100": ""}, status=200)]
    )

    async def _run() -> None:
        await client.submit_order(order_session, _limit_buy())  # type: ignore[arg-type]
        await client.fetch_positions(read_session)  # type: ignore[arg-type]

    asyncio.run(_run())
    assert limiter.n == 2


def test_cancel_sends_full_quantity_cancel_once(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([FakeBrokerResponse(body=_accept_body(), status=200)])
    result = asyncio.run(client.cancel_order(session, org_no="00123", order_no="00012345"))  # type: ignore[arg-type]
    assert result.verdict is SubmitVerdict.ACCEPTED
    assert len(session.requests) == 1
    req = session.requests[0]
    assert req.url.endswith("/uapi/domestic-stock/v1/trading/order-rvsecncl")
    assert req.headers["tr_id"] == "V" + TR_SUFFIX_REVISE_CANCEL
    assert req.json == {
        "CANO": "12345678",
        "ACNT_PRDT_CD": "01",
        "KRX_FWDG_ORD_ORGNO": "00123",
        "ORGN_ODNO": "00012345",
        "ORD_DVSN": "00",
        "RVSE_CNCL_DVSN_CD": "02",
        "ORD_QTY": "0",
        "ORD_UNPR": "0",
        "QTY_ALL_ORD_YN": "Y",
        "EXCG_ID_DVSN_CD": ORDER_EXCHANGE_CODE,
    }


def test_order_log_line_carries_no_secrets(tmp_path, caplog) -> None:
    binding = _mock_binding(KIS_MOCK_APP_KEY="LOGKEY99", KIS_MOCK_APP_SECRET="LOGSEC99")
    client, _, _ = _trading_client(binding, tmp_path)
    session = scripted_session([FakeBrokerResponse(body=_accept_body(), status=200)])
    with caplog.at_level(logging.INFO):
        asyncio.run(client.submit_order(session, _limit_buy()))  # type: ignore[arg-type]
    line = next(rec.message for rec in caplog.records if "stage=kis_order" in rec.message)
    assert "2026-01-02:005930:entry" in line
    assert "accepted" in line
    assert "LOGKEY99" not in line and "LOGSEC99" not in line and "12345678" not in line


def test_empty_order_list_is_empty_result(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session(
        [FakeBrokerResponse(body={"rt_cd": "0", "msg_cd": "70070000", "msg1": "empty", "output1": []}, status=200)]
    )
    assert asyncio.run(client.fetch_daily_orders(session, "2026-01-02")) == []  # type: ignore[arg-type]


def test_order_rows_parsed_with_normalized_numbers(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([_daily_page([_buy_row(), _sell_row()])])

    async def _run() -> list:
        return await client.fetch_daily_orders(session, "2026-01-02")  # type: ignore[arg-type]

    orders = asyncio.run(_run())
    assert len(orders) == 2
    buy, sell = orders
    assert (buy.side, buy.order_no, buy.orig_order_no, buy.org_no) == ("buy", "12345", "12344", "00123")
    assert (buy.order_qty, buy.order_price, buy.filled_qty, buy.filled_amount_krw) == (10, 70000, 10, 700000)
    assert (buy.remaining_qty, buy.rejected_qty, buy.cancelled) == (0, 0, False)
    assert sell.side == "sell" and sell.order_no == "12346" and sell.cancelled is True
    req = session.requests[0]
    assert req.headers["tr_id"] == "V" + TR_SUFFIX_DAILY_ORDERS
    assert req.params["INQR_STRT_DT"] == "20260102" and req.params["INQR_END_DT"] == "20260102"


def test_cancel_child_row_folds_into_parent_and_leaves_no_candidate(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    parent = _buy_row(odno="0000010567", orgn_odno="0000000000", ord_qty="2", tot_ccld_qty="0", tot_ccld_amt="0", rmn_qty="0")
    child = _buy_row(odno="0000010596", orgn_odno="0000010567", ord_qty="2", tot_ccld_qty="0", tot_ccld_amt="0", rmn_qty="0", cncl_yn="Y")
    session = scripted_session([_daily_page([child, parent])])

    async def _run() -> list:
        return await client.fetch_daily_orders(session, "2026-01-02")  # type: ignore[arg-type]

    orders = asyncio.run(_run())
    assert [o.order_no for o in orders] == ["10567"]
    assert orders[0].cancelled is True and orders[0].filled_qty == 0


def test_bad_trading_date_fails_closed(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(FakeBrokerSession(lambda req: FakeBrokerResponse()), "bad-date"))  # type: ignore[arg-type]


def test_unreadable_numeric_field_fails_closed(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([_daily_page([_buy_row(tot_ccld_qty="")])])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]
    session = scripted_session([_daily_page([_buy_row(ord_qty="")])])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]
    session = scripted_session([_daily_page([_buy_row(tot_ccld_amt="")])])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]


def test_unknown_side_code_fails_closed(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([_daily_page([_buy_row(sll_buy_dvsn_cd="03")])])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]


def test_missing_identifiers_fail_closed(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([_daily_page([_buy_row(odno="")])])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]
    session = scripted_session([_daily_page(["not-a-row"])])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]


def test_missing_or_misshaped_output_fails_closed(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([FakeBrokerResponse(body={"rt_cd": "0"}, status=200)])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]
    session = scripted_session([FakeBrokerResponse(body={"rt_cd": "0", "output1": {}}, status=200)])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]
    session = scripted_session([FakeBrokerResponse(body=None, status=200)])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]


def test_continuation_is_followed_to_the_end(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    first = FakeBrokerResponse(
        body={
            "rt_cd": "0",
            "output1": [_buy_row()],
            "ctx_area_fk100": "FK1",
            "ctx_area_nk100": "NK1",
        },
        status=200,
        headers={"tr_cont": "M"},
    )
    second = FakeBrokerResponse(
        body={"rt_cd": "0", "output1": [_sell_row()], "ctx_area_fk100": "", "ctx_area_nk100": ""},
        status=200,
        headers={"tr_cont": "D"},
    )
    session = scripted_session([first, second])
    orders = asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]
    assert [o.order_no for o in orders] == ["12345", "12346"]
    assert len(session.requests) == 2
    assert session.requests[1].headers["tr_cont"] == "N"
    assert session.requests[1].params["CTX_AREA_FK100"] == "FK1"
    assert session.requests[1].params["CTX_AREA_NK100"] == "NK1"


def test_runaway_continuation_fails_closed(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    page = FakeBrokerResponse(
        body={"rt_cd": "0", "output1": [], "ctx_area_fk100": "", "ctx_area_nk100": ""},
        status=200,
        headers={"tr_cont": "M"},
    )
    session = scripted_session([page] * 10)
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]
    assert len(session.requests) == 10


def test_read_retries_throttle_then_succeeds(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session(
        [
            FakeBrokerResponse(body={"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "busy"}, status=500),
            _daily_page([_buy_row()]),
        ]
    )
    orders = asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]
    assert [o.order_no for o in orders] == ["12345"]
    assert len(session.requests) == 2


def test_read_exhausts_throttles_and_fails_closed(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    throttle = FakeBrokerResponse(body={"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "busy"}, status=500)
    session = scripted_session([throttle] * 3)
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]
    assert len(session.requests) == 3


def test_read_error_code_fails_closed(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session(
        [FakeBrokerResponse(body={"rt_cd": "1", "msg_cd": "XXXX", "msg1": "denied"}, status=200)]
    )
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_positions(session))  # type: ignore[arg-type]


def test_read_transport_failure_fails_closed(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)

    class _Down:
        def get(self, _url: str, **_kw: object) -> FakeBrokerResponse:
            raise ValueError("down")

    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_positions(_Down()))  # type: ignore[arg-type]


def test_read_refreshes_token_on_auth_rejection(tmp_path) -> None:
    client, api, _ = _trading_client(_mock_binding(), tmp_path, bearer="OLD")
    session = scripted_session(
        [
            FakeBrokerResponse(body={"rt_cd": "1", "msg_cd": "EGW00123", "msg1": "expired"}, status=200),
            _daily_page([_buy_row()]),
        ],
        token_bodies=[{"access_token": "NEW", "expires_in": 86400}],
        token_path="/oauth2/tokenP",
    )
    orders = asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))  # type: ignore[arg-type]
    assert [o.order_no for o in orders] == ["12345"]
    assert api.token == "NEW"


def test_zero_quantity_holdings_dropped(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    rows = [
        {"pdno": "005930", "hldg_qty": "0", "ord_psbl_qty": "0"},
        {"pdno": "000660", "hldg_qty": "5", "ord_psbl_qty": "3"},
    ]
    session = scripted_session(
        [FakeBrokerResponse(body={"rt_cd": "0", "output1": rows}, status=200, headers={"tr_cont": "D"})]
    )
    positions = asyncio.run(client.fetch_positions(session))  # type: ignore[arg-type]
    assert [(p.symbol, p.holding_qty, p.sellable_qty) for p in positions] == [("000660", 5, 3)]
    assert session.requests[0].headers["tr_id"] == "V" + TR_SUFFIX_BALANCE


def test_balance_continuation_and_runaway(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    first = FakeBrokerResponse(
        body={
            "rt_cd": "0",
            "output1": [{"pdno": "005930", "hldg_qty": "1", "ord_psbl_qty": "1"}],
            "ctx_area_fk100": "F",
            "ctx_area_nk100": "N",
        },
        status=200,
        headers={"tr_cont": "F"},
    )
    second = FakeBrokerResponse(body={"rt_cd": "0", "output1": []}, status=200, headers={"tr_cont": "D"})
    session = scripted_session([first, second])
    assert len(asyncio.run(client.fetch_positions(session))) == 1  # type: ignore[arg-type]
    assert session.requests[1].headers["tr_cont"] == "N"
    page = FakeBrokerResponse(body={"rt_cd": "0", "output1": [], "ctx_area_fk100": "", "ctx_area_nk100": ""}, status=200, headers={"tr_cont": "M"})
    runaway = scripted_session([page] * 10)
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_positions(runaway))  # type: ignore[arg-type]
    assert len(runaway.requests) == 10


def test_balance_bad_rows_fail_closed(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    for rows in (
        [{"pdno": "005930", "hldg_qty": "", "ord_psbl_qty": "1"}],
        ["not-a-row"],
        [{"pdno": "", "hldg_qty": "1", "ord_psbl_qty": "1"}],
    ):
        session = scripted_session(
            [FakeBrokerResponse(body={"rt_cd": "0", "output1": rows}, status=200, headers={"tr_cont": "D"})]
        )
        with pytest.raises(BrokerReadError):
            asyncio.run(client.fetch_positions(session))  # type: ignore[arg-type]
    session = scripted_session([FakeBrokerResponse(body={"rt_cd": "0"}, status=200)])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_positions(session))  # type: ignore[arg-type]
    session = scripted_session([FakeBrokerResponse(body={"rt_cd": "0", "output1": {}}, status=200)])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_positions(session))  # type: ignore[arg-type]


def test_orderable_cash_ignores_margin_fields(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    body = {
        "rt_cd": "0",
        "output": {"ord_psbl_cash": "9950000", "nrcv_buy_amt": None, "max_buy_amt": "25000000"},
    }
    session = scripted_session([FakeBrokerResponse(body=body, status=200)])
    assert asyncio.run(client.fetch_orderable_cash(session, symbol="005930", limit_price=70000)) == 9_950_000  # type: ignore[arg-type]
    assert session.requests[0].headers["tr_id"] == "V" + TR_SUFFIX_BUYING_POWER
    assert session.requests[0].params["PDNO"] == "005930"


def test_orderable_cash_takes_smaller_cash_figure(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    body = {"rt_cd": "0", "output": {"ord_psbl_cash": "9950000", "nrcv_buy_amt": "9000000"}}
    session = scripted_session([FakeBrokerResponse(body=body, status=200)])
    assert asyncio.run(client.fetch_orderable_cash(session, symbol="005930", limit_price=70000)) == 9_000_000  # type: ignore[arg-type]


def test_orderable_cash_without_parseable_figure_fails(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    for output in ({"ord_psbl_cash": None, "nrcv_buy_amt": None}, {"ord_psbl_cash": "abc", "nrcv_buy_amt": ""}):
        session = scripted_session([FakeBrokerResponse(body={"rt_cd": "0", "output": output}, status=200)])
        with pytest.raises(BrokerReadError):
            asyncio.run(client.fetch_orderable_cash(session, symbol="005930", limit_price=70000))  # type: ignore[arg-type]
    session = scripted_session([FakeBrokerResponse(body={"rt_cd": "0"}, status=200)])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_orderable_cash(session, symbol="005930", limit_price=70000))  # type: ignore[arg-type]


def test_price_limits_validated(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    body = {"rt_cd": "0", "output": {"stck_prpr": "263000", "stck_mxpr": "349000", "stck_llam": "188000"}}
    session = scripted_session([FakeBrokerResponse(body=body, status=200)])
    limits = asyncio.run(client.fetch_price_limits(session, "005930"))  # type: ignore[arg-type]
    assert (limits.symbol, limits.last, limits.upper, limits.lower) == ("005930", 263000, 349000, 188000)


def test_price_limits_reject_untrusted_quotes(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    bad_bodies = [
        {"rt_cd": "1", "msg_cd": "XXXX"},
        {"rt_cd": "0"},
        {"rt_cd": "0", "output": {"stck_prpr": "263000", "stck_mxpr": "0", "stck_llam": "188000"}},
        {"rt_cd": "0", "output": {"stck_prpr": "400000", "stck_mxpr": "349000", "stck_llam": "188000"}},
    ]
    for bad in bad_bodies:
        session = scripted_session([FakeBrokerResponse(body=bad, status=200)])
        with pytest.raises(BrokerReadError):
            asyncio.run(client.fetch_price_limits(session, "005930"))  # type: ignore[arg-type]

    class _Down:
        def get(self, _url: str, **_kw: object) -> FakeBrokerResponse:
            raise ValueError("down")

    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_price_limits(_Down(), "005930"))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["2026-13-40", "20260102", "2026-1-02", "2026-02-30"])
def test_invalid_calendar_dates_never_send(tmp_path, bad) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, bad))
    assert session.requests == []


@pytest.mark.parametrize(("field", "value"), [("pdno", None), ("ord_gno_brno", None), ("cncl_yn", None), ("ord_qty", "-1")])
def test_untrustworthy_order_fields_fail_closed(tmp_path, field, value) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([_daily_page([_buy_row(**{field: value})])])
    with pytest.raises(BrokerReadError):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))


def test_continuation_header_case_and_missing_cursor(tmp_path) -> None:
    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([FakeBrokerResponse(body={"rt_cd": "0", "output1": []}, headers={"Tr_Cont": "M"})])
    with pytest.raises(BrokerReadError, match="context"):
        asyncio.run(client.fetch_daily_orders(session, "2026-01-02"))
    assert len(session.requests) == 1


def test_vendor_message_redacts_binding_secrets(tmp_path) -> None:
    binding = _mock_binding()
    client, _, _ = _trading_client(binding, tmp_path, bearer="TOKEN-SENTINEL")
    body = {"rt_cd": "1", "msg_cd": "40100000", "msg1": " ".join((binding.app_key, binding.app_secret, binding.account_no, "TOKEN-SENTINEL"))}
    result = asyncio.run(client.submit_order(scripted_session([body]), _limit_buy()))
    assert result.msg == "*** *** *** ***"


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_fails_closed(timeout) -> None:
    with pytest.raises(TradingConfigError, match="timeout"):
        KisTradingClient(_mock_binding(), http_timeout_seconds=timeout)


@pytest.mark.parametrize("endpoint", ["https://[broken", "https://openapivts.koreainvestment.com:bad", "https://SECRET99@openapivts.koreainvestment.com:29443"])
def test_invalid_endpoint_is_safe_config_error(endpoint) -> None:
    with pytest.raises(TradingConfigError, match="mock") as error:
        _mock_binding(KIS_MOCK_BASE_URL=endpoint)
    assert "SECRET99" not in str(error.value)


def test_injected_live_client_cannot_send_mock_orders(tmp_path) -> None:
    binding = _mock_binding()
    api = KisApiClient(app_key=binding.app_key, app_secret=binding.app_secret, base_url="https://openapi.koreainvestment.com:9443", token_file=tmp_path / "t.json")
    with pytest.raises(TradingConfigError, match="binding"):
        KisTradingClient(binding, http_timeout_seconds=5, api_client=api)


def test_disconnect_on_response_exit_keeps_known_status(tmp_path) -> None:
    class _DisconnectOnExit(FakeBrokerResponse):
        async def __aexit__(self, *exc):
            raise aiohttp.ServerDisconnectedError()

    client, _, _ = _trading_client(_mock_binding(), tmp_path)
    session = scripted_session([_DisconnectOnExit(body=_accept_body(), status=200)])
    result = asyncio.run(client.submit_order(session, _limit_buy()))
    assert result.verdict is SubmitVerdict.AMBIGUOUS
    assert result.http_status == 200
    assert len(session.requests) == 1
