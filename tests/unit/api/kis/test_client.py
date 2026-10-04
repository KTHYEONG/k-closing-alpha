"""src.api.kis.client 모듈 직접 참조 테스트 (호환 파사드 src.api.kis_client 우회).

FID_FAKE_TICK_INCU_YN 누락으로 FHKST03010230 전체 호출이 실패했던 회귀
방지, 그리고 market_div_code 명시 요구(fail-closed) 계약 검증.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.api.kis.client import KisApiClient


class _FakeSession:
    def get(self, *args, **kwargs):
        raise AssertionError("직접 patch된 _handle_request만 사용되어야 한다")


def test_get_historical_minute_chart_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_historical_minute_chart(_FakeSession(), "005930", "20260901")

    asyncio.run(_runner())


def test_get_intraday_minute_chart_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_intraday_minute_chart(_FakeSession(), "005930")

    asyncio.run(_runner())


def test_get_historical_minute_chart_includes_fake_tick_field() -> None:
    """FID_FAKE_TICK_INCU_YN 필드 키 누락 시 KIS가 OPSQ2001로 전체 거부하던 회귀 방지."""
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    handle_request = AsyncMock(return_value={"rt_cd": "0", "output2": []})

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_historical_minute_chart(
                _FakeSession(), "005930", "20260901", market_div_code="J",
            )

    asyncio.run(_runner())

    params = handle_request.await_args.kwargs.get("params", {})
    assert "FID_FAKE_TICK_INCU_YN" in params


def test_get_orderbook_snapshot_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_orderbook_snapshot(_FakeSession(), "005930")

    asyncio.run(_runner())


def test_get_orderbook_snapshot_returns_raw_output() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    handle_request = AsyncMock(return_value={
        "rt_cd": "0",
        "output1": {"askp1": "70100", "bidp1": "70000", "total_askp_rsqn": "1200", "total_bidp_rsqn": "1500"},
        "output2": {},
    })

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_orderbook_snapshot(_FakeSession(), "005930", market_div_code="J")

    res = asyncio.run(_runner())
    assert res["output1"]["askp1"] == "70100"
    params = handle_request.await_args.kwargs.get("params", {})
    assert params.get("FID_COND_MRKT_DIV_CODE") == "J"
    assert params.get("FID_INPUT_ISCD") == "005930"


def test_get_intraday_trade_ticks_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_intraday_trade_ticks(_FakeSession(), "005930")

    asyncio.run(_runner())


def test_get_intraday_trade_ticks_dedupes_by_acml_vol_not_hour() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    page1 = {"rt_cd": "0", "output2": [
        {"stck_cntg_hour": "093000", "acml_vol": "1000", "stck_prpr": "70000"},
        {"stck_cntg_hour": "093000", "acml_vol": "990", "stck_prpr": "69900"},
    ]}
    page2 = {"rt_cd": "0", "output2": [
        {"stck_cntg_hour": "093000", "acml_vol": "990", "stck_prpr": "69900"},
        {"stck_cntg_hour": "090000", "acml_vol": "100", "stck_prpr": "69000"},
    ]}
    handle_request = AsyncMock(side_effect=[page1, page2])

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_intraday_trade_ticks(
                _FakeSession(), "005930", floor_hour="090000", end_hour="153000", market_div_code="J",
            )

    res = asyncio.run(_runner())
    acml_vols = {row["acml_vol"] for row in res["output2"]}
    assert len(res["output2"]) == 3
    assert acml_vols == {"1000", "990", "100"}


def test_get_intraday_trade_ticks_floor_reached_proof() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _fetch(pages, max_pages=10, floor="090000", end="153000"):
        handle = AsyncMock(side_effect=pages)
        with patch.object(client, "_handle_request", handle):
            return await client.get_intraday_trade_ticks(
                _FakeSession(), "005930", floor_hour=floor, end_hour=end, market_div_code="J",
                max_pages=max_pages,
            )

    reached = asyncio.run(_fetch([
        {"rt_cd": "0", "output2": [{"stck_cntg_hour": "130000", "acml_vol": "500"}]},
        {"rt_cd": "0", "output2": [{"stck_cntg_hour": "090000", "acml_vol": "100"}]},
    ]))
    assert reached["floor_reached"] is True

    stalled_at_floor = asyncio.run(_fetch(
        [{"rt_cd": "0", "output2": [{"stck_cntg_hour": "090000", "stck_prpr": "9500"}]}],
        floor="090000", end="090000",
    ))
    assert stalled_at_floor["floor_reached"] is True

    empty_end = asyncio.run(_fetch([
        {"rt_cd": "0", "output2": [{"stck_cntg_hour": "130000", "acml_vol": "500"}]},
        {"rt_cd": "0", "output2": []},
    ]))
    assert empty_end["floor_reached"] is False

    capped = asyncio.run(_fetch(
        [{"rt_cd": "0", "output2": [{"stck_cntg_hour": "130000", "acml_vol": "500"}]}],
        max_pages=1,
    ))
    assert capped["floor_reached"] is False


def test_get_daily_short_sale_history_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_daily_short_sale_history(_FakeSession(), "005930", "20240101", "20240110")

    asyncio.run(_runner())


def test_get_daily_short_sale_history_returns_raw_output_with_date_range_params() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    handle_request = AsyncMock(return_value={"rt_cd": "0", "output2": [
        {"stck_bsop_date": "20240103", "ssts_cntg_qty": "100"},
        {"stck_bsop_date": "20240102", "ssts_cntg_qty": "90"},
    ]})

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_daily_short_sale_history(
                _FakeSession(), "005930", "20240101", "20240110", market_div_code="J",
            )

    res = asyncio.run(_runner())
    dates = [row["stck_bsop_date"] for row in res["output2"]]
    assert dates == ["20240102", "20240103"]
    params = handle_request.await_args.kwargs.get("params", {})
    assert params.get("FID_INPUT_DATE_1") == "20240101"


def test_get_daily_credit_balance_history_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_daily_credit_balance_history(_FakeSession(), "005930", "20240101", "20240110")

    asyncio.run(_runner())


def test_get_daily_credit_balance_history_includes_fixed_screen_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    handle_request = AsyncMock(return_value={"rt_cd": "0", "output": []})

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_daily_credit_balance_history(
                _FakeSession(), "005930", "20240101", "20240110", market_div_code="J",
            )

    asyncio.run(_runner())
    params = handle_request.await_args.kwargs.get("params", {})
    assert params.get("FID_COND_SCR_DIV_CODE") == "20476"


def test_get_daily_credit_balance_history_collects_rows_in_range() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    page = {
        "rt_cd": "0",
        "output": [
            {"deal_date": "20240110", "whol_loan_rdmp_stcn": "50"},
            {"deal_date": "20240102", "whol_loan_rdmp_stcn": "40"},
            {"deal_date": "20231231", "whol_loan_rdmp_stcn": "30"},
        ],
    }
    handle_request = AsyncMock(return_value=page)

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_daily_credit_balance_history(
                _FakeSession(), "005930", "20240101", "20240110", market_div_code="J",
            )

    result = asyncio.run(_runner())

    assert result["rt_cd"] == "0"
    assert [row["deal_date"] for row in result["output"]] == ["20240102", "20240110"]
    assert handle_request.await_count == 1


def test_get_program_trade_daily_history_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_program_trade_daily_history(_FakeSession(), "005930", "20240101", "20240110")

    asyncio.run(_runner())


def test_get_program_trade_daily_history_paginates_backward_by_earliest_date() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    page1 = {"rt_cd": "0", "output": [{"stck_bsop_date": "20240105"}, {"stck_bsop_date": "20240104"}]}
    handle_request = AsyncMock(return_value=page1)

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_program_trade_daily_history(
                _FakeSession(), "005930", "20240101", "20240110", market_div_code="J",
            )

    res = asyncio.run(_runner())
    dates = [row["stck_bsop_date"] for row in res["output"]]
    assert dates == ["20240104", "20240105"]


def test_get_fluctuation_ranking_requires_explicit_market_div_code() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")

    async def _runner() -> None:
        with pytest.raises(ValueError, match="market_div_code"):
            await client.get_fluctuation_ranking(_FakeSession(), rate_min_pct=2.0, rate_max_pct=10.0)

    asyncio.run(_runner())


def test_get_fluctuation_ranking_sends_confirmed_tr_params() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    handle_request = AsyncMock(return_value={"rt_cd": "0", "output": []})

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_fluctuation_ranking(
                _FakeSession(), rate_min_pct=2.0, rate_max_pct=10.0, market_div_code="J",
            )

    asyncio.run(_runner())

    call = handle_request.await_args
    assert call.args[1] == f"{client.base_url}/uapi/domestic-stock/v1/ranking/fluctuation"
    params = call.kwargs.get("params", {})
    assert params["FID_COND_MRKT_DIV_CODE"] == "J"
    assert params["FID_COND_SCR_DIV_CODE"] == "20170"
    assert params["FID_RSFL_RATE1"] == "2"
    assert params["FID_RSFL_RATE2"] == "10"
    headers = call.kwargs.get("headers", {})
    assert headers.get("tr_id") == "FHPST01700000"


def test_get_fluctuation_ranking_returns_raw_rows_with_output_or_output2_fallback() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    rows = [{"stck_shrn_iscd": "005930", "prdy_ctrt": "5.2"}]

    async def _runner_output():
        handle_request = AsyncMock(return_value={"rt_cd": "0", "output": rows})
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_fluctuation_ranking(_FakeSession(), rate_min_pct=2.0, rate_max_pct=10.0, market_div_code="J")

    async def _runner_output2():
        handle_request = AsyncMock(return_value={"rt_cd": "0", "output2": rows})
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_fluctuation_ranking(_FakeSession(), rate_min_pct=2.0, rate_max_pct=10.0, market_div_code="J")

    res1 = asyncio.run(_runner_output())
    res2 = asyncio.run(_runner_output2())
    assert res1["output"] == rows
    assert res2["output"] == rows


def test_get_fluctuation_ranking_propagates_error_response_unchanged() -> None:
    client = KisApiClient(app_key="k", app_secret="s", account_id="a", hts_id="h")
    handle_request = AsyncMock(return_value={"rt_cd": "1", "msg1": "조회 실패"})

    async def _runner():
        with patch.object(client, "_handle_request", handle_request):
            return await client.get_fluctuation_ranking(_FakeSession(), rate_min_pct=2.0, rate_max_pct=10.0, market_div_code="J")

    res = asyncio.run(_runner())
    assert res == {"rt_cd": "1", "msg1": "조회 실패"}


def test_write_token_file_publishes_exact_cache_bytes(tmp_path) -> None:
    import json
    import stat

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "kis_token.json"
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    client._write_token_file("TOK", "2030-01-01T00:00:00+09:00", "2026-10-01T00:00:00+09:00")

    assert token_file.read_text(encoding="utf-8") == json.dumps({
        "access_token": "TOK",
        "expired_at": "2030-01-01T00:00:00+09:00",
        "app_key": "k",
        "issued_at": "2026-10-01T00:00:00+09:00",
    })
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert list(tmp_path.glob("*.tmp")) == []


def test_write_token_file_failure_leaves_no_temp(tmp_path, monkeypatch) -> None:
    import os

    import pytest

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "kis_token.json"
    token_file.write_text("old", encoding="utf-8")
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    def _boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", _boom)

    with pytest.raises(OSError, match="disk gone"):
        client._write_token_file("TOK", "2030-01-01T00:00:00+09:00", "2026-10-01T00:00:00+09:00")

    assert token_file.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.glob("*.tmp")) == []


def test_host_token_lock_acquires_and_releases(tmp_path) -> None:
    import asyncio
    import os

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "kis_token.json"
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    async def _run() -> None:
        async with client._host_token_lock():
            assert os.path.isfile(str(token_file) + ".lock")  # noqa: ASYNC240 - existence probe of the just-created lock file

    asyncio.run(_run())


def test_host_token_lock_times_out_when_held(tmp_path, caplog) -> None:
    import asyncio
    import fcntl
    import logging
    import os
    from time import monotonic

    import pytest

    from src.api.kis.client import KisApiClient
    from src.api.shared_token import TokenStoreLockTimeout

    token_file = tmp_path / "kis_token.json"
    lock_path = str(token_file) + ".lock"
    holder = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        import fcntl as _fcntl

        _fcntl.flock(holder, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

        async def _run() -> None:
            async with client._host_token_lock(deadline=monotonic() + 0.2):
                pass

        with caplog.at_level(logging.ERROR), pytest.raises(TokenStoreLockTimeout, match="token lock not acquired"):
            asyncio.run(_run())
    finally:
        fcntl.flock(holder, fcntl.LOCK_UN)
        os.close(holder)

    assert any("stage=kis_token_lock status=TIMEOUT" in rec.message for rec in caplog.records)


def test_host_token_lock_reraises_unexpected_flock_error(tmp_path, monkeypatch) -> None:
    import asyncio
    import errno
    import fcntl

    import pytest

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "kis_token.json"
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))

    def _boom(fd, op):
        raise OSError(errno.EACCES, "nope")

    monkeypatch.setattr(fcntl, "flock", _boom)

    async def _run() -> None:
        async with client._host_token_lock():
            pass

    with pytest.raises(OSError, match="nope"):
        asyncio.run(_run())


def test_ensure_token_adopts_rotated_cache_without_issuance(tmp_path, caplog) -> None:
    import asyncio
    import logging
    from datetime import UTC, datetime, timedelta

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "kis_token.json"
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    now = datetime.now(UTC)
    client._write_token_file(
        "NEW",
        (now + timedelta(days=1)).isoformat(),
        (now - timedelta(hours=1)).isoformat(),
    )

    async def _run() -> str:
        return await client.ensure_token(None, rejected_token="OLD")

    with caplog.at_level(logging.INFO):
        assert asyncio.run(_run()) == "NEW"

    assert client.token == "NEW"
    assert any("status=ADOPTED_ROTATED" in rec.message for rec in caplog.records)


def test_handle_request_rotates_token_on_auth_rejection(tmp_path, monkeypatch, caplog) -> None:
    import asyncio
    import logging

    from src.api.kis.client import KisApiClient

    token_file = tmp_path / "kis_token.json"
    client = KisApiClient(app_key="k", app_secret="s", token_file=str(token_file))
    client.token = "OLD"

    class _Resp:
        def __init__(self, payload):
            self._payload = payload
            self.status = 200

        async def json(self):
            return self._payload

    class _Ctx:
        def __init__(self, resp):
            self._resp = resp

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *_a):
            return False

    calls = {"n": 0}

    class _FakeSession:
        def get(self, _url, **_kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _Ctx(_Resp({"rt_cd": "1", "msg_cd": "EGW00121", "msg1": "token expired"}))
            return _Ctx(_Resp({"rt_cd": "0", "msg_cd": "MCA00000", "output": []}))

        def post(self, _url, **_kw):
            return _Ctx(_Resp({"access_token": "NEW", "expires_in": 86400}))

    class _Limiter:
        async def acquire(self):
            return None

    monkeypatch.setattr(client, "rate_limiter", _Limiter())
    monkeypatch.setattr("src.api.kis.client.asyncio.sleep", _no_sleep)

    async def _run():
        session = _FakeSession()
        return await client._handle_request(session.get, "https://x", headers={"tr_id": "T", "authorization": "Bearer OLD"})

    with caplog.at_level(logging.WARNING):
        out = asyncio.run(_run())

    assert out["rt_cd"] == "0"
    assert client.token == "NEW"
    assert any("status=AUTH_REJECTED" in rec.message for rec in caplog.records)


async def _no_sleep(_delay):
    return None


def _issuing_session(token: str):
    class _Resp:
        def __init__(self, payload):
            self._payload = payload
            self.status = 200

        async def json(self):
            return self._payload

    class _Ctx:
        def __init__(self, resp):
            self._resp = resp

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def __init__(self):
            self.posts = 0

        def post(self, _url, **_kw):
            self.posts += 1
            return _Ctx(_Resp({"access_token": token, "expires_in": 86400}))

    return _Session()


def test_ensure_token_issues_fresh_then_reuses_cache(tmp_path) -> None:
    import asyncio

    from src.api.kis.client import KisApiClient

    client = KisApiClient(app_key="k", app_secret="s", token_file=str(tmp_path / "kis_token.json"))
    session = _issuing_session("TOK")

    assert asyncio.run(client.ensure_token(session)) == "TOK"
    assert session.posts == 1
    assert asyncio.run(client.ensure_token(session)) == "TOK"
    assert session.posts == 1


def test_ensure_token_force_refresh_reissues(tmp_path) -> None:
    import asyncio

    from src.api.kis.client import KisApiClient

    client = KisApiClient(app_key="k", app_secret="s", token_file=str(tmp_path / "kis_token.json"))
    client.token = "OLD"
    session = _issuing_session("NEW")

    assert asyncio.run(client.ensure_token(session, force_refresh=True)) == "NEW"
    assert session.posts == 1


def test_ensure_token_adopts_sibling_issued_cache(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.kis.client import KisApiClient

    client = KisApiClient(app_key="k", app_secret="s", token_file=str(tmp_path / "kis_token.json"))
    session = _issuing_session("NEW")
    reads = iter([None, "SIBLING"])
    monkeypatch.setattr(client, "_read_cached_token", lambda *a, **k: next(reads))

    assert asyncio.run(client.ensure_token(session)) == "SIBLING"
    assert session.posts == 0


def test_issue_daily_token_issues_when_cache_absent(tmp_path) -> None:
    import asyncio

    from src.api.kis.client import KisApiClient

    client = KisApiClient(app_key="k", app_secret="s", token_file=str(tmp_path / "kis_token.json"))

    assert asyncio.run(client.issue_daily_token(_issuing_session("TOK"))) is True
    assert client.token == "TOK"
