from __future__ import annotations


def test_kiwoom_client_ensure_token() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"token": "mock_tok", "return_code": 0})
    session = AsyncMock()
    session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.post.return_value.__aexit__ = AsyncMock(return_value=False)

    token = asyncio.run(client.ensure_token(session))

    assert token == "mock_tok"
    assert client.token == "mock_tok"


def test_kiwoom_client_ensure_token_concurrent_calls_lock_and_request_once() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    call_count = {"n": 0}

    async def fake_post(*args, **kwargs):
        call_count["n"] += 1
        await asyncio.sleep(0.01)
        mock_resp = AsyncMock()
        mock_resp.json = AsyncMock(return_value={"token": "tok_123", "return_code": 0})
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    session = AsyncMock()
    session.post = fake_post

    async def runner():
        tokens = await asyncio.gather(*[client.ensure_token(session) for _ in range(10)])
        assert all(t == "tok_123" for t in tokens)

    asyncio.run(runner())
    assert call_count["n"] == 1


def test_kiwoom_client_ensure_token_raises_on_empty_token() -> None:
    import asyncio
    from unittest.mock import AsyncMock

    import pytest

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    mock_resp = AsyncMock()
    mock_resp.json = AsyncMock(return_value={"return_code": 3, "return_msg": "invalid appkey"})
    session = AsyncMock()
    session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.post.return_value.__aexit__ = AsyncMock(return_value=False)

    with pytest.raises(RuntimeError, match="Kiwoom token issuance failed"):
        asyncio.run(client.ensure_token(session))


def test_kiwoom_client_get_tick_chart_single_page() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        rows = [
            {"cur_prc": "270000", "trde_qty": "100", "cntr_tm": "20260904153000"},
            {"cur_prc": "269000", "trde_qty": "50", "cntr_tm": "20260904090000"},
        ]
        return ({"return_code": 0, "return_msg": "OK", "stk_tic_chart_qry": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))

    assert res["rt_cd"] == "0"
    assert res["vendor"] == "kiwoom"
    assert res["truncated"] is False
    assert len(res["output2"]) == 2


def test_kiwoom_client_get_tick_chart_paginates_via_cont_yn() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"
    calls = {"n": 0}

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        calls["n"] += 1
        if calls["n"] == 1:
            rows = [{"cur_prc": "270000", "trde_qty": "100", "cntr_tm": "20260904153000"}]
            return ({"return_code": 0, "stk_tic_chart_qry": rows}, {"cont-yn": "Y", "next-key": "abc"})
        rows = [{"cur_prc": "269000", "trde_qty": "50", "cntr_tm": "20260904090000"}]
        return ({"return_code": 0, "stk_tic_chart_qry": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5))

    assert res["rt_cd"] == "0"
    assert calls["n"] == 2
    assert len(res["output2"]) == 2
    assert res["truncated"] is False


def test_kiwoom_get_tick_chart_marks_truncated_when_page_budget_exhausted() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"
    calls = {"n": 0}

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        calls["n"] += 1
        rows = [{"cur_prc": "270000", "trde_qty": "100", "cntr_tm": "20260904143000"}]
        return ({"return_code": 0, "stk_tic_chart_qry": rows}, {"cont-yn": "Y", "next-key": "x"})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=2))

    assert res["rt_cd"] == "0"
    assert res["truncated"] is True
    assert calls["n"] == 2


def test_kiwoom_client_get_tick_chart_filters_rows_outside_target_date() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        rows = [
            {"cur_prc": "270000", "trde_qty": "100", "cntr_tm": "20260904153500"},
            {"cur_prc": "269500", "trde_qty": "80", "cntr_tm": "20260902130600"},
        ]
        return ({"return_code": 0, "stk_tic_chart_qry": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))

    assert res["rt_cd"] == "0"
    assert len(res["output2"]) == 1
    assert res["output2"][0]["cntr_tm"] == "20260904153500"


def test_kiwoom_client_get_tick_chart_returns_soft_failure_on_nonzero_return_code() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        return ({"return_code": 3, "return_msg": "invalid stk_cd"}, {})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_tick_chart(object(), "999999", "2026-09-04"))

    assert res["rt_cd"] == "1"
    assert res["output2"] == []
    assert res["vendor"] == "kiwoom"


def test_kiwoom_client_get_tick_chart_exception_yields_soft_failure() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        raise RuntimeError("network down")

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))

    assert res["rt_cd"] == "1"
    assert res["output2"] == []


def test_kiwoom_post_tr_retries_on_429_then_succeeds(monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from src.api.kis.rate_limit import AsyncRateLimiter
    import src.api.kiwoom.client as kiwoom_client_mod

    client = kiwoom_client_mod.KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def _fast_acquire(self) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr(AsyncRateLimiter, "acquire", _fast_acquire)

    _real_sleep = asyncio.sleep

    async def _fast_sleep(seconds) -> None:
        await _real_sleep(0)

    monkeypatch.setattr(kiwoom_client_mod.asyncio, "sleep", _fast_sleep)

    mock_resp_429 = AsyncMock()
    mock_resp_429.status = 429
    mock_resp_429.json = AsyncMock(return_value={"return_code": 5, "return_msg": "허용된 API 요청 개수를 초과하였습니다. 유량=5"})
    mock_resp_429.headers = {}
    mock_resp_200 = AsyncMock()
    mock_resp_200.status = 200
    mock_resp_200.json = AsyncMock(return_value={"return_code": 0, "return_msg": "OK", "stk_tic_chart_qry": []})
    mock_resp_200.headers = {}

    session = AsyncMock()
    session.post.return_value.__aenter__ = AsyncMock(side_effect=[mock_resp_429, mock_resp_200])
    session.post.return_value.__aexit__ = AsyncMock(return_value=False)

    data, headers = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {"stk_cd": "005930"}))

    assert data.get("return_code") == 0
    assert session.post.call_count == 2


def test_kiwoom_get_fluctuation_ranking_single_page() -> None:
    import asyncio
    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        assert api_id == "ka10027"
        assert path == "/api/dostk/rkinfo"
        rows = [
            {"stk_cd": "004490_AL", "stk_nm": "세방전지", "cur_prc": "+69700", "pred_pre": "+5400", "flu_rt": "+8.40", "now_trde_qty": "143616"},
            {"stk_cd": "005930_AL", "stk_nm": "삼성전자", "cur_prc": "+80000", "pred_pre": "+1000", "flu_rt": "+1.25", "now_trde_qty": "500000"},
        ]
        return ({"return_code": 0, "return_msg": "OK", "pred_pre_flu_rt_upper": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr
    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=2.0, rate_max_pct=10.0))
    assert res["rt_cd"] == "0"
    assert res["vendor"] == "kiwoom"
    assert len(res["output"]) == 1
    assert res["output"][0]["stk_cd"] == "004490_AL"


def test_kiwoom_get_fluctuation_ranking_paginates_via_cont_yn() -> None:
    import asyncio
    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"
    calls = {"n": 0}

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        calls["n"] += 1
        if calls["n"] == 1:
            rows = [{"stk_cd": "004490_AL", "stk_nm": "세방전지", "cur_prc": "+69700", "pred_pre": "+5400", "flu_rt": "+8.40", "now_trde_qty": "143616"}]
            return ({"return_code": 0, "pred_pre_flu_rt_upper": rows}, {"cont-yn": "Y", "next-key": "key2"})
        rows = [{"stk_cd": "249420_AL", "stk_nm": "일동제약", "cur_prc": "+20000", "pred_pre": "+1000", "flu_rt": "+5.26", "now_trde_qty": "100000"}]
        return ({"return_code": 0, "pred_pre_flu_rt_upper": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr
    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=2.0, rate_max_pct=10.0, max_pages=5))
    assert res["rt_cd"] == "0"
    assert calls["n"] == 2
    assert len(res["output"]) == 2


def test_kiwoom_get_fluctuation_ranking_early_exit_on_rate_floor() -> None:
    import asyncio
    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"
    calls = {"n": 0}

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        calls["n"] += 1
        rows = [
            {"stk_cd": "004490_AL", "stk_nm": "세방전지", "cur_prc": "+69700", "pred_pre": "+5400", "flu_rt": "+8.40", "now_trde_qty": "143616"},
            {"stk_cd": "005930_AL", "stk_nm": "삼성전자", "cur_prc": "+80000", "pred_pre": "+100", "flu_rt": "+0.50", "now_trde_qty": "500000"},
        ]
        return ({"return_code": 0, "pred_pre_flu_rt_upper": rows}, {"cont-yn": "Y", "next-key": "more"})

    client._post_tr = fake_post_tr
    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=2.0, rate_max_pct=10.0, max_pages=5))
    assert res["rt_cd"] == "0"
    assert calls["n"] == 1
    assert len(res["output"]) == 1
    assert res["output"][0]["stk_cd"] == "004490_AL"


def test_kiwoom_get_nxt_minute_chart() -> None:
    import asyncio
    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        assert api_id == "ka10080"
        assert body["stk_cd"] == "005930_NX"
        rows = [
            {"cur_prc": "+257000", "trde_qty": "73271", "cntr_tm": "20260904195900", "open_pric": "+256500", "high_pric": "+257000", "low_pric": "+256500"},
            {"cur_prc": "+255000", "trde_qty": "1000", "cntr_tm": "20260904153000", "open_pric": "+255000", "high_pric": "+255000", "low_pric": "+255000"},
            {"cur_prc": "+250000", "trde_qty": "500", "cntr_tm": "20260903195900", "open_pric": "+250000", "high_pric": "+250000", "low_pric": "+250000"},
        ]
        return ({"return_code": 0, "stk_min_pole_chart_qry": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr
    res = asyncio.run(client.get_nxt_minute_chart(object(), "005930", "2026-09-04"))
    assert res["rt_cd"] == "0"
    assert res["vendor"] == "kiwoom"
    assert len(res["output2"]) == 1
    assert res["output2"][0]["cntr_tm"] == "20260904195900"


def test_kiwoom_get_fluctuation_ranking_returns_soft_failure_on_nonzero_return_code() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        return ({"return_code": 3, "return_msg": "invalid param"}, {})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=2.0, rate_max_pct=10.0))

    assert res["rt_cd"] == "1"
    assert res["output"] == []
    assert res["vendor"] == "kiwoom"


def test_kiwoom_get_fluctuation_ranking_skips_unparseable_flu_rt() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        rows = [
            {"stk_cd": "000000_AL", "stk_nm": "불량", "cur_prc": "+1000", "pred_pre": "+10", "flu_rt": "-", "now_trde_qty": "10"},
            {"stk_cd": "004490_AL", "stk_nm": "세방전지", "cur_prc": "+69700", "pred_pre": "+5400", "flu_rt": "+8.40", "now_trde_qty": "143616"},
        ]
        return ({"return_code": 0, "pred_pre_flu_rt_upper": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=2.0, rate_max_pct=10.0))

    assert res["rt_cd"] == "0"
    assert len(res["output"]) == 1
    assert res["output"][0]["stk_cd"] == "004490_AL"


def test_kiwoom_get_fluctuation_ranking_exception_yields_soft_failure() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        raise RuntimeError("network down")

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=2.0, rate_max_pct=10.0))

    assert res["rt_cd"] == "1"
    assert res["output"] == []
    assert res["vendor"] == "kiwoom"


def test_kiwoom_get_nxt_minute_chart_returns_soft_failure_on_nonzero_return_code() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        return ({"return_code": 3, "return_msg": "invalid stk_cd"}, {})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_nxt_minute_chart(object(), "005930", "2026-09-04"))

    assert res["rt_cd"] == "1"
    assert res["output2"] == []
    assert res["vendor"] == "kiwoom"


def test_kiwoom_get_nxt_minute_chart_exception_yields_soft_failure() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        raise RuntimeError("network down")

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_nxt_minute_chart(object(), "005930", "2026-09-04"))

    assert res["rt_cd"] == "1"
    assert res["output2"] == []
    assert res["vendor"] == "kiwoom"


def test_kiwoom_get_nxt_minute_chart_skips_malformed_cntr_tm() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        rows = [
            {"cur_prc": "+257000", "trde_qty": "100", "cntr_tm": "20260904195900", "open_pric": "+256500", "high_pric": "+257000", "low_pric": "+256500"},
            {"cur_prc": "+257000", "trde_qty": "100", "cntr_tm": "20260904", "open_pric": "+256500", "high_pric": "+257000", "low_pric": "+256500"},
        ]
        return ({"return_code": 0, "stk_min_pole_chart_qry": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_nxt_minute_chart(object(), "005930", "2026-09-04"))

    assert res["rt_cd"] == "0"
    assert len(res["output2"]) == 1
    assert res["output2"][0]["cntr_tm"] == "20260904195900"



def test_kiwoom_get_nxt_premarket_chart() -> None:
    import asyncio
    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        assert api_id == "ka10080"
        assert body["stk_cd"] == "005930_NX"
        assert body["tic_scope"] == "1"
        assert body["upd_stkpc_tp"] == "0"
        rows = [
            {"cur_prc": "+251500", "trde_qty": "145344", "cntr_tm": "20260904080000", "open_pric": "+251000", "high_pric": "+253500", "low_pric": "+251000"},
            {"cur_prc": "+252500", "trde_qty": "16736", "cntr_tm": "20260904084900", "open_pric": "+253000", "high_pric": "+253000", "low_pric": "+252500"},
            {"cur_prc": "+257000", "trde_qty": "73271", "cntr_tm": "20260904195900", "open_pric": "+256500", "high_pric": "+257000", "low_pric": "+256500"},
        ]
        return ({"return_code": 0, "stk_min_pole_chart_qry": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr
    res = asyncio.run(client.get_nxt_premarket_chart(object(), "005930", "2026-09-04"))
    assert res["rt_cd"] == "0"
    assert res["vendor"] == "kiwoom"
    assert len(res["output2"]) == 2
    assert res["output2"][0]["cntr_tm"] == "20260904080000"
    assert res["output2"][1]["cntr_tm"] == "20260904084900"


def test_kiwoom_get_nxt_premarket_chart_returns_soft_failure_on_nonzero_return_code() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        return ({"return_code": 3, "return_msg": "invalid stk_cd"}, {})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_nxt_premarket_chart(object(), "005930", "2026-09-04"))

    assert res["rt_cd"] == "1"
    assert res["output2"] == []
    assert res["vendor"] == "kiwoom"


def test_kiwoom_get_nxt_premarket_chart_exception_yields_soft_failure() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        raise RuntimeError("network down")

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_nxt_premarket_chart(object(), "005930", "2026-09-04"))

    assert res["rt_cd"] == "1"
    assert res["output2"] == []
    assert res["vendor"] == "kiwoom"


def test_kiwoom_get_nxt_premarket_chart_skips_malformed_cntr_tm() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        rows = [
            {"cur_prc": "+251500", "trde_qty": "100", "cntr_tm": "20260904080000", "open_pric": "+251000", "high_pric": "+253500", "low_pric": "+251000"},
            {"cur_prc": "+250000", "trde_qty": "100", "cntr_tm": "20260903080000", "open_pric": "+250000", "high_pric": "+250000", "low_pric": "+250000"},
            {"cur_prc": "+251500", "trde_qty": "100", "cntr_tm": "20260904", "open_pric": "+251000", "high_pric": "+253500", "low_pric": "+251000"},
        ]
        return ({"return_code": 0, "stk_min_pole_chart_qry": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_nxt_premarket_chart(object(), "005930", "2026-09-04"))

    assert res["rt_cd"] == "0"
    assert len(res["output2"]) == 1
    assert res["output2"][0]["cntr_tm"] == "20260904080000"
