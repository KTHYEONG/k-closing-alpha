from __future__ import annotations


def test_ls_client_ensure_token() -> None:
    import asyncio
    from unittest.mock import AsyncMock, patch
    from src.api.ls.client import LsApiClient
    client = LsApiClient(app_key="k", app_secret="s")
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"access_token": "mock_tok", "token_type": "Bearer"})
    session = AsyncMock()
    session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.post.return_value.__aexit__ = AsyncMock(return_value=False)
    token = asyncio.run(client.ensure_token(session))
    assert token == "mock_tok"
    assert client.token == "mock_tok"


def test_ls_client_get_minute_chart_single_call() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.api.ls.client import LsApiClient
    client = LsApiClient(app_key="k", app_secret="s")
    client.token = "mock_tok"
    mock_bars = [{"time": "090100", "close": 1000, "jdiff_vol": 50}, {"time": "153000", "close": 1050, "jdiff_vol": 200}]
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"rsp_cd": "00000", "t8412OutBlock1": mock_bars})
    session = AsyncMock()
    session.post.return_value.__aenter__ = AsyncMock(return_value=mock_resp)
    session.post.return_value.__aexit__ = AsyncMock(return_value=False)
    res = asyncio.run(client.get_minute_chart(session, "005930", "2026-09-04"))
    assert res["rt_cd"] == "0"
    assert res["vendor"] == "ls"
    assert len(res["output2"]) == 2
    assert res["output2"][0]["time"] == "090100"


def test_ls_client_get_tick_chart_paginates_with_cts() -> None:
    import asyncio
    from unittest.mock import AsyncMock
    from src.api.ls.client import LsApiClient
    client = LsApiClient(app_key="k", app_secret="s")
    client.token = "mock_tok"
    page1 = {"rsp_cd": "00000", "t8411OutBlock": {"cts_date": "20260904", "cts_time": "151500000"}, "t8411OutBlock1": [{"time": "153000", "close": 1000, "jdiff_vol": 100}]}
    page2 = {"rsp_cd": "00000", "t8411OutBlock": {"cts_date": "", "cts_time": ""}, "t8411OutBlock1": [{"time": "090000", "close": 950, "jdiff_vol": 50}]}
    mock_resp1 = AsyncMock()
    mock_resp1.status = 200
    mock_resp1.json = AsyncMock(return_value=page1)
    mock_resp2 = AsyncMock()
    mock_resp2.status = 200
    mock_resp2.json = AsyncMock(return_value=page2)
    session = AsyncMock()
    session.post.return_value.__aenter__ = AsyncMock(side_effect=[mock_resp1, mock_resp2])
    session.post.return_value.__aexit__ = AsyncMock(return_value=False)
    res = asyncio.run(client.get_tick_chart(session, "005930", "2026-09-04", max_pages=5))
    assert res["rt_cd"] == "0"
    assert res["vendor"] == "ls"
    assert res["truncated"] is False
    assert len(res["output2"]) == 2
    assert res["output2"][0]["time"] == "090000"
    assert res["output2"][1]["time"] == "153000"

def test_ls_get_tick_chart_marks_truncated_when_page_budget_exhausted() -> None:
    import asyncio

    from src.api.ls.client import LsApiClient

    client = LsApiClient(app_key="k", app_secret="s")
    client.token = "t"
    client._min_interval = 0.0

    calls = {"n": 0}

    async def fake_post_tr(session, tr_cd, tr_key, body, tr_cont="N", tr_cont_key="", max_retries=3):
        calls["n"] += 1
        # 항상 09:00 이전에 도달하지 못하는 응답 -- 페이지 예산 소진을 강제한다
        rows = [{"date": "20260904", "time": "143000", "close": 100, "jdiff_vol": 1}]
        return ({"rsp_cd": "00000", "t8411OutBlock1": rows, "t8411OutBlock": {"cts_date": "20260904", "cts_time": "142959"}}, {"tr_cont": "Y", "tr_cont_key": "x"})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=3))

    assert res["rt_cd"] == "0"
    assert res["truncated"] is True
    assert calls["n"] == 3
    assert res["vendor"] == "ls"


def test_ls_post_tr_releases_lock_before_network_roundtrip() -> None:
    import asyncio

    from src.api.ls.client import LsApiClient

    client = LsApiClient(app_key="k", app_secret="s")
    client.token = "t"
    client._min_interval = 0.01

    state = {"in_flight": 0, "max_in_flight": 0}

    class _Resp:
        headers: dict = {}

        async def json(self):
            return {"rsp_cd": "00000"}

        async def __aenter__(self):
            state["in_flight"] += 1
            state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
            await asyncio.sleep(0.05)
            return self

        async def __aexit__(self, *a):
            state["in_flight"] -= 1
            return False

    class _Session:
        def post(self, *a, **kw):
            return _Resp()

    async def _run():
        session = _Session()
        return await asyncio.gather(
            client._post_tr(session, "t8412", "005930", {}),
            client._post_tr(session, "t8412", "000660", {}),
        )

    asyncio.run(_run())

    assert state["max_in_flight"] == 2


def test_ls_client_returns_vendor_native_rows_without_kis_aliases() -> None:
    import asyncio

    from src.api.ls.client import LsApiClient

    client = LsApiClient(app_key="k", app_secret="s")
    client.token = "t"
    client._min_interval = 0.0

    async def fake_post_tr(session, tr_cd, tr_key, body, tr_cont="N", tr_cont_key="", max_retries=3):
        rows = [{"date": "20260904", "time": "090300", "open": 9100, "high": 9100, "low": 9100, "close": 9100, "jdiff_vol": 55226, "value": 498}]
        return ({"rsp_cd": "00000", "t8412OutBlock1": rows}, {})

    client._post_tr = fake_post_tr

    res = asyncio.run(client.get_minute_chart(object(), "009900", "2026-09-04"))

    assert res["rt_cd"] == "0"
    assert res["vendor"] == "ls"
    row = res["output2"][0]
    for kis_alias in ("acml_tr_pbmn", "acml_vol", "stck_prpr", "stck_oprc", "stck_hgpr", "stck_lwpr", "cntg_vol", "cnqn", "stck_cntg_hour"):
        assert kis_alias not in row
    assert row["value"] == 498

