from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from src.api.ls.client import LsApiClient
from src.data.capture_contracts import ChartBudget

_SEOUL = ZoneInfo("Asia/Seoul")
_YMD = "20260904"


def _ls_client() -> LsApiClient:
    client = LsApiClient(app_key="k", app_secret="s")
    client.token = "t"
    client._min_interval = 0.0
    return client


def _budget(**overrides: Any) -> ChartBudget:
    base: dict[str, Any] = {"max_pages": 5, "deadline": None, "request_timeout_seconds": 5.0}
    base.update(overrides)
    return ChartBudget(**base)


def _paging_client(pages: list[tuple[dict, dict]]) -> tuple[LsApiClient, dict[str, Any]]:
    """Fake _post_tr serving scripted pages; any extra call raises."""
    client = _ls_client()
    state: dict[str, Any] = {"calls": 0, "bodies": []}

    async def fake_post_tr(session: Any, tr_cd: str, tr_key: str, body: dict, tr_cont: str = "N", tr_cont_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        assert state["calls"] < len(pages), "unnecessary extra page requested"
        state["calls"] += 1
        state["bodies"].append(body)
        return pages[state["calls"] - 1]

    client._post_tr = fake_post_tr  # type: ignore[method-assign]
    return client, state


def _minute_page(rows: list[dict], cts_date: str, cts_time: str, tr_cont: str = "N", tr_cont_key: str = "") -> tuple[dict, dict]:
    return (
        {"rsp_cd": "00000", "t8412OutBlock": {"cts_date": cts_date, "cts_time": cts_time}, "t8412OutBlock1": rows},
        {"tr_cont": tr_cont, "tr_cont_key": tr_cont_key},
    )


def _tick_page(rows: list[dict], cts_date: str, cts_time: str, tr_cont: str = "N", tr_cont_key: str = "") -> tuple[dict, dict]:
    return (
        {"rsp_cd": "00000", "t8411OutBlock": {"cts_date": cts_date, "cts_time": cts_time}, "t8411OutBlock1": rows},
        {"tr_cont": tr_cont, "tr_cont_key": tr_cont_key},
    )


def test_ls_minute_chart_acquires_morning_and_late_pages() -> None:
    client, state = _paging_client(
        [
            _minute_page(
                [
                    {"date": _YMD, "time": "125100", "close": 1000},
                    {"date": _YMD, "time": "200000", "close": 1010},
                ],
                "20260904",
                "090000",
                "Y",
                "k1",
            ),
            _minute_page([{"date": _YMD, "time": "090000", "close": 950}], "", "", "N", "k1"),
        ]
    )
    seen: list[tuple[Any, ...]] = []

    def _observe(payload: Any, metadata: Any, started: Any, received: Any, page: int, attempt: int) -> None:
        seen.append((payload, dict(metadata), started, received, page, attempt))

    res = asyncio.run(client.get_minute_chart(object(), "005930", "2026-09-04", on_page=_observe))
    assert state["calls"] == 2
    assert res["rt_cd"] == "0"
    assert res["truncated"] is False
    assert res["termination_reason"] == "exhausted"
    assert res["pages_fetched"] == 2
    assert len(res["output2"]) == 3
    assert len(seen) == 2
    assert seen[0][4] == 0 and seen[1][4] == 1
    assert any(r["time"] == "200000" for r in seen[0][0]["t8412OutBlock1"])
    assert seen[0][2].tzinfo is not None and seen[0][3] >= seen[0][2]
    assert res["continuation"] == {"cts_date": "", "cts_time": "", "tr_cont": "N", "tr_cont_key": "k1"}


def test_ls_minute_chart_preserves_out_of_window_evidence() -> None:
    client, _ = _paging_client(
        [_minute_page([{"date": _YMD, "time": "200000", "close": 1010}], "", "", "N", "")],
    )
    seen: list[Any] = []
    res = asyncio.run(
        client.get_minute_chart(object(), "005930", "2026-09-04", on_page=lambda *a: seen.append(a)),
    )
    assert res["rt_cd"] == "0"
    assert seen[0][0]["t8412OutBlock1"][0]["time"] == "200000"
    assert res["output2"][0]["time"] == "200000"


def test_ls_tick_chart_reports_page_budget_partial() -> None:
    client, state = _paging_client(
        [
            _tick_page([{"date": _YMD, "time": "143000", "close": 100}], "20260904", "142959", "Y", "k1"),
            _tick_page([{"date": _YMD, "time": "142900", "close": 99}], "20260904", "142859", "Y", "k2"),
        ]
    )
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=2))
    assert state["calls"] == 2
    assert res["rt_cd"] == "0"
    assert res["truncated"] is True
    assert res["termination_reason"] == "page_budget"


def test_ls_tick_chart_stops_on_nonprogress() -> None:
    page = _tick_page([{"date": _YMD, "time": "143000", "close": 100}], "20260904", "142959", "Y", "k")
    client, state = _paging_client([page, page, page, page, page])
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=10))
    assert state["calls"] == 3
    assert res["truncated"] is True
    assert res["termination_reason"] == "nonprogress"


def test_ls_tick_chart_preserves_trade_multiplicity() -> None:
    trade = {"date": _YMD, "time": "100000", "close": 100, "jdiff_vol": 5}
    client, _ = _paging_client(
        [
            _tick_page([dict(trade), dict(trade)], "20260904", "095959", "Y", "k1"),
            _tick_page([{"date": _YMD, "time": "100001", "close": 101, "jdiff_vol": 7}], "", "", "N", ""),
        ]
    )
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5))
    assert res["rt_cd"] == "0"
    assert res["truncated"] is False
    assert res["termination_reason"] == "exhausted"
    assert len(res["output2"]) == 3
    assert sum(1 for r in res["output2"] if r["time"] == "100000") == 2


def test_ls_chart_rejects_invalid_limits() -> None:
    client = _ls_client()
    with pytest.raises(ValueError, match="invalid target date"):
        asyncio.run(client.get_minute_chart(object(), "005930", "2026-13-45"))
    with pytest.raises(ValueError, match="invalid target date"):
        asyncio.run(client.get_tick_chart(object(), "005930", "not-a-date"))
    with pytest.raises(ValueError, match="invalid tick acquisition limits"):
        asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=0))
    with pytest.raises(ValueError, match="conflicting tick acquisition limits"):
        asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=2, budget=_budget(max_pages=3)))
    ok_client, ok_state = _paging_client(
        [_tick_page([{"date": _YMD, "time": "090000", "close": 1}], "", "", "N", "")],
    )
    res = asyncio.run(
        ok_client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5, budget=_budget(max_pages=5)),
    )
    assert ok_state["calls"] == 1
    assert res["truncated"] is False


def test_ls_chart_deadline_bounds_pagination() -> None:
    client, state = _paging_client([_minute_page([], "", "", "N", "")])
    past = datetime.now(_SEOUL) - timedelta(seconds=10)
    res = asyncio.run(client.get_minute_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=past)))
    assert state["calls"] == 0
    assert res["truncated"] is True
    assert res["termination_reason"] == "deadline"
    assert res["pages_fetched"] == 0

    future = datetime.now(_SEOUL) + timedelta(seconds=60)
    client2, state2 = _paging_client([_minute_page([{"date": _YMD, "time": "090000", "close": 1}], "", "", "N", "")])
    res2 = asyncio.run(client2.get_minute_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=future)))
    assert state2["calls"] == 1
    assert res2["truncated"] is False

    async def _slow(session: Any, *a: Any, **k: Any) -> Any:
        await asyncio.sleep(0.5)
        return _minute_page([], "", "", "N", "")

    client3 = _ls_client()
    client3._post_tr = _slow  # type: ignore[method-assign]
    tight = datetime.now(_SEOUL) + timedelta(seconds=0.05)
    res3 = asyncio.run(client3.get_minute_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=tight)))
    assert res3["truncated"] is True
    assert res3["termination_reason"] == "deadline"


def test_ls_chart_observer_failure_propagates() -> None:
    client, _ = _paging_client([_minute_page([{"date": _YMD, "time": "090000", "close": 1}], "", "", "N", "")])

    def _boom(*args: Any) -> None:
        raise RuntimeError("persistence down")

    with pytest.raises(RuntimeError, match="persistence down"):
        asyncio.run(client.get_minute_chart(object(), "005930", "2026-09-04", on_page=_boom))


def test_ls_chart_vendor_failure_reports_termination() -> None:
    client, _ = _paging_client([({"rsp_cd": "99999", "rsp_msg": "bad"}, {"tr_cont": "N"})])
    seen: list[Any] = []
    res = asyncio.run(
        client.get_minute_chart(object(), "005930", "2026-09-04", on_page=lambda *a: seen.append(a)),
    )
    assert res["rt_cd"] == "1"
    assert res["truncated"] is True
    assert res["termination_reason"] == "vendor_failure"
    assert len(seen) == 1

    async def _down(*a: Any, **k: Any) -> Any:
        raise ConnectionError("wire cut")

    client2 = _ls_client()
    client2._post_tr = _down  # type: ignore[method-assign]
    res2 = asyncio.run(client2.get_tick_chart(object(), "005930", "2026-09-04", max_pages=3))
    assert res2["rt_cd"] == "1"
    assert res2["truncated"] is True
    assert res2["termination_reason"] == "vendor_failure"


def test_ls_chart_crosses_target_date() -> None:
    client, _ = _paging_client(
        [_tick_page([{"date": "20260903", "time": "153000", "close": 100}], "20260903", "152959", "Y", "k1")],
    )
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5))
    assert res["rt_cd"] == "0"
    assert res["truncated"] is False
    assert res["termination_reason"] == "crossed_target_date"
    assert res["output2"] == []


def test_ls_chart_cursor_unknown_on_bare_continuation() -> None:
    client, _ = _paging_client(
        [_minute_page([{"date": _YMD, "time": "120000", "close": 1}], "", "", "Y", "")],
    )
    res = asyncio.run(client.get_minute_chart(object(), "005930", "2026-09-04"))
    assert res["truncated"] is True
    assert res["termination_reason"] == "cursor_unknown"


def test_ls_minute_chart_stops_on_nonprogress_crossing_and_budget() -> None:
    page = _minute_page([{"date": _YMD, "time": "120000", "close": 1}], "20260904", "110000", "Y", "k")
    client, state = _paging_client([page, page, page, page])
    res = asyncio.run(client.get_minute_chart(object(), "005930", "2026-09-04"))
    assert state["calls"] == 3
    assert res["truncated"] is True
    assert res["termination_reason"] == "nonprogress"

    old_client, _ = _paging_client(
        [_minute_page([{"date": "20260903", "time": "153000", "close": 1}], "20260903", "152959", "Y", "k1")],
    )
    old_res = asyncio.run(old_client.get_minute_chart(object(), "005930", "2026-09-04"))
    assert old_res["truncated"] is False
    assert old_res["termination_reason"] == "crossed_target_date"
    assert old_res["output2"] == []

    limited, limited_state = _paging_client(
        [
            _minute_page([{"date": _YMD, "time": "120000", "close": 1}], "20260904", "110000", "Y", "k1"),
            _minute_page([{"date": _YMD, "time": "110000", "close": 1}], "20260904", "100000", "Y", "k2"),
        ]
    )
    limited_res = asyncio.run(
        limited.get_minute_chart(object(), "005930", "2026-09-04", budget=_budget(max_pages=2)),
    )
    assert limited_state["calls"] == 2
    assert limited_res["truncated"] is True
    assert limited_res["termination_reason"] == "page_budget"

    async def _down(*a: Any, **k: Any) -> Any:
        raise ConnectionError("wire cut")

    broken = _ls_client()
    broken._post_tr = _down  # type: ignore[method-assign]
    broken_res = asyncio.run(broken.get_minute_chart(object(), "005930", "2026-09-04"))
    assert broken_res["rt_cd"] == "1"
    assert broken_res["truncated"] is True
    assert broken_res["termination_reason"] == "vendor_failure"


def test_ls_tick_chart_uses_default_budget_deadline_and_observer() -> None:
    default_client, default_state = _paging_client(
        [_tick_page([{"date": _YMD, "time": "090001", "close": 1}], "", "", "N", "")],
    )
    default_res = asyncio.run(default_client.get_tick_chart(object(), "005930", "2026-09-04"))
    assert default_state["calls"] == 1
    assert default_res["truncated"] is False
    assert default_res["termination_reason"] == "exhausted"

    client, state = _paging_client([_tick_page([{"date": _YMD, "time": "120000", "close": 1}], "", "", "N", "")])
    past = datetime.now(_SEOUL) - timedelta(seconds=10)
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=past)))
    assert state["calls"] == 0
    assert res["truncated"] is True
    assert res["termination_reason"] == "deadline"

    future = datetime.now(_SEOUL) + timedelta(seconds=60)
    client2, state2 = _paging_client([_tick_page([{"date": _YMD, "time": "120000", "close": 1}], "", "", "N", "")])
    res2 = asyncio.run(client2.get_tick_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=future)))
    assert state2["calls"] == 1
    assert res2["truncated"] is False

    async def _slow(session: Any, *a: Any, **k: Any) -> Any:
        await asyncio.sleep(0.5)
        return _tick_page([], "", "", "N", "")

    client3 = _ls_client()
    client3._post_tr = _slow  # type: ignore[method-assign]
    tight = datetime.now(_SEOUL) + timedelta(seconds=0.05)
    res3 = asyncio.run(client3.get_tick_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=tight)))
    assert res3["truncated"] is True
    assert res3["termination_reason"] == "deadline"

    seen: list[Any] = []
    client4, _ = _paging_client(
        [_tick_page([{"date": _YMD, "time": "120000", "close": 1}], "", "", "N", "")],
    )
    res4 = asyncio.run(client4.get_tick_chart(object(), "005930", "2026-09-04", on_page=lambda *a: seen.append(a)))
    assert len(seen) == 1
    assert seen[0][4] == 0 and seen[0][5] == 0
    assert res4["continuation"] == {"cts_date": "", "cts_time": "", "tr_cont": "N", "tr_cont_key": ""}

    def _boom(*args: Any) -> None:
        raise RuntimeError("store down")

    client5, _ = _paging_client([_tick_page([], "", "", "N", "")])
    with pytest.raises(RuntimeError, match="store down"):
        asyncio.run(client5.get_tick_chart(object(), "005930", "2026-09-04", on_page=_boom))


def test_ls_tick_chart_reports_cursor_unknown_and_vendor_failure() -> None:
    client, _ = _paging_client(
        [_tick_page([{"date": _YMD, "time": "120000", "close": 1}], "", "", "Y", "")],
    )
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5))
    assert res["truncated"] is True
    assert res["termination_reason"] == "cursor_unknown"

    failing, _ = _paging_client([({"rsp_cd": "99999", "rsp_msg": "bad"}, {"tr_cont": "N"})])
    seen: list[Any] = []
    res2 = asyncio.run(failing.get_tick_chart(object(), "005930", "2026-09-04", on_page=lambda *a: seen.append(a)))
    assert res2["rt_cd"] == "1"
    assert res2["truncated"] is True
    assert res2["termination_reason"] == "vendor_failure"
    assert len(seen) == 1


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



def test_ls_ensure_token_single_flight_issues_once_under_concurrency() -> None:
    import asyncio

    from src.api.ls.client import LsApiClient

    client = LsApiClient(app_key="k", app_secret="s")
    counter = {"token": 0, "tr": 0}

    class _Resp:
        def __init__(self, body):
            self._b = body
            self.status = 200
            self.headers = {}

        async def json(self):
            return self._b

    class _Ctx:
        def __init__(self, body, key):
            self._b = body
            self._k = key

        async def __aenter__(self):
            counter[self._k] += 1
            await asyncio.sleep(0.01)
            return _Resp(self._b)

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, url, **_kw):
            if "oauth2/token" in url:
                return _Ctx({"access_token": "T"}, "token")
            return _Ctx({"rsp_cd": "00000", "t8412OutBlock1": []}, "tr")

    async def _run():
        session = _Session()
        await asyncio.gather(*[client.get_minute_chart(session, "005930", "2026-09-10") for _ in range(10)])

    # When: 세마포어(10) 동시 태스크가 첫 호출을 동시에 시작
    asyncio.run(_run())

    # Then: OAuth 발급은 1회 (기존 회귀: 10회)
    assert counter["token"] == 1
    assert counter["tr"] == 10
    assert client.token == "T"



def test_ls_ensure_token_raises_when_issuance_returns_no_token() -> None:
    import asyncio

    import pytest

    from src.api.ls.client import LsApiClient

    client = LsApiClient(app_key='k', app_secret='s')

    class _Resp:
        status = 200

        async def json(self):
            return {}

    class _Ctx:
        async def __aenter__(self):
            return _Resp()

        async def __aexit__(self, *_a):
            return False

    class _Session:
        def post(self, _url, **_kw):
            return _Ctx()

    with pytest.raises(RuntimeError, match='LS token issuance failed'):
        asyncio.run(client.ensure_token(_Session()))


def test_ls_rate_limit_retries_back_off_exponentially() -> None:
    import asyncio

    from src.api.ls.client import LsApiClient

    client = LsApiClient(app_key="k", app_secret="s")
    client.token = "t"
    client._min_interval = 0.0
    calls = {"n": 0}
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    class _Resp:
        headers: dict = {}

        def __init__(self, body: dict):
            self._b = body

        async def json(self):
            return self._b

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def post(self, *a, **k):
            calls["n"] += 1
            if calls["n"] <= 3:
                return _Resp({"rsp_cd": "IGW00201", "rsp_msg": "rate limited"})
            return _Resp({"rsp_cd": "00000", "t8412OutBlock": {}, "t8412OutBlock1": []})

    async def _run():
        import unittest.mock as mock

        with mock.patch("asyncio.sleep", _fake_sleep):
            return await client._post_tr(_Session(), "t8412", "005930", {})

    data, _ = asyncio.run(_run())
    assert data["rsp_cd"] == "00000"
    assert sleeps == [1.2, 2.4, 4.8]


def test_ls_exhausted_retries_logged_distinctly(caplog) -> None:
    import asyncio
    import logging

    from src.api.ls.client import LsApiClient

    client = LsApiClient(app_key="k", app_secret="s")
    client.token = "t"
    client._min_interval = 0.0

    class _Resp:
        headers: dict = {}

        async def json(self):
            return {"rsp_cd": "IGW00201", "rsp_msg": "rate limited"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def post(self, *a, **k):
            return _Resp()

    async def _run():
        import unittest.mock as mock

        with mock.patch("asyncio.sleep", return_value=asyncio.sleep(0)):
            return await client._post_tr(_Session(), "t8412", "005930", {})

    with caplog.at_level(logging.WARNING, logger="src.api.ls.client"):
        data, _ = asyncio.run(_run())
    assert data["rsp_cd"] == "IGW00201"
    assert any("RATE_LIMITED" in r.getMessage() and "attempts=5" in r.getMessage() for r in caplog.records)
