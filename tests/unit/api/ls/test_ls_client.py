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


def test_ls_minute_chart_continuation_cursor_does_not_drop_boundary_bar() -> None:
    """The server returns strictly older rows than the cursor and reports a cursor one minute before the page's oldest row."""
    minutes = [f"{h:02d}{m:02d}00" for h in range(9, 19) for m in range(60) if "090100" <= f"{h:02d}{m:02d}00" <= "181900"]
    client = _ls_client()

    async def fake_post_tr(session: Any, tr_cd: str, tr_key: str, body: dict, tr_cont: str = "N", tr_cont_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        cursor = body["t8412InBlock"]["cts_time"]
        eligible = [t for t in minutes if not cursor or t < cursor]
        page = eligible[-3:]
        rest = eligible[:-3]
        rows = [{"date": _YMD, "time": t, "close": 1} for t in page]
        if not rest:
            return _minute_page(rows, "", "", "N")
        one_before = f"{int(page[0][:4]) - 1:04d}00"
        return _minute_page(rows, _YMD, one_before, "Y", "k")

    client._post_tr = fake_post_tr  # type: ignore[method-assign]
    res = asyncio.run(client.get_minute_chart(object(), "005930", "2026-09-04", budget=_budget(max_pages=1000)))

    got = sorted(r["time"] for r in res["output2"])
    assert res["termination_reason"] == "exhausted"
    assert got == minutes


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

    from src.api.ls.client import LsApiClient
    from tests.broker_fakes import scripted_session

    client = LsApiClient(app_key="k", app_secret="s")
    session = scripted_session(
        [], token_bodies=[{"access_token": "mock_tok", "token_type": "Bearer"}]
    )
    token = asyncio.run(client.ensure_token(session))
    assert token == "mock_tok"
    assert client.token == "mock_tok"


def test_ls_client_get_minute_chart_single_call() -> None:
    import asyncio

    from src.api.ls.client import LsApiClient
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    client = LsApiClient(app_key="k", app_secret="s")
    client.token = "mock_tok"
    mock_bars = [{"time": "090100", "close": 1000, "jdiff_vol": 50}, {"time": "153000", "close": 1050, "jdiff_vol": 200}]
    session = scripted_session(
        [FakeBrokerResponse(body={"rsp_cd": "00000", "t8412OutBlock1": mock_bars})]
    )
    res = asyncio.run(client.get_minute_chart(session, "005930", "2026-09-04"))
    assert res["rt_cd"] == "0"
    assert res["vendor"] == "ls"
    assert len(res["output2"]) == 2
    assert res["output2"][0]["time"] == "090100"


def test_ls_client_get_tick_chart_paginates_with_cts() -> None:
    import asyncio

    from src.api.ls.client import LsApiClient
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    client = LsApiClient(app_key="k", app_secret="s")
    client.token = "mock_tok"
    page1 = {"rsp_cd": "00000", "t8411OutBlock": {"cts_date": "20260904", "cts_time": "151500000"}, "t8411OutBlock1": [{"time": "153000", "close": 1000, "jdiff_vol": 100}]}
    page2 = {"rsp_cd": "00000", "t8411OutBlock": {"cts_date": "", "cts_time": ""}, "t8411OutBlock1": [{"time": "090000", "close": 950, "jdiff_vol": 50}]}
    session = scripted_session([FakeBrokerResponse(body=page1), FakeBrokerResponse(body=page2)])
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
            await asyncio.sleep(0.3)  # in-flight window >> pacing interval so slow CI runners cannot serialize the overlap
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

    async def _no_sleep(_seconds: float) -> None:
        return None

    async def _run():
        import unittest.mock as mock

        with mock.patch("asyncio.sleep", _no_sleep):
            return await client._post_tr(_Session(), "t8412", "005930", {})

    with caplog.at_level(logging.WARNING, logger="src.api._transport"):
        data, _ = asyncio.run(_run())
    assert data["rsp_cd"] == "IGW00201"
    exhausted = [r.getMessage() for r in caplog.records if "status=RATE_LIMITED" in r.getMessage()]
    assert exhausted == [exhausted[0]]
    assert exhausted[0].startswith("[EXEC] stage=ls_tr status=RATE_LIMITED attempts=5")


def test_ls_client_honors_pacing_overrides_from_settings_instance(monkeypatch) -> None:
    from src.api.ls.client import LsApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "LS_MIN_INTERVAL_SECONDS", 9.9)
    monkeypatch.setattr(settings_instance, "LS_RATE_LIMIT_MAX_RETRIES", 7)
    monkeypatch.setattr(settings_instance, "LS_RATE_LIMIT_BACKOFF_SECONDS", 2.5)

    client = LsApiClient(app_key="k", app_secret="s")

    assert client._min_interval == 9.9
    assert client._rate_limit_max_retries == 7
    assert client._rate_limit_backoff == 2.5


def test_ls_client_explicit_credentials_win_over_instance(monkeypatch) -> None:
    from src.api.ls.client import LsApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "LS_APP_KEY", "inst")
    assert LsApiClient().app_key == "inst"
    assert LsApiClient(app_key="arg").app_key == "arg"

    monkeypatch.setattr(settings_instance, "LS_APP_KEY", "")
    monkeypatch.setenv("LS_APP_KEY", "late")
    assert LsApiClient().app_key == ""


def test_ls_client_requests_use_configured_origin(monkeypatch) -> None:
    import asyncio

    from src.api.ls.client import LsApiClient
    from src.config import settings as settings_instance
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    monkeypatch.setattr(settings_instance, "LS_BASE_URL", "https://ls.example:1")
    client = LsApiClient(app_key="k", app_secret="s")
    session = scripted_session(
        [FakeBrokerResponse(body={"rsp_cd": "00000", "t8412OutBlock1": []})],
        token_bodies=[{"access_token": "t"}],
    )
    res = asyncio.run(client.get_minute_chart(session, "005930", "2026-09-04"))

    assert res["rt_cd"] == "0"
    posted = [request.url for request in session.requests]
    assert posted[0] == "https://ls.example:1/oauth2/token"
    assert posted[1] == "https://ls.example:1/stock/chart"


def test_ls_tick_fallback_budget_equals_chart_budget(monkeypatch) -> None:
    import asyncio

    from src.api.ls.client import LsApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "COLLECTION_CHART_MAX_PAGES", 2)
    client, state = _paging_client(
        [_tick_page([{"date": "20260904", "time": "153000"}], "20260904", "151500", "Y", "k1")] * 2
    )
    res = asyncio.run(client.get_tick_chart(None, "005930", "2026-09-04"))

    assert state["calls"] == 2
    assert res["termination_reason"] == "page_budget"


def test_ls_spacing_enforced_across_instances(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.ls.client import LsApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")
    monkeypatch.setattr(settings_instance, "LS_MIN_INTERVAL_SECONDS", 1.05)

    send_times: list[float] = []

    class _Resp:
        status = 200
        headers: dict = {}

        async def json(self):
            return {"rsp_cd": "00000", "t8412OutBlock": {}, "t8412OutBlock1": []}

        async def __aenter__(self):
            send_times.append(asyncio.get_running_loop().time())
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def post(self, *a, **k):
            return _Resp()

    a = LsApiClient(app_key="k", app_secret="s")
    b = LsApiClient(app_key="k", app_secret="s")
    a.token = "t"
    b.token = "t"

    async def _run():
        session = _Session()
        await asyncio.gather(
            a._post_tr(session, "t8412", "005930", {}),
            b._post_tr(session, "t8412", "005930", {}),
        )

    asyncio.run(_run())
    assert len(send_times) == 2
    assert abs(send_times[1] - send_times[0]) >= 1.05 - 0.05


def test_ls_shared_token_single_issuance(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.api.ls.client import LsApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")

    calls = {"oauth": 0}

    class _Resp:
        status = 200
        headers: dict = {}

        def __init__(self, body):
            self._b = body

        async def json(self):
            return self._b

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _Session:
        def post(self, url, **kw):
            if url.endswith("/oauth2/token"):
                calls["oauth"] += 1
                return _Resp({"access_token": "shared-tok", "expires_in": 86400})
            return _Resp({"rsp_cd": "00000"})

    a = LsApiClient(app_key="k", app_secret="s")
    b = LsApiClient(app_key="k", app_secret="s")

    async def _run():
        session = _Session()
        await asyncio.gather(a.ensure_token(session), b.ensure_token(session))

    asyncio.run(_run())
    assert calls["oauth"] == 1


def _ls_auth_session(tr_responses: list, state: dict) -> Any:
    """Plain fake session: scripted /stock/chart replies, token endpoint issuing tok-1, tok-2, ..."""

    class _Resp:
        def __init__(self, body: Any, status: int, headers: Any) -> None:
            self._body = body
            self.status = status
            self.headers = headers

        async def json(self) -> Any:
            return self._body

        async def __aenter__(self) -> _Resp:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    class _Session:
        def post(self, url: str, **kw: Any) -> _Resp:
            if url.endswith("/oauth2/token"):
                state["token_calls"] += 1
                return _Resp({"access_token": f"tok-{state['token_calls']}", "expires_in": 86400}, 200, {})
            headers = dict(kw.get("headers") or {})
            state["tr_calls"].append({
                "auth": headers.get("authorization"), "cont": headers.get("tr_cont"),
                "key": headers.get("tr_cont_key"), "body": dict(kw.get("json") or {}),
            })
            body, status = tr_responses.pop(0)
            return _Resp(body, status, {})

    return _Session()


def _ls_seeded_client(tmp_path: Any, monkeypatch: Any) -> Any:
    import asyncio

    from src.api.ls.client import LsApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")

    client = LsApiClient(app_key="k", app_secret="s")
    client._min_interval = 0.0
    return client


def test_ls_http_401_refreshes_once_and_replays(tmp_path, monkeypatch) -> None:
    import asyncio

    state: dict[str, Any] = {"token_calls": 0, "tr_calls": []}
    client = _ls_seeded_client(tmp_path, monkeypatch)
    session = _ls_auth_session([({"rsp_cd": "40100"}, 401), ({"rsp_cd": "00000"}, 200)], state)
    asyncio.run(client.ensure_token(session))
    assert state["token_calls"] == 1

    data, _ = asyncio.run(client._post_tr(session, "t8412", "005930", {"t8412InBlock": {"shcode": "005930"}}))

    assert data == {"rsp_cd": "00000"}
    assert state["token_calls"] == 2
    assert client._token_store().read() is not None
    assert client._token_store().read().generation == 2  # type: ignore[union-attr]
    assert [c["auth"] for c in state["tr_calls"]] == ["Bearer tok-1", "Bearer tok-2"]
    assert client.token == "tok-2"


def test_ls_igw001xx_body_codes_refresh_once(tmp_path, monkeypatch) -> None:
    import asyncio
    from pathlib import Path

    for code in ("IGW00101", "IGW00102", "IGW00123"):
        admission_dir = Path(str(tmp_path)) / code
        admission_dir.mkdir()
        state: dict[str, Any] = {"token_calls": 0, "tr_calls": []}
        client = _ls_seeded_client(admission_dir, monkeypatch)
        session = _ls_auth_session([({"rsp_cd": code}, 200), ({"rsp_cd": "00000"}, 200)], state)
        asyncio.run(client.ensure_token(session))

        data, _ = asyncio.run(client._post_tr(session, "t8412", "005930", {"t8412InBlock": {"shcode": "005930"}}))

        assert data == {"rsp_cd": "00000"}, code
        assert state["token_calls"] == 2, code
        assert [c["auth"] for c in state["tr_calls"]] == ["Bearer tok-1", "Bearer tok-2"], code


def test_ls_replay_preserves_request(tmp_path, monkeypatch) -> None:
    import asyncio

    state: dict[str, Any] = {"token_calls": 0, "tr_calls": []}
    client = _ls_seeded_client(tmp_path, monkeypatch)
    session = _ls_auth_session([({"rsp_cd": "40100"}, 401), ({"rsp_cd": "00000"}, 200)], state)
    asyncio.run(client.ensure_token(session))

    body = {"t8412InBlock": {"shcode": "005930", "ncnt": 1}}
    asyncio.run(client._post_tr(session, "t8412", "005930", body, tr_cont="Y", tr_cont_key="k9"))

    assert len(state["tr_calls"]) == 2
    first, second = state["tr_calls"]
    assert first["body"] == second["body"] == {**body, "tr_cd": "t8412"}
    assert (first["cont"], first["key"]) == (second["cont"], second["key"]) == ("Y", "k9")


def test_ls_second_auth_rejection_surfaces(tmp_path, monkeypatch) -> None:
    import asyncio

    state: dict[str, Any] = {"token_calls": 0, "tr_calls": []}
    client = _ls_seeded_client(tmp_path, monkeypatch)
    session = _ls_auth_session(
        [({"rsp_cd": "40100"}, 401), ({"rsp_cd": "IGW00121"}, 401), ({"rsp_cd": "00000"}, 200)], state
    )
    asyncio.run(client.ensure_token(session))

    data, _ = asyncio.run(client._post_tr(session, "t8412", "005930", {}))

    assert data == {"rsp_cd": "IGW00121"}
    assert len(state["tr_calls"]) == 2
    assert state["token_calls"] == 2


def test_ls_peer_rotation_is_adopted(tmp_path, monkeypatch) -> None:
    import asyncio
    from datetime import UTC, datetime

    from src.api.shared_token import IssuedToken, SharedTokenStore, shared_token_path

    state: dict[str, Any] = {"token_calls": 0, "tr_calls": []}
    client = _ls_seeded_client(tmp_path, monkeypatch)

    async def _seed_store() -> None:
        store = SharedTokenStore(
            shared_token_path("ls", "k"),
            lock_timeout_seconds=5.0,
            expiry_margin_seconds=0.0,
            clock=lambda: datetime.now(UTC),
        )
        await store.get_or_issue(lambda: asyncio.sleep(0, result=IssuedToken(access_token="tok-A", expires_in_seconds=86400)))
        await store.replace_rejected("tok-A", lambda: asyncio.sleep(0, result=IssuedToken(access_token="tok-B", expires_in_seconds=86400)))

    session = _ls_auth_session([({"rsp_cd": "40100"}, 401), ({"rsp_cd": "00000"}, 200)], state)
    asyncio.run(_seed_store())
    client.token = "tok-A"

    data, _ = asyncio.run(client._post_tr(session, "t8412", "005930", {}))

    assert data == {"rsp_cd": "00000"}
    assert state["token_calls"] == 0
    assert [c["auth"] for c in state["tr_calls"]] == ["Bearer tok-A", "Bearer tok-B"]


def test_ls_refresh_does_not_consume_rate_limit_attempts(tmp_path, monkeypatch) -> None:
    import asyncio

    state: dict[str, Any] = {"token_calls": 0, "tr_calls": []}
    client = _ls_seeded_client(tmp_path, monkeypatch)
    session = _ls_auth_session(
        [({"rsp_cd": "IGW00201"}, 200), ({"rsp_cd": "40100"}, 401), ({"rsp_cd": "00000"}, 200)], state
    )
    asyncio.run(client.ensure_token(session))

    sleeps: list[float] = []

    async def _record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _record_sleep)

    data, _ = asyncio.run(client._post_tr(session, "t8412", "005930", {}, max_retries=2))

    assert data == {"rsp_cd": "00000"}
    assert len(state["tr_calls"]) == 3
    assert state["token_calls"] == 2
    assert sleeps == [1.2]


def test_ls_non_auth_vendor_errors_are_not_refreshed(tmp_path, monkeypatch) -> None:
    import asyncio
    from pathlib import Path

    for idx, (body, status) in enumerate((({"rsp_cd": "IGW00215"}, 200), ({"rsp_cd": "99999"}, 500))):
        admission_dir = Path(str(tmp_path)) / f"case{idx}"
        admission_dir.mkdir()
        state: dict[str, Any] = {"token_calls": 0, "tr_calls": []}
        client = _ls_seeded_client(admission_dir, monkeypatch)
        session = _ls_auth_session([(body, status)], state)
        asyncio.run(client.ensure_token(session))

        data, _ = asyncio.run(client._post_tr(session, "t8412", "005930", {}))

        assert data == body
        assert len(state["tr_calls"]) == 1
        assert state["token_calls"] == 1


def test_ls_post_tr_converts_mapping_headers(tmp_path, monkeypatch) -> None:
    import asyncio
    from typing import Any

    from multidict import CIMultiDict, CIMultiDictProxy

    client = _ls_seeded_client(tmp_path, monkeypatch)

    class _Resp:
        def __init__(self) -> None:
            self.status = 200
            self.headers = CIMultiDictProxy(CIMultiDict({"tr_cont": "Y", "tr_cont_key": "k1"}))

        async def json(self) -> dict:
            return {"rsp_cd": "00000"}

        async def __aenter__(self) -> _Resp:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    class _TokenResp(_Resp):
        def __init__(self) -> None:
            self.status = 200
            self.headers = {}

        async def json(self) -> dict:
            return {"access_token": "tok-1", "expires_in": 86400}

    class _Session:
        def post(self, url: str, **kw: Any) -> Any:
            if url.endswith("/oauth2/token"):
                return _TokenResp()
            return _Resp()

    data, headers = asyncio.run(client._post_tr(_Session(), "t8412", "005930", {}))  # type: ignore[arg-type]

    assert type(headers) is dict
    assert headers.get("tr_cont") == "Y"
    assert headers.get("tr_cont_key") == "k1"
    assert data == {"rsp_cd": "00000"}


_LS_RL_BODY = {"rsp_cd": "IGW00201"}
_LS_OK_BODY = {"rsp_cd": "00000"}


def _ls_tr_session(tr_replies, token_bodies=None):  # type: ignore[no-untyped-def]
    from tests.broker_fakes import scripted_session

    bodies = (
        list(token_bodies)
        if token_bodies is not None
        else [{"access_token": "tok-1", "expires_in": 86400}, {"access_token": "tok-2", "expires_in": 86400}]
    )
    return scripted_session(list(tr_replies), token_bodies=bodies)


def _ls_record_sleep(monkeypatch):  # type: ignore[no-untyped-def]
    import asyncio

    sleeps: list[float] = []

    async def _record(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _record)
    return sleeps


def _ls_count_acquires(monkeypatch):  # type: ignore[no-untyped-def]
    import src.api.ls.client as ls_client_mod

    calls = {"n": 0}

    async def _count(self) -> None:
        calls["n"] += 1

    monkeypatch.setattr(ls_client_mod.HostPacedRateLimiter, "acquire", _count)
    return calls


def _ls_tr_auths(session):  # type: ignore[no-untyped-def]
    return [
        request.headers.get("authorization", "")
        for request in session.requests
        if not request.url.endswith("/oauth2/token")
    ]


def test_ls_tr_http_429_is_not_retried(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _ls_seeded_client(tmp_path, monkeypatch)
    session = _ls_tr_session([FakeBrokerResponse(body={"rsp_cd": "99999"}, status=429)])
    sleeps = _ls_record_sleep(monkeypatch)
    asyncio.run(client.ensure_token(session))

    data, _ = asyncio.run(client._post_tr(session, "t8412", "005930", {}))

    assert data == {"rsp_cd": "99999"}
    assert len(session.requests_to("/stock/chart")) == 1
    assert sleeps == []


def test_ls_tr_backoff_exponent_uses_attempt_index_across_refresh(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _ls_seeded_client(tmp_path, monkeypatch)
    session = _ls_tr_session(
        [
            FakeBrokerResponse(body=dict(_LS_RL_BODY)),
            FakeBrokerResponse(body=dict(_LS_RL_BODY)),
            FakeBrokerResponse(body={"rsp_cd": "40100"}, status=401),
            FakeBrokerResponse(body=dict(_LS_RL_BODY)),
            FakeBrokerResponse(body=dict(_LS_OK_BODY)),
        ]
    )
    sleeps = _ls_record_sleep(monkeypatch)
    asyncio.run(client.ensure_token(session))

    data, _ = asyncio.run(client._post_tr(session, "t8412", "005930", {}))

    assert data == _LS_OK_BODY
    assert len(session.requests_to("/stock/chart")) == 5
    assert sleeps == [1.2, 2.4, 4.8]
    assert len(session.requests_to("/oauth2/token")) == 2


def test_ls_tr_refresh_replay_rate_limit(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _ls_seeded_client(tmp_path, monkeypatch)
    session = _ls_tr_session(
        [
            FakeBrokerResponse(body={"rsp_cd": "40100"}, status=401),
            FakeBrokerResponse(body=dict(_LS_RL_BODY)),
            FakeBrokerResponse(body=dict(_LS_OK_BODY)),
        ]
    )
    sleeps = _ls_record_sleep(monkeypatch)
    asyncio.run(client.ensure_token(session))

    data, _ = asyncio.run(client._post_tr(session, "t8412", "005930", {}))

    assert data == _LS_OK_BODY
    assert len(session.requests_to("/stock/chart")) == 3
    assert sleeps == [1.2]


def test_ls_tr_acquires_before_every_send(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _ls_seeded_client(tmp_path, monkeypatch)
    session = _ls_tr_session(
        [
            FakeBrokerResponse(body=dict(_LS_RL_BODY)),
            FakeBrokerResponse(body={"rsp_cd": "40100"}, status=401),
            FakeBrokerResponse(body=dict(_LS_OK_BODY)),
        ]
    )
    acquires = _ls_count_acquires(monkeypatch)
    _ls_record_sleep(monkeypatch)
    asyncio.run(client.ensure_token(session))

    asyncio.run(client._post_tr(session, "t8412", "005930", {}))

    assert acquires["n"] == len(session.requests_to("/stock/chart")) == 3


def test_ls_tr_concurrent_rejections_issue_once(tmp_path, monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _ls_seeded_client(tmp_path, monkeypatch)
    _ls_count_acquires(monkeypatch)
    session = _ls_tr_session(
        [
            FakeBrokerResponse(body={"rsp_cd": "40100"}, status=401, enter_delay=0.01),
            FakeBrokerResponse(body={"rsp_cd": "40100"}, status=401, enter_delay=0.01),
            FakeBrokerResponse(body=dict(_LS_OK_BODY)),
            FakeBrokerResponse(body=dict(_LS_OK_BODY)),
        ],
        token_bodies=[
            {"access_token": "tok-1", "expires_in": 86400},
            {"access_token": "tok-2", "expires_in": 86400},
            {"access_token": "tok-3", "expires_in": 86400},
        ],
    )
    asyncio.run(client.ensure_token(session))
    assert client._token_store().read().generation == 1  # type: ignore[union-attr]

    async def _main():  # type: ignore[no-untyped-def]
        return await asyncio.gather(
            client._post_tr(session, "t8412", "005930", {}),
            client._post_tr(session, "t8412", "005930", {}),
        )

    (data_a, _), (data_b, _) = asyncio.run(_main())

    assert data_a == _LS_OK_BODY
    assert data_b == _LS_OK_BODY
    assert len(session.requests_to("/oauth2/token")) == 2
    assert client._token_store().read().generation == 2  # type: ignore[union-attr]
    assert sorted(_ls_tr_auths(session)) == ["Bearer tok-1", "Bearer tok-1", "Bearer tok-2", "Bearer tok-2"]


def test_ls_tr_transport_error_propagates(tmp_path, monkeypatch) -> None:
    import asyncio

    import aiohttp

    from tests.broker_fakes import FakeBrokerResponse

    client = _ls_seeded_client(tmp_path, monkeypatch)
    session = _ls_tr_session([FakeBrokerResponse(enter_error=aiohttp.ServerDisconnectedError("boom"))])
    sleeps = _ls_record_sleep(monkeypatch)

    with pytest.raises(aiohttp.ServerDisconnectedError):
        asyncio.run(client._post_tr(session, "t8412", "005930", {}))

    assert len(session.requests_to("/stock/chart")) == 1
    assert sleeps == []


def test_ls_tr_retry_settings_honored(tmp_path, monkeypatch) -> None:
    import asyncio

    from src.config import settings as settings_instance
    from tests.broker_fakes import FakeBrokerResponse

    monkeypatch.setattr(settings_instance, "LS_RATE_LIMIT_MAX_RETRIES", 2)
    monkeypatch.setattr(settings_instance, "LS_RATE_LIMIT_BACKOFF_SECONDS", 0.5)
    client = _ls_seeded_client(tmp_path, monkeypatch)
    session = _ls_tr_session([FakeBrokerResponse(body=dict(_LS_RL_BODY))] * 3)
    sleeps = _ls_record_sleep(monkeypatch)

    asyncio.run(client._post_tr(session, "t8412", "005930", {}))

    assert len(session.requests_to("/stock/chart")) == 2
    assert sleeps == [0.5]
