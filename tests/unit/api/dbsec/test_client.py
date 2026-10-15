from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from src.api.dbsec.client import DbsecApiClient, DbsecTokenThrottled, _token_issue_state_path
from src.data.capture_contracts import ChartBudget
from tests.broker_fakes import FakeBrokerResponse, scripted_session

_YMD = "20261015"
_PREV = "20261014"


def _seed_admission(tmp_path: Any, monkeypatch: Any) -> None:
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")


def _client() -> DbsecApiClient:
    client = DbsecApiClient(app_key="k", app_secret="s")
    client.token = "t"
    return client


def _budget(**overrides: Any) -> ChartBudget:
    base: dict[str, Any] = {"max_pages": 5, "deadline": None, "request_timeout_seconds": 5.0}
    base.update(overrides)
    return ChartBudget(**base)


def _chart_page(rows: list[dict], cont_yn: str = "N", cont_key: str = "") -> tuple[dict, dict]:
    return ({"rsp_cd": "00000", "Out": rows}, {"cont_yn": cont_yn, "cont_key": cont_key})


def _paging_client(pages: list[tuple[dict, dict]]) -> tuple[DbsecApiClient, dict[str, Any]]:
    client = _client()
    state: dict[str, Any] = {"calls": 0, "cont_keys": []}

    async def fake_post(
        session: Any, path: str, body: dict, *, cont_key: str = "", scope: str, code: str = ""
    ) -> tuple[dict, dict]:
        assert state["calls"] < len(pages), "unnecessary extra page requested"
        state["cont_keys"].append(cont_key)
        state["calls"] += 1
        return pages[state["calls"] - 1]

    client._post = fake_post  # type: ignore[method-assign]
    return client, state


def test_token_issue_throttle_raises_without_vendor_call(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)
    client = DbsecApiClient(app_key="k", app_secret="s")
    path = _token_issue_state_path("k")
    path.write_text(repr(time.time() - 10.0))

    posts = {"n": 0}

    class _Session:
        def post(self, *a: Any, **k: Any) -> Any:
            posts["n"] += 1
            raise AssertionError("vendor must not be contacted")

    with pytest.raises(DbsecTokenThrottled):
        asyncio.run(client.ensure_token(_Session()))  # type: ignore[arg-type]
    assert posts["n"] == 0


def test_token_issue_vendor_throttle_maps_to_token_throttled(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)
    client = DbsecApiClient(app_key="k", app_secret="s")
    session = scripted_session([], token_bodies=[{"error_code": "IGW00201"}])
    with pytest.raises(DbsecTokenThrottled):
        asyncio.run(client.ensure_token(session))


def test_token_failure_message_has_no_secret(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)
    client = DbsecApiClient(app_key="KEY-ABC-123", app_secret="SECRET-XYZ-999")
    session = scripted_session([], token_bodies=[{"error_code": "IGW99999", "detail": "nope"}])
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(client.ensure_token(session))
    text = str(exc.value)
    assert "KEY-ABC-123" not in text
    assert "SECRET-XYZ-999" not in text
    assert "nope" not in text
    assert "IGW99999" in text


def test_throttle_code_retried_on_http_403_and_500(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)
    import asyncio as _asyncio

    sleeps: list[float] = []

    async def _record(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(_asyncio, "sleep", _record)
    client = _client()
    session = scripted_session(
        [
            FakeBrokerResponse(body={"rsp_cd": "IGW00201"}, status=403),
            FakeBrokerResponse(body={"rsp_cd": "IGW00201"}, status=500),
            FakeBrokerResponse(body={"rsp_cd": "00000", "Out": []}, status=200),
        ]
    )
    data, _ = asyncio.run(client._post(session, "/api/v1/quote/kr-chart/min", {}, scope="chart", code="005930"))
    assert data["rsp_cd"] == "00000"
    assert len(session.requests_to("/api/v1/quote/kr-chart/min")) == 3


def test_auth_rejection_refreshes_once_via_cas(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)
    client = _client()
    calls: list[str] = []
    orig = client._refresh_rejected_token

    async def _spy(session: Any, sent: str) -> None:
        calls.append(sent)
        await orig(session, sent)

    client._refresh_rejected_token = _spy  # type: ignore[method-assign]
    session = scripted_session(
        [
            FakeBrokerResponse(body={"rsp_cd": "IGW00123"}, status=200),
            FakeBrokerResponse(body={"rsp_cd": "00000", "Out": []}, status=200),
        ],
        token_bodies=[{"access_token": "tok-2", "expires_in": 86400}],
    )
    data, _ = asyncio.run(client._post(session, "/api/v1/quote/kr-chart/min", {}, scope="chart", code="005930"))
    assert data["rsp_cd"] == "00000"
    assert calls == ["t"]


def test_every_send_takes_scope_and_app_admission(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)
    import src.api.dbsec.client as mod

    counts = {"chart": 0, "quote": 0, "app": 0}
    real = mod.get_host_rate_limiter

    class _Rec:
        def __init__(self, name: str) -> None:
            self._name = name

        async def acquire(self) -> None:
            counts[self._name] += 1

    def _fake(path: Any, rate: float, *a: Any, **k: Any) -> Any:
        name = str(path)
        if name.endswith("_chart.state"):
            return _Rec("chart")
        if name.endswith("_quote.state"):
            return _Rec("quote")
        return _Rec("app")

    monkeypatch.setattr(mod, "get_host_rate_limiter", _fake)
    client = _client()
    pages = [
        _chart_page([{"Date": _YMD, "Hour": "150000"}], "Y", "k1"),
        _chart_page([{"Date": _YMD, "Hour": "090000"}], "N", ""),
    ]
    idx = {"n": 0}

    def _session_post(url: str, **kw: Any) -> FakeBrokerResponse:
        body, headers = pages[idx["n"]]
        idx["n"] += 1
        return FakeBrokerResponse(body=body, status=200, headers=dict(headers))

    class _Session:
        def post(self, url: str, **kw: Any) -> FakeBrokerResponse:
            return _session_post(url, **kw)

    res = asyncio.run(client.get_minute_chart(_Session(), "005930", _YMD))  # type: ignore[arg-type]
    assert res["termination_reason"] in ("exhausted", "crossed_target_date")
    assert counts["chart"] == 2
    assert counts["app"] == 2
    assert counts["quote"] == 0
    monkeypatch.setattr(mod, "get_host_rate_limiter", real)


def test_minute_chart_pages_until_target_date_crossed() -> None:
    page1 = _chart_page([{"Date": _YMD, "Hour": "150000"}, {"Date": _YMD, "Hour": "140000"}], "Y", "k1")
    page2 = _chart_page([{"Date": _YMD, "Hour": "090100"}, {"Date": _PREV, "Hour": "195400"}], "Y", "k2")
    client, state = _paging_client([page1, page2])
    res = asyncio.run(client.get_minute_chart(object(), "005930", _YMD))  # type: ignore[arg-type]
    assert res["termination_reason"] == "crossed_target_date"
    assert all(r["Date"] == _YMD for r in res["output2"])
    assert len(res["output2"]) == 3
    assert state["cont_keys"][1] == "k1"


def test_minute_chart_observer_sees_raw_pages_first() -> None:
    page1 = _chart_page([{"Date": _YMD, "Hour": "150000"}], "Y", "k1")
    page2 = _chart_page([{"Date": _PREV, "Hour": "195400"}], "N", "")
    client, _ = _paging_client([page1, page2])
    seen: list[Any] = []
    res = asyncio.run(
        client.get_minute_chart(object(), "005930", _YMD, on_page=lambda *a: seen.append(a))  # type: ignore[arg-type]
    )
    assert len(seen) == 2
    assert seen[1][0]["Out"][0]["Date"] == _PREV
    assert all(r["Date"] == _YMD for r in res["output2"])


def test_minute_chart_nonprogress_terminates() -> None:
    page = _chart_page([{"Date": _YMD, "Hour": "120000"}], "Y", "k")
    client, state = _paging_client([page, page, page, page])
    res = asyncio.run(client.get_minute_chart(object(), "005930", _YMD))  # type: ignore[arg-type]
    assert state["calls"] == 3
    assert res["termination_reason"] == "nonprogress"
    assert res["truncated"] is True
    assert len(res["output2"]) == 1


def test_minute_chart_page_budget_truncates() -> None:
    page = _chart_page([{"Date": _YMD, "Hour": "120000"}], "Y", "k1")
    client, state = _paging_client([page, page])
    res = asyncio.run(
        client.get_minute_chart(object(), "005930", _YMD, budget=_budget(max_pages=1))  # type: ignore[arg-type]
    )
    assert state["calls"] == 1
    assert res["termination_reason"] == "page_budget"
    assert res["truncated"] is True


def test_auction_path_stops_at_time_floor_and_keeps_duplicates() -> None:
    rows1 = [
        {"Hour": "155900", "Prpr": "1"},
        {"Hour": "153500", "Prpr": "1"},
        {"Hour": "152500", "Prpr": "1"},
    ]
    rows2 = [
        {"Hour": "152500", "Prpr": "2"},
        {"Hour": "152000", "Prpr": "1"},
        {"Hour": "151900", "Prpr": "1"},
    ]
    client, _ = _paging_client([_chart_page(rows1, "Y", "k1"), _chart_page(rows2, "Y", "k2")])
    res = asyncio.run(client.get_auction_path(object(), "005930", floor_hhmmss="152000"))  # type: ignore[arg-type]
    assert res["termination_reason"] == "crossed_time_floor"
    hours = [r["Hour"] for r in res["output2"]]
    assert hours == ["155900", "153500", "152500", "152500", "152000"]
    assert all(h >= "152000" for h in hours)


def test_client_rejects_malformed_code_before_send() -> None:
    client = _client()
    sent = {"n": 0}

    class _Session:
        def post(self, *a: Any, **k: Any) -> Any:
            sent["n"] += 1
            raise AssertionError("must not send")

    with pytest.raises(ValueError, match="invalid code"):
        asyncio.run(client.get_minute_chart(_Session(), "5930", _YMD))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="invalid code"):
        asyncio.run(client.get_auction_path(_Session(), "5930", floor_hhmmss="152000"))  # type: ignore[arg-type]
    assert sent["n"] == 0


def test_token_pacing_helpers_edge_cases(tmp_path, monkeypatch) -> None:
    import os as _os

    from src.api.dbsec.client import _mark_attempt, _read_last_attempt, _token_issue_state_path

    _seed_admission(tmp_path, monkeypatch)
    path = _token_issue_state_path("k")
    path.write_text("not-a-float")
    assert _read_last_attempt(path) is None
    path.write_text("")
    assert _read_last_attempt(path) is None
    monkeypatch.setattr(_os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("denied")))
    assert _read_last_attempt(path) is None
    monkeypatch.undo()
    _mark_attempt(path, 123.0)
    assert _read_last_attempt(path) == 123.0


def test_rows_of_dict_and_missing_out() -> None:
    single = {"Date": _YMD, "Hour": "100000"}
    client, _ = _paging_client([({"rsp_cd": "00000", "Out": single}, {"cont_yn": "N"})])
    res = asyncio.run(client.get_minute_chart(object(), "005930", _YMD))  # type: ignore[arg-type]
    assert len(res["output2"]) == 1
    empty_client, _ = _paging_client([({"rsp_cd": "00000"}, {"cont_yn": "N"})])
    res2 = asyncio.run(empty_client.get_minute_chart(object(), "005930", _YMD))  # type: ignore[arg-type]
    assert res2["termination_reason"] == "exhausted"
    assert res2["output2"] == []


def test_ensure_token_success_and_auth_classifier(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)
    client = DbsecApiClient(app_key="k", app_secret="s")
    session = scripted_session([], token_bodies=[{"access_token": "good", "expires_in": 86400}])
    assert asyncio.run(client.ensure_token(session)) == "good"
    assert DbsecApiClient._is_auth_rejected(401, {}) is True
    assert DbsecApiClient._is_auth_rejected(200, {"rsp_cd": "IGW00121"}) is True
    assert DbsecApiClient._is_auth_rejected(200, {"rsp_cd": "00000"}) is False
    assert DbsecApiClient._is_auth_rejected(200, []) is False  # type: ignore[arg-type]


def test_post_ensures_token_when_missing(tmp_path, monkeypatch) -> None:
    _seed_admission(tmp_path, monkeypatch)
    client = DbsecApiClient(app_key="k", app_secret="s")
    assert client.token is None
    session = scripted_session(
        [FakeBrokerResponse(body={"rsp_cd": "00000", "Out": []}, status=200)],
        token_bodies=[{"access_token": "tok-9", "expires_in": 86400}],
    )
    data, _ = asyncio.run(client._post(session, "/api/v1/quote/kr-chart/min", {}, scope="chart", code="005930"))
    assert data["rsp_cd"] == "00000"
    assert client.token == "tok-9"


def test_minute_chart_deadline_vendor_transport_observer(tmp_path, monkeypatch) -> None:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    _seed_admission(tmp_path, monkeypatch)
    seoul = ZoneInfo("Asia/Seoul")
    past = datetime.now(seoul) - timedelta(seconds=10)
    client, state = _paging_client([_chart_page([{"Date": _YMD, "Hour": "1"}], "N", "")])
    res = asyncio.run(
        client.get_minute_chart(object(), "005930", _YMD, budget=_budget(deadline=past))  # type: ignore[arg-type]
    )
    assert state["calls"] == 0
    assert res["termination_reason"] == "deadline"
    assert res["truncated"] is True

    future = datetime.now(seoul) + timedelta(seconds=60)
    client2, _ = _paging_client([_chart_page([{"Date": _YMD, "Hour": "1"}], "N", "")])
    res2 = asyncio.run(
        client2.get_minute_chart(object(), "005930", _YMD, budget=_budget(deadline=future))  # type: ignore[arg-type]
    )
    assert res2["termination_reason"] == "exhausted"

    bad_client, _ = _paging_client([({"rsp_cd": "99999", "rsp_msg": "bad"}, {"cont_yn": "N"})])
    bad = asyncio.run(bad_client.get_minute_chart(object(), "005930", _YMD))  # type: ignore[arg-type]
    assert bad["rt_cd"] == "1"
    assert bad["termination_reason"] == "vendor_failure"

    async def _down(session: Any, *a: Any, **k: Any) -> Any:
        raise ConnectionError("wire cut")

    broken = _client()
    broken._post = _down  # type: ignore[method-assign]
    res3 = asyncio.run(broken.get_minute_chart(object(), "005930", _YMD))  # type: ignore[arg-type]
    assert res3["rt_cd"] == "1"
    assert res3["termination_reason"] == "vendor_failure"

    obs_client, _ = _paging_client([_chart_page([{"Date": _YMD, "Hour": "1"}], "N", "")])

    def _boom(*a: Any) -> None:
        raise RuntimeError("store down")

    with pytest.raises(RuntimeError, match="store down"):
        asyncio.run(obs_client.get_minute_chart(object(), "005930", _YMD, on_page=_boom))  # type: ignore[arg-type]

    slow = _client()

    async def _slow(session: Any, *a: Any, **k: Any) -> Any:
        await asyncio.sleep(0.5)
        return _chart_page([], "N", "")

    slow._post = _slow  # type: ignore[method-assign]
    tight = datetime.now(seoul) + timedelta(seconds=0.05)
    res4 = asyncio.run(
        slow.get_minute_chart(object(), "005930", _YMD, budget=_budget(deadline=tight))  # type: ignore[arg-type]
    )
    assert res4["termination_reason"] == "deadline"
    assert res4["truncated"] is True


def test_auction_path_branches(tmp_path, monkeypatch) -> None:
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo

    _seed_admission(tmp_path, monkeypatch)
    seoul = ZoneInfo("Asia/Seoul")
    with pytest.raises(ValueError, match="invalid floor"):
        asyncio.run(_client().get_auction_path(object(), "005930", floor_hhmmss="nope"))  # type: ignore[arg-type]

    ok_client, _ = _paging_client([_chart_page([{"Hour": "153000"}, {"Hour": "152500"}], "N", "")])
    ok = asyncio.run(ok_client.get_auction_path(object(), "005930", floor_hhmmss="152000"))  # type: ignore[arg-type]
    assert ok["termination_reason"] == "exhausted"
    assert [r["Hour"] for r in ok["output2"]] == ["153000", "152500"]

    bad_client, _ = _paging_client([({"rsp_cd": "E", "rsp_msg": "bad"}, {"cont_yn": "N"})])
    bad = asyncio.run(bad_client.get_auction_path(object(), "005930", floor_hhmmss="152000"))  # type: ignore[arg-type]
    assert bad["rt_cd"] == "1"
    assert bad["termination_reason"] == "vendor_failure"

    lim_client, lim_state = _paging_client(
        [
            _chart_page([{"Hour": "155900"}], "Y", "k1"),
            _chart_page([{"Hour": "155800"}], "Y", "k2"),
        ]
    )
    lim = asyncio.run(
        lim_client.get_auction_path(object(), "005930", floor_hhmmss="152000", budget=_budget(max_pages=2))  # type: ignore[arg-type]
    )
    assert lim_state["calls"] == 2
    assert lim["termination_reason"] == "page_budget"
    assert lim["truncated"] is True

    page = _chart_page([{"Hour": "155900"}], "Y", "k")
    stall_client, stall_state = _paging_client([page, page, page, page])
    stall = asyncio.run(stall_client.get_auction_path(object(), "005930", floor_hhmmss="152000"))  # type: ignore[arg-type]
    assert stall_state["calls"] == 3
    assert stall["termination_reason"] == "nonprogress"
    assert stall["truncated"] is True
    assert len(stall["output2"]) == 1

    past = datetime.now(seoul) - timedelta(seconds=10)
    dl_client, dl_state = _paging_client([_chart_page([], "N", "")])
    dl = asyncio.run(
        dl_client.get_auction_path(object(), "005930", floor_hhmmss="152000", budget=_budget(deadline=past))  # type: ignore[arg-type]
    )
    assert dl_state["calls"] == 0
    assert dl["termination_reason"] == "deadline"

    fut_client, _ = _paging_client([_chart_page([{"Hour": "153000"}], "N", "")])
    seen: list[Any] = []
    fut = asyncio.run(
        fut_client.get_auction_path(
            object(),  # type: ignore[arg-type]
            "005930",
            floor_hhmmss="152000",
            budget=_budget(deadline=datetime.now(seoul) + timedelta(seconds=60)),
            on_page=lambda *a: seen.append(a),
        )
    )
    assert fut["termination_reason"] == "exhausted"
    assert len(seen) == 1

    async def _down(session: Any, *a: Any, **k: Any) -> Any:
        raise ConnectionError("wire cut")

    broken = _client()
    broken._post = _down  # type: ignore[method-assign]
    res = asyncio.run(broken.get_auction_path(object(), "005930", floor_hhmmss="152000"))  # type: ignore[arg-type]
    assert res["rt_cd"] == "1"
    assert res["termination_reason"] == "vendor_failure"

    obs_client, _ = _paging_client([_chart_page([{"Hour": "153000"}], "N", "")])

    def _boom(*a: Any) -> None:
        raise RuntimeError("persist down")

    with pytest.raises(RuntimeError, match="persist down"):
        asyncio.run(obs_client.get_auction_path(object(), "005930", floor_hhmmss="152000", on_page=_boom))  # type: ignore[arg-type]

    slow = _client()

    async def _slow(session: Any, *a: Any, **k: Any) -> Any:
        await asyncio.sleep(0.5)
        return _chart_page([], "N", "")

    slow._post = _slow  # type: ignore[method-assign]
    tight = datetime.now(seoul) + timedelta(seconds=0.05)
    res2 = asyncio.run(
        slow.get_auction_path(object(), "005930", floor_hhmmss="152000", budget=_budget(deadline=tight))  # type: ignore[arg-type]
    )
    assert res2["termination_reason"] == "deadline"
