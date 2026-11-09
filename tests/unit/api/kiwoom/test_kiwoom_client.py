from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from src.api.kiwoom.client import KiwoomApiClient
from src.data.capture_contracts import ChartBudget, RawCaptureError

_SEOUL = ZoneInfo("Asia/Seoul")
_YMD = "20260904"


def _kiwoom_client() -> KiwoomApiClient:
    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"
    return client


def _budget(**overrides: Any) -> ChartBudget:
    base: dict[str, Any] = {"max_pages": 5, "deadline": None, "request_timeout_seconds": 5.0}
    base.update(overrides)
    return ChartBudget(**base)


def _tick_pages(pages: list[tuple[dict, dict]]) -> Any:
    state: dict[str, Any] = {"calls": 0}

    async def fake_post_tr(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        assert state["calls"] < len(pages), "unnecessary extra page requested"
        state["calls"] += 1
        return pages[state["calls"] - 1]

    return fake_post_tr, state


def _tick_page(rows: list[dict], cont_yn: str = "N", next_key: str = "") -> tuple[dict, dict]:
    return ({"return_code": 0, "return_msg": "OK", "stk_tic_chart_qry": rows}, {"cont-yn": cont_yn, "next-key": next_key})


def test_kiwoom_tick_truncated_success_remains_repairable() -> None:
    from src.api.kiwoom.client import KiwoomApiClient

    client = _kiwoom_client()
    fake, state = _tick_pages(
        [
            _tick_page([{"cur_prc": "270000", "trde_qty": "100", "cntr_tm": "20260904143000"}], "Y", "k1"),
            _tick_page([{"cur_prc": "269000", "trde_qty": "50", "cntr_tm": "20260904140000"}], "Y", "k2"),
        ]
    )
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=2))
    assert state["calls"] == 2
    assert res["rt_cd"] == "0"
    assert res["vendor"] == "kiwoom"
    assert res["truncated"] is True
    assert res["termination_reason"] == "page_budget"
    assert res["pages_fetched"] == 2
    assert res["continuation"] == {"cont-yn": "Y", "next-key": "k2"}
    assert len(res["output2"]) == 2


def test_kiwoom_ranking_observer_sees_prefilter_rows() -> None:
    from src.api.kiwoom.client import KiwoomApiClient

    client = _kiwoom_client()
    rows = [
        {"stk_cd": "004490_AL", "stk_nm": "A", "cur_prc": "+69700", "flu_rt": "+8.40", "now_trde_qty": "10"},
        {"stk_cd": "005930_AL", "stk_nm": "B", "cur_prc": "+80000", "flu_rt": "+1.25", "now_trde_qty": "20"},
    ]

    async def fake_post_tr(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        return ({"return_code": 0, "pred_pre_flu_rt_upper": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr
    seen: list[Any] = []
    res = asyncio.run(
        client.get_fluctuation_ranking(
            object(), rate_min_pct=3.0, rate_max_pct=9.0, on_page=lambda *a: seen.append(a)
        ),
    )
    assert res["rt_cd"] == "0"
    assert len(res["output"]) == 1
    assert len(seen) == 1
    assert len(seen[0][0]["pred_pre_flu_rt_upper"]) == 2
    assert seen[0][1]["cont-yn"] == "N"
    assert seen[0][1]["next-key"] == ""
    assert seen[0][1]["vendor"] == "kiwoom"
    assert seen[0][1]["endpoint"] == "fluctuation-ranking"
    assert seen[0][4] == 0


def test_kiwoom_tick_missing_cursor_cannot_certify_completion() -> None:
    from src.api.kiwoom.client import KiwoomApiClient

    client = _kiwoom_client()
    fake, state = _tick_pages(
        [_tick_page([{"cur_prc": "270000", "trde_qty": "100", "cntr_tm": "20260904143000"}], "Y", "")],
    )
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5))
    assert state["calls"] == 1
    assert res["rt_cd"] == "0"
    assert res["truncated"] is True
    assert res["termination_reason"] == "cursor_unknown"


def test_kiwoom_tick_deadline_blocks_post_deadline_request() -> None:
    from src.api.kiwoom.client import KiwoomApiClient

    client = _kiwoom_client()
    fake, state = _tick_pages([_tick_page([], "N", "")])
    client._post_tr = fake
    past = datetime.now(_SEOUL) - timedelta(seconds=10)
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=past)))
    assert state["calls"] == 0
    assert res["truncated"] is True
    assert res["termination_reason"] == "deadline"
    assert res["pages_fetched"] == 0

    async def _slow(session: Any, *a: Any, **k: Any) -> Any:
        await asyncio.sleep(0.5)
        return _tick_page([], "N", "")

    client2 = _kiwoom_client()
    client2._post_tr = _slow
    tight = datetime.now(_SEOUL) + timedelta(seconds=0.05)
    res2 = asyncio.run(client2.get_tick_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=tight)))
    assert res2["truncated"] is True
    assert res2["termination_reason"] == "deadline"


def test_kiwoom_tick_pages_record_real_clocks() -> None:
    from src.api.kiwoom.client import KiwoomApiClient

    client = _kiwoom_client()
    fake, _ = _tick_pages(
        [
            _tick_page([{"cur_prc": "270000", "trde_qty": "100", "cntr_tm": "20260904153000"}], "Y", "k1"),
            _tick_page([{"cur_prc": "269000", "trde_qty": "50", "cntr_tm": "20260904090000"}], "N", ""),
        ]
    )
    client._post_tr = fake
    seen: list[Any] = []
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5, on_page=lambda *a: seen.append(a)))
    assert res["termination_reason"] == "exhausted"
    assert len(seen) == 2
    assert [call[4] for call in seen] == [0, 1]
    for call in seen:
        assert call[2].tzinfo is not None and call[3].tzinfo is not None
        assert call[3] >= call[2]
    assert seen[1][3] >= seen[0][3]


def test_kiwoom_observer_failure_is_visible() -> None:
    from src.api.kiwoom.client import KiwoomApiClient

    client = _kiwoom_client()

    async def fake_post_tr(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        rows = [{"stk_cd": "004490_AL", "flu_rt": "+8.40"}]
        return ({"return_code": 0, "pred_pre_flu_rt_upper": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr

    def _boom(*args: Any) -> None:
        raise RuntimeError("store down")

    with pytest.raises(RawCaptureError, match="store down"):
        asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=3.0, rate_max_pct=9.0, on_page=_boom))

    client2 = _kiwoom_client()
    fake2, _ = _tick_pages([_tick_page([{"cur_prc": "1", "trde_qty": "1", "cntr_tm": "20260904120000"}], "N", "")])
    client2._post_tr = fake2

    def _raw_boom(*args: Any) -> None:
        raise RawCaptureError("durable store down")

    with pytest.raises(RawCaptureError, match="durable store down"):
        asyncio.run(client2.get_tick_chart(object(), "005930", "2026-09-04", on_page=_raw_boom))

    def _raw_ranking_boom(*args: Any) -> None:
        raise RawCaptureError("ranking store down")

    client3 = _kiwoom_client()
    client3._post_tr = fake_post_tr
    with pytest.raises(RawCaptureError, match="ranking store down"):
        asyncio.run(
            client3.get_fluctuation_ranking(object(), rate_min_pct=3.0, rate_max_pct=9.0, on_page=_raw_ranking_boom)
        )


def test_kiwoom_tick_crosses_target_date() -> None:
    from src.api.kiwoom.client import KiwoomApiClient

    client = _kiwoom_client()
    fake, _ = _tick_pages(
        [_tick_page([{"cur_prc": "270000", "trde_qty": "100", "cntr_tm": "20260903153000"}], "Y", "k1")],
    )
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5))
    assert res["truncated"] is False
    assert res["termination_reason"] == "crossed_target_date"
    assert res["output2"] == []


def test_kiwoom_tick_stops_on_nonprogress_and_rejects_bad_bounds() -> None:
    from src.api.kiwoom.client import KiwoomApiClient

    client = _kiwoom_client()
    page = _tick_page([{"cur_prc": "270000", "trde_qty": "100", "cntr_tm": "20260904143000"}], "Y", "k")
    fake, state = _tick_pages([page, page, page, page])
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=10))
    assert state["calls"] == 3
    assert res["truncated"] is True
    assert res["termination_reason"] == "nonprogress"

    with pytest.raises(ValueError, match="conflicting tick acquisition limits"):
        asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=2, budget=_budget(max_pages=3)))
    with pytest.raises(ValueError, match="invalid tick acquisition limits"):
        asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=0))
    with pytest.raises(ValueError, match="invalid target date"):
        asyncio.run(client.get_tick_chart(object(), "005930", "not-a-date"))


def test_kiwoom_tick_vendor_failure_reports_termination() -> None:
    from src.api.kiwoom.client import KiwoomApiClient

    client = _kiwoom_client()

    async def fake_post_tr(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        return ({"return_code": 3, "return_msg": "bad code"}, {"cont-yn": "N"})

    client._post_tr = fake_post_tr
    seen: list[Any] = []
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", on_page=lambda *a: seen.append(a)))
    assert res["rt_cd"] == "1"
    assert res["truncated"] is True
    assert res["termination_reason"] == "vendor_failure"
    assert len(seen) == 1


def test_kiwoom_client_ensure_token() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient
    from tests.broker_fakes import scripted_session

    client = KiwoomApiClient(app_key="k", secret_key="s")
    session = scripted_session(
        [],
        token_bodies=[{"token": "mock_tok", "return_code": 0, "expires_dt": "20261002151006"}],
    )

    token = asyncio.run(client.ensure_token(session))

    assert token == "mock_tok"
    assert client.token == "mock_tok"


def test_kiwoom_client_ensure_token_concurrent_calls_lock_and_request_once() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    client = KiwoomApiClient(app_key="k", secret_key="s")
    session = scripted_session(
        [],
        token_bodies=[
            FakeBrokerResponse(
                body={"token": "tok_123", "return_code": 0, "expires_dt": "20261002151006"},
                enter_delay=0.01,
            )
        ],
    )

    async def runner():
        tokens = await asyncio.gather(*[client.ensure_token(session) for _ in range(10)])
        assert all(t == "tok_123" for t in tokens)

    asyncio.run(runner())
    assert len(session.requests_to("/oauth2/token")) == 1


def test_kiwoom_client_ensure_token_raises_on_empty_token() -> None:
    import asyncio

    import pytest

    from src.api.kiwoom.client import KiwoomApiClient
    from tests.broker_fakes import scripted_session

    client = KiwoomApiClient(app_key="k", secret_key="s")
    session = scripted_session(
        [], token_bodies=[{"return_code": 3, "return_msg": "invalid appkey"}]
    )

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

    import src.api.kiwoom.client as kiwoom_client_mod
    from tests.broker_fakes import FakeBrokerResponse, scripted_session

    client = kiwoom_client_mod.KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"

    _real_sleep = asyncio.sleep

    async def _fast_sleep(seconds) -> None:
        await _real_sleep(0)

    monkeypatch.setattr(kiwoom_client_mod.asyncio, "sleep", _fast_sleep)

    session = scripted_session(
        [
            FakeBrokerResponse(
                body={"return_code": 5, "return_msg": "허용된 API 요청 개수를 초과하였습니다. 유량=5"},
                status=429,
            ),
            FakeBrokerResponse(
                body={"return_code": 0, "return_msg": "OK", "stk_tic_chart_qry": []}, status=200
            ),
        ]
    )

    data, headers = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {"stk_cd": "005930"}))

    assert data.get("return_code") == 0
    assert len(session.requests) == 2


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
    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=3.0, rate_max_pct=9.0))
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
    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=3.0, rate_max_pct=9.0, max_pages=5))
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
    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=3.0, rate_max_pct=9.0, max_pages=5))
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

    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=3.0, rate_max_pct=9.0))

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

    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=3.0, rate_max_pct=9.0))

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

    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=3.0, rate_max_pct=9.0))

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


def test_kiwoom_get_fluctuation_ranking_flags_truncation_when_pages_exhausted() -> None:
    import asyncio
    import inspect

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"
    calls = {"n": 0}

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        calls["n"] += 1
        rows = [{"stk_cd": f"00000{calls['n']}_AL", "stk_nm": "X", "cur_prc": "+1000", "pred_pre": "+50", "flu_rt": "+5.00", "now_trde_qty": "1000"}]
        return ({"return_code": 0, "pred_pre_flu_rt_upper": rows}, {"cont-yn": "Y", "next-key": f"k{calls['n']}"})

    client._post_tr = fake_post_tr

    # When: 모든 페이지가 밴드 안이고 다음 페이지가 남은 채 예산 소진
    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=3.0, rate_max_pct=9.0, max_pages=3))

    # Then: 무경고 성공 금지
    assert calls["n"] == 3
    assert res["rt_cd"] == "1"
    assert res["truncated"] is True
    assert len(res["output"]) == 3
    assert inspect.signature(KiwoomApiClient.get_fluctuation_ranking).parameters["max_pages"].default == 20

def _trade_value_row(code: str, trde_prica: str) -> dict[str, str]:
    return {"stk_cd": code, "stk_nm": "X", "cur_prc": "+1000", "flu_rt": "-15.00", "now_rank": "1", "trde_prica": trde_prica}


def test_kiwoom_trade_value_ranking_stops_below_floor() -> None:
    import asyncio

    client = _kiwoom_client()
    pages = [
        ([_trade_value_row("005930", "1000"), _trade_value_row("000660", "20000")], {"cont-yn": "Y", "next-key": "k2"}),
        ([_trade_value_row("035420", "15000"), _trade_value_row("051910", "5000")], {"cont-yn": "Y", "next-key": "k3"}),
        ([_trade_value_row("207940", "4000")], {"cont-yn": "N", "next-key": ""}),
    ]
    calls: list[dict[str, Any]] = []

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        calls.append({"cont_yn": cont_yn, "next_key": next_key})
        rows, headers = pages[len(calls) - 1]
        return ({"return_code": 0, "trde_prica_upper": rows}, headers)

    client._post_tr = fake_post_tr
    res = asyncio.run(
        client.get_trade_value_ranking(object(), max_pages=5, min_trade_value_krw=10_000_000_000)
    )

    assert len(calls) == 2
    assert res["rt_cd"] == "0"
    assert res["truncated"] is False
    assert [row["stk_cd"] for row in res["output"]] == ["005930", "000660", "035420", "051910"]


def test_kiwoom_trade_value_ranking_truncated_keeps_rows() -> None:
    import asyncio

    client = _kiwoom_client()
    calls = {"n": 0}

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        calls["n"] += 1
        row = _trade_value_row(f"00000{calls['n']}", "90000")
        return ({"return_code": 0, "trde_prica_upper": [row]}, {"cont-yn": "Y", "next-key": f"k{calls['n']}"})

    client._post_tr = fake_post_tr
    res = asyncio.run(client.get_trade_value_ranking(object(), max_pages=2, min_trade_value_krw=0))

    assert res["rt_cd"] == "1"
    assert res["truncated"] is True
    assert [row["stk_cd"] for row in res["output"]] == ["000001", "000002"]


def test_kiwoom_trade_value_ranking_observer_sees_unfiltered_pages() -> None:
    import asyncio

    client = _kiwoom_client()
    rows = [_trade_value_row("005930", "1000"), _trade_value_row("000660", "10")]

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        return ({"return_code": 0, "trde_prica_upper": rows}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr
    seen: list[Any] = []
    res = asyncio.run(
        client.get_trade_value_ranking(
            object(), max_pages=3, min_trade_value_krw=10_000_000_000, on_page=lambda *a: seen.append(a)
        ),
    )

    assert len(seen) == 1
    assert seen[0][0]["trde_prica_upper"] == rows
    assert seen[0][1]["endpoint"] == "trade-value-ranking"
    assert seen[0][1]["vendor"] == "kiwoom"
    assert seen[0][4] == 0
    assert len(res["output"]) == 2


def test_kiwoom_trade_value_ranking_observer_failure_is_raw_capture_error() -> None:
    import asyncio

    client = _kiwoom_client()

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        return ({"return_code": 0, "trde_prica_upper": [_trade_value_row("005930", "1000")]}, {"cont-yn": "N"})

    def _boom(*args: Any) -> None:
        raise OSError("disk full")

    client._post_tr = fake_post_tr
    with pytest.raises(RawCaptureError):
        asyncio.run(client.get_trade_value_ranking(object(), max_pages=1, min_trade_value_krw=0, on_page=_boom))


def test_kiwoom_trade_value_ranking_vendor_error_returns_soft_failure() -> None:
    import asyncio

    client = _kiwoom_client()

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        return ({"return_code": 3, "return_msg": "rate limited"}, {"cont-yn": "N"})

    client._post_tr = fake_post_tr
    res = asyncio.run(client.get_trade_value_ranking(object(), max_pages=3, min_trade_value_krw=0))

    assert res["rt_cd"] == "1"
    assert res["output"] == []
    assert res["truncated"] is False


def test_kiwoom_trade_value_ranking_request_body_contract() -> None:
    import asyncio

    client = _kiwoom_client()
    sent: list[tuple[str, str, dict[str, Any]]] = []

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        sent.append((api_id, path, body))
        return ({"return_code": 0, "trde_prica_upper": []}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr
    asyncio.run(client.get_trade_value_ranking(object(), max_pages=1, min_trade_value_krw=0))

    assert sent == [
        ("ka10032", "/api/dostk/rkinfo", {"mrkt_tp": "000", "mang_stk_incls": "0", "stex_tp": "1"}),
    ]



def test_kiwoom_client_explicit_credentials_win_over_instance(monkeypatch) -> None:
    from src.api.kiwoom.client import KiwoomApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "KIWOOM_APP_KEY", "inst")
    monkeypatch.setattr(settings_instance, "KIWOOM_BASE_URL", "https://kw.example")

    assert KiwoomApiClient().app_key == "inst"
    assert KiwoomApiClient().base_url == "https://kw.example"
    assert KiwoomApiClient(app_key="arg", base_url="https://arg.example").app_key == "arg"
    assert KiwoomApiClient(app_key="arg", base_url="https://arg.example").base_url == "https://arg.example"

    monkeypatch.setattr(settings_instance, "KIWOOM_APP_KEY", "")
    monkeypatch.setenv("KIWOM_APP_KEY", "late")
    assert KiwoomApiClient().app_key == ""


def test_kiwoom_tick_fallback_budget_equals_chart_budget(monkeypatch) -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "COLLECTION_CHART_MAX_PAGES", 2)
    client = _kiwoom_client()
    rows = [{"cntr_tm": "20260904153000"}]
    fake_post_tr, state = _tick_pages([_tick_page(rows, "Y", "k1"), _tick_page(rows, "Y", "k2")])
    client._post_tr = fake_post_tr  # type: ignore[method-assign]
    res = asyncio.run(client.get_tick_chart(None, "005930", "2026-09-04"))

    assert state["calls"] == 2
    assert res["termination_reason"] == "page_budget"


def test_kiwoom_tick_venue_nxt_sends_nx_code() -> None:
    import asyncio

    client = _kiwoom_client()
    seen: list[dict] = []

    async def fake_post_tr(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        seen.append(dict(body))
        return ({"return_code": 0, "stk_tic_chart_qry": []}, {"cont-yn": "N", "next-key": ""})

    client._post_tr = fake_post_tr
    asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=2, venue="NXT"))
    assert seen[0]["stk_cd"] == "005930_NX"

    client2 = _kiwoom_client()
    seen2: list[dict] = []

    async def fake_post_tr2(session, api_id, path, body, cont_yn="N", next_key="", max_retries=3):
        seen2.append(dict(body))
        return ({"return_code": 0, "stk_tic_chart_qry": []}, {"cont-yn": "N", "next-key": ""})

    client2._post_tr = fake_post_tr2
    asyncio.run(client2.get_tick_chart(object(), "005930", "2026-09-04", max_pages=2, venue="KRX"))
    assert seen2[0]["stk_cd"] == "005930"
    assert seen2[0] == {"stk_cd": "005930", "tic_scope": "1", "upd_stkpc_tp": "1", "base_dt": "20260904"}


def test_kiwoom_tick_floor_stops_pagination_without_truncation() -> None:
    import asyncio

    client = _kiwoom_client()
    fake, state = _tick_pages(
        [
            _tick_page([{"cur_prc": "1", "trde_qty": "1", "cntr_tm": "20260904170000"}], "Y", "k1"),
            _tick_page([{"cur_prc": "1", "trde_qty": "1", "cntr_tm": "20260904153023"}], "Y", "k2"),
        ]
    )
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5, floor_hms="160000"))
    assert state["calls"] == 2
    assert res["termination_reason"] == "crossed_time_floor"
    assert res["truncated"] is False


def test_kiwoom_tick_floor_ignored_on_other_dates() -> None:
    import asyncio

    client = _kiwoom_client()
    fake, _ = _tick_pages(
        [_tick_page([{"cur_prc": "1", "trde_qty": "1", "cntr_tm": "20260903153023"}], "Y", "k1")],
    )
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5, floor_hms="160000"))
    assert res["termination_reason"] == "crossed_target_date"


def test_kiwoom_tick_rejects_unknown_venue_and_malformed_floor() -> None:
    import asyncio

    import pytest

    client = _kiwoom_client()
    with pytest.raises(ValueError, match="venue"):
        asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=1, venue="AL"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="floor_hms"):
        asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=1, floor_hms="1600"))


def _remaining_key(remaining: int, code: str = "005930", ymd: str = _YMD) -> str:
    return f"A{code}{ymd}{remaining:08d}"


def _tick_rows(count: int) -> list[dict]:
    return [{"cur_prc": "270000", "trde_qty": "1", "cntr_tm": f"{_YMD}{140000 + i:06d}"} for i in range(count)]


def test_kiwoom_tick_total_is_parsed_from_remaining_count_and_certifies_exhaustion() -> None:
    client = _kiwoom_client()
    fake, _ = _tick_pages(
        [
            _tick_page(_tick_rows(3), "Y", _remaining_key(4)),
            _tick_page(_tick_rows(3), "Y", _remaining_key(1)),
            _tick_page(_tick_rows(1), "N", _remaining_key(0)),
        ]
    )
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5))
    assert res["vendor_total_ticks"] == 7
    assert res["rows_received"] == 7
    assert res["termination_reason"] == "exhausted"
    assert res["complete_by_total"] is True
    assert res["cursor"] is None


def test_kiwoom_tick_unparseable_key_gives_no_total_and_no_certificate() -> None:
    client = _kiwoom_client()
    fake, _ = _tick_pages([_tick_page(_tick_rows(2), "Y", "opaque-key"), _tick_page(_tick_rows(2), "N", "")])
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5))
    assert res["vendor_total_ticks"] is None
    assert res["complete_by_total"] is False


def test_kiwoom_tick_key_for_other_code_or_date_is_not_parsed() -> None:
    client = _kiwoom_client()
    fake, _ = _tick_pages(
        [_tick_page(_tick_rows(2), "Y", _remaining_key(5, code="000660")), _tick_page(_tick_rows(2), "N", "")]
    )
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5))
    assert res["vendor_total_ticks"] is None


def test_kiwoom_tick_inconsistent_cursor_stops_uncertified() -> None:
    client = _kiwoom_client()
    fake, state = _tick_pages(
        [
            _tick_page(_tick_rows(3), "Y", _remaining_key(4)),
            _tick_page(_tick_rows(3), "Y", _remaining_key(3)),
            _tick_page(_tick_rows(1), "N", _remaining_key(0)),
        ]
    )
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5))
    assert state["calls"] == 2
    assert res["termination_reason"] == "cursor_inconsistent"
    assert res["truncated"] is True
    assert res["complete_by_total"] is False
    assert res["cursor"] is None


def test_kiwoom_tick_budget_exhaustion_returns_resumable_cursor_and_resume_continues_totals() -> None:
    client = _kiwoom_client()
    seen: list[tuple[str, str]] = []
    pages = [
        _tick_page(_tick_rows(3), "Y", _remaining_key(4)),
        _tick_page(_tick_rows(3), "Y", _remaining_key(1)),
        _tick_page(_tick_rows(1), "N", _remaining_key(0)),
    ]
    fake, _ = _tick_pages(pages)

    async def recording(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        seen.append((cont_yn, next_key))
        return await fake(session, api_id, path, body, cont_yn, next_key, max_retries)

    client._post_tr = recording
    first = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=1))
    assert first["truncated"] is True and first["termination_reason"] == "page_budget"
    cursor = first["cursor"]
    assert cursor.next_key == _remaining_key(4)
    assert cursor.rows_received == 3 and cursor.total_ticks == 7
    resumed = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5, resume=cursor))
    assert seen[1] == ("Y", _remaining_key(4))
    assert len(resumed["output2"]) == 4
    assert resumed["vendor_total_ticks"] == 7 and resumed["rows_received"] == 7
    assert resumed["complete_by_total"] is True and resumed["cursor"] is None


def test_kiwoom_tick_resume_with_stale_cursor_is_rejected_by_invariant() -> None:
    from src.api.kiwoom.client import TickCursor

    client = _kiwoom_client()
    fake, _ = _tick_pages([_tick_page(_tick_rows(3), "Y", _remaining_key(9))])
    client._post_tr = fake
    stale = TickCursor(next_key=_remaining_key(10), rows_received=3, total_ticks=7)
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", max_pages=5, resume=stale))
    assert res["termination_reason"] == "cursor_inconsistent"
    assert res["complete_by_total"] is False


def test_kiwoom_tick_remaining_parser_rejects_non_numeric_tail_and_empty_key() -> None:
    from src.api.kiwoom.client import _parse_tick_remaining

    assert _parse_tick_remaining(_remaining_key(12), "005930", _YMD) == 12
    assert _parse_tick_remaining(f"A005930{_YMD}12x", "005930", _YMD) is None
    assert _parse_tick_remaining(f"A005930{_YMD}", "005930", _YMD) is None
    assert _parse_tick_remaining("", "005930", _YMD) is None


def _tape_row(ymd: str, hms: str) -> dict:
    return {"cur_prc": "270000", "trde_qty": "1", "cntr_tm": f"{ymd}{hms}"}


def _tape_key(ymd: str, remaining: int, code: str = "005930") -> str:
    return f"A{code}{ymd}{remaining:08d}"


def _walk(client: KiwoomApiClient, pages: list, **kwargs: Any) -> tuple[dict, list]:
    fake, _ = _tick_pages(pages)
    client._post_tr = fake
    delivered: list = []
    res = asyncio.run(
        client.walk_tick_tape(object(), "005930", stop_before_day="2026-09-03", max_pages=10,
                              on_day_complete=lambda day, rows, cert: delivered.append((day, len(rows), cert)), **kwargs)
    )
    return res, delivered


def test_walk_tick_tape_certifies_dates_across_a_straddling_page() -> None:
    # 09-04: 3 rows (all on page 1 and 2); 09-03: 3 rows; vendor total counts one extra boundary tick (received + 1).
    pages = [
        _tick_page([_tape_row("20260904", "150003"), _tape_row("20260904", "150002")], "Y", _tape_key("20260904", 2)),
        _tick_page([_tape_row("20260904", "150001"), _tape_row("20260903", "150003")], "Y", _tape_key("20260903", 3)),
        _tick_page([_tape_row("20260903", "150002"), _tape_row("20260903", "150001")], "Y", _tape_key("20260903", 1)),
        _tick_page([_tape_row("20260902", "150001")], "N", ""),
    ]
    res, delivered = _walk(_kiwoom_client(), pages)
    assert res["termination_reason"] == "crossed_stop_day" or res["termination_reason"] == "tape_end"
    days = {d: (n, c.complete, c.vendor_total) for d, n, c in delivered}
    assert days["2026-09-04"] == (3, True, 4)
    assert days["2026-09-03"] == (3, True, 4)


def test_walk_tick_tape_short_date_is_not_delivered() -> None:
    pages = [
        _tick_page([_tape_row("20260904", "150003"), _tape_row("20260904", "150002")], "Y", _tape_key("20260904", 5)),
        _tick_page([_tape_row("20260903", "150001")], "Y", _tape_key("20260903", 0)),
        _tick_page([_tape_row("20260902", "150001")], "N", ""),
    ]
    res, delivered = _walk(_kiwoom_client(), pages)
    certs = {c.day: c for c in res["certificates"]}
    assert certs["2026-09-04"].complete is False and certs["2026-09-04"].vendor_total == 7
    assert all(day != "2026-09-04" for day, _, _ in delivered)


def test_walk_tick_tape_nxt_keys_carry_the_nx_suffix_and_certify() -> None:
    client = _kiwoom_client()
    seen: list[str] = []
    pages = [
        _tick_page([_tape_row("20260904", "150002"), _tape_row("20260904", "150001")], "Y", _tape_key("20260904", 1, code="005930_NX")),
        _tick_page([_tape_row("20260903", "150001")], "N", ""),
    ]
    fake, _ = _tick_pages(pages)

    async def recording(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        seen.append(body["stk_cd"])
        return await fake(session, api_id, path, body, cont_yn, next_key, max_retries)

    client._post_tr = recording
    delivered: list = []
    res = asyncio.run(
        client.walk_tick_tape(object(), "005930", venue="NXT", stop_before_day="2026-09-04", max_pages=5,
                              on_day_complete=lambda day, rows, cert: delivered.append((day, cert)))
    )
    assert seen[0] == "005930_NX"
    certs = {c.day: c for c in res["certificates"]}
    assert certs["2026-09-04"].complete is True and certs["2026-09-04"].vendor_total == 3


def test_walk_tick_tape_inconsistent_remaining_stops_and_certifies_nothing_after() -> None:
    pages = [
        _tick_page([_tape_row("20260904", "150003"), _tape_row("20260904", "150002")], "Y", _tape_key("20260904", 3)),
        _tick_page([_tape_row("20260904", "150001")], "Y", _tape_key("20260904", 9)),
        _tick_page([_tape_row("20260903", "150001")], "N", ""),
    ]
    res, delivered = _walk(_kiwoom_client(), pages)
    assert res["termination_reason"] == "cursor_inconsistent" and res["truncated"] is True
    assert delivered == []


def test_walk_tick_tape_certifies_a_low_volume_date_by_bracketing_when_no_key_carries_it() -> None:
    # 09-05 and 09-04 never appear as a page's oldest-row date, so no vendor total exists for them;
    # they are certified by being bracketed by an older date inside the unbroken cursor chain.
    pages = [
        _tick_page(
            [_tape_row("20260905", "150002"), _tape_row("20260905", "150001"), _tape_row("20260904", "150001"), _tape_row("20260903", "150002")],
            "Y", _tape_key("20260903", 2),
        ),
        _tick_page([_tape_row("20260903", "150001")], "Y", _tape_key("20260903", 1)),
        _tick_page([_tape_row("20260902", "150001")], "N", ""),
    ]
    fake, _ = _tick_pages(pages)
    client = _kiwoom_client()
    client._post_tr = fake
    delivered: list = []
    res = asyncio.run(
        client.walk_tick_tape(object(), "005930", stop_before_day="2026-09-03", max_pages=5,
                              on_day_complete=lambda day, rows, cert: delivered.append((day, len(rows), cert.basis)))
    )
    by_day = {d: (n, basis) for d, n, basis in delivered}
    assert by_day["2026-09-05"] == (2, "bracketed")
    assert by_day["2026-09-04"] == (1, "bracketed")
    assert by_day["2026-09-03"][1] == "vendor_total"
    assert res["termination_reason"] == "tape_end"


def test_walk_tick_tape_oldest_date_at_tape_end_needs_its_vendor_total() -> None:
    # The oldest date on the tape may have been cut by vendor retention: bracketing from below is impossible,
    # so without a vendor total it stays uncertified.
    pages = [
        _tick_page([_tape_row("20260905", "150002"), _tape_row("20260904", "150001")], "Y", _tape_key("20260904", 0)),
        _tick_page([_tape_row("20260903", "150001")], "N", ""),
    ]
    fake, _ = _tick_pages(pages)
    client = _kiwoom_client()
    client._post_tr = fake
    delivered: list = []
    res = asyncio.run(
        client.walk_tick_tape(object(), "005930", stop_before_day="2026-09-03", max_pages=5,
                              on_day_complete=lambda day, rows, cert: delivered.append(day))
    )
    certs = {c.day: c for c in res["certificates"]}
    assert certs["2026-09-03"].complete is False
    assert "2026-09-03" not in delivered
    assert "2026-09-05" in delivered


def test_kiwoom_reset_token_forces_reissue() -> None:
    client = _kiwoom_client()
    client.token = "stale"
    client.reset_token()
    assert client.token is None


def test_kiwoom_tape_vendor_failure_logs_return_msg(caplog) -> None:
    client = _kiwoom_client()

    async def fake_post_tr(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        return ({"return_code": 3, "return_msg": "token invalid"}, {"cont-yn": "N"})

    client._post_tr = fake_post_tr
    with caplog.at_level("WARNING"):
        res = asyncio.run(client.walk_tick_tape(object(), "005930", stop_before_day="2026-09-03", max_pages=3))
    assert res["termination_reason"] == "vendor_failure"
    assert "token invalid" in caplog.text


def _scripted(script: list[Any]) -> tuple[KiwoomApiClient, list[tuple[str, str, dict]]]:
    client = _kiwoom_client()
    calls: list[tuple[str, str, dict]] = []

    async def fake_post_tr(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> Any:
        calls.append((cont_yn, next_key, dict(body)))
        assert len(calls) <= len(script), "unexpected extra request"
        item = script[len(calls) - 1]
        if isinstance(item, BaseException):
            raise item
        return item

    client._post_tr = fake_post_tr
    return client, calls


def _transient_page(code: int = 7) -> tuple[dict, dict]:
    return ({"return_code": code, "return_msg": "[1631] 조회 실패"}, {"cont-yn": "N", "next-key": ""})


def _two_day_pages() -> list[tuple[dict, dict]]:
    return [
        _tick_page([_tape_row("20260904", "150003"), _tape_row("20260904", "150002")], "Y", _tape_key("20260904", 2)),
        _tick_page([_tape_row("20260904", "150001"), _tape_row("20260903", "150003")], "Y", _tape_key("20260903", 3)),
        _tick_page([_tape_row("20260903", "150002"), _tape_row("20260903", "150001")], "Y", _tape_key("20260903", 1)),
        _tick_page([_tape_row("20260902", "150001")], "N", ""),
    ]


def _run_tape(client: KiwoomApiClient, **kwargs: Any) -> tuple[dict, list[tuple[str, list[dict], Any]]]:
    delivered: list[tuple[str, list[dict], Any]] = []
    if "budget" not in kwargs:
        kwargs.setdefault("max_pages", 10)
    res = asyncio.run(
        client.walk_tick_tape(
            object(), "005930", stop_before_day="2026-09-03",
            on_day_complete=lambda day, rows, cert: delivered.append((day, rows, cert)), **kwargs,
        )
    )
    return res, delivered


@pytest.mark.parametrize("position", [1, 2, 3])
def test_walk_tick_tape_retry_matches_uninterrupted_walk(position: int) -> None:
    clean = _two_day_pages()
    clean_client, _ = _scripted(clean)
    clean_res, clean_delivered = _run_tape(clean_client)
    client, _ = _scripted([*clean[:position], _transient_page(), *clean[position:]])
    res, delivered = _run_tape(client, page_retries=3, transient_return_codes=(7,))
    assert delivered == clean_delivered
    assert res["certificates"] == clean_res["certificates"]
    assert res["termination_reason"] == clean_res["termination_reason"]
    assert res["page_retries_used"] == 1


def test_walk_tick_tape_retry_resends_the_identical_request() -> None:
    clean = _two_day_pages()
    client, calls = _scripted([clean[0], clean[1], _transient_page(), clean[2], clean[3]])
    _run_tape(client, page_retries=3, transient_return_codes=(7,))
    assert calls[2] == calls[3]
    assert calls[2][:2] == ("Y", _tape_key("20260903", 3))


def test_walk_tick_tape_exhausted_retries_report_vendor_failure_with_details() -> None:
    first = _tick_page(
        [_tape_row("20260905", "150002"), _tape_row("20260905", "150001"), _tape_row("20260904", "150001")],
        "Y", _tape_key("20260904", 1),
    )
    client, calls = _scripted([first, _transient_page(), _transient_page(), _transient_page(), _transient_page()])
    res, delivered = _run_tape(client, page_retries=3, transient_return_codes=(7,))
    assert len(calls) == 5
    assert res["termination_reason"] == "vendor_failure"
    assert res["vendor_return_code"] == 7 and res["transport_error"] is None
    assert res["page_retries_used"] == 3
    assert res["failure_request_key"] == _tape_key("20260904", 1)
    assert res["failure_remaining"] == 1
    assert res["msg1"] == "[1631] 조회 실패"
    assert [day for day, _, _ in delivered] == ["2026-09-05"]
    pending = [c for c in res["certificates"] if not c.complete]
    assert [(c.day, c.basis) for c in pending] == [("2026-09-04", "walk_truncated")]


def test_walk_tick_tape_non_transient_code_is_not_retried() -> None:
    client, calls = _scripted([_transient_page(3)])
    res, _ = _run_tape(client, page_retries=3, transient_return_codes=(7,))
    assert len(calls) == 1
    assert res["termination_reason"] == "vendor_failure"
    assert res["vendor_return_code"] == 3
    assert res["page_retries_used"] == 0


def test_walk_tick_tape_defaults_reproduce_single_request_failure() -> None:
    client, calls = _scripted([_transient_page(7)])
    res, _ = _run_tape(client)
    assert len(calls) == 1
    assert res["termination_reason"] == "vendor_failure"
    assert res["page_retries_used"] == 0


def test_walk_tick_tape_retry_budget_is_per_walk_not_per_page() -> None:
    first = _tick_page([_tape_row("20260905", "150001")], "Y", _tape_key("20260905", 3))
    second = _tick_page([_tape_row("20260905", "150002")], "Y", _tape_key("20260905", 2))
    client, calls = _scripted([first, _transient_page(), second, _transient_page()])
    res, _ = _run_tape(client, page_retries=1, transient_return_codes=(7,))
    assert len(calls) == 4
    assert res["termination_reason"] == "vendor_failure"
    assert res["page_retries_used"] == 1


def test_walk_tick_tape_transport_error_is_retried_in_place() -> None:
    import aiohttp

    clean = _two_day_pages()
    clean_client, _ = _scripted(clean)
    clean_res, clean_delivered = _run_tape(clean_client)
    client, _ = _scripted([clean[0], aiohttp.ClientConnectionError(), clean[1], clean[2], clean[3]])
    res, delivered = _run_tape(client, page_retries=3, transient_return_codes=(7,))
    assert delivered == clean_delivered
    assert res["certificates"] == clean_res["certificates"]
    assert res["termination_reason"] == clean_res["termination_reason"]
    assert res["page_retries_used"] == 1


def test_walk_tick_tape_exhausted_transport_error_reports_transport_name() -> None:
    import aiohttp

    client, calls = _scripted([aiohttp.ClientConnectionError() for _ in range(4)])
    res, _ = _run_tape(client, page_retries=3, transient_return_codes=(7,))
    assert len(calls) == 4
    assert res["termination_reason"] == "vendor_failure"
    assert res["transport_error"] == "ClientConnectionError"
    assert res["vendor_return_code"] is None
    assert res["page_retries_used"] == 3


def test_walk_tick_tape_timeout_without_deadline_is_vendor_failure() -> None:
    client, _ = _scripted([TimeoutError()])
    res, _ = _run_tape(client)
    assert res["termination_reason"] == "vendor_failure"
    assert res["transport_error"] == "TimeoutError"


def test_walk_tick_tape_retry_skipped_when_deadline_is_closer_than_backoff(monkeypatch) -> None:
    sleeps: list[float] = []

    async def _record(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("src.api.kiwoom.client.asyncio.sleep", _record)
    client, calls = _scripted([_transient_page(7), _transient_page(7)])
    budget = _budget(max_pages=10, deadline=datetime.now(ZoneInfo("Asia/Seoul")) + timedelta(seconds=0.5))
    res, _ = _run_tape(client, budget=budget, page_retries=3, transient_return_codes=(7,), retry_backoff_seconds=10.0)
    assert sleeps == []
    assert len(calls) == 1
    assert res["termination_reason"] == "vendor_failure"


def test_walk_tick_tape_backoff_doubles_per_retry(monkeypatch) -> None:
    sleeps: list[float] = []

    async def _record(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("src.api.kiwoom.client.asyncio.sleep", _record)
    client, calls = _scripted([_transient_page(7) for _ in range(4)])
    _run_tape(client, page_retries=3, transient_return_codes=(7,), retry_backoff_seconds=3.0)
    assert len(calls) == 4
    assert sleeps == [3.0, 6.0, 12.0]


def test_walk_tick_tape_observer_receives_page_attempt_and_failed_body() -> None:
    seen: list[tuple] = []
    clean = _two_day_pages()
    client, _ = _scripted([clean[0], _transient_page(), clean[1], clean[2], clean[3]])
    _run_tape(client, page_retries=3, transient_return_codes=(7,), on_page=lambda *a: seen.append(a))
    assert [(a[4], a[5], a[0].get("return_code")) for a in seen] == [(0, 0, 0), (1, 0, 7), (1, 1, 0), (2, 0, 0), (3, 0, 0)]


def test_walk_tick_tape_retried_page_with_inconsistent_total_still_stops() -> None:
    first = _tick_page([_tape_row("20260904", "150003"), _tape_row("20260904", "150002")], "Y", _tape_key("20260904", 3))
    retried = _tick_page([_tape_row("20260904", "150001")], "Y", _tape_key("20260904", 9))
    client, _ = _scripted([first, _transient_page(), retried])
    res, delivered = _run_tape(client, page_retries=1, transient_return_codes=(7,))
    assert res["termination_reason"] == "cursor_inconsistent"
    assert delivered == []
    assert not any(c.complete for c in res["certificates"])


def test_walk_tick_tape_retry_and_failure_logs_carry_identifiers(caplog) -> None:
    client, _ = _scripted([_transient_page(7), _transient_page(7)])
    with caplog.at_level("WARNING"):
        _run_tape(client, page_retries=1, transient_return_codes=(7,))
    retries = [m for m in caplog.messages if "status=PAGE_RETRY" in m]
    failures = [m for m in caplog.messages if "status=VENDOR_FAILURE" in m]
    assert len(retries) == 1 and len(failures) == 1
    assert "rc=7" in retries[0] and "retry=1/1" in retries[0]
    for field in ("code=005930", "venue=KRX", "rc=7", "retries=1", "request_key=-", "return_msg=[1631] 조회 실패"):
        assert field in failures[0]


def test_walk_tick_tape_parse_bug_is_logged_with_traceback(caplog) -> None:
    client, _ = _scripted([(None, {})])
    with caplog.at_level("WARNING"):
        res, _ = _run_tape(client)
    assert res["rt_cd"] == "1" and res["error_type"] == "AttributeError"
    records = [r for r in caplog.records if "status=ERROR" in r.getMessage()]
    assert records and records[0].exc_info is not None


@pytest.mark.parametrize(
    "kwargs",
    [{"page_retries": -1}, {"retry_backoff_seconds": -1.0}, {"retry_backoff_seconds": float("inf")}, {"retry_backoff_seconds": float("nan")}],
)
def test_walk_tick_tape_rejects_invalid_retry_bounds_before_any_request(kwargs) -> None:
    client, calls = _scripted([])
    with pytest.raises(ValueError, match="invalid tape retry bounds"):
        _run_tape(client, **kwargs)
    assert calls == []


class _FakeResp:
    def __init__(self, body: dict, status: int = 200) -> None:
        self._body = body
        self.status = status
        self.headers = {"cont-yn": "N", "next-key": ""}

    async def json(self) -> dict:
        return self._body

    async def __aenter__(self) -> _FakeResp:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _FakeKiwoomSession:
    """Scripted TR responses plus a token endpoint that issues tok-1, tok-2, ..."""

    def __init__(self, tr_responses: list[tuple[dict, int]]) -> None:
        self.tr_responses = list(tr_responses)
        self.tr_calls: list[dict[str, Any]] = []
        self.token_calls = 0

    def post(self, url: str, headers: dict, json: dict) -> _FakeResp:
        if url.endswith("/oauth2/token"):
            self.token_calls += 1
            return _FakeResp({"token": f"tok-{self.token_calls}", "return_code": 0, "expires_dt": "20261002151006"})
        self.tr_calls.append({"auth": headers["authorization"], "cont": headers["cont-yn"], "key": headers["next-key"], "body": dict(json)})
        body, status = self.tr_responses.pop(0)
        return _FakeResp(body, status)


_EXPIRED = ({"return_code": 3, "return_msg": "인증에 실패했습니다[8005:Token이 유효하지 않습니다]"}, 200)
_OK = ({"return_code": 0, "return_msg": "OK"}, 200)


def test_kiwoom_post_tr_refreshes_expired_token_once_and_replays_same_request() -> None:
    client = _kiwoom_client()
    session = _FakeKiwoomSession([_EXPIRED, _OK])
    data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {"stk_cd": "005930"}, cont_yn="Y", next_key="k9"))
    assert data["return_code"] == 0
    assert session.token_calls == 1
    assert [c["auth"] for c in session.tr_calls] == ["Bearer tok", "Bearer tok-1"]
    assert session.tr_calls[0]["body"] == session.tr_calls[1]["body"] == {"stk_cd": "005930"}
    assert {(c["cont"], c["key"]) for c in session.tr_calls} == {("Y", "k9")}


def test_kiwoom_post_tr_second_auth_rejection_is_returned_without_looping() -> None:
    client = _kiwoom_client()
    session = _FakeKiwoomSession([_EXPIRED, _EXPIRED, _OK])
    data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))
    assert data["return_code"] == 3
    assert len(session.tr_calls) == 2
    assert session.token_calls == 1


def test_kiwoom_post_tr_non_auth_vendor_error_is_not_refreshed() -> None:
    client = _kiwoom_client()
    for body in ({"return_code": 3, "return_msg": "other auth problem"}, {"return_code": 5, "return_msg": "8005 lookalike"}):
        session = _FakeKiwoomSession([(body, 200)])
        data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))
        assert data == body
        assert len(session.tr_calls) == 1
        assert session.token_calls == 0


def test_kiwoom_post_tr_refresh_does_not_consume_rate_limit_retries(monkeypatch) -> None:
    async def _no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("src.api.kiwoom.client.asyncio.sleep", _no_sleep)
    client = _kiwoom_client()
    session = _FakeKiwoomSession([({}, 429), _EXPIRED, _OK])
    data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}, max_retries=3))
    assert data["return_code"] == 0
    assert session.token_calls == 1


def test_kiwoom_walk_classifies_empty_nx_tape() -> None:
    client = _kiwoom_client()
    fake, state = _tick_pages([_tick_page([{"cur_prc": "", "trde_qty": "", "cntr_tm": ""}], "N", "")])
    client._post_tr = fake
    seen: list[Any] = []
    res = asyncio.run(client.walk_tick_tape(object(), "031980", venue="NXT", stop_before_day="2026-09-21", max_pages=5, on_page=lambda *a: seen.append(a)))
    assert res["termination_reason"] == "tape_empty"
    assert res["truncated"] is False
    assert res["certificates"] == []
    assert len(seen) == 1 and state["calls"] == 1


def test_kiwoom_walk_malformed_later_page_stays_cursor_inconsistent() -> None:
    client = _kiwoom_client()
    fake, _ = _tick_pages(
        [
            _tick_page([{"cur_prc": "1", "trde_qty": "1", "cntr_tm": "20260930153000"}], "Y", "A005930_NX2026093010"),
            _tick_page([{"cur_prc": "", "trde_qty": "", "cntr_tm": ""}], "N", ""),
        ]
    )
    client._post_tr = fake
    res = asyncio.run(client.walk_tick_tape(object(), "005930", stop_before_day="2026-09-21", max_pages=5))
    assert res["termination_reason"] == "cursor_inconsistent"


def test_kiwoom_walk_empty_first_page_with_continuation_stays_cursor_inconsistent() -> None:
    client = _kiwoom_client()
    fake, _ = _tick_pages([_tick_page([{"cur_prc": "", "trde_qty": "", "cntr_tm": ""}], "Y", "k1")])
    client._post_tr = fake
    res = asyncio.run(client.walk_tick_tape(object(), "005930", stop_before_day="2026-09-21", max_pages=1))
    assert res["termination_reason"] == "cursor_inconsistent"


def test_kiwoom_per_tr_host_bucket_shared_across_clients(tmp_path, monkeypatch) -> None:
    from src.api.kis.rate_limit import host_admission_state_path
    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_DIR", tmp_path)
    (tmp_path / ".host-admission").touch()
    monkeypatch.setattr(settings_instance, "BROKER_ADMISSION_REQUIRE_SHARED", "always")

    from src.api.kiwoom.client import KiwoomApiClient

    a = KiwoomApiClient(app_key="same-key", secret_key="s")
    b = KiwoomApiClient(app_key="same-key", secret_key="s")
    assert a._limiter_for("ka10079")._state_path == b._limiter_for("ka10079")._state_path
    assert a._limiter_for("ka10079")._state_path == host_admission_state_path("kiwoom", "same-key", "ka10079")
    assert a._limiter_for("ka10080")._state_path != a._limiter_for("ka10079")._state_path


def _token_session(body: dict, seen: dict) -> Any:
    class _Resp:
        status = 200

        async def json(self) -> dict:
            return dict(body)

        async def __aenter__(self) -> _Resp:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    class _Session:
        def post(self, url: str, headers: dict, json: dict) -> _Resp:
            seen["url"] = url
            seen["headers"] = dict(headers)
            seen["json"] = dict(json)
            return _Resp()

    return _Session()


def test_kiwoom_issue_token_returns_vendor_expiry() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    seen: dict = {}
    session = _token_session({"token": "live-tok", "return_code": 0, "expires_dt": "20261002151006"}, seen)
    issued = asyncio.run(KiwoomApiClient(app_key="k", secret_key="s").issue_token(session))

    assert issued.token == "live-tok"
    assert issued.expires_at == datetime(2026, 10, 2, 15, 10, 6, tzinfo=_SEOUL)
    assert seen["url"].endswith("/oauth2/token")
    assert seen["headers"]["User-Agent"] == "curl/8.5.0"


def test_kiwoom_issue_token_rejects_malformed_expiry() -> None:
    import asyncio

    import pytest

    from src.api.kiwoom.client import KiwoomApiClient

    for body in (
        {"token": "t", "return_code": 0},
        {"token": "t", "return_code": 0, "expires_dt": "2026-10-02"},
        {"token": "t", "return_code": 0, "expires_dt": "20261301151006"},
    ):
        with pytest.raises(RuntimeError):
            asyncio.run(KiwoomApiClient(app_key="k", secret_key="s").issue_token(_token_session(body, {})))


def test_kiwoom_ensure_token_delegates_to_issue_token() -> None:
    import asyncio

    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    session = _token_session({"token": "live-tok", "return_code": 0, "expires_dt": "20261002151006"}, {})
    assert asyncio.run(client.ensure_token(session)) == "live-tok"
    assert client.token == "live-tok"


def _revoke_session(body: dict, status: int, seen: dict) -> Any:
    class _Resp:
        def __init__(self) -> None:
            self.status = status

        async def json(self) -> dict:
            return dict(body)

        async def __aenter__(self) -> _Resp:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    class _Session:
        def post(self, url: str, headers: dict, json: dict) -> _Resp:
            seen["url"] = url
            seen["headers"] = dict(headers)
            seen["json"] = dict(json)
            return _Resp()

    return _Session()


def test_kiwoom_revoke_token_uses_au10002_contract() -> None:
    import asyncio

    from src.api.kiwoom.client import KIWOOM_REVOKE_PATH, KiwoomApiClient

    seen: dict = {}
    session = _revoke_session({"return_code": 0, "return_msg": "OK"}, 200, seen)
    asyncio.run(KiwoomApiClient(app_key="k", secret_key="s").revoke_token(session, "live-tok"))

    assert seen["url"].endswith(KIWOOM_REVOKE_PATH)
    assert seen["json"] == {"appkey": "k", "secretkey": "s", "token": "live-tok"}
    assert seen["headers"]["User-Agent"] == "curl/8.5.0"


def test_kiwoom_revoke_token_failure_surfaces() -> None:
    import asyncio

    import pytest

    from src.api.kiwoom.client import KiwoomApiClient

    with pytest.raises(RuntimeError):
        asyncio.run(
            KiwoomApiClient(app_key="k", secret_key="s").revoke_token(
                _revoke_session({"return_code": 1, "return_msg": "bad"}, 200, {}), "live-tok"
            )
        )
    with pytest.raises(RuntimeError):
        asyncio.run(
            KiwoomApiClient(app_key="k", secret_key="s").revoke_token(
                _revoke_session({"return_code": 0}, 500, {}), "live-tok"
            )
        )


def test_kiwoom_post_tr_converts_mapping_headers() -> None:
    import asyncio
    from typing import Any

    from multidict import CIMultiDict, CIMultiDictProxy

    from src.api.kiwoom.client import KiwoomApiClient

    class _Resp:
        def __init__(self) -> None:
            self.status = 200
            self.headers = CIMultiDictProxy(CIMultiDict({"cont-yn": "Y", "next-key": "k1"}))

        async def json(self) -> dict:
            return {"return_code": 0, "return_msg": "OK"}

        async def __aenter__(self) -> _Resp:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    class _Session:
        def post(self, url: str, headers: dict, json: dict) -> _Resp:
            return _Resp()

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"
    data, headers = asyncio.run(client._post_tr(_Session(), "ka10079", "/api/dostk/chart", {"stk_cd": "005930"}))

    assert type(headers) is dict
    assert headers.get("cont-yn") == "Y"
    assert headers.get("next-key") == "k1"
    assert data["return_code"] == 0


def test_kiwoom_post_tr_unconvertible_headers_become_empty() -> None:
    import asyncio
    from typing import Any

    from src.api.kiwoom.client import KiwoomApiClient

    body = {"return_code": 0, "return_msg": "OK"}

    class _BadHeaders:
        def items(self):  # noqa: ANN204 - presence of the attribute is the fixture
            return [("a", "1")]

        def __iter__(self):  # noqa: ANN204 - dict() must raise TypeError here
            raise TypeError("not iterable")

    class _Resp:
        def __init__(self) -> None:
            self.status = 200
            self.headers = _BadHeaders()

        async def json(self) -> dict:
            return dict(body)

        async def __aenter__(self) -> _Resp:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    class _Session:
        def post(self, url: str, headers: dict, json: dict) -> _Resp:
            return _Resp()

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"
    data, headers = asyncio.run(client._post_tr(_Session(), "ka10079", "/api/dostk/chart", {"stk_cd": "005930"}))

    assert headers == {}
    assert data == body


_KIWOOM_TOKEN_BODY = {"token": "tok-1", "return_code": 0, "expires_dt": "20261002151006"}
_KIWOOM_EXP_BODY = {"return_code": 3, "return_msg": "인증에 실패했습니다[8005:Token이 유효하지 않습니다]"}


def _kiwoom_seeded_client():  # type: ignore[no-untyped-def]
    from src.api.kiwoom.client import KiwoomApiClient

    client = KiwoomApiClient(app_key="k", secret_key="s")
    client.token = "tok"
    return client


def _kiwoom_tr_session(tr_replies, token_bodies=(_KIWOOM_TOKEN_BODY,)):  # type: ignore[no-untyped-def]
    from tests.broker_fakes import scripted_session

    return scripted_session(list(tr_replies), token_bodies=list(token_bodies))


def _kiwoom_429():  # type: ignore[no-untyped-def]
    from tests.broker_fakes import FakeBrokerResponse

    return FakeBrokerResponse(
        body={"return_code": 5, "return_msg": "허용된 API 요청 개수를 초과하였습니다. 유량=5"}, status=429
    )


def _kiwoom_ok():  # type: ignore[no-untyped-def]
    from tests.broker_fakes import FakeBrokerResponse

    return FakeBrokerResponse(body={"return_code": 0, "return_msg": "OK"}, status=200)


def _kiwoom_exp():  # type: ignore[no-untyped-def]
    from tests.broker_fakes import FakeBrokerResponse

    return FakeBrokerResponse(body=dict(_KIWOOM_EXP_BODY), status=200)


def _record_sleep(monkeypatch):  # type: ignore[no-untyped-def]
    import asyncio

    sleeps: list[float] = []

    async def _record(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _record)
    return sleeps


def _count_acquires(monkeypatch):  # type: ignore[no-untyped-def]
    import src.api.kiwoom.client as kiwoom_client_mod

    calls = {"n": 0}

    async def _count(self) -> None:
        calls["n"] += 1

    monkeypatch.setattr(kiwoom_client_mod.HostPacedRateLimiter, "acquire", _count)
    return calls


def _tr_auths(session) -> list[str]:  # type: ignore[no-untyped-def]
    return [
        request.headers.get("authorization", "")
        for request in session.requests
        if not request.url.endswith("/oauth2/token")
    ]


def test_kiwoom_tr_429_retried_then_succeeds(monkeypatch) -> None:
    import asyncio

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([_kiwoom_429(), _kiwoom_429(), _kiwoom_ok()])
    sleeps = _record_sleep(monkeypatch)

    data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {"stk_cd": "005930"}))

    assert data["return_code"] == 0
    assert len(session.requests) == 3
    assert sleeps == [1.2, 1.2]


def test_kiwoom_tr_429_exhausted_returns_last_body(monkeypatch, caplog) -> None:
    import asyncio
    import logging

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([_kiwoom_429(), _kiwoom_429(), _kiwoom_429()])
    sleeps = _record_sleep(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="src.api._transport"):
        data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {"stk_cd": "005930"}))

    assert data["return_code"] == 5
    assert len(session.requests) == 3
    assert sleeps == [1.2, 1.2]
    assert sum("RATE_LIMITED attempts=3" in r.getMessage() for r in caplog.records) == 1


def test_kiwoom_tr_expired_token_refreshes_once_and_replays(monkeypatch) -> None:
    import asyncio

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([_kiwoom_exp(), _kiwoom_ok()])
    sleeps = _record_sleep(monkeypatch)

    data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {"stk_cd": "005930"}))

    assert data["return_code"] == 0
    assert len(session.requests_to("/api/dostk/chart")) == 2
    assert len(session.requests_to("/oauth2/token")) == 1
    assert _tr_auths(session) == ["Bearer tok", "Bearer tok-1"]
    assert sleeps == []


def test_kiwoom_tr_second_rejection_surfaces(monkeypatch) -> None:
    import asyncio

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([_kiwoom_exp(), _kiwoom_exp(), _kiwoom_ok()])

    data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))

    assert data["return_code"] == 3
    assert len(session.requests_to("/api/dostk/chart")) == 2
    assert len(session.requests_to("/oauth2/token")) == 1


def test_kiwoom_tr_call_cap_is_max_retries_plus_one(monkeypatch) -> None:
    import asyncio

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([_kiwoom_429(), _kiwoom_429(), _kiwoom_exp(), _kiwoom_429()])
    sleeps = _record_sleep(monkeypatch)

    data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))

    assert data["return_code"] == 5
    assert len(session.requests_to("/api/dostk/chart")) == 4
    assert sleeps == [1.2, 1.2]
    assert len(session.requests_to("/oauth2/token")) == 1


def test_kiwoom_tr_replay_rate_limit_continues_retry_budget(monkeypatch) -> None:
    import asyncio

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([_kiwoom_exp(), _kiwoom_429(), _kiwoom_429(), _kiwoom_ok()])
    sleeps = _record_sleep(monkeypatch)

    data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))

    assert data["return_code"] == 0
    assert len(session.requests_to("/api/dostk/chart")) == 4
    assert sleeps == [1.2, 1.2]


def test_kiwoom_tr_http_401_is_not_auth_rejection() -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([FakeBrokerResponse(body={"return_code": 1}, status=401)])

    data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))

    assert data == {"return_code": 1}
    assert len(session.requests_to("/api/dostk/chart")) == 1
    assert session.requests_to("/oauth2/token") == []


def test_kiwoom_tr_acquires_before_every_send(monkeypatch) -> None:
    import asyncio

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([_kiwoom_429(), _kiwoom_exp(), _kiwoom_ok()])
    acquires = _count_acquires(monkeypatch)
    _record_sleep(monkeypatch)

    asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))

    assert acquires["n"] == len(session.requests_to("/api/dostk/chart")) == 3


def test_kiwoom_tr_transport_error_propagates(monkeypatch) -> None:
    import asyncio

    import aiohttp

    from tests.broker_fakes import FakeBrokerResponse

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([FakeBrokerResponse(enter_error=aiohttp.ClientConnectionError("boom"))])
    sleeps = _record_sleep(monkeypatch)

    with pytest.raises(aiohttp.ClientConnectionError):
        asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))

    assert len(session.requests_to("/api/dostk/chart")) == 1
    assert sleeps == []


def test_kiwoom_tr_html_429_raises_content_type_error(monkeypatch) -> None:
    import asyncio
    from unittest.mock import Mock

    import aiohttp

    from tests.broker_fakes import FakeBrokerResponse

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session(
        [FakeBrokerResponse(status=429, json_error=aiohttp.ContentTypeError(Mock(), (), message="no json"))]
    )
    sleeps = _record_sleep(monkeypatch)

    with pytest.raises(aiohttp.ContentTypeError):
        asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))

    assert len(session.requests_to("/api/dostk/chart")) == 1
    assert sleeps == []


def test_kiwoom_tr_retry_settings_honored(monkeypatch) -> None:
    import asyncio

    from src.config import settings as settings_instance

    monkeypatch.setattr(settings_instance, "KIWOOM_RATE_LIMIT_MAX_RETRIES", 2)
    monkeypatch.setattr(settings_instance, "KIWOOM_RATE_LIMIT_BACKOFF_SECONDS", 0.5)
    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([_kiwoom_429(), _kiwoom_429(), _kiwoom_429()])
    sleeps = _record_sleep(monkeypatch)

    asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))

    assert len(session.requests_to("/api/dostk/chart")) == 2
    assert sleeps == [0.5]


def test_kiwoom_tr_explicit_max_retries_wins(monkeypatch) -> None:
    import asyncio

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([_kiwoom_429(), _kiwoom_ok()])
    sleeps = _record_sleep(monkeypatch)

    data, _ = asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}, max_retries=1))

    assert data["return_code"] == 5
    assert len(session.requests_to("/api/dostk/chart")) == 1
    assert sleeps == []


def test_kiwoom_tr_zero_max_retries_rejected() -> None:
    import asyncio

    client = _kiwoom_seeded_client()
    session = _kiwoom_tr_session([_kiwoom_ok()])

    with pytest.raises(ValueError, match="max_attempts"):
        asyncio.run(client._post_tr(session, "ka10079", "/api/dostk/chart", {}, max_retries=0))

    assert session.requests == []


def test_kiwoom_tr_concurrent_rejections_issue_once(monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerSession

    client = _kiwoom_seeded_client()
    _count_acquires(monkeypatch)

    def _respond(request):  # type: ignore[no-untyped-def]
        from tests.broker_fakes import FakeBrokerResponse

        if request.url.endswith("/oauth2/token"):
            return FakeBrokerResponse(body=dict(_KIWOOM_TOKEN_BODY))
        if request.headers.get("authorization") != "Bearer tok":
            return _kiwoom_ok()
        return FakeBrokerResponse(body=dict(_KIWOOM_EXP_BODY), enter_delay=0.01)

    session = FakeBrokerSession(_respond)

    async def _main():  # type: ignore[no-untyped-def]
        return await asyncio.gather(
            client._post_tr(session, "ka10079", "/api/dostk/chart", {}),
            client._post_tr(session, "ka10079", "/api/dostk/chart", {}),
        )

    (data_a, _), (data_b, _) = asyncio.run(_main())

    assert data_a["return_code"] == 0
    assert data_b["return_code"] == 0
    assert len(session.requests_to("/oauth2/token")) == 1
    assert sorted(_tr_auths(session)) == ["Bearer tok", "Bearer tok", "Bearer tok-1", "Bearer tok-1"]


def test_kiwoom_tr_refresh_never_sends_bearer_none(monkeypatch) -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse, FakeBrokerSession

    client = _kiwoom_seeded_client()
    _count_acquires(monkeypatch)
    issuance_started = asyncio.Event()
    release_issuance = asyncio.Event()

    def _respond(request):  # type: ignore[no-untyped-def]
        if request.url.endswith("/oauth2/token"):
            issuance_started.set()
            return FakeBrokerResponse(body=dict(_KIWOOM_TOKEN_BODY), enter_gate=release_issuance)
        auth = request.headers.get("authorization", "")
        if auth == "Bearer tok":
            return _kiwoom_exp()
        return _kiwoom_ok()

    session = FakeBrokerSession(_respond)

    async def _main():  # type: ignore[no-untyped-def]
        task_a = asyncio.create_task(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))
        await issuance_started.wait()
        task_b = asyncio.create_task(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))
        for _ in range(1000):
            if len([r for r in session.requests if r.url.endswith("/api/dostk/chart")]) >= 2:
                break
            await asyncio.sleep(0.001)
        in_flight_auths = _tr_auths(session)
        assert "Bearer None" not in in_flight_auths
        assert in_flight_auths == ["Bearer tok", "Bearer tok"]
        release_issuance.set()
        return await asyncio.gather(task_a, task_b)

    (data_a, _), (data_b, _) = asyncio.run(_main())

    assert data_a["return_code"] == 0
    assert data_b["return_code"] == 0
    assert _tr_auths(session) == ["Bearer tok", "Bearer tok", "Bearer tok-1", "Bearer tok-1"]


def test_kiwoom_tr_cancelled_refresh_keeps_previous_token() -> None:
    import asyncio

    from tests.broker_fakes import FakeBrokerResponse, FakeBrokerSession

    client = _kiwoom_seeded_client()
    issuance_started = asyncio.Event()
    release_issuance = asyncio.Event()

    def _respond(request):  # type: ignore[no-untyped-def]
        if request.url.endswith("/oauth2/token"):
            issuance_started.set()
            return FakeBrokerResponse(body=dict(_KIWOOM_TOKEN_BODY), enter_gate=release_issuance)
        return _kiwoom_exp()

    session = FakeBrokerSession(_respond)

    async def _main() -> None:
        task = asyncio.create_task(client._post_tr(session, "ka10079", "/api/dostk/chart", {}))
        await issuance_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_main())
    assert client.token == "tok"


def test_kiwoom_nxt_chart_methods_keep_positional_session_code_date_order() -> None:
    client = _kiwoom_client()
    seen: list[tuple[str, str, dict]] = []

    async def fake_post_tr(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "") -> tuple[dict, dict]:
        seen.append((api_id, path, body))
        return ({"return_code": 0, "return_msg": "OK", "stk_min_pole_chart_qry": []}, {})

    client._post_tr = fake_post_tr  # type: ignore[method-assign]
    minute = asyncio.run(client.get_nxt_minute_chart(object(), "005930", "20260105"))
    premarket = asyncio.run(client.get_nxt_premarket_chart(object(), "005930", "20260105"))
    assert minute["rt_cd"] == "0"
    assert premarket["rt_cd"] == "0"
    assert len(seen) == 2
    for api_id, _path, body in seen:
        assert api_id == "ka10080"
        assert body["stk_cd"] == "005930_NX"
        assert body["base_dt"] == "20260105"


def test_kiwoom_issued_token_repr_hides_token() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.api.kiwoom.client import KiwoomIssuedToken

    text = repr(KiwoomIssuedToken(token="kw-SECRET-0003", expires_at=datetime(2026, 10, 5, 9, tzinfo=ZoneInfo("Asia/Seoul"))))
    assert "kw-SECRET-0003" not in text


def test_ka10059_request_identity() -> None:
    client = _kiwoom_client()
    seen: list[dict] = []
    body = {"return_code": 0, "stk_invsr_orgn": []}
    headers = {"cont-yn": "N"}

    async def fake_post_tr(session, api_id, path, req_body, cont_yn="N", next_key="", max_retries=None):
        seen.append({"api_id": api_id, "path": path, "body": req_body, "cont_yn": cont_yn, "next_key": next_key, "max_retries": max_retries})
        return (body, headers)

    client._post_tr = fake_post_tr  # type: ignore[method-assign]
    out = asyncio.run(client.get_investor_institution_daily(object(), "005930", "20260910"))
    assert len(seen) == 1
    assert seen[0]["api_id"] == "ka10059"
    assert seen[0]["path"] == "/api/dostk/stkinfo"
    assert list(seen[0]["body"].items()) == [
        ("dt", "20260910"),
        ("stk_cd", "005930"),
        ("amt_qty_tp", "1"),
        ("trde_tp", "0"),
        ("unit_tp", "1000"),
    ]
    assert seen[0]["cont_yn"] == "N" and seen[0]["next_key"] == "" and seen[0]["max_retries"] is None
    assert out is body


def test_ka10059_return_code_passthrough() -> None:
    client = _kiwoom_client()
    body = {"return_code": 1, "return_msg": "x"}

    async def fake_post_tr(session, api_id, path, req_body, cont_yn="N", next_key="", max_retries=None):
        return (body, {})

    client._post_tr = fake_post_tr  # type: ignore[method-assign]
    assert asyncio.run(client.get_investor_institution_daily(object(), "005930", "20260910")) is body


def test_ka10059_exception_passthrough() -> None:
    import aiohttp

    client = _kiwoom_client()

    async def fake_post_tr(session, api_id, path, req_body, cont_yn="N", next_key="", max_retries=None):
        raise aiohttp.ClientError("down")

    client._post_tr = fake_post_tr  # type: ignore[method-assign]
    with pytest.raises(aiohttp.ClientError):
        asyncio.run(client.get_investor_institution_daily(object(), "005930", "20260910"))


def _scripted_post_tr(responses: list[tuple[dict, dict]]) -> tuple[Any, list[tuple[str, str]]]:
    sent: list[tuple[str, str]] = []

    async def fake_post_tr(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        sent.append((cont_yn, next_key))
        return responses[len(sent) - 1]

    return fake_post_tr, sent


def _tick_row(index: int) -> dict[str, str]:
    return {"cur_prc": "270000", "trde_qty": "1", "cntr_tm": f"20260904{143000 - index:06d}"}


def test_kiwoom_tick_mid_walk_failure_keeps_rows_and_cursor() -> None:
    from src.api.kiwoom.client import TickCursor

    client = _kiwoom_client()
    good = [
        ({"return_code": 0, "return_msg": "OK", "stk_tic_chart_qry": [_tick_row(i * 2), _tick_row(i * 2 + 1)]}, {"cont-yn": "Y", "next-key": f"k{i + 1}"})
        for i in range(3)
    ]
    failure = ({"return_code": 7, "return_msg": "서비스를 처리하는 중에 오류가 발생했습니다 [1631]"}, {"cont-yn": "N"})
    fake, sent = _scripted_post_tr([*good, failure])
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))
    assert sent[-1] == ("Y", "k3")
    assert res["rt_cd"] == "1"
    assert res["termination_reason"] == "vendor_failure"
    assert res["vendor_return_code"] == 7
    assert res["rows_received"] == 6
    assert res["output2"] == [row for i in range(6) for row in [_tick_row(i)]]
    assert res["cursor"] == TickCursor(next_key="k3", rows_received=6, total_ticks=None)


def test_kiwoom_tick_first_page_failure_has_no_cursor() -> None:
    client = _kiwoom_client()
    fake, _ = _scripted_post_tr([({"return_code": 3, "return_msg": "bad code"}, {"cont-yn": "N"})])
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))
    assert res["cursor"] is None
    assert res["output2"] == []
    assert res["vendor_return_code"] == 3


def test_kiwoom_tick_resume_pass_failure_returns_resume_key() -> None:
    from src.api.kiwoom.client import TickCursor

    client = _kiwoom_client()
    fake, sent = _scripted_post_tr([({"return_code": 7, "return_msg": "busy"}, {"cont-yn": "N"})])
    client._post_tr = fake
    res = asyncio.run(
        client.get_tick_chart(object(), "005930", "2026-09-04", resume=TickCursor(next_key="K0", rows_received=900, total_ticks=2700))
    )
    assert sent == [("Y", "K0")]
    assert res["cursor"] is not None
    assert res["cursor"].next_key == "K0"
    assert res["cursor"].rows_received == 900
    assert res["output2"] == []


def test_kiwoom_tick_exception_path_has_no_vendor_return_code() -> None:
    client = _kiwoom_client()

    async def boom(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        raise RuntimeError("socket reset")

    client._post_tr = boom
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))
    assert res["rt_cd"] == "1"
    assert "vendor_return_code" not in res
    assert res["output2"] == []


def _post_tr_steps(steps: list[Any]) -> tuple[Any, list[tuple[str, str]]]:
    sent: list[tuple[str, str]] = []

    async def fake_post_tr(session: Any, api_id: str, path: str, body: dict, cont_yn: str = "N", next_key: str = "", max_retries: int = 3) -> tuple[dict, dict]:
        sent.append((cont_yn, next_key))
        step = steps[len(sent) - 1]
        if isinstance(step, BaseException):
            raise step
        return step

    return fake_post_tr, sent


def test_kiwoom_tick_client_error_mid_walk_keeps_rows_and_cursor() -> None:
    import aiohttp

    from src.api.kiwoom.client import TickCursor

    client = _kiwoom_client()
    good = [_tick_page([_tick_row(i * 2), _tick_row(i * 2 + 1)], "Y", f"k{i + 1}") for i in range(3)]
    fake, sent = _post_tr_steps([*good, aiohttp.ClientConnectionError("drop")])
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))
    assert sent[-1] == ("Y", "k3")
    assert res["rt_cd"] == "1"
    assert res["termination_reason"] == "vendor_failure"
    assert res["transport_error"] == "ClientConnectionError"
    assert res["vendor_return_code"] is None
    assert res["output2"] == [_tick_row(i) for i in range(6)]
    assert res["cursor"] == TickCursor(next_key="k3", rows_received=6, total_ticks=None)


def test_kiwoom_tick_transport_error_first_page_has_no_cursor() -> None:
    import aiohttp

    client = _kiwoom_client()
    fake, _ = _post_tr_steps([aiohttp.ClientError("reset")])
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))
    assert res["cursor"] is None
    assert res["output2"] == []
    assert res["transport_error"] == "ClientError"


def test_kiwoom_tick_transport_error_on_resume_returns_resume_key() -> None:
    import aiohttp

    from src.api.kiwoom.client import TickCursor

    client = _kiwoom_client()
    fake, sent = _post_tr_steps([aiohttp.ClientError("reset")])
    client._post_tr = fake
    res = asyncio.run(
        client.get_tick_chart(object(), "005930", "2026-09-04", resume=TickCursor(next_key="K0", rows_received=900, total_ticks=2700))
    )
    assert sent == [("Y", "K0")]
    assert res["cursor"] is not None
    assert res["cursor"].next_key == "K0"
    assert res["cursor"].rows_received == 900
    assert res["cursor"].total_ticks == 2700


def test_kiwoom_tick_timeout_without_deadline_is_transport_not_deadline() -> None:
    client = _kiwoom_client()
    fake, _ = _post_tr_steps([TimeoutError()])
    client._post_tr = fake
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))
    assert res["termination_reason"] == "vendor_failure"
    assert res["transport_error"] == "TimeoutError"


def test_kiwoom_tick_wait_for_timeout_with_deadline_is_deadline() -> None:
    client = _kiwoom_client()

    async def _sleepy(session: Any, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(5)
        return _tick_page([], "N", "")

    client._post_tr = _sleepy
    tight = datetime.now(_SEOUL) + timedelta(seconds=0.05)
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=tight)))
    assert res["termination_reason"] == "deadline"
    assert res["truncated"] is True
    assert res["rt_cd"] == "0"


def test_kiwoom_tick_server_timeout_with_deadline_is_transport() -> None:
    import aiohttp

    client = _kiwoom_client()
    fake, _ = _post_tr_steps([aiohttp.ServerTimeoutError("slow")])
    client._post_tr = fake
    res = asyncio.run(
        client.get_tick_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=datetime.now(_SEOUL) + timedelta(hours=1)))
    )
    assert res["termination_reason"] == "vendor_failure"
    assert res["transport_error"] == "ServerTimeoutError"


def test_kiwoom_tick_observer_timeout_still_raises() -> None:
    client = _kiwoom_client()
    fake, _ = _post_tr_steps([_tick_page([_tick_row(0)], "N", "")])
    client._post_tr = fake

    def _observer(*args: Any) -> None:
        raise TimeoutError("evidence store stalled")

    with pytest.raises(TimeoutError, match="evidence store stalled"):
        asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", on_page=_observer))


def test_kiwoom_tick_parse_bug_logs_traceback_and_error_type(caplog) -> None:
    import logging

    client = _kiwoom_client()
    fake, _ = _post_tr_steps([(None, {})])
    client._post_tr = fake
    with caplog.at_level(logging.WARNING, logger="src.api.kiwoom.client"):
        res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))
    assert res["rt_cd"] == "1"
    assert "vendor_return_code" not in res
    assert res["error_type"] == "AttributeError"
    records = [r for r in caplog.records if "status=ERROR" in r.getMessage()]
    assert len(records) == 1
    assert records[0].exc_info is not None
    assert "symbol=005930" in records[0].getMessage()


def test_kiwoom_tick_transport_log_has_identifiers_and_no_secrets(caplog) -> None:
    import logging

    import aiohttp

    client = KiwoomApiClient(app_key="app-secret-key", secret_key="s")
    client.token = "tok-secret"
    good = [_tick_page([_tick_row(i * 2), _tick_row(i * 2 + 1)], "Y", f"k{i + 1}") for i in range(3)]
    fake, _ = _post_tr_steps([*good, aiohttp.ClientConnectionError("tok-secret app-secret-key")])
    client._post_tr = fake
    with caplog.at_level(logging.WARNING, logger="src.api.kiwoom.client"):
        asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04", venue="KRX"))
    records = [r for r in caplog.records if "status=TRANSPORT_FAILURE" in r.getMessage()]
    assert len(records) == 1
    message = records[0].getMessage()
    for token in ("symbol=005930", "venue=KRX", "error=ClientConnectionError", "pages=3", "request_key=k3"):
        assert token in message
    assert "tok-secret" not in caplog.text
    assert "app-secret-key" not in caplog.text


def test_kiwoom_tick_error_outside_request_is_not_transport(monkeypatch) -> None:
    import aiohttp

    import src.api.kiwoom.client as kiwoom_client_module

    client = _kiwoom_client()
    fake, _ = _post_tr_steps([_tick_page([_tick_row(0)], "Y", "k1")])
    client._post_tr = fake
    real_parse = kiwoom_client_module._parse_tick_remaining

    def _parse_raises_client_error(next_key: str, base_code: str, ymd: str) -> int | None:
        raise aiohttp.ClientError("raised after the page was accumulated")

    monkeypatch.setattr(kiwoom_client_module, "_parse_tick_remaining", _parse_raises_client_error)
    res = asyncio.run(client.get_tick_chart(object(), "005930", "2026-09-04"))
    monkeypatch.setattr(kiwoom_client_module, "_parse_tick_remaining", real_parse)
    assert res["rt_cd"] == "1"
    assert res["error_type"] == "ClientError"
    assert res.get("transport_error") is None
    assert res["cursor"] is None
    assert res["output2"] == []


def test_kiwoom_tick_server_timeout_near_deadline_is_still_transport() -> None:
    import aiohttp

    client = _kiwoom_client()
    fake, _ = _post_tr_steps([aiohttp.ServerTimeoutError("slow")])
    client._post_tr = fake
    res = asyncio.run(
        client.get_tick_chart(object(), "005930", "2026-09-04", budget=_budget(deadline=datetime.now(_SEOUL) + timedelta(seconds=0.5)))
    )
    assert res["termination_reason"] == "vendor_failure"
    assert res["transport_error"] == "ServerTimeoutError"
