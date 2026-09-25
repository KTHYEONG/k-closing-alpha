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
            object(), rate_min_pct=2.0, rate_max_pct=10.0, on_page=lambda *a: seen.append(a)
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
        asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=2.0, rate_max_pct=10.0, on_page=_boom))

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
            client3.get_fluctuation_ranking(object(), rate_min_pct=2.0, rate_max_pct=10.0, on_page=_raw_ranking_boom)
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
    res = asyncio.run(client.get_fluctuation_ranking(object(), rate_min_pct=2.0, rate_max_pct=10.0, max_pages=3))

    # Then: 무경고 성공 금지
    assert calls["n"] == 3
    assert res["rt_cd"] == "1"
    assert res["truncated"] is True
    assert len(res["output"]) == 3
    assert inspect.signature(KiwoomApiClient.get_fluctuation_ranking).parameters["max_pages"].default == 20



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
