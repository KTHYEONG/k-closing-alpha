from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pandas as pd

from src.backfill.kis_flow_backfill import (
    AsyncRateLimiter,
    FLOW_COLUMNS,
    FlowBackfillConfig,
    _plan_missing_fields,
    apply_flow_checkpoints,
    plan_missing_flows,
    run_kis_flow_backfill,
    _fetch_symbol,
    _fetch_symbol_guarded,
)


def _source() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["000001", "000001", "000002"],
        "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
        "foreign_netbuy": [1.0, None, None],
        "inst_netbuy": [1.0, None, None],
        "program_netbuy": [1.0, None, None],
    })


def test_plan_missing_flows_uses_checkpoint() -> None:  # SCENARIO_KIS_FLOW_BACKFILL_01
    checkpoint = _source().iloc[[1]].fillna(2.0)
    plan = plan_missing_flows(_source(), checkpoint)
    assert plan == {"000002": ["20200101"]}


def test_plan_missing_fields_skips_complete_flow_family() -> None:
    source = _source()
    checkpoint = source.iloc[[1]].fillna(2.0)
    assert _plan_missing_fields(source, checkpoint) == {"000001": (False, False), "000002": (True, True)}


def test_plan_missing_fields_detects_partial_program_gap() -> None:
    source = _source().fillna(1.0)
    source.loc[1, "program_netbuy"] = None
    assert _plan_missing_fields(source)["000001"] == (False, True)


def test_fetch_symbol_allows_single_flow_family() -> None:
    async def investor(*args, **kwargs):
        return pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])

    async def program(*args, **kwargs):
        return {}

    inv_patch = patch(
        "src.backfill.kis_flow_backfill.get_investor_trade_daily_async",
        new=AsyncMock(side_effect=investor),
    )
    prog_patch = patch(
        "src.backfill.kis_flow_backfill.get_program_history_async",
        new=AsyncMock(side_effect=program),
    )
    with inv_patch as inv, prog_patch as prog:
        asyncio.run(_fetch_symbol(object(), object(), AsyncRateLimiter(10), "000001", ["20200102"], False, True))
    inv.assert_not_awaited()
    prog.assert_awaited_once()


def test_fetch_symbol_requests_both_flow_families() -> None:
    async def investor(*args, **kwargs):
        return pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])

    async def program(*args, **kwargs):
        return {}

    with patch(
        "src.backfill.kis_flow_backfill.get_investor_trade_daily_async",
        new=AsyncMock(side_effect=investor),
    ) as inv, patch(
        "src.backfill.kis_flow_backfill.get_program_history_async",
        new=AsyncMock(side_effect=program),
    ) as prog:
        asyncio.run(_fetch_symbol(object(), object(), AsyncRateLimiter(10), "000001", ["20200102"]))
    inv.assert_awaited_once()
    prog.assert_awaited_once()


def test_fetch_symbol_guarded_passes_field_requirements(monkeypatch) -> None:
    worker = AsyncMock(return_value=_source().iloc[[1]])
    monkeypatch.setattr("src.backfill.kis_flow_backfill._fetch_symbol", worker)
    asyncio.run(
        _fetch_symbol_guarded(
            object(), object(), AsyncRateLimiter(10), asyncio.Semaphore(1),
            {"000001": (True, False)}, "000001", ["20200102"],
        )
    )
    assert worker.await_args.args[-2:] == (True, False)


def test_fetch_symbol_guarded_isolates_symbol_failure(monkeypatch) -> None:
    worker = AsyncMock(side_effect=RuntimeError("bad payload"))
    monkeypatch.setattr("src.backfill.kis_flow_backfill._fetch_symbol", worker)
    out = asyncio.run(
        _fetch_symbol_guarded(
            object(), object(), AsyncRateLimiter(10), asyncio.Semaphore(1),
            {"000001": (True, True)}, "000001", ["20200102"],
        )
    )
    assert len(out) == 1
    assert out["symbol"].iloc[0] == "000001"
    assert out[list(FLOW_COLUMNS)].isna().all().all()


def test_run_backfill_creates_guarded_tasks(tmp_path, monkeypatch) -> None:
    parquet = tmp_path / "price.parquet"
    _source().iloc[[1]].to_parquet(parquet, index=False)

    class _SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *args):
            return None

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def create_session(self):
            return _SessionContext()

        async def ensure_token(self, session):
            return "token"

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("src.backfill.kis_flow_backfill.KisApiClient", _Client)
    monkeypatch.setattr("src.backfill.kis_flow_backfill._fetch_symbol", AsyncMock(return_value=_source().iloc[[1]]))
    monkeypatch.setattr("src.backfill.kis_flow_backfill.asyncio.to_thread", immediate_to_thread)
    result = asyncio.run(
        run_kis_flow_backfill(parquet, tmp_path / "cp", FlowBackfillConfig(checkpoint_symbols=1))
    )
    assert result.completed_symbols == 1


def test_apply_checkpoints_fills_only_nulls(tmp_path) -> None:  # SCENARIO_KIS_FLOW_BACKFILL_02
    parquet = tmp_path / "price.parquet"
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    _source().to_parquet(parquet, index=False)
    pd.DataFrame({"symbol": ["000001"], "date": pd.to_datetime(["2020-01-02"]), "foreign_netbuy": [9.0], "inst_netbuy": [8.0], "program_netbuy": [7.0]}).to_parquet(checkpoint_dir / "batch_00000.parquet", index=False)
    assert apply_flow_checkpoints(parquet, checkpoint_dir) == 3
    out = pd.read_parquet(parquet)
    assert len(out) == 3
    assert out.loc[0, "foreign_netbuy"] == 1.0
    assert out.loc[1, "program_netbuy"] == 7.0


def test_flow_config_and_limiter_reject_invalid_rate() -> None:
    assert FlowBackfillConfig(requests_per_second=10.0).requests_per_second == 10.0
    try:
        AsyncRateLimiter(0)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid rate must fail closed")


def test_run_backfill_returns_empty_for_complete_source(tmp_path) -> None:
    parquet = tmp_path / "price.parquet"
    source = _source().fillna(1.0)
    source.to_parquet(parquet, index=False)
    result = __import__("asyncio").run(run_kis_flow_backfill(parquet, tmp_path / "cp", FlowBackfillConfig()))
    assert result.planned_symbols == 0



def test_fetch_toss_program_history_pages_backward_until_empty() -> None:
    calls: list[str] = []

    class _FakeToss:
        async def get_program_trades(self, session, symbol, count, until):
            calls.append(until)
            if until == "2020-01-05":
                records = [
                    {"date": "2020-01-05", "arbitrage": {"netBuyVolume": 1.0}, "nonArbitrage": {"netBuyVolume": 1.0}},
                    {"date": "2020-01-03", "arbitrage": {"netBuyVolume": 2.0}, "nonArbitrage": {"netBuyVolume": 0.0}},
                ]
            elif until == "2020-01-02":
                records = [{"date": "2020-01-02", "arbitrage": {"netBuyVolume": 3.0}, "nonArbitrage": {"netBuyVolume": 0.0}}]
            else:
                records = []
            return {"result": {"records": records}}

    from src.backfill.kis_flow_backfill import fetch_toss_program_history

    out = asyncio.run(fetch_toss_program_history(object(), _FakeToss(), "000001", ["20200102", "20200103", "20200105"]))

    assert out == {"20200105": 2.0, "20200103": 2.0, "20200102": 3.0}
    # Then: 2번째 페이지가 요청 최저일(2020-01-02)에 도달 -> 다음 커서(01-01)는 floor 미만이라 3번째 호출 없음
    assert calls == ["2020-01-05", "2020-01-02"]



def test_fetch_toss_program_history_stops_at_requested_floor() -> None:
    calls: list[str] = []

    class _FakeToss:
        async def get_program_trades(self, session, symbol, count, until):
            calls.append(until)
            return {"result": {"records": [{"date": "2019-04-01", "arbitrage": {"netBuyVolume": 5.0}, "nonArbitrage": {"netBuyVolume": 0.0}}]}}

    from src.backfill.kis_flow_backfill import fetch_toss_program_history

    out = asyncio.run(fetch_toss_program_history(object(), _FakeToss(), "000001", ["20190401"]))

    # Then: 요청 최저일에 도달하면(더 과거로 커서를 옮겨도 min(dates) 미만) 추가 페이지를 요청하지 않는다
    assert out == {"20190401": 5.0}
    assert calls == ["2019-04-01"]



def test_fetch_toss_program_history_propagates_vendor_error() -> None:
    import pytest

    from src.backfill.kis_flow_backfill import fetch_toss_program_history
    from src.daily.price_ingest import VendorResponseError

    class _FakeToss:
        async def get_program_trades(self, session, symbol, count, until):
            return {"error": {"code": "invalid-request", "message": "bad symbol"}}

    with pytest.raises(VendorResponseError, match="Toss program-trades"):
        asyncio.run(fetch_toss_program_history(object(), _FakeToss(), "000001", ["20200102"]))



def test_fetch_toss_program_history_returns_empty_for_no_dates() -> None:
    from src.backfill.kis_flow_backfill import fetch_toss_program_history

    class _Boom:
        async def get_program_trades(self, *a, **k):
            raise AssertionError("must not be called with an empty date list")

    assert asyncio.run(fetch_toss_program_history(object(), _Boom(), "000001", [])) == {}



def test_fetch_symbol_splits_program_dates_between_toss_and_kis() -> None:
    class _FakeToss:
        async def get_program_trades(self, session, symbol, count, until):
            return {"result": {"records": [{"date": "2020-01-02", "arbitrage": {"netBuyVolume": 9.0}, "nonArbitrage": {"netBuyVolume": 0.0}}]}}

    kis_program_dates: list[str] = []

    async def investor(*args, **kwargs):
        return pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])

    async def program(session, client, symbol, start, end, *, target_dates, request_slot):
        kis_program_dates.extend(target_dates)
        return dict.fromkeys(target_dates, 1.0)

    from src.backfill.kis_flow_backfill import TOSS_PROGRAM_HISTORY_START_YMD, _fetch_symbol

    with patch("src.backfill.kis_flow_backfill.get_investor_trade_daily_async", new=AsyncMock(side_effect=investor)), \
         patch("src.backfill.kis_flow_backfill.get_program_history_async", new=AsyncMock(side_effect=program)) as prog:
        out = asyncio.run(_fetch_symbol(
            object(), object(), AsyncRateLimiter(10), "000001",
            ["20160104", "20200102"], False, True, toss=_FakeToss(),
        ))

    # Then: 2020-01-02(>=경계)는 Toss가 담당, 2016-01-04(<경계)만 KIS 프로그램 조회로 간다
    assert TOSS_PROGRAM_HISTORY_START_YMD == "20190401"
    assert kis_program_dates == ["20160104"]
    prog.assert_awaited_once()
    row = out.set_index(out["date"].dt.strftime("%Y%m%d"))
    assert row.loc["20200102", "program_netbuy"] == 9.0
    assert row.loc["20160104", "program_netbuy"] == 1.0



def test_fetch_symbol_falls_back_to_kis_when_toss_raises_vendor_error() -> None:
    from src.daily.price_ingest import VendorResponseError

    class _FakeToss:
        async def get_program_trades(self, session, symbol, count, until):
            raise VendorResponseError("Toss program-trades code=rate-limited msg=slow down")

    kis_program_dates: list[str] = []

    async def investor(*args, **kwargs):
        return pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])

    async def program(session, client, symbol, start, end, *, target_dates, request_slot):
        kis_program_dates.extend(target_dates)
        return dict.fromkeys(target_dates, 2.0)

    from src.backfill.kis_flow_backfill import _fetch_symbol

    with patch("src.backfill.kis_flow_backfill.get_investor_trade_daily_async", new=AsyncMock(side_effect=investor)), \
         patch("src.backfill.kis_flow_backfill.get_program_history_async", new=AsyncMock(side_effect=program)):
        out = asyncio.run(_fetch_symbol(
            object(), object(), AsyncRateLimiter(10), "000001",
            ["20200102"], False, True, toss=_FakeToss(),
        ))

    # Then: Toss 실패 -> 애초에 Toss 담당이던 날짜 전부 KIS로 재라우팅(부분 병합 없음)
    assert kis_program_dates == ["20200102"]
    assert out["program_netbuy"].iloc[0] == 2.0



def test_fetch_symbol_fills_toss_depth_gap_from_kis() -> None:
    class _FakeToss:
        async def get_program_trades(self, session, symbol, count, until):
            # 2019-04-01만 반환(2019-04-02 요청분은 실제 Toss 이력 한계처럼 누락)
            return {"result": {"records": [{"date": "2019-04-01", "arbitrage": {"netBuyVolume": 4.0}, "nonArbitrage": {"netBuyVolume": 0.0}}]}}

    kis_program_dates: list[str] = []

    async def investor(*args, **kwargs):
        return pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])

    async def program(session, client, symbol, start, end, *, target_dates, request_slot):
        kis_program_dates.extend(target_dates)
        return dict.fromkeys(target_dates, 7.0)

    from src.backfill.kis_flow_backfill import _fetch_symbol

    with patch("src.backfill.kis_flow_backfill.get_investor_trade_daily_async", new=AsyncMock(side_effect=investor)), \
         patch("src.backfill.kis_flow_backfill.get_program_history_async", new=AsyncMock(side_effect=program)):
        out = asyncio.run(_fetch_symbol(
            object(), object(), AsyncRateLimiter(10), "000001",
            ["20190401", "20190402"], False, True, toss=_FakeToss(),
        ))

    # Then: Toss가 커버 못한 20190402만 KIS로 채워지고(부분 폴백), 20190401은 Toss 값 유지
    assert kis_program_dates == ["20190402"]
    row = out.set_index(out["date"].dt.strftime("%Y%m%d"))
    assert row.loc["20190401", "program_netbuy"] == 4.0
    assert row.loc["20190402", "program_netbuy"] == 7.0



def test_run_backfill_builds_toss_client_unless_disabled(tmp_path, monkeypatch) -> None:
    parquet = tmp_path / "price.parquet"
    _source().iloc[[1]].to_parquet(parquet, index=False)

    class _SessionContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *args):
            return None

    class _Client:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def create_session(self):
            return _SessionContext()

        async def ensure_token(self, session):
            return "token"

    seen_toss: list[object] = []

    async def _fake_fetch_symbol(session, client, limiter, symbol, dates, need_investor, need_program, *, toss=None):
        seen_toss.append(toss)
        return _source().iloc[[1]]

    async def immediate_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("src.backfill.kis_flow_backfill.KisApiClient", _Client)
    monkeypatch.setattr("src.backfill.kis_flow_backfill._fetch_symbol", _fake_fetch_symbol)
    monkeypatch.setattr("src.backfill.kis_flow_backfill.asyncio.to_thread", immediate_to_thread)

    asyncio.run(run_kis_flow_backfill(parquet, tmp_path / "cp1", FlowBackfillConfig(checkpoint_symbols=1)))
    asyncio.run(run_kis_flow_backfill(parquet, tmp_path / "cp2", FlowBackfillConfig(checkpoint_symbols=1, use_toss_program=False)))

    # Then: 기본값은 TossApiClient 인스턴스를 전달, use_toss_program=False면 None
    from src.api.toss.client import TossApiClient

    assert isinstance(seen_toss[0], TossApiClient)
    assert seen_toss[1] is None



def test_fetch_toss_program_history_stops_when_toss_has_no_further_history() -> None:
    calls: list[str] = []

    class _FakeToss:
        async def get_program_trades(self, session, symbol, count, until):
            calls.append(until)
            if until == "2020-01-05":
                return {"result": {"records": [{"date": "2020-01-05", "arbitrage": {"netBuyVolume": 1.0}, "nonArbitrage": {"netBuyVolume": 0.0}}]}}
            # 실제 Toss 이력 하한(2019-03-31)에 해당하는 빈 성공 응답을 재현: 요청 최저일(2016-01-04)에
            # 아직 도달하지 않았지만 Toss가 더 줄 데이터가 없다는 신호이므로 여기서 멈춰야 한다.
            return {"result": {"records": []}}

    from src.backfill.kis_flow_backfill import fetch_toss_program_history

    out = asyncio.run(fetch_toss_program_history(object(), _FakeToss(), "000001", ["20200105", "20160104"]))

    # Then: 빈 응답을 받은 즉시 중단(추가 페이지 요청 없음), 첫 페이지에서 얻은 값만 남는다
    assert out == {"20200105": 1.0}
    assert calls == ["2020-01-05", "2020-01-04"]


def test_main_dispatches_backfill_with_toss_flag_and_defaults(monkeypatch, tmp_path) -> None:
    import sys

    from src.backfill import kis_flow_backfill as mod

    seen: list[mod.FlowBackfillConfig] = []

    async def _fake_run(parquet_path, checkpoint_dir, config, symbols=None):
        seen.append(config)
        return mod.FlowBackfillResult(0, 0, ())

    monkeypatch.setattr(mod, "run_kis_flow_backfill", _fake_run)
    parquet = str(tmp_path / "price.parquet")
    cp = str(tmp_path / "cp")

    # When: 기본(Toss 사용) 그리고 --no-toss-program(비활성화)
    monkeypatch.setattr(sys, "argv", ["kis_flow_backfill", "--parquet", parquet, "--checkpoint-dir", cp])
    mod.main()
    monkeypatch.setattr(sys, "argv", ["kis_flow_backfill", "--parquet", parquet, "--checkpoint-dir", cp, "--no-toss-program"])
    mod.main()

    # Then: FlowBackfillConfig의 4번째 위치 인자(use_toss_program)가 플래그를 따라간다
    assert seen[0].use_toss_program is True
    assert seen[1].use_toss_program is False
