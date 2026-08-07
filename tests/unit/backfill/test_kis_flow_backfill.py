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
