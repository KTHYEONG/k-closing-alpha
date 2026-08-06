from __future__ import annotations

import pandas as pd

from src.backfill.kis_flow_backfill import AsyncRateLimiter, FlowBackfillConfig, apply_flow_checkpoints, plan_missing_flows, run_kis_flow_backfill


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
