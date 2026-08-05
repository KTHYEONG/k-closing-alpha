"""Backtest Evaluation (Baseline 비교 & 연도별 안정성 감사) 단위 테스트.

SCENARIO_BACKTEST_BASELINE_COMPARISON
SCENARIO_BACKTEST_YEARLY_BREAKDOWN
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.backtest_evaluator import run_backtest_evaluation

GROUP_COL = "trade_date"
TARGET_COL = "net_return"

METRIC_KEYS = (
    "top_1_return",
    "top_3_return",
    "win_rate",
    "profit_factor",
    "mean_win",
    "mean_loss",
    "sharpe",
    "cost_adjusted_return",
    "date_weighted_return",
    "capital_weighted_return",
    "turnover",
    "max_drawdown",
)


def _make_oof(
    n_groups: int = 12,
    rows_per_group: int = 6,
    seed: int = 7,
    include_selection_rank: bool = True,
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_groups, freq="D")
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.extend({GROUP_COL: date} for _ in range(rows_per_group))
    df = pd.DataFrame(rows)
    df["pred"] = rng.normal(size=len(df))
    df[TARGET_COL] = 0.5 * df["pred"] + rng.normal(scale=0.05, size=len(df))
    if include_selection_rank:
        df["selection_rank"] = (
            df.groupby(GROUP_COL, sort=False)["pred"].rank(method="first", ascending=True).astype(int)
        )
    return df


def test_run_backtest_evaluation_returns_contract_shape() -> None:
    oof = _make_oof()
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    assert "model_metrics" in result
    assert "baseline_metrics" in result
    assert "yearly_breakdown" in result
    assert "regime_breakdown" in result

    model = result["model_metrics"]
    assert set(model) == set(METRIC_KEYS)
    assert result["baseline_metrics"]["selection_rank"] is not None
    assert result["baseline_metrics"]["equal_weight"] is not None
    assert set(result["baseline_metrics"]["selection_rank"]) == set(METRIC_KEYS)
    assert set(result["baseline_metrics"]["equal_weight"]) == set(METRIC_KEYS)


def test_run_backtest_evaluation_fails_closed_when_selection_rank_missing() -> None:
    """selection_rank baseline 은 필수입니다. 누락 시 조용히 생략하지 않고 ValueError 로 실패합니다."""
    oof = _make_oof(include_selection_rank=False)
    with pytest.raises(ValueError, match="selection_rank"):
        run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)


def test_run_backtest_evaluation_yearly_breakdown() -> None:
    oof = _make_oof(n_groups=15, start_date="2023-12-27")
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    yearly = result["yearly_breakdown"]
    assert set(yearly) == {2023, 2024}
    for entry in yearly.values():
        assert entry is not None
        for key in ("top1_return", "top3_return", "win_rate", "profit_factor", "sharpe"):
            assert key in entry
        assert np.isfinite(entry["top1_return"])


def test_run_backtest_evaluation_yearly_small_sample_is_null() -> None:
    oof = _make_oof(n_groups=9, start_date="2023-12-29")
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    assert result["yearly_breakdown"][2023] is None
    assert result["yearly_breakdown"][2024] is not None


def test_run_backtest_evaluation_profit_factor_inf_when_no_loss() -> None:
    oof = _make_oof(n_groups=8, rows_per_group=4)
    oof[TARGET_COL] = 1.0 + oof["pred"] * 0.01
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    assert np.isinf(result["model_metrics"]["profit_factor"])


def test_run_backtest_evaluation_model_beats_baselines() -> None:
    oof = _make_oof(n_groups=30, rows_per_group=6, seed=3)
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    model = result["model_metrics"]
    selection_rank = result["baseline_metrics"]["selection_rank"]
    equal_weight = result["baseline_metrics"]["equal_weight"]
    assert model["top_1_return"] > selection_rank["top_1_return"]
    assert model["top_1_return"] > equal_weight["top_1_return"]
    assert model["win_rate"] > selection_rank["win_rate"]


def test_run_backtest_evaluation_selection_rank_aligned_equals_model() -> None:
    oof = _make_oof(n_groups=10, rows_per_group=4, seed=5)
    oof["selection_rank"] = (
        oof.groupby(GROUP_COL, sort=False)["pred"].rank(method="first", ascending=False).astype(int)
    )
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    model = result["model_metrics"]
    selection_rank = result["baseline_metrics"]["selection_rank"]
    assert selection_rank["top_1_return"] == pytest.approx(model["top_1_return"])
    assert selection_rank["top_3_return"] == pytest.approx(model["top_3_return"])


def test_run_backtest_evaluation_equal_weight_is_daily_group_mean() -> None:
    oof = _make_oof(n_groups=10, rows_per_group=5, seed=9)
    result = run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)

    expected = oof.groupby(GROUP_COL)[TARGET_COL].mean().mean()
    assert result["baseline_metrics"]["equal_weight"]["top_1_return"] == pytest.approx(expected)


def test_run_backtest_evaluation_rejects_missing_required_column() -> None:
    oof = _make_oof().drop(columns=["pred"])
    with pytest.raises(ValueError, match="missing required columns"):
        run_backtest_evaluation(oof, TARGET_COL, GROUP_COL)
