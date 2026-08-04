"""Quantile Regression & Calibrated Classifier 위험 예측 단위 테스트.

SCENARIO: quantile_monotonicity_check
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.quantile_model import fit_predict_quantile_and_classifier

FEATURE_COLS = ["feature_a", "feature_b"]
TARGET_COL = "net_return"
GROUP_COL = "trade_date"

_PREDICTION_COLS = ["pred_q10", "pred_q50", "pred_q90", "p_good", "p_bad"]


def _make_dataset(n_groups: int = 12, rows_per_group: int = 8, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime([f"2024-03-{d:02d}" for d in range(1, n_groups + 1)])
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.extend({"trade_date": date} for _ in range(rows_per_group))
    df = pd.DataFrame(rows)
    df["feature_a"] = rng.normal(size=len(df))
    df["feature_b"] = rng.normal(size=len(df))
    df["net_return"] = 0.015 * df["feature_a"] + rng.normal(scale=0.008, size=len(df))
    return df


def test_quantile_model_returns_prediction_columns() -> None:
    df = _make_dataset()
    res = fit_predict_quantile_and_classifier(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
    )
    assert set(_PREDICTION_COLS).issubset(res.columns)
    assert {GROUP_COL, TARGET_COL, *FEATURE_COLS} <= set(res.columns)
    assert res.index.is_unique
    assert set(res.index) <= set(df.index)
    assert 0 < len(res) <= len(df)


def test_quantile_predictions_are_monotonic() -> None:
    """분위수 예측 q10 <= q50 <= q90 단조성 보장 (SCENARIO: quantile_monotonicity_check)."""
    df = _make_dataset()
    res = fit_predict_quantile_and_classifier(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
    )
    assert (res["pred_q10"] <= res["pred_q50"]).all()
    assert (res["pred_q50"] <= res["pred_q90"]).all()
    assert (res["pred_q10"] <= res["pred_q90"]).all()


def test_calibrated_probabilities_are_in_unit_interval() -> None:
    """Calibrated p_good/p_bad 는 항상 0.0 ~ 1.0 구간 내에 존재해야 한다."""
    df = _make_dataset()
    res = fit_predict_quantile_and_classifier(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
    )
    assert res["p_good"].between(0.0, 1.0).all()
    assert res["p_bad"].between(0.0, 1.0).all()


def test_quantile_model_rejects_missing_columns() -> None:
    df = _make_dataset().drop(columns=["feature_b"])
    with pytest.raises(ValueError, match="missing columns"):
        fit_predict_quantile_and_classifier(df, feature_cols=FEATURE_COLS, target_col=TARGET_COL, group_col=GROUP_COL)


def test_quantile_model_rejects_empty_feature_cols() -> None:
    df = _make_dataset()
    with pytest.raises(ValueError, match="feature_cols"):
        fit_predict_quantile_and_classifier(df, feature_cols=[], target_col=TARGET_COL, group_col=GROUP_COL)
