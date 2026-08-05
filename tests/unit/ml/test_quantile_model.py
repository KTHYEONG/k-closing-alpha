"""Quantile Regression & Calibrated Classifier 위험 예측 단위 테스트.

SCENARIO: quantile_monotonicity_check
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.quantile_model import (
    _chrono_fit_calibration_split,
    _fit_predict_calibrated,
    fit_predict_quantile_and_classifier,
)

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
    # 순차(chronological) 보정은 fold 단위 커버리지/폭/Brier 진단을 보고합니다.
    diagnostics = res.attrs["calibration_diagnostics"]
    assert diagnostics
    for entry in diagnostics:
        assert "fold" in entry
        assert "q10_q90_coverage" in entry
        assert "interval_width" in entry
        assert "brier" in entry
        assert "brier_date_prior" in entry
        assert "log_loss" in entry


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


def test_chrono_fit_calibration_split_empty_train() -> None:
    """빈 train 인덱스는 빈 fit/calibration 으로 처리됩니다."""
    groups = np.asarray(["2024-01-01"] * 4, dtype=object)
    fit, calib = _chrono_fit_calibration_split(groups, np.array([], dtype=np.intp))
    assert fit.size == 0
    assert calib.size == 0


def test_chrono_fit_calibration_split_embargo_eats_fit() -> None:
    """embargo 가 fit 구간 전체를 소모하면 전체를 calibration 으로 반환합니다."""
    groups = np.asarray(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], dtype=object)
    train_idx = np.arange(4, dtype=np.intp)
    fit, calib = _chrono_fit_calibration_split(groups, train_idx, embargo=10)
    assert fit.size == 0
    assert set(calib.tolist()) == {0, 1, 2, 3}


def test_chrono_fit_calibration_split_orders_segments() -> None:
    """fit 구간은 calibration 구간보다 앞선(이른) 그룹으로 구성됩니다."""
    groups = np.asarray(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"] * 2,
        dtype=object,
    )
    train_idx = np.arange(10, dtype=np.intp)
    fit, calib = _chrono_fit_calibration_split(groups, train_idx, embargo=1)
    assert fit.size > 0
    assert calib.size > 0
    fit_groups = set(groups[fit].tolist())
    calib_groups = set(groups[calib].tolist())
    assert fit_groups.isdisjoint(calib_groups)
    assert max(fit_groups) < min(calib_groups)


def test_fit_predict_calibrated_single_class_calibration_returns_raw_score() -> None:
    """보정 구간이 단일 클래스면 Platt 보정을 건너뛰고 raw score 를 반환합니다."""
    rng = np.random.default_rng(4)
    n_fit, n_calib, n_pred = 40, 8, 5
    fit = pd.DataFrame({"f1": rng.normal(size=n_fit), "f2": rng.normal(size=n_fit)})
    fit["label"] = (rng.uniform(size=n_fit) < 0.5).astype(bool)
    calib = pd.DataFrame({"f1": rng.normal(size=n_calib), "f2": rng.normal(size=n_calib)})
    calib["label"] = np.zeros(n_calib, dtype=bool)
    pred = pd.DataFrame({"f1": rng.normal(size=n_pred), "f2": rng.normal(size=n_pred)})
    out = _fit_predict_calibrated(fit, calib, pred, ["f1", "f2"], "label")
    assert len(out) == n_pred
    assert np.isfinite(out).all()


def test_fit_predict_calibrated_prior_fallback_for_single_class_fit() -> None:
    """fit 구간이 단일 클래스면 사전확률 상수를 반환합니다."""
    rng = np.random.default_rng(5)
    n_fit = 30
    fit = pd.DataFrame({"f1": rng.normal(size=n_fit), "f2": rng.normal(size=n_fit)})
    fit["label"] = np.ones(n_fit, dtype=bool)
    calib = pd.DataFrame({"f1": rng.normal(size=5), "f2": rng.normal(size=5)})
    calib["label"] = np.ones(5, dtype=bool)
    pred = pd.DataFrame({"f1": rng.normal(size=4), "f2": rng.normal(size=4)})
    out = _fit_predict_calibrated(fit, calib, pred, ["f1", "f2"], "label")
    np.testing.assert_allclose(out, np.ones(4, dtype=np.float64))
    """보정 구간에 두 클래스가 존재하면 Platt(sigmoid) 보정 경로가 실행됩니다."""
    rng = np.random.default_rng(3)
    n_fit, n_calib, n_pred = 40, 10, 5
    fit = pd.DataFrame({"f1": rng.normal(size=n_fit), "f2": rng.normal(size=n_fit)})
    fit["label"] = (rng.uniform(size=n_fit) < 0.5).astype(bool)
    calib = pd.DataFrame({"f1": rng.normal(size=n_calib), "f2": rng.normal(size=n_calib)})
    calib["label"] = (np.arange(n_calib) < 5).astype(bool)
    pred = pd.DataFrame({"f1": rng.normal(size=n_pred), "f2": rng.normal(size=n_pred)})
    out = _fit_predict_calibrated(fit, calib, pred, ["f1", "f2"], "label")
    assert len(out) == n_pred
    assert np.isfinite(out).all()
    assert (out >= 0.0).all()
    assert (out <= 1.0).all()
