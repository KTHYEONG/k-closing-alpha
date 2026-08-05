"""LGBMRanker + Quantile 위험 예측 + Utility Sizing 파이프라인 통합 테스트.

Data -> LGBMRanker(순위) + Quantile Regressor(q10/q50/q90) + Calibrated
Classifier(p_good/p_bad) + Utility Sizing 파이프라인 전체가 동일 데이터에서
일관된 계약(컬럼/한도)을 만족하는지 검증합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.model_pipeline import run_model_pipeline, run_sizing_pipeline

FEATURE_COLS = ["feature_a", "feature_b"]
TARGET_COL = "net_return"
GROUP_COL = "trade_date"

_PREDICTION_COLS = ["pred_q10", "pred_q50", "pred_q90", "p_good", "p_bad"]
_GRADE_MULTIPLIERS = {0.0, 0.5, 1.0, 1.5}


def _make_dataset(n_groups: int = 15, rows_per_group: int = 8, seed: int = 19) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime([f"2024-03-{d:02d}" for d in range(1, n_groups + 1)])
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.extend({"trade_date": date} for _ in range(rows_per_group))
    df = pd.DataFrame(rows)
    df["feature_a"] = rng.normal(size=len(df))
    df["feature_b"] = rng.normal(size=len(df))
    df["net_return"] = 0.02 * df["feature_a"] + rng.normal(scale=0.009, size=len(df))
    df["selection_rank"] = df.groupby(GROUP_COL, sort=False).cumcount() + 1
    df["decision_timestamp"] = df[GROUP_COL].map(
        lambda d: pd.Timestamp(d, tz="Asia/Seoul").replace(hour=15, minute=30)
    )
    df["feature_available_timestamp"] = df["decision_timestamp"]
    return df


def test_sizing_pipeline_returns_quantile_and_sizing_contracts() -> None:
    df = _make_dataset()
    sizing = run_sizing_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
    )
    assert set(sizing.keys()) == {"quantile_df", "sizing_df"}

    quantile_df = sizing["quantile_df"]
    assert set(_PREDICTION_COLS).issubset(quantile_df.columns)
    assert (quantile_df["pred_q10"] <= quantile_df["pred_q50"]).all()
    assert (quantile_df["pred_q50"] <= quantile_df["pred_q90"]).all()
    assert quantile_df["p_good"].between(0.0, 1.0).all()
    assert quantile_df["p_bad"].between(0.0, 1.0).all()

    sizing_df = sizing["sizing_df"]
    assert {"utility_score", "grade", "grade_multiplier", "allocation"} <= set(sizing_df.columns)
    assert sizing_df["grade"].isin(["Strong", "Good", "Weak", "Pass"]).all()
    assert sizing_df["grade_multiplier"].isin(_GRADE_MULTIPLIERS).all()
    assert sizing_df["allocation"].ge(0.0).all()


def test_sizing_respects_per_group_risk_limits() -> None:
    df = _make_dataset()
    sizing = run_sizing_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        max_position_pct=0.25,
        max_total_allocation=1.0,
    )
    sizing_df = sizing["sizing_df"]
    for _, group in sizing_df.groupby(GROUP_COL):
        assert group["allocation"].sum() <= 1.0 + 1e-9
        assert group["allocation"].max() <= 0.25 + 1e-9


def test_ranker_and_sizing_pipelines_run_on_same_data() -> None:
    df = _make_dataset()
    ranker = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        model_type="lgb_ranker",
    )
    sizing = run_sizing_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
    )
    assert 0 < len(ranker["oof_predictions"]) <= len(df)
    assert 0 < len(sizing["sizing_df"]) <= len(df)
    assert set(ranker["oof_predictions"].index) == set(sizing["sizing_df"].index)


def test_sizing_pipeline_high_quality_signal_dominates_sizing() -> None:
    """신호가 강하면(고 q50·p_good) Strong/Good 등급이 Pass 보다 높은 비중을 가져야 한다.

    순차 fit/보정 구간 분할(chronological calibration) 후에도 fit 구간에서
    신호가 분리될 수 있도록 충분한 날짜 수와 강한 신호를 사용합니다.
    """
    rng = np.random.default_rng(7)
    dates = pd.to_datetime([f"2024-04-{d:02d}" for d in range(1, 25)])
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.extend({"trade_date": date} for _ in range(12))
    df = pd.DataFrame(rows)
    df["feature_a"] = rng.normal(size=len(df))
    df["feature_b"] = rng.normal(size=len(df))
    df["net_return"] = 0.04 * df["feature_a"] + rng.normal(scale=0.005, size=len(df))

    sizing_df = run_sizing_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
    )["sizing_df"]
    strong_alloc = sizing_df.loc[sizing_df["grade"] == "Strong", "allocation"].sum()
    pass_alloc = sizing_df.loc[sizing_df["grade"] == "Pass", "allocation"].sum()
    assert strong_alloc > pass_alloc
