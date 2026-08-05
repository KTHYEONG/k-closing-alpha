"""Model Pipeline (Purged Walk-Forward CV 학습 + OOF 평가) 단위 테스트.

SCENARIO_MODEL_PIPELINE_TRAIN_EVAL
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.ml.model_pipeline import run_model_pipeline, run_sizing_pipeline
from src.ml.sizing_engine import load_model_artifacts

FEATURE_COLS = ["feature_a", "feature_b"]
TARGET_COL = "net_return"
GROUP_COL = "trade_date"


def _make_dataset(
    n_groups: int = 12,
    rows_per_group: int = 6,
    seed: int = 7,
    include_timestamps: bool = True,
    violate_causality: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime([f"2024-03-{d:02d}" for d in range(1, n_groups + 1)])
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.extend({"trade_date": date} for _ in range(rows_per_group))
    df = pd.DataFrame(rows)
    df["feature_a"] = rng.normal(size=len(df))
    df["feature_b"] = rng.normal(size=len(df))
    df["net_return"] = 0.01 * df["feature_a"] + rng.normal(scale=0.004, size=len(df))
    df["selection_rank"] = df.groupby(GROUP_COL, sort=False).cumcount() + 1
    if include_timestamps:
        df["decision_timestamp"] = df[GROUP_COL].map(
            lambda d: pd.Timestamp(d, tz="Asia/Seoul").replace(hour=15, minute=30)
        )
        df["feature_available_timestamp"] = df["decision_timestamp"]
        if violate_causality:
            df.loc[df.index[0], "feature_available_timestamp"] = pd.Timestamp(
                "2024-03-02 09:00:00", tz="Asia/Seoul"
            )
    return df


@pytest.mark.parametrize(
    "model_type", ["lgb_ranker", "lgb_regressor", "ridge"]
)
def test_run_model_pipeline_returns_contract_shapes(model_type: str) -> None:
    """OOF 출력이 baseline 컬럼과 fold 출처를 보존해 동일 날짜 비교를 지원합니다."""
    df = _make_dataset()
    result = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        model_type=model_type,
    )
    assert "oof_predictions" in result
    assert "metrics" in result

    oof = result["oof_predictions"]
    assert isinstance(oof, pd.DataFrame)
    assert 0 < len(oof) <= len(df)
    assert oof.index.is_unique
    assert set(oof.index) <= set(df.index)
    assert {"pred", "relevance", "selection_rank", "pred_linear", "fold", GROUP_COL, TARGET_COL} <= set(
        oof.columns
    )

    assert result["oof_df"] is oof
    assert len(result["trained_models"]) == 3
    assert result["return_unit"] == "decimal_net"
    assert "feature_manifest" in result
    assert "training_cutoff" in result
    assert result["policy_params"]["purge_gap"] == 1
    assert result["backtest_eval"]["baseline_metrics"]["selection_rank"] is not None
    assert "linear" in result["backtest_eval"]["baseline_metrics"]


def test_run_model_pipeline_rejects_missing_timestamps() -> None:
    df = _make_dataset(include_timestamps=False)
    with pytest.raises(ValueError, match="timestamp column"):
        run_model_pipeline(
            df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
        )


def test_run_model_pipeline_rejects_naive_timestamps() -> None:
    df = _make_dataset()
    df["decision_timestamp"] = df["decision_timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        run_model_pipeline(
            df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
        )


def test_run_model_pipeline_rejects_non_causal_rows() -> None:
    df = _make_dataset(violate_causality=True)
    with pytest.raises(ValueError, match="non-causal"):
        run_model_pipeline(
            df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
        )


@pytest.mark.parametrize(
    "model_type", ["lgb_ranker", "lgb_regressor", "ridge"]
)
def test_run_model_pipeline_metrics_are_valid(model_type: str) -> None:
    df = _make_dataset()
    metrics = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        model_type=model_type,
    )["metrics"]
    for key in ("ndcg_1", "ndcg_3", "rank_ic", "top_1_return", "top_3_return"):
        assert key in metrics
        assert np.isfinite(metrics[key])
    assert 0.0 <= metrics["ndcg_1"] <= 1.0
    assert 0.0 <= metrics["ndcg_3"] <= 1.0


def test_run_model_pipeline_is_deterministic() -> None:
    df = _make_dataset(seed=11)
    kwargs = {
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "group_col": GROUP_COL,
        "n_splits": 3,
        "purge_gap": 1,
        "model_type": "lgb_ranker",
    }
    first = run_model_pipeline(df, **kwargs)
    second = run_model_pipeline(df, **kwargs)
    assert first["metrics"]["rank_ic"] == second["metrics"]["rank_ic"]
    np.testing.assert_array_equal(
        first["oof_predictions"]["pred"].to_numpy(),
        second["oof_predictions"]["pred"].to_numpy(),
    )


def test_run_model_pipeline_signal_ranking_is_positive() -> None:
    """feature_a 가 순수익률과 양의 관계 → Rank IC / NDCG 가 무의미한 수준 이하 금지."""
    df = _make_dataset(seed=3)
    metrics = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        model_type="ridge",
    )["metrics"]
    assert metrics["rank_ic"] > 0.1
    assert metrics["ndcg_3"] > 0.5


def test_run_model_pipeline_rejects_unknown_model_type() -> None:
    df = _make_dataset()
    with pytest.raises(ValueError, match="model_type must be one of"):
        run_model_pipeline(
            df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            model_type="svm",
        )


def test_run_model_pipeline_rejects_missing_columns() -> None:
    df = _make_dataset().drop(columns=["feature_b"])
    with pytest.raises(ValueError, match="missing columns"):
        run_model_pipeline(
            df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
        )


def test_run_sizing_pipeline_exports_model_bundle(tmp_path) -> None:
    """훈련 모드(export_dir)에서 모델 번들을 저장하고 artifact_path 를 반환한다."""
    df = _make_dataset(n_groups=8, rows_per_group=6)
    result = run_sizing_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        export_dir=str(tmp_path),
    )
    assert {"quantile_df", "sizing_df", "artifact_path"}.issubset(result.keys())
    assert {"utility_score", "grade", "grade_multiplier", "allocation"}.issubset(
        result["sizing_df"].columns
    )
    assert tmp_path.joinpath("sizing_pipeline_bundle.joblib").is_file()
    assert os.path.exists(result["artifact_path"])
    loaded = load_model_artifacts(str(tmp_path))
    assert set(loaded["feature_cols"]) == set(FEATURE_COLS)
    assert loaded["return_unit"] == "decimal_net"
    assert loaded["round_trip_cost"] == 0.002
    assert loaded["label_thresholds"] == {"target_good": 0.01, "target_bad": -0.02}
    assert "feature_manifest" in loaded
    assert "training_cutoff" in loaded
    assert "calibration_diagnostics" in loaded
    assert "policy_params" in loaded
