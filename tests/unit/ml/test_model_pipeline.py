"""Model Pipeline (Purged Walk-Forward CV 학습 + OOF 평가) 단위 테스트.

SCENARIO_MODEL_PIPELINE_TRAIN_EVAL
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.ml.model_pipeline import (
    _align_close_morning_oof,
    _calibrate_close_morning_decision_oof,
    _calibrate_oof_policy,
    _close_morning_yearly_breakdown,
    _dominant_recency_config,
    _fit_predict,
    _select_bad_probability_weight,
    _select_recency_ensemble_config,
    calculate_recency_sample_weight,
    evaluate_close_morning_quality,
    run_close_morning_recency_ensemble_experiment,
    run_close_morning_reranker_v2_experiment,
    run_model_pipeline,
    run_sizing_pipeline,
)
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
    """타임스탬프가 없는 현재 후보 패널로도 chronological training 이 성공합니다.

    시간은 고정된 업무 원천 규칙이며 모델 입력·CV 분할·artifact 승인 조건이
    아니므로, 타임스탬프 컬럼이 없어도 walk-forward 학습이 동작해야 합니다.
    """
    df = _make_dataset(include_timestamps=False)
    result = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
    )
    oof = result["oof_predictions"]
    assert 0 < len(oof) <= len(df)
    assert set(oof.columns) >= {"pred", GROUP_COL, TARGET_COL, "fold"}
    assert result["metrics"]["ndcg_1"] >= 0.0


def test_run_model_pipeline_retains_metadata_columns_in_oof() -> None:
    """OOF 에 stock_code/market_type/market_cap_100m 가 보존되어 시장구분·시총 분석이 가능합니다."""
    df = _make_dataset()
    df["stock_code"] = [f"{i:06d}" for i in range(len(df))]
    df["market_type"] = ["KOSPI" if i % 2 == 0 else "KOSDAQ" for i in range(len(df))]
    df["market_cap_100m"] = np.linspace(100.0, 3_000.0, len(df))
    oof = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
    )["oof_predictions"]
    assert {"stock_code", "market_type", "market_cap_100m"}.issubset(oof.columns)


def test_run_model_pipeline_retains_scenario_metadata_in_oof() -> None:
    """OOF 에 chart_analysis 와 시나리오 context 컬럼이 보존됩니다."""
    df = _make_dataset()
    df["chart_analysis"] = ["상따" if i % 3 == 0 else "신고가" for i in range(len(df))]
    df["scenario_count_for_stock_date"] = 1
    df["has_sangtta_for_stock_date"] = (np.arange(len(df)) % 3 == 0).astype(int)
    df["is_multi_scenario_stock_date"] = 0
    oof = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
    )["oof_predictions"]
    assert {
        "chart_analysis",
        "scenario_count_for_stock_date",
        "has_sangtta_for_stock_date",
        "is_multi_scenario_stock_date",
    }.issubset(oof.columns)
    assert oof["chart_analysis"].notna().all()


def test_run_model_pipeline_calibrates_single_stock_policy_on_scenario_panel() -> None:
    """ranker OOF 생성 직후 단일 종목 정책이 인과적으로 보정·영속화됩니다."""
    from src.ml.single_stock_policy import SingleStockPolicy

    df = _make_dataset(n_groups=10, rows_per_group=5, seed=13)
    df["stock_code"] = [f"{i % 5 + 1:06d}" for i in range(len(df))]
    df["chart_analysis"] = ["거래량 폭증", "신고가", "상따", "120 돌파", "신고가 근접"] * (
        len(df) // 5
    )
    df["market_type"] = ["KOSPI" if i % 2 == 0 else "KOSDAQ" for i in range(len(df))]
    result = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
    )
    policy = result["single_stock_policy"]
    assert policy is not None
    assert isinstance(policy, SingleStockPolicy)
    assert policy.candidate in {
        "always_buy_top1",
        "margin_quantile.0.70",
        "margin_quantile.0.90",
    }
    evaluation = result["single_stock_evaluation"]
    assert evaluation is not None
    assert len(evaluation.decisions) == len(result["oof_predictions"][GROUP_COL].unique())

    # 번들 준비 메타데이터: pred→rank_score 매핑과 보정 cutoff 를 명시 기록합니다.
    metadata = result["policy_metadata"]
    assert metadata is not None
    assert metadata["oof_score_col"] == "pred"
    assert metadata["daily_score_col"] == "rank_score"
    assert metadata["calibration_cutoff"] == str(policy.calibration_cutoff)
    assert metadata["policy_version"] == policy.version
    assert metadata["policy_id"] == policy.policy_id
    assert metadata["candidate"] == policy.candidate
    assert "scheduled_mean_return" in metadata["policy_metrics"]


def test_run_model_pipeline_policy_metadata_none_without_identity_columns() -> None:
    """stock_code/chart_analysis 가 없으면 정책 메타데이터는 None (명시 ABSTAIN)."""
    df = _make_dataset()
    result = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
    )
    assert result["single_stock_policy"] is None
    assert result["policy_metadata"] is None


def _aligned_return_oof() -> pd.DataFrame:
    return pd.DataFrame(
        {
            GROUP_COL: ["2024-03-01", "2024-03-01"],
            TARGET_COL: [0.01, 0.02],
            "pred": [0.005, 0.01],
            "stock_code": ["000001", "000002"],
            "chart_analysis": ["상따", "신고가"],
        },
        index=[5, 7],
    )


def _aligned_risk_oof(extra_row: bool = True) -> pd.DataFrame:
    index = [5, 7] if not extra_row else [5, 7, 9]
    rows = {
        GROUP_COL: ["2024-03-01", "2024-03-01"],
        TARGET_COL: [0.01, 0.02],
        "p_good": [0.6, 0.4],
        "p_bad": [0.3, 0.1],
        "stock_code": ["000001", "000002"],
        "chart_analysis": ["상따", "신고가"],
    }
    if extra_row:
        rows[GROUP_COL].append("2024-03-02")
        rows[TARGET_COL].append(0.03)
        rows["p_good"].append(0.7)
        rows["p_bad"].append(0.2)
        rows["stock_code"].append("000003")
        rows["chart_analysis"].append("거래량 폭증")
    return pd.DataFrame(rows, index=index)


def test_align_close_morning_oof_aligns_on_original_index() -> None:
    """return OOF 와 risk OOF 는 날짜가 아닌 원본 행 인덱스로 정렬됩니다."""
    aligned = _align_close_morning_oof(
        _aligned_return_oof(), _aligned_risk_oof(), target_col=TARGET_COL, group_col=GROUP_COL
    )
    assert aligned.index.tolist() == [5, 7]
    assert {"pred", "p_good", GROUP_COL, TARGET_COL, "stock_code", "chart_analysis"} <= set(
        aligned.columns
    )
    np.testing.assert_allclose(aligned["pred"].to_numpy(), [0.005, 0.01])
    np.testing.assert_allclose(aligned["p_good"].to_numpy(), [0.6, 0.4])


def test_align_close_morning_oof_rejects_missing_index() -> None:
    """return 예측 인덱스가 risk OOF 에 없으면 fail-closed 합니다 (누락 대체 금지)."""
    risk = _aligned_risk_oof(extra_row=False).drop(index=[7])
    with pytest.raises(ValueError, match="missing from quantile/classifier OOF"):
        _align_close_morning_oof(
            _aligned_return_oof(), risk, target_col=TARGET_COL, group_col=GROUP_COL
        )


def test_align_close_morning_oof_rejects_group_mismatch() -> None:
    """같은 인덱스에서 trade_date 가 어긋나면 날짜 단독 병합을 금지합니다."""
    risk = _aligned_risk_oof()
    risk.loc[5, GROUP_COL] = "2024-03-02"
    with pytest.raises(ValueError, match="trade_date mismatch"):
        _align_close_morning_oof(
            _aligned_return_oof(), risk, target_col=TARGET_COL, group_col=GROUP_COL
        )


def test_align_close_morning_oof_rejects_target_return_mismatch() -> None:
    """같은 인덱스에서 타깃 수익률이 어긋나면 fail-closed 합니다."""
    risk = _aligned_risk_oof()
    risk.loc[7, TARGET_COL] = 0.99
    with pytest.raises(ValueError, match="net_return mismatch"):
        _align_close_morning_oof(
            _aligned_return_oof(), risk, target_col=TARGET_COL, group_col=GROUP_COL
        )


def test_align_close_morning_oof_rejects_stock_code_mismatch() -> None:
    """같은 인덱스에서 stock_code 가 어긋나면 fail-closed 합니다."""
    risk = _aligned_risk_oof()
    risk.loc[7, "stock_code"] = "000099"
    with pytest.raises(ValueError, match="stock_code mismatch"):
        _align_close_morning_oof(
            _aligned_return_oof(), risk, target_col=TARGET_COL, group_col=GROUP_COL
        )


def test_align_close_morning_oof_rejects_chart_analysis_mismatch() -> None:
    """같은 인덱스에서 chart_analysis 가 어긋나면 fail-closed 합니다."""
    risk = _aligned_risk_oof()
    risk.loc[5, "chart_analysis"] = "상한가 다음날"
    with pytest.raises(ValueError, match="chart_analysis mismatch"):
        _align_close_morning_oof(
            _aligned_return_oof(), risk, target_col=TARGET_COL, group_col=GROUP_COL
        )


def test_align_close_morning_oof_rejects_missing_p_good() -> None:
    """정렬 결과에 누락 p_good 가 있으면 대체하지 않고 거부합니다."""
    risk = _aligned_risk_oof()
    risk.loc[7, "p_good"] = float("nan")
    with pytest.raises(ValueError, match="p_good predictions are missing"):
        _align_close_morning_oof(
            _aligned_return_oof(), risk, target_col=TARGET_COL, group_col=GROUP_COL
        )


def test_align_close_morning_oof_skips_checks_when_columns_absent() -> None:
    """risk OOF 에 타깃/식별 컬럼이 없으면 해당 정렬 검증을 건너뜁니다 (p_good 필수만)."""
    risk = pd.DataFrame(
        {
            GROUP_COL: ["2024-03-01", "2024-03-01"],
            "p_good": [0.6, 0.4],
            "p_bad": [0.3, 0.1],
        },
        index=[5, 7],
    )
    aligned = _align_close_morning_oof(
        _aligned_return_oof(), risk, target_col=TARGET_COL, group_col=GROUP_COL
    )
    assert {"pred", "p_good", "p_bad", "stock_code", "chart_analysis"} <= set(
        aligned.columns
    )
    assert aligned["p_good"].notna().all()
    assert aligned["p_bad"].notna().all()


def test_align_close_morning_oof_rejects_missing_p_bad() -> None:
    """정렬 결과에 누락 p_bad 가 있으면 대체하지 않고 거부합니다."""
    risk = _aligned_risk_oof()
    risk.loc[7, "p_bad"] = float("nan")
    with pytest.raises(ValueError, match="p_bad predictions are missing"):
        _align_close_morning_oof(
            _aligned_return_oof(), risk, target_col=TARGET_COL, group_col=GROUP_COL
        )


def test_calibrate_close_morning_decision_oof_uses_decision_score_end_to_end() -> None:
    """reranker OOF 보정은 decision_score 를 스코어로 사용하고 정책/메타데이터에
    decision_score 매핑을 기록합니다."""
    df = _make_dataset(n_groups=10, rows_per_group=5, seed=13)
    df["stock_code"] = [f"{i % 5 + 1:06d}" for i in range(len(df))]
    df["chart_analysis"] = ["거래량 폭증", "신고가", "상따", "120 돌파", "신고가 근접"] * (
        len(df) // 5
    )
    df["market_type"] = ["KOSPI" if i % 2 == 0 else "KOSDAQ" for i in range(len(df))]
    policy, evaluation, metadata = _calibrate_close_morning_decision_oof(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
    )
    assert policy is not None
    assert policy.score_col == "decision_score"
    assert evaluation is not None
    assert metadata is not None
    assert metadata["oof_score_col"] == "decision_score"
    assert metadata["daily_score_col"] == "decision_score"
    assert metadata["candidate"] == policy.candidate


def test_calibrate_close_morning_decision_oof_returns_none_without_identity() -> None:
    """식별 컬럼(stock_code/chart_analysis)이 없으면 (None, None, None) 을 반환합니다."""
    df = _make_dataset()
    policy, evaluation, metadata = _calibrate_close_morning_decision_oof(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
    )
    assert (policy, evaluation, metadata) == (None, None, None)


def test_calibrate_oof_policy_reranker_returns_decision_score_metadata() -> None:
    """_calibrate_oof_policy(reranker=True) 는 decision_score 매핑을 반환합니다."""
    df = _make_dataset(n_groups=10, rows_per_group=5, seed=13)
    df["stock_code"] = [f"{i % 5 + 1:06d}" for i in range(len(df))]
    df["chart_analysis"] = ["거래량 폭증"] * len(df)
    policy, metadata = _calibrate_oof_policy(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        reranker=True,
    )
    assert policy is not None
    assert policy.score_col == "decision_score"
    assert metadata is not None
    assert metadata["oof_score_col"] == "decision_score"
    assert metadata["daily_score_col"] == "decision_score"


def test_run_model_pipeline_passes_model_params_to_requested_model() -> None:
    """model_params 는 요청된 모델에만 전달되고 random_state=42 가 유지됩니다."""
    df = _make_dataset()
    params = {"learning_rate": 0.02, "num_leaves": 7}
    first = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
        model_params=params,
    )
    second = run_model_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        model_type="lgb_regressor",
        model_params=params,
    )
    np.testing.assert_array_equal(
        first["oof_predictions"]["pred"].to_numpy(),
        second["oof_predictions"]["pred"].to_numpy(),
    )
    assert all(getattr(m, "learning_rate", None) == 0.02 for m in first["trained_models"])


def test_run_model_pipeline_rejects_negative_purge_gap() -> None:
    df = _make_dataset()
    with pytest.raises(ValueError, match="purge_gap"):
        run_model_pipeline(
            df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            purge_gap=-1,
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


def test_run_sizing_pipeline_export_persists_single_stock_policy(tmp_path) -> None:
    """export_dir 훈련 모드는 시나리오 패널이 있으면 정책을 번들에 영속화합니다."""
    df = _make_dataset(n_groups=8, rows_per_group=6, seed=3)
    df["stock_code"] = [f"{i % 6 + 1:06d}" for i in range(len(df))]
    df["chart_analysis"] = ["거래량 폭증"] * len(df)
    result = run_sizing_pipeline(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=3,
        purge_gap=1,
        export_dir=str(tmp_path),
    )
    loaded = load_model_artifacts(str(tmp_path))
    assert loaded["single_stock_policy"] is not None
    assert loaded["policy_metadata"]["oof_score_col"] == "pred"
    assert loaded["policy_metadata"]["daily_score_col"] == "rank_score"
    assert loaded["oof_score_col"] == "pred"
    assert loaded["daily_score_col"] == "rank_score"


def _report_raw_df(n_dates: int = 4, n_candidates: int = 8, seed: int = 5) -> pd.DataFrame:
    """close-morning 품질 보고 테스트용 원본 매매일지(스프레드시트 헤더)."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for d_idx in range(n_dates):
        date = pd.Timestamp(f"2024-0{1 + d_idx}-{5 + d_idx:02d}")
        for c in range(n_candidates):
            prev = 10_000.0 + rng.normal(0, 500)
            open_p = prev * (1 + rng.normal(0, 0.01))
            close = prev * (1 + rng.normal(0, 0.02))
            rows.append(
                {
                    "매수날짜": date,
                    "종목코드": f"{c + 1:06d}",
                    "(시가)": open_p,
                    "(고가)": max(open_p, close) * (1 + abs(rng.normal(0, 0.01))),
                    "(저가)": min(open_p, close) * (1 - abs(rng.normal(0, 0.01))),
                    "(종가)": close,
                    "(전일종가)": prev,
                    "(시가총액, 억)": rng.uniform(300, 3_000),
                    "(거래대금, 억)": rng.uniform(50, 800),
                    "(등락률)": rng.normal(2, 8),
                    "(선정 순위)": float(c + 1),
                    "(기관_순매수)": rng.normal(0, 1e8),
                    "(외국인_순매수)": rng.normal(0, 1e8),
                    "(프로그램_순매수)": rng.normal(0, 5e7),
                    "(체결강도)": rng.uniform(80, 200),
                    "(시장구분)": "KOSPI" if c % 2 == 0 else "KOSDAQ",
                    "(총 종목 수)": float(n_candidates),
                    "(평균 거래대금)": rng.uniform(50, 800),
                    "(kospi, %)": rng.normal(0, 1),
                    "(kosdaq, %)": rng.normal(0, 1),
                    "v_kospi": rng.uniform(12, 25),
                    "v_kosdaq": rng.uniform(12, 25),
                    "(거래량)": rng.uniform(1e5, 5e6),
                    "(테마/섹터)": rng.choice(["테마A", "테마B", "테마C"]),
                    "(차트분석)": rng.choice(["거래량 폭증", "신고가 근접", "상한가 다음날"]),
                    "(매수 가격)": prev * 1.01,
                    "(매도 가격)": prev * 1.03,
                    "(수익률, %)": rng.normal(1.0, 4.0),
                }
            )
    return pd.DataFrame(rows)


def _fake_quality_pipeline_result(scheduled_mean: float) -> dict[str, Any]:
    """evaluate_close_morning_quality 단위 테스트용 run_model_pipeline 페이크 결과."""
    from types import SimpleNamespace

    evaluation = SimpleNamespace(
        metrics={
            "scheduled_mean_return": scheduled_mean,
            "scheduled_win_rate": 0.5,
            "profit_factor": 2.0,
            "scheduled_sharpe": 3.0,
            "active_trade_mean_return": scheduled_mean,
            "active_trade_win_rate": 0.5,
            "n_buy": 100,
            "n_abstain": 20,
            "reason_counts": {"top1_buy": 100, "insufficient_policy_history": 20},
        },
        scheduled_returns=np.array([0.01, -0.005, 0.02, 0.0, 0.01]),
        selected_policy=SimpleNamespace(candidate="always_buy_top1"),
    )
    return {
        "metrics": {"ndcg_1": 0.5, "rank_ic": 0.1},
        "oof_df": pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-05")] * 5}),
        "single_stock_evaluation": evaluation,
    }


def _fake_decision_policy_result(
    scheduled_mean: float,
) -> tuple[Any, Any, dict[str, Any]]:
    """evaluate_close_morning_quality 단위 테스트용 reranker OOF 보정 페이크 결과."""
    from types import SimpleNamespace

    evaluation = SimpleNamespace(
        metrics={
            "scheduled_mean_return": scheduled_mean,
            "scheduled_win_rate": 0.6,
            "profit_factor": 2.5,
            "scheduled_sharpe": 4.0,
            "active_trade_mean_return": scheduled_mean,
            "active_trade_win_rate": 0.6,
            "n_buy": 90,
            "n_abstain": 30,
            "reason_counts": {"top1_buy": 90, "insufficient_policy_history": 30},
        },
        scheduled_returns=np.array([0.02, -0.01, 0.015, 0.0, 0.01]),
        selected_policy=SimpleNamespace(candidate="always_buy_top1"),
    )
    policy = SimpleNamespace(candidate="always_buy_top1", score_col="decision_score")
    metadata = {
        "oof_score_col": "decision_score",
        "daily_score_col": "decision_score",
        "candidate": "always_buy_top1",
    }
    return policy, evaluation, metadata


def test_evaluate_close_morning_quality_reports_close_to_morning_metrics() -> None:
    """보고서가 동일 OOF 날짜에서 피처셋 비교와 100점 스코어를 반환합니다.

    MDD 는 close-to-morning 전략 지표로 명명되며 entry-sequence 프록시가
    아닙니다. 모든 지표는 decimal-net 단위입니다.
    """
    from unittest.mock import patch

    raw = _report_raw_df()
    fake = _fake_quality_pipeline_result(scheduled_mean=0.015)
    decision = _fake_decision_policy_result(scheduled_mean=0.015)
    with (
        patch("src.ml.training.pipelines.run_model_pipeline", return_value=fake),
        patch(
            "src.ml.training.pipelines._calibrate_close_morning_decision_oof",
            return_value=decision,
        ),
    ):
        report = evaluate_close_morning_quality(raw, n_splits=2, purge_gap=1)

    assert set(report["feature_sets"]) == {"base40", "snapshot49", "close_morning61"}
    entry = report["report"]["close_morning61"]
    assert "close_to_morning_mdd" in entry
    assert "entry_sequence_drawdown" not in entry
    assert entry["top1_net_mean"] == pytest.approx(0.015)
    assert entry["scheduled_mean_return"] == pytest.approx(0.015)
    assert entry["reason_counts"]["top1_buy"] == 90
    assert 0 <= entry["n_buy"] <= entry["n_buy"] + entry["n_abstain"]

    score = report["quality_score"]["close_morning61"]
    assert 0 <= score["total"] <= 100
    assert set(score["components"]) == {
        "selection_edge",
        "net_economics",
        "risk_and_stability",
        "validation_independence",
        "daily_deployment_integrity",
    }


def test_evaluate_close_morning_quality_exposes_legacy_and_reranker_metrics() -> None:
    """champion 피처셋은 레거시 rank-only 와 decision-score reranker 정책 지표를
    명확한 이름으로 함께 노출하고, 후보 지표는 decision-score 정책입니다."""
    from unittest.mock import patch

    raw = _report_raw_df()
    fake = _fake_quality_pipeline_result(scheduled_mean=0.011)
    decision = _fake_decision_policy_result(scheduled_mean=0.019)
    with (
        patch("src.ml.training.pipelines.run_model_pipeline", return_value=fake),
        patch(
            "src.ml.training.pipelines._calibrate_close_morning_decision_oof",
            return_value=decision,
        ),
    ):
        report = evaluate_close_morning_quality(raw, n_splits=2, purge_gap=1)

    entry = report["report"]["close_morning61"]
    # 후보 지표 = decision-score reranker 정책
    assert entry["top1_net_mean"] == pytest.approx(0.019)
    assert entry["policy_candidate"] == "always_buy_top1"
    assert entry["reason_counts"]["top1_buy"] == 90
    # 레거시 rank-only 지표는 legacy_ 접두어로 함께 노출됩니다.
    assert entry["legacy_top1_net_mean"] == pytest.approx(0.011)
    assert entry["legacy_sharpe"] == pytest.approx(3.0)
    assert entry["legacy_policy_candidate"] == "always_buy_top1"
    # 그 외 피처셋은 레거시 지표를 후보로 사용하고 legacy_ 접두어도 노출합니다.
    legacy_entry = report["report"]["snapshot49"]
    assert legacy_entry["top1_net_mean"] == pytest.approx(0.011)
    assert legacy_entry["legacy_top1_net_mean"] == pytest.approx(0.011)


def test_evaluate_close_morning_quality_rejects_non_finite_metrics() -> None:
    """비유한 지표(예: scheduled_mean_return=NaN)는 ValueError 로 fail-closed 합니다."""
    from unittest.mock import patch

    raw = _report_raw_df()
    fake = _fake_quality_pipeline_result(scheduled_mean=float("nan"))
    decision = _fake_decision_policy_result(scheduled_mean=float("nan"))
    with (
        patch("src.ml.training.pipelines.run_model_pipeline", return_value=fake),
        patch(
            "src.ml.training.pipelines._calibrate_close_morning_decision_oof",
            return_value=decision,
        ),
        pytest.raises(ValueError, match="non-finite"),
    ):
        evaluate_close_morning_quality(raw, n_splits=2, purge_gap=1)


def test_evaluate_close_morning_quality_missing_policy_rejects() -> None:
    """정책 미보정 상태(식별 컬럼 부재)는 ABSTAIN 메타데이터와 함께 거부됩니다."""
    from unittest.mock import patch

    raw = _report_raw_df()
    fake = _fake_quality_pipeline_result(scheduled_mean=0.015)
    fake["single_stock_evaluation"] = None
    with (
        patch("src.ml.training.pipelines.run_model_pipeline", return_value=fake),
        patch(
            "src.ml.training.pipelines._calibrate_close_morning_decision_oof",
            return_value=(None, None, None),
        ),
        pytest.raises(ValueError, match="non-finite"),
    ):
        evaluate_close_morning_quality(raw, n_splits=2, purge_gap=1)

def _make_reranker_v2_dataset(seed: int = 7, n_groups: int = 52) -> pd.DataFrame:
    """p_bad 패널티가 MDD 만 낮출 수 있는 합성 순서형 패널을 만듭니다.

    ``feature_b > 0.7`` 인 'lottery' 종목은 안전 종목과 동일한 기대수익을
    가지면서 큰 꼬리손실을 가져, 손실 확률(p_bad) 패널티가 평균을 낮추지 않고
    최대 드로다운을 줄이는 구조입니다. ``selection_rank`` 를 포함해
    ``run_model_pipeline`` 의 백테스트 계약을 충족합니다.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_groups, freq="D")
    rows: list[dict[str, object]] = []
    for g, date in enumerate(dates):
        rows.extend(
            {"trade_date": date, "stock_code": f"{g * 6 + i + 1:06d}"}
            for i in range(6)
        )
    df = pd.DataFrame(rows)
    df["feature_a"] = rng.uniform(0, 1, len(df))
    df["feature_b"] = rng.uniform(0, 1, len(df))
    df["chart_analysis"] = ["신고가"] * len(df)
    df["market_type"] = "KOSPI"
    df["selection_rank"] = df.groupby(GROUP_COL, sort=False).cumcount() + 1
    mu = 0.006 + 0.012 * df["feature_a"]
    lottery = df["feature_b"] > 0.7
    up = rng.random(len(df)) < 0.6
    lottery_target = np.where(up, mu + 0.04, mu - 0.06)
    target = np.where(lottery, lottery_target, mu) + rng.normal(0, 0.004, len(df))
    df[TARGET_COL] = np.clip(target, -0.25, 0.25)
    return df


def test_select_bad_probability_weight_rules() -> None:
    """v2 패널티 선택은 보수적 규칙을 준수합니다: 평균 보존 + MDD 엄격 감소만
    유효하며, 최저 MDD → 높은 평균 → 낮은 가중치 순으로 타이브레이크합니다."""
    base = {"scheduled_mean_return": 0.01, "entry_sequence_drawdown": 0.30}
    # MDD 가 동일하면 비영 후보는 미유효 (엄격 감소 요구).
    stats = {
        0.0: dict(base),
        0.5: {"scheduled_mean_return": 0.012, "entry_sequence_drawdown": 0.30},
    }
    assert _select_bad_probability_weight(stats) == 0.0
    # 평균이 v1 미만이면 미유효.
    stats = {
        0.0: dict(base),
        0.5: {"scheduled_mean_return": 0.009, "entry_sequence_drawdown": 0.10},
    }
    assert _select_bad_probability_weight(stats) == 0.0
    # 평균 보존 + MDD 엄격 감소면 선택됩니다.
    stats = {
        0.0: dict(base),
        0.5: {"scheduled_mean_return": 0.012, "entry_sequence_drawdown": 0.20},
    }
    assert _select_bad_probability_weight(stats) == 0.5
    # 최저 MDD 가 우선입니다.
    stats = {
        0.0: dict(base),
        0.5: {"scheduled_mean_return": 0.011, "entry_sequence_drawdown": 0.15},
        1.0: {"scheduled_mean_return": 0.013, "entry_sequence_drawdown": 0.10},
    }
    assert _select_bad_probability_weight(stats) == 1.0
    # MDD/평균 동점이면 낮은 가중치가 우선입니다.
    stats = {
        0.0: dict(base),
        0.5: {"scheduled_mean_return": 0.013, "entry_sequence_drawdown": 0.10},
        1.0: {"scheduled_mean_return": 0.013, "entry_sequence_drawdown": 0.10},
    }
    assert _select_bad_probability_weight(stats) == 0.5
    # NaN 지표는 미충족으로 간주되어 v1 로 fail-closed 합니다.
    stats = {
        0.0: dict(base),
        0.5: {"scheduled_mean_return": float("nan"), "entry_sequence_drawdown": 0.05},
    }
    assert _select_bad_probability_weight(stats) == 0.0


def test_close_morning_reranker_v2_nested_selection_is_causal() -> None:
    """(SCENARIO: test_calibrate_close_morning_decision_oof_uses_decision_score_end_to_end)

    v2 중첩 선택은 각 fold 의 외부 validation 스코어 설정을 그 fold 의 이전
    내부 OOF 역사에서만 고르고, 외부(미래) 수익률은 절대 읽지 않습니다.

    나중 날짜(외부 validation)의 실현 수익률이 선택된 설정을 바꿀 만큼 극단적
    (전량 -19%)으로 바뀌어도 폴드별 ``chosen_weight`` 는 그대로여야 합니다.
    해당 변경은 그 폴드의 외부 평가 지표에는 반영되어야 하므로, 선택이 외부
    validation 레이블을 읽지 않는다는 인과 경계를 증명합니다.
    """
    df = _make_reranker_v2_dataset(seed=7)
    # 마지막 outer fold 의 validation 날짜(어떤 fold 의 train 에도 없는 순수 미래
    # 구간)만 변경해 인과 경계를 검증합니다. test_size = n_groups // (n_splits + 1).
    future_dates = sorted(df[GROUP_COL].unique())[-(len(df[GROUP_COL].unique()) // 5) :]
    mutated = df.copy()
    mutated.loc[mutated[GROUP_COL].isin(future_dates), TARGET_COL] = -0.19

    base = run_close_morning_reranker_v2_experiment(
        df,
        feature_cols=["feature_a", "feature_b"],
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=4,
        min_history_dates=2,
    )
    altered = run_close_morning_reranker_v2_experiment(
        mutated,
        feature_cols=["feature_a", "feature_b"],
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=4,
        min_history_dates=2,
    )

    assert base["contract"]["version"] == "close-morning-reranker-v2-research"
    assert base["contract"]["candidate_weights"] == [0.0, 0.5, 1.0]
    assert len(base["folds"]) == 4
    assert base["chosen_weights"] == altered["chosen_weights"]
    # 적어도 하나의 폴드가 비영(非零) 패널티를 선택해야 테스트가 의미를 가집니다.
    assert any(weight > 0.0 for weight in base["chosen_weights"])
    for fold in base["folds"]:
        assert fold["chosen_weight"] in (0.0, 0.5, 1.0)
        assert set(fold["inner"]["candidate_stats"]) == {0.0, 0.5, 1.0}
        assert {"scheduled_mean_return", "entry_sequence_drawdown"} <= set(
            fold["inner"]["candidate_stats"][0.0]
        )
        assert "scheduled_mean_return" in fold["v1"]["metrics"]
        assert "entry_sequence_drawdown" in fold["v2"]["metrics"]
    assert set(base["aggregate"]) == {"v1", "v2"}
    assert (
        base["aggregate"]["v1"]["n_scheduled_dates"]
        == base["aggregate"]["v2"]["n_scheduled_dates"]
    )
    # 미래 수익률 변경은 해당 폴드의 외부 평가 지표에 반영됩니다 (관측 경계 존재).
    last = len(base["folds"]) - 1
    assert base["folds"][last]["v2"]["metrics"]["scheduled_mean_return"] != pytest.approx(
        altered["folds"][last]["v2"]["metrics"]["scheduled_mean_return"]
    )

def test_close_morning_reranker_v2_rejects_invalid_inputs() -> None:
    """v2 실험은 식별 컬럼 누락, 비정상 가중치/워밍업/purge 를 fail-closed 로 거부합니다."""
    df = _make_reranker_v2_dataset(seed=7, n_groups=16)
    with pytest.raises(ValueError, match="requires stock_code and chart_analysis"):
        run_close_morning_reranker_v2_experiment(
            df.drop(columns=["chart_analysis"]),
            feature_cols=["feature_a", "feature_b"],
            target_col=TARGET_COL,
            group_col=GROUP_COL,
        )
    with pytest.raises(ValueError, match="probability_weight must be in \\(0, 1\\]"):
        run_close_morning_reranker_v2_experiment(
            df,
            feature_cols=["feature_a", "feature_b"],
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            probability_weight=0.0,
        )
    with pytest.raises(ValueError, match="min_history_dates must be >= 1"):
        run_close_morning_reranker_v2_experiment(
            df,
            feature_cols=["feature_a", "feature_b"],
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            min_history_dates=0,
        )
    with pytest.raises(ValueError, match="purge_gap must be >= 0"):
        run_close_morning_reranker_v2_experiment(
            df,
            feature_cols=["feature_a", "feature_b"],
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            purge_gap=-1,
        )


def test_close_morning_reranker_v2_fails_closed_on_insufficient_inner_history() -> None:
    """내부 partition 이 중첩 walk-forward 를 지원할 만큼 충분하지 않으면 v1 로
    fail-closed 합니다 (w_bad=0 선택, 진단 사유 기록)."""
    df = _make_reranker_v2_dataset(seed=7, n_groups=8)
    report = run_close_morning_reranker_v2_experiment(
        df,
        feature_cols=["feature_a", "feature_b"],
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=5,
        min_history_dates=2,
    )
    assert report["folds"][0]["chosen_weight"] == 0.0
    assert (
        report["folds"][0]["inner"]["fail_closed_reason"] == "insufficient_inner_history"
    )
    assert report["folds"][0]["inner"]["candidate_stats"] == {}
    assert len(report["folds"]) == 5

def _make_recency_dataset(seed: int = 7, n_groups: int = 220, rows_per_group: int = 3) -> pd.DataFrame:
    """구간 전환(regime shift) 합성 패널: 전반 고노이즈 구간 / 후반 신호 구간.

    half-life recent 전문가가 초기 고노이즈 구간을 하향 가중해 MDD 를 줄이므로
    recency 앙상블 후보가 baseline(expanding) 을 개선할 수 있는 구조입니다.
    ``selection_rank`` 를 포함해 ``run_model_pipeline`` 의 백테스트 계약을
    충족합니다.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_groups, freq="D")
    rows: list[dict[str, object]] = []
    for g, date in enumerate(dates):
        rows.extend(
            {"trade_date": date, "stock_code": f"{g * rows_per_group + i + 1:06d}"}
            for i in range(rows_per_group)
        )
    df = pd.DataFrame(rows)
    n = len(df)
    df["feature_a"] = rng.uniform(-1, 1, n)
    df["feature_b"] = rng.uniform(-1, 1, n)
    df["chart_analysis"] = "신고가"
    df["market_type"] = "KOSPI"
    df["selection_rank"] = df.groupby(GROUP_COL, sort=False).cumcount() + 1
    positions = df.groupby(GROUP_COL).ngroup().to_numpy()
    frac = positions / n_groups
    signal = np.where(frac >= 0.5, 0.05 * df["feature_a"].to_numpy(), 0.0)
    noise = np.where(frac >= 0.5, 0.004, 0.03)
    df[TARGET_COL] = np.clip(signal + rng.normal(0, noise), -0.2, 0.2)
    return df


def test_calculate_recency_sample_weight_decay_formula_and_mean_one() -> None:
    """decay 공식과 mean-one 정규화, None(expanding) 동작을 검증합니다.

    ``w(age) = exp(-ln(2) * age / half_life)`` 를 정확히 반영하고 정규화 평균이
    1 이며 최신 그룹이 가장 높은 가중치를 받아야 합니다.
    """
    groups = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    weights = calculate_recency_sample_weight(groups, 252)
    assert weights.mean() == pytest.approx(1.0)
    ages = np.array([2.0, 1.0, 0.0])
    expected = np.exp(-np.log(2.0) * ages / 252.0)
    np.testing.assert_allclose(weights, expected / expected.mean())
    assert weights[-1] > weights[0]

    # None 은 기존 expanding 동작(전부 1) 을 반환합니다.
    np.testing.assert_array_equal(calculate_recency_sample_weight(groups, None), np.ones(3))
    # 504 half-life 는 같은 정규화 계약을 유지합니다.
    weights_504 = calculate_recency_sample_weight(groups, 504)
    assert weights_504.mean() == pytest.approx(1.0)
    assert weights_504[-1] > weights_504[0]


def test_calculate_recency_sample_weight_rejects_invalid_inputs() -> None:
    """미지원 half-life, 빈 그룹, 파싱 불가 그룹은 fail-closed 로 거부합니다."""
    groups = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"]))
    with pytest.raises(ValueError, match="one of None, 252, 504"):
        calculate_recency_sample_weight(groups, 100)
    with pytest.raises(ValueError, match="non-empty"):
        calculate_recency_sample_weight(pd.Series([], dtype="datetime64[ns]"), 252)
    with pytest.raises(ValueError, match="parseable"):
        calculate_recency_sample_weight(pd.Series(["2024-01-01", "not-a-date"]), 252)
    with pytest.raises(ValueError, match="parseable"):
        calculate_recency_sample_weight(pd.Series([pd.Timestamp("2024-01-01"), None]), 252)


def test_run_model_pipeline_recency_weights_are_fold_local() -> None:
    """recency 가중치는 fold 의 train 거래일에서만 계산됩니다.

    마지막 외부 validation 날짜의 실현 수익률을 극단(-19%)으로 바꿔도 훈련
    가중치가 바뀌지 않아 OOF 예측이 (LightGBM 스레드 감축 1-ULP 노이즈 이내로)
    동일해야 합니다 — 검증 라벨이 학습에 노출되지 않는다는 인과 경계를 증명합니다.
    """
    df = _make_recency_dataset(seed=7)
    n_groups = df[GROUP_COL].nunique()
    future_dates = sorted(df[GROUP_COL].unique())[-(n_groups // 3) :]
    mutated = df.copy()
    mutated.loc[mutated[GROUP_COL].isin(future_dates), TARGET_COL] = -0.19

    def _preds(data: pd.DataFrame) -> np.ndarray:
        return run_model_pipeline(
            data,
            feature_cols=["feature_a", "feature_b"],
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            n_splits=2,
            model_type="lgb_regressor",
            recency_half_life_groups=252,
        )["oof_predictions"]["pred"].to_numpy()

    np.testing.assert_allclose(_preds(df), _preds(mutated), atol=1e-9)


def test_run_model_pipeline_recency_weights_change_predictions() -> None:
    """half-life recent Huber 가 expanding 과 다른 OOF 예측을 산출합니다."""
    df = _make_recency_dataset(seed=7)
    expanding = run_model_pipeline(
        df,
        feature_cols=["feature_a", "feature_b"],
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=2,
        model_type="lgb_regressor",
    )["oof_predictions"]["pred"].to_numpy()
    recent = run_model_pipeline(
        df,
        feature_cols=["feature_a", "feature_b"],
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=2,
        model_type="lgb_regressor",
        recency_half_life_groups=252,
    )["oof_predictions"]["pred"].to_numpy()
    assert not np.allclose(expanding, recent)


def test_run_model_pipeline_rejects_invalid_recency_configuration() -> None:
    """Ridge/LGBMRanker 는 recency 가중치를 거부하고 미지원 half-life 는 오류입니다."""
    df = _make_dataset(n_groups=12)
    with pytest.raises(ValueError, match="only supported for lgb_regressor"):
        run_model_pipeline(
            df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            model_type="ridge",
            recency_half_life_groups=252,
        )
    with pytest.raises(ValueError, match="only supported for lgb_regressor"):
        run_model_pipeline(
            df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            model_type="lgb_ranker",
            recency_half_life_groups=504,
        )
    with pytest.raises(ValueError, match="one of None, 252, 504"):
        run_model_pipeline(
            df,
            feature_cols=FEATURE_COLS,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            recency_half_life_groups=100,
        )


def test_select_recency_ensemble_config_deterministic_ties() -> None:
    """recency 후보 선택은 보수적 규칙을 준수합니다: 평균 보존 + MDD 엄격 감소만
    유효하며, 낮은 MDD → 높은 mean → 낮은 recent_weight → 긴 half_life 순으로
    타이브레이크합니다. NaN 지표는 미충족으로 간주되어 v1 로 fail-closed 합니다."""
    base = {"scheduled_mean_return": 0.01, "entry_sequence_drawdown": 0.30}
    # MDD 가 동일하면 비영 후보는 미유효 (엄격 감소 요구).
    stats = {
        (None, 0.0): dict(base),
        (252, 0.5): {"scheduled_mean_return": 0.012, "entry_sequence_drawdown": 0.30},
    }
    assert _select_recency_ensemble_config(stats) == (None, 0.0)
    # 평균이 v1 미만이면 미유효.
    stats = {
        (None, 0.0): dict(base),
        (252, 0.5): {"scheduled_mean_return": 0.009, "entry_sequence_drawdown": 0.10},
    }
    assert _select_recency_ensemble_config(stats) == (None, 0.0)
    # 평균 보존 + MDD 엄격 감소면 선택됩니다.
    stats = {
        (None, 0.0): dict(base),
        (504, 0.5): {"scheduled_mean_return": 0.012, "entry_sequence_drawdown": 0.20},
    }
    assert _select_recency_ensemble_config(stats) == (504, 0.5)
    # 최저 MDD 가 우선입니다.
    stats = {
        (None, 0.0): dict(base),
        (252, 0.5): {"scheduled_mean_return": 0.011, "entry_sequence_drawdown": 0.15},
        (504, 1.0): {"scheduled_mean_return": 0.013, "entry_sequence_drawdown": 0.10},
    }
    assert _select_recency_ensemble_config(stats) == (504, 1.0)
    # MDD/mean 동점이면 낮은 recent_weight 가 우선, 같은 alpha 는 긴 half-life.
    stats = {
        (None, 0.0): dict(base),
        (252, 0.75): {"scheduled_mean_return": 0.013, "entry_sequence_drawdown": 0.10},
        (504, 0.75): {"scheduled_mean_return": 0.013, "entry_sequence_drawdown": 0.10},
        (504, 0.25): {"scheduled_mean_return": 0.013, "entry_sequence_drawdown": 0.10},
    }
    assert _select_recency_ensemble_config(stats) == (504, 0.25)
    # NaN 지표는 미충족으로 간주되어 v1 로 fail-closed 합니다.
    stats = {
        (None, 0.0): dict(base),
        (252, 0.5): {"scheduled_mean_return": float("nan"), "entry_sequence_drawdown": 0.05},
    }
    assert _select_recency_ensemble_config(stats) == (None, 0.0)


def test_close_morning_recency_ensemble_nested_selection_is_causal() -> None:
    """recency 앙상블 중첩 선택은 외부 validation 레이블을 읽지 않습니다.

    나중 날짜(마지막 외부 validation fold)의 실현 수익률이 -19% 로 바뀌어도
    폴드별 ``chosen_config`` 는 그대로여야 합니다. 해당 변경은 그 폴드의 외부
    평가 지표에는 반영되어야 하므로, 선택이 외부 validation 레이블을 읽지
    않는다는 인과 경계를 증명합니다.
    """
    df = _make_recency_dataset(seed=7)
    n_groups = df[GROUP_COL].nunique()
    future_dates = sorted(df[GROUP_COL].unique())[-(n_groups // 3) :]
    mutated = df.copy()
    mutated.loc[mutated[GROUP_COL].isin(future_dates), TARGET_COL] = -0.19

    base = run_close_morning_recency_ensemble_experiment(
        df,
        feature_cols=["feature_a", "feature_b"],
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=2,
        min_history_dates=2,
    )
    altered = run_close_morning_recency_ensemble_experiment(
        mutated,
        feature_cols=["feature_a", "feature_b"],
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=2,
        min_history_dates=2,
    )

    assert base["contract"]["version"] == "close-morning-recency-ensemble-research"
    assert base["contract"]["half_lives"] == [252, 504]
    assert base["contract"]["alphas"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert len(base["folds"]) == 2
    assert base["chosen_configs"] == altered["chosen_configs"]
    # 적어도 하나의 폴드가 비-baseline 구성을 선택해야 테스트가 의미를 가집니다.
    assert any(config["half_life"] is not None for config in base["chosen_configs"])
    for fold in base["folds"]:
        assert fold["chosen_config"]["half_life"] in (None, 252, 504)
        assert fold["chosen_config"]["recent_weight"] in (0.0, 0.25, 0.5, 0.75, 1.0)
        assert (None, 0.0) in fold["inner"]["candidate_stats"]
        assert {"scheduled_mean_return", "entry_sequence_drawdown"} <= set(
            fold["inner"]["candidate_stats"][(None, 0.0)]
        )
        assert "scheduled_mean_return" in fold["baseline"]["metrics"]
        assert "entry_sequence_drawdown" in fold["candidate"]["metrics"]
    assert set(base["aggregate"]) == {"baseline", "candidate"}
    assert (
        base["aggregate"]["baseline"]["n_scheduled_dates"]
        == base["aggregate"]["candidate"]["n_scheduled_dates"]
    )
    # 미래 수익률 변경은 해당 폴드의 외부 평가 지표에 반영됩니다 (관측 경계 존재).
    last = len(base["folds"]) - 1
    assert base["folds"][last]["candidate"]["metrics"]["scheduled_mean_return"] != pytest.approx(
        altered["folds"][last]["candidate"]["metrics"]["scheduled_mean_return"]
    )
    assert base["research_bundle"] is None


def test_close_morning_recency_ensemble_rejects_invalid_inputs() -> None:
    """recency 실험은 식별 컬럼 누락, 비정상 가중치/워밍업/purge, 미지원 후보
    half-life/alpha 를 fail-closed 로 거부합니다."""
    df = _make_recency_dataset(seed=7, n_groups=20)
    kwargs = {
        "feature_cols": ["feature_a", "feature_b"],
        "target_col": TARGET_COL,
        "group_col": GROUP_COL,
    }
    with pytest.raises(ValueError, match="requires stock_code and chart_analysis"):
        run_close_morning_recency_ensemble_experiment(
            df.drop(columns=["chart_analysis"]), **kwargs
        )
    with pytest.raises(ValueError, match="probability_weight must be in \\(0, 1\\]"):
        run_close_morning_recency_ensemble_experiment(df, **kwargs, probability_weight=0.0)
    with pytest.raises(ValueError, match="min_history_dates must be >= 1"):
        run_close_morning_recency_ensemble_experiment(df, **kwargs, min_history_dates=0)
    with pytest.raises(ValueError, match="purge_gap must be >= 0"):
        run_close_morning_recency_ensemble_experiment(df, **kwargs, purge_gap=-1)
    with pytest.raises(ValueError, match="half_lives must be a non-empty subset"):
        run_close_morning_recency_ensemble_experiment(df, **kwargs, half_lives=(100,))
    with pytest.raises(ValueError, match="alphas must be a non-empty subset"):
        run_close_morning_recency_ensemble_experiment(df, **kwargs, alphas=(0.5,))
    with pytest.raises(ValueError, match="alphas must be a non-empty subset"):
        run_close_morning_recency_ensemble_experiment(df, **kwargs, alphas=(-0.5, 1.5))


def test_close_morning_recency_ensemble_fails_closed_on_insufficient_inner_history() -> None:
    """내부 partition 이 중첩 walk-forward 를 지원할 만큼 충분하지 않으면 baseline
    으로 fail-closed 합니다 (진단 사유 기록)."""
    df = _make_recency_dataset(seed=7, n_groups=8)
    report = run_close_morning_recency_ensemble_experiment(
        df,
        feature_cols=["feature_a", "feature_b"],
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=5,
        min_history_dates=2,
    )
    assert report["folds"][0]["chosen_config"] == {"half_life": None, "recent_weight": 0.0}
    assert (
        report["folds"][0]["inner"]["fail_closed_reason"] == "insufficient_inner_history"
    )
    assert report["folds"][0]["inner"]["candidate_stats"] == {}
    assert len(report["folds"]) == 5


def test_close_morning_recency_ensemble_builds_research_bundle_when_promoted() -> None:
    """승격 게이트 통과 + 명시 요청 시에만 두 return 모델과 recency 설정을 포함한
    연구 번들이 영속화됩니다 (자동 저장 금지, 프로덕션 기본값 불변)."""
    df = _make_recency_dataset(seed=13)
    report = run_close_morning_recency_ensemble_experiment(
        df,
        feature_cols=["feature_a", "feature_b"],
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        n_splits=2,
        min_history_dates=2,
        build_research_bundle=True,
    )
    assert report["promotion"]["promoted"] is True
    bundle = report["research_bundle"]
    assert bundle is not None
    assert "return_model" in bundle
    assert "recent_return_model" in bundle
    config = bundle["recency_ensemble_config"]
    assert config["version"] == "close-morning-recency-ensemble-research"
    assert config["half_life_groups"] in (252, 504)
    assert config["recent_weight"] in (0.25, 0.5, 0.75, 1.0)
    assert bundle["decision_score_config"]["version"] == "close-morning-reranker-v1"


def test_calculate_recency_sample_weight_rejects_non_finite_mean(monkeypatch) -> None:
    """정규화 분모가 비유한 경우 방어적 fail-closed 로 거부합니다.

    exp 감쇠는 수학적으로 유한하기 때문에 비유한 분모는 정상 입력으로 도달할 수
    없어, np.exp 를 몽키패치해 방어 가드가 실제로 발화하는지 검증합니다.
    """
    import numpy as np

    groups = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
    monkeypatch.setattr(np, "exp", lambda x: np.full(np.asarray(x).size, np.inf))
    with pytest.raises(ValueError, match="not finite"):
        calculate_recency_sample_weight(groups, 252)


def test_fit_predict_rejects_recency_weighting_for_ridge_and_ranker() -> None:
    """_fit_predict 는 Ridge/LGBMRanker 에 비영 recency 가중치를 방어적으로 거부합니다
    (run_model_pipeline 의 사전 검증과 독립적인 이중 방어)."""
    df = _make_dataset(n_groups=10, rows_per_group=6)
    train = df.sort_values(GROUP_COL).iloc[:30]
    val = df.sort_values(GROUP_COL).iloc[30:]
    sample_weight = np.ones(len(train), dtype=np.float64)
    with pytest.raises(ValueError, match="recency sample weighting is not supported for ridge"):
        _fit_predict(
            "ridge", train, val, FEATURE_COLS, TARGET_COL, GROUP_COL, sample_weight=sample_weight
        )
    with pytest.raises(
        ValueError, match="recency sample weighting is not supported for lgb_ranker"
    ):
        _fit_predict(
            "lgb_ranker", train, val, FEATURE_COLS, TARGET_COL, GROUP_COL, sample_weight=sample_weight
        )


def test_dominant_recency_config_deterministic_ties() -> None:
    """연구 번들용 최빈 구성은 높은 빈도 → 낮은 recent_weight → 긴 half_life →
    baseline(None) 우선 순서로 결정적 선택됩니다."""
    # 최빈 구성이 우선합니다.
    configs = [(252, 0.5), (252, 0.5), (504, 0.75), (504, 0.5)]
    assert _dominant_recency_config(configs) == (252, 0.5)
    # 동률이면 낮은 recent_weight 가 우선합니다.
    configs = [(252, 0.75), (504, 0.5)]
    assert _dominant_recency_config(configs) == (504, 0.5)
    # 같은 alpha/빈도이면 긴 half_life 가 우선합니다.
    configs = [(252, 0.5), (504, 0.5)]
    assert _dominant_recency_config(configs) == (504, 0.5)
    # baseline(None) 은 같은 alpha 대비 우선합니다 (recent_weight=0).
    configs = [(None, 0.0), (252, 0.5), (None, 0.0)]
    assert _dominant_recency_config(configs) == (None, 0.0)
    # None half-life 와 같은 alpha 후보는 baseline 이 이깁니다.
    configs = [(None, 0.0), (252, 0.0)]
    assert _dominant_recency_config(configs) == (None, 0.0)


def test_close_morning_yearly_breakdown_handles_small_and_invalid_years() -> None:
    """연도별 분해는 표본 <5년 연도를 null 처리하고 비파싱 연도를 건너뜁니다."""
    dates = np.array(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-06",
            "2025-01-01",
            "garbage-date",
        ],
        dtype=object,
    )
    scheduled = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.01, 0.02])
    out = _close_morning_yearly_breakdown(dates, scheduled)
    assert out[2024] is not None
    assert out[2024]["scheduled_mean_return"] == pytest.approx(0.01)
    assert out[2025] is None
    # 비파싱 연도는 건너뜁니다 (NaN 연도 키 미생성).
    assert all(np.isfinite(year) for year in out)
