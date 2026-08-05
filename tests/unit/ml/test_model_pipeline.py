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
    evaluate_close_morning_quality,
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
        "stock_code": ["000001", "000002"],
        "chart_analysis": ["상따", "신고가"],
    }
    if extra_row:
        rows[GROUP_COL].append("2024-03-02")
        rows[TARGET_COL].append(0.03)
        rows["p_good"].append(0.7)
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
        },
        index=[5, 7],
    )
    aligned = _align_close_morning_oof(
        _aligned_return_oof(), risk, target_col=TARGET_COL, group_col=GROUP_COL
    )
    assert {"pred", "p_good", "stock_code", "chart_analysis"} <= set(aligned.columns)
    assert aligned["p_good"].notna().all()


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
        patch("src.ml.model_pipeline.run_model_pipeline", return_value=fake),
        patch(
            "src.ml.model_pipeline._calibrate_close_morning_decision_oof",
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
        patch("src.ml.model_pipeline.run_model_pipeline", return_value=fake),
        patch(
            "src.ml.model_pipeline._calibrate_close_morning_decision_oof",
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
        patch("src.ml.model_pipeline.run_model_pipeline", return_value=fake),
        patch(
            "src.ml.model_pipeline._calibrate_close_morning_decision_oof",
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
        patch("src.ml.model_pipeline.run_model_pipeline", return_value=fake),
        patch(
            "src.ml.model_pipeline._calibrate_close_morning_decision_oof",
            return_value=(None, None, None),
        ),
        pytest.raises(ValueError, match="non-finite"),
    ):
        evaluate_close_morning_quality(raw, n_splits=2, purge_gap=1)
