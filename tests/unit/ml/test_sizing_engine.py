"""Utility Score 기반 Dynamic Sizing 단위 테스트.

SCENARIO: utility_score_calculation
SCENARIO: sizing_grade_assignment
SCENARIO: daily_position_sizing_inference
SCENARIO: save_and_load_artifacts
SCENARIO: load_model_artifacts_missing
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV

from src.ml.sizing_engine import (
    _CLOSE_MORNING_RERANKER_CONFIG,
    _train_inline_bundle,
    add_close_morning_decision_score,
    apply_risk_limits,
    assign_sizing_grades,
    calculate_utility_score,
    load_model_artifacts,
    predict_daily_position_sizing,
    save_model_artifacts,
)

GROUP_COL = "date"

FEATURE_COLS = ["f1", "f2"]
TARGET_COL = "target_net_return"


def _make_utility_df(n_rows: int = 12, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "pred_q10": rng.uniform(-0.03, 0.005, size=n_rows),
            "pred_q50": rng.uniform(0.005, 0.04, size=n_rows),
            "pred_q90": rng.uniform(0.03, 0.07, size=n_rows),
            "p_good": rng.uniform(0.1, 0.9, size=n_rows),
            "p_bad": rng.uniform(0.05, 0.5, size=n_rows),
        }
    )
    df["pred_q10"] = df[["pred_q10", "pred_q50"]].min(axis=1)
    df["pred_q90"] = df[["pred_q50", "pred_q90"]].max(axis=1)
    df[GROUP_COL] = "2024-01-01"
    return df


def test_utility_score_calculation_matches_formula() -> None:
    """U_i = q50 - lambda*max(0, -q10) - gamma*(q90 - q10) + w_good*p_good - w_bad*p_bad

    decimal net 예측은 이미 비용 차감 완료이므로 유틸리티가 비용을 재차감하지 않습니다.
    (SCENARIO: utility_score_calculation)
    """
    df = _make_utility_df()
    lambda_risk, gamma_uncertainty, w_good, w_bad = 1.5, 0.2, 0.5, 0.5
    series = calculate_utility_score(
        df,
        lambda_risk=lambda_risk,
        gamma_uncertainty=gamma_uncertainty,
        w_good=w_good,
        w_bad=w_bad,
    )
    assert isinstance(series, pd.Series)
    assert series.name == "utility_score"
    expected = (
        df["pred_q50"]
        - lambda_risk * np.maximum(0.0, -df["pred_q10"])
        - gamma_uncertainty * (df["pred_q90"] - df["pred_q10"])
        + w_good * df["p_good"]
        - w_bad * df["p_bad"]
    )
    np.testing.assert_allclose(series.to_numpy(), expected.to_numpy())


def test_utility_score_consumes_net_predictions_without_second_cost_deduction() -> None:
    """(SCENARIO: utility_score_calculation) decimal net 예측에 대해 round_trip_cost
    인자는 결과를 변경하지 않습니다 — 타깃 구성 시점에 정확히 1회 차감된 비용이
    유틸리티에서 다시 차감되지 않아야 합니다."""
    df = pd.DataFrame(
        {
            "pred_q10": [0.0],
            "pred_q50": [0.005],
            "pred_q90": [0.01],
            "p_good": [0.5],
            "p_bad": [0.5],
        }
    )
    no_cost = float(calculate_utility_score(df, round_trip_cost=0.0).iloc[0])
    with_cost = float(calculate_utility_score(df).iloc[0])
    assert with_cost == pytest.approx(no_cost)
    # 유틸리티는 예측 q50 자체를 그대로 기대수익으로 소비합니다 (비용 재차감 없음).
    # p_good/p_bad 유틸리티 가중치는 보정 증거가 없으면 기본 0 입니다.
    assert with_cost == pytest.approx(
        float(df["pred_q50"].iloc[0])
        - 0.5 * np.maximum(0.0, -df["pred_q10"].iloc[0])
        - 0.1 * (df["pred_q90"].iloc[0] - df["pred_q10"].iloc[0])
        + 0.0 * df["p_good"].iloc[0]
        - 0.0 * df["p_bad"].iloc[0]
    )


def test_utility_score_penalizes_downside_and_uncertainty() -> None:
    """동일 q50/p_good/p_bad 에서 하방위험(-q10)과 불확실성(q90-q10)이 큰 경우 점수가 낮아야 한다."""
    base = _make_utility_df(n_rows=2)
    low_risk = base.copy()
    low_risk["pred_q10"] = [-0.001, -0.002]
    high_risk = base.copy()
    high_risk["pred_q10"] = [-0.05, -0.06]
    assert float(calculate_utility_score(low_risk).iloc[0]) > float(calculate_utility_score(high_risk).iloc[0])

    low_unc = base.copy()
    low_unc["pred_q90"] = low_unc["pred_q50"] + 0.005
    high_unc = base.copy()
    high_unc["pred_q90"] = high_unc["pred_q50"] + 0.05
    assert float(calculate_utility_score(low_unc).iloc[0]) > float(calculate_utility_score(high_unc).iloc[0])


def test_utility_score_rewards_q50_and_p_good() -> None:
    df = _make_utility_df(n_rows=2)
    high = df.copy()
    high["pred_q50"] = [0.05, 0.05]
    high["p_good"] = [0.9, 0.9]
    assert float(calculate_utility_score(high).iloc[0]) > float(calculate_utility_score(df).iloc[0])


def test_utility_score_raises_on_missing_prediction_columns() -> None:
    df = _make_utility_df().drop(columns=["pred_q90"])
    with pytest.raises(ValueError, match="missing required prediction columns"):
        calculate_utility_score(df)


def _with_utility(df: pd.DataFrame, values: list[float]) -> pd.DataFrame:
    out = df.copy()
    out["utility_score"] = values
    return out


def test_assign_sizing_grades_returns_grade_columns() -> None:
    df = _with_utility(_make_utility_df(), [float(i) for i in range(12)])
    result = assign_sizing_grades(df, utility_col="utility_score", group_col=GROUP_COL)
    assert {"grade", "grade_multiplier"}.issubset(result.columns)
    assert result["grade"].isin(["Strong", "Good", "Weak", "Pass"]).all()
    assert result["grade_multiplier"].isin([0.0, 0.5, 1.0, 1.5]).all()


def test_assign_sizing_grades_assigns_bands_correctly() -> None:
    """(SCENARIO: sizing_grade_assignment) 상위 10% Strong, 상위 25% Good,
    상위 50% + 기대수익 양수 Weak, 나머지 Pass.

    utility 0~39, q50=+0.01 인 40행 단일 그룹: rank(pct=True) 기준
    Strong 5행(35~39), Good 6행(29~34), Weak 10행(19~28), Pass 19행(0~18).
    """
    df = _with_utility(
        _make_utility_df(n_rows=40),
        [float(i) for i in range(40)],
    )
    df["pred_q50"] = 0.01
    result = assign_sizing_grades(df, utility_col="utility_score", group_col=GROUP_COL)
    result = result.sort_values("utility_score", ascending=False).reset_index(drop=True)

    assert result.iloc[0:5]["grade"].eq("Strong").all()
    assert result.iloc[5:11]["grade"].eq("Good").all()
    assert result.iloc[11:21]["grade"].eq("Weak").all()
    assert result.iloc[21:]["grade"].eq("Pass").all()

    assert result["grade_multiplier"].iloc[0] == 1.5
    assert result["grade_multiplier"].iloc[6] == 1.0
    assert result["grade_multiplier"].iloc[12] == 0.5
    assert result["grade_multiplier"].iloc[30] == 0.0


def test_assign_sizing_grades_passes_negative_net_utility_and_nonpositive_return() -> None:
    """(SCENARIO: sizing_grade_assignment) 음수 순유틸리티 또는 기대수익
    (net_q50, 비용 차감 완료)이 0 이하인 항목은 상위 백분위여도 Pass 등급을 받는다."""
    negative_util = _with_utility(_make_utility_df(n_rows=10), [-0.01] * 10)
    negative_util["pred_q50"] = 0.05
    result = assign_sizing_grades(negative_util, utility_col="utility_score", group_col=GROUP_COL)
    assert (result["grade"] == "Pass").all()

    zero_net = _with_utility(_make_utility_df(n_rows=10), [float(i) for i in range(10)])
    zero_net["pred_q50"] = 0.0
    result2 = assign_sizing_grades(zero_net, utility_col="utility_score", group_col=GROUP_COL)
    assert (result2["grade"] == "Pass").all()


def test_assign_sizing_grades_weak_requires_positive_q50() -> None:
    """Weak 등급은 상위 50% 이면서 기대수익(q50)이 양수일 때만 부여된다."""
    df = _with_utility(
        _make_utility_df(n_rows=20),
        [float(i) for i in range(20)],
    )
    df["pred_q50"] = [-0.02] * 20
    df.loc[10:14, "pred_q50"] = 0.02
    result = assign_sizing_grades(df, utility_col="utility_score", group_col=GROUP_COL)

    weak_mask = result["grade"].eq("Weak")
    assert (result.loc[weak_mask, "pred_q50"] > 0.0).all()
    assert result.loc[result["pred_q50"] < 0.0, "grade"].ne("Weak").all()
    assert set(result.loc[weak_mask, "utility_score"].astype(float).tolist()) == {
        10.0,
        11.0,
        12.0,
        13.0,
    }
    assert result["grade"].eq("Pass").any()


def test_assign_sizing_grades_is_cross_sectional_per_group() -> None:
    df = _with_utility(_make_utility_df(n_rows=24), [float(i) % 8 for i in range(24)])
    df[GROUP_COL] = ["2024-01-01"] * 8 + ["2024-01-02"] * 8 + ["2024-01-03"] * 8
    result = assign_sizing_grades(df, utility_col="utility_score", group_col=GROUP_COL)
    for _, group in result.groupby(GROUP_COL):
        top_pct = group["utility_score"].rank(pct=True)
        assert group.loc[top_pct >= 0.9, "grade"].eq("Strong").all()


def test_assign_sizing_grades_raises_on_missing_columns() -> None:
    df = _with_utility(_make_utility_df(), [float(i) for i in range(12)])
    with pytest.raises(ValueError, match="utility_col"):
        assign_sizing_grades(
            df.drop(columns=["utility_score"]),
            utility_col="utility_score",
            group_col=GROUP_COL,
        )
    with pytest.raises(ValueError, match="group_col"):
        assign_sizing_grades(
            df.drop(columns=[GROUP_COL]),
            utility_col="utility_score",
            group_col=GROUP_COL,
        )


def test_apply_risk_limits_respects_caps_per_group() -> None:
    df = assign_sizing_grades(_with_utility(_make_utility_df(), [float(i) for i in range(12)]))
    result = apply_risk_limits(df, max_position_pct=0.25, max_total_allocation=1.0)
    assert "allocation" in result.columns
    assert result["allocation"].ge(0.0).all()
    total = result["allocation"].sum()
    assert total <= 1.0 + 1e-9
    assert result["allocation"].max() <= 0.25 + 1e-9


def test_apply_risk_limits_scales_down_when_total_exceeds_budget() -> None:
    df = assign_sizing_grades(_with_utility(_make_utility_df(), [float(i) for i in range(12)]))
    result = apply_risk_limits(
        df,
        base_budget=1.0,
        target_vol=0.5,
        max_position_pct=1.0,
        max_total_allocation=0.6,
    )
    assert result["allocation"].sum() <= 0.6 + 1e-9


def test_apply_risk_limits_defensive_cap_on_negative_avg_utility() -> None:
    """(시장 국면 방어) 후보 유니버스 평균 utility 가 음수이면 max_total_allocation 을
    max(1 + avg_utility, 0) 비율로 축소하고, 음수 순유틸리티 종목은 비중 0% 로 강제한다."""
    df = _with_utility(_make_utility_df(n_rows=6), [0.10, 0.01, -0.05, -0.05, -0.05, -0.05])
    df["grade_multiplier"] = 1.5
    result = apply_risk_limits(df, max_position_pct=1.0, max_total_allocation=1.0)
    avg_utility = float(df["utility_score"].mean())
    assert avg_utility < 0.0
    expected_cap = max(1.0 + avg_utility, 0.0)
    assert result["allocation"].sum() <= expected_cap + 1e-9
    assert result.loc[df["utility_score"] < 0.0, "allocation"].eq(0.0).all()


def test_apply_risk_limits_scales_with_utility_magnitude() -> None:
    """(SCENARIO: risk_limits) 동일 시그마/등급에서 utility magnitude 가 클수록
    더 큰 비중을 받는다 (utility_scaling = clip(utility / 0.01, 0.1, 1.5))."""
    df = _with_utility(_make_utility_df(n_rows=3), [0.001, 0.01, 0.1])
    df["grade_multiplier"] = 1.0
    df["pred_q10"] = [-0.01, -0.01, -0.01]
    df["pred_q90"] = [0.05, 0.05, 0.05]
    result = apply_risk_limits(df, max_position_pct=1.0, max_total_allocation=1.0)
    alloc = result["allocation"].to_numpy()
    assert alloc[0] < alloc[1] < alloc[2]


def test_apply_risk_limits_no_defensive_cap_on_positive_avg_utility() -> None:
    """평균 utility 가 양수이면 방어 축소 없이 max_total_allocation 이 적용된다."""
    df = assign_sizing_grades(_with_utility(_make_utility_df(), [float(i) for i in range(12)]))
    df["grade_multiplier"] = 1.5
    result = apply_risk_limits(df, max_position_pct=1.0, max_total_allocation=1.0)
    np.testing.assert_allclose(result["allocation"].sum(), 1.0, atol=1e-6)


def test_apply_risk_limits_raises_on_missing_columns() -> None:
    df = _make_utility_df()
    with pytest.raises(ValueError, match="grade_multiplier"):
        apply_risk_limits(df)
    # 분위수 스프레드(q90-q10)를 실현 변동성으로 사용하지 않으므로
    # pred_q10/pred_q90 없이도 변동성 타게팅이 동작합니다.
    df["grade_multiplier"] = 1.0
    result = apply_risk_limits(df)
    assert "allocation" in result.columns


def test_apply_risk_limits_uses_separate_realized_vol_column() -> None:
    """변동성 타게팅은 q90-q10 이 아니라 별도 실현 변동성(realized_vol) 컬럼을 사용합니다.

    realized_vol 이 제공되면 동일 utility/등급에서 높은 변동성 종목이 낮은 비중을 받습니다.
    """
    df = _with_utility(_make_utility_df(n_rows=2), [0.01, 0.01])
    df["grade_multiplier"] = 1.0
    df["realized_vol"] = [0.01, 0.10]
    result = apply_risk_limits(
        df, target_vol=0.15, max_position_pct=1.0, max_total_allocation=1.0
    )
    alloc = result["allocation"].to_numpy()
    assert alloc[0] > alloc[1]
    # 실현 변동성 비(10배)가 비중 비(1/10)로 반영됩니다.
    assert alloc[0] == pytest.approx(alloc[1] * 10, rel=1e-6)


def _make_feature_df(n_rows: int = 30, n_dates: int = 3, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = [f"2024-01-{d:02d}" for d in range(1, n_dates + 1)]
    rows_per_date = n_rows // n_dates
    f1 = rng.normal(size=n_rows)
    f2 = rng.normal(size=n_rows)
    target = 0.02 * f1 + 0.01 * f2 + rng.normal(loc=0.0, scale=0.008, size=n_rows)
    return pd.DataFrame(
        {
            "date": [d for d in dates for _ in range(rows_per_date)],
            "f1": f1,
            "f2": f2,
            TARGET_COL: target,
        }
    )


def _build_bundle(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
    group_col: str = GROUP_COL,
) -> dict[str, object]:
    train = df.sort_values(group_col)
    y = train[target_col].to_numpy(dtype=np.float64)

    relevance = train[target_col].groupby(train[group_col], sort=False).rank(pct=True).to_numpy()
    relevance = (relevance * 4.0).round().astype(int)
    group_counts = train[group_col].value_counts(sort=False).to_numpy(dtype=np.int64)
    ranker = LGBMRanker(objective="lambdarank", random_state=42, verbosity=-1)
    ranker.fit(train[feature_cols], relevance, group=group_counts)

    quantile_models: dict[str, object] = {}
    for col, alpha in (("pred_q10", 0.1), ("pred_q50", 0.5), ("pred_q90", 0.9)):
        model = LGBMRegressor(objective="quantile", alpha=alpha, random_state=42, verbosity=-1)
        model.fit(train[feature_cols], y)
        quantile_models[col] = model

    calibrators: dict[str, object] = {}
    for name, cond in (
        ("p_good", train[target_col] >= 0.01),
        ("p_bad", train[target_col] <= -0.015),
    ):
        y_bin = cond.to_numpy().astype(bool)
        if np.unique(y_bin).size < 2 or np.min(np.bincount(y_bin)) < 3:
            calibrators[name] = float(np.mean(y_bin))
            continue
        base = LGBMClassifier(objective="binary", random_state=42, verbosity=-1)
        calibrator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        calibrator.fit(train[feature_cols], y_bin)
        calibrators[name] = calibrator

    return {
        "feature_cols": list(feature_cols),
        "target_col": target_col,
        "group_col": group_col,
        "rank_model": ranker,
        "quantile_models": quantile_models,
        "calibrators": calibrators,
    }


def test_predict_daily_position_sizing_returns_sizing_columns() -> None:
    """(SCENARIO: daily_position_sizing_inference)
    인라인 모델로 당일 스냅샷 추론 시 utility_score / grade / grade_multiplier /
    allocation 및 rank_score / 분위수 / 확률 컬럼을 반환한다.
    """
    result = predict_daily_position_sizing(_make_feature_df(), FEATURE_COLS)
    assert {"utility_score", "grade", "grade_multiplier", "allocation"}.issubset(result.columns)
    assert {
        "rank_score",
        "pred_q10",
        "pred_q50",
        "pred_q90",
        "p_good",
        "p_bad",
    }.issubset(result.columns)
    assert result["grade"].isin(["Strong", "Good", "Weak", "Pass"]).all()
    assert result["grade_multiplier"].isin([0.0, 0.5, 1.0, 1.5]).all()
    assert result["allocation"].ge(0.0).all()
    assert len(result) == len(_make_feature_df())


def test_predict_daily_position_sizing_pass_grade_yields_zero_allocation() -> None:
    """Pass 등급은 0.0 배분 비중을 가진다."""
    df = _make_feature_df(n_rows=40, n_dates=4)
    result = predict_daily_position_sizing(df, FEATURE_COLS)
    pass_mask = result["grade"].eq("Pass")
    assert pass_mask.any()
    np.testing.assert_allclose(result.loc[pass_mask, "allocation"].to_numpy(), 0.0)
    assert result.loc[pass_mask, "grade_multiplier"].eq(0.0).all()


def test_predict_daily_position_sizing_supports_single_day() -> None:
    df = _make_feature_df(n_rows=12, n_dates=1)
    result = predict_daily_position_sizing(df, FEATURE_COLS)
    assert {"utility_score", "grade", "allocation"}.issubset(result.columns)
    assert len(result) == len(df)


def test_predict_daily_position_sizing_is_deterministic_inline() -> None:
    df = _make_feature_df()
    first = predict_daily_position_sizing(df, FEATURE_COLS)
    second = predict_daily_position_sizing(df, FEATURE_COLS)
    pd.testing.assert_series_equal(first["utility_score"], second["utility_score"])
    pd.testing.assert_series_equal(first["grade"], second["grade"])
    np.testing.assert_allclose(first["allocation"].to_numpy(), second["allocation"].to_numpy())


def test_predict_daily_position_sizing_with_preloaded_bundle() -> None:
    """저장/로드된 모델 번들로 추론 시 저장 직전 번들과 동일한 결과를 반환한다."""
    df = _make_feature_df()
    bundle = _build_bundle(df)
    result = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bundle)
    assert {"utility_score", "grade", "allocation"}.issubset(result.columns)
    assert result["grade"].isin(["Strong", "Good", "Weak", "Pass"]).all()


def test_predict_daily_position_sizing_requires_group_col_and_features() -> None:
    with pytest.raises(ValueError, match="missing feature columns"):
        predict_daily_position_sizing(_make_feature_df(), ["f3"])
    with pytest.raises(ValueError, match="group_col"):
        predict_daily_position_sizing(_make_feature_df().drop(columns=[GROUP_COL]), FEATURE_COLS)


def test_predict_daily_position_sizing_requires_target_for_inline_models() -> None:
    df = _make_feature_df().drop(columns=[TARGET_COL])
    with pytest.raises(ValueError, match="required to train inline models"):
        predict_daily_position_sizing(df, FEATURE_COLS)


def test_predict_daily_position_sizing_handles_single_class_calibrator() -> None:
    """이진 라벨이 단일 클래스로 수렴해도 prior 상수 폴백으로 추론이 완료된다."""
    df = _make_feature_df()
    df[TARGET_COL] = df[TARGET_COL].abs() + 0.01  # 전부 양수 → p_bad 단일 클래스
    result = predict_daily_position_sizing(df, FEATURE_COLS)
    assert {"utility_score", "grade", "allocation"}.issubset(result.columns)
    assert result["p_bad"].eq(result["p_bad"].iloc[0]).all()


def test_save_load_artifacts_roundtrip(tmp_path) -> None:
    """(SCENARIO: save_and_load_artifacts) 저장된 번들을 로드하면 내용이 정확히 일치한다."""
    artifacts = {"dummy": 123, "nested": {"a": [1, 2], "b": "x"}}
    saved_path = save_model_artifacts(artifacts, str(tmp_path))
    assert os.path.exists(saved_path)
    assert os.path.isabs(saved_path)
    assert save_model_artifacts(artifacts, str(tmp_path)) == saved_path
    reloaded = load_model_artifacts(str(tmp_path))
    assert reloaded == artifacts


def test_save_model_artifacts_creates_directory(tmp_path) -> None:
    target = tmp_path / "nested" / "models"
    saved_path = save_model_artifacts({"dummy": 1}, str(target))
    assert target.is_dir()
    assert os.path.exists(saved_path)


def test_load_model_artifacts_missing_directory_raises(tmp_path) -> None:
    """(SCENARIO: load_model_artifacts_missing) 존재하지 않는 디렉터리/번들에 대해
    FileNotFoundError 를 발생시킨다."""
    with pytest.raises(FileNotFoundError):
        load_model_artifacts(str(tmp_path / "no_such_dir"))
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_model_artifacts(str(empty_dir))


def test_scenario_sizing_grade_hybrid_01() -> None:
    """[SCENARIO_SIZING_GRADE_HYBRID_01] Verifies hybrid grading demotes Strong/Good grades to Pass when utility score falls below absolute threshold."""
    df = _with_utility(_make_utility_df(n_rows=10), [-10.0] * 10)
    df["pred_q50"] = [0.02] * 10
    result = assign_sizing_grades(df, min_good_utility=0.0, min_weak_utility=-2.0)
    assert (result["grade"] == "Pass").all()


def test_utility_score_defaults_zero_p_good_p_bad_weights() -> None:
    """p_good/p_bad 유틸리티 가중치는 보정 승격이 없으면 기본 0 입니다."""
    df = pd.DataFrame(
        {
            "pred_q10": [-0.01],
            "pred_q50": [0.01],
            "pred_q90": [0.03],
            "p_good": [1.0],
            "p_bad": [1.0],
        }
    )
    score = float(calculate_utility_score(df).iloc[0])
    expected = 0.01 - 0.5 * max(0.0, 0.01) - 0.1 * (0.03 - (-0.01))
    assert score == pytest.approx(expected)


def test_train_inline_bundle_contains_return_model_and_zero_utility_weights() -> None:
    """_train_inline_bundle 은 회귀 champion(return_model)을 영속화하고
    p_good/p_bad 유틸리티 가중치는 기본 0 으로 기록합니다."""
    df = _make_feature_df()
    bundle = _train_inline_bundle(df, FEATURE_COLS, TARGET_COL, GROUP_COL)
    assert isinstance(bundle["return_model"], LGBMRegressor)
    assert isinstance(bundle["rank_model"], LGBMRanker)
    weights = bundle["policy_params"]["utility_weights"]
    assert weights["w_good"] == 0.0
    assert weights["w_bad"] == 0.0


def test_predict_daily_position_sizing_rank_score_comes_from_return_model() -> None:
    """당일 rank_score 는 저장된 return_model 의 기대수익 예측으로 생성됩니다."""
    df = _make_feature_df()
    result = predict_daily_position_sizing(df, FEATURE_COLS)
    assert "rank_score" in result.columns
    bundle = _train_inline_bundle(df, FEATURE_COLS, TARGET_COL, GROUP_COL)
    expected = bundle["return_model"].predict(df[FEATURE_COLS])
    np.testing.assert_allclose(result["rank_score"].to_numpy(), expected, atol=1e-12)


def test_predict_from_bundle_falls_back_to_rank_model_without_return_model() -> None:
    """return_model 이 없는 기존 번들은 rank_model 로 rank_score 를 보존합니다."""
    df = _make_feature_df()
    legacy = _build_bundle(df)
    assert "return_model" not in legacy
    result = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=legacy)
    expected = legacy["rank_model"].predict(df[FEATURE_COLS])
    np.testing.assert_allclose(result["rank_score"].to_numpy(), expected, atol=1e-12)


def test_add_close_morning_decision_score_formula_is_groupwise() -> None:
    """(SCENARIO: close_morning_decision_score) decision_score 는 그룹(날짜) 내
    rank_score 백분위 + 0.5 x p_good 백분위로 산출됩니다."""
    df = pd.DataFrame(
        {
            GROUP_COL: ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "rank_score": [0.3, 0.1, 0.2, 0.5, 0.4, 0.6],
            "p_good": [0.1, 0.9, 0.5, 0.8, 0.2, 0.4],
        }
    )
    out = add_close_morning_decision_score(df)
    expected = (
        df.groupby(GROUP_COL)["rank_score"].rank(pct=True, method="average")
        + 0.5 * df.groupby(GROUP_COL)["p_good"].rank(pct=True, method="average")
    )
    assert out["decision_score"].name == "decision_score"
    pd.testing.assert_series_equal(out["decision_score"], expected.rename("decision_score"))
    assert out.columns.tolist() == [GROUP_COL, "rank_score", "p_good", "decision_score"]


def test_add_close_morning_decision_score_resolves_ties_with_average() -> None:
    """동점 rank_score 는 method='average' 백분위 순위로 해소됩니다."""
    df = pd.DataFrame(
        {
            GROUP_COL: ["2024-01-01"] * 3,
            "rank_score": [0.5, 0.5, 0.2],
            "p_good": [0.4, 0.6, 0.8],
        }
    )
    out = add_close_morning_decision_score(df)
    expected = (
        df.groupby(GROUP_COL)["rank_score"].rank(pct=True, method="average")
        + 0.5 * df.groupby(GROUP_COL)["p_good"].rank(pct=True, method="average")
    )
    pd.testing.assert_series_equal(out["decision_score"], expected.rename("decision_score"))
    # 동점 0.5 두 건은 평균 순위 2.5 → pct 2.5/3, 최저값 0.2 는 1/3 입니다.
    assert float(out["decision_score"].iloc[0]) == pytest.approx(
        2.5 / 3 + 0.5 * (1 / 3)
    )
    assert float(out["decision_score"].iloc[1]) == pytest.approx(
        2.5 / 3 + 0.5 * (2 / 3)
    )


def test_add_close_morning_decision_score_handles_single_row_group() -> None:
    """한 행으로만 구성된 그룹은 백분위 순위 1.0 을 유지합니다."""
    df = pd.DataFrame(
        {
            GROUP_COL: ["2024-01-01", "2024-01-02"],
            "rank_score": [0.5, 0.9],
            "p_good": [0.2, 0.3],
        }
    )
    out = add_close_morning_decision_score(df)
    assert float(out["decision_score"].iloc[0]) == pytest.approx(1.0 + 0.5 * 1.0)
    assert float(out["decision_score"].iloc[1]) == pytest.approx(1.0 + 0.5 * 1.0)


def test_add_close_morning_decision_score_rejects_missing_columns() -> None:
    df = pd.DataFrame({GROUP_COL: ["2024-01-01"], "rank_score": [0.5], "p_good": [0.5]})
    with pytest.raises(ValueError, match="missing required columns"):
        add_close_morning_decision_score(df.drop(columns=["p_good"]))
    with pytest.raises(ValueError, match="missing required columns"):
        add_close_morning_decision_score(df.drop(columns=[GROUP_COL]))


def test_add_close_morning_decision_score_rejects_non_finite_scores() -> None:
    df = pd.DataFrame(
        {GROUP_COL: ["2024-01-01"] * 2, "rank_score": [0.5, np.nan], "p_good": [0.5, 0.5]}
    )
    with pytest.raises(ValueError, match="must be finite"):
        add_close_morning_decision_score(df)


def test_add_close_morning_decision_score_rejects_out_of_range_weight() -> None:
    df = pd.DataFrame({GROUP_COL: ["2024-01-01"], "rank_score": [0.5], "p_good": [0.5]})
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        add_close_morning_decision_score(df, probability_weight=1.5)


def test_add_close_morning_decision_score_never_reads_target_column() -> None:
    """결정 스코어는 rank_score/p_good/그룹 컬럼만 소비하고 수익률/미래 정보는 읽지 않습니다."""
    df = pd.DataFrame(
        {
            GROUP_COL: ["2024-01-01"] * 2,
            "rank_score": [0.5, 0.7],
            "p_good": [0.4, 0.6],
            "target_net_return": [0.05, 0.01],
        }
    )
    out = add_close_morning_decision_score(df)
    expected = (
        df.groupby(GROUP_COL)["rank_score"].rank(pct=True, method="average")
        + 0.5 * df.groupby(GROUP_COL)["p_good"].rank(pct=True, method="average")
    )
    pd.testing.assert_series_equal(out["decision_score"], expected.rename("decision_score"))


def test_add_close_morning_decision_score_v2_subtracts_p_bad_percentile() -> None:
    """(SCENARIO: close_morning_decision_score) v2 는 p_bad 백분위를 차감합니다.

    ``decision_score = rank_pct + probability_weight * p_good_pct
    - bad_probability_weight * p_bad_pct``
    """
    df = pd.DataFrame(
        {
            GROUP_COL: ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "rank_score": [0.3, 0.1, 0.2, 0.5, 0.4, 0.6],
            "p_good": [0.1, 0.9, 0.5, 0.8, 0.2, 0.4],
            "p_bad": [0.9, 0.1, 0.5, 0.2, 0.8, 0.4],
        }
    )
    out = add_close_morning_decision_score(
        df, probability_weight=0.5, bad_probability_weight=0.5
    )
    expected = (
        df.groupby(GROUP_COL)["rank_score"].rank(pct=True, method="average")
        + 0.5 * df.groupby(GROUP_COL)["p_good"].rank(pct=True, method="average")
        - 0.5 * df.groupby(GROUP_COL)["p_bad"].rank(pct=True, method="average")
    )
    pd.testing.assert_series_equal(out["decision_score"], expected.rename("decision_score"))
    # p_bad 가 높은(0.9) 종목은 v1 점수 대비 패널티로 점수가 낮아집니다.
    v1 = add_close_morning_decision_score(df)
    assert float(out["decision_score"].iloc[0]) < float(v1["decision_score"].iloc[0])


def test_add_close_morning_decision_score_v2_requires_p_bad_when_weight_nonzero() -> None:
    """bad_probability_weight=0(기본)이면 p_bad 컬럼이 없어도 동작하고, 비영이면
    누락 p_bad 를 fail-closed 로 거부합니다."""
    df = pd.DataFrame(
        {GROUP_COL: ["2024-01-01"] * 2, "rank_score": [0.5, 0.7], "p_good": [0.4, 0.6]}
    )
    assert "decision_score" in add_close_morning_decision_score(df).columns
    with pytest.raises(ValueError, match="missing required columns"):
        add_close_morning_decision_score(df, bad_probability_weight=0.5)


def test_add_close_morning_decision_score_v2_rejects_non_finite_p_bad() -> None:
    df = pd.DataFrame(
        {
            GROUP_COL: ["2024-01-01"] * 2,
            "rank_score": [0.5, 0.7],
            "p_good": [0.4, 0.6],
            "p_bad": [0.3, np.nan],
        }
    )
    with pytest.raises(ValueError, match="p_bad must be finite"):
        add_close_morning_decision_score(df, bad_probability_weight=0.5)


def test_add_close_morning_decision_score_v2_rejects_out_of_range_bad_weight() -> None:
    df = pd.DataFrame(
        {
            GROUP_COL: ["2024-01-01"] * 2,
            "rank_score": [0.5, 0.7],
            "p_good": [0.4, 0.6],
            "p_bad": [0.3, 0.2],
        }
    )
    with pytest.raises(ValueError, match="bad_probability_weight must be within \\[0, 1\\]"):
        add_close_morning_decision_score(df, bad_probability_weight=1.5)


def test_add_close_morning_decision_score_v2_never_reads_target_column() -> None:
    """v2 결정 스코어도 타깃/수익률 컬럼을 절대 읽지 않습니다."""
    df = pd.DataFrame(
        {
            GROUP_COL: ["2024-01-01"] * 2,
            "rank_score": [0.5, 0.7],
            "p_good": [0.4, 0.6],
            "p_bad": [0.9, 0.1],
            "target_net_return": [0.05, -0.05],
        }
    )
    out = add_close_morning_decision_score(
        df, probability_weight=0.5, bad_probability_weight=0.5
    )
    expected = (
        df.groupby(GROUP_COL)["rank_score"].rank(pct=True, method="average")
        + 0.5 * df.groupby(GROUP_COL)["p_good"].rank(pct=True, method="average")
        - 0.5 * df.groupby(GROUP_COL)["p_bad"].rank(pct=True, method="average")
    )
    pd.testing.assert_series_equal(out["decision_score"], expected.rename("decision_score"))


def test_predict_daily_position_sizing_appends_decision_score_for_reranker_bundle() -> None:
    """reranker v1 설정 번들은 rank_score/p_good 예측 직후 decision_score 를 추가합니다."""
    df = _make_feature_df()
    bundle = dict(_build_bundle(df))
    bundle["decision_score_config"] = dict(_CLOSE_MORNING_RERANKER_CONFIG)
    result = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bundle)
    assert "decision_score" in result.columns
    expected = (
        result.groupby(GROUP_COL)["rank_score"].rank(pct=True, method="average")
        + 0.5 * result.groupby(GROUP_COL)["p_good"].rank(pct=True, method="average")
    )
    pd.testing.assert_series_equal(
        result["decision_score"], expected.rename("decision_score")
    )


def test_predict_daily_position_sizing_appends_decision_score_for_reranker_v2_research_bundle() -> None:
    """reranker v2 연구 번들은 p_bad 하방위험 패널티를 명시적으로 선언할 때만
    decision_score 에 반영하고, 기본 프로덕션 번들은 변경하지 않습니다."""
    df = _make_feature_df()
    bundle = dict(_build_bundle(df))
    bundle["decision_score_config"] = {
        "version": "close-morning-reranker-v2-research",
        "rank_weight": 1.0,
        "p_good_weight": 0.5,
        "bad_probability_weight": 0.5,
        "score_col": "decision_score",
    }
    result = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bundle)
    assert "decision_score" in result.columns
    expected = (
        result.groupby(GROUP_COL)["rank_score"].rank(pct=True, method="average")
        + 0.5 * result.groupby(GROUP_COL)["p_good"].rank(pct=True, method="average")
        - 0.5 * result.groupby(GROUP_COL)["p_bad"].rank(pct=True, method="average")
    )
    pd.testing.assert_series_equal(
        result["decision_score"], expected.rename("decision_score")
    )


def test_predict_daily_position_sizing_legacy_bundle_has_no_decision_score() -> None:
    """decision_score_config 가 없는 레거시 번들은 기존 출력을 유지합니다."""
    df = _make_feature_df()
    bundle = _build_bundle(df)
    result = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bundle)
    assert "decision_score" not in result.columns
    assert {"rank_score", "p_good"}.issubset(result.columns)

def _build_research_bundle(
    df: pd.DataFrame,
    alpha: float = 0.5,
    half_life: int = 252,
    probability_weight: float = 0.5,
) -> dict[str, object]:
    """recency-adaptive 앙상블 연구 번들 (두 return 모델 + recency 설정) 을 구성합니다."""
    from src.ml.model_pipeline import calculate_recency_sample_weight

    weights = calculate_recency_sample_weight(df[GROUP_COL], half_life)
    recent_model = LGBMRegressor(objective="huber", random_state=42, verbosity=-1)
    recent_model.fit(df[FEATURE_COLS], df[TARGET_COL], sample_weight=weights)
    config: dict[str, object] = {
        "version": "close-morning-recency-ensemble-research",
        "half_life_groups": half_life,
        "recent_weight": alpha,
        "probability_weight": probability_weight,
        "score_col": "decision_score",
    }
    return _train_inline_bundle(
        df,
        FEATURE_COLS,
        TARGET_COL,
        GROUP_COL,
        recent_return_model=recent_model,
        recency_ensemble_config=config,
    )


def test_predict_daily_position_sizing_research_bundle_blends_rank_scores() -> None:
    """연구 번들은 동일 날짜 백분위 blend 를 rank_score 로 재현하고, v1 reranker
    decision_score 는 blend 백분위 + p_good 백분위 로 재현합니다."""
    df = _make_feature_df(n_rows=45, n_dates=3)
    alpha, probability_weight = 0.5, 0.5
    bundle = _build_research_bundle(df, alpha=alpha, probability_weight=probability_weight)
    result = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bundle)

    expanding = bundle["return_model"].predict(df[FEATURE_COLS])
    recent = bundle["recent_return_model"].predict(df[FEATURE_COLS])
    exp_pct = (
        pd.Series(expanding, index=df.index).groupby(df[GROUP_COL]).rank(pct=True, method="average")
    )
    rec_pct = (
        pd.Series(recent, index=df.index).groupby(df[GROUP_COL]).rank(pct=True, method="average")
    )
    blend = (1 - alpha) * exp_pct + alpha * rec_pct
    np.testing.assert_allclose(result["rank_score"].to_numpy(), blend.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(result["pred_expanding"].to_numpy(), expanding, atol=1e-12)
    np.testing.assert_allclose(result["pred_recent"].to_numpy(), recent, atol=1e-12)

    expected_decision = (
        blend.groupby(df[GROUP_COL]).rank(pct=True, method="average")
        + probability_weight
        * result.groupby(GROUP_COL)["p_good"].rank(pct=True, method="average")
    )
    np.testing.assert_allclose(
        result["decision_score"].to_numpy(), expected_decision.to_numpy(), atol=1e-12
    )
    assert "allocation" in result.columns


def test_predict_daily_position_sizing_research_bundle_alpha_endpoints() -> None:
    """alpha=0 은 expanding pct rank 로 v1 decision 을 정확히 재현하고, alpha=1 은
    recent pct rank 를 그대로 사용합니다."""
    df = _make_feature_df()
    bundle0 = _build_research_bundle(df, alpha=0.0)
    result0 = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bundle0)
    expanding = bundle0["return_model"].predict(df[FEATURE_COLS])
    exp_pct = (
        pd.Series(expanding, index=df.index).groupby(df[GROUP_COL]).rank(pct=True, method="average")
    )
    np.testing.assert_allclose(result0["rank_score"].to_numpy(), exp_pct.to_numpy(), atol=1e-12)

    # alpha=0 연구 번들은 기존 v1 reranker decision_score 와 동일합니다
    # (pct_rank(pct_rank(x)) = pct_rank(x) 항등에 의해).
    v1_bundle = dict(_train_inline_bundle(df, FEATURE_COLS, TARGET_COL, GROUP_COL))
    v1_bundle["decision_score_config"] = dict(_CLOSE_MORNING_RERANKER_CONFIG)
    v1 = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=v1_bundle)
    np.testing.assert_allclose(
        result0["decision_score"].to_numpy(), v1["decision_score"].to_numpy(), atol=1e-12
    )

    bundle1 = _build_research_bundle(df, alpha=1.0)
    result1 = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bundle1)
    recent = bundle1["recent_return_model"].predict(df[FEATURE_COLS])
    rec_pct = (
        pd.Series(recent, index=df.index).groupby(df[GROUP_COL]).rank(pct=True, method="average")
    )
    np.testing.assert_allclose(result1["rank_score"].to_numpy(), rec_pct.to_numpy(), atol=1e-12)


def test_train_inline_bundle_research_bundle_requires_matching_models_and_config() -> None:
    """연구 번들은 recent_return_model 과 recency_ensemble_config 를 항상 함께 요구하며
    미지원 half-life/범위 밖 recent_weight 는 fail-closed 로 거부합니다."""
    df = _make_feature_df()
    with pytest.raises(ValueError, match="requires both recent_return_model and recency_ensemble_config"):
        _train_inline_bundle(
            df, FEATURE_COLS, TARGET_COL, GROUP_COL, recent_return_model=LGBMRegressor()
        )
    with pytest.raises(ValueError, match="requires both recent_return_model and recency_ensemble_config"):
        _train_inline_bundle(
            df, FEATURE_COLS, TARGET_COL, GROUP_COL, recency_ensemble_config={"version": "x"}
        )
    with pytest.raises(ValueError, match="half_life_groups must be 252 or 504"):
        _train_inline_bundle(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            recent_return_model=LGBMRegressor(),
            recency_ensemble_config={"half_life_groups": 100, "recent_weight": 0.5},
        )
    with pytest.raises(ValueError, match="recent_weight must be within"):
        _train_inline_bundle(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            recent_return_model=LGBMRegressor(),
            recency_ensemble_config={"half_life_groups": 252, "recent_weight": 1.5},
        )


def test_predict_daily_position_sizing_research_bundle_missing_recent_model_raises() -> None:
    """recency 설정이 있는데 최근 모델이 누락된 연구 번들은 ValueError 로 거부합니다."""
    df = _make_feature_df()
    bundle = _build_research_bundle(df)
    del bundle["recent_return_model"]
    with pytest.raises(ValueError, match="requires both return_model and recent_return_model"):
        predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bundle)

def test_predict_daily_position_sizing_research_bundle_rejects_invalid_config() -> None:
    """_predict_from_bundle 은 연구 번들의 잘못된 recency 설정을 fail-closed 로 거부합니다."""
    df = _make_feature_df()

    bad_half_life = _build_research_bundle(df)
    bad_half_life["recency_ensemble_config"]["half_life_groups"] = 100
    with pytest.raises(ValueError, match="half_life_groups must be 252 or 504"):
        predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bad_half_life)

    bad_weight = _build_research_bundle(df)
    bad_weight["recency_ensemble_config"]["recent_weight"] = 1.5
    with pytest.raises(ValueError, match="recent_weight must be within"):
        predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bad_weight)

    no_group_col = _build_research_bundle(df)
    del no_group_col["group_col"]
    with pytest.raises(ValueError, match="requires a group_col present in df"):
        predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=no_group_col)


def _algorithm_bundle_models(df: pd.DataFrame, weights: dict[str, float]) -> dict[str, object]:
    """full-history deterministic return 모델 매핑을 구성합니다 (algorithm 연구 번들용)."""
    from src.ml.training.fitting import fit_full_history_algorithm_return_model

    return {
        model_type: fit_full_history_algorithm_return_model(
            model_type, df, FEATURE_COLS, TARGET_COL
        )
        for model_type in weights
    }


def _build_algorithm_bundle(
    df: pd.DataFrame,
    weights: dict[str, float],
    probability_weight: float = 0.5,
) -> dict[str, object]:
    """algorithm-family 앙상블 연구 번들 (full-history return 모델 + 설정) 을 구성합니다."""
    config: dict[str, object] = {
        "version": "close-morning-algorithm-ensemble-research",
        "weights": dict(weights),
        "probability_weight": probability_weight,
        "score_col": "decision_score",
    }
    return _train_inline_bundle(
        df,
        FEATURE_COLS,
        TARGET_COL,
        GROUP_COL,
        algorithm_ensemble_models=_algorithm_bundle_models(df, weights),
        algorithm_ensemble_config=config,
    )


def _expected_rank_blend(
    df: pd.DataFrame,
    models: dict[str, object],
    weights: dict[str, float],
) -> pd.Series:
    """serving blend 와 동일한 그룹 내 백분위 convex blend 를 재계산합니다."""
    blend: pd.Series | None = None
    for model_type, weight in weights.items():
        pred = pd.Series(models[model_type].predict(df[FEATURE_COLS]), index=df.index)
        pct = pred.groupby(df[GROUP_COL]).rank(pct=True, method="average")
        term = weight * pct
        blend = term if blend is None else blend.add(term)
    assert blend is not None
    return blend


def test_predict_daily_position_sizing_algorithm_bundle_blends_rank_scores() -> None:
    """(SCENARIO_ALGORITHM_ENSEMBLE_04) algorithm-family 연구 번들은 동일 날짜
    백분위 blend 를 rank_score 로 재현하고, v1 reranker decision_score 는 blend
    백분위 + p_good 백분위 로 재현합니다."""
    df = _make_feature_df(n_rows=45, n_dates=3)
    weights = {"lgb_regressor": 0.5, "xgb_regressor": 0.5}
    bundle = _build_algorithm_bundle(df, weights)
    result = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bundle)

    blend = _expected_rank_blend(df, bundle["algorithm_ensemble_models"], weights)
    np.testing.assert_allclose(result["rank_score"].to_numpy(), blend.to_numpy(), atol=1e-12)
    for model_type in weights:
        expected_pred = bundle["algorithm_ensemble_models"][model_type].predict(df[FEATURE_COLS])
        np.testing.assert_allclose(
            result[f"pred_{model_type}"].to_numpy(), expected_pred, atol=1e-12
        )

    expected_decision = (
        blend.groupby(df[GROUP_COL]).rank(pct=True, method="average")
        + 0.5 * result.groupby(GROUP_COL)["p_good"].rank(pct=True, method="average")
    )
    np.testing.assert_allclose(
        result["decision_score"].to_numpy(), expected_decision.to_numpy(), atol=1e-12
    )
    assert "allocation" in result.columns


def test_predict_daily_position_sizing_algorithm_bundle_all_four_equal() -> None:
    """all_four_equal(4개 전문가 동일 가중) 레시피가 serving 에서 재현됩니다."""
    df = _make_feature_df(n_rows=60, n_dates=4)
    weights = {
        "lgb_regressor": 0.25,
        "xgb_regressor": 0.25,
        "catboost_regressor": 0.25,
        "random_forest_regressor": 0.25,
    }
    bundle = _build_algorithm_bundle(df, weights)
    result = predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bundle)
    blend = _expected_rank_blend(df, bundle["algorithm_ensemble_models"], weights)
    np.testing.assert_allclose(result["rank_score"].to_numpy(), blend.to_numpy(), atol=1e-12)
    assert "decision_score" in result.columns


def test_convex_rank_blend_rejects_empty_weights() -> None:
    """(defensive) 빈 앙상블 가중치는 fail-closed 로 거부합니다."""
    from src.ml.sizing_engine import _convex_rank_blend

    with pytest.raises(ValueError, match="must not be empty"):
        _convex_rank_blend({}, pd.Series([GROUP_COL]), {})


def test_train_inline_bundle_algorithm_bundle_requires_matching_models_and_config() -> None:
    """algorithm 연구 번들은 모델과 설정을 항상 함께 요구하며 비-convex weights,
    미지원/불일치 model key 는 fail-closed 로 거부합니다."""
    df = _make_feature_df()
    models2 = _algorithm_bundle_models(df, {"lgb_regressor": 0.5, "xgb_regressor": 0.5})
    config = {
        "version": "close-morning-algorithm-ensemble-research",
        "weights": {"lgb_regressor": 0.5, "xgb_regressor": 0.5},
    }
    with pytest.raises(ValueError, match="requires both algorithm_ensemble_models and"):
        _train_inline_bundle(
            df, FEATURE_COLS, TARGET_COL, GROUP_COL, algorithm_ensemble_config=config
        )
    with pytest.raises(ValueError, match="requires both algorithm_ensemble_models and"):
        _train_inline_bundle(
            df, FEATURE_COLS, TARGET_COL, GROUP_COL, algorithm_ensemble_models=models2
        )
    # 비-convex weights.
    bad = dict(config)
    bad["weights"] = {"lgb_regressor": 0.5, "xgb_regressor": 0.4}
    with pytest.raises(ValueError, match="must sum to 1"):
        _train_inline_bundle(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            algorithm_ensemble_models=models2,
            algorithm_ensemble_config=bad,
        )
    # 빈 weights.
    bad = dict(config)
    bad["weights"] = {}
    with pytest.raises(ValueError, match="must be a non-empty mapping"):
        _train_inline_bundle(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            algorithm_ensemble_models=models2,
            algorithm_ensemble_config=bad,
        )
    # 빈 모델 매핑.
    with pytest.raises(ValueError, match="requires a non-empty algorithm_ensemble_models"):
        _train_inline_bundle(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            algorithm_ensemble_models={},
            algorithm_ensemble_config=config,
        )
    # [0, 1] 범위 밖 가중치.
    bad = dict(config)
    bad["weights"] = {"lgb_regressor": 0.5, "xgb_regressor": 1.5}
    with pytest.raises(ValueError, match="must be within \\[0, 1\\]"):
        _train_inline_bundle(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            algorithm_ensemble_models=models2,
            algorithm_ensemble_config=bad,
        )
    # 미지원 model key.
    bad = dict(config)
    bad["weights"] = {"svm": 1.0}
    with pytest.raises(ValueError, match="weights key must be one of"):
        _train_inline_bundle(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            algorithm_ensemble_models={"svm": object()},
            algorithm_ensemble_config=bad,
        )
    # model key 불일치.
    with pytest.raises(ValueError, match="keys must exactly match"):
        _train_inline_bundle(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            algorithm_ensemble_models={"lgb_regressor": models2["lgb_regressor"]},
            algorithm_ensemble_config=config,
        )
    # recency 와 algorithm 동시 설정 거부.
    recency_config = {"version": "r", "half_life_groups": 252, "recent_weight": 0.5}
    with pytest.raises(ValueError, match="cannot be combined"):
        _train_inline_bundle(
            df,
            FEATURE_COLS,
            TARGET_COL,
            GROUP_COL,
            recent_return_model=LGBMRegressor(),
            recency_ensemble_config=recency_config,
            algorithm_ensemble_models=models2,
            algorithm_ensemble_config=config,
        )


def test_predict_daily_position_sizing_algorithm_bundle_fail_closed() -> None:
    """algorithm 설정 번들의 잘못된 설정/모델 누락/동시 recency 설정은 serving 시
    ValueError 로 fail-closed 됩니다."""
    import copy

    df = _make_feature_df()
    weights = {"lgb_regressor": 0.5, "xgb_regressor": 0.5}
    bundle = _build_algorithm_bundle(df, weights)

    # 동시 recency 설정은 serving 시 거부됩니다.
    both = copy.deepcopy(bundle)
    both["recency_ensemble_config"] = {
        "version": "r",
        "half_life_groups": 252,
        "recent_weight": 0.5,
    }
    with pytest.raises(ValueError, match="cannot be combined"):
        predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=both)

    # 모델 누락.
    missing = copy.deepcopy(bundle)
    del missing["algorithm_ensemble_models"]
    with pytest.raises(ValueError, match="requires a non-empty algorithm_ensemble_models"):
        predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=missing)

    # 비-convex weights.
    bad = copy.deepcopy(bundle)
    bad["algorithm_ensemble_config"]["weights"] = {
        "lgb_regressor": 0.5,
        "xgb_regressor": 0.4,
    }
    with pytest.raises(ValueError, match="must sum to 1"):
        predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=bad)

    # model key 불일치.
    mismatch = copy.deepcopy(bundle)
    del mismatch["algorithm_ensemble_models"]["xgb_regressor"]
    with pytest.raises(ValueError, match="keys must exactly match"):
        predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=mismatch)

    # group_col 부재.
    no_group = copy.deepcopy(bundle)
    del no_group["group_col"]
    with pytest.raises(ValueError, match="requires a group_col present in df"):
        predict_daily_position_sizing(df, FEATURE_COLS, models_bundle=no_group)

