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
    """U_i = (q50 - cost) - lambda*max(0, -(q10 - cost)) - gamma*(q90 - q10)
          + w_good*p_good - w_bad*p_bad

    (SCENARIO: utility_score_calculation)
    """
    df = _make_utility_df()
    lambda_risk, gamma_uncertainty, w_good, w_bad = 1.5, 0.2, 0.5, 0.5
    round_trip_cost = 0.0020
    series = calculate_utility_score(
        df,
        lambda_risk=lambda_risk,
        gamma_uncertainty=gamma_uncertainty,
        w_good=w_good,
        w_bad=w_bad,
        round_trip_cost=round_trip_cost,
    )
    assert isinstance(series, pd.Series)
    assert series.name == "utility_score"
    expected = (
        (df["pred_q50"] - round_trip_cost)
        - lambda_risk * np.maximum(0.0, -(df["pred_q10"] - round_trip_cost))
        - gamma_uncertainty * (df["pred_q90"] - df["pred_q10"])
        + w_good * df["p_good"]
        - w_bad * df["p_bad"]
    )
    np.testing.assert_allclose(series.to_numpy(), expected.to_numpy())


def test_utility_score_deducts_round_trip_cost() -> None:
    """(SCENARIO: utility_score_calculation) 기본 round_trip_cost=0.0020 이
    기대수익 q50 에서 차감되어 순유틸리티가 q50 보다 낮고, 비용 없음보다 작다."""
    df = pd.DataFrame(
        {
            "pred_q10": [0.0],
            "pred_q50": [0.005],
            "pred_q90": [0.01],
            "p_good": [0.5],
            "p_bad": [0.5],
        }
    )
    assert float(calculate_utility_score(df).iloc[0]) < 0.005
    no_cost = float(calculate_utility_score(df, round_trip_cost=0.0).iloc[0])
    with_cost = float(calculate_utility_score(df).iloc[0])
    assert with_cost < no_cost


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
    """(SCENARIO: sizing_grade_assignment) 음수 순유틸리티 또는 비용 차감 후
    기대수익(net_q50)이 0 이하인 항목은 상위 백분위여도 Pass 등급을 받는다."""
    negative_util = _with_utility(_make_utility_df(n_rows=10), [-0.01] * 10)
    negative_util["pred_q50"] = 0.05
    result = assign_sizing_grades(negative_util, utility_col="utility_score", group_col=GROUP_COL)
    assert (result["grade"] == "Pass").all()

    zero_net = _with_utility(_make_utility_df(n_rows=10), [float(i) for i in range(10)])
    zero_net["pred_q50"] = 0.0020
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
    graded = assign_sizing_grades(_with_utility(df, [float(i) for i in range(len(df))]))
    with pytest.raises(ValueError, match="pred_q"):
        apply_risk_limits(graded.drop(columns=["pred_q10"]))


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

