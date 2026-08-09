"""Bundle-driven prediction, decision-score, utility, grade, and risk-limit tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.serving.realtime.inference import (
    add_close_morning_decision_score,
    apply_risk_limits,
    assign_sizing_grades,
    calculate_utility_score,
    predict_daily_sizing,
)

from tests.unit.serving.realtime.fixtures import (
    build_fixed_serving_bundle,
    fixed_reranker_bundle,
    scored_snapshot_df,
    snapshot_feature_cols,
)

GROUP_COL = "date"
FEATURE_COLS = snapshot_feature_cols()


def _prediction_frame(n: int = 12) -> pd.DataFrame:
    rng = __import__("numpy").random.default_rng(3)
    df = pd.DataFrame(
        {
            "pred_q10": rng.uniform(-0.03, 0.005, size=n),
            "pred_q50": rng.uniform(0.005, 0.04, size=n),
            "pred_q90": rng.uniform(0.03, 0.07, size=n),
            "p_good": rng.uniform(0.1, 0.9, size=n),
            "p_bad": rng.uniform(0.05, 0.5, size=n),
        }
    )
    df["pred_q10"] = df[["pred_q10", "pred_q50"]].min(axis=1)
    df["pred_q90"] = df[["pred_q50", "pred_q90"]].max(axis=1)
    df[GROUP_COL] = "2026-08-04"
    return df


def test_utility_score_calculation_matches_formula() -> None:
    df = _prediction_frame()
    series = calculate_utility_score(df)
    expected = df["pred_q50"] - 0.5 * (df["pred_q50"] - df["pred_q10"])
    assert series.name == "utility_score"
    assert series.isna().sum() == 0


def test_utility_score_raises_on_missing_prediction_columns() -> None:
    df = _prediction_frame().drop(columns=["pred_q90"])
    with pytest.raises(ValueError, match="missing required prediction columns"):
        calculate_utility_score(df)


def test_assign_sizing_grades_returns_grade_columns() -> None:
    df = _prediction_frame()
    df["utility_score"] = [float(i) for i in range(12)]
    result = assign_sizing_grades(df, group_col=GROUP_COL)
    assert {"grade", "grade_multiplier"}.issubset(result.columns)
    assert result["grade"].isin(["Strong", "Good", "Weak", "Pass"]).all()


def test_apply_risk_limits_returns_nonnegative_allocation() -> None:
    df = _prediction_frame()
    df["utility_score"] = [float(i) for i in range(12)]
    graded = assign_sizing_grades(df, group_col=GROUP_COL)
    result = apply_risk_limits(graded, group_col=GROUP_COL)
    assert result["allocation"].ge(0.0).all()
    assert result["allocation"].max() <= 0.25 + 1e-9


def test_predict_daily_sizing_returns_contract_columns() -> None:
    bundle = build_fixed_serving_bundle(FEATURE_COLS)
    snapshot = scored_snapshot_df(FEATURE_COLS, group_col=GROUP_COL)
    result = predict_daily_sizing(snapshot, bundle, group_col=GROUP_COL)
    assert {"utility_score", "grade", "allocation"}.issubset(result.columns)
    assert result["grade"].isin(["Strong", "Good", "Weak", "Pass"]).all()
    assert result["allocation"].ge(0.0).all()


def test_fixed_bundle_snapshot_output_is_deterministic() -> None:
    """고정 번들 + 스냅샷 픽스처는 결정적으로 동일한 결정 출력을 산출합니다.

    (추출 전후 비교용 고정 픽스처: rank_score/decision_score/utility_score/
    grade/allocation 을 재실행마다 정확히 재현합니다.)
    """
    bundle = fixed_reranker_bundle(FEATURE_COLS)
    snapshot = scored_snapshot_df(FEATURE_COLS, group_col=GROUP_COL)
    first = predict_daily_sizing(snapshot, bundle, group_col=GROUP_COL)
    second = predict_daily_sizing(snapshot, bundle, group_col=GROUP_COL)
    pd.testing.assert_frame_equal(first, second)
    for col in ("rank_score", "decision_score", "utility_score", "grade", "allocation"):
        assert col in first.columns


def test_predict_daily_sizing_adds_missing_group_col() -> None:
    bundle = build_fixed_serving_bundle(FEATURE_COLS)
    snapshot = scored_snapshot_df(FEATURE_COLS, group_col=GROUP_COL).drop(
        columns=[GROUP_COL]
    )
    result = predict_daily_sizing(snapshot, bundle, group_col=GROUP_COL)
    assert GROUP_COL in result.columns
    assert len(result) == len(snapshot)


def test_predict_daily_sizing_fills_missing_features() -> None:
    bundle = build_fixed_serving_bundle(FEATURE_COLS)
    snapshot = scored_snapshot_df(FEATURE_COLS, group_col=GROUP_COL).drop(
        columns=[FEATURE_COLS[1]]
    )
    result = predict_daily_sizing(snapshot, bundle, group_col=GROUP_COL)
    assert len(result) == len(snapshot)
    assert {"utility_score", "grade", "allocation"}.issubset(result.columns)


def test_predict_daily_sizing_raises_without_feature_cols() -> None:
    with pytest.raises(ValueError, match="feature_cols is empty"):
        predict_daily_sizing(
            scored_snapshot_df(FEATURE_COLS, group_col=GROUP_COL),
            {"dummy": 1},
            group_col=GROUP_COL,
        )


def test_predict_daily_sizing_appends_decision_score_only_for_reranker() -> None:
    snapshot = scored_snapshot_df(FEATURE_COLS, group_col=GROUP_COL)
    legacy = build_fixed_serving_bundle(FEATURE_COLS)
    legacy_out = predict_daily_sizing(snapshot, legacy, group_col=GROUP_COL)
    assert "decision_score" not in legacy_out.columns

    reranker = fixed_reranker_bundle(FEATURE_COLS)
    reranker_out = predict_daily_sizing(snapshot, reranker, group_col=GROUP_COL)
    assert "decision_score" in reranker_out.columns
    expected = (
        reranker_out.groupby(GROUP_COL)["rank_score"].rank(pct=True, method="average")
        + 0.5 * reranker_out.groupby(GROUP_COL)["p_good"].rank(pct=True, method="average")
    )
    pd.testing.assert_series_equal(
        reranker_out["decision_score"], expected.rename("decision_score")
    )


def test_add_close_morning_decision_score_raises_on_missing_group() -> None:
    scored = scored_snapshot_df(FEATURE_COLS, group_col=GROUP_COL).drop(
        columns=[GROUP_COL]
    )
    with pytest.raises(ValueError, match="missing required columns"):
        add_close_morning_decision_score(scored, group_col=GROUP_COL)
