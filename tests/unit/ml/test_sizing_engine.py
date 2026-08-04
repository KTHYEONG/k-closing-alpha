"""Utility Score 기반 Dynamic Sizing 단위 테스트.

SCENARIO: utility_score_calculation
SCENARIO: sizing_grade_assignment
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.sizing_engine import apply_risk_limits, assign_sizing_grades, calculate_utility_score

GROUP_COL = "date"


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
    """U_i = q50 - lambda*max(0,-q10) - gamma*(q90-q10) + w_good*p_good - w_bad*p_bad

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


def test_apply_risk_limits_raises_on_missing_columns() -> None:
    df = _make_utility_df()
    with pytest.raises(ValueError, match="grade_multiplier"):
        apply_risk_limits(df)
    graded = assign_sizing_grades(_with_utility(df, [float(i) for i in range(len(df))]))
    with pytest.raises(ValueError, match="pred_q"):
        apply_risk_limits(graded.drop(columns=["pred_q10"]))
