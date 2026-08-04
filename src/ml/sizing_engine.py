"""Utility Score 기반 Dynamic Sizing 및 위험 한도 적용 모듈.

Quantile 예측(q10/q50/q90)과 Calibrated 확률(p_good/p_bad)을 결합한 종합
Utility Score 를 산출하고, Utility 백분위 기반 등급(Strong/Good/Weak/Pass)을
부여한 뒤 변동성 역가중 비중과 개별/총투자 한도(Cap)를 적용합니다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PREDICTION_COLS = ("pred_q10", "pred_q50", "pred_q90", "p_good", "p_bad")

_STRONG_PCT = 0.90
_GOOD_PCT = 0.75
_WEAK_PCT = 0.50

_GRADE_MULTIPLIERS: dict[str, float] = {
    "Strong": 1.5,
    "Good": 1.0,
    "Weak": 0.5,
    "Pass": 0.0,
}


def calculate_utility_score(
    df: pd.DataFrame,
    lambda_risk: float = 1.5,
    gamma_uncertainty: float = 0.2,
    w_good: float = 0.5,
    w_bad: float = 0.5,
) -> pd.Series:
    """하방위험/불확실성 패널티가 적용된 종합 Utility Score 를 산출합니다.

    U_i = q50_i - lambda·max(0, -q10_i) - gamma·(q90_i - q10_i)
          + w_good·p_good_i - w_bad·p_bad_i
    """
    missing = [col for col in _PREDICTION_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required prediction columns in df: {missing}")
    q10 = df["pred_q10"].to_numpy(dtype=np.float64)
    q50 = df["pred_q50"].to_numpy(dtype=np.float64)
    q90 = df["pred_q90"].to_numpy(dtype=np.float64)
    p_good = df["p_good"].to_numpy(dtype=np.float64)
    p_bad = df["p_bad"].to_numpy(dtype=np.float64)
    downside = lambda_risk * np.maximum(0.0, -q10)
    uncertainty = gamma_uncertainty * (q90 - q10)
    utility = q50 - downside - uncertainty + w_good * p_good - w_bad * p_bad
    return pd.Series(utility, index=df.index, name="utility_score")


def assign_sizing_grades(
    df: pd.DataFrame,
    utility_col: str = "utility_score",
    group_col: str = "date",
) -> pd.DataFrame:
    """그룹(날짜) 내 Utility Score 백분위 기준 등급 및 배수를 부여합니다.

    - Strong: 상위 10% (배수 1.5)
    - Good:   상위 25% (배수 1.0)
    - Weak:   상위 50% 이면서 기대수익(q50) 양수 (배수 0.5)
    - Pass:   기타 (배수 0.0)
    """
    if utility_col not in df.columns:
        raise ValueError(f"utility_col {utility_col!r} is missing in df")
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} is missing in df")

    out = df.copy()
    has_q50 = "pred_q50" in out.columns
    grades: list[str] = []
    for idx in out.groupby(group_col, sort=False).groups.values():
        pct = out.loc[idx, utility_col].rank(pct=True, method="average").to_numpy()
        weak = pct >= _WEAK_PCT
        if has_q50:
            weak = weak & (out.loc[idx, "pred_q50"].to_numpy(dtype=np.float64) > 0.0)
        strong = pct >= _STRONG_PCT
        good = pct >= _GOOD_PCT
        grades.extend(np.select([strong, good, weak], ["Strong", "Good", "Weak"], default="Pass").tolist())
    out["grade"] = grades
    out["grade_multiplier"] = out["grade"].map(_GRADE_MULTIPLIERS).to_numpy(dtype=np.float64)
    return out


def apply_risk_limits(
    df: pd.DataFrame,
    base_budget: float = 1.0,
    target_vol: float = 0.15,
    max_position_pct: float = 0.25,
    max_total_allocation: float = 1.0,
    group_col: str = "date",
) -> pd.DataFrame:
    """등급 배수 * 변동성 역가중 비중을 산정하고 위험 한도를 적용합니다.

    Position_i = BaseBudget * GradeMultiplier_i * (TargetVol / sigma_i)

    ``sigma_i`` 는 모델 불확실성 프록시인 분위수 스프레드(q90 - q10)로 추정하며,
    그룹별 합계를 ``max_total_allocation`` 으로, 개별 비중을
    ``max_position_pct`` 로 클리핑합니다.
    """
    if "grade_multiplier" not in df.columns:
        raise ValueError("grade_multiplier is missing in df; run assign_sizing_grades first")
    if "pred_q10" not in df.columns or "pred_q90" not in df.columns:
        raise ValueError("missing required prediction columns in df: pred_q10/pred_q90")

    out = df.copy()
    sigma = np.maximum((out["pred_q90"] - out["pred_q10"]).to_numpy(dtype=np.float64), 1e-8)
    multiplier = out["grade_multiplier"].to_numpy(dtype=np.float64)
    raw = base_budget * multiplier * (target_vol / sigma)
    allocation = np.zeros_like(raw)
    for idx in out.groupby(group_col, sort=False).groups.values():
        pos = out.index.get_indexer(idx)
        group_raw = raw[pos]
        total = float(group_raw.sum())
        if total <= 0.0:
            continue
        scale = min(1.0, max_total_allocation / total)
        scaled = np.minimum(group_raw * scale, max_position_pct)
        allocation[pos] = scaled
    out["allocation"] = allocation
    return out
