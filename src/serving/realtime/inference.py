"""Bundle-driven prediction, dynamic sizing, and risk limits for the live path.

This module reproduces the published bundle's inference contract exactly:
artifact prediction (rank/quantile/calibrated probabilities), close-morning
decision-score, utility score, grade assignment, and risk-limit allocation.
It never trains, calibrates, or saves models.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd

_PREDICTION_COLS = ("pred_q10", "pred_q50", "pred_q90", "p_good", "p_bad")

# 왕복 거래 비용 (매수/매도 수수료 0.028% + 매도 거래세 0.15% + 슬리피지 0.022% ≈ 0.20%).
ROUND_TRIP_COST_RATIO: float = 0.0020

_QUANTILE_COLS = ("pred_q10", "pred_q50", "pred_q90")
_QUANTILE_ALPHAS = (0.10, 0.50, 0.90)

_STRONG_PCT = 0.90
_GOOD_PCT = 0.75
_WEAK_PCT = 0.50

# 변동성 타게팅용 별도 실현 변동성 기본값 (q90-q10 은 불확실성 지표일 뿐
# 실현 변동성이 아니므로 시그마 추정에 사용하지 않습니다).
_DEFAULT_REALIZED_VOL = 0.02
_REALIZED_VOL_COL = "realized_vol"

_GRADE_MULTIPLIERS: dict[str, float] = {
    "Strong": 1.5,
    "Good": 1.0,
    "Weak": 0.5,
    "Pass": 0.0,
}

# close-morning reranker v1 불변 설정 (후보 번들에 영속화되며 추론 시점에만 소비).
_CLOSE_MORNING_RERANKER_CONFIG: dict[str, Any] = {
    "version": "close-morning-reranker-v1",
    "rank_weight": 1.0,
    "p_good_weight": 0.0,
    "score_col": "decision_score",
}

# close-morning reranker v2 연구 설정: ``p_bad`` 하방위험 패널티를 명시적으로
# 선언하는 연구 번들에서만 소비됩니다. 프로덕션 기본값이나 v1 번들은 변경하지
# 않습니다 (+1% good / -2% bad 라벨 비대칭 계약의 제한적 그리드).
_CLOSE_MORNING_RERANKER_V2_RESEARCH_CONFIG: dict[str, Any] = {
    "version": "close-morning-reranker-v2-research",
    "rank_weight": 1.0,
    "p_good_weight": 0.5,
    "bad_probability_weight": 0.5,
    "score_col": "decision_score",
}

# algorithm-family 앙상블 연구(ml_ensemble_improvement)에서 사용 가능한 return
# 추정기 family 목록. ``algorithm_ensemble_config.weights`` 의 key 는 이 목록의
# 하나여야 하며 ``algorithm_ensemble_models`` 의 key 와 정확히 일치해야 합니다.
_ALGORITHM_MODEL_TYPES: tuple[str, ...] = (
    "lgb_regressor",
    "xgb_regressor",
    "catboost_regressor",
    "random_forest_regressor",
)


def _convex_rank_blend(
    preds: dict[str, pd.Series],
    group: pd.Series,
    weights: dict[str, float],
) -> pd.Series:
    """각 모델 예측의 그룹 내 백분위 순위를 convex weighted mean 으로 blend 합니다.

    서로 다른 estimator family 의 raw 예측값은 스케일/손실이 달라 직접 평균할 수
    없으므로, 벡터화된 ``groupby().rank`` 로 백분위 순위로 변환한 뒤 가중 평균을
    계산합니다 (row-wise apply 미사용).
    """
    blend: pd.Series | None = None
    for model_type, weight in weights.items():
        pct = preds[model_type].groupby(group).rank(pct=True, method="average")
        term = weight * pct
        blend = term if blend is None else blend.add(term)
    if blend is None:
        raise ValueError("ensemble weights must not be empty")
    return blend


def _validate_algorithm_ensemble_config(
    algorithm_ensemble_models: Any,
    algorithm_ensemble_config: Any,
) -> dict[str, float]:
    """연구 번들의 algorithm 앙상블 설정을 fail-closed 로 검증합니다.

    ``weights`` 는 비어 있지 않은 convex 매핑(각 원소 [0, 1] 이고 합이 1)이어야
    하고, model key 는 지원 family 중 하나여야 하며 ``algorithm_ensemble_models``
    의 key 와 정확히 일치해야 합니다. 검증된 float weights dict 를 반환합니다.
    """
    weights = algorithm_ensemble_config.get("weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("algorithm_ensemble_config.weights must be a non-empty mapping")
    if not isinstance(algorithm_ensemble_models, dict) or not algorithm_ensemble_models:
        raise ValueError(
            "algorithm_ensemble_config requires a non-empty algorithm_ensemble_models"
        )
    if set(weights) != set(algorithm_ensemble_models):
        raise ValueError(
            "algorithm_ensemble_models keys must exactly match "
            "algorithm_ensemble_config.weights keys"
        )
    total = 0.0
    for model_type, weight in weights.items():
        if model_type not in _ALGORITHM_MODEL_TYPES:
            raise ValueError(
                "algorithm_ensemble_config.weights key must be one of "
                f"{list(_ALGORITHM_MODEL_TYPES)}, got {model_type!r}"
            )
        if not isinstance(weight, (int, float)) or not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"algorithm_ensemble_config.weights[{model_type!r}] must be within [0, 1], "
                f"got {weight!r}"
            )
        total += float(weight)
    if not np.isclose(total, 1.0, atol=1e-9):
        raise ValueError(f"algorithm_ensemble_config.weights must sum to 1, got {total}")
    return {model_type: float(weight) for model_type, weight in weights.items()}


def calculate_utility_score(
    df: pd.DataFrame,
    lambda_risk: float = 0.5,
    gamma_uncertainty: float = 0.1,
    w_good: float = 0.0,
    w_bad: float = 0.0,
    round_trip_cost: float = ROUND_TRIP_COST_RATIO,
) -> pd.Series:
    """하방위험/불확실성 패널티가 적용된 종합 Utility Score 를 산출합니다.

    U_i = q50_i - lambda·max(0, -q10_i) - gamma·(q90_i - q10_i)
          + w_good·p_good_i - w_bad·p_bad_i

    입력 분위수/확률 예측은 이미 decimal net return 단위(비용 차감 완료)이므로
    여기서 거래 비용을 다시 차감하지 않습니다 (``round_trip_cost`` 는 타깃 구성
    시점에 정확히 1회 반영된 비용으로, 이 함수는 예측이 net 기준임을 명시합니다).
    ``q90 - q10`` 은 불확실성 스프레드이며 실현 변동성이 아닙니다.
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
    min_good_utility: float = 0.0030,
    min_weak_utility: float = 0.0010,
    round_trip_cost: float = ROUND_TRIP_COST_RATIO,
) -> pd.DataFrame:
    """그룹(날짜) 내 Utility Score 백분위 + 절대 임계값 혼합(Hybrid) 등급 부여.

    상대백분위(pct)와 절대 Utility Score, 순 기대수익(net_q50) 을 동시에 만족해야
    해당 등급을 부여합니다 (Relative Evaluation Trap 방지). 예측 ``pred_q50`` 은
    이미 decimal net return(비용 차감 완료)이므로 여기서 비용을 추가 차감하지 않고
    ``pred_q50 > 0`` 인 종목만 거래 후보로 승인합니다.

    - Strong: ``pct >= 0.90`` AND ``utility >= min_good_utility`` AND ``net_q50 > 0``
    - Good:   ``pct >= 0.75`` AND ``utility >= min_weak_utility`` AND ``net_q50 > 0``
    - Weak:   ``pct >= 0.50`` AND ``utility >= min_weak_utility`` AND ``net_q50 > 0``
    - Pass:   절대 임계값 미달 또는 ``net_q50 <= 0`` 또는 ``pct < 0.50`` (배수 0.0)
    """
    if utility_col not in df.columns:
        raise ValueError(f"utility_col {utility_col!r} is missing in df")
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} is missing in df")

    out = df.copy()
    has_q50 = "pred_q50" in out.columns
    grades: list[str] = []
    for idx in out.groupby(group_col, sort=False).groups.values():
        u = out.loc[idx, utility_col].to_numpy(dtype=np.float64)
        pct = out.loc[idx, utility_col].rank(pct=True, method="average").to_numpy()
        strong = (pct >= _STRONG_PCT) & (u >= min_good_utility)
        good = (pct >= _GOOD_PCT) & (u >= min_weak_utility)
        weak = (pct >= _WEAK_PCT) & (u >= min_weak_utility)
        if has_q50:
            net_positive = out.loc[idx, "pred_q50"].to_numpy(dtype=np.float64) > 0.0
            strong = strong & net_positive
            good = good & net_positive
            weak = weak & net_positive
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
    utility_col: str = "utility_score",
) -> pd.DataFrame:
    """등급 배수 * 변동성 역가중 * Utility Magnitude 비중을 산정하고 위험 한도를 적용합니다.

    Position_i = BaseBudget * GradeMultiplier_i * (TargetVol / sigma_i) * utility_scaling_i

    ``utility_scaling_i = clip(utility_i / 0.01, 0.1, 1.5)`` 로 Utility Score 의
    절대적 크기를 비중에 반영하며, ``sigma_i`` 는 별도 실현 변동성(``realized_vol``
    컬럼 또는 기본값)입니다. 분위수 스프레드(q90-q10) 는 불확실성 지표이므로
    실현 변동성으로 사용하지 않습니다. 그룹별 합계를 ``max_total_allocation``
    으로, 개별 비중을 ``max_position_pct`` 로 클리핑하고, 음수 순유틸리티
    종목은 비중 0% 로 강제합니다.

    시장 국면 방어: 그룹 후보 유니버스의 평균 Utility Score 가 음수이면
    ``max_total_allocation`` 을 ``max(1 + avg_utility, 0)`` 비율로 축소하여
    불리한 시장 국면에서 자본을 보호합니다. ``utility_col`` 이 없으면
    방어/크기 가중 없이 기존 한도만 적용합니다.
    """
    if "grade_multiplier" not in df.columns:
        raise ValueError("grade_multiplier is missing in df; run assign_sizing_grades first")

    out = df.copy()
    # 변동성 타게팅: 실현 변동성(별도 예측/지연 컬럼) 또는 기본값을 사용합니다.
    # 분위수 스프레드(q90-q10)는 불확실성 지표이므로 실현 변동성으로 사용하지 않습니다.
    if _REALIZED_VOL_COL in out.columns:
        sigma = np.maximum(out[_REALIZED_VOL_COL].to_numpy(dtype=np.float64), 1e-8)
    else:
        sigma = np.full(len(out), _DEFAULT_REALIZED_VOL)
    multiplier = out["grade_multiplier"].to_numpy(dtype=np.float64)
    has_utility = utility_col in out.columns
    utility = (
        out[utility_col].to_numpy(dtype=np.float64)
        if has_utility
        else np.zeros(len(out), dtype=np.float64)
    )
    utility_scaling = (
        np.clip(utility / 0.01, 0.1, 1.5) if has_utility else np.ones(len(out), dtype=np.float64)
    )
    raw = base_budget * multiplier * (target_vol / sigma) * utility_scaling
    allocation = np.zeros_like(raw)
    group_utility = utility if has_utility else None
    for idx in out.groupby(group_col, sort=False).groups.values():
        pos = out.index.get_indexer(idx)
        group_raw = raw[pos]
        total = float(group_raw.sum())
        if total <= 0.0:
            continue
        effective_max_total = max_total_allocation
        if group_utility is not None:
            actionable_mask = (
                (out.loc[idx, "grade"] != "Pass").to_numpy()
                if "grade" in out.columns
                else np.ones(len(pos), dtype=bool)
            )
            actionable_utils = group_utility[pos][actionable_mask]
            avg_utility = (
                float(actionable_utils.mean())
                if len(actionable_utils) > 0
                else float(group_utility[pos].mean())
            )
            if avg_utility < 0.0:
                effective_max_total = max_total_allocation * max(1.0 + avg_utility, 0.0)
        scale = min(1.0, effective_max_total / total)
        scaled = np.minimum(group_raw * scale, max_position_pct)
        allocation[pos] = scaled

    if has_utility:
        if "grade" in out.columns:
            allocation[(utility < 0.0) & (out["grade"] == "Pass")] = 0.0
        else:
            allocation[utility < 0.0] = 0.0
    out["allocation"] = allocation
    return out


def _predict_from_bundle(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_bundle: dict[str, Any],
) -> pd.DataFrame:
    """저장된 번들 모델로 rank/분위수/확률 예측 컬럼을 산출합니다."""
    features = df[feature_cols]
    out = df.copy()

    # rank_score 는 회귀 champion(return_model) 의 기대수익 예측으로 생성하며,
    # return_model 이 없는 기존 번들은 rank_model 로 폴백합니다.
    return_model = models_bundle.get("return_model")
    recency_config = models_bundle.get("recency_ensemble_config")
    algorithm_config = models_bundle.get("algorithm_ensemble_config")
    if algorithm_config is not None and recency_config is not None:
        raise ValueError(
            "recency_ensemble_config and algorithm_ensemble_config cannot be combined"
        )
    if algorithm_config is not None:
        algorithm_models = models_bundle.get("algorithm_ensemble_models")
        if algorithm_models is None:
            raise ValueError(
                "algorithm_ensemble_config requires a non-empty algorithm_ensemble_models"
            )
        weights = _validate_algorithm_ensemble_config(algorithm_models, algorithm_config)
        bundle_group_col = models_bundle.get("group_col")
        if bundle_group_col is None or bundle_group_col not in out.columns:
            raise ValueError(
                "algorithm ensemble research bundle requires a group_col present in df"
            )
        preds = {
            model_type: pd.Series(algorithm_models[model_type].predict(features), index=df.index)
            for model_type in weights
        }
        out["rank_score"] = _convex_rank_blend(preds, out[bundle_group_col], weights)
        for model_type, pred_series in preds.items():
            out[f"pred_{model_type}"] = pred_series
    elif recency_config is not None:
        recent_return_model = models_bundle.get("recent_return_model")
        if return_model is None or recent_return_model is None:
            raise ValueError(
                "recency ensemble research bundle requires both return_model and "
                "recent_return_model"
            )
        half_life = recency_config.get("half_life_groups")
        recent_weight = recency_config.get("recent_weight")
        if half_life not in (252, 504):
            raise ValueError(
                f"recency_ensemble_config.half_life_groups must be 252 or 504, got {half_life!r}"
            )
        if not isinstance(recent_weight, (int, float)) or not 0.0 <= recent_weight <= 1.0:
            raise ValueError(
                f"recency_ensemble_config.recent_weight must be within [0, 1], "
                f"got {recent_weight!r}"
            )
        bundle_group_col = models_bundle.get("group_col")
        if bundle_group_col is None or bundle_group_col not in out.columns:
            raise ValueError(
                "recency ensemble research bundle requires a group_col present in df"
            )
        expanding_pred = pd.Series(return_model.predict(features), index=df.index)
        recent_pred = pd.Series(recent_return_model.predict(features), index=df.index)
        out["rank_score"] = _convex_rank_blend(
            {"expanding": expanding_pred, "recent": recent_pred},
            out[bundle_group_col],
            {"expanding": 1.0 - recent_weight, "recent": recent_weight},
        )
        out["pred_expanding"] = expanding_pred
        out["pred_recent"] = recent_pred
    elif return_model is not None:
        out["rank_score"] = return_model.predict(features)
    else:
        rank_model = models_bundle.get("rank_model")
        if rank_model is not None:
            out["rank_score"] = rank_model.predict(features)

    quantile_models = models_bundle["quantile_models"]
    for col in _QUANTILE_COLS:
        out[col] = quantile_models[col].predict(features)

    for name in ("p_good", "p_bad"):
        calibrator = models_bundle["calibrators"][name]
        if isinstance(calibrator, float):
            out[name] = float(calibrator)
        else:
            proba = calibrator.predict_proba(features)
            positive_idx = list(calibrator.classes_).index(True)
            out[name] = proba[:, positive_idx]

    q10 = out["pred_q10"].to_numpy(dtype=np.float64)
    q50 = out["pred_q50"].to_numpy(dtype=np.float64)
    q90 = out["pred_q90"].to_numpy(dtype=np.float64)
    out["pred_q10"] = np.minimum(np.minimum(q10, q50), q90)
    out["pred_q50"] = np.clip(q50, out["pred_q10"].to_numpy(dtype=np.float64), q90)
    out["pred_q90"] = np.maximum(q90, out["pred_q50"].to_numpy(dtype=np.float64))
    return out


def add_close_morning_decision_score(
    df: pd.DataFrame,
    group_col: str = "date",
    rank_score_col: str = "rank_score",
    p_good_col: str = "p_good",
    p_bad_col: str = "p_bad",
    output_col: str = "decision_score",
    probability_weight: float = 0.5,
    bad_probability_weight: float = 0.0,
) -> pd.DataFrame:
    """close-morning reranker 결정 스코어를 그룹(날짜) 내 횡단면 백분위 순위로 산출합니다.

    ``decision_score = rank(rank_score, pct=True) + probability_weight * rank(p_good, pct=True)
    - bad_probability_weight * rank(p_bad, pct=True)``

    ``bad_probability_weight=0.0``(기본)이면 v1 스코어와 완전히 동일하며 ``p_bad``
    컬럼을 읽지 않습니다. v2 는 동일한 +1%/-2% 라벨 계약의 비대칭 손실 심각도를
    반영해 하방위험 패널티를 추가하는 리스크 통제 실험입니다. 벡터화된
    ``groupby().rank`` 를 사용하며 row-wise apply 를 피하고 타깃/수익률 컬럼을
    절대 읽지 않습니다(미래 정보 금지). 누락 그룹/스코어 컬럼, 비유한 스코어,
    또는 ``[0, 1]`` 을 벗어난 가중치는 ``ValueError`` 로 fail-closed 합니다.
    """
    missing = [col for col in (group_col, rank_score_col, p_good_col) if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns for close-morning decision score: {missing}")
    if not 0.0 <= probability_weight <= 1.0:
        raise ValueError(f"probability_weight must be within [0, 1], got {probability_weight}")
    if not 0.0 <= bad_probability_weight <= 1.0:
        raise ValueError(
            f"bad_probability_weight must be within [0, 1], got {bad_probability_weight}"
        )
    rank_score = df[rank_score_col].to_numpy(dtype=np.float64)
    p_good = df[p_good_col].to_numpy(dtype=np.float64)
    if not np.isfinite(rank_score).all() or not np.isfinite(p_good).all():
        raise ValueError(
            "rank_score and p_good must be finite for close-morning decision score"
        )
    rank_pct = df.groupby(group_col)[rank_score_col].rank(pct=True, method="average")
    p_good_pct = df.groupby(group_col)[p_good_col].rank(pct=True, method="average")
    score = rank_pct + probability_weight * p_good_pct
    if bad_probability_weight != 0.0:
        if p_bad_col not in df.columns:
            raise ValueError(
                f"missing required columns for close-morning decision score: ['{p_bad_col}']"
            )
        p_bad = df[p_bad_col].to_numpy(dtype=np.float64)
        if not np.isfinite(p_bad).all():
            raise ValueError(
                "p_bad must be finite for close-morning decision score"
            )
        p_bad_pct = df.groupby(group_col)[p_bad_col].rank(pct=True, method="average")
        score = score - bad_probability_weight * p_bad_pct
    out = df.copy()
    out[output_col] = score
    return out


def predict_daily_sizing(
    df: pd.DataFrame,
    models_bundle: dict[str, Any],
    group_col: str = "date",
) -> pd.DataFrame:
    """로드된 번들로 당일 스냅샷의 예측·Sizing·위험 한도를 산출합니다.

    모델 학습 시 사용된 ``feature_cols`` 를 번들에서 읽어 누락 컬럼을 0 으로
    채우고, ``group_col`` 이 없으면 오늘 날짜로 단일 그룹을 구성한 뒤
    ``_predict_from_bundle`` → decision-score(v1/v2) → utility → 등급 → 배분을
    계산합니다. 학습(인라인 번들 구성)은 절대 수행하지 않습니다.
    """
    feature_cols = list(models_bundle.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("feature_cols is empty; models_bundle must declare feature_cols")

    work = df.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    if group_col not in work.columns:
        work[group_col] = str(datetime.date.today())

    out = _predict_from_bundle(work, feature_cols, models_bundle)
    decision_score_config = models_bundle.get("decision_score_config")
    if (
        decision_score_config is not None
        and decision_score_config.get("version") == _CLOSE_MORNING_RERANKER_CONFIG["version"]
    ):
        p_good_weight = decision_score_config.get(
            "p_good_weight", _CLOSE_MORNING_RERANKER_CONFIG["p_good_weight"]
        )
        out = add_close_morning_decision_score(
            out, group_col=group_col, probability_weight=p_good_weight
        )
    elif (
        decision_score_config is not None
        and decision_score_config.get("version")
        == _CLOSE_MORNING_RERANKER_V2_RESEARCH_CONFIG["version"]
    ):
        p_good_weight = decision_score_config.get(
            "p_good_weight", _CLOSE_MORNING_RERANKER_CONFIG["p_good_weight"]
        )
        bad_probability_weight = decision_score_config.get(
            "bad_probability_weight",
            _CLOSE_MORNING_RERANKER_V2_RESEARCH_CONFIG["bad_probability_weight"],
        )
        out = add_close_morning_decision_score(
            out,
            group_col=group_col,
            probability_weight=p_good_weight,
            bad_probability_weight=bad_probability_weight,
        )
    out["utility_score"] = calculate_utility_score(out)
    out = assign_sizing_grades(out, group_col=group_col)
    out = apply_risk_limits(out, group_col=group_col)
    return out
