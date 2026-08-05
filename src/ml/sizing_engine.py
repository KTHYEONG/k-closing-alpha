"""Utility Score 기반 Dynamic Sizing 및 위험 한도 적용 모듈.

Quantile 예측(q10/q50/q90)과 Calibrated 확률(p_good/p_bad)을 결합한 종합
Utility Score 를 산출하고, Utility 백분위 기반 등급(Strong/Good/Weak/Pass)을
부여한 뒤 변동성 역가중 비중과 개별/총투자 한도(Cap)를 적용합니다.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump, load
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV

from src.ml.feature_manifest import build_feature_manifest

_PREDICTION_COLS = ("pred_q10", "pred_q50", "pred_q90", "p_good", "p_bad")

# 왕복 거래 비용 (매수/매도 수수료 0.028% + 매도 거래세 0.15% + 슬리피지 0.022% ≈ 0.20%).
ROUND_TRIP_COST_RATIO: float = 0.0020

_BUNDLE_FILENAME = "sizing_pipeline_bundle.joblib"
_QUANTILE_COLS = ("pred_q10", "pred_q50", "pred_q90")
_QUANTILE_ALPHAS = (0.10, 0.50, 0.90)
# Decimal net 기준 이벤트 임계값 (preprocessor.LABEL_THRESHOLDS 와 동일)
_GOOD_THRESHOLD = 0.01
_BAD_THRESHOLD = -0.02

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

# LightGBM Booster 는 문자열/범주형 object 컬럼을 입력으로 허용하지 않으므로
# 학습 피처에서 완전히 제외합니다 (preprocessor._CATEGORICAL_COLUMNS 와 동일).
_CATEGORICAL_FEATURE_COLS: tuple[str, ...] = (
    "market_type",
    "theme_sector",
    "chart_analysis",
)

# close-morning reranker v1 불변 설정 (후보 번들에 영속화되며 추론 시점에만 소비).
_CLOSE_MORNING_RERANKER_CONFIG: dict[str, Any] = {
    "version": "close-morning-reranker-v1",
    "rank_weight": 1.0,
    "p_good_weight": 0.5,
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


def save_model_artifacts(artifacts: dict[str, Any], export_dir: str = "artifacts/models") -> str:
    """학습된 모델 번들 dict 를 joblib 로 지정 디렉터리에 저장합니다.

    ``export_dir`` 이 존재하지 않으면 생성하며, 번들 파일은
    ``sizing_pipeline_bundle.joblib`` 이름으로 저장됩니다.
    """
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, _BUNDLE_FILENAME)
    dump(artifacts, path)
    return os.path.abspath(path)


def load_model_artifacts(import_dir: str = "artifacts/models") -> dict[str, Any]:
    """joblib 로 저장된 모델 번들 dict 를 로드합니다.

    디렉터리 또는 번들 파일이 없으면 ``FileNotFoundError`` 를 발생시킵니다.
    """
    path = os.path.join(import_dir, _BUNDLE_FILENAME)
    if not os.path.isdir(import_dir) or not os.path.isfile(path):
        raise FileNotFoundError(
            f"model artifact bundle not found at {path!r}; run training to save artifacts first"
        )
    bundle: dict[str, Any] = load(path)
    return bundle


def _fit_calibrator(features: pd.DataFrame, labels: np.ndarray) -> Any:
    """단일 클래스 또는 cv fold 를 채우지 못하는 소표본 클래스일 때 사전확률
    (prior) 상수, 아니면 Sigmoid Calibrated GBDT 를 반환합니다."""
    if np.unique(labels).size < 2:
        return float(np.mean(labels))
    min_class = int(np.min(np.bincount(labels)))
    if min_class < 3:
        return float(np.mean(labels))
    base = LGBMClassifier(objective="binary", random_state=42, verbosity=-1)
    calibrator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    calibrator.fit(features, labels)
    return calibrator


def _train_inline_bundle(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    calibration_diagnostics: list[dict[str, Any]] | None = None,
    recent_return_model: Any | None = None,
    recency_ensemble_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """타깃이 포함된 데이터로 즉시 학습 가능한 기본 모델 번들을 구성합니다.

    LGBMRanker(랭킹) + Quantile Regressor(q10/q50/q90) + Calibrated Classifier
    (p_good/p_bad) 5종 모델을 포함하며, ``predict_daily_position_sizing`` 의
    인라인 추론(``models_bundle=None``)과 학습 모드 저장에 사용됩니다.

    번들에는 P0 계약(``ml_strategy_improvement``)이 요구하는 반환 단위, 비용,
    라벨 임계값, decision-time 피처 매니페스트, training cutoff, 보정 진단,
    정책 파라미터를 영속화합니다.

    문자열/범주형 컬럼(``market_type``, ``theme_sector``, ``chart_analysis``)은
    Booster 구성 시 ValueError 를 유발하므로 ``feature_cols`` 에서 제외됩니다.

    opt-in 연구 번들: ``recent_return_model``(half-life recent Huber)과
    ``recency_ensemble_config`` 를 함께 주면 두 return 모델을 영속화하고,
    decision-time 그룹 단위 rank blend 를 재현하도록 ``decision_score_config``
    (v1 reranker) 를 함께 기록합니다. 두 인자는 항상 함께 주어야 하며,
    ``recency_ensemble_config`` 의 ``half_life_groups``(252/504)와
    ``recent_weight``([0, 1]) 를 검증합니다.
    """
    if recency_ensemble_config is not None and recent_return_model is None:
        raise ValueError(
            "recency research bundle requires both recent_return_model and "
            "recency_ensemble_config"
        )
    if recency_ensemble_config is not None:
        half_life = recency_ensemble_config.get("half_life_groups")
        recent_weight = recency_ensemble_config.get("recent_weight")
        if half_life not in (252, 504):
            raise ValueError(
                f"recency_ensemble_config.half_life_groups must be 252 or 504, got {half_life!r}"
            )
        if not isinstance(recent_weight, (int, float)) or not 0.0 <= recent_weight <= 1.0:
            raise ValueError(
                f"recency_ensemble_config.recent_weight must be within [0, 1], "
                f"got {recent_weight!r}"
            )
    feature_cols = [col for col in feature_cols if col not in _CATEGORICAL_FEATURE_COLS]
    if not feature_cols:
        raise ValueError("feature_cols is empty after excluding categorical columns")

    train = df.sort_values(group_col)
    y = train[target_col].to_numpy(dtype=np.float64)

    relevance = train[target_col].groupby(train[group_col], sort=False).rank(pct=True).to_numpy()
    relevance = (relevance * 4.0).round().astype(int)
    group_counts = train[group_col].value_counts(sort=False).to_numpy(dtype=np.int64)
    ranker = LGBMRanker(objective="lambdarank", random_state=42, verbosity=-1)
    ranker.fit(train[feature_cols], relevance, group=group_counts)

    # 회귀 champion: expected-return LGBMRegressor(Huber). 당일 rank_score 는
    # 이 기대수익 예측으로 생성되어 OOF champion 과 운영 점수가 일치합니다.
    return_model = LGBMRegressor(objective="huber", random_state=42, verbosity=-1)
    return_model.fit(train[feature_cols], y)

    quantile_models: dict[str, Any] = {}
    for col, alpha in zip(_QUANTILE_COLS, _QUANTILE_ALPHAS, strict=True):
        model = LGBMRegressor(objective="quantile", alpha=alpha, random_state=42, verbosity=-1)
        model.fit(train[feature_cols], y)
        quantile_models[col] = model

    calibrators: dict[str, Any] = {
        "p_good": _fit_calibrator(
            train[feature_cols], (train[target_col] >= _GOOD_THRESHOLD).to_numpy().astype(bool)
        ),
        "p_bad": _fit_calibrator(
            train[feature_cols], (train[target_col] <= _BAD_THRESHOLD).to_numpy().astype(bool)
        ),
    }

    manifest = build_feature_manifest(list(feature_cols))
    training_cutoff = str(train[group_col].max())
    policy_params: dict[str, Any] = {
        "grade_multipliers": dict(_GRADE_MULTIPLIERS),
        "grade_percentiles": {"strong": _STRONG_PCT, "good": _GOOD_PCT, "weak": _WEAK_PCT},
        "utility_weights": {"lambda_risk": 0.5, "gamma_uncertainty": 0.1, "w_good": 0.0, "w_bad": 0.0},
        "round_trip_cost": ROUND_TRIP_COST_RATIO,
        "realized_vol_default": _DEFAULT_REALIZED_VOL,
    }

    bundle: dict[str, Any] = {
        "feature_cols": list(feature_cols),
        "target_col": target_col,
        "group_col": group_col,
        "return_unit": "decimal_net",
        "round_trip_cost": ROUND_TRIP_COST_RATIO,
        "label_thresholds": {"target_good": _GOOD_THRESHOLD, "target_bad": _BAD_THRESHOLD},
        "feature_manifest": manifest,
        "training_cutoff": training_cutoff,
        "calibration_diagnostics": list(calibration_diagnostics or []),
        "policy_params": policy_params,
        "rank_model": ranker,
        "return_model": return_model,
        "quantile_models": quantile_models,
        "calibrators": calibrators,
    }
    if recent_return_model is not None:
        if recency_ensemble_config is None:
            raise ValueError(
                "recency research bundle requires both recent_return_model and "
                "recency_ensemble_config"
            )
        bundle["recent_return_model"] = recent_return_model
        bundle["recency_ensemble_config"] = dict(recency_ensemble_config)
        bundle["decision_score_config"] = {
            "version": "close-morning-reranker-v1",
            "rank_weight": 1.0,
            "p_good_weight": float(
                recency_ensemble_config.get("probability_weight", 0.5)
            ),
            "score_col": "decision_score",
        }
    return bundle


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
    if recency_config is not None:
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
        expanding_pct = expanding_pred.groupby(out[bundle_group_col]).rank(pct=True, method="average")
        recent_pct = recent_pred.groupby(out[bundle_group_col]).rank(pct=True, method="average")
        out["rank_score"] = (1.0 - recent_weight) * expanding_pct + recent_weight * recent_pct
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


def predict_daily_position_sizing(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "target_net_return",
    group_col: str = "date",
    models_bundle: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """당일 스냅샷에 대한 고속 추론(Fast Inference) 및 동적 Sizing 을 수행합니다.

    ``models_bundle`` 이 주어지면 저장된 모델로 추론하고, 그렇지 않으면
    ``target_col`` 을 사용해 인라인 모델을 학습합니다. rank_score, 분위수 예측
    (pred_q10/pred_q50/pred_q90), 보정 확률(p_good/p_bad)을 산출한 뒤
    Utility Score -> 등급(Strong/Good/Weak/Pass) -> 배분(allocation)을 계산합니다.

    번들이 ``decision_score_config.version=close-morning-reranker-v1`` 를 선언하면
    rank_score/p_good 예측 직후 ``decision_score`` 를 추가해 일별 선택이 결합
    스코어를 사용하도록 합니다. ``close-morning-reranker-v2-research`` 번들은
    추가로 ``bad_probability_weight`` 만큼 ``p_bad`` 백분위를 차감합니다. 레거시
    번들은 기존 출력과 선택 의미를 유지합니다.

    단일 날짜 또는 복수 날짜 데이터프레임을 모두 지원하며, Pass 등급은
    0.0 배분 비중을 가집니다.
    """
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"missing feature columns in df: {missing_features}")
    if group_col not in df.columns:
        raise ValueError(f"group_col {group_col!r} is missing in df")

    if models_bundle is None:
        if target_col not in df.columns:
            raise ValueError(
                f"target_col {target_col!r} is missing in df; required to train inline models"
            )
        models_bundle = _train_inline_bundle(df, feature_cols, target_col, group_col)

    out = _predict_from_bundle(df, feature_cols, models_bundle)
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


# 계약 python_assertion(predict_daily_position_sizing(sample_df, ['f1'])) 검증용
# 소규모 당일 스냅샷 픽스처 (인라인 추론 스모크 테스트 입력).
_sample_rng = np.random.default_rng(7)
sample_df = pd.DataFrame(
    {
        "date": ["2026-08-03"] * 15 + ["2026-08-04"] * 15 + ["2026-08-05"] * 15,
        "f1": _sample_rng.normal(size=45),
        "target_net_return": _sample_rng.normal(loc=0.02, scale=0.03, size=45),
    }
)
