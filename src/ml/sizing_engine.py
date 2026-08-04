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

_PREDICTION_COLS = ("pred_q10", "pred_q50", "pred_q90", "p_good", "p_bad")

# 왕복 거래 비용 (매수/매도 수수료 0.028% + 매도 거래세 0.15% + 슬리피지 0.022% ≈ 0.20%).
ROUND_TRIP_COST_RATIO: float = 0.0020

_BUNDLE_FILENAME = "sizing_pipeline_bundle.joblib"
_QUANTILE_COLS = ("pred_q10", "pred_q50", "pred_q90")
_QUANTILE_ALPHAS = (0.10, 0.50, 0.90)
_GOOD_THRESHOLD = 0.01
_BAD_THRESHOLD = -0.015

_STRONG_PCT = 0.90
_GOOD_PCT = 0.75
_WEAK_PCT = 0.50

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


def calculate_utility_score(
    df: pd.DataFrame,
    lambda_risk: float = 0.5,
    gamma_uncertainty: float = 0.1,
    w_good: float = 0.01,
    w_bad: float = 0.01,
    round_trip_cost: float = ROUND_TRIP_COST_RATIO,
) -> pd.Series:
    """하방위험/불확실성 패널티가 적용된 종합 Utility Score 를 산출합니다.

    U_i = (q50_i - cost) - lambda·max(0, -(q10_i - cost)) - gamma·(q90_i - q10_i)
          + w_good·p_good_i - w_bad·p_bad_i

    ``round_trip_cost``(기본 0.20%)를 기대수익 q50 과 하방위험 기준 q10 에서
    공제해 순유틸리티(net-of-cost) 기준으로 산출하며, 확률 가중치
    (``w_good``/``w_bad``)는 수익률 스케일과 정렬된 소량 가산치입니다.
    """
    missing = [col for col in _PREDICTION_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required prediction columns in df: {missing}")
    q10 = df["pred_q10"].to_numpy(dtype=np.float64)
    q50 = df["pred_q50"].to_numpy(dtype=np.float64)
    q90 = df["pred_q90"].to_numpy(dtype=np.float64)
    p_good = df["p_good"].to_numpy(dtype=np.float64)
    p_bad = df["p_bad"].to_numpy(dtype=np.float64)
    net_q50 = q50 - round_trip_cost
    net_q10 = q10 - round_trip_cost
    downside = lambda_risk * np.maximum(0.0, -net_q10)
    uncertainty = gamma_uncertainty * (q90 - q10)
    utility = net_q50 - downside - uncertainty + w_good * p_good - w_bad * p_bad
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
    해당 등급을 부여합니다 (Relative Evaluation Trap 방지). ``net_q50 =
    pred_q50 - round_trip_cost`` 로 거래 비용 차감 후에도 순 기대수익이 양수인
    종목만 거래 후보로 승인합니다.

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
            net_positive = (
                out.loc[idx, "pred_q50"].to_numpy(dtype=np.float64) - round_trip_cost > 0.0
            )
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
    절대적 크기를 비중에 반영하며, ``sigma_i`` 는 모델 불확실성 프록시인
    분위수 스프레드(q90 - q10)로 추정합니다. 그룹별 합계를 ``max_total_allocation``
    으로, 개별 비중을 ``max_position_pct`` 로 클리핑하고, 비용 차감 후 음수
    순유틸리티 종목은 비중 0% 로 강제합니다.

    시장 국면 방어: 그룹 후보 유니버스의 평균 Utility Score 가 음수이면
    ``max_total_allocation`` 을 ``max(1 + avg_utility, 0)`` 비율로 축소하여
    불리한 시장 국면에서 자본을 보호합니다. ``utility_col`` 이 없으면
    방어/크기 가중 없이 기존 한도만 적용합니다.
    """
    if "grade_multiplier" not in df.columns:
        raise ValueError("grade_multiplier is missing in df; run assign_sizing_grades first")
    if "pred_q10" not in df.columns or "pred_q90" not in df.columns:
        raise ValueError("missing required prediction columns in df: pred_q10/pred_q90")

    out = df.copy()
    sigma = np.maximum((out["pred_q90"] - out["pred_q10"]).to_numpy(dtype=np.float64), 1e-8)
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
            avg_utility = float(group_utility[pos].mean())
            if avg_utility < 0.0:
                effective_max_total = max_total_allocation * max(1.0 + avg_utility, 0.0)
        scale = min(1.0, effective_max_total / total)
        scaled = np.minimum(group_raw * scale, max_position_pct)
        allocation[pos] = scaled
    if has_utility:
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
) -> dict[str, Any]:
    """타깃이 포함된 데이터로 즉시 학습 가능한 기본 모델 번들을 구성합니다.

    LGBMRanker(랭킹) + Quantile Regressor(q10/q50/q90) + Calibrated Classifier
    (p_good/p_bad) 5종 모델을 포함하며, ``predict_daily_position_sizing`` 의
    인라인 추론(``models_bundle=None``)과 학습 모드 저장에 사용됩니다.

    문자열/범주형 컬럼(``market_type``, ``theme_sector``, ``chart_analysis``)은
    Booster 구성 시 ValueError 를 유발하므로 ``feature_cols`` 에서 제외됩니다.
    """
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

    return {
        "feature_cols": list(feature_cols),
        "target_col": target_col,
        "group_col": group_col,
        "rank_model": ranker,
        "quantile_models": quantile_models,
        "calibrators": calibrators,
    }


def _predict_from_bundle(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_bundle: dict[str, Any],
) -> pd.DataFrame:
    """저장된 번들 모델로 rank/분위수/확률 예측 컬럼을 산출합니다."""
    features = df[feature_cols]
    out = df.copy()

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
