"""Purged Walk-Forward CV 기반 ML 모델 학습 및 OOF 백테스트 평가 파이프라인.

Baseline(Ridge) / Primary(LGBMRanker, LGBMRegressor) 모델을 Purged Group
Walk-Forward CV 로 학습하고 OOF 예측과 NDCG@1/NDCG@3, Rank IC, Top-k Net
Return 지표를 산출합니다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from src.ml.backtest_evaluator import run_backtest_evaluation
from src.ml.feature_manifest import build_feature_manifest
from src.ml.purged_cv import PurgedGroupTimeSeriesSplit
from src.ml.quantile_model import fit_predict_quantile_and_classifier
from src.ml.sizing_engine import (
    ROUND_TRIP_COST_RATIO,
    _train_inline_bundle,
    apply_risk_limits,
    assign_sizing_grades,
    calculate_utility_score,
    save_model_artifacts,
)
from src.processing.preprocessor import (
    DECISION_TIMESTAMP_COL,
    FEATURE_AVAILABLE_TIMESTAMP_COL,
    LABEL_THRESHOLDS,
    RETURN_UNIT,
)

logger = logging.getLogger(__name__)

_MODEL_TYPES = ("ridge", "lgb_ranker", "lgb_regressor")


def _group_relevance(target: pd.Series, groups: pd.Series) -> pd.Series:
    """날짜별 Cross-sectional Rank Percentile 을 0~4 등급 relevance 로 동적 변환.

    일자별 후보 수가 상이해도 횡단면 순위 기반이라 바운더리가 왜곡되지 않습니다.
    """
    pct = target.groupby(groups).rank(pct=True, method="average")
    return (pct * 4.0).round().astype(int)


def _ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """단일 그룹의 예측 순서 기준 NDCG@k 를 계산합니다."""
    k = min(k, relevance.size)
    if k <= 0:
        return 0.0
    gains = np.power(2.0, relevance[:k].astype(np.float64)) - 1.0
    discounts = np.log2(np.arange(1, k + 1) + 1.0)
    ideal = np.sort(relevance.astype(np.float64))[::-1][:k]
    idcg = float(np.sum((np.power(2.0, ideal) - 1.0) / discounts))
    if idcg <= 0.0:
        return 0.0
    dcg = float(np.sum(gains / discounts))
    return dcg / idcg


def _group_metric(
    oof: pd.DataFrame,
    group_col: str,
    metric_fn: Callable[[pd.DataFrame], float],
) -> float:
    """그룹별로 metric_fn 을 적용한 값의 평균을 반환합니다."""
    values: list[float] = []
    for _, group in oof.groupby(group_col, sort=False):
        values.append(metric_fn(group))
    return float(np.mean(values))


def _compute_ndcg(oof: pd.DataFrame, group_col: str, k: int) -> float:
    def per_group(group: pd.DataFrame) -> float:
        order = group["pred"].to_numpy().argsort()[::-1]
        return _ndcg_at_k(group["relevance"].to_numpy()[order], k)

    return _group_metric(oof, group_col, per_group)


def _compute_rank_ic(oof: pd.DataFrame, group_col: str, target_col: str) -> float:
    ics: list[float] = []
    for _, group in oof.groupby(group_col, sort=False):
        if len(group) < 2:
            continue
        if float(np.std(group["pred"].to_numpy())) == 0.0:
            continue
        result = spearmanr(group["pred"], group[target_col])
        ic = float(result.statistic)
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def _compute_top_k_return(oof: pd.DataFrame, group_col: str, target_col: str, k: int) -> float:
    def per_group(group: pd.DataFrame) -> float:
        order = group["pred"].to_numpy().argsort()[::-1][:k]
        return float(group[target_col].to_numpy()[order].mean())

    return _group_metric(oof, group_col, per_group)


def _fit_predict(
    model_type: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
) -> Any:
    """model_type 에 따라 fold 단위 모델을 학습하고 OOF 예측을 반환합니다."""
    if model_type == "ridge":
        model = Ridge()
        model.fit(train[feature_cols], train[target_col])
        return model, model.predict(val[feature_cols])

    if model_type == "lgb_regressor":
        model = LGBMRegressor(objective="huber", random_state=42)
        model.fit(train[feature_cols], train[target_col])
        return model, model.predict(val[feature_cols])

    # lgb_ranker: 동일 query(date) 샘플이 연속되도록 정렬 후 group counts 전달
    train_sorted = train.sort_values(group_col)
    relevance = _group_relevance(train_sorted[target_col], train_sorted[group_col]).to_numpy()
    group_counts = train_sorted[group_col].value_counts(sort=False).to_numpy(dtype=np.int64)
    model = LGBMRanker(objective="lambdarank", random_state=42)
    model.fit(train_sorted[feature_cols], relevance, group=group_counts)
    return model, model.predict(val[feature_cols])


def _validate_feature_availability(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """선정 피처가 decision 시각보다 늦지 않게 이용 가능한지 검증합니다.

    Raises:
        ValueError: 타임스탬프 컬럼 누락, timezone-naive, 또는 임의의 행에서
            ``feature_available_timestamp > decision_timestamp`` 인 경우
            (위반 피처 컬럼명을 함께 보고).
    """
    for col in (FEATURE_AVAILABLE_TIMESTAMP_COL, DECISION_TIMESTAMP_COL):
        if col not in df.columns:
            raise ValueError(
                f"timestamp column {col!r} is required for point-in-time integrity; "
                f"record {col} in candidate history before training"
            )
        if getattr(df[col].dtype, "tz", None) is None:
            raise ValueError(
                f"timestamp column {col!r} must be timezone-aware (Asia/Seoul), got {df[col].dtype}"
            )
    available = df[FEATURE_AVAILABLE_TIMESTAMP_COL]
    decision = df[DECISION_TIMESTAMP_COL]
    violation = available > decision
    if violation.any():
        offending = [feature for feature in feature_cols if df.loc[violation, feature].notna().any()]
        raise ValueError(
            "features become available after decision_timestamp; non-causal rows "
            f"({int(violation.sum())} rows) violate feature_available_timestamp <= "
            f"decision_timestamp. Offending feature columns: {offending}"
        )


def _fit_predict_linear_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> np.ndarray:
    """Regularized linear(Ridge) baseline 을 동일 fold 에서 학습·예측합니다."""
    medians = train[feature_cols].median(numeric_only=True).reindex(feature_cols).fillna(0.0)
    train_features = train[feature_cols].fillna(medians)
    val_features = val[feature_cols].fillna(medians)
    model = Ridge()
    model.fit(train_features, train[target_col])
    return np.asarray(model.predict(val_features), dtype=np.float64)


def run_model_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    purge_gap: int = 1,
    model_type: str = "lgb_ranker",
) -> dict[str, Any]:
    """Train ML model using Purged Group Walk-Forward CV and evaluate OOF results.

    Point-in-time 요구사항(P0):
    - ``feature_available_timestamp``/``decision_timestamp`` 가 timezone-aware 로
      존재해야 하고 모든 행에서 ``feature_available <= decision`` 을 만족해야 합니다.
    - OOF 프레임에 ``selection_rank`` 와 regularized linear baseline(``pred_linear``),
      fold 출처를 보존합니다.

    Returns:
        dict containing 'oof_predictions', 'oof_df', 'metrics', 'trained_models',
        'backtest_eval', and bundle metadata (return_unit, round_trip_cost,
        label_thresholds, feature_manifest, training_cutoff, policy_params).
    """
    if model_type not in _MODEL_TYPES:
        raise ValueError(f"model_type must be one of {list(_MODEL_TYPES)}, got {model_type!r}")
    missing_cols = [col for col in [*feature_cols, target_col, group_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns in df: {missing_cols}")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")

    _validate_feature_availability(df, feature_cols)

    work = df.sort_values(group_col).copy()
    splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)

    oof_parts: list[pd.DataFrame] = []
    trained_models: list[Any] = []
    training_cutoff: Any = None

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(work, y=work[target_col], groups=work[group_col])
    ):
        train = work.iloc[train_idx]
        val = work.iloc[val_idx]
        model, pred = _fit_predict(model_type, train, val, feature_cols, target_col, group_col)
        trained_models.append(model)
        training_cutoff = train[group_col].max()

        fold_oof = pd.DataFrame(
            {
                group_col: val[group_col].to_numpy(),
                target_col: val[target_col].to_numpy(),
                "pred": pred,
                "pred_linear": _fit_predict_linear_baseline(train, val, feature_cols, target_col),
                "fold": fold,
            },
            index=val.index,
        )
        if "selection_rank" in work.columns:
            fold_oof["selection_rank"] = val["selection_rank"].to_numpy()
        oof_parts.append(fold_oof)
        logger.info("fold=%d train=%d val=%d", fold, len(train_idx), len(val_idx))

    oof_df = pd.concat(oof_parts, axis=0).sort_values(group_col)
    oof_df["relevance"] = _group_relevance(oof_df[target_col], oof_df[group_col]).to_numpy()

    metrics = {
        "ndcg_1": _compute_ndcg(oof_df, group_col, k=1),
        "ndcg_3": _compute_ndcg(oof_df, group_col, k=3),
        "rank_ic": _compute_rank_ic(oof_df, group_col, target_col),
        "top_1_return": _compute_top_k_return(oof_df, group_col, target_col, k=1),
        "top_3_return": _compute_top_k_return(oof_df, group_col, target_col, k=3),
    }

    return {
        "oof_predictions": oof_df,
        "oof_df": oof_df,
        "metrics": metrics,
        "trained_models": trained_models,
        "backtest_eval": run_backtest_evaluation(oof_df, target_col, group_col),
        "return_unit": RETURN_UNIT,
        "round_trip_cost": ROUND_TRIP_COST_RATIO,
        "label_thresholds": dict(LABEL_THRESHOLDS),
        "feature_manifest": build_feature_manifest(feature_cols),
        "training_cutoff": str(training_cutoff),
        "calibration_diagnostics": [],
        "policy_params": {
            "purge_gap": purge_gap,
            "n_splits": n_splits,
            "model_type": model_type,
        },
    }


def run_sizing_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    purge_gap: int = 1,
    base_budget: float = 1.0,
    target_vol: float = 0.15,
    max_position_pct: float = 0.25,
    max_total_allocation: float = 1.0,
    export_dir: str | None = None,
) -> dict[str, Any]:
    """Quantile 위험 예측 + Utility Dynamic Sizing 통합 파이프라인.

    Quantile Regressor(q10/q50/q90)와 Calibrated Classifier(p_good/p_bad)로
    OOF 위험 프로필을 예측하고, Utility Score -> 등급(Strong/Good/Weak/Pass) ->
    변동성 역가중 비중 및 위험 한도를 적용한 최종 배분을 산출합니다.

    ``export_dir`` 이 주어지면 훈련 모드로 동작해 최종 모델 번들을
    ``save_model_artifacts`` 로 저장하고 결과에 ``artifact_path`` 를 추가합니다.

    Returns:
        dict containing 'quantile_df' (OOF 분위수/확률 예측) and
        'sizing_df' (utility_score, grade, grade_multiplier, allocation 포함).
    """
    quantile_df = fit_predict_quantile_and_classifier(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
    )
    sizing_df = quantile_df.copy()
    sizing_df["utility_score"] = calculate_utility_score(sizing_df)
    sizing_df = assign_sizing_grades(sizing_df, group_col=group_col)
    sizing_df = apply_risk_limits(
        sizing_df,
        base_budget=base_budget,
        target_vol=target_vol,
        max_position_pct=max_position_pct,
        max_total_allocation=max_total_allocation,
        group_col=group_col,
    )
    result: dict[str, Any] = {"quantile_df": quantile_df, "sizing_df": sizing_df}
    if export_dir:
        bundle = _train_inline_bundle(
            df,
            feature_cols,
            target_col,
            group_col,
            calibration_diagnostics=quantile_df.attrs.get("calibration_diagnostics", []),
        )
        result["artifact_path"] = save_model_artifacts(bundle, export_dir)
    return result
