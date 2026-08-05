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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.backtest_evaluator import _max_drawdown, run_backtest_evaluation
from src.ml.feature_manifest import build_feature_manifest
from src.ml.purged_cv import PurgedGroupTimeSeriesSplit
from src.ml.quantile_model import fit_predict_quantile_and_classifier
from src.ml.single_stock_policy import (
    _DEFAULT_MIN_HISTORY_DATES,
    SingleStockPolicy,
    SingleStockPolicyEvaluation,
    default_policy_candidates,
    evaluate_single_stock_policy_oof,
)
from src.ml.sizing_engine import (
    ROUND_TRIP_COST_RATIO,
    _train_inline_bundle,
    add_close_morning_decision_score,
    apply_risk_limits,
    assign_sizing_grades,
    calculate_utility_score,
    save_model_artifacts,
)
from src.processing.preprocessor import (
    LABEL_THRESHOLDS,
    RETURN_UNIT,
    build_ml_dataset,
)

logger = logging.getLogger(__name__)

_MODEL_TYPES = ("ridge", "lgb_ranker", "lgb_regressor")

# 번들에 영속화할 compact OOF 정책 지표 (close-to-morning 전략 지표).
_POLICY_METRIC_KEYS: tuple[str, ...] = (
    "n_scheduled_dates",
    "n_buy",
    "n_abstain",
    "buy_rate",
    "scheduled_mean_return",
    "scheduled_win_rate",
    "profit_factor",
    "scheduled_sharpe",
    "active_trade_mean_return",
    "active_trade_win_rate",
    "entry_sequence_drawdown",
)


def _policy_metadata(
    policy: SingleStockPolicy | None,
    evaluation: SingleStockPolicyEvaluation | None,
    *,
    oof_score_col: str = "pred",
    daily_score_col: str = "rank_score",
) -> dict[str, Any] | None:
    """OOF 정책 보정 결과를 번들용 메타데이터로 요약합니다.

    기본 매핑은 OOF ``pred`` → 일일 ``rank_score`` 이며, close-morning
    reranker 는 ``decision_score`` → ``decision_score`` 를 명시적으로 기록합니다.
    """
    if policy is None or evaluation is None:
        return None
    return {
        "oof_score_col": oof_score_col,
        "daily_score_col": daily_score_col,
        "calibration_cutoff": str(policy.calibration_cutoff),
        "policy_version": policy.version,
        "policy_id": policy.policy_id,
        "candidate": policy.candidate,
        "policy_metrics": {
            key: evaluation.metrics[key]
            for key in _POLICY_METRIC_KEYS
            if key in evaluation.metrics
        },
    }


def _align_close_morning_oof(
    return_oof: pd.DataFrame,
    risk_oof: pd.DataFrame,
    target_col: str,
    group_col: str,
) -> pd.DataFrame:
    """return OOF 와 quantile/classifier OOF 를 원본 행 인덱스로 정렬합니다.

    날짜 단독 병합은 금지합니다. return 예측 행이 risk OOF 에 없거나 p_good
    예측이 누락되거나 group/target/stock/scenario 식별 키가 원본 인덱스에서
    어긋나면 ``ValueError`` 로 fail-closed 합니다 (누락 OOF 예측은 대체하지 않음).
    """
    missing = return_oof.index.difference(risk_oof.index)
    if len(missing):
        raise ValueError(
            "close-morning OOF alignment: return-prediction rows are missing from "
            f"quantile/classifier OOF ({len(missing)} rows); missing OOF predictions "
            "are never substituted"
        )
    aligned = return_oof.join(risk_oof[["p_good"]], how="left")
    if aligned["p_good"].isna().any():
        raise ValueError(
            "close-morning OOF alignment: p_good predictions are missing for aligned "
            "return rows"
        )
    for col in (group_col, target_col):
        if col not in risk_oof.columns:
            continue
        if not aligned[col].equals(risk_oof.loc[aligned.index, col]):
            raise ValueError(
                f"close-morning OOF alignment: {col} mismatch between return and risk "
                "OOF on original index"
            )
    for col in ("stock_code", "chart_analysis"):
        if col not in risk_oof.columns:
            continue
        if not aligned[col].equals(risk_oof.loc[aligned.index, col]):
            raise ValueError(
                f"close-morning OOF alignment: {col} mismatch between return and risk "
                "OOF on original index"
            )
    return aligned


def _calibrate_close_morning_decision_oof(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int,
    purge_gap: int,
) -> tuple[
    SingleStockPolicy | None,
    SingleStockPolicyEvaluation | None,
    dict[str, Any] | None,
]:
    """close-morning reranker OOF 정책 보정 (rank_score + p_good 순위 결합).

    1. 회귀 champion(Huber) OOF ``pred`` 를 기존 purged walk-forward 파이프라인으로 산출하고,
    2. 동일 피처/타깃/그룹/폴드/퍼지 갭으로 시계열 ``p_good`` OOF 를 산출한 뒤
    3. 두 OOF 를 원본 행 인덱스로 정렬해 ``rank_score`` / ``decision_score`` 를 구성하고
    4. ``evaluate_single_stock_policy_oof(..., score_col="decision_score")`` 로 보정합니다.

    정책/메타데이터에는 ``oof_score_col=daily_score_col="decision_score"`` 를
    기록합니다. 식별 컬럼이 없으면 ``(None, None, None)`` 을 반환합니다.
    """
    if not {"stock_code", "chart_analysis"} <= set(df.columns):
        return None, None, None
    result = run_model_pipeline(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
        model_type="lgb_regressor",
    )
    risk_oof = fit_predict_quantile_and_classifier(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
    )
    aligned = _align_close_morning_oof(
        result["oof_predictions"], risk_oof, target_col=target_col, group_col=group_col
    )
    aligned["rank_score"] = aligned["pred"]
    aligned = add_close_morning_decision_score(aligned, group_col=group_col)
    cutoff = str(aligned[group_col].max())
    evaluation = evaluate_single_stock_policy_oof(
        aligned,
        target_col=target_col,
        group_col=group_col,
        stock_col="stock_code",
        policy_candidates=default_policy_candidates(cutoff, score_col="decision_score"),
        min_history_dates=_DEFAULT_MIN_HISTORY_DATES,
        scenario_col="chart_analysis",
        score_col="decision_score",
    )
    policy = evaluation.selected_policy
    metadata = _policy_metadata(
        policy, evaluation, oof_score_col="decision_score", daily_score_col="decision_score"
    )
    return policy, evaluation, metadata


def _calibrate_oof_policy(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int,
    purge_gap: int,
    *,
    reranker: bool = False,
) -> tuple[SingleStockPolicy | None, dict[str, Any] | None]:
    """시나리오 행동 패널에서 purged OOF 정책을 보정하고 번들용 메타데이터를 반환합니다.

    ``reranker=True`` 이면 close-morning 결정 스코어(rank_score + p_good 순위)
    OOF 보정을 사용하고, 그 외에는 기존 ``pred``/``rank_score`` 매핑을
    유지합니다. ``stock_code`` / ``chart_analysis`` 식별 컬럼이 없으면 정책을
    산출할 수 없으므로 ``(None, None)`` 을 반환합니다 — 호출부는 이를 명시적
    ``ABSTAIN(missing_validated_policy)`` 로 이어갑니다 (Top-N 폴백 금지).
    """
    if not {"stock_code", "chart_analysis"} <= set(df.columns):
        return None, None
    if reranker:
        policy, _evaluation, metadata = _calibrate_close_morning_decision_oof(
            df,
            feature_cols=feature_cols,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            purge_gap=purge_gap,
        )
        return policy, metadata
    result = run_model_pipeline(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
        model_type="lgb_regressor",
    )
    return result["single_stock_policy"], result["policy_metadata"]


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
    model_params: dict[str, int | float] | None = None,
) -> Any:
    """model_type 에 따라 fold 단위 모델을 학습하고 OOF 예측을 반환합니다.

    ``model_params`` 는 요청된 모델에만 전달되며 ``random_state=42`` 는 유지합니다.
    """
    params: dict[str, Any] = dict(model_params or {})
    if model_type == "ridge":
        medians = train[feature_cols].median(numeric_only=True).reindex(feature_cols).fillna(0.0)
        train_features = train[feature_cols].fillna(medians)
        val_features = val[feature_cols].fillna(medians)
        model = make_pipeline(StandardScaler(), Ridge(**params))
        model.fit(train_features, train[target_col])
        return model, model.predict(val_features)

    if model_type == "lgb_regressor":
        model = LGBMRegressor(objective="huber", random_state=42, **params)
        model.fit(train[feature_cols], train[target_col])
        return model, model.predict(val[feature_cols])

    # lgb_ranker: 동일 query(date) 샘플이 연속되도록 정렬 후 group counts 전달
    train_sorted = train.sort_values(group_col)
    relevance = _group_relevance(train_sorted[target_col], train_sorted[group_col]).to_numpy()
    group_counts = train_sorted[group_col].value_counts(sort=False).to_numpy(dtype=np.int64)
    model = LGBMRanker(objective="lambdarank", random_state=42, **params)
    model.fit(train_sorted[feature_cols], relevance, group=group_counts)
    return model, model.predict(val[feature_cols])


def _fit_predict_linear_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> np.ndarray:
    """StandardScaler + Ridge linear baseline 을 동일 fold 에서 학습·예측합니다."""
    medians = train[feature_cols].median(numeric_only=True).reindex(feature_cols).fillna(0.0)
    train_features = train[feature_cols].fillna(medians)
    val_features = val[feature_cols].fillna(medians)
    model = make_pipeline(StandardScaler(), Ridge())
    model.fit(train_features, train[target_col])
    return np.asarray(model.predict(val_features), dtype=np.float64)


def run_model_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    purge_gap: int = 1,
    model_type: str = "lgb_regressor",
    model_params: dict[str, int | float] | None = None,
) -> dict[str, Any]:
    """Train ML model using Purged Group Walk-Forward CV and evaluate OOF results.

    시간 컬럼은 필수 검증하지 않습니다. 고정된 업무 원천 규칙이며 모델 입력·CV
    분할·artifact 승인 조건이 아닙니다.

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
    if purge_gap < 0:
        raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")

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
        model, pred = _fit_predict(
            model_type, train, val, feature_cols, target_col, group_col, model_params
        )
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
        for col in (
            "stock_code",
            "market_type",
            "market_cap_100m",
            "chart_analysis",
            "scenario_count_for_stock_date",
            "has_sangtta_for_stock_date",
            "is_multi_scenario_stock_date",
        ):
            if col in work.columns:
                fold_oof[col] = val[col].to_numpy()
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

    # 단일 종목 정책 OOF 보정: ranker OOF 생성 직후, 후보 번들 승격 이전에
    # 저장된 rank_score/OOF pred 만으로 정책 상태를 인과적으로 확정합니다.
    single_stock_policy: SingleStockPolicy | None = None
    single_stock_evaluation: SingleStockPolicyEvaluation | None = None
    if {"stock_code", "chart_analysis"} <= set(oof_df.columns):
        cutoff = str(oof_df[group_col].max())
        single_stock_evaluation = evaluate_single_stock_policy_oof(
            oof_df,
            target_col=target_col,
            group_col=group_col,
            stock_col="stock_code",
            policy_candidates=default_policy_candidates(cutoff, score_col="pred"),
            min_history_dates=_DEFAULT_MIN_HISTORY_DATES,
            scenario_col="chart_analysis",
            score_col="pred",
        )
        single_stock_policy = single_stock_evaluation.selected_policy

    policy_metadata = _policy_metadata(single_stock_policy, single_stock_evaluation)

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
        "single_stock_policy": single_stock_policy,
        "single_stock_evaluation": single_stock_evaluation,
        "policy_metadata": policy_metadata,
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
        policy, policy_metadata = _calibrate_oof_policy(
            df, feature_cols, target_col, group_col, n_splits, purge_gap
        )
        bundle["single_stock_policy"] = policy.model_dump() if policy is not None else None
        bundle["policy_metadata"] = policy_metadata
        bundle["oof_score_col"] = "pred"
        bundle["daily_score_col"] = "rank_score"
        result["artifact_path"] = save_model_artifacts(bundle, export_dir)
    return result


# close-to-morning 품질 보고에서 항상 유한해야 하는 핵심 지표.
_FINITE_QUALITY_METRICS: tuple[str, ...] = (
    "top1_net_mean",
    "win_rate",
    "sharpe",
    "active_trade_win_rate",
    "close_to_morning_mdd",
)

# reranker 후보 정책(decision_score)이 적용되는 champion 피처셋.
_RERANKER_FEATURE_SET = "close_morning61"


def _policy_entry(
    evaluation: SingleStockPolicyEvaluation | None,
    n_oof_dates: int,
    *,
    prefix: str,
) -> dict[str, Any]:
    """OOF 정책 평가를 품질 보고 엔트리 지표 dict 로 변환합니다.

    ``prefix`` 로 지표 이름을 명확히 구분해 레거시 rank-only 와 reranker
    decision-score 정책을 같은 피처셋 엔트리에 함께 노출할 수 있습니다.
    정책이 없으면 관망(ABSTAIN) 지표를 NaN/0 으로 기록합니다.
    """
    if evaluation is None:
        return {
            f"{prefix}top1_net_mean": float("nan"),
            f"{prefix}scheduled_mean_return": float("nan"),
            f"{prefix}active_trade_mean_return": float("nan"),
            f"{prefix}active_trade_win_rate": float("nan"),
            f"{prefix}win_rate": float("nan"),
            f"{prefix}profit_factor": float("nan"),
            f"{prefix}sharpe": float("nan"),
            f"{prefix}close_to_morning_mdd": float("nan"),
            f"{prefix}n_buy": 0,
            f"{prefix}n_abstain": int(n_oof_dates),
            f"{prefix}reason_counts": {"missing_validated_policy": int(n_oof_dates)},
            f"{prefix}policy_candidate": None,
        }
    em = evaluation.metrics
    return {
        f"{prefix}top1_net_mean": float(em["scheduled_mean_return"]),
        f"{prefix}scheduled_mean_return": float(em["scheduled_mean_return"]),
        f"{prefix}active_trade_mean_return": float(em["active_trade_mean_return"]),
        f"{prefix}active_trade_win_rate": float(em["active_trade_win_rate"]),
        f"{prefix}win_rate": float(em["scheduled_win_rate"]),
        f"{prefix}profit_factor": float(em["profit_factor"]),
        f"{prefix}sharpe": float(em["scheduled_sharpe"]),
        f"{prefix}close_to_morning_mdd": _max_drawdown(evaluation.scheduled_returns),
        f"{prefix}n_buy": int(em["n_buy"]),
        f"{prefix}n_abstain": int(em["n_abstain"]),
        f"{prefix}reason_counts": dict(em["reason_counts"]),
        f"{prefix}policy_candidate": (
            evaluation.selected_policy.candidate
            if evaluation.selected_policy is not None
            else None
        ),
    }


def _validate_quality_metrics(feature_set: str, metrics: dict[str, Any]) -> None:
    """보고 지표의 비유한 값은 fail-closed 로 거부합니다."""
    for key in _FINITE_QUALITY_METRICS:
        value = metrics[key]
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(
                f"non-finite close-to-morning metric {key}={value} for feature_set={feature_set}"
            )


def _quality_score(metrics: dict[str, Any]) -> dict[str, Any]:
    """투명한 100점 품질 스코어 (구성 요소 공식 문서화).

    - selection_edge: 30 * NDCG@1 / 0.55 (검증 champion 0.5032 → 약 27)
    - net_economics:  20 * max(top1_net_mean, 0) / 0.015 (champion 1.54% → 약 20)
    - risk_and_stability: 20 * (1 - close_to_morning_mdd) (champion 29.34% → 약 14)
    - validation_independence: 8 (purged walk-forward OOF 확보, 피처 선택이
      전체 이력 재사용 → post-freeze 기간 필요)
    - daily_deployment_integrity: 정책 영속 시 4, 정책 부재 시 0
    """
    selection = min(30.0, 30.0 * metrics["oof_ndcg_1"] / 0.55)
    net = min(20.0, 20.0 * max(metrics["top1_net_mean"], 0.0) / 0.015)
    risk = 20.0 * max(0.0, 1.0 - metrics["close_to_morning_mdd"])
    components = {
        "selection_edge": round(selection, 1),
        "net_economics": round(net, 1),
        "risk_and_stability": round(risk, 1),
        "validation_independence": 8.0,
        "daily_deployment_integrity": 4.0 if metrics.get("policy_candidate") else 0.0,
    }
    return {
        "components": components,
        "total": round(sum(components.values())),
    }


def evaluate_close_morning_quality(
    trade_log_df: pd.DataFrame,
    theme_df: pd.DataFrame | None = None,
    feature_sets: tuple[str, ...] = ("base40", "snapshot49", "close_morning61"),
    panel_mode: str = "scenario_action",
    n_splits: int = 5,
    purge_gap: int = 1,
) -> dict[str, Any]:
    """feature_set 별 close-to-morning 품질을 동일 OOF 날짜에서 비교 보고합니다.

    모든 피처셋은 같은 원천(``trade_log_df``)과 같은 purged walk-forward 분할을
    사용하므로 OOF 날짜 집합이 동일합니다. 보고 지표는 decimal-net 단위의
    close-to-morning 수익/승률/PF/Sharpe/복리 MDD, BUY/ABSTAIN 사유 분포,
    active-trade·scheduled-date 수익, 그리고 100점 투명 스코어입니다. MDD 는
    close-to-morning 전략 지표로 명명합니다 (entry-sequence 프록시 아님).

    비유한(정의되지 않은) 핵심 지표는 ``ValueError`` 로 거부합니다.
    """
    report: dict[str, dict[str, Any]] = {}
    for feature_set in feature_sets:
        panel_x, _targets, cat_features, processed = build_ml_dataset(
            trade_log_df, theme_df, feature_set=feature_set, panel_mode=panel_mode
        )
        feature_cols = [col for col in panel_x.columns if col not in cat_features]
        target_col, group_col = "target_return", "trade_date"
        result = run_model_pipeline(
            processed,
            feature_cols=feature_cols,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            purge_gap=purge_gap,
            model_type="lgb_regressor",
        )
        metrics = result["metrics"]
        evaluation = result["single_stock_evaluation"]
        n_oof_dates = int(result["oof_df"][group_col].nunique())
        entry: dict[str, Any] = {
            "feature_set": feature_set,
            "n_features": len(feature_cols),
            "n_oof_dates": n_oof_dates,
            "oof_ndcg_1": float(metrics["ndcg_1"]),
            "oof_rank_ic": float(metrics["rank_ic"]),
        }
        # 레거시 rank-only 정책 지표는 항상 전방호환 이름으로 노출합니다.
        entry.update(_policy_entry(evaluation, n_oof_dates, prefix="legacy_"))
        # champion 피처셋의 후보(candidate) 지표는 decision-score reranker 정책으로
        # 대체하며, 정렬 검증을 통과한 경우에만 사용합니다(실패 시 fail-closed NaN).
        if feature_set == _RERANKER_FEATURE_SET:
            _policy, decision_evaluation, _metadata = _calibrate_close_morning_decision_oof(
                processed,
                feature_cols=feature_cols,
                target_col=target_col,
                group_col=group_col,
                n_splits=n_splits,
                purge_gap=purge_gap,
            )
            entry.update(_policy_entry(decision_evaluation, n_oof_dates, prefix=""))
        else:
            entry.update(_policy_entry(evaluation, n_oof_dates, prefix=""))
        report[feature_set] = entry
        _validate_quality_metrics(feature_set, entry)

    return {
        "feature_sets": list(feature_sets),
        "panel_mode": panel_mode,
        "report": report,
        "quality_score": {
            feature_set: _quality_score(report[feature_set]) for feature_set in feature_sets
        },
    }
