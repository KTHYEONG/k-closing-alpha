"""run_model_pipeline / run_sizing_pipeline 및 품질 보고 조합 (composition)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.ml.backtest_evaluator import (
    _max_drawdown,
    run_backtest_evaluation,
)
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
    apply_risk_limits,
    assign_sizing_grades,
    calculate_utility_score,
    save_model_artifacts,
)
from src.ml.training.fitting import (
    _compute_ndcg,
    _compute_rank_ic,
    _compute_top_k_return,
    _fit_predict,
    _fit_predict_linear_baseline,
    _group_relevance,
    recency_sample_weight_for_fold,
)
from src.ml.training.policy_calibration import (
    _calibrate_close_morning_decision_oof,
    _calibrate_oof_policy,
    _policy_metadata,
)
from src.ml.training.validation import validate_pipeline_inputs
from src.processing.preprocessor import (
    LABEL_THRESHOLDS,
    RETURN_UNIT,
    build_ml_dataset,
)

logger = logging.getLogger(__name__)


def run_model_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    purge_gap: int = 1,
    model_type: str = "lgb_regressor",
    model_params: dict[str, int | float] | None = None,
    recency_half_life_groups: int | None = None,
) -> dict[str, Any]:
    """Train ML model using Purged Group Walk-Forward CV and evaluate OOF results.

    시간 컬럼은 필수 검증하지 않습니다. 고정된 업무 원천 규칙이며 모델 입력·CV
    분할·artifact 승인 조건이 아닙니다.

    ``recency_half_life_groups`` 가 주어지면 회귀 champion(Huber) 학습 시 각 fold
    의 train 거래일 그룹에서만 계산한 recency sample weight 를 전달합니다.
    검증 행·타깃·미래 날짜는 가중치 계산에 절대 사용되지 않습니다. 지원 값은
    ``None``(기존 expanding 동작), 252, 504 뿐이며, Ridge/LGBMRanker 는 비검증
    동작을 방지하기 위해 비영 recency 가중치를 거부합니다.

    Returns:
        dict containing 'oof_predictions', 'oof_df', 'metrics', 'trained_models',
        'backtest_eval', and bundle metadata (return_unit, round_trip_cost,
        label_thresholds, feature_manifest, training_cutoff, policy_params).
    """
    validate_pipeline_inputs(
        df,
        feature_cols,
        target_col,
        group_col,
        model_type,
        purge_gap,
        recency_half_life_groups,
    )

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
        sample_weight = recency_sample_weight_for_fold(
            train, group_col, recency_half_life_groups
        )
        model, pred = _fit_predict(
            model_type,
            train,
            val,
            feature_cols,
            target_col,
            group_col,
            model_params,
            sample_weight,
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
