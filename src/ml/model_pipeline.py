"""ML 학습 파이프라인 호환성 퍼사드 모듈.

구현은 ``src.ml.training.*`` 로 분리되었으며, 이 모듈은 공개 심볼만 재-export
합니다. 중복 구현이 없고 마이그레이션 기간 동안 기존 import 경로를 보장합니다.
"""

from __future__ import annotations

from src.ml.training.experiments import (
    _aggregate_close_morning_metrics,
    _close_morning_yearly_breakdown,
    _evaluate_close_morning_top1,
    _inner_close_morning_candidate_evaluator,
    _inner_recency_ensemble_candidate_evaluator,
    _recency_ensemble_rank,
    run_close_morning_recency_ensemble_experiment,
    run_close_morning_reranker_v2_experiment,
)
from src.ml.training.fitting import (
    _align_close_morning_oof,
    _compute_ndcg,
    _compute_rank_ic,
    _compute_top_k_return,
    _fit_predict,
    _fit_predict_linear_baseline,
    _group_metric,
    _group_relevance,
    _ndcg_at_k,
)
from src.ml.training.pipelines import (
    _FINITE_QUALITY_METRICS,
    _RERANKER_FEATURE_SET,
    _policy_entry,
    _quality_score,
    _validate_quality_metrics,
    evaluate_close_morning_quality,
    run_model_pipeline,
    run_sizing_pipeline,
)
from src.ml.training.policy_calibration import (
    _POLICY_METRIC_KEYS,
    _calibrate_close_morning_decision_oof,
    _calibrate_oof_policy,
    _dominant_recency_config,
    _policy_metadata,
    _select_bad_probability_weight,
    _select_recency_ensemble_config,
)
from src.ml.training.validation import (
    _MODEL_TYPES,
    calculate_recency_sample_weight,
)

__all__ = [
    "_FINITE_QUALITY_METRICS",
    "_MODEL_TYPES",
    "_POLICY_METRIC_KEYS",
    "_RERANKER_FEATURE_SET",
    "_aggregate_close_morning_metrics",
    "_align_close_morning_oof",
    "_calibrate_close_morning_decision_oof",
    "_calibrate_oof_policy",
    "_close_morning_yearly_breakdown",
    "_compute_ndcg",
    "_compute_rank_ic",
    "_compute_top_k_return",
    "_dominant_recency_config",
    "_evaluate_close_morning_top1",
    "_fit_predict",
    "_fit_predict_linear_baseline",
    "_group_metric",
    "_group_relevance",
    "_inner_close_morning_candidate_evaluator",
    "_inner_recency_ensemble_candidate_evaluator",
    "_ndcg_at_k",
    "_policy_entry",
    "_policy_metadata",
    "_quality_score",
    "_recency_ensemble_rank",
    "_select_bad_probability_weight",
    "_select_recency_ensemble_config",
    "_validate_quality_metrics",
    "calculate_recency_sample_weight",
    "evaluate_close_morning_quality",
    "run_close_morning_recency_ensemble_experiment",
    "run_close_morning_reranker_v2_experiment",
    "run_model_pipeline",
    "run_sizing_pipeline",
]
