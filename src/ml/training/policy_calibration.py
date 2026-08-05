"""OOF 정책 보정/선택 (policy calibration & selection)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.ml.quantile_model import fit_predict_quantile_and_classifier
from src.ml.single_stock_policy import (
    _DEFAULT_MIN_HISTORY_DATES,
    SingleStockPolicy,
    SingleStockPolicyEvaluation,
    default_policy_candidates,
    evaluate_single_stock_policy_oof,
)
from src.ml.sizing_engine import add_close_morning_decision_score
from src.ml.training.fitting import _align_close_morning_oof

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
    from src.ml.training.pipelines import run_model_pipeline

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


def _select_bad_probability_weight(
    candidate_stats: dict[float, dict[str, float]],
) -> float:
    """v2 하방위험 패널티 가중치를 보수적 규칙으로 선택합니다.

    ``candidate_stats`` 는 ``w_bad -> {scheduled_mean_return, entry_sequence_drawdown}``
    매핑입니다. ``w_bad=0``(v1) 은 항상 유효하며, 비영(非零) 후보는 내부 OOF
    scheduled mean 이 v1 이상이고 compounded close-to-morning MDD 가 엄격히
    낮을 때만 유효합니다. 유효 후보 중 최저 MDD → 높은 scheduled mean → 낮은
    ``w_bad`` 순으로 선택하며 조건을 충족하지 못하면 v1(``0.0``) 으로
    fail-closed 합니다. NaN 지표는 미충족으로 간주해 절대 v1 을 대체하지
    않습니다.
    """
    baseline = candidate_stats[0.0]
    base_mean = float(baseline["scheduled_mean_return"])
    base_mdd = float(baseline["entry_sequence_drawdown"])

    eligible: list[float] = [0.0]
    for weight, stats in sorted(candidate_stats.items()):
        if weight == 0.0:
            continue
        mean = float(stats["scheduled_mean_return"])
        mdd = float(stats["entry_sequence_drawdown"])
        mean_ge = np.isfinite(mean) and np.isfinite(base_mean) and mean >= base_mean
        mdd_lt = np.isfinite(mdd) and np.isfinite(base_mdd) and mdd < base_mdd
        if mean_ge and mdd_lt:
            eligible.append(weight)

    def _sort_key(weight: float) -> tuple[float, float, float]:
        stats = candidate_stats[weight]
        mdd = float(stats["entry_sequence_drawdown"])
        mean = float(stats["scheduled_mean_return"])
        return (
            mdd if np.isfinite(mdd) else np.inf,
            -mean if np.isfinite(mean) else -np.inf,
            weight,
        )

    return min(eligible, key=_sort_key)


def _select_recency_ensemble_config(
    candidate_stats: dict[tuple[int | None, float], dict[str, float]],
) -> tuple[int | None, float]:
    """내부 OOF 후보에서 보수적 규칙으로 ``(half_life, recent_weight)`` 를 선택합니다.

    baseline(alpha=0, v1) 은 항상 유효하며, 비영 후보는 내부 OOF scheduled mean
    이 v1 이상이고 compounded MDD 가 엄격히 낮을 때만 유효합니다. 유효 후보 중
    낮은 MDD → 높은 mean → 낮은 recent_weight → 긴 half_life 순으로 선택하고
    조건을 충족하지 못하면 ``(None, 0.0)`` v1 로 fail-closed 합니다. NaN 지표는
    미충족으로 간주해 절대 v1 을 대체하지 않습니다.
    """
    baseline_key: tuple[int | None, float] = (None, 0.0)
    baseline = candidate_stats[baseline_key]
    base_mean = float(baseline["scheduled_mean_return"])
    base_mdd = float(baseline["entry_sequence_drawdown"])

    eligible: list[tuple[int | None, float]] = [baseline_key]
    for config, stats in candidate_stats.items():
        if config == baseline_key:
            continue
        mean = float(stats["scheduled_mean_return"])
        mdd = float(stats["entry_sequence_drawdown"])
        mean_ge = np.isfinite(mean) and np.isfinite(base_mean) and mean >= base_mean
        mdd_lt = np.isfinite(mdd) and np.isfinite(base_mdd) and mdd < base_mdd
        if mean_ge and mdd_lt:
            eligible.append(config)

    def _sort_key(config: tuple[int | None, float]) -> tuple[float, float, float, float]:
        stats = candidate_stats[config]
        mdd = float(stats["entry_sequence_drawdown"])
        mean = float(stats["scheduled_mean_return"])
        half_life, recent_weight = config
        return (
            mdd if np.isfinite(mdd) else np.inf,
            -mean if np.isfinite(mean) else -np.inf,
            recent_weight,
            -float(half_life) if half_life is not None else float("-inf"),
        )

    return min(eligible, key=_sort_key)


def _dominant_recency_config(
    configs: list[tuple[int | None, float]],
) -> tuple[int | None, float]:
    """폴드 선택 구성 중 최빈값을 결정적 순서로 반환합니다 (연구 번들용).

    동률이면 낮은 recent_weight → 긴 half_life 순으로 우선합니다. ``(None, 0.0)``
    baseline 은 recent_weight=0 으로 가장 먼저 우선해, 폴드들이 baseline 으로
    fail-closed 했을 때 번들이 baseline 구성을 유지합니다.
    """
    counts: dict[tuple[int | None, float], int] = {}
    for config in configs:
        counts[config] = counts.get(config, 0) + 1

    def _precedes(a: tuple[int | None, float], b: tuple[int | None, float]) -> bool:
        a_half_life, a_alpha = a
        b_half_life, b_alpha = b
        if a_alpha != b_alpha:
            return a_alpha < b_alpha
        if a_half_life is None or b_half_life is None:
            return a_half_life is None
        return a_half_life > b_half_life

    best = configs[0]
    for config, count in counts.items():
        if count > counts[best] or (count == counts[best] and _precedes(config, best)):
            best = config
    return best


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
    from src.ml.training.pipelines import run_model_pipeline

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
