"""명시적으로 호출되는 연구 실험 (research experiments).

연구 실험 함수는 opt-in 이며 프로덕션 일일 추론 경로에서 실행되지 않습니다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.ml.backtest_evaluator import (
    _MIN_YEAR_SAMPLES,
    _aggregate_metrics,
    _extract_year,
    _max_drawdown,
)
from src.ml.purged_cv import PurgedGroupTimeSeriesSplit
from src.ml.quantile_model import fit_predict_quantile_and_classifier
from src.ml.single_stock_policy import (
    _DEFAULT_MIN_HISTORY_DATES,
    always_buy_policy,
    evaluate_single_stock_policy_oof,
)
from src.ml.sizing_engine import (
    _train_inline_bundle,
    add_close_morning_decision_score,
)
from src.ml.training.fitting import _align_close_morning_oof
from src.ml.training.pipelines import run_model_pipeline
from src.ml.training.policy_calibration import (
    _dominant_recency_config,
    _select_bad_probability_weight,
    _select_recency_ensemble_config,
)
from src.ml.training.validation import calculate_recency_sample_weight


def _evaluate_close_morning_top1(
    panel: pd.DataFrame,
    target_col: str,
    group_col: str,
    *,
    probability_weight: float,
    bad_probability_weight: float,
    min_history_dates: int,
    p_bad_col: str = "p_bad",
) -> dict[str, Any]:
    """``always_buy_top1`` 정책으로 ``panel`` 을 정확히 1회 평가합니다 (연구 전용)."""
    scored = add_close_morning_decision_score(
        panel,
        group_col=group_col,
        p_bad_col=p_bad_col,
        probability_weight=probability_weight,
        bad_probability_weight=bad_probability_weight,
    )
    cutoff = str(scored[group_col].max())
    evaluation = evaluate_single_stock_policy_oof(
        scored,
        target_col=target_col,
        group_col=group_col,
        stock_col="stock_code",
        policy_candidates=(always_buy_policy(cutoff, score_col="decision_score"),),
        min_history_dates=min_history_dates,
        scenario_col="chart_analysis",
        score_col="decision_score",
    )
    return {
        "metrics": dict(evaluation.metrics),
        "scheduled_returns": np.asarray(evaluation.scheduled_returns, dtype=np.float64),
        "dates": evaluation.decisions[group_col].to_numpy(),
    }


def _inner_close_morning_candidate_evaluator(
    inner_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    *,
    n_splits: int,
    purge_gap: int,
    probability_weight: float,
    min_history_dates: int,
    p_bad_col: str = "p_bad",
) -> dict[str, Any]:
    """외부 fold 의 train 분할만 사용하는 내부 purged walk-forward 후보 평가입니다.

    outer-train 날짜 전용 OOF(return ``pred`` + ``p_good``/``p_bad``)를 중첩
    walk-forward 로 산출하고, ``w_bad in {0, w_good, 2*w_good}`` 후보를
    ``always_buy_top1`` 로 평가한 뒤 ``_select_bad_probability_weight`` 의 보수적
    규칙으로 최적 ``w_bad`` 를 결정합니다. 선택은 전적으로 이 partition 의 OOF
    레이블만 사용하며 외부 validation 레이블을 절대 읽지 않습니다. partition 이
    중첩 walk-forward 를 지원할 만큼 충분하지 않으면 ``0.0`` 으로 fail-closed
    합니다.
    """
    inner_groups = int(inner_df[group_col].nunique())
    if inner_groups < 3:
        return {
            "chosen_weight": 0.0,
            "candidate_stats": {},
            "inner_n_groups": inner_groups,
            "inner_n_splits": 0,
            "inner_cutoff": None,
            "fail_closed_reason": "insufficient_inner_history",
        }
    inner_splits = max(1, min(n_splits, inner_groups - 2))

    result = run_model_pipeline(
        inner_df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=inner_splits,
        purge_gap=purge_gap,
        model_type="lgb_regressor",
    )
    risk_oof = fit_predict_quantile_and_classifier(
        inner_df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=inner_splits,
        purge_gap=purge_gap,
    )
    aligned = _align_close_morning_oof(
        result["oof_predictions"], risk_oof, target_col=target_col, group_col=group_col
    )
    aligned["rank_score"] = aligned["pred"]
    cutoff = str(aligned[group_col].max())

    candidate_stats: dict[float, dict[str, float]] = {}
    for weight in (0.0, probability_weight, 2.0 * probability_weight):
        evaluation = _evaluate_close_morning_top1(
            aligned,
            target_col,
            group_col,
            probability_weight=probability_weight,
            bad_probability_weight=weight,
            min_history_dates=min_history_dates,
            p_bad_col=p_bad_col,
        )
        metrics = evaluation["metrics"]
        candidate_stats[float(weight)] = {
            "scheduled_mean_return": float(metrics["scheduled_mean_return"]),
            "entry_sequence_drawdown": float(metrics["entry_sequence_drawdown"]),
            "scheduled_sharpe": float(metrics["scheduled_sharpe"]),
            "profit_factor": float(metrics["profit_factor"]),
            "buy_rate": float(metrics["buy_rate"]),
        }

    return {
        "chosen_weight": float(_select_bad_probability_weight(candidate_stats)),
        "candidate_stats": candidate_stats,
        "inner_n_groups": inner_groups,
        "inner_n_splits": inner_splits,
        "inner_cutoff": cutoff,
        "fail_closed_reason": None,
    }


def _aggregate_close_morning_metrics(
    scheduled: np.ndarray, n_buy: int
) -> dict[str, float]:
    """폴드 전체의 scheduled return 시계열로 실험 수준 지표를 집계합니다."""
    agg = _aggregate_metrics(scheduled)
    return {
        "n_scheduled_dates": int(scheduled.size),
        "n_buy": n_buy,
        "scheduled_mean_return": float(agg["top_1_return"]),
        "scheduled_win_rate": float(agg["win_rate"]),
        "profit_factor": float(agg["profit_factor"]),
        "scheduled_sharpe": float(agg["sharpe"]),
        "entry_sequence_drawdown": float(_max_drawdown(scheduled)),
    }


def _close_morning_yearly_breakdown(
    dates: np.ndarray, scheduled: np.ndarray
) -> dict[int, dict[str, float] | None]:
    """scheduled return 시계열의 연도별 분해를 반환합니다 (표본 <5년 미만 null)."""
    years = _extract_year(pd.Series(dates))
    breakdown: dict[int, dict[str, float] | None] = {}
    for year in np.unique(years):
        if not np.isfinite(year):
            continue
        mask = years == year
        if mask.sum() < _MIN_YEAR_SAMPLES:
            breakdown[int(year)] = None
            continue
        agg = _aggregate_metrics(scheduled[mask])
        breakdown[int(year)] = {
            "scheduled_mean_return": float(agg["top_1_return"]),
            "scheduled_win_rate": float(agg["win_rate"]),
            "profit_factor": float(agg["profit_factor"]),
            "scheduled_sharpe": float(agg["sharpe"]),
        }
    return breakdown


def run_close_morning_reranker_v2_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    purge_gap: int = 1,
    probability_weight: float = 0.5,
    min_history_dates: int = _DEFAULT_MIN_HISTORY_DATES,
    p_bad_col: str = "p_bad",
) -> dict[str, Any]:
    """close-morning reranker v2 중첩 선택 실험을 실행합니다 (연구 전용).

    외부 5-fold ``PurgedGroupTimeSeriesSplit``(purge_gap=1) 로 각 fold 의 외부
    validation 날짜를 1회 평가합니다. 폴드별 선택은 해당 fold 의 외부 train
    partition 만 사용하는 내부 purged walk-forward OOF 에서 ``w_bad`` 를 고르고,
    그 설정을 외부 validation 날짜에 1회 적용합니다. 외부 validation 레이블은
    설정 선택에 절대 사용하지 않습니다. ``save_model_artifacts`` 를 호출하지
    않고 프로덕션 번들이나 reranker 기본값을 변경하지 않습니다.

    Args:
        df: ``stock_code``/``chart_analysis`` 식별 컬럼을 포함한 OOF 패널 입력.
        feature_cols: 학습 피처 컬럼.
        target_col: 일자별 실현 순수익률(decimal net) 컬럼.
        group_col: 거래일 그룹 컬럼.
        n_splits: 외부 walk-forward fold 수.
        purge_gap: 보유기간 만큼의 purge group 수.
        probability_weight: ``w_good`` (v1 p_good 가중치, 기본 0.5).
        min_history_dates: ``always_buy_top1`` 의 기존 워밍업 의미.
        p_bad_col: v2 손실 확률 컬럼.

    Returns:
        dict: ``contract``(버전/후보 그리드/폴드 파라미터), ``folds``(폴드별
        선택 가중치·내부 후보 지표·v1/v2 외부 평가), ``chosen_weights``,
        ``aggregate``(폴드 연결 시계열의 v1/v2 실험 수준 지표) 포함.
    """
    if not {"stock_code", "chart_analysis"} <= set(df.columns):
        raise ValueError(
            "close-morning reranker v2 requires stock_code and chart_analysis columns"
        )
    if not 0.0 < probability_weight <= 1.0:
        raise ValueError(f"probability_weight must be in (0, 1], got {probability_weight}")
    if min_history_dates < 1:
        raise ValueError(f"min_history_dates must be >= 1, got {min_history_dates}")
    if purge_gap < 0:
        raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")

    work = df.sort_values(group_col).copy()
    result = run_model_pipeline(
        work,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
        model_type="lgb_regressor",
    )
    risk_oof = fit_predict_quantile_and_classifier(
        work,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
    )
    outer_aligned = _align_close_morning_oof(
        result["oof_predictions"], risk_oof, target_col=target_col, group_col=group_col
    )
    outer_aligned["rank_score"] = outer_aligned["pred"]

    splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)
    full_date_positions = {
        date: i for i, date in enumerate(sorted(work[group_col].unique()))
    }
    folds: list[dict[str, Any]] = []
    v1_series: list[np.ndarray] = []
    v2_series: list[np.ndarray] = []
    v1_dates: list[np.ndarray] = []
    v2_dates: list[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(work, y=work[target_col], groups=work[group_col])
    ):
        train_groups = set(work.iloc[train_idx][group_col].unique())
        val_groups = set(work.iloc[val_idx][group_col].unique())
        inner_df = work[work[group_col].isin(train_groups)]
        inner = _inner_close_morning_candidate_evaluator(
            inner_df,
            feature_cols=feature_cols,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            purge_gap=purge_gap,
            probability_weight=probability_weight,
            min_history_dates=min_history_dates,
            p_bad_col=p_bad_col,
        )
        chosen = float(inner["chosen_weight"])

        # 워밍업은 전체 타임라인 기준 첫 ``min_history_dates`` 날짜에만 적용됩니다.
        # 외부 validation 패널은 후행 날짜 구간이므로, 패널 내 워밍업 수는 패널
        # 시작일의 전체 위치가 워밍업 창을 벗어나면 0 이 됩니다.
        outer_val = outer_aligned[outer_aligned[group_col].isin(val_groups)]
        panel_start = full_date_positions[outer_val[group_col].min()]
        panel_dates = int(outer_val[group_col].nunique())
        effective_min_history_dates = max(
            1, min(panel_dates, min_history_dates - panel_start)
        )
        v1 = _evaluate_close_morning_top1(
            outer_val,
            target_col,
            group_col,
            probability_weight=probability_weight,
            bad_probability_weight=0.0,
            min_history_dates=effective_min_history_dates,
            p_bad_col=p_bad_col,
        )
        v2 = _evaluate_close_morning_top1(
            outer_val,
            target_col,
            group_col,
            probability_weight=probability_weight,
            bad_probability_weight=chosen,
            min_history_dates=effective_min_history_dates,
            p_bad_col=p_bad_col,
        )
        folds.append(
            {
                "fold": fold,
                "chosen_weight": chosen,
                "inner": inner,
                "v1": {
                    "metrics": v1["metrics"],
                    "n_buy": int(v1["metrics"]["n_buy"]),
                },
                "v2": {
                    "metrics": v2["metrics"],
                    "n_buy": int(v2["metrics"]["n_buy"]),
                },
            }
        )
        v1_series.append(v1["scheduled_returns"])
        v2_series.append(v2["scheduled_returns"])
        v1_dates.append(v1["dates"])
        v2_dates.append(v2["dates"])

    v1_cat = np.concatenate(v1_series)
    v2_cat = np.concatenate(v2_series)
    v1_dates_cat = np.concatenate(v1_dates)
    v2_dates_cat = np.concatenate(v2_dates)
    return {
        "contract": {
            "version": "close-morning-reranker-v2-research",
            "policy_candidate": "always_buy_top1",
            "candidate_weights": [0.0, probability_weight, 2.0 * probability_weight],
            "probability_weight": probability_weight,
            "n_splits": n_splits,
            "purge_gap": purge_gap,
            "min_history_dates": min_history_dates,
            "evaluation_cutoff": str(work[group_col].max()),
        },
        "folds": folds,
        "chosen_weights": [float(fold["chosen_weight"]) for fold in folds],
        "aggregate": {
            "v1": _aggregate_close_morning_metrics(
                v1_cat, int(sum(fold["v1"]["n_buy"] for fold in folds))
            ),
            "v2": _aggregate_close_morning_metrics(
                v2_cat, int(sum(fold["v2"]["n_buy"] for fold in folds))
            ),
        },
        "yearly_breakdown": {
            "v1": _close_morning_yearly_breakdown(v1_dates_cat, v1_cat),
            "v2": _close_morning_yearly_breakdown(v2_dates_cat, v2_cat),
        },
    }


def _recency_ensemble_rank(
    pred_expanding: pd.Series,
    pred_recent: pd.Series,
    group: pd.Series,
    recent_weight: float,
) -> pd.Series:
    """동일 거래일 내 백분위 순위를 convex blend 해 앙상블 순위를 산출합니다.

    ``(1 - recent_weight) * pct_rank(pred_expanding)
    + recent_weight * pct_rank(pred_recent)`` 로 두 전문가의 예측 스케일 차이를
    제거합니다. 그룹 단위 ``groupby().rank`` 벡터화를 사용하며 타깃/수익률
    컬럼은 절대 읽지 않습니다 (미래 정보 금지).
    """
    expanding_pct = pred_expanding.groupby(group).rank(pct=True, method="average")
    recent_pct = pred_recent.groupby(group).rank(pct=True, method="average")
    return (1.0 - recent_weight) * expanding_pct + recent_weight * recent_pct


def _inner_recency_ensemble_candidate_evaluator(
    inner_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    *,
    n_splits: int,
    purge_gap: int,
    probability_weight: float,
    min_history_dates: int,
    half_lives: tuple[int, ...],
    alphas: tuple[float, ...],
) -> dict[str, Any]:
    """외부 fold 의 train 분할만 사용하는 내부 purged walk-forward 후보 평가입니다.

    outer-train 날짜 전용 OOF(expanding + 252/504 half-life recent + p_good)를
    중첩 walk-forward 로 산출하고, ``(half_life, recent_weight)`` 고정 후보를
    ``always_buy_top1`` 로 평가한 뒤 ``_select_recency_ensemble_config`` 의 보수적
    규칙으로 최적 구성을 결정합니다. 선택은 전적으로 이 partition 의 OOF 레이블만
    사용하며 외부 validation 레이블을 절대 읽지 않습니다. partition 이 중첩
    walk-forward 를 지원할 만큼 충분하지 않으면 baseline 으로 fail-closed 합니다.
    """
    inner_groups = int(inner_df[group_col].nunique())
    if inner_groups < 3:
        return {
            "chosen_config": (None, 0.0),
            "candidate_stats": {},
            "inner_n_groups": inner_groups,
            "inner_n_splits": 0,
            "inner_cutoff": None,
            "fail_closed_reason": "insufficient_inner_history",
        }
    inner_splits = max(1, min(n_splits, inner_groups - 2))

    expanding = run_model_pipeline(
        inner_df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=inner_splits,
        purge_gap=purge_gap,
        model_type="lgb_regressor",
    )
    recent_by_h: dict[int, dict[str, Any]] = {
        half_life: run_model_pipeline(
            inner_df,
            feature_cols=feature_cols,
            target_col=target_col,
            group_col=group_col,
            n_splits=inner_splits,
            purge_gap=purge_gap,
            model_type="lgb_regressor",
            recency_half_life_groups=half_life,
        )
        for half_life in half_lives
    }
    risk_oof = fit_predict_quantile_and_classifier(
        inner_df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=inner_splits,
        purge_gap=purge_gap,
    )
    aligned_exp = _align_close_morning_oof(
        expanding["oof_predictions"], risk_oof, target_col=target_col, group_col=group_col
    )
    aligned_recent = {
        half_life: _align_close_morning_oof(
            recent_by_h[half_life]["oof_predictions"],
            risk_oof,
            target_col=target_col,
            group_col=group_col,
        )
        for half_life in half_lives
    }

    def _evaluate(rank_score: pd.Series) -> dict[str, float]:
        panel = aligned_exp.copy()
        panel["rank_score"] = rank_score
        evaluation = _evaluate_close_morning_top1(
            panel,
            target_col,
            group_col,
            probability_weight=probability_weight,
            bad_probability_weight=0.0,
            min_history_dates=min_history_dates,
        )
        metrics = evaluation["metrics"]
        return {
            "scheduled_mean_return": float(metrics["scheduled_mean_return"]),
            "entry_sequence_drawdown": float(metrics["entry_sequence_drawdown"]),
            "scheduled_sharpe": float(metrics["scheduled_sharpe"]),
            "profit_factor": float(metrics["profit_factor"]),
            "buy_rate": float(metrics["buy_rate"]),
        }

    candidate_stats: dict[tuple[int | None, float], dict[str, float]] = {}
    baseline_rank = _recency_ensemble_rank(
        aligned_exp["pred"], aligned_exp["pred"], aligned_exp[group_col], 0.0
    )
    candidate_stats[(None, 0.0)] = _evaluate(baseline_rank)
    for half_life in half_lives:
        for recent_weight in alphas:
            if recent_weight == 0.0:
                continue
            rank_score = _recency_ensemble_rank(
                aligned_exp["pred"],
                aligned_recent[half_life]["pred"],
                aligned_exp[group_col],
                recent_weight,
            )
            candidate_stats[(half_life, recent_weight)] = _evaluate(rank_score)

    return {
        "chosen_config": _select_recency_ensemble_config(candidate_stats),
        "candidate_stats": candidate_stats,
        "inner_n_groups": inner_groups,
        "inner_n_splits": inner_splits,
        "inner_cutoff": str(aligned_exp[group_col].max()),
        "fail_closed_reason": None,
    }


def run_close_morning_recency_ensemble_experiment(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    purge_gap: int = 1,
    probability_weight: float = 0.5,
    min_history_dates: int = _DEFAULT_MIN_HISTORY_DATES,
    p_bad_col: str = "p_bad",
    half_lives: tuple[int, ...] = (252, 504),
    alphas: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0),
    build_research_bundle: bool = False,
) -> dict[str, Any]:
    """close-morning recency-adaptive dual-horizon ensemble 중첩 선택 실험 (연구 전용).

    외부 ``PurgedGroupTimeSeriesSplit(n_splits, purge_gap)`` 로 각 fold 의 외부
    validation 날짜를 1회 평가합니다. 폴드별 ``(half_life, recent_weight)`` 구성은
    해당 fold 의 외부 train partition 만 사용하는 내부 purged walk-forward OOF
    에서 고르고, 그 설정을 외부 validation 날짜에 1회 적용합니다. 외부 validation
    레이블은 설정 선택에 절대 사용하지 않습니다. ``save_model_artifacts`` 를
    호출하지 않고 프로덕션 번들이나 기본값을 변경하지 않습니다.

    ``alpha=0`` 은 v1 expanding baseline 으로 중복 제거되며, ``p_good`` 백분위
    기여와 ``always_buy_top1`` 정책을 유지합니다. ``build_research_bundle=True``
    일 때에만, 승격 게이트(후보가 v1 대비 scheduled mean 우위 + MDD 엄격 감소 +
    평균 양수 + PF>1)를 통과하면 두 return 모델과 ``recency_ensemble_config`` 를
    포함한 연구 번들을 결과에 영속화합니다.

    Args:
        df: ``stock_code``/``chart_analysis`` 식별 컬럼을 포함한 OOF 패널 입력.
        feature_cols: 학습 피처 컬럼.
        target_col: 일자별 실현 순수익률(decimal net) 컬럼.
        group_col: 거래일 그룹 컬럼.
        n_splits: 외부 walk-forward fold 수.
        purge_gap: 보유기간 만큼의 purge group 수.
        probability_weight: ``w_good`` (v1 p_good 가중치, 기본 0.5).
        min_history_dates: ``always_buy_top1`` 의 기존 워밍업 의미.
        p_bad_col: v2 손실 확률 컬럼 (본 실험에서 비영 penalty 미사용).
        half_lives: 최근 전문가 half-life 후보 (252 또는 504).
        alphas: 최근 전문가 가중치 후보 (0.0 baseline 포함).
        build_research_bundle: 승격 게이트 통과 시 연구 번들 영속화 여부.

    Returns:
        dict: ``contract``, ``folds``(폴드별 선택 구성·내부 후보 지표·baseline/
        candidate 외부 평가), ``chosen_configs``, ``aggregate``(폴드 연결 시계열의
        baseline/candidate 지표), ``yearly_breakdown``, ``promotion``(승격 심사),
        ``research_bundle``(선택 시).
    """
    if not {"stock_code", "chart_analysis"} <= set(df.columns):
        raise ValueError(
            "close-morning recency ensemble requires stock_code and chart_analysis columns"
        )
    if not 0.0 < probability_weight <= 1.0:
        raise ValueError(f"probability_weight must be in (0, 1], got {probability_weight}")
    if min_history_dates < 1:
        raise ValueError(f"min_history_dates must be >= 1, got {min_history_dates}")
    if purge_gap < 0:
        raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
    if not half_lives or not all(h in (252, 504) for h in half_lives):
        raise ValueError(
            f"half_lives must be a non-empty subset of (252, 504), got {half_lives!r}"
        )
    if not alphas or 0.0 not in alphas or not all(0.0 <= a <= 1.0 for a in alphas):
        raise ValueError(
            f"alphas must be a non-empty subset of [0, 1] containing 0.0, got {alphas!r}"
        )

    work = df.sort_values(group_col).copy()
    expanding = run_model_pipeline(
        work,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
        model_type="lgb_regressor",
    )
    recent_by_h: dict[int, dict[str, Any]] = {
        half_life: run_model_pipeline(
            work,
            feature_cols=feature_cols,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            purge_gap=purge_gap,
            model_type="lgb_regressor",
            recency_half_life_groups=half_life,
        )
        for half_life in half_lives
    }
    risk_oof = fit_predict_quantile_and_classifier(
        work,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        n_splits=n_splits,
        purge_gap=purge_gap,
    )
    aligned_exp = _align_close_morning_oof(
        expanding["oof_predictions"], risk_oof, target_col=target_col, group_col=group_col
    )
    aligned_recent = {
        half_life: _align_close_morning_oof(
            recent_by_h[half_life]["oof_predictions"],
            risk_oof,
            target_col=target_col,
            group_col=group_col,
        )
        for half_life in half_lives
    }

    splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)
    full_date_positions = {
        date: i for i, date in enumerate(sorted(work[group_col].unique()))
    }
    folds: list[dict[str, Any]] = []
    chosen_config_tuples: list[tuple[int | None, float]] = []
    baseline_series: list[np.ndarray] = []
    candidate_series: list[np.ndarray] = []
    baseline_dates: list[np.ndarray] = []
    candidate_dates: list[np.ndarray] = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(work, y=work[target_col], groups=work[group_col])
    ):
        train_groups = set(work.iloc[train_idx][group_col].unique())
        val_groups = set(work.iloc[val_idx][group_col].unique())
        inner_df = work[work[group_col].isin(train_groups)]
        inner = _inner_recency_ensemble_candidate_evaluator(
            inner_df,
            feature_cols=feature_cols,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            purge_gap=purge_gap,
            probability_weight=probability_weight,
            min_history_dates=min_history_dates,
            half_lives=half_lives,
            alphas=alphas,
        )
        chosen_half_life, chosen_alpha = inner["chosen_config"]

        outer_val = aligned_exp[aligned_exp[group_col].isin(val_groups)]
        panel_start = full_date_positions[outer_val[group_col].min()]
        panel_dates = int(outer_val[group_col].nunique())
        effective_min_history_dates = max(
            1, min(panel_dates, min_history_dates - panel_start)
        )

        baseline_rank = _recency_ensemble_rank(
            outer_val["pred"], outer_val["pred"], outer_val[group_col], 0.0
        )
        baseline_panel = outer_val.copy()
        baseline_panel["rank_score"] = baseline_rank
        baseline_eval = _evaluate_close_morning_top1(
            baseline_panel,
            target_col,
            group_col,
            probability_weight=probability_weight,
            bad_probability_weight=0.0,
            min_history_dates=effective_min_history_dates,
            p_bad_col=p_bad_col,
        )

        if chosen_half_life is None or chosen_alpha == 0.0:
            candidate_eval = baseline_eval
        else:
            recent_outer_val = aligned_recent[chosen_half_life][
                aligned_recent[chosen_half_life][group_col].isin(val_groups)
            ]
            candidate_rank = _recency_ensemble_rank(
                outer_val["pred"],
                recent_outer_val["pred"],
                outer_val[group_col],
                chosen_alpha,
            )
            candidate_panel = outer_val.copy()
            candidate_panel["rank_score"] = candidate_rank
            candidate_eval = _evaluate_close_morning_top1(
                candidate_panel,
                target_col,
                group_col,
                probability_weight=probability_weight,
                bad_probability_weight=0.0,
                min_history_dates=effective_min_history_dates,
                p_bad_col=p_bad_col,
            )

        folds.append(
            {
                "fold": fold,
                "chosen_config": {
                    "half_life": chosen_half_life,
                    "recent_weight": chosen_alpha,
                },
                "inner": inner,
                "baseline": {
                    "metrics": dict(baseline_eval["metrics"]),
                    "n_buy": int(baseline_eval["metrics"]["n_buy"]),
                },
                "candidate": {
                    "metrics": dict(candidate_eval["metrics"]),
                    "n_buy": int(candidate_eval["metrics"]["n_buy"]),
                },
            }
        )
        chosen_config_tuples.append((chosen_half_life, chosen_alpha))
        baseline_series.append(baseline_eval["scheduled_returns"])
        candidate_series.append(candidate_eval["scheduled_returns"])
        baseline_dates.append(baseline_eval["dates"])
        candidate_dates.append(candidate_eval["dates"])

    baseline_cat = np.concatenate(baseline_series)
    candidate_cat = np.concatenate(candidate_series)
    baseline_dates_cat = np.concatenate(baseline_dates)
    candidate_dates_cat = np.concatenate(candidate_dates)

    baseline_agg = _aggregate_close_morning_metrics(
        baseline_cat, int(sum(fold["baseline"]["n_buy"] for fold in folds))
    )
    candidate_agg = _aggregate_close_morning_metrics(
        candidate_cat, int(sum(fold["candidate"]["n_buy"] for fold in folds))
    )

    base_mean = float(baseline_agg["scheduled_mean_return"])
    cand_mean = float(candidate_agg["scheduled_mean_return"])
    base_mdd = float(baseline_agg["entry_sequence_drawdown"])
    cand_mdd = float(candidate_agg["entry_sequence_drawdown"])
    cand_pf = float(candidate_agg["profit_factor"])
    beats_mean = np.isfinite(cand_mean) and np.isfinite(base_mean) and cand_mean > base_mean
    lower_mdd = np.isfinite(cand_mdd) and np.isfinite(base_mdd) and cand_mdd < base_mdd
    positive_mean = np.isfinite(cand_mean) and cand_mean > 0.0
    pf_above_one = np.isfinite(cand_pf) and cand_pf > 1.0
    promotion = {
        "promoted": bool(beats_mean and lower_mdd and positive_mean and pf_above_one),
        "candidate_beats_baseline_mean": bool(beats_mean),
        "candidate_lower_compounded_mdd": bool(lower_mdd),
        "positive_scheduled_net_mean": bool(positive_mean),
        "profit_factor_above_one": bool(pf_above_one),
    }

    research_bundle: dict[str, Any] | None = None
    if build_research_bundle and promotion["promoted"]:
        dominant_half_life, dominant_alpha = _dominant_recency_config(chosen_config_tuples)
        bundle_half_life = (
            dominant_half_life if dominant_half_life is not None else half_lives[0]
        )
        recent_weights = calculate_recency_sample_weight(work[group_col], bundle_half_life)
        recent_return_model = LGBMRegressor(objective="huber", random_state=42)
        recent_return_model.fit(work[feature_cols], work[target_col], sample_weight=recent_weights)
        recency_ensemble_config = {
            "version": "close-morning-recency-ensemble-research",
            "half_life_groups": bundle_half_life,
            "recent_weight": dominant_alpha,
            "probability_weight": probability_weight,
            "score_col": "decision_score",
        }
        research_bundle = _train_inline_bundle(
            work,
            feature_cols,
            target_col,
            group_col,
            calibration_diagnostics=risk_oof.attrs.get("calibration_diagnostics", []),
            recent_return_model=recent_return_model,
            recency_ensemble_config=recency_ensemble_config,
        )

    return {
        "contract": {
            "version": "close-morning-recency-ensemble-research",
            "policy_candidate": "always_buy_top1",
            "half_lives": list(half_lives),
            "alphas": list(alphas),
            "probability_weight": probability_weight,
            "n_splits": n_splits,
            "purge_gap": purge_gap,
            "min_history_dates": min_history_dates,
            "evaluation_cutoff": str(work[group_col].max()),
        },
        "folds": folds,
        "chosen_configs": [
            {"half_life": half_life, "recent_weight": recent_weight}
            for half_life, recent_weight in chosen_config_tuples
        ],
        "aggregate": {
            "baseline": baseline_agg,
            "candidate": candidate_agg,
        },
        "yearly_breakdown": {
            "baseline": _close_morning_yearly_breakdown(baseline_dates_cat, baseline_cat),
            "candidate": _close_morning_yearly_breakdown(candidate_dates_cat, candidate_cat),
        },
        "promotion": promotion,
        "research_bundle": research_bundle,
    }
