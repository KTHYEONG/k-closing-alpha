"""Single stock policy evaluation ported (archival)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ml.metrics import aggregate_metrics, extract_year, max_drawdown
from src.serving.realtime.policy import (
    _ALWAYS_BUY_CANDIDATE,
    _POLICY_VERSION,
    SingleStockPolicy,
    _build_decision_rows,
    _build_panel,
    _candidate_quantile,
    _compute_margins,
    _PanelMargins,
    always_buy_policy,
    margin_quantile_policy,
    resolve_stock_actions,
)

_DEFAULT_MIN_HISTORY_DATES = 252
_DEFAULT_QUANTILE_GRID: tuple[float, ...] = (0.70, 0.90)
_MIN_YEAR_SAMPLES = 5


@dataclass(frozen=True)
class SingleStockPolicyEvaluation:
    """단일 종목 정책의 인과적 OOF 평가 결과."""

    selected_policy: SingleStockPolicy
    decisions: pd.DataFrame
    scheduled_returns: np.ndarray
    metrics: dict[str, Any]
    yearly_breakdown: dict[int, dict[str, float] | None]
    market_type_breakdown: dict[str, dict[str, float]]
    candidate_results: dict[str, dict[str, float]]


def default_policy_candidates(
    calibration_cutoff: str,
    *,
    grid: tuple[float, ...] = _DEFAULT_QUANTILE_GRID,
    score_col: str = "rank_score",
    version: str = _POLICY_VERSION,
) -> tuple[SingleStockPolicy, ...]:
    """버전화된 후보 정책 집합(always_buy + margin quantile grid)을 반환합니다."""
    always = always_buy_policy(calibration_cutoff, version=version, score_col=score_col)
    margins = tuple(
        margin_quantile_policy(q, calibration_cutoff, version=version, score_col=score_col)
        for q in grid
    )
    return (always, *margins)


def _validate_oof(
    oof_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    stock_col: str,
    scenario_col: str,
    score_col: str,
) -> None:
    required = [group_col, target_col, stock_col, scenario_col, score_col]
    missing = [col for col in required if col not in oof_df.columns]
    if missing:
        raise ValueError(f"missing required columns in oof_df: {missing}")
    null_cols = [col for col in required if oof_df[col].isna().any()]
    if null_cols:
        raise ValueError(f"required columns contain nulls: {null_cols}")
    parsed = pd.to_datetime(oof_df[group_col], errors="coerce", format="mixed")
    if parsed.isna().any():
        raise ValueError("group_col contains unparseable dates (chronology contract violation)")
    for col, label in ((score_col, "score"), (target_col, "target")):
        values = oof_df[col].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {label} column {col!r}")


def _validate_candidates(
    policy_candidates: tuple[SingleStockPolicy, ...],
) -> tuple[SingleStockPolicy, ...]:
    if not policy_candidates:
        raise ValueError("policy_candidates must not be empty")
    seen: set[str] = set()
    for cand in policy_candidates:
        if not isinstance(cand, SingleStockPolicy):
            raise ValueError("policy_candidates must be SingleStockPolicy instances")
        if cand.candidate in seen:
            raise ValueError(f"duplicate policy candidate {cand.candidate!r}")
        seen.add(cand.candidate)
    return policy_candidates


def _causal_thresholds(
    margins: np.ndarray,
    q: float,
    min_history_dates: int,
) -> np.ndarray:
    """날짜별 마진 문턱을 이전 날짜(< D)의 마진만으로 계산합니다."""
    n = margins.size
    thresholds = np.full(n, np.nan)
    for i in range(min_history_dates, n):
        prior = margins[:i]
        prior = prior[np.isfinite(prior)]
        if prior.size == 0:
            continue
        thresholds[i] = float(np.quantile(prior, q))
    return thresholds


def _candidate_stats(scheduled: np.ndarray, buy: np.ndarray) -> dict[str, float]:
    agg = aggregate_metrics(scheduled)
    active = scheduled[buy]
    return {
        "scheduled_mean_return": float(agg["top_1_return"]),
        "scheduled_sharpe": float(agg["sharpe"]),
        "scheduled_win_rate": float(agg["win_rate"]),
        "profit_factor": float(agg["profit_factor"]),
        "entry_sequence_drawdown": max_drawdown(scheduled),
        "buy_rate": float(np.mean(buy)) if buy.size else float("nan"),
        "active_trade_mean_return": float(np.mean(active)) if active.size else float("nan"),
        "active_trade_win_rate": float(np.mean(active > 0.0)) if active.size else float("nan"),
    }


def _select_best_candidate(outcomes: dict[str, dict[str, Any]]) -> str:
    """결정적 목적함수 순서로 최적 후보를 선택합니다."""
    rows = [
        {
            "policy_id": pid,
            "mean": item["stats"]["scheduled_mean_return"],
            "sharpe": item["stats"]["scheduled_sharpe"],
            "mdd": item["stats"]["entry_sequence_drawdown"],
            "buy_rate": item["stats"]["buy_rate"],
        }
        for pid, item in outcomes.items()
    ]
    frame = pd.DataFrame(rows)
    frame["mdd"] = frame["mdd"].fillna(0.0)
    frame["sharpe"] = frame["sharpe"].fillna(-np.inf)
    frame = frame.sort_values(
        ["mean", "sharpe", "mdd", "buy_rate", "policy_id"],
        ascending=[False, False, True, False, True],
        kind="mergesort",
    )
    return str(frame.iloc[0]["policy_id"])


def _turnover_selected_codes(codes: np.ndarray) -> float:
    if codes.size <= 1:
        return float("nan")
    changed = codes[1:] != codes[:-1]
    return float(np.mean(changed))


def _finalize_selected_policy(
    candidate: SingleStockPolicy,
    margins: _PanelMargins,
    *,
    n_dates: int,
    calibration_cutoff: str,
) -> SingleStockPolicy:
    if candidate.candidate == _ALWAYS_BUY_CANDIDATE:
        return always_buy_policy(
            calibration_cutoff,
            version=candidate.version,
            score_col=candidate.score_col,
        )
    finite = margins.margin[np.isfinite(margins.margin)]
    q = _candidate_quantile(candidate.candidate)
    threshold: float | None = float(np.quantile(finite, q)) if finite.size else None
    return margin_quantile_policy(
        q,
        calibration_cutoff,
        version=candidate.version,
        score_col=candidate.score_col,
        margin_threshold=threshold,
        reference_margin=tuple(float(v) for v in finite),
        history_length=n_dates,
    )


def _build_evaluation_metrics(
    rows: pd.DataFrame,
    scheduled: np.ndarray,
    *,
    group_col: str,
    stock_col: str,
    market_type_col: str | None,
) -> dict[str, Any]:
    buy = rows["decision"].to_numpy() == "BUY"
    n = int(scheduled.size)
    agg = aggregate_metrics(scheduled)
    active = scheduled[buy]
    active_mean = float(np.mean(active)) if active.size else float("nan")
    active_win = float(np.mean(active > 0.0)) if active.size else float("nan")
    reasons = rows["decision_reason"].to_numpy()
    reason_counts: dict[str, int] = {}
    for value in np.unique(reasons):
        reason_counts[str(value)] = int(np.sum(reasons == value))
    selected = rows[stock_col].to_numpy(dtype=object)[buy]
    return {
        "n_scheduled_dates": n,
        "n_buy": int(buy.sum()),
        "n_abstain": n - int(buy.sum()),
        "buy_rate": float(np.mean(buy)),
        "abstain_rate": float(np.mean(~buy)),
        "reason_counts": reason_counts,
        "scheduled_mean_return": float(agg["top_1_return"]),
        "scheduled_win_rate": float(agg["win_rate"]),
        "profit_factor": float(agg["profit_factor"]),
        "scheduled_sharpe": float(agg["sharpe"]),
        "active_trade_mean_return": active_mean,
        "active_trade_win_rate": active_win,
        "turnover": _turnover_selected_codes(selected),
        "entry_sequence_drawdown": max_drawdown(scheduled),
    }


def _yearly_breakdown(
    daily: np.ndarray,
    years: np.ndarray,
) -> dict[int, dict[str, float] | None]:
    valid = np.isfinite(daily) & np.isfinite(years)
    yv = years[valid]
    top1 = daily[valid]
    breakdown: dict[int, dict[str, float] | None] = {}
    for year in np.unique(yv):
        mask = yv == year
        if mask.sum() < _MIN_YEAR_SAMPLES:
            breakdown[int(year)] = None
            continue
        metrics = aggregate_metrics(top1[mask])
        breakdown[int(year)] = {
            "top1_return": metrics["top_1_return"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "sharpe": metrics["sharpe"],
        }
    return breakdown


def evaluate_single_stock_policy_oof(
    oof_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    stock_col: str,
    policy_candidates: tuple[SingleStockPolicy, ...],
    min_history_dates: int,
    *,
    scenario_col: str = "chart_analysis",
    score_col: str = "rank_score",
) -> SingleStockPolicyEvaluation:
    """OOF 패널에서 단일 종목 정책을 인과적으로 보정·평가합니다."""
    if min_history_dates < 1:
        raise ValueError(f"min_history_dates must be >= 1, got {min_history_dates}")
    _validate_candidates(policy_candidates)
    _validate_oof(oof_df, target_col, group_col, stock_col, scenario_col, score_col)

    resolved = resolve_stock_actions(
        oof_df,
        group_col,
        stock_col=stock_col,
        scenario_col=scenario_col,
        score_col=score_col,
        mode="score_best_action",
    )
    duplicates = resolved.duplicated(subset=[group_col, stock_col], keep=False)
    if duplicates.any():
        raise ValueError("resolved date-stock keys are not unique (data-contract violation)")

    panel = _build_panel(resolved, group_col, stock_col, scenario_col, score_col)
    margins = _compute_margins(
        panel,
        group_col,
        stock_col,
        scenario_col,
        score_col,
        target_col=target_col,
        market_type_col="market_type",
    )
    n = margins.sizes.size
    warm_up = np.arange(n) < min_history_dates

    candidate_outcomes: dict[str, dict[str, Any]] = {}
    for cand in policy_candidates:
        if cand.candidate == _ALWAYS_BUY_CANDIDATE:
            thresholds: np.ndarray | None = None
        else:
            thresholds = _causal_thresholds(
                margins.margin, _candidate_quantile(cand.candidate), min_history_dates
            )
        rows = _build_decision_rows(
            margins,
            cand,
            group_col=group_col,
            stock_col=stock_col,
            scenario_col=scenario_col,
            score_col=score_col,
            thresholds=thresholds,
            warm_up=warm_up,
        )
        buy = rows["decision"].to_numpy() == "BUY"
        assert margins.winner_target is not None
        scheduled = np.where(buy, margins.winner_target, 0.0).astype(np.float64)
        candidate_outcomes[cand.policy_id] = {
            "candidate": cand,
            "rows": rows,
            "scheduled": scheduled,
            "stats": _candidate_stats(scheduled, buy),
        }

    selected_id = _select_best_candidate(candidate_outcomes)
    selected = candidate_outcomes[selected_id]
    calibration_cutoff = str(panel[group_col].max())
    selected_policy = _finalize_selected_policy(
        selected["candidate"], margins, n_dates=n, calibration_cutoff=calibration_cutoff
    )

    rows_df = selected["rows"].copy()
    scheduled = selected["scheduled"].astype(np.float64)
    rows_df["scheduled_return"] = scheduled
    metrics = _build_evaluation_metrics(
        rows_df,
        scheduled,
        group_col=group_col,
        stock_col=stock_col,
        market_type_col="market_type" if "market_type" in rows_df.columns else None,
    )

    years = extract_year(rows_df[group_col])
    yearly = _yearly_breakdown(scheduled, years)

    market_breakdown: dict[str, dict[str, float]] = {}
    if "market_type" in rows_df.columns:
        buy = rows_df["decision"].to_numpy() == "BUY"
        market_values = rows_df["market_type"].to_numpy(dtype=object)
        for value in np.unique(market_values[buy]):
            mask = buy & (market_values == value)
            market_breakdown[str(value)] = aggregate_metrics(scheduled[mask])

    return SingleStockPolicyEvaluation(
        selected_policy=selected_policy,
        decisions=rows_df,
        scheduled_returns=scheduled,
        metrics=metrics,
        yearly_breakdown=yearly,
        market_type_breakdown=market_breakdown,
        candidate_results={pid: item["stats"] for pid, item in candidate_outcomes.items()},
    )
