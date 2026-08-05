"""OOF 백테스트 평가: Baseline 비교 및 연도/국면별 안정성 감사.

`docs/specs/ml_strategy_improvement.md` P0/P1 요구사항:
- ``selection_rank`` 는 필수 baseline 컬럼입니다. 누락 시 경고가 아니라
  ValueError 로 fail-closed 합니다.
- 동일 OOF 날짜 집합에서 selection-rank / equal-weight / regularized-linear
  baseline 을 모두 평가합니다.
- 모든 contender 에 대해 날짜가중·자본가중 수익률, 비용차감 수익률, 턴오버,
  최대 드로다운, 연도/국면(시장구분)/시가총액 분해를 계산합니다.

모든 성능 지표는 NumPy 벡터 연산만 사용합니다 (pd.apply 루프 금지).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DAILY_ANNUALIZATION = float(np.sqrt(252.0))
_MIN_YEAR_SAMPLES = 5
_BASE_METRIC_KEYS = ("top_1_return", "win_rate", "profit_factor", "mean_win", "mean_loss", "sharpe")
_EXTENDED_METRIC_KEYS = (
    *_BASE_METRIC_KEYS,
    "top_3_return",
    "cost_adjusted_return",
    "date_weighted_return",
    "capital_weighted_return",
    "turnover",
    "max_drawdown",
)
_YEARLY_METRIC_KEYS = ("top1_return", "top3_return", "win_rate", "profit_factor", "sharpe")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})")


def _extract_year(groups: pd.Series) -> np.ndarray:
    """그룹 컬럼(거래일) 값에서 연도 배열(float64)을 추출합니다."""
    parsed = pd.to_datetime(groups, errors="coerce", format="mixed")
    years: np.ndarray = np.asarray(parsed.dt.year.to_numpy(dtype=np.float64), dtype=np.float64)
    missing = ~np.isfinite(years)
    if missing.any():
        raw = groups.astype(str).to_numpy()
        for idx in np.flatnonzero(missing):
            match = _YEAR_RE.search(raw[idx])
            years[idx] = float(match.group(1)) if match else np.nan
    return years


def _group_starts(group_vals: np.ndarray) -> np.ndarray:
    """그룹화된(정렬된) 배열에서 각 그룹 시작 인덱스를 반환합니다."""
    change = np.concatenate(([True], group_vals[1:] != group_vals[:-1]))
    return np.flatnonzero(change)


def _positions_within_group(group_vals: np.ndarray) -> np.ndarray:
    """그룹 내 0-based 위치 배열을 반환합니다."""
    starts = _group_starts(group_vals)
    sizes = np.diff(np.concatenate((starts, np.array([group_vals.size]))))
    arange: np.ndarray = np.arange(group_vals.size)
    repeats: np.ndarray = np.repeat(starts, sizes)
    return np.asarray(arange - repeats, dtype=np.intp)


def _group_series(
    oof: pd.DataFrame,
    group_col: str,
    target_col: str,
    score_col: str,
    ascending: bool,
    k: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """score 정렬 기준 일자별 포트폴리오 수익률 시계열과 일자별 연도를 반환합니다.

    k=None 이면 그룹 전체 동일가중 평균(무작위 Baseline)을 사용합니다.
    """
    sorted_df = oof.sort_values([group_col, score_col], ascending=[True, ascending], kind="mergesort")
    targets = sorted_df[target_col].to_numpy(dtype=np.float64)
    group_vals = sorted_df[group_col].to_numpy()
    starts = _group_starts(group_vals)
    sizes = np.diff(np.concatenate((starts, np.array([group_vals.size]))))

    if k is None:
        sums = np.add.reduceat(targets, starts)
        daily = sums / sizes
    else:
        positions = _positions_within_group(group_vals)
        masked = np.where(positions < k, targets, 0.0)
        sums = np.add.reduceat(masked, starts)
        daily = sums / np.minimum(sizes, k)

    years = _extract_year(pd.Series(group_vals[starts]))
    return daily, years


def _aggregate_metrics(daily_returns: np.ndarray) -> dict[str, float]:
    """일자별 수익률 시계열로 전체 기간 성과 지표를 계산합니다."""
    returns = daily_returns[np.isfinite(daily_returns)]
    n = returns.size
    if n == 0:
        return {key: float("nan") for key in _BASE_METRIC_KEYS}

    profits = returns[returns > 0.0]
    loss_mag = -returns[returns < 0.0]
    top_1_return = float(np.mean(returns))
    total_profit = float(np.sum(profits))
    total_loss = float(np.sum(loss_mag))
    if n > 1:
        std = float(np.std(returns, ddof=1))
        sharpe = float(top_1_return / std * _DAILY_ANNUALIZATION) if std > 0.0 else float("nan")
    else:
        sharpe = float("nan")

    return {
        "top_1_return": top_1_return,
        "win_rate": float(profits.size / n),
        "profit_factor": float("inf") if total_loss == 0.0 else total_profit / total_loss,
        "mean_win": float(np.mean(profits)) if profits.size else float("nan"),
        "mean_loss": float(np.mean(loss_mag)) if loss_mag.size else float("nan"),
        "sharpe": sharpe,
    }


def _max_drawdown(daily_returns: np.ndarray) -> float:
    """일자별 수익률 시계열의 최대 드로다운(양수)을 반환합니다."""
    returns = daily_returns[np.isfinite(daily_returns)]
    if returns.size == 0:
        return float("nan")
    cumulative = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(-drawdown.min())


def _turnover(
    oof: pd.DataFrame, group_col: str, score_col: str, ascending: bool, k: int
) -> float:
    """일자별 Top-k 선택 종목의 전일 대비 신규 편입 비율 평균을 반환합니다."""
    sorted_df = oof.sort_values([group_col, score_col], ascending=[True, ascending], kind="mergesort")
    prev: set[Any] | None = None
    turns: list[float] = []
    for _, group in sorted_df.groupby(group_col, sort=False):
        top = group.index[:k]
        if prev is not None:
            keep = len(set(top) & prev)
            turns.append(1.0 - keep / min(k, len(top)))
        prev = set(top)
    return float(np.mean(turns)) if turns else float("nan")


def _capital_weighted_return(
    oof: pd.DataFrame, group_col: str, target_col: str, score_col: str, ascending: bool
) -> float:
    """그룹 내 역순위 가중(상위 점수 종목에 더 큰 가중) 자본가중 일일 수익률 평균."""
    sorted_df = oof.sort_values([group_col, score_col], ascending=[True, ascending], kind="mergesort")
    daily: list[float] = []
    for _, group in sorted_df.groupby(group_col, sort=False):
        n = len(group)
        if n == 0:
            continue
        weights = np.arange(n, 0, -1, dtype=np.float64)
        weights = weights / weights.sum()
        daily.append(float(np.dot(weights, group[target_col].to_numpy(dtype=np.float64))))
    return float(np.mean(daily)) if daily else float("nan")


def _stratified_breakdown(
    oof: pd.DataFrame,
    group_col: str,
    target_col: str,
    score_col: str,
    ascending: bool,
    strat_col: str,
) -> dict[str, dict[str, float]]:
    """일자별 Top-1 선택 종목의 국면(strat_col)별 지표 분해를 반환합니다."""
    sorted_df = oof.sort_values([group_col, score_col], ascending=[True, ascending], kind="mergesort")
    top1 = sorted_df.groupby(group_col, sort=False).head(1)
    top1 = top1.dropna(subset=[strat_col])
    breakdown: dict[str, dict[str, float]] = {}
    for strat, group in top1.groupby(strat_col, sort=False):
        breakdown[str(strat)] = _aggregate_metrics(
            group[target_col].to_numpy(dtype=np.float64)
        )
    return breakdown


def _extended_metrics(
    oof: pd.DataFrame,
    group_col: str,
    target_col: str,
    score_col: str,
    ascending: bool,
    k: int | None,
) -> dict[str, Any]:
    """단일 contender 에 대한 확장 성과 지표(비용/턴오버/드로다운/국면 분해)를 반환합니다.

    ``k=None`` 은 동일가중(그룹 전체 평균) contender 로, 턴오버는 정의되지
    않으므로 NaN 을 반환합니다.
    """
    daily, _ = _group_series(oof, group_col, target_col, score_col, ascending, k)
    if k is None:
        daily3 = daily
        metrics = _aggregate_metrics(daily)
        metrics["top_3_return"] = metrics["top_1_return"]
        metrics["capital_weighted_return"] = metrics["top_1_return"]
        metrics["turnover"] = float("nan")
    else:
        daily3, _ = _group_series(oof, group_col, target_col, score_col, ascending, 3)
        metrics = _aggregate_metrics(daily)
        metrics["top_3_return"] = float(np.mean(daily3)) if daily3.size else float("nan")
        metrics["capital_weighted_return"] = _capital_weighted_return(
            oof, group_col, target_col, score_col, ascending
        )
        metrics["turnover"] = _turnover(oof, group_col, score_col, ascending, k)
    # target_col is decimal net return; transaction cost was deducted during target construction.
    metrics["cost_adjusted_return"] = metrics["top_1_return"]
    metrics["date_weighted_return"] = metrics["top_1_return"]
    metrics["max_drawdown"] = _max_drawdown(daily)
    return metrics


def _yearly_breakdown(
    daily: np.ndarray,
    daily3: np.ndarray,
    years: np.ndarray,
) -> dict[int, dict[str, float] | None]:
    """연도별 지표 분해. 표본(거래일) 수 5개 미만 연도는 null 처리합니다."""
    valid = np.isfinite(daily) & np.isfinite(daily3) & np.isfinite(years)
    yv = years[valid]
    top1 = daily[valid]
    top3 = daily3[valid]

    breakdown: dict[int, dict[str, float] | None] = {}
    for year in np.unique(yv):
        mask = yv == year
        if mask.sum() < _MIN_YEAR_SAMPLES:
            breakdown[int(year)] = None
            continue
        metrics = _aggregate_metrics(top1[mask])
        breakdown[int(year)] = {
            "top1_return": metrics["top_1_return"],
            "top3_return": float(np.mean(top3[mask])),
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "sharpe": metrics["sharpe"],
        }
    return breakdown


def run_backtest_evaluation(
    oof_df: pd.DataFrame,
    target_col: str,
    group_col: str,
) -> dict[str, Any]:
    """OOF 예측 vs Baseline 비교 백테스트를 실행합니다.

    Args:
        oof_df: 컬럼 `group_col`, `target_col`, `pred`, `selection_rank`,
            [선택] `pred_linear` 를 포함해야 합니다.
        target_col: 일자별 실현 순수익률(decimal net) 컬럼.
        group_col: 거래일 그룹 컬럼.

    Returns:
        dict with keys:
          'model_metrics'    : AI 모델 지표 (비용/턴오버/드로다운/국면 분해 포함)
          'baseline_metrics' : selection_rank / equal_weight / [linear] Baseline 지표
          'yearly_breakdown' : 연도별 Top-1 Return, Win Rate, Profit Factor, Sharpe
    """
    required = [group_col, target_col, "pred", "selection_rank"]
    missing = [col for col in required if col not in oof_df.columns]
    if missing:
        raise ValueError(f"missing required columns in oof_df: {missing}")

    work = oof_df[required].copy()
    if "pred_linear" in oof_df.columns:
        work["pred_linear"] = oof_df["pred_linear"].to_numpy()
    if "market_type" in oof_df.columns:
        work["market_type"] = oof_df["market_type"].astype(str)
    if "market_cap_100m" in oof_df.columns:
        work["market_cap_100m"] = oof_df["market_cap_100m"].to_numpy(dtype=np.float64)
        work["market_cap_tercile"] = pd.qcut(
            work["market_cap_100m"], q=3, labels=["low", "mid", "high"], duplicates="drop"
        ).astype(str)

    work = work.dropna(subset=[group_col, target_col, "pred", "selection_rank"])
    if work.empty:
        raise ValueError("oof_df has no usable rows after NaN filtering")

    model_metrics = _extended_metrics(work, group_col, target_col, "pred", False, 1)
    model_daily, model_daily3, model_years = _model_daily_series(work, group_col, target_col)

    ew_metrics = _extended_metrics(work, group_col, target_col, "pred", False, None)

    baseline_metrics: dict[str, Any] = {}
    sr_metrics = _extended_metrics(work, group_col, target_col, "selection_rank", True, 1)
    baseline_metrics["selection_rank"] = sr_metrics
    baseline_metrics["equal_weight"] = ew_metrics

    if "pred_linear" in work.columns:
        linear_metrics = _extended_metrics(work, group_col, target_col, "pred_linear", False, 1)
        baseline_metrics["linear"] = linear_metrics

    return {
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "yearly_breakdown": _yearly_breakdown(model_daily, model_daily3, model_years),
        "regime_breakdown": {
            "market_type": _stratified_breakdown(
                work, group_col, target_col, "pred", False, "market_type"
            )
            if "market_type" in work.columns
            else {},
            "market_cap": _stratified_breakdown(
                work, group_col, target_col, "pred", False, "market_cap_tercile"
            )
            if "market_cap_tercile" in work.columns
            else {},
        },
    }


def _model_daily_series(
    work: pd.DataFrame, group_col: str, target_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LGBM OOF 예측(pred) 정렬 기준 Top-1/Top-3 일별 수익률 시계열과 연도를 반환합니다."""
    daily, years = _group_series(work, group_col, target_col, "pred", ascending=False, k=1)
    daily3, _ = _group_series(work, group_col, target_col, "pred", ascending=False, k=3)
    return daily, daily3, years
