"""Metrics ported (archival) backtest_evaluator + model_pipeline."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_DAILY_ANNUALIZATION = float(np.sqrt(252.0))
_MIN_YEAR_SAMPLES = 5
_BASE_METRIC_KEYS = ("top_1_return", "win_rate", "profit_factor", "mean_win", "mean_loss", "sharpe")
_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})")


def extract_year(groups: pd.Series) -> np.ndarray:
    """그룹 컬럼(거래일) 값에서 연도 배열(float64)을 추출합니다."""
    parsed = pd.to_datetime(groups, errors="coerce", format="mixed")
    years: np.ndarray = np.asarray(
        parsed.dt.year.to_numpy(dtype=np.float64), dtype=np.float64
    ).copy()
    missing = ~np.isfinite(years)
    if missing.any():
        raw = groups.astype(str).to_numpy()
        for idx in np.flatnonzero(missing):
            match = _YEAR_RE.search(raw[idx])
            years[idx] = float(match.group(1)) if match else np.nan
    return years


def aggregate_metrics(daily_returns: np.ndarray) -> dict[str, float]:
    """일자별 수익률 시계열로 전체 기간 성과 지표를 계산합니다."""
    returns = daily_returns[np.isfinite(daily_returns)]
    n = returns.size
    if n == 0:
        return {key: float("nan") for key in _BASE_METRIC_KEYS}  # type: ignore[return-value]
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


def max_drawdown(daily_returns: np.ndarray) -> float:
    """일자별 수익률 시계열의 최대 드로다운(양수)을 반환합니다."""
    returns = daily_returns[np.isfinite(daily_returns)]
    if returns.size == 0:
        return float("nan")
    cumulative = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return float(-drawdown.min())


def group_relevance(target: pd.Series, groups: pd.Series) -> pd.Series:
    """날짜별 Cross-sectional Rank Percentile 을 0~4 등급 relevance 로 변환."""
    pct = target.groupby(groups).rank(pct=True, method="average")
    return (pct * 4.0).round().astype(int)


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """단일 그룹의 예측 순서 기준 NDCG@k."""
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


def rank_ic(oof: pd.DataFrame, group_col: str, target_col: str, score_col: str = "pred") -> float:
    """평균 per-group Spearman(pred,target) 정규화 IC."""
    ics: list[float] = []
    for _, group in oof.groupby(group_col, sort=False):
        if len(group) < 2:
            continue
        if float(np.std(group[score_col].to_numpy())) == 0.0:
            continue
        result = spearmanr(group[score_col], group[target_col])
        ic = float(result.statistic)
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def top_k_return(oof: pd.DataFrame, group_col: str, target_col: str, k: int = 1, score_col: str = "pred") -> float:
    """per-group top-k mean target_return."""
    vals: list[float] = []
    for _, group in oof.groupby(group_col, sort=False):
        order = group[score_col].to_numpy().argsort()[::-1][:k]
        vals.append(float(group[target_col].to_numpy()[order].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def yearly_breakdown(daily: np.ndarray, years: np.ndarray) -> dict[int, dict[str, float] | None]:
    """연도별 분해. 표본 수 5개 미만 연도는 null."""
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
