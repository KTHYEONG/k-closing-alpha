"""OOF 백테스트 평가: Baseline 비교 및 연도별 안정성 감사.

LGBMRanker OOF 예측을 2개 Baseline(selection_rank 휴리스틱, 동일가중 무작위)과
비교하고, 연도별 Top-k 수익률 / 승률 / Profit Factor / Sharpe 를 분해합니다.
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
        oof_df: 컬럼 `group_col`, `target_col`, `pred`, [선택] `selection_rank`.
        target_col: 일자별 실현 순수익률 컬럼.
        group_col: 거래일 그룹 컬럼.

    Returns:
        dict with keys:
          'model_metrics'    : 전체 기간 AI 모델 지표
          'baseline_metrics' : 기존 선정순위 및 동일가중 Baseline 지표
          'yearly_breakdown' : 연도별 Top-1 Return, Win Rate, Profit Factor, Sharpe
    """
    missing = [col for col in (group_col, target_col, "pred") if col not in oof_df.columns]
    if missing:
        raise ValueError(f"missing required columns in oof_df: {missing}")

    work = oof_df[[group_col, target_col, "pred"]].copy()
    if "selection_rank" in oof_df.columns:
        work["selection_rank"] = oof_df["selection_rank"].to_numpy()
    work = work.dropna(subset=[group_col, target_col, "pred"])
    if work.empty:
        raise ValueError("oof_df has no usable rows after NaN filtering")

    model_daily, model_daily3, model_years = _model_daily_series(work, group_col, target_col)
    model_metrics = _aggregate_metrics(model_daily)
    model_metrics["top_3_return"] = float(np.mean(model_daily3))

    ew_daily, _ = _group_series(work, group_col, target_col, "pred", ascending=False, k=None)
    ew_metrics = _aggregate_metrics(ew_daily)
    ew_metrics["top_3_return"] = ew_metrics["top_1_return"]

    baseline_metrics: dict[str, Any] = {}
    if "selection_rank" in work.columns:
        sr_daily, sr_daily3 = _selection_rank_daily_series(work, group_col, target_col)
        sr_metrics = _aggregate_metrics(sr_daily)
        sr_metrics["top_3_return"] = float(np.mean(sr_daily3))
        baseline_metrics["selection_rank"] = sr_metrics
    else:
        logger.warning("selection_rank column missing - skipping selection_rank baseline")
        baseline_metrics["selection_rank"] = None
    baseline_metrics["equal_weight"] = ew_metrics

    return {
        "model_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "yearly_breakdown": _yearly_breakdown(model_daily, model_daily3, model_years),
    }


def _model_daily_series(
    work: pd.DataFrame, group_col: str, target_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LGBM OOF 예측(pred) 정렬 기준 Top-1/Top-3 일별 수익률 시계열과 연도를 반환합니다."""
    daily, years = _group_series(work, group_col, target_col, "pred", ascending=False, k=1)
    daily3, _ = _group_series(work, group_col, target_col, "pred", ascending=False, k=3)
    return daily, daily3, years


def _selection_rank_daily_series(
    work: pd.DataFrame, group_col: str, target_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """selection_rank 오름차순(낮을수록 우선) 정렬 기준 Top-1/Top-3 일별 수익률 시계열."""
    daily, _ = _group_series(work, group_col, target_col, "selection_rank", ascending=True, k=1)
    daily3, _ = _group_series(work, group_col, target_col, "selection_rank", ascending=True, k=3)
    return daily, daily3
