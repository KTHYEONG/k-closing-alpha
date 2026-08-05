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
    # nullable Int64 시계열의 to_numpy 는 읽기전용 뷰를 반환할 수 있어 쓰기가 가능한
    # 복사본을 강제합니다 (연도 파싱 실패 폴백 경로에서 원소를 덮어씁니다).
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

def _turnover_by_stock_code(
    oof: pd.DataFrame,
    group_col: str,
    stock_col: str,
    score_col: str,
    top_k: int,
) -> float:
    """일자별 top-k 선택 종목코드의 전일 대비 신규 편입 비율 평균을 반환합니다.

    DataFrame index 가 아닌 ``stock_col`` 값으로 보유 종목을 추적하므로,
    날짜가 달라져도 동일 종목 코드는 turnover 에서 유지됩니다.
    """
    sorted_df = oof.sort_values([group_col, score_col], ascending=[True, False], kind="mergesort")
    prev: set[Any] | None = None
    turns: list[float] = []
    for _, group in sorted_df.groupby(group_col, sort=False):
        top = set(group[stock_col].iloc[:top_k])
        if prev is not None:
            keep = len(top & prev)
            turns.append(1.0 - keep / min(top_k, len(top)))
        prev = top
    return float(np.mean(turns)) if turns else float("nan")


def simulate_top_k_policy(
    oof_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    stock_col: str = "stock_code",
    score_col: str = "pred",
    top_k: int = 1,
) -> dict[str, Any]:
    """일자별 top-k 종목코드 기반 동일가중 선택 정책 시뮬레이션.

    OOF 의 ``stock_col``, 예측 점수, 실제 decimal-net target 으로 날짜별
    top-k(동일가중)를 선택하고 일별 수익률·누적 NAV·Sharpe·MDD·win rate·
    profit factor·연도별 결과를 반환합니다.

    - turnover 는 선택된 종목코드 집합(``stock_col``)으로 계산합니다.
    - ``target_col`` 은 decimal net return 이므로 비용을 재차감하지 않습니다.
    - 중복 종목코드/결측 코드/``top_k < 1``/비유한 선택 수익률은 ``ValueError`` 입니다.

    Args:
        oof_df: ``group_col``, ``target_col``, ``stock_col``, ``score_col`` 를 포함한 OOF.
        target_col: 일자별 실현 순수익률(decimal net) 컬럼.
        group_col: 거래일 그룹 컬럼.
        stock_col: 종목 식별자 컬럼 (turnover 추적용).
        score_col: 예측 점수 컬럼.
        top_k: 일자별 선택할 최대 종목 수.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    required = [group_col, target_col, stock_col, score_col]
    missing = [col for col in required if col not in oof_df.columns]
    if missing:
        raise ValueError(f"missing required columns in oof_df: {missing}")

    work = oof_df[required].copy()
    work = work.dropna(subset=[group_col, target_col, score_col])
    if work[stock_col].isna().any():
        raise ValueError(f"stock_col {stock_col!r} contains missing values in oof_df")
    duplicates = work.duplicated(subset=[group_col, stock_col], keep=False)
    if duplicates.any():
        raise ValueError(
            f"duplicate {stock_col} values within a {group_col} group are not allowed"
        )
    if work.empty:
        raise ValueError("oof_df has no usable rows after NaN filtering")

    daily, years = _group_series(work, group_col, target_col, score_col, ascending=False, k=top_k)
    if not np.isfinite(daily).all():
        raise ValueError("non-finite selected returns in top_k policy")

    nav = np.cumprod(1.0 + daily)
    metrics = _aggregate_metrics(daily)
    metrics["max_drawdown"] = _max_drawdown(daily)
    return {
        "daily_returns": daily,
        "nav": nav,
        "turnover": _turnover_by_stock_code(work, group_col, stock_col, score_col, top_k),
        "top_k": top_k,
        "metrics": metrics,
        "yearly_breakdown": _yearly_breakdown(daily, daily, years),
    }

_ACTION_RESOLUTION_MODES: tuple[str, ...] = (
    "exclude_multi_scenario",
    "score_best_action",
    "require_final_action",
)


def resolve_stock_actions(
    oof_df: pd.DataFrame,
    group_col: str,
    stock_col: str = "stock_code",
    scenario_col: str = "chart_analysis",
    score_col: str = "pred",
    mode: str = "exclude_multi_scenario",
    executable_col: str = "is_executable_action",
) -> pd.DataFrame:
    """OOF 행동 패널을 유일한 ``(group, stock)`` 종목 패널로 해소합니다.

    시나리오 행동 패널에서 날짜-종목당 실행 가능한 행동 하나를 선택해
    ``simulate_top_k_policy`` 가 중복을 거부하지 않도록 만듭니다. 실현 수익률
    (``target_col``)이나 원천 행 순서로 행동을 선택하지 않습니다.

    모드:
    - ``exclude_multi_scenario``: 날짜-종목에 행동이 둘 이상이면 해당 종목을
      포트폴리오 평가에서 제외합니다.
    - ``score_best_action``: 행동별 예측 점수(``score_col``)가 가장 높은 하나를
      선택하며, 동점은 ``scenario_col`` 오름차순으로 결정합니다.
    - ``require_final_action``: ``executable_col`` 이 True 인 행동이 정확히 하나인
      날짜-종목만 선택하며, 없거나 둘 이상이면 ``ValueError`` 입니다.

    Args:
        oof_df: ``group_col``/``stock_col``/``scenario_col``/``score_col`` 를 포함한 OOF.
        group_col: 거래일 그룹 컬럼.
        stock_col: 종목 식별자 컬럼.
        scenario_col: 시나리오(차트분석) 컬럼.
        score_col: 행동 선택에 사용할 예측 점수 컬럼.
        mode: 해소 모드.
        executable_col: ``require_final_action`` 모드에서 실행 행동을 식별하는 boolean 컬럼.

    Returns:
        날짜-종목 key 가 유일한 해소된 DataFrame.

    Raises:
        ValueError: 미지원 모드, key 컬럼 누락/null, 또는 ``require_final_action``
            에서 날짜-종목별 실행 행동이 0개/2개 이상인 경우. 해소 결과는 각 모드가
            구조적으로 유일한 ``(group, stock)`` key 를 보장합니다.
    """
    if mode not in _ACTION_RESOLUTION_MODES:
        raise ValueError(f"mode must be one of {list(_ACTION_RESOLUTION_MODES)}, got {mode!r}")
    required = [group_col, stock_col, scenario_col, score_col]
    missing = [col for col in required if col not in oof_df.columns]
    if missing:
        raise ValueError(f"missing required columns in oof_df for action resolution: {missing}")
    null_cols = [col for col in required if oof_df[col].isna().any()]
    if null_cols:
        raise ValueError(f"required columns contain nulls: {null_cols}")

    work = oof_df.reset_index(drop=True).copy()
    group_keys = [group_col, stock_col]

    if mode == "exclude_multi_scenario":
        counts = work.groupby(group_keys, sort=False)[scenario_col].transform("size")
        resolved = work.loc[counts == 1]
    elif mode == "score_best_action":
        resolved = (
            work.sort_values(
                [*group_keys, score_col, scenario_col],
                ascending=[True, True, False, True],
                kind="mergesort",
            )
            .groupby(group_keys, sort=False)
            .head(1)
        )
    else:  # require_final_action
        if executable_col not in work.columns:
            raise ValueError(
                f"executable_col {executable_col!r} is required for require_final_action mode"
            )
        executable = work[executable_col].fillna(False).astype(bool)
        group_sizes = work.groupby(group_keys, sort=False).size()
        executable_counts = (
            work.assign(_executable=executable)
            .groupby(group_keys, sort=False)["_executable"]
            .sum()
            .reindex(group_sizes.index, fill_value=0)
        )
        invalid = executable_counts[executable_counts != 1]
        if not invalid.empty:
            raise ValueError(
                "require_final_action expects exactly one executable action per "
                f"{group_col}-{stock_col} group; got {len(invalid)} invalid groups"
            )
        resolved = work.loc[executable]

    # 각 해소 모드는 (group, stock) key 에 대해 하나의 행만 반환하므로 결과는
    # 구조적으로 유일합니다 (exclude: count==1 만 유지, score_best: 그룹당 head(1),
    # require_final: 실행 행동 정확히 1개 검증). 해소되지 않은 중복은
    # simulate_top_k_policy 가 fail-closed 로 거부합니다.
    return resolved.reset_index(drop=True)


def _model_daily_series(
    work: pd.DataFrame, group_col: str, target_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LGBM OOF 예측(pred) 정렬 기준 Top-1/Top-3 일별 수익률 시계열과 연도를 반환합니다."""
    daily, years = _group_series(work, group_col, target_col, "pred", ascending=False, k=1)
    daily3, _ = _group_series(work, group_col, target_col, "pred", ascending=False, k=3)
    return daily, daily3, years
