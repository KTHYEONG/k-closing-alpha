"""Cost-aware top-k model-free regime-gated backtest runner."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet
from src.execution.cost_model import TICK_REFORM_DATE
from src.ml.research.v3_engine import (
    attach_forward_exit_paths,
    build_candidate_universe,
    load_and_prepare_price_history,
)
from src.ml.research.v3_metrics import calculate_series_metrics
from src.strategy.contract import KCA_TOPK_COSTAWARE_001, StrategySpec

logger = logging.getLogger(__name__)

COST_STRESS_TICK_GRID: tuple[float, ...] = (2.0, 3.0, 4.0)
MIN_FEASIBLE_DAY_FRACTION: float = 0.90
MIN_TOP_K: int = 3
MIN_POST_REFORM_T_STAT: float = 2.0


@dataclass(frozen=True)
class RegimeMetrics:
    """Per-regime daily net-return statistics."""

    regime: str
    n_calendar_days: int
    n_days_with_signal: int
    n_days_feasible: int
    feasible_day_fraction: float
    coverage_status: str
    mean_net_bp: float
    median_net_bp: float
    std_net_bp: float
    win_rate: float
    t_stat: float
    sharpe: float
    ci_low_bp: float
    ci_high_bp: float
    dsr: float


@dataclass(frozen=True)
class CostStressPoint:
    """Pass/fail of one round-trip-tick stress level."""

    regime: str
    round_trip_ticks: float
    n_days: int
    mean_net_bp: float
    median_net_bp: float
    t_stat: float
    passes: bool


@dataclass(frozen=True)
class CostAwareTopKReport:
    """Self-describing backtest artifact payload."""

    strategy_id: str
    top_k: int
    universe: dict[str, Any]
    cost: dict[str, Any]
    date_min: str
    date_max: str
    regimes: dict[str, RegimeMetrics]
    cost_stress: list[CostStressPoint]
    verdict: str
    verdict_reasons: list[str]


def select_topk_by_tick_cost(
    cands: pd.DataFrame,
    k: int,
    *,
    cost_col: str = "tick_cost_bp",
    date_col: str = "date",
) -> pd.DataFrame:
    """Select the k cheapest candidates per date by ascending cost.

    Args:
        cands: Candidate pool with date and cost columns.
        k: Number of names to keep per date.
        cost_col: Per-tick cost column name.
        date_col: Date column name.

    Returns:
        Top-k picks per date in ascending cost order.
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    if cost_col not in cands.columns:
        raise ValueError(f"cands missing cost_col {cost_col!r}")
    if date_col not in cands.columns:
        raise ValueError(f"cands missing date_col {date_col!r}")
    ranked = cands.sort_values([date_col, cost_col], ascending=[True, True], kind="stable")
    return ranked.groupby(date_col, sort=False).head(int(k)).reset_index(drop=True)


def compute_net_return(
    picks: pd.DataFrame,
    *,
    round_trip_ticks: float,
    statutory_bp: float = 20.0,
    gross_col: str = "gross_return",
    cost_col: str = "tick_cost_bp",
) -> np.ndarray:
    """Compute D+1-open net return from gross return and PIT tick cost.

    Args:
        picks: Picked candidates with gross and tick-cost columns.
        round_trip_ticks: Round-trip tick multiplier.
        statutory_bp: Statutory cost in bp.
        gross_col: Gross return column name.
        cost_col: Per-tick cost column name.

    Returns:
        Net return array with NaN where inputs are not finite.
    """
    if gross_col not in picks.columns:
        raise ValueError(f"picks missing gross_col {gross_col!r}")
    if cost_col not in picks.columns:
        raise ValueError(f"picks missing cost_col {cost_col!r}")
    gross = picks[gross_col].to_numpy(dtype=np.float64)
    tick = picks[cost_col].to_numpy(dtype=np.float64)
    net = np.full(gross.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(gross) & np.isfinite(tick)
    net[ok] = gross[ok] - (float(statutory_bp) + float(round_trip_ticks) * tick[ok]) / 1e4
    return net


def daily_mean_series(
    picks: pd.DataFrame,
    values: np.ndarray,
    *,
    date_col: str = "date",
) -> pd.Series:
    """Average per-pick values to one daily series.

    Args:
        picks: Picked candidates carrying the date column.
        values: Per-pick values aligned to picks rows.
        date_col: Date column name.

    Returns:
        Daily mean series indexed by a sorted DatetimeIndex.
    """
    if date_col not in picks.columns:
        raise ValueError(f"picks missing date_col {date_col!r}")
    if len(values) != len(picks):
        raise ValueError(f"values length {len(values)} != picks length {len(picks)}")
    aligned = pd.Series(np.asarray(values, dtype=np.float64), index=picks.index)
    dates = pd.to_datetime(picks[date_col])
    grouped = aligned.groupby(dates.to_numpy()).mean()
    grouped.index = pd.DatetimeIndex(pd.to_datetime(grouped.index))
    return grouped.sort_index()


def split_regime_masks(dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Split dates at the tick-reform boundary.

    Args:
        dates: Dates to partition.

    Returns:
        Boolean masks for pre_reform, post_reform and full_history.
    """
    stamps = dates.to_numpy()
    pre = stamps < np.datetime64(TICK_REFORM_DATE)
    post = stamps >= np.datetime64(TICK_REFORM_DATE)
    full = np.ones(len(dates), dtype=bool)
    return {"pre_reform": pre, "post_reform": post, "full_history": full}


def day_level_feasibility(
    cands: pd.DataFrame,
    k: int,
    market_dates: np.ndarray,
    *,
    date_col: str = "date",
) -> pd.Series:
    """Flag calendar days with at least k screen-passing candidates.

    Args:
        cands: Candidate pool with a date column.
        k: Feasibility threshold per day.
        market_dates: Full trading calendar (the denominator).
        date_col: Date column name.

    Returns:
        Boolean series indexed by the full market calendar.
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    if date_col not in cands.columns:
        raise ValueError(f"cands missing date_col {date_col!r}")
    calendar = pd.DatetimeIndex(pd.to_datetime(pd.Series(market_dates)).sort_values().unique())
    counts = cands.groupby(date_col).size()
    aligned = counts.reindex(calendar).fillna(0)
    return pd.Series(aligned.to_numpy() >= int(k), index=calendar)


def compute_regime_metrics(
    daily_net: pd.Series,
    feasibility_full: pd.Series,
    *,
    min_feasible_fraction: float = MIN_FEASIBLE_DAY_FRACTION,
) -> dict[str, RegimeMetrics]:
    """Compute per-regime coverage and net-return statistics.

    Args:
        daily_net: Daily mean net returns on signal days.
        feasibility_full: Day-level feasibility over the full calendar.
        min_feasible_fraction: Coverage gate threshold.

    Returns:
        RegimeMetrics for pre_reform, post_reform and full_history.
    """
    if not isinstance(daily_net.index, pd.DatetimeIndex):
        raise ValueError("daily_net index must be a pd.DatetimeIndex")
    if not isinstance(feasibility_full.index, pd.DatetimeIndex):
        raise ValueError("feasibility_full index must be a pd.DatetimeIndex")
    daily_masks = split_regime_masks(pd.DatetimeIndex(daily_net.index))
    feas_masks = split_regime_masks(pd.DatetimeIndex(feasibility_full.index))
    out: dict[str, RegimeMetrics] = {}
    for regime in ("pre_reform", "post_reform", "full_history"):
        daily_slice = daily_net[daily_masks[regime]]
        feas_slice = feasibility_full[feas_masks[regime]]
        n_calendar_days = len(feas_slice)
        n_days_feasible = int(feas_slice.to_numpy(dtype=bool).sum())
        feasible_day_fraction = float(n_days_feasible / n_calendar_days) if n_calendar_days else float("nan")
        coverage_status = "OK" if feasible_day_fraction >= float(min_feasible_fraction) else "INSUFFICIENT_COVERAGE"
        stats = calculate_series_metrics(daily_slice.to_numpy(dtype=np.float64), cost_ratio=0.0)
        out[regime] = RegimeMetrics(
            regime=regime,
            n_calendar_days=n_calendar_days,
            n_days_with_signal=int(stats["n"]),
            n_days_feasible=n_days_feasible,
            feasible_day_fraction=float(feasible_day_fraction),
            coverage_status=coverage_status,
            mean_net_bp=float(stats["mean_net_bp"]),
            median_net_bp=float(stats["median_net_bp"]),
            std_net_bp=float(stats["std_net_bp"]),
            win_rate=float(stats["win_rate"]),
            t_stat=float(stats["t_stat"]),
            sharpe=float(stats["sharpe"]),
            ci_low_bp=float(stats["ci_low_bp"]),
            ci_high_bp=float(stats["ci_high_bp"]),
            dsr=float(stats["dsr"]),
        )
    return out


def compute_cost_stress(
    picks: pd.DataFrame,
    *,
    statutory_bp: float = 20.0,
    ticks_grid: tuple[float, ...] = COST_STRESS_TICK_GRID,
    regime: str = "post_reform",
) -> list[CostStressPoint]:
    """Stress the round-trip-tick assumption over one regime.

    Args:
        picks: Picked candidates with date, gross and tick-cost columns.
        statutory_bp: Statutory cost in bp.
        ticks_grid: Round-trip tick multipliers to evaluate.
        regime: Regime slice to stress.

    Returns:
        One CostStressPoint per tick level.
    """
    if regime not in ("pre_reform", "post_reform", "full_history"):
        raise ValueError(f"regime must be one of pre_reform/post_reform/full_history, got {regime!r}")
    masks = split_regime_masks(pd.DatetimeIndex(pd.to_datetime(picks["date"])))
    sub = picks[masks[regime]]
    points: list[CostStressPoint] = []
    for ticks in ticks_grid:
        net = compute_net_return(sub, round_trip_ticks=float(ticks), statutory_bp=float(statutory_bp))
        daily = daily_mean_series(sub, net)
        stats = calculate_series_metrics(daily.to_numpy(dtype=np.float64), cost_ratio=0.0)
        passes = bool(stats["mean_net_bp"] > 0.0 and stats["median_net_bp"] > 0.0)
        points.append(
            CostStressPoint(
                regime=regime,
                round_trip_ticks=float(ticks),
                n_days=int(stats["n"]),
                mean_net_bp=float(stats["mean_net_bp"]),
                median_net_bp=float(stats["median_net_bp"]),
                t_stat=float(stats["t_stat"]),
                passes=passes,
            )
        )
    return points


def evaluate_verdict(
    regimes: dict[str, RegimeMetrics],
    cost_stress: list[CostStressPoint],
) -> tuple[str, list[str]]:
    """Decide PASS/FAIL/INSUFFICIENT_COVERAGE from regime gates.

    Args:
        regimes: Per-regime metrics keyed by regime name.
        cost_stress: Round-trip-tick stress points.

    Returns:
        Verdict string and the human-readable reasons.
    """
    reasons: list[str] = []
    post = regimes["post_reform"]
    full = regimes["full_history"]
    pre = regimes.get("pre_reform")
    if pre is not None and pre.coverage_status == "INSUFFICIENT_COVERAGE":
        reasons.append(
            f"pre_reform not gate-eligible: coverage={pre.coverage_status} "
            f"feasible_day_fraction={pre.feasible_day_fraction:.3f}"
        )
    if (full.mean_net_bp > 0.0) != (post.mean_net_bp > 0.0):
        reasons.append("regime break between pre_reform and post_reform: do not pool regimes")
    else:
        reasons.append("full_history carried for transparency only: never pools regimes for certification")
    if post.coverage_status != "OK":
        reasons.append(
            f"post_reform INSUFFICIENT_COVERAGE: feasible_day_fraction={post.feasible_day_fraction} "
            "below gate; statistics not evaluated"
        )
        return ("INSUFFICIENT_COVERAGE", reasons)
    stress_at_3 = next(
        (p for p in cost_stress if p.regime == "post_reform" and math.isclose(p.round_trip_ticks, 3.0)),
        None,
    )
    checks = {
        "post_reform_median_positive": post.median_net_bp > 0.0,
        "post_reform_t_stat": post.t_stat >= MIN_POST_REFORM_T_STAT,
        "post_reform_3tick_stress": stress_at_3 is not None and stress_at_3.passes,
    }
    for name, ok in checks.items():
        if not ok:
            reasons.append(f"failed gate: {name}")
    if all(checks.values()):
        return ("PASS_POST_REFORM", reasons)
    return ("FAIL", reasons)


def run_cost_aware_topk_backtest(
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
    *,
    spec: StrategySpec = KCA_TOPK_COSTAWARE_001,
) -> CostAwareTopKReport:
    """Run the model-free cost-aware top-k backtest over the full panel.

    Args:
        ph: Prepared price-history panel.
        market_dates: Full trading calendar.
        d_to_idx: Date-to-index lookup for forward exits.
        spec: Strategy specification carrying top_k, universe and cost.

    Returns:
        Assembled cost-aware top-k report with regime verdict.
    """
    if int(spec.top_k) < MIN_TOP_K:
        raise ValueError(f"top_k {spec.top_k} below the minimum investable K {MIN_TOP_K}")
    cands, _ = build_candidate_universe(ph, spec.universe)
    cands = attach_forward_exit_paths(cands, ph, market_dates, d_to_idx)
    picks = select_topk_by_tick_cost(cands, int(spec.top_k))
    net = compute_net_return(
        picks,
        round_trip_ticks=float(spec.cost.round_trip_ticks),
        statutory_bp=float(spec.cost.statutory_bp),
    )
    daily_net = daily_mean_series(picks, net)
    # Entry-eligible calendar: the terminal panel date has no D+1 bar, so no
    # entry signal can ever exist there; it is not part of the denominator.
    calendar_all = pd.DatetimeIndex(pd.to_datetime(pd.Series(market_dates)).sort_values().unique())
    feasibility_full = day_level_feasibility(cands, int(spec.top_k), calendar_all[:-1].to_numpy())
    regimes = compute_regime_metrics(daily_net, feasibility_full)
    cost_stress = compute_cost_stress(picks, statutory_bp=float(spec.cost.statutory_bp), regime="post_reform")
    verdict, verdict_reasons = evaluate_verdict(regimes, cost_stress)
    dates_all = pd.to_datetime(ph["date"])
    date_min = str(dates_all.min().date()) if len(dates_all) else ""
    date_max = str(dates_all.max().date()) if len(dates_all) else ""
    return CostAwareTopKReport(
        strategy_id=spec.strategy_id,
        top_k=int(spec.top_k),
        universe=dataclasses.asdict(spec.universe),
        cost=dataclasses.asdict(spec.cost),
        date_min=date_min,
        date_max=date_max,
        regimes=regimes,
        cost_stress=cost_stress,
        verdict=verdict,
        verdict_reasons=list(verdict_reasons),
    )


def report_to_frame(report: CostAwareTopKReport) -> pd.DataFrame:
    """Flatten a report into a self-contained parquet frame.

    Args:
        report: Cost-aware top-k report.

    Returns:
        One row per regime plus one row per cost-stress point.
    """
    rows: list[dict[str, Any]] = []
    for metrics in report.regimes.values():
        row: dict[str, Any] = {"row_type": "regime"}
        row.update(dataclasses.asdict(metrics))
        row["strategy_id"] = report.strategy_id
        row["top_k"] = report.top_k
        row["verdict"] = report.verdict
        rows.append(row)
    for point in report.cost_stress:
        srow: dict[str, Any] = {"row_type": "cost_stress"}
        srow.update(dataclasses.asdict(point))
        srow["strategy_id"] = report.strategy_id
        srow["top_k"] = report.top_k
        srow["verdict"] = report.verdict
        rows.append(srow)
    if not rows:
        return pd.DataFrame(
            columns=["row_type", "strategy_id", "top_k", "verdict"],
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """Run the standalone cost-aware top-k backtest CLI.

    Args:
        argv: Optional argument list for testing.
    """
    import dataclasses as _dataclasses

    parser = argparse.ArgumentParser(description="Cost-aware top-k regime-gated backtest")
    parser.add_argument("--price-history", default=str(settings.PRICE_HISTORY_PARQUET_PATH))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--out", default="artifacts/research/costaware_topk_report.parquet")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.top_k is not None:
        spec = _dataclasses.replace(KCA_TOPK_COSTAWARE_001, top_k=int(args.top_k))
    else:
        spec = KCA_TOPK_COSTAWARE_001
    if not os.path.exists(args.price_history):
        raise ValueError(f"price_history not found: {args.price_history}")
    ph, market_dates, d_to_idx = load_and_prepare_price_history(args.price_history)
    report = run_cost_aware_topk_backtest(ph, market_dates, d_to_idx, spec=spec)
    atomic_write_parquet(report_to_frame(report), Path(args.out))
    logger.info("[EVAL] stage=costaware_topk verdict=%s reasons=%s", report.verdict, report.verdict_reasons)


if __name__ == "__main__":  # pragma: no cover
    main()
