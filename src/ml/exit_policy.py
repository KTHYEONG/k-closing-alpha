"""Research/evaluation only -- no deployed exit behavior."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ml.metrics import aggregate_metrics
from src.ml.robust_eval import CombinatorialPurgedCV, moving_block_bootstrap_delta

DEFAULT_TAKE_PROFIT_GRID: tuple[float, ...] = (0.03, 0.04, 0.05, 0.06, 0.07)
LIMIT_UP_BLOCK_THRESHOLD: float = 0.28
INCUMBENT_EXIT_LABEL: str = "market_next_open"
_REQUIRED_PATH_COLUMNS: frozenset[str] = frozenset({"date", "symbol", "open", "high", "low", "close", "daily_change_pct"})


@dataclass(frozen=True)
class ExitRuleResult:
    take_profit_pct: float
    fallback: str
    n_days: int
    candidate_mean_net: float
    incumbent_mean_net: float
    candidate_sharpe: float
    candidate_win_rate: float
    delta_vs_incumbent: float
    p_value: float
    ci_low: float
    ci_high: float
    cpcv_path_deltas: tuple[float, ...]
    promoted: bool


def attach_next_day_path(
    df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    code_col: str = "stock_code",
) -> pd.DataFrame:
    """Attach entry-day close and next-trading-day OHLC from price history."""
    if date_col not in df.columns or code_col not in df.columns:
        raise ValueError(
            f"df is missing date_col/code_col {(date_col, code_col)}, "
            f"df columns must contain both, got {list(df.columns)!r}"
        )
    missing = [c for c in _REQUIRED_PATH_COLUMNS if c not in price_history_df.columns]
    if missing:
        raise ValueError(
            f"price_history_df is missing required columns {missing}, "
            f"expected columns {_REQUIRED_PATH_COLUMNS!r}"
        )
    ph = price_history_df.copy()
    ph["date"] = pd.to_datetime(ph["date"])
    ph["symbol"] = ph["symbol"].astype(str).str.zfill(6)
    ph = ph.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    change = ph["daily_change_pct"].to_numpy(dtype=np.float64)
    if np.isfinite(change).any() and float(np.nanmedian(np.abs(change[np.isfinite(change)]))) > 1.0:
        ph["daily_change_pct"] = ph["daily_change_pct"].astype(np.float64) / 100.0
    grouped = ph.groupby("symbol", sort=False)
    for col in ("open", "high", "low", "close"):
        ph[f"nd_{col}"] = grouped[col].shift(-1)
    ph["entry_close"] = ph["close"]
    ph["entry_change_ratio"] = ph["daily_change_pct"].astype(np.float64)
    lookup = ph[
        ["symbol", "date", "entry_close", "entry_change_ratio", "nd_open", "nd_high", "nd_low", "nd_close"]
    ]
    out = df.copy()
    out["_merge_date"] = pd.to_datetime(out[date_col])
    out["_merge_symbol"] = out[code_col].astype(str).str.zfill(6)
    merged = out.merge(
        lookup, left_on=["_merge_symbol", "_merge_date"], right_on=["symbol", "date"], how="left", sort=False
    )
    for col in ("entry_close", "entry_change_ratio", "nd_open", "nd_high", "nd_low", "nd_close"):
        out[col] = merged[col].to_numpy(dtype=np.float64)
    return out


def simulate_take_profit_exit(
    entry_close: np.ndarray,
    nd_open: np.ndarray,
    nd_high: np.ndarray,
    nd_close: np.ndarray,
    *,
    take_profit_pct: float,
    fallback: str = "moc",
    fill_probability: float = 1.0,
    fill_haircut: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Simulate a resting take-profit limit with a terminal market fallback."""
    if not 0.0 < take_profit_pct < 0.5:
        raise ValueError(f"take_profit_pct must be in (0, 0.5), got {take_profit_pct!r}")
    if fallback not in ("moc", "next_open"):
        raise ValueError(f"fallback must be one of ('moc', 'next_open'), got {fallback!r}")
    if not 0.0 < fill_probability <= 1.0:
        raise ValueError(f"fill_probability must be in (0, 1.0], got {fill_probability!r}")
    if not 0.0 <= fill_haircut < 0.1:
        raise ValueError(f"fill_haircut must be in [0, 0.1), got {fill_haircut!r}")
    entry = np.asarray(entry_close, dtype=np.float64)
    op = np.asarray(nd_open, dtype=np.float64)
    hi = np.asarray(nd_high, dtype=np.float64)
    cl = np.asarray(nd_close, dtype=np.float64)
    if not (entry.ndim == op.ndim == hi.ndim == cl.ndim == 1 and entry.shape == op.shape == hi.shape == cl.shape):
        raise ValueError(
            f"input arrays must share the same 1-D shape, got {[entry.shape, op.shape, hi.shape, cl.shape]!r}"
        )
    for name, arr in (("entry_close", entry), ("nd_open", op), ("nd_high", hi), ("nd_close", cl)):
        if not np.isfinite(arr).all() or not (arr > 0.0).all():
            raise ValueError(f"{name} must hold only finite positive values, got non-finite or non-positive entries")
    tp = entry * (1.0 + float(take_profit_pct))
    # 돌파 시가는 휴면 지정가가 아닌 관측된 시가로 체결한다 (선행참조 금지).
    gap_fill = op >= tp
    if fill_probability >= 1.0:
        touch_fills = np.ones(entry.shape[0], dtype=bool)
    else:
        touch_fills = np.random.default_rng(seed).random(entry.shape[0]) < fill_probability
    touch_fill = (~gap_fill) & (hi >= tp) & touch_fills
    limit_price = tp * (1.0 - float(fill_haircut))
    fallback_price = cl if fallback == "moc" else op
    exit_price = np.where(gap_fill, op, np.where(touch_fill, limit_price, fallback_price))
    return np.asarray(exit_price / entry - 1.0, dtype=np.float64)


def evaluate_exit_grid(
    oof_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    *,
    group_col: str = "trade_date",
    code_col: str = "stock_code",
    score_col: str = "pred",
    target_col: str = "net_return",
    cost_ratio: float,
    take_profit_grid: tuple[float, ...] = DEFAULT_TAKE_PROFIT_GRID,
    fallback: str = "moc",
    cv: CombinatorialPurgedCV | None = None,
    alpha: float = 0.10,
    fill_probability: float = 1.0,
    fill_haircut: float = 0.0,
    seed: int = 0,
) -> tuple[ExitRuleResult, ...]:
    """Score take-profit exits for the reranker top-1 pick against market-next-open."""
    required = (group_col, code_col, score_col, target_col)
    missing = [c for c in required if c not in oof_df.columns]
    if missing:
        raise ValueError(f"oof_df is missing required columns {missing}, expected {list(required)!r}")
    if cost_ratio < 0:
        raise ValueError(f"cost_ratio must be >= 0, got {cost_ratio!r}")
    grid = tuple(float(v) for v in take_profit_grid)
    if not grid:
        raise ValueError(f"take_profit_grid must be a non-empty strictly increasing tuple in (0, 0.5), got {grid!r}")
    for v in grid:
        if not 0.0 < v < 0.5:
            raise ValueError(f"take_profit_grid values must each be in (0, 0.5), got {v!r}")
    for i in range(1, len(grid)):
        if not grid[i] > grid[i - 1]:
            raise ValueError(f"take_profit_grid must be strictly increasing, got {grid!r}")
    if fallback not in ("moc", "next_open"):
        raise ValueError(f"fallback must be one of ('moc', 'next_open'), got {fallback!r}")
    idx = oof_df.groupby(group_col, sort=True)[score_col].idxmax()
    top1 = oof_df.loc[idx].copy().sort_values(group_col).reset_index(drop=True)
    attached = attach_next_day_path(top1, price_history_df, date_col=group_col, code_col=code_col)
    prices = attached[["entry_close", "nd_open", "nd_high", "nd_low", "nd_close"]].to_numpy(dtype=np.float64)
    finite_positive = np.isfinite(prices).all(axis=1) & (prices > 0.0).all(axis=1)
    change_ratio = attached["entry_change_ratio"].to_numpy(dtype=np.float64)
    # 상한가 잠금 종가는 매수 불가 취급하여 평가에서 제외한다.
    usable = finite_positive & ~(np.abs(change_ratio) >= LIMIT_UP_BLOCK_THRESHOLD)
    kept = attached.loc[usable].copy().reset_index(drop=True)
    n_days = len(kept)
    if n_days < 30:
        raise ValueError(f"evaluation needs >= 30 usable shared days, got {n_days}")
    if cv is not None and int(pd.unique(kept[group_col]).size) < cv.n_groups:
        raise ValueError(
            f"retained distinct days must be >= cv.n_groups={cv.n_groups}, "
            f"got {int(pd.unique(kept[group_col]).size)}"
        )
    entry_close = kept["entry_close"].to_numpy(dtype=np.float64)
    nd_open = kept["nd_open"].to_numpy(dtype=np.float64)
    nd_high = kept["nd_high"].to_numpy(dtype=np.float64)
    nd_close = kept["nd_close"].to_numpy(dtype=np.float64)
    incumbent_net = nd_open / entry_close - 1.0 - float(cost_ratio)
    incumbent_mean_net = float(np.mean(incumbent_net))
    day_groups = kept[group_col].to_numpy()
    results: list[ExitRuleResult] = []
    for tp in grid:
        gross = simulate_take_profit_exit(
            entry_close,
            nd_open,
            nd_high,
            nd_close,
            take_profit_pct=float(tp),
            fallback=fallback,
            fill_probability=fill_probability,
            fill_haircut=fill_haircut,
            seed=seed,
        )
        candidate_net = np.asarray(gross - float(cost_ratio), dtype=np.float64)
        delta = moving_block_bootstrap_delta(candidate_net, incumbent_net)
        if cv is None:
            path_deltas: tuple[float, ...] = ()
        else:
            path_deltas = tuple(
                float(np.mean(candidate_net[test_idx] - incumbent_net[test_idx]))
                for _, test_idx, _ in cv.split(day_groups)
            )
        agg = aggregate_metrics(candidate_net)
        promoted = bool(delta.delta > 0.0 and delta.p_value < alpha and (not path_deltas or min(path_deltas) > 0.0))
        results.append(
            ExitRuleResult(
                take_profit_pct=float(tp),
                fallback=str(fallback),
                n_days=n_days,
                candidate_mean_net=float(np.mean(candidate_net)),
                incumbent_mean_net=incumbent_mean_net,
                candidate_sharpe=float(agg["sharpe"]),
                candidate_win_rate=float(agg["win_rate"]),
                delta_vs_incumbent=float(delta.delta),
                p_value=float(delta.p_value),
                ci_low=float(delta.ci_low),
                ci_high=float(delta.ci_high),
                cpcv_path_deltas=path_deltas,
                promoted=promoted,
            )
        )
    return tuple(results)


def summarize_exit_grid(results: tuple[ExitRuleResult, ...]) -> dict[str, Any]:
    """Format grid results for tuning provenance."""
    if not results:
        raise ValueError(f"results must be a non-empty tuple of ExitRuleResult, got {results!r}")
    grid = []
    for r in results:
        row = dataclasses.asdict(r)
        row["cpcv_path_deltas"] = list(r.cpcv_path_deltas)
        grid.append(row)
    promoted = [r for r in results if r.promoted]
    best = max(promoted, key=lambda r: r.candidate_mean_net) if promoted else None
    best_row = None
    if best is not None:
        best_row = dataclasses.asdict(best)
        best_row["cpcv_path_deltas"] = list(best.cpcv_path_deltas)
    return {
        "incumbent_mean_net": float(results[0].incumbent_mean_net),
        "n_days": int(results[0].n_days),
        "grid": grid,
        "best": best_row,
    }
