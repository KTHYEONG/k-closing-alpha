"""Quantitative and financial metrics module for Research Validation v3.

Provides strictly deterministic, fail-closed financial statistics, discrete
portfolio simulation on actual KRX trading calendar, moving-block bootstrap CIs,
and selection-adjusted Deflated Sharpe Ratio (DSR).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm


def moving_block_bootstrap_ci(
    series: np.ndarray,
    *,
    block_size: int = 10,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Moving block bootstrap 95% confidence interval for serially dependent returns."""
    arr = np.asarray(series, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 10:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    actual_block = max(1, min(block_size, n))
    n_blocks = math.ceil(n / actual_block)
    starts = rng.integers(0, max(1, n - actual_block + 1), size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(actual_block)).reshape(n_boot, -1)[:, :n]
    boot_means = arr[idx].mean(axis=1)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return ci_low, ci_high


def deflated_sharpe_ratio(returns: np.ndarray, n_trials: int = 350) -> float:
    """Calculate Bailey-Lopez de Prado Deflated Sharpe Ratio adjusting for multiple testing.

    Args:
        returns: 1D array of strategy trade or daily returns.
        n_trials: Pre-declared multiple testing budget (number of historical experiments).
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    n = len(finite)
    if n < 20 or np.std(finite, ddof=1) <= 0:
        return 0.0
    sr = float(np.mean(finite) / np.std(finite, ddof=1))
    skew = float(stats.skew(finite))
    kurt = float(stats.kurtosis(finite, fisher=False))
    denom = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / (n - 1)
    if denom <= 0:
        return 0.0
    euler = 0.5772156649
    k = max(float(n_trials), 1.0)
    sr0 = math.sqrt(1.0 / (n - 1)) * (
        (1.0 - euler) * norm.ppf(1.0 - 1.0 / k) + euler * norm.ppf(1.0 - 1.0 / (k * math.e))
    )
    return float(norm.cdf((sr - sr0) / math.sqrt(denom)))


def calculate_geometric_cagr(
    initial_nav: float,
    final_nav: float,
    elapsed_calendar_days: int,
) -> float:
    """Calculate true geometric Compound Annual Growth Rate (CAGR)."""
    if initial_nav <= 0 or final_nav <= 0 or elapsed_calendar_days <= 0:
        return 0.0
    years = elapsed_calendar_days / 365.25
    if years < 0.01:
        return 0.0
    return float((final_nav / initial_nav) ** (1.0 / years) - 1.0)


def calculate_series_metrics(
    returns: np.ndarray,
    cost_ratio: float = 0.0,
    n_trials: int = 350,
) -> dict[str, Any]:
    """Calculate comprehensive statistical metrics on trade returns."""
    arr = np.asarray(returns, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    if n == 0:
        nan = float("nan")
        return {
            "n": 0,
            "mean_gross_bp": nan,
            "mean_net_bp": nan,
            "median_net_bp": nan,
            "std_net_bp": nan,
            "win_rate": nan,
            "profit_factor": nan,
            "t_stat": nan,
            "sharpe": nan,
            "sortino": nan,
            "ci_low_bp": nan,
            "ci_high_bp": nan,
            "dsr": nan,
        }
    net = finite - float(cost_ratio)
    mean_gross = float(np.mean(finite)) * 1e4
    mean_net = float(np.mean(net)) * 1e4
    median_net = float(np.median(net)) * 1e4
    std_net = float(np.std(net, ddof=1)) * 1e4 if n >= 2 else 0.0

    wins = net[net > 0.0]
    losses = -net[net < 0.0]
    win_rate = float(wins.size / n)
    pf = float(np.sum(wins) / np.sum(losses)) if np.sum(losses) > 0.0 else float("inf")

    if std_net > 0.0 and n >= 2:
        t_stat = float(mean_net / (std_net / np.sqrt(n)))
        sharpe = float(mean_net / std_net * np.sqrt(252.0))
        neg_net = net[net < 0.0]
        downside_std = float(np.std(neg_net, ddof=1)) * 1e4 if len(neg_net) > 1 else std_net
        sortino = float(mean_net / downside_std * np.sqrt(252.0)) if downside_std > 0 else float("nan")
    else:
        t_stat = float("nan")
        sharpe = float("nan")
        sortino = float("nan")

    ci_low, ci_high = moving_block_bootstrap_ci(net, block_size=10, n_boot=1000)
    dsr_val = deflated_sharpe_ratio(net, n_trials=n_trials)

    return {
        "n": n,
        "mean_gross_bp": round(mean_gross, 2),
        "mean_net_bp": round(mean_net, 2),
        "median_net_bp": round(median_net, 2),
        "std_net_bp": round(std_net, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 3),
        "t_stat": round(t_stat, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "ci_low_bp": round(ci_low * 1e4, 2) if np.isfinite(ci_low) else float("nan"),
        "ci_high_bp": round(ci_high * 1e4, 2) if np.isfinite(ci_high) else float("nan"),
        "dsr": round(dsr_val, 4),
    }


def simulate_discrete_portfolio(
    daily_trades: pd.DataFrame,
    market_dates: list[pd.Timestamp],
    date_col: str = "date",
    ret_col: str = "net_return_aa",
    n_slots: int = 1,
    initial_capital: float = 100_000_000.0,
) -> dict[str, Any]:
    """Simulate daily compounded NAV on the official KRX trading calendar.

    Args:
        daily_trades: DataFrame of executed trades with date and net return.
        market_dates: Ordered list/array of valid KRX trading dates.
        date_col: Date column name.
        ret_col: Net return column name.
        n_slots: Position slots (1 for Top1, 3 for Top3 EW).
        initial_capital: Starting portfolio equity in KRW.
    """
    if len(market_dates) == 0:
        return {}

    # Map daily trades to series
    daily_ret_series = daily_trades.groupby(date_col)[ret_col].mean()
    m_series = pd.Series(0.0, index=pd.to_datetime(market_dates)).sort_index()

    # Reindex on KRX calendar
    aligned = daily_ret_series.reindex(m_series.index).fillna(0.0)

    nav = initial_capital
    nav_history = [nav]
    for r in aligned:
        # Scale return by number of slots
        r_port = r / float(n_slots)
        nav = nav * (1.0 + r_port)
        nav_history.append(nav)

    nav_arr = np.array(nav_history)
    n_days = len(aligned)
    elapsed_calendar_days = (m_series.index[-1] - m_series.index[0]).days if n_days > 1 else 1

    cagr = calculate_geometric_cagr(initial_capital, float(nav_arr[-1]), elapsed_calendar_days)
    daily_port_rets = nav_arr[1:] / nav_arr[:-1] - 1.0
    mean_daily = float(np.mean(daily_port_rets))
    arithmetic_annual = float(mean_daily * 252.0)
    ann_vol = float(np.std(daily_port_rets, ddof=1) * np.sqrt(252.0)) if len(daily_port_rets) > 1 else 0.0
    sharpe = float(arithmetic_annual / ann_vol) if ann_vol > 0 else float("nan")

    cum_max = np.maximum.accumulate(nav_arr)
    drawdowns = (cum_max - nav_arr) / np.maximum(cum_max, 1e-6)
    mdd = float(np.max(drawdowns))
    calmar = float(cagr / mdd) if mdd > 0 else float("nan")

    neg_rets = daily_port_rets[daily_port_rets < 0]
    downside_vol = float(np.std(neg_rets, ddof=1) * np.sqrt(252.0)) if len(neg_rets) > 1 else ann_vol
    sortino = float(arithmetic_annual / downside_vol) if downside_vol > 0 else float("nan")
    cvar95 = float(np.mean(daily_port_rets[daily_port_rets <= np.percentile(daily_port_rets, 5)]) * 1e4)

    worst_day_bp = float(np.min(daily_port_rets) * 1e4) if len(daily_port_rets) else 0.0
    rolling_5d = pd.Series(daily_port_rets).rolling(5).sum().dropna()
    worst_5d_bp = float(rolling_5d.min() * 1e4) if len(rolling_5d) else 0.0

    active_days = int((aligned != 0.0).sum())
    capital_utilization = float(active_days / n_days) if n_days > 0 else 0.0
    no_trade_rate = 1.0 - capital_utilization

    return {
        "initial_capital": initial_capital,
        "final_nav": round(float(nav_arr[-1]), 2),
        "total_return_pct": round(float((nav_arr[-1] / initial_capital - 1.0) * 100), 2),
        "cagr_pct": round(cagr * 100, 2),
        "arithmetic_annual_return_pct": round(arithmetic_annual * 100, 2),
        "annualized_volatility_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "mdd_pct": round(mdd * 100, 2),
        "cvar_95_bp": round(cvar95, 1),
        "worst_day_bp": round(worst_day_bp, 1),
        "worst_5day_bp": round(worst_5d_bp, 1),
        "trading_days": n_days,
        "active_trade_days": active_days,
        "capital_utilization_pct": round(capital_utilization * 100, 1),
        "no_trade_rate_pct": round(no_trade_rate * 100, 1),
    }
