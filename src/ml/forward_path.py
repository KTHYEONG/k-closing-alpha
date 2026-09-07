"""Forward path attachment and multi-day evaluation utilities (Research-only).

This module provides leak-free, vectorized forward path attachment from price history,
supporting arbitrary forward horizons (D+1, D+2, D+3, etc.), MFE/MAE calculation,
incremental return decomposition, and multi-day exit simulation.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "attach_forward_path",
    "calculate_horizon_metrics",
    "compute_forward_returns",
    "simulate_multiday_tp_exit",
]

_REQUIRED_PRICE_COLUMNS: frozenset[str] = frozenset(
    {"date", "symbol", "open", "high", "low", "close", "daily_change_pct"}
)


def _normalize_symbol(series: pd.Series) -> pd.Series:
    """Normalize symbol to clean 6-digit string without float decimals."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(6)
    )


def attach_forward_path(
    df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    horizons: Sequence[int] = (1, 2, 3),
    *,
    date_col: str = "trade_date",
    code_col: str = "stock_code",
) -> pd.DataFrame:
    """Attach entry-day close and forward trading-day OHLC from price history.

    Args:
        df: Input dataframe with entry date and symbol/code columns.
        price_history_df: Daily price history bars containing date, symbol, OHLC, daily_change_pct.
        horizons: Sequence of forward trading day offsets to attach (e.g. (1, 2, 3)).
        date_col: Name of entry date column in df (will be resolved if missing).
        code_col: Name of symbol/code column in df (will be resolved if missing).

    Returns:
        DataFrame with entry_close, entry_change_ratio, and for each h in horizons:
        d{h}_date, d{h}_open, d{h}_high, d{h}_low, d{h}_close.
    """
    resolved_date = date_col if date_col in df.columns else next(
        (c for c in ("trade_date", "date", "매수날짜", "date_time") if c in df.columns), None
    )
    resolved_code = code_col if code_col in df.columns else next(
        (c for c in ("stock_code", "symbol", "code", "종목코드") if c in df.columns), None
    )
    if resolved_date is None or resolved_code is None:
        raise ValueError(
            f"df is missing date_col/code_col {(date_col, code_col)}, "
            f"available columns: {list(df.columns)!r}"
        )

    missing = [c for c in _REQUIRED_PRICE_COLUMNS if c not in price_history_df.columns]
    if missing:
        raise ValueError(
            f"price_history_df is missing required columns {missing}, "
            f"expected columns {_REQUIRED_PRICE_COLUMNS!r}"
        )

    clean_horizons = tuple(sorted({int(h) for h in horizons}))
    if not clean_horizons or any(h < 1 for h in clean_horizons):
        raise ValueError(f"horizons must contain positive integers >= 1, got {horizons!r}")

    ph = price_history_df.copy()
    ph["date"] = pd.to_datetime(ph["date"])
    ph["symbol"] = _normalize_symbol(ph["symbol"])
    ph = ph.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")

    chg = pd.to_numeric(ph["daily_change_pct"], errors="coerce").to_numpy(dtype=np.float64)
    finite_chg = chg[np.isfinite(chg)]
    if finite_chg.size > 0 and float(np.nanmedian(np.abs(finite_chg))) > 1.0:
        ph["daily_change_pct"] = chg / 100.0

    ph["entry_close"] = pd.to_numeric(ph["close"], errors="coerce").astype(np.float64)
    ph["entry_change_ratio"] = ph["daily_change_pct"].astype(np.float64)

    grouped = ph.groupby("symbol", sort=False)
    # Market trading calendar
    all_market_dates = np.array(sorted(ph["date"].unique()))
    date_to_idx = {d: i for i, d in enumerate(all_market_dates)}
    ph_date_idx = np.array([date_to_idx[d] for d in ph["date"]])
    n_market_dates = len(all_market_dates)

    has_volume = "volume" in ph.columns
    ph_indexed = ph.set_index(["date", "symbol"])

    lookup_cols = ["symbol", "date", "entry_close", "entry_change_ratio"]
    for h in clean_horizons:
        # 1. Symbol next observed bar (diagnostic / provenance)
        observed_date = grouped["date"].shift(-h)
        ph[f"d{h}_observed_date"] = observed_date
        lookup_cols.append(f"d{h}_observed_date")

        for col in ("open", "high", "low", "close"):
            ph[f"d{h}_observed_{col}"] = pd.to_numeric(grouped[col].shift(-h), errors="coerce").astype(np.float64)
            lookup_cols.append(f"d{h}_observed_{col}")

        # 2. Market exchange calendar next trading date
        valid_market_h = (ph_date_idx + h) < n_market_dates
        market_d_date = np.where(valid_market_h, all_market_dates[np.minimum(ph_date_idx + h, n_market_dates - 1)], pd.NaT)
        ph[f"d{h}_market_date"] = market_d_date
        ph[f"d{h}_date"] = np.where(valid_market_h, market_d_date, pd.NaT)
        lookup_cols.extend([f"d{h}_market_date", f"d{h}_date"])

        # Lookup (market_d_date, symbol) directly in ph_indexed
        keys = list(zip(market_d_date, ph["symbol"], strict=False))
        idx_tuples = pd.MultiIndex.from_tuples(keys, names=["date", "symbol"])
        matched_bar = ph_indexed.reindex(idx_tuples)

        m_open = pd.to_numeric(matched_bar["open"], errors="coerce").to_numpy(dtype=np.float64)
        m_high = pd.to_numeric(matched_bar["high"], errors="coerce").to_numpy(dtype=np.float64)
        m_low = pd.to_numeric(matched_bar["low"], errors="coerce").to_numpy(dtype=np.float64)
        m_close = pd.to_numeric(matched_bar["close"], errors="coerce").to_numpy(dtype=np.float64)

        if has_volume:
            m_vol = pd.to_numeric(matched_bar["volume"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            is_halted = (m_vol == 0.0)
        else:
            is_halted = np.zeros(len(ph), dtype=bool)

        has_bar = valid_market_h & np.isfinite(m_open) & (m_open > 0.0)
        tradable = has_bar & (~is_halted)
        suspended = (~valid_market_h) | (~has_bar) | is_halted

        ph[f"d{h}_open"] = np.where(tradable, m_open, np.nan)
        ph[f"d{h}_high"] = np.where(tradable, m_high, np.nan)
        ph[f"d{h}_low"] = np.where(tradable, m_low, np.nan)
        ph[f"d{h}_close"] = np.where(tradable, m_close, np.nan)
        ph[f"d{h}_suspended"] = np.asarray(suspended, dtype=bool)
        ph[f"d{h}_tradable"] = np.asarray(tradable, dtype=bool)

        lookup_cols.extend([f"d{h}_open", f"d{h}_high", f"d{h}_low", f"d{h}_close", f"d{h}_suspended", f"d{h}_tradable"])

    lookup = ph[lookup_cols]

    out = df.copy()
    cols_to_drop = [c for c in lookup_cols if c not in ("symbol", "date") and c in out.columns]
    cols_to_drop.extend([c for c in ("_merge_date", "_merge_symbol") if c in out.columns])
    if cols_to_drop:
        out = out.drop(columns=cols_to_drop)

    out["_merge_date"] = pd.to_datetime(out[resolved_date])
    out["_merge_symbol"] = _normalize_symbol(out[resolved_code])

    merged = out.merge(
        lookup,
        left_on=["_merge_symbol", "_merge_date"],
        right_on=["symbol", "date"],
        how="left",
        sort=False,
    )

    out["entry_close"] = merged["entry_close"].to_numpy(dtype=np.float64)
    out["entry_change_ratio"] = merged["entry_change_ratio"].to_numpy(dtype=np.float64)
    for h in clean_horizons:
        out[f"d{h}_date"] = pd.to_datetime(merged[f"d{h}_date"])
        out[f"d{h}_market_date"] = pd.to_datetime(merged[f"d{h}_market_date"])
        out[f"d{h}_observed_date"] = pd.to_datetime(merged[f"d{h}_observed_date"])
        out[f"d{h}_suspended"] = merged[f"d{h}_suspended"].to_numpy(dtype=bool)
        out[f"d{h}_tradable"] = merged[f"d{h}_tradable"].to_numpy(dtype=bool)
        for col in ("open", "high", "low", "close"):
            out[f"d{h}_{col}"] = merged[f"d{h}_{col}"].to_numpy(dtype=np.float64)
            out[f"d{h}_observed_{col}"] = merged[f"d{h}_observed_{col}"].to_numpy(dtype=np.float64)

    out = out.drop(columns=["_merge_date", "_merge_symbol"])
    return out


def compute_forward_returns(
    df: pd.DataFrame,
    *,
    cost_ratio: float = 0.0046,
    horizons: Sequence[int] = (1, 2, 3),
) -> pd.DataFrame:
    """Compute gross/net forward returns, MFE, MAE, and incremental returns."""
    out = df.copy()
    entry_close = pd.to_numeric(out["entry_close"], errors="coerce").to_numpy(dtype=np.float64)
    valid_entry = np.isfinite(entry_close) & (entry_close > 0.0)

    clean_horizons = tuple(sorted({int(h) for h in horizons}))

    for h in clean_horizons:
        for price_type in ("open", "close"):
            col_name = f"d{h}_{price_type}"
            if col_name in out.columns:
                p = pd.to_numeric(out[col_name], errors="coerce").to_numpy(dtype=np.float64)
                ok = valid_entry & np.isfinite(p) & (p > 0.0)
                gross = np.full(len(out), np.nan, dtype=np.float64)
                gross[ok] = p[ok] / entry_close[ok] - 1.0
                out[f"d{h}_{price_type}_gross"] = gross
                out[f"d{h}_{price_type}_net"] = gross - float(cost_ratio)

    for h in clean_horizons:
        highs = []
        lows = []
        for i in range(1, h + 1):
            if f"d{i}_high" in out.columns:
                highs.append(pd.to_numeric(out[f"d{i}_high"], errors="coerce").to_numpy(dtype=np.float64))
            if f"d{i}_low" in out.columns:
                lows.append(pd.to_numeric(out[f"d{i}_low"], errors="coerce").to_numpy(dtype=np.float64))

        if highs:
            max_high = np.fmax.reduce(highs)
            ok_hi = valid_entry & np.isfinite(max_high) & (max_high > 0.0)
            mfe = np.full(len(out), np.nan, dtype=np.float64)
            mfe[ok_hi] = max_high[ok_hi] / entry_close[ok_hi] - 1.0
            out[f"d{h}_mfe"] = mfe

        if lows:
            min_low = np.fmin.reduce(lows)
            ok_lo = valid_entry & np.isfinite(min_low) & (min_low > 0.0)
            mae = np.full(len(out), np.nan, dtype=np.float64)
            mae[ok_lo] = min_low[ok_lo] / entry_close[ok_lo] - 1.0
            out[f"d{h}_mae"] = mae

    if "d1_open" in out.columns and "d1_close" in out.columns:
        d1_o = pd.to_numeric(out["d1_open"], errors="coerce").to_numpy(dtype=np.float64)
        d1_c = pd.to_numeric(out["d1_close"], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(d1_o) & np.isfinite(d1_c) & (d1_o > 0.0)
        ret = np.full(len(out), np.nan, dtype=np.float64)
        ret[ok] = d1_c[ok] / d1_o[ok] - 1.0
        out["d1_intraday_return"] = ret

    if "d1_close" in out.columns and "d2_close" in out.columns:
        d1_c = pd.to_numeric(out["d1_close"], errors="coerce").to_numpy(dtype=np.float64)
        d2_c = pd.to_numeric(out["d2_close"], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(d1_c) & np.isfinite(d2_c) & (d1_c > 0.0)
        ret = np.full(len(out), np.nan, dtype=np.float64)
        ret[ok] = d2_c[ok] / d1_c[ok] - 1.0
        out["d1_to_d2_close_return"] = ret

    if "d2_close" in out.columns and "d3_close" in out.columns:
        d2_c = pd.to_numeric(out["d2_close"], errors="coerce").to_numpy(dtype=np.float64)
        d3_c = pd.to_numeric(out["d3_close"], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isfinite(d2_c) & np.isfinite(d3_c) & (d2_c > 0.0)
        ret = np.full(len(out), np.nan, dtype=np.float64)
        ret[ok] = d3_c[ok] / d2_c[ok] - 1.0
        out["d2_to_d3_close_return"] = ret

    return out


def calculate_horizon_metrics(
    returns: np.ndarray,
    *,
    cost_ratio: float = 0.0,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute comprehensive statistical metrics for a return series."""
    arr = np.asarray(returns, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    n = finite.size
    if n == 0:
        nan = float("nan")
        return {
            "n": 0,
            "mean_gross_bp": nan,
            "mean_net_bp": nan,
            "median_bp": nan,
            "std_bp": nan,
            "win_rate": nan,
            "profit_factor": nan,
            "t_stat": nan,
            "sharpe": nan,
            "bootstrap_ci_net_bp": (nan, nan),
            "percentiles_bp": dict.fromkeys((10, 25, 50, 75, 90), nan),
        }

    net = finite - float(cost_ratio)
    mean_gross = float(np.mean(finite))
    mean_net = float(np.mean(net))
    median_net = float(np.median(net))
    std_net = float(np.std(net, ddof=1)) if n >= 2 else 0.0

    wins = net[net > 0.0]
    losses = -net[net < 0.0]
    win_rate = float(wins.size / n)
    profit_factor = float(np.sum(wins) / np.sum(losses)) if np.sum(losses) > 0.0 else float("inf")

    if std_net > 0.0 and n >= 2:
        t_stat = float(mean_net / (std_net / np.sqrt(n)))
        sharpe = float(mean_net / std_net * np.sqrt(252.0))
    else:
        t_stat = float("nan")
        sharpe = float("nan")

    rng = np.random.default_rng(seed)
    boot_idx = rng.choice(n, size=(n_boot, n), replace=True)
    boot_means = np.mean(net[boot_idx], axis=1)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    pct_keys = (10, 25, 50, 75, 90)
    pct_vals = np.percentile(net, pct_keys)
    percentiles_bp = {int(p): float(v * 1e4) for p, v in zip(pct_keys, pct_vals, strict=False)}

    return {
        "n": int(n),
        "mean_gross_bp": float(mean_gross * 1e4),
        "mean_net_bp": float(mean_net * 1e4),
        "median_bp": float(median_net * 1e4),
        "std_bp": float(std_net * 1e4),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "t_stat": float(t_stat),
        "sharpe": float(sharpe),
        "bootstrap_ci_net_bp": (float(ci_low * 1e4), float(ci_high * 1e4)),
        "percentiles_bp": percentiles_bp,
    }


def simulate_multiday_tp_exit(
    df: pd.DataFrame,
    *,
    take_profit_pct: float | None = 0.05,
    max_horizon: int = 3,
    fallback: str = "moc",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Simulate multi-day holding with take-profit limit and terminal fallback."""
    n = len(df)
    entry_close = pd.to_numeric(df["entry_close"], errors="coerce").to_numpy(dtype=np.float64)

    gross_returns = np.full(n, np.nan, dtype=np.float64)
    holding_days = np.full(n, np.nan, dtype=np.float64)
    exit_reasons: list[str] = ["unfilled"] * n

    tp_ratio = float(take_profit_pct) if take_profit_pct is not None else None

    for idx in range(n):
        entry = entry_close[idx]
        if not np.isfinite(entry) or entry <= 0.0:
            continue

        target_price = entry * (1.0 + tp_ratio) if tp_ratio is not None else float("inf")

        for d in range(1, max_horizon + 1):
            o_col = f"d{d}_open"
            h_col = f"d{d}_high"
            c_col = f"d{d}_close"

            if o_col not in df.columns or c_col not in df.columns:
                break

            op = float(df[o_col].iloc[idx])
            hi = float(df[h_col].iloc[idx]) if h_col in df.columns else op
            cl = float(df[c_col].iloc[idx])

            if not np.isfinite(op) or op <= 0.0:
                break

            if tp_ratio is not None:
                if op >= target_price:
                    gross_returns[idx] = op / entry - 1.0
                    holding_days[idx] = float(d)
                    exit_reasons[idx] = f"d{d}_tp_gap"
                    break

                if hi >= target_price:
                    gross_returns[idx] = target_price / entry - 1.0
                    holding_days[idx] = float(d)
                    exit_reasons[idx] = f"d{d}_tp_touch"
                    break

            if d == max_horizon:
                fallback_price = cl if fallback == "moc" else op
                if np.isfinite(fallback_price) and fallback_price > 0.0:
                    gross_returns[idx] = fallback_price / entry - 1.0
                    holding_days[idx] = float(d)
                    exit_reasons[idx] = f"d{d}_fallback_{fallback}"
                break

    return gross_returns, holding_days, exit_reasons
