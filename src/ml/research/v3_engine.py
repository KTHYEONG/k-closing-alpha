"""Core research and walk-forward validation engine for Research Validation v3.

Enforces zero lookahead leakage, strict calendar-aware walk-forward validation,
point-in-time universe selection, realistic cost models, and discrete portfolio NAV.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.panel_integrity import prepare_price_panel
from src.strategy.contract import (
    AA_COST,
    DEFAULT_UNIVERSE,
    PA_COST,
    UniverseSpec,
    round_trip_cost_bp,
    select_universe,
)

logger = logging.getLogger("research_v3_engine")

FEATURE_COLS: list[str] = [
    "chg_ratio",
    "log_tv",
    "log_mc",
    "body_ratio",
    "upper_shadow_ratio",
    "intraday_range",
    "inst_density",
    "foreign_density",
    "kospi_pct",
    "kosdaq_pct",
    "v_kospi",
    "tv_rank",
    "inst_rank",
    "chg_rank",
]

# 실측 왕복비용 상단 시나리오(bp). 라벨이 아니라 스트레스 시나리오 전용이므로 PIT 스케줄과 독립이다.
STRESS_COST_BP: float = 46.0


def load_and_prepare_price_history(path: Path | str) -> tuple[pd.DataFrame, np.ndarray, dict[pd.Timestamp, int]]:
    """Load price history and construct trading calendar index."""
    logger.info("Loading price history from %s...", path)
    ph, prov = prepare_price_panel(pd.read_parquet(path))
    logger.info("[DATA] stage=panel_integrity %s", prov.to_log_kv())

    # Baseline trading calendar
    market_dates = np.array(sorted(ph["date"].unique()))
    d_to_idx = {d: i for i, d in enumerate(market_dates)}

    return ph, market_dates, d_to_idx


def build_candidate_universe(
    ph: pd.DataFrame, spec: UniverseSpec = DEFAULT_UNIVERSE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct U0_PIT and U3_PIT universes without future tradability filters.

    Args:
        ph: Prepared price-history panel.
        spec: Universe screen. Defaults to DEFAULT_UNIVERSE; pass COST_AWARE_UNIVERSE
            to add the per-tick cost cap (requires a ``tick_cost_bp`` column).
    """
    logger.info("Filtering PIT candidate universes U0 and U3...")

    # U0_PIT mask: 2% <= chg < 10%, tv >= 100억, mc >= 500억, not ceiling, close > 0, vol > 0
    u0_mask = select_universe(ph, spec)

    u0_df = ph[u0_mask].copy().sort_values(["date", "symbol"]).reset_index(drop=True)

    # U3_PIT mask: U0 + mc >= 5000억 + inst_netbuy > 0 + market index > 0
    is_kosdaq = u0_df["market"].astype(str).str.upper().str.contains("KOSDAQ")
    mkt_idx_pos = np.where(is_kosdaq, u0_df["kosdaq_pct"] > 0, u0_df["kospi_pct"] > 0)
    u3_mask = (u0_df["mc_clean"] >= 5000.0) & (u0_df["inst_netbuy"].fillna(0) > 0) & mkt_idx_pos

    u0_df["is_u3"] = u3_mask
    return u0_df, ph


def attach_forward_exit_paths(
    cands: pd.DataFrame,
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
) -> pd.DataFrame:
    """Attach forward exit prices with suspension tracking and carry rule."""
    logger.info("Attaching calendar-aware forward exit paths...")
    lookup = ph.set_index(["date", "symbol"])[["open", "high", "low", "close", "volume"]]
    n_market_dates = len(market_dates)

    date_indices = np.array([d_to_idx[d] for d in cands["date"]])
    valid_d1 = date_indices + 1 < n_market_dates
    d1_dates = np.where(valid_d1, market_dates[np.minimum(date_indices + 1, n_market_dates - 1)], pd.NaT)
    syms = cands["symbol"].to_numpy()

    # D1 lookup
    keys_d1 = list(zip(d1_dates, syms, strict=False))
    idx_tuples_d1 = pd.MultiIndex.from_tuples(keys_d1, names=["date", "symbol"])
    joined_d1 = lookup.reindex(idx_tuples_d1)

    j_open_d1 = joined_d1["open"].to_numpy(dtype=np.float64)
    j_vol_d1 = joined_d1["volume"].to_numpy(dtype=np.float64)
    tradable_d1 = valid_d1 & np.isfinite(j_open_d1) & (j_open_d1 > 0.0) & (j_vol_d1 > 0.0)

    exit_prices = np.full(len(cands), np.nan, dtype=np.float64)
    exit_status = np.full(len(cands), "EXIT_TRADABLE", dtype=object)
    holding_days = np.ones(len(cands), dtype=np.int32)

    # Direct tradable on D1
    exit_prices[tradable_d1] = j_open_d1[tradable_d1]

    # Handle suspended D1 candidates: search up to 20 days ahead
    suspended_indices = np.where(~tradable_d1 & valid_d1)[0]
    logger.info("Resolving %d suspended/untradable candidates...", len(suspended_indices))

    for idx in suspended_indices:
        sym = syms[idx]
        cur_d_idx = date_indices[idx]
        found = False
        for step in range(2, 21):
            if cur_d_idx + step >= n_market_dates:
                break
            nxt_date = market_dates[cur_d_idx + step]
            try:
                row = lookup.loc[(nxt_date, sym)]
                op = float(row["open"]) if not isinstance(row, pd.DataFrame) else float(row.iloc[0]["open"])
                vl = float(row["volume"]) if not isinstance(row, pd.DataFrame) else float(row.iloc[0]["volume"])
                if np.isfinite(op) and op > 0.0 and vl > 0.0:
                    exit_prices[idx] = op
                    holding_days[idx] = step
                    exit_status[idx] = "EXIT_SUSPENDED"
                    found = True
                    break
            except KeyError:
                continue

        if not found:
            # If never resumes in 20 days, mark as unresolved exit
            exit_prices[idx] = cands.iloc[idx]["close"] * 0.50  # Conservative haircut
            holding_days[idx] = 20
            exit_status[idx] = "UNRESOLVED_EXIT"

    cands["d1_tradable"] = tradable_d1
    cands["exit_price"] = exit_prices
    cands["exit_status"] = exit_status
    cands["holding_days"] = holding_days

    # Calculate returns and costs
    entry_p = cands["close"].to_numpy(dtype=np.float64)
    if "market" not in cands.columns:
        raise ValueError("attach_forward_exit_paths requires a 'market' column for point-in-time tick costing")
    trade_dates = pd.to_datetime(cands["date"]).to_numpy()
    markets = cands["market"].astype(str).to_numpy(dtype=object)
    cost_aa_bp = round_trip_cost_bp(entry_p, trade_dates, markets, AA_COST)
    cost_pa_bp = round_trip_cost_bp(entry_p, trade_dates, markets, PA_COST)
    cost_stress_bp = np.full(len(cands), STRESS_COST_BP, dtype=np.float64)

    gross_ret = exit_prices / entry_p - 1.0
    cands["gross_return"] = gross_ret
    cands["cost_aa_bp"] = cost_aa_bp
    cands["cost_pa_bp"] = cost_pa_bp
    cands["cost_stress_bp"] = cost_stress_bp

    cands["net_return_aa"] = gross_ret - cost_aa_bp / 10000.0
    cands["net_return_pa"] = gross_ret - cost_pa_bp / 10000.0
    cands["net_return_stress"] = gross_ret - cost_stress_bp / 10000.0

    return cands


def compute_derived_features(cands: pd.DataFrame) -> pd.DataFrame:
    """Compute 14 decision-time features strictly using decision candidate set."""
    logger.info("Computing derived decision-time features and cross-sectional ranks...")
    p_close = cands["close"].to_numpy(dtype=np.float64)
    p_open = cands["open"].to_numpy(dtype=np.float64)
    p_high = cands["high"].to_numpy(dtype=np.float64)
    p_low = cands["low"].to_numpy(dtype=np.float64)
    p_vol = cands["volume"].to_numpy(dtype=np.float64)

    rg = np.maximum(p_high - p_low, 1.0)
    cands["body_ratio"] = (p_close - p_open) / rg
    cands["upper_shadow_ratio"] = (p_high - np.maximum(p_open, p_close)) / rg
    cands["intraday_range"] = (p_high - p_low) / p_close
    cands["log_tv"] = np.log1p(np.maximum(cands["tv_clean"].to_numpy(dtype=np.float64), 0.0))
    cands["log_mc"] = np.log1p(np.maximum(cands["mc_clean"].to_numpy(dtype=np.float64), 0.0))

    val_krw = np.maximum(p_close * p_vol, 1.0)
    cands["inst_density"] = np.clip(cands["inst_netbuy"].fillna(0).to_numpy(dtype=np.float64) / val_krw, -1.0, 1.0)
    cands["foreign_density"] = np.clip(cands["foreign_netbuy"].fillna(0).to_numpy(dtype=np.float64) / val_krw, -1.0, 1.0)

    # Cross-sectional ranks across ALL valid candidates on date T
    grouped_date = cands.groupby("date", sort=False)
    cands["tv_rank"] = grouped_date["tv_clean"].rank(pct=True)
    cands["inst_rank"] = grouped_date["inst_netbuy"].rank(pct=True)
    cands["chg_rank"] = grouped_date["chg_ratio"].rank(pct=True)

    return cands


