"""Probe script to test candidate generation without future tradability filtering and inspect suspensions."""
from pathlib import Path
import pandas as pd
import numpy as np

def probe_candidates():
    ph = pd.read_parquet("data/history/price_history.parquet")
    ph["date"] = pd.to_datetime(ph["date"])
    ph["symbol"] = ph["symbol"].astype(str).str.zfill(6)
    ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Clean chg
    chg = ph["daily_change_pct"].to_numpy(dtype=np.float64)
    if np.nanmedian(np.abs(chg[np.isfinite(chg)])) > 1.0:
        chg = chg / 100.0
    ph["chg_ratio"] = chg

    # Clean tv
    tv = ph["trade_value_100m"].to_numpy(dtype=np.float64)
    vol = ph["volume"].to_numpy(dtype=np.float64)
    close = ph["close"].to_numpy(dtype=np.float64)
    high = ph["high"].to_numpy(dtype=np.float64)
    tv_clean = np.where(np.isfinite(tv), tv, close * vol / 1e8)
    ph["tv_clean"] = tv_clean

    # Clean mc without fillna(500)
    # ffill per symbol only, if still NaN then NaN
    ph["mc_clean"] = ph.groupby("symbol")["market_cap_100m"].ffill()

    ceiling = (ph["chg_ratio"] >= 0.29) & (close >= high)
    
    # PIT candidate filter:
    # 2% <= chg < 10%, tv >= 100, mc >= 500 (NaN mc is excluded fail-closed), ~ceiling, close > 0, vol > 0
    u0_mask = (
        (ph["chg_ratio"] >= 0.02)
        & (ph["chg_ratio"] < 0.10)
        & (ph["tv_clean"] >= 100.0)
        & (ph["mc_clean"] >= 500.0)
        & (~ceiling)
        & (close > 0.0)
        & (vol > 0)
    )
    
    cands = ph[u0_mask].copy()
    print(f"Total U0 candidates across history: {len(cands)}")
    print(f"Days with U0 candidates: {cands['date'].nunique()}")
    print(f"Candidates per day: {len(cands) / cands['date'].nunique():.1f}")

    # Check trading dates
    market_dates = np.array(sorted(ph["date"].unique()))
    d_to_idx = {d: i for i, d in enumerate(market_dates)}
    lookup = ph.set_index(["date", "symbol"])

    # Check next-day tradability of all U0 candidates
    date_indices = np.array([d_to_idx[d] for d in cands["date"]])
    valid_d1 = date_indices + 1 < len(market_dates)
    d1_dates = np.where(valid_d1, market_dates[np.minimum(date_indices + 1, len(market_dates) - 1)], pd.NaT)
    syms = cands["symbol"].to_numpy()
    
    keys = list(zip(d1_dates, syms, strict=False))
    idx_tuples = pd.MultiIndex.from_tuples(keys, names=["date", "symbol"])
    joined = lookup.reindex(idx_tuples)
    
    j_open = joined["open"].to_numpy(dtype=np.float64)
    j_vol = joined["volume"].to_numpy(dtype=np.float64)
    tradable = valid_d1 & np.isfinite(j_open) & (j_open > 0.0) & (j_vol > 0.0)
    
    cands["d1_tradable"] = tradable
    n_untradable = np.sum(~tradable & valid_d1)
    print(f"D1 untradable candidates: {n_untradable} / {np.sum(valid_d1)} ({n_untradable / np.sum(valid_d1)*100:.3f}%)")

if __name__ == "__main__":
    probe_candidates()
