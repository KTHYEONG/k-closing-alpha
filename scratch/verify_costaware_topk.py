"""Reproduce the probe's COST_AWARE top-k backtest using shipped production code."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.panel_integrity import load_price_panel
from src.execution.cost_model import TICK_REFORM_DATE
from src.strategy.contract import COST_AWARE_UNIVERSE, DEFAULT_UNIVERSE, select_universe

RNG = np.random.default_rng(42)

ph, prov = load_price_panel("data/history/price_history.parquet")
print(f"[DATA] {prov.to_log_kv()}")

dates = np.array(sorted(ph["date"].unique()))
nxt = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}
lk = ph.set_index(["date", "symbol"])[["open", "volume"]]


def block_ci(x: np.ndarray, block: int = 5, n_boot: int = 2000) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    n = x.size
    if n < block * 3:
        return (np.nan, np.nan)
    n_blocks = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return (float(np.percentile(x[idx].mean(axis=1), 2.5)), float(np.percentile(x[idx].mean(axis=1), 97.5)))


def run(spec, k: int, label: str, post_only: bool = True) -> None:
    s = ph[select_universe(ph, spec)].copy()
    if post_only:
        s = s[s["date"] >= pd.Timestamp(str(TICK_REFORM_DATE))]
    s["d1"] = s["date"].map(nxt)
    j = lk.reindex(pd.MultiIndex.from_arrays([s["d1"], s["symbol"]]))
    s["xo"], s["xv"] = j["open"].to_numpy(np.float64), j["volume"].to_numpy(np.float64)
    s = s[np.isfinite(s["xo"]) & (s["xo"] > 0) & (s["xv"] > 0)].copy()
    if k is not None:
        s = s.sort_values(["date", "tick_cost_bp"], ascending=[True, True]).groupby("date", sort=False).head(k)
    s["gross"] = s["xo"] / s["close"] - 1.0
    s["net"] = s["gross"] - (20.0 + 2.0 * s["tick_cost_bp"]) / 1e4
    daily = s.groupby("date")["net"].mean().to_numpy(np.float64)
    n = daily.size
    if n < 30:
        print(f"[EVAL] {label:32s} days={n} INSUFFICIENT")
        return
    mu, sd = daily.mean(), daily.std(ddof=1)
    lo, hi = block_ci(daily)
    print(f"[EVAL] {label:32s} days={n} net_bp={mu*1e4:7.2f} t={mu/(sd/np.sqrt(n)):5.2f} "
          f"sharpe={mu/sd*np.sqrt(252):5.2f} median_bp={np.median(daily)*1e4:7.2f} "
          f"win={(daily>0).mean():.3f} ci95_bp=[{lo*1e4:6.2f},{hi*1e4:6.2f}]")


print("\n=== post-reform (2023-01-25~), per-row PIT cost, D+1 open exit ===")
run(DEFAULT_UNIVERSE, None, "DEFAULT EW(all)")
run(COST_AWARE_UNIVERSE, None, "COST_AWARE EW(all)")
run(COST_AWARE_UNIVERSE, 1, "COST_AWARE top1 lowest-tick")
run(COST_AWARE_UNIVERSE, 3, "COST_AWARE top3 lowest-tick")
run(COST_AWARE_UNIVERSE, 5, "COST_AWARE top5 lowest-tick")

print("\n=== full history 2016-2026 (regime check) ===")
run(COST_AWARE_UNIVERSE, 3, "COST_AWARE top3 full-history", post_only=False)
