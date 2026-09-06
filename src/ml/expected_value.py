"""Expected-value decision quantity replacing decision_score."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["expected_net_value", "select_by_expected_value"]


def expected_net_value(
    pred_gap: np.ndarray,
    cost_ratio: np.ndarray,
    *,
    fill_prob: np.ndarray,
    adverse_bp: np.ndarray,
) -> np.ndarray:
    """fill_prob * (pred_gap + adverse_bp/1e4 - cost_ratio)."""
    pred = np.asarray(pred_gap, dtype=np.float64).ravel()
    cost = np.asarray(cost_ratio, dtype=np.float64).ravel()
    fill = np.asarray(fill_prob, dtype=np.float64).ravel()
    adv = np.asarray(adverse_bp, dtype=np.float64).ravel()
    if not (pred.shape == cost.shape == fill.shape == adv.shape):
        raise ValueError(f"shape mismatch: pred{pred.shape} cost{cost.shape} fill{fill.shape} adverse{adv.shape}")
    if fill.size == 0:
        raise ValueError("inputs must be non-empty")
    if not np.all(np.isfinite(fill)) or bool(np.any(fill < 0.0) or np.any(fill > 1.0)):
        raise ValueError(f"fill_prob must lie in [0, 1], got {fill!r}")
    return np.asarray(fill * (pred + adv / 1e4 - cost), dtype=np.float64)


def select_by_expected_value(
    df: pd.DataFrame,
    *,
    group_col: str,
    ev_col: str,
    min_ev: float = 0.0,
    max_positions: int = 1,
) -> pd.DataFrame:
    """Per-group keep rows with EV above floor; abstain when none clear."""
    if group_col not in df.columns or ev_col not in df.columns:
        raise ValueError(f"df is missing group_col/ev_col {(group_col, ev_col)}")
    if int(max_positions) < 1:
        raise ValueError(f"max_positions must be >= 1, got {max_positions!r}")
    floor = float(min_ev)
    if not np.isfinite(floor):
        raise ValueError(f"min_ev must be finite, got {min_ev!r}")
    ev = pd.to_numeric(df[ev_col], errors="coerce").to_numpy(dtype=np.float64)
    work = df.copy()
    # Incomparable objects in df.attrs (e.g. a feature_manifest DataFrame) make
    # pd.concat's __finalize__ raise "ambiguous truth value" on the sliced
    # per-group frames below; clear before slicing (see robust_eval.cpcv_oof_predict).
    work.attrs = {}
    work["_ev"] = np.asarray(ev, dtype=np.float64)
    picks: list[pd.DataFrame] = []
    for _key, g in work.groupby(group_col, sort=True):
        elig = g[np.isfinite(g["_ev"].to_numpy(dtype=np.float64)) & (g["_ev"].to_numpy(dtype=np.float64) > floor)]
        if len(elig) == 0:
            continue
        ordered = elig.sort_values("_ev", ascending=False, kind="stable").head(int(max_positions))
        picks.append(ordered)
    out = pd.concat(picks, ignore_index=True) if picks else work.iloc[0:0].copy()
    out = out.drop(columns=["_ev"], errors="ignore")
    n_days = int(out[group_col].nunique()) if len(out) else 0
    n_input_days = int(work[group_col].nunique()) if len(work) else 0
    out.attrs["n_days"] = int(n_days)
    out.attrs["n_input_days"] = int(n_input_days)
    out.attrs["min_ev"] = float(floor)
    ev_vals = pd.to_numeric(out[ev_col], errors="coerce").to_numpy(dtype=np.float64) if len(out) else np.array([], dtype=np.float64)
    ev_vals = ev_vals[np.isfinite(ev_vals)]
    if ev_vals.size >= 2 and float(np.std(ev_vals, ddof=1)) > 0.0:
        from scipy.stats import norm as _norm

        sd = float(np.std(ev_vals, ddof=1))
        out.attrs["mde"] = float((_norm.ppf(0.975) + _norm.ppf(0.80)) * sd / np.sqrt(float(ev_vals.size)))
    else:
        out.attrs["mde"] = float("nan")
    return out
