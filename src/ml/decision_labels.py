"""Decision-grade executable labels: cost + mechanical return (decision-time only)."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.execution.cost_model import estimate_round_trip_cost_bp
from src.ml.exit_policy import attach_next_day_path
from src.serving.realtime.inference import ROUND_TRIP_COST_RATIO

DECISION_LABEL_COLUMNS: frozenset[str] = frozenset(
    {
        "entry_close",
        "entry_change_ratio",
        "nd_open",
        "nd_high",
        "nd_low",
        "nd_close",
        "nd_date",
        "mechanical_gross",
        "has_mechanical_label",
        "label_price_ratio",
        "cost_ratio",
        "cost_measured",
        "eval_net_journaled",
        "eval_net_mechanical",
    }
)


def assert_no_label_leakage(feature_cols: Sequence[str]) -> None:
    """Raise ValueError naming every offending column if any label col leaks."""
    offenders = [c for c in feature_cols if c in DECISION_LABEL_COLUMNS]
    if offenders:
        raise ValueError(f"label leakage into feature_cols: {sorted(offenders)}")


def attach_per_row_cost_ratio(
    df: pd.DataFrame,
    *,
    price_col: str = "close_price",
    impact_col: str | None = None,
    fallback_ratio: float = ROUND_TRIP_COST_RATIO,
) -> pd.DataFrame:
    """Attach per-row cost_ratio with fail-open fallback (never NaN)."""
    out = df.copy()
    out = out.drop(columns=[c for c in ("cost_ratio", "cost_measured") if c in out.columns])
    costed = estimate_round_trip_cost_bp(
        out, price_col=price_col, impact_col=impact_col
    )
    total_bp = pd.to_numeric(
        costed["round_trip_cost_bp"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    finite = np.isfinite(total_bp)
    ratio = np.full(len(out), float(fallback_ratio), dtype=np.float64)
    ratio[finite] = total_bp[finite] / 1e4
    out["cost_ratio"] = np.asarray(ratio, dtype=np.float64)
    out["cost_measured"] = np.asarray(finite, dtype=bool)
    return out


def attach_mechanical_return(
    df: pd.DataFrame,
    price_history_df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    code_col: str = "stock_code",
) -> pd.DataFrame:
    """Attach next-day path + mechanical gross over entry close (both legs adjusted)."""
    out = attach_next_day_path(
        df, price_history_df, date_col=date_col, code_col=code_col
    )
    entry = pd.to_numeric(out["entry_close"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    nd_open = pd.to_numeric(out["nd_open"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    ok = np.isfinite(entry) & np.isfinite(nd_open) & (entry > 0.0)
    gross = np.full(len(out), np.nan, dtype=np.float64)
    gross[ok] = nd_open[ok] / entry[ok] - 1.0
    out["mechanical_gross"] = np.asarray(gross, dtype=np.float64)
    out["has_mechanical_label"] = np.asarray(ok, dtype=bool)
    close = pd.to_numeric(out["close_price"], errors="coerce").to_numpy(dtype=np.float64) if "close_price" in out.columns else np.full(len(out), np.nan)
    ratio_ok = np.isfinite(close) & np.isfinite(entry) & (entry > 0.0)
    ratio = np.full(len(out), np.nan, dtype=np.float64)
    ratio[ratio_ok] = close[ratio_ok] / entry[ratio_ok]
    out["label_price_ratio"] = np.asarray(ratio, dtype=np.float64)
    return out


def build_decision_labels(
    processed: pd.DataFrame,
    price_history_df: pd.DataFrame | None,
    *,
    label_mode: str,
    cost_mode: str,
    clip_lower: float,
    clip_upper: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compose per-row cost + executable labels with fail-closed downgrade guard."""
    if label_mode not in ("journaled", "mechanical"):
        raise ValueError(f"label_mode must be one of journaled/mechanical, got {label_mode!r}")
    if cost_mode not in ("flat", "per_row"):
        raise ValueError(f"cost_mode must be one of flat/per_row, got {cost_mode!r}")
    if label_mode == "mechanical" and price_history_df is None:
        raise ValueError("price_history is required for label_mode='mechanical'")
    n_input = len(processed)
    out = processed.copy()
    out = out.drop(columns=[c for c in DECISION_LABEL_COLUMNS if c in out.columns])
    if cost_mode == "flat":
        out["cost_ratio"] = np.full(len(out), float(ROUND_TRIP_COST_RATIO), dtype=np.float64)
        out["cost_measured"] = np.full(len(out), False, dtype=bool)
    else:
        out = attach_per_row_cost_ratio(out)
    gross_decimal = pd.to_numeric(out["net_return"], errors="coerce").to_numpy(dtype=np.float64)
    cost = pd.to_numeric(out["cost_ratio"], errors="coerce").to_numpy(dtype=np.float64)
    out["eval_net_journaled"] = np.asarray(gross_decimal / 100.0 - cost, dtype=np.float64)
    mechanical_coverage: float = float("nan")
    n_dropped = 0
    if price_history_df is not None:
        out = attach_mechanical_return(out, price_history_df)
        mech = pd.to_numeric(out["mechanical_gross"], errors="coerce").to_numpy(dtype=np.float64)
        out["eval_net_mechanical"] = np.asarray(mech - cost, dtype=np.float64)
        has = out["has_mechanical_label"].to_numpy(dtype=bool) if "has_mechanical_label" in out.columns else np.isfinite(mech)
        mechanical_coverage = float(np.mean(has)) if len(has) else 0.0
    elif label_mode == "journaled":
        mechanical_coverage = float("nan")
    if label_mode == "mechanical":
        has = out["has_mechanical_label"].to_numpy(dtype=bool)
        n_dropped = int(np.sum(~has))
        out = out.loc[has].copy().reset_index(drop=True)
        mechanical_coverage = float(len(out) / n_input) if n_input else 0.0
    eval_col = "eval_net_mechanical" if label_mode == "mechanical" else "eval_net_journaled"
    eval_vals = pd.to_numeric(out[eval_col], errors="coerce").to_numpy(dtype=np.float64)
    out["target_return"] = np.asarray(
        pd.Series(eval_vals).clip(clip_lower, clip_upper).to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    provenance: dict[str, Any] = {
        "label_mode": label_mode,
        "cost_mode": cost_mode,
        "n_rows": len(out),
        "n_input_rows": len(out) + n_dropped,
        "n_dropped_no_mechanical": n_dropped,
        "mechanical_coverage": float(mechanical_coverage),
        "clip_lower": float(clip_lower),
        "clip_upper": float(clip_upper),
        "eval_col": eval_col,
    }
    return out, provenance
