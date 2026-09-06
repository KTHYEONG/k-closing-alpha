"""Passive execution simulation: bar-touch upper-bound fills with adverse selection."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.execution.cost_model import krx_tick_size

__all__ = [
    "INCUMBENT_CROSSING_PROFILE",
    "ExecutionProfile",
    "measure_execution_profile",
    "simulate_passive_entry",
    "simulate_passive_exit",
]

_VALID_MODES: frozenset[str] = frozenset({"cross", "touch", "passive"})


@dataclass(frozen=True)
class ExecutionProfile:
    entry_mode: str
    exit_mode: str
    entry_offset_ticks: int = 0
    exit_offset_ticks: int = 0

    def __post_init__(self) -> None:
        if self.entry_mode not in _VALID_MODES:
            raise ValueError(f"entry_mode must be one of {sorted(_VALID_MODES)}, got {self.entry_mode!r}")
        if self.exit_mode not in _VALID_MODES:
            raise ValueError(f"exit_mode must be one of {sorted(_VALID_MODES)}, got {self.exit_mode!r}")
        if int(self.entry_offset_ticks) < 0 or int(self.exit_offset_ticks) < 0:
            raise ValueError(
                f"offset_ticks must be >= 0, got {(self.entry_offset_ticks, self.exit_offset_ticks)!r}"
            )

    @property
    def round_trip_ticks(self) -> float:
        entry = 1.0 if self.entry_mode == "cross" else 0.0
        exit_ = 1.0 if self.exit_mode == "cross" else 0.0
        return float(entry + exit_)


INCUMBENT_CROSSING_PROFILE: ExecutionProfile = ExecutionProfile(entry_mode="cross", exit_mode="cross")


def _limit_from_close(close: np.ndarray, offset_ticks: int, sign: float) -> tuple[np.ndarray, np.ndarray]:
    tick = krx_tick_size(np.asarray(close, dtype=np.float64))
    limit = np.full(close.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(close) & np.isfinite(tick) & (close > 0.0)
    limit[ok] = close[ok] + float(sign) * float(int(offset_ticks)) * tick[ok]
    return limit, tick


def simulate_passive_entry(
    entries: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    offset_ticks: int,
    window_start_hms: int = 151900,
    window_end_hms: int = 153000,
    price_col: str = "close_price",
    code_col: str = "symbol",
) -> pd.DataFrame:
    """Post a buy limit below the close; bar-touch fill is an upper bound."""
    if int(offset_ticks) < 0:
        raise ValueError(f"offset_ticks must be >= 0, got {offset_ticks!r}")
    if price_col not in entries.columns or code_col not in entries.columns:
        raise ValueError(f"entries is missing price_col/code_col {(price_col, code_col)}")
    out = entries.copy()
    raw_col = next((c for c in ("raw_close_price", "raw_close", "unadjusted_close", "raw_price") if c in out.columns), None)
    ref_col = raw_col if raw_col is not None else price_col
    close = pd.to_numeric(out[ref_col], errors="coerce").to_numpy(dtype=np.float64)
    limit, _ = _limit_from_close(close, int(offset_ticks), sign=-1.0)
    wb = bars.copy()
    if code_col not in wb.columns or "ts_hms" not in wb.columns or "low" not in wb.columns:
        raise ValueError("bars is missing symbol/ts_hms/low columns")
    ts = pd.to_numeric(wb["ts_hms"], errors="coerce").to_numpy(dtype=np.float64)
    wb = wb.assign(_ts=np.asarray(ts, dtype=np.float64))
    wb = wb[(wb["_ts"] >= float(window_start_hms)) & (wb["_ts"] <= float(window_end_hms))]
    low = pd.to_numeric(wb["low"], errors="coerce").to_numpy(dtype=np.float64)
    wb = wb.assign(_low=np.asarray(low, dtype=np.float64))
    min_low: dict[str, float] = {}
    if len(wb):
        for key, g in wb.groupby(code_col, sort=False):
            vals = g["_low"].to_numpy(dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                min_low[str(key)] = float(np.min(finite))
    codes = out[code_col].astype(str).to_numpy()
    filled = np.zeros(len(out), dtype=bool)
    for i, code in enumerate(codes):
        ml = min_low.get(str(code), np.nan)
        if np.isfinite(limit[i]) and np.isfinite(ml) and np.isfinite(close[i]) and close[i] > 0.0 and ml <= limit[i]:
            filled[i] = True
    entry_price = np.full(len(out), np.nan, dtype=np.float64)
    entry_price[filled] = limit[filled]
    saving = np.full(len(out), np.nan, dtype=np.float64)
    ok = filled & np.isfinite(close) & (close > 0.0)
    saving[ok] = (close[ok] - entry_price[ok]) / close[ok] * 1e4
    out["entry_filled"] = np.asarray(filled, dtype=bool)
    out["entry_price"] = np.asarray(entry_price, dtype=np.float64)
    out["entry_saving_bp"] = np.asarray(saving, dtype=np.float64)
    with contextlib.suppress(AttributeError, ValueError):
        out.attrs["tick_from_adjusted_price"] = bool(raw_col is None)
    return out


def simulate_passive_exit(
    exits: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    offset_ticks: int,
    window_start_hms: int = 90000,
    window_end_hms: int = 93000,
    price_col: str = "next_open",
    code_col: str = "symbol",
) -> pd.DataFrame:
    """Post a sell limit above next open; fall back to last close when untouched."""
    if int(offset_ticks) < 0:
        raise ValueError(f"offset_ticks must be >= 0, got {offset_ticks!r}")
    if price_col not in exits.columns or code_col not in exits.columns:
        raise ValueError(f"exits is missing price_col/code_col {(price_col, code_col)}")
    out = exits.copy()
    base = pd.to_numeric(out[price_col], errors="coerce").to_numpy(dtype=np.float64)
    limit, _ = _limit_from_close(base, int(offset_ticks), sign=1.0)
    wb = bars.copy()
    if code_col not in wb.columns or "ts_hms" not in wb.columns or "high" not in wb.columns or "close" not in wb.columns:
        raise ValueError("bars is missing symbol/ts_hms/high/close columns")
    ts = pd.to_numeric(wb["ts_hms"], errors="coerce").to_numpy(dtype=np.float64)
    wb = wb.assign(_ts=np.asarray(ts, dtype=np.float64))
    wb = wb[(wb["_ts"] >= float(window_start_hms)) & (wb["_ts"] <= float(window_end_hms))]
    wb = wb.sort_values(["symbol", "_ts"] if "symbol" in wb.columns else "_ts", kind="stable")
    max_high: dict[str, float] = {}
    last_close: dict[str, float] = {}
    if len(wb):
        high = pd.to_numeric(wb["high"], errors="coerce").to_numpy(dtype=np.float64)
        close_b = pd.to_numeric(wb["close"], errors="coerce").to_numpy(dtype=np.float64)
        wb = wb.assign(_high=np.asarray(high, dtype=np.float64), _close=np.asarray(close_b, dtype=np.float64))
        for key, g in wb.groupby(code_col, sort=False):
            hv = g["_high"].to_numpy(dtype=np.float64)
            hv = hv[np.isfinite(hv)]
            if hv.size:
                max_high[str(key)] = float(np.max(hv))
            cv = g["_close"].to_numpy(dtype=np.float64)
            cv = cv[np.isfinite(cv)]
            if cv.size:
                last_close[str(key)] = float(cv[-1])
    codes = out[code_col].astype(str).to_numpy()
    filled = np.zeros(len(out), dtype=bool)
    exit_price = np.full(len(out), np.nan, dtype=np.float64)
    for i, code in enumerate(codes):
        mh = max_high.get(str(code), np.nan)
        if np.isfinite(limit[i]) and np.isfinite(mh) and mh >= limit[i]:
            filled[i] = True
            exit_price[i] = limit[i]
        else:
            fb = last_close.get(str(code), np.nan)
            if np.isfinite(fb):
                exit_price[i] = fb
            elif np.isfinite(base[i]):
                exit_price[i] = base[i]
    saving = np.full(len(out), np.nan, dtype=np.float64)
    ok = np.isfinite(exit_price) & np.isfinite(base) & (base > 0.0)
    saving[ok] = (exit_price[ok] - base[ok]) / base[ok] * 1e4
    out["exit_filled"] = np.asarray(filled, dtype=bool)
    out["exit_price"] = np.asarray(exit_price, dtype=np.float64)
    out["exit_saving_bp"] = np.asarray(saving, dtype=np.float64)
    return out


def measure_execution_profile(
    panel: pd.DataFrame,
    *,
    gross_col: str = "mechanical_gross",
    filled_col: str = "entry_filled",
    saving_col: str = "entry_saving_bp",
) -> dict[str, Any]:
    """Report fill rate, passive saving, and adverse selection jointly."""
    if gross_col not in panel.columns or filled_col not in panel.columns or saving_col not in panel.columns:
        raise ValueError(f"panel is missing gross/filled/saving columns {(gross_col, filled_col, saving_col)}")
    gross = pd.to_numeric(panel[gross_col], errors="coerce").to_numpy(dtype=np.float64)
    filled = panel[filled_col].to_numpy(dtype=bool)
    saving = pd.to_numeric(panel[saving_col], errors="coerce").to_numpy(dtype=np.float64)
    n_rows = len(panel)
    n_measured = int(np.sum(filled))
    fill_rate = float(np.mean(filled)) if n_rows else 0.0
    finite_saving = saving[np.isfinite(saving)]
    mean_saving = float(np.mean(finite_saving)) if finite_saving.size else float("nan")
    pool_vals = gross[np.isfinite(gross)]
    pool_gross_bp = float(np.mean(pool_vals) * 1e4) if pool_vals.size else float("nan")
    filled_gross = gross[filled]
    filled_gross = filled_gross[np.isfinite(filled_gross)]
    filled_gross_bp = float(np.mean(filled_gross) * 1e4) if filled_gross.size else float("nan")
    if np.isfinite(filled_gross_bp) and np.isfinite(pool_gross_bp):
        adverse = float(filled_gross_bp - pool_gross_bp)
    else:
        adverse = float("nan")
    survives = bool(np.isfinite(mean_saving) and np.isfinite(adverse) and (mean_saving + adverse) > 0.0)
    return {
        "n_rows": int(n_rows),
        "n_measured": int(n_measured),
        "fill_rate": float(fill_rate),
        "mean_saving_bp": float(mean_saving),
        "filled_gross_bp": float(filled_gross_bp),
        "pool_gross_bp": float(pool_gross_bp),
        "adverse_selection_bp": float(adverse),
        "saving_survives_adverse_selection": bool(survives),
        "fill_rate_is_upper_bound": True,
    }
