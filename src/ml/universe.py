"""Parametrised universe screens with model-free baseline provenance."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.strategy.contract import CEILING_CHG_THRESHOLD, MAX_TICK_COST_BP

__all__ = [
    "COST_AWARE_SCREEN",
    "ScreenConfig",
    "build_universe_panel",
    "screen_baseline_stats",
]


@dataclass(frozen=True)
class ScreenConfig:
    change_lower: float
    change_upper: float | None = None
    min_trade_value_100m: float = 0.0
    min_market_cap_100m: float = 0.0
    exclude_ceiling: bool = True
    require_index_up: bool = False
    max_tick_cost_bp: float | None = None

    def __post_init__(self) -> None:
        lower = float(self.change_lower)
        if self.change_upper is not None:
            upper = float(self.change_upper)
            if not upper > lower:
                raise ValueError(f"change_upper ({upper}) must exceed change_lower ({lower})")
        for name in ("min_trade_value_100m", "min_market_cap_100m"):
            v = float(getattr(self, name))
            if not v >= 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {getattr(self, name)!r}")
        if self.max_tick_cost_bp is not None and not float(self.max_tick_cost_bp) > 0.0:
            raise ValueError(f"max_tick_cost_bp must be > 0 when set, got {self.max_tick_cost_bp!r}")


COST_AWARE_SCREEN: ScreenConfig = ScreenConfig(
    change_lower=0.02,
    change_upper=0.10,
    min_trade_value_100m=100.0,
    min_market_cap_100m=500.0,
    exclude_ceiling=True,
    max_tick_cost_bp=MAX_TICK_COST_BP,
)


def _resolve_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_universe_panel(
    price_history_df: pd.DataFrame,
    screen: ScreenConfig,
    *,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild candidate panel from price_history with ceiling exclusion."""
    date_col = _resolve_col(price_history_df, ("date", "trade_date"))
    symbol_col = _resolve_col(price_history_df, ("symbol", "stock_code", "code"))
    if date_col is None or symbol_col is None:
        raise ValueError(f"price_history is missing date/symbol columns: {list(price_history_df.columns)}")
    for col in ("open", "high", "low", "close", "daily_change_pct"):
        if col not in price_history_df.columns:
            raise ValueError(f"price_history is missing required column {col!r}")
    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        raise ValueError(f"start_date/end_date must be parseable, got {start_date!r}/{end_date!r}")
    work = price_history_df.copy()
    work["_entry_date"] = pd.to_datetime(work[date_col], errors="coerce")
    # Attach next open per symbol with a single groupby shift, then reuse.
    ordered = work.sort_values([symbol_col, "_entry_date"], kind="stable").reset_index(drop=True)
    nxt = pd.to_numeric(ordered["open"], errors="coerce").to_numpy(dtype=np.float64)
    syms = ordered[symbol_col].astype(str).to_numpy()
    next_open = np.full(len(ordered), np.nan, dtype=np.float64)
    # Vectorised per-symbol shift without a second groupby pass over bars.
    same_next = np.empty(len(ordered), dtype=object)
    same_next[:-1] = syms[1:]
    same_next[-1] = None
    mask_next = np.array([a == b for a, b in zip(syms, same_next, strict=False)], dtype=bool)
    shifted = np.full(len(ordered), np.nan, dtype=np.float64)
    shifted[:-1] = nxt[1:]
    next_open[mask_next] = shifted[mask_next]
    ordered["_next_open"] = np.asarray(next_open, dtype=np.float64)
    in_range = (ordered["_entry_date"] >= start) & (ordered["_entry_date"] <= end)
    filt = ordered.loc[in_range].copy().reset_index(drop=True)
    n_raw = len(filt)
    # Coverage provenance before any drop (R10).
    if "market_cap_100m" in filt.columns:
        mc = pd.to_numeric(filt["market_cap_100m"], errors="coerce").to_numpy(dtype=np.float64)
        mc_cov = float(np.isfinite(mc).mean()) if len(mc) else 0.0
    else:
        mc_cov = 0.0
    if "trade_value_100m" in filt.columns:
        tv = pd.to_numeric(filt["trade_value_100m"], errors="coerce").to_numpy(dtype=np.float64)
        tv_cov = float(np.isfinite(tv).mean()) if len(tv) else 0.0
    else:
        tv_cov = 0.0
    chg = pd.to_numeric(filt["daily_change_pct"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(filt["close"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(filt["high"], errors="coerce").to_numpy(dtype=np.float64)
    ceiling = np.isfinite(chg) & np.isfinite(close) & np.isfinite(high) & (chg >= CEILING_CHG_THRESHOLD) & (close >= high)
    if bool(screen.exclude_ceiling):
        n_ceiling_excluded = int(np.sum(ceiling))
        keep_ceiling = ~ceiling
    else:
        n_ceiling_excluded = 0
        keep_ceiling = np.ones(len(filt), dtype=bool)
    filt = filt.loc[keep_ceiling].copy().reset_index(drop=True)
    chg = pd.to_numeric(filt["daily_change_pct"], errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(chg) & (chg >= float(screen.change_lower))
    if screen.change_upper is not None:
        keep &= np.isfinite(chg) & (chg <= float(screen.change_upper))
    # tv_clean/mc_clean 우선: 원천 NaN을 복원한 확정 컬럼이 있으면 그것을 쓴다.
    tv_col = "tv_clean" if "tv_clean" in filt.columns else "trade_value_100m"
    if tv_col in filt.columns:
        tvf = pd.to_numeric(filt[tv_col], errors="coerce").to_numpy(dtype=np.float64)
        keep &= np.isfinite(tvf) & (tvf >= float(screen.min_trade_value_100m))
    elif float(screen.min_trade_value_100m) > 0.0:
        keep &= False
    mc_col = "mc_clean" if "mc_clean" in filt.columns else "market_cap_100m"
    if mc_col in filt.columns:
        mcf = pd.to_numeric(filt[mc_col], errors="coerce").to_numpy(dtype=np.float64)
        keep &= np.isfinite(mcf) & (mcf >= float(screen.min_market_cap_100m))
    elif float(screen.min_market_cap_100m) > 0.0:
        keep &= False
    if screen.max_tick_cost_bp is not None:
        if "tick_cost_bp" not in filt.columns:
            raise ValueError(
                "screen.max_tick_cost_bp requires a 'tick_cost_bp' column on price_history_df "
                "(attach it via src.data.panel_integrity.prepare_price_panel before calling "
                "build_universe_panel)"
            )
        tcb = pd.to_numeric(filt["tick_cost_bp"], errors="coerce").to_numpy(dtype=np.float64)
        n_tick_cost_excluded = int((~(np.isfinite(tcb) & (tcb <= float(screen.max_tick_cost_bp)))).sum())
        keep &= np.isfinite(tcb) & (tcb <= float(screen.max_tick_cost_bp))
    else:
        n_tick_cost_excluded = 0
    if bool(screen.require_index_up):
        idx_cols = [c for c in ("kospi_pct", "kosdaq_pct") if c in filt.columns]
        if not idx_cols:
            keep &= False
        else:
            idx_up = np.zeros(len(filt), dtype=bool)
            for c in idx_cols:
                v = pd.to_numeric(filt[c], errors="coerce").to_numpy(dtype=np.float64)
                idx_up |= np.isfinite(v) & (v > 0.0)
            keep &= idx_up
    panel = filt.loc[keep].copy().reset_index(drop=True)
    entry_close = pd.to_numeric(panel["close"], errors="coerce").to_numpy(dtype=np.float64)
    nxt_open = pd.to_numeric(panel["_next_open"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(entry_close) & np.isfinite(nxt_open) & (entry_close > 0.0)
    gross = np.full(len(panel), np.nan, dtype=np.float64)
    gross[ok] = nxt_open[ok] / entry_close[ok] - 1.0
    panel["mechanical_gross"] = np.asarray(gross, dtype=np.float64)
    if "trade_date" not in panel.columns:
        panel["trade_date"] = pd.to_datetime(panel["_entry_date"])
    panel = panel.drop(columns=[c for c in ("_entry_date", "_next_open") if c in panel.columns])
    n_days = int(pd.to_datetime(filt["_entry_date"]).nunique()) if len(filt) else 0
    if n_days < 1:
        n_days = int(pd.to_datetime([start_date, end_date]).nunique()) if n_raw == 0 else 1
    coverage_warning = bool(min(mc_cov, tv_cov) < 0.8)
    provenance: dict[str, Any] = {
        "n_raw": int(n_raw),
        "n_screened": len(panel),
        "n_ceiling_excluded": int(n_ceiling_excluded),
        "n_tick_cost_excluded": int(n_tick_cost_excluded),
        "n_days": int(n_days),
        "per_day": float(len(panel) / n_days) if n_days else 0.0,
        "screen": dataclasses.asdict(screen),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "market_cap_coverage": float(mc_cov),
        "trade_value_coverage": float(tv_cov),
        "coverage_warning": bool(coverage_warning),
        "kanri_filter_unavailable": True,
    }
    return panel, provenance


def screen_baseline_stats(
    panel: pd.DataFrame, *, group_col: str, gross_col: str, cost_ratio: float
) -> dict[str, Any]:
    """Equal-weighted daily unconditional after-cost expectancy."""
    if group_col not in panel.columns or gross_col not in panel.columns:
        raise ValueError(f"panel is missing group_col/gross_col {(group_col, gross_col)}")
    cost = float(cost_ratio)
    if not np.isfinite(cost):
        raise ValueError(f"cost_ratio must be finite, got {cost_ratio!r}")
    if len(panel) == 0:
        raise ValueError("panel is empty: no rows to score")
    gross = pd.to_numeric(panel[gross_col], errors="coerce").to_numpy(dtype=np.float64)
    grouped = panel.assign(_gross=np.asarray(gross, dtype=np.float64)).groupby(group_col, sort=True)["_gross"].mean()
    daily = pd.to_numeric(grouped, errors="coerce").to_numpy(dtype=np.float64)
    daily = daily[np.isfinite(daily)]
    n_days = int(daily.size)
    n_rows = len(panel)
    if n_days < 1:
        raise ValueError("panel has no finite daily gross observations")
    mean = float(np.mean(daily))
    sd = float(np.std(daily, ddof=1)) if n_days >= 2 else float("nan")
    gross_bp = float(mean * 1e4)
    net_bp = float((mean - cost) * 1e4)
    if np.isfinite(sd) and sd > 0.0 and n_days >= 2:
        t_stat = float(mean / (sd / np.sqrt(float(n_days))))
        sharpe = float(mean / sd * np.sqrt(252.0))
    else:
        t_stat = float("nan")
        sharpe = float("nan")
    net_daily = daily - cost
    win_rate = float(np.mean(net_daily > 0.0)) if n_days else float("nan")
    attrs = getattr(panel, "attrs", {})
    coverage_warning = any(
        key in attrs and float(attrs[key]) < 0.8
        for key in ("market_cap_coverage", "trade_value_coverage", "coverage")
        if key in attrs
    )
    return {
        "n_rows": int(n_rows),
        "n_days": int(n_days),
        "per_day": float(n_rows / n_days) if n_days else 0.0,
        "gross_bp": float(gross_bp),
        "net_bp": float(net_bp),
        "t_stat": float(t_stat),
        "sharpe": float(sharpe),
        "win_rate": float(win_rate),
        "coverage_warning": bool(coverage_warning),
    }
