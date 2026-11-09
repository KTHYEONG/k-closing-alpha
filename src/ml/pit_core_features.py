"""Serving-safe PIT-CORE feature math: the 15:20 intraday path, T-1 index lags and stock history.

Leaf module (stdlib, numpy, pandas, src.config only). Research walk-forward modules re-export these names;
serving builds the same columns from live bars and the price history served before the decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Index columns and their T-1 lag suffix; the lag is the unshifted previous trading date.
INDEX_FEATURE_COLS: tuple[str, ...] = ("kospi_pct", "kosdaq_pct", "v_kospi")

LAG1_SUFFIX: str = "_lag1"

SESSION_OPEN_MIN: int = 9 * 60

DECISION_MIN: int = 15 * 60 + 20

SESSION_MINUTES_TO_DECISION: float = float(DECISION_MIN - SESSION_OPEN_MIN)

MIN_BARS_FOR_PATH: int = 30

HISTORY_WINDOW_SHORT: int = 5

HISTORY_WINDOW: int = 20

INTRADAY_FEATURES: tuple[str, ...] = (
    "x_rv_late", "x_rv1m", "x_r30", "x_r60", "x_r_pm", "x_r_am", "x_r_open30", "x_vwap_dev",
    "x_vol_last60", "x_vol_last30", "x_hi_time", "x_lo_time", "x_zero_vol",
)  # fmt: skip

PIT_CORE_PATH_FEATURES: tuple[str, ...] = ("x_rv_late", "x_rv1m", "x_r30", "x_r_pm")
PIT_CORE_HISTORY_FEATURES: tuple[str, ...] = ("h_amihud20", "h_turnover20")
PIT_CORE_INDEX_LAG_FEATURES: tuple[str, ...] = tuple(f"{c}{LAG1_SUFFIX}" for c in INDEX_FEATURE_COLS)
CORE_HISTORY_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "close",
    "prev_close",
    "trade_value_100m",
    "market_cap_100m",
    *INDEX_FEATURE_COLS,
)
_HISTORY_MIN_PERIODS: int = max(3, HISTORY_WINDOW // 2)


def bar_end_minutes(ts_hms: pd.Series, vendor: pd.Series) -> np.ndarray:
    """Minute-of-day at which each bar's interval ends, honouring the vendor stamp convention."""
    from src.config.market_session import BAR_STAMP_START, INTRADAY_BAR_STAMP_CONVENTION

    hms = ts_hms.astype(int).to_numpy()
    minutes = (hms // 10000) * 60 + (hms // 100) % 100
    starts = vendor.astype(str).map(
        lambda v: INTRADAY_BAR_STAMP_CONVENTION.get(v.removesuffix("_cons"), "end") == BAR_STAMP_START
    )
    return np.asarray(minutes + starts.to_numpy().astype(int), dtype=np.int64)


def path_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Intraday path features per symbol from 1m bars carrying symbol, end_min, open, high, low, close, volume.

    Only bars with SESSION_OPEN_MIN < end_min <= DECISION_MIN are used; symbols with fewer than
    MIN_BARS_FOR_PATH bars or a non-positive last close are dropped.
    """
    work = bars[(bars["end_min"] > SESSION_OPEN_MIN) & (bars["end_min"] <= DECISION_MIN)]
    rows = []
    for symbol, g in work.sort_values(["symbol", "end_min"]).groupby("symbol", sort=False):
        t = g["end_min"].to_numpy()
        c = g["close"].to_numpy(dtype=np.float64)
        if len(c) < MIN_BARS_FOR_PATH or c[-1] <= 0:
            continue
        h = g["high"].to_numpy(dtype=np.float64)
        lo = g["low"].to_numpy(dtype=np.float64)
        v = g["volume"].to_numpy(dtype=np.float64)
        first_open = float(g["open"].iloc[0])

        def close_at(minute: int, t: np.ndarray = t, c: np.ndarray = c) -> float:
            k = int(np.searchsorted(t, minute, side="right")) - 1
            return float(c[k]) if k >= 0 else np.nan

        last = float(c[-1])
        vol_total = float(v.sum())
        log_ret = np.diff(np.log(np.maximum(c, 1e-9)))
        late = t > DECISION_MIN - 60
        late_ret = log_ret[late[1:]]
        rows.append(
            {
                "symbol": symbol,
                "x_r30": last / close_at(DECISION_MIN - 30) - 1,
                "x_r60": last / close_at(DECISION_MIN - 60) - 1,
                "x_r_pm": last / close_at(12 * 60) - 1,
                "x_r_am": close_at(12 * 60) / first_open - 1 if first_open > 0 else np.nan,
                "x_r_open30": close_at(SESSION_OPEN_MIN + 30) / first_open - 1 if first_open > 0 else np.nan,
                "x_vwap_dev": last / (((h + lo + c) / 3.0 * v).sum() / vol_total) - 1 if vol_total > 0 else np.nan,
                "x_vol_last60": float(v[late].sum()) / vol_total if vol_total > 0 else np.nan,
                "x_vol_last30": float(v[t > DECISION_MIN - 30].sum()) / vol_total if vol_total > 0 else np.nan,
                "x_hi_time": (t[int(np.argmax(h))] - SESSION_OPEN_MIN) / SESSION_MINUTES_TO_DECISION,
                "x_lo_time": (t[int(np.argmin(lo))] - SESSION_OPEN_MIN) / SESSION_MINUTES_TO_DECISION,
                "x_rv1m": float(np.std(log_ret)) * np.sqrt(len(log_ret)),
                "x_rv_late": float(np.std(late_ret)) if len(late_ret) > HISTORY_WINDOW_SHORT else np.nan,
                "x_zero_vol": float((v <= 0).mean()),
            }
        )
    return pd.DataFrame(rows, columns=["symbol", *INTRADAY_FEATURES])


def attach_lagged_index(frame: pd.DataFrame, index_by_date: pd.DataFrame) -> pd.DataFrame:
    """Attach T-1 market index columns (`<col>_lag1`) from a per-date index table.

    Args:
        frame: Rows with a date column.
        index_by_date: One row per trading date with date and INDEX_FEATURE_COLS (EOD values of that date).

    Returns:
        Copy of frame with INDEX_FEATURE_COLS + LAG1_SUFFIX taken from the previous trading date.

    Raises:
        ValueError: Missing columns or duplicate dates in index_by_date.
    """
    missing = [c for c in ("date", *INDEX_FEATURE_COLS) if c not in index_by_date.columns]
    if missing:
        raise ValueError(f"index_by_date missing columns: {missing}")
    table = index_by_date[["date", *INDEX_FEATURE_COLS]].copy()
    table["date"] = pd.to_datetime(table["date"]).dt.normalize()
    if table["date"].duplicated().any():
        raise ValueError("index_by_date carries duplicate dates")
    table = table.sort_values("date").reset_index(drop=True)
    lagged = table[list(INDEX_FEATURE_COLS)].shift(1)
    lagged.columns = [f"{c}{LAG1_SUFFIX}" for c in INDEX_FEATURE_COLS]
    lagged["date"] = table["date"]
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.merge(lagged, on="date", how="left", validate="many_to_one")


def index_table_from_price_history(ph: pd.DataFrame) -> pd.DataFrame:
    """One row per date with the market-wide EOD index columns carried on every price_history row."""
    missing = [c for c in ("date", *INDEX_FEATURE_COLS) if c not in ph.columns]
    if missing:
        raise ValueError(f"price_history missing index columns: {missing}")
    work = ph[["date", *INDEX_FEATURE_COLS]].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    return work.groupby("date", sort=True)[list(INDEX_FEATURE_COLS)].median().reset_index()


@dataclass(frozen=True)
class PitCoreServingConfig:
    """Serving tolerances for attaching PIT-CORE features; min_path_coverage is the finite x_r30 share required."""

    min_path_coverage: float = 0.8

    def __post_init__(self) -> None:
        if not 0 < self.min_path_coverage <= 1:
            raise ValueError(f"min_path_coverage must be in (0, 1], got {self.min_path_coverage!r}")


def _history_rolls(history: pd.DataFrame) -> pd.DataFrame:
    """Unshifted per-row rolling means of the two stock-history features, sorted by (symbol, date)."""
    d = history.sort_values(["symbol", "date"]).reset_index(drop=True)
    ret = d["close"] / d["prev_close"] - 1
    tv = d["trade_value_100m"].clip(lower=1e-6)
    amihud = ret.abs() / tv
    turnover = tv / d["market_cap_100m"].clip(lower=1e-6)

    def roll(series: pd.Series) -> pd.Series:
        return series.groupby(d["symbol"]).transform(
            lambda x: x.rolling(HISTORY_WINDOW, min_periods=_HISTORY_MIN_PERIODS).mean()
        )

    return pd.DataFrame(
        {
            "date": d["date"],
            "symbol": d["symbol"],
            "h_amihud20": roll(amihud),
            "h_turnover20": roll(turnover),
        }
    )


def core_history_features(ph: pd.DataFrame) -> pd.DataFrame:
    """Per (date, symbol) h_amihud20 and h_turnover20 from the same symbol's strictly earlier rows.

    Args:
        ph: Daily rows with CORE_HISTORY_COLUMNS, one row per (date, symbol).

    Returns:
        Frame with date, symbol (str), h_amihud20, h_turnover20, sorted by (symbol, date).
    """
    rolls = _history_rolls(ph)
    for name in PIT_CORE_HISTORY_FEATURES:
        rolls[name] = rolls.groupby("symbol")[name].shift(1)
    rolls["symbol"] = rolls["symbol"].astype(str)
    return rolls[["date", "symbol", *PIT_CORE_HISTORY_FEATURES]].reset_index(drop=True)


def latest_core_history(history: pd.DataFrame) -> pd.DataFrame:
    """Serving form of core_history_features: the value each symbol's next row would receive.

    Args:
        history: Daily rows with CORE_HISTORY_COLUMNS, all dated before the decision.

    Returns:
        One row per symbol with symbol (str), h_amihud20, h_turnover20, taken from the symbol's last row
        without calendar alignment, as in training.
    """
    rolls = _history_rolls(history)
    last = rolls.groupby("symbol", sort=False).tail(1)
    out = last[["symbol", *PIT_CORE_HISTORY_FEATURES]].copy()
    out["symbol"] = out["symbol"].astype(str)
    return out.reset_index(drop=True)


def latest_index_lag(history: pd.DataFrame) -> dict[str, float]:
    """T-1 index features for the next decision: per-date median at the latest date in history.

    Args:
        history: Daily rows with date and INDEX_FEATURE_COLS, all dated before the decision.

    Returns:
        Mapping from PIT_CORE_INDEX_LAG_FEATURES to finite float values.

    Raises:
        ValueError: When history is empty, misses a column, or any lag value is non-finite.
    """
    missing = [c for c in ("date", *INDEX_FEATURE_COLS) if c not in history.columns]
    if missing:
        raise ValueError(f"history missing index columns: {missing}")
    if history.empty:
        raise ValueError("history is empty; cannot derive index lag features")
    dates = pd.to_datetime(history["date"]).dt.normalize()
    latest = history.loc[dates == dates.max(), list(INDEX_FEATURE_COLS)].median()
    values = {
        lag: float(latest[base]) for base, lag in zip(INDEX_FEATURE_COLS, PIT_CORE_INDEX_LAG_FEATURES, strict=True)
    }
    bad = [name for name, value in values.items() if not np.isfinite(value)]
    if bad:
        raise ValueError(f"non-finite index lag features at latest history date: {bad}")
    return values
