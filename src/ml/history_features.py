"""Multiday trailing feature layer sourced from price_history.parquet."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HISTORY_FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "dist_ma5",
    "dist_ma20",
    "dist_ma60",
    "realized_vol_20d",
    "atr_pct_14d",
    "high_252d_ratio",
    "low_252d_ratio",
    "vol_ratio_5_20",
    "amihud_illiq_20d",
    "inst_netbuy_z_20d",
    "foreign_netbuy_z_20d",
    "up_day_ratio_10d",
    "prev_gap_mean_5d",
)

_REQUIRED_PRICE_HISTORY_COLUMNS: frozenset[str] = frozenset(
    {"date", "symbol", "open", "high", "low", "close", "volume", "trade_value_100m", "inst_netbuy", "foreign_netbuy"}
)

_REALIZED_VOL_FLOOR: float = 0.005
_DEFAULT_REALIZED_VOL_FALLBACK: float = 0.02


def _normalize_price_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df = df.sort_values(["symbol", "date"], ascending=True)
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    df = df.reset_index(drop=True)
    return df


def load_price_history(path: str | os.PathLike[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = _REQUIRED_PRICE_HISTORY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing required price_history columns: {sorted(missing)}")
    return _normalize_price_history_frame(df)


def _group_rolling(series: pd.Series, labels: pd.Series, window: int, min_periods: int, func: str) -> pd.Series:
    # vectorized groupby rolling
    grouped = series.groupby(labels.to_numpy(), sort=False)
    rolled = grouped.rolling(window=window, min_periods=min_periods)
    result = getattr(rolled, func)()
    if isinstance(result.index, pd.MultiIndex):
        result = result.droplevel(0)
    return result.reindex(series.index)


def compute_trailing_frame(price_history: pd.DataFrame) -> pd.DataFrame:
    # Normalize copy sorted by symbol/date
    df = price_history.copy()
    # Ensure date/symbol normalized if caller passed raw (defensive)
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    labels = df["symbol"]
    # Extract numeric series as float64
    close = df["close"].astype(np.float64)
    high = df["high"].astype(np.float64)
    low = df["low"].astype(np.float64)
    open_s = df["open"].astype(np.float64)
    volume = df["volume"].astype(np.float64)
    trade_value = df["trade_value_100m"].astype(np.float64)
    inst = df["inst_netbuy"].astype(np.float64)
    foreign = df["foreign_netbuy"].astype(np.float64)

    # daily log return via pct_change per symbol
    daily_pct = close.groupby(labels.to_numpy(), sort=False).pct_change()
    daily_log_ret = pd.Series(np.log1p(daily_pct.to_numpy(dtype=np.float64)), index=df.index)

    prev_close = close.groupby(labels.to_numpy(), sort=False).shift(1)

    # ret_Nd
    ret_5d = _group_rolling(daily_log_ret, labels, 5, 3, "sum")
    ret_20d = _group_rolling(daily_log_ret, labels, 20, 10, "sum")
    ret_60d = _group_rolling(daily_log_ret, labels, 60, 30, "sum")

    # dist_maN
    ma5 = _group_rolling(close, labels, 5, 3, "mean")
    ma20 = _group_rolling(close, labels, 20, 10, "mean")
    ma60 = _group_rolling(close, labels, 60, 30, "mean")

    def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
        av = a.to_numpy(dtype=np.float64)
        bv = b.to_numpy(dtype=np.float64)
        out = np.full(av.shape, np.nan, dtype=np.float64)
        mask = np.isfinite(bv) & (bv != 0)
        np.divide(av, bv, out=out, where=mask)
        return pd.Series(out, index=a.index)

    dist_ma5 = _safe_div(close, ma5) - 1.0
    dist_ma20 = _safe_div(close, ma20) - 1.0
    dist_ma60 = _safe_div(close, ma60) - 1.0

    # realized_vol_20d
    realized_vol_20d = _group_rolling(daily_log_ret, labels, 20, 10, "std")

    # atr_pct_14d
    hl = high - low
    abs_h_pc = (high - prev_close).abs()
    abs_l_pc = (low - prev_close).abs()
    # Use pandas max to handle NaN for first row (high-low should win)
    tr_frame = pd.concat([hl, abs_h_pc, abs_l_pc], axis=1)
    true_range = tr_frame.max(axis=1, skipna=True)
    # ensure float64
    true_range = true_range.astype(np.float64)
    atr_mean = _group_rolling(true_range, labels, 14, 7, "mean")
    atr_pct_14d = _safe_div(atr_mean, close)

    # high_252d_ratio / low_252d_ratio
    high_roll_max = _group_rolling(high, labels, 252, 60, "max")
    low_roll_min = _group_rolling(low, labels, 252, 60, "min")
    high_252d_ratio = _safe_div(close, high_roll_max)
    low_252d_ratio = _safe_div(close, low_roll_min)

    # vol_ratio_5_20
    vol_ma5 = _group_rolling(volume, labels, 5, 3, "mean")
    vol_ma20 = _group_rolling(volume, labels, 20, 10, "mean")
    vol_ratio_5_20 = _safe_div(vol_ma5, vol_ma20)

    # amihud_illiq_20d
    trade_value_no0 = trade_value.replace(0, np.nan)
    amihud_raw = _safe_div(daily_log_ret.abs(), trade_value_no0)
    amihud_illiq_20d = _group_rolling(amihud_raw, labels, 20, 10, "mean")

    # inst/foreign z
    inst_mean = _group_rolling(inst, labels, 20, 10, "mean")
    inst_std = _group_rolling(inst, labels, 20, 10, "std")
    foreign_mean = _group_rolling(foreign, labels, 20, 10, "mean")
    foreign_std = _group_rolling(foreign, labels, 20, 10, "std")
    inst_netbuy_z_20d = _safe_div(inst - inst_mean, inst_std)
    foreign_netbuy_z_20d = _safe_div(foreign - foreign_mean, foreign_std)
    # clip [-5,5]
    inst_netbuy_z_20d = inst_netbuy_z_20d.clip(lower=-5, upper=5)
    foreign_netbuy_z_20d = foreign_netbuy_z_20d.clip(lower=-5, upper=5)

    # up_day_ratio_10d
    up_flag = (daily_log_ret > 0).astype(np.float64)
    # NaN >0 is False, keep 0 for NaN rows? But rolling min_periods handles
    up_day_ratio_10d = _group_rolling(pd.Series(up_flag, index=df.index), labels, 10, 5, "mean")

    # prev_gap_mean_5d
    gap_raw = _safe_div(open_s - prev_close, prev_close)
    prev_gap_mean_5d = _group_rolling(gap_raw, labels, 5, 3, "mean")

    out = pd.DataFrame(
        {
            "symbol": df["symbol"],
            "date": df["date"],
            "ret_5d": ret_5d.astype(np.float64),
            "ret_20d": ret_20d.astype(np.float64),
            "ret_60d": ret_60d.astype(np.float64),
            "dist_ma5": dist_ma5.astype(np.float64),
            "dist_ma20": dist_ma20.astype(np.float64),
            "dist_ma60": dist_ma60.astype(np.float64),
            "realized_vol_20d": realized_vol_20d.astype(np.float64),
            "atr_pct_14d": atr_pct_14d.astype(np.float64),
            "high_252d_ratio": high_252d_ratio.astype(np.float64),
            "low_252d_ratio": low_252d_ratio.astype(np.float64),
            "vol_ratio_5_20": vol_ratio_5_20.astype(np.float64),
            "amihud_illiq_20d": amihud_illiq_20d.astype(np.float64),
            "inst_netbuy_z_20d": inst_netbuy_z_20d.astype(np.float64),
            "foreign_netbuy_z_20d": foreign_netbuy_z_20d.astype(np.float64),
            "up_day_ratio_10d": up_day_ratio_10d.astype(np.float64),
            "prev_gap_mean_5d": prev_gap_mean_5d.astype(np.float64),
        }
    )
    # Replace inf with NaN, do not fill
    numeric_cols = list(HISTORY_FEATURE_COLUMNS)
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    # Ensure column order
    out = out[["symbol", "date", *HISTORY_FEATURE_COLUMNS]]
    # Ensure float64 dtypes
    for col in HISTORY_FEATURE_COLUMNS:
        out[col] = out[col].astype(np.float64)
    return out


def attach_history_features(
    panel: pd.DataFrame, price_history: pd.DataFrame, *, date_col: str = "trade_date", code_col: str = "stock_code"
) -> pd.DataFrame:
    if date_col not in panel.columns:
        raise ValueError(f"panel missing date_col {date_col!r}")
    if code_col not in panel.columns:
        raise ValueError(f"panel missing code_col {code_col!r}")

    # normalize price_history inline (coerce date/symbol as in load_price_history)
    ph_norm = price_history.copy()
    ph_norm["date"] = pd.to_datetime(ph_norm["date"])
    ph_norm["symbol"] = ph_norm["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    # trailing frame handles its own sorting
    trailing = compute_trailing_frame(ph_norm)

    # Build stable positional key
    out = panel.copy()
    origin_index = out.index
    origin_pos = np.arange(len(out))
    out["_pos"] = origin_pos
    out["join_symbol"] = out[code_col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    out["join_date"] = pd.to_datetime(out[date_col])

    left_sorted = out.sort_values("join_date")
    right_sorted = trailing.sort_values("date")

    merged = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on="join_date",
        right_on="date",
        left_by="join_symbol",
        right_by="symbol",
        direction="backward",
        allow_exact_matches=False,
    )

    # Re-index back to original order
    merged = merged.sort_values("_pos")
    merged.index = origin_index

    # realized_vol derived from realized_vol_20d
    rv = merged["realized_vol_20d"].to_numpy(dtype=np.float64)
    # finite and >0 -> floored, else fallback
    realized = np.where(np.isfinite(rv) & (rv > 0), np.maximum(rv, _REALIZED_VOL_FLOOR), _DEFAULT_REALIZED_VOL_FALLBACK)
    merged["realized_vol"] = realized.astype(np.float64)

    # Drop helper columns: join_date, join_symbol, symbol, date (right side)
    # Note: after merge, columns include join_date, join_symbol, _pos, symbol, date
    # Use errors='ignore' to be safe
    cols_to_drop = [col for col in ["join_date", "join_symbol", "symbol", "date", "_pos"] if col in merged.columns]
    merged = merged.drop(columns=cols_to_drop)

    # Ensure HISTORY_FEATURE_COLUMNS remain float64 and realized_vol exists
    for col in HISTORY_FEATURE_COLUMNS:
        if col in merged.columns:
            merged[col] = merged[col].astype(np.float64)
    merged["realized_vol"] = merged["realized_vol"].astype(np.float64)

    return merged
