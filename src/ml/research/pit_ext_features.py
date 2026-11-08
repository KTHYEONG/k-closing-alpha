"""Candidate decision-time features beyond the certified ranker set, for the 15:20 walk-forward research.

Four families, each observable at the 15:20 decision:
- intraday path (`x_*`): from regular-session 1m bars whose interval ends at or before 15:20;
- stock history (`h_*`): from daily rows strictly before T;
- market state (`m_*` at 15:20 from the decision panel cross-section, `i_*` from T-1 index levels) and
  calendar (`c_*`, known in advance).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SESSION_OPEN_MIN: int = 9 * 60
DECISION_MIN: int = 15 * 60 + 20
SESSION_MINUTES_TO_DECISION: float = float(DECISION_MIN - SESSION_OPEN_MIN)
MIN_BARS_FOR_PATH: int = 30
MAX_ABS_DAILY_CHANGE: float = 0.31
HISTORY_WINDOW_SHORT: int = 5
HISTORY_WINDOW: int = 20
INDEX_Z_WINDOW: int = 60
EXPIRY_DAY_FIRST: int = 8
EXPIRY_DAY_LAST: int = 14
THURSDAY: int = 3

INTRADAY_FEATURES: tuple[str, ...] = (
    "x_rv_late", "x_rv1m", "x_r30", "x_r60", "x_r_pm", "x_r_am", "x_r_open30", "x_vwap_dev",
    "x_vol_last60", "x_vol_last30", "x_hi_time", "x_lo_time", "x_zero_vol",
)  # fmt: skip
HISTORY_FEATURES: tuple[str, ...] = (
    "h_amihud20", "h_turnover20", "h_rv20", "h_vol_surge", "h_prog_dens", "h_prog_cum5", "h_prev_ret",
)  # fmt: skip
MARKET_FEATURES: tuple[str, ...] = (
    "m_breadth", "m_cap_chg", "m_eq_chg", "m_disp", "i_vk_d1", "i_vk_d5", "i_vk_z60", "i_kosdaq_sma20",
    "i_kospi_sma20", "i_kosdaq_r5", "c_gap_days", "c_dow", "c_expiry",
)  # fmt: skip
EXT_FEATURES: tuple[str, ...] = (*INTRADAY_FEATURES, *HISTORY_FEATURES, *MARKET_FEATURES)


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


def market_state(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-date 15:20 cross-section state of the decision panel (breadth, cap/equal-weight change, dispersion)."""
    p = panel[["date", "close", "prev_close", "market_cap_100m"]].copy()
    p["chg"] = p["close"] / p["prev_close"] - 1
    p = p[np.isfinite(p["chg"]) & (p["chg"].abs() < MAX_ABS_DAILY_CHANGE)]
    p["cap"] = p["market_cap_100m"].clip(lower=0)
    p["wchg"] = p["chg"] * p["cap"]
    p["up"] = (p["chg"] > 0).astype(float)
    g = p.groupby("date")
    out = pd.DataFrame(
        {
            "m_breadth": g["up"].mean(),
            "m_cap_chg": g["wchg"].sum() / g["cap"].sum(),
            "m_eq_chg": g["chg"].mean(),
            "m_disp": g["chg"].std(),
        }
    )
    return out.reset_index()


def stock_history(ph: pd.DataFrame) -> pd.DataFrame:
    """Per (date, symbol) features from daily rows strictly before the date (every value is shifted by one row)."""
    d = ph.sort_values(["symbol", "date"]).reset_index(drop=True)
    ret = d["close"] / d["prev_close"] - 1
    tv = d["trade_value_100m"].clip(lower=1e-6)

    def roll(series: pd.Series, n: int, how: str) -> pd.Series:
        return series.groupby(d["symbol"]).transform(lambda x: getattr(x.rolling(n, min_periods=max(3, n // 2)), how)())

    out = pd.DataFrame({"date": d["date"], "symbol": d["symbol"].astype(str)})
    raw = {
        "h_prev_ret": ret,
        "h_rv20": roll(ret, HISTORY_WINDOW, "std"),
        "h_vol_surge": d["volume"] / roll(d["volume"], HISTORY_WINDOW, "mean"),
        "h_turnover20": roll(tv / d["market_cap_100m"].clip(lower=1e-6), HISTORY_WINDOW, "mean"),
        "h_amihud20": roll(ret.abs() / tv, HISTORY_WINDOW, "mean"),
        "h_prog_dens": d["program_netbuy"] / tv,
        "h_prog_cum5": roll(d["program_netbuy"], HISTORY_WINDOW_SHORT, "sum") / roll(tv, HISTORY_WINDOW_SHORT, "sum"),
    }
    for name, series in raw.items():
        out[name] = series.groupby(d["symbol"]).shift(1).to_numpy()
    return out


def index_and_calendar(ph: pd.DataFrame) -> pd.DataFrame:
    """Per-date T-1 index trend/volatility features and calendar flags (gap to the next session, weekday, expiry)."""
    idx = ph.groupby("date")[["kospi_pct", "kosdaq_pct", "v_kospi"]].median().sort_index()
    kospi = (1 + idx["kospi_pct"]).cumprod()
    kosdaq = (1 + idx["kosdaq_pct"]).cumprod()
    vk = idx["v_kospi"]
    out = pd.DataFrame(index=idx.index)
    out["i_kospi_sma20"] = (kospi / kospi.rolling(HISTORY_WINDOW).mean() - 1).shift(1)
    out["i_kosdaq_sma20"] = (kosdaq / kosdaq.rolling(HISTORY_WINDOW).mean() - 1).shift(1)
    out["i_kosdaq_r5"] = (kosdaq / kosdaq.shift(HISTORY_WINDOW_SHORT) - 1).shift(1)
    out["i_vk_d1"] = (vk - vk.shift(1)).shift(1)
    out["i_vk_d5"] = (vk - vk.shift(HISTORY_WINDOW_SHORT)).shift(1)
    out["i_vk_z60"] = ((vk - vk.rolling(INDEX_Z_WINDOW).mean()) / vk.rolling(INDEX_Z_WINDOW).std()).shift(1)
    dates = pd.Series(idx.index, index=idx.index)
    nxt = dates.shift(-1)
    out["c_gap_days"] = (nxt - dates).dt.days
    out["c_dow"] = idx.index.dayofweek
    day = idx.index.day
    out["c_expiry"] = ((idx.index.dayofweek == THURSDAY) & (day >= EXPIRY_DAY_FIRST) & (day <= EXPIRY_DAY_LAST)).astype(
        float
    )
    return out.reset_index()


def _load_day_bars(day: str, research_root: Path) -> pd.DataFrame:  # pragma: no cover - partition I/O
    from src.data.pit1520_panel import load_consolidated_bars, load_regular_bars

    parts = [load_regular_bars(day)]
    for sub in ("toss_offsession", "toss_consolidated"):
        path = research_root / sub / f"{day}.parquet"
        if path.exists():
            extra = pd.read_parquet(path)
            if len(extra.columns):
                parts.append(extra[[c for c in parts[0].columns if c in extra.columns]])
    parts.append(load_consolidated_bars(day))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    bars = pd.concat(parts, ignore_index=True)
    bars["symbol"] = bars["symbol"].astype(str)
    bars["end_min"] = bar_end_minutes(bars["ts_hms"], bars["vendor"])
    first_vendor = bars.groupby("symbol")["vendor"].transform("first")
    bars = bars[bars["vendor"] == first_vendor]
    return bars.drop_duplicates(["symbol", "end_min"], keep="first")


def _day_path(args: tuple[str, list[str], str]) -> pd.DataFrame:  # pragma: no cover - partition I/O
    day, symbols, root = args
    bars = _load_day_bars(day, Path(root))
    if bars.empty:
        return pd.DataFrame()
    out = path_features(bars[bars["symbol"].isin(set(symbols))])
    out.insert(0, "date", pd.Timestamp(day))
    return out


def build_ext_features(
    keys: pd.DataFrame, *, panel: pd.DataFrame, price_history: pd.DataFrame, research_root: Path, workers: int = 7
) -> pd.DataFrame:  # pragma: no cover - multi-year store orchestration, exercised by the research runs
    """All EXT_FEATURES for the (date, symbol) keys of the 15:20 rank pool."""
    k = keys[["date", "symbol"]].copy()
    k["symbol"] = k["symbol"].astype(str)
    by_day: dict[str, Sequence[str]] = k.groupby(k["date"].dt.strftime("%Y-%m-%d"))["symbol"].apply(list).to_dict()
    jobs = [(day, list(syms), str(research_root)) for day, syms in by_day.items()]
    with ProcessPoolExecutor(max_workers=int(workers)) as ex:
        parts = list(ex.map(_day_path, jobs, chunksize=8))
    path = pd.concat([p for p in parts if len(p)], ignore_index=True)
    panel_work = panel.assign(date=pd.to_datetime(panel["date"]))
    ph = price_history.assign(symbol=price_history["symbol"].astype(str))
    out = k.merge(path, on=["date", "symbol"], how="left").merge(stock_history(ph), on=["date", "symbol"], how="left")
    out = out.merge(market_state(panel_work), on="date", how="left").merge(
        index_and_calendar(ph), on="date", how="left"
    )
    logger.info("[DATA] stage=ext_features rows=%d path_coverage=%.4f", len(out), float(out["x_r30"].notna().mean()))
    return out
