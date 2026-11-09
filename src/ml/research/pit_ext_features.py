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

from src.ml.pit_core_features import (
    DECISION_MIN as DECISION_MIN,
)
from src.ml.pit_core_features import (
    HISTORY_WINDOW as HISTORY_WINDOW,
)
from src.ml.pit_core_features import (
    HISTORY_WINDOW_SHORT as HISTORY_WINDOW_SHORT,
)
from src.ml.pit_core_features import (
    INTRADAY_FEATURES as INTRADAY_FEATURES,
)
from src.ml.pit_core_features import (
    MIN_BARS_FOR_PATH as MIN_BARS_FOR_PATH,
)
from src.ml.pit_core_features import (
    SESSION_MINUTES_TO_DECISION as SESSION_MINUTES_TO_DECISION,
)
from src.ml.pit_core_features import (
    SESSION_OPEN_MIN as SESSION_OPEN_MIN,
)
from src.ml.pit_core_features import (
    bar_end_minutes as bar_end_minutes,
)
from src.ml.pit_core_features import (
    path_features as path_features,
)

logger = logging.getLogger(__name__)

MAX_ABS_DAILY_CHANGE: float = 0.31
INDEX_Z_WINDOW: int = 60
EXPIRY_DAY_FIRST: int = 8
EXPIRY_DAY_LAST: int = 14
THURSDAY: int = 3

HISTORY_FEATURES: tuple[str, ...] = (
    "h_amihud20", "h_turnover20", "h_rv20", "h_vol_surge", "h_prog_dens", "h_prog_cum5", "h_prev_ret",
)  # fmt: skip
MARKET_FEATURES: tuple[str, ...] = (
    "m_breadth", "m_cap_chg", "m_eq_chg", "m_disp", "i_vk_d1", "i_vk_d5", "i_vk_z60", "i_kosdaq_sma20",
    "i_kospi_sma20", "i_kosdaq_r5", "c_gap_days", "c_dow", "c_expiry",
)  # fmt: skip
EXT_FEATURES: tuple[str, ...] = (*INTRADAY_FEATURES, *HISTORY_FEATURES, *MARKET_FEATURES)


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
