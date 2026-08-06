"""지수 수익률 / VKOSPI 보강 (index-return & VKOSPI enrichment)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backfill.price.config import FetchConfig
from src.backfill.price.sources import _fetch_index_returns, _fetch_vkospi_proxy


def compute_vkospi_proxy(
    index_close_df: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: int = 20,
    output_col: str = "v_kospi",
) -> pd.DataFrame:
    """Build V-KOSPI proxy (historical volatility) from index close prices."""
    if index_close_df is None or index_close_df.empty:
        return pd.DataFrame(columns=["date", output_col])

    if "date" not in index_close_df.columns or "close" not in index_close_df.columns:
        return pd.DataFrame(columns=["date", output_col])

    out = index_close_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    if out.empty:
        return pd.DataFrame(columns=["date", output_col])

    close_ratio = pd.to_numeric(out["close"] / out["close"].shift(1), errors="coerce")
    log_ret = np.where(close_ratio > 0, np.log(close_ratio), np.nan)
    roll_std = pd.Series(log_ret, index=out.index).rolling(
        window=int(window),
        min_periods=int(min_periods),
    ).std(ddof=0)
    out[output_col] = roll_std * np.sqrt(252.0) * 100.0
    return out[["date", output_col]]


def _merge_index_returns(
    history: pd.DataFrame,
    fetch_cfg: FetchConfig,
) -> pd.DataFrame:
    if history.empty:
        return history
    buffer_days = max(30, int(fetch_cfg.lookback_trading_days) * 3)
    start = pd.Timestamp(history["date"].min()) - pd.Timedelta(days=buffer_days)
    end = pd.Timestamp(history["date"].max())

    kospi = _fetch_index_returns(start, end, code="1001", out_col="kospi_pct", fetch_cfg=fetch_cfg)
    kosdaq = _fetch_index_returns(start, end, code="2001", out_col="kosdaq_pct", fetch_cfg=fetch_cfg)
    # Historical proxy volatility is calculated from KOSPI/KOSDAQ index closes;
    # do not depend on pykrx's separate volatility-index ticker metadata.
    vkospi = _fetch_vkospi_proxy(start, end, fetch_cfg=fetch_cfg, index_code="1001", output_col="v_kospi")
    vkosdaq = _fetch_vkospi_proxy(start, end, fetch_cfg=fetch_cfg, index_code="2001", output_col="v_kosdaq")

    out = history.merge(kospi, on="date", how="left")
    out = out.merge(kosdaq, on="date", how="left")
    out = out.merge(vkospi, on="date", how="left")
    out = out.merge(vkosdaq, on="date", how="left")
    return out
