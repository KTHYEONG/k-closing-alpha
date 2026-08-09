from __future__ import annotations

import pandas as pd

from src.backfill.price.config import FetchConfig
from src.backfill.price.universe import _build_symbol_windows


def test_new_symbol_uses_requested_lookback_window() -> None:
    universe = pd.DataFrame({"symbol": ["005930"], "market": ["KOSPI"]})
    cfg = FetchConfig(
        lookback_trading_days=10,
        fixed_start_date=pd.Timestamp("2016-01-01"),
        fixed_end_date=pd.Timestamp("2025-01-31"),
    )

    windows = _build_symbol_windows(universe, fetch_cfg=cfg)

    assert len(windows) == 1
    symbol, start, end, market = windows[0]
    assert symbol == "005930"
    assert start == pd.Timestamp("2025-01-17")
    assert end == pd.Timestamp("2025-01-31")
    assert market == "KOSPI"


def test_existing_symbol_keeps_incremental_overlap_window() -> None:
    universe = pd.DataFrame({"symbol": ["005930"], "market": ["KOSPI"]})
    cfg = FetchConfig(
        lookback_trading_days=10,
        calendar_buffer_days=30,
        fixed_start_date=pd.Timestamp("2016-01-01"),
        fixed_end_date=pd.Timestamp("2025-01-31"),
    )

    windows = _build_symbol_windows(
        universe,
        fetch_cfg=cfg,
        existing_last_dates={"005930": pd.Timestamp("2025-01-20")},
    )

    assert windows[0][1] == pd.Timestamp("2024-12-21")
