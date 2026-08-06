from __future__ import annotations

import pandas as pd

from src.backfill.price import sources
from src.backfill.price.config import FetchConfig


def test_safe_market_wrappers_and_flow_fetchers(monkeypatch) -> None:
    monkeypatch.setattr(sources.stock, "get_market_ohlcv_by_date", lambda *a, **k: pd.DataFrame({"종가": [1]}))
    monkeypatch.setattr(sources.stock, "get_market_cap_by_date", lambda *a, **k: pd.DataFrame({"시가총액": [2]}))
    monkeypatch.setattr(sources.stock, "get_market_trading_value_by_date", lambda *a, **k: pd.DataFrame({"기관합계": [3], "외국인합계": [4]}))
    cfg = FetchConfig(retries=1, request_sleep_sec=0)
    assert not sources._safe_get_market_ohlcv_by_date("20200101", "20200102", "000001", cfg).empty
    assert not sources._safe_get_market_cap_by_date("20200101", "20200102", "000001", cfg).empty
    assert not sources._safe_get_trading_value_by_date("20200101", "20200102", "000001", cfg).empty

    monkeypatch.setattr(sources, "_resolve_program_history_func", lambda: lambda *a, **k: {"20200102": 5})
    monkeypatch.setattr(sources, "_resolve_investor_history_func", lambda: lambda *a, **k: pd.DataFrame({"date": ["20200102"], "foreign_netbuy": [6], "inst_netbuy": [7]}))
    program = sources._fetch_program_history_by_date("000001", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), cfg)
    investor = sources._fetch_investor_history_by_date("000001", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), cfg)
    assert program.loc[0, "program_netbuy"] == 5
    assert investor.loc[0, "foreign_netbuy"] == 6
