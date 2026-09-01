from pathlib import Path

import pandas as pd

from src.backfill.altdata import shorting
from src.backfill.altdata.config import AltDataFetchConfig


def test_collect_shorting_builds_symbol_date_panel(monkeypatch) -> None:
    vol = pd.DataFrame({"공매도": [100.0], "매수": [10000.0], "비중": [1.0]}, index=pd.Index(["005930"], name="티커"))
    bal = pd.DataFrame({"공매도잔고": [500.0], "상장주식수": [1e8], "공매도금액": [3e7], "시가총액": [6e12], "비중": [0.5]}, index=pd.Index(["005930"], name="티커"))
    monkeypatch.setattr(shorting.stock, "get_shorting_volume_by_ticker", lambda *a, **k: vol)
    monkeypatch.setattr(shorting.stock, "get_shorting_balance_by_ticker", lambda *a, **k: bal)
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
    )
    out = shorting.collect_shorting(cfg, [pd.Timestamp("2024-01-02")])
    assert {"date", "symbol", "short_volume", "short_balance_qty"}.issubset(out.columns)
    assert out["symbol"].iloc[0] == "005930"
    assert out["short_volume"].iloc[0] == 100.0


from src.backfill.altdata import fundamental


def test_collect_fundamental_maps_valuation_columns(monkeypatch) -> None:
    frame = pd.DataFrame(
        {"BPS": [50000.0], "PER": [12.5], "PBR": [1.4], "EPS": [4000.0], "DIV": [2.1], "DPS": [1500.0]},
        index=pd.Index(["000660"], name="티커"),
    )
    monkeypatch.setattr(fundamental.stock, "get_market_fundamental_by_ticker", lambda *a, **k: frame)
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
    )
    out = fundamental.collect_fundamental(cfg, [pd.Timestamp("2024-01-02")])
    row = out.iloc[0]
    assert row["symbol"] == "000660"
    assert row["per"] == 12.5 and row["pbr"] == 1.4 and row["bps"] == 50000.0 and row["div_yield"] == 2.1
