from pathlib import Path

import pandas as pd

from src.backfill.altdata import shorting
from src.backfill.altdata.config import AltDataFetchConfig


def test_collect_shorting_builds_symbol_date_panel(monkeypatch) -> None:
    from unittest.mock import AsyncMock, patch

    from src.backfill.altdata import shorting

    async def _fake_history(session, code, start_date, end_date, market_div_code=None):
        return {"rt_cd": "0", "output2": [
            {"stck_bsop_date": "20240102", "ssts_cntg_qty": "100", "ssts_tr_pbmn": "3000000",
             "acml_vol": "10000", "ssts_vol_rlim": "1.0"},
        ]}

    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
        universe_symbols=frozenset({"005930"}),
    )

    with (
        patch.object(shorting.KisApiClient, "get_daily_short_sale_history", AsyncMock(side_effect=_fake_history)),
        patch.object(shorting.KisApiClient, "create_session") as mock_create_session,
        patch.object(shorting.KisApiClient, "ensure_token", AsyncMock(return_value="tok")),
    ):
        mock_create_session.return_value.__aenter__ = AsyncMock(return_value=object())
        mock_create_session.return_value.__aexit__ = AsyncMock(return_value=False)
        out = shorting.collect_shorting(cfg, [pd.Timestamp("2024-01-02")])

    assert {"date", "symbol", "short_volume", "short_balance_qty"}.issubset(out.columns)
    assert out["symbol"].iloc[0] == "005930"
    assert out["short_volume"].iloc[0] == 100.0
    assert pd.isna(out["short_balance_qty"].iloc[0])


def test_collect_shorting_returns_empty_panel_without_universe_symbols() -> None:
    from src.backfill.altdata import shorting

    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
    )

    out = shorting.collect_shorting(cfg, [pd.Timestamp("2024-01-02")])

    assert out.empty
    assert "symbol" in out.columns


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
