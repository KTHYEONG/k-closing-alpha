from __future__ import annotations


def test_collect_program_trade_daily_maps_kis_fields_with_universe_symbols() -> None:
    from pathlib import Path
    from unittest.mock import AsyncMock, patch

    import pandas as pd

    from src.backfill.altdata import program_trade_daily
    from src.backfill.altdata.config import AltDataFetchConfig

    async def _fake_history(session, code, start_date, end_date, market_div_code=None):
        return {"rt_cd": "0", "output": [
            {"stck_bsop_date": "20240102", "whol_smtn_seln_vol": "1000", "whol_smtn_shnu_vol": "800",
             "whol_smtn_ntby_qty": "-200", "whol_smtn_seln_tr_pbmn": "5000000",
             "whol_smtn_shnu_tr_pbmn": "4000000", "whol_smtn_ntby_tr_pbmn": "-1000000"},
        ]}

    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
        universe_symbols=frozenset({"005930"}),
    )

    with (
        patch.object(program_trade_daily.KisApiClient, "get_program_trade_daily_history", AsyncMock(side_effect=_fake_history)),
        patch.object(program_trade_daily.KisApiClient, "create_session") as mock_create_session,
        patch.object(program_trade_daily.KisApiClient, "ensure_token", AsyncMock(return_value="tok")),
    ):
        mock_create_session.return_value.__aenter__ = AsyncMock(return_value=object())
        mock_create_session.return_value.__aexit__ = AsyncMock(return_value=False)
        out = program_trade_daily.collect_program_trade_daily(cfg, [pd.Timestamp("2024-01-02")])

    assert out["symbol"].iloc[0] == "005930"
    assert out["program_net_vol"].iloc[0] == -200.0
