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

    from src.api.kis.client import KisApiClient

    with (
        patch.object(KisApiClient, "get_program_trade_daily_history", AsyncMock(side_effect=_fake_history)),
        patch.object(KisApiClient, "create_session") as mock_create_session,
        patch.object(KisApiClient, "ensure_token", AsyncMock(return_value="tok")),
    ):
        mock_create_session.return_value.__aenter__ = AsyncMock(return_value=object())
        mock_create_session.return_value.__aexit__ = AsyncMock(return_value=False)
        out = program_trade_daily.collect_program_trade_daily(cfg, [pd.Timestamp("2024-01-02")])

    assert out["symbol"].iloc[0] == "005930"
    assert out["program_net_vol"].iloc[0] == -200.0


def test_collect_program_trade_daily_fans_out_across_keys(monkeypatch) -> None:
    """다중 키 선언 시 팬아웃 헬퍼로 위임하고 결과를 기존 스키마로 조립한다."""
    from pathlib import Path

    import pandas as pd

    from src.backfill.altdata import program_trade_daily
    from src.backfill.altdata.config import AltDataFetchConfig

    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
        universe_symbols=frozenset({"005930"}),
        extra_client_kwargs=(("k2", "s2", "h2"),),
    )
    seen: dict[str, object] = {}

    async def _fake_fan_out(cfg_arg, symbols, call, on_error):
        seen["symbols"] = list(symbols)

        class _Stub:
            async def get_program_trade_daily_history(self, session, code, start, end, market_div_code=None):
                return {"rt_cd": "0", "output": []}

        res = await call(_Stub(), object(), "005930")
        assert res["rt_cd"] == "0"
        assert on_error("005930", RuntimeError("x")) == {"rt_cd": "9", "output": []}
        return [("005930", {"rt_cd": "0", "output": [
            {"stck_bsop_date": "20240102", "whol_smtn_seln_vol": "1000", "whol_smtn_shnu_vol": "800",
             "whol_smtn_ntby_qty": "-200", "whol_smtn_seln_tr_pbmn": "5000000",
             "whol_smtn_shnu_tr_pbmn": "4000000", "whol_smtn_ntby_tr_pbmn": "-1000000"},
        ]})]

    monkeypatch.setattr(program_trade_daily, "fan_out_symbol_calls", _fake_fan_out)
    out = program_trade_daily.collect_program_trade_daily(cfg, [pd.Timestamp("2024-01-02")])
    assert seen["symbols"] == ["005930"]
    assert out["symbol"].iloc[0] == "005930"
    assert out["program_net_vol"].iloc[0] == -200.0
