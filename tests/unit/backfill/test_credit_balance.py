from __future__ import annotations


def test_collect_credit_balance_panel_schema_has_ten_columns() -> None:
    from pathlib import Path

    import pandas as pd

    from src.backfill.altdata import credit_balance
    from src.backfill.altdata.config import AltDataFetchConfig

    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
    )

    out = credit_balance.collect_credit_balance(cfg, [pd.Timestamp("2024-01-02")])

    expected = {
        "date", "symbol", "loan_new_qty", "loan_redemption_qty", "loan_balance_qty",
        "loan_balance_amt", "loan_balance_rate", "stln_balance_qty", "stln_balance_amt", "stln_balance_rate",
    }
    assert expected.issubset(set(out.columns))
    assert out.empty


def test_collect_credit_balance_maps_kis_fields_with_universe_symbols() -> None:
    from pathlib import Path
    from unittest.mock import AsyncMock, patch

    import pandas as pd

    from src.backfill.altdata import credit_balance
    from src.backfill.altdata.config import AltDataFetchConfig

    async def _fake_history(session, code, start_date, end_date, market_div_code=None):
        return {"rt_cd": "0", "output": [
            {"deal_date": "20240102", "whol_loan_new_stcn": "100", "whol_loan_rdmp_stcn": "50",
             "whol_loan_rmnd_stcn": "1000", "whol_loan_rmnd_amt": "50000", "whol_loan_rmnd_rate": "0.5",
             "whol_stln_rmnd_stcn": "20", "whol_stln_rmnd_amt": "1000", "whol_stln_rmnd_rate": "0.01"},
        ]}

    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), markets=("KOSPI",), retries=1, retry_sleep_sec=0.0,
        universe_symbols=frozenset({"005930"}),
    )

    from src.api.kis.client import KisApiClient

    with (
        patch.object(KisApiClient, "get_daily_credit_balance_history", AsyncMock(side_effect=_fake_history)),
        patch.object(KisApiClient, "create_session") as mock_create_session,
        patch.object(KisApiClient, "ensure_token", AsyncMock(return_value="tok")),
    ):
        mock_create_session.return_value.__aenter__ = AsyncMock(return_value=object())
        mock_create_session.return_value.__aexit__ = AsyncMock(return_value=False)
        out = credit_balance.collect_credit_balance(cfg, [pd.Timestamp("2024-01-02")])

    assert out["symbol"].iloc[0] == "005930"
    assert out["loan_balance_qty"].iloc[0] == 1000.0


def test_collect_credit_balance_fans_out_across_keys(monkeypatch) -> None:
    """다중 키 선언 시 팬아웃 헬퍼로 위임하고 결과를 기존 스키마로 조립한다."""
    from pathlib import Path

    import pandas as pd

    from src.backfill.altdata import credit_balance
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
            async def get_daily_credit_balance_history(self, session, code, start, end, market_div_code=None):
                return {"rt_cd": "0", "output": []}

        res = await call(_Stub(), object(), "005930")
        assert res["rt_cd"] == "0"
        assert on_error("005930", RuntimeError("x")) == {"rt_cd": "9", "output": []}
        return [("005930", {"rt_cd": "0", "output": [
            {"deal_date": "20240102", "whol_loan_new_stcn": "100", "whol_loan_rdmp_stcn": "50",
             "whol_loan_rmnd_stcn": "1000", "whol_loan_rmnd_amt": "50000", "whol_loan_rmnd_rate": "0.5",
             "whol_stln_rmnd_stcn": "20", "whol_stln_rmnd_amt": "1000", "whol_stln_rmnd_rate": "0.01"},
        ]})]

    monkeypatch.setattr(credit_balance, "fan_out_symbol_calls", _fake_fan_out)
    out = credit_balance.collect_credit_balance(cfg, [pd.Timestamp("2024-01-02")])
    assert seen["symbols"] == ["005930"]
    assert out["symbol"].iloc[0] == "005930"
    assert out["loan_balance_qty"].iloc[0] == 1000.0
