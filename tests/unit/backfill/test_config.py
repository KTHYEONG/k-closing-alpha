from __future__ import annotations


def test_altdata_panels_registry_includes_credit_balance_and_program_trade_daily() -> None:
    from src.backfill.altdata.config import _ALTDATA_PANELS, AltDataFetchConfig

    assert "credit_balance" in _ALTDATA_PANELS
    assert "program_trade_daily" in _ALTDATA_PANELS
    assert _ALTDATA_PANELS["credit_balance"]["key_cols"] == ("date", "symbol")
    assert _ALTDATA_PANELS["program_trade_daily"]["key_cols"] == ("date", "symbol")

    default_cfg = AltDataFetchConfig.__dataclass_fields__["sources"].default
    assert "credit_balance" in default_cfg
    assert "program_trade_daily" in default_cfg
