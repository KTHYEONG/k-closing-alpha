from __future__ import annotations




def test_market_session_premarket_constants() -> None:
    from src.config.market_session import (
        INTRADAY_SESSION_NXT_PREMARKET,
        NXT_PREMARKET_HOUR_CEIL,
        NXT_PREMARKET_HOUR_FLOOR,
    )

    assert INTRADAY_SESSION_NXT_PREMARKET == "nxt_premarket"
    assert NXT_PREMARKET_HOUR_FLOOR == "080000"
    assert NXT_PREMARKET_HOUR_CEIL == "085000"


def test_toss_stamp_convention() -> None:
    from src.config.market_session import BAR_STAMP_END, INTRADAY_BAR_STAMP_CONVENTION

    assert INTRADAY_BAR_STAMP_CONVENTION["toss"] == BAR_STAMP_END
    assert INTRADAY_BAR_STAMP_CONVENTION["kis"] == "start"
    assert INTRADAY_BAR_STAMP_CONVENTION["kiwoom"] == "start"
    assert INTRADAY_BAR_STAMP_CONVENTION["ls"] == "end"


def test_route_certification() -> None:
    from src.config.market_session import VERIFIED_CHART_ROUTES

    assert VERIFIED_CHART_ROUTES["toss:toss-candles"] == "KRX"
    assert VERIFIED_CHART_ROUTES["ls:t8412"] == "KRX"
    assert VERIFIED_CHART_ROUTES["kiwoom:ka10079"] == "KRX"
