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
