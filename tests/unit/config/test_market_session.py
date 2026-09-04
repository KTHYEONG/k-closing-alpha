from __future__ import annotations

from src.config import market_session


def test_market_session_constants_cover_regular_and_aftermarket_windows() -> None:
    assert market_session.KRX_CLOSE_MARKET_DIV_CODE == "J"
    assert market_session.NXT_MARKET_DIV_CODE == "NX"
    assert market_session.DECISION_PRICE_MARKET_DIV_CODES == ("J", "NX")
    assert market_session.DEFAULT_BAR_INTERVAL_MINUTES == 1
    assert market_session.INTRADAY_SESSION_REGULAR == "regular"
    assert market_session.INTRADAY_SESSION_NXT_AFTERMARKET == "nxt_aftermarket"
    assert market_session.KRX_REGULAR_HOUR_FLOOR == "090000"
    assert market_session.KRX_REGULAR_HOUR_CEIL == "153000"
    assert market_session.NXT_AFTERMARKET_HOUR_FLOOR == "154000"
    assert market_session.NXT_AFTERMARKET_HOUR_CEIL == "200000"
