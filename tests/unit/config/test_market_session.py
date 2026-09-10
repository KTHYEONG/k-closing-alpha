from __future__ import annotations

from src.config import market_session


def test_market_session_constants_cover_regular_and_aftermarket_windows() -> None:
    assert market_session.KRX_CLOSE_MARKET_DIV_CODE == "J"
    assert market_session.NXT_MARKET_DIV_CODE == "NX"
    assert not hasattr(market_session, "DECISION_PRICE_MARKET_DIV_CODES")
    assert market_session.DECISION_WINDOW_START_HHMMSS == "152000"
    assert market_session.DECISION_WINDOW_END_HHMMSS == "153000"
    assert market_session.DEFAULT_BAR_INTERVAL_MINUTES == 1
    assert market_session.INTRADAY_SESSION_REGULAR == "regular"
    assert market_session.INTRADAY_SESSION_NXT_AFTERMARKET == "nxt_aftermarket"
    assert market_session.KRX_REGULAR_HOUR_FLOOR == "090000"
    assert market_session.KRX_REGULAR_HOUR_CEIL == "153000"
    assert market_session.NXT_AFTERMARKET_HOUR_FLOOR == "154000"
    assert market_session.NXT_AFTERMARKET_HOUR_CEIL == "200000"


def test_session_constants_have_one_definition_without_import_cycle() -> None:
    import inspect

    from src.config import market_session
    from src.execution import cost_model
    from src.ml import buyability

    # Then: the int forms derive from the string forms so they cannot drift.
    assert market_session.DECISION_WINDOW_START_HMS == 152000
    assert market_session.DECISION_WINDOW_END_HMS == 153000
    assert int(market_session.DECISION_WINDOW_START_HHMMSS) == market_session.DECISION_WINDOW_START_HMS
    assert int(market_session.DECISION_WINDOW_END_HHMMSS) == market_session.DECISION_WINDOW_END_HMS

    # And: both private duplicates are gone.
    assert not hasattr(cost_model, "_AUCTION_CLOSE_HMS")
    assert not hasattr(buyability, "_AUCTION_CLOSE_HMS")

    # And: cost_model must not import strategy.contract - contract already imports cost_model.
    cost_model_source = inspect.getsource(cost_model)
    assert "from src.strategy.contract" not in cost_model_source
    assert "from src.config.market_session" in cost_model_source

    # And: the keyword defaults now come from the session module.
    assert (
        inspect.signature(buyability.attach_entry_auction_liquidity).parameters["auction_start_hms"].default
        == market_session.DECISION_WINDOW_START_HMS
    )


def test_krx_aftermarket_session_constants_are_disjoint_from_regular() -> None:
    from src.config.market_session import (
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
        INTRADAY_SESSION_REGULAR,
        KRX_AFTERMARKET_HOUR_CEIL,
        KRX_AFTERMARKET_HOUR_FLOOR,
        KRX_AFTERMARKET_START_DATE,
        KRX_REGULAR_HOUR_CEIL,
    )

    assert KRX_AFTERMARKET_HOUR_FLOOR == "160000"
    assert KRX_AFTERMARKET_HOUR_CEIL == "200000"
    assert KRX_AFTERMARKET_HOUR_FLOOR > KRX_REGULAR_HOUR_CEIL
    assert KRX_AFTERMARKET_START_DATE == "2026-09-14"
    assert INTRADAY_SESSION_KRX_AFTERMARKET not in (INTRADAY_SESSION_REGULAR, INTRADAY_SESSION_NXT_AFTERMARKET)
