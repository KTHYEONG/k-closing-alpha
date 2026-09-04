"""장 구분/세션 상수 (KRX 정규세션, NXT 애프터마켓, 의사결정 듀얼벤뉴)."""

from __future__ import annotations

KRX_CLOSE_MARKET_DIV_CODE: str = "J"
NXT_MARKET_DIV_CODE: str = "NX"
DECISION_PRICE_MARKET_DIV_CODES: tuple[str, str] = ("J", "NX")
DEFAULT_BAR_INTERVAL_MINUTES: int = 1
INTRADAY_SESSION_REGULAR: str = "regular"
INTRADAY_SESSION_NXT_AFTERMARKET: str = "nxt_aftermarket"
KRX_REGULAR_HOUR_FLOOR: str = "090000"
KRX_REGULAR_HOUR_CEIL: str = "153000"
NXT_AFTERMARKET_HOUR_FLOOR: str = "154000"
NXT_AFTERMARKET_HOUR_CEIL: str = "200000"
