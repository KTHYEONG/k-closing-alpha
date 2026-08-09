"""KIS API 호환성 퍼사드 모듈.

구현은 ``src.api.kis.client`` (HTTP/토큰/요청), ``src.api.kis.rate_limit``
(AsyncRateLimiter), ``src.api.kis.indicators`` (SMA/EMA/변동성) 로 분리되었으며,
이 모듈은 공개 심볼만 재-export 합니다. 중복 구현이 없고 마이그레이션 기간 동안
기존 import 경로를 보장합니다.
"""

from __future__ import annotations

from src.api.kis.client import KisApiClient
from src.api.kis.indicators import (
    calculate_all_moving_averages,
    calculate_multiple_emas,
    calculate_stock_ema,
    calculate_stock_sma,
    fetch_index_and_calculate_volatility,
    fetch_kospi200_and_calculate_vkospi,
    prefetch_ohlcv_for_sma120,
)
from src.api.kis.rate_limit import AsyncRateLimiter

__all__ = [
    "AsyncRateLimiter",
    "KisApiClient",
    "calculate_all_moving_averages",
    "calculate_multiple_emas",
    "calculate_stock_ema",
    "calculate_stock_sma",
    "fetch_index_and_calculate_volatility",
    "fetch_kospi200_and_calculate_vkospi",
    "prefetch_ohlcv_for_sma120",
]
