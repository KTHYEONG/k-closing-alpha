"""KIS order-routing settings domain (KisTradingSettings)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import SettingsConfigDict

from src.config._env import EnvSettings


class KisTradingSettings(EnvSettings):
    """KIS order-routing environment selector, mock credentials and order risk limits."""

    model_config = SettingsConfigDict(hide_input_in_errors=True)

    KIS_TRADING_ENV: Literal["off", "mock", "live"] = "off"
    KIS_MOCK_BASE_URL: str = "https://openapivts.koreainvestment.com:29443"
    KIS_MOCK_APP_KEY: str = Field(default="", repr=False)
    KIS_MOCK_APP_SECRET: str = Field(default="", repr=False)
    KIS_MOCK_ACCOUNT_NO: str = Field(default="", repr=False)
    KIS_MOCK_ACCOUNT_PRODUCT_CODE: str = "01"
    KIS_ACCOUNT_PRODUCT_CODE: str = "01"
    KIS_LIVE_TRADING_CONFIRM: str = ""
    KIS_MOCK_REST_TPS: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    ORDER_SEED_CAPITAL: int = Field(default=10_000_000, gt=0)
    ORDER_ENTRY_LIMIT_BUFFER_BP: float = Field(default=300.0, ge=0.0, le=3000.0, allow_inf_nan=False)
    ORDER_MAX_NOTIONAL_PER_ORDER_KRW: int = Field(default=5_000_000, gt=0)
    ORDER_MAX_SENDS_PER_RUN: int = Field(default=10, gt=0)
    ORDER_SEND_MAX_ATTEMPTS: int = Field(default=3, ge=1)
    ORDER_HTTP_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0, allow_inf_nan=False)
    ORDER_UNKNOWN_RESOLVE_POLLS: int = Field(default=3, ge=1)
    ORDER_UNKNOWN_RESOLVE_INTERVAL_SECONDS: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    ORDER_EXIT_RESEND_ON_AMBIGUOUS: bool = False
    ORDER_JOURNAL_LOCK_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _cap_within_seed(self) -> KisTradingSettings:
        if self.ORDER_MAX_NOTIONAL_PER_ORDER_KRW > self.ORDER_SEED_CAPITAL:
            raise ValueError("ORDER_MAX_NOTIONAL_PER_ORDER_KRW must not exceed ORDER_SEED_CAPITAL")
        return self
