"""토스증권 OpenAPI 설정 도메인 (TossSettings)."""

from __future__ import annotations

from pydantic import Field

from src.config._env import EnvSettings


class TossSettings(EnvSettings):
    """토스증권 OpenAPI 접속 설정."""

    TOSS_APP_KEY: str = Field(default="")
    TOSS_APP_SECRET: str = Field(default="")
    TOSS_BASE_URL: str = "https://openapi.tossinvest.com"
    TOSS_RATE_LIMIT_MAX_RETRIES: int = Field(default=3, ge=1)
    """Rate-limit attempts per GET (HTTP 429). An auth-refresh replay does not consume one."""
    TOSS_RATE_LIMIT_BACKOFF_SECONDS: float = Field(default=1.2, gt=0.0)
    """Wait after a Toss 429 that carries no usable Retry-After header."""
