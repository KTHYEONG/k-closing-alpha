"""LS증권 OpenAPI 설정 도메인 (LsSettings)."""

from __future__ import annotations

from pydantic import Field

from src.config._env import EnvSettings


class LsSettings(EnvSettings):
    """LS증권 OpenAPI 접속 설정."""

    LS_APP_KEY: str = Field(default="")
    LS_APP_SECRET: str = Field(default="")
    LS_BASE_URL: str = "https://openapi.ls-sec.co.kr:8080"
    LS_MIN_INTERVAL_SECONDS: float = Field(default=1.05, gt=0.0)
    LS_RATE_LIMIT_MAX_RETRIES: int = Field(default=5, ge=1)
    LS_RATE_LIMIT_BACKOFF_SECONDS: float = Field(default=1.2, gt=0.0)
