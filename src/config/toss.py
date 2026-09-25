"""토스증권 OpenAPI 설정 도메인 (TossSettings)."""

from __future__ import annotations

from pydantic import Field

from src.config._env import EnvSettings


class TossSettings(EnvSettings):
    """토스증권 OpenAPI 접속 설정."""

    TOSS_APP_KEY: str = Field(default="")
    TOSS_APP_SECRET: str = Field(default="")
    TOSS_BASE_URL: str = "https://openapi.tossinvest.com"
