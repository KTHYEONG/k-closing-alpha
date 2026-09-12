"""토스증권 OpenAPI 설정 도메인 (TossSettings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TossSettings(BaseSettings):
    """토스증권 OpenAPI 접속 설정."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    TOSS_APP_KEY: str = Field(default="")
    TOSS_APP_SECRET: str = Field(default="")
    TOSS_BASE_URL: str = "https://openapi.tossinvest.com"
