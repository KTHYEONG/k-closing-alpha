"""키움증권 REST API 설정 도메인 (KiwoomSettings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class KiwoomSettings(BaseSettings):
    """키움증권 REST API 접속 설정."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    KIWOM_APP_KEY: str = Field(default="")
    KIWOM_SECRET_KEY: str = Field(default="")
    KIWOM_BASE_URL: str = "https://api.kiwoom.com"
    KIWOM_TICK_MAX_PAGES: int = Field(default=30)
