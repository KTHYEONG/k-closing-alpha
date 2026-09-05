"""LS증권 OpenAPI 설정 도메인 (LsSettings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class LsSettings(BaseSettings):
    """LS증권 OpenAPI 접속 설정."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LS_APP_KEY: str = Field(default="")
    LS_APP_SECRET: str = Field(default="")
    LS_BASE_URL: str = "https://openapi.ls-sec.co.kr:8080"
