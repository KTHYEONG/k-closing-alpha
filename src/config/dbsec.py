"""DB Securities OpenAPI settings domain (DbsecSettings)."""

from __future__ import annotations

from pydantic import Field

from src.config._env import EnvSettings


class DbsecSettings(EnvSettings):
    """DB Securities OpenAPI connection, pacing and paging settings."""

    DB_APP_KEY: str = Field(default="")
    DB_APP_SECRET: str = Field(default="")
    DB_BASE_URL: str = "https://openapi.dbsec.co.kr:8443"
    DB_RATE_LIMIT_MAX_RETRIES: int = Field(default=5, ge=1)
    DB_RATE_LIMIT_BACKOFF_SECONDS: float = Field(default=1.2, gt=0.0)
    DB_AUCTION_PATH_ENABLED: bool = Field(default=False)
    DB_AUCTION_PATH_MAX_PAGES: int = Field(default=60, ge=1)
    DB_AUCTION_PATH_DEADLINE_HHMMSS: str = Field(default="190000", pattern=r"^\d{6}$")
    DB_AUCTION_PATH_CONCURRENCY: int = Field(default=2, ge=1)
