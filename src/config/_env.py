"""Shared pydantic-settings base for domain settings classes."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class EnvSettings(BaseSettings):
    """Shared pydantic-settings base that loads the project-root `.env` file.

    Every domain settings class inherits this base so the env-file location, encoding and
    `extra="ignore"` policy are declared once. `extra="ignore"` is required: the runtime env files
    carry keys owned by other consumers (KIS key pools, systemd), and the combined `Settings` class
    must not reject them.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
