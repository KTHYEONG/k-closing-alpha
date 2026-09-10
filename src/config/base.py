"""전역 경로 설정 도메인 (PathSettings).

프로젝트 루트 기준 경로(BASE_DIR, DATA_DIR, CONFIGS_DIR, MODELS_DIR)와
경로로부터 파생되는 computed 경로들을 담당합니다.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PathSettings(BaseSettings):
    """프로젝트 루트 경로 및 파생 경로 설정."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------
    # [경로 설정]
    # ---------------------------------------------------------
    BASE_DIR: Path = _PROJECT_ROOT
    DATA_DIR: Path = _PROJECT_ROOT / "data"
    CONFIGS_DIR: Path = _PROJECT_ROOT / "configs"

    # ---------------------------------------------------------
    # [파생 경로]
    # ---------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def MODELS_DIR(self) -> Path:
        return self.BASE_DIR / "artifacts" / "models"
    @computed_field  # type: ignore[prop-decorator]
    @property
    def PARQUET_DIR(self) -> Path:
        return self.DATA_DIR / "parquet"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def TRADE_LOG_PARQUET_PATH(self) -> Path:
        return self.PARQUET_DIR / "trade_log.parquet"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def THEME_PARQUET_PATH(self) -> Path:
        return self.PARQUET_DIR / "theme.parquet"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def DAILY_DIR(self) -> Path:
        return self.DATA_DIR / "daily"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def HISTORY_PARQUET_PATH(self) -> Path:
        return self.HISTORY_DIR / "archive.parquet"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def TOKEN_FILE(self) -> Path:
        return self.CONFIGS_DIR / "kis_token_cache.json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def HISTORY_DIR(self) -> Path:
        return self.DATA_DIR / "history"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ORDERBOOK_DIR(self) -> Path:
        return self.HISTORY_DIR / "orderbook"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def LABEL_ENCODER_PATH(self) -> Path:
        return self.MODELS_DIR / "best_stock_rg_cat_encoders.json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def MODEL_PATH(self) -> Path:
        return self.MODELS_DIR / "best_stock_rg_cat.joblib"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ALTDATA_DIR(self) -> Path:
        return self.HISTORY_DIR / "altdata"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def PRICE_HISTORY_PARQUET_PATH(self) -> Path:
        return self.HISTORY_DIR / "price_history.parquet"
