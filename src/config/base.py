"""전역 경로 설정 도메인 (PathSettings).

프로젝트 루트 기준 경로(BASE_DIR, DATA_DIR, CONFIGS_DIR, MODELS_DIR)와
경로로부터 파생되는 computed 경로들을 담당합니다.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field
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
    # 학습 모델 아티팩트는 artifacts/models/ 로 이관 (데이터 아티팩트와 분리)
    MODELS_DIR: Path = _PROJECT_ROOT / "artifacts" / "models"

    # LS 틱 차트 페이지 예산 (100페이지 ~= 심볼당 ~105초 @1.05초 페이싱)
    LS_TICK_MAX_PAGES: int = Field(default=100)

    # ---------------------------------------------------------
    # [파생 경로]
    # ---------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def STOCK_DB_PATH(self) -> Path:
        return self.DATA_DIR / "stock.db"

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
    def CONDITION_PARQUET_PATH(self) -> Path:
        return self.DAILY_DIR / "daily_stocks.parquet"

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

    # ---------------------------------------------------------
    # [스펙 하위 호환 별칭] (spec: base_dir, data_dir, models_dir, ...)
    # ---------------------------------------------------------
    @property
    def base_dir(self) -> Path:
        return self.BASE_DIR

    @property
    def data_dir(self) -> Path:
        return self.DATA_DIR

    @property
    def artifacts_dir(self) -> Path:
        return self.BASE_DIR / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.MODELS_DIR
