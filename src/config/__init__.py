"""도메인별 설정 패키지 (Settings 싱글톤).

각 도메인 모듈(base/kis/gsheet/trading)의 설정을 통합한 `Settings` 싱글톤과
기존 `from src import settings` / `from src.settings import ...` 하위 호환
재수출을 제공합니다.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import computed_field
from pydantic_settings import SettingsConfigDict

from src.config.altdata import AltDataSettings
from src.config.base import PathSettings
from src.config.gsheet import GSheetSettings
from src.config.kis import KisSettings
from src.config.trading import TradingSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(PathSettings, KisSettings, GSheetSettings, TradingSettings, AltDataSettings):
    """프로젝트 전역 설정. `.env` 파일에서 자동 로드.

    도메인별 설정 모듈을 통합한 싱글톤으로, 모든 소비자 모듈은
    `from src import settings` 후 `settings.XXX`로 참조합니다.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------
    # [도메인 간 파생 설정] (통합 싱글톤에서만 정의)
    # ---------------------------------------------------------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def GOOGLE_KEY_PATH(self) -> Path:
        """GSPREAD_KEY_PATH 환경변수를 절대 경로로 변환 (Windows 역슬래시 보정)."""
        raw = self.GSPREAD_KEY_PATH_ENV.strip().replace("\\", "/")
        if not raw:
            return Path("")
        key_path = Path(raw)
        return key_path if key_path.is_absolute() else self.BASE_DIR / key_path

    @computed_field  # type: ignore[prop-decorator]
    @property
    def HISTORY_DB_PATH(self) -> Path:
        return self.HISTORY_DIR / "archive.db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def HISTORY_CSV_PATH(self) -> Path:
        return self.HISTORY_DIR / "archive.csv"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def CONDITION_CSV_PATH(self) -> Path:
        return self.DAILY_DIR / "daily_stocks.csv"


settings = Settings()


# =========================================================
# [모듈 레벨 하위 호환 재수출]
# 기존 소비자 모듈(`from src import settings` 후 `settings.XXX` 참조)을 깨지 않도록
# Singleton 인스턴스의 속성을 모듈 레벨로 재수출합니다.
# =========================================================
BASE_DIR = settings.BASE_DIR
DATA_DIR = settings.DATA_DIR
CONFIGS_DIR = settings.CONFIGS_DIR
MODELS_DIR = settings.MODELS_DIR
KIS_APP_KEY = settings.KIS_APP_KEY
KIS_APP_SECRET = settings.KIS_APP_SECRET
KIS_ACCOUNT_ID = settings.KIS_ACCOUNT_ID
KIS_HTS_ID = settings.KIS_HTS_ID
KIS_BASE_URL = settings.KIS_BASE_URL
KIS_API_CONFIG = settings.KIS_API_CONFIG
GSPREAD_KEY_PATH_ENV = settings.GSPREAD_KEY_PATH_ENV
GSPREAD_SA_JSON = settings.GSPREAD_SA_JSON
GOOGLE_KEY_PATH = settings.GOOGLE_KEY_PATH
GOOGLE_SHEET_NAME = settings.GOOGLE_SHEET_NAME
TRADE_WORKSHEETS = settings.TRADE_WORKSHEETS
THEME_WORKSHEET_NAME = settings.THEME_WORKSHEET_NAME
GOTTEN_COLS = settings.GOTTEN_COLS
TARGET_CONDITION_NAME = settings.TARGET_CONDITION_NAME
OVERHEATED_CONDITION_NAME = settings.OVERHEATED_CONDITION_NAME
NEW_HIGH_CONDITION_NAME = settings.NEW_HIGH_CONDITION_NAME
NEAR_NEW_HIGH_CONDITION_NAME = settings.NEAR_NEW_HIGH_CONDITION_NAME
UPPER_LIMIT_NEXT_DAY_CONDITION_NAME = settings.UPPER_LIMIT_NEXT_DAY_CONDITION_NAME
UPPER_LIMIT_CONDITION_NAME = settings.UPPER_LIMIT_CONDITION_NAME
API_SEMAPHORE_LIMIT = settings.API_SEMAPHORE_LIMIT
EMA_PERIOD = settings.EMA_PERIOD
SMA_PERIOD = settings.SMA_PERIOD
SMA60_PERIOD = settings.SMA60_PERIOD
CANDLE_BODY_RATIO_THRESHOLD = settings.CANDLE_BODY_RATIO_THRESHOLD
GAP_UP_THRESHOLD = settings.GAP_UP_THRESHOLD
SMA_LOOKBACK_DAYS = settings.SMA_LOOKBACK_DAYS
SMA60_LOOKBACK_DAYS = settings.SMA60_LOOKBACK_DAYS
EMA_LOOKBACK_DAYS = settings.EMA_LOOKBACK_DAYS
DEFAULT_SCENARIOS = settings.DEFAULT_SCENARIOS
DAY_NAME_MAP = settings.DAY_NAME_MAP
STOCK_DB_PATH = settings.STOCK_DB_PATH
PARQUET_DIR = settings.PARQUET_DIR
TRADE_LOG_PARQUET_PATH = settings.TRADE_LOG_PARQUET_PATH
THEME_PARQUET_PATH = settings.THEME_PARQUET_PATH
CONDITION_PARQUET_PATH = settings.CONDITION_PARQUET_PATH
TOKEN_FILE = settings.TOKEN_FILE
DAILY_DIR = settings.DAILY_DIR
HISTORY_PARQUET_PATH = settings.HISTORY_PARQUET_PATH
HISTORY_DIR = settings.HISTORY_DIR
HISTORY_DB_PATH = settings.HISTORY_DB_PATH
HISTORY_CSV_PATH = settings.HISTORY_CSV_PATH
PRICE_HISTORY_PARQUET_PATH = settings.PRICE_HISTORY_PARQUET_PATH
LABEL_ENCODER_PATH = settings.LABEL_ENCODER_PATH
MODEL_PATH = settings.MODEL_PATH
CONDITION_CSV_PATH = settings.CONDITION_CSV_PATH
ALTDATA_DIR = settings.ALTDATA_DIR
DART_API_KEY = settings.DART_API_KEY
OPENDART_API_KEY = settings.OPENDART_API_KEY
KRX_OPENAPI_KEY = settings.KRX_OPENAPI_KEY

# 스펙 하위 호환 별칭 (소문자)
base_dir = settings.base_dir
data_dir = settings.data_dir
artifacts_dir = settings.artifacts_dir
models_dir = settings.models_dir
gspread_key_path = settings.gspread_key_path
google_sheet_name = settings.google_sheet_name
kis_app_key = settings.kis_app_key
kis_app_secret = settings.kis_app_secret
kis_account_id = settings.kis_account_id

__all__ = [
    "ALTDATA_DIR",
    "API_SEMAPHORE_LIMIT",
    "BASE_DIR",
    "CANDLE_BODY_RATIO_THRESHOLD",
    "CONDITION_CSV_PATH",
    "CONDITION_PARQUET_PATH",
    "CONFIGS_DIR",
    "DAILY_DIR",
    "DART_API_KEY",
    "DATA_DIR",
    "DAY_NAME_MAP",
    "DEFAULT_SCENARIOS",
    "EMA_LOOKBACK_DAYS",
    "EMA_PERIOD",
    "GAP_UP_THRESHOLD",
    "GOOGLE_KEY_PATH",
    "GOOGLE_SHEET_NAME",
    "GOTTEN_COLS",
    "GSPREAD_KEY_PATH_ENV",
    "GSPREAD_SA_JSON",
    "HISTORY_CSV_PATH",
    "HISTORY_DB_PATH",
    "HISTORY_DIR",
    "HISTORY_PARQUET_PATH",
    "KIS_ACCOUNT_ID",
    "KIS_API_CONFIG",
    "KIS_APP_KEY",
    "KIS_APP_SECRET",
    "KIS_BASE_URL",
    "KIS_HTS_ID",
    "KRX_OPENAPI_KEY",
    "LABEL_ENCODER_PATH",
    "MODELS_DIR",
    "MODEL_PATH",
    "NEAR_NEW_HIGH_CONDITION_NAME",
    "NEW_HIGH_CONDITION_NAME",
    "OPENDART_API_KEY",
    "OVERHEATED_CONDITION_NAME",
    "PARQUET_DIR",
    "PRICE_HISTORY_PARQUET_PATH",
    "SMA60_LOOKBACK_DAYS",
    "SMA60_PERIOD",
    "SMA_LOOKBACK_DAYS",
    "SMA_PERIOD",
    "STOCK_DB_PATH",
    "TARGET_CONDITION_NAME",
    "THEME_PARQUET_PATH",
    "THEME_WORKSHEET_NAME",
    "TOKEN_FILE",
    "TRADE_LOG_PARQUET_PATH",
    "TRADE_WORKSHEETS",
    "UPPER_LIMIT_CONDITION_NAME",
    "UPPER_LIMIT_NEXT_DAY_CONDITION_NAME",
    "AltDataSettings",
    "GSheetSettings",
    "KisSettings",
    "PathSettings",
    "Settings",
    "TradingSettings",
    "artifacts_dir",
    "base_dir",
    "data_dir",
    "google_sheet_name",
    "gspread_key_path",
    "kis_account_id",
    "kis_app_key",
    "kis_app_secret",
    "models_dir",
    "settings",
]
