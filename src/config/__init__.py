"""도메인별 설정 패키지 (Settings 싱글톤).

각 도메인 모듈(base/kis/trading)의 설정을 통합한 `Settings` 싱글톤과
기존 `from src import settings` / `from src.settings import ...` 하위 호환
재수출을 제공합니다.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import SettingsConfigDict

from src.config.altdata import AltDataSettings
from src.config.base import PathSettings
from src.config.kis import KisSettings
from src.config.kiwoom import KiwoomSettings
from src.config.ls import LsSettings
from src.config.trading import TradingSettings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(PathSettings, KisSettings, LsSettings, TradingSettings, AltDataSettings, KiwoomSettings):
    """프로젝트 전역 설정. `.env` 파일에서 자동 로드.

    도메인별 설정 모듈을 통합한 싱글톤으로, 모든 소비자 모듈은
    `from src import settings` 후 `settings.XXX`로 참조합니다.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )



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
LS_APP_KEY = settings.LS_APP_KEY
LS_APP_SECRET = settings.LS_APP_SECRET
LS_BASE_URL = settings.LS_BASE_URL
KIWOM_APP_KEY = settings.KIWOM_APP_KEY
KIWOM_SECRET_KEY = settings.KIWOM_SECRET_KEY
KIWOM_BASE_URL = settings.KIWOM_BASE_URL
KIWOM_TICK_MAX_PAGES = settings.KIWOM_TICK_MAX_PAGES
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
PARQUET_DIR = settings.PARQUET_DIR
TRADE_LOG_PARQUET_PATH = settings.TRADE_LOG_PARQUET_PATH
THEME_PARQUET_PATH = settings.THEME_PARQUET_PATH
TOKEN_FILE = settings.TOKEN_FILE
DAILY_DIR = settings.DAILY_DIR
HISTORY_PARQUET_PATH = settings.HISTORY_PARQUET_PATH
HISTORY_DIR = settings.HISTORY_DIR
ORDERBOOK_DIR = settings.ORDERBOOK_DIR
PAPER_DIR = settings.PAPER_DIR
PAPER_SEED_CAPITAL = settings.PAPER_SEED_CAPITAL
LS_TICK_MAX_PAGES = settings.LS_TICK_MAX_PAGES
PRICE_HISTORY_PARQUET_PATH = settings.PRICE_HISTORY_PARQUET_PATH
LABEL_ENCODER_PATH = settings.LABEL_ENCODER_PATH
MODEL_PATH = settings.MODEL_PATH
ALTDATA_DIR = settings.ALTDATA_DIR
DART_API_KEY = settings.DART_API_KEY
OPENDART_API_KEY = settings.OPENDART_API_KEY
KRX_OPENAPI_KEY = settings.KRX_OPENAPI_KEY
KRX_OPENAPI_BASE_URL = settings.KRX_OPENAPI_BASE_URL
KRX_OPENAPI_BASE_URLS = settings.KRX_OPENAPI_BASE_URLS
KRX_OPENAPI_ENDPOINTS = settings.KRX_OPENAPI_ENDPOINTS

__all__ = [
    "ALTDATA_DIR",
    "API_SEMAPHORE_LIMIT",
    "BASE_DIR",
    "CANDLE_BODY_RATIO_THRESHOLD",
    "CONFIGS_DIR",
    "DAILY_DIR",
    "DART_API_KEY",
    "DATA_DIR",
    "DAY_NAME_MAP",
    "DEFAULT_SCENARIOS",
    "EMA_LOOKBACK_DAYS",
    "EMA_PERIOD",
    "GAP_UP_THRESHOLD",
    "HISTORY_DIR",
    "HISTORY_PARQUET_PATH",
    "KIS_ACCOUNT_ID",
    "KIS_API_CONFIG",
    "KIS_APP_KEY",
    "KIS_APP_SECRET",
    "KIS_BASE_URL",
    "KIS_HTS_ID",
    "KIWOM_APP_KEY",
    "KIWOM_BASE_URL",
    "KIWOM_SECRET_KEY",
    "KIWOM_TICK_MAX_PAGES",
    "KRX_OPENAPI_BASE_URL",
    "KRX_OPENAPI_BASE_URLS",
    "KRX_OPENAPI_ENDPOINTS",
    "KRX_OPENAPI_KEY",
    "LABEL_ENCODER_PATH",
    "LS_APP_KEY",
    "LS_APP_SECRET",
    "LS_BASE_URL",
    "LS_TICK_MAX_PAGES",
    "MODELS_DIR",
    "MODEL_PATH",
    "NEAR_NEW_HIGH_CONDITION_NAME",
    "NEW_HIGH_CONDITION_NAME",
    "OPENDART_API_KEY",
    "ORDERBOOK_DIR",
    "OVERHEATED_CONDITION_NAME",
    "PAPER_DIR",
    "PAPER_SEED_CAPITAL",
    "PARQUET_DIR",
    "PRICE_HISTORY_PARQUET_PATH",
    "SMA60_LOOKBACK_DAYS",
    "SMA60_PERIOD",
    "SMA_LOOKBACK_DAYS",
    "SMA_PERIOD",
    "TARGET_CONDITION_NAME",
    "THEME_PARQUET_PATH",
    "TOKEN_FILE",
    "TRADE_LOG_PARQUET_PATH",
    "UPPER_LIMIT_CONDITION_NAME",
    "UPPER_LIMIT_NEXT_DAY_CONDITION_NAME",
    "AltDataSettings",
    "KisSettings",
    "KiwoomSettings",
    "LsSettings",
    "PathSettings",
    "Settings",
    "TradingSettings",
    "settings",
]
