"""프로젝트 전역 설정 (pydantic-settings Singleton).

모듈 임포트 시점의 글로벌 사이드 이펙트(폴더 생성, load_dotenv 등)를 완전히 제거하고,
`Settings` Singleton 인스턴스로 모든 설정을 타입 안전하게 관리합니다.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """프로젝트 전역 설정. `.env` 파일에서 자동 로드.

    모든 소비자 모듈은 `from src import settings` 후 `settings.XXX`로 참조하며,
    개별 파일에서 상대 경로를 하드코딩하지 않습니다.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================================================
    # [경로 설정]
    # =========================================================
    BASE_DIR: Path = _PROJECT_ROOT
    DATA_DIR: Path = _PROJECT_ROOT / "data"
    CONFIGS_DIR: Path = _PROJECT_ROOT / "configs"
    # 학습 모델 아티팩트는 artifacts/models/ 로 이관 (데이터 아티팩트와 분리)
    MODELS_DIR: Path = _PROJECT_ROOT / "artifacts" / "models"

    # KIS API
    KIS_APP_KEY: str = Field(default="")
    KIS_APP_SECRET: str = Field(default="")
    KIS_ACCOUNT_ID: str = Field(default="")
    KIS_HTS_ID: str = Field(default="")
    KIS_BASE_URL: str = "https://openapi.koreainvestment.com:9443"

    # Google Sheets
    GSPREAD_KEY_PATH_ENV: str = Field(default="", validation_alias="GSPREAD_KEY_PATH")
    GSPREAD_SA_JSON: str = Field(default="")
    GOOGLE_SHEET_NAME: str = "Stock"
    TRADE_WORKSHEETS: list[str] = ["Trade", "Trade2"]
    THEME_WORKSHEET_NAME: str = "코드_테마_DB"

    # Google Sheets 컬럼명 설정
    GOTTEN_COLS: dict[str, str] = {
        "DATE": "(매수날짜)",
        "CODE": "(종목코드)",
        "PROGRAM": "(프로그램_순매수)",
        "INST": "(기관_순매수)",
        "FOREIGN": "(외국인_순매수)",
        "V_KOSPI": "(v-kospi)",
        "V_KOSDAQ": "(v-kosdaq)",
    }

    # =========================================================
    # [데이터 수집 설정 (collect_data)]
    # =========================================================
    TARGET_CONDITION_NAME: str = "종가매매"
    OVERHEATED_CONDITION_NAME: str = "단기과열"
    NEW_HIGH_CONDITION_NAME: str = "신고가"
    NEAR_NEW_HIGH_CONDITION_NAME: str = "신고가 근접"
    UPPER_LIMIT_NEXT_DAY_CONDITION_NAME: str = "상한가 다음날"
    UPPER_LIMIT_CONDITION_NAME: str = "상한가"

    # API 요청 제한 및 지연 시간
    API_SEMAPHORE_LIMIT: int = 4
    API_SLEEP_INTERVAL: float = 0.2

    # 차트 필터링 설정
    EMA_PERIOD: int = 20
    SMA_PERIOD: int = 120
    SMA60_PERIOD: int = 60
    CANDLE_BODY_RATIO_THRESHOLD: float = 0.5
    GAP_UP_THRESHOLD: float = 0.1
    SMA_LOOKBACK_DAYS: int = 200
    SMA60_LOOKBACK_DAYS: int = 120
    EMA_LOOKBACK_DAYS: int = 60

    # AI 분석 기본 시나리오 (시트에서 새로운 유형 기록시 추가 필요)
    DEFAULT_SCENARIOS: list[str] = [
        "신고가",
        "상따",
        "신고가 근접",
        "거래량 폭증",
        "상한가 다음날",
        "120 돌파",
        "상승형 음봉",
    ]

    # 한글 요일 매핑
    DAY_NAME_MAP: dict[int, str] = {
        0: "월요일",
        1: "화요일",
        2: "수요일",
        3: "목요일",
        4: "금요일",
        5: "토요일",
        6: "일요일",
    }

    # =========================================================
    # [파생 경로]
    # =========================================================
    @computed_field  # type: ignore[prop-decorator]
    @property
    def KIS_API_CONFIG(self) -> dict[str, str]:
        """KIS API 접속 정보 딕셔너리 (app_key/app_secret/account_id/hts_id)."""
        return {
            "app_key": self.KIS_APP_KEY,
            "app_secret": self.KIS_APP_SECRET,
            "account_id": self.KIS_ACCOUNT_ID,
            "hts_id": self.KIS_HTS_ID,
        }

    @computed_field  # type: ignore[prop-decorator]
    @property
    def STOCK_DB_PATH(self) -> Path:
        return self.DATA_DIR / "stock.db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def TOKEN_FILE(self) -> Path:
        return self.CONFIGS_DIR / "kis_token_cache.json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def CHART_PASS_CACHE_FILE(self) -> Path:
        return self.DATA_DIR / "chart_pass_cache.json"

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
    def HISTORY_DIR(self) -> Path:
        return self.DATA_DIR / "history"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def HISTORY_DB_PATH(self) -> Path:
        return self.HISTORY_DIR / f"condition_history_{self.TARGET_CONDITION_NAME}.db"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def HISTORY_CSV_PATH(self) -> Path:
        return self.HISTORY_DIR / f"condition_history_{self.TARGET_CONDITION_NAME}.csv"

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
    def CONDITION_EXCEL_PATH(self) -> Path:
        return self.DATA_DIR / f"condition_{self.TARGET_CONDITION_NAME}.xlsx"

    # =========================================================
    # [스펙 하위 호환 별칭] (spec: base_dir, data_dir, models_dir, ...)
    # =========================================================
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

    @property
    def gspread_key_path(self) -> str:
        return self.GSPREAD_KEY_PATH_ENV

    @property
    def google_sheet_name(self) -> str:
        return self.GOOGLE_SHEET_NAME

    @property
    def kis_app_key(self) -> str:
        return self.KIS_APP_KEY

    @property
    def kis_app_secret(self) -> str:
        return self.KIS_APP_SECRET

    @property
    def kis_account_id(self) -> str:
        return self.KIS_ACCOUNT_ID


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
API_SLEEP_INTERVAL = settings.API_SLEEP_INTERVAL
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
TOKEN_FILE = settings.TOKEN_FILE
CHART_PASS_CACHE_FILE = settings.CHART_PASS_CACHE_FILE
HISTORY_DIR = settings.HISTORY_DIR
HISTORY_DB_PATH = settings.HISTORY_DB_PATH
HISTORY_CSV_PATH = settings.HISTORY_CSV_PATH
LABEL_ENCODER_PATH = settings.LABEL_ENCODER_PATH
MODEL_PATH = settings.MODEL_PATH
CONDITION_EXCEL_PATH = settings.CONDITION_EXCEL_PATH

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
