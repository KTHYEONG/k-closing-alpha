import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# =========================================================
# [경로 설정]
# =========================================================
# 프로젝트 루트 디렉토리 설정 (src 폴더의 부모 폴더)
BASE_DIR = Path(__file__).resolve().parent.parent

# 데이터 및 설정 폴더 경로
DATA_DIR = BASE_DIR / "data"
CONFIGS_DIR = BASE_DIR / "configs"
MODELS_DIR = BASE_DIR / "models"

# DB 경로
STOCK_DB_PATH = DATA_DIR / "stock.db"
# 필요한 폴더들이 없으면 생성
for folder in [DATA_DIR, CONFIGS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================================================
# [API 설정]
# =========================================================
# KIS API 정보 (configs/kis_config.py에서 우선적으로 가져옴)
try:
    import configs.kis_config as kis_config
    KIS_API_CONFIG = kis_config.real_investment
except (ImportError, AttributeError):
    KIS_API_CONFIG = {
        "app_key": os.getenv("KIS_APP_KEY", ""),
        "app_secret": os.getenv("KIS_APP_SECRET", ""),
        "account_id": os.getenv("KIS_ACCOUNT_ID", ""),
        "hts_id": os.getenv("KIS_HTS_ID", "")
    }

KIS_BASE_URL = "https://openapi.koreainvestment.com:9443"
TOKEN_FILE = CONFIGS_DIR / "kis_token_cache.json"

# =========================================================
# [데이터 수집 설정 (Daily_Get_Data)]
# =========================================================
TARGET_CONDITION_NAME = "종가매매"
CHART_PASS_CACHE_FILE = DATA_DIR / "chart_pass_cache.json"

# API 요청 제한 및 지연 시간
API_SEMAPHORE_LIMIT = 2
API_SLEEP_INTERVAL = 0.2

# 히스토리 관리 설정
HISTORY_DIR = DATA_DIR / "history"
HISTORY_DB_PATH = HISTORY_DIR / f"condition_history_{TARGET_CONDITION_NAME}.db"
HISTORY_CSV_PATH = HISTORY_DIR / f"condition_history_{TARGET_CONDITION_NAME}.csv"

# =========================================================
# [구글 시트 설정]
# =========================================================
# .env 파일에서 GSPREAD_KEY_PATH를 가져와 절대 경로로 변환
GSPREAD_KEY_PATH_ENV = os.getenv("GSPREAD_KEY_PATH", "")
if GSPREAD_KEY_PATH_ENV and not os.path.isabs(GSPREAD_KEY_PATH_ENV):
    GOOGLE_KEY_PATH = BASE_DIR / GSPREAD_KEY_PATH_ENV
else:
    GOOGLE_KEY_PATH = Path(GSPREAD_KEY_PATH_ENV)

GOOGLE_SHEET_NAME = "Stock"
TRADE2_WORKSHEET_NAME = "Trade2"

# Google Sheets 컬럼명 설정
GOTTEN_COLS = {
    "DATE": "(매수날짜)",
    "CODE": "(종목코드)",
    "PROGRAM": "(프로그램_순매수)",
    "INST": "(기관_순매수)",
    "FOREIGN": "(외국인_순매수)"
}

# =========================================================
# [AI 분석 설정 (Daily_Pos_AI)]
# =========================================================
LABEL_ENCODER_PATH = MODELS_DIR / "best_stock_rg_label_encoders.json"
MODEL_PATH = MODELS_DIR / "best_stock_rg.joblib"
CONDITION_EXCEL_PATH = DATA_DIR / f"condition_{TARGET_CONDITION_NAME}.xlsx"

DEFAULT_SCENARIOS = [
    "신고가",
    "상따",
    "신고가 근접",
    "거래량 폭증",
    "상한가 다음날",
    "120 돌파",
    "상승형 음봉",
]

# 한글 요일 매핑
DAY_NAME_MAP = {
    0: "월요일",
    1: "화요일",
    2: "수요일",
    3: "목요일",
    4: "금요일",
    5: "토요일",
    6: "일요일",
}
