import os
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
OVERHEATED_CONDITION_NAME = "단기과열"  # 단기과열 조건검색명 (예고+본지정)
NEW_HIGH_CONDITION_NAME = "신고가"
NEAR_NEW_HIGH_CONDITION_NAME = "신고가 근접"
UPPER_LIMIT_NEXT_DAY_CONDITION_NAME = "상한가 다음날"
UPPER_LIMIT_CONDITION_NAME = "상한가"  # 상한가 조건검색명
CHART_PASS_CACHE_FILE = DATA_DIR / "chart_pass_cache.json"

# API 요청 제한 및 지연 시간
API_SEMAPHORE_LIMIT = 4  # 동시 요청 수 (기본: 2 → 6 사용 시 ServerDisconnectedError 발생 → 4로 안정화)
API_SLEEP_INTERVAL = 0.2  # 요청 간 대기 시간(초) (기본: 0.2 → 최적화: 0.05)
# 성능: 약 3배 빠름 / 안전성: KIS API 제한(초당 20~30요청) 내에서 안전
# Semaphore=4 + 종목당 4 API = 최대 16개 동시 요청 (안전 범위)

# 차트 필터링 설정
EMA_PERIOD = 20  # EMA 기간 (신규 종목 필터링)
SMA_PERIOD = 120  # SMA 기간 (이평선 필터링)
SMA60_PERIOD = 60  # SMA60 기간
CANDLE_BODY_RATIO_THRESHOLD = 0.5  # 캔들 몸통 비율 임계값 (50%)
GAP_UP_THRESHOLD = 0.1  # 시가 갭 상승 예외 기준 (10% 이상 갭 상승 시 캔들 몸통 필터 무시)
SMA_LOOKBACK_DAYS = 200  # SMA 120 계산을 위한 과거 조회일수
SMA60_LOOKBACK_DAYS = 120  # SMA 60 계산을 위한 과거 조회일수
EMA_LOOKBACK_DAYS = 60  # EMA 계산을 위한 과거 조회일수

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
TRADE_WORKSHEETS = ["Trade", "Trade2"]
THEME_WORKSHEET_NAME = "코드_테마_DB"

# Google Sheets 컬럼명 설정
GOTTEN_COLS = {
    "DATE": "(매수날짜)",
    "CODE": "(종목코드)",
    "PROGRAM": "(프로그램_순매수)",
    "INST": "(기관_순매수)",
    "FOREIGN": "(외국인_순매수)",
    "V_KOSPI": "(v-kospi)",
    "V_KOSDAQ": "(v-kosdaq)"
}

# =========================================================
# [AI 분석 설정 (Daily_Pos_AI)]
# =========================================================
LABEL_ENCODER_PATH = MODELS_DIR / "best_stock_rg_cat_encoders.json"
MODEL_PATH = MODELS_DIR / "best_stock_rg_cat.joblib"
CONDITION_EXCEL_PATH = DATA_DIR / f"condition_{TARGET_CONDITION_NAME}.xlsx"

# AI 분석 기본 시나리오 (시트에서 새로운 유형 기록시 추가 필요)
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
