import json
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import datetime
import unicodedata

from joblib import load

# 구글 시트 연동을 위한 라이브러리
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# .env 파일에서 환경변수를 불러오기 위한 라이브러리
from dotenv import load_dotenv


# 터미널 색상 코드
class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    GRAY = "\033[90m"


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "best_stock_rg_label_encoders.json")


def load_label_encoder_map(path):
    """Load saved label encoder classes and build mapping."""
    if not os.path.exists(path):
        print(
            f"{Colors.YELLOW}[Warning] Label encoder file not found: {path}. "
            "Categorical encoding may fail during inference."
            f"{Colors.RESET}"
        )
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as exc:
        print(
            f"{Colors.RED}[Error] Failed to read label encoder file: {exc}{Colors.RESET}"
        )
        return {}

    encoder_map = {}
    for col, classes in raw_data.items():
        mapping = {str(cls): idx for idx, cls in enumerate(classes)}
        unknown_idx = mapping.get("Unknown", len(mapping))
        encoder_map[col] = {"mapping": mapping, "unknown": unknown_idx}
    print(
        f"{Colors.CYAN}Loaded label encoders for columns: {list(encoder_map.keys())}{Colors.RESET}"
    )
    return encoder_map


LABEL_ENCODER_MAP = load_label_encoder_map(LABEL_ENCODER_PATH)


def apply_label_encodings(df, encoder_map):
    """Apply label encoding mappings to categorical columns in-place."""
    if not encoder_map:
        object_cols = df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            df[col] = pd.Categorical(df[col]).codes.astype(float)
        return df

    for col, info in encoder_map.items():
        if col not in df.columns:
            continue
        mapping = info["mapping"]
        unknown_idx = info["unknown"]
        df[col] = (
            df[col]
            .astype(str)
            .apply(lambda val: mapping.get(val, unknown_idx))
            .astype(float)
        )
    return df


# 1. 분석할 조건검색 결과 파일 (data 폴더)
CONDITION_EXCEL_PATH = os.path.join(DATA_DIR, "condition_종가매매.xlsx")

# 2. AI 모델 파일 (models 폴더)
MODEL_PATH = os.path.join(MODELS_DIR, "best_stock_rg.cbm")

# .env 파일에서 환경변수 로드
env_file_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(env_file_path)

# 3. 구글 시트 인증 키 파일 (환경변수 사용)
google_key_env = os.getenv("GSPREAD_KEY_PATH")
if not google_key_env:
    print(
        f"{Colors.RED}[Error] 'GSPREAD_KEY_PATH' 환경변수를 찾을 수 없습니다. (.env: {env_file_path}){Colors.RESET}"
    )
    sys.exit(1)

if not os.path.isabs(google_key_env):
    # 상대 경로를 입력한 경우 프로젝트 루트 기준으로 보정
    google_key_env = os.path.join(PROJECT_ROOT, google_key_env)

GOOGLE_KEY_PATH = os.path.normpath(google_key_env)
GOOGLE_SHEET_NAME = "Stock"
WORKSHEET_NAME = "코드_테마_DB"

### 모든 종목에 대해 분석할 기본 시나리오 리스트 ###
DEFAULT_SCENARIOS = [
    "신고가",
    "상따",
    "신고가 근접",
    "거래량 폭증",
    "상한가 다음날",
    "120 돌파",
    "상승형 음봉",
]


def get_decision_color(decision):
    """decision 값에 따라 색상 코드를 반환"""
    d = decision.lower()
    if "max" in d:
        return Colors.RED + Colors.BOLD
    if "expand" in d:
        return Colors.MAGENTA
    if "neutral" in d:
        return Colors.WHITE
    if "reduce" in d:
        return Colors.YELLOW
    return Colors.RESET


# [한글/영문 혼합 문자열의 실제 화면 너비 계산 함수
def get_display_width(s):
    width = 0
    for char in s:
        if unicodedata.east_asian_width(char) in ["F", "W", "A"]:
            width += 2
        else:
            width += 1
    return width


# 화면 너비 기준으로 문자열 정렬(Padding) 함수
def pad_str(s, width, align="left"):
    s = str(s)
    current_width = get_display_width(s)
    padding_size = max(0, width - current_width)

    if align == "center":
        left = padding_size // 2
        right = padding_size - left
        return " " * left + s + " " * right
    elif align == "right":
        return " " * padding_size + s
    else:  # left
        return s + " " * padding_size


# =========================================================
# 1. 데이터 로드 및 전처리
# =========================================================


def load_theme_from_gsheet(key_path, sheet_name, worksheet_name):
    """
    [NEW] 구글 시트에서 실시간으로 데이터를 가져와 {종목코드: 테마} 딕셔너리 반환
    """
    if not os.path.exists(key_path):
        print(f"{Colors.RED}[Error] 인증 키 파일({key_path})이 없습니다.{Colors.RESET}")
        return {}

    print(
        f"{Colors.CYAN}구글 시트 연결 중... ({sheet_name} > {worksheet_name}){Colors.RESET}"
    )

    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
        client = gspread.authorize(creds)
        sh = client.open(sheet_name)
        ws = sh.worksheet(worksheet_name)
        records = ws.get_all_records()
        df = pd.DataFrame(records)

        if df.empty:
            print(f"{Colors.YELLOW}[Warning] 시트가 비어있습니다.{Colors.RESET}")
            return {}

        code_col = None
        theme_col = None
        for col in df.columns:
            if "코드" in str(col):
                code_col = col
            if "테마" in str(col) or "섹터" in str(col):
                theme_col = col

        if not code_col or not theme_col:
            print(
                f"{Colors.RED}시트에서 '종목코드' 또는 '테마' 컬럼을 찾을 수 없습니다.{Colors.RESET}"
            )
            return {}

        df[code_col] = df[code_col].apply(
            lambda x: str(x).strip().split(".")[0].zfill(6)
        )
        theme_map = dict(zip(df[code_col], df[theme_col]))
        print(f"✅ 구글 시트 데이터 로드 완료: 총 {len(theme_map)}개 매핑")
        return theme_map

    except Exception as e:
        print(f"{Colors.RED}[Error] 구글 시트 연동 실패: {e}{Colors.RESET}")
        return {}


def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        print(f"{Colors.RED}Error: {file_path} 파일을 찾을 수 없습니다.{Colors.RESET}")
        sys.exit(1)

    print(f"{Colors.CYAN}조건검색 데이터 로드 중... ({file_path}){Colors.RESET}")

    try:
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        print(f"{Colors.RED}엑셀 파일 로드 실패: {e}{Colors.RESET}")
        sys.exit(1)

    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].apply(lambda x: str(x).zfill(6))

    rename_map = {
        # preprocessor.py의 rename_map과 유사하게 구성
        "시가총액(억)": "시가총액",
        "거래대금(억)": "거래대금",
        "등락률": "등락률",
        "순위": "선정순위",
        "기관_순매수(억)": "기관_순매수",
        "외국인_순매수(억)": "외국인_순매수",
        "프로그램_순매수(억)": "프로그램_순매수",
        "KOSPI등락률": "kospi",
        "KOSDAQ등락률": "kosdaq",
        "전체종목수": "총_종목수",
        "평균거래대금(억)": "평균거래대금_억",
    }
    df.rename(columns=rename_map, inplace=True)

    # 단위 변환 (억 -> 원)
    for col in [
        "기관_순매수",
        "외국인_순매수",
        "프로그램_순매수",
        "거래대금",
        "평균거래대금_억",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: float(x) * 100_000_000 if pd.notna(x) else 0
            )

    # preprocessor.py의 로직을 따라 '평균_거래대금'으로 컬럼명 최종 변경
    if "평균거래대금_억" in df.columns:
        df.rename(columns={"평균거래대금_억": "평균_거래대금"}, inplace=True)

    print(f"✅ 데이터 로드 완료: 총 {len(df)}개 종목")
    return df


def explain_prediction(model, X_input, explainer=None):
    """SHAP을 사용하여 예측에 대한 기여도를 설명합니다."""
    if explainer is None:
        return [], []

    try:
        # SHAP 값 계산
        shap_values = explainer.shap_values(X_input)

        # 피처 이름과 SHAP 값을 결합
        feature_names = X_input.columns
        feature_shap_values = list(zip(feature_names, shap_values[0]))

        # 기여도에 따라 정렬
        feature_shap_values.sort(key=lambda x: x[1], reverse=True)

        # 긍정적/부정적 요인 상위 3개 추출
        pos_factors = [
            f"{name}({val:.2f})" for name, val in feature_shap_values[:3] if val > 0
        ]
        neg_factors = [
            f"{name}({val:.2f})"
            for name, val in reversed(feature_shap_values[-3:])
            if val < 0
        ]

        return pos_factors, neg_factors
    except Exception as e:
        print(f"{Colors.YELLOW}[Warning] SHAP 값 계산 중 오류: {e}{Colors.RESET}")
        return [], []


def print_table(results_list, title, minimal=False):
    """결과 리스트를 테이블 형태로 출력"""
    if not results_list:
        return

    results_list.sort(key=lambda x: x["Score"], reverse=True)

    if minimal:
        W_RANK, W_NAME = 6, 12
        W_PROB, W_DECISION = 8, 14
        header = (
            f"| {pad_str('Rank', W_RANK, 'center')} "
            f"| {pad_str('Name', W_NAME, 'center')} "
            f"| {pad_str('Score', W_PROB, 'center')} "
            f"| {pad_str('Decision', W_DECISION, 'center')} |"
        )
    else:
        W_RANK, W_NAME, W_THEME = 6, 12, 10
        W_SCENARIO, W_PROB, W_DECISION = 12, 8, 14
        header = (
            f"| {pad_str('Rank', W_RANK, 'center')} "
            f"| {pad_str('Name', W_NAME, 'center')} "
            f"| {pad_str('Theme', W_THEME, 'center')} "
            f"| {pad_str('Scenario', W_SCENARIO, 'center')} "
            f"| {pad_str('Prob', W_PROB, 'center')} "
            f"| {pad_str('Decision', W_DECISION, 'center')} |"
        )
    divider = "-" * get_display_width(header)

    print(f"\n{Colors.BOLD}=== {title} ==={Colors.RESET}")
    print(divider)
    print(Colors.BOLD + header + Colors.RESET)
    print(divider)

    previous_stock_name = None

    for res in results_list:
        dec_color = get_decision_color(res["Decision"])
        if previous_stock_name is not None and res["Name"] != previous_stock_name:
            print(divider)
        previous_stock_name = res["Name"]

        name_display = res["Name"]
        if get_display_width(name_display) > W_NAME:
            name_display = name_display[:6] + ".."
        score_str = f"{res['Score']:.4f}"

        if minimal:
            row_str = (
                f"| {pad_str(res['Rank'], W_RANK, 'center')} "
                f"| {pad_str(name_display, W_NAME, 'left')} "
                f"| {pad_str(score_str, W_PROB, 'center')} "
                f"| {dec_color}{pad_str(res['Decision'], W_DECISION, 'center')}{Colors.RESET} |"
            )
        else:
            theme_display = res["Theme"]
            if get_display_width(theme_display) > W_THEME:
                theme_display = theme_display[:6] + ".."
            scenario_display = res["Scenario"]
            if get_display_width(scenario_display) > W_SCENARIO:
                scenario_display = scenario_display[:4] + ".."

            row_str = (
                f"| {pad_str(res['Rank'], W_RANK, 'center')} "
                f"| {pad_str(name_display, W_NAME, 'left')} "
                f"| {pad_str(theme_display, W_THEME, 'left')} "
                f"| {pad_str(scenario_display, W_SCENARIO, 'left')} "
                f"| {pad_str(score_str, W_PROB, 'center')} "
                f"| {dec_color}{pad_str(res['Decision'], W_DECISION, 'center')}{Colors.RESET} |"
            )
        print(row_str)
    print(divider)


def print_explanations(results_list, title):
    """각 종목별 점수에 영향을 준 주요 요인을 설명"""
    if not results_list:
        return

    print(f"\n{Colors.BOLD}--- {title}: Feature Contributions ---{Colors.RESET}")
    for res in sorted(results_list, key=lambda x: x["Score"], reverse=True):
        name = res["Name"]
        scenario = res["Scenario"]
        score = res["Score"]
        pos_factors = res.get("Pos_Factors", [])
        neg_factors = res.get("Neg_Factors", [])

        print(
            f"{Colors.CYAN}{name} ({scenario}){Colors.RESET} " f"- Score: {score:.4f}"
        )
        if pos_factors:
            print(
                f"  {Colors.GREEN}+ 영향 요인:{Colors.RESET} {', '.join(pos_factors)}"
            )
        if neg_factors:
            print(f"  {Colors.RED}- 부담 요인:{Colors.RESET} {', '.join(neg_factors)}")
        if not pos_factors and not neg_factors:
            print("  (설명 정보를 불러오지 못했습니다.)")


# =========================================================
# 2. 메인 실행 함수
# =========================================================


def main():
    # 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(
            f"{Colors.YELLOW}[Warning] 모델 파일({MODEL_PATH})이 없습니다.{Colors.RESET}"
        )
        model = None
    else:
        print(f"{Colors.GREEN}모델 로드 중... ({MODEL_PATH}){Colors.RESET}")
        try:
            model = load(MODEL_PATH)
        except Exception as e:
            print(f"{Colors.RED}모델 로드 오류: {e}{Colors.RESET}")
            sys.exit(1)

    # SHAP Explainer 초기화 (모델이 로드된 후에)
    explainer = None
    # ExtraTrees, CatBoost 등 트리 기반 모델에 대해 TreeExplainer 초기화
    if model and (
        "extratrees" in str(type(model)).lower()
        or "catboost" in str(type(model)).lower()
    ):
        import shap

        explainer = shap.TreeExplainer(model)
    # 데이터 로드
    df_condition = load_and_preprocess_data(CONDITION_EXCEL_PATH)
    theme_map = load_theme_from_gsheet(
        GOOGLE_KEY_PATH, GOOGLE_SHEET_NAME, WORKSHEET_NAME
    )

    # condition_*.xlsx 파일에 있는 모든 종목에 대해 분석 수행
    print(
        f"\n분석 대상: {Colors.BOLD}condition 파일 내 모든 종목 ({len(df_condition)}개){Colors.RESET}"
    )
    print(f"적용 시나리오: {Colors.BOLD}{DEFAULT_SCENARIOS}{Colors.RESET}")

    final_ordered_results = []

    # 모델이 학습할 때 사용한 Feature 순서를 가져옴.
    if model and hasattr(model, "feature_names_in_"):
        train_feature_names = list(model.feature_names_in_)
    else:
        train_feature_names = None

    for index, matched_row_series in df_condition.iterrows():
        # Series를 DataFrame으로 변환
        matched_row = matched_row_series.to_frame().T

        code_input = matched_row.iloc[0]["종목코드"]
        stock_name = matched_row.iloc[0]["종목명"]
        theme = theme_map.get(code_input, "테마 없음")
        if not theme or theme == "테마 없음":
            print(f"{Colors.YELLOW}{stock_name} 테마없음{Colors.RESET}")
            continue

        # 모든 종목에 대해 DEFAULT_SCENARIOS를 적용
        for scenario in DEFAULT_SCENARIOS:
            # 1. DataFrame을 복사하여 시나리오별 입력 데이터 생성
            X_input = matched_row.copy()

            # 2. 예측 시점에 생성되는 Feature들을 DataFrame에 직접 추가
            today = datetime.datetime.now()
            day_name_map = {
                0: "월요일",
                1: "화요일",
                2: "수요일",
                3: "목요일",
                4: "금요일",
                5: "토요일",
                6: "일요일",
            }

            X_input["테마_섹터"] = theme
            X_input["차트분석"] = str(scenario)
            X_input["day_name"] = day_name_map[today.weekday()]
            X_input["day_of_month"] = float(today.day)
            X_input["month"] = float(today.month)

            # '상따' 시나리오일 경우 등락률을 29.9로 강제 조정
            if "상따" in str(scenario):
                X_input["등락률"] = 29.9

            # 학습 시 사용되지 않은 컬럼 제거
            X_input.drop(columns=["종목명"], inplace=True, errors="ignore")

            # 모델에 없는 'real_trade' 같은 컬럼이 있다면 기본값으로 추가
            if "real_trade" not in X_input.columns:
                X_input["real_trade"] = 0.0

            # 범주형 컬럼에 저장된 인코더 적용
            X_input = apply_label_encodings(X_input, LABEL_ENCODER_MAP)

            # 3. 모델이 원하는 컬럼 순서대로 강제 재정렬 (Reindexing)
            if model and train_feature_names:
                try:
                    # 학습시 사용한 컬럼들로만 구성하고 순서를 맞춤
                    X_input = X_input[train_feature_names]
                except KeyError as e:
                    print(
                        f"{Colors.RED}[Critical] 학습 데이터와 예측 데이터의 컬럼이 다릅니다. 누락된 컬럼: {e}{Colors.RESET}"
                    )
                    continue

            if model:
                try:
                    score = float(model.predict(X_input)[0])
                    pos_factors, neg_factors = explain_prediction(
                        model, X_input, explainer
                    )
                except Exception as e:
                    print(
                        f"{Colors.RED}Predict Error ({stock_name}): {e}{Colors.RESET}"
                    )
                    continue
            else:
                score = 0.5
                pos_factors, neg_factors = [], []

            # Decision Logic (비중 조절 로직)
            if score >= 1.9:
                decision = "Max_Expand"
            elif score >= 1.6:
                decision = "Expand"
            elif score >= 1.2:
                decision = "Neutral"
            else:
                decision = "Reduce"

            applied_rate = X_input["등락률"].values[0]

            final_ordered_results.append(
                {
                    "Rank": X_input["선정순위"].values[0],
                    "Name": stock_name,
                    "Theme": theme,
                    "Scenario": str(scenario),
                    "Score": score,
                    "Decision": decision,
                    "KOSPI_Rate": X_input["kospi"].values[0],
                    "KOSDAQ_Rate": X_input["kosdaq"].values[0],
                    "Applied_Rate": applied_rate,
                    "Pos_Factors": pos_factors,
                    "Neg_Factors": neg_factors,
                }
            )

    # 결과 출력
    normal_results = [r for r in final_ordered_results if "상따" not in r["Scenario"]]
    sangdda_results = [r for r in final_ordered_results if "상따" in r["Scenario"]]

    print_table(normal_results, "일반 분석 결과")
    print_table(sangdda_results, "상따(29.9%) 시나리오 결과", minimal=True)

    print_explanations(normal_results, "일반 분석 결과")
    # print_explanations(sangdda_results, "상따(29.9%) 시나리오 결과")


if __name__ == "__main__":
    main()
