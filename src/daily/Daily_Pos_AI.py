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

# DB 로더 임포트
from src.data.db_loader import load_theme_from_db

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
MODEL_PATH = os.path.join(MODELS_DIR, "best_stock_rg.joblib")

# .env 파일에서 환경변수 로드
env_file_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(env_file_path)

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
        "(차트통과)": "차트통과",
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


# =========================================================
# 2. 메인 실행 함수
# =========================================================


def main():
    # 1. 모델 및 Explainer 로드 (기존과 동일)
    if not os.path.exists(MODEL_PATH):
        print(f"{Colors.YELLOW}[Warning] 모델 파일이 없습니다.{Colors.RESET}")
        model = None
    else:
        print(f"{Colors.GREEN}모델 로드 중...{Colors.RESET}")
        model = load(MODEL_PATH)

    explainer = None
    if model and (
        "extratrees" in str(type(model)).lower()
        or "catboost" in str(type(model)).lower()
    ):
        import shap

        explainer = shap.TreeExplainer(model)

    # 2. 데이터 로드 및 테마 일괄 매핑
    df_condition = load_and_preprocess_data(CONDITION_EXCEL_PATH)
    theme_map = load_theme_from_db()

    # 루프 밖에서 테마 정보를 일괄적으로 입힙니다.
    df_condition["테마_섹터"] = (
        df_condition["종목코드"].map(theme_map).fillna("테마 없음")
    )
    df_condition = df_condition[df_condition["테마_섹터"] != "테마 없음"].copy()

    if df_condition.empty:
        print(f"{Colors.RED}분석 대상 종목이 없습니다.{Colors.RESET}")
        return

    # 3. [핵심] 모든 시나리오를 데이터프레임으로 확장 (Cross Join)
    # 종목 10개 * 시나리오 7개 = 70개 행을 한 번에 만듭니다.
    scenario_df = pd.DataFrame({"Scenario_Base": DEFAULT_SCENARIOS})
    df_all = (
        df_condition.assign(key=1)
        .merge(scenario_df.assign(key=1), on="key")
        .drop("key", axis=1)
    )

    # 4. 벡터화된 피처 엔지니어링 (루프 없음)
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

    df_all["day_name"] = day_name_map[today.weekday()]
    df_all["day_of_month"] = float(today.day)
    df_all["month"] = float(today.month)
    df_all["차트분석"] = (
        df_all["Scenario_Base"]
        + "_"
        + df_all["차트통과"].fillna(1).astype(int).astype(str)
    )

    # '상따' 시나리오 일괄 적용
    df_all.loc[df_all["Scenario_Base"].str.contains("상따"), "등락률"] = 29.9
    if "real_trade" not in df_all.columns:
        df_all["real_trade"] = 0.0

    # 5. 모델 입력 준비 및 배치 추론
    # 인코딩도 일괄 처리
    X_processed = apply_label_encodings(df_all.copy(), LABEL_ENCODER_MAP)

    train_feature_names = (
        list(model.feature_names_in_)
        if model and hasattr(model, "feature_names_in_")
        else []
    )

    if model and train_feature_names:
        X_final = X_processed[train_feature_names]
        # 단 한 번의 호출로 모든 종목/시나리오 예측
        df_all["Score"] = model.predict(X_final)
    else:
        df_all["Score"] = 0.5

    # 6. 의사결정 로직 일괄 적용
    df_all["Decision"] = pd.cut(
        df_all["Score"],
        bins=[-np.inf, 1.2, 1.6, 1.9, np.inf],
        labels=["Reduce", "Neutral", "Expand", "Max_Expand"],
    ).astype(str)

    # 7. 결과 리스트 생성
    final_results = []
    for _, row in df_all.iterrows():
        res = {
            "Rank": row["선정순위"],
            "Name": row["종목명"],
            "Theme": row["테마_섹터"],
            "Scenario": row["차트분석"],
            "Score": row["Score"],
            "Decision": row["Decision"],
            "kospi": row.get("kospi", 0),
            "kosdaq": row.get("kosdaq", 0),
            "Applied_Rate": row["등락률"],
        }
        final_results.append(res)

    # 8. 결과 출력
    normal_results = [r for r in final_results if "상따" not in r["Scenario"]]
    sangdda_results = [r for r in final_results if "상따" in r["Scenario"]]

    print_table(normal_results, "일반 분석 결과")
    print_table(sangdda_results, "상따(29.9%) 시나리오 결과", minimal=True)


if __name__ == "__main__":
    main()
