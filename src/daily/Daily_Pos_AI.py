import json
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import datetime

from joblib import load

# DB 로더 임포트
from src.data.db_loader import load_theme_from_db
from src.utils.display import Colors, print_table, apply_label_encodings

# .env 파일에서 환경변수를 불러오기 위한 라이브러리
from dotenv import load_dotenv


from src import settings

LABEL_ENCODER_PATH = str(settings.LABEL_ENCODER_PATH)
CONDITION_EXCEL_PATH = str(settings.CONDITION_EXCEL_PATH)
MODEL_PATH = str(settings.MODEL_PATH)
DEFAULT_SCENARIOS = settings.DEFAULT_SCENARIOS
DAY_NAME_MAP = settings.DAY_NAME_MAP


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


# 분석할 조건검색 결과 파일 및 모델 파일은 위에서 설정됨


# .env 파일에서 환경변수 로드
env_file_path = os.path.join(settings.BASE_DIR, ".env")
load_dotenv(env_file_path)



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

    print(f"{Colors.GREEN}✅ 데이터 로드 완료: 총 {len(df)}개 종목{Colors.RESET}")
    return df


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

    # 2. 데이터 로드 및 테마 일괄 매핑
    df_condition = load_and_preprocess_data(CONDITION_EXCEL_PATH)
    theme_map = load_theme_from_db()

    # 루프 밖에서 테마 정보를 일괄적으로 입힙니다.
    df_condition["테마_섹터"] = (
        df_condition["종목코드"].map(theme_map).fillna("테마 없음")
    )

    # 테마가 없는 종목 알림 출력
    no_theme_stocks = df_condition[df_condition["테마_섹터"] == "테마 없음"]["종목명"].tolist()
    if no_theme_stocks:
        print(f"{Colors.YELLOW}[알림] 테마 미매칭으로 분석 제외: {', '.join(no_theme_stocks)}{Colors.RESET}")

    # 테마가 있는 종목만 분석 대상으로 유지
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
    import datetime
    today = datetime.datetime.now()
    df_all["day_name"] = DAY_NAME_MAP[today.weekday()]
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
