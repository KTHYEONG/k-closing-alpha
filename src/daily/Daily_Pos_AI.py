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

# DB 및 시트 관련 임포트
from src.data.db_loader import load_theme_from_db
from src.data.sync_gsheet_to_db import sync_theme_only
from src.utils.display import Colors, print_table, apply_label_encodings

# .env 파일에서 환경변수를 불러오기 위한 라이브러리
from dotenv import load_dotenv

# SHAP (모델 해석용) - 필요 시 설치: pip install shap
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


from src import settings

LABEL_ENCODER_PATH = str(settings.LABEL_ENCODER_PATH)
CONDITION_EXCEL_PATH = str(settings.CONDITION_EXCEL_PATH)
MODEL_PATH = str(settings.MODEL_PATH)
DEFAULT_SCENARIOS = settings.DEFAULT_SCENARIOS
DAY_NAME_MAP = settings.DAY_NAME_MAP
GOOGLE_SHEET_NAME = settings.GOOGLE_SHEET_NAME
THEME_WORKSHEET_NAME = settings.THEME_WORKSHEET_NAME


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


def explain_predictions_with_shap(model, X_final, stock_names, top_n=3):
    """
    SHAP 분석을 통해 각 종목의 예측 점수에 가장 큰 영향을 준 피처를 출력합니다.
    
    Args:
        model: 학습된 모델 객체
        X_final: 모델 입력 데이터 (DataFrame)
        stock_names: 종목명 리스트 (분석 결과 표시용)
        top_n: 표시할 상위 피처 개수 (기본값: 3)
    """
    if not HAS_SHAP:
        print(f"{Colors.YELLOW}[SHAP] shap 라이브러리가 설치되지 않았습니다. 'pip install shap'로 설치하세요.{Colors.RESET}")
        return
    
    print(f"\n{Colors.CYAN}[SHAP Analysis] 예측 근거 분석 중...{Colors.RESET}")
    
    try:
        # TreeExplainer 생성 (CatBoost, RandomForest 등 트리 기반 모델용)
        explainer = shap.TreeExplainer(model)
        
        # SHAP 값 계산 (시간이 걸릴 수 있음)
        shap_values = explainer.shap_values(X_final)
        
        # 각 종목별로 상위 기여 피처 출력
        print(f"\n{'='*80}")
        print(f"{Colors.GREEN}📊 주요 예측 근거 (Top {top_n} Features){Colors.RESET}")
        print(f"{'='*80}")
        
        printed_stocks = set()
        
        for idx, stock_name in enumerate(stock_names):
            if stock_name in printed_stocks:
                continue
            
            printed_stocks.add(stock_name)

            # 해당 종목의 SHAP 값 (절대값 기준 정렬)
            shap_row = shap_values[idx]
            abs_shap = np.abs(shap_row)
            top_indices = np.argsort(abs_shap)[-top_n:][::-1]
            
            print(f"\n[{stock_name}]")
            for rank, feat_idx in enumerate(top_indices, 1):
                feat_name = X_final.columns[feat_idx]
                feat_value = X_final.iloc[idx, feat_idx]
                shap_impact = shap_row[feat_idx]
                direction = "↑" if shap_impact > 0 else "↓"
                
                # 값이 문자열인지 숫자인지 확인하여 적절한 포맷 사용
                if isinstance(feat_value, str):
                    value_str = f"{feat_value:20s}"
                else:
                    try:
                        value_str = f"{float(feat_value):8.2f}"
                    except:
                        value_str = f"{str(feat_value):20s}"
                
                print(f"  {rank}. {feat_name:20s} = {value_str}  {direction} {abs(shap_impact):+.4f}")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"{Colors.RED}[SHAP Error] {e}{Colors.RESET}")


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

    # 2. 데이터 로드 및 테마 매핑
    df_condition = load_and_preprocess_data(CONDITION_EXCEL_PATH)
    theme_map = load_theme_from_db()

    # [Smart Sync] 현재 분석 종목 중 테마가 없는 종목이 있는지 확인
    missing_themes = df_condition[~df_condition["종목코드"].map(theme_map).isin(theme_map.values())]
    
    if not missing_themes.empty:
        print(f"{Colors.CYAN}신규 또는 미분류 종목 발견! 구글 시트에서 테마 정보를 동기화합니다...{Colors.RESET}")
        sync_theme_only()
        # 동기화 후 다시 로드
        theme_map = load_theme_from_db()

    # 최종 테마 정보를 데이터프레임에 입힙니다.
    df_condition["테마_섹터"] = (
        df_condition["종목코드"].map(theme_map).fillna("테마 없음")
    )

    # 테마가 없는 종목 알림 출력
    df_no_theme = df_condition[df_condition["테마_섹터"] == "테마 없음"]
    if not df_no_theme.empty:
        no_theme_names = df_no_theme["종목명"].tolist()
        print(f"{Colors.YELLOW}[알림] 테마 미매칭으로 분석 제외: {', '.join(no_theme_names)}{Colors.RESET}")

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

    # 4. [Feature Engineering Upgrade] 고도화된 전처리 적용
    # preprocessor.py 와 동일한 로직을 적용하여 모델 입력 정합성 확보
    
    # (1) 날짜 관련 피처 생성
    import datetime
    today = datetime.datetime.now()
    # 단일 날짜(오늘)이므로 list로 만들어서 Series 생성 후 처리
    # 실제로는 historical 데이터가 아니므로 diff/shift 기능은 제한적이지만, 최대한 포맷 유지
    # 영업일 로직은 단일 날짜에서는 '평일' 기준으로 가정
    
    # 1. 주기성 인코딩
    # 여기서는 dt 접근자 대신 단일 값 계산으로 처리
    days_in_month_map = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    # 윤년 간단 처리
    if today.year % 4 == 0 and (today.year % 100 != 0 or today.year % 400 == 0):
        days_in_month_map[2] = 29
        
    days_in_month = days_in_month_map.get(today.month, 30)
    
    # [Fix] 누락된 단순 날짜 피처 추가
    df_all["month"] = float(today.month)
    df_all["day_of_month"] = float(today.day)
    
    df_all["month_sin"] = np.sin(2 * np.pi * today.month / 12)
    df_all["month_cos"] = np.cos(2 * np.pi * today.month / 12)
    df_all["day_sin"] = np.sin(2 * np.pi * today.day / days_in_month)
    df_all["day_cos"] = np.cos(2 * np.pi * today.day / days_in_month)
    
    # 2. 영업일 관련 (단일 시점이라 shift 불가 -> 보수적 기본값 적용)
    # 금요일(4)이면 다음 거래일까지 3일(토,일,월), 그 외는 1일로 가정
    if today.weekday() == 4: # 금요일
        df_all["date_diff"] = 3
    else:
        df_all["date_diff"] = 1
        
    # 월말 여부 (오늘이 마지막 날인지 체크)
    is_month_end = 1 if today.day == days_in_month else 0
    # 주말이 껴서 마지막 영업일인 경우까지 계산하긴 복잡하므로 달력 기준 근사치 사용
    df_all["is_trading_month_end"] = is_month_end
    
    # 3. 요일 (Categorical)
    df_all["weekday"] = today.weekday()

    # (2) 로그 스케일링 (자릿수 압축)
    # 선정순위 원본값 백업 (결과 출력용)
    if "선정순위" in df_all.columns:
        df_all["선정순위_원본"] = pd.to_numeric(df_all["선정순위"], errors="coerce").fillna(0).astype(int)
    
    log_cols = ["시가총액", "거래대금", "평균_거래대금", "총_종목수", "선정순위"]
    for col in log_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0)
            df_all[col] = np.log1p(df_all[col].clip(lower=0))

    # (3) Signed Log (수급 데이터)
    # preprocessor.py에서는 _apply_signed_log_scaling
    signed_cols = ["기관_순매수", "외국인_순매수", "프로그램_순매수"]
    for col in signed_cols:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0)
            df_all[col] = np.sign(df_all[col]) * np.log1p(np.abs(df_all[col]))

    # (4) 도메인 특화 비율 (Custom Ratios)
    # 4-1. 체결강도 (100 기준 로그)
    if "체결강도" in df_all.columns:
        df_all["체결강도"] = pd.to_numeric(df_all["체결강도"], errors="coerce").fillna(100)
        df_all["체결강도"] = np.log(df_all["체결강도"].clip(lower=1) / 100.0)

    # 4-2. 선정순위 상대화
    if "선정순위" in df_all.columns and "총_종목수" in df_all.columns:
        # 이미 위에서 log변환 되었으므로 expm1로 복원 후 계산
        rank_raw = np.expm1(df_all["선정순위"])
        total_raw = np.expm1(df_all["총_종목수"])
        df_all["선정순위_상대"] = rank_raw / total_raw.clip(lower=1)

    # 4-3. 상대 등락률 & 방어 강도
    if "등락률" in df_all.columns:
        if "시장구분" in df_all.columns:
             market_ref = np.where(df_all["시장구분"].str.contains("KOSDAQ", na=False, case=False), 
                                  df_all["kosdaq"], df_all["kospi"])
        else:
            market_ref = df_all["kospi"]
        
        df_all["상대_등락률"] = df_all["등락률"] - market_ref
        df_all["방어_강도"] = np.where(market_ref < 0, df_all["상대_등락률"], 0)

    # 4-4. 상대 거래대금 (당일/평균)
    if "거래대금" in df_all.columns and "평균_거래대금" in df_all.columns:
        raw_trade = np.expm1(df_all["거래대금"])
        raw_avg_trade = np.expm1(df_all["평균_거래대금"])
        df_all["상대_거래대금"] = np.log((raw_trade + 1) / (raw_avg_trade + 1).clip(lower=1))

    # 4-5. 수급 질적 분석 (메이저 밀도, 프로그램 주도성)
    buy_cols = ["기관_순매수", "외국인_순매수", "프로그램_순매수"]
    if all(c in df_all.columns for c in buy_cols) and "거래대금" in df_all.columns:
        # 이미 Signed Log 변환되었으므로 복원
        raw_inst = np.sign(df_all["기관_순매수"]) * np.expm1(np.abs(df_all["기관_순매수"]))
        raw_fore = np.sign(df_all["외국인_순매수"]) * np.expm1(np.abs(df_all["외국인_순매수"]))
        raw_prog = np.sign(df_all["프로그램_순매수"]) * np.expm1(np.abs(df_all["프로그램_순매수"]))
        raw_trade_for_ratio = np.expm1(df_all["거래대금"]).clip(lower=1)
        
        # 메이저 밀도 (기관+외국인)
        df_all["메이저_밀도"] = (raw_inst + raw_fore) / raw_trade_for_ratio
        # 프로그램 주도성
        df_all["프로그램_주도성"] = raw_prog / raw_trade_for_ratio

    # 4-6. 차트분석 피처 생성 (학습 시 사용된 핵심 피처)
    # Scenario_Base(시나리오명)와 차트통과(Y/N)를 결합
    df_all["차트분석"] = (
        df_all["Scenario_Base"].astype(str)
        + "_"
        + df_all["차트통과"].fillna("N").astype(str)
    )

    # '상따' 시나리오 일괄 적용 (기존 로직 유지)
    df_all.loc[df_all["Scenario_Base"].str.contains("상따"), "등락률"] = 29.9
    if "real_trade" not in df_all.columns:
        df_all["real_trade"] = 0.0

    # 5. 모델 입력 준비 및 배치 추론
    # [CatBoost 전용] 범주형 변수는 문자열로 변환 (학습 시와 동일)
    X_processed = df_all.copy()
    
    # LABEL_ENCODER_MAP에 정의된 범주형 컬럼들을 문자열로 변환
    cat_cols_from_encoder = list(LABEL_ENCODER_MAP.keys()) if LABEL_ENCODER_MAP else []
    for col in cat_cols_from_encoder:
        if col in X_processed.columns:
            X_processed[col] = X_processed[col].astype(str)

    # [Feature Names Check] 모델 학습 시 사용된 피처 이름 확보
    train_feature_names = []
    if model:
        if hasattr(model, "feature_names_in_"):
            train_feature_names = list(model.feature_names_in_)
        elif hasattr(model, "feature_names_"):
            train_feature_names = list(model.feature_names_)
    
    if model and train_feature_names:
        # 누락된 컬럼 채우기 (범주형/숫자형 구분)
        for col in train_feature_names:
            if col not in X_processed.columns:
                # 범주형이면 "0" (문자열), 숫자형이면 0.0 (실수)
                if col in cat_cols_from_encoder:
                    X_processed[col] = "0"
                else:
                    X_processed[col] = 0.0
                
        X_final = X_processed[train_feature_names]
        # 단 한 번의 호출로 모든 종목/시나리오 예측
        df_all["Score"] = model.predict(X_final)
        
        # ============================================================
        # [SHAP 분석] 예측 근거 확인 (필요 시 아래 주석 해제)
        # ============================================================
        # 각 종목의 점수가 왜 높게/낮게 나왔는지 상위 3개 피처를 분석합니다.
        # 주의: 종목 수가 많으면 시간이 오래 걸릴 수 있습니다.
        explain_predictions_with_shap(
            model=model,
            X_final=X_final,
            stock_names=df_all["종목명"].tolist(),
            top_n=3
        )
        # ============================================================

    else:
        df_all["Score"] = 0.5

    # 6. 의사결정 로직 일괄 적용
    # [Optimized Thresholds] 타겟 클리핑 후 재학습된 모델 기준
    # - 이전 (클리핑 전): [-inf, 0.2, 0.4, 0.8, inf]
    # - AI 추천 (클리핑 후): [-inf, 0.36, 0.46, 0.54, inf]
    # - 적용 (안전마진): [-inf, 0.4, 0.5, 0.6, inf]
    # → 모델이 극단값 학습을 피하고 안정적 수익 패턴에 집중하도록 개선됨
    df_all["Decision"] = pd.cut(
        df_all["Score"],
        bins=[-np.inf, 0.35, 0.45, 0.55, np.inf],
        labels=["Reduce", "Neutral", "Expand", "Max_Expand"],
    ).astype(str)

    # 7. 결과 리스트 생성
    final_results = []
    for _, row in df_all.iterrows():
        res = {
            "Rank": int(row.get("선정순위_원본", row["선정순위"])),
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
