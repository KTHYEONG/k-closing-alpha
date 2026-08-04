import datetime
import json
import logging
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DB 및 시트 관련 임포트
from src.data.db_loader import load_theme_from_db
from src.data.sync_sheet_db import sync_theme_only
from src.ml.model_pipeline import run_model_pipeline  # noqa: F401  (Purged Walk-Forward CV 학습 파이프라인 진입점)
from src.ml.sizing_engine import load_model_artifacts, predict_daily_position_sizing
from src.processing.preprocessor import build_ml_dataset  # noqa: F401  (파퀘 기반 학습 파이프라인 진입점)
from src.utils.display import Colors, print_table

# run_model_pipeline(df, feature_cols, target_col, group_col)

# SHAP (모델 해석용) - 필요 시 설치: pip install shap
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


from src import settings

LABEL_ENCODER_PATH = str(settings.LABEL_ENCODER_PATH)
CONDITION_CSV_PATH = str(settings.CONDITION_CSV_PATH)
DEFAULT_SCENARIOS = settings.DEFAULT_SCENARIOS
DAY_NAME_MAP = settings.DAY_NAME_MAP
GOOGLE_SHEET_NAME = settings.GOOGLE_SHEET_NAME
THEME_WORKSHEET_NAME = settings.THEME_WORKSHEET_NAME


def load_label_encoder_map(path):
    """Load saved label encoder classes and build mapping."""
    if not os.path.exists(path):
        logger.info(
            f"{Colors.YELLOW}[Warning] Label encoder file not found: {path}. "
            "Categorical encoding may fail during inference."
            f"{Colors.RESET}"
        )
        return {}

    try:
        with open(path, encoding="utf-8") as f:
            raw_data = json.load(f)
    except Exception as exc:
        logger.info(
            f"{Colors.RED}[Error] Failed to read label encoder file: {exc}{Colors.RESET}"
        )
        return {}

    encoder_map = {}
    for col, classes in raw_data.items():
        mapping = {str(cls): idx for idx, cls in enumerate(classes)}
        unknown_idx = mapping.get("Unknown", len(mapping))
        encoder_map[col] = {"mapping": mapping, "unknown": unknown_idx}
    logger.info(
        f"{Colors.CYAN}Loaded label encoders for columns: {list(encoder_map.keys())}{Colors.RESET}"
    )
    return encoder_map


LABEL_ENCODER_MAP = load_label_encoder_map(LABEL_ENCODER_PATH)


# 분석할 조건검색 결과 파일 및 모델 파일은 위에서 설정됨


def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        logger.info(f"{Colors.RED}Error: {file_path} 파일을 찾을 수 없습니다.{Colors.RESET}")
        sys.exit(1)

    logger.info(f"{Colors.CYAN}조건검색 데이터 로드 중... ({file_path}){Colors.RESET}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.info(f"{Colors.RED}CSV 파일 로드 실패: {e}{Colors.RESET}")
        sys.exit(1)

    # [Fix] 최신 엑셀 컬럼명 형식(괄호 포함) 대응을 위한 rename_map 업데이트
    rename_map = {
        "(시가총액, 억)": "시가총액",
        "시가총액(억)": "시가총액",
        "(거래대금, 억)": "거래대금",
        "거래대금(억)": "거래대금",
        "(등락률)": "등락률",
        "등락률": "등락률",
        "(선정 순위)": "선정순위",
        "순위": "선정순위",
        "(기관_순매수)": "기관_순매수",
        "기관_순매수(억)": "기관_순매수",
        "(외국인_순매수)": "외국인_순매수",
        "외국인_순매수(억)": "외국인_순매수",
        "(프로그램_순매수)": "프로그램_순매수",
        "프로그램_순매수(억)": "프로그램_순매수",
        "(kospi, %)": "kospi",
        "KOSPI등락률": "kospi",
        "(kosdaq, %)": "kosdaq",
        "KOSDAQ등락률": "kosdaq",
        "(총 종목 수)": "총_종목수",
        "전체종목수": "총_종목수",
        "(평균 거래대금)": "평균_거래대금",
        "평균거래대금(억)": "평균_거래대금",
        "(차트통과)": "차트통과",
        "(종목코드)": "종목코드",
        "(종가)": "종가",
        "(체결강도)": "체결강도",
        "(시장구분)": "시장구분",
    }
    df.rename(columns=rename_map, inplace=True)

    # 종목코드 포맷팅 (6자리 zero-fill)
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].apply(lambda x: str(x).zfill(6))

    # 단위 변환 (억 -> 원)
    for col in [
        "기관_순매수",
        "외국인_순매수",
        "프로그램_순매수",
        "거래대금",
        "평균_거래대금",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: float(x) * 100_000_000 if pd.notna(x) else 0
            )

    # [New] 상장일수 부족 종목 제외 로직 (EMA 20 계산 불가능한 경우)
    if "(상장일수)" in df.columns:
        original_count = len(df)
        df["(상장일수)"] = pd.to_numeric(df["(상장일수)"], errors="coerce").fillna(0)
        df = df[df["(상장일수)"] >= settings.EMA_PERIOD].copy()
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info(
                f"{Colors.YELLOW}⚠️ 상장일수 부족 ({settings.EMA_PERIOD}일 미만): {filtered_count}개 종목 제외{Colors.RESET}"
            )

    # "상따" 시나리오 종목도 함께 로드 (나중에 필터링)
    logger.info(
        f"{Colors.GREEN}✅ 데이터 로드 완료: 분석 대상 {len(df)}개 종목{Colors.RESET}"
    )
    return df


def explain_predictions_with_shap(model, X_final, stock_names, top_n=3):
    """SHAP 분석을 통해 각 종목의 예측 점수에 가장 큰 영향을 준 피처를 출력합니다.

    Args:
        model: 학습된 모델 객체
        X_final: 모델 입력 데이터 (DataFrame)
        stock_names: 종목명 리스트 (분석 결과 표시용)
        top_n: 표시할 상위 피처 개수 (기본값: 3)

    """
    if not HAS_SHAP:
        logger.info(
            f"{Colors.YELLOW}[SHAP] shap 라이브러리가 설치되지 않았습니다. 'pip install shap'로 설치하세요.{Colors.RESET}"
        )
        return

    logger.info(f"\n{Colors.CYAN}[SHAP Analysis] 예측 근거 분석 중...{Colors.RESET}")

    try:
        # TreeExplainer 생성 (CatBoost, RandomForest 등 트리 기반 모델용)
        explainer = shap.TreeExplainer(model)

        # SHAP 값 계산 (시간이 걸릴 수 있음)
        shap_values = explainer.shap_values(X_final)

        # 각 종목별로 상위 기여 피처 출력
        logger.info(f"\n{'='*80}")
        logger.info(f"{Colors.GREEN}📊 주요 예측 근거 (Top {top_n} Features){Colors.RESET}")
        logger.info(f"{'='*80}")

        printed_stocks = set()

        for idx, stock_name in enumerate(stock_names):
            if stock_name in printed_stocks:
                continue

            printed_stocks.add(stock_name)

            # 해당 종목의 SHAP 값 (절대값 기준 정렬)
            shap_row = shap_values[idx]
            abs_shap = np.abs(shap_row)
            top_indices = np.argsort(abs_shap)[-top_n:][::-1]

            logger.info(f"\n[{stock_name}]")
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
                        value_str = f"{feat_value!s:20s}"

                logger.info(
                    f"  {rank}. {feat_name:20s} = {value_str}  {direction} {abs(shap_impact):+.4f}"
                )

        logger.info(f"\n{'='*80}\n")

    except Exception as e:
        logger.info(f"{Colors.RED}[SHAP Error] {e}{Colors.RESET}")


# =========================================================
# 2. 메인 실행 함수
# =========================================================


def run_daily_sizing_inference(
    df: pd.DataFrame,
    models_bundle: dict[str, Any],
    feature_cols: list[str] | None = None,
    group_col: str = "date",
) -> pd.DataFrame:
    """저장된 모델 아티팩트로 당일 스냅샷에 Fast Inference + Dynamic Sizing 을 수행합니다.

    모델 학습 시 사용된 ``feature_cols`` 를 기준으로 누락 컬럼을 0 으로 채우고,
    ``group_col`` 이 없으면 오늘 날짜로 단일 그룹을 구성하여
    ``predict_daily_position_sizing`` 을 호출합니다.
    """
    if feature_cols is None:
        feature_cols = list(models_bundle.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("feature_cols is empty; models_bundle must declare feature_cols")

    work = df.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    if group_col not in work.columns:
        work[group_col] = str(datetime.date.today())

    return predict_daily_position_sizing(
        work[[*feature_cols, group_col]],
        feature_cols,
        group_col=group_col,
        models_bundle=models_bundle,
    )


def main():
    # 1. 데이터 로드 및 테마 매핑
    df_condition = load_and_preprocess_data(settings.CONDITION_CSV_PATH)
    theme_map = load_theme_from_db()

    # [Smart Sync] 현재 분석 종목 중 테마가 없는 종목이 있는지 확인
    missing_themes = df_condition[
        ~df_condition["종목코드"].map(theme_map).isin(theme_map.values())
    ]

    if not missing_themes.empty:
        logger.info(
            f"{Colors.CYAN}신규 또는 미분류 종목 발견! 구글 시트에서 테마 정보를 동기화합니다...{Colors.RESET}"
        )
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
        logger.info(
            f"{Colors.YELLOW}[알림] 테마 미매칭으로 분석 제외: {', '.join(no_theme_names)}{Colors.RESET}"
        )

    # 테마가 있는 종목만 분석 대상으로 유지
    df_condition = df_condition[df_condition["테마_섹터"] != "테마 없음"].copy()

    if df_condition.empty:
        logger.info(f"{Colors.RED}분석 대상 종목이 없습니다.{Colors.RESET}")
        return

    # 차트통과 데이터는 이미 Daily_Get_Data.py에서 SMA 120 기준으로 설정됨

    # [New] 실시간 V-KOSPI & V-KOSDAQ 계산
    current_vkospi, current_vkospi_change = 0.0, 0.0
    current_vkosdaq, current_vkosdaq_change = 0.0, 0.0

    try:
        import asyncio

        from src.api.kis_client import fetch_index_and_calculate_volatility

        logger.info(
            f"{Colors.CYAN}실시간 변동성(V-KOSPI/V-KOSDAQ) 데이터를 가져옵니다...{Colors.RESET}"
        )

        # 1. V-KOSPI (1028)
        current_vkospi, current_vkospi_change = asyncio.run(
            fetch_index_and_calculate_volatility("1028")
        )
        logger.info(
            f"  > V-KOSPI : {current_vkospi:.2f} (Change: {current_vkospi_change:+.2%})"
        )

        # 2. V-KOSDAQ (2203)
        current_vkosdaq, current_vkosdaq_change = asyncio.run(
            fetch_index_and_calculate_volatility("2203")
        )
        logger.info(
            f"  > V-KOSDAQ: {current_vkosdaq:.2f} (Change: {current_vkosdaq_change:+.2%})"
        )

    except Exception as e:
        logger.info(f"{Colors.YELLOW}[Warning] 변동성 계산 중 오류 발생: {e}{Colors.RESET}")
        logger.info("  > 변동성 관련 피처는 0으로 처리됩니다.")

    # 3. [핵심] 시나리오 확장 - 수집 단계에서 저장된 (시나리오)를 그대로 사용
    # 차트통과=1: 저장된 시나리오 유지
    # 차트통과=0: 저장된 시나리오 유지 (미지정은 "거래량 폭증" 기본)

    df_passed = df_condition[df_condition["차트통과"] == 1].copy()  # 통과한 종목
    df_filtered = df_condition[df_condition["차트통과"] == 0].copy()  # 필터된 종목

    df_all_parts = []

    # 3-1. 차트통과한 종목: 시나리오 리스트 생성
    if not df_passed.empty:

        def get_scenario_list(row):
            assigned = row.get("(시나리오)")

            # [규칙 1] 엑셀에서 "상따"로 지정된 경우 -> 일반분석 제외하고 "상따"만 적용
            if assigned == "상따":
                return ["상따"]

            scenarios = []

            # [규칙 2] 그 외 일반적인 경우
            if pd.notna(assigned) and assigned != "" and assigned is not None:
                scenarios.append(assigned)
            else:
                scenarios.append("거래량 폭증")  # 기본값

            # [규칙 3] 등락률 20% 이상이면 "상따" 시나리오 추가 (일반 + 상따 함께 분석)
            if row.get("등락률", 0) >= 20:
                if "상따" not in scenarios:
                    scenarios.append("상따")

            return scenarios

        df_passed["Scenario_List"] = df_passed.apply(get_scenario_list, axis=1)
        df_passed_exploded = df_passed.explode("Scenario_List")
        df_passed_exploded["Scenario_Base"] = df_passed_exploded["Scenario_List"]

        # 분석 대상에 추가
        df_all_parts.append(df_passed_exploded.drop(columns=["Scenario_List"]))
        logger.info(
            f"{Colors.CYAN}📌 분석 대상 종목(차트통과): {len(df_passed)}개 (시나리오 확장 포함 총 {len(df_passed_exploded)}건){Colors.RESET}"
        )

    # 3-2. 차트통과=0인 종목: (시나리오) 우선 확인 후 SMA 120 분석
    if not df_filtered.empty:
        # [Step 1] "상따" 시나리오 지정 종목 우선 분리 및 처리
        df_filtered_sangdda = pd.DataFrame()
        if "(시나리오)" in df_filtered.columns:
            df_filtered_sangdda = df_filtered[
                df_filtered["(시나리오)"] == "상따"
            ].copy()
            # 상따 제외한 나머지
            df_remaining = df_filtered[df_filtered["(시나리오)"] != "상따"].copy()
        else:
            df_remaining = df_filtered.copy()

        # 상따 지정 종목 -> 바로 추가 (일반 분석 제외)
        if not df_filtered_sangdda.empty:
            df_filtered_sangdda["Scenario_Base"] = "상따"
            df_all_parts.append(df_filtered_sangdda)
            logger.info(
                f"{Colors.MAGENTA}📌 차트통과=0 but 상따 지정: {len(df_filtered_sangdda)}개 (상따 전용){Colors.RESET}"
            )

        # [Step 2] 나머지 종목 중 시나리오 지정 vs 미지정 분리
        df_filtered_with_scenario = (
            df_remaining[
                df_remaining["(시나리오)"].notna() & (df_remaining["(시나리오)"] != "")
            ].copy()
            if "(시나리오)" in df_remaining.columns
            else pd.DataFrame()
        )

        df_filtered_without_scenario = (
            df_remaining[
                df_remaining["(시나리오)"].isna() | (df_remaining["(시나리오)"] == "")
            ].copy()
            if "(시나리오)" in df_remaining.columns
            else df_remaining.copy()
        )

        # [Step 3] 시나리오 지정 종목 (상따 제외) 처리
        if not df_filtered_with_scenario.empty:

            def get_scenario_list_filtered(row):
                scenarios = [row["(시나리오)"]]
                # 등락률 20% 이상이면 상따 시나리오 추가
                if row.get("등락률", 0) >= 20:
                    if "상따" not in scenarios:
                        scenarios.append("상따")
                return scenarios

            df_filtered_with_scenario["Scenario_List"] = (
                df_filtered_with_scenario.apply(get_scenario_list_filtered, axis=1)
            )
            df_filtered_with_scenario_exploded = df_filtered_with_scenario.explode(
                "Scenario_List"
            )
            df_filtered_with_scenario_exploded["Scenario_Base"] = (
                df_filtered_with_scenario_exploded["Scenario_List"]
            )
            df_all_parts.append(
                df_filtered_with_scenario_exploded.drop(columns=["Scenario_List"])
            )
            logger.info(
                f"{Colors.MAGENTA}📌 차트통과=0 but 시나리오 지정: {len(df_filtered_with_scenario)}개 (지정+상따){Colors.RESET}"
            )

        # [Step 4] 시나리오 미지정 차트통과=0 종목: 기본 시나리오 적용
        # (수집 단계에서 시나리오가 정형화되므로 중복 SMA 120 재계산 없이 "거래량 폭증" 기본값 사용)
        if not df_filtered_without_scenario.empty:

            def get_default_scenarios(row):
                scenarios = ["거래량 폭증"]
                if row.get("등락률", 0) >= 20:
                    if "상따" not in scenarios:
                        scenarios.append("상따")
                return scenarios

            df_filtered_without_scenario["Scenario_List"] = (
                df_filtered_without_scenario.apply(get_default_scenarios, axis=1)
            )
            df_default_exploded = df_filtered_without_scenario.explode("Scenario_List")
            df_default_exploded["Scenario_Base"] = df_default_exploded["Scenario_List"]
            df_all_parts.append(df_default_exploded.drop(columns=["Scenario_List"]))
            logger.info(
                f"{Colors.MAGENTA}📌 차트통과=0 but 시나리오 미지정: {len(df_filtered_without_scenario)}개 → 거래량 폭증 기본 적용{Colors.RESET}"
            )

    # 3-3. 모든 부분 결합
    if df_all_parts:
        df_all = pd.concat(df_all_parts, ignore_index=True)
    else:
        logger.info(f"{Colors.RED}분석할 데이터가 없습니다.{Colors.RESET}")
        return

    # 결과 요약
    logger.info(f"\n{Colors.CYAN}[시나리오 적용 완료]{Colors.RESET}")
    logger.info(f"  ✅ 차트통과: {len(df_passed)}개 × 시나리오 확장")
    if df_filtered is not None and not df_filtered.empty:
        logger.info(f"  📊 차트통과=0: {len(df_filtered)}개 × 저장 시나리오 유지")
    logger.info()

    # [New] v-kospi & v-kosdaq 피처 주입
    df_all["v_kospi"] = current_vkospi
    df_all["v_kospi_change"] = current_vkospi_change
    df_all["v_kosdaq"] = current_vkosdaq
    df_all["v_kosdaq_change"] = current_vkosdaq_change

    # 4. [Feature Engineering Upgrade] 고도화된 전처리 적용
    # preprocessor.py 와 동일한 로직을 적용하여 모델 입력 정합성 확보

    # (1) 날짜 관련 피처 생성

    today = datetime.datetime.now()
    # 단일 날짜(오늘)이므로 list로 만들어서 Series 생성 후 처리
    # 실제로는 historical 데이터가 아니므로 diff/shift 기능은 제한적이지만, 최대한 포맷 유지
    # 영업일 로직은 단일 날짜에서는 '평일' 기준으로 가정

    # 1. 주기성 인코딩
    # 여기서는 dt 접근자 대신 단일 값 계산으로 처리
    days_in_month_map = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
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
    if today.weekday() == 4:  # 금요일
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
    # [V1 Feature Set] 선정순위 원본값 백업 (결과 출력용) - 모델 입력에서는 제외
    if "선정순위" in df_all.columns:
        df_all["선정순위_원본"] = (
            pd.to_numeric(df_all["선정순위"], errors="coerce").fillna(0).astype(int)
        )

    # [V1 Feature Set] 로그 스케일링 (선정순위 포함)
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
        df_all["체결강도"] = pd.to_numeric(df_all["체결강도"], errors="coerce").fillna(
            100
        )
        df_all["체결강도"] = np.log(df_all["체결강도"].clip(lower=1) / 100.0)

    # 4-2. [V1 Feature Set] 선정순위_상대 생성 (학습 모델과 동일)
    if "선정순위" in df_all.columns and "총_종목수" in df_all.columns:
        rank_raw = np.expm1(df_all["선정순위"])
        total_raw = np.expm1(df_all["총_종목수"])
        df_all["선정순위_상대"] = rank_raw / total_raw.clip(lower=1)

    # 4-3. [V1 Feature Set] 상대 등락률 & 방어 강도 (시장구분 미사용)
    # 시장구분 피처를 사용하지 않으므로 KOSPI를 기본 참조로 사용
    if "등락률" in df_all.columns:
        market_ref = df_all["kospi"]
        df_all["상대_등락률"] = df_all["등락률"] - market_ref
        df_all["방어_강도"] = np.where(market_ref < 0, df_all["상대_등락률"], 0)

    # 4-4. 상대 거래대금 (당일/평균)
    if "거래대금" in df_all.columns and "평균_거래대금" in df_all.columns:
        raw_trade = np.expm1(df_all["거래대금"])
        raw_avg_trade = np.expm1(df_all["평균_거래대금"])
        df_all["상대_거래대금"] = np.log(
            (raw_trade + 1) / (raw_avg_trade + 1).clip(lower=1)
        )

    # 4-5. 수급 질적 분석 (메이저 밀도, 프로그램 주도성)
    buy_cols = ["기관_순매수", "외국인_순매수", "프로그램_순매수"]
    if all(c in df_all.columns for c in buy_cols) and "거래대금" in df_all.columns:
        # 이미 Signed Log 변환되었으므로 복원
        raw_inst = np.sign(df_all["기관_순매수"]) * np.expm1(
            np.abs(df_all["기관_순매수"])
        )
        raw_fore = np.sign(df_all["외국인_순매수"]) * np.expm1(
            np.abs(df_all["외국인_순매수"])
        )
        raw_prog = np.sign(df_all["프로그램_순매수"]) * np.expm1(
            np.abs(df_all["프로그램_순매수"])
        )
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

    # 5. Fast Inference & Dynamic Sizing (저장된 모델 아티팩트 로드)
    # 레거시 GMM/Static 판단 로직은 제거되고, artifacts/models/ 의 모델 번들을
    # 로드하여 Utility Score 기반 Sizing Grade(Strong/Good/Weak/Pass) 및 배분을 산출합니다.
    try:
        models_bundle = load_model_artifacts()
    except FileNotFoundError as exc:
        logger.info(
            f"{Colors.YELLOW}[Warning] 모델 아티팩트가 없어 예측을 건너뜁니다: {exc}{Colors.RESET}"
        )
        logger.info(
            f"{Colors.CYAN}[Guide] 먼저 run_sizing_pipeline(export_dir=...) 로 학습 후 "
            "save_model_artifacts() 를 실행해 artifacts/models/ 에 저장하세요.{Colors.RESET}"
        )
        return

    logger.info(
        f"{Colors.GREEN}모델 아티팩트 로드 완료. Fast Inference + Dynamic Sizing 실행 중...{Colors.RESET}"
    )
    sizing_df = run_daily_sizing_inference(df_all, models_bundle)

    # 6. 결과 리스트 생성 (utility_score / grade / allocation)
    final_results = []
    for _, row in sizing_df.iterrows():
        res = {
            "Rank": int(row.get("선정순위_원본", row.get("선정순위", 0))),
            "Name": row["종목명"],
            "Theme": row["테마_섹터"],
            "Scenario": row["차트분석"],
            "RankScore": round(float(row.get("rank_score", 0.0)), 4),
            "Utility": round(float(row["utility_score"]), 4),
            "Grade": row["grade"],
            "Alloc%": round(float(row["allocation"]) * 100.0, 2),
            "kospi": row.get("kospi", 0),
            "kosdaq": row.get("kosdaq", 0),
            "Applied_Rate": row["등락률"],
        }
        final_results.append(res)

    # 7. 결과 출력
    normal_results = [r for r in final_results if "상따" not in r["Scenario"]]
    sangdda_results = [r for r in final_results if "상따" in r["Scenario"]]

    print_table(normal_results, "일반 분석 결과 (Dynamic Sizing)")
    print_table(sangdda_results, "상따(29.9%) 시나리오 결과", minimal=True)


if __name__ == "__main__":
    main()
