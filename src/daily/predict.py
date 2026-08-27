import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DB 및 시트 관련 임포트
from src.data.db_loader import load_theme_from_db
from src.data.sync_sheet_db import sync_theme_only
from src.processing.schema import normalize_column_names
from src.serving.realtime.artifacts import load_model_bundle
from src.serving.realtime.features import build_snapshot_features
from src.serving.realtime.inference import predict_daily_sizing
from src.utils.display import Colors, print_table

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

    logger.debug("조건검색 데이터 로드 중... (%s)", file_path)

    try:
        # 종목코드(0155E0 등 지수 표기 및 0 누락 방지)를 문자열로 명시적 로드
        df = pd.read_csv(file_path, dtype={"종목코드": str})
    except Exception as e:
        logger.info(f"{Colors.RED}CSV 파일 로드 실패: {e}{Colors.RESET}")
        sys.exit(1)

    # 종목코드 포맷팅 (6자리 zero-fill, 불필요 소수점 제거)
    # Time Complexity: O(N) | Space Complexity: O(N)
    if "종목코드" in df.columns:
        df["종목코드"] = (
            df["종목코드"]
            .astype(str)
            .str.strip()
            .str.split(".")
            .str[0]
            .str.zfill(6)
        )

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
    if "상장일수" in df.columns:
        original_count = len(df)
        df["상장일수"] = pd.to_numeric(df["상장일수"], errors="coerce").fillna(0)
        df = df[df["상장일수"] >= settings.EMA_PERIOD].copy()
        filtered_count = original_count - len(df)
        if filtered_count > 0:
            logger.info(
                f"{Colors.YELLOW}⚠️ 상장일수 부족 ({settings.EMA_PERIOD}일 미만): {filtered_count}개 종목 제외{Colors.RESET}"
            )

    # "상따" 시나리오 종목도 함께 로드 (나중에 필터링)
    logger.debug("데이터 로드 완료: %d개 종목", len(df))
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
# 2. 표준 ML 피처 스키마 정합성 + 출력 포맷 헬퍼
# =========================================================

# 출력 테이블 기본 상위 후보 수 (Top N)
_DEFAULT_TOP_N = 15


def build_result_rows(sizing_df: pd.DataFrame) -> list[dict[str, Any]]:
    """``sizing_df`` 를 출력용 결과 리스트(display 스키마)로 변환합니다.

    표준 컬럼(``selection_rank``, ``theme_sector``, ``chart_analysis``,
    ``change_rate``)을 우선 사용하고, 레거시 한글 컬럼명을 폴백으로 지원합니다.
    """
    final_results: list[dict[str, Any]] = []
    for _, row in sizing_df.iterrows():
        res = {
            "Rank": int(row.get("selection_rank", row.get("선정순위", 0)) or 0),
            "Name": row.get("종목명", ""),
            "Theme": row.get("theme_sector", row.get("테마_섹터", "")),
            "Scenario": row.get("chart_analysis", row.get("차트분석", "")),
            "RankScore": round(float(row.get("rank_score", 0.0)), 4),
            "Score": round(float(row["utility_score"]), 4),
            "Utility": round(float(row["utility_score"]), 4),
            "Grade": row["grade"],
            "Decision": f"{row['grade']} ({round(float(row['allocation']) * 100.0, 1)}%)",
            "Alloc%": round(float(row["allocation"]) * 100.0, 2),
            "kospi": row.get("kospi", 0),
            "kosdaq": row.get("kosdaq", 0),
            "Applied_Rate": row.get("change_rate", row.get("등락률", 0)),
        }
        final_results.append(res)
    return final_results


def select_top_actionable(
    results: list[dict[str, Any]], top_n: int = _DEFAULT_TOP_N
) -> list[dict[str, Any]]:
    """Pass 등급을 제외한 액션 가능 후보 중 Utility Score 기준 상위 ``top_n`` 을 선별하며,
    액션 가능 종목이 없으면 상위 ``top_n`` 종목을 관찰용 표로 반환합니다.
    """
    actionable = [r for r in results if r["Grade"] != "Pass"]
    if not actionable and results:
        return sorted(results, key=lambda r: r["Score"], reverse=True)[:top_n]
    return sorted(actionable, key=lambda r: r["Score"], reverse=True)[:top_n]


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

    # [New] 실시간 V-KOSPI & V-KOSDAQ 계산
    current_vkospi, current_vkospi_change = 0.0, 0.0
    current_vkosdaq, current_vkosdaq_change = 0.0, 0.0

    try:
        import asyncio

        from src.api.kis_client import fetch_index_and_calculate_volatility

        # 1. V-KOSPI (1028)
        current_vkospi, current_vkospi_change = asyncio.run(
            fetch_index_and_calculate_volatility("1028")
        )

        # 2. V-KOSDAQ (2203)
        current_vkosdaq, current_vkosdaq_change = asyncio.run(
            fetch_index_and_calculate_volatility("2203")
        )

    except Exception as e:
        logger.warning(f"[Warning] 변동성 계산 중 오류 발생: {e}")

    # 3. [핵심] 시나리오 확장 - 수집 단계에서 저장된 표준 시나리오를 그대로 사용
    def get_scenario_list(row):
        assigned = row.get("시나리오")

        # [규칙 1] "상따"로 지정된 경우 -> 일반분석 제외하고 "상따"만 적용
        if assigned == "상따":
            return ["상따"]

        scenarios = []

        # [규칙 2] 그 외 일반적인 경우
        if pd.notna(assigned) and assigned != "" and assigned is not None:
            scenarios.append(assigned)
        else:
            scenarios.append("거래량 폭증")  # 기본값

        # [규칙 3] 등락률 20% 이상이면 "상따" 시나리오 추가 (일반 + 상따 함께 분석)
        rate = float(row.get("change_rate", row.get("등락률", 0)) or 0)
        if rate >= 20 and "상따" not in scenarios:
            scenarios.append("상따")

        return scenarios

    df_condition["Scenario_List"] = df_condition.apply(get_scenario_list, axis=1)
    df_all = df_condition.explode("Scenario_List").reset_index(drop=True)
    df_all["Scenario_Base"] = df_all["Scenario_List"]
    df_all = df_all.drop(columns=["Scenario_List"])

    # [New] v-kospi & v-kosdaq 피처 주입
    df_all["v_kospi"] = current_vkospi
    df_all["v_kospi_change"] = current_vkospi_change
    df_all["v_kosdaq"] = current_vkosdaq
    df_all["v_kosdaq_change"] = current_vkosdaq_change

    # 4. [Feature Engineering] 표준 ML 피처 정합성 확보
    df_all["차트분석"] = df_all["Scenario_Base"].astype(str)
    # 먼저 컬럼명을 정규화하여 '등락률' 등을 'change_rate'로 통합한 후 '상따' 29.9%를 적용
    df_all = normalize_column_names(df_all)

    sangdda_mask = df_all["Scenario_Base"].str.contains("상따", na=False)
    if "change_rate" in df_all.columns:
        df_all.loc[sangdda_mask, "change_rate"] = 29.9
    if "prev_close_price" in df_all.columns:
        limit_up_price = np.round(df_all.loc[sangdda_mask, "prev_close_price"] * 1.299)
        if "close_price" in df_all.columns:
            df_all.loc[sangdda_mask, "close_price"] = limit_up_price
        if "buy_price" in df_all.columns:
            df_all.loc[sangdda_mask, "buy_price"] = limit_up_price
        if "high_price" in df_all.columns:
            df_all.loc[sangdda_mask, "high_price"] = np.maximum(
                df_all.loc[sangdda_mask, "high_price"], limit_up_price
            )

    decision_date = pd.Timestamp.today().normalize()
    df_all = build_snapshot_features(df_all, decision_date=decision_date)

    # 5. Fast Inference & Dynamic Sizing (저장된 모델 아티팩트 로드)
    # 레거시 GMM/Static 판단 로직은 제거되고, artifacts/models/ 의 모델 번들을
    # 로드하여 Utility Score 기반 Sizing Grade(Strong/Good/Weak/Pass) 및 배분을 산출합니다.
    try:
        models_bundle = load_model_bundle()
    except (FileNotFoundError, ValueError) as exc:
        logger.info(
            f"{Colors.YELLOW}[Warning] 모델 아티팩트 로드/검증 실패로 예측을 건너뜁니다: {exc}{Colors.RESET}"
        )
        return

    start = time.perf_counter()
    df_normal = df_all[~df_all["Scenario_Base"].str.contains("상따", na=False)].copy()
    df_sangdda = df_all[df_all["Scenario_Base"].str.contains("상따", na=False)].copy()

    normal_sizing = predict_daily_sizing(df_normal, models_bundle)
    sangdda_sizing = (
        predict_daily_sizing(df_sangdda, models_bundle)
        if not df_sangdda.empty
        else pd.DataFrame()
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{Colors.GREEN}🚀 [예측 완료] 대상: {len(df_condition)}종목({len(df_all)}건) "
        f"| 변동성: VKOSPI {current_vkospi:.2f}({current_vkospi_change:+.2%}), VKOSDAQ {current_vkosdaq:.2f}({current_vkosdaq_change:+.2%}) "
        f"| 추론: {elapsed_ms:.0f}ms{Colors.RESET}\n"
    )

    # 6. 결과 리스트 생성 및 출력 (Top N 액션 가능 후보만 표시)
    normal_results = build_result_rows(normal_sizing)
    sangdda_results = (
        build_result_rows(sangdda_sizing) if not df_sangdda.empty else []
    )

    print_table(
        select_top_actionable(normal_results), "일반 분석 결과 (Dynamic Sizing)"
    )
    print_table(
        select_top_actionable(sangdda_results), "상따(29.9%) 시나리오 결과", minimal=True
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
