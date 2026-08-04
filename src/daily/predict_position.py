import datetime
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
from joblib import load

logger = logging.getLogger(__name__)

# DB 및 시트 관련 임포트
from src.data.db_loader import load_theme_from_db
from src.data.sync_sheet_db import sync_theme_only
from src.processing.preprocessor_v2 import build_ml_dataset  # noqa: F401  (파퀘 기반 학습 파이프라인 진입점)
from src.utils.display import Colors, print_table

# SHAP (모델 해석용) - 필요 시 설치: pip install shap
try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# GMM (동적 의사결정용)
try:
    from sklearn.mixture import GaussianMixture

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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
        df = pd.read_excel(file_path, engine="openpyxl")
    except Exception as e:
        logger.info(f"{Colors.RED}엑셀 파일 로드 실패: {e}{Colors.RESET}")
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


# =========================================================
# 2. 동적 의사결정 로직 (GMM & Safety Floor)
# =========================================================

# GMM 적용을 위한 최소/최적화 파라미터 (추후 Find_Opt_Threshold 등으로 최적화 가능)
MIN_SAMPLES_FOR_GMM = 10  # 최소 샘플 수
MIN_STD_FOR_GMM = 0.05  # 최소 표준편차 (변별력 기준 1)
MIN_RANGE_FOR_GMM = 0.10  # 최소 점수 범위 (변별력 기준 2)

# Safety Floor (AI 최적화 임계값 - Find_Opt_Threshold.py 결과)
# 최적 Bins: [-inf, 0.5868, 0.8717, 1.0726, inf]
SAFETY_MAX_FLOOR = 1.07  # Max_Expand가 되기 위한 그룹 평균 최소값 (AI 추천)
SAFETY_EXPAND_FLOOR = 0.87  # Expand가 되기 위한 그룹 평균 최소값 (AI 추천)
ABSOLUTE_MIN_SCORE = 0.59  # 이 점수 미만은 무조건 Reduce (AI 추천)


def get_decision_batch(scores: pd.Series) -> pd.Series:
    """GMM을 사용하여 동적으로 등급을 부여하되, 데이터가 부족하거나 변별력이 없으면
    기존의 고정 임계값 방식을 사용합니다. 또한 Safety Floor를 적용하여 리스크를 관리합니다.
    """

    # 1. 고정 임계값 로직 (Fallback & Basic)
    def apply_static_logic(s):
        # AI 최적화 임계값 (Find_Opt_Threshold.py 결과)
        # Reduce (~0.59), Neutral (0.59~0.87), Expand (0.87~1.07), Max (1.07~)
        if s < 0.59:
            return "Reduce"
        elif s < 0.87:
            return "Neutral"
        elif s < 1.07:
            return "Expand"
        else:
            return "Max_Expand"

    # GMM 사용 불가 조건 체크
    if not HAS_SKLEARN:
        return scores.apply(apply_static_logic)

    n_samples = len(scores)
    score_std = scores.std()
    score_range = scores.max() - scores.min()

    # 2. GMM 적용 여부 판단
    # 샘플이 너무 적거나, 점수가 다들 비슷비슷하면(표준편차/범위 미달) 굳이 억지로 나누지 않음
    if (
        n_samples < MIN_SAMPLES_FOR_GMM
        or score_std < MIN_STD_FOR_GMM
        or score_range < MIN_RANGE_FOR_GMM
    ):

        # 로깅 (필요 시 주석 해제)
        # logger.info(f"  > [Static Logic] Samples={n_samples}, Std={score_std:.4f}, Range={score_range:.4f}")
        return scores.apply(apply_static_logic)

    # 3. GMM Clustering
    try:
        # 데이터 shape 변환 (N, 1)
        X = scores.values.reshape(-1, 1)

        # 컴포넌트 수는 최대 4개, 샘플 수보다 작아야 함
        n_components = min(4, n_samples)

        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)

        # 4. 클러스터 의미 매핑 (Labels 0,1,2,3 -> Reduce, Neutral, Expand, Max_Expand)
        # 각 클러스터의 평균 점수를 구해서 오름차순 정렬
        cluster_means = []
        for i in range(n_components):
            mean_score = X[labels == i].mean()
            cluster_means.append((mean_score, i))

        # 평균 점수가 낮은 순서대로 정렬
        cluster_means.sort(key=lambda x: x[0])

        # 순위 매핑 (Rank 0 = Lowest Score Group)
        rank_map = {
            original_label: rank
            for rank, (_, original_label) in enumerate(cluster_means)
        }

        # 가능한 등급 리스트 (4단계)
        decision_levels = ["Reduce", "Neutral", "Expand", "Max_Expand"]

        final_decisions = []
        for i, score in enumerate(scores):
            # 개별 절대 과락 체크 (Safety Floor 1)
            if score < ABSOLUTE_MIN_SCORE:
                final_decisions.append("Reduce")
                continue

            cluster_label = labels[i]
            rank = rank_map[
                cluster_label
            ]  # 0, 1, 2, 3 중 하나 (또는 n_components-1 까지)

            # 컴포넌트 개수가 4개 미만일 경우 처리를 위해 매핑 조정
            # 예: 2개 그룹이면 -> 0(Reduce), 1(Neutral/Expand?) -> 단순 매핑보다 그룹 평균 기반 매핑이 나음
            # 여기서는 간단히 4등분 논리를 적용하되, 그룹 평균 점수를 Safety Floor로 재검증

            # 현재 그룹의 평균 점수
            group_mean = cluster_means[rank][0]

            # 기본 등급 할당 (그룹 순위에 따라 최대치 부여)
            # n_components가 4이면: 0->Red, 1->Neu, 2->Exp, 3->Max
            # n_components가 3이면: 0->Red, 1->Neu, 2->Exp (Max 없음)

            # 인덱스 스케일링: (rank / (n_components-1)) * 3 -> 0~3 사이 실수
            if n_components > 1:
                scaled_rank = int(round((rank / (n_components - 1)) * 3))
            else:
                scaled_rank = 0  # 그룹이 1개면 다 Reduce? -> 아니면 평균 점수 따라감

            base_decision = decision_levels[scaled_rank]

            # 5. Safety Floor 적용 (강등 로직 - Group Mean 기준)
            # Max_Expand가 되려면 그룹 평균이 SAFETY_MAX_FLOOR(0.6) 이상이어야 함
            if base_decision == "Max_Expand" and group_mean < SAFETY_MAX_FLOOR:
                base_decision = "Expand"  # 1단계 강등

            # Expand가 되려면 그룹 평균이 SAFETY_EXPAND_FLOOR(0.4) 이상이어야 함
            if base_decision == "Expand" and group_mean < SAFETY_EXPAND_FLOOR:
                base_decision = "Neutral"  # 1단계 강등

            # 6. Safety Pass (승격 로직 - Individual Score 기준)
            # 상대평가(GMM)로 인해 억울하게 강등된 고득점 종목 구제
            if score >= 1.07:  # AI 추천: Max_Expand 임계값
                base_decision = "Max_Expand"
            elif score >= 0.87 and base_decision in [
                "Reduce",
                "Neutral",
            ]:  # AI 추천: Expand 임계값
                base_decision = "Expand"

            final_decisions.append(base_decision)

        return pd.Series(final_decisions, index=scores.index)

    except Exception as e:
        logger.info(
            f"{Colors.YELLOW}[Warning] GMM Error: {e} -> Fallback to Static{Colors.RESET}"
        )
        return scores.apply(apply_static_logic)


def main():
    # 1. 모델 및 Explainer 로드 (기존과 동일)
    if not os.path.exists(MODEL_PATH):
        logger.info(f"{Colors.YELLOW}[Warning] 모델 파일이 없습니다.{Colors.RESET}")
        model = None
    else:
        logger.info(f"{Colors.GREEN}모델 로드 중...{Colors.RESET}")
        model = load(MODEL_PATH)

    # 2. 데이터 로드 및 테마 매핑
    df_condition = load_and_preprocess_data(CONDITION_EXCEL_PATH)
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

    # 3. [핵심] 시나리오 확장 - 차트통과=0인 종목은 SMA 120 실시간 계산
    # 차트통과=1: 모든 시나리오
    # 차트통과=0 & SMA 120 미만: "120 돌파" 시나리오만
    # 차트통과=0 & SMA 120 이상 (캔들 필터만): 모든 시나리오

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

        # [Step 4] 시나리오 미지정 차트통과=0 종목: SMA 120 기준으로 분류
        if not df_filtered_without_scenario.empty:
            logger.info(
                f"\n{Colors.CYAN}[차트통과=0 & 시나리오 미지정 종목 SMA 120 분석 중...]{Colors.RESET}"
            )

            import asyncio

            from src.api.kis_client import calculate_stock_sma

            df_sma_below = []  # SMA 120 미만
            df_candle_only = []  # 캔들 필터만 (SMA 120 이상)

            async def check_sma_for_filtered_stocks():
                for idx, row in df_filtered_without_scenario.iterrows():
                    ticker = str(row["종목코드"]).zfill(6)
                    stock_name = row["종목명"]
                    current_price = row.get("종가", row.get("현재가", 0))

                    sma_120, success = await calculate_stock_sma(ticker, sma_period=120)

                    if success and sma_120 > 0:
                        if current_price < sma_120:
                            df_sma_below.append(row)
                            logger.info(
                                f"  📉 {stock_name}: 현재가 {current_price:,}원 < SMA120 {sma_120:,.0f}원 → 120돌파 시나리오만"
                            )
                        else:
                            df_candle_only.append(row)
                            logger.info(
                                f"  📊 {stock_name}: 현재가 {current_price:,}원 >= SMA120 {sma_120:,.0f}원 → 거래량 폭증"
                            )
                    else:
                        df_candle_only.append(row)
                        logger.info(f"  ⚠️  {stock_name}: SMA 계산 실패 → 거래량 폭증")

            asyncio.run(check_sma_for_filtered_stocks())

            # 3-2-B-1. SMA 120 미만 종목: "120 돌파" + "상따" 시나리오
            if df_sma_below:
                df_sma_below_df = pd.DataFrame(df_sma_below)

                def get_sma_below_scenarios(row):
                    # 120 돌파 관련 시나리오 자동 선택
                    breakthrough_scenarios = [
                        s for s in DEFAULT_SCENARIOS if "120" in s or "돌파" in s
                    ]
                    if not breakthrough_scenarios:
                        breakthrough_scenarios = [DEFAULT_SCENARIOS[0]]

                    scenarios = breakthrough_scenarios[:]
                    if row.get("등락률", 0) >= 20:
                        if "상따" not in scenarios:
                            scenarios.append("상따")
                    return scenarios

                df_sma_below_df["Scenario_List"] = df_sma_below_df.apply(
                    get_sma_below_scenarios, axis=1)
                df_sma_below_exploded = df_sma_below_df.explode("Scenario_List")
                df_sma_below_exploded["Scenario_Base"] = df_sma_below_exploded[
                    "Scenario_List"
                ]
                df_all_parts.append(
                    df_sma_below_exploded.drop(columns=["Scenario_List"])
                )

            # 3-2-B-2. 캔들 필터만 종목: "거래량 폭증" + "상따" 시나리오
            if df_candle_only:
                df_candle_only_df = pd.DataFrame(df_candle_only)

                def get_candle_only_scenarios(row):
                    scenarios = ["거래량 폭증"]
                    if row.get("등락률", 0) >= 20:
                        if "상따" not in scenarios:
                            scenarios.append("상따")
                    return scenarios

                df_candle_only_df["Scenario_List"] = df_candle_only_df.apply(
                    get_candle_only_scenarios, axis=1)
                df_candle_only_exploded = df_candle_only_df.explode("Scenario_List")
                df_candle_only_exploded["Scenario_Base"] = df_candle_only_exploded[
                    "Scenario_List"
                ]
                df_all_parts.append(
                    df_candle_only_exploded.drop(columns=["Scenario_List"])
                )

    # 3-3. 모든 부분 결합
    if df_all_parts:
        df_all = pd.concat(df_all_parts, ignore_index=True)
    else:
        logger.info(f"{Colors.RED}분석할 데이터가 없습니다.{Colors.RESET}")
        return

    # 결과 요약
    logger.info(f"\n{Colors.CYAN}[시나리오 적용 완료]{Colors.RESET}")
    logger.info(f"  ✅ 차트통과: {len(df_passed)}개 × {len(DEFAULT_SCENARIOS)}개 시나리오")

    # [Fix] 변수가 조건문 내부에서만 정의되므로 안전하게 참조
    if df_filtered is not None and not df_filtered.empty:
        # df_sma_below와 df_candle_only는 리스트로 정의되었으므로 존재 여부 확인 후 len() 호출
        sma_below_count = (
            len(df_sma_below) if "df_sma_below" in locals() and df_sma_below else 0
        )
        candle_only_count = (
            len(df_candle_only)
            if "df_candle_only" in locals() and df_candle_only
            else 0
        )
        logger.info(f"  📉 SMA 120 미만: {sma_below_count}개 × 120돌파 시나리오만")
        logger.info(
            f"  📊 캔들 필터만: {candle_only_count}개 × {len(DEFAULT_SCENARIOS)}개 시나리오"
        )
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
        # [V1 Feature Set] 제외 피처 확인 및 제거 (V1_Drop_Rank: 시장구분, 선정순위 제외)
        V1_EXCLUDE_FEATURES = ["시장구분", "선정순위"]

        # 누락된 컬럼 채우기 (범주형/숫자형 구분)
        for col in train_feature_names:
            if col not in X_processed.columns:
                # 범주형이면 "0" (문자열), 숫자형이면 0.0 (실수)
                if col in cat_cols_from_encoder:
                    X_processed[col] = "0"
                else:
                    X_processed[col] = 0.0

        # V1 제외 피처가 train_feature_names에 없는지 확인 (디버깅용)
        excluded_in_model = [f for f in V1_EXCLUDE_FEATURES if f in train_feature_names]
        if excluded_in_model:
            logger.info(
                f"{Colors.YELLOW}[Warning] 학습 모델에 제외되어야 할 피처가 포함됨: {excluded_in_model}{Colors.RESET}"
            )

        X_final = X_processed[train_feature_names]

        # 단 한 번의 호출로 모든 종목/시나리오 예측
        df_all["Score"] = model.predict(X_final)

        # [Debug] 피처 정합성 확인 로그
        logger.info(
            f"{Colors.GREEN}✅ 모델 입력 피처 수: {len(train_feature_names)}{Colors.RESET}"
        )
        logger.info(f"   👉 V1 제외 피처: {V1_EXCLUDE_FEATURES}")

        # ============================================================
        # [SHAP 분석] 예측 근거 확인 (필요 시 아래 주석 해제)
        # ============================================================
        # 각 종목의 점수가 왜 높게/낮게 나왔는지 상위 3개 피처를 분석합니다.
        # 주의: 종목 수가 많으면 시간이 오래 걸릴 수 있습니다.
        # explain_predictions_with_shap(
        #     model=model,
        #     X_final=X_final,
        #     stock_names=df_all["종목명"].tolist(),
        #     top_n=3
        # )
        # ============================================================

    else:
        df_all["Score"] = 0.5

    # 6. 의사결정 로직 일괄 적용 (GMM Dynamic + Safety Floor)
    # [Optimized Thresholds] AI 추천 및 실전 보정 임계값 적용 (2025-12-27 업데이트)
    # GMM을 통해 그날의 시장 난이도에 맞게 상대평가하되, 절대 하한선(Safety)을 지킴
    logger.info(f"{Colors.CYAN}의사결정 등급 산정 중 (GMM Dynamic Logic)...{Colors.RESET}")
    df_all["Decision"] = get_decision_batch(df_all["Score"])

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
