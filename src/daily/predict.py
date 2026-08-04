import datetime
import json
import logging
import os
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DB 및 시트 관련 임포트
from src.data.db_loader import load_theme_from_db
from src.data.sync_sheet_db import sync_theme_only
from src.ml.model_pipeline import run_model_pipeline  # noqa: F401  (Purged Walk-Forward CV 학습 파이프라인 진입점)
from src.ml.sizing_engine import (
    _train_inline_bundle,
    load_model_artifacts,
    predict_daily_position_sizing,
    save_model_artifacts,
)
from src.processing.preprocessor import (  # noqa: F401  (파퀘 기반 학습 파이프라인 진입점)
    _ROBUST_Z_COLUMNS,
    _apply_robust_z,
    build_ml_dataset,
    clean_column_names,
    engineer_features,
)
from src.processing.schema import normalize_column_names
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
# 3. 메인 실행 함수 (Fast Inference + Dynamic Sizing)
# =========================================================


# =========================================================
# 2. 표준 ML 피처 스키마 정합성 + 출력 포맷 헬퍼
# =========================================================

# 출력 테이블 기본 상위 후보 수 (Top N)
_DEFAULT_TOP_N = 15


def apply_standard_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """당일 스냅샷을 학습 파이프라인과 1:1 동일한 표준 ML 피처 스키마로 정규화합니다.

    ``normalize_column_names`` 단일 정규화로 일일 CSV 의 한글/괄호 헤더를 표준
    영문 컬럼으로 변환한 뒤 ``engineer_features`` / ``_apply_robust_z`` 를
    적용합니다. 당일 스냅샷에는 존재하지 않는 ``trade_date``(오늘)와
    ``buy_price``(전일종가 기준 중립값)를 보강하고, 표시용 메타데이터
    (``종목명`` 등)는 보존합니다.
    """
    work = df.copy()
    work = normalize_column_names(work)
    if "trade_date" not in work.columns:
        work["trade_date"] = pd.Timestamp.today().normalize()
    if "buy_price" not in work.columns:
        work["buy_price"] = work["prev_close_price"]
    work = engineer_features(work)
    # 학습 파이프라인(build_ml_dataset)과 1:1 동일한 횡단면 Robust Z-Score
    # 피처(change_rate_z, major_density_z 등)를 생성합니다.
    return _apply_robust_z(work, _ROBUST_Z_COLUMNS)


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
        work,
        feature_cols,
        group_col=group_col,
        models_bundle=models_bundle,
    )


# 모델 번들 검증 상수: 단위 테스트 픽스처가 생성하는 더미 피처 패턴 및
# 실데이터 학습 번들이 반드시 포함해야 하는 모델 키 목록.
_DUMMY_FEATURE_PATTERN = re.compile(r"^f\d+$")
_MODEL_BUNDLE_KEYS = ("rank_model", "quantile_models", "calibrators")


def _is_dummy_feature_cols(feature_cols: list[str]) -> bool:
    """더미 테스트 피처(예: ['f1', 'f2', 'f3']) 여부를 결정적으로 판별합니다."""
    return any(_DUMMY_FEATURE_PATTERN.match(col) for col in feature_cols)


def _is_valid_real_bundle(models_bundle: dict[str, Any]) -> bool:
    """저장된 번들이 실데이터 학습 산출물인지 검증합니다.

    실데이터 번들은 비어있지 않은 수치 피처 목록과 rank/quantile/calibrator
    모델 키를 모두 포함해야 합니다. 더미 피처만으로 구성된 번들은 무효로
    간주합니다.
    """
    feature_cols = list(models_bundle.get("feature_cols", []))
    if not feature_cols or _is_dummy_feature_cols(feature_cols):
        return False
    return all(key in models_bundle for key in _MODEL_BUNDLE_KEYS)


def train_and_save_real_model_bundle(
    export_dir: str = "artifacts/models",
    trade_log_path: str | os.PathLike[str] | None = None,
    theme_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """``trade_log.parquet`` 실데이터로 표준 ML 번들을 학습·저장한 뒤 반환합니다.

    ``build_ml_dataset`` 산출 feature_cols 에서 범주형 문자열 컬럼
    (``market_type``, ``theme_sector``, ``chart_analysis``)을 제외한 수치
    피처만 LightGBM 학습에 사용해 Booster 구성 오류를 방지합니다.
    """
    trade_log_path = str(trade_log_path or settings.TRADE_LOG_PARQUET_PATH)
    theme_path = str(theme_path or settings.THEME_PARQUET_PATH)
    trade_log_df = pd.read_parquet(trade_log_path)
    theme_df = pd.read_parquet(theme_path) if os.path.exists(theme_path) else None
    X, targets, cat_features, processed = build_ml_dataset(trade_log_df, theme_df)
    feature_cols = [col for col in X.columns if col not in cat_features]
    target_col = "target_return"
    group_col = "trade_date"
    bundle = _train_inline_bundle(
        processed[[*feature_cols, target_col, group_col]],
        feature_cols,
        target_col,
        group_col,
    )
    save_model_artifacts(bundle, export_dir)
    logger.info(
        f"{Colors.GREEN}실데이터 모델 번들 재학습·저장 완료: "
        f"feature_cols={len(feature_cols)}개 (export_dir={export_dir}){Colors.RESET}"
    )
    return bundle


def ensure_valid_model_bundle(models_bundle: dict[str, Any]) -> dict[str, Any]:
    """더미/무효 모델 번들을 감지하면 실데이터로 재학습한 번들로 대체합니다."""
    if _is_valid_real_bundle(models_bundle):
        return models_bundle
    logger.info(
        f"{Colors.YELLOW}[Warning] 모델 번들이 더미 테스트 피처이거나 무효합니다. "
        f"trade_log.parquet 로 실데이터 재학습을 시작합니다.{Colors.RESET}"
    )
    return train_and_save_real_model_bundle()


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

    # 결과 요약
    logger.info(
        f"{Colors.CYAN}분석 대상: {len(df_condition)}개 종목 "
        f"(시나리오 확장 포함 {len(df_all)}건){Colors.RESET}"
    )

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

    df_all = apply_standard_feature_engineering(df_all)

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

    # 모델 번들 검증: 더미 테스트 피처('f1'/'f2'/'f3') 또는 무효 번들이면
    # trade_log.parquet 실데이터로 자동 재학습합니다.
    models_bundle = ensure_valid_model_bundle(models_bundle)

    start = time.perf_counter()
    df_normal = df_all[~df_all["시나리오"].str.contains("상따", na=False)].copy()
    df_sangdda = df_all[df_all["시나리오"].str.contains("상따", na=False)].copy()

    normal_sizing = run_daily_sizing_inference(df_normal, models_bundle)
    sangdda_sizing = (
        run_daily_sizing_inference(df_sangdda, models_bundle)
        if not df_sangdda.empty
        else pd.DataFrame()
    )
    sizing_df = (
        pd.concat([normal_sizing, sangdda_sizing], ignore_index=True)
        if not sangdda_sizing.empty
        else normal_sizing
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{Colors.GREEN}추론 완료: {len(sizing_df)}건 ({elapsed_ms:.0f}ms){Colors.RESET}"
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
