"""일일 예측 입력 준비와 Fast Inference + Dynamic Sizing 서비스."""

from __future__ import annotations

import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.ml.feature_manifest import build_feature_manifest
from src.ml.sizing_engine import predict_daily_position_sizing
from src.processing.feature_catalog import build_causal_feature_matrix
from src.processing.preprocessor import (
    _ROBUST_Z_COLUMNS,
    _apply_robust_z,
    engineer_features,
)
from src.processing.schema import normalize_column_names

logger = logging.getLogger(__name__)


def apply_standard_feature_engineering(
    df: pd.DataFrame, price_history_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """당일 스냅샷을 학습 파이프라인과 1:1 동일한 표준 ML 피처 스키마로 정규화합니다.

    ``normalize_column_names`` 단일 정규화로 일일 CSV 의 한글/괄호 헤더를 표준
    영문 컬럼으로 변환한 뒤 ``engineer_features`` / ``_apply_robust_z`` 를
    적용합니다. 당일 스냅샷에는 존재하지 않는 ``trade_date``(오늘)를 보강합니다.
    ``buy_price`` 가 없으면 이전 종가 중립값 대신 유한 양수 ``close_price`` 로
    대체합니다(학습 대비 서빙 피처 정합성). 명시적으로 공급된 ``buy_price`` 는
    변경하지 않습니다. 표시용 메타데이터(``종목명`` 등)는 보존합니다.

    ``price_history_df`` 가 주어지면 ``causal_expanded_v1`` 카탈로그를 재현해
    당일 스냅샷 + prior 이력으로 후보 행렬을 추가하고 카탈로그 메타데이터
    매니페스트를 attrs 에 기록합니다.
    """
    work = df.copy()
    work = normalize_column_names(work)
    if "trade_date" not in work.columns:
        work["trade_date"] = pd.Timestamp.today().normalize()
    if "buy_price" not in work.columns:
        if "close_price" not in work.columns:
            raise ValueError(
                "buy_price is absent and close_price is missing; cannot derive buy_price"
            )
        close_price = pd.to_numeric(work["close_price"], errors="coerce")
        if close_price.isna().any() or not np.isfinite(close_price.to_numpy(dtype=np.float64)).all():
            raise ValueError(
                "buy_price is absent and close_price is non-finite; cannot derive buy_price"
            )
        if (close_price <= 0.0).any():
            raise ValueError(
                "buy_price is absent and close_price is non-positive; cannot derive buy_price"
            )
        work["buy_price"] = close_price
    work = engineer_features(work)
    # 학습 파이프라인(build_ml_dataset)과 1:1 동일한 횡단면 Robust Z-Score
    # 피처(change_rate_z, major_density_z 등)를 생성합니다.
    work = _apply_robust_z(work, _ROBUST_Z_COLUMNS)
    if price_history_df is not None:
        catalog_matrix, catalog_manifest = build_causal_feature_matrix(work, price_history_df)
        work = work.join(catalog_matrix, how="left")
        manifest = build_feature_manifest(
            list(catalog_matrix.columns), catalog_metadata=catalog_manifest
        )
        work.attrs["feature_manifest"] = manifest
        work.attrs["catalog_version"] = catalog_matrix.attrs["catalog_version"]
        work.attrs["catalog_hash"] = catalog_matrix.attrs["catalog_hash"]
    return work


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


# 출력 테이블 기본 상위 후보 수 (Top N)
_DEFAULT_TOP_N = 15


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

    ``feature_selection_version`` 을 선언한 번들(선택 인지 번들)은 매니페스트가
    available 로 선언한 선별 피처가 누락되거나 결정 시점 이용 가능해야 하는
    피처가 유한하지 않으면 ``ValueError`` 로 fail-closed 합니다. 레거시 번들은
    기존 0-fill 동작을 유지합니다.
    """
    if feature_cols is None:
        feature_cols = list(models_bundle.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("feature_cols is empty; models_bundle must declare feature_cols")

    work = df.copy()
    selection_version = models_bundle.get("feature_selection_version")
    if selection_version is not None:
        missing = [col for col in feature_cols if col not in work.columns]
        if missing:
            raise ValueError(
                f"missing selected features in serving frame: {missing}; "
                "selection-aware serving fails closed on absent selected features"
            )
        manifest = models_bundle.get("feature_manifest")
        if manifest is not None and "feature_name" in manifest.columns and "availability_rule" in manifest.columns:
            rule_map = dict(
                zip(
                    manifest["feature_name"].astype(str),
                    manifest["availability_rule"].astype(str),
                    strict=False,
                )
            )
            for col in feature_cols:
                if rule_map.get(col) == "at_decision_time":
                    arr = work[col].to_numpy(dtype=np.float64)
                    if not np.isfinite(arr).all():
                        raise ValueError(
                            f"selected feature {col!r} must be finite at decision time; "
                            "selection-aware serving fails closed"
                        )
    else:
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
