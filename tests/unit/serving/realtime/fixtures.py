"""고정 번들/스냅샷 픽스처 (live serving 테스트 전용).

실시간 추론 경로는 절대 학습하지 않으므로, 테스트는 결정적(고정 시드)으로
구성된 고정 번들을 사용합니다. 학습 파이프라인(legacy)을 import 하지 않고
lightgbm 모델을 직접 구성합니다. 이 모듈은 ``test_*.py`` 가 아니므로 pytest
수집 대상이 아닙니다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.serving.realtime.inference import _QUANTILE_ALPHAS, _QUANTILE_COLS

_QUANTILE_ALPHAS = _QUANTILE_ALPHAS
_QUANTILE_COLS = _QUANTILE_COLS


def build_fixed_serving_bundle(
    feature_cols: list[str],
    n_rows: int = 60,
    seed: int = 7,
) -> dict[str, Any]:
    """고정 시드로 훈련된 소형 예측 모델 번들을 결정적으로 구성합니다.

    ``return_model``/``rank_model`` 은 Huber 회귀, ``quantile_models`` 는
    분위수 회귀, ``calibrators`` 는 상수 prior 로 구성해 추론 경로의
    decision/utility/grade/allocation 산출을 검증합니다.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.normal(size=n_rows) for col in feature_cols})
    y = 0.05 * df[feature_cols[0]] + rng.normal(scale=0.01, size=n_rows)

    return_model = LGBMRegressor(objective="huber", random_state=42, verbosity=-1)
    return_model.fit(df, y)

    quantile_models: dict[str, Any] = {}
    for col, alpha in zip(_QUANTILE_COLS, _QUANTILE_ALPHAS, strict=True):
        model = LGBMRegressor(objective="quantile", alpha=alpha, random_state=42, verbosity=-1)
        model.fit(df, y)
        quantile_models[col] = model

    return {
        "feature_cols": list(feature_cols),
        "return_model": return_model,
        "rank_model": return_model,
        "quantile_models": quantile_models,
        "calibrators": {"p_good": 0.5, "p_bad": 0.1},
    }


def fixed_reranker_bundle(feature_cols: list[str]) -> dict[str, Any]:
    """close-morning reranker v1 설정이 포함된 고정 번들을 반환합니다."""
    bundle = build_fixed_serving_bundle(feature_cols)
    bundle["decision_score_config"] = {
        "version": "close-morning-reranker-v1",
        "rank_weight": 1.0,
        "p_good_weight": 0.5,
        "score_col": "decision_score",
    }
    return bundle


def scored_snapshot_df(
    feature_cols: list[str],
    group_col: str = "date",
    n_groups: int = 2,
    n_rows: int = 12,
    seed: int = 11,
) -> pd.DataFrame:
    """그룹별 후보 행으로 구성된 스코어링용 스냅샷 DataFrame 을 반환합니다."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for g in range(n_groups):
        for _ in range(n_rows):
            row = {group_col: f"2026-08-0{g + 1}"}
            for col in feature_cols:
                row[col] = float(rng.normal())
            row["stock_code"] = f"{len(rows) + 1:06d}"
            row["chart_analysis"] = "거래량 폭증"
            rows.append(row)
    return pd.DataFrame(rows)


def daily_snapshot_df() -> pd.DataFrame:
    """일일 CSV(daily_stocks.csv)와 동일한 스프레드시트 컬럼명의 당일 스냅샷을 생성합니다."""
    return pd.DataFrame(
        {
            "시나리오": ["거래량 폭증", "상따"],
            "종목명": ["AAA", "BBB"],
            "종목코드": ["000001", "000002"],
            "시가": [10_000.0, 20_000.0],
            "고가": [11_000.0, 22_000.0],
            "저가": [9_000.0, 19_000.0],
            "종가": [10_500.0, 21_000.0],
            "전일종가": [10_000.0, 20_000.0],
            "시가총액": [1_000.0, 2_000.0],
            "거래대금": [100.0, 200.0],
            "등락률": [22.0, 20.0],
            "선정순위": [1, 2],
            "기관_순매수": [10.0, 20.0],
            "외국인_순매수": [5.0, 10.0],
            "프로그램_순매수": [2.0, 4.0],
            "체결강도": [110.0, 120.0],
            "시장구분": ["KOSPI", "KOSDAQ"],
            "총_종목수": [50, 50],
            "평균_거래대금": [80.0, 80.0],
            "kospi": [0.5, 0.5],
            "kosdaq": [0.3, 0.3],
            "v_kospi": [15.0, 15.0],
            "v_kosdaq": [18.0, 18.0],
            "거래량": [1_000_000, 2_000_000],
            "테마_섹터": ["테마A", "테마A"],
            "차트분석": ["거래량 폭증", "상따"],
        }
    )


def snapshot_feature_cols() -> list[str]:
    """``build_snapshot_features`` 가 생성하는 고정 번들용 피처 컬럼."""
    return [
        "change_rate",
        "rank_ratio",
        "major_density",
        "log_market_cap_100m",
        "log_trade_value_100m",
        "turnover",
    ]
