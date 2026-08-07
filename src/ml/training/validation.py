"""ML 학습 파이프라인 입력 및 인과성 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd

_MODEL_TYPES = (
    "ridge",
    "lgb_ranker",
    "lgb_regressor",
    "xgb_regressor",
    "catboost_regressor",
    "random_forest_regressor",
)

# algorithm-family ensemble 실험(ml_ensemble_improvement)에 참여하는 결정적
# numeric-only return 추정기 family 목록입니다. 모든 family 는 동일한 수치형
# close_morning61 피처 컬럼을 사용합니다 (범주형 피처 미사용).
_ALGORITHM_FAMILIES: tuple[str, ...] = (
    "lgb_regressor",
    "xgb_regressor",
    "catboost_regressor",
    "random_forest_regressor",
)


def calculate_recency_sample_weight(
    groups: pd.Series, half_life_groups: int | None
) -> np.ndarray:
    """최근 데이터에 지수 감쇠 가중치를 부여한 row 단위 sample weight 를 반환합니다.

    ``groups`` 의 유일 거래일을 시간순으로 정렬해 최신 그룹 age=0, 과거로 갈수록
    age 가 커집니다. 각 그룹 가중치는 ``exp(-ln(2) * age / half_life_groups)`` 이며
    평균이 정확히 1 이 되도록 정규화해 가중 Huber 학습 시 순수익 타깃의 레이블
    크기가 변하지 않도록 합니다. ``None`` 이면 전부 1 (기존 expanding 동작) 을
    반환합니다.

    ``half_life_groups`` 는 252(1 policy-history 년) 또는 504(2년) 만 허용하며,
    그 외 값·비어 있는 그룹·시간순 파싱 불가 그룹·비유한 출력은 ``ValueError`` 로
    fail-closed 합니다.
    """
    if half_life_groups is None:
        return np.ones(len(groups), dtype=np.float64)
    if half_life_groups not in (252, 504):
        raise ValueError(
            f"recency_half_life_groups must be one of None, 252, 504, got {half_life_groups!r}"
        )
    if len(groups) == 0:
        raise ValueError("recency sample weight requires non-empty trade-date groups")
    unique_groups = pd.unique(groups)
    parsed = pd.to_datetime(pd.Series(unique_groups), errors="coerce")
    if parsed.isna().any():
        raise ValueError(
            "recency sample weight requires parseable chronological trade-date groups"
        )
    order = np.argsort(parsed.to_numpy(), kind="stable")
    sorted_unique = unique_groups[order]
    ages = np.arange(len(sorted_unique), dtype=np.float64)[::-1]
    decay = np.exp(-np.log(2.0) * ages / float(half_life_groups))
    mean_decay = float(decay.mean())
    if not np.isfinite(mean_decay) or mean_decay <= 0.0:
        raise ValueError("recency sample weights are not finite")
    weights = decay / mean_decay
    weight_by_group = dict(zip(sorted_unique, weights.tolist(), strict=True))
    return np.asarray([weight_by_group[g] for g in groups], dtype=np.float64)


def validate_pipeline_inputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    model_type: str,
    purge_gap: int,
    recency_half_life_groups: int | None,
) -> None:
    """``run_model_pipeline`` 입력을 검증합니다 (model_type/컬럼/purge/recency)."""
    if model_type not in _MODEL_TYPES:
        raise ValueError(f"model_type must be one of {list(_MODEL_TYPES)}, got {model_type!r}")
    if recency_half_life_groups is not None:
        if model_type != "lgb_regressor":
            raise ValueError(
                "recency_half_life_groups is only supported for lgb_regressor, "
                f"got model_type={model_type!r}"
            )
        if recency_half_life_groups not in (252, 504):
            raise ValueError(
                "recency_half_life_groups must be one of None, 252, 504, "
                f"got {recency_half_life_groups!r}"
            )
    missing_cols = [col for col in [*feature_cols, target_col, group_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns in df: {missing_cols}")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")
    if purge_gap < 0:
        raise ValueError(f"purge_gap must be >= 0, got {purge_gap}")
