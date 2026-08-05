"""폴드 학습과 OOF 조립 (fold training + OOF assembly)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker, LGBMRegressor
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.training.validation import calculate_recency_sample_weight


def _group_relevance(target: pd.Series, groups: pd.Series) -> pd.Series:
    """날짜별 Cross-sectional Rank Percentile 을 0~4 등급 relevance 로 동적 변환.

    일자별 후보 수가 상이해도 횡단면 순위 기반이라 바운더리가 왜곡되지 않습니다.
    """
    pct = target.groupby(groups).rank(pct=True, method="average")
    return (pct * 4.0).round().astype(int)


def _ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """단일 그룹의 예측 순서 기준 NDCG@k 를 계산합니다."""
    k = min(k, relevance.size)
    if k <= 0:
        return 0.0
    gains = np.power(2.0, relevance[:k].astype(np.float64)) - 1.0
    discounts = np.log2(np.arange(1, k + 1) + 1.0)
    ideal = np.sort(relevance.astype(np.float64))[::-1][:k]
    idcg = float(np.sum((np.power(2.0, ideal) - 1.0) / discounts))
    if idcg <= 0.0:
        return 0.0
    dcg = float(np.sum(gains / discounts))
    return dcg / idcg


def _group_metric(
    oof: pd.DataFrame,
    group_col: str,
    metric_fn: Callable[[pd.DataFrame], float],
) -> float:
    """그룹별로 metric_fn 을 적용한 값의 평균을 반환합니다."""
    values: list[float] = []
    for _, group in oof.groupby(group_col, sort=False):
        values.append(metric_fn(group))
    return float(np.mean(values))


def _compute_ndcg(oof: pd.DataFrame, group_col: str, k: int) -> float:
    def per_group(group: pd.DataFrame) -> float:
        order = group["pred"].to_numpy().argsort()[::-1]
        return _ndcg_at_k(group["relevance"].to_numpy()[order], k)

    return _group_metric(oof, group_col, per_group)


def _compute_rank_ic(oof: pd.DataFrame, group_col: str, target_col: str) -> float:
    ics: list[float] = []
    for _, group in oof.groupby(group_col, sort=False):
        if len(group) < 2:
            continue
        if float(np.std(group["pred"].to_numpy())) == 0.0:
            continue
        result = spearmanr(group["pred"], group[target_col])
        ic = float(result.statistic)
        if not np.isnan(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def _compute_top_k_return(oof: pd.DataFrame, group_col: str, target_col: str, k: int) -> float:
    def per_group(group: pd.DataFrame) -> float:
        order = group["pred"].to_numpy().argsort()[::-1][:k]
        return float(group[target_col].to_numpy()[order].mean())

    return _group_metric(oof, group_col, per_group)


def _fit_predict(
    model_type: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    model_params: dict[str, int | float] | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """model_type 에 따라 fold 단위 모델을 학습하고 OOF 예측을 반환합니다.

    ``model_params`` 는 요청된 모델에만 전달되며 ``random_state=42`` 는 유지합니다.
    ``sample_weight`` 는 회귀 champion(Huber) 경로에서만 지원되며, 검증 라벨은
    가중치 계산에 절대 사용되지 않습니다. Ridge/LGBMRanker 는 비검증된 동작을
    방지하기 위해 비영(非零) recency 가중치를 ``ValueError`` 로 거부합니다.
    """
    params: dict[str, Any] = dict(model_params or {})
    if model_type == "ridge":
        if sample_weight is not None:
            raise ValueError("recency sample weighting is not supported for ridge")
        medians = train[feature_cols].median(numeric_only=True).reindex(feature_cols).fillna(0.0)
        train_features = train[feature_cols].fillna(medians)
        val_features = val[feature_cols].fillna(medians)
        model = make_pipeline(StandardScaler(), Ridge(**params))
        model.fit(train_features, train[target_col])
        return model, model.predict(val_features)

    if model_type == "lgb_regressor":
        model = LGBMRegressor(objective="huber", random_state=42, **params)
        model.fit(train[feature_cols], train[target_col], sample_weight=sample_weight)
        return model, model.predict(val[feature_cols])

    if sample_weight is not None:
        raise ValueError("recency sample weighting is not supported for lgb_ranker")
    # lgb_ranker: 동일 query(date) 샘플이 연속되도록 정렬 후 group counts 전달
    train_sorted = train.sort_values(group_col)
    relevance = _group_relevance(train_sorted[target_col], train_sorted[group_col]).to_numpy()
    group_counts = train_sorted[group_col].value_counts(sort=False).to_numpy(dtype=np.int64)
    model = LGBMRanker(objective="lambdarank", random_state=42, **params)
    model.fit(train_sorted[feature_cols], relevance, group=group_counts)
    return model, model.predict(val[feature_cols])


def _fit_predict_linear_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> np.ndarray:
    """StandardScaler + Ridge linear baseline 을 동일 fold 에서 학습·예측합니다."""
    medians = train[feature_cols].median(numeric_only=True).reindex(feature_cols).fillna(0.0)
    train_features = train[feature_cols].fillna(medians)
    val_features = val[feature_cols].fillna(medians)
    model = make_pipeline(StandardScaler(), Ridge())
    model.fit(train_features, train[target_col])
    return np.asarray(model.predict(val_features), dtype=np.float64)


def _align_close_morning_oof(
    return_oof: pd.DataFrame,
    risk_oof: pd.DataFrame,
    target_col: str,
    group_col: str,
) -> pd.DataFrame:
    """return OOF 와 quantile/classifier OOF 를 원본 행 인덱스로 정렬합니다.

    날짜 단독 병합은 금지합니다. return 예측 행이 risk OOF 에 없거나 p_good /
    p_bad 예측이 누락되거나 group/target/stock/scenario 식별 키가 원본 인덱스에서
    어긋나면 ``ValueError`` 로 fail-closed 합니다 (누락 OOF 예측은 대체하지 않음).
    """
    missing = return_oof.index.difference(risk_oof.index)
    if len(missing):
        raise ValueError(
            "close-morning OOF alignment: return-prediction rows are missing from "
            f"quantile/classifier OOF ({len(missing)} rows); missing OOF predictions "
            "are never substituted"
        )
    aligned = return_oof.join(risk_oof[["p_good", "p_bad"]], how="left")
    for col in ("p_good", "p_bad"):
        if aligned[col].isna().any():
            raise ValueError(
                "close-morning OOF alignment: "
                f"{col} predictions are missing for aligned return rows"
            )
    for col in (group_col, target_col):
        if col not in risk_oof.columns:
            continue
        if not aligned[col].equals(risk_oof.loc[aligned.index, col]):
            raise ValueError(
                f"close-morning OOF alignment: {col} mismatch between return and risk "
                "OOF on original index"
            )
    for col in ("stock_code", "chart_analysis"):
        if col not in risk_oof.columns:
            continue
        if not aligned[col].equals(risk_oof.loc[aligned.index, col]):
            raise ValueError(
                f"close-morning OOF alignment: {col} mismatch between return and risk "
                "OOF on original index"
            )
    return aligned


def assemble_fold_oof(
    val: pd.DataFrame,
    pred: np.ndarray,
    pred_linear: np.ndarray,
    fold: int,
    group_col: str,
    target_col: str,
) -> pd.DataFrame:
    """단일 폴드의 OOF 예측 DataFrame 을 조립합니다 (metadata 컬럼은 호출부가 주입)."""
    return pd.DataFrame(
        {
            group_col: val[group_col].to_numpy(),
            target_col: val[target_col].to_numpy(),
            "pred": pred,
            "pred_linear": pred_linear,
            "fold": fold,
        },
        index=val.index,
    )


def recency_sample_weight_for_fold(
    train: pd.DataFrame,
    group_col: str,
    recency_half_life_groups: int | None,
) -> np.ndarray | None:
    """폴드 train 그룹에 대한 recency sample weight (None 이면 None)."""
    if recency_half_life_groups is None:
        return None
    return calculate_recency_sample_weight(train[group_col], recency_half_life_groups)
