"""Quantile Regression & Calibrated Classifier 기반 위험 예측 모듈.

후보 종목의 미래 수익률 분위수(pred_q10/pred_q50/pred_q90)를 LightGBM
Quantile Regression 으로 예측하고, 목표달성/손실 확률(p_good/p_bad)을
Sigmoid(Platt) Calibrated GBDT 분류기로 추정합니다. Purged Group Walk-Forward
CV 로 fold 단위 학습되어 미래 정보 누수(Data Leakage)를 차단합니다.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV

from src.ml.purged_cv import PurgedGroupTimeSeriesSplit

logger = logging.getLogger(__name__)

_QUANTILE_ALPHAS = (0.10, 0.50, 0.90)
_QUANTILE_COLS = ("pred_q10", "pred_q50", "pred_q90")
_GOOD_THRESHOLD = 0.01
_BAD_THRESHOLD = -0.015
_GOOD_COL = "_y_good"
_BAD_COL = "_y_bad"
_PREDICTION_COLS = ("pred_q10", "pred_q50", "pred_q90", "p_good", "p_bad")


def _fit_predict_quantile(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    alpha: float,
) -> np.ndarray:
    """단일 fold 에서 alpha 분위수 LGBMRegressor 를 학습하고 OOF 예측을 반환합니다."""
    model = LGBMRegressor(objective="quantile", alpha=alpha, random_state=42, verbosity=-1)
    model.fit(train[feature_cols], train[target_col].to_numpy())
    return np.asarray(model.predict(val[feature_cols]), dtype=np.float64)


def _fit_predict_calibrated(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> np.ndarray:
    """Sigmoid(Platt) CalibratedClassifierCV 로 목표 이벤트 확률을 예측합니다.

    Train fold 에 단일 클래스만 존재하면 분포 손실을 방지하기 위해 사전확률(prior)을
    상수 예측으로 반환합니다.
    """
    y_train = train[target_col].to_numpy().astype(bool)
    if np.unique(y_train).size < 2 or int(np.min(np.bincount(y_train))) < 3:
        return np.full(len(val), float(np.mean(y_train)))
    base = LGBMClassifier(objective="binary", random_state=42, verbosity=-1)
    calibrator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    calibrator.fit(train[feature_cols], y_train)
    proba = calibrator.predict_proba(val[feature_cols])
    positive_idx = list(calibrator.classes_).index(True)
    return np.asarray(proba[:, positive_idx], dtype=np.float64)


def fit_predict_quantile_and_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    purge_gap: int = 1,
) -> pd.DataFrame:
    """Quantile Regression + Calibrated Classifier 를 Purged Walk-Forward CV 로 학습.

    Parameters
    ----------
    df : pd.DataFrame
        ``group_col``, ``target_col``, ``feature_cols`` 컬럼을 포함한 후보 종목 데이터.
    feature_cols : list[str]
        모델 입력 Feature 컬럼 목록.
    target_col : str
        (수익률) Target 컬럼.
    group_col : str
        (날짜) 그룹 컬럼 — 동일 그룹은 항상 같은 fold 에 속합니다.
    n_splits : int
        Walk-Forward fold 수.
    purge_gap : int
        Train/Validation 경계에서 제외할 group 수(보유기간 purge).

    Returns
    -------
    pd.DataFrame
        OOF 예측 대상 행에 대해 원본 키 컬럼과 함께
        ``pred_q10``, ``pred_q50``, ``pred_q90``, ``p_good``, ``p_bad`` 컬럼을 포함.
        분위수 단조성(q10 <= q50 <= q90)은 클리핑으로 보장됩니다.
    """
    missing_cols = [col for col in [*feature_cols, target_col, group_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns in df: {missing_cols}")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")

    work = df.sort_values(group_col).copy()
    work[_GOOD_COL] = work[target_col] >= _GOOD_THRESHOLD
    work[_BAD_COL] = work[target_col] <= _BAD_THRESHOLD

    splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)
    pred_parts: list[pd.DataFrame] = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(work, y=work[target_col], groups=work[group_col])):
        train = work.iloc[train_idx]
        val = work.iloc[val_idx]

        quantile_preds: dict[str, np.ndarray] = {}
        for col, alpha in zip(_QUANTILE_COLS, _QUANTILE_ALPHAS, strict=True):
            quantile_preds[col] = _fit_predict_quantile(train, val, feature_cols, target_col, alpha)

        fold = pd.DataFrame({col: quantile_preds[col] for col in _QUANTILE_COLS}, index=val.index)
        fold["p_good"] = _fit_predict_calibrated(train, val, feature_cols, _GOOD_COL)
        fold["p_bad"] = _fit_predict_calibrated(train, val, feature_cols, _BAD_COL)
        pred_parts.append(fold)
        logger.info("fold=%d train=%d val=%d", fold, len(train_idx), len(val_idx))

    pred_df = pd.concat(pred_parts).sort_index()
    out = work.loc[pred_df.index].copy()
    for col in _PREDICTION_COLS:
        out[col] = pred_df[col].to_numpy()

    q10 = out["pred_q10"].to_numpy(dtype=np.float64)
    q50 = out["pred_q50"].to_numpy(dtype=np.float64)
    q90 = out["pred_q90"].to_numpy(dtype=np.float64)
    q10 = np.minimum(np.minimum(q10, q50), q90)
    q50 = np.clip(q50, q10, q90)
    q90 = np.maximum(q90, q50)
    out["pred_q10"] = q10
    out["pred_q50"] = q50
    out["pred_q90"] = q90

    return out.drop(columns=[_GOOD_COL, _BAD_COL])
