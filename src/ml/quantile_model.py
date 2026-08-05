"""Quantile Regression & Calibrated Classifier 기반 위험 예측 모듈.

후보 종목의 미래 수익률 분위수(pred_q10/pred_q50/pred_q90)를 LightGBM
Quantile Regression 으로 예측하고, 목표달성/손실 확률(p_good/p_bad)을
순차(chronological) 보정된 분류기로 추정합니다.

P0/P1(`docs/specs/ml_strategy_improvement.md`):
- 타깃은 decimal net return 단일 권위 단위이며, 라벨 임계값도 decimal net
  (``_GOOD_THRESHOLD``/``_BAD_THRESHOLD``) 기준입니다.
- 각 outer fold 내부에서 학습 기간을 '이른 순차 fit 구간'과 '이후 보정(calibration)
  구간'으로 나누고, 분위수 잔차 보정(conformal)과 확률 보정(sigmoid/Platt)을
  보정 구간에서만 학습합니다. random KFold/StratifiedKFold/검증 라벨을
  보정이나 임계값 적합에 사용하지 않습니다.
- fold 단위 q10-q90 커버리지/구간 폭/Brier/log-loss 진단을 반환합니다.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression

from src.ml.purged_cv import PurgedGroupTimeSeriesSplit

logger = logging.getLogger(__name__)

_QUANTILE_ALPHAS = (0.10, 0.50, 0.90)
_QUANTILE_COLS = ("pred_q10", "pred_q50", "pred_q90")
# Decimal net 기준 이벤트 임계값 (preprocessor.LABEL_THRESHOLDS 와 동일)
_GOOD_THRESHOLD = 0.01
_BAD_THRESHOLD = -0.02
_GOOD_COL = "_y_good"
_BAD_COL = "_y_bad"
_PREDICTION_COLS = ("pred_q10", "pred_q50", "pred_q90", "p_good", "p_bad")
_CALIB_FRAC = 0.3
_MIN_CALIB_ROWS = 5
_MIN_FIT_GROUPS = 2


def _fit_predict_quantile(
    train: pd.DataFrame,
    pred: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    alpha: float,
) -> np.ndarray:
    """alpha 분위수 LGBMRegressor 를 fit 구간에서 학습하고 ``pred`` 행을 예측합니다."""
    model = LGBMRegressor(objective="quantile", alpha=alpha, random_state=42, verbosity=-1)
    model.fit(train[feature_cols], train[target_col].to_numpy())
    return np.asarray(model.predict(pred[feature_cols]), dtype=np.float64)


def _chrono_fit_calibration_split(
    group_values: np.ndarray,
    train_idx: np.ndarray,
    calib_frac: float = _CALIB_FRAC,
    embargo: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Train 인덱스를 이른 fit 구간과 이후 보정 구간으로 순차 분할합니다.

    보정 구간은 Train 의 가장 최근 ``calib_frac`` 그룹이고, fit 구간은
    보정 경계에서 ``embargo``(보유기간 purge)만큼 앞선 그룹까지만 사용합니다.
    분할이 불가능한 소표본이면 전체를 fit 으로 반환합니다.
    """
    if train_idx.size == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.intp)
    train_groups = group_values[train_idx]
    unique = np.unique(train_groups)
    n = len(unique)
    if n < _MIN_FIT_GROUPS + 1:
        return train_idx, np.array([], dtype=np.intp)
    # fit 구간이 항상 1개 이상 그룹을 갖도록 보정 구간 크기를 제한합니다.
    calib_size = max(1, min(n - 1, int(np.ceil(n * calib_frac))))
    fit_size = n - calib_size
    calib_groups = set(unique[fit_size:].tolist())
    fit_groups = set(unique[: max(0, fit_size - embargo)].tolist())
    if not fit_groups:
        return np.array([], dtype=np.intp), train_idx
    fit_mask = np.isin(train_groups, list(fit_groups))
    calib_mask = np.isin(train_groups, list(calib_groups))
    return train_idx[fit_mask], train_idx[calib_mask]


def _conformal_shifts(
    y_calib: np.ndarray, raw_calib: dict[str, np.ndarray]
) -> dict[str, float]:
    """보정 구간 잔차(y - pred)의 alpha 분위수로 scalar 잔차 보정값을 산출합니다."""
    return {
        col: float(np.quantile(y_calib - raw_calib[col], alpha))
        for col, alpha in zip(_QUANTILE_COLS, _QUANTILE_ALPHAS, strict=True)
    }


def _fit_predict_calibrated(
    fit: pd.DataFrame,
    calib: pd.DataFrame,
    pred: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> np.ndarray:
    """fit 구간에서 base 분류기를 학습하고, 보정 구간에서 Platt(sigmoid) 보정을
    적합하여 ``pred`` 행의 보정 확률을 반환합니다.

    fit 구간에 단일 클래스만 존재하면 사전확률(prior) 상수를, 보정 구간이
    소표본/단일 클래스면 raw score 를 반환합니다.
    """
    y_fit = fit[label_col].to_numpy().astype(bool)
    if np.unique(y_fit).size < 2 or int(np.min(np.bincount(y_fit))) < 3:
        return np.full(len(pred), float(np.mean(y_fit)))
    base = LGBMClassifier(objective="binary", random_state=42, verbosity=-1)
    base.fit(fit[feature_cols], y_fit)
    positive_idx = int(np.argmax(np.unique(y_fit)))

    proba_pred = np.asarray(base.predict_proba(pred[feature_cols]), dtype=np.float64)
    score_pred = proba_pred[:, positive_idx]
    if len(calib) < _MIN_CALIB_ROWS:
        return score_pred
    y_calib = calib[label_col].to_numpy().astype(bool)
    if np.unique(y_calib).size < 2:
        return score_pred
    proba_calib = np.asarray(base.predict_proba(calib[feature_cols]), dtype=np.float64)
    score_calib = proba_calib[:, positive_idx]
    platts = LogisticRegression(max_iter=1000)
    platts.fit(score_calib.reshape(-1, 1), y_calib)
    return np.asarray(platts.predict_proba(score_pred.reshape(-1, 1))[:, 1], dtype=np.float64)


def _fold_diagnostics(
    calib: pd.DataFrame,
    group_col: str,
    raw_calib: dict[str, np.ndarray],
    shifts: dict[str, float],
    p_good_calib: np.ndarray,
    p_bad_calib: np.ndarray,
    fold: int,
) -> dict[str, Any]:
    """보정 구간에서 q10-q90 커버리지/구간 폭/Brier/log-loss 진단을 계산합니다.

    Brier/log-loss 는 해당 날짜의 date-prior(라벨 평균) baseline 과 함께 보고됩니다.
    """
    y = calib[_GOOD_COL].to_numpy().astype(bool)
    y_bad = calib[_BAD_COL].to_numpy().astype(bool)
    q10 = np.asarray(raw_calib[_QUANTILE_COLS[0]]) + shifts[_QUANTILE_COLS[0]]
    q50 = np.asarray(raw_calib[_QUANTILE_COLS[1]]) + shifts[_QUANTILE_COLS[1]]
    q90 = np.asarray(raw_calib[_QUANTILE_COLS[2]]) + shifts[_QUANTILE_COLS[2]]
    q10 = np.minimum(np.minimum(q10, q50), q90)
    q90 = np.maximum(q90, q50)
    covered = float(((y >= q10) & (y <= q90)).mean())
    width = float(np.mean(q90 - q10))

    p = np.clip(p_good_calib, 1e-7, 1.0 - 1e-7)
    brier = float(np.mean((p_good_calib - y) ** 2))
    log_loss = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    date_prior = calib.groupby(group_col)[_GOOD_COL].transform("mean").to_numpy(dtype=np.float64)
    brier_prior = float(np.mean((date_prior - y) ** 2))
    return {
        "fold": fold,
        "calibration_rows": len(calib),
        "q10_q90_coverage": covered,
        "interval_width": width,
        "brier": brier,
        "brier_date_prior": brier_prior,
        "log_loss": log_loss,
        "p_bad_brier": float(np.mean((np.clip(p_bad_calib, 0.0, 1.0) - y_bad) ** 2)),
    }


def fit_predict_quantile_and_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    n_splits: int = 5,
    purge_gap: int = 1,
) -> pd.DataFrame:
    """Quantile Regression + Calibrated Classifier 를 순차 보정 Walk-Forward CV 로 학습.

    각 outer fold 의 Train 을 이른 fit 구간과 이후 보정 구간으로 분할하고,
    분위수 conformal 잔차 보정과 확률 Platt 보정을 보정 구간에서만 적합합니다.
    Validation 은 각 fold 에서 정확히 1회 예측됩니다.

    Returns
    -------
    pd.DataFrame
        OOF 예측 행에 원본 키/피처 컬럼과 ``pred_q10``, ``pred_q50``,
        ``pred_q90``, ``p_good``, ``p_bad``, ``fold`` 컬럼을 포함합니다.
        ``attrs['calibration_diagnostics']`` 에 fold 단위 커버리지/폭/Brier/
        log-loss 진단 목록이 저장됩니다.
    """
    missing_cols = [col for col in [*feature_cols, target_col, group_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns in df: {missing_cols}")
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")

    work = df.sort_values(group_col).copy()
    work[_GOOD_COL] = work[target_col] >= _GOOD_THRESHOLD
    work[_BAD_COL] = work[target_col] <= _BAD_THRESHOLD
    group_values = work[group_col].to_numpy()

    splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)
    pred_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []

    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(work, y=work[target_col], groups=work[group_col])
    ):
        val = work.iloc[val_idx]
        fit_idx, calib_idx = _chrono_fit_calibration_split(
            group_values, train_idx, embargo=purge_gap
        )
        fit = work.iloc[fit_idx]
        calib = work.iloc[calib_idx]

        raw_val: dict[str, np.ndarray] = {}
        raw_calib: dict[str, np.ndarray] = {}
        shifts: dict[str, float] = {}
        for col, alpha in zip(_QUANTILE_COLS, _QUANTILE_ALPHAS, strict=True):
            raw_val[col] = _fit_predict_quantile(fit, val, feature_cols, target_col, alpha)

        use_calibration = len(calib_idx) >= _MIN_CALIB_ROWS
        if use_calibration:
            for col, alpha in zip(_QUANTILE_COLS, _QUANTILE_ALPHAS, strict=True):
                raw_calib[col] = _fit_predict_quantile(fit, calib, feature_cols, target_col, alpha)
            shifts = _conformal_shifts(
                calib[target_col].to_numpy(dtype=np.float64), raw_calib
            )
        else:
            shifts = dict.fromkeys(_QUANTILE_COLS, 0.0)

        p_good = _fit_predict_calibrated(fit, calib, val, feature_cols, _GOOD_COL)
        p_bad = _fit_predict_calibrated(fit, calib, val, feature_cols, _BAD_COL)

        if use_calibration:
            p_good_calib = _fit_predict_calibrated(fit, calib, calib, feature_cols, _GOOD_COL)
            p_bad_calib = _fit_predict_calibrated(fit, calib, calib, feature_cols, _BAD_COL)
            diagnostics.append(
                _fold_diagnostics(calib, group_col, raw_calib, shifts, p_good_calib, p_bad_calib, fold)
            )

        fold_frame = pd.DataFrame(
            {col: raw_val[col] + shifts[col] for col in _QUANTILE_COLS},
            index=val.index,
        )
        fold_frame["p_good"] = p_good
        fold_frame["p_bad"] = p_bad
        fold_frame["fold"] = fold
        pred_parts.append(fold_frame)
        logger.info(
            "fold=%d fit=%d calib=%d val=%d",
            fold,
            len(fit_idx),
            len(calib_idx),
            len(val_idx),
        )

    pred_df = pd.concat(pred_parts).sort_index()
    out = work.loc[pred_df.index].copy()
    for col in (*_PREDICTION_COLS, "fold"):
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

    out = out.drop(columns=[_GOOD_COL, _BAD_COL])
    out.attrs["calibration_diagnostics"] = diagnostics
    out.attrs["label_thresholds"] = {"target_good": _GOOD_THRESHOLD, "target_bad": _BAD_THRESHOLD}
    return out
