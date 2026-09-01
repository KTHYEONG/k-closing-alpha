"""OOF generation with chronological calibration and weighting."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression

from src.ml.purged_cv import PurgedGroupTimeSeriesSplit, chrono_fit_calibration_split

_GOOD_THRESHOLD = 0.01
_BAD_THRESHOLD = -0.02
_MIN_CALIB_ROWS = 5


def _finite_nan(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """피처의 비유한 값을 NaN 으로 치환."""
    out = frame.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        arr = out[col].to_numpy(dtype=np.float64)
        if np.isinf(arr).any():
            out[col] = np.where(np.isfinite(arr), arr, np.nan)
    return out


def fit_chrono_calibrator(
    features: pd.DataFrame,
    labels: np.ndarray,
    group_values: np.ndarray,
    feature_cols: list[str],
    calib_frac: float = 0.3,
    embargo: int = 1,
    model_params: dict[str, Any] | None = None,
) -> Any:
    """시계열 분할 기반 플랫 보정 분류기 또는 prior float 를 반환합니다."""
    n = len(features)
    if n == 0:
        return 0.0
    train_idx = np.arange(n)
    fit_idx, calib_idx = chrono_fit_calibration_split(group_values, train_idx, calib_frac=calib_frac, embargo=embargo)
    # Fallback rules mirror legacy _fit_predict_calibrated
    y = np.asarray(labels).astype(bool)
    # consider fit slice for class check
    if fit_idx.size == 0:
        return float(np.mean(y)) if y.size else 0.0
    y_fit = y[fit_idx]
    if np.unique(y_fit).size < 2 or int(np.min(np.bincount(y_fit))) < 3:
        return float(np.mean(y_fit))
    if calib_idx.size < _MIN_CALIB_ROWS:
        # still train base and return raw scores? But spec says return bare float prior when fit slice degenerate OR calib slice small. Actually when calib <5 return raw? For fit_chrono_calibrator we return Platt? Let's follow spec: returns float prior when fit slice has <2 classes or calib <5 or calib <2 classes.
        # But for fit_chrono_calibrator spec: fail-closed float when calib <5 or <2 classes. For small calib we return prior? Let's interpret: if calib <5 -> float prior? Wait spec says: Returns bare float prior when fit slice has <2 classes or minority count <3, or calib slice has <5 rows or <2 classes.
        # So if calib small we return float prior, not raw score.
        # However legacy _fit_predict_calibrated returns raw score when calib small. But spec explicitly says float prior for calib <5.
        # We'll follow spec.
        return float(np.mean(y_fit))
    y_calib = y[calib_idx]
    if np.unique(y_calib).size < 2:
        return float(np.mean(y_fit))
    # Train base on fit slice
    params = model_params or {}
    base = LGBMClassifier(objective="binary", random_state=42, verbosity=-1, **params)
    fit_features = _finite_nan(features.iloc[fit_idx], feature_cols)[feature_cols]
    base.fit(fit_features, y_fit)
    # Platt on calib slice
    calib_features = _finite_nan(features.iloc[calib_idx], feature_cols)[feature_cols]
    proba_calib = np.asarray(base.predict_proba(calib_features), dtype=np.float64)
    # positive index is where class == True
    classes = base.classes_
    positive_idx = int(np.where(classes)[0][0]) if np.any(classes) else 1
    score_calib = proba_calib[:, positive_idx]
    platts = LogisticRegression(max_iter=1000)
    platts.fit(score_calib.reshape(-1, 1), y_calib)
    # Return wrapper
    from src.ml.bundle import ChronoCalibratedClassifier

    return ChronoCalibratedClassifier(base=base, platt=platts, positive_index=positive_idx)


def sample_weight_for_fold(
    train_groups: pd.Series,
    weighting_mode: str,
    recency_half_life_groups: int | None,
) -> np.ndarray | None:
    """폴드 train 그룹 기반 샘플 가중치를 반환합니다."""
    if weighting_mode == "current" and recency_half_life_groups is None:
        return None
    if weighting_mode not in ("current", "date_balanced"):
        raise ValueError(f"weighting_mode must be one of current/date_balanced, got {weighting_mode!r}")
    if recency_half_life_groups is not None and recency_half_life_groups not in (252, 504):
        raise ValueError(
            f"recency_half_life_groups must be one of None, 252, 504, got {recency_half_life_groups!r}"
        )
    if len(train_groups) == 0:
        raise ValueError("sample weight requires non-empty groups")
    parsed = pd.to_datetime(pd.Series(train_groups), errors="coerce")
    if parsed.isna().any():
        raise ValueError("sample weight requires parseable chronological trade-date groups")
    if weighting_mode == "current":
        # recency only
        assert recency_half_life_groups is not None
        unique_groups = pd.unique(train_groups)
        parsed_unique = pd.to_datetime(pd.Series(unique_groups), errors="coerce")
        if parsed_unique.isna().any():
            raise ValueError("sample weight requires parseable groups")
        order = np.argsort(parsed_unique.to_numpy(), kind="stable")
        sorted_unique = unique_groups[order]
        ages = np.arange(len(sorted_unique), dtype=np.float64)[::-1]
        decay = np.exp(-np.log(2.0) * ages / float(recency_half_life_groups))
        mean_decay = float(decay.mean())
        if not np.isfinite(mean_decay) or mean_decay <= 0.0:
            raise ValueError("recency sample weights are not finite")
        weights = decay / mean_decay
        weight_by_group = dict(zip(sorted_unique, weights.tolist(), strict=True))
        return np.asarray([weight_by_group[g] for g in train_groups], dtype=np.float64)
    # date_balanced
    parsed_all = pd.to_datetime(pd.Series(train_groups), errors="coerce")
    group_id = pd.factorize(parsed_all, sort=True)[0]
    counts = pd.Series(group_id).value_counts(sort=False).to_numpy(dtype=np.float64)
    row_counts = counts[group_id]
    weights_arr: np.ndarray = np.asarray(1.0 / row_counts, dtype=np.float64)
    if recency_half_life_groups is not None:
        unique_group_ids = np.unique(group_id)
        age_map = {
            int(gid): float(len(unique_group_ids) - 1 - int(pos))
            for pos, gid in enumerate(unique_group_ids)
        }
        ages = np.asarray([age_map[gid] for gid in group_id], dtype=np.float64)
        decay = np.exp(-np.log(2.0) * ages / float(recency_half_life_groups))
        weights_arr = weights_arr * decay
    mean_weight = float(weights_arr.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        raise ValueError("sample weights are not finite")
    weights_arr = weights_arr / mean_weight
    if not np.isfinite(weights_arr).all() or float(weights_arr.sum()) <= 0.0:
        raise ValueError("sample weights are not finite")
    return weights_arr


def purged_oof_predict(
    dev_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    *,
    n_splits: int,
    purge_gap: int,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
    weighting_mode: str = "current",
    recency_half_life_groups: int | None = None,
    predict_proba: bool = True,
) -> pd.DataFrame:
    """Purged walk-forward OOF predictions with optional calibrated probabilities."""
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")
    missing = [c for c in [*feature_cols, target_col, group_col] if c not in dev_df.columns]
    if missing:
        raise ValueError(f"missing columns in dev_df: {missing}")
    work = dev_df.sort_values(group_col).copy()
    splitter = PurgedGroupTimeSeriesSplit(n_splits=n_splits, purge_gap=purge_gap)
    parts: list[pd.DataFrame] = []
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(work, y=work[target_col], groups=work[group_col])
    ):
        train = work.iloc[train_idx]
        val = work.iloc[val_idx]
        # sample weight
        train_groups = train[group_col]
        sw = sample_weight_for_fold(train_groups, weighting_mode, recency_half_life_groups)
        # Fit regressor
        params = dict(model_params or {})
        # ensure finite
        train_f = _finite_nan(train, feature_cols)
        val_f = _finite_nan(val, feature_cols)
        reg = LGBMRegressor(objective="huber", alpha=huber_delta, random_state=42, verbosity=-1, **params)
        reg.fit(train_f[feature_cols], train_f[target_col].to_numpy(dtype=np.float64), sample_weight=sw)
        pred = reg.predict(val_f[feature_cols])
        fold_df = pd.DataFrame({"pred": pred, "fold": fold}, index=val.index)
        if predict_proba:
            # p_good / p_bad calibrated
            # Prepare labels for good/bad
            y_good = (train[target_col] >= _GOOD_THRESHOLD).to_numpy().astype(bool)
            y_bad = (train[target_col] <= _BAD_THRESHOLD).to_numpy().astype(bool)
            # Use chrono split for calibration within train
            # Fit calibrators on this fold's train groups only
            # Build features DataFrames for calibrator
            train_features = train[feature_cols]
            # For p_good
            calib_good = fit_chrono_calibrator(
                train_features, y_good, train[group_col].to_numpy(), feature_cols, model_params=None
            )
            calib_bad = fit_chrono_calibrator(
                train_features, y_bad, train[group_col].to_numpy(), feature_cols, model_params=None
            )
            # Predict for val
            val_features = val[feature_cols]
            if isinstance(calib_good, float):
                p_good = np.full(len(val), float(calib_good), dtype=np.float64)
            else:
                p_good = calib_good.predict_proba(val_features)[:, list(calib_good.classes_).index(True)]
            if isinstance(calib_bad, float):
                p_bad = np.full(len(val), float(calib_bad), dtype=np.float64)
            else:
                p_bad = calib_bad.predict_proba(val_features)[:, list(calib_bad.classes_).index(True)]
            fold_df["p_good"] = p_good
            fold_df["p_bad"] = p_bad
        parts.append(fold_df)
    if not parts:
        return pd.DataFrame()
    oof = pd.concat(parts).sort_index()
    # passthrough columns
    out = work.loc[oof.index].copy()
    for col in oof.columns:
        out[col] = oof[col].to_numpy()
    # Ensure passthrough identity cols preserved
    return out
