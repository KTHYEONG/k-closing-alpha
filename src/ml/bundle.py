"""Bundle assembly for serving compatibility."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV

from src.ml.feature_manifest import build_feature_manifest
from src.ml.oof import _finite_nan, fit_chrono_calibrator
from src.serving.realtime.inference import (
    _DEFAULT_REALIZED_VOL,
    _GOOD_PCT,
    _GRADE_MULTIPLIERS,
    _QUANTILE_ALPHAS,
    _QUANTILE_COLS,
    _STRONG_PCT,
    _WEAK_PCT,
    ROUND_TRIP_COST_RATIO,
)

_GOOD_THRESHOLD = 0.01
_BAD_THRESHOLD = -0.02
_CATEGORICAL_FEATURE_COLS: tuple[str, ...] = (
    "market_type",
    "theme_sector",
    "chart_analysis",
)

CHAMPION_DEFAULT_MODEL_PARAMS: dict[str, Any] = {
    "num_leaves": 15,
    "min_child_samples": 40,
    "n_estimators": 350,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


class SeedEnsembleModel:
    """Seed ensemble averaging predictions."""

    def __init__(self, models: list[Any], seeds: tuple[int, ...]) -> None:
        if not models or len(models) != len(seeds):
            raise ValueError("SeedEnsembleModel requires non-empty models with len(models)==len(seeds)")
        self.models = models
        self._seeds = tuple(seeds)

    @property
    def seeds(self) -> tuple[int, ...]:
        return self._seeds

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:  # noqa: N803
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.asarray(preds.mean(axis=1), dtype=np.float64)


class ChronoCalibratedClassifier:
    """Chronological Platt calibrated classifier."""

    def __init__(self, base: Any, platt: Any, positive_index: int) -> None:
        self.base = base
        self.platt = platt
        self.positive_index = positive_index
        # classes_ as [False, True]
        self._classes = np.array([False, True])

    @property
    def classes_(self) -> np.ndarray:
        return self._classes

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:  # noqa: N803
        base_proba = self.base.predict_proba(X)
        scores = base_proba[:, self.positive_index]
        platt_proba = self.platt.predict_proba(scores.reshape(-1, 1))
        # platt classes are [False, True] mapped to 0/1 via training labels
        # Return (n,2) with columns [False, True]
        # platt predicts class True probability at index 1
        n = len(scores)
        out = np.zeros((n, 2), dtype=np.float64)
        # platt's positive column is where class == True
        # logistic regression trained on bool labels; its classes_ is [False, True] if both present
        # We'll map proba[:,1] to True column
        true_proba = platt_proba[:, 1] if platt_proba.shape[1] > 1 else platt_proba[:, 0]
        out[:, 0] = 1.0 - true_proba
        out[:, 1] = true_proba
        return out


def _fit_calibrator_cv(features: pd.DataFrame, labels: np.ndarray) -> Any:
    """Legacy CV calibrator for cv3 mode."""
    if np.unique(labels).size < 2:
        return float(np.mean(labels))
    min_class = int(np.min(np.bincount(labels.astype(bool))))
    if min_class < 3:
        return float(np.mean(labels))
    base = LGBMClassifier(objective="binary", random_state=42, verbosity=-1)
    calibrator = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    calibrator.fit(features, labels)
    return calibrator


def fit_seed_ensemble(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seeds: tuple[int, ...],
    model_params: dict[str, Any],
    huber_delta: float,
    sample_weight: np.ndarray | None = None,
) -> SeedEnsembleModel:
    """One LGBMRegressor per seed."""
    models: list[Any] = []
    for seed in seeds:
        params = dict(model_params or {})
        params.setdefault("alpha", huber_delta)
        train_f = _finite_nan(train_df, feature_cols)
        model = LGBMRegressor(objective="huber", random_state=seed, verbosity=-1, **params)  # type: ignore[arg-type]
        model.fit(train_f[feature_cols], train_f[target_col].to_numpy(dtype=np.float64), sample_weight=sample_weight)
        models.append(model)
    return SeedEnsembleModel(models, seeds)


def build_inline_bundle(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    *,
    return_model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
    seeds: tuple[int, ...] | None = None,
    calibrator_mode: str = "cv3",
    calib_group_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train inline bundle with optional seed ensemble and chrono calibrators."""
    if calibrator_mode not in ("cv3", "chrono"):
        raise ValueError(f"calibrator_mode must be one of cv3/chrono, got {calibrator_mode!r}")
    feature_cols = [c for c in feature_cols if c not in _CATEGORICAL_FEATURE_COLS]
    if not feature_cols:
        raise ValueError("feature_cols is empty after excluding categorical columns")
    train = df.sort_values(group_col)
    y = train[target_col].to_numpy(dtype=np.float64)
    base_return_params = {**CHAMPION_DEFAULT_MODEL_PARAMS, **(return_model_params or {})}
    # ranker
    relevance = train[target_col].groupby(train[group_col], sort=False).rank(pct=True).to_numpy()
    relevance = (relevance * 4.0).round().astype(int)
    group_counts = train[group_col].value_counts(sort=False).to_numpy(dtype=np.int64)
    ranker = LGBMRanker(objective="lambdarank", random_state=42, verbosity=-1)
    ranker.fit(train[feature_cols], relevance, group=group_counts)

    # return model
    if seeds is not None:
        return_model = fit_seed_ensemble(
            train, feature_cols, target_col, seeds, base_return_params, huber_delta
        )
    else:
        train_f = _finite_nan(train, feature_cols)
        merged_params: dict[str, object] = dict(base_return_params)
        merged_params.setdefault("alpha", huber_delta)
        return_model = LGBMRegressor(objective="huber", random_state=42, verbosity=-1, **merged_params)  # type: ignore[arg-type]
        return_model.fit(train_f[feature_cols], y)

    quantile_models: dict[str, Any] = {}
    for col, alpha in zip(_QUANTILE_COLS, _QUANTILE_ALPHAS, strict=True):
        model = LGBMRegressor(objective="quantile", alpha=alpha, random_state=42, verbosity=-1)
        model.fit(train[feature_cols], y)
        quantile_models[col] = model

    # calibrators
    calibrators: dict[str, Any] = {}
    if calibrator_mode == "cv3":
        for name, thresh in (("p_good", _GOOD_THRESHOLD), ("p_bad", _BAD_THRESHOLD)):
            labels = (train[target_col] >= thresh).to_numpy().astype(bool) if name == "p_good" else (train[target_col] <= thresh).to_numpy().astype(bool)
            calibrators[name] = _fit_calibrator_cv(train[feature_cols], labels)
    else:
        # chrono
        if calib_group_values is None:
            calib_group_values = train[group_col].to_numpy()
        for name, thresh in (("p_good", _GOOD_THRESHOLD), ("p_bad", _BAD_THRESHOLD)):
            labels = (train[target_col] >= thresh).to_numpy().astype(bool) if name == "p_good" else (train[target_col] <= thresh).to_numpy().astype(bool)
            # Use fit_chrono_calibrator with those group values
            # need features DataFrame
            calibrators[name] = fit_chrono_calibrator(
                train[feature_cols], labels, calib_group_values, feature_cols
            )

    manifest = build_feature_manifest(list(feature_cols))
    training_cutoff = str(train[group_col].max())
    policy_params: dict[str, Any] = {
        "grade_multipliers": dict(_GRADE_MULTIPLIERS),
        "grade_percentiles": {"strong": _STRONG_PCT, "good": _GOOD_PCT, "weak": _WEAK_PCT},
        "utility_weights": {"lambda_risk": 0.5, "gamma_uncertainty": 0.1, "w_good": 0.0, "w_bad": 0.0},
        "round_trip_cost": ROUND_TRIP_COST_RATIO,
        "realized_vol_default": _DEFAULT_REALIZED_VOL,
    }
    bundle: dict[str, Any] = {
        "feature_cols": list(feature_cols),
        "target_col": target_col,
        "group_col": group_col,
        "return_unit": "decimal_net",
        "round_trip_cost": ROUND_TRIP_COST_RATIO,
        "label_thresholds": {"target_good": _GOOD_THRESHOLD, "target_bad": _BAD_THRESHOLD},
        "feature_manifest": manifest,
        "training_cutoff": training_cutoff,
        "calibration_diagnostics": [],
        "policy_params": policy_params,
        "rank_model": ranker,
        "return_model": return_model,
        "quantile_models": quantile_models,
        "calibrators": calibrators,
    }
    return bundle


def save_bundle(bundle: dict[str, Any], export_dir: str) -> str:
    """joblib dump to <export_dir>/sizing_pipeline_bundle.joblib."""
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
    dump(bundle, path)
    return os.path.abspath(path)
