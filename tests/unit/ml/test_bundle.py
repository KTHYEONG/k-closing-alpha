"""bundle: regularized default return-model params reach the deployed model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.ml.bundle import (
    CHAMPION_DEFAULT_MODEL_PARAMS,
    SeedEnsembleModel,
    build_inline_bundle,
    fit_seed_ensemble,
)


def _panel(n_dates: int = 60, per_day: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = np.repeat(pd.bdate_range("2023-01-02", periods=n_dates), per_day)
    m = len(dates)
    f1, f2 = rng.normal(size=m), rng.normal(size=m)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "f1": f1,
            "f2": f2,
            "target_return": 0.01 * f1 - 0.004 * f2 + rng.normal(scale=0.02, size=m),
        }
    )


def test_champion_default_model_params_are_regularized() -> None:
    # Regularized relative to bare LightGBM defaults (num_leaves 31, n_estimators 100).
    assert CHAMPION_DEFAULT_MODEL_PARAMS["num_leaves"] == 15
    assert CHAMPION_DEFAULT_MODEL_PARAMS["n_estimators"] == 350
    assert CHAMPION_DEFAULT_MODEL_PARAMS["min_child_samples"] >= 20
    assert 0.0 < CHAMPION_DEFAULT_MODEL_PARAMS["subsample"] <= 1.0


def test_build_inline_bundle_applies_regularized_default_params() -> None:
    df = _panel()

    b_default = build_inline_bundle(df, ["f1", "f2"], "target_return", "trade_date")
    rm = b_default["return_model"].get_params()
    assert rm["num_leaves"] == CHAMPION_DEFAULT_MODEL_PARAMS["num_leaves"]
    assert rm["n_estimators"] == CHAMPION_DEFAULT_MODEL_PARAMS["n_estimators"]

    # An explicit param dict overrides per-key; unspecified keys keep the regularized default.
    b_override = build_inline_bundle(
        df, ["f1", "f2"], "target_return", "trade_date", return_model_params={"num_leaves": 63}
    )
    ro = b_override["return_model"].get_params()
    assert ro["num_leaves"] == 63
    assert ro["n_estimators"] == CHAMPION_DEFAULT_MODEL_PARAMS["n_estimators"]


def test_build_inline_bundle_seed_ensemble_uses_regularized_default() -> None:
    df = _panel()
    bundle = build_inline_bundle(
        df, ["f1", "f2"], "target_return", "trade_date", seeds=(11, 23), calibrator_mode="chrono"
    )
    model = bundle["return_model"]
    assert isinstance(model, SeedEnsembleModel)
    for member in model.models:
        assert member.get_params()["num_leaves"] == CHAMPION_DEFAULT_MODEL_PARAMS["num_leaves"]
        assert member.get_params()["n_estimators"] == CHAMPION_DEFAULT_MODEL_PARAMS["n_estimators"]


def test_fit_seed_ensemble_passes_params_through_verbatim() -> None:
    # fit_seed_ensemble itself does not inject CHAMPION_DEFAULT_MODEL_PARAMS;
    # build_inline_bundle is the layer that merges them before calling here.
    df = _panel()
    model = fit_seed_ensemble(df, ["f1", "f2"], "target_return", (1, 2), {"num_leaves": 8, "n_estimators": 40}, 0.9)
    assert isinstance(model, SeedEnsembleModel)
    for member in model.models:
        assert member.get_params()["num_leaves"] == 8
        assert member.get_params()["n_estimators"] == 40
