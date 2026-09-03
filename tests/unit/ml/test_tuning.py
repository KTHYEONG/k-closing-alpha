"""ChampionTuningConfig 계약: rank_ic 기본 목적함수 및 feature-selection 설정."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.tuning import ChampionTuningConfig, tune_return_model_params


def test_default_hpo_objective_is_rank_ic() -> None:
    cfg = ChampionTuningConfig()
    assert cfg.hpo_objective == "rank_ic"
    assert cfg.feature_selection_top_n is None
    assert cfg.feature_selection_min_folds == 2


def test_feature_selection_top_n_lower_bound() -> None:
    assert ChampionTuningConfig(feature_selection_top_n=5).feature_selection_top_n == 5
    with pytest.raises(ValueError, match="feature_selection_top_n"):  # noqa: PT011
        ChampionTuningConfig(feature_selection_top_n=4)


def test_feature_selection_min_folds_lower_bound() -> None:
    with pytest.raises(ValueError, match="feature_selection_min_folds"):  # noqa: PT011
        ChampionTuningConfig(feature_selection_min_folds=0)


def test_invalid_hpo_objective_rejected() -> None:
    with pytest.raises(ValueError, match="hpo_objective"):  # noqa: PT011
        ChampionTuningConfig(hpo_objective="sharpe")


def test_tune_return_model_params_rank_ic_objective_runs() -> None:
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.bdate_range("2023-01-02", periods=90):
        for _ in range(8):
            sig = float(rng.normal())
            rows.append({"trade_date": d, "signal": sig, "noise": float(rng.normal()),
                         "target_return": 0.02 * sig + 0.003 * float(rng.normal())})
    df = pd.DataFrame(rows)
    cfg = ChampionTuningConfig(hpo_trials=3, inner_n_splits=2, hpo_objective="rank_ic")

    res = tune_return_model_params(df, ["signal", "noise"], "target_return", "trade_date", cfg)

    assert res.objective == "rank_ic"
    assert np.isfinite(res.best_value)
    assert set(res.best_params).issuperset({"num_leaves", "learning_rate", "n_estimators"})
