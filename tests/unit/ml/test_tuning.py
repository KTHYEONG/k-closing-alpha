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


def test_tune_return_model_params_rank_ic_value_unchanged_after_vectorization() -> None:
    import pytest
    import numpy as np
    import pandas as pd

    from src.ml.metrics import mean_group_rank_ic
    from src.ml.oof import purged_oof_predict
    from src.ml.tuning import ChampionTuningConfig, tune_return_model_params

    rng = np.random.default_rng(11)
    rows = []
    for d in pd.bdate_range("2025-01-01", periods=60):
        for _ in range(8):
            sig = rng.normal()
            rows.append({"trade_date": d.strftime("%Y-%m-%d"), "signal": sig, "noise": rng.normal(), "target_return": 0.02 * sig + 0.003 * rng.normal()})
    df = pd.DataFrame(rows)

    cfg = ChampionTuningConfig(hpo_trials=4, inner_n_splits=2, hpo_objective="rank_ic")
    res_a = tune_return_model_params(df, ["signal", "noise"], "target_return", "trade_date", cfg)
    res_b = tune_return_model_params(df, ["signal", "noise"], "target_return", "trade_date", cfg)

    assert res_a.best_params == res_b.best_params
    assert res_a.best_value == pytest.approx(res_b.best_value, abs=1e-12)

    oof = purged_oof_predict(df, ["signal", "noise"], "target_return", "trade_date", n_splits=cfg.inner_n_splits, purge_gap=cfg.purge_gap, model_params=res_a.best_params, huber_delta=cfg.huber_delta, predict_proba=False)
    direct = mean_group_rank_ic(oof, ["trade_date"], "pred", "target_return", min_group_size=2)
    assert res_a.best_value == pytest.approx(direct, abs=1e-9)
