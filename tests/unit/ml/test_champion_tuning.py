import numpy as np
import pandas as pd

from src.ml.dataset import retarget_with_clip


def test_retarget_with_clip_applies_bounds_and_cost() -> None:
    from src.serving.realtime.inference import ROUND_TRIP_COST_RATIO

    df = pd.DataFrame({
        "net_return": [40.0, -50.0, 3.0],
        "target_return": [0.10, -0.10, 0.028],
        "target_good": [1, 0, 1],
        "target_bad": [0, 1, 0],
    })
    out = retarget_with_clip(df, -0.15, 0.30)
    assert np.isclose(out["target_return"].iloc[0], 0.30)
    assert np.isclose(out["target_return"].iloc[1], -0.15)
    assert np.isclose(out["target_return"].iloc[2], 3.0 / 100 - ROUND_TRIP_COST_RATIO)
    assert out["target_good"].tolist() == [1, 0, 1]
    assert out["target_bad"].tolist() == [0, 1, 0]

import numpy as np
import pandas as pd

from src.ml.bundle import SeedEnsembleModel, fit_seed_ensemble


def test_seed_ensemble_predict_is_member_mean() -> None:
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"f1": rng.normal(size=200), "f2": rng.normal(size=200)})
    train["target_return"] = 0.5 * train["f1"] - 0.3 * train["f2"] + rng.normal(scale=0.01, size=200)
    seeds = (1, 2, 3)
    model = fit_seed_ensemble(train, ["f1", "f2"], "target_return", seeds, {"n_estimators": 40}, 0.9)
    assert isinstance(model, SeedEnsembleModel)
    assert model.seeds == seeds and len(model.models) == 3
    member = np.column_stack([m.predict(train[["f1", "f2"]]) for m in model.models])
    np.testing.assert_allclose(model.predict(train[["f1", "f2"]]), member.mean(axis=1))

import numpy as np
import pandas as pd

from src.ml.bundle import ChronoCalibratedClassifier
from src.ml.oof import fit_chrono_calibrator


def test_fit_chrono_calibrator_is_serving_compatible() -> None:
    rng = np.random.default_rng(3)
    n = 400
    feats = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    score = feats["f1"] - feats["f2"] + rng.normal(scale=0.5, size=n)
    labels = (score > score.median()).to_numpy()
    groups = np.repeat(pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(40), unit="D"), n // 40)
    calib = fit_chrono_calibrator(feats, labels, groups, ["f1", "f2"])
    assert isinstance(calib, ChronoCalibratedClassifier)
    assert list(calib.classes_) == [False, True]
    proba = calib.predict_proba(feats[["f1", "f2"]])
    assert proba.shape == (n, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    prior = fit_chrono_calibrator(feats, np.zeros(n, dtype=bool), groups, ["f1", "f2"])
    assert isinstance(prior, float) and prior == 0.0

import numpy as np
import pandas as pd

from src.ml.oof import purged_oof_predict


def _panel(n_dates: int = 70, per_day: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=n_dates), per_day)
    m = len(dates)
    f1, f2 = rng.normal(size=m), rng.normal(size=m)
    return pd.DataFrame({
        "trade_date": dates,
        "stock_code": [f"{i % 50:06d}" for i in range(m)],
        "chart_analysis": "volume_surge",
        "selection_rank": rng.integers(1, per_day + 1, size=m),
        "f1": f1, "f2": f2,
        "target_return": 0.02 * f1 - 0.01 * f2 + rng.normal(scale=0.03, size=m),
    })


def test_purged_oof_predict_has_no_future_leak() -> None:
    df = _panel()
    oof = purged_oof_predict(
        df, ["f1", "f2"], "target_return", "trade_date",
        n_splits=4, purge_gap=1, model_params={"n_estimators": 40},
        weighting_mode="date_balanced", predict_proba=True,
    )
    assert set(oof.index).issubset(set(df.index))
    assert np.isfinite(oof["pred"].to_numpy()).all()
    p = oof["p_good"].to_numpy()
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert oof["fold"].nunique() >= 2


