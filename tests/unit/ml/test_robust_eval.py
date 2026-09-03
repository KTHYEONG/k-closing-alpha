import numpy as np
import pandas as pd

from src.ml.robust_eval import CombinatorialPurgedCV


def test_cpcv_split_no_train_test_leakage() -> None:
    # Given a 10-day panel with 3 rows per day
    days = pd.to_datetime([f"2024-01-{d:02d}" for d in range(1, 11)])
    groups = pd.Series(np.repeat(days.values, 3))
    cv = CombinatorialPurgedCV(n_groups=10, k_test=2, purge_gap=1, embargo_gap=1)
    unique_sorted = np.array(sorted(groups.unique()))

    # When we iterate every fold
    folds = list(cv.split(groups))
    assert len(folds) == 45  # C(10, 2)

    for train_idx, test_idx, _fold_id in folds:
        train_groups = set(groups.to_numpy()[train_idx].tolist())
        test_groups = set(groups.to_numpy()[test_idx].tolist())
        # Then train and test never share a date-group
        assert train_groups.isdisjoint(test_groups)
        # And the purge/embargo neighbours of each contiguous test block are absent from train
        test_positions = sorted(np.where(np.isin(unique_sorted, list(test_groups)))[0].tolist())
        for pos in test_positions:
            for neighbour in (pos - 1, pos + 1):
                if 0 <= neighbour < len(unique_sorted):
                    g = unique_sorted[neighbour]
                    if g not in test_groups:
                        assert g not in train_groups


import numpy as np
import pandas as pd

from src.ml.robust_eval import CombinatorialPurgedCV


def test_cpcv_path_count_and_coverage() -> None:
    days = pd.to_datetime([f"2024-02-{d:02d}" for d in range(1, 17)])
    groups = pd.Series(np.repeat(days.values, 2))
    cv = CombinatorialPurgedCV(n_groups=8, k_test=2, purge_gap=1, embargo_gap=1)

    assert cv.n_paths() == 7
    folds = list(cv.split(groups))
    assert len(folds) == 28

    # Bin the 16 unique days into 8 contiguous groups; each bin must be tested 7 times
    unique_sorted = np.array(sorted(groups.unique()))
    bin_of_day = {d: i * 8 // len(unique_sorted) for i, d in enumerate(unique_sorted)}
    counts = dict.fromkeys(range(8), 0)
    for _train_idx, test_idx, _fold_id in folds:
        tested_bins = {bin_of_day[d] for d in groups.to_numpy()[test_idx]}
        for b in tested_bins:
            counts[b] += 1
    assert set(counts.values()) == {7}


import numpy as np

from src.ml.robust_eval import moving_block_bootstrap_delta


def test_moving_block_bootstrap_delta_detects_shift() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(0.0, 0.03, size=240)

    shifted = moving_block_bootstrap_delta(base + 0.002, base, block_size=10, n_boot=4000, seed=0)
    assert abs(shifted.delta - 0.002) < 5e-4
    assert shifted.p_value < 0.05
    assert shifted.ci_low <= shifted.delta <= shifted.ci_high
    assert shifted.n_obs == 240

    null = moving_block_bootstrap_delta(base, base.copy(), block_size=10, n_boot=4000, seed=0)
    assert null.p_value > 0.5


import numpy as np
import pytest

from src.ml.robust_eval import moving_block_bootstrap_delta


def test_moving_block_bootstrap_delta_fail_closed() -> None:
    a = np.zeros(50)
    with pytest.raises(ValueError, match='share'):
        moving_block_bootstrap_delta(a, np.zeros(49))
    bad = a.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        moving_block_bootstrap_delta(bad, np.zeros(50))
    with pytest.raises(ValueError, match='30'):
        moving_block_bootstrap_delta(np.zeros(20), np.zeros(20))


import numpy as np
import pytest

from src.ml.robust_eval import deflated_sharpe_ratio


def test_deflated_sharpe_ratio_penalizes_trials() -> None:
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0008, 0.02, size=500)

    dsr_1 = deflated_sharpe_ratio(returns, n_independent_trials=1)
    dsr_200 = deflated_sharpe_ratio(returns, n_independent_trials=200)

    assert 0.0 <= dsr_200 <= dsr_1 <= 1.0
    with pytest.raises(ValueError, match='trial'):
        deflated_sharpe_ratio(returns, n_independent_trials=0)
    with pytest.raises(ValueError, match='20'):
        deflated_sharpe_ratio(np.zeros(10), n_independent_trials=5)


import numpy as np
import pandas as pd

from src.ml.robust_eval import path_top1_returns


def test_path_top1_returns_selects_argmax_per_day() -> None:
    oof = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                ["2024-03-01", "2024-03-01", "2024-03-02", "2024-03-02", "2024-03-01", "2024-03-01"]
            ),
            "target_return": [0.05, -0.02, 0.01, 0.09, 0.04, -0.10],
            "pred": [0.9, 0.1, 0.2, 0.8, 0.7, 0.6],
            "cpcv_fold": [0, 0, 0, 0, 1, 1],
        }
    )

    paths = path_top1_returns(oof, "trade_date", "target_return", score_col="pred", fold_col="cpcv_fold")

    assert set(paths) == {0, 1}
    np.testing.assert_allclose(sorted(paths[0]), sorted([0.05, 0.09]))
    np.testing.assert_allclose(paths[1], [0.04])



import numpy as np
import pandas as pd

from src.ml.robust_eval import CombinatorialPurgedCV, cpcv_oof_predict


def test_cpcv_oof_predict_survives_incomparable_attrs() -> None:
    # Given a dev panel whose .attrs holds an object pandas cannot compare (a DataFrame),
    # mirroring build_ml_dataset output (feature_manifest / scenario_action_rejects).
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2023-01-02", periods=48)
    rows = []
    for d in days:
        for _ in range(5):
            f1, f2 = rng.normal(), rng.normal()
            rows.append({"trade_date": d, "f1": f1, "f2": f2, "target_return": 0.01 * f1 - 0.004 * f2 + rng.normal(scale=0.02)})
    dev = pd.DataFrame(rows)
    dev.attrs["feature_manifest"] = pd.DataFrame({"name": ["f1", "f2"]})

    cv = CombinatorialPurgedCV(n_groups=8, k_test=2, purge_gap=1, embargo_gap=1)
    oof = cpcv_oof_predict(dev, ["f1", "f2"], "target_return", "trade_date", cv=cv, model_params={"n_estimators": 30})

    assert not oof.empty
    assert {"pred", "cpcv_fold"}.issubset(oof.columns)
    assert np.isfinite(oof["pred"].to_numpy()).all()
    assert oof["cpcv_fold"].nunique() == 28
