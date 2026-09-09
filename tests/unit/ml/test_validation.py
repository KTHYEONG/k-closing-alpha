"""Auto-generated scenario tests (see docs/specs/ml_validation_overhaul_contract.json)."""
from __future__ import annotations

def test_minimum_detectable_effect_matches_closed_form() -> None:
    import numpy as np
    import pytest
    from scipy.stats import norm

    from src.ml.validation import minimum_detectable_effect

    expected = (norm.ppf(0.975) + norm.ppf(0.80)) * 4.667 / np.sqrt(250.0)
    assert np.isclose(minimum_detectable_effect(4.667, 250), expected)

    # Measured project noise: top-1 sd 4.667%/day, daily rank-IC sd 0.2745
    assert minimum_detectable_effect(4.667, 250) > 0.25  # 25bp/day is unresolvable
    assert minimum_detectable_effect(0.2745, 250) < 0.10  # a 0.10 rank-IC is resolvable

    with pytest.raises(ValueError, match="n_obs"):
        minimum_detectable_effect(4.667, 0)
    with pytest.raises(ValueError, match="sd"):
        minimum_detectable_effect(0.0, 250)

def test_paired_t_p_value_matches_scipy() -> None:
    import numpy as np
    import pytest
    from scipy import stats

    from src.ml.validation import paired_t_p_value

    rng = np.random.default_rng(7)
    delta = rng.normal(loc=0.002, scale=0.045, size=400)

    expected = float(stats.ttest_1samp(delta, 0.0).pvalue)
    assert np.isclose(paired_t_p_value(delta), expected)

    with pytest.raises(ValueError, match="finite"):
        paired_t_p_value(np.array([0.01, np.nan, 0.02]))
    with pytest.raises(ValueError, match="n_obs"):
        paired_t_p_value(np.array([0.01]))


def test_cpcv_path_evidence_reports_paired_path_rates() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.validation import cpcv_path_evidence

    rng = np.random.default_rng(11)
    days = pd.to_datetime(pd.date_range("2024-01-01", periods=120, freq="D"))
    rows = 6
    n = len(days) * rows
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    eval_net = 0.03 * f1 + 0.01 * f2 + rng.normal(scale=0.02, size=n)
    dev = pd.DataFrame({
        "trade_date": np.repeat(days.to_numpy(), rows),
        "stock_code": [f"{i % 40:06d}" for i in range(n)],
        "f1": f1,
        "f2": f2,
        "target_return": eval_net,
        "eval_net_mechanical": eval_net,
    })

    cv = CombinatorialPurgedCV(n_groups=7, k_test=2, purge_gap=1, embargo_gap=1)
    ev = cpcv_path_evidence(
        dev, ["f1", "f2"], "target_return", "eval_net_mechanical", "trade_date",
        cv=cv,
        candidate_params={"n_estimators": 20, "num_leaves": 7},
        control_params={"n_estimators": 5, "num_leaves": 2, "learning_rate": 0.05},
        huber_delta=0.9,
        control_huber_delta=0.9,
    )

    assert ev["n_path_deltas"] == 21  # C(7, 2) test-bin combinations
    assert len(ev["path_deltas"]) == 21
    assert 0.0 <= ev["ic_path_win_rate"] <= 1.0
    assert 0.0 <= ev["top1_path_win_rate"] <= 1.0
    assert 0.0 <= ev["p_bootstrap"] <= 1.0
    assert 0.0 <= ev["p_paired_t"] <= 1.0
    assert np.isfinite(ev["mde_top1"]) and ev["mde_top1"] > 0.0
    assert np.isfinite(ev["mde_ic"]) and ev["mde_ic"] > 0.0
    assert np.isfinite(ev["ic_candidate"]) and np.isfinite(ev["ic_control"])
    assert np.isclose(ev["ic_delta"], ev["ic_candidate"] - ev["ic_control"])
    assert np.isclose(ev["pooled_delta"], float(np.mean(ev["path_deltas"])))

def test_cpcv_path_evidence_skips_days_with_unlabelled_top1_pick() -> None:
    """Regression: a top-1 pick lacking a mechanical label (NaN eval_col) must
    not crash the bootstrap/paired-t significance tests (measured 2026-09-06
    on the real panel: mechanical label coverage is 99.68%, not 100%)."""
    import numpy as np
    import pandas as pd

    from src.ml.robust_eval import CombinatorialPurgedCV
    from src.ml.validation import cpcv_path_evidence

    rng = np.random.default_rng(11)
    days = pd.to_datetime(pd.date_range("2024-01-01", periods=120, freq="D"))
    rows = 6
    n = len(days) * rows
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    eval_net = 0.03 * f1 + 0.01 * f2 + rng.normal(scale=0.02, size=n)
    # Blank out the eval label for the single best-scoring row on a handful of
    # dates so their top-1 pick has no mechanical label, exactly as a missing
    # next-trading-day price does in build_decision_labels.
    dev = pd.DataFrame({
        "trade_date": np.repeat(days.to_numpy(), rows),
        "stock_code": [f"{i % 40:06d}" for i in range(n)],
        "f1": f1,
        "f2": f2,
        "target_return": eval_net,
        "eval_net_mechanical": eval_net,
    })
    top_idx = dev.groupby("trade_date")["f1"].idxmax()
    dev.loc[top_idx[:15], "eval_net_mechanical"] = np.nan

    cv = CombinatorialPurgedCV(n_groups=7, k_test=2, purge_gap=1, embargo_gap=1)
    ev = cpcv_path_evidence(
        dev, ["f1", "f2"], "target_return", "eval_net_mechanical", "trade_date",
        cv=cv,
        candidate_params={"n_estimators": 20, "num_leaves": 7},
        control_params={"n_estimators": 5, "num_leaves": 2, "learning_rate": 0.05},
        huber_delta=0.9,
        control_huber_delta=0.9,
    )

    assert 0.0 <= ev["p_bootstrap"] <= 1.0
    assert 0.0 <= ev["p_paired_t"] <= 1.0
    assert np.isfinite(ev["pooled_delta"])
    assert np.isfinite(ev["mde_top1"]) and ev["mde_top1"] > 0.0


