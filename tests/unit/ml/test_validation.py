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

def test_run_promotion_gate_blocks_underpowered_oos_ic() -> None:
    from src.ml.validation import PromotionDecision, ValidationConfig, run_promotion_gate

    cpcv = {
        "path_deltas": [0.003] * 28,
        "n_path_deltas": 28,
        "top1_path_win_rate": 0.86,
        "ic_path_win_rate": 0.96,
        "pooled_delta": 0.0033,
        "p_bootstrap": 0.0,
        "p_paired_t": 0.0012,
        "ic_candidate": 0.20,
        "ic_control": 0.17,
        "ic_delta": 0.03,
        "mde_top1": 0.0027,
        "mde_ic": 0.016,
    }
    oos = {
        "n_days": 244, "n_rows": 3800, "top1_mean": 0.004, "top1_sd": 0.046,
        "top1_sharpe": 1.4, "rank_ic": 0.02, "mde_top1": 0.83, "mde_ic": 0.049,
        "first_date": "2025-09-01", "last_date": "2026-08-25",
    }
    fillable = {"n_days": 200, "top1_mean": 0.003, "rank_ic": 0.02, "measured_share": 0.90}
    config = ValidationConfig(oos_reserve_start="2025-09-01")

    decision = run_promotion_gate(
        cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600,
        fillable=fillable, config=config,
    )

    assert isinstance(decision, PromotionDecision)
    assert decision.deployable is False
    assert decision.verdict == "research_only"
    assert "oos_rank_ic_above_mde" in decision.failed_gates
    # Every gate is reported, passed or not, so the verdict is auditable
    assert {g.name for g in decision.gates} >= {
        "cpcv_ic_path_win_rate", "cpcv_top1_path_win_rate", "cpcv_delta_significance",
        "selection_dsr", "oos_min_days", "oos_rank_ic_above_mde", "oos_top1_sign",
        "fillable_top1_sign",
    }

def test_run_promotion_gate_promotes_when_every_gate_passes() -> None:
    from src.ml.validation import GateOutcome, ValidationConfig, run_promotion_gate

    cpcv = {
        "path_deltas": [0.003] * 28,
        "n_path_deltas": 28,
        "top1_path_win_rate": 0.86,
        "ic_path_win_rate": 0.96,
        "pooled_delta": 0.0033,
        "p_bootstrap": 0.0,
        "p_paired_t": 0.0012,
        "ic_candidate": 0.20,
        "ic_control": 0.17,
        "ic_delta": 0.03,
        "mde_top1": 0.0027,
        "mde_ic": 0.016,
    }
    oos = {
        "n_days": 244, "n_rows": 3800, "top1_mean": 0.006, "top1_sd": 0.046,
        "top1_sharpe": 2.0, "rank_ic": 0.13, "mde_top1": 0.83, "mde_ic": 0.049,
        "first_date": "2025-09-01", "last_date": "2026-08-25",
    }
    fillable = {"n_days": 200, "top1_mean": 0.004, "rank_ic": 0.11, "measured_share": 0.90}
    config = ValidationConfig(oos_reserve_start="2025-09-01")

    decision = run_promotion_gate(
        cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600,
        fillable=fillable, config=config,
        temporal={"sign_consistent": True, "first_mean": 0.002, "second_mean": 0.0011},
        execution_profile={"fill_rate": 0.4, "saving_survives_adverse_selection": True},
    )

    assert decision.deployable is True
    assert decision.verdict == "deployable"
    assert decision.failed_gates == ()
    assert all(isinstance(g, GateOutcome) for g in decision.gates)
    assert all(g.passed for g in decision.gates)

def test_run_promotion_gate_uses_the_conservative_p_value() -> None:
    from src.ml.validation import ValidationConfig, run_promotion_gate

    cpcv = {
        "path_deltas": [0.003] * 28,
        "n_path_deltas": 28,
        "top1_path_win_rate": 0.86,
        "ic_path_win_rate": 0.96,
        "pooled_delta": 0.0033,
        "p_bootstrap": 0.0,      # block-bootstrap says "certain"
        "p_paired_t": 0.20,      # analytic paired-t disagrees
        "ic_candidate": 0.20,
        "ic_control": 0.17,
        "ic_delta": 0.03,
        "mde_top1": 0.0027,
        "mde_ic": 0.016,
    }
    oos = {
        "n_days": 244, "n_rows": 3800, "top1_mean": 0.006, "top1_sd": 0.046,
        "top1_sharpe": 2.0, "rank_ic": 0.13, "mde_top1": 0.83, "mde_ic": 0.049,
        "first_date": "2025-09-01", "last_date": "2026-08-25",
    }
    fillable = {"n_days": 200, "top1_mean": 0.004, "rank_ic": 0.11, "measured_share": 0.90}
    config = ValidationConfig(oos_reserve_start="2025-09-01", promotion_alpha=0.05)

    decision = run_promotion_gate(
        cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600,
        fillable=fillable, config=config,
    )

    assert decision.deployable is False
    assert "cpcv_delta_significance" in decision.failed_gates
    gate = next(g for g in decision.gates if g.name == "cpcv_delta_significance")
    assert gate.observed == 0.20

def test_run_promotion_gate_blocks_saturated_selection_dsr() -> None:
    import pytest

    from src.ml.validation import ValidationConfig, run_promotion_gate

    cpcv = {
        "path_deltas": [0.003] * 28,
        "n_path_deltas": 28,
        "top1_path_win_rate": 0.86,
        "ic_path_win_rate": 0.96,
        "pooled_delta": 0.0033,
        "p_bootstrap": 0.0,
        "p_paired_t": 0.0012,
        "ic_candidate": 0.20,
        "ic_control": 0.17,
        "ic_delta": 0.03,
        "mde_top1": 0.0027,
        "mde_ic": 0.016,
    }
    oos = {
        "n_days": 244, "n_rows": 3800, "top1_mean": 0.006, "top1_sd": 0.046,
        "top1_sharpe": 2.0, "rank_ic": 0.13, "mde_top1": 0.83, "mde_ic": 0.049,
        "first_date": "2025-09-01", "last_date": "2026-08-25",
    }
    fillable = {"n_days": 200, "top1_mean": 0.004, "rank_ic": 0.11, "measured_share": 0.90}
    config = ValidationConfig(oos_reserve_start="2025-09-01")

    decision = run_promotion_gate(
        cpcv=cpcv, oos=oos, selection_dsr=0.80, n_selection_trials=600,
        fillable=fillable, config=config,
    )
    assert decision.deployable is False
    assert "selection_dsr" in decision.failed_gates

    # A missing DSR is a hard failure, never a silent pass
    missing = run_promotion_gate(
        cpcv=cpcv, oos=oos, selection_dsr=None, n_selection_trials=600,
        fillable=fillable, config=config,
    )
    assert missing.deployable is False
    assert "selection_dsr" in missing.failed_gates

    with pytest.raises(ValueError, match="n_selection_trials"):
        run_promotion_gate(
            cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=0,
            fillable=fillable, config=config,
        )

def test_validation_config_rejects_out_of_domain_settings() -> None:
    import pytest

    from src.ml.validation import ValidationConfig

    cfg = ValidationConfig(oos_reserve_start="2025-09-01")
    assert cfg.eval_col == "eval_net_mechanical"
    assert cfg.min_ic_path_win_rate == 0.75
    assert cfg.min_top1_path_win_rate == 0.60
    assert cfg.min_oos_days == 60
    assert cfg.require_execution_profile is True

    with pytest.raises(ValueError, match="oos_reserve_start"):
        ValidationConfig(oos_reserve_start="not-a-date")
    with pytest.raises(ValueError, match="min_ic_path_win_rate"):
        ValidationConfig(oos_reserve_start="2025-09-01", min_ic_path_win_rate=1.5)
    with pytest.raises(ValueError, match="promotion_alpha"):
        ValidationConfig(oos_reserve_start="2025-09-01", promotion_alpha=0.0)
    with pytest.raises(ValueError, match="min_oos_days"):
        ValidationConfig(oos_reserve_start="2025-09-01", min_oos_days=0)
    with pytest.raises(ValueError, match="require_execution_profile"):
        ValidationConfig(oos_reserve_start="2025-09-01", require_execution_profile="yes")  # type: ignore[arg-type]

def test_evaluate_locked_oos_rejects_dev_overlap_and_scores_once() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.validation import evaluate_locked_oos

    rng = np.random.default_rng(3)
    days = pd.to_datetime(pd.date_range("2024-01-01", periods=180, freq="D"))
    rows = 5

    def _panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
        n = len(dates) * rows
        f1 = rng.normal(size=n)
        f2 = rng.normal(size=n)
        eval_net = 0.02 * f1 + rng.normal(scale=0.01, size=n)
        return pd.DataFrame({
            "trade_date": np.repeat(dates.to_numpy(), rows),
            "stock_code": [f"{i % 50:06d}" for i in range(n)],
            "f1": f1,
            "f2": f2,
            "target_return": eval_net,
            "eval_net_mechanical": eval_net,
        })

    dev = _panel(days[:150])
    oos = _panel(days[150:])

    result = evaluate_locked_oos(
        dev, oos, ["f1", "f2"], "target_return", "eval_net_mechanical", "trade_date",
        model_params={"n_estimators": 20, "num_leaves": 7}, huber_delta=0.9, seeds=(13, 29),
    )

    assert result["n_days"] == 30
    assert result["n_rows"] == 30 * rows
    assert np.isfinite(result["top1_mean"])
    assert np.isfinite(result["rank_ic"])
    assert -1.0 <= result["rank_ic"] <= 1.0
    assert np.isfinite(result["mde_ic"]) and result["mde_ic"] > 0.0
    assert str(result["first_date"])[:10] == "2024-05-30"

    with pytest.raises(ValueError, match="overlap"):
        evaluate_locked_oos(
            dev, dev, ["f1", "f2"], "target_return", "eval_net_mechanical", "trade_date",
            model_params={"n_estimators": 20}, huber_delta=0.9, seeds=(13,),
        )

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


def test_evaluate_locked_oos_skips_days_with_unlabelled_top1_pick() -> None:
    """Regression: same NaN-top1 gap on the locked OOS window."""
    import numpy as np
    import pandas as pd

    from src.ml.validation import evaluate_locked_oos

    rng = np.random.default_rng(3)
    days = pd.to_datetime(pd.date_range("2024-01-01", periods=180, freq="D"))
    rows = 5

    def _panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
        n = len(dates) * rows
        f1 = rng.normal(size=n)
        f2 = rng.normal(size=n)
        eval_net = 0.02 * f1 + rng.normal(scale=0.01, size=n)
        return pd.DataFrame({
            "trade_date": np.repeat(dates.to_numpy(), rows),
            "stock_code": [f"{i % 50:06d}" for i in range(n)],
            "f1": f1,
            "f2": f2,
            "target_return": eval_net,
            "eval_net_mechanical": eval_net,
        })

    dev = _panel(days[:150])
    oos = _panel(days[150:])
    top_idx = oos.groupby("trade_date")["f1"].idxmax()
    oos.loc[top_idx[:5], "eval_net_mechanical"] = np.nan

    result = evaluate_locked_oos(
        dev, oos, ["f1", "f2"], "target_return", "eval_net_mechanical", "trade_date",
        model_params={"n_estimators": 20, "num_leaves": 7}, huber_delta=0.9, seeds=(13, 29),
    )

    assert result["n_days"] == 25  # 30 OOS dates minus the 5 unlabelled top-1 picks
    assert np.isfinite(result["top1_mean"])
    assert np.isfinite(result["top1_sd"])


def test_publish_bundle_refuses_when_not_deployable(tmp_path) -> None:
    import json

    from src.ml.validation import GateOutcome, PromotionDecision, publish_bundle

    candidate_dir = tmp_path / "candidate"
    production_dir = tmp_path / "production"
    production_dir.mkdir()
    incumbent = production_dir / "sizing_pipeline_bundle.joblib"
    incumbent.write_bytes(b"incumbent")

    decision = PromotionDecision(
        deployable=False,
        verdict="research_only",
        failed_gates=("oos_rank_ic_above_mde",),
        gates=(GateOutcome(name="oos_rank_ic_above_mde", passed=False, observed=0.02, threshold=0.049, detail={}),),
        evidence={},
    )

    result = publish_bundle({"feature_cols": ["f1"]}, str(candidate_dir), str(production_dir), decision)

    assert result["published"] is False
    assert result["failed_gates"] == ["oos_rank_ic_above_mde"]
    # The candidate artifact is still written for research
    assert (candidate_dir / "sizing_pipeline_bundle.joblib").exists()
    # Production is untouched: same bytes, no record, no backup
    assert incumbent.read_bytes() == b"incumbent"
    assert not (production_dir / "promotion_record.json").exists()
    assert not list(production_dir.glob("*.bak_*"))
    assert json.dumps(result)  # result stays JSON-serialisable for logging

def test_publish_bundle_backs_up_incumbent_and_writes_record(tmp_path) -> None:
    import json

    import joblib

    from src.ml.validation import GateOutcome, PromotionDecision, publish_bundle

    candidate_dir = tmp_path / "candidate"
    production_dir = tmp_path / "production"
    production_dir.mkdir()
    (production_dir / "sizing_pipeline_bundle.joblib").write_bytes(b"incumbent")

    decision = PromotionDecision(
        deployable=True,
        verdict="deployable",
        failed_gates=(),
        gates=(GateOutcome(name="oos_rank_ic_above_mde", passed=True, observed=0.13, threshold=0.049, detail={}),),
        evidence={"oos_reserve_start": "2025-09-01"},
    )
    bundle = {"feature_cols": ["f1"], "training_cutoff": "2025-08-29", "round_trip_cost": 0.0046}

    result = publish_bundle(bundle, str(candidate_dir), str(production_dir), decision)

    assert result["published"] is True
    backups = list(production_dir.glob("sizing_pipeline_bundle.joblib.bak_*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == b"incumbent"

    promoted = joblib.load(production_dir / "sizing_pipeline_bundle.joblib")
    assert promoted["feature_cols"] == ["f1"]

    record = json.loads((production_dir / "promotion_record.json").read_text(encoding="utf-8"))
    assert record["verdict"] == "deployable"
    assert record["training_cutoff"] == "2025-08-29"
    assert record["round_trip_cost"] == 0.0046
    assert record["oos_reserve_start"] == "2025-09-01"
    assert len(record["sha256"]) == 64
    assert [g["name"] for g in record["gates"]] == ["oos_rank_ic_above_mde"]


def test_validation_config_extra_domains_fail_closed() -> None:
    import pytest
    from src.ml.validation import ValidationConfig
    with pytest.raises(ValueError, match="eval_col"):
        ValidationConfig(oos_reserve_start="2025-09-01", eval_col="")
    with pytest.raises(ValueError, match="cpcv_n_groups"):
        ValidationConfig(oos_reserve_start="2025-09-01", cpcv_n_groups=3)
    with pytest.raises(ValueError, match="cpcv_k_test"):
        ValidationConfig(oos_reserve_start="2025-09-01", cpcv_k_test=8)
    with pytest.raises(ValueError, match="purge_gap"):
        ValidationConfig(oos_reserve_start="2025-09-01", purge_gap=-1)
    with pytest.raises(ValueError, match="min_oos_rank_ic"):
        ValidationConfig(oos_reserve_start="2025-09-01", min_oos_rank_ic=1.5)
    with pytest.raises(ValueError, match="target_notional_100m"):
        ValidationConfig(oos_reserve_start="2025-09-01", target_notional_100m=0.0)


def test_evaluate_locked_oos_rejects_empty_window() -> None:
    import pandas as pd
    import pytest
    from src.ml.validation import evaluate_locked_oos
    dev = pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-01"]), "f1": [0.1], "target_return": [0.01], "eval_net_mechanical": [0.01]})
    oos = dev.iloc[0:0].copy()
    with pytest.raises(ValueError, match="empty"):
        evaluate_locked_oos(dev, oos, ["f1"], "target_return", "eval_net_mechanical", "trade_date", model_params={"n_estimators": 5}, huber_delta=0.9, seeds=(13,))


def test_run_promotion_gate_fillable_none_and_opt_out() -> None:
    from src.ml.validation import ValidationConfig, run_promotion_gate
    cpcv = {"path_deltas": [0.003]*28, "n_path_deltas": 28, "top1_path_win_rate": 0.86, "ic_path_win_rate": 0.96, "pooled_delta": 0.0033, "p_bootstrap": 0.0, "p_paired_t": 0.0012, "ic_candidate": 0.20, "ic_control": 0.17, "ic_delta": 0.03, "mde_top1": 0.0027, "mde_ic": 0.016}
    oos = {"n_days": 244, "n_rows": 3800, "top1_mean": 0.006, "top1_sd": 0.046, "top1_sharpe": 2.0, "rank_ic": 0.13, "mde_top1": 0.83, "mde_ic": 0.049, "first_date": "2025-09-01", "last_date": "2026-08-25"}
    cfg = ValidationConfig(oos_reserve_start="2025-09-01")
    missing = run_promotion_gate(cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600, fillable=None, config=cfg)
    assert missing.deployable is False
    assert "fillable_top1_sign" in missing.failed_gates
    cfg_opt = ValidationConfig(oos_reserve_start="2025-09-01", require_fillable_sleeve=False)
    opt = run_promotion_gate(cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600, fillable=None, config=cfg_opt)
    assert "fillable_top1_sign" not in opt.failed_gates


def test_temporal_sign_consistency_blocks_a_half_sample_flip() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.validation import temporal_sign_consistency

    idx = pd.to_datetime(pd.date_range("2019-01-01", periods=400, freq="D"))
    # positive in the first half, negative in the second -- the measured failure shape
    flipping = pd.Series(np.concatenate([np.full(200, 0.002), np.full(200, -0.0015)]), index=idx)
    stable = pd.Series(np.full(400, 0.0012), index=idx)

    bad = temporal_sign_consistency(flipping, split_date="2019-07-20")
    good = temporal_sign_consistency(stable, split_date="2019-07-20")

    assert bad["sign_consistent"] is False
    assert bad["first_mean"] > 0.0 > bad["second_mean"]
    assert bad["n_first"] > 0 and bad["n_second"] > 0
    assert good["sign_consistent"] is True
    assert np.isclose(good["first_mean"], 0.0012)


def test_run_promotion_gate_blocks_temporal_sign_flip() -> None:
    from src.ml.validation import ValidationConfig, run_promotion_gate

    cpcv = {
        "path_deltas": [0.003] * 28, "n_path_deltas": 28,
        "top1_path_win_rate": 0.86, "ic_path_win_rate": 0.96,
        "pooled_delta": 0.0033, "p_bootstrap": 0.0, "p_paired_t": 0.0012,
        "ic_candidate": 0.20, "ic_control": 0.17, "ic_delta": 0.03,
        "mde_top1": 0.0027, "mde_ic": 0.016,
    }
    oos = {"n_days": 244, "n_rows": 3800, "top1_mean": 0.006, "top1_sd": 0.046,
           "top1_sharpe": 2.0, "rank_ic": 0.13, "mde_top1": 0.83, "mde_ic": 0.049,
           "first_date": "2025-09-01", "last_date": "2026-08-25"}
    fillable = {"n_days": 200, "top1_mean": 0.004, "rank_ic": 0.11, "measured_share": 0.90}
    config = ValidationConfig(oos_reserve_start="2025-09-01")

    flipped = run_promotion_gate(
        cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600, fillable=fillable,
        config=config, temporal={"sign_consistent": False, "first_mean": 0.002, "second_mean": -0.0015},
    )
    assert flipped.deployable is False
    assert "temporal_sign_consistency" in flipped.failed_gates

    held = run_promotion_gate(
        cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600, fillable=fillable,
        config=config, temporal={"sign_consistent": True, "first_mean": 0.002, "second_mean": 0.0011},
        execution_profile={"fill_rate": 0.4, "saving_survives_adverse_selection": True},
    )
    assert held.deployable is True


def test_run_promotion_gate_blocks_unmeasured_execution_profile() -> None:
    """R1: promotion requires execution to have been measured at all -- the
    gate does not require the measurement to be favourable, since
    fill_rate_is_upper_bound already caps how optimistic it can be."""
    from src.ml.validation import ValidationConfig, run_promotion_gate

    cpcv = {
        "path_deltas": [0.003] * 28, "n_path_deltas": 28,
        "top1_path_win_rate": 0.86, "ic_path_win_rate": 0.96,
        "pooled_delta": 0.0033, "p_bootstrap": 0.0, "p_paired_t": 0.0012,
        "ic_candidate": 0.20, "ic_control": 0.17, "ic_delta": 0.03,
        "mde_top1": 0.0027, "mde_ic": 0.016,
    }
    oos = {"n_days": 244, "n_rows": 3800, "top1_mean": 0.006, "top1_sd": 0.046,
           "top1_sharpe": 2.0, "rank_ic": 0.13, "mde_top1": 0.83, "mde_ic": 0.049,
           "first_date": "2025-09-01", "last_date": "2026-08-25"}
    fillable = {"n_days": 200, "top1_mean": 0.004, "rank_ic": 0.11, "measured_share": 0.90}
    temporal = {"sign_consistent": True, "first_mean": 0.002, "second_mean": 0.0011}
    config = ValidationConfig(oos_reserve_start="2025-09-01")

    unmeasured = run_promotion_gate(
        cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600, fillable=fillable,
        config=config, temporal=temporal, execution_profile=None,
    )
    assert unmeasured.deployable is False
    assert "execution_profile_measured" in unmeasured.failed_gates

    # Unfavourable savings still count as "measured" -- the gate is about
    # measurement having happened, not about the number being good.
    unfavourable = run_promotion_gate(
        cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600, fillable=fillable,
        config=config, temporal=temporal,
        execution_profile={"fill_rate": 0.3, "saving_survives_adverse_selection": False},
    )
    assert unfavourable.deployable is True
    assert "execution_profile_measured" not in unfavourable.failed_gates

    # Opting out of the requirement passes even without a measurement.
    opted_out_config = ValidationConfig(oos_reserve_start="2025-09-01", require_execution_profile=False)
    opted_out = run_promotion_gate(
        cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600, fillable=fillable,
        config=opted_out_config, temporal=temporal, execution_profile=None,
    )
    assert opted_out.deployable is True


def test_evaluate_screen_grid_charges_the_selection_budget() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.universe import ScreenConfig
    from src.ml.validation import evaluate_screen_grid

    rng = np.random.default_rng(1)
    days = pd.to_datetime(pd.date_range("2024-01-01", periods=120, freq="D"))
    rows = 6
    n = len(days) * rows
    panel = pd.DataFrame({
        "trade_date": np.repeat(days.to_numpy(), rows),
        "symbol": [f"{i % 40:06d}" for i in range(n)],
        "daily_change_pct": rng.uniform(0.0, 0.25, size=n),
        "trade_value_100m": rng.uniform(50.0, 5000.0, size=n),
        "market_cap_100m": rng.uniform(600.0, 50000.0, size=n),
        "mechanical_gross": rng.normal(loc=0.002, scale=0.02, size=n),
    })
    grid = (
        ScreenConfig(change_lower=0.10, min_trade_value_100m=100.0, min_market_cap_100m=500.0),
        ScreenConfig(change_lower=0.02, change_upper=0.15, min_trade_value_100m=100.0, min_market_cap_100m=500.0),
        ScreenConfig(change_lower=0.05, change_upper=0.15, min_trade_value_100m=1000.0, min_market_cap_100m=500.0),
    )

    result = evaluate_screen_grid(panel, grid, group_col="trade_date", gross_col="mechanical_gross", cost_ratio=0.0046)

    assert result["n_screens"] == 3
    assert len(result["screens"]) == 3
    # every screen reports its model-free baseline, so a negative pool is never hidden
    for row in result["screens"]:
        assert {"net_bp", "t_stat", "per_day", "n_days"} <= set(row)
    assert result["selection_trials_multiplier"] == 3


def test_validation_fail_closed_edges() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.universe import ScreenConfig
    from src.ml.validation import ValidationConfig, evaluate_screen_grid, run_promotion_gate, temporal_sign_consistency

    with pytest.raises(ValueError, match="temporal_split_date"):
        ValidationConfig(oos_reserve_start="2025-09-01", temporal_split_date="not-a-date")
    with pytest.raises(ValueError, match="require_temporal_consistency"):
        ValidationConfig(oos_reserve_start="2025-09-01", require_temporal_consistency="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="split_date"):
        temporal_sign_consistency(pd.Series([0.01, 0.02]), split_date="not-a-date")
    idx = pd.to_datetime(pd.date_range("2023-01-01", periods=10, freq="D"))
    with pytest.raises(ValueError, match="empty"):
        temporal_sign_consistency(pd.Series(np.full(10, 0.01), index=idx), split_date="2025-01-01")
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_screen_grid(
            pd.DataFrame({"trade_date": idx[:2], "mechanical_gross": [0.01, 0.02]}),
            (), group_col="trade_date", gross_col="mechanical_gross", cost_ratio=0.0046,
        )

    # A screen matching nothing still reports a row with zero days
    days = pd.to_datetime(pd.date_range("2024-01-01", periods=10, freq="D"))
    panel = pd.DataFrame({
        "trade_date": np.repeat(days.to_numpy(), 2),
        "symbol": ["000001", "000002"] * 10,
        "daily_change_pct": np.full(20, 0.01),
        "trade_value_100m": np.full(20, 500.0),
        "market_cap_100m": np.full(20, 1000.0),
        "close": np.full(20, 10000.0),
        "high": np.concatenate([np.full(19, 10000.0), np.array([10000.0])]),
        "mechanical_gross": np.full(20, 0.002),
    })
    res = evaluate_screen_grid(
        panel, (ScreenConfig(change_lower=0.90),), group_col="trade_date", gross_col="mechanical_gross", cost_ratio=0.0046,
    )
    assert res["screens"][0]["n_days"] == 0
    # A ceiling close is dropped inside the grid when price legs are present
    panel2 = panel.copy()
    panel2.loc[0, "daily_change_pct"] = 0.30
    res2 = evaluate_screen_grid(
        panel2, (ScreenConfig(change_lower=0.0),), group_col="trade_date", gross_col="mechanical_gross", cost_ratio=0.0046,
    )
    assert res2["screens"][0]["n_rows"] == 19

    cpcv = {"path_deltas": [0.003] * 28, "n_path_deltas": 28, "top1_path_win_rate": 0.86, "ic_path_win_rate": 0.96, "pooled_delta": 0.0033, "p_bootstrap": 0.0, "p_paired_t": 0.0012, "ic_candidate": 0.20, "ic_control": 0.17, "ic_delta": 0.03, "mde_top1": 0.0027, "mde_ic": 0.016}
    oos = {"n_days": 244, "n_rows": 3800, "top1_mean": 0.006, "top1_sd": 0.046, "top1_sharpe": 2.0, "rank_ic": 0.13, "mde_top1": 0.83, "mde_ic": 0.049, "first_date": "2025-09-01", "last_date": "2026-08-25"}
    fillable = {"n_days": 200, "top1_mean": 0.004, "rank_ic": 0.11, "measured_share": 0.90}
    opt_out = ValidationConfig(oos_reserve_start="2025-09-01", require_temporal_consistency=False)
    passed = run_promotion_gate(cpcv=cpcv, oos=oos, selection_dsr=0.99, n_selection_trials=600, fillable=fillable, config=opt_out)
    assert "temporal_sign_consistency" not in passed.failed_gates
