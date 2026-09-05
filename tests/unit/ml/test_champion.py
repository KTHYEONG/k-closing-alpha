"""champion 오케스트레이터의 feature_set / price_history 배선 계약."""
from __future__ import annotations

import inspect

from src.ml.champion import train_champion_bundle, train_tuned_champion_bundle


def test_champion_entrypoints_accept_feature_set_and_price_history() -> None:
    for fn in (train_champion_bundle, train_tuned_champion_bundle):
        params = inspect.signature(fn).parameters
        assert params["feature_set"].default == "close_morning61"
        assert "price_history_df" in params
        assert params["price_history_df"].default is None


import numpy as np

from src.ml.champion import evaluate_promotion


def test_evaluate_promotion_is_significance_based() -> None:
    rng = np.random.default_rng(11)
    ctrl = rng.normal(0.0, 0.03, size=300)
    strong_cand = ctrl + 0.004

    promoted = evaluate_promotion(strong_cand, ctrl, alpha=0.10)
    assert promoted["promoted"] is True
    assert promoted["p_value"] < 0.10
    assert promoted["delta"] > 0.0
    assert promoted["method"] == "moving_block_bootstrap"
    assert promoted["n_obs"] == 300

    tie = evaluate_promotion(ctrl.copy(), ctrl, alpha=0.10)
    assert tie["promoted"] is False

    noisy_cand = ctrl + rng.normal(0.0005, 0.03, size=300)
    marginal = evaluate_promotion(noisy_cand, ctrl, alpha=0.10)
    assert marginal["promoted"] == (marginal["delta"] > 0.0 and marginal["p_value"] < 0.10)


import numpy as np
import pandas as pd

from src.ml.champion import train_tuned_champion_bundle
from src.ml.tuning import ChampionTuningConfig


def _raw_trade_log(n_dates: int = 90, per_day: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for d in pd.bdate_range("2023-01-02", periods=n_dates):
        for j in range(per_day):
            e = rng.normal()
            rows.append(
                {
                    "\ub9e4\uc218\ub0a0\uc9dc": d.strftime("%Y-%m-%d"),
                    "\uc885\ubaa9\ucf54\ub4dc": f"{j:06d}",
                    "(\uc2dc\uac00)": "10000",
                    "(\uace0\uac00)": "10400",
                    "(\uc800\uac00)": "9800",
                    "(\uc885\uac00)": "10200",
                    "(\uc804\uc77c\uc885\uac00)": "10000",
                    "(\uc2dc\uac00\ucd1d\uc561, \uc5b5)": "5000",
                    "(\uac70\ub798\ub300\uae08, \uc5b5)": "300",
                    "(\ub4f1\ub77d\ub960)": f"{2 + e:.2f}",
                    "(\uc120\uc815 \uc21c\uc704)": str(j + 1),
                    "(\uae30\uad00_\uc21c\ub9e4\uc218)": f"{e * 100:.0f}",
                    "(\uc678\uad6d\uc778_\uc21c\ub9e4\uc218)": f"{e * 80:.0f}",
                    "(\ud504\ub85c\uadf8\ub7a8_\uc21c\ub9e4\uc218)": f"{e * 50:.0f}",
                    "(\uccb4\uacb0\uac15\ub3c4)": "120",
                    "(\uc2dc\uc7a5\uad6c\ubd84)": "KOSPI",
                    "(\ucd1d \uc885\ubaa9 \uc218)": str(per_day),
                    "(\ud3c9\uade0 \uac70\ub798\ub300\uae08)": "250",
                    "(kospi, %)": "0.3",
                    "(kosdaq, %)": "0.1",
                    "v_kospi": "18",
                    "v_kosdaq": "20",
                    "(\uac70\ub798\ub7c9)": "100000",
                    "(\ud14c\ub9c8/\uc139\ud130)": "\ubc18\ub3c4\uccb4",
                    "(\ucc28\ud2b8\ubd84\uc11d)": "\uac70\ub798\ub7c9 \ud3ed\uc99d",
                    "(\ub9e4\uc218 \uac00\uaca9)": "10200",
                    "(\ub9e4\ub3c4 \uac00\uaca9)": f"{10200 * (1 + 0.01 * e):.0f}",
                    "(\uc218\uc775\ub960, %)": f"{e:.2f}",
                }
            )
    return pd.DataFrame(rows)


def test_tuned_champion_provenance_records_bootstrap_gate() -> None:
    trade_log = _raw_trade_log()
    cfg = ChampionTuningConfig(hpo_trials=2, seed_ensemble=(13, 29), require_beats_control=False, min_history_dates=20)

    bundle = train_tuned_champion_bundle(trade_log, None, cfg, export_dir="tmp/spec_champion")

    cvc = bundle["tuning_provenance"]["control_vs_candidate"]
    assert "p_value" in cvc
    assert "delta" in cvc
    assert "ci_low" in cvc and "ci_high" in cvc
    assert cvc["promotion_alpha"] == cfg.promotion_alpha
    assert isinstance(cvc["promoted"], bool)


def test_close_morning_reranker_config_p_good_weight_is_zero() -> None:
    from src.serving.realtime.inference import (
        _CLOSE_MORNING_RERANKER_CONFIG,
        _CLOSE_MORNING_RERANKER_V2_RESEARCH_CONFIG,
    )

    assert _CLOSE_MORNING_RERANKER_CONFIG["p_good_weight"] == 0.0
    assert _CLOSE_MORNING_RERANKER_V2_RESEARCH_CONFIG["p_good_weight"] == 0.5


def test_train_tuned_champion_skips_hpo_with_model_params_override(monkeypatch) -> None:
    import src.ml.champion as champ
    from src.ml.bundle import CHAMPION_DEFAULT_MODEL_PARAMS

    calls: list[int] = []
    monkeypatch.setattr(champ, "tune_return_model_params", lambda *a, **k: calls.append(1))

    cfg = ChampionTuningConfig(
        seed_ensemble=(13, 29), require_beats_control=False, min_history_dates=20,
        model_params_override=dict(CHAMPION_DEFAULT_MODEL_PARAMS),
    )
    bundle = train_tuned_champion_bundle(_raw_trade_log(), None, cfg, export_dir="tmp/spec_override")

    assert not calls
    prov = bundle["tuning_provenance"]
    assert prov["objective"] == "override"
    assert prov["n_trials"] == 0


def test_champion_provenance_defaults_to_skipped_without_notional() -> None:
    from src.ml.champion import ChampionTuningConfig

    # Given / When: the default config
    config = ChampionTuningConfig()

    # Then: the research knob is opt-in, so no existing caller changes behavior
    assert config.buyability_target_notional_100m is None



def test_champion_buyability_evaluated_path_records_provenance() -> None:
    import pandas as pd

    from src.ml.buyability import evaluate_buyability_sleeves, summarize_buyability_sleeves
    from src.ml.champion import train_tuned_champion_bundle
    from src.ml.tuning import ChampionTuningConfig
    from tests.unit.ml.test_champion import _raw_trade_log

    cfg = ChampionTuningConfig(
        hpo_trials=2,
        seed_ensemble=(13, 29),
        require_beats_control=False,
        min_history_dates=20,
        model_params_override={"num_leaves": 7},
        buyability_target_notional_100m=1.0,
    )
    bundle = train_tuned_champion_bundle(
        _raw_trade_log(n_dates=40, per_day=4), None, cfg, export_dir="tmp/spec_buyability"
    )
    prov = bundle["tuning_provenance"]["buyability_sleeves"]
    assert prov["status"] == "evaluated"
    assert set(prov["sleeves"]) == {"fillable", "ceiling", "pooled"}

    # Direct sleeve call keeps wiring import live
    oof = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "stock_code": ["005930", "000660"],
            "pred": [0.5, 0.6],
            "net_return": [0.2, 0.3],
            "close_price": [11000.0, 11000.0],
            "prev_close_price": [10000.0, 10000.0],
            "high_price": [11200.0, 11200.0],
            "auction_value_100m": [40.0, 40.0],
            "auction_vol_share": [0.01, 0.01],
            "auction_bars_found": [True, True],
        }
    )
    assert "fillable" in summarize_buyability_sleeves(evaluate_buyability_sleeves(oof, target_notional_100m=1.0))["sleeves"]


def test_champion_buyability_value_error_degrades_to_skipped(monkeypatch) -> None:
    import src.ml.champion as champ
    from src.ml.tuning import ChampionTuningConfig
    from tests.unit.ml.test_champion import _raw_trade_log

    def _boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(champ, "evaluate_buyability_sleeves", _boom)
    cfg = ChampionTuningConfig(
        hpo_trials=2,
        seed_ensemble=(13, 29),
        require_beats_control=False,
        min_history_dates=20,
        model_params_override={"num_leaves": 7},
        buyability_target_notional_100m=1.0,
    )
    bundle = train_tuned_champion_bundle(
        _raw_trade_log(n_dates=40, per_day=4), None, cfg, export_dir="tmp/spec_buyability_boom"
    )
    prov = bundle["tuning_provenance"]["buyability_sleeves"]
    assert prov["status"] == "skipped"
    assert prov["reason"] == "boom"


def test_champion_execution_cost_evaluated_path_records_provenance() -> None:
    from src.ml.champion import train_tuned_champion_bundle
    from src.ml.tuning import ChampionTuningConfig
    from tests.unit.ml.test_champion import _raw_trade_log

    cfg = ChampionTuningConfig(
        hpo_trials=2,
        seed_ensemble=(13, 29),
        require_beats_control=False,
        min_history_dates=20,
        model_params_override={"num_leaves": 7},
    )
    bundle = train_tuned_champion_bundle(
        _raw_trade_log(n_dates=40, per_day=4), None, cfg, export_dir="tmp/spec_execution_cost"
    )
    prov = bundle["tuning_provenance"]["execution_cost"]
    assert prov["status"] == "evaluated"
    assert prov["n_rows"] > 0
    assert prov["n_impact_measured"] == 0
    assert prov["breakeven_cost_bp"] == prov["breakeven_cost_bp"]


def test_champion_execution_cost_value_error_degrades_to_skipped(monkeypatch) -> None:
    import src.ml.champion as champ
    from src.ml.tuning import ChampionTuningConfig
    from tests.unit.ml.test_champion import _raw_trade_log

    def _boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(champ, "estimate_round_trip_cost_bp", _boom)
    cfg = ChampionTuningConfig(
        hpo_trials=2,
        seed_ensemble=(13, 29),
        require_beats_control=False,
        min_history_dates=20,
        model_params_override={"num_leaves": 7},
    )
    bundle = train_tuned_champion_bundle(
        _raw_trade_log(n_dates=40, per_day=4), None, cfg, export_dir="tmp/spec_execution_cost_boom"
    )
    prov = bundle["tuning_provenance"]["execution_cost"]
    assert prov["status"] == "skipped"
    assert prov["reason"] == "boom"
