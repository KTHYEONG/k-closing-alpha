import numpy as np
import pandas as pd

from src.ml.champion import train_champion_bundle
from src.serving.realtime.inference import predict_daily_sizing
from tests.integration.ml.conftest import synthetic_trade_log


def test_train_champion_bundle_reproduces_serving_contract(tmp_path) -> None:
    trade_log = synthetic_trade_log(n_dates=200, per_day=8)
    bundle = train_champion_bundle(trade_log, None, export_dir=str(tmp_path))
    assert bundle["feature_set"] == "close_morning61"
    assert bundle["decision_score_config"]["version"] == "close-morning-reranker-v1"
    for key in ("rank_model", "return_model", "quantile_models", "calibrators", "single_stock_policy"):
        assert key in bundle
    assert not (tmp_path / "sizing_pipeline_bundle.joblib").exists()
    snapshot = pd.DataFrame({col: [0.0, 0.1] for col in bundle["feature_cols"]})
    snapshot["date"] = "2026-09-01"
    scored = predict_daily_sizing(snapshot, bundle, group_col="date")
    assert "decision_score" in scored.columns and len(scored) == 2

import pandas as pd

from src.ml.bundle import ChronoCalibratedClassifier, SeedEnsembleModel
from src.ml.champion import train_tuned_champion_bundle
from src.ml.tuning import ChampionTuningConfig
from src.serving.realtime.inference import predict_daily_sizing
from tests.integration.ml.conftest import synthetic_trade_log


def test_train_tuned_champion_bundle_provenance_and_serving_load(tmp_path) -> None:
    trade_log = synthetic_trade_log(n_dates=260, per_day=10)
    cfg = ChampionTuningConfig(
        n_splits=4, inner_n_splits=3, hpo_trials=3, seed_ensemble=(42, 7),
        p_good_weight_grid=(0.0, 0.5, 1.0), weighting_mode="date_balanced",
        oos_reserve_start="2023-11-01", require_beats_control=False,
    )
    bundle = train_tuned_champion_bundle(trade_log, None, cfg, export_dir=str(tmp_path))
    prov = bundle["tuning_provenance"]
    assert prov["oos_reserve_start"] == "2023-11-01" and prov["oos_row_count"] > 0
    assert set(prov["best_params"]).issubset({
        "num_leaves", "min_child_samples", "learning_rate", "n_estimators",
        "reg_alpha", "reg_lambda", "subsample", "colsample_bytree", "subsample_freq",
        "min_split_gain", "path_smooth",
    })
    assert bundle["decision_score_config"]["p_good_weight"] in cfg.p_good_weight_grid
    assert isinstance(bundle["return_model"], SeedEnsembleModel)
    assert isinstance(bundle["calibrators"]["p_good"], (ChronoCalibratedClassifier, float))
    assert "promoted" in prov["control_vs_candidate"]
    snapshot = pd.DataFrame({col: [0.0, 0.1] for col in bundle["feature_cols"]})
    snapshot["date"] = "2026-09-01"
    scored = predict_daily_sizing(snapshot, bundle, group_col="date")
    assert "decision_score" in scored.columns and len(scored) == 2

import pathlib

import pytest

from src.ml.champion import train_tuned_champion_bundle
from src.ml.tuning import ChampionTuningConfig
from tests.integration.ml.conftest import synthetic_trade_log


def test_tuned_gate_blocks_write_when_control_wins(tmp_path, monkeypatch) -> None:
    import numpy as np

    trade_log = synthetic_trade_log(n_dates=220, per_day=8)
    import src.ml.champion as champ
    from src.ml.tuning import TunedSearchResult

    monkeypatch.setattr(
        champ, "tune_return_model_params",
        lambda *a, **k: TunedSearchResult(
            best_params={"num_leaves": 31}, best_value=0.0, objective="top1_return", n_trials=1, trials=()
        ),
    )
    monkeypatch.setattr(champ, "calibrate_blend_weight", lambda *a, **k: champ.BlendWeightResult(0.5, {0.5: {}}))

    _dates = np.arange("2023-01-02", "2023-04-02", dtype="datetime64[D]")

    def _fake_eval(*_a, **kw):
        # 후보(model_params 존재)는 대조군보다 낮은 스케줄 수익률을 내도록 강제
        weak = kw.get("model_params") is not None
        level = -0.01 if weak else 0.01
        return {
            "metrics": {"scheduled_mean_return": level},
            "scheduled_returns": np.full(_dates.size, level, dtype=np.float64),
            "dates": _dates.copy(),
            "policy": None,
            "oof": None,
        }

    monkeypatch.setattr(champ, "evaluate_config_oof", _fake_eval)
    cfg = ChampionTuningConfig(n_splits=4, inner_n_splits=3, hpo_trials=1, require_beats_control=True)
    with pytest.raises(ValueError, match="does not beat identical-date control"):
        train_tuned_champion_bundle(trade_log, None, cfg, export_dir=str(tmp_path))
    assert not any(pathlib.Path(tmp_path).rglob("sizing_pipeline_bundle.joblib"))
