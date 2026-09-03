# ruff: noqa: PERF401, B007
import numpy as np
import pandas as pd

from src.ml.champion import train_tuned_champion_bundle
from src.ml.tuning import ChampionTuningConfig
from tests.integration.ml.conftest import synthetic_trade_log


def test_train_tuned_champion_bundle_records_exit_policy_grid(tmp_path) -> None:
    # Given a synthetic trade log and a matching synthetic price history
    trade_log = synthetic_trade_log(n_dates=260, per_day=6)
    codes = sorted(trade_log["\uc885\ubaa9\ucf54\ub4dc"].unique())
    log_dates = pd.to_datetime(sorted(trade_log["\ub9e4\uc218\ub0a0\uc9dc"].unique()))
    ph_dates = pd.bdate_range(log_dates.min(), log_dates.max() + pd.Timedelta(days=7))
    rng = np.random.default_rng(0)
    rows = []
    for code in codes:
        for d in ph_dates:
            rows.append(
                {
                    "date": d, "symbol": code,
                    "open": 100.4 + rng.normal(0, 0.1), "high": 106.0 + rng.normal(0, 0.3),
                    "low": 96.0, "close": 100.0 + rng.normal(0, 0.2),
                    "daily_change_pct": 0.03,
                }
            )
    price_history = pd.DataFrame(rows)
    cfg = ChampionTuningConfig(
        n_splits=4, inner_n_splits=3, hpo_trials=3, seed_ensemble=(42, 7),
        p_good_weight_grid=(0.0, 0.5, 1.0), oos_reserve_start="2023-11-01",
        require_beats_control=False,
    )

    # When
    bundle = train_tuned_champion_bundle(trade_log, None, cfg, export_dir=str(tmp_path), price_history_df=price_history)

    # Then
    prov = bundle["tuning_provenance"]["exit_policy_grid"]
    assert prov["status"] in {"evaluated", "skipped"}
    if prov["status"] == "evaluated":
        assert "grid" in prov and len(prov["grid"]) >= 1
        assert "incumbent_mean_net" in prov

    # And with no price history the champion flow still records a skipped status
    bundle_none = train_tuned_champion_bundle(trade_log, None, cfg, export_dir=str(tmp_path))
    assert bundle_none["tuning_provenance"]["exit_policy_grid"]["status"] == "skipped"


