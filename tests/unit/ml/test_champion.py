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


def test_champion_excludes_ceiling_rows_from_dev_and_control() -> None:
    import pandas as pd

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
    baseline_log = _raw_trade_log(n_dates=40, per_day=6)
    bundle_baseline = train_tuned_champion_bundle(
        baseline_log, None, cfg, export_dir="tmp/spec_ceiling_pool_baseline"
    )
    prov_baseline = bundle_baseline["tuning_provenance"]["ceiling_excluded_from_pool"]

    # Given: the same log plus one extra ceiling-close row (close/prev_close>=1.29,
    # close==high) per day, on a stock code absent from the baseline log.
    dates = sorted(baseline_log["매수날짜"].unique())
    ceiling_rows = pd.DataFrame(
        [
            {
                "매수날짜": d,
                "종목코드": "999999",
                "(시가)": "12900",
                "(고가)": "12900",
                "(저가)": "12000",
                "(종가)": "12900",
                "(전일종가)": "10000",
                "(시가총액, 억)": "5000",
                "(거래대금, 억)": "300",
                "(등락률)": "29.00",
                "(선정 순위)": "1",
                "(기관_순매수)": "0",
                "(외국인_순매수)": "0",
                "(프로그램_순매수)": "0",
                "(체결강도)": "120",
                "(시장구분)": "KOSPI",
                "(총 종목 수)": "6",
                "(평균 거래대금)": "250",
                "(kospi, %)": "0.3",
                "(kosdaq, %)": "0.1",
                "v_kospi": "18",
                "v_kosdaq": "20",
                "(거래량)": "100000",
                "(테마/섹터)": "반도체",
                "(차트분석)": "상한가 다음날",
                "(매수 가격)": "12900",
                "(매도 가격)": "12900",
                "(수익률, %)": "0.00",
            }
            for d in dates
        ]
    )
    augmented_log = pd.concat([baseline_log, ceiling_rows], ignore_index=True)

    # When
    bundle = train_tuned_champion_bundle(
        augmented_log, None, cfg, export_dir="tmp/spec_ceiling_pool"
    )
    prov = bundle["tuning_provenance"]["ceiling_excluded_from_pool"]

    # Then: the added ceiling rows never reach dev/control_dev -- row counts match
    # the ceiling-free baseline exactly, not the baseline plus the added rows.
    assert prov["n_dev_rows"] == prov_baseline["n_dev_rows"]
    assert prov["n_control_dev_rows"] == prov_baseline["n_control_dev_rows"]


def test_promotion_delta_invariant_to_common_cost_shift() -> None:
    import numpy as np
    import pytest

    from src.ml.champion import evaluate_promotion

    # Given: paired daily top-1 returns for candidate and control
    rng = np.random.default_rng(0)
    ctrl = rng.normal(0.004, 0.03, size=200)
    cand = ctrl + rng.normal(0.001, 0.005, size=200)

    # When: apply an identical additional round-trip cost to both legs
    extra_cost = 0.0046 - 0.0020
    base = evaluate_promotion(cand, ctrl, alpha=0.10)
    shifted = evaluate_promotion(cand - extra_cost, ctrl - extra_cost, alpha=0.10)

    # Then: delta is unchanged by a common additive shift
    assert shifted["delta"] == pytest.approx(base["delta"], abs=1e-12)


def test_champion_candidate_oof_computed_once(monkeypatch) -> None:
    import src.ml.champion as champ
    import src.ml.tuning as tuning_mod
    from src.ml.tuning import ChampionTuningConfig
    from tests.unit.ml.test_champion import _raw_trade_log

    real = champ.purged_oof_predict
    calls = {"outer": 0}

    def _spy(*a, **kw):
        if kw.get("predict_proba") is True and kw.get("n_splits") == 5:
            calls["outer"] += 1
        return real(*a, **kw)

    # evaluate_config_oof resolves purged_oof_predict in the tuning namespace,
    # so both namespaces share the spy to count global outer OOFs.
    monkeypatch.setattr(champ, "purged_oof_predict", _spy)
    monkeypatch.setattr(tuning_mod, "purged_oof_predict", _spy)
    cfg = ChampionTuningConfig(hpo_trials=2, n_splits=5, require_beats_control=False, min_history_dates=20, model_params_override={"num_leaves": 7, "n_estimators": 40})
    champ.train_tuned_champion_bundle(_raw_trade_log(n_dates=70, per_day=6), None, cfg, export_dir="tmp/spec_oof_once")

    assert calls["outer"] == 2


def test_champion_tuning_config_accepts_decision_label_modes() -> None:
    import pytest

    from src.ml.tuning import ChampionTuningConfig
    from src.ml.validation import ValidationConfig

    default = ChampionTuningConfig()
    assert default.label_mode == "journaled"
    assert default.cost_mode == "flat"
    assert default.validation is None

    cfg = ChampionTuningConfig(
        oos_reserve_start="2025-09-01",
        label_mode="mechanical",
        cost_mode="per_row",
        validation=ValidationConfig(oos_reserve_start="2025-09-01"),
    )
    assert cfg.validation is not None
    assert cfg.validation.oos_reserve_start == "2025-09-01"

    with pytest.raises(ValueError, match="label_mode"):
        ChampionTuningConfig(label_mode="operator")
    with pytest.raises(ValueError, match="cost_mode"):
        ChampionTuningConfig(cost_mode="guess")
    # A validation protocol without a matching locked window is a contradiction
    with pytest.raises(ValueError, match="oos_reserve_start"):
        ChampionTuningConfig(validation=ValidationConfig(oos_reserve_start="2025-09-01"))
    with pytest.raises(ValueError, match="oos_reserve_start"):
        ChampionTuningConfig(
            oos_reserve_start="2025-01-01",
            validation=ValidationConfig(oos_reserve_start="2025-09-01"),
        )


def test_champion_tuning_config_accepts_screen() -> None:
    import pytest

    from src.ml.tuning import ChampionTuningConfig
    from src.ml.universe import BAND_2_15_SCREEN

    default = ChampionTuningConfig()
    assert default.screen is None

    cfg = ChampionTuningConfig(screen=BAND_2_15_SCREEN)
    assert cfg.screen is BAND_2_15_SCREEN

    with pytest.raises(ValueError, match="screen"):
        ChampionTuningConfig(screen="operator_legacy")  # type: ignore[arg-type]


def _synthetic_price_history(n_dates: int, per_day: int) -> pd.DataFrame:
    """Minimal price_history matching _raw_trade_log's dates/codes for eval_net_mechanical."""
    dates = pd.bdate_range("2023-01-02", periods=n_dates + 1)
    rows = [
        {
            "date": d,
            "symbol": f"{j:06d}",
            "open": 10000.0,
            "high": 10400.0,
            "low": 9800.0,
            "close": 10200.0,
            "daily_change_pct": 0.02,
        }
        for d in dates
        for j in range(per_day)
    ]
    return pd.DataFrame(rows)


def test_champion_validation_requires_mechanical_eval_column(tmp_path) -> None:
    """Fail-closed: config.validation needs eval_net_mechanical, which requires price_history_df."""
    import pytest

    import src.ml.champion as champ
    from src.ml.tuning import ChampionTuningConfig
    from src.ml.validation import ValidationConfig

    cfg = ChampionTuningConfig(
        hpo_trials=2, seed_ensemble=(13,), require_beats_control=False, min_history_dates=10,
        model_params_override={"num_leaves": 7, "n_estimators": 10}, label_mode="journaled",
        cost_mode="flat", oos_reserve_start="2025-09-01",
        validation=ValidationConfig(oos_reserve_start="2025-09-01"),
    )
    log = _raw_trade_log(n_dates=70, per_day=6)

    # label_mode='journaled' does not require price_history_df on its own, but a
    # configured validation protocol always scores eval_net_mechanical -- without
    # price_history_df that column never exists, so this must fail closed.
    with pytest.raises(ValueError, match="eval_net_mechanical"):
        champ.train_tuned_champion_bundle(
            log, None, cfg, export_dir=str(tmp_path / "cand"), production_dir=str(tmp_path / "prod")
        )


def test_champion_validation_gate_wiring_with_mocks(monkeypatch, tmp_path) -> None:
    import src.ml.champion as champ
    from src.ml.tuning import ChampionTuningConfig
    from src.ml.validation import ValidationConfig

    monkeypatch.setattr(champ, "cpcv_path_evidence", lambda *a, **k: {"path_deltas": [0.001]*6, "n_path_deltas": 6, "top1_path_win_rate": 0.9, "ic_path_win_rate": 0.9, "pooled_delta": 0.001, "p_bootstrap": 0.01, "p_paired_t": 0.01, "ic_candidate": 0.1, "ic_control": 0.05, "ic_delta": 0.05, "mde_top1": 0.01, "mde_ic": 0.01})
    monkeypatch.setattr(champ, "evaluate_locked_oos", lambda *a, **k: {"n_days": 80, "n_rows": 100, "top1_mean": 0.005, "top1_sd": 0.02, "top1_sharpe": 1.0, "rank_ic": 0.08, "mde_top1": 0.01, "mde_ic": 0.02, "first_date": "2025-09-01", "last_date": "2026-01-01"})

    class _Fill:
        sleeve = "fillable"
        n_days = 80
        n_rows = 100
        top1_mean = 0.004
        rank_ic = 0.05

    monkeypatch.setattr(champ, "evaluate_buyability_sleeves", lambda *a, **k: (_Fill(),))
    monkeypatch.setattr(champ, "summarize_buyability_sleeves", lambda *a, **k: {"measured_share": 0.9})
    monkeypatch.setattr(champ, "fit_seed_ensemble", lambda *a, **k: type("M", (), {"predict": lambda self, X: [0.0]*len(X)})())

    from src.ml.validation import PromotionDecision, GateOutcome
    fake_decision = PromotionDecision(deployable=False, verdict="research_only", failed_gates=("oos_rank_ic_above_mde",), gates=(GateOutcome(name="oos_rank_ic_above_mde", passed=False, observed=0.01, threshold=0.02, detail={}),), evidence={"oos_reserve_start": "2025-09-01"})
    monkeypatch.setattr(champ, "run_promotion_gate", lambda *a, **k: fake_decision)
    published = {}
    def _fake_publish(bundle, cand_dir, prod_dir, decision):
        published["called"] = True
        from src.ml.bundle import save_bundle
        save_bundle(dict(bundle), str(cand_dir))
        return {"published": False}
    monkeypatch.setattr(champ, "publish_bundle", _fake_publish)

    cfg = ChampionTuningConfig(hpo_trials=2, seed_ensemble=(13,), require_beats_control=False, min_history_dates=10, model_params_override={"num_leaves": 7, "n_estimators": 10}, label_mode="journaled", cost_mode="flat", oos_reserve_start="2025-09-01", validation=ValidationConfig(oos_reserve_start="2025-09-01"), feature_selection_top_n=5)
    log = _raw_trade_log(n_dates=70, per_day=6)
    price_history = _synthetic_price_history(n_dates=70, per_day=6)
    bundle = champ.train_tuned_champion_bundle(log, None, cfg, price_history_df=price_history, export_dir=str(tmp_path / "cand"), production_dir=str(tmp_path / "prod"))
    assert published.get("called") is True
    assert bundle["promotion_decision"]["verdict"] == "research_only"
    assert bundle["label_mode"] == "journaled"


def test_measure_oos_execution_profile_reports_fill_and_adverse_selection(monkeypatch) -> None:
    """R1: with real bars, the profile is actually computed (not just skipped)."""
    import numpy as np
    import pandas as pd

    import src.ml.champion as champ

    oos_scored = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-01-02", "2026-01-05"]),
        "stock_code": ["000001", "000002"],
        "close_price": [10000.0, 20000.0],
        "pred": [0.01, 0.02],
        "eval_net_mechanical": [0.005, -0.010],
    })
    bars = pd.DataFrame({
        "symbol": ["000001", "000001", "000002", "000002"],
        "ts_hms": [152000, 152500, 152000, 152500],
        "low": [9985.0, 9990.0, 19999.0, 19998.0],  # 000001 touches a 1-tick limit; 000002 does not
        "high": [10010.0, 10005.0, 20010.0, 20005.0],
        "close": [10000.0, 10000.0, 20000.0, 20000.0],
    })
    monkeypatch.setattr(champ, "_load_normalized_bars_for_entries", lambda *a, **k: bars)

    profile = champ.measure_oos_execution_profile(oos_scored, "eval_net_mechanical")

    assert profile is not None
    assert profile["fill_rate"] == 0.5
    assert profile["fill_rate_is_upper_bound"] is True
    assert np.isfinite(profile["adverse_selection_bp"])


def test_measure_oos_execution_profile_fails_open_without_bars(monkeypatch) -> None:
    import pandas as pd

    import src.ml.champion as champ

    oos_scored = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-01-02"]),
        "stock_code": ["000001"],
        "close_price": [10000.0],
        "pred": [0.01],
        "eval_net_mechanical": [0.005],
    })
    monkeypatch.setattr(champ, "_load_normalized_bars_for_entries", lambda *a, **k: pd.DataFrame())

    assert champ.measure_oos_execution_profile(oos_scored, "eval_net_mechanical") is None
    assert champ.measure_oos_execution_profile(oos_scored.iloc[0:0], "eval_net_mechanical") is None


def test_measure_oos_execution_profile_fails_open_on_all_nat_dates() -> None:
    import numpy as np
    import pandas as pd

    import src.ml.champion as champ

    oos_scored = pd.DataFrame({
        "trade_date": pd.to_datetime([None, None]),
        "stock_code": ["000001", "000002"],
        "close_price": [10000.0, 20000.0],
        "pred": [0.01, 0.02],
        "eval_net_mechanical": [0.005, -0.010],
    })
    assert champ.measure_oos_execution_profile(oos_scored, "eval_net_mechanical") is None


def test_load_normalized_bars_for_entries_normalizes_raw_kis_partitions(tmp_path, monkeypatch) -> None:
    """Regression: read_intraday_range concatenates raw vendor partitions verbatim
    (242/243 on-disk partitions are unnormalized KIS output keyed by '종목코드',
    not 'symbol'/'ts_hms') -- simulate_passive_entry would silently never fill
    without normalizing first, exactly like the 2026-09-06 buyability.py bug."""
    import pandas as pd

    import src.ml.champion as champ
    from src import settings

    monkeypatch.setattr(settings, "HISTORY_DIR", str(tmp_path))
    partition_dir = tmp_path / "intraday" / "1m" / "regular" / "2026-01"
    partition_dir.mkdir(parents=True)
    raw = pd.DataFrame({
        "종목코드": ["000001", "000001", "000002"],
        "stck_bsop_date": ["20260102", "20260102", "20260102"],
        "stck_cntg_hour": ["152000", "152500", "152000"],
        "stck_oprc": ["10000", "9990", "20000"],
        "stck_hgpr": ["10010", "10005", "20010"],
        "stck_lwpr": ["9985", "9990", "19999"],
        "stck_prpr": ["10000", "10000", "20000"],
        "cntg_vol": ["100", "50", "200"],
        "acml_tr_pbmn": ["1000000", "1500000", "4000000"],
    })
    raw.to_parquet(partition_dir / "2026-01-02.parquet")
    # 2026-01-05: already-canonical partition (some partitions are correctly written)
    canonical = pd.DataFrame({
        "snapshot_date": ["2026-01-05"], "symbol": ["000004"], "ts_hms": [152000],
        "open": [5000.0], "high": [5010.0], "low": [4990.0], "close": [5000.0],
        "volume": [10], "value_krw": [50000.0], "has_trade": [True], "vendor": ["kis"],
    })
    canonical.to_parquet(partition_dir / "2026-01-05.parquet")
    # 2026-01-06: empty partition
    pd.DataFrame(columns=["종목코드", "stck_cntg_hour"]).to_parquet(partition_dir / "2026-01-06.parquet")
    # 2026-01-07: partition with neither 'symbol' nor '종목코드'
    pd.DataFrame({"unrelated_col": [1, 2]}).to_parquet(partition_dir / "2026-01-07.parquet")

    entries = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]),
        # 000003 is requested but absent from the 2026-01-02 partition (len(sub)==0 branch);
        # 2026-01-08 has no partition file at all (path.exists() is False branch).
        "symbol": ["000001", "000003", "000004", "000005", "000006", "000007"],
        "close_price": [10000.0, 10000.0, 5000.0, 6000.0, 7000.0, 8000.0],
    })

    bars = champ._load_normalized_bars_for_entries(entries)

    assert not bars.empty
    assert {"symbol", "ts_hms", "low", "high", "close"} <= set(bars.columns)
    assert "000001" in set(bars["symbol"].unique())
    assert "000004" in set(bars["symbol"].unique())  # survived the already-canonical branch
    assert "000003" not in set(bars["symbol"].unique())  # requested but absent that day
    assert (bars["ts_hms"] == 152000).any()


def test_load_normalized_bars_for_entries_returns_empty_when_nothing_resolves(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.ml.champion as champ
    from src import settings

    monkeypatch.setattr(settings, "HISTORY_DIR", str(tmp_path))
    entries = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-02-01"]),
        "symbol": ["000001"],
        "close_price": [10000.0],
    })

    bars = champ._load_normalized_bars_for_entries(entries)

    assert bars.empty


def test_train_tuned_champion_applies_screen_filter() -> None:
    """Regression: --screen was parsed but config.screen was never applied to dev."""
    from src.ml.tuning import ChampionTuningConfig
    from src.ml.universe import ScreenConfig

    permissive = ScreenConfig(change_lower=0.0, min_trade_value_100m=0.0, min_market_cap_100m=0.0)
    cfg = ChampionTuningConfig(
        hpo_trials=2, seed_ensemble=(13,), require_beats_control=False, min_history_dates=10,
        model_params_override={"num_leaves": 7, "n_estimators": 10}, screen=permissive,
    )
    bundle = train_tuned_champion_bundle(
        _raw_trade_log(n_dates=40, per_day=4), None, cfg, export_dir="tmp/spec_screen_filter"
    )
    assert bundle["tuning_provenance"]["screen"] == {
        "change_lower": 0.0, "change_upper": None, "min_trade_value_100m": 0.0,
        "min_market_cap_100m": 0.0, "exclude_ceiling": True, "require_index_up": False,
    }
    # A strict screen that excludes every row must not silently keep training on the old pool.
    strict = ScreenConfig(change_lower=0.99)
    cfg_strict = ChampionTuningConfig(
        hpo_trials=2, seed_ensemble=(13,), require_beats_control=False, min_history_dates=1,
        model_params_override={"num_leaves": 7, "n_estimators": 10}, screen=strict,
    )
    import pytest

    with pytest.raises(ValueError, match="unique groups"):
        train_tuned_champion_bundle(
            _raw_trade_log(n_dates=40, per_day=4), None, cfg_strict, export_dir="tmp/spec_screen_strict"
        )


def test_champion_screen_baseline_and_grid_degrade_to_skipped_on_error(monkeypatch) -> None:
    import src.ml.champion as champ
    from src.ml.tuning import ChampionTuningConfig

    def _boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(champ, "screen_baseline_stats", _boom)
    monkeypatch.setattr(champ, "evaluate_screen_grid", _boom)

    price_history = _synthetic_price_history(n_dates=40, per_day=4)
    cfg = ChampionTuningConfig(
        hpo_trials=2, seed_ensemble=(13,), require_beats_control=False, min_history_dates=10,
        model_params_override={"num_leaves": 7, "n_estimators": 10}, label_mode="journaled", cost_mode="flat",
    )
    bundle = champ.train_tuned_champion_bundle(
        _raw_trade_log(n_dates=40, per_day=4), None, cfg, price_history_df=price_history, export_dir="tmp/spec_screen_boom"
    )
    assert bundle["tuning_provenance"]["screen_baseline"] == {"status": "skipped", "reason": "boom"}
    assert bundle["tuning_provenance"]["screen_grid"] == {"status": "skipped", "reason": "boom"}


def test_champion_expected_value_policy_degrades_to_skipped_on_error(monkeypatch) -> None:
    import src.ml.champion as champ
    from src.ml.tuning import ChampionTuningConfig

    def _boom(*a, **k):
        raise ValueError("boom")

    monkeypatch.setattr(champ, "select_by_expected_value", _boom)

    cfg = ChampionTuningConfig(
        hpo_trials=2, seed_ensemble=(13,), require_beats_control=False, min_history_dates=10,
        model_params_override={"num_leaves": 7, "n_estimators": 10},
    )
    bundle = champ.train_tuned_champion_bundle(
        _raw_trade_log(n_dates=40, per_day=4), None, cfg, export_dir="tmp/spec_ev_boom"
    )
    assert bundle["tuning_provenance"]["expected_value_policy"] == {"status": "skipped", "reason": "boom"}
