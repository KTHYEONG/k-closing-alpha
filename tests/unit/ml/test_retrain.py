"""Retrain CLI contract (ranker-only, fail-closed)."""
from __future__ import annotations

import pytest

from src.ml.retrain import main


def test_retrain_rejects_unknown_feature_set(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--feature-set", "bogus"])
    assert exc.value.code != 0
    assert "feature-set" in capsys.readouterr().err


def test_retrain_raises_when_no_action_flag_given() -> None:
    import pytest

    from src.ml.retrain import main

    with pytest.raises(ValueError, match="no action flag given"):
        main([])


def test_retrain_champion_only_cli_flags_are_removed() -> None:
    from src.ml.retrain import build_arg_parser

    parser = build_arg_parser()
    dests = {action.dest for action in parser._actions}

    for gone in (
        "trade_log", "theme", "tuned", "feature_selection_top_n", "weighting_mode",
        "recency_half_life", "hpo_trials", "no_gate", "eval_mode", "hpo_objective",
        "promotion_alpha", "no_hpo", "no_restore_panel", "label_mode", "screen",
        "scenario_source", "cost_mode", "publish", "production_dir",
        "target_notional_100m", "min_ic_path_win_rate", "min_top1_path_win_rate",
        "min_oos_days",
    ):
        assert gone not in dests, gone

    for kept in (
        "export_dir", "feature_set", "oos_reserve_start", "universe_research",
        "cost_aware_backtest", "train_ranker_bundle", "ranker_topk_research",
        "ranker_train_start", "exit_grid_revalidation",
    ):
        assert kept in dests, kept


def _price_history_file(path) -> None:
    import pandas as pd

    dates = pd.bdate_range("2023-01-02", periods=20)
    pd.DataFrame({
        "date": list(dates) * 2,
        "symbol": ["000001"] * 20 + ["000002"] * 20,
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "prev_close": 100.0,
        "market_cap_100m": 900.0, "trade_value_100m": 200.0, "daily_change_pct": 0.005,
        "market": "KOSPI", "volume": 1e5, "foreign_netbuy": 0.0, "inst_netbuy": 0.0,
        "program_netbuy": 0.0, "kospi_pct": 0.001, "kosdaq_pct": 0.001,
        "v_kospi": 18.0, "v_kosdaq": 22.0,
    }).to_parquet(path)


def test_retrain_universe_research_mode_dispatches(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    seen: dict[str, object] = {}

    def _fake_grid(price_history_df, screens, **kwargs):
        seen["n_screens"] = len(screens)
        seen["rows"] = len(price_history_df)
        return pd.DataFrame([{"screen_name": "operator_legacy", "ranked_top1_net_bp": -10.0}])

    monkeypatch.setattr(mod, "run_universe_screen_grid", _fake_grid)

    main(["--universe-research", "--export-dir", str(tmp_path)])

    assert seen["n_screens"] >= 5 and seen["rows"] == 40
    assert (tmp_path / "universe_grid.parquet").exists()


def test_retrain_universe_research_missing_price_history_raises(tmp_path, monkeypatch) -> None:
    import pytest

    import src.ml.retrain as mod
    from src.ml.retrain import main

    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "nope.parquet")

    with pytest.raises(ValueError, match="price_history not found"):
        main(["--universe-research", "--export-dir", str(tmp_path)])


def test_retrain_universe_research_prepares_price_panel(tmp_path, monkeypatch) -> None:
    import numpy as np
    import pandas as pd

    import src.ml.retrain as mod
    from src.ml.retrain import main

    dates = pd.bdate_range("2023-02-01", periods=6)
    prev = 10000.0
    rows = []
    for sym in ("000001", "000002"):
        px = prev
        for d in dates:
            nxt = px * 1.08
            rows.append({
                "date": d, "symbol": sym, "open": px, "high": nxt * 1.01, "low": px * 0.99,
                "close": nxt, "prev_close": px, "market_cap_100m": 900.0,
                "trade_value_100m": 300.0, "daily_change_pct": 8.0, "market": "KOSPI",
                "volume": 1e5, "foreign_netbuy": 0.0, "inst_netbuy": 0.0, "program_netbuy": 0.0,
                "kospi_pct": 0.001, "kosdaq_pct": 0.001, "v_kospi": 18.0, "v_kosdaq": 22.0,
            })
            px = nxt
    ph_path = tmp_path / "price_history.parquet"
    pd.DataFrame(rows).to_parquet(ph_path)

    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    seen: dict[str, object] = {}

    def _fake_grid(price_history_df, screens, **kwargs):
        seen["max_abs_chg"] = float(np.nanmax(np.abs(price_history_df["daily_change_pct"].to_numpy(dtype=float))))
        seen["has_tick_cost"] = "tick_cost_bp" in price_history_df.columns
        seen["has_chg_ratio"] = "chg_ratio" in price_history_df.columns
        seen["rows"] = len(price_history_df)
        return pd.DataFrame([{"screen_name": "operator_legacy", "ranked_top1_net_bp": -10.0}])

    monkeypatch.setattr(mod, "run_universe_screen_grid", _fake_grid)

    main(["--universe-research", "--export-dir", str(tmp_path)])

    assert seen["rows"] == 12
    assert seen["has_tick_cost"] is True
    assert seen["has_chg_ratio"] is True
    assert float(seen["max_abs_chg"]) < 1.0
    assert np.isclose(float(seen["max_abs_chg"]), 0.08)


def test_retrain_cost_aware_backtest_mode_dispatches(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.ml.retrain as mod

    ph_path = tmp_path / "price_history.parquet"
    pd.DataFrame({
        "date": pd.to_datetime(["2023-02-01"]), "symbol": ["000001"], "open": [100.0], "high": [101.0],
        "low": [99.0], "close": [100.0], "prev_close": [99.0], "volume": [1000.0],
        "market_cap_100m": [900.0], "trade_value_100m": [300.0], "market": ["KOSPI"],
    }).to_parquet(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)

    seen: dict[str, object] = {}

    def _fake_run(ph, market_dates, d_to_idx, **kwargs):
        seen["called"] = True
        seen["rows"] = len(ph)
        from src.ml.costaware_topk import CostAwareTopKReport
        return CostAwareTopKReport(
            strategy_id="KCA-TOPK-COSTAWARE-001", top_k=3, universe={}, cost={}, date_min="2023-02-01",
            date_max="2023-02-01", regimes={}, cost_stress=[], verdict="INSUFFICIENT_COVERAGE", verdict_reasons=["synthetic"],
        )

    monkeypatch.setattr(mod, "run_cost_aware_topk_backtest", _fake_run)

    mod.main(["--cost-aware-backtest", "--export-dir", str(tmp_path)])

    assert seen.get("called") is True
    assert seen["rows"] == 1
    assert (tmp_path / "costaware_topk_report.parquet").exists()


def test_retrain_cost_aware_backtest_missing_price_history_raises(tmp_path, monkeypatch) -> None:
    import pytest

    import src.ml.retrain as mod

    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "nope.parquet")

    with pytest.raises(ValueError, match="price_history not found"):
        mod.main(["--cost-aware-backtest", "--export-dir", str(tmp_path)])


def test_retrain_ranker_topk_research_dispatches(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    seen: dict[str, object] = {}

    class _Ev:
        top_k = 3
        n_paths = 28
        path_win_rate = 0.893
        mean_path_delta_bp = 8.04
        pooled_delta_bp = 10.97
        p_paired_t = 0.054

    class _Report:
        verdict = "PASS_POST_REFORM"
        verdict_reasons = ["ok"]
        top_k = 3
        path_evidence = _Ev()

    def _fake_run(ph, market_dates, d_to_idx, **kwargs):
        seen["rows"] = len(ph)
        return _Report()

    def _fake_frame(report):
        seen["framed"] = report.verdict
        return pd.DataFrame([{"row_type": "path_evidence", "verdict": report.verdict}])

    monkeypatch.setattr(mod, "run_topk_ranker_backtest", _fake_run)
    monkeypatch.setattr(mod, "topk_ranker_report_to_frame", _fake_frame)

    main(["--ranker-topk-research", "--export-dir", str(tmp_path)])

    assert seen["rows"] == 40
    assert seen["framed"] == "PASS_POST_REFORM"
    assert (tmp_path / "topk_ranker_report.parquet").exists()


def test_retrain_ranker_topk_research_missing_price_history_raises(tmp_path, monkeypatch) -> None:
    import pytest

    import src.ml.retrain as mod
    from src.ml.retrain import main

    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "nope.parquet")

    with pytest.raises(ValueError, match="price_history not found"):
        main(["--ranker-topk-research", "--export-dir", str(tmp_path)])


def test_retrain_ranker_topk_research_passes_explicit_train_start(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    seen: dict[str, object] = {}

    class _Ev:
        top_k = 3
        n_paths = 28
        n_folds_total = 28
        n_folds_scored = 28
        path_win_rate = 1.0
        mean_path_delta_bp = 9.94
        pooled_delta_bp = 9.94
        p_paired_t = 0.068

    class _Report:
        verdict = "PASS_POST_REFORM"
        verdict_reasons = ["ok"]
        top_k = 3
        train_start = "2021-01-01"
        path_evidence = _Ev()

    def _fake_run(ph, market_dates, d_to_idx, **kwargs):
        seen["train_start"] = kwargs.get("train_start")
        return _Report()

    def _fake_frame(report):
        seen["framed"] = report.train_start
        return pd.DataFrame([{"row_type": "path_evidence", "verdict": report.verdict}])

    monkeypatch.setattr(mod, "run_topk_ranker_backtest", _fake_run)
    monkeypatch.setattr(mod, "topk_ranker_report_to_frame", _fake_frame)

    main(["--ranker-topk-research", "--ranker-train-start", "2021-01-01", "--export-dir", str(tmp_path)])

    assert seen["train_start"] == pd.Timestamp("2021-01-01")
    assert (tmp_path / "topk_ranker_report.parquet").exists()

    seen.clear()
    main(["--ranker-topk-research", "--export-dir", str(tmp_path)])
    assert seen["train_start"] is None


def test_retrain_train_ranker_bundle_dispatches(tmp_path, monkeypatch) -> None:
    import src.ml.retrain as mod
    from src.ml.retrain import main
    from src.ml.retrain_gate import PromotionVerdict

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    saved: list[str] = []

    def _fake_train(ph, market_dates, d_to_idx, **kwargs):
        return {"feature_cols": ["f1"], "rank_model": object(), "quantile_models": {}, "calibrators": {}, "top_k": 3}

    def _fake_save(bundle, export_dir):
        saved.append(export_dir)
        return f"{export_dir}/sizing_pipeline_bundle.joblib"

    monkeypatch.setattr(mod, "train_production_bundle", _fake_train)
    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)
    monkeypatch.setattr(mod, "load_current_bundle", lambda export_dir: None)
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda ph, market_dates, d_to_idx: "eval-frame")
    seen: dict = {}

    def _approve(candidate, current, eval_frame):
        seen["eval_frame"] = eval_frame
        seen["current"] = current
        return PromotionVerdict(promote=True, reasons=(), agreement=0.99)

    monkeypatch.setattr(mod, "evaluate_retrain_promotion", _approve)

    # When
    main(["--train-ranker-bundle", "--export-dir", str(tmp_path)])

    # Then
    assert saved == [str(tmp_path / "topk_ranker")]
    assert seen == {"eval_frame": "eval-frame", "current": None}


def test_retrain_train_ranker_bundle_rejected_by_gate_keeps_live_bundle(tmp_path, monkeypatch) -> None:
    import pytest

    import src.ml.retrain as mod
    from src.ml.retrain import main
    from src.ml.retrain_gate import PromotionVerdict

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    saved: list[str] = []

    def _fake_train(ph, market_dates, d_to_idx, **kwargs):
        return {"feature_cols": ["f1"], "rank_model": object(), "quantile_models": {}, "calibrators": {}, "top_k": 3}

    def _fake_save(bundle, export_dir):
        saved.append(export_dir)
        return f"{export_dir}/sizing_pipeline_bundle.joblib"

    monkeypatch.setattr(mod, "train_production_bundle", _fake_train)
    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)
    monkeypatch.setattr(mod, "load_current_bundle", lambda export_dir: None)
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda ph, market_dates, d_to_idx: "eval-frame")
    monkeypatch.setattr(
        mod,
        "evaluate_retrain_promotion",
        lambda candidate, current, eval_frame: PromotionVerdict(promote=False, reasons=("prediction agreement 0.810 below 0.950",), agreement=0.81),
    )

    # When / Then
    with pytest.raises(RuntimeError, match="promotion gate rejected"):
        main(["--train-ranker-bundle", "--export-dir", str(tmp_path)])

    # And: 후보는 rejected 에만 저장되고 라이브 경로는 건드리지 않는다
    assert saved == [str(tmp_path / "topk_ranker" / "rejected")]


def test_retrain_skip_promotion_gate_publishes_without_evaluating(tmp_path, monkeypatch) -> None:
    import src.ml.retrain as mod
    from src.ml.retrain import main
    from src.ml.retrain_gate import PromotionVerdict

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    saved: list[str] = []

    def _fake_train(ph, market_dates, d_to_idx, **kwargs):
        return {"feature_cols": ["f1"], "rank_model": object(), "quantile_models": {}, "calibrators": {}, "top_k": 3}

    def _fake_save(bundle, export_dir):
        saved.append(export_dir)
        return f"{export_dir}/sizing_pipeline_bundle.joblib"

    monkeypatch.setattr(mod, "train_production_bundle", _fake_train)
    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)
    monkeypatch.setattr(mod, "load_current_bundle", lambda export_dir: None)
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda ph, market_dates, d_to_idx: "eval-frame")

    def _never(*args, **kwargs):
        raise AssertionError("gate must not run with --skip-promotion-gate")

    monkeypatch.setattr(mod, "evaluate_retrain_promotion", _never)
    monkeypatch.setattr(mod, "build_gate_eval_frame", _never)

    # When
    main(["--train-ranker-bundle", "--skip-promotion-gate", "--export-dir", str(tmp_path)])

    # Then
    assert saved == [str(tmp_path / "topk_ranker")]


def test_retrain_train_ranker_bundle_missing_price_history_raises(tmp_path, monkeypatch) -> None:
    import pytest

    import src.ml.retrain as mod
    from src.ml.retrain import main

    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "nope.parquet")

    with pytest.raises(ValueError, match="price_history not found"):
        main(["--train-ranker-bundle", "--export-dir", str(tmp_path)])


def test_retrain_exit_grid_revalidation_dispatches(tmp_path, monkeypatch) -> None:
    import src.ml.research.exit_grid_revalidation as exit_grid_mod
    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    seen: dict[str, object] = {}

    def _fake_run(*, export_dir=None, **kwargs):
        seen["export_dir"] = export_dir
        return {"n_days": 44, "cost_ratio": 0.00469, "incumbent_mean_net": -0.001, "grid": [], "best": None}

    monkeypatch.setattr(exit_grid_mod, "run_exit_grid_revalidation", _fake_run)

    main(["--exit-grid-revalidation", "--export-dir", str(tmp_path)])

    assert seen["export_dir"] == str(tmp_path)


def test_retrain_exit_grid_revalidation_missing_price_history_raises(tmp_path, monkeypatch) -> None:
    import pytest

    import src.ml.retrain as mod
    from src.ml.retrain import main

    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "nope.parquet")

    with pytest.raises(ValueError, match="price_history not found"):
        main(["--exit-grid-revalidation", "--export-dir", str(tmp_path)])


def test_retrain_main_configures_logging_before_dispatch(monkeypatch) -> None:
    import logging

    import pytest

    import src.ml.retrain as mod
    from src.ml.retrain import main

    calls: list[tuple[tuple, dict]] = []

    def fake_basic_config(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(mod.logging, "basicConfig", fake_basic_config)

    with pytest.raises(ValueError, match="no action flag given"):
        main([])

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs == {"level": logging.INFO, "format": "%(message)s"}

