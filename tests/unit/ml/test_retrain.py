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


def _classification_panel_for(dates, symbols) -> None:
    """Stage an all-screenable classification panel in the isolated ALTDATA_DIR."""
    from pathlib import Path

    import pandas as pd

    from src import settings

    out = Path(settings.ALTDATA_DIR) / "security_classification.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": [d for d in dates for _ in symbols],
        "symbol": [s for _ in dates for s in symbols],
        "is_screenable": True,
    }).to_parquet(out)


def _stage_classification_all_screenable() -> None:
    import pandas as pd

    dates = pd.bdate_range("2023-01-02", periods=60)
    _classification_panel_for(dates, ["000001", "000002"])


def test_retrain_universe_research_mode_dispatches(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    seen: dict[str, object] = {}

    def _fake_grid(price_history_df, screens, **kwargs):
        seen["n_screens"] = len(screens)
        seen["rows"] = len(price_history_df)
        return pd.DataFrame([{"screen_name": "operator_legacy", "ranked_top1_net_bp": -10.0}])

    monkeypatch.setattr("src.ml.universe_research.run_universe_screen_grid", _fake_grid)

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

    monkeypatch.setattr("src.ml.universe_research.run_universe_screen_grid", _fake_grid)

    main(["--universe-research", "--export-dir", str(tmp_path)])

    assert seen["rows"] == 12
    assert seen["has_tick_cost"] is True
    assert seen["has_chg_ratio"] is True
    assert float(seen["max_abs_chg"]) < 1.0
    assert np.isclose(float(seen["max_abs_chg"]), 0.08)


def test_retrain_import_excludes_universe_research() -> None:
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[3]
    code = "import sys, src.ml.retrain; print(sorted(sys.modules))"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repo_root)},
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "src.ml.universe_research" not in proc.stdout


def test_retrain_bundle_path_never_imports_universe_research(tmp_path, monkeypatch) -> None:
    import os
    import sys

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()

    def _fake_train(ph, market_dates, d_to_idx, **kwargs):
        return {
            "strategy_id": "KCA-TOPK-COSTAWARE-001", "training_cutoff": "2026-09-18 00:00:00",
            "train_start": "2016-01-04", "feature_cols": ["f1"], "top_k": 3,
        }

    def _fake_save(bundle, export_dir):
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
        with open(path, "wb") as fh:
            fh.write(repr(sorted(bundle)).encode())
        return path

    monkeypatch.setattr(mod, "train_production_bundle", _fake_train)
    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)
    monkeypatch.delitem(sys.modules, "src.ml.universe_research", raising=False)

    main(["--train-ranker-bundle", "--skip-promotion-gate", "--export-dir", str(tmp_path)])

    assert "src.ml.universe_research" not in sys.modules


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
    _classification_panel_for(pd.to_datetime(["2023-02-01"]), ["000001"])

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
    _stage_classification_all_screenable()
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
    _stage_classification_all_screenable()
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
    _stage_classification_all_screenable()
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



def test_retrain_train_ranker_bundle_dispatches(tmp_path, monkeypatch) -> None:

    import json

    import src.ml.retrain as mod
    from src.ml.retrain import main
    from src.ml.retrain_gate import PromotionVerdict

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    monkeypatch.setenv("KCA_CODE_COMMIT", "abc123")
    saved: list[str] = []
    trained: list[dict] = []

    def _fake_train(ph, market_dates, d_to_idx, **kwargs):
        bundle = {"strategy_id": "KCA-TOPK-COSTAWARE-001", "training_cutoff": "2026-09-18 00:00:00", "train_start": "2016-01-04",
                  "feature_cols": ["f1"], "rank_model": object(), "quantile_models": {}, "calibrators": {}, "top_k": 3}
        trained.append(bundle)
        return bundle

    def _fake_save(bundle, export_dir):
        import os

        saved.append(export_dir)
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
        with open(path, "wb") as fh:
            fh.write(repr(sorted(bundle)).encode())
        return path

    monkeypatch.setattr(mod, "train_production_bundle", _fake_train)
    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)
    monkeypatch.setattr(mod, "load_current_bundle", lambda export_dir: None)
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda ph, market_dates, d_to_idx: "eval-frame")
    registry = tmp_path / "topk_ranker" / "retrain_registry.jsonl"

    seen: dict = {}

    def _approve(candidate, current, eval_frame, **kwargs):
        seen["eval_frame"] = eval_frame
        seen["current"] = current
        return PromotionVerdict(promote=True, reasons=(), agreement=0.99)

    monkeypatch.setattr(mod, "evaluate_retrain_promotion", _approve)

    # When
    main(["--train-ranker-bundle", "--export-dir", str(tmp_path)])

    # Then
    assert saved == [str(tmp_path / "topk_ranker")]
    assert seen == {"eval_frame": "eval-frame", "current": None}
    assert trained[0]["trained_at"].endswith("+09:00")
    rows = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "PROMOTED"
    assert rows[0]["agreement"] == 0.99
    assert rows[0]["reasons"] == []
    assert rows[0]["code_commit"] == "abc123"
    assert rows[0]["trained_at"] == trained[0]["trained_at"]
    assert rows[0]["bundle_path"] == str(tmp_path / "topk_ranker" / "sizing_pipeline_bundle.joblib")
    assert len(rows[0]["bundle_sha"]) == 12


def test_retrain_train_ranker_bundle_rejected_by_gate_keeps_live_bundle(tmp_path, monkeypatch) -> None:
    import pytest

    import json

    import src.ml.retrain as mod
    from src.ml.retrain import main
    from src.ml.retrain_gate import PromotionVerdict

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    monkeypatch.setenv("KCA_CODE_COMMIT", "abc123")
    saved: list[str] = []
    trained: list[dict] = []

    def _fake_train(ph, market_dates, d_to_idx, **kwargs):
        bundle = {"strategy_id": "KCA-TOPK-COSTAWARE-001", "training_cutoff": "2026-09-18 00:00:00", "train_start": "2016-01-04",
                  "feature_cols": ["f1"], "rank_model": object(), "quantile_models": {}, "calibrators": {}, "top_k": 3}
        trained.append(bundle)
        return bundle

    def _fake_save(bundle, export_dir):
        import os

        saved.append(export_dir)
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
        with open(path, "wb") as fh:
            fh.write(repr(sorted(bundle)).encode())
        return path

    monkeypatch.setattr(mod, "train_production_bundle", _fake_train)
    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)
    monkeypatch.setattr(mod, "load_current_bundle", lambda export_dir: None)
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda ph, market_dates, d_to_idx: "eval-frame")
    registry = tmp_path / "topk_ranker" / "retrain_registry.jsonl"

    monkeypatch.setattr(
        mod,
        "evaluate_retrain_promotion",
        lambda candidate, current, eval_frame, **kwargs: PromotionVerdict(promote=False, reasons=("prediction agreement 0.810 below 0.950",), agreement=0.81),
    )

    # When / Then
    with pytest.raises(RuntimeError, match="promotion gate rejected"):
        main(["--train-ranker-bundle", "--export-dir", str(tmp_path)])

    # And: 후보는 rejected 에만 저장되고, 거부 이력이 레지스트리에 남는다
    assert saved == [str(tmp_path / "topk_ranker" / "rejected")]
    rows = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["outcome"] == "REJECTED"
    assert rows[0]["agreement"] == 0.81
    assert rows[0]["reasons"] == ["prediction agreement 0.810 below 0.950"]
    assert rows[0]["bundle_path"] == str(tmp_path / "topk_ranker" / "rejected" / "sizing_pipeline_bundle.joblib")


def test_retrain_skip_promotion_gate_publishes_without_evaluating(tmp_path, monkeypatch) -> None:

    import json

    import src.ml.retrain as mod
    from src.ml.retrain import main
    from src.ml.retrain_gate import PromotionVerdict

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    monkeypatch.setenv("KCA_CODE_COMMIT", "abc123")
    saved: list[str] = []
    trained: list[dict] = []

    def _fake_train(ph, market_dates, d_to_idx, **kwargs):
        bundle = {"strategy_id": "KCA-TOPK-COSTAWARE-001", "training_cutoff": "2026-09-18 00:00:00", "train_start": "2016-01-04",
                  "feature_cols": ["f1"], "rank_model": object(), "quantile_models": {}, "calibrators": {}, "top_k": 3}
        trained.append(bundle)
        return bundle

    def _fake_save(bundle, export_dir):
        import os

        saved.append(export_dir)
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
        with open(path, "wb") as fh:
            fh.write(repr(sorted(bundle)).encode())
        return path

    monkeypatch.setattr(mod, "train_production_bundle", _fake_train)
    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)
    monkeypatch.setattr(mod, "load_current_bundle", lambda export_dir: None)
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda ph, market_dates, d_to_idx: "eval-frame")
    registry = tmp_path / "topk_ranker" / "retrain_registry.jsonl"

    def _never(*args, **kwargs):
        raise AssertionError("gate must not run with --skip-promotion-gate")

    monkeypatch.setattr(mod, "evaluate_retrain_promotion", _never)
    monkeypatch.setattr(mod, "build_gate_eval_frame", _never)

    # When
    main(["--train-ranker-bundle", "--skip-promotion-gate", "--export-dir", str(tmp_path)])

    # Then
    assert saved == [str(tmp_path / "topk_ranker")]
    rows = [json.loads(line) for line in registry.read_text(encoding="utf-8").splitlines()]
    assert [r["outcome"] for r in rows] == ["PROMOTED_UNGATED"]
    assert rows[0]["agreement"] is None


def _cutover_live_bundle(live_dir) -> dict:
    import dataclasses

    from joblib import dump

    from src.strategy.contract import COST_AWARE_UNIVERSE

    bundle = {"strategy_id": "KCA-TOPK-COSTAWARE-001", "training_cutoff": "2026-09-18 00:00:00",
              "train_start": "2016-01-04", "feature_cols": ["f1"], "rank_model": None,
              "quantile_models": {}, "calibrators": {}, "top_k": 3,
              "select_universe": dataclasses.asdict(COST_AWARE_UNIVERSE),
              "feature_contract_version": "1"}
    live_dir.mkdir(parents=True, exist_ok=True)
    dump(bundle, live_dir / "sizing_pipeline_bundle.joblib")
    return bundle


def _cutover_candidate_bundle() -> dict:
    import dataclasses

    from src.strategy.contract import PRODUCTION_STRATEGY

    return {"strategy_id": "KCA-TOPK-COSTAWARE-002", "training_cutoff": "2026-09-18 00:00:00",
            "train_start": "2016-01-04", "feature_cols": ["f1"], "rank_model": None,
            "quantile_models": {}, "calibrators": {}, "top_k": 3,
            "select_universe": dataclasses.asdict(PRODUCTION_STRATEGY.universe),
            "feature_contract_version": "1"}


def test_retrain_cutover_rejects_002_candidate_and_keeps_live_bundle(tmp_path, monkeypatch) -> None:
    import json
    import os

    import pandas as pd
    import pytest

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    monkeypatch.setenv("KCA_CODE_COMMIT", "abc123")

    live_dir = tmp_path / "topk_ranker"
    _cutover_live_bundle(live_dir)
    before = (live_dir / "sizing_pipeline_bundle.joblib").read_bytes()

    candidate = _cutover_candidate_bundle()
    monkeypatch.setattr(mod, "train_production_bundle", lambda ph, market_dates, d_to_idx, **kw: dict(candidate))
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-09-01"] * 5), "f1": [1.0, 2.0, 3.0, 4.0, 5.0]})
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda ph, market_dates, d_to_idx: frame)

    saved: list[str] = []

    def _fake_save(bundle, export_dir):
        saved.append(export_dir)
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
        with open(path, "wb") as fh:
            fh.write(repr(sorted(bundle)).encode())
        return path

    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)

    with pytest.raises(RuntimeError, match="promotion gate rejected"):
        main(["--train-ranker-bundle", "--export-dir", str(tmp_path)])

    assert (live_dir / "sizing_pipeline_bundle.joblib").read_bytes() == before
    assert saved == [str(live_dir / "rejected")]
    rows = [json.loads(line) for line in (live_dir / "retrain_registry.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["outcome"] for r in rows] == ["REJECTED"]
    assert any("strategy or screen changed" in r for r in rows[0]["reasons"])


def test_retrain_cutover_manual_promotion_publishes_002_bundle(tmp_path, monkeypatch) -> None:
    import json

    from joblib import load

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    monkeypatch.setenv("KCA_CODE_COMMIT", "abc123")

    live_dir = tmp_path / "topk_ranker"
    _cutover_live_bundle(live_dir)

    candidate = _cutover_candidate_bundle()
    monkeypatch.setattr(mod, "train_production_bundle", lambda ph, market_dates, d_to_idx, **kw: dict(candidate))

    def _never(*args, **kwargs):
        raise AssertionError("gate must not run with --skip-promotion-gate")

    monkeypatch.setattr(mod, "evaluate_retrain_promotion", _never)
    monkeypatch.setattr(mod, "build_gate_eval_frame", _never)

    main(["--train-ranker-bundle", "--skip-promotion-gate", "--export-dir", str(tmp_path)])

    rows = [json.loads(line) for line in (live_dir / "retrain_registry.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["outcome"] for r in rows] == ["PROMOTED_UNGATED"]
    assert rows[0]["agreement"] is None
    assert load(live_dir / "sizing_pipeline_bundle.joblib")["strategy_id"] == "KCA-TOPK-COSTAWARE-002"


def test_retrain_pit_flags_defaults() -> None:
    from src import settings
    from src.ml.retrain import build_arg_parser

    args = build_arg_parser().parse_args([])

    assert args.pit_certification is False
    assert args.pit_augment is False
    assert args.pit_gate_mode == "advisory"
    assert args.pit_panel_dir == str(settings.HISTORY_DIR)


def test_retrain_pit_certification_dispatches(tmp_path, monkeypatch) -> None:
    import src.ml.research.pit_certification as pit_mod
    import src.ml.retrain as mod
    from src.ml.retrain import main

    seen: dict = {}

    def _fake_main(*, export_dir, panel_dir, augment, train_start):
        seen.update(export_dir=export_dir, panel_dir=panel_dir, augment=augment, train_start=train_start)
        from src.ml.pit_report import PitHaircutReport

        return PitHaircutReport()

    monkeypatch.setattr(pit_mod, "main_pit_certification", _fake_main)

    def _never(*args, **kwargs):
        raise AssertionError("no bundle training may happen for --pit-certification")

    monkeypatch.setattr(mod, "train_production_bundle", _never)

    main(["--pit-certification", "--pit-augment", "--export-dir", str(tmp_path),
          "--pit-panel-dir", str(tmp_path / "panel"), "--ranker-train-start", "2023-01-25"])

    assert seen["augment"] is True
    assert seen["export_dir"] == str(tmp_path)
    assert str(seen["panel_dir"]) == str(tmp_path / "panel")
    assert seen["train_start"] is not None


class _PitColumnModel:
    def predict(self, features):
        import numpy as np

        return features["f1"].to_numpy(dtype=np.float64)


def _pit_live_bundle(**overrides) -> dict:
    import dataclasses

    from src.strategy.contract import PRODUCTION_STRATEGY

    bundle = {
        "strategy_id": "KCA-TOPK-COSTAWARE-002",
        "training_cutoff": "2026-09-18 00:00:00",
        "train_start": "2023-01-25",
        "feature_cols": ["f1", "f2"],
        "rank_model": object(),
        "quantile_models": {},
        "calibrators": {},
        "top_k": 3,
        "select_universe": dataclasses.asdict(PRODUCTION_STRATEGY.universe),
        "feature_contract_version": "1",
        "model_params": {"n_estimators": 10},
        "seeds": [1],
        "return_model": _PitColumnModel(),
    }
    bundle.update(overrides)
    return bundle


def _write_pit_report(live_dir, native_mean=5.0, native_ic=0.02) -> None:
    import dataclasses
    from datetime import datetime
    from pathlib import Path
    from zoneinfo import ZoneInfo

    import pandas as pd

    from src.ml.pit_report import PairedDelta, PitHaircutReport, PitReportStatus, save_pit_haircut_report
    from src.strategy.contract import PRODUCTION_STRATEGY

    report = PitHaircutReport(
        generated_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        strategy_id="KCA-TOPK-COSTAWARE-002",
        strategy_fingerprint="fp",
        top_k=3,
        select_universe=dataclasses.asdict(PRODUCTION_STRATEGY.universe),
        feature_contract_version="1",
        model_params={"n_estimators": 10},
        seeds=(1,),
        status=PitReportStatus.OK,
        panel_date_min="2026-06-01",
        panel_date_max="2026-09-01",
        n_usable_days=60,
        n_paired_days=60,
        n_live_days=60,
        n_eod_index_days=0,
        mean_net_bp={"eod_full": 8.0, "eod_matched": 8.0, "pit_feature": 6.0, "pit_native": native_mean},
        haircut=PairedDelta(delta=8.0 - native_mean, ci_low=0.0, ci_high=5.0, p_value=0.1, n_days=60),
        coverage_component=PairedDelta(delta=0.0, ci_low=0.0, ci_high=0.0, p_value=1.0, n_days=60),
        feature_component=PairedDelta(delta=1.0, ci_low=0.0, ci_high=2.0, p_value=0.2, n_days=60),
        selection_component=PairedDelta(delta=1.0, ci_low=0.0, ci_high=2.0, p_value=0.2, n_days=60),
        pit_native_vs_zero=PairedDelta(delta=native_mean, ci_low=0.0, ci_high=9.0, p_value=0.01, n_days=60),
        rank_ic_mean={"eod_full": 0.03, "eod_matched": 0.03, "pit_feature": 0.02, "pit_native": native_ic},
        rank_ic_haircut=PairedDelta(delta=0.01, ci_low=0.0, ci_high=0.02, p_value=0.2, n_days=60),
        pick_overlap_mean={"pit_feature": 0.9, "pit_native": 0.8},
        haircut_by_index_basis={"live_1520": 2.0, "eod_fallback": float("nan")},
        augmentation=None,
    )
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-01"]), "arm": ["eod_full"], "topk_net_bp": [8.0],
        "rank_ic": [0.03], "n_pool": [20], "overlap_vs_eod": [1.0], "index_basis": ["live_1520"],
    })
    save_pit_haircut_report(report, daily, out_dir=Path(live_dir))


def _pit_eval_frame():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    dates = np.repeat(pd.bdate_range("2026-08-17", periods=20).to_numpy(), 10)
    return pd.DataFrame({"date": dates, "f1": rng.normal(size=200), "f2": rng.normal(size=200)})


def test_retrain_train_bundle_stamps_pit_metadata(tmp_path, monkeypatch) -> None:
    import os

    from joblib import load

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    monkeypatch.setenv("KCA_CODE_COMMIT", "abc123")
    live_dir = tmp_path / "topk_ranker"
    live_dir.mkdir(parents=True, exist_ok=True)
    _write_pit_report(live_dir)

    candidate = _pit_live_bundle()
    monkeypatch.setattr(mod, "train_production_bundle", lambda ph, market_dates, d_to_idx, **kw: dict(candidate))
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda ph, market_dates, d_to_idx: _pit_eval_frame())

    saved: list[str] = []

    def _fake_save(bundle, export_dir):
        saved.append(export_dir)
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
        from joblib import dump

        dump(bundle, path)
        return path

    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)

    main(["--train-ranker-bundle", "--export-dir", str(tmp_path)])

    stored = load(live_dir / "sizing_pipeline_bundle.joblib")
    assert stored["pit_certification"]["gate_mode"] == "advisory"
    assert stored["pit_certification"]["gate_status"] == "PASS"
    assert stored["pit_certification"]["status"] == "OK"
    assert saved == [str(live_dir)]


def test_retrain_enforce_pit_gate_rejects_and_stamps(tmp_path, monkeypatch) -> None:
    import os

    import pytest
    from joblib import load

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    monkeypatch.setenv("KCA_CODE_COMMIT", "abc123")
    live_dir = tmp_path / "topk_ranker"
    live_dir.mkdir(parents=True, exist_ok=True)
    _write_pit_report(live_dir, native_mean=-5.0, native_ic=-0.01)

    candidate = _pit_live_bundle()
    monkeypatch.setattr(mod, "train_production_bundle", lambda ph, market_dates, d_to_idx, **kw: dict(candidate))
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda ph, market_dates, d_to_idx: _pit_eval_frame())

    def _fake_save(bundle, export_dir):
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
        from joblib import dump

        dump(bundle, path)
        return path

    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)

    with pytest.raises(RuntimeError, match="promotion gate rejected"):
        main(["--train-ranker-bundle", "--pit-gate-mode", "enforce", "--export-dir", str(tmp_path)])

    rejected = load(live_dir / "rejected" / "sizing_pipeline_bundle.joblib")
    assert rejected["pit_certification"]["gate_status"] == "FAIL"


def test_retrain_skip_gate_records_ungated_pit_status(tmp_path, monkeypatch) -> None:
    import os

    from joblib import load

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    monkeypatch.setenv("KCA_CODE_COMMIT", "abc123")

    candidate = _pit_live_bundle()
    monkeypatch.setattr(mod, "train_production_bundle", lambda ph, market_dates, d_to_idx, **kw: dict(candidate))

    def _never(*args, **kwargs):
        raise AssertionError("gate must not run with --skip-promotion-gate")

    monkeypatch.setattr(mod, "evaluate_retrain_promotion", _never)
    monkeypatch.setattr(mod, "build_gate_eval_frame", _never)

    def _fake_save(bundle, export_dir):
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
        from joblib import dump

        dump(bundle, path)
        return path

    monkeypatch.setattr(mod, "save_production_bundle", _fake_save)

    main(["--train-ranker-bundle", "--skip-promotion-gate", "--export-dir", str(tmp_path)])

    stored = load(tmp_path / "topk_ranker" / "sizing_pipeline_bundle.joblib")
    assert stored["pit_certification"]["gate_status"] == "UNGATED"


def test_retrain_corrupt_pit_report_evaluates_as_missing_without_crashing(tmp_path, monkeypatch, caplog) -> None:
    import logging

    import pytest

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    live_dir = tmp_path / "topk_ranker"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "pit_haircut_report.json").write_text('{"schema_version": 1,', encoding="utf-8")
    seen: dict = {}

    class _StopError(Exception):
        pass

    def _capture(bundle, current, eval_frame, *, pit_report, pit_gate, **kw):
        seen["pit_report"] = pit_report
        seen["mode"] = pit_gate.mode
        raise _StopError

    monkeypatch.setattr(mod, "train_production_bundle", lambda ph, market_dates, d_to_idx, **kw: {"a": 1})
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda *a, **k: None)
    monkeypatch.setattr(mod, "evaluate_retrain_promotion", _capture)

    for mode in ("advisory", "enforce"):
        with caplog.at_level(logging.WARNING, logger=mod.logger.name), pytest.raises(_StopError):
            main(["--train-ranker-bundle", "--export-dir", str(tmp_path), "--pit-gate-mode", mode])
        assert seen["pit_report"] is None
        assert seen["mode"].value == mode
        assert any("status=REPORT_UNREADABLE" in r.getMessage() for r in caplog.records)
        caplog.clear()


def test_retrain_train_bundle_scores_adopted_arm2_report(tmp_path, monkeypatch) -> None:
    import pandas as pd

    import src.ml.retrain as mod
    from src.ml.retrain import main

    ph_path = tmp_path / "price_history.parquet"
    _price_history_file(ph_path)
    monkeypatch.setattr(mod.settings, "PRICE_HISTORY_PARQUET_PATH", ph_path)
    _stage_classification_all_screenable()
    live_dir = tmp_path / "topk_ranker"
    live_dir.mkdir(parents=True, exist_ok=True)
    _write_pit_report(live_dir)
    seen: dict = {}

    class _StopError(Exception):
        pass

    def _capture(bundle, current, eval_frame, *, pit_report, pit_gate, **kw):
        seen.update(pit_report=pit_report, recon_adopted=kw["pit_recon_adopted"],
                    recon_detail=kw["pit_recon_detail"], recon_report=kw["pit_recon_report"])
        raise _StopError

    monkeypatch.setattr(mod, "train_production_bundle", lambda ph, market_dates, d_to_idx, **kw: {"a": 1})
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda *a, **k: None)
    monkeypatch.setattr(mod, "evaluate_retrain_promotion", _capture)

    import hashlib
    from datetime import datetime
    from zoneinfo import ZoneInfo

    import pytest

    from src.ml.pit_report import (
        RECON_ARM_DIRNAME,
        CalibrationStability,
        CertificationBindings,
        PairedDelta,
        ReconstructionCertification,
        save_reconstruction_certification,
    )

    config_path = tmp_path / "nxt_decomposition_config.json"
    config_path.write_bytes(b'{"ewma_alpha": 0.5}')
    monkeypatch.setattr(
        "src.data.pit1520_panel.default_decomposition_config_path", lambda: config_path)
    arm_dir = live_dir / RECON_ARM_DIRNAME
    arm_dir.mkdir(exist_ok=True)
    _write_pit_report(arm_dir, native_mean=9.0)
    arm2_sha = hashlib.sha256((arm_dir / "pit_haircut_report.json").read_bytes()).hexdigest()
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    cert = ReconstructionCertification(
        generated_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        exact_dir="exact",
        recon_dir="recon",
        paired_days=("2026-03-02",),
        dropped_days=(),
        coverage_improvement=PairedDelta(delta=4.0, ci_low=2.0, ci_high=6.0, p_value=0.001, n_days=60),
        reconstruction_feature=PairedDelta(delta=0.2, ci_low=-1.0, ci_high=1.4, p_value=0.6, n_days=60),
        stability=CalibrationStability(
            passed=True, rel_err_p90=0.10, coverage=0.80, bias_drift=0.01,
            alpha_at_boundary=False, n_holdout_rows=40, detail="",
        ),
        coverage_by_year_and_basis={},
        bindings=CertificationBindings(
            decomposition_config_sha256=config_sha,
            calibration_table_sha256="b" * 64,
            exact_report_sha256="c" * 64,
            recon_report_sha256=arm2_sha,
        ),
        fidelity={
            "chg_rank_correlation": 0.99, "tv_rank_correlation": 0.98, "top_k_overlap": 0.97,
            "n_paired_days": 60.0, "n_symbol_days": 600.0, "n_no_share": 2.0,
        },
        holdout_start="2026-08-25",
        holdout_end="2026-08-31",
        gate_config={"feature_margin_bp": 5.0},
        gate_verdict="ADOPT",
        gate_reasons=(),
    )
    save_reconstruction_certification(cert, out_path=live_dir / "reconstruction_certification.json")

    with pytest.raises(_StopError):
        main(["--train-ranker-bundle", "--export-dir", str(tmp_path)])

    assert seen["recon_adopted"] is True
    assert seen["recon_report"] is not None
    assert float(seen["recon_report"].mean_net_bp["pit_native"]) == 9.0
    assert seen["pit_report"] is not None

    import dataclasses
    from pathlib import Path

    from joblib import dump, load

    from src.ml.retrain_gate import evaluate_retrain_promotion

    monkeypatch.setattr(mod, "evaluate_retrain_promotion", evaluate_retrain_promotion)
    monkeypatch.setattr(mod, "train_production_bundle", lambda *args, **kwargs: _pit_live_bundle())
    monkeypatch.setattr(mod, "build_gate_eval_frame", lambda *args, **kwargs: _pit_eval_frame())
    monkeypatch.setenv("KCA_CODE_COMMIT", "audit-regression")

    def save_bundle(bundle, export_dir):
        target = Path(export_dir) / "sizing_pipeline_bundle.joblib"
        target.parent.mkdir(parents=True, exist_ok=True)
        dump(bundle, target)
        return str(target)

    monkeypatch.setattr(mod, "save_production_bundle", save_bundle)
    for cert_verdict, expected_arm, expected_mean in (("ADOPT", "arm2", 9.0), ("REJECT", "arm1", 5.0)):
        save_reconstruction_certification(dataclasses.replace(cert, gate_verdict=cert_verdict),
                                         out_path=live_dir / "reconstruction_certification.json")
        main(["--train-ranker-bundle", "--export-dir", str(tmp_path)])
        metadata = load(live_dir / "sizing_pipeline_bundle.joblib")["pit_certification"]
        assert metadata["arm"] == expected_arm
        assert metadata["mean_net_bp"]["pit_native"] == expected_mean
        assert metadata["gate_status"] == "PASS"
