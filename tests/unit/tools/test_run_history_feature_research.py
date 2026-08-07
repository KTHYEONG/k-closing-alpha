from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.ml.feature_selection import FeatureSelectionConfig
from src.ml.history_features import HistoryFeatureExecutionConfig
from src.tools import run_history_feature_research as runner


def test_scenario_history_feature_pipeline_readiness_01_defaults_are_bounded() -> None:
    """SCENARIO_HISTORY_FEATURE_PIPELINE_READINESS_01: safe defaults are explicit."""
    args = runner.build_parser().parse_args([])
    assert args.memory_budget_gib == 8.0
    assert args.parquet_batch_rows == 100_000
    assert args.screening_device == "cpu"


def test_scenario_history_feature_pipeline_readiness_02_wires_parquet_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO_HISTORY_FEATURE_PIPELINE_READINESS_02 and SCENARIO_HISTORY_FEATURE_RESEARCH_OBSERVABILITY_01."""
    trade_path = tmp_path / "trade.parquet"
    history_path = tmp_path / "history.parquet"
    theme_path = tmp_path / "theme.parquet"
    for path in (trade_path, history_path, theme_path):
        path.touch()
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.DataFrame({"x": [str(path)]}))
    captured: dict[str, object] = {}

    def fake_experiment(*args: object, **kwargs: object) -> dict[str, object]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        progress = kwargs["progress_callback"]
        assert callable(progress)
        progress("dataset_built", {"processed_rows": 10})
        return {
            "build_metrics": {"batch_count": 2},
            "candidate": {"final_features": ["f1", "f2"], "metrics": {"mean": 0.2}},
            "control": {"metrics": {"mean": 0.1}},
            "comparison": {"identical_oof_dates": True},
            "candidate_bundle_path": "research/model.joblib",
        }

    monkeypatch.setattr(runner, "run_history_feature_research_experiment", fake_experiment)
    args = argparse.Namespace(
        trade_log_path=trade_path,
        history_path=history_path,
        theme_path=theme_path,
        export_dir=tmp_path / "research",
        status_path=None,
        memory_budget_gib=8.0,
        parquet_batch_rows=100_000,
        n_splits=5,
        purge_gap=1,
        screening_device="cpu",
    )

    summary = runner.run_research(args)

    assert captured["kwargs"] == {
        "price_history_path": str(history_path),
        "n_splits": 5,
        "purge_gap": 1,
        "feature_selection_config": FeatureSelectionConfig(screening_device="cpu"),
        "execution_config": HistoryFeatureExecutionConfig(
            memory_budget_bytes=8 * 1024**3,
            parquet_batch_rows=100_000,
            enforce_memory_budget=True,
        ),
        "export_dir": str(tmp_path / "research"),
        "progress_callback": captured["kwargs"]["progress_callback"],
    }
    assert summary["selected_feature_count"] == 2
    assert summary["identical_oof_dates"] is True
    status = json.loads((tmp_path / "research" / "run_status.json").read_text())
    assert status["state"] == "completed"
    events = (tmp_path / "research" / "run_events.jsonl").read_text().splitlines()
    assert any(json.loads(line)["stage"] == "dataset_built" for line in events)


def test_scenario_history_feature_research_observability_02_records_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SCENARIO_HISTORY_FEATURE_RESEARCH_OBSERVABILITY_02: errors survive terminal loss."""
    for name in ("trade.parquet", "history.parquet", "theme.parquet"):
        (tmp_path / name).touch()
    monkeypatch.setattr(pd, "read_parquet", lambda _path: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(
        runner,
        "run_history_feature_research_experiment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fold failure")),
    )
    args = argparse.Namespace(
        trade_log_path=tmp_path / "trade.parquet",
        history_path=tmp_path / "history.parquet",
        theme_path=tmp_path / "theme.parquet",
        export_dir=tmp_path / "research",
        status_path=None,
        memory_budget_gib=8.0,
        parquet_batch_rows=100_000,
        n_splits=5,
        purge_gap=1,
        screening_device="cpu",
    )

    with pytest.raises(RuntimeError, match="fold failure"):
        runner.run_research(args)

    status = json.loads((tmp_path / "research" / "run_status.json").read_text())
    assert status["state"] == "failed"
    assert status["details"]["exception_type"] == "RuntimeError"
