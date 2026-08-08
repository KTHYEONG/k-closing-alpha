"""Run the bounded causal-history feature research experiment on local Parquet data."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.feature_selection import FeatureSelectionConfig
from src.ml.history_feature_research import run_history_feature_research_experiment
from src.ml.history_features import HistoryFeatureExecutionConfig


@dataclass(frozen=True)
class ResearchRunObserver:
    """연구 실행 상태를 원자적 snapshot과 append-only event log로 기록합니다."""

    status_path: Path
    events_path: Path

    def __call__(self, stage: str, details: Mapping[str, Any]) -> None:
        self._record("running", stage, details)

    def complete(self, summary: dict[str, object]) -> None:
        self._record("completed", "completed", summary)

    def fail(self, exc: BaseException) -> None:
        self._record(
            "failed",
            "failed",
            {
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )

    def _record(self, state: str, stage: str, details: Mapping[str, Any]) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "state": state,
            "stage": stage,
            "details": details,
        }
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(self.status_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2))
        temporary.replace(self.status_path)
        with self.events_path.open("a", encoding="utf-8") as events_file:
            events_file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Return the explicit, research-only command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-log-path", type=Path, default=Path("data/parquet/trade_log.parquet"))
    parser.add_argument("--theme-path", type=Path, default=Path("data/parquet/theme.parquet"))
    parser.add_argument("--history-path", type=Path, default=Path("data/history/price_history.parquet"))
    parser.add_argument("--export-dir", type=Path, default=Path("artifacts/models/research"))
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--memory-budget-gib", type=float, default=8.0)
    parser.add_argument("--parquet-batch-rows", type=int, default=100_000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--purge-gap", type=int, default=1)
    parser.add_argument("--screening-device", choices=("cpu", "gpu", "auto"), default="cpu")
    parser.add_argument("--research-cutoff", type=str, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=("confirmation", "discovery"),
        default="confirmation",
        help="confirmation은 선형 기준선 포함 확정 실행, discovery는 후보 속도 모드.",
    )
    parser.add_argument("--wall-time-budget-seconds", type=float, default=1800.0)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.memory_budget_gib <= 0.0:
        raise ValueError("memory_budget_gib must be positive")
    if args.parquet_batch_rows < 1:
        raise ValueError("parquet_batch_rows must be positive")
    if args.n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if args.purge_gap < 0:
        raise ValueError("purge_gap must be non-negative")
    if args.wall_time_budget_seconds <= 0.0:
        raise ValueError("wall_time_budget_seconds must be positive")
    for path in (args.trade_log_path, args.history_path):
        if not path.is_file():
            raise ValueError(f"required input does not exist: {path}")
    if args.theme_path is not None and not args.theme_path.is_file():
        raise ValueError(f"theme input does not exist: {args.theme_path}")


def _summary(result: dict[str, Any]) -> dict[str, object]:
    return {
        "mode": result["contract"]["mode"],
        "evaluation_cutoff": result["contract"]["evaluation_cutoff"],
        "excluded_rows_after_cutoff": result["contract"]["excluded_rows_after_cutoff"],
        "cache_state": result["build_metrics"].get("cache_state"),
        "build_metrics": result["build_metrics"],
        "selected_feature_count": len(result["candidate"]["final_features"]),
        "control_metrics": result["control"]["metrics"],
        "candidate_metrics": result["candidate"]["metrics"],
        "identical_oof_dates": result["comparison"]["identical_oof_dates"],
        "promotion": result["promotion"],
        "candidate_bundle_path": result["candidate_bundle_path"],
    }


def run_research(args: argparse.Namespace) -> dict[str, object]:
    """Execute research through the streaming Parquet history path and summarize it."""
    status_path = args.status_path or args.export_dir / "run_status.json"
    observer = ResearchRunObserver(
        status_path=status_path,
        events_path=status_path.with_name("run_events.jsonl"),
    )
    try:
        _validate_args(args)
        observer("started", {"memory_budget_gib": args.memory_budget_gib})
        trade_log = pd.read_parquet(args.trade_log_path)
        theme = pd.read_parquet(args.theme_path) if args.theme_path is not None else None
        observer(
            "inputs_loaded",
            {"trade_log_rows": len(trade_log), "theme_rows": 0 if theme is None else len(theme)},
        )
        execution = HistoryFeatureExecutionConfig(
            memory_budget_bytes=int(args.memory_budget_gib * 1024**3),
            parquet_batch_rows=args.parquet_batch_rows,
            enforce_memory_budget=True,
        )
        result = run_history_feature_research_experiment(
            trade_log,
            theme,
            price_history_path=str(args.history_path),
            n_splits=args.n_splits,
            purge_gap=args.purge_gap,
            feature_selection_config=FeatureSelectionConfig(screening_device=args.screening_device),
            execution_config=execution,
            research_cutoff=args.research_cutoff,
            cache_dir=str(args.cache_dir) if args.cache_dir is not None else None,
            mode=args.mode,
            wall_time_budget_seconds=args.wall_time_budget_seconds,
            export_dir=str(args.export_dir),
            progress_callback=observer,
        )
        summary = _summary(result)
        observer.complete(summary)
        return summary
    except BaseException as exc:
        observer.fail(exc)
        raise


def main() -> None:
    args = build_parser().parse_args()
    sys.stdout.write(json.dumps(run_research(args), ensure_ascii=False, default=str, indent=2) + "\n")


if __name__ == "__main__":
    main()
