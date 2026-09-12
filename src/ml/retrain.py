"""CLI entrypoint for cost-aware top-k ranker research and production bundle training."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet
from src.data.panel_integrity import load_price_panel
from src.ml.costaware_topk import report_to_frame, run_cost_aware_topk_backtest
from src.ml.topk_ranker_research import (
    run_topk_ranker_backtest,
    save_production_bundle,
    topk_ranker_report_to_frame,
    train_production_bundle,
)
from src.ml.universe_research import DEFAULT_RESEARCH_SCREENS, run_universe_screen_grid

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the retrain CLI parser (unit-testable contract)."""
    parser = argparse.ArgumentParser(description="Cost-aware top-k ranker research and bundle training")
    parser.add_argument("--export-dir", default="artifacts/models")
    parser.add_argument("--feature-set", default="close_morning61", choices=["close_morning61", "close_morning_history", "close_morning_sector"])
    parser.add_argument("--oos-reserve-start", default=None)
    parser.add_argument("--universe-research", action="store_true", help="reconstruct full-market panels for a ScreenConfig family, train the ranker on each, print/save the model-free-vs-ranked-vs-CPCV comparison")
    parser.add_argument("--cost-aware-backtest", action="store_true", help="run the model-free COST_AWARE top-k regime-gated backtest against full price_history")
    parser.add_argument("--train-ranker-bundle", action="store_true", help="train and persist the certified top-3 cost-aware production bundle (build_inline_bundle on the certification-regime population via train_production_bundle)")
    parser.add_argument("--ranker-topk-research", action="store_true", help="train the ranker on the wide screen pool, select top-k from the cost-capped pool, and score it against the model-free cost-sort control on the post-reform regime")
    parser.add_argument("--ranker-train-start", default=None, help="widen the ranker training window to this YYYY-MM-DD start; augments training only and never moves the certification boundary (default: the certification regime start)")
    parser.add_argument("--exit-grid-revalidation", action="store_true", help="re-validate the TP5%%+MOC next-day exit-timing lever (src/ml/exit_policy.py) under the certified ranker's own CPCV(8,2) OOF pipeline and real PIT cost, without touching run_topk_ranker_backtest itself")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse retrain arguments and dispatch to the ranker research pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.universe_research:
        if not os.path.exists(settings.PRICE_HISTORY_PARQUET_PATH):
            raise ValueError(f"price_history not found: {settings.PRICE_HISTORY_PARQUET_PATH}")
        price_history_df, panel_prov = load_price_panel(settings.PRICE_HISTORY_PARQUET_PATH)
        logger.info("[DATA] stage=panel_integrity %s", panel_prov.to_log_kv())
        universe_grid = run_universe_screen_grid(
            price_history_df,
            DEFAULT_RESEARCH_SCREENS,
            feature_set=args.feature_set,
            start_date=str(pd.to_datetime(price_history_df["date"]).min().date()),
            end_date=str(pd.to_datetime(price_history_df["date"]).max().date()),
            oos_reserve_start=args.oos_reserve_start,
            model_params=None,
        )
        atomic_write_parquet(universe_grid, Path(args.export_dir) / "universe_grid.parquet")
        logger.info(universe_grid.to_string())
        return

    if args.cost_aware_backtest:
        from src.ml.research.v3_engine import load_and_prepare_price_history

        if not os.path.exists(settings.PRICE_HISTORY_PARQUET_PATH):
            raise ValueError(f"price_history not found: {settings.PRICE_HISTORY_PARQUET_PATH}")
        ph, market_dates, d_to_idx = load_and_prepare_price_history(settings.PRICE_HISTORY_PARQUET_PATH)
        report = run_cost_aware_topk_backtest(ph, market_dates, d_to_idx)
        atomic_write_parquet(report_to_frame(report), Path(args.export_dir) / "costaware_topk_report.parquet")
        logger.info("[EVAL] stage=costaware_topk verdict=%s reasons=%s", report.verdict, report.verdict_reasons)
        return

    if args.train_ranker_bundle:
        from src.ml.research.v3_engine import load_and_prepare_price_history

        if not os.path.exists(settings.PRICE_HISTORY_PARQUET_PATH):
            raise ValueError(f"price_history not found: {settings.PRICE_HISTORY_PARQUET_PATH}")
        ph, market_dates, d_to_idx = load_and_prepare_price_history(settings.PRICE_HISTORY_PARQUET_PATH)
        bundle = train_production_bundle(ph, market_dates, d_to_idx)
        path = save_production_bundle(bundle, export_dir=os.path.join(args.export_dir, "topk_ranker"))
        logger.info("[EVAL] stage=train_ranker_bundle path=%s top_k=%s train_start=%s", path, bundle.get("top_k"), bundle.get("train_start"))
        return

    if args.ranker_topk_research:
        from src.ml.research.v3_engine import load_and_prepare_price_history

        if not os.path.exists(settings.PRICE_HISTORY_PARQUET_PATH):
            raise ValueError(f"price_history not found: {settings.PRICE_HISTORY_PARQUET_PATH}")
        ph, market_dates, d_to_idx = load_and_prepare_price_history(settings.PRICE_HISTORY_PARQUET_PATH)
        train_start = pd.Timestamp(args.ranker_train_start) if args.ranker_train_start else None
        report = run_topk_ranker_backtest(ph, market_dates, d_to_idx, train_start=train_start)
        atomic_write_parquet(topk_ranker_report_to_frame(report), Path(args.export_dir) / "topk_ranker_report.parquet")
        logger.info("[EVAL] stage=topk_ranker verdict=%s reasons=%s", report.verdict, report.verdict_reasons)
        return

    if args.exit_grid_revalidation:
        if not os.path.exists(settings.PRICE_HISTORY_PARQUET_PATH):
            raise ValueError(f"price_history not found: {settings.PRICE_HISTORY_PARQUET_PATH}")
        from src.ml.research.exit_grid_revalidation import run_exit_grid_revalidation

        summary = run_exit_grid_revalidation(export_dir=args.export_dir)
        promoted = summary["best"] is not None
        logger.info(
            "[EVAL] stage=exit_grid_revalidation promoted=%s n_days=%s cost_ratio=%s best=%s",
            promoted, summary["n_days"], summary["cost_ratio"], summary["best"],
        )
        return

    raise ValueError("no action flag given; choose one of --universe-research/--cost-aware-backtest/--train-ranker-bundle/--ranker-topk-research/--exit-grid-revalidation")


if __name__ == "__main__":
    main()
