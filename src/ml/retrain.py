"""CLI entrypoint for offline champion bundle retraining."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from src import settings
from src.data.candidate_panel import build_restored_trade_log, check_price_history_freshness
from src.data.io_utils import atomic_write_parquet
from src.ml.bundle import CHAMPION_DEFAULT_MODEL_PARAMS
from src.ml.champion import train_champion_bundle, train_tuned_champion_bundle
from src.ml.tuning import ChampionTuningConfig
from src.ml.universe import SCREEN_REGISTRY
from src.ml.universe_research import DEFAULT_RESEARCH_SCREENS, run_universe_screen_grid
from src.ml.validation import ValidationConfig

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the retrain CLI parser (unit-testable contract)."""
    parser = argparse.ArgumentParser(description="Champion bundle retraining")
    parser.add_argument("--trade-log", default=str(settings.TRADE_LOG_PARQUET_PATH))
    parser.add_argument("--theme", default=str(settings.THEME_PARQUET_PATH))
    parser.add_argument("--export-dir", default="artifacts/models")
    parser.add_argument("--tuned", action="store_true")
    parser.add_argument("--feature-set", default="close_morning61", choices=["close_morning61", "close_morning_history", "close_morning_sector"])
    parser.add_argument("--feature-selection-top-n", type=int, default=None)
    parser.add_argument("--oos-reserve-start", default=None)
    parser.add_argument("--weighting-mode", default="current")
    parser.add_argument("--recency-half-life", default=None)
    parser.add_argument("--hpo-trials", type=int, default=40)
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--eval-mode", default="walkforward", choices=["walkforward", "cpcv"])
    parser.add_argument("--hpo-objective", default="rank_ic", choices=["rank_ic", "top1_return", "cpcv_top1"])
    parser.add_argument("--promotion-alpha", type=float, default=0.10)
    parser.add_argument("--no-hpo", action="store_true", help="skip Optuna; use CHAMPION_DEFAULT_MODEL_PARAMS")
    parser.add_argument("--no-restore-panel", action="store_true", help="train on the raw trade log only; skip condition_history/archive panel restoration")
    parser.add_argument("--universe-research", action="store_true", help="reconstruct full-market panels for a ScreenConfig family, train the ranker on each, print/save the model-free-vs-ranked-vs-CPCV comparison; skips champion training")
    parser.add_argument("--label-mode", default="mechanical", choices=["journaled", "mechanical"])
    parser.add_argument("--screen", default="operator_legacy", choices=["operator_legacy", "band_2_15", "band_5_15_highvalue"])
    parser.add_argument("--scenario-source", default="manual", choices=["manual", "auto", "none"], help="manual: keep the journaled 차트분석; auto: derive it from price_history; none: drop the scenario feature block")
    parser.add_argument("--cost-mode", default="per_row", choices=["flat", "per_row"])
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--production-dir", default="artifacts/models")
    parser.add_argument("--target-notional-100m", type=float, default=0.5)
    parser.add_argument("--min-ic-path-win-rate", type=float, default=0.75)
    parser.add_argument("--min-top1-path-win-rate", type=float, default=0.60)
    parser.add_argument("--min-oos-days", type=int, default=60)
    _orig_parse = parser.parse_args

    def _guarded_parse(args: list[str] | None = None, namespace: argparse.Namespace | None = None) -> argparse.Namespace:
        parsed = _orig_parse(args, namespace) if namespace is not None else _orig_parse(args)
        if bool(getattr(parsed, "publish", False)) and not getattr(parsed, "oos_reserve_start", None):
            parser.error("--publish requires --oos-reserve-start")
        # scenario_source=auto needs price_history, which is not loaded when the
        # panel restore is skipped for a non-history feature_set.
        if (
            getattr(parsed, "scenario_source", "manual") == "auto"
            and bool(getattr(parsed, "no_restore_panel", False))
            and getattr(parsed, "feature_set", "") not in ("close_morning_history", "close_morning_sector")
        ):
            parser.error("--scenario-source auto requires price_history (drop --no-restore-panel)")
        return parsed

    parser.parse_args = _guarded_parse  # type: ignore[method-assign]
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse retrain arguments and dispatch to the champion training pipeline."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    # promotion_alpha_wiring: forward CLI flag to ValidationConfig gate.
    validation = ValidationConfig(oos_reserve_start=args.oos_reserve_start, min_ic_path_win_rate=args.min_ic_path_win_rate, min_top1_path_win_rate=args.min_top1_path_win_rate, min_oos_days=args.min_oos_days, target_notional_100m=args.target_notional_100m, promotion_alpha=args.promotion_alpha) if args.oos_reserve_start else None

    trade_log_df = pd.read_parquet(args.trade_log)
    theme_df = pd.read_parquet(args.theme) if os.path.exists(args.theme) else None
    price_history_df = pd.read_parquet(settings.PRICE_HISTORY_PARQUET_PATH) if (args.feature_set in ("close_morning_history", "close_morning_sector") or not args.no_restore_panel) and os.path.exists(settings.PRICE_HISTORY_PARQUET_PATH) else None
    if not args.no_restore_panel and price_history_df is None:
        logger.warning(
            "[DATA] stage=panel_restore status=skipped reason=price_history_missing path=%s",
            settings.PRICE_HISTORY_PARQUET_PATH,
        )
    elif not args.no_restore_panel:
        freshness = check_price_history_freshness(price_history_df)
        if freshness.get("is_stale"):
            logger.warning(
                "[DATA] stage=ml_panel_freshness status=stale max_date=%s staleness_days=%s threshold_days=%s",
                freshness.get("max_date"),
                freshness.get("staleness_days"),
                5,
            )
        trade_log_df = build_restored_trade_log(trade_log_df, price_history_df, theme_df=theme_df)
        prov_restore = trade_log_df.attrs.get("panel_restoration", {})
        logger.info(
            "[DATA] stage=panel_restore execution_offset_pct=%s restored_rows=%s restored_dates=%s restored_date_min=%s restored_date_max=%s",
            prov_restore.get("execution_offset_pct"),
            prov_restore.get("restored_rows"),
            prov_restore.get("restored_dates"),
            prov_restore.get("restored_date_min"),
            prov_restore.get("restored_date_max"),
        )

    if args.universe_research:
        if price_history_df is None:
            if not os.path.exists(settings.PRICE_HISTORY_PARQUET_PATH):
                raise ValueError(f"price_history not found: {settings.PRICE_HISTORY_PARQUET_PATH}")
            price_history_df = pd.read_parquet(settings.PRICE_HISTORY_PARQUET_PATH)
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

    if args.tuned:
        recency = int(args.recency_half_life) if args.recency_half_life is not None else None
        cfg = ChampionTuningConfig(
            oos_reserve_start=args.oos_reserve_start,
            weighting_mode=args.weighting_mode,
            recency_half_life_groups=recency,
            hpo_trials=args.hpo_trials,
            require_beats_control=not args.no_gate,
            feature_selection_top_n=args.feature_selection_top_n,
            eval_mode=args.eval_mode,
            hpo_objective=args.hpo_objective,
            promotion_alpha=args.promotion_alpha,
            model_params_override=(CHAMPION_DEFAULT_MODEL_PARAMS if args.no_hpo else None),
            label_mode=args.label_mode,
            cost_mode=args.cost_mode,
            validation=validation,
            buyability_target_notional_100m=float(args.target_notional_100m),
            screen=SCREEN_REGISTRY[args.screen],
            scenario_source=args.scenario_source,
        )
        bundle = train_tuned_champion_bundle(trade_log_df, theme_df, cfg, export_dir=args.export_dir, feature_set=args.feature_set, price_history_df=price_history_df, production_dir=args.production_dir)
        prov = bundle.get("tuning_provenance", {})
        cvc = prov.get("control_vs_candidate", {})
        logger.info(
            f"[EVAL] cand={cvc.get('cand_mean')} ctrl={cvc.get('ctrl_mean')} shared_dates={cvc.get('shared_dates')} promoted={cvc.get('promoted')}"
        )
        logger.info(f"tuned bundle saved: {bundle.get('training_cutoff')} provenance={prov}")
    else:
        bundle = train_champion_bundle(trade_log_df, theme_df, export_dir=args.export_dir, feature_set=args.feature_set, price_history_df=price_history_df, scenario_source=args.scenario_source)
        logger.info(f"champion bundle saved: {bundle.get('training_cutoff')}")


if __name__ == "__main__":
    main()
