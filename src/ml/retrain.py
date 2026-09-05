"""CLI entrypoint for offline champion bundle retraining."""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from src import settings
from src.data.candidate_panel import build_restored_trade_log, check_price_history_freshness
from src.ml.bundle import CHAMPION_DEFAULT_MODEL_PARAMS
from src.ml.champion import train_champion_bundle, train_tuned_champion_bundle
from src.ml.tuning import ChampionTuningConfig

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Parse retrain arguments and dispatch to the champion training pipeline."""
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
    args = parser.parse_args(argv)

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
        )
        bundle = train_tuned_champion_bundle(trade_log_df, theme_df, cfg, export_dir=args.export_dir, feature_set=args.feature_set, price_history_df=price_history_df)
        prov = bundle.get("tuning_provenance", {})
        cvc = prov.get("control_vs_candidate", {})
        logger.info(
            f"[EVAL] cand={cvc.get('cand_mean')} ctrl={cvc.get('ctrl_mean')} shared_dates={cvc.get('shared_dates')} promoted={cvc.get('promoted')}"
        )
        logger.info(f"tuned bundle saved: {bundle.get('training_cutoff')} provenance={prov}")
    else:
        bundle = train_champion_bundle(trade_log_df, theme_df, export_dir=args.export_dir, feature_set=args.feature_set, price_history_df=price_history_df)
        logger.info(f"champion bundle saved: {bundle.get('training_cutoff')}")


if __name__ == "__main__":
    main()
