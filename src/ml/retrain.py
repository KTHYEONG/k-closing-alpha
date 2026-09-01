"""CLI entrypoint for offline champion bundle retraining."""
from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from src import settings
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
    parser.add_argument("--oos-reserve-start", default=None)
    parser.add_argument("--weighting-mode", default="current")
    parser.add_argument("--recency-half-life", default=None)
    parser.add_argument("--hpo-trials", type=int, default=40)
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args(argv)

    trade_log_df = pd.read_parquet(args.trade_log)
    theme_df = pd.read_parquet(args.theme) if os.path.exists(args.theme) else None

    if args.tuned:
        recency = int(args.recency_half_life) if args.recency_half_life is not None else None
        cfg = ChampionTuningConfig(
            oos_reserve_start=args.oos_reserve_start,
            weighting_mode=args.weighting_mode,
            recency_half_life_groups=recency,
            hpo_trials=args.hpo_trials,
            require_beats_control=not args.no_gate,
        )
        bundle = train_tuned_champion_bundle(trade_log_df, theme_df, cfg, export_dir=args.export_dir)
        prov = bundle.get("tuning_provenance", {})
        cvc = prov.get("control_vs_candidate", {})
        logger.info(
            f"[EVAL] cand={cvc.get('cand_mean')} ctrl={cvc.get('ctrl_mean')} shared_dates={cvc.get('shared_dates')} promoted={cvc.get('promoted')}"
        )
        logger.info(f"tuned bundle saved: {bundle.get('training_cutoff')} provenance={prov}")
    else:
        bundle = train_champion_bundle(trade_log_df, theme_df, export_dir=args.export_dir)
        logger.info(f"champion bundle saved: {bundle.get('training_cutoff')}")


if __name__ == "__main__":
    main()
