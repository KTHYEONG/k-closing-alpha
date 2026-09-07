"""Top-level executable script for K-Closing Alpha Research Validation v3.

Produces docs/research/v3/research_validation_v3_metrics.json as the Single Source of Truth.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.ml.research.v3_engine import (
    attach_forward_exit_paths,
    build_candidate_universe,
    compute_derived_features,
    evaluate_all_pipelines,
    evaluate_decision_gates,
    execute_walk_forward_oof,
    load_and_prepare_price_history,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_research_v3")


def run_full_validation_v3() -> dict[str, Any]:
    t_start = time.perf_counter()
    logger.info("Starting Research Validation v3 execution...")

    # 1. Load data
    ph_path = Path("data/history/price_history.parquet")
    ph, market_dates, d_to_idx = load_and_prepare_price_history(ph_path)

    # 2. Build universe
    u0_df, ph = build_candidate_universe(ph)

    # 3. Attach forward paths with suspension handling
    cands = attach_forward_exit_paths(u0_df, ph, market_dates, d_to_idx)

    # 4. Compute derived features
    cands = compute_derived_features(cands)

    # 5. Execute Walk-Forward OOF
    oof_df, fold_manifest, model_eval = execute_walk_forward_oof(cands, n_splits=5, purge_gap=2)

    # 6. Evaluate all pipelines
    pipelines = evaluate_all_pipelines(oof_df, market_dates, holdout_start="2025-09-01")

    # 7. Evaluate gates
    p1 = pipelines["P1"]
    gates, research_verdict, production_verdict = evaluate_decision_gates(
        pipelines, p1, model_eval, fold_manifest, oof_df
    )

    # 8. Sample Attrition Table
    n_raw = len(ph)
    n_u0 = len(u0_df)
    n_valid_cands = len(cands)
    n_oof = len(oof_df)
    n_top1 = p1["n_signals"]
    n_d1_tradable = int(oof_df.groupby("date").apply(lambda g: g.loc[g["oof_score_lgbm"].idxmax()]["d1_tradable"]).sum())
    n_d1_susp = p1["suspension_count"]
    n_unres = p1["unresolved_exit_count"]

    attrition_table = {
        "raw_universe_rows": n_raw,
        "u0_pit_rows": n_u0,
        "valid_feature_candidates": n_valid_cands,
        "oof_evaluated_rows": n_oof,
        "top1_signals": n_top1,
        "top1_d1_tradable": n_d1_tradable,
        "top1_d1_suspended": n_d1_susp,
        "top1_unresolved_exits": n_unres,
    }

    # 9. Return Buckets Analysis on full liquid universe for context
    logger.info("Computing return bucket breakdowns across full liquid universe...")
    bins = [-float("inf"), 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.29]
    labels = ["< 0%", "0~2%", "2~5%", "5~10%", "10~15%", "15~20%", "20~25%", "25~29%"]
    ph_liquid_mask = (ph["tv_clean"] >= 100.0) & (ph["mc_clean"] >= 500.0) & (~ph["is_ceiling"]) & (ph["close"] > 0)
    ph_liq = ph[ph_liquid_mask].copy()
    ph_liq["bucket"] = pd.cut(ph_liq["chg_ratio"], bins=bins, labels=labels, right=False)

    bucket_stats = {}
    for b in labels:
        sub_b = ph_liq[ph_liq["bucket"] == b]
        bucket_stats[b] = {
            "n_obs": len(sub_b),
            "candidates_per_day": round(len(sub_b) / len(market_dates), 1),
        }

    # 10. Incremental values
    u3_incremental_net_bp = round(pipelines["P4"]["net_bp"] - p1["net_bp"], 2)
    ev_incremental_net_bp = round(pipelines["P3"]["net_bp"] - p1["net_bp"], 2)
    ev_incremental_mdd_pct = round(pipelines["P3"]["mdd_pct"] - p1["mdd_pct"], 2)
    pa_incremental_net_bp = round(pipelines["P1_PA"]["filled_trade_net_bp"] - p1["net_bp"], 2)

    # 11. Compile Master Metrics Document
    out_dir = Path("docs/research/v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    master_metrics: dict[str, Any] = {
        "spec": {
            "version": "v3.0.0",
            "decision_timestamp": "15:20:00 KST",
            "candidate_generation_timestamp": "15:18:00 KST",
            "order_submission_timestamp": "15:20:00 KST",
            "primary_execution": "AA (Closing Auction at 15:30:00 KST)",
            "primary_exit": "Open(T+1) at 09:00:00 KST",
            "multi_testing_trials": 350,
        },
        "data_coverage": {
            "total_trading_days": len(market_dates),
            "date_range": [str(market_dates[0].strftime("%Y-%m-%d")), str(market_dates[-1].strftime("%Y-%m-%d"))],
            "oof_evaluated_days": oof_df["date"].nunique(),
            "oof_date_range": [str(oof_df["date"].min().strftime("%Y-%m-%d")), str(oof_df["date"].max().strftime("%Y-%m-%d"))],
            "intraday_1m_days": 243,
            "intraday_date_range": ["2025-09-04", "2026-09-04"],
            "survivorship_delisted_count": 39,
            "survivorship_free": False,
            "future_tradability_leakage_removed": True,
        },
        "candidate_attrition": attrition_table,
        "universe_buckets": bucket_stats,
        "walk_forward_folds": fold_manifest,
        "ranking_evaluation": model_eval,
        "pipelines": pipelines,
        "incremental_analysis": {
            "u3_incremental_net_bp": u3_incremental_net_bp,
            "ev_incremental_net_bp": ev_incremental_net_bp,
            "ev_incremental_mdd_pct": ev_incremental_mdd_pct,
            "pa_incremental_net_bp": pa_incremental_net_bp,
        },
        "gates": gates,
        "research_verdict": research_verdict,
        "production_verdict": production_verdict,
        "limitations": [
            "Historical price data (2016-2025) uses EOD daily summary proxy due to lack of 15:20 historical snapshots",
            "Survivorship bias present: only 39 delisted stocks present in price_history.parquet vs 400+ KRX historical delistings",
            "Prospective forward/shadow data not available for production live qualification",
        ],
        "invalidated_metrics": [
            "Top-1 +254.7bp (final model in-sample rescoring)",
            "Full Rank IC 0.3737 (in-sample rescoring)",
            "CAGR +345.2% (arithmetic annualized return misrepresentation)",
            "Take Profit +270~290bp (unrealistic 100% intraday touch fill assumption)",
            "U0 ML + U3 standalone effect additive sum",
            "Constant PA fill rate 87.8% / adverse selection +14.87bp blindly applied",
        ],
    }

    metrics_path = out_dir / "research_validation_v3_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, indent=2, ensure_ascii=False)

    elapsed = time.perf_counter() - t_start
    logger.info("Research validation v3 successfully executed in %.2f seconds.", elapsed)
    logger.info("Saved master metrics JSON to %s", metrics_path)
    return master_metrics


if __name__ == "__main__":
    run_full_validation_v3()
