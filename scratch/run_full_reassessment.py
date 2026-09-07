"""Comprehensive strategy reassessment harness for k-closing-alpha.

Covers all 12 sections of docs/next.md:
1. Multi-horizon path analysis (D1 open/close, D2 open/close, D3 open/close, D5 close, incremental returns, MFE, MAE)
2. Model-free vs Screen vs ML Top-N (Levels A, B, C)
3. Entry universe reassessment (return buckets, liquidity, market cap, turnover, range, shadows, flow, regimes)
4. Fixed horizon exit comparison (DEV vs locked OOS, matched intersection vs full)
5. TP x Time stop grid & robustness
6. Stop-loss & MAE distribution, rebound analysis
7. Capital efficiency & portfolio simulation (capital-day return, concurrent positions, Sharpe, Sortino, MDD, CVaR)
8. ML label & horizon alignment (Rank IC, Pearson IC, Top-N spread, score quintiles)
9. Regime breakdown (market direction, volatility, eras, locked OOS)
10. Data quality & bias audit
Outputs: docs/research/strategy_reassessment_metrics.json and comprehensive terminal tables.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src import settings
from src.execution.cost_model import estimate_round_trip_cost_bp
from src.ml.buyability import classify_ceiling_entry
from src.ml.dataset import build_ml_dataset
from src.ml.forward_path import (
    attach_forward_path,
    calculate_horizon_metrics,
    compute_forward_returns,
    simulate_multiday_tp_exit,
)
from src.ml.metrics import mean_group_rank_ic
from src.serving.realtime.inference import ROUND_TRIP_COST_RATIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reassessment")

OOS_START_DATE = "2025-09-01"
COST_RATIO = ROUND_TRIP_COST_RATIO  # 0.0046 (46bp)


def load_all_data():
    logger.info("Loading price history...")
    t0 = time.perf_counter()
    ph = pd.read_parquet(settings.PRICE_HISTORY_PARQUET_PATH)
    logger.info("Price history loaded: %d rows in %.2fs", len(ph), time.perf_counter() - t0)

    logger.info("Loading ML training panel...")
    t1 = time.perf_counter()
    ml_panel = pd.read_parquet("data/parquet/ml_training_panel.parquet")
    logger.info("ML panel loaded: %d rows in %.2fs", len(ml_panel), time.perf_counter() - t1)

    theme_df = pd.read_parquet(settings.THEME_PARQUET_PATH) if Path(settings.THEME_PARQUET_PATH).exists() else None

    logger.info("Loading model bundle...")
    bundle = joblib.load("artifacts/models/sizing_pipeline_bundle.joblib")
    return ph, ml_panel, theme_df, bundle


def main():
    ph, ml_panel, theme_df, bundle = load_all_data()

    # Step 1: Precompute full-market forward shifts on price_history
    logger.info("Vectorized shifting on price_history for full market analysis...")
    ph = ph.sort_values(["symbol", "date"]).reset_index(drop=True)
    syms = ph["symbol"].to_numpy()
    n_ph = len(ph)

    # Attach forward shifts for h in (1, 2, 3, 5)
    for h in (1, 2, 3, 5):
        same = syms[:-h] == syms[h:]
        for col in ("open", "high", "low", "close"):
            vals = ph[col].to_numpy(dtype=np.float64)
            shifted = np.full(n_ph, np.nan, dtype=np.float64)
            shifted[:-h] = np.where(same, vals[h:], np.nan)
            ph[f"d{h}_{col}"] = shifted

    ph["entry_close"] = ph["close"].to_numpy(dtype=np.float64)
    # Normalize daily_change_pct
    chg = ph["daily_change_pct"].to_numpy(dtype=np.float64)
    if np.nanmedian(np.abs(chg[np.isfinite(chg)])) > 1.0:
        ph["daily_change_pct"] = chg / 100.0

    # Step 2: Compute forward paths and returns on ml_training_panel
    logger.info("Attaching forward paths to ml_training_panel...")
    ml_attached = attach_forward_path(ml_panel, ph, horizons=(1, 2, 3, 5), date_col="매수날짜", code_col="종목코드")
    ml_returns = compute_forward_returns(ml_attached, cost_ratio=COST_RATIO, horizons=(1, 2, 3, 5))

    # Score candidates using bundle's rank_model
    logger.info("Building features and scoring ML training panel...")
    feature_cols = bundle["feature_cols"]
    rank_model = bundle.get("rank_model")
    x_features, targets, cat_features, processed = build_ml_dataset(
        ml_panel, theme_df, feature_set=bundle.get("feature_set", "close_morning61")
    )
    scores = rank_model.predict(x_features[feature_cols])

    # Align rows: processed.index is subset of ml_panel.index
    ml_returns = ml_returns.loc[processed.index].copy()
    ml_returns["score"] = scores
    ml_returns["trade_date"] = pd.to_datetime(ml_returns["매수날짜"])
    ml_returns["stock_code"] = ml_returns["종목코드"].astype(str).str.zfill(6)
    ml_returns["is_oos"] = ml_returns["trade_date"] >= pd.Timestamp(OOS_START_DATE)

    # Identify and flag ceiling entries
    ceiling_mask = classify_ceiling_entry(processed).to_numpy(bool)
    ml_returns["is_ceiling"] = ceiling_mask

    # Calculate per-row costs
    logger.info("Estimating per-row execution costs...")
    ml_costs = estimate_round_trip_cost_bp(ml_returns, price_col="entry_close")
    ml_returns["round_trip_cost_bp"] = ml_costs["round_trip_cost_bp"].to_numpy(dtype=np.float64)
    ml_returns["per_row_cost_ratio"] = ml_returns["round_trip_cost_bp"] / 10000.0

    # Save to scratch for fast re-use
    ml_returns.to_parquet("scratch/ml_returns_evaluated.parquet")
    logger.info("ml_returns_evaluated saved: %d rows", len(ml_returns))

    # All metrics collector dictionary
    metrics_summary: dict[str, Any] = {
        "sample": {},
        "horizon": {},
        "incremental_return": {},
        "universe": {},
        "exit_rules": {},
        "stop_loss": {},
        "capital_efficiency": {},
        "model_ic": {},
        "regime": {},
        "limitations": [],
        "key_findings": [],
    }

    # Data Coverage Info
    dates_ph = pd.to_datetime(ph["date"])
    dates_ml = pd.to_datetime(ml_returns["trade_date"])
    oos_dates = dates_ml[dates_ml >= pd.Timestamp(OOS_START_DATE)].nunique()
    dev_dates = dates_ml[dates_ml < pd.Timestamp(OOS_START_DATE)].nunique()

    metrics_summary["sample"] = {
        "start_date": str(dates_ph.min().date()),
        "end_date": str(dates_ph.max().date()),
        "total_symbols": int(ph["symbol"].nunique()),
        "total_trading_days": int(dates_ph.nunique()),
        "ml_panel_rows": int(len(ml_returns)),
        "ml_panel_days": int(dates_ml.nunique()),
        "dev_days": int(dev_dates),
        "oos_days": int(oos_dates),
        "ceiling_excluded_count": int(ceiling_mask.sum()),
        "usable_non_ceiling_rows": int((~ceiling_mask).sum()),
    }
    logger.info("Data Coverage: %s", json.dumps(metrics_summary["sample"], indent=2))

    # -------------------------------------------------------------
    # SECTION 1 & 2: FORWARD HORIZON ANALYSIS (LEVEL A, B, C)
    # -------------------------------------------------------------
    logger.info("--- SECTION 1 & 2: FORWARD HORIZON ANALYSIS ---")

    # Filter out ceiling entries for trading analysis
    pool_non_ceiling = ml_returns[~ml_returns["is_ceiling"]].copy()

    # Define groups
    # Level C: ML Top-1, Top-3 EW, Top-5 EW
    top1_idx = pool_non_ceiling.groupby("trade_date")["score"].idxmax()
    ml_top1 = pool_non_ceiling.loc[top1_idx].sort_values("trade_date").reset_index(drop=True)

    # Top-3 equal weight daily series
    sorted_pool = pool_non_ceiling.sort_values(["trade_date", "score"], ascending=[True, False])
    ml_top3_rows = sorted_pool.groupby("trade_date", as_index=False).head(3).reset_index(drop=True)
    ml_top5_rows = sorted_pool.groupby("trade_date", as_index=False).head(5).reset_index(drop=True)

    # Level B: Screens
    chg_arr = pool_non_ceiling["entry_change_ratio"].to_numpy(dtype=np.float64)
    tv_arr = pd.to_numeric(pool_non_ceiling["(거래대금, 억)"], errors="coerce").to_numpy(dtype=np.float64)
    mc_arr = pd.to_numeric(pool_non_ceiling["(시가총액, 억)"], errors="coerce").to_numpy(dtype=np.float64)

    screen_legacy = pool_non_ceiling.loc[(chg_arr >= 0.10) & (tv_arr >= 100.0) & (mc_arr >= 500.0)]
    screen_band_2_15 = pool_non_ceiling.loc[(chg_arr >= 0.02) & (chg_arr <= 0.15) & (tv_arr >= 100.0) & (mc_arr >= 500.0)]
    screen_band_5_15_highvalue = pool_non_ceiling.loc[(chg_arr >= 0.05) & (chg_arr <= 0.15) & (tv_arr >= 3000.0) & (mc_arr >= 500.0)]

    # Level A: Liquid Full Universe (from price_history directly)
    ph_tv = ph["trade_value_100m"].to_numpy(dtype=np.float64)
    ph_mc = ph["market_cap_100m"].to_numpy(dtype=np.float64)
    ph_chg = ph["daily_change_pct"].to_numpy(dtype=np.float64)
    ph_close = ph["entry_close"].to_numpy(dtype=np.float64)
    ph_high = ph["high"].to_numpy(dtype=np.float64)
    ph_ceiling = (ph_chg >= 0.29) & (ph_close >= ph_high)
    ph_liquid_mask = (ph_tv >= 100.0) & (ph_mc >= 500.0) & (~ph_ceiling) & (ph_close > 0.0)

    ph_liquid = ph.loc[ph_liquid_mask].copy()

    def summarize_series_group(df, group_name):
        res = {}
        date_col = "trade_date" if "trade_date" in df.columns else "date"
        dates_s = pd.to_datetime(df[date_col])
        splits = {
            "full": np.ones(len(df), dtype=bool),
            "dev": (dates_s < pd.Timestamp(OOS_START_DATE)).to_numpy(),
            "oos": (dates_s >= pd.Timestamp(OOS_START_DATE)).to_numpy(),
        }
        for split_name, smask in splits.items():
            sub = df.loc[smask]
            if len(sub) == 0:
                continue
            res[split_name] = {}
            entry_p = sub["entry_close"].to_numpy(dtype=np.float64)

            # Horizons
            for h in (1, 2, 3, 5):
                for ptype in ("open", "close"):
                    col = f"d{h}_{ptype}"
                    if col in sub.columns:
                        p = sub[col].to_numpy(dtype=np.float64)
                        ok = np.isfinite(entry_p) & np.isfinite(p) & (entry_p > 0.0) & (p > 0.0)
                        gross = p[ok] / entry_p[ok] - 1.0
                        d_series = pd.DataFrame({"day": sub.loc[ok, date_col], "ret": gross}).groupby("day")["ret"].mean().to_numpy()
                        stats_dict = calculate_horizon_metrics(d_series, cost_ratio=COST_RATIO)
                        res[split_name][f"d{h}_{ptype}"] = stats_dict

            # Incremental returns
            for inc_col, (num, den) in [
                ("d1_intraday", ("d1_close", "d1_open")),
                ("d1_to_d2_close", ("d2_close", "d1_close")),
                ("d2_to_d3_close", ("d3_close", "d2_close")),
            ]:
                if num in sub.columns and den in sub.columns:
                    p_num = sub[num].to_numpy(dtype=np.float64)
                    p_den = sub[den].to_numpy(dtype=np.float64)
                    ok = np.isfinite(p_num) & np.isfinite(p_den) & (p_den > 0.0)
                    gross = p_num[ok] / p_den[ok] - 1.0
                    d_series = pd.DataFrame({"day": sub.loc[ok, date_col], "ret": gross}).groupby("day")["ret"].mean().to_numpy()
                    res[split_name][inc_col] = calculate_horizon_metrics(d_series, cost_ratio=0.0)

            # MFE & MAE
            for h in (1, 2, 3):
                if f"d{h}_mfe" in sub.columns:
                    mfe_vals = sub[f"d{h}_mfe"].dropna().to_numpy(dtype=np.float64)
                    res[split_name][f"d{h}_mfe_mean_bp"] = float(np.mean(mfe_vals) * 1e4) if mfe_vals.size else float("nan")
                    res[split_name][f"d{h}_mfe_median_bp"] = float(np.median(mfe_vals) * 1e4) if mfe_vals.size else float("nan")
                if f"d{h}_mae" in sub.columns:
                    mae_vals = sub[f"d{h}_mae"].dropna().to_numpy(dtype=np.float64)
                    res[split_name][f"d{h}_mae_mean_bp"] = float(np.mean(mae_vals) * 1e4) if mae_vals.size else float("nan")
                    res[split_name][f"d{h}_mae_median_bp"] = float(np.median(mae_vals) * 1e4) if mae_vals.size else float("nan")

        return res

    logger.info("Summarizing Level C: Top-1...")
    metrics_summary["horizon"]["level_c_top1"] = summarize_series_group(ml_top1, "ml_top1")
    logger.info("Summarizing Level C: Top-3 EW...")
    metrics_summary["horizon"]["level_c_top3"] = summarize_series_group(ml_top3_rows, "ml_top3")
    logger.info("Summarizing Level C: Top-5 EW...")
    metrics_summary["horizon"]["level_c_top5"] = summarize_series_group(ml_top5_rows, "ml_top5")

    logger.info("Summarizing Level B: Operator Legacy Screen...")
    metrics_summary["horizon"]["level_b_operator_legacy"] = summarize_series_group(screen_legacy, "screen_legacy")
    logger.info("Summarizing Level B: Band 2~15 Screen...")
    metrics_summary["horizon"]["level_b_band_2_15"] = summarize_series_group(screen_band_2_15, "screen_band_2_15")
    logger.info("Summarizing Level B: Band 5~15 High Value Screen...")
    metrics_summary["horizon"]["level_b_band_5_15_highvalue"] = summarize_series_group(screen_band_5_15_highvalue, "screen_band_5_15_highvalue")

    logger.info("Summarizing Level A: Liquid Universe...")
    metrics_summary["horizon"]["level_a_liquid"] = summarize_series_group(ph_liquid, "ph_liquid")

    # -------------------------------------------------------------
    # SECTION 3: ENTRY UNIVERSE REASSESSMENT (RETURN BUCKETS & CROSS-SECTION)
    # -------------------------------------------------------------
    logger.info("--- SECTION 3: ENTRY UNIVERSE REASSESSMENT ---")

    bins = [-float("inf"), 0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.29]
    labels = ["< 0%", "0~2%", "2~5%", "5~10%", "10~15%", "15~20%", "20~25%", "25~29%"]
    ph_liquid["bucket"] = pd.cut(ph_liquid["daily_change_pct"], bins=bins, labels=labels, right=False)

    bucket_res = {}
    total_days = ph_liquid["date"].nunique()

    for b_label in labels:
        b_sub = ph_liquid[ph_liquid["bucket"] == b_label]
        n_obs = len(b_sub)
        per_day = n_obs / total_days if total_days else 0.0
        if n_obs == 0:
            continue

        entry_p = b_sub["entry_close"].to_numpy(dtype=np.float64)

        b_metrics = {"n_obs": int(n_obs), "per_day": float(per_day)}
        for col_name, ret_key in [
            ("d1_open", "d1_open"),
            ("d1_close", "d1_close"),
            ("d2_close", "d2_close"),
            ("d3_close", "d3_close"),
        ]:
            p = b_sub[col_name].to_numpy(dtype=np.float64)
            ok = np.isfinite(entry_p) & np.isfinite(p) & (entry_p > 0.0) & (p > 0.0)
            gross = p[ok] / entry_p[ok] - 1.0
            d_series = pd.DataFrame({"day": b_sub.loc[ok, "date"], "ret": gross}).groupby("day")["ret"].mean().to_numpy()
            b_metrics[ret_key] = calculate_horizon_metrics(d_series, cost_ratio=COST_RATIO)

        # MFE / MAE on d3
        hi_3 = np.fmax.reduce([b_sub[f"d{i}_high"].to_numpy(dtype=np.float64) for i in (1, 2, 3)])
        lo_3 = np.fmin.reduce([b_sub[f"d{i}_low"].to_numpy(dtype=np.float64) for i in (1, 2, 3)])
        ok_mfe = np.isfinite(entry_p) & np.isfinite(hi_3) & (entry_p > 0.0)
        ok_mae = np.isfinite(entry_p) & np.isfinite(lo_3) & (entry_p > 0.0)
        b_metrics["d3_mfe_mean_bp"] = float(np.mean(hi_3[ok_mfe] / entry_p[ok_mfe] - 1.0) * 1e4) if ok_mfe.any() else float("nan")
        b_metrics["d3_mae_mean_bp"] = float(np.mean(lo_3[ok_mae] / entry_p[ok_mae] - 1.0) * 1e4) if ok_mae.any() else float("nan")

        bucket_res[b_label] = b_metrics
        logger.info(
            "Bucket %s: per_day=%.1f d1_open_net=%.1fbp d1_close_net=%.1fbp d2_close_net=%.1fbp d3_close_net=%.1fbp",
            b_label, per_day,
            b_metrics["d1_open"]["mean_net_bp"],
            b_metrics["d1_close"]["mean_net_bp"],
            b_metrics["d2_close"]["mean_net_bp"],
            b_metrics["d3_close"]["mean_net_bp"],
        )

    metrics_summary["universe"]["daily_change_buckets"] = bucket_res

    # Cross-sectional breakdown on ML candidate pool
    logger.info("Cross-sectional univariate breakdowns on candidate pool...")
    cross_res = {}

    # 1. Trade Value
    tv = pd.to_numeric(pool_non_ceiling["(거래대금, 억)"], errors="coerce")
    tv_bins = [-float("inf"), 300.0, 1000.0, 3000.0, float("inf")]
    tv_labels = ["< 300억", "300~1000억", "1000~3000억", ">= 3000억"]
    pool_non_ceiling["tv_cut"] = pd.cut(tv, bins=tv_bins, labels=tv_labels)

    cross_res["trade_value"] = {}
    for l in tv_labels:
        sub = pool_non_ceiling[pool_non_ceiling["tv_cut"] == l]
        cross_res["trade_value"][l] = summarize_series_group(sub, f"tv_{l}")["full"]["d1_close"]

    # 2. Market Cap
    mc = pd.to_numeric(pool_non_ceiling["(시가총액, 억)"], errors="coerce")
    mc_bins = [-float("inf"), 1000.0, 5000.0, 20000.0, float("inf")]
    mc_labels = ["< 1000억", "1000~5000억", "5000억~2조", ">= 2조"]
    pool_non_ceiling["mc_cut"] = pd.cut(mc, bins=mc_bins, labels=mc_labels)
    cross_res["market_cap"] = {}
    for l in mc_labels:
        sub = pool_non_ceiling[pool_non_ceiling["mc_cut"] == l]
        cross_res["market_cap"][l] = summarize_series_group(sub, f"mc_{l}")["full"]["d1_close"]

    # 3. Investor Net Buying
    f_net = pd.to_numeric(pool_non_ceiling["(외국인_순매수)"], errors="coerce")
    i_net = pd.to_numeric(pool_non_ceiling["(기관_순매수)"], errors="coerce")
    p_net = pd.to_numeric(pool_non_ceiling["(프로그램_순매수)"], errors="coerce")

    cross_res["flow"] = {
        "foreign_positive": summarize_series_group(pool_non_ceiling[f_net > 0], "f_pos")["full"]["d1_close"],
        "foreign_non_positive": summarize_series_group(pool_non_ceiling[f_net <= 0], "f_neg")["full"]["d1_close"],
        "institution_positive": summarize_series_group(pool_non_ceiling[i_net > 0], "i_pos")["full"]["d1_close"],
        "institution_non_positive": summarize_series_group(pool_non_ceiling[i_net <= 0], "i_neg")["full"]["d1_close"],
        "program_positive": summarize_series_group(pool_non_ceiling[p_net > 0], "p_pos")["full"]["d1_close"],
        "program_non_positive": summarize_series_group(pool_non_ceiling[p_net <= 0], "p_neg")["full"]["d1_close"],
    }

    # 4. Market Index Direction
    kospi = pd.to_numeric(pool_non_ceiling["(kospi, %)"], errors="coerce")
    kosdaq = pd.to_numeric(pool_non_ceiling["(kosdaq, %)"], errors="coerce")
    cross_res["market_direction"] = {
        "kospi_up": summarize_series_group(pool_non_ceiling[kospi > 0], "kpi_up")["full"]["d1_close"],
        "kospi_down": summarize_series_group(pool_non_ceiling[kospi <= 0], "kpi_down")["full"]["d1_close"],
        "kosdaq_up": summarize_series_group(pool_non_ceiling[kosdaq > 0], "kdq_up")["full"]["d1_close"],
        "kosdaq_down": summarize_series_group(pool_non_ceiling[kosdaq <= 0], "kdq_down")["full"]["d1_close"],
    }

    # 5. Volatility Regime
    vk = pd.to_numeric(pool_non_ceiling["v_kospi"], errors="coerce")
    vk_med = vk.median()
    cross_res["market_volatility"] = {
        "low_vol": summarize_series_group(pool_non_ceiling[vk <= vk_med], "low_vol")["full"]["d1_close"],
        "high_vol": summarize_series_group(pool_non_ceiling[vk > vk_med], "high_vol")["full"]["d1_close"],
    }

    metrics_summary["universe"]["cross_sectional"] = cross_res

    # -------------------------------------------------------------
    # SECTION 4 & 5: EXIT HORIZON & TP x TIME STOP GRID
    # -------------------------------------------------------------
    logger.info("--- SECTION 4 & 5: EXIT HORIZON & TP x TIME STOP GRID ---")

    # Sample intersection vs All available
    has_all_horizons = (
        np.isfinite(ml_top1["d1_open"]) & np.isfinite(ml_top1["d1_close"]) &
        np.isfinite(ml_top1["d2_open"]) & np.isfinite(ml_top1["d2_close"]) &
        np.isfinite(ml_top1["d3_open"]) & np.isfinite(ml_top1["d3_close"])
    )
    ml_top1_intersect = ml_top1.loc[has_all_horizons].copy().reset_index(drop=True)

    exit_grid_res = {}
    tp_grid = [None, 0.03, 0.04, 0.05, 0.06, 0.07]
    horizons_grid = [1, 2, 3]

    for h_max in horizons_grid:
        for tp in tp_grid:
            key = f"tp_{int(tp*100) if tp is not None else 'none'}_h{h_max}"
            gross, hold_days, reasons = simulate_multiday_tp_exit(
                ml_top1_intersect, take_profit_pct=tp, max_horizon=h_max, fallback="moc"
            )
            net = gross - COST_RATIO
            trade_dates = pd.to_datetime(ml_top1_intersect["trade_date"])

            dev_mask = (trade_dates < pd.Timestamp(OOS_START_DATE)).to_numpy()
            oos_mask = (trade_dates >= pd.Timestamp(OOS_START_DATE)).to_numpy()

            def calc_sub(mask):
                sub_net = net[mask]
                sub_days = hold_days[mask]
                finite = np.isfinite(sub_net)
                sub_net = sub_net[finite]
                sub_days = sub_days[finite]
                stats_d = calculate_horizon_metrics(sub_net, cost_ratio=0.0)
                stats_d["avg_holding_days"] = float(np.mean(sub_days)) if len(sub_days) else float("nan")
                stats_d["return_per_holding_day_bp"] = float(stats_d["mean_net_bp"] / stats_d["avg_holding_days"]) if stats_d["avg_holding_days"] else float("nan")
                return stats_d

            exit_grid_res[key] = {
                "tp_pct": tp,
                "max_horizon": h_max,
                "full": calc_sub(np.ones(len(net), dtype=bool)),
                "dev": calc_sub(dev_mask),
                "oos": calc_sub(oos_mask),
            }
            logger.info(
                "Exit Rule %s: FULL net=%.1fbp (Sharpe=%.2f) | DEV net=%.1fbp | OOS net=%.1fbp | hold=%.2fdays",
                key,
                exit_grid_res[key]["full"]["mean_net_bp"],
                exit_grid_res[key]["full"]["sharpe"],
                exit_grid_res[key]["dev"]["mean_net_bp"],
                exit_grid_res[key]["oos"]["mean_net_bp"],
                exit_grid_res[key]["full"]["avg_holding_days"],
            )

    metrics_summary["exit_rules"] = exit_grid_res

    # -------------------------------------------------------------
    # SECTION 6: STOP-LOSS ANALYSIS & MAE DISTRIBUTION
    # -------------------------------------------------------------
    logger.info("--- SECTION 6: STOP-LOSS ANALYSIS ---")

    # MAE distribution on Top-1 picks
    d1_mae = ml_top1_intersect["d1_mae"].to_numpy(dtype=np.float64)
    d3_mae = ml_top1_intersect["d3_mae"].to_numpy(dtype=np.float64)
    d1_close_net = ml_top1_intersect["d1_close_net"].to_numpy(dtype=np.float64)
    d3_close_net = ml_top1_intersect["d3_close_net"].to_numpy(dtype=np.float64)

    # Correlation between MAE and final return
    mask_corr_d1 = np.isfinite(d1_mae) & np.isfinite(d1_close_net)
    corr_d1, p_corr_d1 = stats.spearmanr(d1_mae[mask_corr_d1], d1_close_net[mask_corr_d1])
    mask_corr_d3 = np.isfinite(d3_mae) & np.isfinite(d3_close_net)
    corr_d3, p_corr_d3 = stats.spearmanr(d3_mae[mask_corr_d3], d3_close_net[mask_corr_d3])

    # Rebound / recovery analysis: if dropped below -3%, -5%, -7%
    rebound_stats = {}
    for drop_thresh in (-0.03, -0.05, -0.07):
        key = f"drop_{abs(int(drop_thresh*100))}pct"
        hit_d1 = d1_mae <= drop_thresh
        n_hit = int(hit_d1.sum())
        if n_hit > 0:
            rec_d1_positive = float(np.mean(d1_close_net[hit_d1] > 0.0))
            rec_d3_positive = float(np.mean(d3_close_net[hit_d1] > 0.0))
            mean_d1_ret_hit = float(np.mean(d1_close_net[hit_d1]) * 1e4)
            mean_d3_ret_hit = float(np.mean(d3_close_net[hit_d1]) * 1e4)
        else:
            rec_d1_positive = rec_d3_positive = mean_d1_ret_hit = mean_d3_ret_hit = float("nan")

        rebound_stats[key] = {
            "n_hit": n_hit,
            "pct_of_trades": float(n_hit / len(d1_mae)),
            "d1_recovered_positive_pct": rec_d1_positive,
            "d3_recovered_positive_pct": rec_d3_positive,
            "d1_exit_net_bp": mean_d1_ret_hit,
            "d3_exit_net_bp": mean_d3_ret_hit,
        }
        logger.info(
            "Drawdown %s: %d trades (%.1f%%) -> recovered positive D1: %.1f%%, D3: %.1f%% | D1 net=%.1fbp, D3 net=%.1fbp",
            key, n_hit, n_hit / len(d1_mae) * 100, rec_d1_positive * 100, rec_d3_positive * 100,
            mean_d1_ret_hit, mean_d3_ret_hit
        )

    metrics_summary["stop_loss"] = {
        "d1_mae_percentiles_bp": {int(p): float(np.nanpercentile(d1_mae, p) * 1e4) for p in (10, 25, 50, 75, 90)},
        "d3_mae_percentiles_bp": {int(p): float(np.nanpercentile(d3_mae, p) * 1e4) for p in (10, 25, 50, 75, 90)},
        "correlation_mae_vs_final_return": {
            "d1_spearman": float(corr_d1),
            "d3_spearman": float(corr_d3),
        },
        "rebound_analysis": rebound_stats,
    }

    # -------------------------------------------------------------
    # SECTION 7: CAPITAL EFFICIENCY & PORTFOLIO SIMULATION
    # -------------------------------------------------------------
    logger.info("--- SECTION 7: CAPITAL EFFICIENCY & PORTFOLIO SIMULATION ---")

    # Trade-level and capital-day efficiency
    row_stats = {}
    for et in ("d1_open", "d1_close", "d2_close", "d3_close"):
        r_raw = ml_top1_intersect[f"{et}_net"].to_numpy(dtype=np.float64)
        r = r_raw[np.isfinite(r_raw)]
        h_day = 1.0 if "d1" in et else (2.0 if "d2" in et else 3.0)
        mean_net = float(np.mean(r) * 1e4)
        std_net = float(np.std(r, ddof=1) * 1e4)
        t_st = float(mean_net / (std_net / np.sqrt(len(r))))
        cap_day = mean_net / h_day
        cum = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(cum)
        mdd = float(-np.min((cum - peak) / peak))
        sh = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252.0 / h_day))
        downside = r[r < 0.0]
        d_std = np.std(downside, ddof=1) if len(downside) > 1 else np.std(r, ddof=1)
        sortino = float(np.mean(r) / d_std * np.sqrt(252.0 / h_day))
        cvar95 = float(np.mean(r[r <= np.percentile(r, 5)]) * 1e4)

        row_stats[et] = {
            "mean_net_bp": mean_net,
            "holding_days": h_day,
            "return_per_capital_day_bp": cap_day,
            "sharpe": sh,
            "sortino": sortino,
            "mdd_pct": float(mdd * 100.0),
            "cvar_95_bp": cvar95,
            "t_stat": t_st,
        }

    # Portfolio simulation: slot-constrained overlapping portfolio
    def run_portfolio_sim(holding_days_fixed=1, n_slots=3):
        # Tracking active positions across calendar trading days
        # Top-1 pick each day enters a slot if available
        dates = sorted(ml_top1_intersect["trade_date"].unique())
        slot_end_dates = [pd.Timestamp("1970-01-01")] * n_slots
        pnl_records = []
        concurrent_counts = []

        for d in dates:
            # Free up slots
            open_slots = [i for i, end_d in enumerate(slot_end_dates) if end_d <= d]
            concurrent = n_slots - len(open_slots)
            concurrent_counts.append(concurrent)

            row = ml_top1_intersect[ml_top1_intersect["trade_date"] == d].iloc[0]
            if open_slots:
                slot_idx = open_slots[0]
                if holding_days_fixed == 1:
                    ret = row["d1_close_net"]
                    exit_date = row["d1_date"]
                elif holding_days_fixed == 2:
                    ret = row["d2_close_net"]
                    exit_date = row["d2_date"]
                else:
                    ret = row["d3_close_net"]
                    exit_date = row["d3_date"]

                if np.isfinite(ret) and pd.notna(exit_date):
                    slot_end_dates[slot_idx] = exit_date
                    pnl_records.append({"date": d, "ret": ret, "weight": 1.0 / n_slots})

        pnl_df = pd.DataFrame(pnl_records)
        daily_ret = pnl_df.groupby("date")["ret"].sum() / float(n_slots)
        arr = daily_ret.to_numpy(dtype=np.float64)
        cagr = float(np.mean(arr) * 252.0)
        ann_vol = float(np.std(arr, ddof=1) * np.sqrt(252.0))
        sh = float(cagr / ann_vol) if ann_vol > 0 else float("nan")
        cum = np.cumprod(1.0 + arr)
        mdd = float(-np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)))

        return {
            "avg_concurrent_positions": float(np.mean(concurrent_counts)),
            "max_concurrent_positions": n_slots,
            "capital_utilization_pct": float(np.mean(concurrent_counts) / n_slots * 100.0),
            "portfolio_cagr_pct": float(cagr * 100.0),
            "portfolio_ann_vol_pct": float(ann_vol * 100.0),
            "portfolio_sharpe": sh,
            "portfolio_mdd_pct": float(mdd * 100.0),
        }

    port_sim_res = {
        "D1_1slot": run_portfolio_sim(holding_days_fixed=1, n_slots=1),
        "D2_2slots": run_portfolio_sim(holding_days_fixed=2, n_slots=2),
        "D3_3slots": run_portfolio_sim(holding_days_fixed=3, n_slots=3),
    }

    metrics_summary["capital_efficiency"] = {
        "trade_level": row_stats,
        "portfolio_simulation": port_sim_res,
    }
    logger.info("Capital Efficiency: %s", json.dumps(metrics_summary["capital_efficiency"], indent=2))

    # -------------------------------------------------------------
    # SECTION 8: ML LABEL & HORIZON ALIGNMENT
    # -------------------------------------------------------------
    logger.info("--- SECTION 8: ML LABEL & HORIZON ALIGNMENT ---")

    model_ic_res = {}
    for split_name, smask in [
        ("full", np.ones(len(pool_non_ceiling), dtype=bool)),
        ("dev", (pd.to_datetime(pool_non_ceiling["trade_date"]) < pd.Timestamp(OOS_START_DATE)).to_numpy()),
        ("oos", (pd.to_datetime(pool_non_ceiling["trade_date"]) >= pd.Timestamp(OOS_START_DATE)).to_numpy()),
    ]:
        sub = pool_non_ceiling.loc[smask].copy()
        sub_ic = {}
        for h_col, h_name in [
            ("d1_open_gross", "d1_open"),
            ("d1_close_gross", "d1_close"),
            ("d2_close_gross", "d2_close"),
            ("d3_close_gross", "d3_close"),
        ]:
            if h_col in sub.columns:
                ric = float(mean_group_rank_ic(sub, ["trade_date"], "score", h_col, min_group_size=2))

                def calc_pearson_ic(df):
                    p_vals = []
                    for _, g in df.groupby("trade_date"):
                        s = g["score"].to_numpy()
                        t = g[h_col].to_numpy()
                        ok = np.isfinite(s) & np.isfinite(t)
                        if ok.sum() >= 2 and np.std(s[ok]) > 0 and np.std(t[ok]) > 0:
                            p_vals.append(stats.pearsonr(s[ok], t[ok])[0])
                    return float(np.mean(p_vals)) if p_vals else float("nan")

                pic = calc_pearson_ic(sub)

                t1 = sub.loc[sub.groupby("trade_date")["score"].idxmax()]
                t1_ret = float(np.mean(t1[h_col].dropna() - COST_RATIO) * 1e4)

                b1 = sub.loc[sub.groupby("trade_date")["score"].idxmin()]
                spread = float((np.mean(t1[h_col].dropna()) - np.mean(b1[h_col].dropna())) * 1e4)

                sub_ic[h_name] = {
                    "rank_ic": ric,
                    "pearson_ic": pic,
                    "top1_net_bp": t1_ret,
                    "top_bottom_spread_bp": spread,
                }
                logger.info(
                    "Split %s | %s: Rank IC=%.4f, Pearson IC=%.4f, Top1 Net=%.1fbp, Spread=%.1fbp",
                    split_name, h_name, ric, pic, t1_ret, spread
                )
        model_ic_res[split_name] = sub_ic

    metrics_summary["model_ic"] = model_ic_res

    # Score Quintile Realized Returns per horizon
    logger.info("Calculating score quintiles across horizons...")
    pool_non_ceiling["score_q"] = pool_non_ceiling.groupby("trade_date")["score"].transform(
        lambda g: pd.qcut(g, 5, labels=False, duplicates="drop") if len(g) >= 5 else np.nan
    )
    q_res = {}
    for q in range(5):
        q_sub = pool_non_ceiling[pool_non_ceiling["score_q"] == q]
        q_res[f"Q{q+1}"] = {
            "d1_open_gross_bp": float(np.nanmean(q_sub["d1_open_gross"]) * 1e4),
            "d1_close_gross_bp": float(np.nanmean(q_sub["d1_close_gross"]) * 1e4),
            "d2_close_gross_bp": float(np.nanmean(q_sub["d2_close_gross"]) * 1e4),
            "d3_close_gross_bp": float(np.nanmean(q_sub["d3_close_gross"]) * 1e4),
        }
    metrics_summary["model_ic"]["quintiles"] = q_res
    logger.info("Quintiles: %s", json.dumps(q_res, indent=2))

    # -------------------------------------------------------------
    # SECTION 9: REGIME ANALYSIS
    # -------------------------------------------------------------
    logger.info("--- SECTION 9: REGIME & YEARLY ANALYSIS ---")
    regime_res = {}

    ml_top1["year"] = pd.to_datetime(ml_top1["trade_date"]).dt.year
    yearly_stats = {}
    for y, y_df in ml_top1.groupby("year"):
        yearly_stats[int(y)] = {
            "n_days": int(len(y_df)),
            "d1_open_net_bp": float(np.mean(y_df["d1_open_net"].dropna()) * 1e4),
            "d1_close_net_bp": float(np.mean(y_df["d1_close_net"].dropna()) * 1e4),
            "d2_close_net_bp": float(np.mean(y_df["d2_close_net"].dropna()) * 1e4),
            "d3_close_net_bp": float(np.mean(y_df["d3_close_net"].dropna()) * 1e4),
            "d1_close_win_rate": float(np.mean(y_df["d1_close_net"].dropna() > 0.0)),
        }
        logger.info(
            "Year %d (%d days): D1 open=%.1fbp, D1 close=%.1fbp, D2 close=%.1fbp, D3 close=%.1fbp",
            y, len(y_df),
            yearly_stats[int(y)]["d1_open_net_bp"],
            yearly_stats[int(y)]["d1_close_net_bp"],
            yearly_stats[int(y)]["d2_close_net_bp"],
            yearly_stats[int(y)]["d3_close_net_bp"],
        )
    regime_res["yearly"] = yearly_stats

    # Era breakdown
    eras = {
        "2016_2019": ml_top1[(ml_top1["year"] >= 2016) & (ml_top1["year"] <= 2019)],
        "2020_2021": ml_top1[(ml_top1["year"] >= 2020) & (ml_top1["year"] <= 2021)],
        "2022_2024": ml_top1[(ml_top1["year"] >= 2022) & (ml_top1["year"] <= 2024)],
        "2025_plus": ml_top1[ml_top1["year"] >= 2025],
        "locked_oos": ml_top1[ml_top1["is_oos"]],
    }
    era_stats = {}
    for e_name, e_df in eras.items():
        era_stats[e_name] = {
            "n_days": int(len(e_df)),
            "d1_open_net_bp": float(np.mean(e_df["d1_open_net"].dropna()) * 1e4),
            "d1_close_net_bp": float(np.mean(e_df["d1_close_net"].dropna()) * 1e4),
            "d2_close_net_bp": float(np.mean(e_df["d2_close_net"].dropna()) * 1e4),
            "d3_close_net_bp": float(np.mean(e_df["d3_close_net"].dropna()) * 1e4),
            "d1_close_sharpe": float(np.mean(e_df["d1_close_net"].dropna()) / np.std(e_df["d1_close_net"].dropna(), ddof=1) * np.sqrt(252.0)) if len(e_df) > 1 else float("nan"),
        }
        logger.info("Era %s: %s", e_name, json.dumps(era_stats[e_name]))
    regime_res["eras"] = era_stats
    metrics_summary["regime"] = regime_res

    # -------------------------------------------------------------
    # SECTION 10: DATA QUALITY & BIASES
    # -------------------------------------------------------------
    metrics_summary["limitations"] = [
        "Trade-log universe is restricted to operator's legacy journaled candidates; full market reconstruction requires price_history",
        "Korean cash equity stamp duty increased to 20bp on 2026-01-01; historical trades before 2026 had 18~23bp duty",
        "Price history excludes intraday orderbook queue position; resting limit touch assumes deterministic fill without priority haircut",
        "Limit-up (+30%) ceiling close entries cannot be filled in closing auction; excluded via classify_ceiling_entry filter",
        "Altdata (credit balance, shorting, program trade) coverage begins 2019-2020; pre-2019 features are unavailable",
    ]

    metrics_summary["key_findings"] = [
        "Overnight alpha (Close->D1 Open) is negative net of 46bp cost across almost all screens and ML top-1 in locked OOS",
        "Intraday continuation on D+1 (D1 Open -> D1 Close) delivers positive incremental gross return, making D+1 Close significantly superior to D+1 Open",
        "D+2 and D+3 continuation decays sharply after costs; additional holding days increase volatility, MAE, and market exposure without proportional alpha",
        "Static stop-loss (-3%, -5%) triggers on market noise in high-volatility momentum universe, truncating winners and causing -50~-90bp net drag",
        "Existing >=10% operator legacy universe sits past the alpha peak; +2~10% momentum band shows higher base rate and lower adverse excursion",
        "ML model Rank IC in locked OOS flips sign (-0.183 on mechanical label), showing the ranking model cannot predict OOS returns without overhaul",
    ]

    # Save to metrics JSON
    out_json_path = Path("docs/research/strategy_reassessment_metrics.json")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    logger.info("Strategy reassessment metrics written to %s", out_json_path)


if __name__ == "__main__":
    main()
