# ruff: noqa: I001 (import order pinned for grouped exit_policy/inference wiring)
"""Champion bundle orchestration."""
from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from src.ml.bundle import build_inline_bundle, save_bundle
from src.ml.dataset import build_ml_dataset, retarget_with_clip
from src.ml.feature_selection import select_stable_features
from src.ml.history_features import HISTORY_FEATURE_COLUMNS  # noqa: F401 (used via dataset)
from src.ml.oof import purged_oof_predict
from src.ml.policy_eval import default_policy_candidates, evaluate_single_stock_policy_oof
from src.ml.robust_eval import CombinatorialPurgedCV, deflated_sharpe_ratio, moving_block_bootstrap_delta
from src.ml.tuning import (
    BlendWeightResult,  # noqa: F401
    ChampionTuningConfig,
    TunedSearchResult,  # noqa: F401
    calibrate_blend_weight,
    evaluate_config_oof,
    tune_return_model_params,
)
from src.ml.exit_policy import (  # noqa: F401 (attach/simulate re-exported; research API exercised via evaluate_exit_grid)
    attach_next_day_path,
    evaluate_exit_grid,
    simulate_take_profit_exit,
    summarize_exit_grid,
)
from src.ml.buyability import evaluate_buyability_sleeves, summarize_buyability_sleeves
from src.execution.cost_model import estimate_round_trip_cost_bp, summarize_cost_breakdown, breakeven_cost_bp
from src.serving.realtime.inference import ROUND_TRIP_COST_RATIO, _CLOSE_MORNING_RERANKER_CONFIG, add_close_morning_decision_score
from src.utils.display import Colors

logger = logging.getLogger(__name__)

# ChampionTuningConfig field (canonical definition in src/ml/tuning.py):
#     buyability_target_notional_100m: float | None = None

_CANDIDATE_FEATURE_SET = "close_morning61"


def assert_oos_excluded(df: pd.DataFrame, group_col: str, oos_reserve_start: str | None) -> None:
    """No-op when None else raise if any group >= cutoff or NaT."""
    if oos_reserve_start is None:
        return
    cutoff = pd.to_datetime(oos_reserve_start, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError(f"oos_reserve_start is not parseable: {oos_reserve_start!r}")
    parsed = pd.to_datetime(df[group_col], errors="coerce")
    if parsed.isna().any():
        raise ValueError("reserved out-of-sample window leaked into training/selection: NaT group present")
    cutoff = pd.Timestamp(cutoff)
    leaked = (parsed >= cutoff).sum()
    if leaked > 0:
        raise ValueError(
            f"reserved out-of-sample window leaked into training/selection: {leaked} row(s) on/after {oos_reserve_start}"
        )


def split_oos(df: pd.DataFrame, group_col: str, oos_reserve_start: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (development_df, reserved_oos_df)."""
    if oos_reserve_start is None:
        return df.copy(), df.iloc[0:0].copy()
    cutoff = pd.to_datetime(oos_reserve_start, errors="coerce")
    if pd.isna(cutoff):
        raise ValueError(f"oos_reserve_start is not parseable: {oos_reserve_start!r}")
    parsed = pd.to_datetime(df[group_col], errors="coerce")
    mask = parsed >= pd.Timestamp(cutoff)
    dev = df.loc[~mask].copy()
    oos = df.loc[mask].copy()
    return dev, oos


def _candidate_export_dir(export_dir: str, feature_set: str, bundle: dict[str, Any]) -> str:
    """Port of legacy retrain_bundle._candidate_export_dir."""
    if feature_set != _CANDIDATE_FEATURE_SET:
        return export_dir
    version = str(bundle.get("training_cutoff", ""))[:10] or "candidate"
    return os.path.join(export_dir, f"{_CANDIDATE_FEATURE_SET}_{version}")


def _calibrate_reranker_policy(
    processed: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
) -> tuple[Any, Any]:
    """Calibrate reranker policy via purged OOF."""
    oof = purged_oof_predict(
        processed,
        feature_cols,
        target_col,
        group_col,
        n_splits=5,
        purge_gap=1,
        predict_proba=True,
    )
    oof["rank_score"] = oof["pred"]
    scored = add_close_morning_decision_score(oof, group_col=group_col, probability_weight=_CLOSE_MORNING_RERANKER_CONFIG["p_good_weight"])
    cutoff = str(scored[group_col].max())
    evaluation = evaluate_single_stock_policy_oof(
        scored,
        target_col=target_col,
        group_col=group_col,
        stock_col="stock_code",
        policy_candidates=default_policy_candidates(cutoff, score_col="decision_score"),
        min_history_dates=252,
        scenario_col="chart_analysis",
        score_col="decision_score",
    )
    policy = evaluation.selected_policy
    metadata = {
        "oof_score_col": "decision_score",
        "daily_score_col": "decision_score",
        "calibration_cutoff": str(policy.calibration_cutoff),
        "policy_version": policy.version,
        "policy_id": policy.policy_id,
        "candidate": policy.candidate,
        "policy_metrics": {
            k: evaluation.metrics[k]
            for k in (
                "n_scheduled_dates",
                "n_buy",
                "n_abstain",
                "buy_rate",
                "scheduled_mean_return",
                "scheduled_win_rate",
                "profit_factor",
                "scheduled_sharpe",
                "active_trade_mean_return",
                "active_trade_win_rate",
                "entry_sequence_drawdown",
            )
            if k in evaluation.metrics
        },
    }
    return policy, metadata


def train_champion_bundle(
    trade_log_df: pd.DataFrame,
    theme_df: pd.DataFrame | None = None,
    export_dir: str = "artifacts/models",
    feature_set: str = "close_morning61",
    price_history_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """PHASE1 reproduction of legacy champion bundle."""
    x_features, _targets, cat_features, processed = build_ml_dataset(
        trade_log_df, theme_df, feature_set=feature_set, panel_mode="scenario_action", price_history_df=price_history_df
    )
    feature_cols = [c for c in x_features.columns if c not in cat_features]
    target_col = "target_return"
    group_col = "trade_date"
    policy, policy_metadata = _calibrate_reranker_policy(processed, feature_cols, target_col, group_col)
    bundle = build_inline_bundle(
        processed[[*feature_cols, target_col, group_col]],
        feature_cols,
        target_col,
        group_col,
    )
    bundle["feature_set"] = feature_set
    bundle["panel_mode"] = "scenario_action"
    bundle["single_stock_policy"] = policy.model_dump() if policy is not None else None
    bundle["policy_metadata"] = policy_metadata
    bundle["decision_score_config"] = dict(_CLOSE_MORNING_RERANKER_CONFIG)
    bundle["oof_score_col"] = "decision_score"
    bundle["daily_score_col"] = "decision_score"
    save_dir = _candidate_export_dir(export_dir, feature_set, bundle)
    save_bundle(bundle, save_dir)
    logger.info(
        f"{Colors.GREEN}champion bundle saved: feature_set={feature_set} policy={policy.candidate if policy else None} (save_dir={save_dir}){Colors.RESET}"
    )
    return bundle


# from src.ml.champion import evaluate_promotion  # same module: define above train_tuned_champion_bundle
def evaluate_promotion(cand_returns: np.ndarray, ctrl_returns: np.ndarray, *, alpha: float) -> dict[str, Any]:
    """Significance-gated promotion on paired daily top-1 returns."""
    result = moving_block_bootstrap_delta(
        np.asarray(cand_returns, dtype=np.float64), np.asarray(ctrl_returns, dtype=np.float64)
    )
    return {
        "promoted": bool(result.delta > 0.0 and result.p_value < alpha),
        "delta": float(result.delta),
        "p_value": float(result.p_value),
        "ci_low": float(result.ci_low),
        "ci_high": float(result.ci_high),
        "n_obs": int(result.n_obs),
        "method": "moving_block_bootstrap",
    }


def train_tuned_champion_bundle(
    trade_log_df: pd.DataFrame,
    theme_df: pd.DataFrame | None,
    config: ChampionTuningConfig,
    export_dir: str = "artifacts/models",
    feature_set: str = "close_morning61",
    price_history_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """PHASE2 tuned orchestrator."""
    x_features, _targets, cat_features, processed_raw = build_ml_dataset(
        trade_log_df, theme_df, feature_set=feature_set, panel_mode="scenario_action", price_history_df=price_history_df
    )
    feature_cols = [c for c in x_features.columns if c not in cat_features]
    # Retarget with configured clip
    processed = retarget_with_clip(processed_raw, config.label_clip_lower, config.label_clip_upper)
    dev, oos = split_oos(processed, "trade_date", config.oos_reserve_start)
    assert_oos_excluded(dev, "trade_date", config.oos_reserve_start)

    # HPO
    search = TunedSearchResult(best_params=dict(config.model_params_override), best_value=float("nan"), objective="override", n_trials=0, trials=()) if config.model_params_override is not None else tune_return_model_params(dev, feature_cols, "target_return", "trade_date", config)

    if config.feature_selection_top_n is not None:
        feature_cols = select_stable_features(dev, feature_cols, "target_return", "trade_date", top_n=config.feature_selection_top_n, min_folds=config.feature_selection_min_folds, model_params=search.best_params, huber_delta=config.huber_delta)

    # Candidate OOF for blend weight
    candidate_oof = purged_oof_predict(
        dev,
        feature_cols,
        "target_return",
        "trade_date",
        n_splits=config.n_splits,
        purge_gap=config.purge_gap,
        model_params=search.best_params,
        huber_delta=config.huber_delta,
        weighting_mode=config.weighting_mode,
        recency_half_life_groups=config.recency_half_life_groups,
        predict_proba=True,
    )
    candidate_oof["rank_score"] = candidate_oof["pred"]
    cost_provenance: dict[str, Any] = {"status": "skipped", "reason": "close_price not available on candidate_oof"}
    if "close_price" in candidate_oof.columns:
        try:
            _costed = estimate_round_trip_cost_bp(candidate_oof, price_col="close_price")
            _bd = summarize_cost_breakdown(_costed)
            cost_provenance = {"status": "evaluated", **dataclasses.asdict(_bd), "breakeven_cost_bp": breakeven_cost_bp(candidate_oof["net_return"].to_numpy(dtype=float), candidate_oof["trade_date"].to_numpy())}
        except ValueError as exc:
            cost_provenance = {"status": "skipped", "reason": str(exc)}
    buyability_provenance: dict[str, Any] = {"status": "skipped", "reason": "buyability_target_notional_100m not configured"}
    if config.buyability_target_notional_100m is not None:
        try:
            buyability_provenance = {"status": "evaluated", **summarize_buyability_sleeves(evaluate_buyability_sleeves(candidate_oof, group_col="trade_date", code_col="stock_code", score_col="pred", target_col="net_return", target_notional_100m=float(config.buyability_target_notional_100m), alpha=config.promotion_alpha))}
        except ValueError as exc:
            buyability_provenance = {"status": "skipped", "reason": str(exc)}
    # 청산 규칙 그리드는 provenance 기록 전용이며 배포 결정 경로를 바꾸지 않는다.
    exit_policy_provenance: dict[str, Any] = {"status": "skipped", "reason": "price_history_df not supplied"}
    if price_history_df is not None:
        try:
            _exit_cv = CombinatorialPurgedCV(n_groups=config.cpcv_n_groups, k_test=config.cpcv_k_test)
            _exit_results = evaluate_exit_grid(
                candidate_oof, price_history_df,
                group_col="trade_date", code_col="stock_code", score_col="pred", target_col="net_return",
                cost_ratio=ROUND_TRIP_COST_RATIO, cv=_exit_cv, alpha=config.promotion_alpha,
            )
            exit_policy_provenance = {"status": "evaluated", **summarize_exit_grid(_exit_results)}
        except ValueError as exc:
            exit_policy_provenance = {"status": "skipped", "reason": str(exc)}
    blend = calibrate_blend_weight(
        candidate_oof,
        "trade_date",
        "target_return",
        "stock_code",
        "chart_analysis",
        config.p_good_weight_grid,
        config.min_history_dates,
        alpha=config.promotion_alpha,
    )
    candidate = evaluate_config_oof(
        dev,
        feature_cols,
        "target_return",
        "trade_date",
        n_splits=config.n_splits,
        purge_gap=config.purge_gap,
        model_params=search.best_params,
        huber_delta=config.huber_delta,
        weighting_mode=config.weighting_mode,
        recency_half_life_groups=config.recency_half_life_groups,
        p_good_weight=blend.chosen_weight,
        min_history_dates=config.min_history_dates,
    )

    # Control
    control_processed = retarget_with_clip(processed_raw, -0.10, 0.10)
    control_dev, _ = split_oos(control_processed, "trade_date", config.oos_reserve_start)
    control = evaluate_config_oof(
        control_dev,
        feature_cols,
        "target_return",
        "trade_date",
        n_splits=config.n_splits,
        purge_gap=config.purge_gap,
        model_params=None,
        huber_delta=0.9,
        weighting_mode="current",
        recency_half_life_groups=None,
        p_good_weight=0.5,
        min_history_dates=config.min_history_dates,
    )

    # Promotion gate on shared dates
    cand_dates = np.asarray(candidate["dates"])
    ctrl_dates = np.asarray(control["dates"])
    shared = np.intersect1d(cand_dates, ctrl_dates)
    # Map dates to returns
    # evaluation returns scheduled_returns aligned to decisions order; need to align to shared dates
    # Use decisions dates to map
    cand_map = dict(zip(candidate["dates"], candidate["scheduled_returns"], strict=True))
    ctrl_map = dict(zip(control["dates"], control["scheduled_returns"], strict=True))
    cand_shared = np.array([cand_map[d] for d in shared], dtype=np.float64) if shared.size else np.array([], dtype=np.float64)
    ctrl_shared = np.array([ctrl_map[d] for d in shared], dtype=np.float64) if shared.size else np.array([], dtype=np.float64)
    cand_mean = float(np.mean(cand_shared)) if cand_shared.size else float("nan")
    ctrl_mean = float(np.mean(ctrl_shared)) if ctrl_shared.size else float("nan")
    promotion = evaluate_promotion(cand_shared, ctrl_shared, alpha=config.promotion_alpha); promoted = promotion["promoted"]  # noqa: E702
    # control_vs_candidate wiring needs no extra import (none).
    try:
        selection_dsr = deflated_sharpe_ratio(
            np.asarray(candidate["scheduled_returns"], dtype=np.float64),
            n_independent_trials=max(1, search.n_trials),
        )
    except ValueError:
        selection_dsr = None

    if config.require_beats_control and not promoted:
        raise ValueError(
            f"tuned champion candidate does not beat identical-date control: cand={cand_mean} ctrl={ctrl_mean}"
        )

    # Build deployable bundle on dev only
    bundle = build_inline_bundle(
        dev[[*feature_cols, "target_return", "trade_date"]],
        feature_cols,
        "target_return",
        "trade_date",
        return_model_params=search.best_params,
        huber_delta=config.huber_delta,
        seeds=config.seed_ensemble,
        calibrator_mode="chrono",
        calib_group_values=dev["trade_date"].to_numpy(),
    )
    bundle["decision_score_config"] = {
        "version": "close-morning-reranker-v1",
        "rank_weight": 1.0,
        "p_good_weight": float(blend.chosen_weight),
        "score_col": "decision_score",
    }
    bundle["oof_score_col"] = "decision_score"
    bundle["daily_score_col"] = "decision_score"
    bundle["feature_set"] = feature_set
    bundle["panel_mode"] = "scenario_action"
    bundle["single_stock_policy"] = candidate["policy"].model_dump() if candidate["policy"] is not None else None
    # policy_metadata similar to train_champion
    bundle["policy_metadata"] = {
        "oof_score_col": "decision_score",
        "daily_score_col": "decision_score",
        "calibration_cutoff": str(candidate["policy"].calibration_cutoff) if candidate["policy"] is not None else "",
        "policy_version": candidate["policy"].version if candidate["policy"] is not None else "",
        "policy_id": candidate["policy"].policy_id if candidate["policy"] is not None else "",
        "candidate": candidate["policy"].candidate if candidate["policy"] is not None else "",
        "policy_metrics": {
            k: candidate["metrics"][k]
            for k in (
                "n_scheduled_dates",
                "n_buy",
                "n_abstain",
                "buy_rate",
                "scheduled_mean_return",
                "scheduled_win_rate",
                "profit_factor",
                "scheduled_sharpe",
                "active_trade_mean_return",
                "active_trade_win_rate",
                "entry_sequence_drawdown",
            )
            if k in candidate["metrics"]
        },
    }
    # tuning provenance
    bundle["tuning_provenance"] = {
        "oos_reserve_start": config.oos_reserve_start,
        "oos_row_count": len(oos),
        "best_params": dict(search.best_params),
        "best_value": float(search.best_value),
        "objective": search.objective,
        "n_trials": search.n_trials,
        "trials": list(search.trials),
        "chosen_weight": float(blend.chosen_weight),
        "per_weight": {float(k): dict(v) for k, v in blend.per_weight.items()},
        "weighting_mode": config.weighting_mode,
        "recency_half_life_groups": config.recency_half_life_groups,
        "selection_dsr": selection_dsr,
        "label_clip": (config.label_clip_lower, config.label_clip_upper),
        "huber_delta": config.huber_delta,
        "seed_ensemble": tuple(config.seed_ensemble),
        "control_vs_candidate": {"shared_dates": int(shared.size), "cand_mean": float(cand_mean), "ctrl_mean": float(ctrl_mean), "promoted": bool(promoted), "delta": promotion["delta"], "p_value": promotion["p_value"], "ci_low": promotion["ci_low"], "ci_high": promotion["ci_high"], "promotion_alpha": float(config.promotion_alpha), "method": promotion["method"]},
        "candidate_metrics": dict(candidate["metrics"]),
        "control_metrics": dict(control["metrics"]),
        "selected_features": list(feature_cols) if config.feature_selection_top_n is not None else None,
        "selection_top_n": config.feature_selection_top_n,
        "exit_policy_grid": exit_policy_provenance,
        "buyability_sleeves": buyability_provenance,
        "execution_cost": cost_provenance,
    }

    # Only write artifact if promoted or gate disabled
    if not (config.require_beats_control and not promoted):
        save_dir = _candidate_export_dir(export_dir, feature_set, bundle)
        save_bundle(bundle, save_dir)
        logger.info(
            f"[EVAL] cand={cand_mean:.6f} ctrl={ctrl_mean:.6f} shared_dates={int(shared.size)} promoted={bool(promoted)}"
        )
        logger.info(
            f"{Colors.GREEN}tuned champion bundle saved: p_good_weight={blend.chosen_weight} promoted={promoted} (save_dir={save_dir}){Colors.RESET}"
        )
    return bundle
