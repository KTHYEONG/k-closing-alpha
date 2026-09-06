# ruff: noqa: I001 (import order pinned for grouped exit_policy/inference wiring)
"""Champion bundle orchestration."""
from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from src.ml.bundle import build_inline_bundle, fit_seed_ensemble, save_bundle
from src.ml.dataset import build_ml_dataset
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
from src.ml.buyability import classify_ceiling_entry, evaluate_buyability_sleeves, summarize_buyability_sleeves
from src.ml.decision_labels import assert_no_label_leakage, build_decision_labels
from src.execution.cost_model import estimate_round_trip_cost_bp, summarize_cost_breakdown, breakeven_cost_bp
from src.execution.cost_model import measure_auction_impact_bp
from src.ml.validation import cpcv_path_evidence, evaluate_locked_oos, evaluate_screen_grid, publish_bundle, run_promotion_gate, temporal_sign_consistency
from src.ml.universe import SCREEN_REGISTRY, apply_screen_mask, screen_baseline_stats
from src.ml.expected_value import expected_net_value, select_by_expected_value
from src.execution.passive_fill import measure_execution_profile, simulate_passive_entry, simulate_passive_exit
from src.data.intraday_store import intraday_partition_path
from src.data.intraday_schema import CANONICAL_BAR_COLUMNS, normalize_bar_frame
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


def _load_normalized_bars_for_entries(entries: pd.DataFrame, *, date_col: str = "trade_date", bar_interval_minutes: int = 1, session: str = "regular") -> pd.DataFrame:
    """Load only the needed (date, symbol) bars, normalizing raw vendor partitions.

    242/243 on-disk partitions are unnormalized KIS output keyed by '종목코드'
    (see src/ml/buyability.py's attach_entry_auction_liquidity for the same
    fix) -- read_intraday_range concatenates them verbatim, so simulate_passive_entry
    would see neither 'symbol' nor 'ts_hms' without this normalization pass.
    Reads one partition per distinct OOS date rather than the whole range.
    """
    dates = pd.to_datetime(entries[date_col]).dt.strftime("%Y-%m-%d").unique().tolist()
    wanted_by_date: dict[str, set[str]] = {}
    for d, sym in zip(pd.to_datetime(entries[date_col]).dt.strftime("%Y-%m-%d"), entries["symbol"].astype(str).str.zfill(6), strict=True):
        wanted_by_date.setdefault(d, set()).add(sym)
    parts: list[pd.DataFrame] = []
    for d in dates:
        path = intraday_partition_path(bar_interval_minutes, d, session)
        if not path.exists():
            continue
        try:
            raw = pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - unreadable partition
            logger.warning("[DATA] execution_profile partition read failed date=%s path=%s: %s", d, path, exc)
            continue
        if raw is None or len(raw) == 0:
            continue
        if set(CANONICAL_BAR_COLUMNS).issubset(set(raw.columns)):
            frame = raw.copy()
            frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
            parts.append(frame[frame["symbol"].isin(wanted_by_date[d])])
            continue
        vendor = "ls" if "jdiff_vol" in raw.columns else "kis"
        raw_symbol_col = "symbol" if "symbol" in raw.columns else ("종목코드" if "종목코드" in raw.columns else None)
        if raw_symbol_col is None:
            continue
        for sym in wanted_by_date[d]:
            sub = raw[raw[raw_symbol_col].astype(str).str.zfill(6) == sym]
            if len(sub) == 0:
                continue
            try:
                parts.append(normalize_bar_frame(sub, vendor, d, sym))
            except Exception as exc:  # pragma: no cover - malformed partition
                logger.warning("[DATA] execution_profile normalize failed date=%s symbol=%s: %s", d, sym, exc)
                continue
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def measure_oos_execution_profile(oos_scored: pd.DataFrame, eval_col: str) -> dict[str, Any] | None:
    """R1: measure fill rate/adverse selection for the OOS top-1 pick.

    Best-effort against the OOS-window intraday store; returns None (fail-open)
    when bars are unavailable so run_promotion_gate's execution_profile_measured
    gate fails closed rather than crashing.
    """
    try:
        if len(oos_scored) == 0 or "pred" not in oos_scored.columns:
            raise ValueError("oos_scored is empty or missing pred")
        oos_top1_idx = oos_scored.groupby("trade_date", sort=True)["pred"].idxmax()
        entries = oos_scored.loc[oos_top1_idx, ["trade_date", "stock_code", "close_price"]].copy()
        entries["symbol"] = entries["stock_code"].astype(str)
        if pd.to_datetime(entries["trade_date"]).isna().all():
            raise ValueError("no OOS dates to measure execution against")
        bars = _load_normalized_bars_for_entries(entries)
        if len(bars) == 0:
            raise ValueError("no intraday bars available in the OOS window")
        filled = simulate_passive_entry(entries, bars, offset_ticks=1)
        filled["mechanical_gross"] = oos_scored.loc[oos_top1_idx, eval_col].to_numpy(dtype=np.float64)
        entry_profile = measure_execution_profile(filled)
        exit_defaults: dict[str, Any] = {
            "exit_fill_rate": float("nan"),
            "exit_mean_saving_bp": float("nan"),
            "exit_filled_gross_bp": float("nan"),
            "exit_pool_gross_bp": float("nan"),
            "exit_adverse_selection_bp": float("nan"),
            "exit_saving_survives_adverse_selection": False,
        }
        if "nd_date" not in oos_scored.columns:
            return {**entry_profile, **exit_defaults}
        if "nd_open" in oos_scored.columns:
            next_open_vals = pd.to_numeric(oos_scored.loc[oos_top1_idx, "nd_open"], errors="coerce").to_numpy(dtype=np.float64)
        else:
            next_open_vals = pd.to_numeric(oos_scored.loc[oos_top1_idx, "close_price"], errors="coerce").to_numpy(dtype=np.float64)
        exits = pd.DataFrame({
            "symbol": oos_scored.loc[oos_top1_idx, "stock_code"].astype(str).to_numpy(),
            "next_open": next_open_vals,
            "nd_date": pd.to_datetime(oos_scored.loc[oos_top1_idx, "nd_date"]),
        })
        exits["mechanical_gross"] = oos_scored.loc[oos_top1_idx, eval_col].to_numpy(dtype=np.float64)
        try:
            exit_bars = _load_normalized_bars_for_entries(exits, date_col="nd_date")
            if len(exit_bars) == 0:
                return {**entry_profile, **exit_defaults}
            exit_filled = simulate_passive_exit(exits, exit_bars, offset_ticks=1)
            exit_profile = measure_execution_profile(exit_filled, filled_col="exit_filled", saving_col="exit_saving_bp")
        except (ValueError, KeyError) as exc:
            logger.info("[EVAL] execution_profile exit leg skipped reason=%s", str(exc))
            return {**entry_profile, **exit_defaults}
        return {
            **entry_profile,
            "exit_fill_rate": float(exit_profile["fill_rate"]),
            "exit_mean_saving_bp": float(exit_profile["mean_saving_bp"]),
            "exit_filled_gross_bp": float(exit_profile["filled_gross_bp"]),
            "exit_pool_gross_bp": float(exit_profile["pool_gross_bp"]),
            "exit_adverse_selection_bp": float(exit_profile["adverse_selection_bp"]),
            "exit_saving_survives_adverse_selection": bool(exit_profile["saving_survives_adverse_selection"]),
        }
    except (ValueError, KeyError) as exc:
        logger.info("[EVAL] execution_profile status=skipped reason=%s", str(exc))
        return None


def train_tuned_champion_bundle(
    trade_log_df: pd.DataFrame,
    theme_df: pd.DataFrame | None,
    config: ChampionTuningConfig,
    export_dir: str = "artifacts/models",
    feature_set: str = "close_morning61",
    price_history_df: pd.DataFrame | None = None,
    production_dir: str = "artifacts/models",
) -> dict[str, Any]:
    """PHASE2 tuned orchestrator."""
    x_features, _targets, cat_features, processed_raw = build_ml_dataset(
        trade_log_df, theme_df, feature_set=feature_set, panel_mode="scenario_action", price_history_df=price_history_df
    )
    feature_cols = [c for c in x_features.columns if c not in cat_features]
    assert_no_label_leakage(feature_cols)
    processed, label_provenance = build_decision_labels(processed_raw, price_history_df, label_mode=config.label_mode, cost_mode=config.cost_mode, clip_lower=config.label_clip_lower, clip_upper=config.label_clip_upper)
    dev, oos = split_oos(processed, "trade_date", config.oos_reserve_start)
    dev = dev[~classify_ceiling_entry(dev).to_numpy(dtype=bool)]
    dev_prescreen = dev
    if config.screen is not None:
        dev = dev[apply_screen_mask(dev, config.screen)]
    assert_oos_excluded(dev, "trade_date", config.oos_reserve_start)
    screen_baseline: dict[str, Any] = {"status": "skipped", "reason": "mechanical_gross not available"}
    if "mechanical_gross" in dev.columns and len(dev) > 0:
        try:
            screen_baseline = {"status": "evaluated", **screen_baseline_stats(dev, group_col="trade_date", gross_col="mechanical_gross", cost_ratio=float(ROUND_TRIP_COST_RATIO))}
        except ValueError as exc:
            screen_baseline = {"status": "skipped", "reason": str(exc)}
    screen_grid_provenance: dict[str, Any] = {"status": "skipped", "reason": "mechanical_gross not available"}
    n_screen_trials = 1
    if "mechanical_gross" in dev.columns and len(dev) > 0:
        try:
            _grid_panel = dev_prescreen.assign(
                daily_change_pct=pd.to_numeric(dev_prescreen["change_rate"], errors="coerce") / 100.0,
                close=dev_prescreen["close_price"],
                high=dev_prescreen["high_price"],
            )
            screen_grid_provenance = {"status": "evaluated", **evaluate_screen_grid(
                _grid_panel, tuple(SCREEN_REGISTRY.values()), group_col="trade_date", gross_col="mechanical_gross", cost_ratio=float(ROUND_TRIP_COST_RATIO),
            )}
            n_screen_trials = int(screen_grid_provenance.get("selection_trials_multiplier", 1))
        except (ValueError, KeyError) as exc:
            screen_grid_provenance = {"status": "skipped", "reason": str(exc)}

    # HPO
    search = TunedSearchResult(best_params=dict(config.model_params_override), best_value=float("nan"), objective="override", n_trials=0, trials=()) if config.model_params_override is not None else tune_return_model_params(dev, feature_cols, "target_return", "trade_date", config)

    if config.feature_selection_top_n is not None:
        feature_cols = select_stable_features(dev, feature_cols, "target_return", "trade_date", top_n=config.feature_selection_top_n, min_folds=config.feature_selection_min_folds, model_params=search.best_params, huber_delta=config.huber_delta)
        assert_no_label_leakage(feature_cols)

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
            _costed = estimate_round_trip_cost_bp(measure_auction_impact_bp(candidate_oof), price_col="close_price", impact_col="auction_impact_bp")
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
            _exit_cv = CombinatorialPurgedCV(n_groups=config.cpcv_n_groups, k_test=config.cpcv_k_test, purge_gap=config.purge_gap)
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
        precomputed_oof=candidate_oof,
    )

    # Control
    control_dev, _ = split_oos(processed, "trade_date", config.oos_reserve_start)
    control_dev = control_dev[~classify_ceiling_entry(control_dev).to_numpy(dtype=bool)]
    if config.screen is not None:
        control_dev = control_dev[apply_screen_mask(control_dev, config.screen)]
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
    n_policy_candidates = len(default_policy_candidates(str(dev["trade_date"].max()) if len(dev) else "1970-01-01"))
    # R3: a screen grid search is a selection decision like HPO/blend/policy and
    # must inflate the DSR trial count by the same multiplier.
    n_selection_trials = max(1, int(config.hpo_trials) * len(config.p_good_weight_grid) * int(n_policy_candidates) * n_screen_trials)
    try:
        selection_dsr = deflated_sharpe_ratio(
            np.asarray(candidate["scheduled_returns"], dtype=np.float64),
            n_independent_trials=n_selection_trials,
        )
    except ValueError:
        selection_dsr = None

    # R4: expected-value policy alongside always_buy_top1, reported regardless
    # of config.policy_mode so the always-buy default's cost is always visible.
    expected_value_provenance: dict[str, Any] = {"status": "skipped", "reason": "candidate_oof missing cost_ratio/pred"}
    if {"cost_ratio", "pred"}.issubset(candidate_oof.columns):
        try:
            # EV must be scored on the out-of-fold prediction, never on the
            # realized target_return -- that would be look-ahead.
            ev = expected_net_value(
                candidate_oof["pred"].to_numpy(dtype=np.float64),
                candidate_oof["cost_ratio"].to_numpy(dtype=np.float64),
                fill_prob=np.ones(len(candidate_oof), dtype=np.float64),
                adverse_bp=np.zeros(len(candidate_oof), dtype=np.float64),
            )
            ev_df = candidate_oof[["trade_date", "target_return"]].copy()
            ev_df["ev"] = ev
            ev_picks = select_by_expected_value(ev_df, group_col="trade_date", ev_col="ev", min_ev=0.0, max_positions=1)
            n_days_total = int(ev_df["trade_date"].nunique())
            expected_value_provenance = {
                "status": "evaluated",
                "n_days_total": n_days_total,
                "n_days_with_position": int(ev_picks.attrs.get("n_days", 0)),
                "buy_rate": float(ev_picks.attrs.get("n_days", 0) / n_days_total) if n_days_total else float("nan"),
                "mean_ev": float(ev_picks["ev"].mean()) if len(ev_picks) else float("nan"),
                # realized outcome on the days the EV policy would have traded
                "mean_realized_return": float(ev_picks["target_return"].mean()) if len(ev_picks) else float("nan"),
                "mde": float(ev_picks.attrs.get("mde", float("nan"))),
            }
        except (ValueError, KeyError) as exc:
            expected_value_provenance = {"status": "skipped", "reason": str(exc)}

    cpcv_evidence: dict[str, Any] | None = None
    oos_result: dict[str, Any] | None = None
    oos_fillable: dict[str, Any] | None = None
    execution_profile: dict[str, Any] | None = None
    decision = None
    if config.validation is not None:
        try:
            temporal = temporal_sign_consistency(pd.Series(candidate["scheduled_returns"], index=pd.Index(candidate["dates"])), split_date=config.validation.temporal_split_date)
        except ValueError:
            temporal = {"sign_consistent": False, "first_mean": float("nan"), "second_mean": float("nan"), "n_first": 0, "n_second": 0}
        _eval_col = config.validation.eval_col
        if _eval_col not in dev.columns or _eval_col not in oos.columns:
            raise ValueError(
                f"config.validation requires {_eval_col!r} on dev/oos; pass price_history_df so "
                "build_decision_labels can attach the executable label (label_mode='journaled' does "
                "not require it, but a validation protocol always does)"
            )
        _cv = CombinatorialPurgedCV(
            n_groups=config.validation.cpcv_n_groups,
            k_test=config.validation.cpcv_k_test,
            purge_gap=config.validation.purge_gap,
        )
        cpcv_evidence = cpcv_path_evidence(dev, feature_cols, "target_return", _eval_col, "trade_date", cv=_cv, candidate_params=dict(search.best_params), control_params=None, huber_delta=config.huber_delta, control_huber_delta=0.9)
        oos_result = evaluate_locked_oos(dev, oos, feature_cols, "target_return", _eval_col, "trade_date", model_params=dict(search.best_params), huber_delta=config.huber_delta, seeds=config.seed_ensemble)
        try:
            _ens = fit_seed_ensemble(dev, feature_cols, "target_return", config.seed_ensemble, dict(search.best_params), config.huber_delta)
            _oos_scored = oos.copy()
            _oos_scored["pred"] = np.asarray(_ens.predict(oos[feature_cols]), dtype=np.float64)
            _sleeves = evaluate_buyability_sleeves(_oos_scored, group_col="trade_date", code_col="stock_code", score_col="pred", target_col=_eval_col, target_notional_100m=float(config.validation.target_notional_100m), alpha=config.validation.promotion_alpha)
            _summary = summarize_buyability_sleeves(_sleeves)
            _fill = next((r for r in _sleeves if r.sleeve == "fillable"), _sleeves[0])
            oos_fillable = {"n_days": int(_fill.n_days), "n_rows": int(_fill.n_rows), "top1_mean": float(_fill.top1_mean), "rank_ic": float(_fill.rank_ic), "measured_share": float(_summary.get("measured_share", float("nan")))}
        except ValueError:  # pragma: no cover - fillable fail-closed guard
            oos_fillable = None
        # R1: measure execution before ANY promotion. Best-effort against the
        # OOS-window intraday store; fail-open to None (gate fails closed).
        execution_profile = measure_oos_execution_profile(_oos_scored, _eval_col)
        decision = run_promotion_gate(cpcv=cpcv_evidence, oos=oos_result, selection_dsr=selection_dsr, n_selection_trials=n_selection_trials, fillable=oos_fillable, config=config.validation, temporal=temporal, execution_profile=execution_profile)

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
        "ceiling_excluded_from_pool": {"n_dev_rows": int(len(dev)), "n_control_dev_rows": int(len(control_dev))},  # noqa: RUF046
        "execution_cost": cost_provenance,
        "screen": dataclasses.asdict(config.screen) if config.screen is not None else None,
        "screen_baseline": screen_baseline,
        "screen_grid": screen_grid_provenance,
        "n_screen_trials": int(n_screen_trials),
        "expected_value_policy": expected_value_provenance,
        "execution_profile": execution_profile,
    }

    bundle["label_mode"] = config.label_mode
    bundle["cost_mode"] = config.cost_mode
    bundle["label_provenance"] = dict(label_provenance)
    if decision is not None:
        bundle["promotion_decision"] = {
            "deployable": bool(decision.deployable),
            "verdict": str(decision.verdict),
            "failed_gates": list(decision.failed_gates),
            "gates": [
                {"name": g.name, "passed": bool(g.passed), "observed": float(g.observed), "threshold": float(g.threshold), "detail": dict(g.detail)}
                for g in decision.gates
            ],
        }
        publish_result = publish_bundle(bundle, _candidate_export_dir(export_dir, feature_set, bundle), production_dir, decision)
        logger.info(
            f"[EVAL] cand={cand_mean:.6f} ctrl={ctrl_mean:.6f} shared_dates={int(shared.size)} promoted={bool(promoted)} published={publish_result.get('published')}"
        )
        return bundle

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
