"""Decision-time (15:20) re-certification of the top-k ranker on the reconstructed panel.

Certification trains and scores CPCV fold models on end-of-day features, but serving decides at 15:20,
before the closing auction moves price and trade value. This module keeps the fold models exactly as
certified, scores them on 15:20-native inputs built by the serving feature code for every covered
certification date, and reports the paired gap (pit_haircut) with day-block bootstrap intervals. Labels
are unchanged: entry is the closing-auction fill (final close) and exit the next-day open, so only the
information set differs between arms.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.daily.universe_screen import build_screen_frame
from src.data.nxt_decomposition import (
    DECOMPOSITION_FIT_REPORT_FILENAME,
    DecompositionConfig,
    FitDiagnostics,
    load_decomposition_config,
    load_fit_diagnostics,
    predict_share,
)
from src.data.pit1520_panel import (
    INDEX_BASIS_EOD_FALLBACK,
    INDEX_BASIS_LIVE,
    PIT1520_PANEL_COLUMNS,
    PanelSource,
    panel_to_decision_input,
)
from src.ml.oof import _finite_nan
from src.ml.pit_report import (
    PIT_HAIRCUT_DAILY_FILENAME,
    PIT_HAIRCUT_REPORT_FILENAME,
    RECONSTRUCTION_CERTIFICATION_FILENAME,
    AugmentationSummary,
    CalibrationStability,
    CertificationBindings,
    PairedDelta,
    PitHaircutReport,
    PitReportStatus,
    ReconstructionCertification,
    load_pit_haircut_report,
    save_pit_haircut_report,
    save_reconstruction_certification,
)
from src.ml.research.v3_engine import attach_forward_exit_paths
from src.ml.retrain_gate import MIN_NAMES_PER_EVAL_DAY
from src.ml.robust_eval import CombinatorialPurgedCV, moving_block_bootstrap_delta
from src.ml.topk_contract import (
    RANKER_FEATURE_COLS,
    TOPK_FEATURE_CONTRACT_VERSION,
    compute_derived_features,
    select_topk_by_score,
)
from src.ml.topk_history_features import HISTORY_LOOKBACK_CALENDAR_DAYS, HISTORY_REQUIRED_COLUMNS
from src.ml.topk_ranker_research import (
    CERT_REGIME_START,
    HISTORY_SEAM_EMBARGO_DAYS,
    RANKER_MODEL_PARAMS,
    RANKER_SEEDS,
    TRAIN_POOL_MIN_ROWS,
    assert_unique_date_symbol,
    attach_pit_net_label,
    build_dual_pool,
    cpcv_score_with_history,
    dedupe_cpcv_oof,
    demean_label_by_date,
    split_regime_frames,
)
from src.serving.realtime.features import build_topk_ranker_features
from src.strategy.contract import (
    MIN_TOP_K,
    PRODUCTION_STRATEGY,
    SCREENABLE_CLASS_COL,
    CostSpec,
    StrategySpec,
    UniverseSpec,
    derive_chg_ratio,
    select_universe,
    tick_cost_bp,
    training_universe,
)

logger = logging.getLogger(__name__)

_KST = ZoneInfo("Asia/Seoul")


_EOD_FULL = "eod_full"
_EOD_MATCHED = "eod_matched"
_PIT_FEATURE = "pit_feature"
_PIT_NATIVE = "pit_native"
_ARMS: tuple[str, ...] = (_EOD_FULL, _EOD_MATCHED, _PIT_FEATURE, _PIT_NATIVE)


@dataclass(frozen=True)
class AuctionNoiseConfig:
    """Experimental augmentation: perturb close-derived training features with empirical auction moves.

    Attributes:
        seed: Base seed; each fold draws with (seed, fold_id).
        min_source_days: Minimum panel days available to a fold's source distribution; fewer fails the
            experiment closed (status INSUFFICIENT_SOURCE).
        perturb_trade_value: Also scale trade value by (1 - auction trade-value share).
        declared_trials: Number of augmentation variants tried in total (multiple-testing denominator).
        alpha: Family-wise significance level before the declared_trials correction.
    """

    seed: int = 0
    min_source_days: int = 20
    perturb_trade_value: bool = True
    declared_trials: int = 1
    alpha: float = 0.05


@dataclass(frozen=True)
class PitCertificationConfig:
    """Parameters of the 15:20 re-certification.

    Attributes:
        min_day_coverage: Bar-derived panel days need superset_coverage at least this; live days always qualify.
        min_paired_days: Minimum paired days for status OK (moving_block_bootstrap_delta needs >= 30).
        min_names_for_ic: Minimum names in an arm's pool for a daily rank IC.
        bootstrap_block_days: Block length in trading days.
        bootstrap_n_boot: Bootstrap resamples.
        bootstrap_seed: Bootstrap seed.
        progress_every_days: Heartbeat interval for per-day feature construction.
        augmentation: AuctionNoiseConfig to run the experiment, None to skip it.
    """

    min_day_coverage: float = 0.95
    min_paired_days: int = 30
    min_names_for_ic: int = MIN_NAMES_PER_EVAL_DAY
    bootstrap_block_days: int = 10
    bootstrap_n_boot: int = 5000
    bootstrap_seed: int = 0
    progress_every_days: int = 20
    augmentation: AuctionNoiseConfig | None = None


def _normalize_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def select_usable_panel_days(
    panel_days: pd.DataFrame, *, cert_dates: Sequence[pd.Timestamp], config: PitCertificationConfig
) -> list[pd.Timestamp]:
    """Return panel dates eligible for paired scoring.

    A date qualifies when it is a certification-pool date (on/after CERT_REGIME_START with labels) and its
    panel day is either live-sourced or bar-sourced with superset_coverage >= config.min_day_coverage.

    Raises:
        ValueError: Missing panel_days columns.
    """
    required = ("date", "source", "superset_coverage", "index_basis")
    missing = [c for c in required if c not in panel_days.columns]
    if missing:
        raise ValueError(f"select_usable_panel_days panel_days missing required columns: {missing}")
    cert_set = {pd.Timestamp(d).normalize() for d in cert_dates}
    bound = pd.Timestamp(CERT_REGIME_START).normalize()
    out: set[pd.Timestamp] = set()
    for _, row in panel_days.iterrows():
        day = pd.Timestamp(row["date"]).normalize() if pd.notna(row["date"]) else pd.NaT
        if pd.isna(day) or day not in cert_set or day < bound:
            continue
        if str(row["source"]) == PanelSource.LIVE_DECISION.value:
            out.add(day)
            continue
        cov = float(row["superset_coverage"])
        if np.isfinite(cov) and cov >= float(config.min_day_coverage):
            out.add(day)
    return sorted(out)


def _history_window(
    sorted_hist: pd.DataFrame, date_vals: np.ndarray, day: pd.Timestamp, *, symbols: set[str]
) -> pd.DataFrame:
    start = day - pd.Timedelta(days=int(HISTORY_LOOKBACK_CALENDAR_DAYS))
    lo = int(np.searchsorted(date_vals, np.datetime64(start)))
    hi = int(np.searchsorted(date_vals, np.datetime64(day)))
    window = sorted_hist.iloc[lo:hi]
    if symbols:
        window = window[window["symbol"].astype(str).isin(symbols)]
    return window[list(HISTORY_REQUIRED_COLUMNS)].copy()


def _check_ranker_features(frame: pd.DataFrame, *, label: str) -> None:
    missing = [c for c in RANKER_FEATURE_COLS if c not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing ranker feature columns: {missing}")


def build_pit_feature_frames(
    *,
    panel: pd.DataFrame,
    usable_days: Sequence[pd.Timestamp],
    eod_pool: pd.DataFrame,
    sel_mask: np.ndarray,
    serving_history: pd.DataFrame,
    spec: StrategySpec,
    progress_every_days: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build 15:20 feature frames for the PIT-feature and PIT-native arms through the serving code path.

    PIT-feature isolates feature-value noise: the EOD pool's own rows on T (same symbols, same EOD
    selection mask) with features rebuilt from their 15:20 panel state. PIT-native reproduces serving:
    the 15:20 rank pool (training_universe(spec.universe) on 15:20 values), features over that pool, admission by the
    strategy screen on 15:20 values.

    Args:
        panel: Decision-time panel rows (PIT1520_PANEL_COLUMNS).
        usable_days: Output of select_usable_panel_days.
        eod_pool: Labeled wide pool (build_dual_pool + attach_pit_net_label) with date, symbol.
        sel_mask: Select mask aligned to eod_pool rows.
        serving_history: Raw price_history rows with HISTORY_REQUIRED_COLUMNS (serving's history view).
        spec: Strategy whose universe defines admission.

    Returns:
        (pit_feature, pit_native) frames with date, symbol, RANKER_FEATURE_COLS, selectable (bool) and,
        for pit_native, index_basis; one row per (date, symbol).

    Raises:
        ValueError: Propagated from the serving builder on missing columns.
    """
    missing_panel = [c for c in PIT1520_PANEL_COLUMNS if c not in panel.columns]
    if missing_panel:
        raise ValueError(f"build_pit_feature_frames panel missing required columns: {missing_panel}")
    missing_pool = [c for c in ("date", "symbol") if c not in eod_pool.columns]
    if missing_pool:
        raise ValueError(f"build_pit_feature_frames eod_pool missing required columns: {missing_pool}")
    missing_hist = [c for c in HISTORY_REQUIRED_COLUMNS if c not in serving_history.columns]
    if missing_hist:
        raise ValueError(f"build_pit_feature_frames serving_history missing required columns: {missing_hist}")
    if len(sel_mask) != len(eod_pool):
        raise ValueError(
            f"build_pit_feature_frames sel_mask length {len(sel_mask)} != eod_pool rows {len(eod_pool)}"
        )
    days = sorted({pd.Timestamp(d).normalize() for d in usable_days})
    panel_work = panel.copy()
    panel_work["date"] = _normalize_dates(panel_work["date"])
    pool_work = eod_pool.copy()
    pool_work["date"] = _normalize_dates(pool_work["date"])
    pool_work["symbol"] = pool_work["symbol"].astype(str)
    sel = np.asarray(sel_mask, dtype=bool)
    hist_sorted = serving_history.copy()
    hist_sorted["date"] = _normalize_dates(hist_sorted["date"])
    hist_sorted["symbol"] = hist_sorted["symbol"].astype(str)
    hist_sorted = hist_sorted.sort_values("date", kind="stable").reset_index(drop=True)
    date_vals = hist_sorted["date"].to_numpy(dtype="datetime64[ns]")
    rank_screen = training_universe(spec.universe)

    feature_parts: list[pd.DataFrame] = []
    native_parts: list[pd.DataFrame] = []
    total = len(days)
    for done, day in enumerate(days, start=1):
        panel_day = panel_work[panel_work["date"] == day].copy()
        if panel_day.empty:
            continue
        snapshot_full = panel_to_decision_input(panel_day)
        screen_full = build_screen_frame(snapshot_full, decision_date=day)
        rank_mask = np.asarray(select_universe(screen_full, rank_screen), dtype=bool)
        selectable_native = np.asarray(select_universe(screen_full, spec.universe), dtype=bool)
        # Serving restricts the snapshot to the rank pool before building features (predict.restrict_to_rank_pool),
        # so cross-sectional ranks (chg_rank, tv_rank) must be computed over the pool, not the fetch superset.
        snapshot_pool = snapshot_full.loc[rank_mask].reset_index(drop=True)
        day_symbols = set(snapshot_pool["종목코드"].astype(str).tolist())
        hist_window = _history_window(hist_sorted, date_vals, day, symbols=day_symbols)
        native_rows = build_topk_ranker_features(snapshot_pool, day, price_history=hist_window).copy()
        _check_ranker_features(native_rows, label=f"pit_native features {day.date()}")
        native_rows["date"] = day
        native_rows["symbol"] = native_rows["symbol"].astype(str)
        native_rows["selectable"] = selectable_native[rank_mask]
        basis = {
            str(s): str(b)
            for s, b in zip(
                panel_day["symbol"].astype(str).tolist(),
                panel_day["index_basis"].astype(str).tolist(),
                strict=True,
            )
        }
        native_rows["index_basis"] = native_rows["symbol"].map(basis)
        native_parts.append(
            native_rows[["date", "symbol", *RANKER_FEATURE_COLS, "selectable", "index_basis"]]
        )

        pool_day_pos = np.flatnonzero((pool_work["date"] == day).to_numpy())
        pool_day = pool_work.iloc[pool_day_pos]
        sel_day = sel[pool_day_pos]
        eod_symbols = set(pool_day["symbol"].tolist())
        panel_symbols = set(panel_day["symbol"].astype(str).tolist())
        matched = sorted(eod_symbols & set(snapshot_full["종목코드"].astype(str).tolist()) & panel_symbols)
        if matched:
            snapshot_matched = snapshot_full[
                snapshot_full["종목코드"].astype(str).isin(matched)
            ].reset_index(drop=True)
            hist_matched = _history_window(hist_sorted, date_vals, day, symbols=set(matched))
            feature_feats = build_topk_ranker_features(snapshot_matched, day, price_history=hist_matched)
            _check_ranker_features(feature_feats, label=f"pit_feature features {day.date()}")
            sel_by_symbol = {
                str(s): bool(v)
                for s, v in zip(pool_day["symbol"].tolist(), sel_day.tolist(), strict=True)
            }
            feature_feats["date"] = day
            feature_feats["symbol"] = feature_feats["symbol"].astype(str)
            feature_feats["selectable"] = feature_feats["symbol"].map(sel_by_symbol).fillna(False).astype(bool).to_numpy()
            feature_parts.append(feature_feats[["date", "symbol", *RANKER_FEATURE_COLS, "selectable"]])
        if done % max(1, int(progress_every_days)) == 0 or done == total:
            logger.info(
                "[ALGO] stage=pit_features done=%d/%d date=%s n_native=%d n_feature=%d",
                done,
                total,
                day.date(),
                sum(len(p) for p in native_parts),
                sum(len(p) for p in feature_parts),
            )
    feature_frame = (
        pd.concat(feature_parts, ignore_index=True)
        if feature_parts
        else pd.DataFrame({c: pd.Series(dtype="object") for c in ["date", "symbol", *RANKER_FEATURE_COLS, "selectable"]})
    )
    native_frame = (
        pd.concat(native_parts, ignore_index=True)
        if native_parts
        else pd.DataFrame({
            c: pd.Series(dtype="object")
            for c in ["date", "symbol", *RANKER_FEATURE_COLS, "selectable", "index_basis"]
        })
    )
    return feature_frame, native_frame


def attach_eod_labels(
    keys: pd.DataFrame,
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
    *,
    cost: CostSpec,
) -> pd.DataFrame:
    """Attach net_pit labels to arbitrary (date, symbol) keys from price_history.

    The entry is the final close of T (the closing-auction fill), never the 15:20 price; exits and costs
    follow attach_forward_exit_paths and attach_pit_net_label exactly as certification does.

    Returns:
        keys with gross_return, tick_cost_bp, net_pit; keys absent from ph are dropped and counted in the log.
    """
    work = keys[["date", "symbol"]].copy()
    work["date"] = _normalize_dates(work["date"])
    work["symbol"] = work["symbol"].astype(str)
    pool_cols = ["date", "symbol", "close", "market", "tick_cost_bp"]
    if "close_raw" in ph.columns:
        pool_cols.append("close_raw")
    right = ph[pool_cols].copy()
    right["date"] = _normalize_dates(right["date"])
    right["symbol"] = right["symbol"].astype(str)
    merged = work.merge(right, on=["date", "symbol"], how="left", validate="many_to_one")
    n_missing = int(merged["close"].isna().sum())
    if n_missing:
        logger.info(
            "[ALGO] stage=pit_certification attach_eod_labels dropped=%d keys_missing_from_price_history",
            n_missing,
        )
    merged = merged[~merged["close"].isna()].reset_index(drop=True)
    if merged.empty:
        out = merged.copy()
        out["gross_return"] = pd.Series(dtype="float64")
        out["net_pit"] = pd.Series(dtype="float64")
        return out
    with_exits = attach_forward_exit_paths(merged, ph, market_dates, d_to_idx)
    return attach_pit_net_label(with_exits, cost=cost)


def measure_auction_moves(
    panel: pd.DataFrame, ph: pd.DataFrame, *, usable_days: Sequence[pd.Timestamp]
) -> pd.DataFrame:
    """Return per symbol-day auction_move = close_raw_EOD / close_1520 - 1 and auction_tv_share = 1 - tv_1520 / tv_EOD.

    Both prices are raw basis; auction_tv_share is clipped to [0, 1]; rows with non-finite inputs are dropped.
    Used only as the augmentation source distribution and for reporting; never as a feature.
    """
    days = {pd.Timestamp(d).normalize() for d in usable_days}
    panel_work = panel[panel["date"].apply(lambda d: pd.Timestamp(d).normalize() in days)].copy()
    eod = ph[ph["date"].apply(lambda d: pd.Timestamp(d).normalize() in days)][
        ["date", "symbol", "tv_clean", *([] if "close_raw" not in ph.columns else ["close_raw"]), "close"]
    ].copy()
    eod["date"] = _normalize_dates(eod["date"])
    eod["symbol"] = eod["symbol"].astype(str)
    panel_work["date"] = _normalize_dates(panel_work["date"])
    panel_work["symbol"] = panel_work["symbol"].astype(str)
    level_col = "close_raw" if "close_raw" in eod.columns else "close"
    eod_small = eod.rename(columns={level_col: "close_eod", "tv_clean": "tv_eod"})[["date", "symbol", "close_eod", "tv_eod"]]
    panel_small = panel_work.rename(columns={"close": "close_1520", "trade_value_100m": "tv_1520"})[
        ["date", "symbol", "close_1520", "tv_1520"]
    ]
    joined = panel_small.merge(eod_small, on=["date", "symbol"], how="inner", validate="many_to_one")
    close_eod = joined["close_eod"].to_numpy(dtype=np.float64)
    close_1520 = joined["close_1520"].to_numpy(dtype=np.float64)
    tv_eod = joined["tv_eod"].to_numpy(dtype=np.float64)
    tv_1520 = joined["tv_1520"].to_numpy(dtype=np.float64)
    ok_price = (
        np.isfinite(close_eod) & np.isfinite(close_1520) & (close_eod > 0.0) & (close_1520 > 0.0)
    )
    ok_tv = np.isfinite(tv_eod) & np.isfinite(tv_1520) & (tv_eod > 0.0)
    joined["auction_move"] = np.where(ok_price, close_eod / np.where(ok_price, close_1520, 1.0) - 1.0, np.nan)
    joined["auction_tv_share"] = np.clip(
        np.where(ok_tv, 1.0 - tv_1520 / np.where(ok_tv, tv_eod, 1.0), np.nan), 0.0, 1.0
    )
    keep = np.isfinite(joined["auction_move"].to_numpy(dtype=np.float64)) & np.isfinite(
        joined["auction_tv_share"].to_numpy(dtype=np.float64)
    )
    return joined.loc[keep, ["date", "symbol", "auction_move", "auction_tv_share"]].reset_index(drop=True)


def perturb_auction_noise(
    train: pd.DataFrame, draws: pd.DataFrame, *, perturb_trade_value: bool
) -> pd.DataFrame:
    """Simulate the 15:20 state of EOD training rows by removing a drawn auction move.

    Args:
        train: Fold training rows with date, symbol, market, open, high, low, close, close_raw, prev_close,
            tv_clean and RANKER_FEATURE_COLS.
        draws: One (auction_move, auction_tv_share) pair per train row, aligned by position.
        perturb_trade_value: Scale tv_clean by (1 - auction_tv_share).

    Returns:
        Copy of train (same index and order) with close-derived features recomputed: chg_ratio,
        body_ratio, upper_shadow_ratio, intraday_range, log_tv, tv_rank, chg_rank, f_tick_cost, f_log_close.
        Labels, history features (including f_dist_high60) and log_mc are unchanged (OD-4).

    Raises:
        ValueError: len(draws) != len(train).
    """
    if len(draws) != len(train):
        raise ValueError(f"perturb_auction_noise draws {len(draws)} != train rows {len(train)}")
    out = train.copy()
    move = draws["auction_move"].to_numpy(dtype=np.float64)
    share = draws["auction_tv_share"].to_numpy(dtype=np.float64)
    close = out["close"].to_numpy(dtype=np.float64)
    low = out["low"].to_numpy(dtype=np.float64)
    high = out["high"].to_numpy(dtype=np.float64)
    ok_move = np.isfinite(move)
    raw = np.where(ok_move, close / (1.0 + np.where(ok_move, move, 0.0)), close)
    ok_band = np.isfinite(low) & np.isfinite(high) & (low > 0.0) & (high >= low)
    new_close = np.where(ok_band, np.clip(raw, low, high), raw)
    new_close = np.where(np.isfinite(close) & (close > 0.0), new_close, close)
    factor = np.where(np.isfinite(close) & (close > 0.0), new_close / np.where(close > 0.0, close, 1.0), 1.0)
    out["close"] = np.asarray(new_close, dtype=np.float64)
    if "close_raw" in out.columns:
        raw_level = pd.to_numeric(out["close_raw"], errors="coerce").to_numpy(dtype=np.float64)
        out["close_raw"] = np.where(
            np.isfinite(raw_level), raw_level * factor, np.asarray(new_close, dtype=np.float64)
        )
    if perturb_trade_value:
        tv = out["tv_clean"].to_numpy(dtype=np.float64)
        ok_share = np.isfinite(share)
        out["tv_clean"] = np.where(ok_share, tv * (1.0 - np.where(ok_share, share, 0.0)), tv)
    prev = out["prev_close"].to_numpy(dtype=np.float64)
    out["chg_ratio"] = np.asarray(derive_chg_ratio(new_close, prev), dtype=np.float64)
    out = compute_derived_features(out)
    level = (
        pd.to_numeric(out["close_raw"], errors="coerce").to_numpy(dtype=np.float64)
        if "close_raw" in out.columns
        else np.asarray(new_close, dtype=np.float64)
    )
    trade_dates = pd.to_datetime(out["date"], errors="coerce").to_numpy()
    markets = out["market"].astype(str).to_numpy(dtype=object)
    out["f_tick_cost"] = np.asarray(tick_cost_bp(level, trade_dates, markets), dtype=np.float64)
    out["f_log_close"] = np.log(np.where(level > 0.0, level, np.nan))
    return out


def _nan_delta(n_days: int = 0) -> PairedDelta:
    nan = float("nan")
    return PairedDelta(delta=nan, ci_low=nan, ci_high=nan, p_value=nan, n_days=int(n_days))


def _paired_delta(
    first: np.ndarray, second: np.ndarray, *, config: PitCertificationConfig
) -> PairedDelta:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    n = int(a.size)
    if n == 0:
        return _nan_delta(0)
    mean = float(np.nanmean(a - b)) if n else float("nan")
    if n < 30 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return PairedDelta(delta=mean, ci_low=float("nan"), ci_high=float("nan"), p_value=float("nan"), n_days=n)
    try:
        res = moving_block_bootstrap_delta(
            a,
            b,
            block_size=min(int(config.bootstrap_block_days), n),
            n_boot=int(config.bootstrap_n_boot),
            seed=int(config.bootstrap_seed),
        )
    except ValueError:
        return PairedDelta(delta=mean, ci_low=float("nan"), ci_high=float("nan"), p_value=float("nan"), n_days=n)
    return PairedDelta(delta=res.delta, ci_low=res.ci_low, ci_high=res.ci_high, p_value=res.p_value, n_days=res.n_obs)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 2:
        return float("nan")
    try:
        rho = float(spearmanr(a[ok], b[ok]).statistic)
    except Exception:
        return float("nan")
    return rho if np.isfinite(rho) else float("nan")


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "arm": pd.Series(dtype="object"),
        "topk_net_bp": pd.Series(dtype="float64"),
        "rank_ic": pd.Series(dtype="float64"),
        "n_pool": pd.Series(dtype="int64"),
        "overlap_vs_eod": pd.Series(dtype="float64"),
        "index_basis": pd.Series(dtype="object"),
    })


def _insufficient_report(
    *,
    status: PitReportStatus,
    usable_days: Sequence[pd.Timestamp],
    strategy_id: str,
    fingerprint: str,
    top_k: int,
    select_universe: dict[str, Any],
    feature_contract_version: str,
    model_params: dict[str, Any],
    seeds: tuple[int, ...],
    generated_at: str,
    augmentation: AugmentationSummary | None,
    basis_by_day: dict[pd.Timestamp, str],
) -> tuple[PitHaircutReport, pd.DataFrame]:
    days = sorted({pd.Timestamp(d).normalize() for d in usable_days})
    nan_map = {arm: float("nan") for arm in _ARMS}
    basis_means = {INDEX_BASIS_LIVE: float("nan"), INDEX_BASIS_EOD_FALLBACK: float("nan")}
    report = PitHaircutReport(
        generated_at=generated_at,
        strategy_id=strategy_id,
        strategy_fingerprint=fingerprint,
        top_k=int(top_k),
        select_universe=dict(select_universe),
        feature_contract_version=str(feature_contract_version),
        model_params=dict(model_params),
        seeds=tuple(seeds),
        status=status,
        panel_date_min=str(days[0].date()) if days else "",
        panel_date_max=str(days[-1].date()) if days else "",
        n_usable_days=len(days),
        n_paired_days=0,
        n_live_days=sum(1 for d in days if basis_by_day.get(d) == INDEX_BASIS_LIVE),
        n_eod_index_days=sum(1 for d in days if basis_by_day.get(d) == INDEX_BASIS_EOD_FALLBACK),
        mean_net_bp=dict(nan_map),
        haircut=_nan_delta(0),
        coverage_component=_nan_delta(0),
        feature_component=_nan_delta(0),
        selection_component=_nan_delta(0),
        pit_native_vs_zero=_nan_delta(0),
        rank_ic_mean=dict(nan_map),
        rank_ic_haircut=_nan_delta(0),
        pick_overlap_mean={_PIT_FEATURE: float("nan"), _PIT_NATIVE: float("nan")},
        haircut_by_index_basis=dict(basis_means),
        augmentation=augmentation,
    )
    return report, _empty_daily()


def _fold_test_dates(
    cert_df: pd.DataFrame, *, group_col: str, cv: CombinatorialPurgedCV
) -> tuple[pd.DataFrame, list[set[pd.Timestamp]]]:
    target = cert_df["train_label"].to_numpy(dtype=np.float64)
    work = cert_df[np.isfinite(target)].sort_values(group_col).copy()
    groups = _normalize_dates(work[group_col])
    per_fold: list[set[pd.Timestamp]] = []
    for _train_idx, test_idx, _fold_id in cv.split(groups):
        per_fold.append({pd.Timestamp(d).normalize() for d in groups.iloc[test_idx].tolist()})
    return work, per_fold


def _causal_source_days(
    source: pd.DataFrame,
    pool_dates: list[pd.Timestamp],
    test_dates: set[pd.Timestamp],
    *,
    embargo_days: int,
) -> pd.DataFrame:
    if source.empty:
        return source
    pos = {d: i for i, d in enumerate(pool_dates)}
    test_pos = sorted(pos[d] for d in test_dates if d in pos)
    if not test_pos:
        return source
    src_dates = _normalize_dates(source["date"])
    keep = []
    for d in src_dates.tolist():
        day = pd.Timestamp(d).normalize()
        i = pos.get(day)
        if i is None:
            keep.append(False)
            continue
        if all(abs(i - t) > int(embargo_days) for t in test_pos):
            keep.append(True)
        else:
            keep.append(False)
    return source[np.asarray(keep, dtype=bool)].reset_index(drop=True)


def _with_screenable_class(panel: pd.DataFrame, ph: pd.DataFrame) -> pd.DataFrame:
    """Attach the training panel's PIT class verdict to decision-time panel rows (fail-closed: unknown -> False)."""
    if SCREENABLE_CLASS_COL not in ph.columns:
        raise ValueError(f"run_pit_certification ph lacks {SCREENABLE_CLASS_COL}; load it via load_and_prepare_price_history")
    verdict = ph[["date", "symbol", SCREENABLE_CLASS_COL]].copy()
    verdict["date"] = _normalize_dates(verdict["date"])
    verdict["symbol"] = verdict["symbol"].astype(str)
    out = panel.drop(columns=[SCREENABLE_CLASS_COL], errors="ignore").copy()
    out["date"] = _normalize_dates(out["date"])
    out["symbol"] = out["symbol"].astype(str)
    out = out.merge(verdict, on=["date", "symbol"], how="left")
    out[SCREENABLE_CLASS_COL] = out[SCREENABLE_CLASS_COL].fillna(False).astype(bool)
    return out


def run_pit_certification(
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
    *,
    panel: pd.DataFrame,
    panel_days: pd.DataFrame,
    serving_history: pd.DataFrame,
    spec: StrategySpec = PRODUCTION_STRATEGY,
    train_spec: UniverseSpec | None = None,
    train_start: pd.Timestamp | None = None,
    cv: CombinatorialPurgedCV | None = None,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
    seeds: tuple[int, ...] = RANKER_SEEDS,
    config: PitCertificationConfig = PitCertificationConfig(),  # noqa: B008
    now: datetime | None = None,
) -> tuple[PitHaircutReport, pd.DataFrame]:
    """Re-score certified CPCV fold models on 15:20-native features and report the paired haircut.

    Args:
        ph: Prepared price-history panel (load_and_prepare_price_history).
        market_dates: Full trading calendar.
        d_to_idx: Date-to-index lookup.
        panel: Decision-time panel rows.
        panel_days: Panel day-coverage sidecar.
        serving_history: Raw price_history with HISTORY_REQUIRED_COLUMNS (what serving reads).
        spec, train_spec, train_start, cv, model_params, huber_delta, seeds: Exactly the
            run_topk_ranker_backtest arguments; fold models are trained identically. spec defaults to
            PRODUCTION_STRATEGY and train_spec (None) to training_universe(spec.universe), so the report
            certifies the same strategy identity the weekly retrain and the PIT gate compare against.
        config: Re-certification parameters.
        now: Aware clock for generated_at; None uses Asia/Seoul now.

    Returns:
        (report, daily evidence frame with date, arm, topk_net_bp, rank_ic, n_pool, overlap_vs_eod,
        index_basis).

    Raises:
        ValueError: spec.top_k below MIN_TOP_K, or propagated contract violations; insufficient evidence is
            NOT an exception (status INSUFFICIENT_DAYS / INSUFFICIENT_COVERAGE).
    """
    if int(spec.top_k) < MIN_TOP_K:
        raise ValueError(f"top_k {spec.top_k} below the minimum investable K {MIN_TOP_K}")
    k = int(spec.top_k)
    eff_params = dict(RANKER_MODEL_PARAMS) if model_params is None else dict(model_params)
    eff_seeds = tuple(seeds)
    stamp = now if now is not None else datetime.now(_KST)
    generated_at = pd.Timestamp(stamp).isoformat()
    fingerprint = spec.fingerprint()
    select_screen: dict[str, Any] = dataclasses.asdict(spec.universe)
    strategy_id = str(spec.strategy_id)
    feature_contract_version = str(TOPK_FEATURE_CONTRACT_VERSION)
    basis_by_day: dict[pd.Timestamp, str] = {}
    if {"date", "index_basis"}.issubset(set(panel_days.columns)):
        for _, row in panel_days.iterrows():
            if pd.isna(row["date"]):
                continue
            basis_by_day[pd.Timestamp(row["date"]).normalize()] = str(row["index_basis"])

    eff_train_spec = training_universe(spec.universe) if train_spec is None else train_spec
    panel = _with_screenable_class(panel, ph) if spec.universe.exclude_non_screenable_class else panel
    eff_train_start = (
        pd.Timestamp(pd.to_datetime(ph["date"]).min()) if train_start is None else pd.Timestamp(train_start)
    )
    pool, sel_mask = build_dual_pool(
        ph, market_dates, d_to_idx, train_spec=eff_train_spec, select_spec=spec.universe
    )
    labeled = demean_label_by_date(attach_pit_net_label(pool, cost=spec.cost))
    cert_df, hist_df = split_regime_frames(labeled, train_start=eff_train_start)
    cert_dates = sorted(
        {pd.Timestamp(d).normalize() for d in pd.to_datetime(cert_df["date"]).tolist()}
    )
    usable = select_usable_panel_days(panel_days, cert_dates=cert_dates, config=config)
    no_aug = AugmentationSummary(
        status="INSUFFICIENT_DAYS",
        improvement_bp=None,
        declared_trials=int(config.augmentation.declared_trials) if config.augmentation else 0,
        alpha=float(config.augmentation.alpha) if config.augmentation else 0.05,
        verdict="NOT_EVALUATED",
    )
    if not usable:
        report, daily = _insufficient_report(
            status=PitReportStatus.INSUFFICIENT_COVERAGE,
            usable_days=[],
            strategy_id=strategy_id,
            fingerprint=fingerprint,
            top_k=k,
            select_universe=select_screen,
            feature_contract_version=feature_contract_version,
            model_params=eff_params,
            seeds=eff_seeds,
            generated_at=generated_at,
            augmentation=None if config.augmentation is None else no_aug,
            basis_by_day=basis_by_day,
        )
        _log_headline(report)
        return report, daily

    pit_feature, pit_native = build_pit_feature_frames(
        progress_every_days=int(config.progress_every_days),
        panel=panel,
        usable_days=usable,
        eod_pool=labeled,
        sel_mask=np.asarray(sel_mask, dtype=bool),
        serving_history=serving_history,
        spec=spec,
    )
    splitter = cv if cv is not None else CombinatorialPurgedCV(n_groups=8, k_test=2, purge_gap=1, embargo_gap=1)

    pit_feature_records: list[pd.DataFrame] = []
    pit_native_records: list[pd.DataFrame] = []
    feat_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    nat_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    if len(pit_feature):
        for _day, _frame in pit_feature.groupby("date", sort=False):
            feat_by_date[pd.Timestamp(_day).normalize()] = _frame
    if len(pit_native):
        for _day, _frame in pit_native.groupby("date", sort=False):
            nat_by_date[pd.Timestamp(_day).normalize()] = _frame

    def _observe(fold_id: int, test_rows: pd.DataFrame, model: Any) -> None:
        test_dates = {pd.Timestamp(d).normalize() for d in pd.to_datetime(test_rows["date"]).tolist()}
        for arm, by_date, store in (
            (_PIT_FEATURE, feat_by_date, pit_feature_records),
            (_PIT_NATIVE, nat_by_date, pit_native_records),
        ):
            frames = [by_date[d] for d in test_dates if d in by_date]
            if not frames:
                continue
            sub = pd.concat(frames, ignore_index=True)
            if sub.empty:
                continue
            vals = _finite_nan(sub, list(RANKER_FEATURE_COLS))
            preds = np.asarray(model.predict(vals[list(RANKER_FEATURE_COLS)]), dtype=np.float64)
            got = pd.DataFrame({
                "date": _normalize_dates(sub["date"]).to_numpy(),
                "symbol": sub["symbol"].astype(str).to_numpy(),
                "arm": arm,
                "cpcv_fold": int(fold_id),
                "pred": preds,
            })
            store.append(got)

    oof = cpcv_score_with_history(
        cert_df,
        hist_df,
        RANKER_FEATURE_COLS,
        target_col="train_label",
        group_col="date",
        cv=splitter,
        model_params=eff_params,
        huber_delta=float(huber_delta),
        min_train_rows=TRAIN_POOL_MIN_ROWS,
        seeds=eff_seeds,
        fold_observer=_observe,
    )
    eod_dedup = dedupe_cpcv_oof(oof, value_cols=("net_pit", "gross_return", "tick_cost_bp"))
    assert_unique_date_symbol(eod_dedup)
    eod_dedup["date"] = _normalize_dates(eod_dedup["date"])
    eod_dedup["symbol"] = eod_dedup["symbol"].astype(str)

    sel_keys = {
        (pd.Timestamp(d).normalize(), str(s))
        for d, s, v in zip(
            pd.to_datetime(labeled["date"]).tolist(),
            labeled["symbol"].astype(str).tolist(),
            np.asarray(sel_mask, dtype=bool).tolist(),
            strict=True,
        )
        if bool(v)
    }
    eod_dedup = eod_dedup[
        [k in sel_keys for k in zip(eod_dedup["date"].tolist(), eod_dedup["symbol"].tolist(), strict=True)]
    ].reset_index(drop=True)

    feature_scored = (
        pd.concat(pit_feature_records, ignore_index=True) if pit_feature_records else _empty_scored()
    )
    native_scored = (
        pd.concat(pit_native_records, ignore_index=True) if pit_native_records else _empty_scored()
    )
    feature_dedup = _dedupe_scored(feature_scored)
    native_dedup = _dedupe_scored(native_scored)

    labeled_small = labeled.copy()
    labeled_small["date"] = _normalize_dates(labeled_small["date"])
    labeled_small["symbol"] = labeled_small["symbol"].astype(str)
    label_by_key = {
        (d, s): (float(n), bool(v))
        for d, s, n, v in zip(
            labeled_small["date"].tolist(),
            labeled_small["symbol"].tolist(),
            labeled_small["net_pit"].to_numpy(dtype=np.float64).tolist(),
            np.asarray(sel_mask, dtype=bool).tolist(),
            strict=True,
        )
    }
    native_labels = attach_eod_labels(
        native_dedup[["date", "symbol"]].drop_duplicates(),
        ph,
        market_dates,
        d_to_idx,
        cost=spec.cost,
    )
    native_label_by_key = (
        {
            (pd.Timestamp(d).normalize(), str(s)): float(n)
            for d, s, n in zip(
                pd.to_datetime(native_labels["date"]).tolist(),
                native_labels["symbol"].astype(str).tolist(),
                native_labels["net_pit"].to_numpy(dtype=np.float64).tolist(),
                strict=True,
            )
        }
        if len(native_labels)
        else {}
    )
    nat_attr = pit_native[["date", "symbol", "selectable", "index_basis"]].copy() if len(pit_native) else pit_native
    if len(nat_attr):
        nat_attr["date"] = _normalize_dates(nat_attr["date"])
        nat_attr["symbol"] = nat_attr["symbol"].astype(str)
    nat_attr_by_key = {
        (d, s): (bool(v), str(b))
        for d, s, v, b in zip(
            nat_attr["date"].tolist() if len(nat_attr) else [],
            nat_attr["symbol"].tolist() if len(nat_attr) else [],
            nat_attr["selectable"].tolist() if len(nat_attr) else [],
            nat_attr["index_basis"].tolist() if len(nat_attr) else [],
            strict=True,
        )
    }

    panel_symbols_by_day: dict[pd.Timestamp, set[str]] = {}
    panel_work = panel.copy()
    panel_work["date"] = _normalize_dates(panel_work["date"])
    panel_work["symbol"] = panel_work["symbol"].astype(str)
    for day, group in panel_work.groupby("date"):
        panel_symbols_by_day[pd.Timestamp(day).normalize()] = set(group["symbol"].tolist())

    per_day: dict[pd.Timestamp, dict[str, Any]] = {}
    for day in usable:
        eod_day = eod_dedup[eod_dedup["date"] == day].copy()
        day_mask = np.array(
            [k in sel_keys for k in zip(eod_day["date"].tolist(), eod_day["symbol"].tolist(), strict=True)],
            dtype=bool,
        )
        eod_day = eod_day[day_mask].copy()
        panel_syms = panel_symbols_by_day.get(day, set())
        matched_day = eod_day[eod_day["symbol"].isin(panel_syms)].copy()
        feat_day = feature_dedup[feature_dedup["date"] == day].copy()
        feat_day = feat_day[feat_day["symbol"].isin(set(matched_day["symbol"].tolist()))].copy()
        feat_day["net_pit"] = feat_day["symbol"].map(
            lambda s, _day=day: label_by_key.get((_day, s), (float("nan"), False))[0]
        ).astype(np.float64)
        feat_day["selectable"] = feat_day["symbol"].map(
            lambda s, _day=day: label_by_key.get((_day, s), (float("nan"), False))[1]
        ).astype(bool)
        nat_day = native_dedup[native_dedup["date"] == day].copy()
        nat_day["selectable"] = nat_day["symbol"].map(
            lambda s, _day=day: nat_attr_by_key.get((_day, s), (False, ""))[0]
        ).astype(bool)
        nat_day["net_pit"] = nat_day["symbol"].map(
            lambda s, _day=day: native_label_by_key.get((_day, s), float("nan"))
        ).astype(np.float64)
        arms_selectable = {
            _EOD_FULL: eod_day,
            _EOD_MATCHED: matched_day,
            _PIT_FEATURE: feat_day[feat_day["selectable"]].copy() if len(feat_day) else feat_day,
            _PIT_NATIVE: nat_day[nat_day["selectable"]].copy() if len(nat_day) else nat_day,
        }
        per_day[day] = {
            "eod_day": eod_day,
            "matched_day": matched_day,
            "feat_day": feat_day,
            "nat_day": nat_day,
            "selectable": arms_selectable,
            "index_basis": basis_by_day.get(day, ""),
        }

    def _selectable_finite_count(day: pd.Timestamp, arm: str) -> int:
        vals = per_day[day]["selectable"][arm]["net_pit"].to_numpy(dtype=np.float64)
        return int(np.isfinite(vals).sum())

    paired = sorted(day for day in usable if all(_selectable_finite_count(day, arm) >= k for arm in _ARMS))

    augmentation: AugmentationSummary | None = None
    if config.augmentation is not None and len(paired) < int(config.min_paired_days):
        augmentation = no_aug

    if len(paired) < int(config.min_paired_days):
        report, daily = _insufficient_report(
            status=PitReportStatus.INSUFFICIENT_DAYS,
            usable_days=usable,
            strategy_id=strategy_id,
            fingerprint=fingerprint,
            top_k=k,
            select_universe=select_screen,
            feature_contract_version=feature_contract_version,
            model_params=eff_params,
            seeds=eff_seeds,
            generated_at=generated_at,
            augmentation=augmentation,
            basis_by_day=basis_by_day,
        )
        _log_headline(report)
        return report, daily

    daily_topk: dict[str, list[float]] = {arm: [] for arm in _ARMS}
    daily_ic: dict[str, list[float]] = {arm: [] for arm in _ARMS}
    daily_n: dict[str, list[int]] = {arm: [] for arm in _ARMS}
    overlaps: dict[str, list[float]] = {_PIT_FEATURE: [], _PIT_NATIVE: []}
    eod_picks_by_day: dict[pd.Timestamp, set[str]] = {}
    for day in paired:
        entry = per_day[day]
        picks_by_arm: dict[str, pd.DataFrame] = {}
        for arm in _ARMS:
            sel_rows = entry["selectable"][arm]
            finite = sel_rows[np.isfinite(sel_rows["net_pit"].to_numpy(dtype=np.float64))].copy()
            picks = select_topk_by_score(finite, k)
            vals = picks["net_pit"].to_numpy(dtype=np.float64)
            daily_topk[arm].append(float(np.mean(vals[np.isfinite(vals)])))
            pool_vals = sel_rows
            ic = _spearman(
                pool_vals["pred"].to_numpy(dtype=np.float64),
                pool_vals["net_pit"].to_numpy(dtype=np.float64),
            ) if len(pool_vals) >= int(config.min_names_for_ic) else float("nan")
            daily_ic[arm].append(ic)
            daily_n[arm].append(len(sel_rows))
            picks_by_arm[arm] = picks
        eod_picks = set(picks_by_arm[_EOD_FULL]["symbol"].astype(str).tolist())
        eod_picks_by_day[day] = eod_picks
        for arm in (_PIT_FEATURE, _PIT_NATIVE):
            got = set(picks_by_arm[arm]["symbol"].astype(str).tolist())
            overlaps[arm].append(len(got & eod_picks) / float(k))

    bp = {arm: np.asarray(vals, dtype=np.float64) * 1e4 for arm, vals in daily_topk.items()}
    mean_net_bp = {arm: float(np.mean(vals)) for arm, vals in bp.items()}
    haircut = _paired_delta(bp[_EOD_FULL], bp[_PIT_NATIVE], config=config)
    coverage = _paired_delta(bp[_EOD_FULL], bp[_EOD_MATCHED], config=config)
    feature_comp = _paired_delta(bp[_EOD_MATCHED], bp[_PIT_FEATURE], config=config)
    selection = _paired_delta(bp[_PIT_FEATURE], bp[_PIT_NATIVE], config=config)
    native_zero = _paired_delta(bp[_PIT_NATIVE], np.zeros_like(bp[_PIT_NATIVE]), config=config)
    ic_arrays = {arm: np.asarray(vals, dtype=np.float64) for arm, vals in daily_ic.items()}
    rank_ic_mean = {
        arm: float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")
        for arm, vals in ic_arrays.items()
    }
    ic_pair_mask = np.isfinite(ic_arrays[_EOD_FULL]) & np.isfinite(ic_arrays[_PIT_NATIVE])
    if int(ic_pair_mask.sum()) >= 30:
        rank_ic_haircut = _paired_delta(
            ic_arrays[_EOD_FULL][ic_pair_mask], ic_arrays[_PIT_NATIVE][ic_pair_mask], config=config
        )
    else:
        rank_ic_haircut = _nan_delta(int(ic_pair_mask.sum()))
    pick_overlap_mean = {arm: float(np.mean(vals)) for arm, vals in overlaps.items()}
    basis_means: dict[str, float] = {}
    for basis in (INDEX_BASIS_LIVE, INDEX_BASIS_EOD_FALLBACK):
        vals = [
            float(bp[_EOD_FULL][i] - bp[_PIT_NATIVE][i])
            for i, day in enumerate(paired)
            if per_day[day]["index_basis"] == basis
        ]
        basis_means[basis] = float(np.mean(vals)) if vals else float("nan")

    if config.augmentation is not None:
        augmentation = _run_augmentation(
            cert_df=cert_df,
            hist_df=hist_df,
            splitter=splitter,
            eff_params=eff_params,
            huber_delta=float(huber_delta),
            eff_seeds=eff_seeds,
            panel=panel,
            ph=ph,
            usable=usable,
            paired=paired,
            per_day=per_day,
            native_features=pit_native,
            baseline_native_bp=bp[_PIT_NATIVE],
            aug_config=config.augmentation,
            config=config,
            k=k,
        )

    report = PitHaircutReport(
        generated_at=generated_at,
        strategy_id=strategy_id,
        strategy_fingerprint=fingerprint,
        top_k=k,
        select_universe=dict(select_screen),
        feature_contract_version=feature_contract_version,
        model_params=dict(eff_params),
        seeds=tuple(eff_seeds),
        status=PitReportStatus.OK,
        panel_date_min=str(paired[0].date()),
        panel_date_max=str(paired[-1].date()),
        n_usable_days=len(usable),
        n_paired_days=len(paired),
        n_live_days=sum(1 for d in usable if basis_by_day.get(d) == INDEX_BASIS_LIVE),
        n_eod_index_days=sum(1 for d in usable if basis_by_day.get(d) == INDEX_BASIS_EOD_FALLBACK),
        mean_net_bp=dict(mean_net_bp),
        haircut=haircut,
        coverage_component=coverage,
        feature_component=feature_comp,
        selection_component=selection,
        pit_native_vs_zero=native_zero,
        rank_ic_mean=dict(rank_ic_mean),
        rank_ic_haircut=rank_ic_haircut,
        pick_overlap_mean=dict(pick_overlap_mean),
        haircut_by_index_basis=dict(basis_means),
        augmentation=augmentation,
    )
    daily = _build_daily(
        usable=usable,
        paired_days=set(paired),
        per_day=per_day,
        daily_topk=daily_topk,
        daily_ic=daily_ic,
        daily_n=daily_n,
        overlaps=overlaps,
        eod_picks_by_day=eod_picks_by_day,
    )
    _log_headline(report)
    return report, daily


def _empty_scored() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "symbol": pd.Series(dtype="object"),
        "arm": pd.Series(dtype="object"),
        "cpcv_fold": pd.Series(dtype="int64"),
        "pred": pd.Series(dtype="float64"),
    })


def _dedupe_scored(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        out = _empty_scored().copy()
        out["date"] = pd.Series(dtype="datetime64[ns]")
        return out
    work = scored.copy()
    work["date"] = _normalize_dates(work["date"])
    work["symbol"] = work["symbol"].astype(str)
    agg: dict[str, str] = {"pred": "mean", "arm": "first", "cpcv_fold": "first"}
    out = work.groupby(["date", "symbol"], sort=False).agg(agg).reset_index()
    return out


def _build_daily(
    *,
    usable: list[pd.Timestamp],
    paired_days: set[pd.Timestamp],
    per_day: dict[pd.Timestamp, dict[str, Any]],
    daily_topk: dict[str, list[float]],
    daily_ic: dict[str, list[float]],
    daily_n: dict[str, list[int]],
    overlaps: dict[str, list[float]],
    eod_picks_by_day: dict[pd.Timestamp, set[str]],
) -> pd.DataFrame:
    paired = sorted(paired_days)
    rows: list[dict[str, Any]] = []
    for i, day in enumerate(paired):
        entry = per_day[day]
        for arm in _ARMS:
            overlap: float
            if arm == _EOD_FULL:
                overlap = 1.0
            elif arm == _EOD_MATCHED:
                got = set(
                    select_topk_by_score(
                        entry["selectable"][arm][
                            np.isfinite(entry["selectable"][arm]["net_pit"].to_numpy(dtype=np.float64))
                        ],
                        len(eod_picks_by_day[day]),
                    )["symbol"].astype(str).tolist()
                ) if len(entry["selectable"][arm]) else set()
                overlap = len(got & eod_picks_by_day[day]) / float(len(eod_picks_by_day[day])) if eod_picks_by_day[day] else float("nan")
            else:
                overlap = float(overlaps[arm][i])
            rows.append({
                "date": day,
                "arm": arm,
                "topk_net_bp": float(daily_topk[arm][i] * 1e4),
                "rank_ic": float(daily_ic[arm][i]),
                "n_pool": int(daily_n[arm][i]),
                "overlap_vs_eod": overlap,
                "index_basis": str(entry["index_basis"]),
            })
    void = [d for d in usable if d not in paired_days]
    for day in void:
        entry = per_day[day]
        rows.extend(
            {
                "date": day,
                "arm": arm,
                "topk_net_bp": float("nan"),
                "rank_ic": float("nan"),
                "n_pool": len(entry["selectable"][arm]),
                "overlap_vs_eod": float("nan"),
                "index_basis": str(entry["index_basis"]),
            }
            for arm in _ARMS
        )
    frame = pd.DataFrame(rows, columns=["date", "arm", "topk_net_bp", "rank_ic", "n_pool", "overlap_vs_eod", "index_basis"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["date", "arm"], kind="stable").reset_index(drop=True)


def _augmentation_verdict(improvement: PairedDelta, *, declared_trials: int, alpha: float) -> str:
    """Decide adoption under the family-wise Bonferroni correction."""
    if (
        np.isfinite(improvement.ci_low)
        and improvement.ci_low > 0.0
        and np.isfinite(improvement.p_value)
        and improvement.p_value < float(alpha) / max(1, int(declared_trials))
    ):
        return "ADOPT_CANDIDATE"
    return "REJECT"


def _run_augmentation(
    *,
    cert_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    splitter: CombinatorialPurgedCV,
    eff_params: dict[str, Any],
    huber_delta: float,
    eff_seeds: tuple[int, ...],
    panel: pd.DataFrame,
    ph: pd.DataFrame,
    usable: list[pd.Timestamp],
    paired: list[pd.Timestamp],
    per_day: dict[pd.Timestamp, dict[str, Any]],
    native_features: pd.DataFrame,
    baseline_native_bp: np.ndarray,
    aug_config: AuctionNoiseConfig,
    config: PitCertificationConfig,
    k: int,
) -> AugmentationSummary:
    source = measure_auction_moves(panel, ph, usable_days=usable)
    cert_work, test_dates_per_fold = _fold_test_dates(cert_df, group_col="date", cv=splitter)
    pool_dates = sorted({pd.Timestamp(d).normalize() for d in pd.to_datetime(cert_work["date"]).tolist()})
    fold_sources: list[pd.DataFrame] = []
    for fold_id in range(len(test_dates_per_fold)):
        src = _causal_source_days(
            source, pool_dates, test_dates_per_fold[fold_id], embargo_days=HISTORY_SEAM_EMBARGO_DAYS
        )
        fold_sources.append(src)
        if src["date"].nunique() < int(aug_config.min_source_days):
            return AugmentationSummary(
                status="INSUFFICIENT_SOURCE",
                improvement_bp=None,
                declared_trials=int(aug_config.declared_trials),
                alpha=float(aug_config.alpha),
                verdict="NOT_EVALUATED",
            )

    def _transform(fold_id: int, train: pd.DataFrame) -> pd.DataFrame:
        src = fold_sources[int(fold_id)]
        rng = np.random.default_rng((int(aug_config.seed), int(fold_id)))
        picks = rng.integers(0, len(src), size=len(train))
        draws = src.iloc[picks][["auction_move", "auction_tv_share"]].reset_index(drop=True)
        return perturb_auction_noise(train, draws, perturb_trade_value=bool(aug_config.perturb_trade_value))

    native_only: list[pd.DataFrame] = []
    nat_feat_groups: dict[pd.Timestamp, pd.DataFrame] = {}
    if len(native_features):
        feat_work = native_features.copy()
        feat_work["date"] = _normalize_dates(feat_work["date"])
        for day, group in feat_work.groupby("date"):
            nat_feat_groups[pd.Timestamp(day).normalize()] = group

    def _observe_native(fold_id: int, test_rows: pd.DataFrame, model: Any) -> None:
        test_dates = {pd.Timestamp(d).normalize() for d in pd.to_datetime(test_rows["date"]).tolist()}
        frames = [nat_feat_groups[d] for d in test_dates if d in nat_feat_groups]
        if not frames:
            return
        sub = pd.concat(frames, ignore_index=True)
        if sub.empty:
            return
        vals = _finite_nan(sub, list(RANKER_FEATURE_COLS))
        preds = np.asarray(model.predict(vals[list(RANKER_FEATURE_COLS)]), dtype=np.float64)
        native_only.append(pd.DataFrame({
            "date": _normalize_dates(sub["date"]).to_numpy(),
            "symbol": sub["symbol"].astype(str).to_numpy(),
            "arm": _PIT_NATIVE,
            "cpcv_fold": int(fold_id),
            "pred": preds,
        }))

    cpcv_score_with_history(
        cert_df,
        hist_df,
        RANKER_FEATURE_COLS,
        target_col="train_label",
        group_col="date",
        cv=splitter,
        model_params=eff_params,
        huber_delta=float(huber_delta),
        min_train_rows=TRAIN_POOL_MIN_ROWS,
        seeds=eff_seeds,
        fold_train_transform=_transform,
        fold_observer=_observe_native,
    )
    aug_scored = _dedupe_scored(
        pd.concat(native_only, ignore_index=True) if native_only else _empty_scored()
    )
    aug_daily: list[float] = []
    for day in paired:
        entry = per_day[day]
        base_nat = entry["nat_day"]
        aug_preds = aug_scored[aug_scored["date"] == day][["symbol", "pred"]].copy()
        if aug_preds.empty:
            aug_daily.append(float("nan"))
            continue
        by_symbol = {str(s): float(p) for s, p in zip(aug_preds["symbol"].tolist(), aug_preds["pred"].tolist(), strict=True)}
        sel_rows = base_nat[base_nat["selectable"]].copy()
        sel_rows["pred"] = sel_rows["symbol"].map(by_symbol)
        sel_rows = sel_rows[np.isfinite(sel_rows["pred"].to_numpy(dtype=np.float64))]
        finite = sel_rows[np.isfinite(sel_rows["net_pit"].to_numpy(dtype=np.float64))].copy()
        if len(finite) < k:
            aug_daily.append(float("nan"))
            continue
        picks = select_topk_by_score(finite, k)
        vals = picks["net_pit"].to_numpy(dtype=np.float64)
        aug_daily.append(float(np.mean(vals[np.isfinite(vals)]) * 1e4))
    aug_arr = np.asarray(aug_daily, dtype=np.float64)
    base_arr = np.asarray(baseline_native_bp, dtype=np.float64)
    mask = np.isfinite(aug_arr) & np.isfinite(base_arr)
    if int(mask.sum()) < 30:
        improvement = _nan_delta(int(mask.sum()))
        verdict = "REJECT"
    else:
        improvement = _paired_delta(aug_arr[mask], base_arr[mask], config=config)
        verdict = _augmentation_verdict(
            improvement, declared_trials=int(aug_config.declared_trials), alpha=float(aug_config.alpha)
        )
    logger.info(
        "[ALGO] stage=pit_augmentation status=%s paired_days=%d improvement_bp=%.2f improvement_ci=[%.2f,%.2f] verdict=%s",
        "OK",
        int(mask.sum()),
        improvement.delta if np.isfinite(improvement.delta) else float("nan"),
        improvement.ci_low if np.isfinite(improvement.ci_low) else float("nan"),
        improvement.ci_high if np.isfinite(improvement.ci_high) else float("nan"),
        verdict,
    )
    return AugmentationSummary(
        status="OK",
        improvement_bp=improvement,
        declared_trials=int(aug_config.declared_trials),
        alpha=float(aug_config.alpha),
        verdict=verdict,
    )


def _log_headline(report: PitHaircutReport) -> None:
    ic = report.rank_ic_mean or {}
    overlap = report.pick_overlap_mean or {}
    logger.info(
        "[ALGO] stage=pit_certification status=%s paired_days=%d eod_full_bp=%.2f pit_native_bp=%.2f haircut_bp=%.2f haircut_ci=[%.2f,%.2f] ic_eod=%.4f ic_pit=%.4f overlap_native=%.3f",
        report.status.value if isinstance(report.status, PitReportStatus) else str(report.status),
        int(report.n_paired_days),
        float(report.mean_net_bp.get(_EOD_FULL, float("nan"))),
        float(report.mean_net_bp.get(_PIT_NATIVE, float("nan"))),
        float(report.haircut.delta),
        float(report.haircut.ci_low),
        float(report.haircut.ci_high),
        float(ic.get(_EOD_FULL, float("nan"))),
        float(ic.get(_PIT_NATIVE, float("nan"))),
        float(overlap.get(_PIT_NATIVE, float("nan"))),
    )


def main_pit_certification(
    *, export_dir: str, panel_dir: Path, augment: bool, train_start: pd.Timestamp | None
) -> PitHaircutReport:
    """CLI body used by src.ml.retrain --pit-certification: load inputs, run, persist, log.

    Raises:
        FileNotFoundError: price_history or panel files missing.
    """
    from src import settings
    from src.ml.research.v3_engine import load_and_prepare_price_history

    price_path = Path(settings.PRICE_HISTORY_PARQUET_PATH)
    if not price_path.exists():
        raise FileNotFoundError(f"price_history not found: {price_path}")
    panel_path = Path(panel_dir) / "pit1520_panel.parquet"
    days_path = Path(panel_dir) / "pit1520_panel_days.parquet"
    for path in (panel_path, days_path):
        if not path.exists():
            raise FileNotFoundError(f"decision-time panel file not found: {path}")
    ph, market_dates, d_to_idx = load_and_prepare_price_history(price_path)
    panel = pd.read_parquet(panel_path)
    panel_days = pd.read_parquet(days_path)
    serving_history = pd.read_parquet(price_path, columns=list(HISTORY_REQUIRED_COLUMNS))
    cfg = PitCertificationConfig(
        augmentation=AuctionNoiseConfig() if bool(augment) else None
    )
    report, daily = run_pit_certification(
        ph,
        market_dates,
        d_to_idx,
        panel=panel,
        panel_days=panel_days,
        serving_history=serving_history,
        train_start=train_start,
        config=cfg,
    )
    save_pit_haircut_report(report, daily, out_dir=Path(export_dir) / "topk_ranker")
    return report


# ---------------------------------------------------------------------------
# NXT reconstruction three-arm certification (Part 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconstructionGateConfig:
    """Adoption thresholds for the consolidated-tape reconstruction.

    Attributes:
        feature_margin_bp: Non-inferiority margin: the reconstruction may lose at most this
            much daily top-k net bp versus the hidden-exact feature set.
        min_tv_rank_correlation: Sanity floor on trade-value rank fidelity.
        min_top_k_overlap: Sanity floor on top-k pick overlap.
        max_holdout_volume_rel_err_p90: Ceiling on the holdout volume relative-error p90.
        min_holdout_coverage: Floor on holdout prediction coverage.
        max_bias_drift: Ceiling on |bias_holdout / bias_fit - 1|.
        min_coverage_improvement_bp: Minimum arm-1 minus arm-2 coverage-loss improvement.
        min_paired_days: Minimum paired days for an estimable paired delta.
        bootstrap_block_days, bootstrap_n_boot, bootstrap_seed: Day-block bootstrap of paired deltas.
    """

    feature_margin_bp: float = 5.0
    min_tv_rank_correlation: float = 0.95
    min_top_k_overlap: float = 0.90
    max_holdout_volume_rel_err_p90: float = 0.15
    min_holdout_coverage: float = 0.50
    max_bias_drift: float = 0.03
    min_coverage_improvement_bp: float = 0.0
    min_paired_days: int = 30
    bootstrap_block_days: int = 10
    bootstrap_n_boot: int = 5000
    bootstrap_seed: int = 0


def gate_config_to_dict(config: ReconstructionGateConfig) -> dict[str, float]:
    """Render gate thresholds as a JSON-serializable mapping for the certificate."""
    return {
        "feature_margin_bp": float(config.feature_margin_bp),
        "min_tv_rank_correlation": float(config.min_tv_rank_correlation),
        "min_top_k_overlap": float(config.min_top_k_overlap),
        "max_holdout_volume_rel_err_p90": float(config.max_holdout_volume_rel_err_p90),
        "min_holdout_coverage": float(config.min_holdout_coverage),
        "max_bias_drift": float(config.max_bias_drift),
        "min_coverage_improvement_bp": float(config.min_coverage_improvement_bp),
        "min_paired_days": float(config.min_paired_days),
    }


@dataclass(frozen=True)
class ReconstructionGateVerdict:
    """Adoption outcome evaluated by the report: ADOPT only when every criterion holds."""

    verdict: str
    reasons: tuple[str, ...]
    coverage_improvement: PairedDelta
    reconstruction_feature: PairedDelta
    stability_passed: bool


@dataclass(frozen=True)
class InTheLoopResult:
    """Decision-feature fidelity of reconstructed versus hidden-exact symbol-days."""

    n_symbol_days: int
    n_paired_days: int
    chg_rank_correlation: float
    tv_rank_correlation: float
    top_k_overlap: float
    feature_component: PairedDelta


@dataclass(frozen=True)
class LoopFrames:
    """In-the-loop decision inputs derived from the calibration window."""

    exact: pd.DataFrame
    recon: pd.DataFrame
    n_no_share: int


def _bootstrap_view(config: ReconstructionGateConfig) -> PitCertificationConfig:
    return PitCertificationConfig(
        bootstrap_block_days=int(config.bootstrap_block_days),
        bootstrap_n_boot=int(config.bootstrap_n_boot),
        bootstrap_seed=int(config.bootstrap_seed),
    )


def pair_certification_days(
    daily_exact: pd.DataFrame, daily_recon: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Intersect arm dates; days present in one arm only are reported and dropped from paired statistics.

    Raises:
        ValueError: Either daily frame lacks the date column.
    """
    for label, frame in (("exact", daily_exact), ("recon", daily_recon)):
        if "date" not in frame.columns:
            raise ValueError(f"pair_certification_days {label} daily missing required column 'date'")

    def _day_set(frame: pd.DataFrame) -> set[pd.Timestamp]:
        parsed = pd.to_datetime(frame["date"], errors="coerce")
        return {pd.Timestamp(d).normalize() for d in parsed.tolist() if pd.notna(d)}

    exact_days = _day_set(daily_exact)
    recon_days = _day_set(daily_recon)
    paired = sorted(exact_days & recon_days)
    dropped = sorted(exact_days ^ recon_days)
    return [d.strftime("%Y-%m-%d") for d in paired], [d.strftime("%Y-%m-%d") for d in dropped]


def paired_coverage_improvement(
    daily_exact: pd.DataFrame, daily_recon: pd.DataFrame, *, config: ReconstructionGateConfig
) -> PairedDelta:
    """Arm-1 minus arm-2 daily coverage loss (eod_full minus eod_matched top-k net bp) on paired days.

    Raises:
        ValueError: Either daily frame lacks date, arm or topk_net_bp columns.
    """
    for label, frame in (("exact", daily_exact), ("recon", daily_recon)):
        missing = [c for c in ("date", "arm", "topk_net_bp") if c not in frame.columns]
        if missing:
            raise ValueError(f"paired_coverage_improvement {label} daily missing columns: {missing}")
    paired, _dropped = pair_certification_days(daily_exact, daily_recon)

    def _daily_loss(frame: pd.DataFrame) -> dict[str, float]:
        work = frame.copy()
        work["day"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out: dict[str, float] = {}
        for day, group in work.groupby("day"):
            full = pd.to_numeric(group[group["arm"] == _EOD_FULL]["topk_net_bp"], errors="coerce").to_numpy(dtype=np.float64)
            matched = pd.to_numeric(group[group["arm"] == _EOD_MATCHED]["topk_net_bp"], errors="coerce").to_numpy(dtype=np.float64)
            if len(full) and len(matched) and np.isfinite(full[0]) and np.isfinite(matched[0]):
                out[str(day)] = float(full[0] - matched[0])
        return out

    loss_exact = _daily_loss(daily_exact)
    loss_recon = _daily_loss(daily_recon)
    improvements = [loss_exact[d] - loss_recon[d] for d in paired if d in loss_exact and d in loss_recon]
    if not improvements:
        return _nan_delta(0)
    return _paired_delta(
        np.asarray(improvements, dtype=np.float64),
        np.zeros(len(improvements), dtype=np.float64),
        config=_bootstrap_view(config),
    )


def coverage_by_year_and_basis(panel_days: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Mean superset coverage by calendar year and index basis (report-only context).

    Raises:
        ValueError: Missing date, index_basis or superset_coverage columns.
    """
    missing = [c for c in ("date", "index_basis", "superset_coverage") if c not in panel_days.columns]
    if missing:
        raise ValueError(f"coverage_by_year_and_basis panel_days missing columns: {missing}")
    work = panel_days.copy()
    work["year"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y")
    work["coverage"] = pd.to_numeric(work["superset_coverage"], errors="coerce")
    out: dict[str, dict[str, float]] = {}
    for (year, basis), group in work.groupby(["year", "index_basis"]):
        finite = group["coverage"].to_numpy(dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        out.setdefault(str(year), {})[str(basis)] = float(np.mean(finite)) if len(finite) else float("nan")
    return out


_LOOP_FRAME_COLUMNS: tuple[str, ...] = ("date", "symbol", "trade_value", "chg")
_LOOP_CALIBRATION_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "v_krx_1520",
    "v_cons_1520",
    "eod_volume",
    "close_krx_1519",
    "close_cons_1519",
)


def calibration_to_loop_frames(
    calibration: pd.DataFrame,
    *,
    config: DecompositionConfig,
    prev_closes: Mapping[tuple[str, str], float],
    score_start: str,
    score_end: str,
) -> LoopFrames:
    """Derive hidden-truth versus reconstructed decision inputs on the holdout window only.

    The exact arm uses KRX volumes and closes; the reconstruction arm predicts each day's share
    from strictly earlier per-symbol proxies (within max_gap_days, like the production predictor)
    and scales the consolidated volume. Only symbol-days in [score_start, score_end] produce
    frame rows; both frames hold the same symbol-day set where reconstruction is possible and a
    symbol-day without a prediction is counted in n_no_share and absent from both frames.

    Raises:
        ValueError: Missing calibration columns, unparseable dates, a non-finite auction
            fraction, or score_start after score_end.
    """
    missing = [c for c in _LOOP_CALIBRATION_COLUMNS if c not in calibration.columns]
    if missing:
        raise ValueError(f"calibration_to_loop_frames calibration missing columns: {missing}")
    auction_mean = float(config.auction_fraction_mean)
    if not np.isfinite(auction_mean) or auction_mean < 0.0 or auction_mean >= 1.0:
        raise ValueError(f"calibration_to_loop_frames config carries an unusable auction fraction {auction_mean!r}")
    try:
        start = pd.Timestamp(str(score_start))
        end = pd.Timestamp(str(score_end))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"calibration_to_loop_frames carries unparseable score window: {exc}") from exc
    if pd.isna(start) or pd.isna(end):
        raise ValueError("calibration_to_loop_frames carries unparseable score window")
    start, end = start.normalize(), end.normalize()
    if start > end:
        raise ValueError(f"calibration_to_loop_frames score_start {score_start!r} after score_end {score_end!r}")
    work = calibration.copy()
    stamps = pd.to_datetime(work["date"], errors="coerce", format="mixed")
    if bool(stamps.isna().any()):
        raise ValueError("calibration_to_loop_frames calibration carries unparseable dates")
    work["day"] = stamps.dt.strftime("%Y-%m-%d")
    work["ordinal_day"] = stamps.dt.normalize()
    for column in ("v_krx_1520", "v_cons_1520", "eod_volume", "close_krx_1519", "close_cons_1519"):
        work[column] = pd.to_numeric(work[column], errors="coerce").astype(np.float64)
    work["symbol"] = work["symbol"].astype(str)
    exact_rows: list[dict[str, object]] = []
    recon_rows: list[dict[str, object]] = []
    n_no_share = 0
    for symbol, group in work.groupby("symbol", sort=True):
        ordered = group.sort_values("ordinal_day", kind="stable").reset_index(drop=True)
        proxy_by_day: list[tuple[pd.Timestamp, float]] = []
        for row in ordered.to_dict(orient="records"):
            day = str(row["day"])
            day_ts = pd.Timestamp(day).normalize()
            v_krx = float(row["v_krx_1520"])
            v_cons = float(row["v_cons_1520"])
            eod = float(row["eod_volume"])
            close_krx = float(row["close_krx_1519"])
            close_cons = float(row["close_cons_1519"])
            if np.isfinite(eod) and eod > 0.0 and np.isfinite(v_cons) and v_cons > 0.0:
                proxy_by_day.append((day_ts, float(eod * (1.0 - auction_mean) / v_cons)))
            if day_ts < start or day_ts > end:
                continue
            try:
                prev = float(prev_closes.get((day, str(symbol)), float("nan")))
            except (TypeError, ValueError):
                prev = float("nan")
            exact_ok = (
                np.isfinite(v_krx) and v_krx > 0.0
                and np.isfinite(close_krx) and close_krx > 0.0
                and np.isfinite(prev) and prev > 0.0
            )
            history = pd.Series(
                [p for _, p in proxy_by_day if _ < day_ts],
                index=pd.Index([d.strftime("%Y-%m-%d") for d, _ in proxy_by_day if d < day_ts], dtype=object),
                dtype=np.float64,
            )
            share = predict_share(history, day, config=config) if len(history) else None
            recon_ok = (
                share is not None
                and np.isfinite(v_cons) and v_cons > 0.0
                and np.isfinite(close_cons) and close_cons > 0.0
                and np.isfinite(prev) and prev > 0.0
            )
            if exact_ok and recon_ok:
                assert share is not None
                exact_rows.append({
                    "date": day,
                    "symbol": str(symbol),
                    "trade_value": float(v_krx * close_krx),
                    "chg": float(close_krx / prev - 1.0),
                })
                recon_rows.append({
                    "date": day,
                    "symbol": str(symbol),
                    "trade_value": float(share * v_cons * close_cons),
                    "chg": float(close_cons / prev - 1.0),
                })
            elif exact_ok:
                n_no_share += 1
    exact = pd.DataFrame(exact_rows, columns=list(_LOOP_FRAME_COLUMNS))
    recon = pd.DataFrame(recon_rows, columns=list(_LOOP_FRAME_COLUMNS))
    return LoopFrames(exact=exact, recon=recon, n_no_share=int(n_no_share))


def reconstruction_in_the_loop(
    *,
    exact: pd.DataFrame,
    recon: pd.DataFrame,
    labels: pd.DataFrame,
    top_k: int,
    config: ReconstructionGateConfig,
) -> InTheLoopResult:
    """Compare reconstructed versus hidden-exact decision features on KRX-truth symbol-days only.

    Per paired day: Spearman correlation of chg ranks and tv ranks, top-K (by trade value) overlap,
    and the top-K rule net bp under each feature set. The feature component is the paired delta of
    reconstructed minus exact rule returns.

    Raises:
        ValueError: Missing columns, top_k below one, or duplicate (date, symbol) keys.
    """
    for label, frame in (("exact", exact), ("recon", recon)):
        missing = [c for c in _LOOP_FRAME_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"reconstruction_in_the_loop {label} missing columns: {missing}")
    missing_labels = [c for c in ("date", "symbol", "net_return") if c not in labels.columns]
    if missing_labels:
        raise ValueError(f"reconstruction_in_the_loop labels missing columns: {missing_labels}")
    if int(top_k) < 1:
        raise ValueError(f"reconstruction_in_the_loop top_k must be >= 1, got {top_k!r}")

    def _keyed(frame: pd.DataFrame, label: str) -> dict[tuple[str, str], dict[str, float]]:
        work = frame.copy()
        work["day"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        work["symbol"] = work["symbol"].astype(str)
        if bool(work.duplicated(["day", "symbol"]).any()):
            raise ValueError(f"reconstruction_in_the_loop {label} carries duplicate (date, symbol) rows")
        out: dict[tuple[str, str], dict[str, float]] = {}
        for row in work.to_dict(orient="records"):
            out[(str(row["day"]), str(row["symbol"]))] = {
                "trade_value": float(row["trade_value"]),
                "chg": float(row["chg"]),
            }
        return out

    exact_map = _keyed(exact, "exact")
    recon_map = _keyed(recon, "recon")
    label_work = labels.copy()
    label_work["day"] = pd.to_datetime(label_work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    label_work["symbol"] = label_work["symbol"].astype(str)
    label_map = {
        (str(day), str(symbol)): float(net)
        for day, symbol, net in zip(
            label_work["day"].tolist(),
            label_work["symbol"].tolist(),
            pd.to_numeric(label_work["net_return"], errors="coerce").tolist(),
            strict=True,
        )
        if np.isfinite(float(net))
    }
    paired_keys = [k for k in set(exact_map) & set(recon_map) if k in label_map]
    by_day: dict[str, list[tuple[str, str]]] = {}
    for day, symbol in paired_keys:
        by_day.setdefault(day, []).append((day, symbol))
    chg_ics: list[float] = []
    tv_ics: list[float] = []
    overlaps: list[float] = []
    recon_bp: list[float] = []
    exact_bp: list[float] = []
    n_days = 0
    for day in sorted(by_day):
        keys = by_day[day]
        if len(keys) < int(top_k):
            continue
        tv_exact = np.asarray([exact_map[k]["trade_value"] for k in keys], dtype=np.float64)
        tv_recon = np.asarray([recon_map[k]["trade_value"] for k in keys], dtype=np.float64)
        chg_exact = np.asarray([exact_map[k]["chg"] for k in keys], dtype=np.float64)
        chg_recon = np.asarray([recon_map[k]["chg"] for k in keys], dtype=np.float64)
        if not (np.isfinite(tv_exact).all() and np.isfinite(tv_recon).all() and np.isfinite(chg_exact).all() and np.isfinite(chg_recon).all()):
            continue
        chg_ics.append(_spearman(chg_exact, chg_recon))
        tv_ics.append(_spearman(tv_exact, tv_recon))
        order_exact = np.argsort(-tv_exact, kind="stable")[: int(top_k)]
        order_recon = np.argsort(-tv_recon, kind="stable")[: int(top_k)]
        picks_exact = {keys[i] for i in order_exact.tolist()}
        picks_recon = {keys[i] for i in order_recon.tolist()}
        overlaps.append(len(picks_exact & picks_recon) / float(top_k))
        exact_bp.append(float(np.mean([label_map[k] for k in picks_exact]) * 1e4))
        recon_bp.append(float(np.mean([label_map[k] for k in picks_recon]) * 1e4))
        n_days += 1
    feature = (
        _paired_delta(
            np.asarray(recon_bp, dtype=np.float64),
            np.asarray(exact_bp, dtype=np.float64),
            config=_bootstrap_view(config),
        )
        if recon_bp
        else _nan_delta(0)
    )
    return InTheLoopResult(
        n_symbol_days=len(paired_keys),
        n_paired_days=int(n_days),
        chg_rank_correlation=float(np.nanmean(chg_ics)) if chg_ics else float("nan"),
        tv_rank_correlation=float(np.nanmean(tv_ics)) if tv_ics else float("nan"),
        top_k_overlap=float(np.mean(overlaps)) if overlaps else float("nan"),
        feature_component=feature,
    )


def evaluate_backcast_transfer(
    diagnostics: FitDiagnostics, *, config: ReconstructionGateConfig
) -> CalibrationStability:
    """Score the holdout-excluded fit on the isolated holdout window in the direction of use.

    Never raises on weak evidence: a failed outcome carries the reason.
    """
    reasons: list[str] = []
    try:
        p90 = float(diagnostics.holdout_rel_err_p90)
        coverage = float(diagnostics.holdout_coverage)
        abar_fit = float(diagnostics.abar_fit)
        abar_holdout = float(diagnostics.abar_holdout)
        bias_fit = float(diagnostics.bias_fit)
        bias_holdout = float(diagnostics.bias_holdout)
        boundary = bool(diagnostics.alpha_at_boundary)
        n_rows = int(diagnostics.n_holdout_rows)
    except (TypeError, ValueError, AttributeError):
        return CalibrationStability(
            passed=False,
            rel_err_p90=float("nan"),
            coverage=float("nan"),
            bias_drift=float("nan"),
            alpha_at_boundary=False,
            n_holdout_rows=0,
            detail="diagnostics carry non-numeric transfer inputs",
        )
    finite_inputs = all(np.isfinite(v) for v in (p90, coverage, bias_fit, bias_holdout, abar_fit, abar_holdout))
    if not finite_inputs:
        reasons.append("non-finite holdout transfer inputs")
    if np.isfinite(p90) and p90 > float(config.max_holdout_volume_rel_err_p90):
        reasons.append(
            f"holdout p90 {p90:.4f} above ceiling {float(config.max_holdout_volume_rel_err_p90):.4f}"
        )
    if np.isfinite(coverage) and coverage < float(config.min_holdout_coverage):
        reasons.append(
            f"holdout coverage {coverage:.4f} below floor {float(config.min_holdout_coverage):.4f}"
        )
    if np.isfinite(bias_fit) and np.isfinite(bias_holdout) and bias_fit > 0.0:
        drift = abs(bias_holdout / bias_fit - 1.0)
    else:
        drift = float("nan")
    if np.isfinite(drift) and drift > float(config.max_bias_drift):
        reasons.append(f"bias drift {drift:.4f} above ceiling {float(config.max_bias_drift):.4f}")
    if not np.isfinite(drift):
        reasons.append("bias drift is not estimable")
    if boundary:
        reasons.append("alpha_at_boundary")
    passed = not reasons
    return CalibrationStability(
        passed=passed,
        rel_err_p90=p90,
        coverage=coverage,
        bias_drift=drift,
        alpha_at_boundary=boundary,
        n_holdout_rows=n_rows,
        detail="; ".join(reasons),
    )


def evaluate_reconstruction_gate(
    *,
    coverage_improvement: PairedDelta,
    reconstruction_feature: PairedDelta,
    stability: CalibrationStability,
    tv_rank_correlation: float,
    top_k_overlap: float,
    config: ReconstructionGateConfig,
) -> ReconstructionGateVerdict:
    """Evaluate adoption: ADOPT only when coverage improves, arm-3 is non-inferior within the
    declared margin, fidelity floors hold, and the holdout transfer passes."""
    reasons: list[str] = []
    horizon = int(config.min_paired_days)
    improvement = coverage_improvement
    if improvement.n_days < horizon:
        reasons.append(f"coverage improvement covers only {improvement.n_days} paired days (minimum {horizon})")
    elif not (np.isfinite(improvement.delta) and np.isfinite(improvement.ci_low) and np.isfinite(improvement.ci_high)):
        reasons.append("coverage improvement is not estimable")
    elif improvement.ci_low <= 0.0:
        reasons.append(
            f"coverage improvement CI [{improvement.ci_low:.2f}, {improvement.ci_high:.2f}] includes zero"
        )
    elif improvement.delta <= float(config.min_coverage_improvement_bp):
        reasons.append(
            f"coverage improvement {improvement.delta:.2f}bp at or below the minimum {float(config.min_coverage_improvement_bp):.2f}bp"
        )
    feature = reconstruction_feature
    margin = float(config.feature_margin_bp)
    if feature.n_days < horizon:
        reasons.append(f"arm-3 feature covers only {feature.n_days} paired days (minimum {horizon})")
    elif not (np.isfinite(feature.delta) and np.isfinite(feature.ci_low) and np.isfinite(feature.ci_high)):
        reasons.append("arm-3 feature is not estimable")
    elif not np.isfinite(margin):
        reasons.append("feature margin is not finite")
    elif feature.ci_low <= -margin:
        reasons.append(
            f"arm-3 feature CI [{feature.ci_low:.2f}, {feature.ci_high:.2f}] worse than margin -{margin:.2f}bp"
        )
    try:
        tv_corr = float(tv_rank_correlation)
    except (TypeError, ValueError):
        tv_corr = float("nan")
    try:
        overlap = float(top_k_overlap)
    except (TypeError, ValueError):
        overlap = float("nan")
    if not np.isfinite(tv_corr) or tv_corr < float(config.min_tv_rank_correlation):
        reasons.append(
            f"tv_rank_correlation {tv_corr:.4f} below floor {float(config.min_tv_rank_correlation):.4f}"
        )
    if not np.isfinite(overlap) or overlap < float(config.min_top_k_overlap):
        reasons.append(f"top_k_overlap {overlap:.4f} below floor {float(config.min_top_k_overlap):.4f}")
    if not bool(stability.passed):
        reasons.append(f"stability failed: {stability.detail or 'no detail'}")
    for metric, value, limit, above in (
        ("holdout p90", stability.rel_err_p90, config.max_holdout_volume_rel_err_p90, True),
        ("holdout coverage", stability.coverage, config.min_holdout_coverage, False),
        ("bias drift", stability.bias_drift, config.max_bias_drift, True),
    ):
        if not np.isfinite(value) or (value > limit if above else value < limit):
            reasons.append(f"stability {metric} violates current threshold {limit:.4f}")
    if stability.alpha_at_boundary:
        reasons.append("stability alpha_at_boundary")
    if stability.n_holdout_rows <= 0:
        reasons.append("stability has no holdout rows")
    return ReconstructionGateVerdict(
        verdict="ADOPT" if not reasons else "REJECT",
        reasons=tuple(reasons),
        coverage_improvement=improvement,
        reconstruction_feature=feature,
        stability_passed=bool(stability.passed),
    )


def _sha256_file_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _arm_identity(report: PitHaircutReport) -> dict[str, Any]:
    return {
        "strategy_id": str(report.strategy_id),
        "top_k": int(report.top_k),
        "select_universe": dict(report.select_universe),
        "feature_contract_version": str(report.feature_contract_version),
        "model_params": dict(report.model_params),
        "seeds": tuple(int(s) for s in report.seeds),
    }


def _arm_identity_detail(exact: PitHaircutReport, recon: PitHaircutReport) -> str:
    left = _arm_identity(exact)
    right = _arm_identity(recon)
    diffs = sorted(k for k in left if left[k] != right[k])
    if not diffs:
        return ""
    return f"identity fields differ: {', '.join(diffs)}"


def run_reconstruction_certification(
    *,
    daily_exact: pd.DataFrame,
    daily_recon: pd.DataFrame,
    loop: InTheLoopResult,
    n_no_share: int,
    fit_diagnostics: FitDiagnostics,
    bindings: CertificationBindings,
    arm_identity_equal: bool,
    arm_identity_detail: str = "",
    coverage: dict[str, dict[str, dict[str, float]]] | None = None,
    config: ReconstructionGateConfig | None = None,
    exact_dir: str = "",
    recon_dir: str = "",
    generated_at: str = "",
) -> ReconstructionCertification:
    """Combine the three arms into one bound adoption artifact on identical (paired) dates."""
    gate = config if config is not None else ReconstructionGateConfig()
    paired, dropped = pair_certification_days(daily_exact, daily_recon)
    improvement = paired_coverage_improvement(daily_exact, daily_recon, config=gate)
    stability = evaluate_backcast_transfer(fit_diagnostics, config=gate)
    verdict = evaluate_reconstruction_gate(
        coverage_improvement=improvement,
        reconstruction_feature=loop.feature_component,
        stability=stability,
        tv_rank_correlation=float(loop.tv_rank_correlation),
        top_k_overlap=float(loop.top_k_overlap),
        config=gate,
    )
    reasons = list(verdict.reasons)
    if not bool(arm_identity_equal):
        reasons.append(f"arms certify different configurations: {arm_identity_detail or 'identity mismatch'}")
    final = "ADOPT" if not reasons else "REJECT"
    stamp = generated_at or datetime.now(_KST).isoformat()
    fidelity = {
        "chg_rank_correlation": float(loop.chg_rank_correlation),
        "tv_rank_correlation": float(loop.tv_rank_correlation),
        "top_k_overlap": float(loop.top_k_overlap),
        "n_paired_days": float(loop.n_paired_days),
        "n_symbol_days": float(loop.n_symbol_days),
        "n_no_share": float(n_no_share),
    }
    return ReconstructionCertification(
        generated_at=str(stamp),
        exact_dir=str(exact_dir),
        recon_dir=str(recon_dir),
        paired_days=tuple(paired),
        dropped_days=tuple(dropped),
        coverage_improvement=improvement,
        reconstruction_feature=loop.feature_component,
        stability=stability,
        coverage_by_year_and_basis=dict(coverage or {}),
        bindings=bindings,
        fidelity=dict(fidelity),
        holdout_start=str(fit_diagnostics.holdout_start),
        holdout_end=str(fit_diagnostics.holdout_end),
        gate_config=gate_config_to_dict(gate),
        gate_verdict=final,
        gate_reasons=tuple(reasons),
    )


def _read_cert_daily(report_dir: Path) -> pd.DataFrame:
    path = Path(report_dir) / PIT_HAIRCUT_DAILY_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"certification daily evidence not found: {path}")
    return pd.read_parquet(path)


def _prev_close_map(price_history: pd.DataFrame) -> dict[tuple[str, str], float]:
    if "close_raw" not in price_history.columns:
        raise ValueError("_prev_close_map price_history missing required column 'close_raw'")
    missing = [c for c in ("date", "symbol") if c not in price_history.columns]
    if missing:
        raise ValueError(f"_prev_close_map price_history missing columns: {missing}")
    work = price_history.copy()
    work["day"] = pd.to_datetime(work["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    work["symbol"] = work["symbol"].astype(str)
    work["close_raw"] = pd.to_numeric(work["close_raw"], errors="coerce").astype(np.float64)
    ordered = work.sort_values(["symbol", "day"], kind="stable")
    raw = ordered["close_raw"].to_numpy(dtype=np.float64)
    prev = ordered.groupby("symbol")["close_raw"].shift(1).to_numpy(dtype=np.float64)
    out: dict[tuple[str, str], float] = {}
    for day, symbol, cur, prv in zip(
        ordered["day"].tolist(), ordered["symbol"].tolist(), raw.tolist(), prev.tolist(), strict=True
    ):
        if pd.isna(day):
            continue
        if not (np.isfinite(cur) and cur > 0.0 and np.isfinite(prv) and prv > 0.0):
            continue
        out[(str(day), str(symbol))] = float(prv)
    return out


def main_reconstruction_certification(
    *,
    exact_dir: str | Path,
    recon_dir: str | Path,
    out_dir: str | Path,
    calibration_path: str | Path,
    config_path: str | Path,
    price_history_path: str | Path,
    top_k: int | None = None,
    exact_days_path: str | Path | None = None,
    recon_days_path: str | Path | None = None,
    gate_config: ReconstructionGateConfig | None = None,
) -> ReconstructionCertification:
    """Certification entry for a panel pair: exact arm plus exact-with-reconstruction arm.

    Each report directory holds one arm's pit_haircut report and daily evidence, produced by two
    standard --pit-certification runs (exact panel, then the reconstructed panel). The holdout
    in-the-loop arm derives from the calibration table, the holdout-excluded decomposition
    config and forward labels attached from price_history here.

    Raises:
        FileNotFoundError: An arm report, the config, its fit diagnostics, the table or labels are absent.
        ValueError: The fit diagnostics table_sha256 differs from the supplied calibration table file.
    """
    from src.ml.research.v3_engine import load_and_prepare_price_history

    gate = gate_config if gate_config is not None else ReconstructionGateConfig()
    for label, path in (
        ("exact report", Path(exact_dir) / PIT_HAIRCUT_REPORT_FILENAME),
        ("recon report", Path(recon_dir) / PIT_HAIRCUT_REPORT_FILENAME),
        ("decomposition config", Path(config_path)),
        ("calibration table", Path(calibration_path)),
        ("price history", Path(price_history_path)),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"reconstruction certification {label} not found: {path}")
    fit_path = Path(config_path).parent / DECOMPOSITION_FIT_REPORT_FILENAME
    if not fit_path.exists():
        raise FileNotFoundError(f"reconstruction certification fit diagnostics not found: {fit_path}")
    diagnostics = load_fit_diagnostics(fit_path)
    table_sha = hashlib.sha256(Path(calibration_path).read_bytes()).hexdigest()
    if str(diagnostics.table_sha256) != table_sha:
        raise ValueError("reconstruction certification config was not fitted on this calibration table")
    exact_report = load_pit_haircut_report(Path(exact_dir))
    recon_report = load_pit_haircut_report(Path(recon_dir))
    if exact_report is None or recon_report is None:
        raise FileNotFoundError("reconstruction certification arm report not found")
    daily_exact = _read_cert_daily(Path(exact_dir))
    daily_recon = _read_cert_daily(Path(recon_dir))
    calibration = pd.read_parquet(calibration_path)
    decomp = load_decomposition_config(config_path)
    if (decomp.fit_start, decomp.calibrated_through, decomp.holdout_start, decomp.holdout_end) != (
        diagnostics.fit_start, diagnostics.fit_end, diagnostics.holdout_start, diagnostics.holdout_end
    ):
        raise ValueError("decomposition config and fit diagnostics window mismatch")
    if not decomp.holdout_start <= decomp.holdout_end < decomp.fit_start <= decomp.calibrated_through:
        raise ValueError("decomposition fit and holdout windows are not temporally separated")
    ph, market_dates, d_to_idx = load_and_prepare_price_history(price_history_path)
    prev_closes = _prev_close_map(ph)
    frames = calibration_to_loop_frames(
        calibration,
        config=decomp,
        prev_closes=prev_closes,
        score_start=str(decomp.holdout_start),
        score_end=str(decomp.holdout_end),
    )
    loop_keys = frames.exact[["date", "symbol"]].drop_duplicates().reset_index(drop=True)
    labeled = attach_eod_labels(loop_keys, ph, market_dates, d_to_idx, cost=PRODUCTION_STRATEGY.cost)
    loop_labels = labeled[["date", "symbol", "net_pit"]].rename(columns={"net_pit": "net_return"})
    loop = reconstruction_in_the_loop(
        exact=frames.exact,
        recon=frames.recon,
        labels=loop_labels,
        top_k=int(top_k) if top_k is not None else int(PRODUCTION_STRATEGY.top_k),
        config=gate,
    )
    identity_detail = _arm_identity_detail(exact_report, recon_report)
    bindings = CertificationBindings(
        decomposition_config_sha256=_sha256_file_bytes(Path(config_path)),
        calibration_table_sha256=table_sha,
        exact_report_sha256=_sha256_file_bytes(Path(exact_dir) / PIT_HAIRCUT_REPORT_FILENAME),
        recon_report_sha256=_sha256_file_bytes(Path(recon_dir) / PIT_HAIRCUT_REPORT_FILENAME),
    )
    coverage: dict[str, dict[str, dict[str, float]]] = {}
    if exact_days_path is not None:
        coverage["exact"] = coverage_by_year_and_basis(pd.read_parquet(exact_days_path))
    if recon_days_path is not None:
        coverage["with_reconstruction"] = coverage_by_year_and_basis(pd.read_parquet(recon_days_path))
    cert = run_reconstruction_certification(
        daily_exact=daily_exact,
        daily_recon=daily_recon,
        loop=loop,
        n_no_share=int(frames.n_no_share),
        fit_diagnostics=diagnostics,
        bindings=bindings,
        arm_identity_equal=not bool(identity_detail),
        arm_identity_detail=identity_detail,
        coverage=coverage,
        config=gate,
        exact_dir=str(exact_dir),
        recon_dir=str(recon_dir),
    )
    save_reconstruction_certification(cert, out_path=Path(out_dir) / RECONSTRUCTION_CERTIFICATION_FILENAME)
    logger.info(
        "[ALGO] stage=reconstruction_certification verdict=%s paired_days=%d dropped_days=%d n_no_share=%d reasons=%s",
        cert.gate_verdict,
        len(cert.paired_days),
        len(cert.dropped_days),
        int(frames.n_no_share),
        list(cert.gate_reasons),
    )
    return cert
