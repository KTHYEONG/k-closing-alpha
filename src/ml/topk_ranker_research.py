"""Top-K cost-aware ranker research harness (wide-train / cost-screened-select)."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import ttest_rel

from src import settings
from src.data.io_utils import atomic_write_parquet
from src.execution.cost_model import TICK_REFORM_DATE
from src.ml.costaware_topk import (
    MIN_POST_REFORM_T_STAT,
    MIN_TOP_K,
    CostStressPoint,
    RegimeMetrics,
    compute_cost_stress,
    compute_net_return,
    compute_regime_metrics,
    daily_mean_series,
    day_level_feasibility,
    select_topk_by_tick_cost,
)
from src.ml.oof import _finite_nan
from src.ml.research.v3_engine import (
    FEATURE_COLS,
    attach_forward_exit_paths,
    build_candidate_universe,
    compute_derived_features,
    load_and_prepare_price_history,
)
from src.ml.research.v3_metrics import calculate_series_metrics
from src.ml.robust_eval import CombinatorialPurgedCV, cpcv_oof_predict
from src.strategy.contract import (
    DEFAULT_UNIVERSE,
    KCA_TOPK_COSTAWARE_001,
    CostSpec,
    StrategySpec,
    UniverseSpec,
    select_universe,
)

logger = logging.getLogger(__name__)

TRAIN_POOL_MIN_ROWS: int = 2000
CPCV_N_GROUPS: int = 8
CPCV_K_TEST: int = 2
MIN_PATH_WIN_RATE: float = 0.60
RANKER_MODEL_PARAMS: dict[str, Any] = {"n_estimators": 60, "learning_rate": 0.03}
LABEL_CLIP: float = 0.10
CERT_REGIME_START: pd.Timestamp = pd.Timestamp(TICK_REFORM_DATE)
MIN_SCORED_FOLD_FRACTION: float = 0.90


@dataclass(frozen=True)
class YearMetrics:
    """Per-calendar-year stability record for one arm."""

    year: int
    n_days: int
    mean_net_bp: float
    median_net_bp: float
    t_stat: float
    sharpe: float


@dataclass(frozen=True)
class ArmMetrics:
    """One selection arm scored with costaware_topk primitives."""

    arm: str
    top_k: int
    regimes: dict[str, RegimeMetrics]
    by_year: list[YearMetrics]
    cost_stress: list[CostStressPoint]


@dataclass(frozen=True)
class PathEvidence:
    """Per-fold CPCV paired evidence of the ranker against the cost-sort control."""

    top_k: int
    n_paths: int
    path_win_rate: float
    mean_path_delta_bp: float
    pooled_delta_bp: float
    p_paired_t: float
    n_folds_total: int = 0
    n_folds_scored: int = 0


@dataclass(frozen=True)
class TopKRankerReport:
    """Self-describing artifact payload for the ranker-vs-control harness."""

    strategy_id: str
    top_k: int
    train_universe: dict[str, Any]
    select_universe: dict[str, Any]
    cost: dict[str, Any]
    date_min: str
    date_max: str
    n_train_rows: int
    n_select_rows: int
    ranker: ArmMetrics
    control: ArmMetrics
    path_evidence: PathEvidence
    verdict: str
    verdict_reasons: list[str]
    train_start: str = ""
    certification_regime_start: str = ""


def assert_nested_universe_specs(train_spec: UniverseSpec, select_spec: UniverseSpec) -> None:
    """Fail closed unless the specs differ only in the tick-cost cap.

    Args:
        train_spec: Wide training screen; its max_tick_cost_bp must be None.
        select_spec: Cost-capped selection screen; its max_tick_cost_bp must be finite.

    Returns:
        None when the pair is nested.

    Raises:
        ValueError: Naming every differing field, or the cap violation.
    """
    train_d = dataclasses.asdict(train_spec)
    select_d = dataclasses.asdict(select_spec)
    if train_d.get("max_tick_cost_bp") is not None:
        raise ValueError(f"train_spec.max_tick_cost_bp must be None, got {train_d.get('max_tick_cost_bp')!r}")
    sel_cap = select_d.get("max_tick_cost_bp")
    if sel_cap is None or not math.isfinite(float(sel_cap)):
        raise ValueError(f"select_spec.max_tick_cost_bp must be a finite float, got {sel_cap!r}")
    differing = [k for k in train_d if k != "max_tick_cost_bp" and train_d[k] != select_d[k]]
    if differing:
        raise ValueError(f"universe specs are not nested; differing fields: {differing}")
    return None


def build_dual_pool(
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
    *,
    train_spec: UniverseSpec,
    select_spec: UniverseSpec,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the wide train pool once and derive the cost-capped select mask.

    Args:
        ph: Prepared price-history panel.
        market_dates: Full trading calendar.
        d_to_idx: Date-to-index lookup for forward exits.
        train_spec: Wide screen without the cost cap.
        select_spec: Nested screen carrying max_tick_cost_bp.

    Returns:
        The wide train pool and the boolean select mask over that same frame.

    Raises:
        ValueError: When the spec pair is not nested.
    """
    # 넓은 학습 풀 1회 구축, 선택 마스크는 동일 프레임에서 파생
    assert_nested_universe_specs(train_spec, select_spec)
    pool, _ = build_candidate_universe(ph, train_spec)
    pool = attach_forward_exit_paths(pool, ph, market_dates, d_to_idx)
    pool = compute_derived_features(pool)
    sel_mask = select_universe(pool, select_spec)
    return pool, np.asarray(sel_mask, dtype=bool)


def attach_pit_net_label(
    cands: pd.DataFrame, *, cost: CostSpec, label_clip: float = LABEL_CLIP
) -> pd.DataFrame:
    """Attach the PIT net return and the clipped training label.

    Args:
        cands: Candidate pool with gross_return and PIT tick_cost_bp columns.
        cost: Cost specification carrying round-trip ticks and statutory bp.
        label_clip: Symmetric clip bound for the training label.

    Returns:
        Copy of cands with net_pit and train_label columns.

    Raises:
        ValueError: When tick_cost_bp or gross_return is missing.
    """
    if "tick_cost_bp" not in cands.columns:
        raise ValueError("cands missing tick_cost_bp column for PIT net label")
    if "gross_return" not in cands.columns:
        raise ValueError("cands missing gross_return column for PIT net label")
    # PIT 틱 비용으로 순수익 계산, 결측은 NaN 전파
    out = cands.copy()
    net = compute_net_return(
        out,
        round_trip_ticks=float(cost.round_trip_ticks),
        statutory_bp=float(cost.statutory_bp),
    )
    out["net_pit"] = np.asarray(net, dtype=np.float64)
    clip = float(label_clip)
    out["train_label"] = np.clip(out["net_pit"].to_numpy(dtype=np.float64), -clip, clip)
    return out


def select_topk_by_score(
    cands: pd.DataFrame, k: int, *, score_col: str = "pred", date_col: str = "date"
) -> pd.DataFrame:
    """Select the k highest-scoring candidates per date in descending order.

    Args:
        cands: Candidate pool with date and score columns.
        k: Number of names to keep per date.
        score_col: Prediction score column name.
        date_col: Date column name.

    Returns:
        Top-k picks per date in descending score order.

    Raises:
        ValueError: For k < 1 or a missing column.
    """
    if int(k) < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    if score_col not in cands.columns:
        raise ValueError(f"cands missing score_col {score_col!r}")
    if date_col not in cands.columns:
        raise ValueError(f"cands missing date_col {date_col!r}")
    ranked = cands.sort_values([date_col, score_col], ascending=[True, False], kind="stable")
    return ranked.groupby(date_col, sort=False).head(int(k)).reset_index(drop=True)


def assert_unique_date_symbol(
    df: pd.DataFrame, *, date_col: str = "date", symbol_col: str = "symbol"
) -> None:
    """Fail closed when any (date, symbol) pair repeats.

    Args:
        df: Frame to guard before a pooled top-k selection.
        date_col: Date column name.
        symbol_col: Symbol column name.

    Returns:
        None when every pair is unique.

    Raises:
        ValueError: Reporting the duplicate pair count and max multiplicity.
    """
    # CPCV 풀 중복 붕괴 방지 가드
    counts = df.groupby([date_col, symbol_col], sort=False).size()
    dup = counts[counts > 1]
    if len(dup):
        raise ValueError(
            f"duplicate (date, symbol) pairs: {len(dup)} pairs, "
            f"max multiplicity {int(dup.max())}"
        )
    return None


def dedupe_cpcv_oof(
    oof: pd.DataFrame,
    *,
    value_cols: tuple[str, ...],
    date_col: str = "date",
    symbol_col: str = "symbol",
    score_col: str = "pred",
) -> pd.DataFrame:
    """Collapse a pooled CPCV frame to one row per (date, symbol).

    Args:
        oof: Pooled out-of-fold predictions with one row per (date, symbol, fold).
        value_cols: Columns carried with the first value across folds.
        date_col: Date column name.
        symbol_col: Symbol column name.
        score_col: Score column averaged across folds.

    Returns:
        Deduplicated frame satisfying assert_unique_date_symbol.
    """
    # 폴드별 예측은 평균, 값 컬럼은 첫 값으로 축소
    agg: dict[str, str] = {score_col: "mean"}
    for col in value_cols:
        if col != score_col and col not in (date_col, symbol_col):
            agg[col] = "first"
    for col in oof.columns:
        if col not in agg and col not in (date_col, symbol_col):
            agg[col] = "first"
    return oof.groupby([date_col, symbol_col], sort=False).agg(agg).reset_index()


def score_pool_cpcv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "train_label",
    group_col: str = "date",
    cv: CombinatorialPurgedCV | None = None,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
    min_train_rows: int = TRAIN_POOL_MIN_ROWS,
) -> pd.DataFrame:
    """Score the train pool with purged CPCV out-of-fold predictions.

    Args:
        train_df: Labelled train pool.
        feature_cols: Decision-time feature columns.
        target_col: Training label column.
        group_col: Date-group column for purging.
        cv: CPCV splitter; defaults to (8, 2) with purge/embargo gaps of 1.
        model_params: LightGBM params; defaults to RANKER_MODEL_PARAMS.
        huber_delta: Huber alpha for the ranker.
        min_train_rows: Fail-closed floor on finite-target rows.

    Returns:
        Out-of-fold predictions carrying pred and cpcv_fold columns.

    Raises:
        ValueError: When finite-target rows fall below min_train_rows.
    """
    splitter = cv if cv is not None else CombinatorialPurgedCV(n_groups=CPCV_N_GROUPS, k_test=CPCV_K_TEST, purge_gap=1, embargo_gap=1)
    params = dict(RANKER_MODEL_PARAMS) if model_params is None else dict(model_params)
    # 비유한 라벨 행 제거 후 CPCV 위임
    target = train_df[target_col].to_numpy(dtype=np.float64)
    work = train_df[np.isfinite(target)]
    if len(work) < int(min_train_rows):
        raise ValueError(
            f"finite-target rows {len(work)} below min_train_rows {min_train_rows}"
        )
    return cpcv_oof_predict(
        work,
        list(feature_cols),
        target_col,
        group_col,
        cv=splitter,
        model_params=params,
        huber_delta=float(huber_delta),
    )


def split_regime_frames(
    labeled: pd.DataFrame,
    *,
    train_start: pd.Timestamp,
    cert_start: pd.Timestamp = CERT_REGIME_START,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the labelled pool into certification rows and training-only history.

    Args:
        labeled: Labelled pool with a date column.
        train_start: Training-window start; rows in [train_start, cert_start) augment training.
        cert_start: Certification boundary; rows on/after it are scored.
        date_col: Date column name.

    Returns:
        Tuple of (cert_df, hist_df); cert_df holds every row on/after cert_start
        and hist_df holds rows in [train_start, cert_start).

    Raises:
        ValueError: When date_col is missing or train_start is after cert_start.
    """
    if date_col not in labeled.columns:
        raise ValueError(f"labeled missing date_col {date_col!r}")
    train_ts = pd.Timestamp(train_start)
    cert_ts = pd.Timestamp(cert_start)
    if train_ts > cert_ts:
        raise ValueError(
            f"train_start {train_ts.date()} after cert_start {cert_ts.date()}; "
            "widening must augment training, never shrink the certification regime"
        )
    # 인증 구간은 하한 포함, 히스토리는 구간 하한 포함·상한 제외
    dates = pd.to_datetime(labeled[date_col])
    cert_df = labeled[dates >= cert_ts]
    hist_df = labeled[(dates >= train_ts) & (dates < cert_ts)]
    return cert_df, hist_df


def cpcv_score_with_history(
    cert_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target_col: str = "train_label",
    group_col: str = "date",
    cv: CombinatorialPurgedCV | None = None,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
    min_train_rows: int = TRAIN_POOL_MIN_ROWS,
) -> pd.DataFrame:
    """Score certification rows with CPCV bins, fitting on history plus fold rows.

    Args:
        cert_df: Certification-regime rows; bins are formed over this frame only.
        hist_df: Training-only history; enters fitting, never scoring.
        feature_cols: Decision-time feature columns.
        target_col: Training label column.
        group_col: Date-group column for purging.
        cv: CPCV splitter; defaults to (8, 2) with purge/embargo gaps of 1.
        model_params: LightGBM params; defaults to RANKER_MODEL_PARAMS.
        huber_delta: Huber alpha for the ranker.
        min_train_rows: Fail-closed floor on per-fold training rows.

    Returns:
        Out-of-fold predictions covering cert_df rows only, carrying pred and
        cpcv_fold with the original index preserved.

    Raises:
        ValueError: When a fold's training rows fall below min_train_rows.
    """
    splitter = cv if cv is not None else CombinatorialPurgedCV(n_groups=CPCV_N_GROUPS, k_test=CPCV_K_TEST, purge_gap=1, embargo_gap=1)
    params = dict(RANKER_MODEL_PARAMS) if model_params is None else dict(model_params)
    # 양쪽 프레임에서 비유한 라벨 제거, 인증 인덱스는 그대로 유지
    cert_target = cert_df[target_col].to_numpy(dtype=np.float64)
    cert_work = cert_df[np.isfinite(cert_target)].sort_values(group_col).copy()
    cert_work.attrs = {}
    hist_target = hist_df[target_col].to_numpy(dtype=np.float64) if len(hist_df) else np.empty(0, dtype=np.float64)
    hist_work = hist_df[np.isfinite(hist_target)] if len(hist_df) else hist_df
    parts: list[pd.DataFrame] = []
    for train_idx, test_idx, fold_id in splitter.split(cert_work[group_col]):
        train_cert = cert_work.iloc[train_idx]
        val = cert_work.iloc[test_idx]
        train_full = pd.concat([hist_work, train_cert]) if len(hist_work) else train_cert
        if len(train_full) < int(min_train_rows):
            raise ValueError(
                f"fold {int(fold_id)} training rows {len(train_full)} below min_train_rows {min_train_rows}"
            )
        train_f = _finite_nan(train_full, list(feature_cols))
        val_f = _finite_nan(val, list(feature_cols))
        reg = LGBMRegressor(
            objective="huber", alpha=float(huber_delta), random_state=42, verbosity=-1, **params
        )
        reg.fit(train_f[list(feature_cols)], train_f[target_col].to_numpy(dtype=np.float64))
        fold_df = cert_work.loc[val.index].copy()
        fold_df.attrs = {}
        fold_df["pred"] = np.asarray(reg.predict(val_f[list(feature_cols)]), dtype=np.float64)
        fold_df["cpcv_fold"] = int(fold_id)
        parts.append(fold_df)
    return pd.concat(parts)


def compute_yearly_stability(daily_net: pd.Series) -> list[YearMetrics]:
    """Compute per-calendar-year finite-masked return statistics.

    Args:
        daily_net: Daily mean net returns indexed by date.

    Returns:
        One YearMetrics per calendar year, ascending by year.
    """
    # 연도별 안정성은 합치지 않고 그대로 보고
    idx = pd.DatetimeIndex(pd.to_datetime(daily_net.index))
    out: list[YearMetrics] = []
    for year in sorted(set(idx.year)):
        vals = daily_net[idx.year == int(year)].to_numpy(dtype=np.float64)
        stats = calculate_series_metrics(vals[np.isfinite(vals)], cost_ratio=0.0)
        out.append(
            YearMetrics(
                year=int(year),
                n_days=int(stats["n"]),
                mean_net_bp=float(stats["mean_net_bp"]),
                median_net_bp=float(stats["median_net_bp"]),
                t_stat=float(stats["t_stat"]),
                sharpe=float(stats["sharpe"]),
            )
        )
    return out


def compute_arm_metrics(
    picks: pd.DataFrame,
    feasibility_full: pd.Series,
    *,
    arm: str,
    top_k: int,
    cost: CostSpec,
    value_col: str = "net_pit",
) -> ArmMetrics:
    """Score one selection arm with regime, yearly and stress metrics.

    Args:
        picks: Top-k picks carrying the value column.
        feasibility_full: Day-level feasibility over the full calendar.
        arm: Arm name (ranker or costsort).
        top_k: Names selected per date.
        cost: Cost specification for the stress grid.
        value_col: Per-pick net return column.

    Returns:
        Assembled ArmMetrics for the arm.
    """
    # 비유한 일별 값 명시적 마스킹 후 집계
    vals = picks[value_col].to_numpy(dtype=np.float64)
    finite = np.isfinite(vals)
    daily_net = daily_mean_series(picks[finite], vals[finite])
    regimes = compute_regime_metrics(daily_net, feasibility_full)
    by_year = compute_yearly_stability(daily_net)
    cost_stress = compute_cost_stress(
        picks, statutory_bp=float(cost.statutory_bp), regime="post_reform"
    )
    return ArmMetrics(
        arm=arm, top_k=int(top_k), regimes=regimes, by_year=by_year, cost_stress=cost_stress
    )


def compute_path_evidence(
    sel_oof: pd.DataFrame,
    *,
    top_k: int,
    score_col: str = "pred",
    control_col: str = "tick_cost_bp",
    value_col: str = "net_pit",
    date_col: str = "date",
    fold_col: str = "cpcv_fold",
    cert_regime_start: pd.Timestamp = CERT_REGIME_START,
) -> PathEvidence:
    """Compare the ranker against the cost-sort control fold by fold.

    Args:
        sel_oof: Select-pool out-of-fold predictions with a fold column.
        top_k: Names selected per date inside each fold.
        score_col: Ranker score column.
        control_col: Per-tick cost column for the model-free control.
        value_col: Per-pick net return column.
        date_col: Date column name.
        fold_col: CPCV fold id column.
        cert_regime_start: Certification boundary; any earlier row fails closed.

    Returns:
        PathEvidence with per-fold win rate and pooled paired statistics.

    Raises:
        ValueError: When the fold column is absent, when any row predates the
            certification regime, or when the scored-fold fraction falls below
            MIN_SCORED_FOLD_FRACTION.
    """
    if fold_col not in sel_oof.columns:
        raise ValueError(f"sel_oof missing fold_col {fold_col!r}")
    if (pd.to_datetime(sel_oof[date_col]) < pd.Timestamp(cert_regime_start)).any():
        raise ValueError(
            f"sel_oof carries pre-certification-regime rows before the "
            f"{pd.Timestamp(cert_regime_start).date()} certification regime boundary; "
            "pooling regimes into fold deltas is forbidden"
        )
    k = int(top_k)
    fold_ids = sorted(pd.unique(sel_oof[fold_col]).tolist())
    deltas: list[float] = []
    wins = 0
    # 폴드별 top-k 대결, 풀 결합 선택 금지
    for fid in fold_ids:
        fold = sel_oof[sel_oof[fold_col] == fid]
        r_picks = select_topk_by_score(fold, k, score_col=score_col, date_col=date_col)
        c_picks = select_topk_by_tick_cost(fold, k, cost_col=control_col, date_col=date_col)
        r_vals = r_picks[value_col].to_numpy(dtype=np.float64)
        c_vals = c_picks[value_col].to_numpy(dtype=np.float64)
        r_daily = daily_mean_series(r_picks[np.isfinite(r_vals)], r_vals[np.isfinite(r_vals)])
        c_daily = daily_mean_series(c_picks[np.isfinite(c_vals)], c_vals[np.isfinite(c_vals)])
        paired = pd.concat([r_daily, c_daily], axis=1, join="inner").dropna()
        diff = (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy(dtype=np.float64)
        delta_bp = float(np.mean(diff)) * 1e4 if len(diff) else float("nan")
        deltas.append(delta_bp)
        wins += int(delta_bp > 0.0)
    arr = np.asarray(deltas, dtype=np.float64)
    mean_delta = float(arr[np.isfinite(arr)].mean()) if np.isfinite(arr).any() else float("nan")
    # 중복 제거 프레임에서 날짜 기준 paired 통계
    dedup = dedupe_cpcv_oof(sel_oof, value_cols=(value_col, control_col))
    assert_unique_date_symbol(dedup, date_col=date_col)
    r_pool = select_topk_by_score(dedup, k, score_col=score_col, date_col=date_col)
    c_pool = select_topk_by_tick_cost(dedup, k, cost_col=control_col, date_col=date_col)
    rp_vals = r_pool[value_col].to_numpy(dtype=np.float64)
    cp_vals = c_pool[value_col].to_numpy(dtype=np.float64)
    rp_daily = daily_mean_series(r_pool[np.isfinite(rp_vals)], rp_vals[np.isfinite(rp_vals)])
    cp_daily = daily_mean_series(c_pool[np.isfinite(cp_vals)], cp_vals[np.isfinite(cp_vals)])
    pooled = pd.concat([rp_daily, cp_daily], axis=1, join="inner").dropna()
    p_diff = (pooled.iloc[:, 0] - pooled.iloc[:, 1]).to_numpy(dtype=np.float64)
    pooled_delta = float(np.mean(p_diff)) * 1e4 if len(p_diff) else float("nan")
    p_value = float(ttest_rel(pooled.iloc[:, 0], pooled.iloc[:, 1]).pvalue)
    n_folds_total = len(fold_ids)
    n_folds_scored = int(np.isfinite(arr).sum())
    if n_folds_total and n_folds_scored / n_folds_total < MIN_SCORED_FOLD_FRACTION:
        raise ValueError(
            f"only {n_folds_scored}/{n_folds_total} folds scored below "
            f"MIN_SCORED_FOLD_FRACTION {MIN_SCORED_FOLD_FRACTION}; "
            "refusing to certify on a thinned path set"
        )
    n_paths = n_folds_scored
    return PathEvidence(
        top_k=k,
        n_paths=n_paths,
        path_win_rate=float(wins / n_paths) if n_paths else float("nan"),
        mean_path_delta_bp=float(mean_delta),
        pooled_delta_bp=float(pooled_delta),
        p_paired_t=float(p_value),
        n_folds_total=n_folds_total,
        n_folds_scored=n_folds_scored,
    )


def evaluate_ranker_verdict(
    ranker: ArmMetrics, control: ArmMetrics, evidence: PathEvidence
) -> tuple[str, list[str]]:
    """Decide the post-reform certification verdict for the ranker arm.

    Args:
        ranker: Ranker arm metrics.
        control: Model-free cost-sort control metrics at the same top_k.
        evidence: Per-fold CPCV paired evidence.

    Returns:
        Verdict string and the human-readable reasons.
    """
    reasons: list[str] = []
    post = ranker.regimes["post_reform"]
    # 커버리지 미달은 통계 판단 이전에 차단
    if post.coverage_status != "OK":
        reasons.append(
            f"post_reform INSUFFICIENT_COVERAGE: feasible_day_fraction={post.feasible_day_fraction} "
            "below gate; statistics not evaluated"
        )
        return ("INSUFFICIENT_COVERAGE", reasons)
    stress_at_3 = next(
        (
            p
            for p in ranker.cost_stress
            if p.regime == "post_reform" and math.isclose(p.round_trip_ticks, 3.0)
        ),
        None,
    )
    ctrl_post = control.regimes["post_reform"]
    checks = {
        "post_reform_median_positive": post.median_net_bp > 0.0,
        "post_reform_t_stat": post.t_stat >= MIN_POST_REFORM_T_STAT,
        "post_reform_3tick_stress": stress_at_3 is not None and stress_at_3.passes,
        "path_win_rate": evidence.path_win_rate >= MIN_PATH_WIN_RATE,
        "ranker_beats_control": post.mean_net_bp > ctrl_post.mean_net_bp,
    }
    for name, ok in checks.items():
        if not ok:
            reasons.append(f"failed gate: {name}")
    if all(checks.values()):
        return ("PASS_POST_REFORM", reasons)
    return ("FAIL", reasons)


def run_topk_ranker_backtest(
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
    *,
    spec: StrategySpec = KCA_TOPK_COSTAWARE_001,
    train_spec: UniverseSpec = DEFAULT_UNIVERSE,
    cv: CombinatorialPurgedCV | None = None,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
    min_train_rows: int = TRAIN_POOL_MIN_ROWS,
    train_start: pd.Timestamp | None = None,
) -> TopKRankerReport:
    """Run the wide-train / cost-screened-select ranker harness over the full panel.

    Args:
        ph: Prepared price-history panel.
        market_dates: Full trading calendar.
        d_to_idx: Date-to-index lookup for forward exits.
        spec: Strategy specification carrying top_k, select universe and cost.
        train_spec: Wide training screen without the cost cap.
        cv: CPCV splitter override.
        model_params: LightGBM params override.
        huber_delta: Huber alpha for the ranker.
        min_train_rows: Fail-closed floor on finite-target rows.
        train_start: Training-window start; augments training only and never
            moves the certification boundary. None selects CERT_REGIME_START.

    Returns:
        Assembled ranker-vs-control report with regime verdict.

    Raises:
        ValueError: When spec.top_k is below the investable minimum.
    """
    if int(spec.top_k) < MIN_TOP_K:
        raise ValueError(f"top_k {spec.top_k} below the minimum investable K {MIN_TOP_K}")
    k = int(spec.top_k)
    eff_train_start = CERT_REGIME_START if train_start is None else pd.Timestamp(train_start)
    # 이중 풀 → PIT 라벨 → 인증구간 비닝+히스토리 증강 스코어 → 선택 마스크 제한 → 양 팔 평가
    pool, sel_mask = build_dual_pool(
        ph, market_dates, d_to_idx, train_spec=train_spec, select_spec=spec.universe
    )
    labeled = attach_pit_net_label(pool, cost=spec.cost)
    cert_df, hist_df = split_regime_frames(labeled, train_start=eff_train_start)
    oof = cpcv_score_with_history(
        cert_df,
        hist_df,
        FEATURE_COLS,
        cv=cv,
        model_params=model_params,
        huber_delta=huber_delta,
        min_train_rows=min_train_rows,
    )
    sel_oof = oof[sel_mask[oof.index.to_numpy()]]
    evidence = compute_path_evidence(sel_oof, top_k=k)
    dedup = dedupe_cpcv_oof(sel_oof, value_cols=("net_pit", "gross_return", "tick_cost_bp"))
    assert_unique_date_symbol(dedup)
    ranker_picks = select_topk_by_score(dedup, k)
    control_picks = select_topk_by_tick_cost(dedup, k)
    # 진입 가능 달력: D+1 봉 없는 종료일은 분모 제외
    calendar_all = pd.DatetimeIndex(pd.to_datetime(pd.Series(market_dates)).sort_values().unique())
    feasibility_full = day_level_feasibility(pool.loc[sel_mask], k, calendar_all[:-1].to_numpy())
    ranker_arm = compute_arm_metrics(
        ranker_picks, feasibility_full, arm="ranker", top_k=k, cost=spec.cost
    )
    control_arm = compute_arm_metrics(
        control_picks, feasibility_full, arm="costsort", top_k=k, cost=spec.cost
    )
    verdict, verdict_reasons = evaluate_ranker_verdict(ranker_arm, control_arm, evidence)
    dates_all = pd.to_datetime(ph["date"])
    date_min = str(dates_all.min().date()) if len(dates_all) else ""
    date_max = str(dates_all.max().date()) if len(dates_all) else ""
    return TopKRankerReport(
        strategy_id=spec.strategy_id,
        top_k=k,
        train_universe=dataclasses.asdict(train_spec),
        select_universe=dataclasses.asdict(spec.universe),
        cost=dataclasses.asdict(spec.cost),
        date_min=date_min,
        date_max=date_max,
        n_train_rows=len(pool),
        n_select_rows=int(sel_mask.sum()),
        ranker=ranker_arm,
        control=control_arm,
        path_evidence=evidence,
        verdict=verdict,
        verdict_reasons=list(verdict_reasons),
        train_start=str(eff_train_start.date()),
        certification_regime_start=str(CERT_REGIME_START.date()),
    )


def topk_ranker_report_to_frame(report: TopKRankerReport) -> pd.DataFrame:
    """Flatten a ranker report into a self-contained parquet frame.

    Args:
        report: Top-K ranker report.

    Returns:
        One row per regime, cost-stress point, year and path-evidence record.
    """
    # 행 타입별 평탄화, arm·전략·판정 식별자 부착
    rows: list[dict[str, Any]] = []
    for arm_metrics in (report.ranker, report.control):
        for metrics in arm_metrics.regimes.values():
            row: dict[str, Any] = {"row_type": "regime", "arm": arm_metrics.arm}
            row.update(dataclasses.asdict(metrics))
            row["strategy_id"] = report.strategy_id
            row["top_k"] = report.top_k
            row["verdict"] = report.verdict
            rows.append(row)
        for point in arm_metrics.cost_stress:
            srow: dict[str, Any] = {"row_type": "cost_stress", "arm": arm_metrics.arm}
            srow.update(dataclasses.asdict(point))
            srow["strategy_id"] = report.strategy_id
            srow["top_k"] = report.top_k
            srow["verdict"] = report.verdict
            rows.append(srow)
        for yearly in arm_metrics.by_year:
            yrow: dict[str, Any] = {"row_type": "by_year", "arm": arm_metrics.arm}
            yrow.update(dataclasses.asdict(yearly))
            yrow["strategy_id"] = report.strategy_id
            yrow["top_k"] = report.top_k
            yrow["verdict"] = report.verdict
            rows.append(yrow)
    ev: dict[str, Any] = {"row_type": "path_evidence", "arm": "ranker_vs_costsort"}
    ev.update(dataclasses.asdict(report.path_evidence))
    ev["strategy_id"] = report.strategy_id
    ev["top_k"] = report.top_k
    ev["verdict"] = report.verdict
    rows.append(ev)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """Run the standalone top-K ranker research CLI.

    Args:
        argv: Optional argument list for testing.

    Raises:
        ValueError: When the price_history path does not exist.
    """
    import dataclasses as _dataclasses

    parser = argparse.ArgumentParser(description="Top-K cost-aware ranker research harness")
    parser.add_argument("--price-history", default=str(settings.PRICE_HISTORY_PARQUET_PATH))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--out", default="artifacts/research/topk_ranker_report.parquet")
    parser.add_argument("--train-start", default=None, help="training-window start YYYY-MM-DD; augments training only and never moves the certification boundary (default: the certification regime start)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if args.top_k is not None:
        spec = _dataclasses.replace(KCA_TOPK_COSTAWARE_001, top_k=int(args.top_k))
    else:
        spec = KCA_TOPK_COSTAWARE_001
    if not os.path.exists(args.price_history):
        raise ValueError(f"price_history not found: {args.price_history}")
    ph, market_dates, d_to_idx = load_and_prepare_price_history(args.price_history)
    train_start = pd.Timestamp(args.train_start) if args.train_start else None
    report = run_topk_ranker_backtest(ph, market_dates, d_to_idx, spec=spec, train_start=train_start)
    atomic_write_parquet(topk_ranker_report_to_frame(report), Path(args.out))
    logger.info("[EVAL] stage=topk_ranker verdict=%s reasons=%s", report.verdict, report.verdict_reasons)


if __name__ == "__main__":  # pragma: no cover
    main()
