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
from scipy.stats import ttest_rel

from src import settings
from src.data.io_utils import atomic_write_parquet
from src.execution.cost_model import TICK_REFORM_DATE
from src.ml.bundle import build_inline_bundle, fit_seed_ensemble
from src.ml.costaware_topk import (
    MIN_POST_REFORM_T_STAT,
    MIN_TOP_K,
    CostStressPoint,
    RegimeMetrics,
    assert_screen_constructible,
    compute_cost_stress,
    compute_net_return,
    compute_regime_metrics,
    daily_mean_series,
    day_level_feasibility,
    select_topk_by_tick_cost,
)
from src.ml.oof import _finite_nan
from src.ml.research.v3_engine import (
    attach_forward_exit_paths,
    build_candidate_universe,
    compute_derived_features,
    load_and_prepare_price_history,
)
from src.ml.research.v3_metrics import calculate_series_metrics
from src.ml.robust_eval import CombinatorialPurgedCV, cpcv_oof_predict
from src.ml.topk_history_features import TOPK_FEATURE_COLS_V2, attach_topk_features
from src.strategy.contract import (
    COST_AWARE_UNIVERSE,
    DEFAULT_UNIVERSE,
    KCA_TOPK_CAPFREE_001,
    KCA_TOPK_COSTAWARE_001,
    MIN_PATH_WIN_RATE,
    CostSpec,
    StrategySpec,
    UniverseSpec,
    select_universe,
)

logger = logging.getLogger(__name__)

TRAIN_POOL_MIN_ROWS: int = 2000
CPCV_N_GROUPS: int = 8
CPCV_K_TEST: int = 2
# 인증·서빙 공용 단일 설정: 인접 하이퍼파라미터 6점이 모두 동일 성능 고원에 있음을 CPCV로 확인한 값
RANKER_MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.02,
    "num_leaves": 15,
    "min_child_samples": 300,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
}
# subsample<1 이라 단일 시드 결과가 임의적 → 5시드 평균으로 분산 제거
RANKER_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
RANKER_FEATURE_COLS: list[str] = list(TOPK_FEATURE_COLS_V2)
# top-3 는 매일 투자하므로 날짜 공통 드리프트는 선택과 무관 → 날짜내 차감 라벨
LABEL_MODE: str = "date_demeaned"
LABEL_CLIP: float = 0.10
CERT_REGIME_START: pd.Timestamp = pd.Timestamp(TICK_REFORM_DATE)
MIN_SCORED_FOLD_FRACTION: float = 0.90
# 개편전 반증 판정에 필요한 최소 신호일 수 (약 1년).
MIN_FALSIFICATION_DAYS: int = 250
TOPK_RANKER_BUNDLE_DIR: str = "artifacts/models/topk_ranker"


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
    falsification_status: str = ""
    falsification_reasons: list[str] = dataclasses.field(default_factory=list)
    screen_day_fractions: dict[str, float] = dataclasses.field(default_factory=dict)


def assert_nested_universe_specs(train_spec: UniverseSpec, select_spec: UniverseSpec) -> None:
    """Fail closed unless the specs differ only in the tick-cost cap.

    Args:
        train_spec: Wide training screen; its max_tick_cost_bp must be None.
        select_spec: Selection screen; its max_tick_cost_bp is None (cap-free) or finite.

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
    if sel_cap is not None and not math.isfinite(float(sel_cap)):
        raise ValueError(f"select_spec.max_tick_cost_bp must be None or a finite float, got {sel_cap!r}")
    differing = [k for k in train_d if k != "max_tick_cost_bp" and train_d[k] != select_d[k]]
    if differing:
        raise ValueError(f"universe specs are not nested; differing fields: {differing}")
    return None


def _screen_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    return float(value)


def assert_bundle_screen_parity(bundle: dict[str, Any], spec: UniverseSpec = COST_AWARE_UNIVERSE) -> None:
    screened = bundle.get("select_universe")
    if not isinstance(screened, dict):
        raise ValueError(f"bundle select_universe is not certified: got {screened!r}")
    live = dataclasses.asdict(spec)
    differing = [k for k in live if _screen_value(screened.get(k)) != _screen_value(live[k])]
    if differing:
        bundle_vals = {k: screened.get(k) for k in differing}
        live_vals = {k: live[k] for k in differing}
        raise ValueError(
            f"bundle select_universe differs from live screen in fields {differing}: "
            f"bundle={bundle_vals} live={live_vals}"
        )
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
    pool = attach_topk_features(pool, ph)
    sel_mask = select_universe(pool, select_spec)
    return pool, np.asarray(sel_mask, dtype=bool)


def attach_pit_net_label(
    cands: pd.DataFrame, *, cost: CostSpec, label_clip: float = LABEL_CLIP
) -> pd.DataFrame:
    """Attach the PIT net return and the clipped training label.

    Args:
        cands: Candidate pool with gross_return and PIT tick_cost_bp columns.
        cost: Cost specification carrying the round-trip tick multiplier.
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
    if "date" not in cands.columns:
        raise ValueError("cands missing date column for PIT statutory cost")
    # PIT 틱 비용으로 순수익 계산, 결측은 NaN 전파
    out = cands.copy()
    net = compute_net_return(
        out,
        round_trip_ticks=float(cost.round_trip_ticks),
    )
    out["net_pit"] = np.asarray(net, dtype=np.float64)
    clip = float(label_clip)
    out["train_label"] = np.clip(out["net_pit"].to_numpy(dtype=np.float64), -clip, clip)
    return out


def demean_label_by_date(
    labeled: pd.DataFrame,
    *,
    label_clip: float = LABEL_CLIP,
    date_col: str = "date",
    net_col: str = "net_pit",
    label_col: str = "train_label",
) -> pd.DataFrame:
    """Replace the training label with the date-demeaned, clipped net return.

    Args:
        labeled: Wide pool carrying the PIT net return column.
        label_clip: Symmetric clip bound applied after demeaning.
        date_col: Date column defining the cross-section.
        net_col: Net return column; left untouched for evaluation.
        label_col: Training label column to overwrite.

    Returns:
        Copy of labeled whose label_col is clip(net - mean_date(net)); NaN net stays NaN.

    Raises:
        ValueError: When date_col or net_col is missing.
    """
    missing = [c for c in (date_col, net_col) if c not in labeled.columns]
    if missing:
        raise ValueError(f"labeled missing required columns: {missing}")
    out = labeled.copy()
    net = out[net_col].astype("float64")
    # 날짜 평균은 유한 순수익 행만으로 계산 (광역풀 단면 기준)
    day_mean = net.groupby(out[date_col], sort=False).transform("mean")
    clip = float(label_clip)
    out[label_col] = (net - day_mean).clip(-clip, clip)
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
    seeds: tuple[int, ...] = RANKER_SEEDS,
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
        seeds: LightGBM seeds averaged per fold (the serving bundle uses the same).

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
        val_f = _finite_nan(val, list(feature_cols))
        reg = fit_seed_ensemble(train_full, list(feature_cols), target_col, tuple(seeds), params, float(huber_delta))
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
        picks, regime="post_reform"
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


def evaluate_falsification(
    arm: ArmMetrics, *, min_t_stat: float = MIN_POST_REFORM_T_STAT, min_days: int = MIN_FALSIFICATION_DAYS
) -> tuple[str, list[str]]:
    """Stress the cost model out-of-regime: pre-reform must not certify profit.

    Args:
        arm: Model-free full-history control arm metrics.
        min_t_stat: Significance threshold mirrored from the post-reform gate.
        min_days: Minimum pre-reform signal days for a judgement.

    Returns:
        Falsification status and the human-readable reasons.
    """
    # 개편전은 인증 표본이 아니라 역외 경제성 반증 시험이다.
    pre = arm.regimes["pre_reform"]
    if int(pre.n_days_with_signal) < int(min_days):
        return ("INSUFFICIENT_PRE_REFORM_SAMPLE", [f"pre_reform n_days_with_signal={pre.n_days_with_signal} below min_days={min_days}; falsification not evaluated"])
    if pre.mean_net_bp > 0.0 and pre.t_stat >= float(min_t_stat):
        return ("REFUTED", [f"pre_reform net is significantly positive under PIT cost: mean_net_bp={pre.mean_net_bp} t_stat={pre.t_stat}; a 46bp-cost regime cannot be profitable, so the cost model or the label is wrong"])
    return ("CONSISTENT", [])


def evaluate_ranker_verdict(
    ranker: ArmMetrics, control: ArmMetrics, evidence: PathEvidence, *, falsification_status: str = "CONSISTENT"
) -> tuple[str, list[str]]:
    """Decide the post-reform certification verdict for the ranker arm.

    Args:
        ranker: Ranker arm metrics.
        control: Model-free cost-sort control metrics at the same top_k.
        evidence: Per-fold CPCV paired evidence.
        falsification_status: Out-of-regime falsification tier outcome.

    Returns:
        Verdict string and the human-readable reasons.
    """
    reasons: list[str] = []
    if falsification_status == "REFUTED":
        reasons.append("falsification tier REFUTED: the selection rule certifies positive net return in the pre-reform cost regime")
        return ("REFUTED", reasons)
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
    seeds: tuple[int, ...] = RANKER_SEEDS,
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
            moves the certification boundary. None selects the panel minimum date.
        seeds: LightGBM seeds averaged per CPCV fold.

    Returns:
        Assembled ranker-vs-control report with regime verdict.

    Raises:
        ValueError: When spec.top_k is below the investable minimum.
    """
    if int(spec.top_k) < MIN_TOP_K:
        raise ValueError(f"top_k {spec.top_k} below the minimum investable K {MIN_TOP_K}")
    k = int(spec.top_k)
    # 기본 학습 시작은 패널 최소일. 비용이 PIT라 개편전도 올바르게 라벨링되며, 인증 경계는 split_regime_frames가 별도로 고정한다.
    eff_train_start = pd.Timestamp(pd.to_datetime(ph["date"]).min()) if train_start is None else pd.Timestamp(train_start)
    # 이중 풀 → PIT 라벨 → 인증구간 비닝+히스토리 증강 스코어 → 선택 마스크 제한 → 양 팔 평가
    pool, sel_mask = build_dual_pool(
        ph, market_dates, d_to_idx, train_spec=train_spec, select_spec=spec.universe
    )
    constructible_regimes = ("pre_reform", "post_reform") if spec.universe.max_tick_cost_bp is None else ("post_reform",)
    screen_day_fractions = assert_screen_constructible(pool.loc[sel_mask], top_k=k, regimes=constructible_regimes)
    labeled = demean_label_by_date(attach_pit_net_label(pool, cost=spec.cost))
    cert_df, hist_df = split_regime_frames(labeled, train_start=eff_train_start)
    oof = cpcv_score_with_history(
        cert_df,
        hist_df,
        RANKER_FEATURE_COLS,
        cv=cv,
        model_params=model_params,
        huber_delta=huber_delta,
        min_train_rows=min_train_rows,
        seeds=seeds,
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
    full_sel = labeled[sel_mask]
    control_full_arm = compute_arm_metrics(
        select_topk_by_tick_cost(full_sel, k), feasibility_full, arm="costsort_full", top_k=k, cost=spec.cost
    )
    falsification_status, falsification_reasons = evaluate_falsification(control_full_arm)
    verdict, verdict_reasons = evaluate_ranker_verdict(ranker_arm, control_arm, evidence, falsification_status=falsification_status)
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
        falsification_status=falsification_status,
        falsification_reasons=list(falsification_reasons),
        screen_day_fractions=dict(screen_day_fractions),
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
    ev["falsification_status"] = report.falsification_status
    ev["falsification_reasons"] = "; ".join(report.falsification_reasons)
    rows.append(ev)
    for regime, fraction in sorted(report.screen_day_fractions.items()):
        rows.append(
            {
                "row_type": "screen_constructibility",
                "arm": "select",
                "regime": regime,
                "constructible_day_fraction": float(fraction),
                "strategy_id": report.strategy_id,
                "top_k": report.top_k,
                "verdict": report.verdict,
            }
        )
    return pd.DataFrame(rows)


def train_production_bundle(
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
    *,
    spec: StrategySpec = KCA_TOPK_COSTAWARE_001,
    train_spec: UniverseSpec = DEFAULT_UNIVERSE,
    train_start: pd.Timestamp | None = None,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
    min_train_rows: int = TRAIN_POOL_MIN_ROWS,
    seeds: tuple[int, ...] = RANKER_SEEDS,
) -> dict[str, Any]:
    """Train one final production bundle on the certification-regime wide pool.

    Args:
        ph: Prepared price-history panel.
        market_dates: Full trading calendar.
        d_to_idx: Date-to-index lookup for forward exits.
        spec: Strategy specification carrying top_k, select universe and cost.
        train_spec: Wide training screen without the cost cap.
        train_start: Training-window start; None selects the panel minimum date.
        model_params: LightGBM params forwarded to build_inline_bundle; None
            selects RANKER_MODEL_PARAMS, the configuration CPCV certifies.
        huber_delta: Huber alpha for the return model.
        min_train_rows: Fail-closed floor on finite-label rows.
        seeds: Seed ensemble for the return model, matching certification.

    Returns:
        Extended bundle dict with audit provenance keys.

    Raises:
        ValueError: When spec.top_k is below MIN_TOP_K or finite rows are short.
    """
    if int(spec.top_k) < MIN_TOP_K:
        raise ValueError(f"top_k {spec.top_k} below the minimum investable K {MIN_TOP_K}")
    eff_train_start = pd.Timestamp(pd.to_datetime(ph["date"]).min()) if train_start is None else pd.Timestamp(train_start)
    # 인증구간 와이드 풀 조립 후 PIT 라벨 부착
    pool, _sel_mask = build_dual_pool(
        ph, market_dates, d_to_idx, train_spec=train_spec, select_spec=spec.universe
    )
    labeled = demean_label_by_date(attach_pit_net_label(pool, cost=spec.cost))
    cert_df, hist_df = split_regime_frames(labeled, train_start=eff_train_start)
    train_df = pd.concat([hist_df, cert_df]) if len(hist_df) else cert_df
    labels = train_df["train_label"].to_numpy(dtype=np.float64)
    fit_df = train_df[np.isfinite(labels)]
    if len(fit_df) < int(min_train_rows):
        raise ValueError(
            f"finite train_label rows {len(fit_df)} below min_train_rows {min_train_rows}"
        )
    # 인증 CPCV 와 동일 파라미터·시드로 학습 (서빙=인증 모델 정합)
    eff_params = dict(RANKER_MODEL_PARAMS) if model_params is None else dict(model_params)
    bundle = build_inline_bundle(
        fit_df,
        list(RANKER_FEATURE_COLS),
        "train_label",
        "date",
        return_model_params=eff_params,
        huber_delta=huber_delta,
        seeds=tuple(seeds),
    )
    bundle["strategy_id"] = spec.strategy_id
    bundle["top_k"] = int(spec.top_k)
    bundle["train_start"] = str(eff_train_start.date())
    bundle["certification_regime_start"] = str(CERT_REGIME_START.date())
    bundle["select_universe"] = dataclasses.asdict(spec.universe)
    bundle["label_mode"] = LABEL_MODE
    bundle["model_params"] = eff_params
    bundle["seeds"] = list(seeds)
    return bundle


def save_production_bundle(bundle: dict[str, Any], export_dir: str = TOPK_RANKER_BUNDLE_DIR) -> str:
    """Persist a production bundle under the reranker artifact directory.

    Args:
        bundle: Extended bundle dict from train_production_bundle.
        export_dir: Destination directory, distinct from champion's directory.

    Returns:
        Saved joblib path as a string.
    """
    from joblib import dump

    # 챔피언 번들과 파일명 충돌 방지용 별도 디렉터리
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
    dump(bundle, path)
    return path


def select_topk_equal_weight(
    df: pd.DataFrame, bundle: dict[str, Any], *, top_k: int, date_col: str = "date", admitted_col: str = "admitted"
) -> pd.DataFrame:
    """Select the certified top-k by point-estimate rank with equal weights.

    Args:
        df: Live snapshot with the bundle's feature columns.
        bundle: Production bundle carrying return/quantile/calibrator models.
        top_k: Names to select; must equal the certified MIN_TOP_K.
        date_col: Date column name for per-date selection.
        admitted_col: Admission flag column; only flagged rows are selectable.

    Returns:
        Top-k picks with pred, diagnostic columns and uniform allocation.

    Raises:
        ValueError: When top_k is not MIN_TOP_K, feature_cols is empty, or a
            declared feature column is missing from the snapshot.
    """
    if int(top_k) != MIN_TOP_K:
        raise ValueError(f"top_k {top_k!r} is not the certified MIN_TOP_K {MIN_TOP_K}")
    assert_bundle_screen_parity(bundle)
    feature_cols = list(bundle.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("bundle feature_cols is empty; refusing to select")
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"snapshot is missing bundle feature columns: {missing}")
    work = df.copy()
    features = work[feature_cols]
    work["pred"] = np.asarray(bundle["return_model"].predict(features), dtype=np.float64)
    q_models = bundle["quantile_models"]
    q10 = np.asarray(q_models["pred_q10"].predict(features), dtype=np.float64)
    q50 = np.asarray(q_models["pred_q50"].predict(features), dtype=np.float64)
    q90 = np.asarray(q_models["pred_q90"].predict(features), dtype=np.float64)
    work["pred_q10"] = np.minimum(np.minimum(q10, q50), q90)
    work["pred_q50"] = np.clip(q50, work["pred_q10"].to_numpy(dtype=np.float64), q90)
    work["pred_q90"] = np.maximum(q90, work["pred_q50"].to_numpy(dtype=np.float64))
    for name in ("p_good", "p_bad"):
        calibrator = bundle["calibrators"][name]
        if isinstance(calibrator, float):
            work[name] = float(calibrator)
        else:
            proba = calibrator.predict_proba(features)
            positive_idx = list(calibrator.classes_).index(True)
            work[name] = proba[:, positive_idx]
    pool = work
    if admitted_col in work.columns:
        pool = work[np.asarray(work[admitted_col], dtype=bool)]
    admitted_counts = pool.groupby(date_col, sort=False).size()
    certified_dates = admitted_counts[admitted_counts >= int(top_k)].index
    pool = pool[pool[date_col].isin(certified_dates)]
    picks = select_topk_by_score(pool, int(top_k), score_col="pred", date_col=date_col)
    picks["allocation"] = 1.0 / float(top_k)
    return picks


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
    parser.add_argument("--train-start", default=None, help="training-window start YYYY-MM-DD; augments training only and never moves the certification boundary (default: the panel minimum date)")
    parser.add_argument("--capfree", action="store_true", help="select with the cap-free universe (KCA-TOPK-CAPFREE-001) instead of the tick-cost-capped one")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    base_spec = KCA_TOPK_CAPFREE_001 if args.capfree else KCA_TOPK_COSTAWARE_001
    spec = _dataclasses.replace(base_spec, top_k=int(args.top_k)) if args.top_k is not None else base_spec
    if not os.path.exists(args.price_history):
        raise ValueError(f"price_history not found: {args.price_history}")
    ph, market_dates, d_to_idx = load_and_prepare_price_history(args.price_history)
    train_start = pd.Timestamp(args.train_start) if args.train_start else None
    report = run_topk_ranker_backtest(ph, market_dates, d_to_idx, spec=spec, train_start=train_start)
    atomic_write_parquet(topk_ranker_report_to_frame(report), Path(args.out))
    logger.info("[EVAL] stage=topk_ranker verdict=%s falsification=%s reasons=%s", report.verdict, report.falsification_status, report.verdict_reasons)


if __name__ == "__main__":  # pragma: no cover
    main()
