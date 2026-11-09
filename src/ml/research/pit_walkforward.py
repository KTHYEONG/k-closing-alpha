"""Walk-forward backtest of the closing-auction top-k strategy on the 15:20 decision-time panel.

The certified backtest screened and featurized on end-of-day values, which carry the 15:20->15:30 auction
and off-session volume that are unobservable at the decision. This harness evaluates only 15:20-native rows
built through the serving feature code. Pre-panel EOD rows may augment training, never evaluation. Every
evaluation day is scored by a model fitted on rows whose labels resolved before that day, retrained on a
fixed cadence like production, and every evaluated configuration is appended to a trial registry so the
deflated Sharpe uses the true number of trials.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data.nxt_decomposition import BASIS_KRX_RECONSTRUCTED
from src.ml.topk_contract import RANKER_FEATURE_COLS

logger = logging.getLogger(__name__)

_KST = ZoneInfo("Asia/Seoul")

TIER_EXACT: str = "exact"
TIER_RECOVERED: str = "recovered"
TIER_RECONSTRUCTED: str = "reconstructed"
TIER_EOD: str = "eod"
PIT_TIERS: tuple[str, ...] = (TIER_EXACT, TIER_RECOVERED, TIER_RECONSTRUCTED)

INDEX_FEATURE_COLS: tuple[str, ...] = ("kospi_pct", "kosdaq_pct", "v_kospi")
FEATURE_SET_BASE: str = "base"
FEATURE_SET_INDEX_LAG1: str = "idx_lag1"
FEATURE_SET_INDEX_DROP: str = "idx_drop"
FEATURE_SETS: tuple[str, ...] = (FEATURE_SET_BASE, FEATURE_SET_INDEX_LAG1, FEATURE_SET_INDEX_DROP)

TRAIN_PIT_ONLY: str = "pit_only"
TRAIN_PIT_EOD_AUG: str = "pit_eod_aug"
TRAIN_EOD_ALL: str = "eod_all"
TRAIN_SOURCES: tuple[str, ...] = (TRAIN_PIT_ONLY, TRAIN_PIT_EOD_AUG, TRAIN_EOD_ALL)

OBJECTIVE_HUBER: str = "huber"
OBJECTIVE_LAMBDARANK: str = "lambdarank"
OBJECTIVES: tuple[str, ...] = (OBJECTIVE_HUBER, OBJECTIVE_LAMBDARANK)
LAMBDARANK_GRADES: int = 5
SAMPLE_WEIGHT_COL: str = "_sample_weight"
_MODEL_EXTENSION_DEFAULTS: dict[str, Any] = {
    "extra_features": (),
    "objective": "huber",
    "train_window_days": None,
    "decay_half_life_days": None,
}

WEIGHT_EQUAL: str = "equal"
WEIGHT_INV_TICK: str = "inv_tick"
WEIGHTINGS: tuple[str, ...] = (WEIGHT_EQUAL, WEIGHT_INV_TICK)

DATASET_KEY_COLS: tuple[str, ...] = ("date", "symbol")
DATASET_LABEL_COLS: tuple[str, ...] = ("net_pit", "train_label", "exit_date")
LAG1_SUFFIX: str = "_lag1"
NEWEY_WEST_LAGS: int = 5
TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_RESEARCH_DIRNAME: str = "research/pit_walkforward"


@dataclass(frozen=True)
class Period:
    """Inclusive evaluation window."""

    name: str
    start: str
    end: str


DISCOVERY: Period = Period("discovery", "2024-01-02", "2025-12-30")
HOLDOUT: Period = Period("holdout", "2026-01-02", "2099-12-31")


@dataclass(frozen=True)
class ModelConfig:
    """Everything that changes the fitted model (one walk-forward fit per distinct value).

    Attributes:
        feature_set: FEATURE_SETS member; idx_lag1 replaces the same-day index columns by their T-1 values,
            idx_drop removes them.
        train_source: TRAIN_SOURCES member. pit_only fits 15:20 rows; pit_eod_aug adds pre-panel EOD rows;
            eod_all reproduces the deployed pipeline (EOD rows for every training date, 15:20 rows only scored).
        retrain_every_days: Evaluation days per fitted model (production retrains weekly = 5).
        embargo_days: Trading days between the last resolved training label and the block start.
        min_train_days: Distinct training dates required before a block is evaluated.
        model_params: LightGBM parameters (None = the strategy store ranker parameters).
        seeds: Seed ensemble (None = the strategy store seeds). Fitting refuses the public example store.
        huber_delta: Huber alpha.
        extra_features: Dataset columns appended to the feature set (e.g. intraday path features).
        objective: OBJECTIVES member; huber regresses the demeaned label, lambdarank ranks within each date on
            per-date label quantile grades.
        train_window_days: Rolling window in trading days before the block start (None = expanding).
        decay_half_life_days: Exponential sample-weight half-life in trading days of age (None = uniform).
    """

    feature_set: str = FEATURE_SET_BASE
    train_source: str = TRAIN_PIT_EOD_AUG
    retrain_every_days: int = 20
    embargo_days: int = 1
    min_train_days: int = 120
    model_params: tuple[tuple[str, Any], ...] | None = None
    seeds: tuple[int, ...] | None = None
    huber_delta: float = 0.9
    extra_features: tuple[str, ...] = ()
    objective: str = OBJECTIVE_HUBER
    train_window_days: int | None = None
    decay_half_life_days: float | None = None

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.train_window_days is not None and self.train_window_days < 1:
            raise ValueError(f"train_window_days must be >= 1 when set, got {self.train_window_days}")
        if self.decay_half_life_days is not None and not self.decay_half_life_days > 0:
            raise ValueError(f"decay_half_life_days must be positive when set, got {self.decay_half_life_days}")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {self.feature_set!r}")
        if self.train_source not in TRAIN_SOURCES:
            raise ValueError(f"train_source must be one of {TRAIN_SOURCES}, got {self.train_source!r}")
        if self.retrain_every_days < 1 or self.embargo_days < 0 or self.min_train_days < 1:
            raise ValueError("retrain_every_days >= 1, embargo_days >= 0 and min_train_days >= 1 are required")

    def identity(self) -> dict[str, Any]:
        """Registry identity; extension fields at their defaults are omitted so earlier ids stay stable."""
        out = dataclasses.asdict(self)
        for name, default in _MODEL_EXTENSION_DEFAULTS.items():
            if out[name] == default:
                del out[name]
        return out

    def key(self) -> str:
        return _digest(self.identity())

    def features(self) -> list[str]:
        base = feature_columns(self.feature_set)
        return [*base, *(c for c in self.extra_features if c not in base)]


@dataclass(frozen=True)
class PortfolioConfig:
    """Selection and sizing applied to one model's predictions (no refit).

    Attributes:
        top_k: Names held per day.
        weighting: WEIGHTINGS member; inv_tick weights proportional to 1 / one-tick cost.
        max_tick_cost_bp: Selection tick-cost cap (the rank pool itself is uncapped).
        abstain_below_k: Production abstains when fewer than top_k names are selectable.
        regime_lookback: When set, abstain on days whose trailing mean selectable-pool net return over the
            previous `regime_lookback` evaluated days is <= 0. Day t-1's pool return resolves at the t open,
            so the gate only uses information available at t's decision. None disables the gate.
    """

    top_k: int = 3
    weighting: str = WEIGHT_EQUAL
    max_tick_cost_bp: float = 12.0
    abstain_below_k: bool = True
    regime_lookback: int | None = None

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {self.weighting!r}")
        if not math.isfinite(self.max_tick_cost_bp) or self.max_tick_cost_bp <= 0:
            raise ValueError(f"max_tick_cost_bp must be positive, got {self.max_tick_cost_bp}")
        if self.regime_lookback is not None and self.regime_lookback < 1:
            raise ValueError(f"regime_lookback must be >= 1 when set, got {self.regime_lookback}")

    def identity(self) -> dict[str, Any]:
        """Registry identity; an unset gate is omitted so ungated trial ids stay stable."""
        out = dataclasses.asdict(self)
        if out["regime_lookback"] is None:
            del out["regime_lookback"]
        return out


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def feature_columns(feature_set: str) -> list[str]:
    """Return the model feature list of a feature set, preserving the certified column order."""
    if feature_set == FEATURE_SET_BASE:
        return list(RANKER_FEATURE_COLS)
    if feature_set == FEATURE_SET_INDEX_LAG1:
        return [f"{c}{LAG1_SUFFIX}" if c in INDEX_FEATURE_COLS else c for c in RANKER_FEATURE_COLS]
    if feature_set == FEATURE_SET_INDEX_DROP:
        return [c for c in RANKER_FEATURE_COLS if c not in INDEX_FEATURE_COLS]
    raise ValueError(f"unknown feature_set {feature_set!r}")


def attach_lagged_index(frame: pd.DataFrame, index_by_date: pd.DataFrame) -> pd.DataFrame:
    """Attach T-1 market index columns (`<col>_lag1`) from a per-date index table.

    Args:
        frame: Rows with a date column.
        index_by_date: One row per trading date with date and INDEX_FEATURE_COLS (EOD values of that date).

    Returns:
        Copy of frame with INDEX_FEATURE_COLS + LAG1_SUFFIX taken from the previous trading date.

    Raises:
        ValueError: Missing columns or duplicate dates in index_by_date.
    """
    missing = [c for c in ("date", *INDEX_FEATURE_COLS) if c not in index_by_date.columns]
    if missing:
        raise ValueError(f"index_by_date missing columns: {missing}")
    table = index_by_date[["date", *INDEX_FEATURE_COLS]].copy()
    table["date"] = pd.to_datetime(table["date"]).dt.normalize()
    if table["date"].duplicated().any():
        raise ValueError("index_by_date carries duplicate dates")
    table = table.sort_values("date").reset_index(drop=True)
    lagged = table[list(INDEX_FEATURE_COLS)].shift(1)
    lagged.columns = [f"{c}{LAG1_SUFFIX}" for c in INDEX_FEATURE_COLS]
    lagged["date"] = table["date"]
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.merge(lagged, on="date", how="left", validate="many_to_one")


def index_table_from_price_history(ph: pd.DataFrame) -> pd.DataFrame:
    """One row per date with the market-wide EOD index columns carried on every price_history row."""
    missing = [c for c in ("date", *INDEX_FEATURE_COLS) if c not in ph.columns]
    if missing:
        raise ValueError(f"price_history missing index columns: {missing}")
    work = ph[["date", *INDEX_FEATURE_COLS]].copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    return work.groupby("date", sort=True)[list(INDEX_FEATURE_COLS)].median().reset_index()


def train_tier_mask(tiers: pd.Series, train_source: str) -> np.ndarray:
    """Rows a train source may fit on: 15:20 tiers (pit_only), every row of the view (pit_eod_aug), EOD (eod_all)."""
    values = tiers.astype(str).to_numpy()
    if train_source == TRAIN_PIT_ONLY:
        return np.isin(values, PIT_TIERS)
    if train_source == TRAIN_PIT_EOD_AUG:
        return np.ones(len(values), dtype=bool)
    return np.asarray(values == TIER_EOD, dtype=bool)


def training_rows(
    dataset: pd.DataFrame, *, calendar: np.ndarray, block_start: pd.Timestamp, config: ModelConfig
) -> pd.DataFrame:
    """Rows admissible to fit the model that scores the block starting at block_start.

    A row is admissible only when its label resolved strictly before the embargo boundary, i.e. its exit
    date precedes the trading date `embargo_days` sessions before block_start. Tier membership follows
    config.train_source.

    Raises:
        ValueError: block_start absent from calendar.
    """
    days = pd.to_datetime(pd.Series(calendar)).dt.normalize().to_numpy(dtype="datetime64[ns]")
    pos = int(np.searchsorted(days, np.datetime64(pd.Timestamp(block_start).normalize())))
    if pos >= len(days) or days[pos] != np.datetime64(pd.Timestamp(block_start).normalize()):
        raise ValueError(f"block_start {block_start} is not a calendar date")
    boundary = days[max(0, pos - int(config.embargo_days))]
    exits = pd.to_datetime(dataset["exit_date"]).to_numpy(dtype="datetime64[ns]")
    resolved = ~np.isnat(exits) & (exits < boundary)
    if config.train_window_days is not None:
        window_start = days[max(0, pos - int(config.train_window_days))]
        resolved &= pd.to_datetime(dataset["date"]).to_numpy(dtype="datetime64[ns]") >= window_start
    labels = dataset["train_label"].to_numpy(dtype=np.float64)
    tier_ok = train_tier_mask(dataset["tier"], config.train_source)
    return dataset[resolved & np.isfinite(labels) & tier_ok]


def portfolio_weights(picks: pd.DataFrame, weighting: str) -> np.ndarray:
    """Normalized weights of one day's picks; inv_tick uses 1 / f_tick_cost (falls back to equal if unusable)."""
    n = len(picks)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if weighting == WEIGHT_EQUAL:
        return np.full(n, 1.0 / n)
    tick = picks["f_tick_cost"].to_numpy(dtype=np.float64)
    if not np.isfinite(tick).all() or (tick <= 0).any():
        return np.full(n, 1.0 / n)
    inv = 1.0 / tick
    return np.asarray(inv / float(inv.sum()), dtype=np.float64)


def evaluate_portfolio(scored: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    """Daily realized net returns of the model portfolio and the same-day model-free baselines.

    Args:
        scored: Evaluation rows with date, symbol, pred, net_pit, f_tick_cost (selection cap is applied here).
        config: Portfolio rules.

    Returns:
        One row per date: n_selectable, active, model_net (0 when abstaining), pool_mean (equal-weight mean of
        every selectable name), costsort_net (k cheapest ticks), picks (comma-joined symbols).
    """
    rows: list[dict[str, Any]] = []
    k = int(config.top_k)
    for day, group in scored.groupby("date", sort=True):
        tick = group["f_tick_cost"].to_numpy(dtype=np.float64)
        sel = group[
            np.isfinite(tick)
            & (tick <= float(config.max_tick_cost_bp))
            & np.isfinite(group["net_pit"].to_numpy(dtype=np.float64))
        ]
        n_sel = len(sel)
        pool_mean = float(sel["net_pit"].mean()) if n_sel else 0.0
        if n_sel == 0 or (config.abstain_below_k and n_sel < k):
            rows.append(
                {
                    "date": day,
                    "n_selectable": n_sel,
                    "active": False,
                    "model_net": 0.0,
                    "pool_mean": pool_mean,
                    "costsort_net": 0.0,
                    "picks": "",
                }
            )
            continue
        top = sel.sort_values(["pred", "symbol"], ascending=[False, True], kind="stable").head(k)
        w = portfolio_weights(top, config.weighting)
        cheap = sel.sort_values(["f_tick_cost", "symbol"], kind="stable").head(k)
        rows.append(
            {
                "date": day,
                "n_selectable": n_sel,
                "active": True,
                "model_net": float(np.dot(w, top["net_pit"].to_numpy(dtype=np.float64))),
                "pool_mean": pool_mean,
                "costsort_net": float(cheap["net_pit"].mean()),
                "picks": ",".join(top["symbol"].astype(str).tolist()),
            }
        )
    out = pd.DataFrame(rows)
    if config.regime_lookback is not None and len(out):
        lookback = int(config.regime_lookback)
        trailing = out["pool_mean"].shift(1).rolling(lookback, min_periods=max(1, lookback // 2)).mean()
        gated = trailing.notna().to_numpy() & (trailing.to_numpy() <= 0.0) & out["active"].to_numpy(dtype=bool)
        out.loc[gated, ["model_net", "costsort_net"]] = 0.0
        out.loc[gated, "active"] = False
        out.loc[gated, "picks"] = ""
    return out


def newey_west_t(x: np.ndarray, lags: int = NEWEY_WEST_LAGS) -> float:
    """t-statistic of the mean with a Bartlett-kernel HAC variance."""
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 3 or float(np.ptp(v)) == 0.0:
        return float("nan")
    d = v - v.mean()
    gamma0 = float(d @ d) / n
    var = gamma0
    for lag in range(1, min(int(lags), n - 1) + 1):
        cov = float(d[lag:] @ d[:-lag]) / n
        var += 2.0 * (1.0 - lag / (lags + 1.0)) * cov
    # Bartlett weights keep the HAC variance non-negative; a constant series was rejected above.
    return float(v.mean() / math.sqrt(var / n))


def series_metrics(returns: np.ndarray, active: np.ndarray | None = None) -> dict[str, float]:
    """Headline metrics of a daily decimal return series (bp units for means)."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"n_days": 0}
    std = float(r.std(ddof=1)) if r.size > 1 else float("nan")
    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    out = {
        "n_days": int(r.size),
        "mean_bp": float(r.mean() * 1e4),
        "median_bp": float(np.median(r) * 1e4),
        "std_bp": std * 1e4,
        "t_nw": newey_west_t(r),
        "sharpe_ann": float(r.mean() / std * math.sqrt(TRADING_DAYS_PER_YEAR)) if std and std > 0 else float("nan"),
        "hit_rate": float((r > 0).mean()),
        "mdd": float(1.0 - (wealth / peak).min()),
        "cum_log": float(np.log1p(r).sum()),
    }
    if active is not None:
        out["active_share"] = float(np.asarray(active, dtype=bool).mean())
    return out


def summarize_daily(daily: pd.DataFrame) -> dict[str, Any]:
    """Model, baseline and selection-alpha metrics of one evaluated portfolio."""
    act = daily["active"].to_numpy(dtype=bool)
    model = daily["model_net"].to_numpy(dtype=np.float64)
    pool = daily["pool_mean"].to_numpy(dtype=np.float64)
    alpha = model[act] - pool[act]
    return {
        "model": series_metrics(model, act),
        "pool": series_metrics(pool),
        "costsort": series_metrics(daily["costsort_net"].to_numpy(dtype=np.float64), act),
        "selection_alpha": {
            "mean_bp": float(alpha.mean() * 1e4) if alpha.size else float("nan"),
            "t_nw": newey_west_t(alpha),
            "n_days": int(alpha.size),
        },
    }


def _fit_and_score(
    dataset: pd.DataFrame,
    *,
    eval_days: Sequence[pd.Timestamp],
    calendar: np.ndarray,
    config: ModelConfig,
    fit_fn: Callable[[pd.DataFrame, list[str]], Any],
    progress: bool = True,
) -> pd.DataFrame:
    feats = config.features()
    missing = [c for c in feats if c not in dataset.columns]
    if missing:
        raise ValueError(f"dataset missing feature columns: {missing}")
    pit = dataset[dataset["tier"].isin(PIT_TIERS)]
    blocks = [
        list(eval_days[i : i + int(config.retrain_every_days)])
        for i in range(0, len(eval_days), int(config.retrain_every_days))
    ]
    scored_parts: list[pd.DataFrame] = []
    for b_idx, block in enumerate(blocks, start=1):
        start = pd.Timestamp(block[0])
        train = training_rows(dataset, calendar=calendar, block_start=start, config=config)
        n_train_days = int(pd.to_datetime(train["date"]).dt.normalize().nunique())
        if n_train_days < int(config.min_train_days):
            logger.info(
                "[ALGO] stage=walkforward block=%d start=%s status=SKIP train_days=%d",
                b_idx,
                start.date(),
                n_train_days,
            )
            continue
        if config.decay_half_life_days is not None:
            train = train.assign(
                **{
                    SAMPLE_WEIGHT_COL: sample_age_weights(
                        train["date"],
                        calendar=calendar,
                        block_start=start,
                        half_life_days=float(config.decay_half_life_days),
                    )
                }
            )
        model = fit_fn(train, feats)
        test = pit[pit["date"].isin(block)].copy()
        test["pred"] = np.asarray(model.predict(test[feats]), dtype=np.float64)
        test["train_cutoff"] = pd.to_datetime(train["exit_date"]).max()
        scored_parts.append(test)
        if progress:
            logger.info(
                "[ALGO] stage=walkforward block=%d/%d start=%s train_rows=%d train_days=%d test_rows=%d",
                b_idx,
                len(blocks),
                start.date(),
                len(train),
                n_train_days,
                len(test),
            )
    if not scored_parts:
        return pd.DataFrame(columns=[*DATASET_KEY_COLS, "pred"])
    return pd.concat(scored_parts, ignore_index=True)


def sample_age_weights(
    dates: pd.Series, *, calendar: np.ndarray, block_start: pd.Timestamp, half_life_days: float
) -> np.ndarray:
    """Exponential weights 0.5 ** (age / half_life) with age in trading days before block_start."""
    days = pd.to_datetime(pd.Series(calendar)).dt.normalize().to_numpy(dtype="datetime64[ns]")
    pos_start = int(np.searchsorted(days, np.datetime64(pd.Timestamp(block_start).normalize())))
    pos = np.searchsorted(days, pd.to_datetime(dates).dt.normalize().to_numpy(dtype="datetime64[ns]"))
    age = np.maximum(pos_start - pos, 0).astype(np.float64)
    return np.asarray(np.power(0.5, age / float(half_life_days)), dtype=np.float64)


def lambdarank_grades(train: pd.DataFrame, n_grades: int = LAMBDARANK_GRADES) -> np.ndarray:
    """Per-date quantile grades 0..n_grades-1 of train_label (higher = better), for the ranking objective."""
    by_date = train.groupby("date")["train_label"]
    pos = by_date.rank(method="first").to_numpy(dtype=np.float64) - 1.0
    size = by_date.transform("size").to_numpy(dtype=np.float64)
    return np.asarray(np.minimum((pos / size * n_grades).astype(np.int64), n_grades - 1), dtype=np.int64)


def default_fit_fn(config: ModelConfig) -> Callable[[pd.DataFrame, list[str]], Any]:
    """Seed-ensemble LightGBM: Huber regression as the production bundle, or a per-date LambdaRank ranker."""
    from lightgbm import LGBMRanker

    from src.ml.bundle import SeedEnsembleModel, fit_seed_ensemble
    from src.ml.oof import _finite_nan
    from src.strategy.store import require_production_store

    # Research results are compared against the deployed strategy, so placeholder values must never fit a model.
    values = require_production_store()
    params = dict(config.model_params) if config.model_params is not None else dict(values.ranker_model_params)
    seeds = tuple(config.seeds) if config.seeds is not None else tuple(values.ranker_seeds)

    def _fit(train: pd.DataFrame, feats: list[str]) -> Any:
        weights = train[SAMPLE_WEIGHT_COL].to_numpy(dtype=np.float64) if SAMPLE_WEIGHT_COL in train else None
        if config.objective == OBJECTIVE_HUBER:
            return fit_seed_ensemble(
                train, feats, "train_label", seeds, params, float(config.huber_delta), sample_weight=weights
            )
        ordered = train.sort_values("date", kind="stable")
        group_sizes = ordered.groupby("date", sort=True).size().to_numpy()
        grades = lambdarank_grades(ordered)
        x = _finite_nan(ordered, feats)[feats]
        w = ordered[SAMPLE_WEIGHT_COL].to_numpy(dtype=np.float64) if SAMPLE_WEIGHT_COL in ordered else None
        models = []
        for seed in seeds:
            ranker = LGBMRanker(objective="lambdarank", random_state=seed, verbosity=-1, **params)
            ranker.fit(x, grades, group=group_sizes, sample_weight=w)
            models.append(ranker)
        return SeedEnsembleModel(models, seeds)

    return _fit


def run_model(
    dataset: pd.DataFrame,
    *,
    period: Period,
    config: ModelConfig,
    calendar: np.ndarray | None = None,
    fit_fn: Callable[[pd.DataFrame, list[str]], Any] | None = None,
) -> pd.DataFrame:
    """Walk-forward predictions of one model configuration over a period's 15:20 evaluation days.

    Returns:
        Scored 15:20 rows (date, symbol, tier, features, net_pit, f_tick_cost, pred, train_cutoff).
    """
    pit_dates = pd.to_datetime(dataset.loc[dataset["tier"].isin(PIT_TIERS), "date"]).dt.normalize()
    lo, hi = pd.Timestamp(period.start), pd.Timestamp(period.end)
    eval_days = sorted(d for d in pit_dates.unique() if lo <= d <= hi)
    if calendar is None:
        calendar = np.array(sorted(pd.to_datetime(dataset["date"]).dt.normalize().unique()))
    fit = fit_fn if fit_fn is not None else default_fit_fn(config)
    return _fit_and_score(dataset, eval_days=eval_days, calendar=calendar, config=config, fit_fn=fit)


def run_cpcv(
    dataset: pd.DataFrame,
    *,
    start: str,
    end: str,
    config: ModelConfig,
    n_groups: int = 8,
    k_test: int = 2,
    embargo_groups: int = 1,
    fit_fn: Callable[[pd.DataFrame, list[str]], Any] | None = None,
) -> pd.DataFrame:
    """Combinatorial purged CV over the 15:20 days in [start, end], the certified methodology on PIT rows.

    Days are split into n_groups contiguous groups; every k_test-subset is a test fold. Training excludes the
    test groups, `embargo_groups` neighbours on each side, rows dated after `end`, and rows whose label resolves
    inside a test group. A day's prediction is the mean over the folds that test it.

    Returns:
        Scored 15:20 rows like run_model (pred averaged across folds) plus n_folds.

    Raises:
        ValueError: Fewer days than groups or invalid k_test.
    """
    from itertools import combinations

    if not 1 <= int(k_test) < int(n_groups):
        raise ValueError(f"k_test must satisfy 1 <= k_test < n_groups, got {k_test}")
    feats = config.features()
    fit = fit_fn if fit_fn is not None else default_fit_fn(config)
    dates = pd.to_datetime(dataset["date"]).dt.normalize()
    is_pit = dataset["tier"].isin(PIT_TIERS).to_numpy()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    days = np.array(sorted(d for d in dates[is_pit].unique() if lo <= d <= hi))
    if len(days) < int(n_groups):
        raise ValueError(f"run_cpcv needs at least n_groups days, got {len(days)}")
    groups = np.array_split(days, int(n_groups))
    spans = [(g[0], g[-1]) for g in groups]
    group_of = pd.Series(np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)]), index=days)
    row_group = dates.map(group_of)
    exits = pd.to_datetime(dataset["exit_date"])
    labels = dataset["train_label"].to_numpy(dtype=np.float64)
    tier_ok = train_tier_mask(dataset["tier"], config.train_source)
    base_ok = tier_ok & np.isfinite(labels) & (dates <= hi).to_numpy() & exits.notna().to_numpy()
    parts: list[pd.DataFrame] = []
    for fold, test_ids in enumerate(combinations(range(int(n_groups)), int(k_test)), start=1):
        banned = {j for i in test_ids for j in range(i - int(embargo_groups), i + int(embargo_groups) + 1)}
        in_banned = row_group.isin(banned).to_numpy()
        purge = np.zeros(len(dataset), dtype=bool)
        for i in test_ids:
            purge |= ((exits >= spans[i][0]) & (dates <= spans[i][1])).to_numpy()
        train = dataset[base_ok & ~in_banned & ~purge]
        model = fit(train, feats)
        test = dataset[is_pit & row_group.isin(set(test_ids)).to_numpy()].copy()
        test["pred"] = np.asarray(model.predict(test[feats]), dtype=np.float64)
        test["fold"] = fold
        parts.append(test)
        logger.info(
            "[ALGO] stage=cpcv fold=%d test_groups=%s train_rows=%d test_rows=%d", fold, test_ids, len(train), len(test)
        )
    scored = pd.concat(parts, ignore_index=True)
    keep = [c for c in scored.columns if c not in ("pred", "fold")]
    agg = (
        scored.groupby(["date", "symbol"], sort=True)
        .agg(pred=("pred", "mean"), n_folds=("fold", "nunique"))
        .reset_index()
    )
    first = scored.drop_duplicates(["date", "symbol"])[keep]
    return first.merge(agg, on=["date", "symbol"], how="inner", validate="one_to_one")


def breakdown(daily: pd.DataFrame, scored: pd.DataFrame) -> dict[str, Any]:
    """Per-year and per-tier (share of reconstructed rows among picks) model and pool means in bp."""
    out: dict[str, Any] = {}
    d = daily.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    for year, g in d.groupby("year"):
        a = g["active"].to_numpy(dtype=bool)
        out[f"year_{year}"] = {
            "model_bp": float(g["model_net"].mean() * 1e4),
            "pool_bp": float(g["pool_mean"].mean() * 1e4),
            "alpha_bp": float((g.loc[a, "model_net"] - g.loc[a, "pool_mean"]).mean() * 1e4)
            if a.any()
            else float("nan"),
            "n_days": len(g),
        }
    tiers = scored.groupby("date")["tier"].agg(lambda s: bool((s == TIER_RECONSTRUCTED).any()))
    d["has_recon"] = d["date"].map(tiers).fillna(False).astype(bool)
    for flag, g in d.groupby("has_recon"):
        out[f"recon_day_{bool(flag)}"] = {"model_bp": float(g["model_net"].mean() * 1e4), "n_days": len(g)}
    return out


class TrialRegistry:
    """Append-only JSONL log of every evaluated (model, portfolio, period) trial."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def n_trials(self, period: str) -> int:
        return len({r["trial_id"] for r in self.load() if r.get("period") == period})


def trial_id(model: ModelConfig, portfolio: PortfolioConfig) -> str:
    return _digest({"model": model.identity(), "portfolio": portfolio.identity()})


def deflated_sharpe(returns: np.ndarray, n_trials: int) -> float:
    """Deflated Sharpe ratio (probability the true Sharpe exceeds the best-of-n_trials null)."""
    from src.ml.robust_eval import deflated_sharpe_ratio

    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 20 or float(r.std(ddof=1)) <= 0:
        return float("nan")
    return float(deflated_sharpe_ratio(r, n_independent_trials=max(1, int(n_trials))))


def record_trial(
    registry: TrialRegistry,
    *,
    model: ModelConfig,
    portfolio: PortfolioConfig,
    period: Period,
    daily: pd.DataFrame,
    scored: pd.DataFrame,
    daily_dir: Path,
) -> dict[str, Any]:
    """Summarize, persist the daily series and append one registry record."""
    tid = trial_id(model, portfolio)
    summary = summarize_daily(daily)
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_path = daily_dir / f"{period.name}_{tid}.parquet"
    daily.to_parquet(daily_path, index=False)
    record = {
        "trial_id": tid,
        "period": period.name,
        "period_start": period.start,
        "period_end": period.end,
        "model": dataclasses.asdict(model),
        "portfolio": dataclasses.asdict(portfolio),
        "summary": summary,
        "breakdown": breakdown(daily, scored),
        "daily_path": str(daily_path),
        "created_at": datetime.now(_KST).isoformat(),
    }
    registry.append(record)
    return record


def build_dataset(  # pragma: no cover - multi-year store orchestration, exercised by the CLI research runs
    *,
    panel: pd.DataFrame,
    panel_days: pd.DataFrame,
    recovered_keys: frozenset[tuple[str, str]],
    min_day_coverage: float,
    eod_until: pd.Timestamp,
    spec: Any = None,
) -> pd.DataFrame:
    """Assemble 15:20 rows (serving features, tiered) and EOD rows (pre-panel and full, for training arms).

    Args:
        panel: Decision-time panel (PIT1520_PANEL_COLUMNS, optionally with a reconstruction `basis`).
        panel_days: Panel day sidecar.
        recovered_keys: (YYYY-MM-DD, symbol) symbol-days whose bars came from the off-session recovery.
        min_day_coverage: Bar days need superset_coverage at least this.
        eod_until: EOD rows dated on/after this are tagged eod but used only by the eod_all train source.
        spec: Strategy whose screen defines the rank pool (None = PRODUCTION_STRATEGY); research only.

    Returns:
        Frame with date, symbol, tier, RANKER_FEATURE_COLS, index lag columns, net_pit, train_label, exit_date.
    """
    from src import settings
    from src.ml.research.pit_certification import (
        PitCertificationConfig,
        _with_screenable_class,
        attach_eod_labels,
        build_pit_feature_frames,
        select_usable_panel_days,
    )
    from src.ml.research.v3_engine import load_and_prepare_price_history
    from src.ml.topk_history_features import HISTORY_REQUIRED_COLUMNS
    from src.ml.topk_ranker_research import (
        attach_pit_net_label,
        build_dual_pool,
        demean_label_by_date,
    )
    from src.strategy.contract import PRODUCTION_STRATEGY, training_universe
    from src.strategy.store import require_production_store

    spec = PRODUCTION_STRATEGY if spec is None else spec
    clip = float(require_production_store().label_clip)
    ph, market_dates, d_to_idx = load_and_prepare_price_history(settings.PRICE_HISTORY_PARQUET_PATH)
    pool, sel_mask = build_dual_pool(
        ph, market_dates, d_to_idx, train_spec=training_universe(spec.universe), select_spec=spec.universe
    )
    eod = demean_label_by_date(attach_pit_net_label(pool, cost=spec.cost), label_clip=clip)
    serving_history = pd.read_parquet(settings.PRICE_HISTORY_PARQUET_PATH, columns=list(HISTORY_REQUIRED_COLUMNS))
    work_panel = _with_screenable_class(panel, ph) if spec.universe.exclude_non_screenable_class else panel
    cert_dates = sorted(pd.to_datetime(eod["date"]).dt.normalize().unique())
    usable = select_usable_panel_days(
        panel_days, cert_dates=cert_dates, config=PitCertificationConfig(min_day_coverage=float(min_day_coverage))
    )
    logger.info("[DATA] stage=pit_dataset usable_days=%d of panel_days=%d", len(usable), len(panel_days))
    _feature_arm, native = build_pit_feature_frames(
        panel=work_panel,
        usable_days=usable,
        eod_pool=eod,
        sel_mask=np.asarray(sel_mask, dtype=bool),
        serving_history=serving_history,
        spec=spec,
    )
    labeled = attach_eod_labels(native[["date", "symbol"]], ph, market_dates, d_to_idx, cost=spec.cost)
    labeled = demean_label_by_date(labeled, label_clip=clip)
    native = native.merge(
        labeled[["date", "symbol", "net_pit", "train_label", "exit_date"]],
        on=["date", "symbol"],
        how="inner",
        validate="one_to_one",
    )
    basis = panel[["date", "symbol"]].copy()
    basis["date"] = pd.to_datetime(basis["date"]).dt.normalize()
    basis["symbol"] = basis["symbol"].astype(str)
    basis["recon"] = (
        panel["basis"].astype(str).eq(BASIS_KRX_RECONSTRUCTED).to_numpy() if "basis" in panel.columns else False
    )
    native = native.merge(basis, on=["date", "symbol"], how="left", validate="one_to_one")
    keys = list(zip(native["date"].dt.strftime("%Y-%m-%d"), native["symbol"].astype(str), strict=True))
    tier = np.where(native["recon"].fillna(False).astype(bool), TIER_RECONSTRUCTED, TIER_EXACT)
    tier = np.where([k in recovered_keys for k in keys], TIER_RECOVERED, tier)
    native["tier"] = tier
    native = native.drop(columns=["recon"])
    eod_rows = eod[["date", "symbol", *RANKER_FEATURE_COLS, "net_pit", "train_label", "exit_date"]].copy()
    eod_rows["date"] = pd.to_datetime(eod_rows["date"]).dt.normalize()
    eod_rows["tier"] = TIER_EOD
    eod_rows["pre_panel"] = eod_rows["date"] < pd.Timestamp(eod_until)
    native["pre_panel"] = False
    out = pd.concat([native.drop(columns=["selectable", "index_basis"], errors="ignore"), eod_rows], ignore_index=True)
    out = attach_lagged_index(out, index_table_from_price_history(ph))
    logger.info(
        "[DATA] stage=pit_dataset rows=%d tiers=%s",
        len(out),
        out["tier"].value_counts().to_dict(),
    )
    return out


def recovered_bars_loader(recovered_dir: Path | None) -> Callable[[str], pd.DataFrame]:
    """Loader of off-session-recovered Toss regular bars (one parquet per date, canonical bar columns)."""

    def _load(day: str) -> pd.DataFrame:
        if recovered_dir is None:
            return pd.DataFrame()
        path = Path(recovered_dir) / f"{day}.parquet"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        return frame if len(frame.columns) else pd.DataFrame()

    return _load


def build_research_panel(  # pragma: no cover - multi-year store orchestration, exercised by the CLI research runs
    *,
    start: str,
    end: str,
    out_dir: Path,
    recovered_dir: Path | None,
    recovered_consolidated_dir: Path | None = None,
    reconstruct_consolidated: bool = True,
) -> pd.DataFrame:
    """Build the 15:20 panel from stored bars plus recovered Toss bars, with NXT-era KRX reconstruction.

    Bars only (no live capture): on live days the bar path reproduces the live snapshot exactly
    (measured 2026-09-15..11-05: close and high identical on every overlapping row).

    Returns:
        The recovered (date, symbol) keys actually present in the panel (also written as recovered_keys.parquet).
    """
    from src import settings
    from src.config.collection import CollectionSettings
    from src.data.eod_superset import EodSupersetScreen
    from src.data.nxt_decomposition import load_decomposition_config
    from src.data.panel_integrity import load_price_panel
    from src.data.pit1520_panel import (
        _BAR_REQUIRED_COLUMNS,
        PanelSourcePolicy,
        Pit1520PanelConfig,
        _shift_minutes,
        build_pit1520_panel,
        default_decomposition_config_path,
        known_consolidated_symbol_days,
        load_consolidated_bars,
        load_regular_bars,
    )

    config = Pit1520PanelConfig(reconstruct_consolidated=bool(reconstruct_consolidated))
    load_recovered = recovered_bars_loader(recovered_dir)
    floor, ceil = int(config.floor_hhmmss), _shift_minutes(int(config.cutoff_hhmmss), 1)
    recovered_seen: list[tuple[str, str]] = []

    def _bars(day: str) -> pd.DataFrame:
        base = load_regular_bars(day, config=config)
        extra = load_recovered(day)
        if extra.empty:
            return base
        extra = extra[list(_BAR_REQUIRED_COLUMNS)]
        extra = extra[(extra["ts_hms"].astype(int) >= floor) & (extra["ts_hms"].astype(int) <= ceil)]
        present = set(base["symbol"].astype(str)) if len(base) else set()
        extra = extra[~extra["symbol"].astype(str).isin(present)]
        recovered_seen.extend((day, s) for s in sorted(set(extra["symbol"].astype(str))))
        return pd.concat([base, extra], ignore_index=True) if len(base) else extra.reset_index(drop=True)

    load_recovered_cons = recovered_bars_loader(recovered_consolidated_dir)

    def _cons(day: str) -> pd.DataFrame:
        base = load_consolidated_bars(day, config=config)
        extra = load_recovered_cons(day)
        if extra.empty:
            return base
        extra = extra[list(_BAR_REQUIRED_COLUMNS)]
        extra = extra[(extra["ts_hms"].astype(int) >= floor) & (extra["ts_hms"].astype(int) <= ceil)]
        present = set(base["symbol"].astype(str)) if len(base) else set()
        extra = extra[~extra["symbol"].astype(str).isin(present)]
        return pd.concat([base, extra], ignore_index=True) if len(base) else extra.reset_index(drop=True)

    decomp = load_decomposition_config(default_decomposition_config_path()) if reconstruct_consolidated else None
    price_history, _prov = load_price_panel(settings.PRICE_HISTORY_PARQUET_PATH)
    calendar = sorted(pd.to_datetime(price_history["date"]).dt.normalize().unique().tolist())
    wanted = [
        pd.Timestamp(d).strftime("%Y-%m-%d") for d in calendar if start <= pd.Timestamp(d).strftime("%Y-%m-%d") <= end
    ]
    result = build_pit1520_panel(
        price_history=price_history,
        dates=wanted,
        bars_loader=_bars,
        live_loader=lambda _day: None,
        screen=EodSupersetScreen.from_profile(CollectionSettings()),
        config=config,
        source_policy=PanelSourcePolicy.BARS_ONLY,
        consolidated_loader=_cons if reconstruct_consolidated else None,
        decomposition_config=decomp,
        consolidated_symbol_days=known_consolidated_symbol_days() if reconstruct_consolidated else None,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result.panel.to_parquet(out_dir / "pit1520_panel.parquet", index=False)
    result.days.to_parquet(out_dir / "pit1520_panel_days.parquet", index=False)
    result.exclusions.to_parquet(out_dir / "pit1520_panel_exclusions.parquet", index=False)
    in_panel = set(
        zip(
            pd.to_datetime(result.panel["date"]).dt.strftime("%Y-%m-%d"),
            result.panel["symbol"].astype(str),
            strict=True,
        )
    )
    keys = pd.DataFrame(sorted(k for k in set(recovered_seen) if k in in_panel), columns=["date", "symbol"])
    keys.to_parquet(out_dir / "recovered_keys.parquet", index=False)
    logger.info(
        "[DATA] stage=research_panel dates=%d rows=%d excluded=%d recovered_in_panel=%d",
        len(wanted),
        len(result.panel),
        len(result.exclusions),
        len(keys),
    )
    return keys


def restrict_augmentation(dataset: pd.DataFrame) -> pd.DataFrame:
    """Drop EOD rows inside the panel era from the pit_eod_aug view (they would leak post-15:20 values)."""
    keep = (dataset["tier"] != TIER_EOD) | dataset["pre_panel"].astype(bool)
    return dataset[keep]


def model_view(dataset: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    """The dataset rows a model configuration may see (eod_all keeps every EOD row, others only pre-panel EOD)."""
    if config.train_source == TRAIN_EOD_ALL:
        return dataset
    return restrict_augmentation(dataset)


def default_grid() -> tuple[list[ModelConfig], list[PortfolioConfig]]:
    """The pre-registered discovery grid: feature set x train source, then K x weighting x tick cap."""
    models = [
        ModelConfig(feature_set=f, train_source=t) for f in FEATURE_SETS for t in (TRAIN_PIT_ONLY, TRAIN_PIT_EOD_AUG)
    ]
    models.append(ModelConfig(feature_set=FEATURE_SET_BASE, train_source=TRAIN_EOD_ALL))
    portfolios = [
        PortfolioConfig(top_k=k, weighting=w, max_tick_cost_bp=c)
        for k in (3, 5)
        for w in WEIGHTINGS
        for c in (12.0, 15.0)
    ]
    return models, portfolios


def research_dir() -> Path:  # pragma: no cover - settings path wiring
    from src import settings

    return Path(settings.HISTORY_DIR) / DEFAULT_RESEARCH_DIRNAME


def run_grid(
    dataset: pd.DataFrame,
    *,
    period: Period,
    models: Sequence[ModelConfig],
    portfolios: Sequence[PortfolioConfig],
    out_dir: Path,
    fit_fn: Callable[[pd.DataFrame, list[str]], Any] | None = None,
) -> list[dict[str, Any]]:
    """Fit each model once over the period, evaluate every portfolio on its predictions, register all trials."""
    registry = TrialRegistry(out_dir / "trials.jsonl")
    records: list[dict[str, Any]] = []
    calendar = np.array(sorted(pd.to_datetime(dataset["date"]).dt.normalize().unique()))
    for m_idx, model in enumerate(models, start=1):
        pred_path = out_dir / "preds" / f"{period.name}_{model.key()}.parquet"
        if pred_path.exists():
            scored = pd.read_parquet(pred_path)
        else:
            logger.info("[ALGO] stage=walkforward model=%d/%d config=%s", m_idx, len(models), dataclasses.asdict(model))
            scored = run_model(
                model_view(dataset, model), period=period, config=model, calendar=calendar, fit_fn=fit_fn
            )
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            scored.to_parquet(pred_path, index=False)
        for portfolio in portfolios:
            daily = evaluate_portfolio(scored, portfolio)
            rec = record_trial(
                registry,
                model=model,
                portfolio=portfolio,
                period=period,
                daily=daily,
                scored=scored,
                daily_dir=out_dir / "daily",
            )
            records.append(rec)
            s = rec["summary"]
            logger.info(
                "[ALGO] stage=trial id=%s period=%s feat=%s train=%s k=%d w=%s cap=%.0f mean_bp=%.2f t=%.2f alpha_bp=%.2f pool_bp=%.2f",
                rec["trial_id"],
                period.name,
                model.feature_set,
                model.train_source,
                portfolio.top_k,
                portfolio.weighting,
                portfolio.max_tick_cost_bp,
                s["model"].get("mean_bp", float("nan")),
                s["model"].get("t_nw", float("nan")),
                s["selection_alpha"]["mean_bp"],
                s["pool"].get("mean_bp", float("nan")),
            )
    return records


def leaderboard(out_dir: Path, period: str) -> pd.DataFrame:
    """Latest record per trial for a period with the deflated Sharpe over all trials of that period."""
    registry = TrialRegistry(out_dir / "trials.jsonl")
    records = {r["trial_id"]: r for r in registry.load() if r.get("period") == period}
    n = len(records)
    rows = []
    for tid, r in records.items():
        daily = pd.read_parquet(r["daily_path"])
        s = r["summary"]
        rows.append(
            {
                "trial_id": tid,
                "feature_set": r["model"]["feature_set"],
                "train_source": r["model"]["train_source"],
                "top_k": r["portfolio"]["top_k"],
                "weighting": r["portfolio"]["weighting"],
                "cap_bp": r["portfolio"]["max_tick_cost_bp"],
                "gate": r["portfolio"].get("regime_lookback"),
                "mean_bp": s["model"].get("mean_bp"),
                "median_bp": s["model"].get("median_bp"),
                "t_nw": s["model"].get("t_nw"),
                "sharpe": s["model"].get("sharpe_ann"),
                "mdd": s["model"].get("mdd"),
                "active": s["model"].get("active_share"),
                "alpha_bp": s["selection_alpha"]["mean_bp"],
                "alpha_t": s["selection_alpha"]["t_nw"],
                "pool_bp": s["pool"].get("mean_bp"),
                "costsort_bp": s["costsort"].get("mean_bp"),
                "dsr": deflated_sharpe(daily["model_net"].to_numpy(dtype=np.float64), n),
                "n_trials": n,
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI wiring
    from src.utils.cli_logging import configure_cli_logging

    parser = argparse.ArgumentParser(description="15:20 walk-forward backtest of the top-k closing strategy")
    sub = parser.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("panel", help="build the 15:20 panel from stored + recovered bars")
    pp.add_argument("--start", default="2023-01-02")
    pp.add_argument("--end", required=True)
    pp.add_argument("--recovered-dir", default=None)
    pp.add_argument("--recovered-consolidated-dir", default=None)
    pp.add_argument("--no-reconstruct", action="store_true")
    pp.add_argument("--out-name", default="panel", help="subdirectory under the research dir")
    b = sub.add_parser("build", help="assemble the tiered 15:20 + EOD dataset")
    b.add_argument("--panel-dir", required=True)
    b.add_argument("--recovered-keys", default=None, help="parquet with date, symbol of off-session-recovered bars")
    b.add_argument("--min-day-coverage", type=float, default=0.80)
    b.add_argument("--eod-until", default="2023-01-25")
    g = sub.add_parser("grid", help="run the pre-registered grid on a period")
    g.add_argument("--period", choices=("discovery", "holdout"), default="discovery")
    g.add_argument("--retrain-every", type=int, default=20)
    one = sub.add_parser("run", help="run explicitly chosen configurations (the only way to touch the holdout)")
    one.add_argument("--period", choices=("discovery", "holdout"), required=True)
    one.add_argument("--feature-set", choices=FEATURE_SETS, required=True)
    one.add_argument("--train-source", choices=TRAIN_SOURCES, required=True)
    one.add_argument("--retrain-every", type=int, default=5)
    one.add_argument(
        "--portfolio",
        action="append",
        required=True,
        help="K:weighting:cap[:gate_lookback], e.g. 3:equal:12 or 3:equal:12:20 (repeatable)",
    )
    lb = sub.add_parser("leaderboard")
    lb.add_argument("--period", choices=("discovery", "holdout"), default="discovery")
    args = parser.parse_args(argv)
    configure_cli_logging()
    out_dir = research_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset.parquet"
    if args.cmd == "panel":
        build_research_panel(
            start=str(args.start),
            end=str(args.end),
            out_dir=out_dir / str(args.out_name),
            recovered_dir=Path(args.recovered_dir) if args.recovered_dir else None,
            recovered_consolidated_dir=Path(args.recovered_consolidated_dir)
            if args.recovered_consolidated_dir
            else None,
            reconstruct_consolidated=not bool(args.no_reconstruct),
        )
        return
    if args.cmd == "build":
        pdir = Path(args.panel_dir)
        panel = pd.read_parquet(pdir / "pit1520_panel.parquet")
        days = pd.read_parquet(pdir / "pit1520_panel_days.parquet")
        keys: frozenset[tuple[str, str]] = frozenset()
        if args.recovered_keys:
            kf = pd.read_parquet(args.recovered_keys)
            keys = frozenset(
                zip(pd.to_datetime(kf["date"]).dt.strftime("%Y-%m-%d"), kf["symbol"].astype(str), strict=True)
            )
        ds = build_dataset(
            panel=panel,
            panel_days=days,
            recovered_keys=keys,
            min_day_coverage=float(args.min_day_coverage),
            eod_until=pd.Timestamp(args.eod_until),
        )
        ds.to_parquet(dataset_path, index=False)
        return
    if args.cmd == "grid":
        ds = pd.read_parquet(dataset_path)
        period = DISCOVERY if args.period == "discovery" else HOLDOUT
        models, portfolios = default_grid()
        models = [dataclasses.replace(m, retrain_every_days=int(args.retrain_every)) for m in models]
        run_grid(ds, period=period, models=models, portfolios=portfolios, out_dir=out_dir)
        return
    if args.cmd == "run":
        ds = pd.read_parquet(dataset_path)
        period = DISCOVERY if args.period == "discovery" else HOLDOUT
        model = ModelConfig(
            feature_set=args.feature_set, train_source=args.train_source, retrain_every_days=int(args.retrain_every)
        )
        portfolios = []
        for spec_text in args.portfolio:
            parts = spec_text.split(":")
            gate = int(parts[3]) if len(parts) > 3 else None
            portfolios.append(
                PortfolioConfig(
                    top_k=int(parts[0]), weighting=parts[1], max_tick_cost_bp=float(parts[2]), regime_lookback=gate
                )
            )
        run_grid(ds, period=period, models=[model], portfolios=portfolios, out_dir=out_dir)
        return
    if args.cmd == "leaderboard":
        print(leaderboard(out_dir, args.period).to_string())  # noqa: T201 - CLI report output is the operator contract


if __name__ == "__main__":  # pragma: no cover
    main()
