"""Sparse-panel robustness: CPCV evaluation, block bootstrap gate, DSR discount."""
from __future__ import annotations

import itertools
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import norm

from src.ml.oof import _finite_nan


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    """Combinatorial purged CV splitter; group = whole date-group, never split."""

    n_groups: int = 8
    k_test: int = 2
    purge_gap: int = 1
    embargo_gap: int = 1

    def __post_init__(self) -> None:
        if self.n_groups < 4:
            raise ValueError(f"n_groups must be >= 4, got {self.n_groups}")
        if not 2 <= self.k_test < self.n_groups:
            raise ValueError(f"k_test must satisfy 2 <= k_test < n_groups, got {self.k_test}")
        if self.purge_gap < 0:
            raise ValueError(f"purge_gap must be >= 0, got {self.purge_gap}")
        if self.embargo_gap < 0:
            raise ValueError(f"embargo_gap must be >= 0, got {self.embargo_gap}")
        # 최악 조합(테스트 빈이 최대로 분산)에서도 train 빈이 최소 1개 남도록 보장.
        min_groups = self.k_test * (1 + self.purge_gap + self.embargo_gap) + 1
        if self.n_groups < min_groups:
            raise ValueError(
                f"n_groups must be >= k_test*(1+purge_gap+embargo_gap)+1 = {min_groups} "
                f"to avoid an empty train split, got {self.n_groups}"
            )

    def n_paths(self) -> int:
        return math.comb(self.n_groups - 1, self.k_test - 1)

    def split(self, groups: pd.Series | np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        arr = pd.Series(groups).to_numpy() if isinstance(groups, pd.Series) else np.asarray(groups)
        unique_sorted = np.array(sorted(pd.unique(arr)))
        if len(unique_sorted) < self.n_groups:
            raise ValueError(f"CPCV needs >= n_groups unique groups, got {len(unique_sorted)}")
        order = {v: i for i, v in enumerate(unique_sorted.tolist())}
        positions = np.array([order[v] for v in pd.Series(arr).tolist()], dtype=np.int64)
        bin_index = np.minimum((positions * self.n_groups) // len(unique_sorted), self.n_groups - 1)
        fold_combos = itertools.combinations(range(self.n_groups), self.k_test)
        for fold_id, combo in enumerate(fold_combos):
            test_bins = set(combo)
            purged = set(combo)
            for b in combo:
                for gap in range(1, self.purge_gap + 1):
                    if b - gap >= 0:
                        purged.add(b - gap)
                for gap in range(1, self.embargo_gap + 1):
                    if b + gap < self.n_groups:
                        purged.add(b + gap)
            train_bins = set(range(self.n_groups)) - purged
            test_idx = np.flatnonzero(np.isin(bin_index, list(test_bins)))
            train_idx = np.flatnonzero(np.isin(bin_index, list(train_bins)))
            yield train_idx, test_idx, fold_id


@dataclass(frozen=True)
class BootstrapDelta:
    delta: float
    p_value: float
    ci_low: float
    ci_high: float
    n_obs: int


def moving_block_bootstrap_delta(
    series_a: np.ndarray,
    series_b: np.ndarray,
    *,
    block_size: int = 10,
    n_boot: int = 5000,
    seed: int = 0,
) -> BootstrapDelta:
    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"paired series must share 1-D shape, got {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired series must hold only finite values")
    n = int(a.size)
    if n < 30:
        raise ValueError(f"bootstrap needs n_obs >= 30, got {n}")
    if block_size < 1 or block_size > n:
        raise ValueError(f"block_size must satisfy 1 <= block_size <= n, got {block_size}")
    d = a - b
    delta = float(d.mean())
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_size)
    starts = rng.integers(0, n - block_size + 1, size=(n_boot, n_blocks))
    idx = (starts[..., None] + np.arange(block_size)).reshape(n_boot, -1)[:, :n]
    boot_means = d[idx].mean(axis=1)
    frac_le = float(np.mean(boot_means <= 0.0))
    frac_ge = float(np.mean(boot_means >= 0.0))
    p_value = min(1.0, 2.0 * min(frac_le, frac_ge))
    ci_low, ci_high = (float(v) for v in np.quantile(boot_means, [0.025, 0.975]))
    return BootstrapDelta(delta=delta, p_value=p_value, ci_low=ci_low, ci_high=ci_high, n_obs=n)


def deflated_sharpe_ratio(
    returns: np.ndarray, *, n_independent_trials: int, periods_per_year: int = 252
) -> float:
    x = np.asarray(returns, dtype=np.float64)
    if n_independent_trials < 1:
        raise ValueError(f"n_independent_trials must be >= 1, got {n_independent_trials}")
    if x.size < 20:
        raise ValueError(f"DSR needs returns.size >= 20, got {x.size}")
    if not np.isfinite(x).all():
        raise ValueError("returns must hold only finite values")
    std = float(np.std(x, ddof=1))
    if std <= 0.0 or not np.isfinite(std):
        raise ValueError("returns must have positive finite variance")
    _ = periods_per_year
    n = x.size
    sr = float(np.mean(x) / std)
    skew = float((((x - x.mean()) / std) ** 3).mean())
    kurt = float((((x - x.mean()) / std) ** 4).mean())
    denom = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / (n - 1)
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("Sharpe standard error is not finite")
    if n_independent_trials <= 1:
        sr0 = 0.0
    else:
        k = float(n_independent_trials)
        euler = 0.5772156649
        sr0 = math.sqrt(1.0 / (n - 1)) * (
            (1.0 - euler) * norm.ppf(1.0 - 1.0 / k) + euler * norm.ppf(1.0 - 1.0 / (k * math.e))
        )
    return float(norm.cdf((sr - sr0) / math.sqrt(denom)))


def cpcv_oof_predict(
    dev_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    *,
    cv: CombinatorialPurgedCV,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.9,
) -> pd.DataFrame:
    if not feature_cols:
        raise ValueError("feature_cols must not be empty")
    missing = [c for c in [*feature_cols, target_col, group_col] if c not in dev_df.columns]
    if missing:
        raise ValueError(f"missing columns in dev_df: {missing}")
    work = dev_df.sort_values(group_col).copy()
    # df.attrs 에 비교 불가 객체(feature_manifest/DataFrame 등)가 있으면 pd.concat 이
    # __finalize__ 에서 ValueError 를 던지므로 슬라이스 이전에 제거.
    work.attrs = {}
    parts: list[pd.DataFrame] = []
    for train_idx, test_idx, fold_id in cv.split(work[group_col]):
        train = work.iloc[train_idx]
        val = work.iloc[test_idx]
        train_f = _finite_nan(train, feature_cols)
        val_f = _finite_nan(val, feature_cols)
        reg = LGBMRegressor(
            objective="huber", alpha=huber_delta, random_state=42, verbosity=-1, **dict(model_params or {})
        )
        reg.fit(train_f[feature_cols], train_f[target_col].to_numpy(dtype=np.float64))
        fold_df = work.loc[val.index].copy()
        fold_df.attrs = {}
        fold_df["pred"] = np.asarray(reg.predict(val_f[feature_cols]), dtype=np.float64)
        fold_df["cpcv_fold"] = int(fold_id)
        parts.append(fold_df)
    if not parts:
        return dev_df.iloc[0:0].copy()
    return pd.concat(parts)


def path_top1_returns(
    oof_df: pd.DataFrame,
    group_col: str,
    target_col: str,
    *,
    score_col: str = "pred",
    fold_col: str = "cpcv_fold",
) -> dict[int, np.ndarray]:
    missing = [c for c in (group_col, target_col, score_col, fold_col) if c not in oof_df.columns]
    if missing:
        raise ValueError(f"missing columns in oof_df: {missing}")
    paths: dict[int, np.ndarray] = {}
    for fold_id, fold_df in oof_df.groupby(fold_col, sort=True):
        vals: list[float] = []
        for _, g in fold_df.groupby(group_col, sort=True):
            best = int(g[score_col].to_numpy().argmax())
            vals.append(float(g[target_col].to_numpy()[best]))
        paths[int(fold_id)] = np.asarray(vals, dtype=np.float64)
    return paths
