"""Permutation rankIC importance and stable feature selection."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _rank_ic_for_df(df: pd.DataFrame, pred: np.ndarray, target_col: str, group_col: str) -> float:
    groups = df[group_col].to_numpy()
    target = df[target_col].to_numpy(dtype=np.float64)
    ics: list[float] = []
    tmp = pd.DataFrame({"pred": pred, "target": target, group_col: groups})
    for _, g in tmp.groupby(group_col, sort=False):
        if len(g) < 3:
            continue
        p = g["pred"].to_numpy(dtype=np.float64)
        t = g["target"].to_numpy(dtype=np.float64)
        if float(np.std(p)) == 0.0:
            continue
        if float(np.std(t)) == 0.0:
            continue
        stat = spearmanr(p, t).statistic
        if stat is None or not np.isfinite(stat):
            continue
        ics.append(float(stat))
    if not ics:
        return float("nan")
    return float(np.mean(ics))


def permutation_rank_ic_importance(
    model: Any,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    *,
    random_state: int = 42,
) -> pd.Series:
    # base prediction
    x_base = eval_df[feature_cols]
    pred_base = np.asarray(model.predict(x_base), dtype=np.float64)
    base_ic = _rank_ic_for_df(eval_df, pred_base, target_col, group_col)
    # if base is nan, treat as 0? Keep nan propagation
    importances: dict[str, float] = {}
    rng = np.random.default_rng(int(random_state))

    # Precompute group positions for shuffling within groups
    groups = eval_df[group_col].to_numpy()
    # map group -> indices positions (iloc positions)
    group_to_pos: dict[Any, np.ndarray] = {}
    unique_groups = pd.unique(groups)
    for g in unique_groups:
        pos = np.where(groups == g)[0]
        group_to_pos[g] = pos

    for feat in feature_cols:
        # copy column values
        vals = eval_df[feat].to_numpy()
        permuted_vals = vals.copy()
        for pos in group_to_pos.values():
            if len(pos) <= 1:
                continue
            perm = rng.permutation(len(pos))
            permuted_vals[pos] = vals[pos[perm]]
        # create shuffled df copy for prediction
        df_shuffled = eval_df.copy()
        df_shuffled[feat] = permuted_vals
        pred_perm = np.asarray(model.predict(df_shuffled[feature_cols]), dtype=np.float64)
        perm_ic = _rank_ic_for_df(df_shuffled, pred_perm, target_col, group_col)
        # drop = base - perm
        if not np.isfinite(base_ic) or not np.isfinite(perm_ic):
            # if either nan, define drop 0 if both nan? Else nan? For date-constant feature, both should be same -> 0
            # If base nan and perm nan, drop 0
            if not np.isfinite(base_ic) and not np.isfinite(perm_ic):
                drop = 0.0
            elif not np.isfinite(base_ic):
                drop = float("nan")
            else:
                # perm is nan, treat drop as base - perm => nan? Use 0? Let's set drop = 0 if both undefined
                drop = 0.0 if not np.isfinite(perm_ic) and not np.isfinite(base_ic) else float("nan") if not np.isfinite(perm_ic) else float(base_ic) - float(perm_ic)
        else:
            drop = float(base_ic) - float(perm_ic)
        # For date-constant feature, permuted values are identical to original within group (all same), so perm_ic == base_ic -> drop 0
        # Ensure floating and handle small numerical noise: if values identical, drop should be exactly 0.0
        # Our permutation for constant group values will produce identical array, so pred same -> drop 0
        if drop is not None and np.isfinite(drop) and abs(drop) < 1e-15:
            drop = 0.0
        importances[feat] = float(drop) if np.isfinite(drop) else 0.0

    # Actually handle case where drop is nan -> set 0.0 for stability? For constant feature we expect 0.0 not nan
    # Re-evaluate: if base_ic or perm_ic nan, set drop 0.0
    for k, v in list(importances.items()):
        if not np.isfinite(v):
            importances[k] = 0.0

    series = pd.Series(importances, dtype=np.float64)
    # ensure order as feature_cols
    series = series.reindex(feature_cols).astype(np.float64)
    return series


def select_stable_features(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    *,
    top_n: int = 30,
    inner_splits: int = 3,
    purge_gap: int = 1,
    min_folds: int = 2,
    model_params: dict[str, Any] | None = None,
    huber_delta: float = 0.02,
    random_state: int = 42,
) -> list[str]:
    from lightgbm import LGBMRegressor

    from src.ml.purged_cv import PurgedGroupTimeSeriesSplit

    splitter = PurgedGroupTimeSeriesSplit(n_splits=int(inner_splits), purge_gap=int(purge_gap))
    groups = train_df[group_col]

    top_per_fold: list[list[str]] = []
    sum_importance: dict[str, float] = dict.fromkeys(feature_cols, 0.0)

    for train_idx, val_idx in splitter.split(train_df, groups=groups):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        train_part = train_df.iloc[train_idx]
        val_part = train_df.iloc[val_idx]
        params = dict(model_params or {})
        model = LGBMRegressor(
            objective="huber",
            alpha=float(huber_delta),
            random_state=int(random_state),
            verbosity=-1,
            **params,
        )
        model.fit(train_part[feature_cols], train_part[target_col].to_numpy(dtype=np.float64))
        imp = permutation_rank_ic_importance(model, val_part, feature_cols, target_col, group_col, random_state=int(random_state))
        # accumulate sum
        for c in feature_cols:
            v = float(imp.get(c, 0.0))
            if np.isfinite(v):
                sum_importance[c] += v
        # top_n descending
        sorted_feats = imp.sort_values(ascending=False).index.tolist()
        top = sorted_feats[: int(top_n)]
        top_per_fold.append(top)

    if not top_per_fold:
        # fallback: return first top_n in original order? But spec says fallback to sum importance top_n
        sorted_by_sum = sorted(feature_cols, key=lambda c: sum_importance.get(c, 0.0), reverse=True)
        return sorted_by_sum[: int(top_n)]

    # count occurrences
    counts: dict[str, int] = dict.fromkeys(feature_cols, 0)
    for top in top_per_fold:
        for c in top:
            counts[c] += 1

    stable = [c for c in feature_cols if counts.get(c, 0) >= int(min_folds)]

    if not stable:
        # fallback to sum importance
        sorted_by_sum = sorted(feature_cols, key=lambda c: sum_importance.get(c, 0.0), reverse=True)
        fallback = sorted_by_sum[: int(top_n)]
        # return in original order
        fallback_set = set(fallback)
        return [c for c in feature_cols if c in fallback_set]

    # trim to top_n if stable > top_n? spec says selected are those >=min_folds among top_n per fold, but could exceed top_n? Keep at most top_n sorted by sum importance?
    # If stable length > top_n, keep top_n by sum importance but preserve original order
    if len(stable) > int(top_n):
        # rank stable by sum importance descending, take top_n
        ranked = sorted(stable, key=lambda c: sum_importance.get(c, 0.0), reverse=True)[: int(top_n)]
        ranked_set = set(ranked)
        stable = [c for c in feature_cols if c in ranked_set]

    # Preserve original order
    stable_set = set(stable)
    ordered = [c for c in feature_cols if c in stable_set]
    return ordered
