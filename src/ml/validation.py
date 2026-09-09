"""Decision-grade validation: locked OOS + power-matched CPCV promotion gate."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from src.ml.metrics import mean_group_rank_ic
from src.ml.robust_eval import (
    CombinatorialPurgedCV,
    cpcv_oof_predict,
    moving_block_bootstrap_delta,
)

__all__ = [
    "cpcv_path_evidence",
    "minimum_detectable_effect",
    "paired_t_p_value",
]



def minimum_detectable_effect(
    sd: float, n_obs: int, *, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Two-sided 80%-power MDE for a mean: (z_{1-a/2}+z_power)*sd/sqrt(n)."""
    sd_f = float(sd)
    if not np.isfinite(sd_f) or sd_f <= 0.0:
        raise ValueError(f"sd must be finite and > 0, got {sd!r}")
    if int(n_obs) < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs!r}")
    return float((norm.ppf(1.0 - float(alpha) / 2.0) + norm.ppf(float(power))) * sd_f / np.sqrt(float(n_obs)))


def paired_t_p_value(delta: np.ndarray) -> float:
    """Two-sided one-sample t-test p-value on the paired daily difference."""
    arr = np.asarray(delta, dtype=np.float64).ravel()
    if arr.size < 2:
        raise ValueError(f"paired t-test needs n_obs >= 2, got {arr.size}")
    if not np.isfinite(arr).all():
        raise ValueError("paired t-test requires only finite values")
    return float(stats.ttest_1samp(arr, 0.0).pvalue)


def _daily_top1(
    df: pd.DataFrame, group_col: str, value_col: str, score_col: str = "pred"
) -> pd.Series:
    idx = df.groupby(group_col, sort=True)[score_col].idxmax()
    top = df.loc[idx].sort_values(group_col)
    return pd.Series(
        top[value_col].to_numpy(dtype=np.float64),
        index=pd.Index(pd.to_datetime(top[group_col])),
    )


def _daily_ic(
    df: pd.DataFrame, group_col: str, score_col: str, target_col: str
) -> pd.Series:
    vals: list[float] = []
    keys: list[Any] = []
    for key, g in df.groupby(group_col, sort=True):
        s = pd.to_numeric(g[score_col], errors="coerce").to_numpy(dtype=np.float64)
        t = pd.to_numeric(g[target_col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(s) & np.isfinite(t)
        if int(finite.sum()) < 2:  # pragma: no cover - degenerate group guard
            continue
        sf, tf = s[finite], t[finite]
        if float(np.std(sf)) == 0.0 or float(np.std(tf)) == 0.0:  # pragma: no cover - zero-variance guard
            continue
        stat = stats.spearmanr(sf, tf).statistic
        if np.isfinite(stat):
            vals.append(float(stat))
            keys.append(key)
    return pd.Series(np.asarray(vals, dtype=np.float64), index=pd.Index(keys))


def cpcv_path_evidence(
    dev_df: pd.DataFrame,
    feature_cols: list[str],
    train_target_col: str,
    eval_col: str,
    group_col: str,
    *,
    cv: CombinatorialPurgedCV,
    candidate_params: dict[str, Any] | None,
    control_params: dict[str, Any] | None,
    huber_delta: float,
    control_huber_delta: float,
) -> dict[str, Any]:
    """Score candidate vs control on identical CPCV folds with paired path rates."""
    cand_oof = cpcv_oof_predict(
        dev_df,
        feature_cols,
        train_target_col,
        group_col,
        cv=cv,
        model_params=dict(candidate_params) if candidate_params is not None else None,
        huber_delta=float(huber_delta),
    )
    ctrl_oof = cpcv_oof_predict(
        dev_df,
        feature_cols,
        train_target_col,
        group_col,
        cv=cv,
        model_params=dict(control_params) if control_params is not None else None,
        huber_delta=float(control_huber_delta),
    )
    fold_ids = sorted(pd.unique(cand_oof["cpcv_fold"]).tolist())
    path_deltas: list[float] = []
    ic_wins = 0
    top1_wins = 0
    for fid in fold_ids:
        cand_f = cand_oof[cand_oof["cpcv_fold"] == fid]
        ctrl_f = ctrl_oof[ctrl_oof["cpcv_fold"] == fid]
        cand_top = _daily_top1(cand_f, group_col, eval_col).to_numpy(dtype=np.float64)
        ctrl_top = _daily_top1(ctrl_f, group_col, eval_col).to_numpy(dtype=np.float64)
        # A top-1 pick can land on a row without a mechanical label (no next
        # trading day); skip those dates rather than let one NaN null the fold.
        cand_finite = cand_top[np.isfinite(cand_top)]
        ctrl_finite = ctrl_top[np.isfinite(ctrl_top)]
        cand_mean = float(np.mean(cand_finite)) if cand_finite.size else float("nan")
        ctrl_mean = float(np.mean(ctrl_finite)) if ctrl_finite.size else float("nan")
        delta = float(cand_mean - ctrl_mean)
        path_deltas.append(delta)
        if np.isfinite(delta) and delta > 0.0:
            top1_wins += 1
        cand_ic = mean_group_rank_ic(cand_f, [group_col], "pred", eval_col, min_group_size=2)
        ctrl_ic = mean_group_rank_ic(ctrl_f, [group_col], "pred", eval_col, min_group_size=2)
        if np.isfinite(cand_ic) and np.isfinite(ctrl_ic) and cand_ic > ctrl_ic:
            ic_wins += 1
    n_paths = len(path_deltas)
    finite_path_deltas = np.asarray(path_deltas, dtype=np.float64)
    finite_path_deltas = finite_path_deltas[np.isfinite(finite_path_deltas)]
    pooled = float(np.mean(finite_path_deltas)) if finite_path_deltas.size else float("nan")
    cand_daily = _daily_top1(cand_oof, group_col, eval_col)
    ctrl_daily = _daily_top1(ctrl_oof, group_col, eval_col)
    common = cand_daily.index.intersection(ctrl_daily.index)
    cand_common = cand_daily.loc[common].to_numpy(dtype=np.float64)
    ctrl_common = ctrl_daily.loc[common].to_numpy(dtype=np.float64)
    # Skip dates where either arm's top-1 pick has no mechanical label so the
    # significance tests below never see a non-finite paired value.
    pair_finite = np.isfinite(cand_common) & np.isfinite(ctrl_common)
    cand_aligned = cand_common[pair_finite]
    ctrl_aligned = ctrl_common[pair_finite]
    if cand_aligned.size >= 30:
        boot = moving_block_bootstrap_delta(cand_aligned, ctrl_aligned)
        p_boot = float(boot.p_value)
    else:  # pragma: no cover - tiny-dev fallback
        p_boot = 1.0
    if cand_aligned.size >= 2:
        try:
            p_t = float(paired_t_p_value(cand_aligned - ctrl_aligned))
        except ValueError:  # pragma: no cover - degenerate delta guard
            p_t = 1.0
    else:  # pragma: no cover - tiny-dev fallback
        p_t = 1.0
    ic_cand = float(mean_group_rank_ic(cand_oof, [group_col], "pred", eval_col, min_group_size=2))
    ic_ctrl = float(mean_group_rank_ic(ctrl_oof, [group_col], "pred", eval_col, min_group_size=2))
    delta_daily = cand_aligned - ctrl_aligned
    sd_top1 = float(np.std(delta_daily, ddof=1)) if delta_daily.size >= 2 else float("nan")
    n_daily = int(cand_aligned.size) if cand_aligned.size else int(n_paths)
    if np.isfinite(sd_top1) and sd_top1 > 0.0 and n_daily >= 1:
        mde_top1 = float(minimum_detectable_effect(sd_top1, n_daily))
    else:  # pragma: no cover - degenerate variance guard
        mde_top1 = float("nan")
    cand_ic_daily = _daily_ic(cand_oof, group_col, "pred", eval_col)
    ctrl_ic_daily = _daily_ic(ctrl_oof, group_col, "pred", eval_col)
    ic_common = cand_ic_daily.index.intersection(ctrl_ic_daily.index)
    if len(ic_common) >= 2:
        ic_delta_daily = (
            cand_ic_daily.loc[ic_common].to_numpy(dtype=np.float64)
            - ctrl_ic_daily.loc[ic_common].to_numpy(dtype=np.float64)
        )
        sd_ic = float(np.std(ic_delta_daily, ddof=1))
        if np.isfinite(sd_ic) and sd_ic > 0.0:
            mde_ic = float(minimum_detectable_effect(sd_ic, len(ic_common)))
        else:  # pragma: no cover - degenerate IC variance guard
            mde_ic = float(minimum_detectable_effect(0.2745, len(ic_common)))
    else:  # pragma: no cover - tiny-dev fallback
        mde_ic = float("nan")
    return {
        "path_deltas": list(path_deltas),
        "n_path_deltas": int(n_paths),
        "top1_path_win_rate": float(top1_wins / n_paths) if n_paths else float("nan"),
        "ic_path_win_rate": float(ic_wins / n_paths) if n_paths else float("nan"),
        "pooled_delta": float(pooled),
        "p_bootstrap": float(p_boot),
        "p_paired_t": float(p_t),
        "ic_candidate": float(ic_cand),
        "ic_control": float(ic_ctrl),
        "ic_delta": float(ic_cand - ic_ctrl),
        "mde_top1": float(mde_top1),
        "mde_ic": float(mde_ic),
    }
