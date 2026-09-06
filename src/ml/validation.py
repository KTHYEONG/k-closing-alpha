"""Decision-grade validation: locked OOS + power-matched CPCV promotion gate."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

from src.ml.bundle import fit_seed_ensemble, save_bundle
from src.ml.metrics import mean_group_rank_ic
from src.ml.robust_eval import (
    CombinatorialPurgedCV,
    cpcv_oof_predict,
    moving_block_bootstrap_delta,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationConfig:
    oos_reserve_start: str
    eval_col: str = "eval_net_mechanical"
    cpcv_n_groups: int = 8
    cpcv_k_test: int = 2
    purge_gap: int = 1
    promotion_alpha: float = 0.05
    min_ic_path_win_rate: float = 0.75
    min_top1_path_win_rate: float = 0.60
    min_selection_dsr: float = 0.95
    min_oos_days: int = 60
    min_oos_rank_ic: float = 0.0
    require_fillable_sleeve: bool = True
    target_notional_100m: float = 0.5

    def __post_init__(self) -> None:
        parsed = pd.to_datetime(self.oos_reserve_start, errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"oos_reserve_start is not parseable: {self.oos_reserve_start!r}")
        if not isinstance(self.eval_col, str) or not self.eval_col:
            raise ValueError(f"eval_col must be a non-empty str, got {self.eval_col!r}")
        if self.cpcv_n_groups < 4:
            raise ValueError(f"cpcv_n_groups must be >= 4, got {self.cpcv_n_groups}")
        if not 2 <= self.cpcv_k_test < self.cpcv_n_groups:
            raise ValueError(
                f"cpcv_k_test must satisfy 2 <= cpcv_k_test < cpcv_n_groups, got {self.cpcv_k_test}"
            )
        if self.purge_gap < 0:
            raise ValueError(f"purge_gap must be >=0, got {self.purge_gap}")
        if not 0.0 < self.promotion_alpha <= 0.5:
            raise ValueError(f"promotion_alpha must be in (0, 0.5], got {self.promotion_alpha}")
        for name in ("min_ic_path_win_rate", "min_top1_path_win_rate", "min_selection_dsr"):
            v = float(getattr(self, name))
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        if not 0.0 <= float(self.min_oos_rank_ic) <= 1.0:
            raise ValueError(f"min_oos_rank_ic must be in [0, 1], got {self.min_oos_rank_ic}")
        if self.min_oos_days < 1:
            raise ValueError(f"min_oos_days must be >=1, got {self.min_oos_days}")
        if not np.isfinite(float(self.target_notional_100m)) or float(self.target_notional_100m) <= 0.0:
            raise ValueError(
                f"target_notional_100m must be finite and > 0, got {self.target_notional_100m!r}"
            )


@dataclass(frozen=True)
class GateOutcome:
    name: str
    passed: bool
    observed: float
    threshold: float
    detail: dict[str, Any]


@dataclass(frozen=True)
class PromotionDecision:
    deployable: bool
    verdict: str
    failed_gates: tuple[str, ...]
    gates: tuple[GateOutcome, ...]
    evidence: dict[str, Any]


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


def evaluate_locked_oos(
    dev_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    feature_cols: list[str],
    train_target_col: str,
    eval_col: str,
    group_col: str,
    *,
    model_params: dict[str, Any] | None,
    huber_delta: float,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    """Fit on dev only and score the reserved window exactly once."""
    if len(oos_df) == 0:
        raise ValueError("oos_df is empty: locked window has no rows to score")
    dev_groups = set(pd.unique(pd.to_datetime(dev_df[group_col], errors="coerce").dropna()))
    oos_groups = set(pd.unique(pd.to_datetime(oos_df[group_col], errors="coerce").dropna()))
    if dev_groups & oos_groups:
        raise ValueError(
            f"dev/oos overlap: {len(dev_groups & oos_groups)} shared group value(s) leak into the locked window"
        )
    ensemble = fit_seed_ensemble(
        dev_df,
        list(feature_cols),
        train_target_col,
        tuple(seeds),
        dict(model_params) if model_params is not None else {},
        float(huber_delta),
    )
    oos = oos_df.copy()
    feats = oos[list(feature_cols)]
    oos["pred"] = np.asarray(ensemble.predict(feats), dtype=np.float64)
    daily = _daily_top1(oos, group_col, eval_col)
    raw_top1 = daily.to_numpy(dtype=np.float64)
    # A day's top-1 pick can lack a mechanical label (no next trading day);
    # drop those days rather than let one NaN null the whole OOS statistic.
    top1 = raw_top1[np.isfinite(raw_top1)]
    n_days = int(top1.size)
    n_rows = len(oos)
    top1_mean = float(np.mean(top1)) if top1.size else float("nan")
    top1_sd = float(np.std(top1, ddof=1)) if top1.size >= 2 else 0.0
    top1_sharpe = float(top1_mean / top1_sd * np.sqrt(252.0)) if top1_sd > 0.0 else float("nan")
    rank_ic = float(mean_group_rank_ic(oos, [group_col], "pred", eval_col, min_group_size=2))
    mde_top1 = float(minimum_detectable_effect(top1_sd, n_days)) if top1_sd > 0.0 and n_days >= 1 else float("nan")
    ic_daily = _daily_ic(oos, group_col, "pred", eval_col)
    sd_ic = float(np.std(ic_daily.to_numpy(dtype=np.float64), ddof=1)) if ic_daily.size >= 2 else float("nan")
    if np.isfinite(sd_ic) and sd_ic > 0.0:
        mde_ic = float(minimum_detectable_effect(sd_ic, int(ic_daily.size)))
    else:  # pragma: no cover - degenerate OOS IC guard
        mde_ic = float(minimum_detectable_effect(0.2745, max(1, n_days)))
    parsed = pd.to_datetime(oos[group_col], errors="coerce")
    return {
        "n_days": int(n_days),
        "n_rows": int(n_rows),
        "top1_mean": float(top1_mean),
        "top1_sd": float(top1_sd),
        "top1_sharpe": float(top1_sharpe),
        "rank_ic": float(rank_ic),
        "mde_top1": float(mde_top1),
        "mde_ic": float(mde_ic),
        "first_date": str(parsed.min()),
        "last_date": str(parsed.max()),
    }


def run_promotion_gate(
    *,
    cpcv: dict[str, Any],
    oos: dict[str, Any],
    selection_dsr: float | None,
    n_selection_trials: int,
    fillable: dict[str, Any] | None,
    config: ValidationConfig,
) -> PromotionDecision:
    """Compose the eight fail-closed promotion gates with per-gate MDE evidence."""
    if int(n_selection_trials) < 1:
        raise ValueError(f"n_selection_trials must be >= 1, got {n_selection_trials!r}")
    gates: list[GateOutcome] = []
    p_boot = float(cpcv.get("p_bootstrap", 1.0))
    p_t = float(cpcv.get("p_paired_t", 1.0))
    p_cons = float(max(p_boot, p_t))
    gates.append(
        GateOutcome(
            name="cpcv_ic_path_win_rate",
            passed=bool(float(cpcv.get("ic_path_win_rate", 0.0)) >= float(config.min_ic_path_win_rate)),
            observed=float(cpcv.get("ic_path_win_rate", float("nan"))),
            threshold=float(config.min_ic_path_win_rate),
            detail={"mde": float(cpcv.get("mde_ic", float("nan")))},
        )
    )
    gates.append(
        GateOutcome(
            name="cpcv_top1_path_win_rate",
            passed=bool(float(cpcv.get("top1_path_win_rate", 0.0)) >= float(config.min_top1_path_win_rate)),
            observed=float(cpcv.get("top1_path_win_rate", float("nan"))),
            threshold=float(config.min_top1_path_win_rate),
            detail={"mde": float(cpcv.get("mde_top1", float("nan")))},
        )
    )
    gates.append(
        GateOutcome(
            name="cpcv_delta_significance",
            passed=bool(p_cons < float(config.promotion_alpha)),
            observed=float(p_cons),
            threshold=float(config.promotion_alpha),
            detail={"mde": float(cpcv.get("mde_top1", float("nan"))), "p_bootstrap": p_boot, "p_paired_t": p_t},
        )
    )
    if selection_dsr is None or not np.isfinite(float(selection_dsr)):
        dsr_passed = False
        dsr_obs = float("nan")
    else:
        dsr_obs = float(selection_dsr)
        dsr_passed = bool(dsr_obs >= float(config.min_selection_dsr))
    gates.append(
        GateOutcome(
            name="selection_dsr",
            passed=bool(dsr_passed),
            observed=float(dsr_obs),
            threshold=float(config.min_selection_dsr),
            detail={"mde": float("nan"), "n_selection_trials": int(n_selection_trials)},
        )
    )
    gates.append(
        GateOutcome(
            name="oos_min_days",
            passed=bool(int(oos.get("n_days", 0)) >= int(config.min_oos_days)),
            observed=float(oos.get("n_days", 0)),
            threshold=float(config.min_oos_days),
            detail={"mde": float("nan")},
        )
    )
    oos_mde_ic = float(oos.get("mde_ic", float("nan")))
    oos_ic = float(oos.get("rank_ic", float("nan")))
    ic_threshold = float(max(oos_mde_ic if np.isfinite(oos_mde_ic) else 0.0, float(config.min_oos_rank_ic)))
    gates.append(
        GateOutcome(
            name="oos_rank_ic_above_mde",
            passed=bool(np.isfinite(oos_ic) and np.isfinite(oos_mde_ic) and oos_ic > oos_mde_ic and oos_ic > float(config.min_oos_rank_ic)),
            observed=float(oos_ic),
            threshold=float(ic_threshold),
            detail={"mde": float(oos_mde_ic)},
        )
    )
    gates.append(
        GateOutcome(
            name="oos_top1_sign",
            passed=bool(np.isfinite(float(oos.get("top1_mean", float("nan")))) and float(oos.get("top1_mean")) > 0.0),
            observed=float(oos.get("top1_mean", float("nan"))),
            threshold=0.0,
            detail={"mde": float(oos.get("mde_top1", float("nan")))},
        )
    )
    if not bool(config.require_fillable_sleeve):
        fill_passed = True
        fill_obs = float(fillable.get("top1_mean", float("nan"))) if fillable else float("nan")
    elif fillable is None:
        fill_passed = False
        fill_obs = float("nan")
    else:
        fill_obs = float(fillable.get("top1_mean", float("nan")))
        fill_passed = bool(np.isfinite(fill_obs) and fill_obs > 0.0)
    gates.append(
        GateOutcome(
            name="fillable_top1_sign",
            passed=bool(fill_passed),
            observed=float(fill_obs),
            threshold=0.0,
            detail={"mde": float("nan"), "measured_share": float(fillable.get("measured_share", float("nan"))) if fillable else float("nan")},
        )
    )
    for g in gates:
        logger.info(
            "[EVAL] gate=%s passed=%s observed=%.4f threshold=%.4f mde=%.4f",
            g.name,
            g.passed,
            float(g.observed) if np.isfinite(g.observed) else 0.0,
            float(g.threshold) if np.isfinite(g.threshold) else 0.0,
            float(g.detail.get("mde", float("nan"))) if np.isfinite(float(g.detail.get("mde", float("nan")))) else 0.0,
        )
    failed = tuple(g.name for g in gates if not g.passed)
    verdict = "deployable" if not failed else "research_only"
    logger.info("[EVAL] verdict=%s failed_gates=%s", verdict, len(failed))
    evidence: dict[str, Any] = {
        "cpcv": dict(cpcv),
        "oos": dict(oos),
        "selection_dsr": selection_dsr,
        "n_selection_trials": int(n_selection_trials),
        "fillable": dict(fillable) if fillable else None,
        "oos_reserve_start": config.oos_reserve_start,
    }
    return PromotionDecision(
        deployable=bool(not failed),
        verdict=str(verdict),
        failed_gates=tuple(failed),
        gates=tuple(gates),
        evidence=dict(evidence),
    )


def publish_bundle(
    bundle: dict[str, Any],
    candidate_dir: str,
    production_dir: str,
    decision: PromotionDecision,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Always write the candidate; publish to production only when deployable."""
    candidate_path = save_bundle(dict(bundle), candidate_dir)
    with open(candidate_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    should_publish = bool(decision.deployable or force)
    logger.info(
        "[SYS] stage=publish published=%s production_dir=%s",
        bool(should_publish),
        str(production_dir),
    )
    result: dict[str, Any] = {
        "published": bool(should_publish),
        "candidate_dir": str(candidate_dir),
        "candidate_path": str(candidate_path),
        "production_dir": str(production_dir),
        "sha256": str(digest),
        "verdict": str(decision.verdict),
        "gates": [
            {
                "name": g.name,
                "passed": bool(g.passed),
                "observed": float(g.observed),
                "threshold": float(g.threshold),
                "detail": dict(g.detail),
            }
            for g in decision.gates
        ],
        "failed_gates": [str(v) for v in decision.failed_gates],
    }
    if not should_publish:
        return json.loads(json.dumps(result))
    os.makedirs(production_dir, exist_ok=True)
    incumbent = os.path.join(production_dir, "sizing_pipeline_bundle.joblib")
    if os.path.exists(incumbent):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = f"{incumbent}.bak_{stamp}"
        shutil.copy2(incumbent, backup)
        result["backup_path"] = str(backup)
    shutil.copy2(candidate_path, incumbent)
    record = {
        "published_at": datetime.now(UTC).isoformat(),
        "candidate_dir": str(candidate_dir),
        "sha256": str(digest),
        "verdict": str(decision.verdict),
        "gates": result["gates"],
        "failed_gates": result["failed_gates"],
        "label_mode": bundle.get("label_mode"),
        "cost_mode": bundle.get("cost_mode"),
        "round_trip_cost": bundle.get("round_trip_cost"),
        "training_cutoff": bundle.get("training_cutoff"),
        "oos_reserve_start": decision.evidence.get("oos_reserve_start") if decision.evidence else bundle.get("oos_reserve_start"),
    }
    with open(os.path.join(production_dir, "promotion_record.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    result["promotion_record"] = str(os.path.join(production_dir, "promotion_record.json"))
    return json.loads(json.dumps(result))
