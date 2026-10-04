"""Train/serve contract for the cost-aware top-k ranker.

Owns everything the live 15:20 decision path and the production bundle trainer must agree on:
the ordered feature list, the base decision-time feature definitions, the bundle directory and
atomic persistence, the certified-screen parity check, and the scoring/selection rules applied to
a loaded bundle. It is deliberately a leaf module (stdlib, numpy, pandas, src.strategy.contract
only) so that serving never imports research harnesses, CLIs, or scipy. Research modules re-export
these names for backward compatibility.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.strategy.contract import MIN_TOP_K, PRODUCTION_STRATEGY, UniverseSpec

logger = logging.getLogger(__name__)

TOPK_RANKER_BUNDLE_DIR: str = "artifacts/models/topk_ranker"

FEATURE_COLS: list[str] = [
    "chg_ratio",
    "log_tv",
    "log_mc",
    "body_ratio",
    "upper_shadow_ratio",
    "intraday_range",
    "kospi_pct",
    "kosdaq_pct",
    "v_kospi",
    "tv_rank",
    "chg_rank",
]

TOPK_COST_FEATURE_COLS: tuple[str, ...] = ("f_tick_cost", "f_log_close")
TOPK_FLOW_FEATURE_COLS: tuple[str, ...] = ("inst_density", "inst_rank")
TOPK_HISTORY_FEATURE_COLS: tuple[str, ...] = (
    "f_ret5",
    "f_ret20",
    "f_ret60",
    "f_dist_high60",
    "f_upcount20",
    "f_on_mean20",
    "f_on_mean60",
    "f_upnext_on60",
    "f_gap",
    "f_id_mean20",
    "f_inst_cum5",
    "f_foreign_cum5",
)
# 순서 고정: colsample_bytree 가 열 순서에 의존하므로 인증 순서를 그대로 유지한다.
TOPK_FEATURE_COLS_V2: list[str] = [
    *FEATURE_COLS,
    *TOPK_FLOW_FEATURE_COLS,
    *TOPK_COST_FEATURE_COLS,
    *TOPK_HISTORY_FEATURE_COLS,
]
RANKER_FEATURE_COLS: list[str] = list(TOPK_FEATURE_COLS_V2)


class FeatureUnit(enum.StrEnum):
    """Physical unit of a ranker feature value as fed to the model."""

    DECIMAL_RETURN = "decimal_return"
    DECIMAL_RATIO = "decimal_ratio"
    LOG1P_100M_KRW = "log1p_100m_krw"
    LOG_KRW = "log_krw"
    LOG_RETURN_SUM = "log_return_sum"
    PCT_RANK = "pct_rank"
    INDEX_LEVEL = "index_level"
    BASIS_POINTS = "basis_points"
    COUNT = "count"


class FeatureTiming(enum.StrEnum):
    """Point-in-time availability class of a ranker feature at the 15:20 decision."""

    DECISION_SNAPSHOT = "decision_snapshot"
    PRIOR_CONFIRMED = "prior_confirmed"
    SNAPSHOT_AND_PRIOR = "snapshot_and_prior"


@dataclasses.dataclass(frozen=True)
class TopkFeatureSpec:
    """Declared definition of one ranker feature; the authoritative unit/timing record."""

    name: str
    unit: FeatureUnit
    timing: FeatureTiming
    definition: str


TOPK_FEATURE_SPECS: tuple[TopkFeatureSpec, ...] = (
    TopkFeatureSpec("chg_ratio", FeatureUnit.DECIMAL_RETURN, FeatureTiming.DECISION_SNAPSHOT, "close_T / prev_close_T - 1; NaN beyond the KRX daily limit"),
    TopkFeatureSpec("log_tv", FeatureUnit.LOG1P_100M_KRW, FeatureTiming.DECISION_SNAPSHOT, "log1p(max(trade value in 100M KRW, 0))"),
    TopkFeatureSpec("log_mc", FeatureUnit.LOG1P_100M_KRW, FeatureTiming.DECISION_SNAPSHOT, "log1p(max(market cap in 100M KRW, 0))"),
    TopkFeatureSpec("body_ratio", FeatureUnit.DECIMAL_RATIO, FeatureTiming.DECISION_SNAPSHOT, "signed (close - open) / max(high - low, 1 KRW)"),
    TopkFeatureSpec("upper_shadow_ratio", FeatureUnit.DECIMAL_RATIO, FeatureTiming.DECISION_SNAPSHOT, "(high - max(open, close)) / max(high - low, 1 KRW)"),
    TopkFeatureSpec("intraday_range", FeatureUnit.DECIMAL_RATIO, FeatureTiming.DECISION_SNAPSHOT, "(high - low) / close"),
    TopkFeatureSpec("kospi_pct", FeatureUnit.DECIMAL_RETURN, FeatureTiming.DECISION_SNAPSHOT, "KOSPI index day change as a fraction"),
    TopkFeatureSpec("kosdaq_pct", FeatureUnit.DECIMAL_RETURN, FeatureTiming.DECISION_SNAPSHOT, "KOSDAQ index day change as a fraction"),
    TopkFeatureSpec("v_kospi", FeatureUnit.INDEX_LEVEL, FeatureTiming.DECISION_SNAPSHOT, "V-KOSPI index level"),
    TopkFeatureSpec("tv_rank", FeatureUnit.PCT_RANK, FeatureTiming.DECISION_SNAPSHOT, "within-date percentile rank of trade value over the candidate cross-section"),
    TopkFeatureSpec("chg_rank", FeatureUnit.PCT_RANK, FeatureTiming.DECISION_SNAPSHOT, "within-date percentile rank of chg_ratio over the candidate cross-section"),
    TopkFeatureSpec("inst_density", FeatureUnit.DECIMAL_RATIO, FeatureTiming.PRIOR_CONFIRMED, "T-1 institutional net buy / T-1 traded value in KRW, clipped to [-1, 1]"),
    TopkFeatureSpec("inst_rank", FeatureUnit.PCT_RANK, FeatureTiming.PRIOR_CONFIRMED, "within-date percentile rank of T-1 institutional net buy"),
    TopkFeatureSpec("f_tick_cost", FeatureUnit.BASIS_POINTS, FeatureTiming.DECISION_SNAPSHOT, "point-in-time one-tick cost at the decision price"),
    TopkFeatureSpec("f_log_close", FeatureUnit.LOG_KRW, FeatureTiming.DECISION_SNAPSHOT, "ln(raw close price level)"),
    TopkFeatureSpec("f_ret5", FeatureUnit.LOG_RETURN_SUM, FeatureTiming.PRIOR_CONFIRMED, "sum of ln(1 + chg) over [t-5, t-1]"),
    TopkFeatureSpec("f_ret20", FeatureUnit.LOG_RETURN_SUM, FeatureTiming.PRIOR_CONFIRMED, "sum of ln(1 + chg) over [t-20, t-1]"),
    TopkFeatureSpec("f_ret60", FeatureUnit.LOG_RETURN_SUM, FeatureTiming.PRIOR_CONFIRMED, "sum of ln(1 + chg) over [t-60, t-1]"),
    TopkFeatureSpec("f_dist_high60", FeatureUnit.DECIMAL_RETURN, FeatureTiming.SNAPSHOT_AND_PRIOR, "exp(cumulative log return at T - its trailing 60-row max including T) - 1"),
    TopkFeatureSpec("f_upcount20", FeatureUnit.COUNT, FeatureTiming.PRIOR_CONFIRMED, "number of days with chg >= DEFAULT_UNIVERSE.chg_min over [t-20, t-1]"),
    TopkFeatureSpec("f_on_mean20", FeatureUnit.DECIMAL_RETURN, FeatureTiming.PRIOR_CONFIRMED, "mean overnight gap open/prev_close - 1 over [t-20, t-1]"),
    TopkFeatureSpec("f_on_mean60", FeatureUnit.DECIMAL_RETURN, FeatureTiming.PRIOR_CONFIRMED, "mean overnight gap open/prev_close - 1 over [t-60, t-1]"),
    TopkFeatureSpec("f_upnext_on60", FeatureUnit.DECIMAL_RETURN, FeatureTiming.SNAPSHOT_AND_PRIOR, "mean next-day overnight gap after up days s <= t-1 within 60 rows (includes gap_T); NaN below MIN_CONDITIONAL_OBS"),
    TopkFeatureSpec("f_gap", FeatureUnit.DECIMAL_RETURN, FeatureTiming.DECISION_SNAPSHOT, "open_T / prev_close_T - 1"),
    TopkFeatureSpec("f_id_mean20", FeatureUnit.DECIMAL_RETURN, FeatureTiming.PRIOR_CONFIRMED, "mean intraday return close/open - 1 over [t-20, t-1]"),
    TopkFeatureSpec("f_inst_cum5", FeatureUnit.DECIMAL_RATIO, FeatureTiming.PRIOR_CONFIRMED, "sum institutional net buy / sum traded value over [t-5, t-1], clipped to [-1, 1]"),
    TopkFeatureSpec("f_foreign_cum5", FeatureUnit.DECIMAL_RATIO, FeatureTiming.PRIOR_CONFIRMED, "sum foreign net buy / sum traded value over [t-5, t-1], clipped to [-1, 1]"),
)

# Bump only when a feature's definition, unit, timing or input semantics change for
# any name in TOPK_FEATURE_SPECS. A bump requires retraining and updating the version
# pins in tests/unit/ml/test_topk_contract.py.
TOPK_FEATURE_CONTRACT_VERSION: str = "1"
# This is the definition set in force when the key was introduced (2026-10). Bundles
# without the key were trained under it. Never bump it.
LEGACY_FEATURE_CONTRACT_VERSION: str = "1"
FEATURE_CONTRACT_VERSION_KEY: str = "feature_contract_version"


def build_topk_feature_manifest(feature_cols: Sequence[str]) -> pd.DataFrame:
    """Render the declared feature table for a bundle's feature list.

    The manifest is an audit record persisted inside the bundle; it is derived from
    TOPK_FEATURE_SPECS, never from feature-name heuristics.

    Args:
        feature_cols: Bundle feature names, in model column order.

    Returns:
        One row per feature in the given order with columns feature_name, source_column,
        availability_rule, unit, panel_scope, definition (source_column = feature_name,
        availability_rule = timing value, panel_scope = "candidate_panel").

    Raises:
        TypeError: When feature_cols is a str.
        ValueError: Naming every feature absent from TOPK_FEATURE_SPECS.
    """
    if isinstance(feature_cols, str):
        raise TypeError("feature_cols must be a sequence of feature names, not a string")
    by_name = {spec.name: spec for spec in TOPK_FEATURE_SPECS}
    unknown = [name for name in feature_cols if name not in by_name]
    if unknown:
        raise ValueError(f"features unknown to the top-k contract: {unknown}")
    rows = [
        {
            "feature_name": spec.name,
            "source_column": spec.name,
            "availability_rule": spec.timing.value,
            "unit": spec.unit.value,
            "panel_scope": "candidate_panel",
            "definition": spec.definition,
        }
        for spec in (by_name[name] for name in feature_cols)
    ]
    return pd.DataFrame(
        rows,
        columns=["feature_name", "source_column", "availability_rule", "unit", "panel_scope", "definition"],
    )


def bundle_feature_contract_version(bundle: Mapping[str, Any]) -> str:
    """Resolve the feature-contract version a bundle was trained under.

    Bundles published before the version key existed were trained under the baseline
    definitions and resolve to LEGACY_FEATURE_CONTRACT_VERSION.

    Args:
        bundle: Loaded or freshly trained bundle.

    Returns:
        The stamped version, or LEGACY_FEATURE_CONTRACT_VERSION when the key is absent.

    Raises:
        ValueError: When the key is present but its value is not a non-empty str.
    """
    if FEATURE_CONTRACT_VERSION_KEY not in bundle:
        return LEGACY_FEATURE_CONTRACT_VERSION
    value = bundle[FEATURE_CONTRACT_VERSION_KEY]
    if not isinstance(value, str) or value == "":
        raise ValueError(
            f"bundle {FEATURE_CONTRACT_VERSION_KEY!r} must be a non-empty str, got {value!r}"
        )
    return value


def feature_contract_issue(bundle: Mapping[str, Any]) -> str | None:
    """Describe why a bundle's feature contract is incompatible with this code, if it is.

    Args:
        bundle: Loaded or freshly trained bundle.

    Returns:
        None when the resolved version equals TOPK_FEATURE_CONTRACT_VERSION and a keyless bundle is
        acceptable (TOPK_FEATURE_CONTRACT_VERSION == LEGACY_FEATURE_CONTRACT_VERSION); otherwise a
        one-line message naming the bundle version (or "missing") and the code version.
    """
    try:
        resolved = bundle_feature_contract_version(bundle)
    except ValueError as exc:
        return str(exc)
    if FEATURE_CONTRACT_VERSION_KEY not in bundle:
        if TOPK_FEATURE_CONTRACT_VERSION == LEGACY_FEATURE_CONTRACT_VERSION:
            return None
        return (
            f"bundle feature contract version missing (legacy baseline "
            f"{LEGACY_FEATURE_CONTRACT_VERSION!r}) != code version {TOPK_FEATURE_CONTRACT_VERSION!r}"
        )
    if resolved != TOPK_FEATURE_CONTRACT_VERSION:
        return (
            f"bundle feature contract version {resolved!r} != "
            f"code version {TOPK_FEATURE_CONTRACT_VERSION!r}"
        )
    return None


def assert_feature_contract(bundle: Mapping[str, Any], *, source: str) -> None:
    """Fail closed when a bundle's features were defined differently from this code's.

    A name-identical feature whose formula or unit changed silently feeds the model mis-scaled
    inputs; the explicit version is the only signal that survives such a change.

    Args:
        bundle: Loaded bundle.
        source: Bundle location, logged for correlation.

    Raises:
        ValueError: With the feature_contract_issue message when incompatible.

    Side effects:
        Logs one `[ALGO]` WARNING when a keyless bundle is accepted as the legacy baseline.
    """
    issue = feature_contract_issue(bundle)
    if issue is not None:
        raise ValueError(issue)
    if FEATURE_CONTRACT_VERSION_KEY not in bundle:
        logger.warning(
            "[ALGO] stage=bundle_load feature_contract=legacy_assumed version=%s source=%s",
            LEGACY_FEATURE_CONTRACT_VERSION,
            source,
        )
    return None


def compute_derived_features(cands: pd.DataFrame) -> pd.DataFrame:
    """Compute 11 decision-time features strictly using decision candidate set."""
    logger.info("Computing derived decision-time features and cross-sectional ranks...")
    p_close = cands["close"].to_numpy(dtype=np.float64)
    p_open = cands["open"].to_numpy(dtype=np.float64)
    p_high = cands["high"].to_numpy(dtype=np.float64)
    p_low = cands["low"].to_numpy(dtype=np.float64)

    rg = np.maximum(p_high - p_low, 1.0)
    cands["body_ratio"] = (p_close - p_open) / rg
    cands["upper_shadow_ratio"] = (p_high - np.maximum(p_open, p_close)) / rg
    cands["intraday_range"] = (p_high - p_low) / p_close
    cands["log_tv"] = np.log1p(np.maximum(cands["tv_clean"].to_numpy(dtype=np.float64), 0.0))
    cands["log_mc"] = np.log1p(np.maximum(cands["mc_clean"].to_numpy(dtype=np.float64), 0.0))

    # Cross-sectional ranks across ALL valid candidates on date T
    grouped_date = cands.groupby("date", sort=False)
    cands["tv_rank"] = grouped_date["tv_clean"].rank(pct=True)
    cands["chg_rank"] = grouped_date["chg_ratio"].rank(pct=True)

    return cands


def _screen_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    return float(value)


SCREEN_PARITY_CLASS_FILTER_GRANDFATHERED_STRATEGY_IDS: frozenset[str] = frozenset({"KCA-TOPK-COSTAWARE-001"})
"""Strategies whose bundles may serve a class-filtered live screen although certified without it.

Since 87e3eab the live cohort excludes non-screenable security classes regardless of the
bundle, so serving these bundles on a class-filtered pool is the pre-existing live state.
The allowance covers only exclude_non_screenable_class moving False -> True; every other
field stays strict.
"""


def certified_screen(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bundle's certified universe with omitted fields read as UniverseSpec defaults.

    Args:
        bundle: Production bundle carrying ``select_universe``.

    Returns:
        Dict over every UniverseSpec field with numeric values normalized as parity compares
        them (bool/None kept, numbers as float).

    Raises:
        ValueError: When select_universe is not a dict or carries fields unknown to UniverseSpec.
    """
    screened = bundle.get("select_universe")
    if not isinstance(screened, dict):
        raise ValueError(f"bundle select_universe is not certified: got {screened!r}")
    live = dataclasses.asdict(PRODUCTION_STRATEGY.universe)
    unknown = sorted(set(screened) - set(live))
    if unknown:
        raise ValueError(f"bundle select_universe carries fields unknown to live screen: {unknown}")
    defaults = dataclasses.asdict(UniverseSpec())
    return {k: _screen_value(screened.get(k, defaults[k])) for k in live}


def assert_bundle_screen_parity(bundle: dict[str, Any], spec: UniverseSpec = PRODUCTION_STRATEGY.universe) -> None:
    """Fail closed unless the bundle was certified under the live selection screen.

    Args:
        bundle: Production bundle carrying ``select_universe`` (asdict of the certified
            UniverseSpec) and ``strategy_id``.
        spec: Live selection screen; defaults to PRODUCTION_STRATEGY.universe.

    Returns:
        None when every live screen field matches. A field absent from the bundle is read as
        the UniverseSpec default (the bundle predates that field). A bundle whose strategy_id
        is in SCREEN_PARITY_CLASS_FILTER_GRANDFATHERED_STRATEGY_IDS also passes when the only
        difference is exclude_non_screenable_class certified False while live is True; that
        acceptance is logged at WARNING.

    Raises:
        ValueError: When select_universe is not a dict, carries fields unknown to the live
            screen, or any field differs from the live value outside the grandfather rule.
    """
    screened = bundle.get("select_universe")
    if not isinstance(screened, dict):
        raise ValueError(f"bundle select_universe is not certified: got {screened!r}")
    live = dataclasses.asdict(spec)
    unknown = sorted(set(screened) - set(live))
    if unknown:
        raise ValueError(f"bundle select_universe carries fields unknown to live screen: {unknown}")
    # 번들에 없는 키 = 필드 도입 이전 학습 → 기본값(도입 이전 동작)으로 학습된 것으로 간주
    defaults = dataclasses.asdict(UniverseSpec())
    certified = {k: screened.get(k, defaults[k]) for k in live}
    differing = [k for k in live if _screen_value(certified[k]) != _screen_value(live[k])]
    if differing == ["exclude_non_screenable_class"]:
        strategy_id = bundle.get("strategy_id")
        if (
            strategy_id in SCREEN_PARITY_CLASS_FILTER_GRANDFATHERED_STRATEGY_IDS
            and _screen_value(certified["exclude_non_screenable_class"]) is False
            and _screen_value(live["exclude_non_screenable_class"]) is True
        ):
            logger.warning(
                "[RISK] stage=screen_parity status=GRANDFATHERED strategy_id=%s "
                "field=exclude_non_screenable_class certified=False live=True",
                strategy_id,
            )
            return None
    if differing:
        bundle_vals = {k: certified[k] for k in differing}
        live_vals = {k: live[k] for k in differing}
        raise ValueError(
            f"bundle select_universe differs from live screen in fields {differing}: "
            f"bundle={bundle_vals} live={live_vals}"
        )
    return None


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


def save_production_bundle(bundle: dict[str, Any], export_dir: str = TOPK_RANKER_BUNDLE_DIR) -> str:
    """Persist a production bundle under the reranker artifact directory.

    Args:
        bundle: Extended bundle dict from train_production_bundle.
        export_dir: Destination directory, distinct from champion's directory.

    Returns:
        Saved joblib path as a string.
    """
    from joblib import dump

    from src.data.io_utils import atomic_output_path

    # 챔피언 번들과 파일명 충돌 방지용 별도 디렉터리
    path = os.path.join(export_dir, "sizing_pipeline_bundle.joblib")
    with atomic_output_path(path, mode=None) as tmp_path:
        dump(bundle, tmp_path)
    # 추론 쪽이 부분 기록된 번들을 읽지 않도록 같은 디렉터리에서 원자적으로 교체한다
    return path


def score_topk_candidates(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """Score every snapshot row with the production bundle models.

    Args:
        df: Live snapshot with the bundle's feature columns.
        bundle: Production bundle carrying return/quantile/calibrator models.

    Returns:
        Copy of df with pred, quantile and calibration score columns.

    Raises:
        ValueError: When feature_cols is empty or a declared feature column
            is missing from the snapshot.
    """
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
    return work


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
    work = score_topk_candidates(df, bundle)
    pool = work
    if admitted_col in work.columns:
        pool = work[np.asarray(work[admitted_col], dtype=bool)]
    admitted_counts = pool.groupby(date_col, sort=False).size()
    certified_dates = admitted_counts[admitted_counts >= int(top_k)].index
    pool = pool[pool[date_col].isin(certified_dates)]
    picks = select_topk_by_score(pool, int(top_k), score_col="pred", date_col=date_col)
    picks["allocation"] = 1.0 / float(top_k)
    return picks
