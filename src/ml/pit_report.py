"""Contract of the decision-time (15:20) certification report shared by research, gate and bundle metadata."""

from __future__ import annotations

import enum
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.io_utils import atomic_write_parquet, atomic_write_text

PIT_HAIRCUT_REPORT_FILENAME: str = "pit_haircut_report.json"
PIT_HAIRCUT_DAILY_FILENAME: str = "pit_haircut_daily.parquet"
PIT_REPORT_SCHEMA_VERSION: int = 1
PIT_CERTIFICATION_BUNDLE_KEY: str = "pit_certification"


class PitReportStatus(enum.StrEnum):
    """Whether a report carries usable paired evidence."""

    OK = "OK"
    INSUFFICIENT_DAYS = "INSUFFICIENT_DAYS"
    INSUFFICIENT_COVERAGE = "INSUFFICIENT_COVERAGE"


@dataclass(frozen=True)
class PairedDelta:
    """Mean paired daily difference (first minus second) with a day-block bootstrap interval, in bp or IC units."""

    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n_days: int


@dataclass(frozen=True)
class AugmentationSummary:
    """Report-only outcome of the auction-noise augmentation experiment on the 15:20 panel."""

    status: str  # "OK" | "INSUFFICIENT_SOURCE" | "INSUFFICIENT_DAYS"
    improvement_bp: PairedDelta | None  # augmented PIT-native minus baseline PIT-native
    declared_trials: int
    alpha: float
    verdict: str  # "ADOPT_CANDIDATE" | "REJECT" | "NOT_EVALUATED"


@dataclass(frozen=True)
class PitHaircutReport:
    """Paired EOD-vs-15:20 certification evidence for one strategy/feature/model configuration.

    Attributes:
        schema_version: PIT_REPORT_SCHEMA_VERSION at write time.
        generated_at: Aware ISO-8601 KST timestamp of the run.
        strategy_id, strategy_fingerprint, top_k, select_universe: Strategy identity (StrategySpec).
        feature_contract_version, model_params, seeds: Model identity the fold models were trained under.
        status: PitReportStatus value.
        panel_date_min, panel_date_max: Usable panel date span (YYYY-MM-DD; "" when none).
        n_usable_days, n_paired_days, n_live_days, n_eod_index_days: Day counts (live = index_basis live_1520).
        mean_net_bp: Arm -> daily top-k mean net_pit in bp over paired days
            (keys eod_full, eod_matched, pit_feature, pit_native).
        haircut: eod_full minus pit_native (bp).
        coverage_component, feature_component, selection_component: Decomposition (bp, PairedDelta each).
        pit_native_vs_zero: pit_native minus 0 (bp) — its CI bounds absolute 15:20 performance.
        rank_ic_mean: Arm -> mean daily Spearman(pred, net_pit) over the arm's selectable pool.
        rank_ic_haircut: eod_full IC minus pit_native IC (PairedDelta).
        pick_overlap_mean: "pit_feature" and "pit_native" -> mean |picks ∩ eod picks| / top_k.
        haircut_by_index_basis: "live_1520" / "eod_fallback" -> mean haircut bp (report-only).
        augmentation: AugmentationSummary or None when the experiment was not requested.
    """

    schema_version: int = PIT_REPORT_SCHEMA_VERSION
    generated_at: str = ""
    strategy_id: str = ""
    strategy_fingerprint: str = ""
    top_k: int = 0
    select_universe: dict[str, Any] = None  # type: ignore[assignment]
    feature_contract_version: str = ""
    model_params: dict[str, Any] = None  # type: ignore[assignment]
    seeds: tuple[int, ...] = ()
    status: PitReportStatus = PitReportStatus.INSUFFICIENT_COVERAGE
    panel_date_min: str = ""
    panel_date_max: str = ""
    n_usable_days: int = 0
    n_paired_days: int = 0
    n_live_days: int = 0
    n_eod_index_days: int = 0
    mean_net_bp: dict[str, float] = None  # type: ignore[assignment]
    haircut: PairedDelta = None  # type: ignore[assignment]
    coverage_component: PairedDelta = None  # type: ignore[assignment]
    feature_component: PairedDelta = None  # type: ignore[assignment]
    selection_component: PairedDelta = None  # type: ignore[assignment]
    pit_native_vs_zero: PairedDelta = None  # type: ignore[assignment]
    rank_ic_mean: dict[str, float] = None  # type: ignore[assignment]
    rank_ic_haircut: PairedDelta = None  # type: ignore[assignment]
    pick_overlap_mean: dict[str, float] = None  # type: ignore[assignment]
    haircut_by_index_basis: dict[str, float] = None  # type: ignore[assignment]
    augmentation: AugmentationSummary | None = None


def _dump_float(value: float) -> float | None:
    val = float(value)
    return None if not math.isfinite(val) else val


def _load_float(value: float | None, *, field_name: str) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"pit_haircut report field {field_name!r} must be a number or null, got {value!r}")
    return float(value)


def _dump_delta(delta: PairedDelta) -> dict[str, Any]:
    return {
        "delta": _dump_float(delta.delta),
        "ci_low": _dump_float(delta.ci_low),
        "ci_high": _dump_float(delta.ci_high),
        "p_value": _dump_float(delta.p_value),
        "n_days": int(delta.n_days),
    }


def _load_delta(payload: Any, *, field_name: str) -> PairedDelta:
    if not isinstance(payload, Mapping):
        raise ValueError(f"pit_haircut report field {field_name!r} must be a mapping, got {type(payload).__name__}")
    unknown = set(payload) - {"delta", "ci_low", "ci_high", "p_value", "n_days"}
    if unknown:
        raise ValueError(f"pit_haircut report field {field_name!r} carries unknown keys: {sorted(unknown)}")
    try:
        n_days = int(payload["n_days"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"pit_haircut report field {field_name!r}.n_days must be an int") from exc
    return PairedDelta(
        delta=_load_float(payload.get("delta"), field_name=f"{field_name}.delta"),
        ci_low=_load_float(payload.get("ci_low"), field_name=f"{field_name}.ci_low"),
        ci_high=_load_float(payload.get("ci_high"), field_name=f"{field_name}.ci_high"),
        p_value=_load_float(payload.get("p_value"), field_name=f"{field_name}.p_value"),
        n_days=n_days,
    )


def _dump_augmentation(summary: AugmentationSummary | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    return {
        "status": str(summary.status),
        "improvement_bp": _dump_delta(summary.improvement_bp) if summary.improvement_bp is not None else None,
        "declared_trials": int(summary.declared_trials),
        "alpha": _dump_float(float(summary.alpha)),
        "verdict": str(summary.verdict),
    }


def _load_augmentation(payload: Any) -> AugmentationSummary | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("pit_haircut report field 'augmentation' must be a mapping or null")
    unknown = set(payload) - {"status", "improvement_bp", "declared_trials", "alpha", "verdict"}
    if unknown:
        raise ValueError(f"pit_haircut report field 'augmentation' carries unknown keys: {sorted(unknown)}")
    improvement = payload.get("improvement_bp")
    return AugmentationSummary(
        status=str(payload.get("status", "")),
        improvement_bp=_load_delta(improvement, field_name="augmentation.improvement_bp") if improvement is not None else None,
        declared_trials=int(payload.get("declared_trials", 0)),
        alpha=_load_float(payload.get("alpha"), field_name="augmentation.alpha"),
        verdict=str(payload.get("verdict", "")),
    )


def _dump_float_map(mapping: Mapping[str, float]) -> dict[str, float | None]:
    return {str(key): _dump_float(value) for key, value in mapping.items()}


def _load_float_map(payload: Any, *, field_name: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"pit_haircut report field {field_name!r} must be a mapping")
    return {str(key): _load_float(value, field_name=f"{field_name}.{key}") for key, value in payload.items()}


_REPORT_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(PitHaircutReport))


def _report_to_payload(report: PitHaircutReport) -> dict[str, Any]:
    if report.select_universe is None or report.model_params is None or report.mean_net_bp is None:
        raise ValueError("pit_haircut report has unpopulated required fields; refusing to persist a partial report")
    for name in ("haircut", "coverage_component", "feature_component", "selection_component", "pit_native_vs_zero", "rank_ic_haircut"):
        if getattr(report, name) is None:
            raise ValueError(f"pit_haircut report field {name!r} is unpopulated; refusing to persist a partial report")
    if report.rank_ic_mean is None or report.pick_overlap_mean is None or report.haircut_by_index_basis is None:
        raise ValueError("pit_haircut report has unpopulated required fields; refusing to persist a partial report")
    return {
        "schema_version": int(report.schema_version),
        "generated_at": str(report.generated_at),
        "strategy_id": str(report.strategy_id),
        "strategy_fingerprint": str(report.strategy_fingerprint),
        "top_k": int(report.top_k),
        "select_universe": json.loads(json.dumps(dict(report.select_universe))),
        "feature_contract_version": str(report.feature_contract_version),
        "model_params": json.loads(json.dumps(dict(report.model_params))),
        "seeds": [int(s) for s in report.seeds],
        "status": PitReportStatus(report.status).value,
        "panel_date_min": str(report.panel_date_min),
        "panel_date_max": str(report.panel_date_max),
        "n_usable_days": int(report.n_usable_days),
        "n_paired_days": int(report.n_paired_days),
        "n_live_days": int(report.n_live_days),
        "n_eod_index_days": int(report.n_eod_index_days),
        "mean_net_bp": _dump_float_map(report.mean_net_bp),
        "haircut": _dump_delta(report.haircut),
        "coverage_component": _dump_delta(report.coverage_component),
        "feature_component": _dump_delta(report.feature_component),
        "selection_component": _dump_delta(report.selection_component),
        "pit_native_vs_zero": _dump_delta(report.pit_native_vs_zero),
        "rank_ic_mean": _dump_float_map(report.rank_ic_mean),
        "rank_ic_haircut": _dump_delta(report.rank_ic_haircut),
        "pick_overlap_mean": _dump_float_map(report.pick_overlap_mean),
        "haircut_by_index_basis": _dump_float_map(report.haircut_by_index_basis),
        "augmentation": _dump_augmentation(report.augmentation),
    }


def _payload_to_report(payload: Mapping[str, Any]) -> PitHaircutReport:
    unknown = set(payload) - set(_REPORT_FIELDS)
    if unknown:
        raise ValueError(f"pit_haircut report carries unknown keys: {sorted(unknown)}")
    version = payload.get("schema_version")
    if version != PIT_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"pit_haircut report schema_version {version!r} != supported {PIT_REPORT_SCHEMA_VERSION}"
        )
    missing = [name for name in _REPORT_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"pit_haircut report is missing keys: {missing}")
    status_raw = payload.get("status")
    try:
        status = PitReportStatus(str(status_raw))
    except ValueError as exc:
        raise ValueError(f"pit_haircut report status {status_raw!r} is unknown") from exc
    select_universe = payload.get("select_universe")
    model_params = payload.get("model_params")
    if not isinstance(select_universe, Mapping):
        raise ValueError("pit_haircut report field 'select_universe' must be a mapping")
    if not isinstance(model_params, Mapping):
        raise ValueError("pit_haircut report field 'model_params' must be a mapping")
    seeds_raw = payload.get("seeds")
    if not isinstance(seeds_raw, Sequence) or isinstance(seeds_raw, str):
        raise ValueError("pit_haircut report field 'seeds' must be a sequence")
    return PitHaircutReport(
        schema_version=int(version),
        generated_at=str(payload.get("generated_at", "")),
        strategy_id=str(payload.get("strategy_id", "")),
        strategy_fingerprint=str(payload.get("strategy_fingerprint", "")),
        top_k=int(payload.get("top_k", 0)),
        select_universe=dict(select_universe),
        feature_contract_version=str(payload.get("feature_contract_version", "")),
        model_params=dict(model_params),
        seeds=tuple(int(s) for s in seeds_raw),
        status=status,
        panel_date_min=str(payload.get("panel_date_min", "")),
        panel_date_max=str(payload.get("panel_date_max", "")),
        n_usable_days=int(payload.get("n_usable_days", 0)),
        n_paired_days=int(payload.get("n_paired_days", 0)),
        n_live_days=int(payload.get("n_live_days", 0)),
        n_eod_index_days=int(payload.get("n_eod_index_days", 0)),
        mean_net_bp=_load_float_map(payload.get("mean_net_bp"), field_name="mean_net_bp"),
        haircut=_load_delta(payload.get("haircut"), field_name="haircut"),
        coverage_component=_load_delta(payload.get("coverage_component"), field_name="coverage_component"),
        feature_component=_load_delta(payload.get("feature_component"), field_name="feature_component"),
        selection_component=_load_delta(payload.get("selection_component"), field_name="selection_component"),
        pit_native_vs_zero=_load_delta(payload.get("pit_native_vs_zero"), field_name="pit_native_vs_zero"),
        rank_ic_mean=_load_float_map(payload.get("rank_ic_mean"), field_name="rank_ic_mean"),
        rank_ic_haircut=_load_delta(payload.get("rank_ic_haircut"), field_name="rank_ic_haircut"),
        pick_overlap_mean=_load_float_map(payload.get("pick_overlap_mean"), field_name="pick_overlap_mean"),
        haircut_by_index_basis=_load_float_map(payload.get("haircut_by_index_basis"), field_name="haircut_by_index_basis"),
        augmentation=_load_augmentation(payload.get("augmentation")),
    )


def save_pit_haircut_report(report: PitHaircutReport, daily: pd.DataFrame, *, out_dir: Path) -> tuple[Path, Path]:
    """Atomically write the JSON report and the per-day evidence parquet into out_dir.

    Raises:
        OSError: Persistence fails (nothing partial is visible).
    """
    payload = _report_to_payload(report)
    text = json.dumps(payload, indent=2, sort_keys=True)
    out = Path(out_dir)
    report_path = out / PIT_HAIRCUT_REPORT_FILENAME
    daily_path = out / PIT_HAIRCUT_DAILY_FILENAME
    atomic_write_text(report_path, text, mode=None)
    atomic_write_parquet(daily.reset_index(drop=True), daily_path)
    return report_path, daily_path


def load_pit_haircut_report(out_dir: Path) -> PitHaircutReport | None:
    """Load the report from out_dir, or None when the file does not exist.

    Raises:
        ValueError: The file exists but is malformed or has an unknown schema_version (fail closed; a corrupt
            report must never read as absent).
    """
    path = Path(out_dir) / PIT_HAIRCUT_REPORT_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"pit_haircut report at {path} is malformed: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"pit_haircut report at {path} must hold a JSON object")
    try:
        return _payload_to_report(payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"pit_haircut report at {path} is malformed: {exc}") from exc


def pit_certification_metadata(
    report: PitHaircutReport | None, *, gate_mode: str, gate_status: str, gate_reasons: Sequence[str]
) -> dict[str, Any]:
    """Render the bundle-metadata view of a report plus the gate outcome.

    Returns:
        JSON-serializable dict: status ("MISSING" when report is None), generated_at, panel_date_min/max,
        n_paired_days, mean_net_bp, haircut (delta/ci_low/ci_high/p_value), rank_ic_mean, pick_overlap_mean,
        gate_mode, gate_status, gate_reasons (list).
    """
    if report is None:
        return {
            "status": "MISSING",
            "generated_at": None,
            "panel_date_min": "",
            "panel_date_max": "",
            "n_paired_days": 0,
            "mean_net_bp": {},
            "haircut": {"delta": None, "ci_low": None, "ci_high": None, "p_value": None},
            "rank_ic_mean": {},
            "pick_overlap_mean": {},
            "gate_mode": str(gate_mode),
            "gate_status": str(gate_status),
            "gate_reasons": [str(r) for r in gate_reasons],
        }
    haircut = report.haircut
    return {
        "status": PitReportStatus(report.status).value,
        "generated_at": str(report.generated_at),
        "panel_date_min": str(report.panel_date_min),
        "panel_date_max": str(report.panel_date_max),
        "n_paired_days": int(report.n_paired_days),
        "mean_net_bp": _dump_float_map(report.mean_net_bp),
        "haircut": {
            "delta": _dump_float(haircut.delta),
            "ci_low": _dump_float(haircut.ci_low),
            "ci_high": _dump_float(haircut.ci_high),
            "p_value": _dump_float(haircut.p_value),
        },
        "rank_ic_mean": _dump_float_map(report.rank_ic_mean),
        "pick_overlap_mean": _dump_float_map(report.pick_overlap_mean),
        "gate_mode": str(gate_mode),
        "gate_status": str(gate_status),
        "gate_reasons": [str(r) for r in gate_reasons],
    }


def bundle_pit_certification(bundle: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the bundle's pit_certification mapping, or None for bundles that predate the key.

    Raises:
        ValueError: The key is present but not a mapping.
    """
    if PIT_CERTIFICATION_BUNDLE_KEY not in bundle:
        return None
    value = bundle[PIT_CERTIFICATION_BUNDLE_KEY]
    if not isinstance(value, Mapping):
        raise ValueError(
            f"bundle {PIT_CERTIFICATION_BUNDLE_KEY!r} must be a mapping, got {type(value).__name__}"
        )
    return value
