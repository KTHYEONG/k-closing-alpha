"""Weekly retrain promotion gate.

A retrained ranker bundle replaces the live one only when it is structurally
certified for the live screen and ranks the most recent selection pool
consistently with the bundle it replaces. This guards the unattended weekly
retrain against silently promoting a model fit on corrupted or truncated data;
it does not re-certify alpha (that remains the manual CPCV certification).
"""

from __future__ import annotations

import enum
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.ml.pit_report import PitHaircutReport
from src.ml.topk_contract import (
    assert_bundle_screen_parity,
    bundle_feature_contract_version,
    certified_screen,
    feature_contract_issue,
)
from src.ml.topk_ranker_research import build_dual_pool
from src.strategy.contract import PRODUCTION_STRATEGY, training_universe

RETRAIN_GATE_EVAL_DAYS: int = 20
# 실측(2026-09-13, 최근 20거래일 선택풀): 주간 연속 재학습 간 일별 순위상관 평균 0.983~0.989,
# 최근 1년 시가를 ±10% 오염시켜 학습하면 0.808 -> 정상 변동과 데이터 오염을 가르는 하한
RETRAIN_MIN_PREDICTION_AGREEMENT: float = 0.95
# 순위상관은 표본 4개 미만인 날에 의미가 없어 집계에서 제외한다
MIN_NAMES_PER_EVAL_DAY: int = 4
BUNDLE_FILENAME: str = "sizing_pipeline_bundle.joblib"

_KST = ZoneInfo("Asia/Seoul")


class PitGateMode(enum.StrEnum):
    """How the decision-time certification participates in promotion."""

    OFF = "off"
    ADVISORY = "advisory"
    ENFORCE = "enforce"


@dataclass(frozen=True)
class PitGateConfig:
    """Promotion criterion on the latest pit_haircut report.

    Attributes:
        mode: OFF skips evaluation; ADVISORY records the outcome without blocking; ENFORCE blocks on non-PASS.
        max_report_age_days: Calendar days after which a report is STALE.
        min_pit_native_mean_net_bp: Floor on the 15:20-native top-k daily mean net return (bp).
        min_pit_rank_ic: Floor on the 15:20-native mean daily rank IC.
        max_haircut_bp: Optional ceiling on eod_full minus pit_native (bp); None disables it.
    """

    mode: PitGateMode = PitGateMode.OFF
    max_report_age_days: int = 14
    min_pit_native_mean_net_bp: float = 0.0
    min_pit_rank_ic: float = 0.0
    max_haircut_bp: float | None = None


@dataclass(frozen=True)
class PromotionVerdict:
    """Promotion decision for a retrained candidate bundle.

    Attributes:
        promote: True only when no reason blocks promotion.
        reasons: Human-readable blocking reasons (empty when promote is True).
        agreement: Mean daily rank agreement with the live bundle; None when not measured.
        pit_status: OFF, MISSING, MALFORMED, STALE, MISMATCH, INSUFFICIENT, PASS or FAIL.
        pit_reasons: Human-readable PIT findings (blocking only in ENFORCE mode).
    """

    promote: bool
    reasons: tuple[str, ...]
    agreement: float | None
    pit_status: str = "OFF"
    pit_reasons: tuple[str, ...] = ()


def load_current_bundle(export_dir: str) -> dict[str, Any] | None:
    """Load the live bundle from export_dir, or None when none has been published.

    Args:
        export_dir: Directory holding the published sizing_pipeline_bundle.joblib.

    Returns:
        The deserialized bundle, or None when the file does not exist.
    """
    from joblib import load

    path = os.path.join(export_dir, BUNDLE_FILENAME)
    if not os.path.isfile(path):
        return None
    bundle: dict[str, Any] = load(path)
    return bundle


def build_gate_eval_frame(
    ph: pd.DataFrame,
    market_dates: np.ndarray,
    d_to_idx: dict[pd.Timestamp, int],
    *,
    eval_days: int = RETRAIN_GATE_EVAL_DAYS,
) -> pd.DataFrame:
    """Return the live-screen selection pool rows of the most recent eval_days dates.

    Args:
        ph: Prepared price-history panel.
        market_dates: Full trading calendar.
        d_to_idx: Date-to-index lookup for forward exits.
        eval_days: Number of most recent selection dates to keep.

    Returns:
        Selection-pool rows (with decision-time features) for the recent dates.
    """
    pool, sel_mask = build_dual_pool(
        ph, market_dates, d_to_idx, train_spec=training_universe(PRODUCTION_STRATEGY.universe), select_spec=PRODUCTION_STRATEGY.universe
    )
    selected = pool.loc[np.asarray(sel_mask, dtype=bool)]
    dates = pd.to_datetime(selected["date"])
    recent = sorted(dates.unique())[-int(eval_days):]
    return selected[dates.isin(recent)].reset_index(drop=True)


def _resolved_contract(bundle: dict[str, Any]) -> str | None:
    try:
        return bundle_feature_contract_version(bundle)
    except ValueError:
        return None


def _mean_daily_rank_agreement(dates: np.ndarray, first: np.ndarray, second: np.ndarray) -> float | None:
    frame = pd.DataFrame({"date": dates, "first": first, "second": second})
    per_day = [
        group["first"].rank().corr(group["second"].rank())
        for _, group in frame.groupby("date", sort=True)
        if len(group) >= MIN_NAMES_PER_EVAL_DAY
    ]
    finite = [value for value in per_day if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _as_sequence(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def evaluate_pit_gate(
    report: PitHaircutReport | None,
    candidate: Mapping[str, Any],
    *,
    config: PitGateConfig,
    now: datetime,
) -> tuple[str, tuple[str, ...]]:
    """Judge whether the decision-time evidence supports the candidate's configuration.

    The report certifies CPCV fold models, not the weekly candidate itself (scoring the candidate on panel
    days would be in-sample); it is therefore accepted only when it was produced recently for the same
    strategy, screen, feature contract, model parameters and seeds as the candidate.

    Returns:
        (status, reasons): OFF when config.mode is OFF; MISSING without a report; MALFORMED when generated_at
        is not an aware ISO-8601 timestamp; STALE when older than
        max_report_age_days at now; MISMATCH naming each differing identity field (strategy_id, top_k,
        select_universe, feature contract version, model_params, seeds); INSUFFICIENT when the report status
        is not OK; FAIL naming each violated threshold; PASS otherwise.
    """
    if config.mode == PitGateMode.OFF:
        return "OFF", ()
    if now.tzinfo is None:
        raise ValueError(f"pit gate now must be timezone-aware, got {now!r}")
    if report is None:
        return "MISSING", ("no pit_haircut report available for the candidate configuration",)
    # A malformed report is a status, not an exception: ENFORCE blocks on any non-PASS status (fail-closed)
    # while ADVISORY must never stop the weekly retrain.
    try:
        generated = datetime.fromisoformat(str(report.generated_at))
    except ValueError:
        return "MALFORMED", (f"pit_haircut report generated_at {report.generated_at!r} is not ISO-8601",)
    if generated.tzinfo is None:
        return "MALFORMED", (f"pit_haircut report generated_at {report.generated_at!r} is not timezone-aware",)
    age_days = (now.date() - generated.date()).days
    if age_days > int(config.max_report_age_days):
        return "STALE", (
            f"pit_haircut report is {age_days} calendar days old (max {int(config.max_report_age_days)})",
        )
    mismatch: list[str] = []
    if candidate.get("strategy_id") != report.strategy_id:
        mismatch.append("strategy_id")
    if candidate.get("top_k") != report.top_k:
        mismatch.append("top_k")
    if "select_universe" not in candidate:
        mismatch.append("select_universe (missing from candidate)")
    elif dict(candidate["select_universe"]) != dict(report.select_universe):
        mismatch.append("select_universe")
    try:
        candidate_contract = bundle_feature_contract_version(candidate)
    except ValueError:
        candidate_contract = None
    if candidate_contract != report.feature_contract_version:
        mismatch.append("feature contract version")
    if "model_params" not in candidate:
        mismatch.append("model_params (missing from candidate)")
    elif dict(candidate["model_params"]) != dict(report.model_params):
        mismatch.append("model_params")
    candidate_seeds = _as_sequence(candidate.get("seeds"))
    report_seeds = _as_sequence(report.seeds)
    if candidate_seeds is None:
        mismatch.append("seeds (missing from candidate)")
    elif report_seeds is None or candidate_seeds != report_seeds:
        mismatch.append("seeds")
    if mismatch:
        return "MISMATCH", tuple(
            f"identity field {name} does not match the pit_haircut report" for name in mismatch
        )
    if str(report.status) != "OK":
        return "INSUFFICIENT", (f"pit_haircut report status is {report.status}, no usable paired evidence",)
    failures: list[str] = []
    native_mean = float(report.mean_net_bp.get("pit_native", float("nan")))
    native_ic = float(report.rank_ic_mean.get("pit_native", float("nan")))
    if not np.isfinite(native_mean) or native_mean < float(config.min_pit_native_mean_net_bp):
        failures.append(
            f"pit_native mean_net_bp {native_mean:.2f} below floor {float(config.min_pit_native_mean_net_bp):.2f}"
        )
    if not np.isfinite(native_ic) or native_ic < float(config.min_pit_rank_ic):
        failures.append(
            f"pit_native rank_ic {native_ic:.4f} below floor {float(config.min_pit_rank_ic):.4f}"
        )
    if config.max_haircut_bp is not None:
        haircut = float(report.haircut.delta)
        if not np.isfinite(haircut) or haircut > float(config.max_haircut_bp):
            failures.append(
                f"pit haircut {haircut:.2f}bp above ceiling {float(config.max_haircut_bp):.2f}bp"
            )
    if failures:
        return "FAIL", tuple(failures)
    return "PASS", ()


def evaluate_retrain_promotion(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
    eval_frame: pd.DataFrame,
    *,
    min_agreement: float = RETRAIN_MIN_PREDICTION_AGREEMENT,
    date_col: str = "date",
    pit_report: PitHaircutReport | None = None,
    pit_gate: PitGateConfig = PitGateConfig(),  # noqa: B008
    now: datetime | None = None,
) -> PromotionVerdict:
    """Decide whether a retrained candidate may replace the live bundle.

    Rank agreement is only meaningful between bundles certified for the same strategy and
    screen; a strategy_id or certified-screen change is a certification event, so the gate
    refuses it and names the manual path (certify, then rerun with --skip-promotion-gate).

    Args:
        candidate: Freshly trained bundle.
        current: Live bundle, or None on first publication (bootstrap).
        eval_frame: Recent selection-pool rows from build_gate_eval_frame.
        min_agreement: Minimum mean daily rank agreement with the live bundle.
        date_col: Date column in eval_frame.
        pit_report: Latest pit_haircut report, or None.
        pit_gate: PIT criterion configuration; the default (OFF) reproduces the legacy verdict exactly.
        now: Aware clock for report staleness; None uses Asia/Seoul now.

    Returns:
        PromotionVerdict with the blocking reasons, if any.
    """
    eff_now = now if now is not None else datetime.now(_KST)
    pit_status, pit_reasons = evaluate_pit_gate(pit_report, candidate, config=pit_gate, now=eff_now)

    def _verdict(promote: bool, reasons: tuple[str, ...], agreement: float | None) -> PromotionVerdict:
        if pit_gate.mode == PitGateMode.ENFORCE and pit_status != "PASS":
            gated = [f"pit gate {pit_status}: {reason}" for reason in pit_reasons] or [
                f"pit gate {pit_status}"
            ]
            return PromotionVerdict(
                promote=False,
                reasons=tuple(reasons) + tuple(gated),
                agreement=agreement,
                pit_status=pit_status,
                pit_reasons=pit_reasons,
            )
        return PromotionVerdict(
            promote=promote, reasons=reasons, agreement=agreement, pit_status=pit_status, pit_reasons=pit_reasons
        )

    reasons: list[str] = []
    try:
        assert_bundle_screen_parity(candidate)
    except ValueError as exc:
        reasons.append(f"screen parity: {exc}")
    feature_cols = list(candidate.get("feature_cols", []))
    if not feature_cols:
        reasons.append("candidate feature_cols is empty")
    missing = [col for col in feature_cols if col not in eval_frame.columns]
    if missing:
        reasons.append(f"gate eval frame is missing features {missing}")
    if eval_frame.empty:
        reasons.append("gate eval frame is empty")
    candidate_issue = feature_contract_issue(candidate)
    if candidate_issue is not None:
        reasons.append(f"candidate feature contract: {candidate_issue}")
    if reasons:
        return _verdict(False, tuple(reasons), None)

    if current is not None:
        if current.get("strategy_id") != candidate.get("strategy_id"):
            reasons.append(
                f"strategy or screen changed vs live bundle ({current.get('strategy_id')} -> {candidate.get('strategy_id')}); "
                "certify manually and rerun with --skip-promotion-gate"
            )
            return _verdict(False, tuple(reasons), None)
        try:
            current_screen = certified_screen(current)
            candidate_screen = certified_screen(candidate)
        except ValueError as exc:
            reasons.append(
                f"strategy or screen changed vs live bundle ({current.get('strategy_id')} -> {candidate.get('strategy_id')}); "
                f"certify manually and rerun with --skip-promotion-gate: {exc}"
            )
            return _verdict(False, tuple(reasons), None)
        if current_screen != candidate_screen:
            reasons.append(
                f"strategy or screen changed vs live bundle ({current.get('strategy_id')} -> {candidate.get('strategy_id')}); "
                "certify manually and rerun with --skip-promotion-gate"
            )
            return _verdict(False, tuple(reasons), None)

    features = eval_frame[feature_cols]
    dates = eval_frame[date_col].to_numpy()
    candidate_pred = np.asarray(candidate["return_model"].predict(features), dtype=np.float64)
    if not np.all(np.isfinite(candidate_pred)):
        reasons.append("candidate predictions contain non-finite values")
    else:
        spread = pd.DataFrame({"date": dates, "pred": candidate_pred}).groupby("date")["pred"].agg(["size", "std"])
        eligible = spread[spread["size"] >= MIN_NAMES_PER_EVAL_DAY]
        if bool((eligible["std"].fillna(0.0) <= 0.0).any()):
            reasons.append("candidate predictions are constant within an eval day")
    if current is None:
        return _verdict(not reasons, tuple(reasons), None)
    if list(current.get("feature_cols", [])) != feature_cols or _resolved_contract(current) != _resolved_contract(candidate):
        reasons.append("feature contract changed vs live bundle; certify manually and rerun with --skip-promotion-gate")
        return _verdict(False, tuple(reasons), None)
    current_pred = np.asarray(current["return_model"].predict(features), dtype=np.float64)
    agreement = _mean_daily_rank_agreement(dates, candidate_pred, current_pred)
    if agreement is None:
        reasons.append(f"no eval day has at least {MIN_NAMES_PER_EVAL_DAY} names to measure agreement")
    elif agreement < float(min_agreement):
        reasons.append(f"prediction agreement {agreement:.3f} below {float(min_agreement):.3f}")
    return _verdict(not reasons, tuple(reasons), agreement)
