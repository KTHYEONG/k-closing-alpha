"""Weekly retrain promotion gate.

A retrained ranker bundle replaces the live one only when it is structurally
certified for the live screen and ranks the most recent selection pool
consistently with the bundle it replaces. This guards the unattended weekly
retrain against silently promoting a model fit on corrupted or truncated data;
it does not re-certify alpha (that remains the manual CPCV certification).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.ml.topk_ranker_research import assert_bundle_screen_parity, build_dual_pool
from src.strategy.contract import DEFAULT_UNIVERSE, KCA_TOPK_COSTAWARE_001

RETRAIN_GATE_EVAL_DAYS: int = 20
# 실측(2026-09-13, 최근 20거래일 선택풀): 주간 연속 재학습 간 일별 순위상관 평균 0.983~0.989,
# 최근 1년 시가를 ±10% 오염시켜 학습하면 0.808 -> 정상 변동과 데이터 오염을 가르는 하한
RETRAIN_MIN_PREDICTION_AGREEMENT: float = 0.95
# 순위상관은 표본 4개 미만인 날에 의미가 없어 집계에서 제외한다
MIN_NAMES_PER_EVAL_DAY: int = 4
BUNDLE_FILENAME: str = "sizing_pipeline_bundle.joblib"


@dataclass(frozen=True)
class PromotionVerdict:
    """Promotion decision for a retrained candidate bundle.

    Attributes:
        promote: True only when no reason blocks promotion.
        reasons: Human-readable blocking reasons (empty when promote is True).
        agreement: Mean daily rank agreement with the live bundle; None when not measured.
    """

    promote: bool
    reasons: tuple[str, ...]
    agreement: float | None


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
        ph, market_dates, d_to_idx, train_spec=DEFAULT_UNIVERSE, select_spec=KCA_TOPK_COSTAWARE_001.universe
    )
    selected = pool.loc[np.asarray(sel_mask, dtype=bool)]
    dates = pd.to_datetime(selected["date"])
    recent = sorted(dates.unique())[-int(eval_days):]
    return selected[dates.isin(recent)].reset_index(drop=True)


def _mean_daily_rank_agreement(dates: np.ndarray, first: np.ndarray, second: np.ndarray) -> float | None:
    frame = pd.DataFrame({"date": dates, "first": first, "second": second})
    per_day = [
        group["first"].rank().corr(group["second"].rank())
        for _, group in frame.groupby("date", sort=True)
        if len(group) >= MIN_NAMES_PER_EVAL_DAY
    ]
    finite = [value for value in per_day if np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def evaluate_retrain_promotion(
    candidate: dict[str, Any],
    current: dict[str, Any] | None,
    eval_frame: pd.DataFrame,
    *,
    min_agreement: float = RETRAIN_MIN_PREDICTION_AGREEMENT,
    date_col: str = "date",
) -> PromotionVerdict:
    """Decide whether a retrained candidate may replace the live bundle.

    Args:
        candidate: Freshly trained bundle.
        current: Live bundle, or None on first publication (bootstrap).
        eval_frame: Recent selection-pool rows from build_gate_eval_frame.
        min_agreement: Minimum mean daily rank agreement with the live bundle.
        date_col: Date column in eval_frame.

    Returns:
        PromotionVerdict with the blocking reasons, if any.
    """
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
    if reasons:
        return PromotionVerdict(promote=False, reasons=tuple(reasons), agreement=None)

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
        return PromotionVerdict(promote=not reasons, reasons=tuple(reasons), agreement=None)
    if list(current.get("feature_cols", [])) != feature_cols:
        reasons.append("feature contract changed vs live bundle; certify manually and rerun with --skip-promotion-gate")
        return PromotionVerdict(promote=False, reasons=tuple(reasons), agreement=None)
    current_pred = np.asarray(current["return_model"].predict(features), dtype=np.float64)
    agreement = _mean_daily_rank_agreement(dates, candidate_pred, current_pred)
    if agreement is None:
        reasons.append(f"no eval day has at least {MIN_NAMES_PER_EVAL_DAY} names to measure agreement")
    elif agreement < float(min_agreement):
        reasons.append(f"prediction agreement {agreement:.3f} below {float(min_agreement):.3f}")
    return PromotionVerdict(promote=not reasons, reasons=tuple(reasons), agreement=agreement)
