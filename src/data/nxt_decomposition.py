"""Causal consolidated-to-KRX decomposition estimator (decision cutoff 15:20).

Reconstructs KRX-only bars from the consolidated tape using only information
observable at the decision time plus strictly earlier history. A day-T estimate
scales consolidated bars at or before the cutoff by a KRX share predicted from
earlier days; day-T EOD volume and post-cutoff bars never participate.

The estimator is fitted on the later dates of the KIS calibration window and
measured on an isolated holdout of the earliest calibration dates, reproducing
the direction of use (reconstruction applies to a period before the window).
Scoring always runs over proxy history ``EOD x (1 - a_bar) / V_cons(<=15:20)``,
never over true KRX shares.
"""

from __future__ import annotations

import json
import logging
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.market_session import DECISION_WINDOW_START_HHMMSS

logger = logging.getLogger(__name__)

CUTOFF_HHMMSS: int = int(DECISION_WINDOW_START_HHMMSS)

BASIS_KRX_RECONSTRUCTED: str = "krx_reconstructed"

_BASIS_RECONSTRUCTED: str = BASIS_KRX_RECONSTRUCTED

DECOMPOSITION_CONFIG_FILENAME: str = "nxt_decomposition_config.json"
DECOMPOSITION_FIT_REPORT_FILENAME: str = "nxt_decomposition_fit_report.json"

_REQUIRED_CALIBRATION_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "v_krx_1520",
    "auction_volume",
    "v_cons_1520",
    "eod_volume",
    "close_krx_1519",
    "close_cons_1519",
)

_REQUIRED_BAR_COLUMNS: tuple[str, ...] = ("ts_hms", "volume", "high", "low", "close")

_ELIGIBILITY_VOLUME_COLUMNS: tuple[str, ...] = (
    "v_krx_1520",
    "auction_volume",
    "v_cons_1520",
    "eod_volume",
    "close_krx_1519",
    "close_cons_1519",
)


@dataclass(frozen=True)
class DecompositionConfig:
    """Structural parameters of the consolidated-to-KRX measurement basis.

    `fit_start`/`calibrated_through` bound the fit window; `holdout_start`/`holdout_end`
    bound the isolated earliest-date holdout the error quantiles were measured on.
    """

    ewma_alpha: float
    min_prior_days: int
    max_gap_days: int
    auction_fraction_mean: float
    bias_correction: float
    volume_rel_err_p90: float
    close_bp_err_p90: float
    calibrated_through: str
    fit_start: str
    holdout_start: str
    holdout_end: str


@dataclass(frozen=True)
class FrontierPoint:
    """One (alpha, min_prior_days, max_gap_days) candidate scored on the selection rows.

    coverage is predicted rows / selection rows; errors are |pred x bias - true| / true over predicted rows, where bias is
    this candidate's own median true/raw ratio. Error quantiles are None when coverage is zero.
    """

    ewma_alpha: float
    min_prior_days: int
    max_gap_days: int
    coverage: float
    rel_err_p50: float | None
    rel_err_p90: float | None
    rel_err_p99: float | None


@dataclass(frozen=True)
class FitDiagnostics:
    """Evidence behind a fitted config, persisted beside it.

    holdout_* are measured on the isolated holdout rows with the chosen parameters; abar_* and bias_* compare the
    fit-window estimate with the same statistic recomputed on the holdout (transfer to the earlier gap period);
    alpha_at_boundary flags a chosen alpha equal to the smallest or largest grid value; frontier lists every scored
    candidate; table_sha256 identifies the calibration table the fit consumed.
    """

    fit_start: str
    fit_end: str
    holdout_start: str
    holdout_end: str
    n_fit_rows: int
    n_holdout_rows: int
    n_symbols: int
    holdout_rel_err_p50: float
    holdout_rel_err_p90: float
    holdout_rel_err_p99: float
    holdout_coverage: float
    abar_fit: float
    abar_holdout: float
    bias_fit: float
    bias_holdout: float
    alpha_at_boundary: bool
    frontier: tuple[FrontierPoint, ...]
    table_sha256: str


@dataclass(frozen=True)
class DecompositionFit:
    """A fitted config together with the diagnostics evidencing it."""

    config: DecompositionConfig
    diagnostics: FitDiagnostics


@dataclass(frozen=True)
class _HistoryRow:
    """One eligible symbol-day inside a proxy history, ordered by calendar day."""

    ordinal: int
    proxy: float
    true: float


@dataclass(frozen=True)
class _ScoredCandidate:
    """A ranked candidate: selection-set scores plus its own bias for downstream use."""

    ewma_alpha: float
    min_prior_days: int
    max_gap_days: int
    coverage: float
    rel_err_p50: float
    rel_err_p90: float
    rel_err_p99: float
    bias: float


def _ewma(values: Sequence[float], alpha: float) -> float:
    ema = float(values[0])
    for value in values[1:]:
        ema = float(alpha) * float(value) + (1.0 - float(alpha)) * ema
    return ema


def _parse_day(value: object) -> pd.Timestamp | None:
    try:
        parsed = pd.Timestamp(str(value))
    except (ValueError, TypeError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.normalize()


def _as_float(name: str, value: Any, *, caller: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{caller} carries a mistyped {name}: {value!r}")
    return float(value)


def _as_int(name: str, value: Any, *, caller: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{caller} carries a mistyped {name}: {value!r}")
    return int(value)


def _checked_day(name: str, value: object, *, caller: str) -> str:
    text = str(value) if isinstance(value, str) else ""
    if not text.strip() or _parse_day(text) is None:
        raise ValueError(f"{caller} carries an invalid {name}: {value!r}")
    return text


def _eligible_frame(calibration: pd.DataFrame, identity_tolerance: float, *, caller: str) -> pd.DataFrame:
    """Eligible calibration rows: finite positive volumes, tolerable identity residual.

    Ineligible rows are dropped outright and never enter `a_bar`, any proxy history,
    any score or any reported count; no NaN is converted to zero.
    """
    missing = [c for c in _REQUIRED_CALIBRATION_COLUMNS if c not in calibration.columns]
    if missing:
        raise ValueError(f"{caller} missing required columns: {missing}")
    if calibration.empty:
        raise ValueError(f"{caller} carries no calibration rows")
    work = calibration.copy()
    stamps = pd.to_datetime(work["date"], errors="coerce", format="mixed")
    if bool(stamps.isna().any()):
        raise ValueError(f"{caller} carries unparseable dates")
    work["day"] = stamps.dt.strftime("%Y-%m-%d")
    epoch = pd.Timestamp("1970-01-01")
    work["ordinal"] = ((stamps.dt.normalize() - epoch) // pd.Timedelta(days=1)).astype(np.int64)
    for column in _ELIGIBILITY_VOLUME_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce").astype(np.float64)
    work["symbol"] = work["symbol"].astype(str)
    eod = work["eod_volume"].to_numpy(dtype=np.float64)
    v_krx = work["v_krx_1520"].to_numpy(dtype=np.float64)
    v_cons = work["v_cons_1520"].to_numpy(dtype=np.float64)
    auction = work["auction_volume"].to_numpy(dtype=np.float64)
    safe_eod = np.where(eod > 0.0, eod, 1.0)
    residual = np.where(eod > 0.0, np.abs(eod - (v_krx + auction)) / safe_eod, np.inf)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        true_share = v_krx / v_cons
    valid = (
        np.isfinite(eod)
        & (eod > 0.0)
        & np.isfinite(v_krx)
        & (v_krx > 0.0)
        & np.isfinite(v_cons)
        & (v_cons > 0.0)
        & np.isfinite(auction)
        & (auction >= 0.0)
        & np.isfinite(residual)
        & (residual <= float(identity_tolerance))
        & np.isfinite(true_share)
        & (true_share > 0.0)
    )
    kept = work.loc[valid].copy().reset_index(drop=True)
    kept["true_share"] = np.asarray(true_share)[valid]
    return kept


def _build_history(frame: pd.DataFrame, auction_fraction_mean: float) -> dict[str, list[_HistoryRow]]:
    """Per-symbol proxy histories over eligible rows, oldest first.

    The proxy is the production quantity `EOD x (1 - a_bar) / V_cons(<=15:20)`; every
    input row is eligible so each proxy is finite, but prediction still skips
    non-finite entries exactly like the production predictor.
    """
    scale = 1.0 - float(auction_fraction_mean)
    ordered = frame.sort_values(["symbol", "ordinal"], kind="stable")
    history: dict[str, list[_HistoryRow]] = {}
    for row in ordered.to_dict(orient="records"):
        symbol = str(row["symbol"])
        proxy = float(row["eod_volume"]) * scale / float(row["v_cons_1520"])
        history.setdefault(symbol, []).append(
            _HistoryRow(ordinal=int(row["ordinal"]), proxy=proxy, true=float(row["true_share"]))
        )
    return history


def _predict_raw(
    priors: list[_HistoryRow],
    decision_ordinal: int,
    *,
    min_prior_days: int,
    max_gap_days: int,
    alpha: float,
) -> float | None:
    """Unbiased EWMA over strictly earlier proxies, mirroring `predict_share` history use."""
    values = [
        row.proxy
        for row in priors
        if row.ordinal < decision_ordinal
        and decision_ordinal - row.ordinal <= int(max_gap_days)
        and np.isfinite(row.proxy)
    ]
    if len(values) < int(min_prior_days):
        return None
    return _prediction_value(values, alpha=alpha, bias=1.0, enforce_range=False)


def _prediction_value(
    values: Sequence[float], *, alpha: float, bias: float, enforce_range: bool = True
) -> float | None:
    """Shared EWMA and validity gate for production and calibration predictions."""
    raw = _ewma(values, alpha)
    corrected = raw * bias
    if not np.isfinite(corrected) or corrected <= 0.0 or (enforce_range and corrected > 1.0):
        return None
    return corrected


def _candidate_sort_key(candidate: _ScoredCandidate) -> tuple[float, float, int, int, float]:
    return (
        -candidate.coverage,
        candidate.rel_err_p90,
        candidate.min_prior_days,
        -candidate.max_gap_days,
        candidate.ewma_alpha,
    )


def _score_candidates(
    history: Mapping[str, Sequence[_HistoryRow]],
    scope: Sequence[tuple[str, int, float]],
    *,
    alphas: Sequence[float],
    structures: Sequence[tuple[int, int]],
) -> list[_ScoredCandidate]:
    """Score every (alpha, structure) candidate on scope rows from strictly earlier history.

    Returns every candidate sorted by the choice order (coverage, then p90, then smaller
    min_prior_days, then larger max_gap_days, then smaller alpha). A candidate's bias is
    its own median true/raw ratio; coverage counts predictions whose bias-corrected share
    lands in (0, 1], exactly the production predictor's range gate.
    """
    scored: list[_ScoredCandidate] = []
    for alpha in [float(a) for a in alphas]:
        for min_prior_days, max_gap_days in structures:
            raws: list[tuple[float, float]] = []
            for symbol, decision_ordinal, true in scope:
                raw = _predict_raw(
                    list(history.get(symbol, [])),
                    int(decision_ordinal),
                    min_prior_days=int(min_prior_days),
                    max_gap_days=int(max_gap_days),
                    alpha=float(alpha),
                )
                if raw is not None and raw > 0.0 and np.isfinite(true) and true > 0.0:
                    raws.append((raw, float(true)))
            ratios = [true / raw for raw, true in raws]
            bias = float(np.median(ratios)) if ratios else float("nan")
            predicted: list[float] = []
            if ratios and np.isfinite(bias) and bias > 0.0:
                for raw, true in raws:
                    corrected = raw * bias
                    if np.isfinite(corrected) and 0.0 < corrected <= 1.0:
                        predicted.append(abs(corrected - true) / true)
            coverage = float(len(predicted) / len(scope)) if scope else 0.0
            if predicted:
                errors = np.asarray(predicted, dtype=np.float64)
                p50, p90, p99 = (float(np.quantile(errors, q)) for q in (0.5, 0.9, 0.99))
            else:
                p50 = p90 = p99 = float("inf")
            scored.append(
                _ScoredCandidate(
                    ewma_alpha=float(alpha),
                    min_prior_days=int(min_prior_days),
                    max_gap_days=int(max_gap_days),
                    coverage=coverage,
                    rel_err_p50=p50,
                    rel_err_p90=p90,
                    rel_err_p99=p99,
                    bias=bias,
                )
            )
    scored.sort(key=_candidate_sort_key)
    return scored


def fit_decomposition(
    calibration: pd.DataFrame,
    *,
    alphas: Sequence[float],
    structures: Sequence[tuple[int, int]],
    identity_tolerance: float,
    holdout_fraction: float,
    min_holdout_days: int,
    max_selection_rel_err_p90: float,
    table_sha256: str,
) -> DecompositionFit:
    """Fit the consolidated-to-KRX share estimator and measure it out of sample in the direction it will be used.

    The estimator is the production predictor: bias-corrected EWMA of strictly earlier per-symbol proxies
    EOD x (1 - a_bar) / V_cons(<=15:20). The parameters describe the vendors' measurement basis, not alpha. The EARLIEST
    holdout_fraction of distinct calibration dates is isolated; parameters are fitted on the remaining (later) dates
    because the reconstruction is applied to a period before the calibration window.

    Args:
        calibration: Calibration table (`CALIBRATION_TABLE_COLUMNS`).
        alphas: Candidate EWMA coefficients in (0, 1]; at least three distinct values.
        structures: Candidate (min_prior_days, max_gap_days) pairs.
        identity_tolerance: Maximum |EOD - (V_krx + A)| / EOD kept.
        holdout_fraction: Fraction of distinct dates held out, in (0, 0.5].
        min_holdout_days: Minimum distinct holdout dates.
        max_selection_rel_err_p90: Ceiling on a candidate's selection-set p90 relative error.
        table_sha256: Digest of the calibration table, recorded verbatim.

    Returns:
        The fitted config (fit-window parameters, holdout-measured error fields) and its diagnostics.

    Raises:
        ValueError: Invalid grids or fractions, missing columns, fewer holdout dates than min_holdout_days, too few
            fit rows or symbols after identity filtering, or no candidate within max_selection_rel_err_p90.
    """
    caller = "fit_decomposition"
    candidates = [float(a) for a in alphas]
    if len(set(candidates)) < 3 or any(not np.isfinite(a) or a <= 0.0 or a > 1.0 for a in candidates):
        raise ValueError(f"{caller} alphas must hold at least three distinct values within (0, 1]")
    try:
        pairs = [(operator.index(m), operator.index(g)) for m, g in [tuple(s) for s in structures]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{caller} structures must hold (min_prior_days, max_gap_days) pairs") from exc
    if not pairs:
        raise ValueError(f"{caller} structures must hold (min_prior_days, max_gap_days) pairs")
    for min_prior_days, max_gap_days in pairs:
        if int(min_prior_days) < 1:
            raise ValueError(f"{caller} min_prior_days must be >= 1")
        if int(max_gap_days) < 0:
            raise ValueError(f"{caller} max_gap_days must be >= 0")
    structures_deduped = sorted(set(pairs))
    if not np.isfinite(float(identity_tolerance)) or float(identity_tolerance) <= 0.0:
        raise ValueError(f"{caller} identity_tolerance must be positive")
    fraction = float(holdout_fraction)
    if not np.isfinite(fraction) or fraction <= 0.0 or fraction > 0.5:
        raise ValueError(f"{caller} holdout_fraction must lie within (0, 0.5]")
    min_holdout = _as_int("min_holdout_days", min_holdout_days, caller=caller)
    if min_holdout < 1:
        raise ValueError(f"{caller} min_holdout_days must be >= 1")
    ceiling = float(max_selection_rel_err_p90)
    if not np.isfinite(ceiling) or ceiling < 0.0:
        raise ValueError(f"{caller} max_selection_rel_err_p90 must be a finite ceiling >= 0")
    if not isinstance(table_sha256, str):
        raise ValueError(f"{caller} table_sha256 must be a string")

    kept = _eligible_frame(calibration, float(identity_tolerance), caller=caller)
    dates = sorted(kept["day"].unique().tolist())
    if not dates:
        raise ValueError(f"{caller} keeps 0 symbol-days over 0 symbols after identity filtering")
    n_holdout_dates = math.ceil(fraction * len(dates))
    if n_holdout_dates < min_holdout:
        raise ValueError(
            f"{caller} holds out {n_holdout_dates} date(s), fewer than min_holdout_days={min_holdout}"
        )
    holdout_dates = dates[:n_holdout_dates]
    fit_dates = dates[n_holdout_dates:]
    holdout_set = set(holdout_dates)
    fit = kept.loc[~kept["day"].isin(holdout_set)].reset_index(drop=True)
    holdout = kept.loc[kept["day"].isin(holdout_set)].reset_index(drop=True)
    if len(fit_dates) < 2 or len(fit) < 2 or int(fit["symbol"].nunique()) < 2:
        raise ValueError(
            f"{caller} keeps {len(fit)} fit symbol-days "
            f"over {int(fit['symbol'].nunique()) if len(fit) else 0} symbols after identity filtering"
        )

    abar_fit = float(np.mean((fit["auction_volume"] / fit["eod_volume"]).to_numpy(dtype=np.float64)))
    fit_history = _build_history(fit, abar_fit)
    selection_days = set(fit_dates[max(1, len(fit_dates) // 2):])
    selection_scope = [
        (str(symbol), int(ordinal), float(true))
        for symbol, ordinal, day, true in zip(
            fit["symbol"].tolist(),
            fit["ordinal"].tolist(),
            fit["day"].tolist(),
            fit["true_share"].tolist(),
            strict=True,
        )
        if day in selection_days
    ]
    scored = _score_candidates(
        fit_history, selection_scope, alphas=candidates, structures=structures_deduped
    )
    frontier = tuple(
        FrontierPoint(
            ewma_alpha=c.ewma_alpha,
            min_prior_days=c.min_prior_days,
            max_gap_days=c.max_gap_days,
            coverage=c.coverage,
            rel_err_p50=c.rel_err_p50 if c.coverage else None,
            rel_err_p90=c.rel_err_p90 if c.coverage else None,
            rel_err_p99=c.rel_err_p99 if c.coverage else None,
        )
        for c in scored
    )
    qualifying = [c for c in scored if c.rel_err_p90 <= ceiling]
    if not qualifying:
        raise ValueError(
            f"{caller} finds no candidate within max_selection_rel_err_p90={ceiling} ({len(scored)} scored)"
        )
    best = qualifying[0]

    holdout_history = _build_history(holdout, abar_fit)
    holdout_scope = [
        (str(symbol), int(ordinal), float(true))
        for symbol, ordinal, true in zip(
            holdout["symbol"].tolist(), holdout["ordinal"].tolist(), holdout["true_share"].tolist(), strict=True
        )
    ]
    holdout_errors: list[float] = []
    holdout_ratios: list[float] = []
    for symbol, decision_ordinal, true in holdout_scope:
        raw = _predict_raw(
            holdout_history.get(symbol, []),
            int(decision_ordinal),
            min_prior_days=int(best.min_prior_days),
            max_gap_days=int(best.max_gap_days),
            alpha=float(best.ewma_alpha),
        )
        if raw is None or raw <= 0.0:
            continue
        corrected = raw * float(best.bias)
        if not np.isfinite(corrected) or corrected <= 0.0 or corrected > 1.0:
            continue
        holdout_errors.append(abs(corrected - true) / true)
        holdout_ratios.append(true / raw)
    if not holdout_errors:
        raise ValueError(f"{caller} cannot score any holdout row with the chosen parameters")
    error_array = np.asarray(holdout_errors, dtype=np.float64)
    holdout_p50 = float(np.quantile(error_array, 0.5))
    holdout_p90 = float(np.quantile(error_array, 0.9))
    holdout_p99 = float(np.quantile(error_array, 0.99))
    holdout_coverage = float(len(holdout_errors) / len(holdout_scope)) if holdout_scope else 0.0
    abar_holdout = float(np.mean((holdout["auction_volume"] / holdout["eod_volume"]).to_numpy(dtype=np.float64)))
    bias_holdout = float(np.median(holdout_ratios))
    if not all(np.isfinite(value) for value in (
        holdout_p50, holdout_p90, holdout_p99, abar_fit, abar_holdout, best.bias, bias_holdout,
    )) or bias_holdout <= 0.0:
        raise ValueError(f"{caller} cannot compute finite holdout diagnostics")

    close_krx = holdout["close_krx_1519"].to_numpy(dtype=np.float64)
    close_cons = holdout["close_cons_1519"].to_numpy(dtype=np.float64)
    close_ok = np.isfinite(close_krx) & np.isfinite(close_cons) & (close_krx > 0.0) & (close_cons > 0.0)
    if not bool(close_ok.any()):
        raise ValueError(f"{caller} finds no holdout close pair with finite positive closes")
    close_bp_err_p90 = float(np.quantile(np.abs(close_cons[close_ok] - close_krx[close_ok]) / close_krx[close_ok] * 1e4, 0.9))
    if not np.isfinite(close_bp_err_p90):
        raise ValueError(f"{caller} cannot compute finite holdout close diagnostics")

    distinct_alphas = sorted(set(candidates))
    alpha_at_boundary = float(best.ewma_alpha) == distinct_alphas[0] or float(best.ewma_alpha) == distinct_alphas[-1]
    fit_start = str(min(fit_dates))
    fit_end = str(max(fit_dates))
    config = DecompositionConfig(
        ewma_alpha=float(best.ewma_alpha),
        min_prior_days=int(best.min_prior_days),
        max_gap_days=int(best.max_gap_days),
        auction_fraction_mean=float(abar_fit),
        bias_correction=float(best.bias),
        volume_rel_err_p90=float(holdout_p90),
        close_bp_err_p90=float(close_bp_err_p90),
        calibrated_through=fit_end,
        fit_start=fit_start,
        holdout_start=str(min(holdout_dates)),
        holdout_end=str(max(holdout_dates)),
    )
    diagnostics = FitDiagnostics(
        fit_start=fit_start,
        fit_end=fit_end,
        holdout_start=str(min(holdout_dates)),
        holdout_end=str(max(holdout_dates)),
        n_fit_rows=len(fit),
        n_holdout_rows=len(holdout),
        n_symbols=int(kept["symbol"].nunique()),
        holdout_rel_err_p50=float(holdout_p50),
        holdout_rel_err_p90=float(holdout_p90),
        holdout_rel_err_p99=float(holdout_p99),
        holdout_coverage=float(holdout_coverage),
        abar_fit=float(abar_fit),
        abar_holdout=float(abar_holdout),
        bias_fit=float(best.bias),
        bias_holdout=float(bias_holdout),
        alpha_at_boundary=bool(alpha_at_boundary),
        frontier=frontier,
        table_sha256=str(table_sha256),
    )
    logger.info(
        "[DATA] stage=nxt_decomposition status=FITTED alpha=%.3f structure=(%d,%d) "
        "fit_rows=%d holdout_rows=%d symbols=%d holdout_p90=%.4f coverage=%.4f",
        float(best.ewma_alpha),
        int(best.min_prior_days),
        int(best.max_gap_days),
        len(fit),
        len(holdout),
        int(kept["symbol"].nunique()),
        float(holdout_p90),
        float(holdout_coverage),
    )
    return DecompositionFit(config=config, diagnostics=diagnostics)


def _cons_volume_at_or_below(bars: pd.DataFrame) -> float:
    if bars is None or bars.empty or not {"ts_hms", "volume"} <= set(bars.columns):
        return float("nan")
    stamps = pd.to_numeric(bars["ts_hms"], errors="coerce").to_numpy(dtype=np.float64)
    volumes = pd.to_numeric(bars["volume"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(stamps)):
        return float("nan")
    observable = volumes[stamps <= float(CUTOFF_HHMMSS)]
    if not np.all(np.isfinite(observable)):
        return float("nan")
    return float(np.sum(observable))


def share_proxy_history(
    cons_bars_by_day: Mapping[str, pd.DataFrame],
    eod_volume_by_day: Mapping[str, float],
    *,
    config: DecompositionConfig,
) -> pd.Series:
    """Per earlier day, the KRX share proxy `EOD_d x (1 - a_bar) / V_cons_d(<=15:20)`. Days with missing or non-positive inputs yield NaN (never a default)."""
    days = sorted({str(d) for d in list(cons_bars_by_day.keys()) + list(eod_volume_by_day.keys())})
    auction_mean = float(config.auction_fraction_mean)
    scale = (1.0 - auction_mean) if np.isfinite(auction_mean) and 0.0 <= auction_mean < 1.0 else float("nan")
    values: list[float] = []
    for day in days:
        bars = cons_bars_by_day.get(day)
        try:
            eod = float(eod_volume_by_day.get(day, float("nan")))
        except (TypeError, ValueError):
            eod = float("nan")
        v_cons = _cons_volume_at_or_below(bars) if bars is not None else float("nan")
        if not np.isfinite(scale) or not np.isfinite(eod) or eod <= 0.0 or not np.isfinite(v_cons) or v_cons <= 0.0:
            values.append(float("nan"))
        else:
            values.append(float(eod * scale / v_cons))
    return pd.Series(values, index=pd.Index(days, dtype=object), dtype=np.float64, name="share_proxy")


def predict_share(
    proxy_history: pd.Series, decision_date: str, *, config: DecompositionConfig
) -> float | None:
    """The EWMA of strictly earlier proxy values, bias-corrected. Only entries dated before `decision_date` participate. Returns None (fail closed) when fewer than `min_prior_days` entries lie within `max_gap_days` calendar days before `decision_date`."""
    decision = _parse_day(decision_date)
    if decision is None:
        return None
    alpha = float(config.ewma_alpha)
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        return None
    dated: list[tuple[pd.Timestamp, float]] = []
    for label, value in proxy_history.items():
        day = _parse_day(label)
        if day is None or day >= decision:
            continue
        gap = int((decision - day).days)
        if gap < 1 or gap > int(config.max_gap_days):
            continue
        try:
            share = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(share):
            dated.append((day, share))
    if len(dated) < int(config.min_prior_days):
        return None
    dated.sort(key=lambda item: item[0])
    return _prediction_value([share for _, share in dated], alpha=alpha, bias=float(config.bias_correction))


def reconstruct_krx_bars(
    cons_bars: pd.DataFrame, *, share: float, config: DecompositionConfig
) -> pd.DataFrame:
    """Scale each consolidated bar volume by `share`, rebuild bar value from the scaled volume and the typical price, keep prices as the consolidated prices, and stamp the frame with `basis="krx_reconstructed"`, `recon_volume_rel_err_p90` and `recon_close_bp_err_p90` copied from the config. Bars after the 15:20 cutoff are not accepted."""
    share_value = float(share)
    if not np.isfinite(share_value) or share_value <= 0.0 or share_value > 1.0:
        raise ValueError(f"reconstruct_krx_bars requires share within (0, 1], got {share!r}")
    missing = [c for c in _REQUIRED_BAR_COLUMNS if c not in cons_bars.columns]
    if missing:
        raise ValueError(f"reconstruct_krx_bars missing required columns: {missing}")
    if cons_bars.empty:
        raise ValueError("reconstruct_krx_bars carries no bars")
    stamps = pd.to_numeric(cons_bars["ts_hms"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(stamps)):
        raise ValueError("reconstruct_krx_bars carries non-numeric ts_hms")
    if bool(np.any(stamps > float(CUTOFF_HHMMSS))):
        raise ValueError(f"reconstruct_krx_bars rejects bars after the 15:20 cutoff ({CUTOFF_HHMMSS})")
    volume = pd.to_numeric(cons_bars["volume"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(cons_bars["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(cons_bars["low"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(cons_bars["close"], errors="coerce").to_numpy(dtype=np.float64)
    if not bool(np.all(np.isfinite(high) & np.isfinite(low) & np.isfinite(close))):
        raise ValueError("reconstruct_krx_bars carries non-finite prices")
    if bool(np.any((high <= 0.0) | (low <= 0.0) | (close <= 0.0))):
        raise ValueError("reconstruct_krx_bars carries non-positive prices")
    if bool(np.any(volume < 0.0)) or not bool(np.all(np.isfinite(volume))):
        raise ValueError("reconstruct_krx_bars carries invalid volumes")
    scaled = volume * share_value
    typical = (high + low + close) / 3.0
    rebuilt = scaled * typical
    out = cons_bars.copy().reset_index(drop=True)
    out["volume"] = scaled
    if "value_krw" in out.columns:
        out["value_krw"] = rebuilt
    elif "value" in out.columns:
        out["value"] = rebuilt
    else:
        out["value_krw"] = rebuilt
    out["basis"] = _BASIS_RECONSTRUCTED
    out["recon_volume_rel_err_p90"] = float(config.volume_rel_err_p90)
    out["recon_close_bp_err_p90"] = float(config.close_bp_err_p90)
    return out


_CONFIG_FLOAT_FIELDS: tuple[str, ...] = (
    "ewma_alpha",
    "auction_fraction_mean",
    "bias_correction",
    "volume_rel_err_p90",
    "close_bp_err_p90",
)

_CONFIG_INT_FIELDS: tuple[str, ...] = ("min_prior_days", "max_gap_days")

_CONFIG_DATE_FIELDS: tuple[str, ...] = ("calibrated_through", "fit_start", "holdout_start", "holdout_end")


def decomposition_config_to_payload(config: DecompositionConfig) -> dict[str, Any]:
    """Render a fitted config as a JSON-serializable mapping."""
    return dict(asdict(config))


def decomposition_config_from_payload(payload: Mapping[str, Any]) -> DecompositionConfig:
    """Rebuild a config persisted by `decomposition_config_to_payload`.

    Raises:
        ValueError: Unknown, missing or mistyped fields (fail closed; a corrupt config must never
            read as a default).
    """
    caller = "decomposition config"
    if not isinstance(payload, Mapping):
        raise ValueError(f"decomposition config must hold a JSON object, got {type(payload).__name__}")
    known = {f.name for f in fields(DecompositionConfig)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"decomposition config carries unknown keys: {sorted(unknown)}")
    missing = [name for name in known if name not in payload]
    if missing:
        raise ValueError(f"decomposition config is missing keys: {sorted(missing)}")
    try:
        values: dict[str, Any] = {name: _as_float(name, payload[name], caller=caller) for name in _CONFIG_FLOAT_FIELDS}
        values.update({name: _as_int(name, payload[name], caller=caller) for name in _CONFIG_INT_FIELDS})
        dates = {name: _checked_day(name, payload[name], caller=caller) for name in _CONFIG_DATE_FIELDS}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decomposition config carries mistyped fields: {exc}") from exc
    for name in _CONFIG_FLOAT_FIELDS:
        if not np.isfinite(values[name]):
            raise ValueError(f"decomposition config carries a non-finite {name}: {values[name]!r}")
    if not 0.0 < values["ewma_alpha"] <= 1.0:
        raise ValueError(f"decomposition config carries an out-of-range ewma_alpha: {values['ewma_alpha']!r}")
    if values["min_prior_days"] < 1:
        raise ValueError(f"decomposition config carries an out-of-range min_prior_days: {values['min_prior_days']!r}")
    if values["max_gap_days"] < 0:
        raise ValueError(f"decomposition config carries an out-of-range max_gap_days: {values['max_gap_days']!r}")
    if values["bias_correction"] <= 0.0:
        raise ValueError(f"decomposition config carries an out-of-range bias_correction: {values['bias_correction']!r}")
    if values["volume_rel_err_p90"] < 0.0 or values["close_bp_err_p90"] < 0.0:
        raise ValueError("decomposition config carries a negative error quantile")
    values.update(dates)
    return DecompositionConfig(**values)


def save_decomposition_config(config: DecompositionConfig, path: Path | str) -> Path:
    """Atomically persist a fitted config as JSON.

    Raises:
        OSError: Persistence fails (no partial file is ever visible).
    """
    from src.data.io_utils import atomic_write_text

    target = Path(path)
    payload = decomposition_config_to_payload(config)
    decomposition_config_from_payload(payload)
    atomic_write_text(target, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), mode=None)
    return target


def load_decomposition_config(path: Path | str) -> DecompositionConfig:
    """Load a config persisted by `save_decomposition_config`.

    Raises:
        FileNotFoundError: No config file exists at the path.
        ValueError: The file exists but is malformed (fail closed).
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"decomposition config not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"decomposition config at {target} is malformed: {exc}") from exc
    try:
        return decomposition_config_from_payload(payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"decomposition config at {target} is malformed: {exc}") from exc


_DIAGNOSTICS_FLOAT_FIELDS: tuple[str, ...] = (
    "holdout_rel_err_p50",
    "holdout_rel_err_p90",
    "holdout_rel_err_p99",
    "holdout_coverage",
    "abar_fit",
    "abar_holdout",
    "bias_fit",
    "bias_holdout",
)

_DIAGNOSTICS_INT_FIELDS: tuple[str, ...] = ("n_fit_rows", "n_holdout_rows", "n_symbols")

_DIAGNOSTICS_DATE_FIELDS: tuple[str, ...] = ("fit_start", "fit_end", "holdout_start", "holdout_end")


def fit_diagnostics_to_payload(diagnostics: FitDiagnostics) -> dict[str, Any]:
    """Render fit diagnostics as a JSON-serializable mapping, frontier included."""
    payload = asdict(diagnostics)
    payload["frontier"] = [dict(asdict(point)) for point in diagnostics.frontier]
    return payload


def _frontier_point_from_payload(payload: Mapping[str, Any]) -> FrontierPoint:
    caller = "fit diagnostics frontier"
    if not isinstance(payload, Mapping):
        raise ValueError(f"{caller} must hold JSON objects, got {type(payload).__name__}")
    known = {f.name for f in fields(FrontierPoint)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"{caller} carries unknown keys: {sorted(unknown)}")
    missing = [name for name in known if name not in payload]
    if missing:
        raise ValueError(f"{caller} is missing keys: {sorted(missing)}")
    try:
        alpha = _as_float("ewma_alpha", payload["ewma_alpha"], caller=caller)
        min_prior_days = _as_int("min_prior_days", payload["min_prior_days"], caller=caller)
        max_gap_days = _as_int("max_gap_days", payload["max_gap_days"], caller=caller)
        coverage = _as_float("coverage", payload["coverage"], caller=caller)
        errors = {
            name: None if payload[name] is None else _as_float(name, payload[name], caller=caller)
            for name in ("rel_err_p50", "rel_err_p90", "rel_err_p99")
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{caller} carries mistyped fields: {exc}") from exc
    if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError(f"{caller} carries an out-of-range ewma_alpha: {alpha!r}")
    if min_prior_days < 1 or max_gap_days < 0:
        raise ValueError(f"{caller} carries an out-of-range structure: {(min_prior_days, max_gap_days)!r}")
    if not np.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
        raise ValueError(f"{caller} carries an out-of-range coverage: {coverage!r}")
    for name, value in errors.items():
        if value is None:
            if coverage != 0.0:
                raise ValueError(f"{caller} requires {name} for positive coverage")
        elif not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{caller} carries an invalid {name}: {value!r}")
    return FrontierPoint(
        ewma_alpha=alpha,
        min_prior_days=min_prior_days,
        max_gap_days=max_gap_days,
        coverage=coverage,
        **errors,
    )


def fit_diagnostics_from_payload(payload: Mapping[str, Any]) -> FitDiagnostics:
    """Rebuild diagnostics persisted by `fit_diagnostics_to_payload`.

    Raises:
        ValueError: Unknown, missing or mistyped fields, or non-finite required numbers (fail closed).
    """
    caller = "fit diagnostics"
    if not isinstance(payload, Mapping):
        raise ValueError(f"fit diagnostics must hold a JSON object, got {type(payload).__name__}")
    known = {f.name for f in fields(FitDiagnostics)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"fit diagnostics carries unknown keys: {sorted(unknown)}")
    missing = [name for name in known if name not in payload]
    if missing:
        raise ValueError(f"fit diagnostics is missing keys: {sorted(missing)}")
    try:
        values: dict[str, Any] = {
            name: _as_float(name, payload[name], caller=caller) for name in _DIAGNOSTICS_FLOAT_FIELDS
        }
        values.update({name: _as_int(name, payload[name], caller=caller) for name in _DIAGNOSTICS_INT_FIELDS})
        values.update({name: _checked_day(name, payload[name], caller=caller) for name in _DIAGNOSTICS_DATE_FIELDS})
        if not isinstance(payload["alpha_at_boundary"], bool):
            raise ValueError(f"{caller} carries a mistyped alpha_at_boundary: {payload['alpha_at_boundary']!r}")
        values["alpha_at_boundary"] = payload["alpha_at_boundary"]
        if not isinstance(payload["table_sha256"], str):
            raise ValueError(f"{caller} carries a mistyped table_sha256: {payload['table_sha256']!r}")
        values["table_sha256"] = payload["table_sha256"]
        frontier_raw = payload["frontier"]
        if not isinstance(frontier_raw, (list, tuple)) or not frontier_raw:
            raise ValueError(f"{caller} carries an empty frontier")
        values["frontier"] = tuple(_frontier_point_from_payload(item) for item in frontier_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fit diagnostics carries mistyped fields: {exc}") from exc
    for name in ("holdout_rel_err_p50", "holdout_rel_err_p90", "holdout_rel_err_p99"):
        if not np.isfinite(values[name]) or values[name] < 0.0:
            raise ValueError(f"fit diagnostics carries a non-finite {name}: {values[name]!r}")
    if not np.isfinite(values["holdout_coverage"]) or not 0.0 <= values["holdout_coverage"] <= 1.0:
        raise ValueError(f"fit diagnostics carries an out-of-range holdout_coverage: {values['holdout_coverage']!r}")
    for name in ("abar_fit", "abar_holdout"):
        if not np.isfinite(values[name]):
            raise ValueError(f"fit diagnostics carries a non-finite {name}: {values[name]!r}")
    for name in ("bias_fit", "bias_holdout"):
        if not np.isfinite(values[name]) or values[name] <= 0.0:
            raise ValueError(f"fit diagnostics carries an out-of-range {name}: {values[name]!r}")
    for name in _DIAGNOSTICS_INT_FIELDS:
        if values[name] < 0:
            raise ValueError(f"fit diagnostics carries an out-of-range {name}: {values[name]!r}")
    return FitDiagnostics(**values)


def save_fit_diagnostics(diagnostics: FitDiagnostics, path: Path | str) -> Path:
    """Atomically persist fit diagnostics as JSON.

    Raises:
        OSError: Persistence fails (no partial file is ever visible).
    """
    from src.data.io_utils import atomic_write_text

    target = Path(path)
    payload = fit_diagnostics_to_payload(diagnostics)
    fit_diagnostics_from_payload(payload)
    atomic_write_text(target, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), mode=None)
    return target


def load_fit_diagnostics(path: Path | str) -> FitDiagnostics:
    """Load diagnostics persisted by `save_fit_diagnostics`.

    Raises:
        FileNotFoundError: No report file exists at the path.
        ValueError: The file exists but is malformed (fail closed).
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"fit diagnostics not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"fit diagnostics at {target} is malformed: {exc}") from exc
    try:
        return fit_diagnostics_from_payload(payload)
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"fit diagnostics at {target} is malformed: {exc}") from exc
