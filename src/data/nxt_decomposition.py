"""Causal consolidated-to-KRX decomposition estimator (decision cutoff 15:20).

Reconstructs KRX-only bars from the consolidated tape using only information
observable at the decision time plus strictly earlier history. A day-T estimate
scales consolidated bars at or before the cutoff by a KRX share predicted from
earlier days; day-T EOD volume and post-cutoff bars never participate.
"""

from __future__ import annotations

import json
import logging
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


@dataclass(frozen=True)
class DecompositionConfig:
    """Structural parameters of the consolidated-to-KRX measurement basis."""

    ewma_alpha: float
    min_prior_days: int
    max_gap_days: int
    auction_fraction_mean: float
    bias_correction: float
    volume_rel_err_p90: float
    close_bp_err_p90: float
    calibrated_through: str


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


def _scored_errors(
    frame: pd.DataFrame,
    score_dates: set[str],
    alpha: float,
) -> tuple[list[float], list[float]]:
    """Walk-forward relative errors and true/raw ratios on scoring rows.

    Each scoring row is predicted from strictly earlier rows of the same
    symbol, so the row's own volumes never inform its prediction.
    """
    ordered = frame.sort_values(["symbol", "date"], kind="stable")
    history: dict[str, list[tuple[str, float]]] = {}
    errors: list[float] = []
    ratios: list[float] = []
    for row in ordered.to_dict(orient="records"):
        symbol = str(row["symbol"])
        day = str(row["day"])
        true_share = float(row["true_share"])
        prior = [share for _, share in history.get(symbol, [])]
        history.setdefault(symbol, []).append((day, true_share))
        if day not in score_dates or not prior:
            continue
        raw = _ewma(prior, alpha)
        if not np.isfinite(raw) or raw <= 0.0 or not np.isfinite(true_share) or true_share <= 0.0:
            continue
        errors.append(abs(raw - true_share) / true_share)
        ratios.append(true_share / raw)
    return errors, ratios


def _raw_true_pairs(
    frame: pd.DataFrame, score_dates: set[str], alpha: float
) -> list[tuple[float, float]]:
    ordered = frame.sort_values(["symbol", "date"], kind="stable")
    history: dict[str, list[float]] = {}
    pairs: list[tuple[float, float]] = []
    for row in ordered.to_dict(orient="records"):
        symbol = str(row["symbol"])
        day = str(row["day"])
        true_share = float(row["true_share"])
        prior = list(history.get(symbol, []))
        history.setdefault(symbol, []).append(true_share)
        if day not in score_dates or not prior:
            continue
        raw = _ewma(prior, alpha)
        if np.isfinite(raw) and raw > 0.0 and np.isfinite(true_share) and true_share > 0.0:
            pairs.append((raw, true_share))
    return pairs


def fit_decomposition_config(
    calibration: pd.DataFrame,
    *,
    alphas: Sequence[float],
    min_prior_days: int,
    max_gap_days: int,
    identity_tolerance: float,
) -> DecompositionConfig:
    """Fit the few structural parameters of the consolidated-to-KRX decomposition on symbol-days where both tapes exist (KIS window). The parameters describe the measurement basis of the vendors, not alpha, and are low-dimensional by construction (EWMA coefficient, mean auction fraction, median bias, error quantiles). Selection of `ewma_alpha` minimizes median relative error on a walk-forward split by date (fit on earlier dates, score on later ones), never on the fitting rows.

    Raises:
        ValueError: When fewer than a declared minimum of symbol-days or symbols remain after identity filtering.
    """
    candidates = [float(a) for a in alphas]
    if not candidates or any(not np.isfinite(a) or a <= 0.0 or a > 1.0 for a in candidates):
        raise ValueError("alphas must be a nonempty sequence within (0, 1]")
    if int(min_prior_days) < 1:
        raise ValueError("min_prior_days must be >= 1")
    if int(max_gap_days) < 0:
        raise ValueError("max_gap_days must be >= 0")
    if not np.isfinite(float(identity_tolerance)) or float(identity_tolerance) <= 0.0:
        raise ValueError("identity_tolerance must be positive")
    missing = [c for c in _REQUIRED_CALIBRATION_COLUMNS if c not in calibration.columns]
    if missing:
        raise ValueError(f"fit_decomposition_config missing required columns: {missing}")
    if calibration.empty:
        raise ValueError("fit_decomposition_config carries no calibration rows")

    work = calibration.copy()
    day_labels = pd.to_datetime(work["date"], errors="coerce", format="mixed").dt.strftime("%Y-%m-%d")
    work["day"] = day_labels
    if bool(work["day"].isna().any()):
        raise ValueError("fit_decomposition_config carries unparseable dates")
    for column in ("v_krx_1520", "auction_volume", "v_cons_1520", "eod_volume", "close_krx_1519", "close_cons_1519"):
        work[column] = pd.to_numeric(work[column], errors="coerce").astype(np.float64)
    work["symbol"] = work["symbol"].astype(str)
    eod = work["eod_volume"].to_numpy(dtype=np.float64)
    v_krx = work["v_krx_1520"].to_numpy(dtype=np.float64)
    v_cons = work["v_cons_1520"].to_numpy(dtype=np.float64)
    auction = work["auction_volume"].to_numpy(dtype=np.float64)
    residual = np.where(eod > 0.0, np.abs(eod - (v_krx + auction)) / np.where(eod > 0.0, eod, 1.0), np.inf)
    valid = (
        np.isfinite(eod) & (eod > 0.0) & np.isfinite(v_krx) & (v_krx > 0.0) & np.isfinite(v_cons) & (v_cons > 0.0)
        & np.isfinite(auction) & (auction >= 0.0) & np.isfinite(residual) & (residual <= float(identity_tolerance))
    )
    kept = work.loc[valid].copy().reset_index(drop=True)
    if len(kept) < max(int(min_prior_days), 2) or int(kept["symbol"].nunique()) < 2:
        raise ValueError(
            f"fit_decomposition_config keeps {len(kept)} symbol-days "
            f"over {int(kept['symbol'].nunique()) if len(kept) else 0} symbols after identity filtering"
        )
    kept["true_share"] = kept["v_krx_1520"].to_numpy(dtype=np.float64) / kept["v_cons_1520"].to_numpy(dtype=np.float64)
    kept = kept.loc[np.isfinite(kept["true_share"].to_numpy(dtype=np.float64)) & (kept["true_share"] > 0.0)]
    kept = kept.reset_index(drop=True)
    if len(kept) < max(int(min_prior_days), 2) or int(kept["symbol"].nunique()) < 2:
        raise ValueError("fit_decomposition_config keeps too few positive-share rows after identity filtering")

    auction_fraction = kept["auction_volume"].to_numpy(dtype=np.float64) / kept["eod_volume"].to_numpy(dtype=np.float64)
    auction_fraction_mean = float(np.mean(auction_fraction[np.isfinite(auction_fraction)]))

    close_krx = kept["close_krx_1519"].to_numpy(dtype=np.float64)
    close_cons = kept["close_cons_1519"].to_numpy(dtype=np.float64)
    close_ok = np.isfinite(close_krx) & np.isfinite(close_cons) & (close_krx > 0.0) & (close_cons > 0.0)
    close_bp_err_p90 = (
        float(np.quantile(np.abs(close_cons[close_ok] - close_krx[close_ok]) / close_krx[close_ok] * 1e4, 0.9))
        if bool(close_ok.any())
        else float("nan")
    )

    dates = sorted(kept["day"].unique().tolist())
    primary_scores = set(dates[max(1, len(dates) // 2):]) if len(dates) >= 2 else set(dates)
    medians: dict[float, float] = {}
    per_alpha: dict[float, tuple[list[float], list[float]]] = {}
    effective_scores: dict[float, set[str]] = {}
    for alpha in candidates:
        errors, ratios = _scored_errors(kept, primary_scores, float(alpha))
        used = set(primary_scores)
        if not errors and len(dates) >= 2:
            errors, ratios = _scored_errors(kept, set(dates), float(alpha))
            used = set(dates)
        medians[float(alpha)] = float(np.median(errors)) if errors else float("inf")
        per_alpha[float(alpha)] = (errors, ratios)
        effective_scores[float(alpha)] = used
    if all(not np.isfinite(m) or m == float("inf") for m in medians.values()):
        raise ValueError("fit_decomposition_config cannot score any alpha walk-forward")
    best_alpha = min(candidates, key=lambda a: (medians[float(a)], float(a)))
    _, best_ratios = per_alpha[float(best_alpha)]
    finite_ratios = [r for r in best_ratios if np.isfinite(r) and r > 0.0]
    bias_correction = float(np.median(finite_ratios)) if finite_ratios else 1.0
    pairs = _raw_true_pairs(kept, effective_scores[float(best_alpha)], float(best_alpha))
    corrected = [abs(raw * bias_correction - true) / true for raw, true in pairs]
    volume_rel_err_p90 = float(np.quantile(corrected, 0.9)) if corrected else float("nan")

    calibrated_through = str(max(dates))
    logger.info(
        "[DATA] stage=nxt_decomposition status=FITTED alpha=%.3f rows=%d symbols=%d",
        float(best_alpha),
        len(kept),
        int(kept["symbol"].nunique()),
    )
    return DecompositionConfig(
        ewma_alpha=float(best_alpha),
        min_prior_days=int(min_prior_days),
        max_gap_days=int(max_gap_days),
        auction_fraction_mean=float(auction_fraction_mean),
        bias_correction=float(bias_correction),
        volume_rel_err_p90=float(volume_rel_err_p90),
        close_bp_err_p90=float(close_bp_err_p90),
        calibrated_through=calibrated_through,
    )


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
    raw = _ewma([share for _, share in dated], alpha)
    corrected = float(raw * float(config.bias_correction))
    if not np.isfinite(corrected) or corrected <= 0.0 or corrected > 1.0:
        return None
    return corrected


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


def decomposition_config_to_payload(config: DecompositionConfig) -> dict[str, Any]:
    """Render a fitted config as a JSON-serializable mapping (records `calibrated_through`)."""
    return dict(asdict(config))


def decomposition_config_from_payload(payload: Mapping[str, Any]) -> DecompositionConfig:
    """Rebuild a config persisted by `decomposition_config_to_payload`.

    Raises:
        ValueError: Unknown, missing or mistyped fields (fail closed; a corrupt config must never
            read as a default).
    """
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
        values: dict[str, Any] = {name: float(payload[name]) for name in _CONFIG_FLOAT_FIELDS}
        values.update({name: int(payload[name]) for name in _CONFIG_INT_FIELDS})
        calibrated_through = str(payload["calibrated_through"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decomposition config carries mistyped fields: {exc}") from exc
    if not calibrated_through.strip():
        raise ValueError("decomposition config carries an empty calibrated_through")
    return DecompositionConfig(calibrated_through=calibrated_through, **values)


def save_decomposition_config(config: DecompositionConfig, path: Path | str) -> Path:
    """Atomically persist a fitted config as JSON.

    Raises:
        OSError: Persistence fails (no partial file is ever visible).
    """
    from src.data.io_utils import atomic_write_text

    target = Path(path)
    atomic_write_text(target, json.dumps(decomposition_config_to_payload(config), indent=2, sort_keys=True), mode=None)
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
