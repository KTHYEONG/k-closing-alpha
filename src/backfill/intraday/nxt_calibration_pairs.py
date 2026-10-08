"""NXT consolidated-tape calibration pairs: KRX truth vs consolidated Toss bars.

For stored vendor-`kis` (KRX-only) symbol-days of NXT-listed symbols, pair the KIS bars
with the consolidated Toss bars of the same day, so the volume decomposition model can be
calibrated and validated against ground truth. Writes nothing to production partitions
or the production ledger: Toss evidence goes to the explicit `evidence_store`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.backfill.intraday.toss_equivalence_audit import (
    _KIS_FULL_SESSION_FIRST_HHMMSS,
    _KIS_FULL_SESSION_LAST_HHMMSS,
)
from src.backfill.intraday.toss_regular import acquire_toss_regular_bars
from src.config.collection import CollectionSettings
from src.data.capture_contracts import CaptureStatus
from src.data.capture_store import CaptureStore

logger = logging.getLogger(__name__)

_CONSOLIDATED_REASON: str = "toss_consolidated_tape"

# KIS start-stamp cutoffs: the continuous session ends at 15:20 and the 15:30 print is the auction.
_KRX_CONTINUOUS_CUTOFF_HHMMSS: int = 151900
_KRX_AUCTION_TS_HHMMSS: int = 153000
_KRX_CLOSE_REF_HHMMSS: int = 151900
# Toss end-stamps lag KIS start-stamps by one minute: label 15:20:00 is the last continuous minute, the decision-time cutoff.
_CONS_CONTINUOUS_CUTOFF_HHMMSS: int = 152000
_CONS_CLOSE_REF_HHMMSS: int = 152000

# Relative bound on |EOD - (V_krx + A)|; measured median EOD/(V+A) is 1.0001 (p95 1.016).
DEFAULT_IDENTITY_RESIDUAL_TOLERANCE: float = 0.05

CALIBRATION_TABLE_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "v_krx_1520",
    "auction_volume",
    "v_cons_1520",
    "v_cons_full",
    "close_krx_1519",
    "close_cons_1519",
    "high_krx",
    "low_krx",
    "high_cons",
    "low_cons",
    "eod_volume",
    "identity_residual",
)


@dataclass(frozen=True)
class CalibrationSample:
    """One candidate symbol-day carrying its KRX truth and point-in-time basis state."""

    snapshot_date: str
    symbol: str
    kis_bars: pd.DataFrame
    eod_volume: float
    is_raw_basis: bool = True


def _is_full_session_kis(frame: pd.DataFrame) -> bool:
    if frame is None or frame.empty or not {"symbol", "vendor", "ts_hms"} <= set(frame.columns):
        return False
    if {str(v) for v in frame["vendor"].astype(str).tolist()} != {"kis"}:
        return False
    stamps = pd.to_numeric(frame["ts_hms"], errors="coerce").dropna()
    if stamps.empty:
        return False
    return int(stamps.min()) <= _KIS_FULL_SESSION_FIRST_HHMMSS and int(stamps.max()) >= _KIS_FULL_SESSION_LAST_HHMMSS


def _total_volume(frame: pd.DataFrame) -> float:
    if frame is None or frame.empty or "volume" not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame["volume"], errors="coerce").fillna(0).sum())


def _volumes_at_or_below(frame: pd.DataFrame, cutoff: int) -> float:
    if frame is None or frame.empty or not {"ts_hms", "volume"} <= set(frame.columns):
        return 0.0
    stamps = pd.to_numeric(frame["ts_hms"], errors="coerce")
    volumes = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
    return float(volumes[stamps <= cutoff].sum())


def _volume_at(frame: pd.DataFrame, ts: int) -> float:
    if frame is None or frame.empty or not {"ts_hms", "volume"} <= set(frame.columns):
        return 0.0
    stamps = pd.to_numeric(frame["ts_hms"], errors="coerce")
    volumes = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
    matched = volumes[stamps == ts]
    return float(matched.sum()) if len(matched) else 0.0


def _close_at(frame: pd.DataFrame, ts: int) -> float:
    if frame is None or frame.empty or not {"ts_hms", "close"} <= set(frame.columns):
        return float("nan")
    stamps = pd.to_numeric(frame["ts_hms"], errors="coerce")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    matched = closes[stamps == ts]
    return float(matched.iloc[0]) if len(matched) else float("nan")


def _side_range(frame: pd.DataFrame) -> tuple[float, float]:
    if frame is None or frame.empty or not {"high", "low"} <= set(frame.columns):
        return float("nan"), float("nan")
    high = pd.to_numeric(frame["high"], errors="coerce").dropna()
    low = pd.to_numeric(frame["low"], errors="coerce").dropna()
    if high.empty or low.empty:
        return float("nan"), float("nan")
    return float(high.max()), float(low.min())


async def collect_calibration_pairs(
    *,
    samples: Sequence[CalibrationSample],
    client: Any,
    session: Any,
    profile: CollectionSettings,
    evidence_store: CaptureStore,
    run_id: str,
) -> pd.DataFrame:
    """Pair stored KRX-truth symbol-days with their consolidated Toss bars.

    Only symbol-days with exact KRX truth are fetched: vendor-`kis` full-session bars on the
    raw price basis. Of those, only NXT-listed days (gate verdict `toss_consolidated_tape`) are
    paired; Toss evidence lands in the explicit temporary `evidence_store` and no production
    partition or ledger is touched.

    Returns:
        One row per paired symbol-day with `date`, `symbol`, `kis_bars`, `toss_bars` and
        `eod_volume`; empty when nothing qualifies.
    """
    if not str(run_id).strip():
        raise ValueError("run_id must be nonempty")
    if evidence_store is None:
        raise ValueError("evidence_store must be an explicit temporary CaptureStore")
    rows: list[dict[str, Any]] = []
    skipped_truth = 0
    skipped_gate = 0
    for sample in samples:
        day = str(sample.snapshot_date)
        symbol = str(sample.symbol).zfill(6)
        if not sample.is_raw_basis or not _is_full_session_kis(sample.kis_bars):
            skipped_truth += 1
            continue
        toss_frame, entry = await acquire_toss_regular_bars(
            client,
            session,
            symbol,
            day,
            eod_volume=sample.eod_volume,
            profile=profile,
            capture_store=evidence_store,
            run_id=str(run_id),
        )
        if entry.status != CaptureStatus.NOT_APPLICABLE or entry.reason != _CONSOLIDATED_REASON:
            skipped_gate += 1
            continue
        if toss_frame is None or toss_frame.empty:
            skipped_gate += 1
            continue
        rows.append(
            {
                "date": day,
                "symbol": symbol,
                "kis_bars": sample.kis_bars.reset_index(drop=True),
                "toss_bars": toss_frame.reset_index(drop=True),
                "eod_volume": float(sample.eod_volume),
            }
        )
    logger.info(
        "[DATA] stage=nxt_calibration_pairs status=COLLECTED pairs=%d skipped_truth=%d skipped_gate=%d",
        len(rows),
        skipped_truth,
        skipped_gate,
    )
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "kis_bars", "toss_bars", "eod_volume"])
    return pd.DataFrame(rows, columns=["date", "symbol", "kis_bars", "toss_bars", "eod_volume"])


def calibration_row_from_pair(row: Mapping[str, Any]) -> dict[str, Any]:
    """Derive ONE calibration-table row from one collected pair.

    Pure function of the pair's bars: per-side volumes at the 15:20 observability
    boundary, the 15:30 auction print, 15:19 reference closes, full-session high/low
    of both sides, EOD volume and the identity residual `EOD - (V_krx + A)`.
    The identity exclusion itself stays with the table builder; this function never
    filters.

    Raises:
        KeyError: The pair lacks `kis_bars`, `toss_bars`, `eod_volume`, `date` or `symbol`.
    """
    kis = row["kis_bars"]
    toss = row["toss_bars"]
    eod = float(row["eod_volume"])
    v_krx = _volumes_at_or_below(kis, _KRX_CONTINUOUS_CUTOFF_HHMMSS)
    auction = _volume_at(kis, _KRX_AUCTION_TS_HHMMSS)
    high_krx, low_krx = _side_range(kis)
    high_cons, low_cons = _side_range(toss)
    return {
        "date": str(row["date"]),
        "symbol": str(row["symbol"]),
        "v_krx_1520": v_krx,
        "auction_volume": auction,
        "v_cons_1520": _volumes_at_or_below(toss, _CONS_CONTINUOUS_CUTOFF_HHMMSS),
        "v_cons_full": _total_volume(toss),
        "close_krx_1519": _close_at(kis, _KRX_CLOSE_REF_HHMMSS),
        "close_cons_1519": _close_at(toss, _CONS_CLOSE_REF_HHMMSS),
        "high_krx": high_krx,
        "low_krx": low_krx,
        "high_cons": high_cons,
        "low_cons": low_cons,
        "eod_volume": eod,
        "identity_residual": eod - (v_krx + auction),
    }


def build_calibration_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Derive the per-symbol-day calibration table from collected pairs.

    The table is a pure function of `pairs`: per-side volumes at the 15:20 observability
    boundary, the 15:30 auction print, 15:19 reference closes, full-session high/low of both
    sides, EOD volume and the identity residual `EOD - (V_krx + A)`. Rows whose relative
    residual exceeds `DEFAULT_IDENTITY_RESIDUAL_TOLERANCE` are excluded and counted in
    `attrs["n_identity_excluded"]`, never silently kept.

    Returns:
        Calibration rows with `CALIBRATION_TABLE_COLUMNS`; empty when no pair survives.
    """
    if pairs is None or pairs.empty:
        table = pd.DataFrame(columns=list(CALIBRATION_TABLE_COLUMNS))
        table.attrs["n_identity_excluded"] = 0
        return table
    derived: list[dict[str, Any]] = []
    for _, row in pairs.iterrows():
        derived.append(calibration_row_from_pair(row))
    table = pd.DataFrame(derived, columns=list(CALIBRATION_TABLE_COLUMNS))
    denom = pd.to_numeric(table["eod_volume"], errors="coerce").to_numpy(dtype=float)
    residual = pd.to_numeric(table["identity_residual"], errors="coerce").to_numpy(dtype=float)
    rel = np.where(denom > 0, np.abs(residual) / np.where(denom > 0, denom, 1.0), np.inf)
    bad = rel > float(DEFAULT_IDENTITY_RESIDUAL_TOLERANCE)
    excluded = int(np.sum(bad))
    kept = table.loc[~bad].reset_index(drop=True)
    if excluded:
        logger.warning(
            "[DATA] stage=nxt_calibration_pairs status=IDENTITY_EXCLUDED excluded=%d kept=%d",
            excluded,
            len(kept),
        )
    kept.attrs["n_identity_excluded"] = excluded
    return kept
