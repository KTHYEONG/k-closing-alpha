"""Measured per-instrument execution cost model (decision-time only).

Every component is computable at the entry-day close: tick/spread read only
the entry price, and the auction impact reads only entry-day bars with
``ts_hms <= 153000``. No next-day bar or realized exit price is ever read.
This module is measurement-only and never changes ``ROUND_TRIP_COST_RATIO``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.intraday_schema import CANONICAL_BAR_COLUMNS, normalize_bar_frame
from src.data.intraday_store import intraday_partition_path

logger = logging.getLogger(__name__)

# Public entry points: estimate_* consume an impact column produced by
# measure_auction_impact_bp (the producer runs ahead of selection; its
# output joins back via impact_col, so no in-module call site exists).
__all__ = [
    "KRX_TICK_BANDS",
    "STATUTORY_COST_BP",
    "CostBreakdown",
    "breakeven_cost_bp",
    "estimate_round_trip_cost_bp",
    "krx_tick_size",
    "measure_auction_impact_bp",
    "spread_cost_bp",
    "summarize_cost_breakdown",
]

KRX_TICK_BANDS: tuple[tuple[float, float], ...] = (
    (2000.0, 1.0),
    (5000.0, 5.0),
    (20000.0, 10.0),
    (50000.0, 50.0),
    (200000.0, 100.0),
    (500000.0, 500.0),
    (float("inf"), 1000.0),
)

STATUTORY_COST_BP: float = 21.0

_AUCTION_CLOSE_HMS: int = 153000


@dataclass(frozen=True)
class CostBreakdown:
    statutory_bp: float
    spread_bp: float
    auction_impact_bp: float
    total_bp: float
    n_rows: int
    n_impact_measured: int


def krx_tick_size(price: np.ndarray) -> np.ndarray:
    """Map entry price to the KRX cash-equity tick (``price < bound`` wins)."""
    arr = np.asarray(price, dtype=np.float64)
    tick = np.full(arr.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(arr) & (arr > 0.0)
    for bound, size in KRX_TICK_BANDS:
        take = valid & np.isnan(tick) & (arr < float(bound))
        tick[take] = float(size)
    return tick


def spread_cost_bp(price: np.ndarray, *, round_trip_ticks: float = 2.0) -> np.ndarray:
    """Round-trip spread cost in bp; bad prices propagate NaN, never 0."""
    arr = np.asarray(price, dtype=np.float64)
    tick = krx_tick_size(arr)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(tick) & np.isfinite(arr) & (arr > 0.0)
    out[ok] = float(round_trip_ticks) * tick[ok] / arr[ok] * 10000.0
    return out


def _candidate_partition_paths(
    bar_interval_minutes: int, snapshot_date: str, session: str, intraday_root: Path | None
) -> list[Path]:
    if intraday_root is None:
        return [intraday_partition_path(int(bar_interval_minutes), str(snapshot_date), str(session))]
    root = Path(intraday_root)
    month = str(snapshot_date)[:7]
    interval = f"{int(bar_interval_minutes)}m"
    return [
        root / "intraday" / interval / str(session) / month / f"{snapshot_date}.parquet",
        root / interval / str(session) / month / f"{snapshot_date}.parquet",
    ]


def measure_auction_impact_bp(
    df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    code_col: str = "stock_code",
    decision_hms: int = 152000,
    bar_interval_minutes: int = 1,
    intraday_root: Path | None = None,
) -> pd.DataFrame:
    """Attach measured decision-to-auction drift; fail-open when unmeasured."""
    if date_col not in df.columns or code_col not in df.columns:
        raise ValueError(f"df is missing date_col/code_col {(date_col, code_col)}")
    out = df.copy()
    for col in ("auction_impact_bp", "impact_measured"):
        if col in out.columns:
            out = out.drop(columns=[col])
    dates = pd.to_datetime(out[date_col], errors="coerce")
    snap_dates = dates.dt.strftime("%Y-%m-%d")
    codes = out[code_col].astype(str).str.zfill(6)

    impact = np.full(len(out), np.nan, dtype=np.float64)
    measured = np.zeros(len(out), dtype=bool)

    for snap in sorted(pd.unique(snap_dates.dropna())):
        snap_str = str(snap)
        paths = _candidate_partition_paths(
            bar_interval_minutes, snap_str, "regular", intraday_root
        )
        target = next((p for p in paths if p.exists()), None)
        if target is None:
            logger.warning("[DATA] cost_model intraday partition missing date=%s path=%s", snap_str, paths[0])
            continue
        try:
            try:
                raw = pd.read_parquet(target, columns=list(CANONICAL_BAR_COLUMNS), engine="pyarrow")
            except Exception:
                raw = pd.read_parquet(target)
        except Exception as exc:
            logger.warning("[DATA] cost_model intraday read failed date=%s path=%s: %s", snap_str, target, exc)
            continue
        if raw is None or len(raw) == 0:
            logger.warning("[DATA] cost_model intraday partition empty date=%s path=%s", snap_str, target)
            continue
        if set(CANONICAL_BAR_COLUMNS).issubset(set(raw.columns)):
            frame = pd.DataFrame(
                {
                    "symbol": raw["symbol"].astype(str).str.zfill(6),
                    "ts_hms": pd.to_numeric(raw["ts_hms"], errors="coerce"),
                    "close": pd.to_numeric(raw["close"], errors="coerce"),
                }
            )
        else:
            # Almost every on-disk partition today (242/243) is raw,
            # unnormalized vendor output keyed by '종목코드' (KIS) rather
            # than the canonical 'symbol' -- normalize per symbol group
            # rather than mixing every stock's cumulative series together.
            vendor = "ls" if "jdiff_vol" in raw.columns else "kis"
            raw_symbol_col = "symbol" if "symbol" in raw.columns else ("종목코드" if "종목코드" in raw.columns else None)
            if raw_symbol_col is None:
                logger.warning("[DATA] cost_model raw partition missing symbol column date=%s path=%s", snap_str, target)
                continue
            symbols = raw[raw_symbol_col].astype(str).str.zfill(6).unique().tolist()
            parts: list[pd.DataFrame] = []
            for symbol in symbols:
                sub = raw[raw[raw_symbol_col].astype(str).str.zfill(6) == str(symbol)]
                try:
                    norm = normalize_bar_frame(sub, vendor, snap_str, symbol)
                except Exception as exc:  # pragma: no cover
                    logger.warning("[DATA] cost_model normalize failed date=%s symbol=%s: %s", snap_str, symbol, exc)  # pragma: no cover
                    continue  # pragma: no cover
                parts.append(pd.DataFrame({"symbol": norm["symbol"], "ts_hms": norm["ts_hms"], "close": norm["close"]}))
            if not parts:
                logger.warning("[DATA] cost_model raw partition yielded no usable symbols date=%s path=%s", snap_str, target)
                continue
            frame = pd.concat(parts, ignore_index=True)
        frame = frame[frame["ts_hms"] <= _AUCTION_CLOSE_HMS]
        per_symbol: dict[str, float] = {}
        for symbol, group in frame.groupby("symbol", sort=False):
            ordered = group.sort_values("ts_hms", kind="stable")
            cont = ordered[ordered["ts_hms"] < int(decision_hms)]
            auc = ordered[ordered["ts_hms"] >= int(decision_hms)]
            if len(cont) < 10 or len(auc) == 0:
                continue
            last_cont = float(cont["close"].iloc[-1])
            auction_close = float(auc["close"].iloc[-1])
            if not np.isfinite(last_cont) or not np.isfinite(auction_close):
                continue
            if last_cont <= 0.0 or auction_close <= 0.0:
                continue
            per_symbol[str(symbol)] = float(auction_close / last_cont - 1.0) * 10000.0
        mask = (snap_dates == snap_str).to_numpy()
        for idx in np.flatnonzero(mask):
            key = str(codes.iloc[idx])
            if key not in per_symbol:
                continue
            impact[idx] = np.float64(per_symbol[key])
            measured[idx] = True

    out["auction_impact_bp"] = np.asarray(impact, dtype=np.float64)
    out["impact_measured"] = np.asarray(measured, dtype=bool)
    return out


def estimate_round_trip_cost_bp(
    df: pd.DataFrame,
    *,
    price_col: str = "close_price",
    round_trip_ticks: float = 2.0,
    statutory_bp: float = STATUTORY_COST_BP,
    impact_col: str | None = None,
) -> pd.DataFrame:
    """Compose per-row round-trip cost; fail-open on unmeasured inputs."""
    if price_col not in df.columns:
        raise ValueError(f"df is missing price_col {price_col!r}")
    out = df.copy()
    prices = pd.to_numeric(out[price_col], errors="coerce").to_numpy(dtype=np.float64)
    tick = krx_tick_size(prices)
    spread = spread_cost_bp(prices, round_trip_ticks=float(round_trip_ticks))
    out["tick_krw"] = np.asarray(tick, dtype=np.float64)
    out["spread_bp"] = np.asarray(spread, dtype=np.float64)
    out["statutory_bp"] = np.full(len(out), float(statutory_bp), dtype=np.float64)
    if impact_col is None or impact_col not in out.columns:
        out["auction_impact_bp"] = np.full(len(out), np.nan, dtype=np.float64)
        impact_term = np.zeros(len(out), dtype=np.float64)
    else:
        vals = pd.to_numeric(out[impact_col], errors="coerce").to_numpy(dtype=np.float64)
        out["auction_impact_bp"] = np.asarray(vals, dtype=np.float64)
        impact_term = np.where(np.isfinite(vals), vals, 0.0).astype(np.float64)
    total = out["statutory_bp"].to_numpy(dtype=np.float64) + out["spread_bp"].to_numpy(dtype=np.float64) + impact_term
    out["round_trip_cost_bp"] = np.asarray(total, dtype=np.float64)
    return out


def summarize_cost_breakdown(df: pd.DataFrame) -> CostBreakdown:
    """Mean cost components plus the measured-impact denominator."""
    n_rows = len(df)
    if "auction_impact_bp" in df.columns:
        auction_vals = pd.to_numeric(df["auction_impact_bp"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        auction_vals = np.full(n_rows, np.nan, dtype=np.float64)
    n_measured = int(np.isfinite(auction_vals).sum())
    if "statutory_bp" in df.columns:
        statutory = float(np.nanmean(pd.to_numeric(df["statutory_bp"], errors="coerce").to_numpy(dtype=np.float64)))
    else:
        statutory = float(STATUTORY_COST_BP)
    if "spread_bp" in df.columns:
        spread = float(np.nanmean(pd.to_numeric(df["spread_bp"], errors="coerce").to_numpy(dtype=np.float64)))
    else:
        spread = float("nan")
    auction = float(np.nanmean(auction_vals)) if n_measured else float("nan")
    if "round_trip_cost_bp" in df.columns:
        total = float(np.nanmean(pd.to_numeric(df["round_trip_cost_bp"], errors="coerce").to_numpy(dtype=np.float64)))
    else:
        total = float("nan")
    return CostBreakdown(
        statutory_bp=statutory,
        spread_bp=spread,
        auction_impact_bp=auction,
        total_bp=total,
        n_rows=n_rows,
        n_impact_measured=n_measured,
    )


def breakeven_cost_bp(gross_return_pct: np.ndarray, groups: np.ndarray) -> float:
    """Equal-weighted per-group mean gross return scaled to bp."""
    rets = np.asarray(gross_return_pct, dtype=np.float64).ravel()
    grp = np.asarray(groups).ravel()
    if rets.shape != grp.shape:
        raise ValueError(f"gross_return_pct and groups must align, got {rets.shape} vs {grp.shape}")
    day_means: list[float] = []
    for key in np.unique(grp):
        vals = rets[grp == key]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        day_means.append(float(np.mean(finite)))
    if len(day_means) < 30:
        return float("nan")
    return float(np.mean(np.asarray(day_means, dtype=np.float64))) * 100.0
