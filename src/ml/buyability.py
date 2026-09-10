"""Buyability gating and sleeve evaluation (research/provenance only).

Decision-time only: every quantity is computable at the entry-day close.
No next-day bars, no exit-path columns, no serving mutation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.market_session import DECISION_WINDOW_END_HMS, DECISION_WINDOW_START_HMS, INTRADAY_SESSION_REGULAR
from src.data.intraday_schema import CANONICAL_BAR_COLUMNS, normalize_bar_frame
from src.data.intraday_store import intraday_partition_path

logger = logging.getLogger(__name__)

CEILING_RATIO_THRESHOLD: float = 1.29
DEFAULT_PARTICIPATION_CAP: float = 0.10


def classify_ceiling_entry(
    df: pd.DataFrame,
    *,
    close_col: str = "close_price",
    prev_close_col: str = "prev_close_price",
    high_col: str = "high_price",
    ratio_threshold: float = CEILING_RATIO_THRESHOLD,
) -> pd.Series:
    """Flag closes at the session high and at the +30% daily limit."""
    for col in (close_col, prev_close_col, high_col):
        if col not in df.columns:
            raise ValueError(f"df is missing required column {col!r}")
    close = pd.to_numeric(df[close_col], errors="coerce").to_numpy(dtype=np.float64)
    prev = pd.to_numeric(df[prev_close_col], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(df[high_col], errors="coerce").to_numpy(dtype=np.float64)
    valid_prev = np.isfinite(prev) & (prev > 0.0)
    ratio = np.full(close.shape, np.nan, dtype=np.float64)
    np.divide(close, prev, out=ratio, where=valid_prev)
    at_limit = np.isfinite(ratio) & (ratio >= float(ratio_threshold))
    at_high = np.isfinite(close) & np.isfinite(high) & (close >= high)
    flag = at_limit & at_high & valid_prev
    return pd.Series(np.asarray(flag, dtype=bool), index=df.index, dtype=bool)


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


def attach_entry_auction_liquidity(
    df: pd.DataFrame,
    *,
    date_col: str = "trade_date",
    code_col: str = "stock_code",
    bar_interval_minutes: int = 1,
    auction_start_hms: int = DECISION_WINDOW_START_HMS,
    intraday_root: Path | None = None,
) -> pd.DataFrame:
    """Attach entry-day auction liquidity; fail-open with NaN when unmeasured."""
    if date_col not in df.columns or code_col not in df.columns:
        raise ValueError(f"df is missing date_col/code_col {(date_col, code_col)}")
    out = df.copy()
    # Overwrite semantics for idempotency (R11): drop existing outputs first.
    for col in ("auction_value_100m", "auction_vol_share", "auction_bars_found"):
        if col in out.columns:
            out = out.drop(columns=[col])
    dates = pd.to_datetime(out[date_col], errors="coerce")
    codes = out[code_col].astype(str).str.zfill(6)
    snap_dates = dates.dt.strftime("%Y-%m-%d")

    auction_value = np.full(len(out), np.nan, dtype=np.float64)
    auction_share = np.full(len(out), np.nan, dtype=np.float64)
    bars_found = np.zeros(len(out), dtype=bool)

    # One partition at a time, aggregated to (date, symbol) before any join.
    for snap in sorted(pd.unique(snap_dates.dropna())):
        snap_str = str(snap)
        paths = _candidate_partition_paths(bar_interval_minutes, snap_str, INTRADAY_SESSION_REGULAR, intraday_root)
        target = next((p for p in paths if p.exists()), None)
        if target is None:
            logger.warning("[DATA] buyability intraday partition missing date=%s path=%s", snap_str, paths[0])
            continue
        try:
            try:
                raw = pd.read_parquet(target, columns=list(CANONICAL_BAR_COLUMNS))
            except Exception:
                raw = pd.read_parquet(target)
        except Exception as exc:  # pragma: no cover
            logger.warning("[DATA] buyability intraday read failed date=%s path=%s: %s", snap_str, target, exc)  # pragma: no cover
            continue  # pragma: no cover
        if raw is None or len(raw) == 0:
            logger.warning("[DATA] buyability intraday partition empty date=%s path=%s", snap_str, target)  # pragma: no cover
            continue  # pragma: no cover
        vendor = "ls" if "jdiff_vol" in raw.columns else "kis"
        per_symbol: dict[str, tuple[float, float, float]] = {}
        try:
            if set(CANONICAL_BAR_COLUMNS).issubset(set(raw.columns)):
                frame = raw.copy()
                frame["symbol"] = frame["symbol"].astype(str).str.zfill(6)
                frame["ts_hms"] = pd.to_numeric(frame["ts_hms"], errors="coerce")
                frame["value_krw"] = pd.to_numeric(frame["value_krw"], errors="coerce")
                frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
                for symbol, group in frame.groupby("symbol", sort=False):
                    entry = group[(group["ts_hms"] <= DECISION_WINDOW_END_HMS)]
                    day_vol = float(np.nansum(entry["volume"].to_numpy(dtype=np.float64)))
                    if not np.isfinite(day_vol) or day_vol <= 0.0:
                        continue
                    auc = entry[(entry["ts_hms"] >= int(auction_start_hms))]
                    auc_value = float(np.nansum(auc["value_krw"].to_numpy(dtype=np.float64)))
                    auc_vol = float(np.nansum(auc["volume"].to_numpy(dtype=np.float64)))
                    per_symbol[str(symbol)] = (auc_value, auc_vol, day_vol)
            else:
                # Raw vendor partitions in this repo key the stock code under either
                # the canonical 'symbol' or the raw KIS column '종목코드' -- almost
                # every on-disk partition today (242/243) is unnormalized KIS output
                # carrying only '종목코드'. Checking 'symbol' alone silently pooled
                # every stock in the file into one ungrouped frame.
                raw_symbol_col = "symbol" if "symbol" in raw.columns else ("종목코드" if "종목코드" in raw.columns else None)
                symbols = raw[raw_symbol_col].astype(str).str.zfill(6).unique().tolist() if raw_symbol_col else []
                groups: list[tuple[str, pd.DataFrame]] = []
                if symbols and raw_symbol_col is not None:
                    for symbol in symbols:
                        sub = raw[raw[raw_symbol_col].astype(str).str.zfill(6) == str(symbol)]
                        groups.append((str(symbol), sub))
                else:
                    groups.append(("", raw))
                for symbol, sub in groups:
                    if not symbol:
                        # No resolvable per-row stock code: skip rather than mix
                        # multiple symbols' cumulative volumes into one series.
                        logger.warning("[DATA] buyability raw partition missing symbol column date=%s path=%s", snap_str, target)
                        continue
                    try:
                        norm = normalize_bar_frame(sub, vendor, snap_str, symbol)
                    except Exception as exc:  # pragma: no cover
                        logger.warning("[DATA] buyability normalize failed date=%s symbol=%s: %s", snap_str, symbol, exc)  # pragma: no cover
                        continue  # pragma: no cover
                    entry = norm[norm["ts_hms"] <= DECISION_WINDOW_END_HMS]
                    day_vol = float(entry["volume"].to_numpy(dtype=np.float64).sum())
                    if not np.isfinite(day_vol) or day_vol <= 0.0:
                        continue
                    auc = entry[entry["ts_hms"] >= int(auction_start_hms)]
                    auc_value = float(auc["value_krw"].to_numpy(dtype=np.float64).sum())
                    auc_vol = float(auc["volume"].to_numpy(dtype=np.float64).sum())
                    per_symbol[str(symbol)] = (auc_value, auc_vol, day_vol)
        except Exception as exc:  # pragma: no cover
            logger.warning("[DATA] buyability intraday aggregate failed date=%s path=%s: %s", snap_str, target, exc)  # pragma: no cover
            continue  # pragma: no cover
        mask = snap_dates.to_numpy() == np.datetime64(snap_str) if False else (snap_dates == snap_str).to_numpy()
        for idx in np.flatnonzero(mask):
            key = str(codes.iloc[idx])
            if key not in per_symbol:
                continue
            auc_value, auc_vol, day_vol = per_symbol[key]
            auction_value[idx] = np.float64(auc_value / 1e8)
            auction_share[idx] = np.float64(auc_vol / day_vol)
            bars_found[idx] = True

    out["auction_value_100m"] = np.asarray(auction_value, dtype=np.float64)
    out["auction_vol_share"] = np.asarray(auction_share, dtype=np.float64)
    out["auction_bars_found"] = np.asarray(bars_found, dtype=bool)
    return out


def estimate_fill_ratio(
    auction_value_100m: np.ndarray,
    target_notional_100m: float,
    *,
    participation_cap: float = DEFAULT_PARTICIPATION_CAP,
) -> np.ndarray:
    """Capped participation share of the target notional; NaN stays NaN."""
    target = float(target_notional_100m)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError(f"target_notional_100m must be finite and > 0, got {target_notional_100m!r}")
    cap = float(participation_cap)
    if not np.isfinite(cap) or not 0.0 < cap <= 1.0:
        raise ValueError(f"participation_cap must be in (0.0, 1.0], got {participation_cap!r}")
    auction = np.asarray(auction_value_100m, dtype=np.float64)
    scaled = np.full(auction.shape, np.nan, dtype=np.float64)
    np.multiply(auction, cap, out=scaled, where=~np.isnan(auction))
    ratio = np.full(auction.shape, np.nan, dtype=np.float64)
    np.divide(scaled, target, out=ratio, where=~np.isnan(auction))
    clipped = np.clip(ratio, 0.0, 1.0)
    clipped[np.isnan(ratio)] = np.nan
    return np.asarray(clipped, dtype=np.float64)


def apply_buyability_gate(
    df: pd.DataFrame,
    *,
    target_notional_100m: float,
    min_fill_ratio: float = 1.0,
    participation_cap: float = DEFAULT_PARTICIPATION_CAP,
    require_auction_data: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Annotate (never drop) with ceiling/liquidity/fill flags plus provenance."""
    target = float(target_notional_100m)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError(f"target_notional_100m must be finite and > 0, got {target_notional_100m!r}")
    cap = float(participation_cap)
    if not np.isfinite(cap) or not 0.0 < cap <= 1.0:
        raise ValueError(f"participation_cap must be in (0.0, 1.0], got {participation_cap!r}")
    out = df.copy()
    has_auction = {"auction_value_100m", "auction_vol_share", "auction_bars_found"}.issubset(set(out.columns))
    if not has_auction:
        date_col = "trade_date" if "trade_date" in out.columns else None
        code_col = "stock_code" if "stock_code" in out.columns else None
        if date_col is None or code_col is None:
            raise ValueError("df must contain trade_date/stock_code or pre-attached auction columns")
        attached = attach_entry_auction_liquidity(out, date_col=date_col, code_col=code_col)
        for col in ("auction_value_100m", "auction_vol_share", "auction_bars_found"):
            out[col] = attached[col].to_numpy() if col != "auction_bars_found" else attached[col].to_numpy(dtype=bool)
        out["auction_value_100m"] = out["auction_value_100m"].to_numpy(dtype=np.float64)
        out["auction_vol_share"] = out["auction_vol_share"].to_numpy(dtype=np.float64)
        out["auction_bars_found"] = out["auction_bars_found"].to_numpy(dtype=bool)
    else:
        out["auction_value_100m"] = pd.to_numeric(out["auction_value_100m"], errors="coerce").to_numpy(dtype=np.float64)
        out["auction_vol_share"] = pd.to_numeric(out["auction_vol_share"], errors="coerce").to_numpy(dtype=np.float64)
        out["auction_bars_found"] = out["auction_bars_found"].fillna(False).to_numpy(dtype=bool)
    is_ceiling = classify_ceiling_entry(out)
    out["is_ceiling_entry"] = is_ceiling.to_numpy(dtype=bool)
    expected = estimate_fill_ratio(out["auction_value_100m"].to_numpy(dtype=np.float64), target, participation_cap=cap)
    out["expected_fill_ratio"] = np.asarray(expected, dtype=np.float64)
    found = out["auction_bars_found"].to_numpy(dtype=bool)
    buyable = np.asarray(expected >= float(min_fill_ratio))
    buyable[~found] = bool(not require_auction_data)
    # NaN expected with found=True stays False (measured unfillable).
    buyable = np.asarray(buyable, dtype=bool)
    out["is_buyable"] = buyable
    n_rows = len(out)
    n_ceiling = int(np.sum(out["is_ceiling_entry"].to_numpy(dtype=bool)))
    n_unmeasured = int(np.sum(~found))
    n_blocked = int(np.sum(~buyable))
    blocked_vals = out.loc[~buyable, "auction_value_100m"].to_numpy(dtype=np.float64)
    blocked_measured = blocked_vals[np.isfinite(blocked_vals)]
    if blocked_measured.size:
        med = float(np.median(blocked_measured))
        p25 = float(np.quantile(blocked_measured, 0.25))
    else:
        med = float("nan")
        p25 = float("nan")
    provenance: dict[str, Any] = {
        "n_rows": n_rows,
        "n_ceiling": n_ceiling,
        "n_unmeasured": n_unmeasured,
        "n_blocked": n_blocked,
        "blocked_median_auction_value_100m": med,
        "blocked_p25_auction_value_100m": p25,
        "median_blocked_auction_value_100m": med,
        "p25_blocked_auction_value_100m": p25,
    }
    return out, provenance
