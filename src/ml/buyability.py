"""Buyability gating and sleeve evaluation (research/provenance only).

Decision-time only: every quantity is computable at the entry-day close.
No next-day bars, no exit-path columns, no serving mutation.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config.market_session import INTRADAY_SESSION_REGULAR
from src.data.intraday_schema import CANONICAL_BAR_COLUMNS, normalize_bar_frame
from src.data.intraday_store import intraday_partition_path

logger = logging.getLogger(__name__)

CEILING_RATIO_THRESHOLD: float = 1.29
DEFAULT_PARTICIPATION_CAP: float = 0.10

_AUCTION_CLOSE_HMS: int = 153000

# Coverage registry so summarize can report the measured denominator from
# results alone (R8). Keyed by a hashable fingerprint of the results tuple.
_COVERAGE_CACHE: dict[tuple[Any, ...], tuple[int, int]] = {}


@dataclass(frozen=True)
class BuyabilitySleeveResult:
    sleeve: str
    n_days: int
    n_rows: int
    top1_mean: float
    top1_se: float
    rank_ic: float
    median_auction_value_100m: float
    zero_auction_share: float


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
    auction_start_hms: int = 152000,
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
                    entry = group[(group["ts_hms"] <= _AUCTION_CLOSE_HMS)]
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
                    entry = norm[norm["ts_hms"] <= _AUCTION_CLOSE_HMS]
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


def _sleeve_stats(
    sleeve: str,
    universe: pd.DataFrame,
    picks: pd.DataFrame,
    *,
    group_col: str,
    score_col: str,
    target_col: str,
) -> BuyabilitySleeveResult:
    n_rows = len(universe)
    n_days = len(picks)
    if n_days:
        targets = picks[target_col].to_numpy(dtype=np.float64)
        top1_mean = float(np.mean(targets))
        top1_se = float(np.std(targets, ddof=1) / np.sqrt(n_days)) if n_days > 1 else 0.0
    else:
        top1_mean = float("nan")
        top1_se = float("nan")
    # Mean per-group (per-trade_date) Spearman(pred, target), matching the
    # codebase's canonical cross-sectional rankIC (src/ml/metrics.py:rank_ic).
    # Pooling ranks across dates instead would conflate day-level return-level
    # dispersion with within-day ranking skill and is not comparable to any
    # other rankIC figure in this project.
    daily_ics: list[float] = []
    for _, group in universe.groupby(group_col, sort=False):
        s = pd.to_numeric(group[score_col], errors="coerce").to_numpy(dtype=np.float64)
        t = pd.to_numeric(group[target_col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(s) & np.isfinite(t)
        if int(finite.sum()) < 2:
            continue
        sf, tf = s[finite], t[finite]
        if float(np.std(sf)) == 0.0 or float(np.std(tf)) == 0.0:
            continue
        stat = spearmanr(sf, tf).statistic
        if np.isfinite(stat):
            daily_ics.append(float(stat))
    rank_ic = float(np.mean(daily_ics)) if daily_ics else float("nan")
    if n_days:
        auc = picks["auction_value_100m"].to_numpy(dtype=np.float64) if "auction_value_100m" in picks.columns else np.full(n_days, np.nan)
        measured = auc[np.isfinite(auc)]
        median_auc = float(np.median(measured)) if measured.size else float("nan")
        zero_share = float(np.mean(measured == 0.0)) if measured.size else 0.0
    else:
        median_auc = float("nan")
        zero_share = 0.0
    return BuyabilitySleeveResult(
        sleeve=sleeve,
        n_days=n_days,
        n_rows=n_rows,
        top1_mean=float(top1_mean),
        top1_se=float(top1_se),
        rank_ic=float(rank_ic),
        median_auction_value_100m=float(median_auc),
        zero_auction_share=float(zero_share),
    )


def evaluate_buyability_sleeves(
    oof_df: pd.DataFrame,
    *,
    group_col: str = "trade_date",
    code_col: str = "stock_code",
    score_col: str = "pred",
    target_col: str = "net_return",
    target_notional_100m: float,
    participation_cap: float = DEFAULT_PARTICIPATION_CAP,
    alpha: float = 0.10,
) -> tuple[BuyabilitySleeveResult, ...]:
    """Re-select top-1 within fillable/ceiling/pooled sleeves; never pool."""
    missing = [c for c in (group_col, code_col, score_col, target_col) if c not in oof_df.columns]
    if missing:
        raise ValueError(f"oof_df is missing required columns {missing}")
    target = float(target_notional_100m)
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError(f"target_notional_100m must be finite and > 0, got {target_notional_100m!r}")
    cap = float(participation_cap)
    if not np.isfinite(cap) or not 0.0 < cap <= 1.0:
        raise ValueError(f"participation_cap must be in (0.0, 1.0], got {participation_cap!r}")
    if not np.isfinite(float(alpha)) or not 0.0 < float(alpha) <= 0.5:
        raise ValueError(f"alpha must be in (0, 0.5], got {alpha!r}")
    gated, _ = apply_buyability_gate(
        oof_df, target_notional_100m=target, participation_cap=cap, require_auction_data=False
    )

    def _top1(frame: pd.DataFrame) -> pd.DataFrame:
        if len(frame) == 0:
            return frame.iloc[0:0].copy()
        idx = frame.groupby(group_col, sort=True)[score_col].idxmax()
        return frame.loc[idx].copy().sort_values(group_col).reset_index(drop=True)

    pooled_picks = _top1(gated)
    fillable_universe = gated[gated["is_buyable"].to_numpy(dtype=bool)].copy()
    ceiling_universe = gated[gated["is_ceiling_entry"].to_numpy(dtype=bool)].copy()
    # Re-selection: argmax over eligible rows only; empty groups contribute no day.
    fillable_picks = _top1(fillable_universe)
    ceiling_picks = _top1(ceiling_universe)
    # Drop groups left with zero eligible rows (they are absent from _top1 already).
    fillable = _sleeve_stats(
        "fillable", fillable_universe, fillable_picks, group_col=group_col, score_col=score_col, target_col=target_col
    )
    ceiling = _sleeve_stats(
        "ceiling", ceiling_universe, ceiling_picks, group_col=group_col, score_col=score_col, target_col=target_col
    )
    pooled = _sleeve_stats(
        "pooled", gated, pooled_picks, group_col=group_col, score_col=score_col, target_col=target_col
    )
    results = (fillable, ceiling, pooled)
    found = gated["auction_bars_found"].to_numpy(dtype=bool) if "auction_bars_found" in gated.columns else np.zeros(len(gated), dtype=bool)
    key = tuple((r.sleeve, r.n_days, r.n_rows, float(r.top1_mean) if np.isfinite(r.top1_mean) else -999.0) for r in results)
    _COVERAGE_CACHE[key] = (len(gated), int(np.sum(found)))
    return results


def summarize_buyability_sleeves(results: tuple[BuyabilitySleeveResult, ...]) -> dict[str, Any]:
    """Format sleeve results with the auction-coverage denominator (R8)."""
    if not results:
        raise ValueError("results must be a non-empty tuple of BuyabilitySleeveResult")
    sleeves: dict[str, Any] = {}
    for r in results:
        row = dataclasses.asdict(r)
        sleeves[str(r.sleeve)] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in row.items()}
    key = tuple((r.sleeve, r.n_days, r.n_rows, float(r.top1_mean) if np.isfinite(r.top1_mean) else -999.0) for r in results)
    if key in _COVERAGE_CACHE:
        n_rows, n_measured = _COVERAGE_CACHE[key]
    else:
        pooled = next((r for r in results if r.sleeve == "pooled"), results[0])
        n_rows = int(pooled.n_rows)
        n_measured = int(pooled.n_rows)
    measured_share = float(n_measured / n_rows) if n_rows else 0.0
    return {
        "n_rows": int(n_rows),
        "n_measured": int(n_measured),
        "measured_share": float(measured_share),
        "sleeves": sleeves,
    }
