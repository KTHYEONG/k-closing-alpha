"""Decision-time (15:20) daily panel reconstructed from 1m bars and live decision inputs.

Training historically screened and derived features from final EOD values while serving decides at
15:20 on the live capture; the 15:20->15:30 closing auction moves price and trade value between the two.
This panel restores the serving information set for past days: each row holds the state observable at
the decision cutoff (DECISION_WINDOW_START_HHMMSS), never the auction outcome, with T-1 confirmed values
for quantities that are only published after the close. Rows that cannot be reconstructed are excluded
with a reason, never imputed. Index columns are the single declared exception (EOD fallback, flagged).
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config.market_session import (
    BAR_STAMP_END,
    BAR_STAMP_START,
    DECISION_WINDOW_END_HHMMSS,
    DECISION_WINDOW_START_HHMMSS,
    DEFAULT_BAR_INTERVAL_MINUTES,
    INTRADAY_BAR_STAMP_CONVENTION,
    INTRADAY_SESSION_REGULAR,
    INTRADAY_SESSION_REGULAR_CONSOLIDATED,
    KRX_REGULAR_HOUR_FLOOR,
)
from src.data.capture_contracts import CaptureStatus
from src.data.capture_store import CaptureStore
from src.data.eod_superset import EodSupersetScreen, eod_superset_mask
from src.data.nxt_decomposition import (
    BASIS_KRX_RECONSTRUCTED,
    DECOMPOSITION_CONFIG_FILENAME,
    DecompositionConfig,
    load_decomposition_config,
    predict_share,
    reconstruct_krx_bars,
    share_proxy_history,
)
from src.strategy.contract import SCREENABLE_CLASS_COL

logger = logging.getLogger(__name__)

PIT1520_PANEL_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "market",
    "open",
    "high",
    "low",
    "close",
    "close_raw",
    "prev_close",
    "volume",
    "trade_value_100m",
    "market_cap_100m",
    "inst_netbuy",
    "foreign_netbuy",
    "inst_netbuy_prev",
    "foreign_netbuy_prev",
    "kospi_pct",
    "kosdaq_pct",
    "v_kospi",
    "v_kosdaq",
    "index_basis",
    "source",
    "bars_vendor",
    "n_bars",
    "first_bar_hms",
    "last_bar_hms",
    "capture_run_id",
)

PANEL_FLOAT_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "close_raw",
    "prev_close",
    "volume",
    "trade_value_100m",
    "market_cap_100m",
    "inst_netbuy",
    "foreign_netbuy",
    "inst_netbuy_prev",
    "foreign_netbuy_prev",
    "kospi_pct",
    "kosdaq_pct",
    "v_kospi",
    "v_kosdaq",
)

PANEL_STRING_COLUMNS: tuple[str, ...] = (
    "market",
    "index_basis",
    "source",
    "bars_vendor",
    "first_bar_hms",
    "last_bar_hms",
    "capture_run_id",
)

RECON_TOSS_VENDOR: str = "toss_cons"
BASIS_EXACT: str = "exact"
BASIS_RECONSTRUCTED: str = BASIS_KRX_RECONSTRUCTED

PIT1520_RECON_COLUMNS: tuple[str, ...] = (
    *PIT1520_PANEL_COLUMNS,
    "basis",
    "recon_volume_rel_err_p90",
    "recon_close_bp_err_p90",
)

RECON_FLOAT_COLUMNS: tuple[str, ...] = (
    "recon_volume_rel_err_p90",
    "recon_close_bp_err_p90",
)

INDEX_BASIS_LIVE: str = "live_1520"
INDEX_BASIS_EOD_FALLBACK: str = "eod_fallback"

_TOSS_LEDGER_FILENAME: str = "toss_regular.parquet"
# Part 1 basis-gate verdict marking a symbol-day as consolidated tape.
_KNOWN_CONSOLIDATED_REASON: str = "toss_consolidated_tape"

_BAR_REQUIRED_COLUMNS: tuple[str, ...] = (
    "symbol",
    "ts_hms",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "value_krw",
    "has_trade",
    "vendor",
)

_AGGREGATE_COLUMNS: tuple[str, ...] = (
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_value_100m",
    "bars_vendor",
    "n_bars",
    "first_bar_hms",
    "last_bar_hms",
    "head_low",
    "head_high",
)

_LIVE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "종목코드",
    "시가",
    "고가",
    "저가",
    "종가",
    "전일종가",
    "거래량",
    "거래대금",
    "시가총액",
    "시장구분",
    "기관_순매수",
    "외국인_순매수",
    "kospi",
    "kosdaq",
    "v_kospi",
)

_PRICE_HISTORY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "market",
    "open",
    "close",
    "close_raw",
    "prev_close",
    "mc_clean",
    "inst_netbuy",
    "foreign_netbuy",
    "kospi_pct",
    "kosdaq_pct",
    "v_kospi",
    "v_kosdaq",
    "volume",
    "chg_ratio",
    "tv_clean",
)


class PanelSource(enum.StrEnum):
    """Origin of a panel row."""

    LIVE_DECISION = "live_decision"
    BARS = "bars"


class PanelSourcePolicy(enum.StrEnum):
    """Which source a build may use per date."""

    PREFER_LIVE = "prefer_live"
    BARS_ONLY = "bars_only"


class PanelExclusionReason(enum.StrEnum):
    """Why a symbol-day has no panel row."""

    NOT_FETCHED = "not_fetched"
    PRICE_BASIS_ADJUSTED = "price_basis_adjusted"
    NO_BARS_BEFORE_CUTOFF = "no_bars_before_cutoff"
    UNKNOWN_BAR_STAMP = "unknown_bar_stamp"
    MIXED_VENDOR = "mixed_vendor"
    VENDOR_EXCLUDED = "vendor_excluded"
    HEAD_TRUNCATED = "head_truncated"
    VENDOR_MINUTE_GAP = "vendor_minute_gap"
    INVALID_PRICE = "invalid_price"
    NOT_IN_PRICE_HISTORY = "not_in_price_history"
    PREV_CLOSE_UNAVAILABLE = "prev_close_unavailable"
    PREV_MARKET_CAP_UNAVAILABLE = "prev_market_cap_unavailable"
    SHARE_UNAVAILABLE = "share_unavailable"
    CONSOLIDATED_MISSING = "consolidated_missing"


@dataclass(frozen=True)
class Pit1520PanelConfig:
    """Construction parameters of the decision-time panel.

    Attributes:
        cutoff_hhmmss: Decision cutoff; a bar is included only when its interval END is at or before it.
        floor_hhmmss: Regular-session open; a bar is included only when its interval START is at or after it.
        stamp_conventions: Vendor -> bar timestamp convention (start/end).
        max_open_mismatch_ticks: Head-truncation guard tolerance, in KRX ticks, around the price range traded
            in the head window.
        value_reconstructed_vendors: Vendors whose per-bar traded value is not a trustworthy per-minute figure at the source and is rebuilt as volume x typical price ((high+low+close)/3). LS reports it in KRW millions per minute, so day sums run ~15% low at p5 (measured 2026-09: p5 0.997, 97% within 1% after rebuild). Toss exposes no value field at all; the schema's close x volume approximation is biased by the intra-minute range, so it is rebuilt the same way as LS.
        excluded_bar_vendors: Vendors whose bar symbol-days are dropped from the panel with reason `vendor_excluded`. It exists for ablation (panel with and without a vendor) and rollback, not as a quality filter.
        minute_gap_min_symbols: A day is checked for vendor-wide missing minutes only when at least this many
            symbols traded (small samples have legitimately empty minutes).
        minute_gap_min_share: A minute between the open and the cutoff is a vendor gap when fewer symbols traded
            in it than this share of the day's median per-minute count. A gap biases every symbol's cumulative
            volume and value, so the whole bar-derived day is excluded (measured: LS 2026-09-21/28 lack 12:50).
        head_window_minutes: Head-truncation guard window after the session open. A symbol-day is kept only if
            it traded in [open, open + window) and the official open (raw basis) lies within that window's
            low-high range. The range, not the first bar's open, is compared because a vendor may fold a
            pre-open off-hours print (at the previous close) into the first minute of a gap day.
        live_available_by_hhmmss: A live decision input qualifies only if completed at or before T at this time.
        reconstruct_consolidated: Reconstruct KRX-only bars for superset symbol-days with no exact bars but
            with `regular_consolidated` bars (Part 2 estimator). Off by default; the production panel is
            unchanged until the adoption gate passes.
        decomposition_config_path: Fitted decomposition config location (informational; the builder takes
            the loaded config object, the CLI resolves this path through settings).
    """

    cutoff_hhmmss: str = DECISION_WINDOW_START_HHMMSS
    floor_hhmmss: str = KRX_REGULAR_HOUR_FLOOR
    stamp_conventions: Mapping[str, str] = dataclasses.field(
        default_factory=lambda: dict(INTRADAY_BAR_STAMP_CONVENTION)
    )
    max_open_mismatch_ticks: int = 1
    head_window_minutes: int = 5
    value_reconstructed_vendors: frozenset[str] = frozenset({"ls", "toss"})
    excluded_bar_vendors: frozenset[str] = frozenset()
    minute_gap_min_symbols: int = 30
    minute_gap_min_share: float = 0.2
    live_available_by_hhmmss: str = DECISION_WINDOW_END_HHMMSS
    reconstruct_consolidated: bool = False
    decomposition_config_path: str = ""


@dataclass(frozen=True)
class Pit1520PanelResult:
    """Panel rows, per-symbol exclusions and per-day coverage of one build."""

    panel: pd.DataFrame
    exclusions: pd.DataFrame
    days: pd.DataFrame


def default_panel_paths() -> tuple[Path, Path, Path]:
    """Return (panel, exclusions, days) paths under settings.HISTORY_DIR.

    Returns:
        HISTORY_DIR/pit1520_panel.parquet, HISTORY_DIR/pit1520_panel_exclusions.parquet,
        HISTORY_DIR/pit1520_panel_days.parquet.
    """
    from src import settings

    base = Path(settings.HISTORY_DIR)
    return (
        base / "pit1520_panel.parquet",
        base / "pit1520_panel_exclusions.parquet",
        base / "pit1520_panel_days.parquet",
    )


def _hhmmss_to_seconds(hms: int) -> int:
    hms = int(hms)
    return (hms // 10000) * 3600 + ((hms // 100) % 100) * 60 + (hms % 100)


def _seconds_to_hhmmss(seconds: int) -> int:
    seconds = int(seconds)
    return (seconds // 3600) * 10000 + ((seconds % 3600) // 60) * 100 + (seconds % 60)


def _shift_minutes(hms: int, delta_minutes: int) -> int:
    return _seconds_to_hhmmss(_hhmmss_to_seconds(int(hms)) + int(delta_minutes) * 60)


def _coerce_float(values: pd.Series) -> np.ndarray:
    return np.asarray(pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64), dtype=np.float64)


def _empty_aggregates() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in _AGGREGATE_COLUMNS})


def _empty_exclusions() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ("symbol", "reason", "detail")})


def _has_trade_flag(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return np.asarray(values.to_numpy(dtype=bool), dtype=bool)
    if values.dtype.kind in "iufb":
        return np.asarray(
            pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) != 0.0, dtype=bool
        )
    return np.asarray(
        values.astype(str).str.strip().str.lower().isin({"true", "1", "t", "yes"}).to_numpy(dtype=bool), dtype=bool
    )


def vendor_minute_gaps(bars: pd.DataFrame, *, config: Pit1520PanelConfig) -> list[int]:
    """Return decision-window minutes (HHMMSS of the minute start) that the vendor dropped for the whole market.

    Args:
        bars: Canonical bar rows of one snapshot date.
        config: Cutoff, floor, stamp conventions and gap thresholds.

    Returns:
        Ascending gap minutes; empty when fewer than config.minute_gap_min_symbols symbols traded or a bar's
        vendor has no declared stamp convention (those rows are excluded per symbol elsewhere).
    """
    if bars.empty:
        return []
    conventions = dict(config.stamp_conventions)
    vendors = bars["vendor"].astype(str)
    if not vendors.map(lambda v: conventions.get(v) in (BAR_STAMP_START, BAR_STAMP_END)).all():
        return []
    traded = bars.loc[_has_trade_flag(bars["has_trade"])]
    if traded["symbol"].astype(str).nunique() < int(config.minute_gap_min_symbols):
        return []
    stamps = pd.to_numeric(traded["ts_hms"], errors="coerce")
    seconds = (stamps // 10000) * 3600 + ((stamps // 100) % 100) * 60 + (stamps % 100)
    end_stamped = traded["vendor"].astype(str).map(lambda v: conventions[v] == BAR_STAMP_END).to_numpy(dtype=bool)
    start_seconds = seconds.to_numpy(dtype=np.float64) - np.where(end_stamped, 60.0, 0.0)
    floor_s = _hhmmss_to_seconds(int(config.floor_hhmmss))
    cutoff_s = _hhmmss_to_seconds(int(config.cutoff_hhmmss))
    window = (start_seconds >= floor_s) & (start_seconds + 60.0 <= cutoff_s)
    per_minute = (
        pd.DataFrame({"m": start_seconds[window], "s": traded["symbol"].astype(str).to_numpy()[window]})
        .groupby("m")["s"].nunique()
        .reindex(np.arange(floor_s, cutoff_s, 60, dtype=np.float64), fill_value=0)
    )
    median = float(per_minute.median())
    if median <= 0.0:
        return []
    gaps = per_minute.index[per_minute.to_numpy() < float(config.minute_gap_min_share) * median]
    return [_seconds_to_hhmmss(int(m)) for m in gaps]


def aggregate_decision_bars(
    bars: pd.DataFrame, *, config: Pit1520PanelConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate one day's regular 1m bars to the decision cutoff per symbol.

    Bars are vendor-stamped differently (KIS/Kiwoom at the minute start, LS at the minute end), so the
    cutoff is applied to each bar's interval end resolved through config.stamp_conventions; applying the
    raw ts_hms would drop KIS's last continuous minute or admit an auction print.

    Args:
        bars: Canonical bar rows of one snapshot date (CANONICAL_BAR_COLUMNS: symbol, ts_hms, open, high,
            low, close, volume, value_krw, has_trade, vendor).
        config: Cutoff, floor and stamp conventions.

    Returns:
        (aggregates, exclusions). aggregates has one row per symbol with columns symbol, open, high, low,
        close, volume, trade_value_100m, bars_vendor, n_bars, first_bar_hms, last_bar_hms; exclusions has
        columns symbol, reason, detail.

    Raises:
        ValueError: Naming missing required columns.
    """
    missing = [c for c in _BAR_REQUIRED_COLUMNS if c not in bars.columns]
    if missing:
        raise ValueError(f"aggregate_decision_bars missing required columns: {missing}")
    if bars.empty:
        return _empty_aggregates(), _empty_exclusions()
    work = bars[list(_BAR_REQUIRED_COLUMNS)].copy().reset_index(drop=True)
    work["symbol"] = work["symbol"].astype(str)
    ts_all = _coerce_float(work["ts_hms"])
    vendors_all = work["vendor"].astype(str).to_numpy(dtype=object)
    traded_all = _has_trade_flag(work["has_trade"])
    open_all = _coerce_float(work["open"])
    high_all = _coerce_float(work["high"])
    low_all = _coerce_float(work["low"])
    close_all = _coerce_float(work["close"])
    volume_all = _coerce_float(work["volume"].fillna(0.0))
    value_all = _coerce_float(work["value_krw"].fillna(0.0))
    rebuilt = np.isin(vendors_all.astype(str), sorted(config.value_reconstructed_vendors))
    if bool(rebuilt.any()):
        typical = (high_all + low_all + close_all) / 3.0
        value_all = np.where(rebuilt, np.nan_to_num(volume_all * typical, nan=0.0), value_all)
    floor = int(config.floor_hhmmss)
    cutoff = int(config.cutoff_hhmmss)
    head_end = _shift_minutes(floor, int(config.head_window_minutes))
    conventions = dict(config.stamp_conventions)

    rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for key, group in work.groupby("symbol", sort=True):
        symbol = str(key)
        pos = group.index.to_numpy()
        group_vendors = sorted({str(vendors_all[i]) for i in pos})
        if set(group_vendors) & set(config.excluded_bar_vendors):
            excluded.append({
                "symbol": symbol,
                "reason": PanelExclusionReason.VENDOR_EXCLUDED.value,
                "detail": f"vendors={group_vendors}",
            })
            continue
        unknown = [v for v in group_vendors if conventions.get(v) not in (BAR_STAMP_START, BAR_STAMP_END)]
        if unknown:
            excluded.append({
                "symbol": symbol,
                "reason": PanelExclusionReason.UNKNOWN_BAR_STAMP.value,
                "detail": f"vendors={group_vendors}",
            })
            continue
        if len(group_vendors) > 1:
            excluded.append({
                "symbol": symbol,
                "reason": PanelExclusionReason.MIXED_VENDOR.value,
                "detail": f"vendors={group_vendors}",
            })
            continue
        finite = np.flatnonzero(np.isfinite(ts_all[pos]))
        stamps = ts_all[pos][finite].astype(np.int64)
        if conventions[group_vendors[0]] == BAR_STAMP_START:
            starts = stamps
            ends = np.array([_shift_minutes(int(v), 1) for v in stamps], dtype=np.int64)
        else:
            starts = np.array([_shift_minutes(int(v), -1) for v in stamps], dtype=np.int64)
            ends = stamps
        admitted = (starts >= floor) & (ends <= cutoff)
        slots = [j for j in admitted.nonzero()[0] if bool(traded_all[pos[finite[j]]])]
        use = np.zeros(len(pos), dtype=bool)
        for slot in slots:
            use[finite[slot]] = True
        n_included = int(admitted.sum())
        if not bool(use.any()):
            excluded.append({
                "symbol": symbol,
                "reason": PanelExclusionReason.NO_BARS_BEFORE_CUTOFF.value,
                "detail": f"n_included={n_included}",
            })
            continue
        traded_pos = pos[use]
        start_of = dict(zip(finite.tolist(), starts.tolist(), strict=True))
        ordered = traded_pos[np.argsort([start_of[int(k)] for k in np.flatnonzero(use)], kind="stable")]
        opens = open_all[ordered]
        closes = close_all[ordered]
        highs = high_all[ordered]
        lows = low_all[ordered]
        if not (
            np.isfinite(opens[0])
            and np.isfinite(closes[-1])
            and bool(np.isfinite(highs).all())
            and bool(np.isfinite(lows).all())
            and opens[0] > 0.0
            and closes[-1] > 0.0
            and highs.max() > 0.0
            and lows.min() > 0.0
        ):
            excluded.append({
                "symbol": symbol,
                "reason": PanelExclusionReason.INVALID_PRICE.value,
                "detail": f"open={opens[0]!r} close={closes[-1]!r}",
            })
            continue
        in_window = np.zeros(len(work), dtype=bool)
        for slot in admitted.nonzero()[0]:
            in_window[pos[finite[slot]]] = True
        ordered_starts = np.asarray([start_of[int(k)] for k in np.flatnonzero(use)], dtype=np.int64)
        ordered_starts.sort(kind="stable")
        head = ordered_starts < head_end
        head_low = float(lows[head].min()) if bool(head.any()) else float("nan")
        head_high = float(highs[head].max()) if bool(head.any()) else float("nan")
        rows.append({
            "symbol": symbol,
            "open": float(opens[0]),
            "high": float(highs.max()),
            "low": float(lows.min()),
            "close": float(closes[-1]),
            "volume": float(volume_all[in_window].sum()),
            "trade_value_100m": float(value_all[in_window].sum()) / 1e8,
            "bars_vendor": group_vendors[0],
            "n_bars": int(use.sum()),
            "first_bar_hms": str(int(ts_all[ordered[0]])),
            "last_bar_hms": str(int(ts_all[ordered[-1]])),
            "head_low": head_low,
            "head_high": head_high,
        })
    gap_bars = bars[~bars["vendor"].astype(str).isin(config.excluded_bar_vendors)] if config.excluded_bar_vendors else bars
    gaps = vendor_minute_gaps(gap_bars, config=config)
    if gaps:
        detail = f"vendor gap minutes={gaps[:5]} n={len(gaps)}"
        excluded.extend(
            {"symbol": str(r["symbol"]), "reason": PanelExclusionReason.VENDOR_MINUTE_GAP.value, "detail": detail}
            for r in rows
        )
        rows = []
    aggregates = pd.DataFrame(rows, columns=list(_AGGREGATE_COLUMNS)) if rows else _empty_aggregates()
    exclusions = pd.DataFrame(excluded, columns=["symbol", "reason", "detail"]) if excluded else _empty_exclusions()
    return aggregates, exclusions


def live_input_to_panel_rows(frame: pd.DataFrame, decision_date: pd.Timestamp, *, run_id: str) -> pd.DataFrame:
    """Map a verified live decision input (Korean columns, percent-unit indices) onto panel columns.

    Args:
        frame: Output of CaptureStore.read_decision for decision_date.
        decision_date: Trading date T.
        run_id: Capture run identity stamped into capture_run_id.

    Returns:
        PIT1520_PANEL_COLUMNS frame, source "live_decision", index_basis "live_1520".

    Raises:
        ValueError: Naming missing required Korean columns or duplicate 종목코드.
    """
    missing = [c for c in _LIVE_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"live_input_to_panel_rows missing required columns: {missing}")
    day = pd.Timestamp(decision_date).normalize()
    symbols = frame["종목코드"].astype(str).str.zfill(6)
    dup = symbols[symbols.duplicated(keep=False)].unique().tolist()
    if dup:
        raise ValueError(f"live_input_to_panel_rows carries duplicate 종목코드: {sorted(dup)[:5]}")
    n = len(frame)
    v_kosdaq = (
        _coerce_float(frame["v_kosdaq"]) if "v_kosdaq" in frame.columns else np.full(n, np.nan, dtype=np.float64)
    )
    out = pd.DataFrame({
        "date": pd.to_datetime([day] * n),
        "symbol": symbols.to_numpy(dtype=object),
        "market": frame["시장구분"].astype(str).to_numpy(dtype=object),
        "open": _coerce_float(frame["시가"]),
        "high": _coerce_float(frame["고가"]),
        "low": _coerce_float(frame["저가"]),
        "close": _coerce_float(frame["종가"]),
        "close_raw": _coerce_float(frame["종가"]),
        "prev_close": _coerce_float(frame["전일종가"]),
        "volume": _coerce_float(frame["거래량"]),
        "trade_value_100m": _coerce_float(frame["거래대금"]),
        "market_cap_100m": _coerce_float(frame["시가총액"]),
        "inst_netbuy": np.full(n, np.nan, dtype=np.float64),
        "foreign_netbuy": np.full(n, np.nan, dtype=np.float64),
        "inst_netbuy_prev": np.full(n, np.nan, dtype=np.float64),
        "foreign_netbuy_prev": np.full(n, np.nan, dtype=np.float64),
        "kospi_pct": _coerce_float(frame["kospi"]) / 100.0,
        "kosdaq_pct": _coerce_float(frame["kosdaq"]) / 100.0,
        "v_kospi": _coerce_float(frame["v_kospi"]),
        "v_kosdaq": np.asarray(v_kosdaq, dtype=np.float64),
        "index_basis": np.full(n, INDEX_BASIS_LIVE, dtype=object),
        "source": np.full(n, PanelSource.LIVE_DECISION.value, dtype=object),
        "bars_vendor": np.full(n, "", dtype=object),
        "n_bars": np.zeros(n, dtype=np.int64),
        "first_bar_hms": np.full(n, "", dtype=object),
        "last_bar_hms": np.full(n, "", dtype=object),
        "capture_run_id": np.full(n, str(run_id), dtype=object),
    })
    return out[list(PIT1520_PANEL_COLUMNS)]


def panel_to_decision_input(panel_day: pd.DataFrame) -> pd.DataFrame:
    """Render one date of panel rows as the Korean-column snapshot the serving builder consumes.

    This adapter is the single bridge from the panel to build_topk_ranker_features /
    build_screen_frame / rank_pool_mask, so research scoring runs the exact serving feature code.

    Args:
        panel_day: Panel rows of exactly one date.

    Returns:
        Frame with 종목코드, 시장구분, 시가, 고가, 저가, 종가, 전일종가, 거래량, 거래대금, 시가총액,
        기관_순매수, 외국인_순매수, kospi, kosdaq, v_kospi (percent units for kospi/kosdaq), plus
        is_screenable when panel_day carries it (the class screen of a class-filtering UniverseSpec reads it),
        row order preserved.

    Raises:
        ValueError: When panel_day spans more than one date or misses a panel column.
    """
    missing = [c for c in PIT1520_PANEL_COLUMNS if c not in panel_day.columns]
    if missing:
        raise ValueError(f"panel_to_decision_input missing required columns: {missing}")
    dates = pd.to_datetime(panel_day["date"], errors="coerce").dt.normalize().unique()
    if len(dates) != 1:
        raise ValueError(f"panel_to_decision_input requires exactly one date, got {len(dates)}")
    snapshot = pd.DataFrame({
        "종목코드": panel_day["symbol"].astype(str).to_numpy(dtype=object),
        "시장구분": panel_day["market"].astype(str).to_numpy(dtype=object),
        "시가": _coerce_float(panel_day["open"]),
        "고가": _coerce_float(panel_day["high"]),
        "저가": _coerce_float(panel_day["low"]),
        "종가": _coerce_float(panel_day["close"]),
        "전일종가": _coerce_float(panel_day["prev_close"]),
        "거래량": _coerce_float(panel_day["volume"]),
        "거래대금": _coerce_float(panel_day["trade_value_100m"]),
        "시가총액": _coerce_float(panel_day["market_cap_100m"]),
        "기관_순매수": _coerce_float(panel_day["inst_netbuy"]),
        "외국인_순매수": _coerce_float(panel_day["foreign_netbuy"]),
        "kospi": _coerce_float(panel_day["kospi_pct"]) * 100.0,
        "kosdaq": _coerce_float(panel_day["kosdaq_pct"]) * 100.0,
        "v_kospi": _coerce_float(panel_day["v_kospi"]),
    })
    if SCREENABLE_CLASS_COL in panel_day.columns:
        snapshot[SCREENABLE_CLASS_COL] = panel_day[SCREENABLE_CLASS_COL].fillna(False).astype(bool).to_numpy()
    return snapshot.reset_index(drop=True)


def _empty_panel(*, columns: Sequence[str] | None = None) -> pd.DataFrame:
    cols = list(columns) if columns is not None else list(PIT1520_PANEL_COLUMNS)
    out = pd.DataFrame({c: pd.Series(dtype="object") for c in cols})
    return _cast_panel_dtypes(out, columns=cols)


def _empty_day_exclusions() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in ("date", "symbol", "reason", "detail")})


def _empty_days() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.Series(dtype="datetime64[ns]"),
        "source": pd.Series(dtype="object"),
        "n_rows": pd.Series(dtype="int64"),
        "n_excluded": pd.Series(dtype="int64"),
        "n_superset": pd.Series(dtype="int64"),
        "n_superset_present": pd.Series(dtype="int64"),
        "superset_coverage": pd.Series(dtype="float64"),
        "index_basis": pd.Series(dtype="object"),
    })


def _cast_panel_dtypes(panel: pd.DataFrame, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    out = panel.copy()
    cols = list(columns) if columns is not None else list(PIT1520_PANEL_COLUMNS)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").astype("datetime64[ns]")
    out["symbol"] = out["symbol"].astype(str)
    for col in PANEL_FLOAT_COLUMNS:
        out[col] = _coerce_float(out[col])
    for col in PANEL_STRING_COLUMNS:
        out[col] = out[col].astype(str)
    if "basis" in out.columns:
        out["basis"] = out["basis"].astype(str)
    for col in RECON_FLOAT_COLUMNS:
        if col in out.columns:
            out[col] = _coerce_float(out[col])
    out["n_bars"] = pd.to_numeric(out["n_bars"], errors="coerce").fillna(0).astype(np.int64)
    return out[cols]


def _prepare_price_history_work(price_history: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _PRICE_HISTORY_REQUIRED_COLUMNS if c not in price_history.columns]
    if missing:
        raise ValueError(f"build_pit1520_panel price_history missing required columns: {missing}")
    work = price_history[list(_PRICE_HISTORY_REQUIRED_COLUMNS)].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce", format="mixed").dt.normalize()
    if bool(work["date"].isna().any()):
        raise ValueError("build_pit1520_panel price_history carries unparseable dates")
    work["symbol"] = work["symbol"].astype(str).str.zfill(6)
    dup = int(work.duplicated(["date", "symbol"]).sum())
    if dup:
        raise ValueError(f"build_pit1520_panel price_history carries {dup} duplicate (date, symbol) rows")
    return work


def _superset_symbols(day_ph: pd.DataFrame, screen: EodSupersetScreen) -> set[str]:
    mask = np.asarray(eod_superset_mask(day_ph, screen), dtype=bool)
    return set(day_ph.loc[mask, "symbol"].astype(str).tolist())


def _build_bar_day_rows(
    day_label: str,
    day: pd.Timestamp,
    aggregates: pd.DataFrame,
    day_ph: pd.DataFrame,
    prev_ph: pd.DataFrame | None,
    *,
    config: Pit1520PanelConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    from src.execution.cost_model import krx_tick_size_asof

    rows: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []

    def _drop(symbol: str, reason: PanelExclusionReason, detail: str) -> None:
        excluded.append({"date": day, "symbol": symbol, "reason": reason.value, "detail": detail})

    if aggregates.empty:
        return rows, excluded
    agg = aggregates.copy()
    agg["symbol"] = agg["symbol"].astype(str)
    assert not bool(agg["symbol"].duplicated().any()), f"duplicate aggregate symbols on {day_label}"
    ts = day_ph.set_index("symbol")
    prev = prev_ph.set_index("symbol") if prev_ph is not None and len(prev_ph) else None
    day_np = np.asarray([day.to_datetime64()], dtype="datetime64[ns]")
    for record in agg.to_dict(orient="records"):
        symbol = str(record["symbol"])
        if symbol not in ts.index:
            _drop(symbol, PanelExclusionReason.NOT_IN_PRICE_HISTORY, "absent from price_history on T")
            continue
        trow = ts.loc[symbol]
        close_ph = float(_coerce_float(pd.Series([trow["close"]]))[0])
        close_raw_ph = float(_coerce_float(pd.Series([trow["close_raw"]]))[0])
        prev_close_ph = float(_coerce_float(pd.Series([trow["prev_close"]]))[0])
        factor = (
            close_raw_ph / close_ph
            if np.isfinite(close_raw_ph) and np.isfinite(close_ph) and close_ph != 0.0
            else np.nan
        )
        prev_close = prev_close_ph * factor if np.isfinite(prev_close_ph) and np.isfinite(factor) else np.nan
        if not np.isfinite(prev_close) or prev_close <= 0.0:
            _drop(symbol, PanelExclusionReason.PREV_CLOSE_UNAVAILABLE, f"prev_close={prev_close!r}")
            continue
        open_ph = float(_coerce_float(pd.Series([trow["open"]]))[0])
        expected_open = open_ph * factor if np.isfinite(open_ph) and np.isfinite(factor) else np.nan
        market = str(trow["market"])
        head_low = float(_coerce_float(pd.Series([record.get("head_low")]))[0])
        head_high = float(_coerce_float(pd.Series([record.get("head_high")]))[0])
        if not (np.isfinite(head_low) and np.isfinite(head_high)):
            _drop(symbol, PanelExclusionReason.HEAD_TRUNCATED, f"no traded bar in head window first_bar={record['first_bar_hms']}")
            continue
        if not np.isfinite(expected_open) or expected_open <= 0.0:
            _drop(symbol, PanelExclusionReason.HEAD_TRUNCATED, f"official open unavailable expected_open={expected_open!r}")
            continue
        tick = float(
            krx_tick_size_asof(
                np.asarray([expected_open], dtype=np.float64),
                day_np,
                np.asarray([market], dtype=object),
            )[0]
        )
        tolerance = float(config.max_open_mismatch_ticks) * tick if np.isfinite(tick) and tick > 0.0 else 0.0
        if not (head_low - tolerance <= expected_open <= head_high + tolerance):
            _drop(
                symbol,
                PanelExclusionReason.HEAD_TRUNCATED,
                f"expected_open={expected_open!r} outside head range [{head_low!r}, {head_high!r}] tick={tick!r}",
            )
            continue
        close = float(record["close"])
        mc_prev = float(_coerce_float(pd.Series([prev.loc[symbol]["mc_clean"]]))[0]) if prev is not None and symbol in prev.index else np.nan
        if not np.isfinite(mc_prev):
            _drop(symbol, PanelExclusionReason.PREV_MARKET_CAP_UNAVAILABLE, "mc_clean(P) unavailable")
            continue
        inst_prev = float(_coerce_float(pd.Series([prev.loc[symbol]["inst_netbuy"]]))[0]) if prev is not None and symbol in prev.index else np.nan
        foreign_prev = float(_coerce_float(pd.Series([prev.loc[symbol]["foreign_netbuy"]]))[0]) if prev is not None and symbol in prev.index else np.nan
        rows.append({
            "date": day,
            "symbol": symbol,
            "market": market,
            # Official open (raw basis): fixed at 09:00 so observable at the cutoff, and identical in meaning to the
            # live input's 시가; the first bar's open can carry a folded pre-open print on gap days.
            "open": float(expected_open),
            "high": float(record["high"]),
            "low": float(record["low"]),
            "close": close,
            "close_raw": close,
            "prev_close": float(prev_close),
            "volume": float(record["volume"]),
            "trade_value_100m": float(record["trade_value_100m"]),
            "market_cap_100m": float(mc_prev * close / prev_close),
            "inst_netbuy": np.nan,
            "foreign_netbuy": np.nan,
            "inst_netbuy_prev": float(inst_prev),
            "foreign_netbuy_prev": float(foreign_prev),
            "kospi_pct": float(_coerce_float(pd.Series([trow["kospi_pct"]]))[0]),
            "kosdaq_pct": float(_coerce_float(pd.Series([trow["kosdaq_pct"]]))[0]),
            "v_kospi": float(_coerce_float(pd.Series([trow["v_kospi"]]))[0]),
            "v_kosdaq": float(_coerce_float(pd.Series([trow["v_kosdaq"]]))[0]),
            "index_basis": INDEX_BASIS_EOD_FALLBACK,
            "source": PanelSource.BARS.value,
            "bars_vendor": str(record["bars_vendor"]),
            "n_bars": int(record["n_bars"]),
            "first_bar_hms": str(record["first_bar_hms"]),
            "last_bar_hms": str(record["last_bar_hms"]),
            "capture_run_id": "",
        })
    return rows, excluded


def build_pit1520_panel(
    *,
    price_history: pd.DataFrame,
    dates: Sequence[str],
    bars_loader: Callable[[str], pd.DataFrame],
    live_loader: Callable[[str], tuple[pd.DataFrame, str] | None],
    screen: EodSupersetScreen,
    config: Pit1520PanelConfig = Pit1520PanelConfig(),  # noqa: B008
    source_policy: PanelSourcePolicy = PanelSourcePolicy.PREFER_LIVE,
    consolidated_loader: Callable[[str], pd.DataFrame] | None = None,
    decomposition_config: DecompositionConfig | None = None,
    consolidated_symbol_days: frozenset[tuple[str, str]] | None = None,
) -> Pit1520PanelResult:
    """Build the decision-time panel for the given dates.

    Args:
        price_history: prepare_price_panel output covering every requested date and its previous trading
            date (columns date, symbol, market, open, close, close_raw, prev_close, mc_clean, inst_netbuy,
            foreign_netbuy, kospi_pct, kosdaq_pct, v_kospi, v_kosdaq, chg_ratio, tv_clean, volume).
        dates: Trading dates (YYYY-MM-DD) to build; each must be a price_history date.
        bars_loader: Returns one date's regular 1m bars (empty frame when no partition exists).
        live_loader: Returns (verified live input, run_id) for a date, or None when none qualifies.
        screen: EOD fetch superset used for day coverage and NOT_FETCHED attribution only.
        config: Construction parameters.
        source_policy: PREFER_LIVE (production) or BARS_ONLY (validation of the bar path on live days).
        consolidated_loader: Returns one date's consolidated-tape 1m bars; required only when
            config.reconstruct_consolidated is set.
        decomposition_config: Fitted Part 2 estimator; required only when config.reconstruct_consolidated
            is set. With the flag off the panel carries PIT1520_PANEL_COLUMNS exactly; with the flag on
            it carries PIT1520_RECON_COLUMNS (basis provenance plus calibrated error quantiles).
        consolidated_symbol_days: (date, symbol) pairs the Part 1 ledger verdicts as consolidated tape;
            only those may attribute `consolidated_missing`.

    Returns:
        Panel (sorted by date then symbol, unique (date, symbol)), exclusions
        (date, symbol, reason, detail) and days (date, source, n_rows, n_excluded, n_superset,
        n_superset_present, superset_coverage, index_basis).

    Raises:
        ValueError: A requested date absent from price_history, duplicate (date, symbol) in inputs, or
            reconstruction enabled without its loader and fitted config.
    """
    work = _prepare_price_history_work(price_history)
    calendar = sorted(work["date"].dropna().unique().tolist())
    available = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in calendar}
    ordered_days = sorted({str(d) for d in dates})
    for day_label in ordered_days:
        if day_label not in available:
            raise ValueError(f"build_pit1520_panel date absent from price_history: {day_label!r}")
    recon_on = bool(config.reconstruct_consolidated)
    decomp = decomposition_config
    cons_loader = consolidated_loader
    if recon_on and decomp is None:
        raise ValueError("build_pit1520_panel reconstruct_consolidated requires decomposition_config")
    if recon_on and cons_loader is None:
        raise ValueError("build_pit1520_panel reconstruct_consolidated requires consolidated_loader")
    known_consolidated = consolidated_symbol_days if consolidated_symbol_days is not None else frozenset()
    cons_cache: dict[str, pd.DataFrame] = {}
    out_columns = list(PIT1520_RECON_COLUMNS) if recon_on else list(PIT1520_PANEL_COLUMNS)

    panel_frames: list[pd.DataFrame] = []
    exclusion_frames: list[pd.DataFrame] = []
    day_rows: list[dict[str, object]] = []
    total = len(ordered_days)
    for done, day_label in enumerate(ordered_days, start=1):
        day = pd.Timestamp(day_label).normalize()
        day_ph = work[work["date"] == day].copy()
        past = [d for d in calendar if d < day]
        prev_ph = work[work["date"] == max(past)].copy() if past else None
        superset = _superset_symbols(day_ph, screen)
        live = live_loader(day_label) if source_policy == PanelSourcePolicy.PREFER_LIVE else None
        if live is not None:
            live_frame, run_id = live
            day_panel = live_input_to_panel_rows(live_frame, day, run_id=run_id)
            if prev_ph is not None and len(prev_ph):
                prev_indexed = prev_ph.set_index("symbol")
                inst_map = pd.to_numeric(prev_indexed["inst_netbuy"], errors="coerce")
                foreign_map = pd.to_numeric(prev_indexed["foreign_netbuy"], errors="coerce")
                day_panel["inst_netbuy_prev"] = day_panel["symbol"].map(inst_map).astype(np.float64)
                day_panel["foreign_netbuy_prev"] = day_panel["symbol"].map(foreign_map).astype(np.float64)
            day_panel = _cast_panel_dtypes(day_panel)
            if recon_on:
                assert decomp is not None
                day_panel = _cast_panel_dtypes(
                    _stamp_recon_basis(day_panel, recon_symbols=set(), decomposition_config=decomp),
                    columns=out_columns,
                )
            day_exclusions = _empty_day_exclusions()
            source = PanelSource.LIVE_DECISION.value
            index_basis = INDEX_BASIS_LIVE
        else:
            bars = bars_loader(day_label)
            aggregates, bar_exclusions = aggregate_decision_bars(bars, config=config)
            recon_symbols: set[str] = set()
            if recon_on:
                assert decomp is not None
                assert cons_loader is not None
                exact_symbols = set(bars["symbol"].astype(str).tolist()) if len(bars) else set()
                recon_aggregates, recon_bar_exclusions, recon_symbols, recon_extra = _reconstruct_consolidated_day(
                    day_label,
                    day,
                    superset=superset,
                    exact_symbols=exact_symbols,
                    work=work,
                    cons_cache=cons_cache,
                    consolidated_loader=cons_loader,
                    decomposition_config=decomp,
                    config=config,
                    known_consolidated=known_consolidated,
                )
                if len(recon_aggregates):
                    aggregates = (
                        pd.concat([aggregates, recon_aggregates], ignore_index=True)
                        if len(aggregates)
                        else recon_aggregates
                    )
                if len(recon_bar_exclusions):
                    bar_exclusions = (
                        pd.concat([bar_exclusions, recon_bar_exclusions], ignore_index=True)
                        if len(bar_exclusions)
                        else recon_bar_exclusions
                    )
            bar_rows, join_excluded = _build_bar_day_rows(day_label, day, aggregates, day_ph, prev_ph, config=config)
            if recon_on:
                join_excluded = [*join_excluded, *recon_extra]
            day_panel = (
                _cast_panel_dtypes(pd.DataFrame(bar_rows, columns=list(PIT1520_PANEL_COLUMNS)))
                if bar_rows
                else _empty_panel()
            )
            if recon_on:
                assert decomp is not None
                day_panel = _cast_panel_dtypes(
                    _stamp_recon_basis(day_panel, recon_symbols=recon_symbols, decomposition_config=decomp),
                    columns=out_columns,
                )
            excluded_symbols = {str(s) for s in bar_exclusions["symbol"].tolist()} if len(bar_exclusions) else set()
            excluded_symbols |= {str(e["symbol"]) for e in join_excluded}
            panel_symbols = set(day_panel["symbol"].astype(str).tolist()) if len(day_panel) else set()
            attributed: list[dict[str, object]] = []
            for missing_symbol in sorted(superset - panel_symbols - excluded_symbols):
                trow = day_ph.loc[day_ph["symbol"] == missing_symbol].iloc[0]
                close_v = float(_coerce_float(pd.Series([trow["close"]]))[0])
                raw_v = float(_coerce_float(pd.Series([trow["close_raw"]]))[0])
                adjusted = (
                    np.isfinite(close_v)
                    and np.isfinite(raw_v)
                    and round(float(close_v)) != round(float(raw_v))
                )
                attributed.append({
                    "date": day,
                    "symbol": missing_symbol,
                    "reason": (
                        PanelExclusionReason.PRICE_BASIS_ADJUSTED.value if adjusted
                        else PanelExclusionReason.NOT_FETCHED.value
                    ),
                    "detail": f"close={close_v!r} close_raw={raw_v!r}",
                })
            parts = [bar_exclusions.assign(date=day)[["date", "symbol", "reason", "detail"]] if len(bar_exclusions) else _empty_day_exclusions()]
            if join_excluded:
                parts.append(pd.DataFrame(join_excluded, columns=["date", "symbol", "reason", "detail"]))
            if attributed:
                parts.append(pd.DataFrame(attributed, columns=["date", "symbol", "reason", "detail"]))
            day_exclusions = pd.concat(parts, ignore_index=True)
            source = PanelSource.BARS.value
            index_basis = INDEX_BASIS_EOD_FALLBACK
        panel_symbols = set(day_panel["symbol"].astype(str).tolist()) if len(day_panel) else set()
        n_present = len(superset & panel_symbols)
        coverage = float(n_present / len(superset)) if superset else float("nan")
        day_rows.append({
            "date": day,
            "source": source,
            "n_rows": len(day_panel),
            "n_excluded": len(day_exclusions),
            "n_superset": len(superset),
            "n_superset_present": int(n_present),
            "superset_coverage": coverage,
            "index_basis": index_basis,
        })
        panel_frames.append(day_panel)
        exclusion_frames.append(day_exclusions)
        logger.info(
            "[DATA] stage=pit1520_panel date=%s source=%s rows=%d excluded=%d superset_coverage=%.4f done=%d/%d",
            day_label,
            source,
            len(day_panel),
            len(day_exclusions),
            coverage,
            done,
            total,
        )
    panel = (
        _cast_panel_dtypes(pd.concat(panel_frames, ignore_index=True), columns=out_columns).sort_values(
            ["date", "symbol"], kind="stable"
        ).reset_index(drop=True)
        if panel_frames
        else _empty_panel(columns=out_columns)
    )
    assert not bool(panel.duplicated(["date", "symbol"]).any()), "duplicate (date, symbol) rows"
    exclusions = (
        pd.concat(exclusion_frames, ignore_index=True)[["date", "symbol", "reason", "detail"]].sort_values(
            ["date", "symbol"], kind="stable"
        ).reset_index(drop=True)
        if exclusion_frames
        else _empty_day_exclusions()
    )
    if len(exclusions):
        exclusions["date"] = pd.to_datetime(exclusions["date"], errors="coerce")
        panel_keys = (
            set(zip(panel["date"].astype(str).tolist(), panel["symbol"].astype(str).tolist(), strict=True))
            if len(panel)
            else set()
        )
        bad = [
            (str(d), str(s))
            for d, s in zip(exclusions["date"].astype(str).tolist(), exclusions["symbol"].astype(str).tolist(), strict=True)
            if (str(d), str(s)) in panel_keys
        ]
        assert not bad, f"exclusion overlaps panel rows: {bad[:3]}"
    days = (
        pd.DataFrame(day_rows).sort_values("date", kind="stable").reset_index(drop=True)
        if day_rows
        else _empty_days()
    )
    if len(days):
        days["date"] = pd.to_datetime(days["date"], errors="coerce")
    return Pit1520PanelResult(panel=panel, exclusions=exclusions, days=days)


def load_regular_bars(snapshot_date: str, *, config: Pit1520PanelConfig = Pit1520PanelConfig()) -> pd.DataFrame:  # noqa: B008
    """Read one date's regular 1m partition column-pruned with ts_hms predicate pushdown.

    Returns:
        Canonical bar rows with floor <= ts_hms <= cutoff + one minute (the widest window any stamp
        convention can need); an empty canonical frame when the partition does not exist.

    Raises:
        OSError: The partition exists but is unreadable (fail loud; never treated as empty).
    """
    from src.data.intraday_store import intraday_partition_path

    target = intraday_partition_path(
        int(DEFAULT_BAR_INTERVAL_MINUTES), str(snapshot_date), str(INTRADAY_SESSION_REGULAR)
    )
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in _BAR_REQUIRED_COLUMNS})
    if not target.exists():
        return empty
    floor = int(config.floor_hhmmss)
    ceil = _shift_minutes(int(config.cutoff_hhmmss), 1)
    try:
        try:
            frame = pd.read_parquet(
                target,
                columns=list(_BAR_REQUIRED_COLUMNS),
                filters=[("ts_hms", ">=", floor), ("ts_hms", "<=", ceil)],
            )
        except Exception:
            frame = pd.read_parquet(target, columns=list(_BAR_REQUIRED_COLUMNS))
    except Exception as exc:
        raise OSError(f"Cannot read intraday partition evidence: {target}") from exc
    if frame is None or len(frame) == 0:
        return empty
    ts = pd.to_numeric(frame["ts_hms"], errors="coerce")
    keep = ((ts >= floor) & (ts <= ceil)).to_numpy()
    return frame.loc[keep].reset_index(drop=True)


def default_decomposition_config_path() -> Path:
    """Fitted decomposition config location under settings.HISTORY_DIR."""
    from src import settings

    return Path(settings.HISTORY_DIR) / DECOMPOSITION_CONFIG_FILENAME


def default_toss_ledger_path() -> Path:
    """Toss backfill ledger location holding the Part 1 consolidated-tape verdicts."""
    from src import settings

    return Path(settings.HISTORY_DIR) / "intraday" / "backfill_ledger" / _TOSS_LEDGER_FILENAME


def load_consolidated_bars(snapshot_date: str, *, config: Pit1520PanelConfig = Pit1520PanelConfig()) -> pd.DataFrame:  # noqa: B008
    """Read one date's consolidated-tape 1m partition, column-pruned like the regular loader.

    Returns:
        Canonical bar rows with floor <= ts_hms <= cutoff + one minute; an empty canonical frame
        when the partition does not exist.

    Raises:
        OSError: The partition exists but is unreadable (fail loud; never treated as empty).
    """
    from src.data.intraday_store import intraday_partition_path

    target = intraday_partition_path(
        int(DEFAULT_BAR_INTERVAL_MINUTES), str(snapshot_date), str(INTRADAY_SESSION_REGULAR_CONSOLIDATED)
    )
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in _BAR_REQUIRED_COLUMNS})
    if not target.exists():
        return empty
    floor = int(config.floor_hhmmss)
    ceil = _shift_minutes(int(config.cutoff_hhmmss), 1)
    try:
        try:
            frame = pd.read_parquet(
                target,
                columns=list(_BAR_REQUIRED_COLUMNS),
                filters=[("ts_hms", ">=", floor), ("ts_hms", "<=", ceil)],
            )
        except Exception:
            frame = pd.read_parquet(target, columns=list(_BAR_REQUIRED_COLUMNS))
    except Exception as exc:
        raise OSError(f"Cannot read intraday partition evidence: {target}") from exc
    if frame is None or len(frame) == 0:
        return empty
    ts = pd.to_numeric(frame["ts_hms"], errors="coerce")
    keep = ((ts >= floor) & (ts <= ceil)).to_numpy()
    return frame.loc[keep].reset_index(drop=True)


def known_consolidated_symbol_days(ledger_path: Path | str | None = None) -> frozenset[tuple[str, str]]:
    """Symbol-days the Toss ledger verdicts as consolidated tape (the Part 1 basis gate).

    Tolerant reader for panel wiring: a missing or unreadable ledger means no symbol-day is claimed
    consolidated, so `consolidated_missing` never fires and attribution stays `not_fetched`.
    """
    path = Path(ledger_path) if ledger_path is not None else default_toss_ledger_path()
    if not path.exists():
        return frozenset()
    try:
        frame = pd.read_parquet(path, columns=["snapshot_date", "session", "symbol", "status", "reason"])
    except Exception as exc:
        logger.warning(
            "[DATA] stage=pit1520_panel status=CONSOLIDATED_KNOWN_UNREADABLE path=%s error=%s",
            path,
            type(exc).__name__,
        )
        return frozenset()
    hits = frame[
        (frame["session"].astype(str) == INTRADAY_SESSION_REGULAR)
        & (frame["status"].astype(str) == CaptureStatus.NOT_APPLICABLE.value)
        & (frame["reason"].astype(str) == _KNOWN_CONSOLIDATED_REASON)
    ]
    if hits.empty:
        return frozenset()
    latest = hits.drop_duplicates(subset=["snapshot_date", "symbol"], keep="last")
    days = pd.to_datetime(latest["snapshot_date"], errors="coerce", format="mixed")
    ok = ~days.isna()
    return frozenset(
        (pd.Timestamp(d).strftime("%Y-%m-%d"), str(s).zfill(6))
        for d, s, keep in zip(days.tolist(), latest["symbol"].tolist(), ok.tolist(), strict=True)
        if bool(keep)
    )


def _stamp_recon_basis(
    day_panel: pd.DataFrame, *, recon_symbols: set[str], decomposition_config: DecompositionConfig
) -> pd.DataFrame:
    """Stamp every row's provenance: reconstructed rows carry the calibrated error quantiles."""
    out = day_panel.copy()
    is_recon = out["symbol"].astype(str).isin(recon_symbols).to_numpy(dtype=bool) if len(out) else np.zeros(0, dtype=bool)
    out["basis"] = np.where(is_recon, BASIS_RECONSTRUCTED, BASIS_EXACT)
    out["recon_volume_rel_err_p90"] = np.where(is_recon, float(decomposition_config.volume_rel_err_p90), np.nan)
    out["recon_close_bp_err_p90"] = np.where(is_recon, float(decomposition_config.close_bp_err_p90), np.nan)
    return out


def _predict_consolidated_share(
    *,
    day_label: str,
    day: pd.Timestamp,
    symbol: str,
    work: pd.DataFrame,
    cons_cache: dict[str, pd.DataFrame],
    consolidated_loader: Callable[[str], pd.DataFrame],
    decomposition_config: DecompositionConfig,
) -> float | None:
    """Predict the KRX share for one consolidated symbol-day from strictly earlier history.

    History bars come from earlier consolidated partitions and EOD volumes from price_history; the
    decision day's EOD volume is never consulted, so randomizing it cannot move the prediction.
    """
    gap = int(decomposition_config.max_gap_days)
    start = day - pd.Timedelta(days=gap)
    earlier = work[
        (work["date"] < day) & (work["date"] >= start) & (work["symbol"] == symbol)
    ].sort_values("date", kind="stable")
    if earlier.empty:
        return None
    eod_volumes = pd.to_numeric(earlier["volume"], errors="coerce").tolist()
    cons_map: dict[str, pd.DataFrame] = {}
    eod_map: dict[str, float] = {}
    for row_day, eod in zip(
        pd.to_datetime(earlier["date"], errors="coerce").dt.strftime("%Y-%m-%d").tolist(),
        eod_volumes,
        strict=True,
    ):
        if row_day in cons_map or not np.isfinite(float(eod)) or float(eod) <= 0.0:
            continue
        frame = cons_cache.get(row_day)
        if frame is None:
            frame = consolidated_loader(row_day)
            cons_cache[row_day] = frame
        if frame.empty:
            continue
        sym_rows = frame[frame["symbol"].astype(str) == symbol]
        if sym_rows.empty:
            continue
        cons_map[row_day] = sym_rows
        eod_map[row_day] = float(eod)
    if not cons_map:
        return None
    history = share_proxy_history(cons_map, eod_map, config=decomposition_config)
    return predict_share(history, day_label, config=decomposition_config)


def _reconstruct_consolidated_day(
    day_label: str,
    day: pd.Timestamp,
    *,
    superset: set[str],
    exact_symbols: set[str],
    work: pd.DataFrame,
    cons_cache: dict[str, pd.DataFrame],
    consolidated_loader: Callable[[str], pd.DataFrame],
    decomposition_config: DecompositionConfig,
    config: Pit1520PanelConfig,
    known_consolidated: frozenset[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, set[str], list[dict[str, object]]]:
    """Reconstruct KRX-only bars for superset symbol-days that carry no exact bars.

    Symbol-days with any exact bar row stay on the exact path (exact wins); only symbols wholly
    absent from the regular bars are candidates. Returns (aggregates, exclusions, reconstructed
    symbols, symbol-day exclusions for history failures).
    """
    cutoff = float(int(config.cutoff_hhmmss))
    window_start = day - pd.Timedelta(days=int(decomposition_config.max_gap_days))
    # One window/symbol slice per decision day instead of one full-history scan per symbol.
    work = work[(work["date"] < day) & (work["date"] >= window_start) & work["symbol"].isin(superset - exact_symbols)]
    cons_day = cons_cache.get(day_label)
    if cons_day is None:
        cons_day = consolidated_loader(day_label)
        cons_cache[day_label] = cons_day
    recon_frames: list[pd.DataFrame] = []
    extra_excluded: list[dict[str, object]] = []
    for symbol in sorted(superset - exact_symbols):
        rows = cons_day[cons_day["symbol"].astype(str) == symbol] if len(cons_day) else cons_day
        if rows.empty:
            if (day_label, symbol) in known_consolidated:
                extra_excluded.append({
                    "date": day,
                    "symbol": symbol,
                    "reason": PanelExclusionReason.CONSOLIDATED_MISSING.value,
                    "detail": "known consolidated symbol-day without consolidated bars",
                })
            continue
        stamps = pd.to_numeric(rows["ts_hms"], errors="coerce")
        observable = rows[stamps <= cutoff]
        if observable.empty:
            if (day_label, symbol) in known_consolidated:
                extra_excluded.append({
                    "date": day,
                    "symbol": symbol,
                    "reason": PanelExclusionReason.CONSOLIDATED_MISSING.value,
                    "detail": "known consolidated symbol-day without observable consolidated bars",
                })
            continue
        stored_vendors = {str(v) for v in observable["vendor"].astype(str).tolist()}
        if stored_vendors != {"toss"}:
            extra_excluded.append({
                "date": day,
                "symbol": symbol,
                "reason": PanelExclusionReason.MIXED_VENDOR.value,
                "detail": f"consolidated bars carry vendors={sorted(stored_vendors)}",
            })
            continue
        if stored_vendors & set(config.excluded_bar_vendors) or RECON_TOSS_VENDOR in config.excluded_bar_vendors:
            extra_excluded.append({
                "date": day,
                "symbol": symbol,
                "reason": PanelExclusionReason.VENDOR_EXCLUDED.value,
                "detail": "consolidated toss bars excluded by vendor",
            })
            continue
        share = _predict_consolidated_share(
            day_label=day_label,
            day=day,
            symbol=symbol,
            work=work,
            cons_cache=cons_cache,
            consolidated_loader=consolidated_loader,
            decomposition_config=decomposition_config,
        )
        if share is None:
            extra_excluded.append({
                "date": day,
                "symbol": symbol,
                "reason": PanelExclusionReason.SHARE_UNAVAILABLE.value,
                "detail": f"insufficient share history before {day_label}",
            })
            continue
        frame = observable.copy()
        frame["vendor"] = RECON_TOSS_VENDOR
        try:
            recon_frames.append(reconstruct_krx_bars(frame, share=share, config=decomposition_config))
        except ValueError as exc:
            extra_excluded.append({
                "date": day,
                "symbol": symbol,
                "reason": PanelExclusionReason.INVALID_PRICE.value,
                "detail": f"reconstruction rejected consolidated bars: {exc}",
            })
    if not recon_frames:
        return _empty_aggregates(), _empty_exclusions(), set(), extra_excluded
    merged = pd.concat(recon_frames, ignore_index=True)
    recon_config = dataclasses.replace(
        config,
        stamp_conventions={**dict(config.stamp_conventions), RECON_TOSS_VENDOR: BAR_STAMP_END},
    )
    recon_aggregates, recon_exclusions = aggregate_decision_bars(merged, config=recon_config)
    recon_symbols = set(recon_aggregates["symbol"].astype(str).tolist()) if len(recon_aggregates) else set()
    return recon_aggregates, recon_exclusions, recon_symbols, extra_excluded


def load_live_decision_input(
    store: CaptureStore, snapshot_date: str, *, config: Pit1520PanelConfig = Pit1520PanelConfig()  # noqa: B008
) -> tuple[pd.DataFrame, str] | None:
    """Return the verified live decision input observable by T at config.live_available_by_hhmmss.

    Returns:
        (frame, capture_run_id), or None when CaptureStore.read_decision raises FileNotFoundError.

    Raises:
        ValueError: Evidence exists but fails hash/cohort verification (propagated; never silently skipped).
    """
    hhmmss = str(config.live_available_by_hhmmss)
    day = pd.Timestamp(str(snapshot_date)).normalize()
    available_by = datetime(
        day.year, day.month, day.day, int(hhmmss[0:2]), int(hhmmss[2:4]), int(hhmmss[4:6]),
        tzinfo=ZoneInfo("Asia/Seoul"),
    )
    try:
        frame = store.read_decision(str(snapshot_date), available_by=available_by)
    except FileNotFoundError:
        return None
    run_id = str(frame["capture_run_id"].iloc[0]) if "capture_run_id" in frame.columns and len(frame) else ""
    return frame, run_id


def write_pit1520_panel(result: Pit1520PanelResult, *, paths: tuple[Path, Path, Path], replace_dates: Sequence[str]) -> None:
    """Atomically replace the given dates in the three panel files, keeping every other date.

    Raises:
        OSError: Persistence fails (no partial file is ever visible).
    """
    from src.data.io_utils import atomic_write_parquet, read_existing_parquet

    wanted = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in replace_dates}
    frames = (result.panel, result.exclusions, result.days)
    sort_keys: tuple[list[str], list[str], list[str]] = (["date", "symbol"], ["date", "symbol"], ["date"])
    for path, frame, keys in zip(paths, frames, sort_keys, strict=True):
        existing = read_existing_parquet(Path(path))
        if len(existing):
            existing = existing.copy()
            existing["_day"] = pd.to_datetime(existing["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            kept = existing.loc[~existing["_day"].isin(wanted)].drop(columns=["_day"]).reset_index(drop=True)
        else:
            kept = existing
        if len(kept) and len(frame):
            merged = pd.concat([kept, frame], ignore_index=True)
        elif len(frame):
            merged = frame.copy()
        else:
            merged = kept.copy()
        if len(merged):
            merged = merged.copy()
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged = merged.sort_values(keys, kind="stable").reset_index(drop=True)
        atomic_write_parquet(merged, Path(path))


def _earliest_regular_partition_date() -> str | None:
    from src import settings

    root = (
        Path(settings.HISTORY_DIR)
        / "intraday"
        / f"{int(DEFAULT_BAR_INTERVAL_MINUTES)}m"
        / str(INTRADAY_SESSION_REGULAR)
    )
    if not root.exists():
        return None
    stems = sorted(p.stem for p in root.rglob("*.parquet") if p.is_file())
    return stems[0] if stems else None


def main(argv: list[str] | None = None) -> None:
    """CLI: build the decision-time panel for a date range and persist it.

    Flags:
        --start YYYY-MM-DD (default: earliest date with a regular 1m partition)
        --end YYYY-MM-DD (default: latest price_history date)
        --source-policy {prefer_live,bars_only} (default prefer_live)
        --exclude-bar-vendor VENDOR (repeatable; drops that vendor's bar symbol-days as vendor_excluded)
        --reconstruct-consolidated (rebuild KRX-only bars for consolidated symbol-days via Part 2)
        --decomposition-config PATH (fitted config; default settings.HISTORY_DIR/nxt_decomposition_config.json)
        --out-dir PATH (default settings.HISTORY_DIR)
    """
    from src import settings
    from src.config.collection import CollectionSettings
    from src.data.capture_store import resolve_capture_root
    from src.data.panel_integrity import load_price_panel

    parser = argparse.ArgumentParser(description="Build the decision-time (15:20) panel")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--source-policy", choices=("prefer_live", "bars_only"), default="prefer_live")
    parser.add_argument("--exclude-bar-vendor", action="append", default=[], dest="exclude_bar_vendor")
    parser.add_argument("--reconstruct-consolidated", action="store_true")
    parser.add_argument("--decomposition-config", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    config = Pit1520PanelConfig(
        excluded_bar_vendors=frozenset(args.exclude_bar_vendor),
        reconstruct_consolidated=bool(args.reconstruct_consolidated),
    )
    decomp: DecompositionConfig | None = None
    cons_loader: Callable[[str], pd.DataFrame] | None = None
    known: frozenset[tuple[str, str]] | None = None
    if args.reconstruct_consolidated:
        cfg_path = Path(args.decomposition_config) if args.decomposition_config else default_decomposition_config_path()
        decomp = load_decomposition_config(cfg_path)

        def _load_cons(day: str) -> pd.DataFrame:
            return load_consolidated_bars(day, config=config)

        cons_loader = _load_cons
        known = known_consolidated_symbol_days()
    price_history, _prov = load_price_panel(settings.PRICE_HISTORY_PARQUET_PATH)
    calendar = sorted(pd.to_datetime(price_history["date"]).dt.normalize().unique().tolist())
    if not calendar:
        raise ValueError("build_pit1520_panel price_history carries no dates")
    start = args.start or _earliest_regular_partition_date() or pd.Timestamp(calendar[0]).strftime("%Y-%m-%d")
    end = args.end or pd.Timestamp(calendar[-1]).strftime("%Y-%m-%d")
    wanted = [
        pd.Timestamp(d).strftime("%Y-%m-%d") for d in calendar if start <= pd.Timestamp(d).strftime("%Y-%m-%d") <= end
    ]
    if not wanted:
        logger.info("[DATA] stage=pit1520_panel status=DONE dates=0 rows=0 excluded=0 live_days=0 bar_days=0")
        return
    screen = EodSupersetScreen.from_profile(CollectionSettings())
    store = CaptureStore(resolve_capture_root())
    policy = PanelSourcePolicy(args.source_policy)

    def _load_bars(day: str) -> pd.DataFrame:
        return load_regular_bars(day, config=config)

    def _load_live(day: str) -> tuple[pd.DataFrame, str] | None:
        return load_live_decision_input(store, day, config=config)

    result = build_pit1520_panel(
        price_history=price_history,
        dates=wanted,
        bars_loader=_load_bars,
        live_loader=_load_live,
        screen=screen,
        config=config,
        source_policy=policy,
        consolidated_loader=cons_loader,
        decomposition_config=decomp,
        consolidated_symbol_days=known,
    )
    out_dir = Path(args.out_dir) if args.out_dir else Path(settings.HISTORY_DIR)
    paths = (
        out_dir / "pit1520_panel.parquet",
        out_dir / "pit1520_panel_exclusions.parquet",
        out_dir / "pit1520_panel_days.parquet",
    )
    write_pit1520_panel(result, paths=paths, replace_dates=wanted)
    coverage = pd.to_numeric(result.days["superset_coverage"], errors="coerce").to_numpy(dtype=np.float64)
    finite = coverage[np.isfinite(coverage)]
    min_cov = float(np.min(finite)) if len(finite) else float("nan")
    med_cov = float(np.median(finite)) if len(finite) else float("nan")
    live_days = int((result.days["source"] == PanelSource.LIVE_DECISION.value).sum()) if len(result.days) else 0
    bar_days = int((result.days["source"] == PanelSource.BARS.value).sum()) if len(result.days) else 0
    n_recon = int((result.panel["basis"] == BASIS_RECONSTRUCTED).sum()) if "basis" in result.panel.columns and len(result.panel) else 0
    n_share_missing = int((result.exclusions["reason"] == PanelExclusionReason.SHARE_UNAVAILABLE.value).sum()) if len(result.exclusions) else 0
    logger.info(
        "[DATA] stage=pit1520_panel status=DONE dates=%d rows=%d excluded=%d live_days=%d bar_days=%d min_coverage=%.4f median_coverage=%.4f",
        len(wanted),
        len(result.panel),
        len(result.exclusions),
        live_days,
        bar_days,
        min_cov,
        med_cov,
    )
    if args.reconstruct_consolidated:
        logger.info(
            "[DATA] stage=pit1520_panel status=RECONSTRUCTED recon_rows=%d share_unavailable=%d",
            n_recon,
            n_share_missing,
        )


if __name__ == "__main__":  # pragma: no cover
    from src.utils.cli_logging import configure_cli_logging

    configure_cli_logging()
    main()
