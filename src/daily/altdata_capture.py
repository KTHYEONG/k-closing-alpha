"""Rolling slow-data refresh without trade-based coverage."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import uuid
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src import settings
from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.runner import run_altdata_backfill
from src.config.collection import CollectionSettings
from src.daily.price_ingest import fetch_krx_daily
from src.data.capture_contracts import SEOUL, CaptureManifest, CaptureStatus
from src.data.capture_store import CaptureStore

logger = logging.getLogger(__name__)


def _capture_root(profile: CollectionSettings) -> Path:
    if profile.COLLECTION_ROOT is not None:
        return Path(profile.COLLECTION_ROOT)
    return Path(settings.HISTORY_DIR) / "capture"


def _rolling_bounds(trading_day: date, lookback_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(trading_day).normalize()
    start = end - pd.Timedelta(days=int(lookback_days) - 1)
    return start, end


_UNIVERSE_LOOKBACK_DAYS: int = 7


def _krx_listed_universe(window_end: pd.Timestamp, cfg: AltDataFetchConfig) -> frozenset[str] | None:
    """Fetch the most recently published full KRX-listed symbol set (no screen/trade bias).

    KRX's daily open API publishes a trading date's rows with a settlement lag
    (same-day queries for the run date itself observably return zero rows), so
    this walks back from window_end until a published day is found. Returns
    None only after exhausting the lookback (e.g. a multi-day holiday cluster),
    leaving the caller's symbol-scoped panels to skip that date.

    ETN/ELW/non-6-digit codes are dropped: AltDataFetchConfig.universe_symbols
    requires plain 6-digit tickers, and these alt-data endpoints are scoped to
    common stock, not derivative/ETN listings.
    """
    for offset in range(_UNIVERSE_LOOKBACK_DAYS):
        candidate = window_end - pd.Timedelta(days=offset)
        frame = fetch_krx_daily(candidate, cfg)
        if frame.empty:
            continue
        symbols = {s for s in frame["symbol"].astype(str).str.strip().tolist() if len(s) == 6 and s.isdigit()}
        if symbols:
            return frozenset(symbols)
    return None


def run_altdata_capture(trading_date: date, *, profile: CollectionSettings, store: CaptureStore, cfg: AltDataFetchConfig) -> CaptureManifest:
    """Refresh the declared rolling window without using selected trades as coverage.

    Args:
        trading_date: Run anchor date in Asia/Seoul.
        profile: Explicit slow-data enablement and lookback limits.
        store: Owner-local durable observation store.
        cfg: Existing source credentials, rates and inclusive date bounds.
    Returns:
        Aggregate manifest reconciling each configured source.
    Raises:
        ValueError: Disabled profile or mismatched rolling bounds.
        RawCaptureError: Durable output failed.
    """
    if not (bool(profile.COLLECTION_RAW_ENABLED) and bool(profile.COLLECTION_ALTDATA_ENABLED)):
        raise ValueError("altdata capture requires enabled raw and altdata collection")
    lookback = int(profile.COLLECTION_ALTDATA_LOOKBACK_DAYS)
    window_start, window_end = _rolling_bounds(trading_date, lookback)
    if pd.Timestamp(cfg.start).normalize() != window_start or pd.Timestamp(cfg.end).normalize() != window_end:
        raise ValueError("cfg rolling bounds must equal the declared inclusive window")
    if cfg.universe_symbols is None:
        universe = _krx_listed_universe(window_end, cfg)
        if universe is not None:
            cfg = dataclasses.replace(cfg, universe_symbols=universe)
    run_id = f"altdata-{window_end.strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}"
    started = datetime.now(SEOUL)
    report = run_altdata_backfill(cfg, capture_store=store, run_id=run_id, reobserve=True)
    elapsed = (datetime.now(SEOUL) - started).total_seconds()
    logger.info(
        "[DATA] stage=altdata_capture date=%s window=%s..%s elapsed=%.1fs capture=%s",
        window_end.date().isoformat(),
        window_start.date().isoformat(),
        window_end.date().isoformat(),
        elapsed,
        report.get("capture"),
    )
    for manifest in reversed(store.read_manifests(window_end.date().isoformat())):
        if manifest.context.run_id == run_id:
            return manifest
    raise ValueError(f"altdata capture manifest missing for run {run_id!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optional scheduled slow-data refresh.

    Args:
        argv: Optional --date ISO run date; omission selects today's Korean date.
    Returns:
        Zero for a complete configured-source run, nonzero for incomplete acquisition.
    Raises:
        ValueError: Invalid date or configured profile.
    """
    parser = argparse.ArgumentParser(description="Rolling slow-data refresh (research, no trading)")
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)
    try:
        trading_day = date.fromisoformat(args.date) if args.date else datetime.now(SEOUL).date()
    except ValueError:
        raise ValueError(f"Invalid date: {args.date!r}") from None
    profile = CollectionSettings()
    if not (bool(profile.COLLECTION_RAW_ENABLED) and bool(profile.COLLECTION_ALTDATA_ENABLED)):
        logger.info("[DATA] stage=altdata_capture status=SKIP reason=disabled")
        return 0
    window_start, window_end = _rolling_bounds(trading_day, int(profile.COLLECTION_ALTDATA_LOOKBACK_DAYS))
    end_ts = window_end if window_end > window_start else window_start + pd.Timedelta(hours=12)
    cfg = AltDataFetchConfig(
        start=window_start,
        end=end_ts,
        out_dir=Path(settings.ALTDATA_DIR),
        dart_api_key=str(settings.OPENDART_API_KEY or settings.DART_API_KEY),
        krx_api_key=settings.KRX_OPENAPI_KEY,
    )
    store = CaptureStore(_capture_root(profile))
    manifest = run_altdata_capture(trading_day, profile=profile, store=store, cfg=cfg)
    return 0 if manifest.status == CaptureStatus.COMPLETE else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
