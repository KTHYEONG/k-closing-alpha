"""Rolling slow-data refresh without trade-based coverage."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src import settings
from src.api.kis.key_pool import resolve_research_credentials
from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.dart_keys import resolve_dart_key_pool
from src.backfill.altdata.runner import run_altdata_backfill
from src.config.collection import CollectionSettings
from src.daily.price_ingest import fetch_krx_daily
from src.data.altdata_health import AltdataVerdict, altdata_verdict
from src.data.capture_contracts import SEOUL, CaptureManifest, CaptureStatus
from src.data.capture_store import CaptureStore
from src.data.session_calendar import SessionKind, resolve_session_day
from src.data.trading_calendar import is_kis_trading_day_sync

logger = logging.getLogger(__name__)


def _capture_root(profile: CollectionSettings) -> Path:
    if profile.COLLECTION_ROOT is not None:
        return Path(profile.COLLECTION_ROOT)
    return Path(settings.HISTORY_DIR) / "capture"


def _rolling_bounds(trading_day: date, lookback_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(trading_day).normalize()
    start = end - pd.Timedelta(days=int(lookback_days) - 1)
    return start, end


def _extra_altdata_client_kwargs(profile: CollectionSettings, env: Mapping[str, str]) -> tuple[tuple[str, str, str], ...]:
    """추가 alt-data 슬롯의 자격증명을 수집 설정 형식으로 변환한다.

    Args:
        profile: 추가 슬롯 선언을 담은 수집 설정.
        env: 자격증명 소스.

    Returns:
        수집 설정에 대입 가능한 자격증명 튜플.
    """
    if len(profile.COLLECTION_ALTDATA_EXTRA_SLOTS) == 0:
        return ()
    creds = resolve_research_credentials(env, slots=tuple(profile.COLLECTION_ALTDATA_EXTRA_SLOTS))
    return tuple((cred.app_key, cred.app_secret, cred.hts_id) for cred in creds)


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
    if not cfg.extra_client_kwargs:
        extra = _extra_altdata_client_kwargs(profile, dict(os.environ))
        if extra:
            cfg = dataclasses.replace(cfg, extra_client_kwargs=extra)
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


def main(argv: Sequence[str] | None = None, *, trading_day_fn: Callable[[str], bool] | None = None) -> int:
    """Run the optional scheduled slow-data refresh.

    Args:
        argv: Optional --date ISO run date; omission selects today's Korean date.
        trading_day_fn: Trading-day oracle; defaults to the synchronous KIS lookup.
    Returns:
        Zero for a COMPLETE or DEGRADED run, nonzero for a FAILED run.
        A skipped-by-disabled run returns zero unchanged.
    Raises:
        ValueError: Invalid date or configured profile.
    """
    import aiohttp

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
    if trading_day.weekday() >= 5:
        logger.info("[DATA] stage=altdata_capture status=SKIP reason=weekend date=%s", trading_day.isoformat())
        return 0
    if resolve_session_day(trading_day).kind is SessionKind.CLOSED:
        logger.info("[DATA] stage=altdata_capture status=SKIP reason=non_trading_day date=%s", trading_day.isoformat())
        return 0
    oracle = trading_day_fn if trading_day_fn is not None else is_kis_trading_day_sync
    try:
        is_trading = oracle(trading_day.isoformat())
    except (RuntimeError, OSError, aiohttp.ClientError) as exc:
        logger.warning("[DATA] stage=altdata_capture calendar_lookup=FAIL reason=%s proceed=true", type(exc).__name__)
    else:
        if not is_trading:
            logger.info(
                "[DATA] stage=altdata_capture status=SKIP reason=non_trading_day date=%s",
                trading_day.isoformat(),
            )
            return 0
    window_start, window_end = _rolling_bounds(trading_day, int(profile.COLLECTION_ALTDATA_LOOKBACK_DAYS))
    end_ts = window_end if window_end > window_start else window_start + pd.Timedelta(hours=12)
    cfg = AltDataFetchConfig(
        start=window_start,
        end=end_ts,
        out_dir=Path(settings.ALTDATA_DIR),
        dart_key_pool=resolve_dart_key_pool(
            primary=settings.OPENDART_API_KEY,
            secondary=settings.OPENDART_API_KEY_2,
            legacy=settings.DART_API_KEY,
        ),
        krx_api_key=settings.KRX_OPENAPI_KEY,
    )
    logger.info("[DATA] stage=dart_key_pool labels=%s n=%d", ",".join(cfg.dart_key_pool.labels), len(cfg.dart_key_pool.labels))
    store = CaptureStore(_capture_root(profile))
    manifest = run_altdata_capture(trading_day, profile=profile, store=store, cfg=cfg)
    verdict = altdata_verdict(manifest)
    if verdict == AltdataVerdict.COMPLETE:
        return 0
    if verdict == AltdataVerdict.DEGRADED:
        degraded = ",".join(
            f"{e.dataset.value}:{e.reason}" for e in manifest.entries if e.status != CaptureStatus.COMPLETE
        )
        logger.warning("[DATA] stage=altdata_capture status=DEGRADED sources=%s", degraded)
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    # 설정이 없으면 INFO(SKIP·키 풀 구성)가 저널에 남지 않아 게이트 동작을 확인할 수 없다.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
