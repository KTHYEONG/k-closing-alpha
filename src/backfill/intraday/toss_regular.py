"""Toss regular-session 1m backfill acquisition with KRX-basis gate."""

from __future__ import annotations

import logging
import math
import re
import uuid
from itertools import pairwise
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Sequence

import pandas as pd

from src.config.collection import CollectionSettings
from src.config.market_session import INTRADAY_SESSION_REGULAR
from src.data.capture_contracts import (
    SEOUL,
    ArtifactRef,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureStatus,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.intraday_schema import normalize_bar_frame

logger = logging.getLogger(__name__)

TOSS_CANDLES_ENDPOINT: str = "toss-candles"
TOSS_CANDLES_PAGE_MAX: int = 200
TOSS_REGULAR_FIRST_LABEL_HHMMSS: str = "090100"
TOSS_REGULAR_LAST_LABEL_HHMMSS: str = "153000"
TOSS_REGULAR_EXPECTED_BARS: int = 390
TOSS_MAX_PAGES_PER_DAY: int = 3

TOSS_BASIS_RATIO_MIN: float = 0.95
TOSS_BASIS_RATIO_TOLERANCE: float = 0.02

_ERROR_RE = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class TossBasisVerdict:
    accepted: bool
    reason: str
    volume_ratio: float | None


def toss_basis_verdict(
    frame: pd.DataFrame, eod_volume: float | None, *, ratio_min: float, ratio_tolerance: float
) -> TossBasisVerdict:
    """Decide whether Toss bars represent KRX-only tape via EOD volume ratio comparison.

    Args:
        frame: canonical Toss bars covering the full regular session.
        eod_volume: the KRX EOD volume of the same symbol-day.
        ratio_min: lower bound below which bars are considered missing.
        ratio_tolerance: non-negative slack on the upper bound.

    Returns:
        verdict with `accepted`, a machine reason (`ok`, `toss_consolidated_tape`, `toss_volume_shortfall`, `toss_basis_unverifiable`) and the measured ratio (None when unverifiable).
    """
    if eod_volume is None:
        return TossBasisVerdict(accepted=False, reason="toss_basis_unverifiable", volume_ratio=None)
    try:
        eod = float(eod_volume)
    except (TypeError, ValueError):
        return TossBasisVerdict(accepted=False, reason="toss_basis_unverifiable", volume_ratio=None)
    if not math.isfinite(eod) or eod <= 0:
        return TossBasisVerdict(accepted=False, reason="toss_basis_unverifiable", volume_ratio=None)
    if frame is None or frame.empty or "volume" not in frame.columns:
        total = 0.0
    else:
        total = float(pd.to_numeric(frame["volume"], errors="coerce").fillna(0).sum())
    ratio = total / eod
    if not math.isfinite(ratio):
        return TossBasisVerdict(accepted=False, reason="toss_basis_unverifiable", volume_ratio=None)
    if ratio < float(ratio_min):
        return TossBasisVerdict(accepted=False, reason="toss_volume_shortfall", volume_ratio=ratio)
    if ratio > 1.0 + float(ratio_tolerance):
        return TossBasisVerdict(accepted=False, reason="toss_consolidated_tape", volume_ratio=ratio)
    return TossBasisVerdict(accepted=True, reason="ok", volume_ratio=ratio)


def _gate_thresholds(profile: CollectionSettings) -> tuple[float, float]:
    ratio_min = getattr(profile, "COLLECTION_TOSS_BASIS_VOLUME_RATIO_MIN", TOSS_BASIS_RATIO_MIN)
    tolerance = getattr(profile, "COLLECTION_TOSS_BASIS_VOLUME_TOLERANCE", TOSS_BASIS_RATIO_TOLERANCE)
    return float(ratio_min), float(tolerance)


def _redacted_error(exc: BaseException) -> str:
    name = type(exc).__name__ or "error"
    cleaned = _ERROR_RE.sub("-", name).strip("-")
    return cleaned or "error"


def _resolve_profile(profile: CollectionSettings | None) -> CollectionSettings:
    return profile if profile is not None else CollectionSettings()


def _resolve_store(capture_store: CaptureStore | None, profile: CollectionSettings) -> CaptureStore:
    if capture_store is not None:
        return capture_store
    return CaptureStore(_capture_root(profile))


def _venue_for_toss(profile: CollectionSettings) -> str:
    routes = profile.COLLECTION_VERIFIED_CHART_ROUTES or {}
    mapped = routes.get(f"toss:{TOSS_CANDLES_ENDPOINT}")
    if mapped is not None:
        return str(mapped)
    return "UNKNOWN"


def _parse_toss_label(timestamp: str) -> tuple[datetime, str, str]:
    dt = datetime.fromisoformat(str(timestamp))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SEOUL)
    day = dt.date().isoformat()
    hhmmss = f"{dt.hour:02d}{dt.minute:02d}{dt.second:02d}"
    return dt, day, hhmmss


def _before_for(snapshot_date: str, hhmmss: str) -> str:
    total = int(hhmmss[0:2]) * 60 + int(hhmmss[2:4]) - 1
    total = max(total, 0)
    return f"{snapshot_date}T{total // 60:02d}:{total % 60:02d}:00+09:00"


def _empty_frame(snapshot_date: str, symbol: str) -> pd.DataFrame:
    return normalize_bar_frame(pd.DataFrame(), "toss", snapshot_date, symbol)


def _terminal_entry(
    *,
    symbol: str,
    venue: str,
    status: CaptureStatus,
    rows: int,
    reason: str,
    refs: list[Any],
) -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.MINUTE_BARS,
        venue=venue,
        session=INTRADAY_SESSION_REGULAR,
        scheduled_at=None,
        status=status,
        rows=int(rows),
        first_event_time=None,
        last_event_time=None,
        reason=reason,
        raw_refs=tuple(refs),
    )


def _log(symbol: str, snapshot_date: str, status: str, reason: str) -> None:
    logger.info("[DATA] stage=toss_regular symbol=%s date=%s status=%s reason=%s", symbol, snapshot_date, status, reason)


def _extract_candles(payload: Any) -> list[dict[str, Any]] | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    candles = result.get("candles")
    if not isinstance(candles, list):
        return None
    return [dict(c) for c in candles if isinstance(c, dict)]


def _error_code(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return str(code) if code is not None else ""


async def acquire_toss_regular_bars(
    client: Any,
    session: Any,
    symbol: str,
    snapshot_date: str,
    *,
    eod_volume: float | None,
    profile: CollectionSettings | None = None,
    capture_store: CaptureStore | None = None,
    run_id: str,
) -> tuple[pd.DataFrame, CoverageEntry]:
    """Fetch one past symbol-day of KRX regular-session 1m bars (labels 09:01..15:30) from Toss on the RAW price basis, with raw page evidence retained. Past dates only. The two-page walk requests the newest page first and continues strictly older until the first label is reached.

    Returns `(frame, entry)`: COMPLETE with canonical bars (vendor `toss`, END-stamped) when the basis gate accepts; a NOT_APPLICABLE `toss_consolidated_tape` rejection also carries its canonical frame so the consolidated-tape partition can persist it without refetching; every other rejection returns an empty frame with a non-COMPLETE entry whose reason names the cause. The frame is never written to any partition by this function.
    """
    if not str(run_id).strip():
        raise ValueError("run_id must be nonempty")
    trading_day = date.fromisoformat(str(snapshot_date))
    today = datetime.now(SEOUL).date().isoformat()
    if str(snapshot_date) >= today:
        raise ValueError(f"snapshot_date must be strictly past: {snapshot_date!r}")
    code = str(symbol).zfill(6)
    prof = _resolve_profile(profile)
    store = _resolve_store(capture_store, prof)
    venue = _venue_for_toss(prof)
    context = CaptureContext(
        trading_date=trading_day,
        run_id=str(run_id),
        dataset=CaptureDataset.MINUTE_BARS,
        vendor="toss",
        endpoint=TOSS_CANDLES_ENDPOINT,
        symbol=code,
        venue=venue,
        session=INTRADAY_SESSION_REGULAR,
        capture_reason="intraday-minute_bars",
        cohort_id=None,
        scheduled_at=None,
    )
    refs: list[ArtifactRef] = []
    retained: list[dict[str, Any]] = []
    total_raw = 0

    before = f"{snapshot_date}T15:30:00+09:00"
    count = TOSS_CANDLES_PAGE_MAX
    for page_index in range(TOSS_MAX_PAGES_PER_DAY):
        started = datetime.now(SEOUL)
        try:
            payload = await client.get_candles(
                session, code, interval="1m", count=int(count), before=before, adjusted=False
            )
        except Exception as exc:
            tag = _redacted_error(exc)
            _log(code, snapshot_date, "FAILED", f"transport:{tag}")
            return _empty_frame(snapshot_date, code), _terminal_entry(
                symbol=code, venue=venue, status=CaptureStatus.FAILED,
                rows=0, reason=f"transport:{tag}", refs=refs,
            )
        received = datetime.now(SEOUL)
        body = dict(payload) if isinstance(payload, dict) else None
        refs.append(
            store.append_response(
                CapturedResponse(
                    context=context,
                    request_started_at=started,
                    received_at=received,
                    payload=body,
                    status=CaptureStatus.COMPLETE if body is not None else CaptureStatus.FAILED,
                    source_timestamp=None,
                    source_published_at=None,
                    page_index=int(page_index),
                    attempt_index=0,
                    continuation={"before": str(before), "count": str(int(count))},
                    error_type=None,
                )
            )
        )
        err = _error_code(payload)
        if err is not None:
            if err == "stock-not-found":
                _log(code, snapshot_date, "NOT_APPLICABLE", "toss_stock_not_found")
                return _empty_frame(snapshot_date, code), _terminal_entry(
                    symbol=code, venue=venue, status=CaptureStatus.NOT_APPLICABLE,
                    rows=0, reason="toss_stock_not_found", refs=refs,
                )
            reason = f"vendor_failure:{err}" if err else "vendor_failure:unknown"
            _log(code, snapshot_date, "FAILED", reason)
            return _empty_frame(snapshot_date, code), _terminal_entry(
                symbol=code, venue=venue, status=CaptureStatus.FAILED,
                rows=0, reason=reason, refs=refs,
            )
        candles = _extract_candles(payload)
        if candles is None:
            _log(code, snapshot_date, "FAILED", "vendor_failure:unknown")
            return _empty_frame(snapshot_date, code), _terminal_entry(
                symbol=code, venue=venue, status=CaptureStatus.FAILED,
                rows=0, reason="vendor_failure:unknown", refs=refs,
            )
        if not candles:
            break
        try:
            parsed = [_parse_toss_label(str(c.get("timestamp", ""))) for c in candles]
        except Exception:
            _log(code, snapshot_date, "FAILED", "toss_malformed_page")
            return _empty_frame(snapshot_date, code), _terminal_entry(
                symbol=code, venue=venue, status=CaptureStatus.FAILED,
                rows=0, reason="toss_malformed_page", refs=refs,
            )
        dts = [item[0] for item in parsed]
        for dt in dts:
            if dt.second != 0 or dt.microsecond != 0:
                _log(code, snapshot_date, "FAILED", "toss_malformed_page")
                return _empty_frame(snapshot_date, code), _terminal_entry(
                    symbol=code, venue=venue, status=CaptureStatus.FAILED,
                    rows=0, reason="toss_malformed_page", refs=refs,
                )
        # Newest-first requires strictly decreasing timestamps.
        if any(b >= a for a, b in pairwise(dts)):
            _log(code, snapshot_date, "FAILED", "toss_malformed_page")
            return _empty_frame(snapshot_date, code), _terminal_entry(
                symbol=code, venue=venue, status=CaptureStatus.FAILED,
                rows=0, reason="toss_malformed_page", refs=refs,
            )
        total_raw += len(candles)
        if parsed[0][1] < str(snapshot_date):
            break
        for candle, (_, day, hhmmss) in zip(candles, parsed):
            if day == str(snapshot_date) and TOSS_REGULAR_FIRST_LABEL_HHMMSS <= hhmmss <= TOSS_REGULAR_LAST_LABEL_HHMMSS:
                retained.append(candle)
        if retained:
            oldest = min(_parse_toss_label(str(c.get("timestamp", "")))[2] for c in retained)
            if oldest <= TOSS_REGULAR_FIRST_LABEL_HHMMSS:
                break
            before = _before_for(str(snapshot_date), oldest)
            count = min(TOSS_CANDLES_PAGE_MAX, max(1, TOSS_REGULAR_EXPECTED_BARS - len(retained)))
        else:
            oldest_raw = min(item[2] for item in parsed if item[1] == str(snapshot_date)) if any(
                item[1] == str(snapshot_date) for item in parsed
            ) else None
            if oldest_raw is not None:
                before = _before_for(str(snapshot_date), oldest_raw)
                count = min(TOSS_CANDLES_PAGE_MAX, max(1, TOSS_REGULAR_EXPECTED_BARS - len(retained)))
            else:
                break
    if total_raw == 0:
        _log(code, snapshot_date, "FAILED", "toss_empty_without_proof")
        return _empty_frame(snapshot_date, code), _terminal_entry(
            symbol=code, venue=venue, status=CaptureStatus.FAILED,
            rows=0, reason="toss_empty_without_proof", refs=refs,
        )
    if not retained:
        _log(code, snapshot_date, "FAILED", "toss_no_regular_bars")
        return _empty_frame(snapshot_date, code), _terminal_entry(
            symbol=code, venue=venue, status=CaptureStatus.FAILED,
            rows=0, reason="toss_no_regular_bars", refs=refs,
        )
    try:
        frame = normalize_bar_frame(pd.DataFrame(retained), "toss", str(snapshot_date), code)
    except Exception:
        _log(code, snapshot_date, "FAILED", "toss_malformed_page")
        return _empty_frame(snapshot_date, code), _terminal_entry(
            symbol=code, venue=venue, status=CaptureStatus.FAILED,
            rows=0, reason="toss_malformed_page", refs=refs,
        )
    if frame.empty:
        _log(code, snapshot_date, "FAILED", "toss_no_regular_bars")
        return _empty_frame(snapshot_date, code), _terminal_entry(
            symbol=code, venue=venue, status=CaptureStatus.FAILED,
            rows=0, reason="toss_no_regular_bars", refs=refs,
        )
    if venue == "UNKNOWN":
        frag_ref = store.publish_frame(
            frame,
            context=CaptureContext(
                trading_date=trading_day,
                run_id=f"{run_id}-uncertified_venue-{uuid.uuid4().hex[:6]}",
                dataset=CaptureDataset.MINUTE_BARS,
                vendor="owner-local",
                endpoint="staged-fragment",
                symbol=code,
                venue="UNKNOWN",
                session=INTRADAY_SESSION_REGULAR,
                capture_reason="intraday-minute_bars",
                cohort_id=None,
                scheduled_at=None,
            ),
        )
        refs.append(frag_ref)
        _log(code, snapshot_date, "UNKNOWN", "uncertified_venue")
        return _empty_frame(snapshot_date, code), _terminal_entry(
            symbol=code, venue=venue, status=CaptureStatus.UNKNOWN,
            rows=0, reason="uncertified_venue", refs=refs,
        )
    ratio_min, ratio_tolerance = _gate_thresholds(prof)
    verdict = toss_basis_verdict(frame, eod_volume, ratio_min=ratio_min, ratio_tolerance=ratio_tolerance)
    if not verdict.accepted:
        if verdict.reason == "toss_consolidated_tape":
            consolidated = frame.sort_values("ts_hms", kind="stable").reset_index(drop=True)
            _log(code, snapshot_date, "NOT_APPLICABLE", verdict.reason)
            return consolidated, _terminal_entry(
                symbol=code, venue=venue, status=CaptureStatus.NOT_APPLICABLE,
                rows=len(consolidated), reason=verdict.reason, refs=refs,
            )
        _log(code, snapshot_date, "FAILED", verdict.reason)
        return _empty_frame(snapshot_date, code), _terminal_entry(
            symbol=code, venue=venue, status=CaptureStatus.FAILED,
            rows=0, reason=verdict.reason, refs=refs,
        )
    frame = frame.sort_values("ts_hms", kind="stable").reset_index(drop=True)
    reason = f"toss_regular:{len(frame)}"
    _log(code, snapshot_date, "COMPLETE", reason)
    return frame, _terminal_entry(
        symbol=code, venue=venue, status=CaptureStatus.COMPLETE,
        rows=len(frame), reason=reason, refs=refs,
    )


async def probe_toss_retention_floor(
    client: Any, session: Any, *, trading_days: Sequence[str], reference_symbol: str
) -> str | None:
    """Find the earliest trading day for which Toss still serves bars, using one liquid reference symbol and a bisection over the supplied ascending trading calendar. Returns that date, or None when even the newest supplied day is empty (vendor outage: callers must treat as "unknown", not "no history")."""
    days = [str(d) for d in (trading_days or [])]
    if not days:
        return None
    symbol = str(reference_symbol).zfill(6)

    async def _served(day: str) -> bool:
        payload = await client.get_candles(
            session, symbol, interval="1m", count=TOSS_CANDLES_PAGE_MAX,
            before=f"{day}T15:30:00+09:00", adjusted=False,
        )
        err = _error_code(payload)
        if err is not None:
            raise RuntimeError(f"toss probe vendor_failure:{err}" if err else "toss probe vendor_failure:unknown")
        candles = _extract_candles(payload)
        if candles is None:
            raise RuntimeError("toss probe vendor_failure:unknown")
        for candle in candles:
            try:
                _, candle_day, hhmmss = _parse_toss_label(str(candle.get("timestamp", "")))
            except Exception:
                continue
            if candle_day == str(day) and TOSS_REGULAR_FIRST_LABEL_HHMMSS <= hhmmss <= TOSS_REGULAR_LAST_LABEL_HHMMSS:
                return True
        return False

    async def _counted(day: str) -> bool:
        return await _served(day)

    if not await _counted(days[-1]):
        return None
    if len(days) == 1:
        return days[0]
    lo = 0
    hi = len(days) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if await _counted(days[mid]):
            hi = mid
        else:
            lo = mid + 1
    return days[hi]
