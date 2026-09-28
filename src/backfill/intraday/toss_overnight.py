"""Toss-sourced NXT evening/premarket 1m bars for entry days older than the KIS/Kiwoom retention window.

adjusted=false with zero-volume bars removed and one-minute label shift reproduces Kiwoom NXT bars exactly,
and Toss is a consolidated tape so it is valid only while KRX has no evening session.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.config.market_session import (
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_NXT_PREMARKET,
    KRX_AFTERMARKET_START_DATE,
    KRX_REGULAR_HOUR_CEIL,
    NXT_AFTERMARKET_HOUR_CEIL,
    NXT_AFTERMARKET_HOUR_FLOOR,
    NXT_PREMARKET_HOUR_CEIL,
    NXT_PREMARKET_HOUR_FLOOR,
    NXT_START_DATE,
)
from src.data.capture_contracts import (
    SEOUL,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureStatus,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.intraday_schema import assert_canonical_bars, normalize_bar_frame

logger = logging.getLogger(__name__)

# vendor per-call maximum.
TOSS_CANDLE_MAX_COUNT: int = 200
# hard bound on calls per (symbol, entry day) window.
TOSS_PAIR_MAX_CALLS: int = 3

_TOSS_REQUIRED: tuple[str, ...] = (
    "timestamp",
    "openPrice",
    "highPrice",
    "lowPrice",
    "closePrice",
    "volume",
)

_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def _hhmmss_to_minutes(hhmmss: str) -> int:
    return int(hhmmss[0:2]) * 60 + int(hhmmss[2:4])


def _hhmmss_to_hhmm(hhmmss: str) -> str:
    return f"{hhmmss[0:2]}:{hhmmss[2:4]}"


def _minutes_to_hhmm(total: int) -> str:
    total %= 24 * 60
    return f"{total // 60:02d}:{total % 60:02d}"


def _minutes_to_hhmmss(total: int) -> int:
    total %= 24 * 60
    return (total // 60) * 10000 + (total % 60) * 100


def _parse_end_label(value: str) -> str:
    text = str(value)
    if _HHMM_RE.fullmatch(text) is None:
        raise ValueError(f"Invalid stop_label: {value!r}")
    return text


def session_end_label_window(session: str) -> tuple[str, str]:
    """Return the first and last Toss end-label ("HH:MM") of an NXT session.

    Toss labels a bar by its end minute while stored NXT bars use the start minute, so a session's Toss window is one
    minute later than its stored window at both ends: for the evening 15:41 through 20:00, for the premarket 08:01
    through 08:50. Derived from the market-session constants so the two conventions cannot drift apart.

    Args:
        session: `nxt_aftermarket` or `nxt_premarket`.

    Returns:
        (first end-label, last end-label).

    Raises:
        ValueError: Unknown session.
    """
    if session == INTRADAY_SESSION_NXT_AFTERMARKET:
        floor, ceil = NXT_AFTERMARKET_HOUR_FLOOR, NXT_AFTERMARKET_HOUR_CEIL
    elif session == INTRADAY_SESSION_NXT_PREMARKET:
        floor, ceil = NXT_PREMARKET_HOUR_FLOOR, NXT_PREMARKET_HOUR_CEIL
    else:
        raise ValueError(f"Unknown session: {session!r}")
    first = _minutes_to_hhmm(_hhmmss_to_minutes(str(floor)) + 1)
    last = _hhmmss_to_hhmm(str(ceil))
    return (first, last)


@dataclass(frozen=True)
class TossWindow:
    candles: tuple[dict[str, Any], ...]
    calls: int
    reached_stop: bool
    error: str | None


def _seoul_now() -> datetime:
    return datetime.now(SEOUL)


def _parse_before(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Invalid before: {value!r}") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Invalid before: {value!r}")
    return parsed.astimezone(SEOUL)


def _parse_stop_day(value: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Invalid stop_day: {value!r}") from None


def _candle_timestamp(candle: Any) -> str | None:
    """Return a valid ISO timestamp, or None for a malformed vendor bar (dropped once, at merge time)."""
    if not isinstance(candle, dict) or candle.get("timestamp") is None:
        return None
    ts = str(candle["timestamp"])
    try:
        datetime.fromisoformat(ts)
    except ValueError:
        return None
    return ts


def _split_timestamp(ts: str) -> tuple[str, str]:
    """Split an ISO timestamp into (date ISO, HH:MM end-label)."""
    try:
        parsed = datetime.fromisoformat(str(ts)).astimezone(SEOUL)
    except ValueError:
        raise ValueError(f"Invalid candle timestamp: {ts!r}") from None
    return (parsed.date().isoformat(), f"{parsed.hour:02d}:{parsed.minute:02d}")


def _stop_reached(oldest_day: str, oldest_label: str, stop_day: str, stop_label: str) -> bool:
    if oldest_day < stop_day:
        return True
    return oldest_day == stop_day and oldest_label <= stop_label


async def fetch_toss_window(
    toss: Any,
    http_session: Any,
    symbol: str,
    *,
    before: str,
    stop_day: str,
    stop_label: str,
    on_page: Callable[[dict[str, Any] | None, datetime, datetime, int], None] | None = None,
) -> TossWindow:
    """Page backwards through Toss 1m candles until a stop boundary is proven or the call budget is spent.

    Toss returns newest-first pages and its `before` cursor is inclusive, so consecutive pages overlap by one bar and are
    de-duplicated by timestamp. The stop boundary is a proof of coverage, not a bar count: the walk ends only when the
    oldest bar reaches `stop_label` on `stop_day` (or earlier), when a page is shorter than requested (vendor history
    exhausted), or when a page is empty.

    Args:
        toss: Client exposing `get_candles`.
        http_session: Open HTTP session.
        symbol: Six-digit KRX code.
        before: ISO-8601 inclusive upper bound of the first page.
        stop_day: ISO date whose `stop_label` must be reached.
        stop_label: `HH:MM` end-label of the earliest bar that must be present.
        on_page: Observer receiving every raw response (payload or None on transport failure), request start, receipt
            time and page index, called before any interpretation so evidence is retained even for error bodies.

    Returns:
        The de-duplicated window and whether the stop boundary was proven. `error` is set to `vendor_error:<code>`,
        `transport:<ExceptionName>` or `call_budget` when the walk could not finish; candles gathered so far are kept.

    Raises:
        ValueError: Malformed `before`, `stop_day` or `stop_label`.
    """
    cursor = _parse_before(before)
    stop = _parse_stop_day(stop_day)
    stop_text = _parse_end_label(stop_label)
    stop_day_text = stop.isoformat()
    stop_dt = datetime.fromisoformat(f"{stop_day_text}T{stop_text}:00+09:00")

    merged: dict[str, dict[str, Any]] = {}
    calls = 0
    error: str | None = None
    reached_stop = False
    exhausted = False
    current_before = str(before)
    count = TOSS_CANDLE_MAX_COUNT

    for page_index in range(TOSS_PAIR_MAX_CALLS):
        started = _seoul_now()
        try:
            payload = await toss.get_candles(
                http_session, symbol, interval="1m", count=int(count), before=current_before, adjusted=False
            )
            received = _seoul_now()
        except Exception as exc:  # noqa: BLE001 - transport envelope, never propagates
            received = _seoul_now()
            if on_page is not None:
                on_page(None, started, received, page_index)
            error = f"transport:{type(exc).__name__}"
            calls += 1
            break
        calls += 1
        if on_page is not None:
            on_page(payload if isinstance(payload, dict) else None, started, received, page_index)
        if isinstance(payload, dict) and "error" in payload:
            err = payload.get("error")
            code = err.get("code") if isinstance(err, dict) else None
            error = f"vendor_error:{code}" if code else "vendor_error:unknown"
            break
        result = (payload.get("result") or {}) if isinstance(payload, dict) else {}
        page_candles = list(result.get("candles") or [])
        if not page_candles:
            exhausted = True
            break
        for candle in page_candles:
            ts = _candle_timestamp(candle)
            if ts is None:
                continue
            if ts not in merged:
                merged[ts] = dict(candle)
        ordered_ts = sorted(merged.keys())
        oldest_ts = ordered_ts[0]
        oldest_day, oldest_label = _split_timestamp(oldest_ts)
        if _stop_reached(oldest_day, oldest_label, stop_day_text, stop_text):
            reached_stop = True
            break
        if len(page_candles) < int(count):
            exhausted = True
            break
        oldest_dt = datetime.fromisoformat(str(oldest_ts)).astimezone(SEOUL)
        diff_minutes = int((oldest_dt - stop_dt).total_seconds() // 60)
        count = min(TOSS_CANDLE_MAX_COUNT, max(1, diff_minutes + 2))
        current_before = str(oldest_ts)
    else:
        pass

    if not reached_stop and error is None and not exhausted and calls >= TOSS_PAIR_MAX_CALLS:
        if merged:
            error = "call_budget"
    ordered = tuple(merged[key] for key in sorted(merged.keys()))
    return TossWindow(candles=ordered, calls=calls, reached_stop=reached_stop, error=error)


def toss_session_frame(
    candles: Sequence[Mapping[str, Any]],
    symbol: str,
    session_date: str,
    session: str,
) -> pd.DataFrame:
    """Convert Toss end-labeled candles of one NXT session into the canonical start-labeled traded-bars frame.

    Only bars dated `session_date` inside the session's end-labeled window are kept. Zero-volume bars are Toss's
    forward-filled placeholders (flat OHLC at the previous close), not trades, and are dropped so the frame has the same
    sparse traded-minutes shape as Kiwoom/KIS NXT partitions. Labels move one minute earlier to the stored start-label
    convention, and traded value is close * volume as for every non-KIS vendor.

    Args:
        candles: Toss candle dicts (timestamp, open/high/low/closePrice, volume).
        symbol: Six-digit KRX code.
        session_date: ISO date of the session.
        session: `nxt_aftermarket` or `nxt_premarket`.

    Returns:
        Canonical bars (vendor "toss", has_trade True) sorted by ts_hms; empty canonical frame when nothing traded.

    Raises:
        ValueError: Unknown session or a candle missing required Toss fields.
    """
    first_label, last_label = session_end_label_window(session)
    try:
        target = date.fromisoformat(str(session_date)).isoformat()
    except ValueError:
        raise ValueError(f"Invalid session_date: {session_date!r}") from None
    code = str(symbol).zfill(6)

    kept: list[dict[str, Any]] = []
    for candle in candles:
        if not isinstance(candle, dict):
            raise ValueError(f"Missing required Toss fields in candle for {code}")
        missing = [key for key in _TOSS_REQUIRED if key not in candle]
        if missing:
            raise ValueError(f"Missing required Toss fields {missing} for {code}")
        day, label = _split_timestamp(str(candle["timestamp"]))
        if day != target:
            continue
        if not (first_label <= label <= last_label):
            continue
        try:
            volume = int(float(str(candle["volume"])))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid Toss volume for {code}: {candle.get('volume')!r}") from None
        if volume <= 0:
            continue
        kept.append(dict(candle))

    frame = normalize_bar_frame(pd.DataFrame(kept), "toss", target, code)
    if frame.empty:
        assert_canonical_bars(frame)
        return frame

    shifted = frame.copy()
    ts_vals = pd.to_numeric(shifted["ts_hms"], errors="coerce").astype("Int64")
    minutes = (ts_vals // 10000) * 60 + ((ts_vals % 10000) // 100)
    shifted_minutes = ((minutes - 1) % (24 * 60)).astype("int64")
    shifted["ts_hms"] = (((shifted_minutes // 60) * 100 + (shifted_minutes % 60)) * 100).astype("int32")
    shifted = shifted.drop_duplicates(subset=["symbol", "ts_hms"], keep="last")

    if session == INTRADAY_SESSION_NXT_AFTERMARKET:
        floor_hms, ceil_hms = NXT_AFTERMARKET_HOUR_FLOOR, NXT_AFTERMARKET_HOUR_CEIL
    else:
        floor_hms, ceil_hms = NXT_PREMARKET_HOUR_FLOOR, NXT_PREMARKET_HOUR_CEIL
    floor = int(str(floor_hms))
    # ceil start-label as HHMMSS int (e.g. 195900 / 084900)
    ceil_val = int(_minutes_to_hhmmss(_hhmmss_to_minutes(str(ceil_hms)) - 1))
    shifted = shifted[(shifted["ts_hms"] >= floor) & (shifted["ts_hms"] <= ceil_val)]
    shifted = shifted.sort_values("ts_hms", kind="stable").reset_index(drop=True)
    assert_canonical_bars(shifted)
    return shifted[list(normalize_bar_frame(pd.DataFrame([]), "toss", target, code).columns)]


@dataclass(frozen=True)
class TossPairResult:
    evening: tuple[pd.DataFrame, CoverageEntry]
    premarket: tuple[pd.DataFrame, CoverageEntry]
    calls: int
    failed: bool


def _krx_close_end_label() -> str:
    return _hhmmss_to_hhmm(str(KRX_REGULAR_HOUR_CEIL))


def _candle_day_label(candle: Mapping[str, Any]) -> tuple[str, str]:
    return _split_timestamp(str(candle.get("timestamp")))


def _has_any_bar(candles: Sequence[Mapping[str, Any]], day: str, session: str) -> bool:
    first_label, last_label = session_end_label_window(session)
    for candle in candles:
        candle_day, candle_label = _candle_day_label(candle)
        if candle_day == day and first_label <= candle_label <= last_label:
            return True
    return False


def _empty_frame(day: str, code: str) -> pd.DataFrame:
    return normalize_bar_frame(pd.DataFrame([]), "toss", day, code)


def _terminal_entry(
    *,
    symbol: str,
    session: str,
    status: CaptureStatus,
    rows: int,
    reason: str,
    refs: Sequence[Any],
) -> CoverageEntry:
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.MINUTE_BARS,
        venue="NXT",
        session=session,
        scheduled_at=None,
        status=status,
        rows=int(rows),
        first_event_time=None,
        last_event_time=None,
        reason=reason,
        raw_refs=tuple(refs),
    )


async def fetch_toss_overnight_pair(
    toss: Any,
    http_session: Any,
    symbol: str,
    entry_day: str,
    next_day: str,
    *,
    store: CaptureStore,
    run_id: str,
) -> TossPairResult:
    """Fetch and classify one symbol's NXT evening (entry day) and premarket (next trading day) in at most three calls.

    A long-hold candidate bought at the entry-day close is exposed to the evening tape and the next premarket. Toss
    returns both in one contiguous walk from the next-day premarket close backwards, so a single window serves both
    sessions. Classification per session: no bar in the session window and traversal proven -> NOT_APPLICABLE
    (`nxt_not_listed`); bars but no traded volume -> NO_TRADES; traded bars -> COMPLETE. An unproven traversal or any
    vendor/transport error is never certified: it yields PARTIAL/FAILED with the raw evidence attached so the ledger retries it.

    Args:
        toss: Client exposing `get_candles`.
        http_session: Open HTTP session.
        symbol: Six-digit KRX code.
        entry_day: ISO entry trading day T; must be within [NXT_START_DATE, KRX_AFTERMARKET_START_DATE).
        next_day: ISO next trading day T+1 (from the trading calendar, never guessed).
        store: Capture store; every raw response is appended verbatim.
        run_id: Acquisition identity for the raw context.

    Returns:
        Frames and coverage entries for both sessions.

    Raises:
        ValueError: entry_day outside the valid range or next_day not after entry_day.
        OSError: Raw evidence persistence fails (fail loud; nothing is certified).
    """
    try:
        entry = date.fromisoformat(str(entry_day))
    except ValueError:
        raise ValueError(f"Invalid entry_day: {entry_day!r}") from None
    try:
        nxt = date.fromisoformat(str(next_day))
    except ValueError:
        raise ValueError(f"Invalid next_day: {next_day!r}") from None
    if not (NXT_START_DATE <= entry.isoformat() < KRX_AFTERMARKET_START_DATE):
        raise ValueError(f"entry_day outside Toss consolidated-free range: {entry_day!r}")
    if not (nxt > entry):
        raise ValueError(f"next_day must be after entry_day: {next_day!r}")
    if not str(run_id).strip():
        raise ValueError("run_id must be nonempty")
    code = str(symbol).zfill(6)

    evening_first, _ = session_end_label_window(INTRADAY_SESSION_NXT_AFTERMARKET)
    _, premarket_last = session_end_label_window(INTRADAY_SESSION_NXT_PREMARKET)
    before = f"{nxt.isoformat()}T{premarket_last}:00.000+09:00"

    context = CaptureContext(
        trading_date=entry,
        run_id=str(run_id),
        dataset=CaptureDataset.MINUTE_BARS,
        vendor="toss",
        endpoint="candles",
        symbol=code,
        venue="NXT",
        session=INTRADAY_SESSION_NXT_AFTERMARKET,
        capture_reason="toss-overnight",
        cohort_id=None,
        scheduled_at=None,
    )
    refs: list[Any] = []

    def _on_page(
        payload: dict[str, Any] | None,
        started: datetime,
        received: datetime,
        page_index: int,
    ) -> None:
        body = dict(payload) if isinstance(payload, dict) else None
        response = CapturedResponse(
            context=context,
            request_started_at=started,
            received_at=received,
            payload=body,
            status=CaptureStatus.COMPLETE if body is not None else CaptureStatus.FAILED,
            source_timestamp=None,
            source_published_at=None,
            page_index=int(page_index),
            attempt_index=0,
            continuation={},
            error_type=None,
        )
        refs.append(store.append_response(response))

    window = await fetch_toss_window(
        toss,
        http_session,
        code,
        before=before,
        stop_day=entry.isoformat(),
        stop_label=evening_first,
        on_page=_on_page,
    )

    if window.error is not None and window.error != "call_budget":
        reason = f"toss:{window.error}"
        evening_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_AFTERMARKET,
            status=CaptureStatus.FAILED, rows=0, reason=reason, refs=refs,
        )
        premarket_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_PREMARKET,
            status=CaptureStatus.FAILED, rows=0, reason=reason, refs=refs,
        )
        return TossPairResult(
            evening=(_empty_frame(entry.isoformat(), code), evening_entry),
            premarket=(_empty_frame(nxt.isoformat(), code), premarket_entry),
            calls=window.calls,
            failed=True,
        )

    if len(window.candles) == 0:
        evening_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_AFTERMARKET,
            status=CaptureStatus.NOT_APPLICABLE, rows=0, reason="toss_empty", refs=refs,
        )
        premarket_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_PREMARKET,
            status=CaptureStatus.NOT_APPLICABLE, rows=0, reason="toss_empty", refs=refs,
        )
        return TossPairResult(
            evening=(_empty_frame(entry.isoformat(), code), evening_entry),
            premarket=(_empty_frame(nxt.isoformat(), code), premarket_entry),
            calls=window.calls,
            failed=False,
        )

    evening_frame = toss_session_frame(window.candles, code, entry.isoformat(), INTRADAY_SESSION_NXT_AFTERMARKET)
    premarket_frame = toss_session_frame(window.candles, code, nxt.isoformat(), INTRADAY_SESSION_NXT_PREMARKET)

    krx_close = _krx_close_end_label()
    traversal_proof = window.reached_stop or any(
        (candle_day == entry.isoformat() and candle_label <= krx_close)
        for candle_day, candle_label in (_candle_day_label(c) for c in window.candles)
    )

    premarket_first, _ = session_end_label_window(INTRADAY_SESSION_NXT_PREMARKET)
    # 벤더 오류·호출 한도 소진(call_budget)이 아닌데 정지 경계에 못 미쳤다면 벤더 이력이 소진된 것이다
    exhausted = window.error is None and not window.reached_stop
    earlier_than_premarket = False
    for candle in window.candles:
        candle_day, candle_label = _candle_day_label(candle)
        if candle_day < nxt.isoformat() or (candle_day == nxt.isoformat() and candle_label < premarket_first):
            earlier_than_premarket = True
            break
    premarket_proof = earlier_than_premarket or exhausted or window.reached_stop

    evening_has_any = _has_any_bar(window.candles, entry.isoformat(), INTRADAY_SESSION_NXT_AFTERMARKET)
    premarket_has_any = _has_any_bar(window.candles, nxt.isoformat(), INTRADAY_SESSION_NXT_PREMARKET)

    if not traversal_proof or not window.reached_stop:
        evening_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_AFTERMARKET,
            status=CaptureStatus.PARTIAL, rows=0, reason="incomplete_window", refs=refs,
        )
        evening_out = _empty_frame(entry.isoformat(), code)
    elif not evening_has_any:
        evening_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_AFTERMARKET,
            status=CaptureStatus.NOT_APPLICABLE, rows=0, reason="nxt_not_listed", refs=refs,
        )
        evening_out = _empty_frame(entry.isoformat(), code)
    elif evening_frame.empty:
        evening_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_AFTERMARKET,
            status=CaptureStatus.NO_TRADES, rows=0, reason="no_trades_in_window", refs=refs,
        )
        evening_out = _empty_frame(entry.isoformat(), code)
    else:
        evening_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_AFTERMARKET,
            status=CaptureStatus.COMPLETE, rows=len(evening_frame),
            reason=f"complete:traded={len(evening_frame)}", refs=refs,
        )
        evening_out = evening_frame

    if not premarket_proof:
        premarket_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_PREMARKET,
            status=CaptureStatus.PARTIAL, rows=0, reason="incomplete_window", refs=refs,
        )
        premarket_out = _empty_frame(nxt.isoformat(), code)
    elif not premarket_has_any:
        premarket_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_PREMARKET,
            status=CaptureStatus.NOT_APPLICABLE, rows=0, reason="nxt_not_listed", refs=refs,
        )
        premarket_out = _empty_frame(nxt.isoformat(), code)
    elif premarket_frame.empty:
        premarket_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_PREMARKET,
            status=CaptureStatus.NO_TRADES, rows=0, reason="no_trades_in_window", refs=refs,
        )
        premarket_out = _empty_frame(nxt.isoformat(), code)
    else:
        premarket_entry = _terminal_entry(
            symbol=code, session=INTRADAY_SESSION_NXT_PREMARKET,
            status=CaptureStatus.COMPLETE, rows=len(premarket_frame),
            reason=f"complete:traded={len(premarket_frame)}", refs=refs,
        )
        premarket_out = premarket_frame

    # 미완결 창(call_budget 등)은 비종료(PARTIAL)로 재시도되지만 벤더 장애가 아니므로 연속 오류 차단기에 세지 않는다.
    return TossPairResult(
        evening=(evening_out, evening_entry),
        premarket=(premarket_out, premarket_entry),
        calls=window.calls,
        failed=False,
    )
