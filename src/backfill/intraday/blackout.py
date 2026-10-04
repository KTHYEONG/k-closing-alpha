"""Operator blackout windows for jobs that share a host with the decision path: KST minute-of-day windows, wrap-around supported, optional weekday restriction."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any

from src.data.capture_contracts import SEOUL


def _minute_in_window(minute: int, window: tuple[int, int]) -> bool:
    start, end = window
    if end <= start:
        return minute >= start or minute < end
    return start <= minute < end


def parse_blackout_windows(items: Sequence[str]) -> list[tuple[int, int]]:
    """Parse `HHMM-HHMM` KST windows into minute-of-day bounds.

    Args:
        items: Window specs such as `"0850-0940"`; an end of `"2400"` means midnight.

    Returns:
        List of `(start_minute, end_minute)` pairs in input order.

    Raises:
        ValueError: Any item is not a well-formed `HHMM-HHMM` window.
    """
    parsed: list[tuple[int, int]] = []
    for spec in items or []:
        text = str(spec).strip()
        try:
            start_raw, end_raw = text.split("-", 1)
            if len(start_raw) != 4 or len(end_raw) != 4:
                raise ValueError(text)
            if not (start_raw.isdecimal() and end_raw.isdecimal()):
                raise ValueError(text)
            start = int(start_raw[:2]) * 60 + int(start_raw[2:])
            end = int(end_raw[:2]) * 60 + int(end_raw[2:])
            start_ok = 0 <= int(start_raw[:2]) <= 23 and 0 <= int(start_raw[2:]) <= 59
            end_ok = (0 <= int(end_raw[:2]) <= 23 and 0 <= int(end_raw[2:]) <= 59) or end_raw == "2400"
        except ValueError:
            raise ValueError(f"Invalid blackout window: {spec!r} (expected HHMM-HHMM)") from None
        if not (start_ok and end_ok):
            raise ValueError(f"Invalid blackout window: {spec!r} (expected HHMM-HHMM)")
        parsed.append((start, end))
    return parsed


def _as_seoul(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        return now
    return now.astimezone(SEOUL)


def in_blackout(now: datetime, windows: Sequence[tuple[int, int]], *, weekdays_only: bool = True) -> bool:
    """True when `now` falls inside any window; weekends are never blackout when `weekdays_only` is set."""
    moment = _as_seoul(now)
    if weekdays_only and moment.weekday() >= 5:
        return False
    minute = moment.hour * 60 + moment.minute
    return any(_minute_in_window(minute, window) for window in windows)


def blackout_end(
    now: datetime, windows: Sequence[tuple[int, int]], *, weekdays_only: bool = True
) -> datetime | None:
    """End of the blackout containing `now`, or None when `now` is outside every window."""
    moment = _as_seoul(now)
    if weekdays_only and moment.weekday() >= 5:
        return None
    minute = moment.hour * 60 + moment.minute
    ends: list[datetime] = []
    for start, end in windows:
        if not _minute_in_window(minute, (start, end)):
            continue
        if end <= start:
            target = moment.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            target += timedelta(minutes=end)
            if minute < end:
                target = moment.replace(hour=end // 60, minute=end % 60, second=0, microsecond=0)
        elif end >= 24 * 60:
            target = moment.replace(hour=23, minute=59, second=59, microsecond=0) + timedelta(seconds=1)
        else:
            target = moment.replace(hour=end // 60, minute=end % 60, second=0, microsecond=0)
        ends.append(target)
    return max(ends) if ends else None


async def wait_for_blackout(
    windows: Sequence[tuple[int, int]],
    *,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], Any] | None = None,
    weekdays_only: bool = True,
) -> None:
    """Sleep until the containing blackout ends; returns immediately outside every window."""
    clock = now_fn if now_fn is not None else (lambda: datetime.now(SEOUL))
    sleep = sleep_fn if sleep_fn is not None else asyncio.sleep
    while True:
        end = blackout_end(clock(), windows, weekdays_only=weekdays_only)
        if end is None:
            return
        await sleep(max(0.0, (end - clock()).total_seconds()))
