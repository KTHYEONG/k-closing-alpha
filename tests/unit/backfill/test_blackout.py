"""Invariant guards for the hoisted operator blackout windows."""

from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.backfill.intraday import blackout

_SEOUL = ZoneInfo("Asia/Seoul")


def _kst(day: str, hh: int, mm: int) -> datetime:
    return datetime.fromisoformat(f"{day}T{hh:02d}:{mm:02d}:00+09:00")


def test_wrap_around_and_weekday_rules() -> None:
    """A window crossing midnight resolves as before; weekends are not blackout by default."""
    windows = [(23 * 60, 60)]
    assert blackout.in_blackout(_kst("2026-09-10", 23, 30), windows) is True
    assert blackout.in_blackout(_kst("2026-09-11", 0, 30), windows) is True
    assert blackout.in_blackout(_kst("2026-09-11", 2, 0), windows) is False
    saturday = _kst("2026-09-12", 23, 30)
    assert blackout.in_blackout(saturday, windows) is False
    assert blackout.in_blackout(saturday, windows, weekdays_only=False) is True
    assert blackout.blackout_end(saturday, windows) is None
    assert blackout.blackout_end(saturday, windows, weekdays_only=False) is not None


def test_midnight_end_resolves_like_legacy() -> None:
    """A window ending at 24:00 ends at midnight, exactly as the previous implementation."""
    night = _kst("2026-09-10", 23, 30)
    end = blackout.blackout_end(night, [(23 * 60, 24 * 60)])
    assert end is not None and (end.day, end.hour, end.minute) == (11, 0, 0)
    assert blackout.blackout_end(night, [(9 * 60, 10 * 60)]) is None
    wrap_end = blackout.blackout_end(_kst("2026-09-11", 0, 30), [(23 * 60, 60)])
    assert wrap_end is not None and (wrap_end.day, wrap_end.hour, wrap_end.minute) == (11, 1, 0)


def test_parse_errors_are_loud() -> None:
    """Well-formed HHMM-HHMM parses to minutes; every malformed spelling raises."""
    assert blackout.parse_blackout_windows(("0850-0940",)) == [(8 * 60 + 50, 9 * 60 + 40)]
    assert blackout.parse_blackout_windows(()) == []
    assert blackout.parse_blackout_windows(("2300-2400",)) == [(23 * 60, 24 * 60)]
    for bad in ("8:50", "0850", "0850-0940-1000", "2500-2600", "0860-0940", "0850-2460", "", "abcd-efgh"):
        with pytest.raises(ValueError, match="blackout"):
            blackout.parse_blackout_windows((bad,))


def test_wait_returns_immediately_outside_and_sleeps_once_inside() -> None:
    """Outside every window no sleep happens; inside, one sleep covers the whole blackout."""
    slept: list[float] = []

    async def _sleep(seconds: float) -> None:
        slept.append(seconds)

    asyncio.run(blackout.wait_for_blackout([], now_fn=lambda: _kst("2026-09-10", 12, 0), sleep_fn=_sleep))
    assert slept == []
    calls = {"n": 0}

    def _now() -> datetime:
        calls["n"] += 1
        return _kst("2026-09-10", 9, 0) if calls["n"] < 3 else _kst("2026-09-10", 9, 41)

    asyncio.run(blackout.wait_for_blackout([(8 * 60 + 50, 9 * 60 + 40)], now_fn=_now, sleep_fn=_sleep))
    assert len(slept) == 1 and slept[0] > 0


def test_tape_recovery_delegation_matches_legacy_behavior() -> None:
    """The previous tape-recovery entry points keep their exact behavior through delegation."""
    from src.backfill.intraday import tape_recovery as tr

    assert tr._in_blackout(30, (1380, 60)) is True
    assert tr._in_blackout(120, (1380, 60)) is False
    night = datetime(2026, 9, 10, 23, 30, tzinfo=_SEOUL)
    end = tr._blackout_end(night, [(1380, 60)])
    assert end is not None and (end.day, end.hour, end.minute) == (11, 1, 0)
    assert tr._blackout_end(night, [(540, 600)]) is None
    midnight = tr._blackout_end(night, [(1380, 1440)])
    assert midnight is not None and (midnight.day, midnight.hour, midnight.minute) == (11, 0, 0)
