from __future__ import annotations

from datetime import datetime
from datetime import time as clock_time
from datetime import timedelta
from zoneinfo import ZoneInfo

_SEOUL = ZoneInfo("Asia/Seoul")


def _kst(day: int, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 9, 29 + day, hour, minute, second, tzinfo=_SEOUL)


def test_blackout_remaining_zero_outside_windows() -> None:
    from src.tools.deploy_window import blackout_remaining

    assert blackout_remaining(_kst(0, 12, 0)) == timedelta(0)


def test_blackout_remaining_inside_decision_window() -> None:
    from src.tools.deploy_window import blackout_remaining

    now = datetime(2026, 9, 25, 15, 20, 49, tzinfo=_SEOUL)

    assert blackout_remaining(now) == timedelta(minutes=22, seconds=11)


def test_blackout_remaining_inside_paper_exit_window() -> None:
    from src.tools.deploy_window import blackout_remaining

    assert blackout_remaining(_kst(0, 9, 23)) == timedelta(minutes=17)


def test_blackout_half_open_boundaries() -> None:
    from src.tools.deploy_window import blackout_remaining

    assert blackout_remaining(_kst(0, 15, 10)) == timedelta(minutes=33)
    assert blackout_remaining(_kst(0, 15, 43)) == timedelta(0)


def test_blackout_ignores_weekends() -> None:
    from datetime import date

    from src.tools.deploy_window import blackout_remaining

    saturday = date(2026, 10, 3)
    assert saturday.weekday() == 5
    now = datetime(2026, 10, 3, 15, 20, tzinfo=_SEOUL)

    assert blackout_remaining(now) == timedelta(0)


def test_blackout_converts_utc_input() -> None:
    from zoneinfo import ZoneInfo

    from src.tools.deploy_window import blackout_remaining

    utc_now = datetime(2026, 9, 29, 6, 20, tzinfo=ZoneInfo("UTC"))
    kst_now = datetime(2026, 9, 29, 15, 20, tzinfo=_SEOUL)

    assert blackout_remaining(utc_now) == blackout_remaining(kst_now)
    assert blackout_remaining(utc_now) > timedelta(0)


def test_blackout_rejects_naive_datetime() -> None:
    import pytest

    from src.tools.deploy_window import blackout_remaining

    with pytest.raises(ValueError, match="timezone-aware"):
        blackout_remaining(datetime(2026, 9, 29, 15, 20))


def test_windows_derive_from_market_session_constants() -> None:
    import importlib

    import src.config.market_session as market_session
    import src.tools.deploy_window as deploy_window

    original = market_session.DECISION_WINDOW_START_HHMMSS
    market_session.DECISION_WINDOW_START_HHMMSS = "150000"
    try:
        reloaded = importlib.reload(deploy_window)
        assert reloaded.DEPLOY_BLACKOUT_WINDOWS[1][0] == clock_time(14, 50)
    finally:
        market_session.DECISION_WINDOW_START_HHMMSS = original
        importlib.reload(deploy_window)


def test_main_without_wait_exit_code(monkeypatch) -> None:
    import src.tools.deploy_window as deploy_window

    monkeypatch.setattr(deploy_window, "_now", lambda: _kst(0, 15, 20))
    assert deploy_window.main([]) == 3

    monkeypatch.setattr(deploy_window, "_now", lambda: _kst(0, 12, 0))
    assert deploy_window.main([]) == 0


def test_now_returns_timezone_aware_seoul_time() -> None:
    import src.tools.deploy_window as deploy_window

    now = deploy_window._now()

    assert now.tzinfo is not None and now.utcoffset() is not None
    assert now.astimezone(_SEOUL).utcoffset() == now.utcoffset()


def test_main_wait_clears_immediately_outside_window(monkeypatch, caplog) -> None:
    import logging

    import src.tools.deploy_window as deploy_window

    monkeypatch.setattr(deploy_window, "_now", lambda: _kst(0, 12, 0))

    with caplog.at_level(logging.INFO):
        assert deploy_window.main(["--wait"]) == 0

    assert any("stage=deploy_window status=CLEAR" in rec.message for rec in caplog.records)


def test_main_wait_heartbeats_until_blackout_ends(monkeypatch, caplog) -> None:
    import logging
    import time

    import src.tools.deploy_window as deploy_window

    ticks = iter([_kst(0, 15, 20), _kst(0, 15, 44)])
    monkeypatch.setattr(deploy_window, "_now", lambda: next(ticks))
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    with caplog.at_level(logging.INFO):
        assert deploy_window.main(["--wait"]) == 0

    assert sleeps == [60]
    assert any("stage=deploy_window status=WAIT" in rec.message for rec in caplog.records)
    assert any("stage=deploy_window status=CLEAR" in rec.message for rec in caplog.records)
