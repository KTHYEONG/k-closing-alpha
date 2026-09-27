"""Deploy blackout window: defer VPS convergence while trading decisions run."""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from datetime import time as clock_time
from zoneinfo import ZoneInfo

from src.config import market_session as _session
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)

DEPLOY_BLACKOUT_GUARD_MINUTES: int = 10
DEPLOY_WAIT_HEARTBEAT_SECONDS: int = 60

_SEOUL = ZoneInfo("Asia/Seoul")
_MINUTES_PER_DAY: int = 24 * 60


def _shift_hhmmss(hhmmss: str, *, delta_minutes: int) -> clock_time:
    """Shift an HHMMSS wall-clock by whole minutes, preserving seconds."""
    total = int(hhmmss[0:2]) * 60 + int(hhmmss[2:4]) + delta_minutes
    total %= _MINUTES_PER_DAY
    return clock_time(total // 60, total % 60, int(hhmmss[4:6]))


def _build_windows() -> tuple[tuple[clock_time, clock_time], ...]:
    guard = DEPLOY_BLACKOUT_GUARD_MINUTES
    return (
        (
            _shift_hhmmss(_session.KRX_REGULAR_HOUR_FLOOR, delta_minutes=-guard),
            _shift_hhmmss(_session.PAPER_EXIT_WINDOW_END_HHMMSS, delta_minutes=guard),
        ),
        (
            _shift_hhmmss(_session.DECISION_WINDOW_START_HHMMSS, delta_minutes=-guard),
            _shift_hhmmss(_session.CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS, delta_minutes=guard),
        ),
    )


DEPLOY_BLACKOUT_WINDOWS: tuple[tuple[clock_time, clock_time], ...] = _build_windows()


def blackout_remaining(
    now: datetime,
    *,
    windows: Sequence[tuple[clock_time, clock_time]] = DEPLOY_BLACKOUT_WINDOWS,
) -> timedelta:
    """Return how long a deploy must wait before it may converge the VPS.

    A deploy swaps the image tag and host checkout that every scheduled unit
    reads at start. Converging while the open-auction exit or the closing
    decision chain is running would split one trading day's decision across
    two code versions, so convergence is deferred until the blackout ends.
    Weekends carry no blackout; exchange holidays are treated like weekdays
    because the cost is only a short delay.

    Args:
        now: Timezone-aware current time; converted to Asia/Seoul.
        windows: Half-open [start, end) KST wall-clock windows on Mon-Fri.

    Returns:
        timedelta(0) outside every window, otherwise end-of-window minus now.

    Raises:
        ValueError: now is naive.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    seoul = now.astimezone(_SEOUL)
    if seoul.weekday() >= 5:
        return timedelta(0)
    current = seoul.time()
    for start, end in windows:
        if start <= current < end:
            return datetime.combine(seoul.date(), end, tzinfo=_SEOUL) - seoul
    return timedelta(0)


def _now() -> datetime:
    return datetime.now(_SEOUL)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `python -m src.tools.deploy_window --wait`.

    Without --wait: exit 0 outside a blackout, exit 3 inside one.
    With --wait: sleep until the blackout ends, emitting a heartbeat log
    line every DEPLOY_WAIT_HEARTBEAT_SECONDS, then exit 0.
    """
    parser = argparse.ArgumentParser(description="Deploy blackout gate for VPS convergence")
    parser.add_argument("--wait", action="store_true", help="sleep until the blackout ends")
    args = parser.parse_args(argv)
    remaining = blackout_remaining(_now())
    if not args.wait:
        if remaining > timedelta(0):
            logger.warning(
                "[SYS] stage=deploy_window status=BLACKOUT remaining_s=%d",
                int(remaining.total_seconds()),
            )
            return 3
        logger.info("[SYS] stage=deploy_window status=CLEAR remaining_s=0")
        return 0
    while remaining > timedelta(0):
        logger.info(
            "[SYS] stage=deploy_window status=WAIT remaining_s=%d",
            int(remaining.total_seconds()),
        )
        time.sleep(min(DEPLOY_WAIT_HEARTBEAT_SECONDS, remaining.total_seconds()))
        remaining = blackout_remaining(_now())
    logger.info("[SYS] stage=deploy_window status=CLEAR remaining_s=0")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    raise SystemExit(main())
