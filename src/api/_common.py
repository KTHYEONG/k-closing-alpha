"""Vendor-neutral helpers shared by the Kiwoom, LS and Toss REST clients.

Only pure, behavior-identical helpers live here; per-vendor auth classification, rate-limit detection and
return shapes stay in each client because they encode vendor contracts.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from src.data.capture_contracts import ChartBudget

SEOUL_TZ: ZoneInfo = ZoneInfo("Asia/Seoul")
"""Asia/Seoul; KRX session clocks, vendor expiry stamps and page timestamps are all KST."""


def now_seoul() -> datetime:
    """Return the current aware Asia/Seoul time.

    Single clock read point for page `started`/`received` stamps and deadline arithmetic in the broker clients.
    """
    return datetime.now(SEOUL_TZ)


def validate_target_ymd(target_date: str) -> str:
    """Normalize a market date to YYYYMMDD.

    Args:
        target_date: "YYYY-MM-DD" or "YYYYMMDD" (any '-' characters are stripped before parsing).

    Returns:
        The 8-digit YYYYMMDD string.

    Raises:
        ValueError: The stripped value is not a valid calendar date; message is
            "invalid target date: <repr(target_date)>" and the strptime cause is suppressed.
    """
    ymd = str(target_date).replace("-", "")
    try:
        datetime.strptime(ymd, "%Y%m%d")
    except ValueError:
        raise ValueError(f"invalid target date: {target_date!r}") from None
    return ymd


def resolve_chart_budget(budget: ChartBudget | None, default_pages: int) -> tuple[int, datetime | None]:
    """Resolve the page cap and absolute deadline for one chart sweep.

    Args:
        budget: Explicit budget, or None to fall back to `default_pages` with no deadline.
        default_pages: Configured page cap used when `budget` is None.

    Returns:
        (max_pages, deadline): max_pages is clamped to at least 1; deadline is `budget.deadline` or None.
    """
    if budget is None:
        return max(1, int(default_pages)), None
    return max(1, int(budget.max_pages)), budget.deadline


def deadline_remaining(deadline: datetime | None) -> float | None:
    """Seconds until `deadline` measured on `now_seoul()`; None when there is no deadline.

    A non-positive result means the deadline has passed; callers stop before issuing another request.
    """
    if deadline is None:
        return None
    return (deadline - now_seoul()).total_seconds()


def parse_expires_in(raw: object) -> float | None:
    """Parse an OAuth `expires_in` field into seconds, accepting exactly what LS/Toss historically accepted.

    Vendors send the lifetime as a JSON number or a numeric string. Anything unparsable yields None, which the
    shared token store treats as "no declared expiry".

    Args:
        raw: The decoded `expires_in` value (any JSON type, or None when absent).

    Returns:
        float seconds, or None. Booleans are rejected (JSON true/false must not become 1.0/0.0).
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str) and raw.strip().lstrip("+-").replace(".", "", 1).isdigit():
        try:
            return float(raw.strip())
        except ValueError:
            return None
    return None
