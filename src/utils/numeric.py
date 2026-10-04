"""Vendor-agnostic numeric parsing helpers (no src imports)."""

from __future__ import annotations


def safe_float(value: object, default: float = 0.0) -> float:
    """Parse a vendor numeric field that may be None, blank, or comma-grouped.

    KIS/Kiwoom REST payloads encode numbers as strings with thousands separators and use empty strings or None
    for missing values; callers choose the fallback because 0 is a valid price for some fields and a sentinel for
    others.

    Args:
        value: Raw field value.
        default: Returned unchanged when value is None or not parseable.

    Returns:
        float(str(value) with "," removed), or default.
    """
    if value is None:
        return default
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return default
