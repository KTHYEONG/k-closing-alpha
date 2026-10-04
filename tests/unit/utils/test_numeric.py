from __future__ import annotations

import pytest

from src.utils.numeric import safe_float


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1,234.5", 1234.5), ("-0.25", -0.25), (7, 7.0), (" 12 ", 12.0), ("1,000,000", 1_000_000.0)],
)
def test_safe_float_parses_vendor_numeric_strings(raw: object, expected: float) -> None:
    assert safe_float(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "abc", "1.2.3", object()])
def test_safe_float_returns_caller_default_when_unparseable(raw: object) -> None:
    assert safe_float(raw) == 0.0
    assert safe_float(raw, default=-1.0) == -1.0


def test_safe_float_preserves_default_identity_type() -> None:
    sentinel = 0
    result = safe_float(None, default=sentinel)
    assert result == 0
    assert type(result) is int


def test_safe_float_reexported_from_collect_is_same_object() -> None:
    from src.daily import collect

    assert collect.safe_float is safe_float
