"""Invariant scenarios for the price-panel adjustment reference."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backfill.intraday.price_basis import PriceReference


def _ref(rows: list[tuple[str, str, float, float]]) -> PriceReference:
    return PriceReference.from_price_history(pd.DataFrame(rows, columns=["date", "symbol", "close", "close_raw"]))


def test_reference_distinguishes_adjusted_raw_and_unknown_days() -> None:
    ref = _ref([
        ("2025-09-29", "5930", 84200.0, 84200.0),
        ("2025-09-29", "196170", 358565.0, 466000.0),
        ("2025-09-29", "000001", 1000.0, float("nan")),
        ("2025-09-29", "000002", 0.0, 0.0),
    ])
    assert ref.is_known("2025-09-29", "005930") and not ref.is_adjusted("2025-09-29", "005930")
    assert ref.is_known("2025-09-29", "196170") and ref.is_adjusted("2025-09-29", "196170")
    assert not ref.is_known("2025-09-29", "000001") and not ref.is_adjusted("2025-09-29", "000001")
    assert not ref.is_known("2025-09-29", "000002")
    assert not ref.is_known("2025-09-30", "196170")


def test_reference_accepts_timestamp_dates() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-03-02"]), "symbol": ["000001"], "close": [5300.0], "close_raw": [1060.0],
    })
    assert PriceReference.from_price_history(frame).is_adjusted("2026-03-02", "000001")


def test_reference_rejects_missing_columns_and_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        PriceReference.from_price_history(pd.DataFrame({"date": [], "symbol": [], "close": []}))
    with pytest.raises(ValueError, match="Duplicate"):
        _ref([("2025-09-29", "000001", 1.0, 1.0), ("2025-09-29", "1", 1.0, 1.0)])
