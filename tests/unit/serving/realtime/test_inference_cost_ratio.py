"""Measured round-trip cost ratio contract (46bp decomposed)."""
from __future__ import annotations


def test_round_trip_cost_ratio_is_measured_46bp_decomposed() -> None:
    import pytest

    from src.serving.realtime import inference

    # Given / When: module-level constants
    statutory = inference._STATUTORY_COST_RATIO
    spread = inference._SPREAD_COST_RATIO

    # Then: measured round-trip cost, decomposed (no magic number)
    assert statutory == pytest.approx(0.0020)
    assert spread == pytest.approx(0.0026)
    assert inference.ROUND_TRIP_COST_RATIO == pytest.approx(statutory + spread)  # noqa: SIM300
    assert inference.ROUND_TRIP_COST_RATIO == pytest.approx(0.0046)  # noqa: SIM300
