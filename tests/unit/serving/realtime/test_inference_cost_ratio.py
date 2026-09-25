"""Measured round-trip cost ratio contract (46.73bp decomposed)."""
from __future__ import annotations


def test_round_trip_cost_ratio_is_measured_46bp_decomposed() -> None:
    import pytest

    from src.execution import cost_model
    from src.serving.realtime import inference

    # Given both modules; When compared
    # Then: serving ratio is the cost_model constant (same object)
    assert inference.ROUND_TRIP_COST_RATIO is cost_model.ROUND_TRIP_COST_RATIO
    assert inference.ROUND_TRIP_COST_RATIO == 0.004672792
    assert pytest.approx(
        cost_model.STATUTORY_COST_BP / 1e4
        + cost_model.LEGACY_FLAT_SPREAD_COST_BP / 1e4
        + cost_model.BROKERAGE_FEE_BP / 1e4
    ) == inference.ROUND_TRIP_COST_RATIO
