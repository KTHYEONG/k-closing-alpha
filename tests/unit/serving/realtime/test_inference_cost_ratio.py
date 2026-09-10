"""Measured round-trip cost ratio contract (46.73bp decomposed)."""
from __future__ import annotations


def test_round_trip_cost_ratio_is_measured_46bp_decomposed() -> None:
    import pytest

    from src.serving.realtime import inference

    # Given / When: module-level constants
    statutory = inference._STATUTORY_COST_RATIO
    spread = inference._SPREAD_COST_RATIO
    brokerage = inference._BROKERAGE_FEE_RATIO

    # Then: measured round-trip cost, decomposed (no magic number)
    assert statutory == pytest.approx(0.0020)
    assert spread == pytest.approx(0.0026)
    assert brokerage == pytest.approx(0.000036396 * 2)
    assert inference.ROUND_TRIP_COST_RATIO == pytest.approx(statutory + spread + brokerage)  # noqa: SIM300
    assert inference.ROUND_TRIP_COST_RATIO == pytest.approx(0.004672792)  # noqa: SIM300
