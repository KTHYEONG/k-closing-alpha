"""Auto-generated scenario tests (see docs/specs/closing_alpha_architecture_v2_contract.json)."""
from __future__ import annotations


def test_simulate_passive_entry_fills_only_when_window_low_touches_limit() -> None:
    import numpy as np
    import pandas as pd

    from src.execution.passive_fill import simulate_passive_entry

    # 10,000원 -> KRX tick 10원; a 1-tick passive limit sits at 9,990
    bars = pd.DataFrame({
        "symbol": ["000001", "000001", "000002", "000002"],
        "ts_hms": [152000, 152500, 152000, 152500],
        "low":    [9995.0, 9985.0, 9995.0, 9996.0],
        "high":   [10010.0, 10005.0, 10010.0, 10005.0],
        "close":  [10000.0, 10000.0, 10000.0, 10000.0],
    })
    entries = pd.DataFrame({"symbol": ["000001", "000002"], "close_price": [10000.0, 10000.0]})

    out = simulate_passive_entry(entries, bars, offset_ticks=1, window_start_hms=151900, window_end_hms=153000)

    # 000001 traded to 9,985 <= 9,990 -> filled at the limit; 000002 never did
    assert out["entry_filled"].tolist() == [True, False]
    assert np.isclose(out["entry_price"].iloc[0], 9990.0)
    assert not np.isfinite(out["entry_price"].iloc[1])
    # filling 1 tick better than the close is worth 10bp on a 10,000원 name
    assert np.isclose(out["entry_saving_bp"].iloc[0], 10.0)


def test_simulate_passive_exit_fills_only_when_window_high_touches_limit() -> None:
    import numpy as np
    import pandas as pd

    from src.execution.passive_fill import simulate_passive_exit

    bars = pd.DataFrame({
        "symbol": ["000001", "000001", "000002", "000002"],
        "ts_hms": [90100, 90500, 90100, 90500],
        "low":    [10090.0, 10095.0, 10090.0, 10090.0],
        "high":   [10105.0, 10120.0, 10105.0, 10102.0],
        "close":  [10100.0, 10110.0, 10100.0, 10095.0],
    })
    exits = pd.DataFrame({"symbol": ["000001", "000002"], "next_open": [10100.0, 10100.0]})

    out = simulate_passive_exit(exits, bars, offset_ticks=1, window_start_hms=90000, window_end_hms=93000)

    # 10,100원 -> tick 10원 -> limit 10,110; 000001 reached 10,120, 000002 topped at 10,105
    assert out["exit_filled"].tolist() == [True, False]
    assert np.isclose(out["exit_price"].iloc[0], 10110.0)
    # unfilled positions fall back to the last observed price in the window, never to NaN
    assert np.isfinite(out["exit_price"].iloc[1])
    assert np.isclose(out["exit_price"].iloc[1], 10095.0)


def test_measure_execution_profile_reports_fill_rate_and_adverse_selection() -> None:
    import numpy as np
    import pandas as pd

    from src.execution.passive_fill import measure_execution_profile

    # 4 names: the two that fill happen to be the two with the worse overnight gap
    panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-01-02"] * 4),
        "symbol": ["000001", "000002", "000003", "000004"],
        "close_price": [10000.0] * 4,
        "mechanical_gross": [0.000, 0.002, 0.020, 0.030],
        "entry_filled": [True, True, False, False],
        "entry_saving_bp": [10.0, 10.0, np.nan, np.nan],
    })

    prof = measure_execution_profile(panel, gross_col="mechanical_gross")

    assert np.isclose(prof["fill_rate"], 0.5)
    assert np.isclose(prof["mean_saving_bp"], 10.0)
    # filled mean 10bp vs pool mean 130bp -> adverse selection of -120bp swamps the 10bp saving
    assert np.isclose(prof["filled_gross_bp"], 10.0)
    assert np.isclose(prof["pool_gross_bp"], 130.0)
    assert np.isclose(prof["adverse_selection_bp"], -120.0)
    assert prof["saving_survives_adverse_selection"] is False
    assert prof["fill_rate_is_upper_bound"] is True


def test_execution_profile_rejects_unknown_mode() -> None:
    import pytest

    from src.execution.passive_fill import INCUMBENT_CROSSING_PROFILE, ExecutionProfile

    assert INCUMBENT_CROSSING_PROFILE.entry_mode == "cross"
    assert INCUMBENT_CROSSING_PROFILE.exit_mode == "cross"
    assert INCUMBENT_CROSSING_PROFILE.round_trip_ticks == 2.0

    passive = ExecutionProfile(entry_mode="passive", exit_mode="passive", entry_offset_ticks=1, exit_offset_ticks=1)
    assert passive.round_trip_ticks == 0.0

    with pytest.raises(ValueError, match="entry_mode"):
        ExecutionProfile(entry_mode="iceberg", exit_mode="cross")
    with pytest.raises(ValueError, match="offset_ticks"):
        ExecutionProfile(entry_mode="passive", exit_mode="cross", entry_offset_ticks=-1)


def test_passive_fill_fail_closed_edges() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.execution.passive_fill import (
        ExecutionProfile,
        measure_execution_profile,
        simulate_passive_entry,
        simulate_passive_exit,
    )

    with pytest.raises(ValueError, match="exit_mode"):
        ExecutionProfile(entry_mode="cross", exit_mode="iceberg")
    with pytest.raises(ValueError, match="offset_ticks"):
        ExecutionProfile(entry_mode="cross", exit_mode="cross", exit_offset_ticks=-2)
    with pytest.raises(ValueError, match="offset_ticks"):
        simulate_passive_entry(
            pd.DataFrame({"symbol": ["000001"], "close_price": [10000.0]}),
            pd.DataFrame({"symbol": ["000001"], "ts_hms": [152000], "low": [9000.0], "high": [10010.0], "close": [10000.0]}),
            offset_ticks=-1,
        )
    with pytest.raises(ValueError, match="offset_ticks"):
        simulate_passive_exit(
            pd.DataFrame({"symbol": ["000001"], "next_open": [10100.0]}),
            pd.DataFrame({"symbol": ["000001"], "ts_hms": [90100], "high": [10120.0], "low": [10090.0], "close": [10100.0]}),
            offset_ticks=-1,
        )
    with pytest.raises(ValueError, match="missing"):
        simulate_passive_entry(
            pd.DataFrame({"symbol": ["000001"]}),
            pd.DataFrame({"symbol": ["000001"], "ts_hms": [152000], "low": [9000.0], "high": [10010.0], "close": [10000.0]}),
            offset_ticks=1,
        )
    with pytest.raises(ValueError, match="missing"):
        simulate_passive_entry(
            pd.DataFrame({"symbol": ["000001"], "close_price": [10000.0]}),
            pd.DataFrame({"symbol": ["000001"], "ts_hms": [152000]}),
            offset_ticks=1,
        )
    with pytest.raises(ValueError, match="missing"):
        simulate_passive_exit(
            pd.DataFrame({"symbol": ["000001"]}),
            pd.DataFrame({"symbol": ["000001"], "ts_hms": [90100], "high": [10120.0], "low": [10090.0], "close": [10100.0]}),
            offset_ticks=1,
        )
    with pytest.raises(ValueError, match="missing"):
        simulate_passive_exit(
            pd.DataFrame({"symbol": ["000001"], "next_open": [10100.0]}),
            pd.DataFrame({"symbol": ["000001"], "ts_hms": [90100]}),
            offset_ticks=1,
        )

    # No bars inside the window: entry stays unfilled with NaN price/saving
    out = simulate_passive_entry(
        pd.DataFrame({"symbol": ["000001"], "close_price": [10000.0]}),
        pd.DataFrame({"symbol": ["000001"], "ts_hms": [140000], "low": [9000.0], "high": [10010.0], "close": [10000.0]}),
        offset_ticks=1,
    )
    assert out["entry_filled"].tolist() == [False]
    assert not np.isfinite(out["entry_price"].iloc[0])
    # Exit with no bars falls back to next_open, never NaN
    outx = simulate_passive_exit(
        pd.DataFrame({"symbol": ["000001"], "next_open": [10100.0]}),
        pd.DataFrame({"symbol": ["000001"], "ts_hms": [80000], "high": [10120.0], "low": [10090.0], "close": [10100.0]}),
        offset_ticks=1,
    )
    assert outx["exit_filled"].tolist() == [False]
    assert np.isclose(outx["exit_price"].iloc[0], 10100.0)

    with pytest.raises(ValueError, match="missing"):
        measure_execution_profile(pd.DataFrame({"a": [1.0]}))

    # An all-NaN gross pool reports NaN adverse selection instead of crashing
    nan_panel = pd.DataFrame({
        "trade_date": pd.to_datetime(["2026-01-02"] * 2),
        "symbol": ["000001", "000002"],
        "mechanical_gross": [np.nan, np.nan],
        "entry_filled": [True, False],
        "entry_saving_bp": [10.0, np.nan],
    })
    nan_prof = measure_execution_profile(nan_panel, gross_col="mechanical_gross")
    assert not np.isfinite(nan_prof["adverse_selection_bp"])
    assert nan_prof["fill_rate_is_upper_bound"] is True
