"""Auto-generated scenario tests (see docs/specs/closing_alpha_architecture_v2_contract.json)."""
from __future__ import annotations


def test_expected_net_value_subtracts_cost_and_adverse_selection() -> None:
    import numpy as np

    from src.ml.expected_value import expected_net_value

    pred = np.array([0.010, 0.010, 0.002])
    cost = np.array([0.0046, 0.0046, 0.0046])

    ev = expected_net_value(pred, cost, fill_prob=np.array([1.0, 0.8, 1.0]),
                            adverse_bp=np.array([0.0, -10.0, 0.0]))

    assert np.isclose(ev[0], 0.010 - 0.0046)
    assert np.isclose(ev[1], 0.8 * (0.010 - 0.0010 - 0.0046))
    # a +20bp prediction cannot clear a 46bp round trip: EV is negative, not small
    assert ev[2] < 0.0
    assert np.isclose(ev[2], 0.002 - 0.0046)


def test_select_by_expected_value_abstains_when_no_name_clears_cost() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.expected_value import select_by_expected_value

    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3 + ["2024-01-04"] * 3),
        "symbol": [f"{i:06d}" for i in range(9)],
        "ev": [0.008, 0.003, -0.001,      # day 1: two names clear
               -0.002, -0.005, -0.010,    # day 2: none clear -> abstain
               0.001, -0.004, -0.006],    # day 3: one name clears
    })

    picks = select_by_expected_value(df, group_col="trade_date", ev_col="ev", min_ev=0.0, max_positions=2)

    assert len(picks) == 3
    assert pd.Timestamp("2024-01-03") not in set(picks["trade_date"])
    assert picks[picks["trade_date"] == pd.Timestamp("2024-01-02")]["symbol"].tolist() == ["000000", "000001"]
    assert np.isclose(picks["ev"].min(), 0.001)
    # buy_rate is an outcome of the EV floor, never a target to optimise
    assert set(picks.columns) >= {"trade_date", "symbol", "ev"}


def test_expected_value_fail_closed_edges() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.expected_value import expected_net_value, select_by_expected_value

    with pytest.raises(ValueError, match="shape mismatch"):
        expected_net_value(np.array([0.01]), np.array([0.0046, 0.0046]), fill_prob=np.array([1.0]), adverse_bp=np.array([0.0]))
    with pytest.raises(ValueError, match="non-empty"):
        expected_net_value(np.array([]), np.array([]), fill_prob=np.array([]), adverse_bp=np.array([]))
    with pytest.raises(ValueError, match="fill_prob"):
        expected_net_value(np.array([0.01]), np.array([0.0046]), fill_prob=np.array([1.5]), adverse_bp=np.array([0.0]))
    with pytest.raises(ValueError, match="missing"):
        select_by_expected_value(pd.DataFrame({"a": [1.0]}), group_col="trade_date", ev_col="ev")
    with pytest.raises(ValueError, match="max_positions"):
        select_by_expected_value(
            pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-02"]), "ev": [0.01]}),
            group_col="trade_date", ev_col="ev", max_positions=0,
        )
    with pytest.raises(ValueError, match="min_ev"):
        select_by_expected_value(
            pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-02"]), "ev": [0.01]}),
            group_col="trade_date", ev_col="ev", min_ev=float("nan"),
        )

    # A fully negative pool abstains everywhere; MDE is undefined on no picks
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 2),
        "symbol": ["000000", "000001"],
        "ev": [-0.01, -0.02],
    })
    picks = select_by_expected_value(df, group_col="trade_date", ev_col="ev", min_ev=0.0)
    assert len(picks) == 0
    assert picks.attrs["n_days"] == 0
    assert not np.isfinite(picks.attrs["mde"])


def test_select_by_expected_value_survives_incomparable_frame_attrs() -> None:
    """Regression: an incomparable object in df.attrs (e.g. a feature_manifest
    DataFrame attached upstream by build_feature_manifest) made pd.concat's
    __finalize__ raise 'ambiguous truth value' on the per-group sliced frames."""
    import numpy as np
    import pandas as pd

    from src.ml.expected_value import select_by_expected_value

    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        "symbol": ["000000", "000001", "000002"],
        "ev": [0.008, 0.003, 0.001],
    })
    # Same class of object (a DataFrame) that build_feature_manifest attaches
    # to processed panels upstream -- comparing two DataFrames with `==`
    # raises inside pandas' attrs-equality check unless cleared first.
    df.attrs["feature_manifest"] = pd.DataFrame({"col": ["a"]})

    picks = select_by_expected_value(df, group_col="trade_date", ev_col="ev", min_ev=0.0, max_positions=1)

    assert len(picks) == 2
    assert picks["symbol"].tolist() == ["000000", "000002"]
