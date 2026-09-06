"""Auto-generated scenario tests (see docs/specs/ml_validation_overhaul_contract.json)."""
from __future__ import annotations

def test_attach_per_row_cost_ratio_falls_back_on_bad_price() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.decision_labels import attach_per_row_cost_ratio
    from src.serving.realtime.inference import ROUND_TRIP_COST_RATIO

    # Given two tradable closes and two unusable ones
    df = pd.DataFrame({"close_price": [1500.0, 30000.0, np.nan, 0.0]})

    out = attach_per_row_cost_ratio(df)

    # Then no row is ever NaN-costed and unmeasurable rows fall back to the flat constant
    assert np.isfinite(out["cost_ratio"].to_numpy(dtype=float)).all()
    assert out["cost_measured"].tolist() == [True, True, False, False]
    assert np.isclose(out["cost_ratio"].iloc[2], ROUND_TRIP_COST_RATIO)
    assert np.isclose(out["cost_ratio"].iloc[3], ROUND_TRIP_COST_RATIO)
    # 1,500원 -> tick 1원 -> 2-tick round trip 13.33bp + 20bp statutory
    assert np.isclose(out["cost_ratio"].iloc[0], (20.0 + 2.0 * 1.0 / 1500.0 * 1e4) / 1e4)
    # 30,000원 -> tick 50원 -> 33.33bp + 20bp statutory
    assert np.isclose(out["cost_ratio"].iloc[1], (20.0 + 2.0 * 50.0 / 30000.0 * 1e4) / 1e4)
    # The point of per-row costing: identical picks do not carry identical cost
    assert not np.isclose(out["cost_ratio"].iloc[0], out["cost_ratio"].iloc[1])

def test_build_decision_labels_flat_journaled_matches_retarget_with_clip() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.dataset import retarget_with_clip
    from src.ml.decision_labels import build_decision_labels

    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        "stock_code": ["000001", "000002", "000001"],
        "net_return": [3.0, -50.0, 40.0],
        "target_return": [0.0, 0.0, 0.0],
        "close_price": [1500.0, 30000.0, 1500.0],
    })

    legacy = retarget_with_clip(df, -0.10, 0.10)
    out, prov = build_decision_labels(
        df, None, label_mode="journaled", cost_mode="flat", clip_lower=-0.10, clip_upper=0.10
    )

    # Then the legacy training label is reproduced exactly
    assert np.allclose(
        out["target_return"].to_numpy(dtype=float), legacy["target_return"].to_numpy(dtype=float)
    )
    # And the unclipped evaluation label is exposed alongside it
    assert np.isclose(out["eval_net_journaled"].iloc[2], 40.0 / 100.0 - out["cost_ratio"].iloc[2])
    assert prov["label_mode"] == "journaled"
    assert prov["cost_mode"] == "flat"
    assert prov["n_rows"] == 3

def test_build_decision_labels_mechanical_requires_price_history() -> None:
    import pandas as pd
    import pytest

    from src.ml.decision_labels import build_decision_labels

    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"]),
        "stock_code": ["000001"],
        "net_return": [3.0],
        "target_return": [0.0],
        "close_price": [1500.0],
    })

    with pytest.raises(ValueError, match="price_history"):
        build_decision_labels(
            df, None, label_mode="mechanical", cost_mode="per_row", clip_lower=-0.10, clip_upper=0.10
        )
    with pytest.raises(ValueError, match="label_mode"):
        build_decision_labels(
            df, None, label_mode="operator", cost_mode="flat", clip_lower=-0.10, clip_upper=0.10
        )
    with pytest.raises(ValueError, match="cost_mode"):
        build_decision_labels(
            df, None, label_mode="journaled", cost_mode="guess", clip_lower=-0.10, clip_upper=0.10
        )

def test_build_decision_labels_mechanical_prices_next_open_over_entry_close() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.decision_labels import attach_mechanical_return, build_decision_labels

    price_history = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "symbol": ["000001", "000001", "000001"],
        "open": [990.0, 1030.0, 1040.0],
        "high": [1010.0, 1050.0, 1060.0],
        "low": [980.0, 1020.0, 1030.0],
        "close": [1000.0, 1040.0, 1050.0],
        "daily_change_pct": [0.010, 0.040, 0.0096],
    })
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"]),
        "stock_code": ["000001"],
        "net_return": [5.0],
        "target_return": [0.0],
        "close_price": [1000.0],
    })

    attached = attach_mechanical_return(df, price_history)
    assert np.isclose(attached["mechanical_gross"].iloc[0], 1030.0 / 1000.0 - 1.0)
    assert bool(attached["has_mechanical_label"].iloc[0]) is True

    out, prov = build_decision_labels(
        df, price_history, label_mode="mechanical", cost_mode="per_row",
        clip_lower=-0.10, clip_upper=0.10,
    )

    # 1,000원 -> tick 1원 -> 20bp spread + 20bp statutory = 40bp
    assert np.isclose(out["cost_ratio"].iloc[0], 0.0040)
    assert np.isclose(out["eval_net_mechanical"].iloc[0], 0.03 - 0.0040)
    assert np.isclose(out["target_return"].iloc[0], 0.026)
    # The journaled label survives on the same per-row cost so the A/B is paired
    assert np.isclose(out["eval_net_journaled"].iloc[0], 0.05 - 0.0040)
    assert np.isclose(prov["mechanical_coverage"], 1.0)
    assert prov["n_dropped_no_mechanical"] == 0

def test_build_decision_labels_mechanical_drops_rows_without_next_day_price() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.decision_labels import build_decision_labels

    price_history = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02"]),
        "symbol": ["000001", "000001", "000002"],
        "open": [990.0, 1030.0, 500.0],
        "high": [1010.0, 1050.0, 520.0],
        "low": [980.0, 1020.0, 495.0],
        "close": [1000.0, 1040.0, 510.0],
        "daily_change_pct": [0.010, 0.040, 0.020],
    })
    # 000002 has no next trading day in price_history -> no mechanical label
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "stock_code": ["000001", "000002"],
        "net_return": [5.0, 7.0],
        "target_return": [0.0, 0.0],
        "close_price": [1000.0, 510.0],
    })

    out, prov = build_decision_labels(
        df, price_history, label_mode="mechanical", cost_mode="per_row",
        clip_lower=-0.10, clip_upper=0.10,
    )

    assert len(out) == 1
    assert out["stock_code"].tolist() == ["000001"]
    assert prov["n_dropped_no_mechanical"] == 1
    assert np.isclose(prov["mechanical_coverage"], 0.5)
    assert np.isfinite(out["target_return"].to_numpy(dtype=float)).all()

def test_assert_no_label_leakage_rejects_next_day_columns() -> None:
    import pytest

    from src.ml.dataset import _EXCLUDED_FROM_X
    from src.ml.decision_labels import DECISION_LABEL_COLUMNS, assert_no_label_leakage

    # Given a clean decision-time feature list, the guard is silent
    assert_no_label_leakage(["close_position", "body_ratio", "relative_flow_strength"])

    # Then every next-day / evaluation column is registered and rejected
    for col in ("nd_open", "nd_high", "nd_close", "entry_close", "mechanical_gross",
                "eval_net_mechanical", "eval_net_journaled", "cost_ratio"):
        assert col in DECISION_LABEL_COLUMNS
        assert col in _EXCLUDED_FROM_X
    with pytest.raises(ValueError, match="nd_open"):
        assert_no_label_leakage(["close_position", "nd_open"])
    with pytest.raises(ValueError, match="eval_net_mechanical"):
        assert_no_label_leakage(["eval_net_mechanical"])
