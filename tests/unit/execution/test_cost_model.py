"""Execution cost model scenarios (contract: execution_cost_model)."""

from __future__ import annotations


def test_krx_tick_size_follows_band_ladder_and_rejects_bad_price() -> None:
    import numpy as np

    from src.execution.cost_model import krx_tick_size

    # Given: one price inside each band, a band boundary, and two bad prices
    price = np.array([1500.0, 3000.0, 12000.0, 30000.0, 100000.0, 300000.0, 900000.0, 2000.0, 0.0, np.nan])

    # When
    tick = krx_tick_size(price)

    # Then: ladder values, boundary takes the HIGHER band (price < bound is the lower band)
    assert tick[:7].tolist() == [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    assert tick[7] == 5.0
    assert np.isnan(tick[8])
    assert np.isnan(tick[9])


def test_spread_cost_bp_scales_with_ticks_and_is_caller_visible() -> None:
    import numpy as np
    import pytest

    from src.execution.cost_model import spread_cost_bp

    # Given: 10,000원 -> tick 10 -> one tick = 10bp
    price = np.array([10000.0, 0.0])

    # When / Then: default is two ticks round trip
    assert spread_cost_bp(price)[0] == pytest.approx(20.0)
    # And: the assumption is a caller-visible knob, not hardcoded
    assert spread_cost_bp(price, round_trip_ticks=1.0)[0] == pytest.approx(10.0)
    assert spread_cost_bp(price, round_trip_ticks=4.0)[0] == pytest.approx(40.0)
    assert np.isnan(spread_cost_bp(price)[1])


def test_measure_auction_impact_bp_distinguishes_unmeasured_from_zero(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.execution.cost_model import measure_auction_impact_bp

    # Given: 12 continuous bars ending at 10,000, then an auction print at 10,050 (+50bp)
    snap = "2026-03-02"
    part_dir = tmp_path / "intraday" / "1m" / "regular" / snap[:7]
    part_dir.mkdir(parents=True, exist_ok=True)
    ts = list(range(150000, 151200, 100))
    bars = pd.DataFrame(
        {
            "snapshot_date": [snap] * (len(ts) + 1),
            "symbol": ["005930"] * (len(ts) + 1),
            "ts_hms": [*ts, 153000],
            "open": [10000] * len(ts) + [10050],
            "high": [10000] * len(ts) + [10050],
            "low": [10000] * len(ts) + [10050],
            "close": [10000] * len(ts) + [10050],
            "volume": [100] * (len(ts) + 1),
            "value_krw": [1_000_000] * (len(ts) + 1),
            "has_trade": [True] * (len(ts) + 1),
            "vendor": ["kis"] * (len(ts) + 1),
        }
    )
    bars.to_parquet(part_dir / f"{snap}.parquet")
    df = pd.DataFrame(
        {"trade_date": pd.to_datetime([snap, snap]), "stock_code": ["005930", "000660"]}
    )

    # When
    out = measure_auction_impact_bp(df, intraday_root=tmp_path)

    # Then
    assert len(out) == 2
    assert out["impact_measured"].tolist() == [True, False]
    assert out.loc[0, "auction_impact_bp"] == pytest.approx(50.0, abs=1e-6)
    assert np.isnan(out.loc[1, "auction_impact_bp"])


def test_estimate_round_trip_cost_bp_composes_and_is_idempotent() -> None:
    import pandas as pd
    import pytest

    from src.execution.cost_model import STATUTORY_COST_BP, estimate_round_trip_cost_bp

    # Given: 10,000원 (tick 10 -> 20bp spread) and 100,000원 (tick 100 -> 20bp spread)
    df = pd.DataFrame({"close_price": [10000.0, 100000.0]})

    # When: no measured impact supplied
    out = estimate_round_trip_cost_bp(df)

    # Then
    assert len(out) == 2
    assert out["spread_bp"].tolist() == pytest.approx([20.0, 20.0])
    assert out["round_trip_cost_bp"].tolist() == pytest.approx([STATUTORY_COST_BP + 20.0] * 2)
    # unmeasured impact stays visible rather than becoming a measured zero
    assert "auction_impact_bp" in out.columns
    assert out["auction_impact_bp"].isna().all()

    # And: idempotent
    again = estimate_round_trip_cost_bp(out)
    pd.testing.assert_frame_equal(out, again[out.columns])


def test_estimate_round_trip_cost_bp_adds_measured_impact_only_where_measured() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.execution.cost_model import STATUTORY_COST_BP, estimate_round_trip_cost_bp

    # Given: one row with a measured +30bp auction move, one unmeasured
    df = pd.DataFrame(
        {"close_price": [10000.0, 10000.0], "auction_impact_bp": [30.0, np.nan]}
    )

    # When
    out = estimate_round_trip_cost_bp(df, impact_col="auction_impact_bp")

    # Then: impact enters the total only where it was measured
    assert out["round_trip_cost_bp"][0] == pytest.approx(STATUTORY_COST_BP + 20.0 + 30.0)
    assert out["round_trip_cost_bp"][1] == pytest.approx(STATUTORY_COST_BP + 20.0)
    assert np.isnan(out["auction_impact_bp"][1])


def test_breakeven_cost_bp_is_per_day_and_refuses_small_samples() -> None:
    import numpy as np
    import pytest

    from src.execution.cost_model import breakeven_cost_bp

    # Given: 40 days; each day one row of +1.0% except day 0 which has 9 extra rows of -1.0%.
    # Per-day means: day0 = -0.8%, days 1..39 = +1.0%  -> mean = (-0.8 + 39*1.0)/40 = 0.955%
    groups = np.array([0] * 10 + list(range(1, 40)))
    rets = np.array([1.0] + [-1.0] * 9 + [1.0] * 39)

    # When
    be = breakeven_cost_bp(rets, groups)

    # Then: per-day equal weighting, not per-row
    assert be == pytest.approx(95.5, abs=1e-6)

    # And: too few groups is NaN, never a number computed on a handful of days
    assert np.isnan(breakeven_cost_bp(np.array([1.0, 2.0]), np.array([0, 1])))


def test_summarize_cost_breakdown_reports_impact_coverage() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.execution.cost_model import estimate_round_trip_cost_bp, summarize_cost_breakdown

    # Given: 4 rows, 2 with a measured impact
    df = pd.DataFrame(
        {
            "close_price": [10000.0] * 4,
            "auction_impact_bp": [10.0, 30.0, np.nan, np.nan],
        }
    )
    costed = estimate_round_trip_cost_bp(df, impact_col="auction_impact_bp")

    # When
    bd = summarize_cost_breakdown(costed)

    # Then
    assert bd.n_rows == 4
    assert bd.n_impact_measured == 2
    assert bd.spread_bp == pytest.approx(20.0)
    assert bd.auction_impact_bp == pytest.approx(20.0)
