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


def test_estimate_round_trip_cost_bp_uses_updated_statutory_rate() -> None:
    import pandas as pd
    import pytest

    from src.execution.cost_model import STATUTORY_COST_BP, estimate_round_trip_cost_bp

    # Given: 10,000원 (tick 10 -> 2-tick round trip spread = 20bp)
    df = pd.DataFrame({"close_price": [10000.0]})

    # When
    out = estimate_round_trip_cost_bp(df)

    # Then
    assert STATUTORY_COST_BP == pytest.approx(20.0)  # noqa: SIM300
    assert out["spread_bp"][0] == pytest.approx(20.0)
    assert out["round_trip_cost_bp"][0] == pytest.approx(40.0)

def test_krx_tick_size_asof_switches_bands_at_reform_date() -> None:
    # Given
    import numpy as np

    from src.execution.cost_model import TICK_REFORM_DATE, krx_tick_size_asof

    price = np.array([15000.0, 15000.0, 15000.0], dtype=np.float64)
    trade_date = np.array(
        [
            np.datetime64("2022-12-30"),
            TICK_REFORM_DATE,
            np.datetime64("2026-09-04"),
        ],
        dtype="datetime64[ns]",
    )
    market = np.array(["KOSPI", "KOSPI", "KOSPI"], dtype=object)

    # When
    tick = krx_tick_size_asof(price, trade_date, market)

    # Then: 개편 이전 10,000~50,000 구간은 50원, 개편 이후 10,000~20,000 구간은 10원
    np.testing.assert_allclose(tick, [50.0, 10.0, 10.0])


def test_krx_tick_size_asof_uses_kosdaq_table_before_reform() -> None:
    # Given
    import numpy as np

    from src.execution.cost_model import krx_tick_size_asof

    price = np.array([150000.0, 150000.0, 150000.0, 150000.0], dtype=np.float64)
    trade_date = np.array(
        [
            np.datetime64("2020-06-01"),
            np.datetime64("2020-06-01"),
            np.datetime64("2020-06-01"),
            np.datetime64("2026-01-05"),
        ],
        dtype="datetime64[ns]",
    )
    market = np.array(["KOSDAQ", "KOSPI", "UNKNOWN", "KOSDAQ"], dtype=object)

    # When
    tick = krx_tick_size_asof(price, trade_date, market)

    # Then: UNKNOWN 은 보수적으로 KOSPI 테이블(500원)로 처리한다
    np.testing.assert_allclose(tick, [100.0, 500.0, 500.0, 100.0])
    assert tick[2] == tick[1]

    # KSQ150 도 KOSDAQ 계열로 취급
    ksq = krx_tick_size_asof(
        np.array([150000.0]),
        np.array([np.datetime64("2020-06-01")], dtype="datetime64[ns]"),
        np.array(["KSQ150"], dtype=object),
    )
    np.testing.assert_allclose(ksq, [100.0])


def test_krx_tick_size_asof_propagates_nan_never_zero() -> None:
    # Given
    import numpy as np

    from src.execution.cost_model import krx_tick_size_asof, tick_cost_bp

    price = np.array([0.0, -100.0, np.nan, np.inf, 15000.0, 15000.0], dtype=np.float64)
    trade_date = np.array(
        [
            np.datetime64("2026-01-05"),
            np.datetime64("2026-01-05"),
            np.datetime64("2026-01-05"),
            np.datetime64("2026-01-05"),
            np.datetime64("NaT", "ns"),
            np.datetime64("2026-01-05"),
        ],
        dtype="datetime64[ns]",
    )
    market = np.array(["KOSPI"] * 6, dtype=object)

    # When
    tick = krx_tick_size_asof(price, trade_date, market)
    cost = tick_cost_bp(price, trade_date, market)

    # Then
    assert np.isnan(tick[:5]).all()
    assert np.isnan(cost[:5]).all()
    assert not np.any(tick[:5] == 0.0)
    assert not np.any(cost[:5] == 0.0)
    assert np.isfinite(tick[5]) and np.isfinite(cost[5])


def test_tick_cost_bp_returns_single_tick_bp() -> None:
    # Given
    import numpy as np

    from src.execution.cost_model import spread_cost_bp, tick_cost_bp

    price = np.array([15000.0, 3000.0, 150000.0], dtype=np.float64)
    trade_date = np.array([np.datetime64("2026-01-05")] * 3, dtype="datetime64[ns]")
    market = np.array(["KOSPI"] * 3, dtype=object)

    # When
    per_tick = tick_cost_bp(price, trade_date, market)

    # Then: 개편 후 틱은 10 / 5 / 100
    np.testing.assert_allclose(
        per_tick,
        [10.0 / 15000.0 * 1e4, 5.0 / 3000.0 * 1e4, 100.0 / 150000.0 * 1e4],
    )
    # 15,000원 종목의 1틱 비용은 6.67bp 로 7.5bp 상한을 통과한다
    assert per_tick[0] < 7.5
    assert per_tick[1] > 7.5
    # 왕복 2틱은 개편 후 구간에서 기존 spread_cost_bp 와 동일
    np.testing.assert_allclose(2.0 * per_tick, spread_cost_bp(price, round_trip_ticks=2.0))
