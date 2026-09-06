"""Auto-generated scenario tests (see docs/specs/closing_alpha_architecture_v2_contract.json)."""
from __future__ import annotations


def test_screen_config_rejects_invalid_bands() -> None:
    import pytest

    from src.ml.universe import OPERATOR_LEGACY_SCREEN, ScreenConfig

    # The inherited screen is pinned as the control arm, never as a default
    assert OPERATOR_LEGACY_SCREEN.change_lower == 0.10
    assert OPERATOR_LEGACY_SCREEN.change_upper is None
    assert OPERATOR_LEGACY_SCREEN.min_trade_value_100m == 100.0
    assert OPERATOR_LEGACY_SCREEN.min_market_cap_100m == 500.0
    assert OPERATOR_LEGACY_SCREEN.exclude_ceiling is True

    band = ScreenConfig(change_lower=0.02, change_upper=0.15, min_trade_value_100m=1000.0)
    assert band.change_upper == 0.15

    with pytest.raises(ValueError, match="change_upper"):
        ScreenConfig(change_lower=0.15, change_upper=0.05)
    with pytest.raises(ValueError, match="min_trade_value_100m"):
        ScreenConfig(change_lower=0.02, min_trade_value_100m=-1.0)
    with pytest.raises(ValueError, match="min_market_cap_100m"):
        ScreenConfig(change_lower=0.02, min_market_cap_100m=-1.0)


def test_build_universe_panel_applies_screen_and_excludes_ceiling() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.universe import ScreenConfig, build_universe_panel

    ph = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 4),
        "symbol": ["000001", "000002", "000003", "000004"] * 2,
        "open":       [9900.0, 9900.0, 9900.0, 9900.0, 10300.0, 10100.0, 10100.0, 10100.0],
        "high":       [10200.0, 10000.0, 13000.0, 10000.0, 10400.0, 10200.0, 13100.0, 10200.0],
        "low":        [9800.0, 9800.0, 9800.0, 9800.0, 10000.0, 10000.0, 10000.0, 10000.0],
        "close":      [10000.0, 10000.0, 13000.0, 10000.0, 10200.0, 10100.0, 13050.0, 10100.0],
        "daily_change_pct": [0.08, 0.001, 0.30, 0.08, 0.02, 0.01, 0.004, 0.02],
        "market_cap_100m":  [1000.0, 1000.0, 1000.0, 100.0, 1000.0, 1000.0, 1000.0, 100.0],
        "trade_value_100m": [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
        "market": ["KOSPI"] * 8,
        "kospi_pct": [0.01] * 8,
        "kosdaq_pct": [0.01] * 8,
    })
    screen = ScreenConfig(change_lower=0.05, change_upper=0.15,
                          min_trade_value_100m=100.0, min_market_cap_100m=500.0)

    panel, prov = build_universe_panel(ph, screen, start_date="2024-01-02", end_date="2024-01-02")

    # 000002 fails the change floor, 000003 is a +30% ceiling close, 000004 fails market cap
    assert panel["symbol"].tolist() == ["000001"]
    # mechanical label = next-day open over entry close, both from price_history
    assert np.isclose(panel["mechanical_gross"].iloc[0], 10300.0 / 10000.0 - 1.0)
    assert prov["n_screened"] == 1
    assert prov["n_ceiling_excluded"] == 1
    assert prov["screen"]["change_lower"] == 0.05


def test_screen_baseline_stats_reports_unconditional_edge() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.universe import screen_baseline_stats

    rng = np.random.default_rng(0)
    days = pd.to_datetime(pd.date_range("2024-01-01", periods=200, freq="D"))
    rows = 5
    n = len(days) * rows
    # A pool whose gross overnight drift (+20bp) does not cover a 46bp round trip
    panel = pd.DataFrame({
        "trade_date": np.repeat(days.to_numpy(), rows),
        "symbol": [f"{i % 30:06d}" for i in range(n)],
        "mechanical_gross": 0.0020 + rng.normal(scale=0.02, size=n),
    })

    stats = screen_baseline_stats(panel, group_col="trade_date", gross_col="mechanical_gross", cost_ratio=0.0046)

    assert stats["n_days"] == 200
    assert np.isclose(stats["per_day"], 5.0)
    # gross ~ +20bp, net ~ -26bp: the pool loses money before any model runs
    assert stats["net_bp"] < 0.0
    assert np.isclose(stats["net_bp"], (stats["gross_bp"] - 46.0), atol=1.0)
    assert np.isfinite(stats["t_stat"]) and np.isfinite(stats["sharpe"])


def test_universe_fail_closed_edges() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.universe import ScreenConfig, build_universe_panel, screen_baseline_stats

    ph = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 4),
        "symbol": ["000001", "000002", "000003", "000004"] * 2,
        "open": [9900.0, 9900.0, 9900.0, 9900.0, 10300.0, 10100.0, 10100.0, 10100.0],
        "high": [10200.0, 10000.0, 13000.0, 10000.0, 10400.0, 10200.0, 13100.0, 10200.0],
        "low": [9800.0, 9800.0, 9800.0, 9800.0, 10000.0, 10000.0, 10000.0, 10000.0],
        "close": [10000.0, 10000.0, 13000.0, 10000.0, 10200.0, 10100.0, 13050.0, 10100.0],
        "daily_change_pct": [0.08, 0.001, 0.30, 0.08, 0.02, 0.01, 0.004, 0.02],
        "market_cap_100m": [1000.0, 1000.0, 1000.0, 100.0, 1000.0, 1000.0, 1000.0, 100.0],
        "trade_value_100m": [500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0],
        "market": ["KOSPI"] * 8,
        "kospi_pct": [0.01] * 8,
        "kosdaq_pct": [0.01] * 8,
    })
    wide = ScreenConfig(change_lower=0.05, change_upper=None, min_trade_value_100m=0.0, min_market_cap_100m=0.0, exclude_ceiling=False)
    panel_keep, prov_keep = build_universe_panel(ph, wide, start_date="2024-01-02", end_date="2024-01-02")
    assert prov_keep["n_ceiling_excluded"] == 0
    assert len(panel_keep) == 3  # 000001, 000003 (ceiling kept), 000004
    assert prov_keep["kanri_filter_unavailable"] is True

    strict = ScreenConfig(change_lower=0.05, change_upper=0.15, min_trade_value_100m=100.0, min_market_cap_100m=500.0)
    panel_idx, _ = build_universe_panel(
        ph, ScreenConfig(change_lower=0.05, change_upper=0.15, min_trade_value_100m=100.0, min_market_cap_100m=500.0, require_index_up=True),
        start_date="2024-01-02", end_date="2024-01-02",
    )
    assert len(panel_idx) == 1
    no_idx = ph.drop(columns=["kospi_pct", "kosdaq_pct"])
    panel_noidx, _ = build_universe_panel(
        no_idx, ScreenConfig(change_lower=0.05, change_upper=0.15, min_trade_value_100m=100.0, min_market_cap_100m=500.0, require_index_up=True),
        start_date="2024-01-02", end_date="2024-01-02",
    )
    assert len(panel_noidx) == 0
    no_mc = ph.drop(columns=["market_cap_100m"])
    panel_nomc, prov_nomc = build_universe_panel(no_mc, strict, start_date="2024-01-02", end_date="2024-01-02")
    assert len(panel_nomc) == 0
    assert prov_nomc["market_cap_coverage"] == 0.0
    assert prov_nomc["coverage_warning"] is True
    no_tv = ph.drop(columns=["trade_value_100m"])
    panel_notv, prov_notv = build_universe_panel(no_tv, strict, start_date="2024-01-02", end_date="2024-01-02")
    assert len(panel_notv) == 0
    assert prov_notv["trade_value_coverage"] == 0.0
    panel_empty, prov_empty = build_universe_panel(ph, strict, start_date="2030-01-02", end_date="2030-01-03")
    assert len(panel_empty) == 0
    assert prov_empty["n_raw"] == 0
    assert prov_empty["per_day"] == 0.0

    with pytest.raises(ValueError, match="missing"):
        build_universe_panel(ph.drop(columns=["close"]), strict, start_date="2024-01-02", end_date="2024-01-02")
    with pytest.raises(ValueError, match="missing"):
        build_universe_panel(ph.drop(columns=["date"]), strict, start_date="2024-01-02", end_date="2024-01-02")
    with pytest.raises(ValueError, match="parseable"):
        build_universe_panel(ph, strict, start_date="not-a-date", end_date="2024-01-02")
    with pytest.raises(ValueError, match="missing"):
        screen_baseline_stats(pd.DataFrame({"a": [1.0]}), group_col="trade_date", gross_col="mechanical_gross", cost_ratio=0.0046)
    with pytest.raises(ValueError, match="empty"):
        screen_baseline_stats(ph.iloc[0:0].assign(trade_date=pd.to_datetime([]), mechanical_gross=[]), group_col="trade_date", gross_col="mechanical_gross", cost_ratio=0.0046)
    with pytest.raises(ValueError, match="cost_ratio"):
        screen_baseline_stats(panel_keep.assign(trade_date=pd.to_datetime(["2024-01-02"] * len(panel_keep))), group_col="trade_date", gross_col="mechanical_gross", cost_ratio=float("nan"))
    with pytest.raises(ValueError, match="finite daily"):
        screen_baseline_stats(
            pd.DataFrame({"trade_date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "mechanical_gross": [np.nan, np.nan]}),
            group_col="trade_date", gross_col="mechanical_gross", cost_ratio=0.0046,
        )

    covered = panel_keep.copy()
    covered["trade_date"] = pd.to_datetime(["2024-01-02"] * len(covered))
    covered.attrs["market_cap_coverage"] = 0.5
    warn = screen_baseline_stats(covered, group_col="trade_date", gross_col="mechanical_gross", cost_ratio=0.0046)
    assert warn["coverage_warning"] is True


def test_apply_screen_mask_uses_percent_scaled_change_rate() -> None:
    """Regression: apply_screen_mask filters an already-logged (trade-log-shaped)
    panel using change_rate in PERCENT units, unlike build_universe_panel's
    daily_change_pct which is a fraction."""
    import pandas as pd
    import pytest

    from src.ml.universe import BAND_2_15_SCREEN, OPERATOR_LEGACY_SCREEN, apply_screen_mask

    df = pd.DataFrame({
        "change_rate": [1.5, 9.9, 12.0, 20.0],  # percent units, as logged in the trade log
        "trade_value_100m": [500.0, 500.0, 500.0, 500.0],
        "market_cap_100m": [1000.0, 1000.0, 1000.0, 1000.0],
    })

    legacy_mask = apply_screen_mask(df, OPERATOR_LEGACY_SCREEN)
    # OPERATOR_LEGACY_SCREEN.change_lower=0.10 -> 10% -> only 12.0% and 20.0% qualify
    assert legacy_mask.tolist() == [False, False, True, True]

    band_mask = apply_screen_mask(df, BAND_2_15_SCREEN)
    # BAND_2_15_SCREEN is [2%, 15%] -> only 12.0% qualifies (9.9% is below 2%? no: 9.9 >= 2)
    assert band_mask.tolist() == [False, True, True, False]

    with pytest.raises(ValueError, match="change_col"):
        apply_screen_mask(df.drop(columns=["change_rate"]), OPERATOR_LEGACY_SCREEN)

    # Missing liquidity columns with a positive floor fails closed (nothing qualifies)
    no_liquidity = df.drop(columns=["trade_value_100m", "market_cap_100m"])
    assert apply_screen_mask(no_liquidity, OPERATOR_LEGACY_SCREEN).tolist() == [False, False, False, False]
