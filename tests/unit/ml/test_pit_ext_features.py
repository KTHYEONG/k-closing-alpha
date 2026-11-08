import numpy as np
import pandas as pd
import pytest

from src.ml.research import pit_ext_features as xf


def _bars(symbol: str = "000001", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = np.arange(xf.SESSION_OPEN_MIN + 1, xf.DECISION_MIN + 1)
    close = 10_000 + np.cumsum(rng.integers(-20, 21, size=len(end)))
    return pd.DataFrame(
        {
            "symbol": symbol,
            "end_min": end,
            "open": close,
            "high": close + 10,
            "low": close - 10,
            "close": close,
            "volume": rng.integers(0, 1000, size=len(end)).astype(float),
        }
    )


def test_bar_end_minutes_follow_vendor_stamp_convention() -> None:
    ts = pd.Series([151900, 152000])
    kis = xf.bar_end_minutes(ts, pd.Series(["kis", "kis"]))
    toss = xf.bar_end_minutes(ts, pd.Series(["toss", "toss_cons"]))
    assert kis.tolist() == [15 * 60 + 20, 15 * 60 + 21]
    assert toss.tolist() == [15 * 60 + 19, 15 * 60 + 20]


def test_path_features_ignore_bars_after_the_decision() -> None:
    base = _bars()
    extra = base.tail(10).copy()
    extra["end_min"] = np.arange(xf.DECISION_MIN + 1, xf.DECISION_MIN + 11)
    extra[["close", "high", "low", "volume"]] = [1.0, 1e9, 1.0, 1e9]
    later = pd.concat([base, extra], ignore_index=True)
    a = xf.path_features(base).set_index("symbol")
    b = xf.path_features(later).set_index("symbol")
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == list(xf.INTRADAY_FEATURES)


def test_path_features_values_and_short_series_drop() -> None:
    bars = _bars()
    bars["close"] = 100.0
    bars.loc[bars["end_min"] == xf.DECISION_MIN, "close"] = 110.0
    out = xf.path_features(bars).iloc[0]
    assert out["x_r30"] == pytest.approx(0.10)
    assert 0.0 <= out["x_hi_time"] <= 1.0
    assert xf.path_features(_bars().head(xf.MIN_BARS_FOR_PATH - 1)).empty


def _daily(n: int = 40) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(1)
    close = 1000 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": "000001",
            "close": close,
            "prev_close": np.r_[close[0], close[:-1]],
            "volume": rng.integers(1000, 2000, n).astype(float),
            "trade_value_100m": rng.uniform(100, 200, n),
            "market_cap_100m": 5000.0,
            "program_netbuy": rng.normal(0, 10, n),
            "kospi_pct": rng.normal(0, 0.01, n),
            "kosdaq_pct": rng.normal(0, 0.01, n),
            "v_kospi": 15 + rng.normal(0, 1, n),
        }
    )


def test_stock_history_and_index_features_never_use_the_same_or_later_day() -> None:
    ph = _daily()
    cut = ph["date"].iloc[30]
    perturbed = ph.copy()
    late = perturbed["date"] >= cut
    perturbed.loc[
        late, ["close", "volume", "trade_value_100m", "program_netbuy", "kospi_pct", "kosdaq_pct", "v_kospi"]
    ] = 7.0
    for fn in (xf.stock_history, xf.index_and_calendar):
        a = fn(ph).set_index("date").loc[:cut].drop(columns=["c_gap_days"], errors="ignore")
        b = fn(perturbed).set_index("date").loc[:cut].drop(columns=["c_gap_days"], errors="ignore")
        pd.testing.assert_frame_equal(a, b)


def test_calendar_flags_gap_days_and_expiry_thursday() -> None:
    ph = _daily(15)
    cal = xf.index_and_calendar(ph).set_index("date")
    assert cal.loc[pd.Timestamp("2024-01-05"), "c_gap_days"] == 3
    assert cal.loc[pd.Timestamp("2024-01-11"), "c_expiry"] == 1.0
    assert cal.loc[pd.Timestamp("2024-01-04"), "c_expiry"] == 0.0


def test_market_state_breadth_and_cap_weighting() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "close": [110.0, 90.0, 100.0],
            "prev_close": [100.0, 100.0, 100.0],
            "market_cap_100m": [300.0, 100.0, 100.0],
        }
    )
    row = xf.market_state(panel).iloc[0]
    assert row["m_breadth"] == pytest.approx(1 / 3)
    assert row["m_cap_chg"] == pytest.approx((0.1 * 300 - 0.1 * 100) / 500)
    assert row["m_eq_chg"] == pytest.approx(0.0)
