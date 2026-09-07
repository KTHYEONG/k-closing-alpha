import numpy as np
import pandas as pd
import pytest

from src.ml.forward_path import (
    attach_forward_path,
    calculate_horizon_metrics,
    compute_forward_returns,
    simulate_multiday_tp_exit,
)


def _make_sample_price_history() -> pd.DataFrame:
    # 5 trading days for two symbols, skipping a weekend/holiday
    # Dates: 2024-01-02 (Tue), 2024-01-03 (Wed), 2024-01-04 (Thu), 2024-01-05 (Fri), 2024-01-08 (Mon)
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    rows = []
    # Symbol 1: 000660
    for i, d in enumerate(dates):
        rows.append({
            "date": d,
            "symbol": "000660",
            "open": 100.0 + i * 10.0,
            "high": 105.0 + i * 10.0,
            "low": 98.0 + i * 10.0,
            "close": 102.0 + i * 10.0,
            "daily_change_pct": 0.02,
        })
    # Symbol 2: 005930
    for i, d in enumerate(dates):
        rows.append({
            "date": d,
            "symbol": "005930",
            "open": 50.0 + i * 5.0,
            "high": 53.0 + i * 5.0,
            "low": 49.0 + i * 5.0,
            "close": 51.0 + i * 5.0,
            "daily_change_pct": 0.01,
        })
    return pd.DataFrame(rows)


def test_attach_forward_path_trading_day_and_symbol_shift() -> None:
    ph = _make_sample_price_history()
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "stock_code": ["000660", "005930"],
    })

    out = attach_forward_path(df, ph, horizons=(1, 2, 3))

    assert len(out) == 2
    row0 = out.iloc[0]
    assert row0["entry_close"] == 102.0
    # D+1 is 2024-01-03
    assert row0["d1_date"] == pd.Timestamp("2024-01-03")
    assert row0["d1_open"] == 110.0
    assert row0["d1_high"] == 115.0
    assert row0["d1_low"] == 108.0
    assert row0["d1_close"] == 112.0
    # D+2 is 2024-01-04
    assert row0["d2_date"] == pd.Timestamp("2024-01-04")
    assert row0["d2_close"] == 122.0
    # D+3 is 2024-01-05
    assert row0["d3_date"] == pd.Timestamp("2024-01-05")
    assert row0["d3_close"] == 132.0

    # Verify symbol 2 does not bleed into symbol 1
    row1 = out.iloc[1]
    assert row1["entry_close"] == 51.0
    assert row1["d1_open"] == 55.0
    assert row1["d2_close"] == 61.0
    assert row1["d3_close"] == 66.0


def test_attach_forward_path_holiday_and_last_date_nan() -> None:
    ph = _make_sample_price_history()
    # Entry on 2024-01-05 (Friday), D+1 should be 2024-01-08 (Monday, skipping weekend)
    # D+2 and D+3 are beyond available history and should be NaN
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-05"]),
        "stock_code": ["000660"],
    })

    out = attach_forward_path(df, ph, horizons=(1, 2, 3))
    row = out.iloc[0]
    assert row["d1_date"] == pd.Timestamp("2024-01-08")
    assert row["d1_close"] == 142.0
    assert pd.isna(row["d2_date"])
    assert np.isnan(row["d2_open"])
    assert np.isnan(row["d3_close"])


def test_attach_forward_path_symbol_normalization_and_duplicates() -> None:
    ph = _make_sample_price_history()
    # Add a duplicate row in price_history
    dup_row = ph.iloc[0:1].copy()
    dup_row["close"] = 999.0
    ph_with_dup = pd.concat([ph, dup_row], ignore_index=True)

    # df has non-zero padded symbol, float-like symbol, and duplicate entries
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
        "stock_code": ["660", "660.0", "005930"],
        "tag": ["first", "second", "third"],
    })

    out = attach_forward_path(df, ph_with_dup, horizons=(1, 2))
    assert len(out) == 3
    assert list(out["tag"]) == ["first", "second", "third"]
    assert out.iloc[0]["entry_close"] == 999.0
    assert out.iloc[1]["entry_close"] == 999.0
    assert out.iloc[2]["entry_close"] == 51.0


def test_attach_forward_path_missing_bar_or_unlisted_symbol() -> None:
    ph = _make_sample_price_history()
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "stock_code": ["000660", "999999"],  # 999999 does not exist in ph
    })

    out = attach_forward_path(df, ph, horizons=(1, 2))
    assert len(out) == 2
    assert np.isnan(out.iloc[1]["entry_close"])
    assert np.isnan(out.iloc[1]["d1_open"])
    assert pd.isna(out.iloc[1]["d1_date"])


def test_compute_forward_returns_and_incremental() -> None:
    ph = _make_sample_price_history()
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"]),
        "stock_code": ["000660"],
    })
    attached = attach_forward_path(df, ph, horizons=(1, 2, 3))
    returns_df = compute_forward_returns(attached, cost_ratio=0.0046, horizons=(1, 2, 3))

    # entry_close = 102.0
    # d1_open = 110.0 -> gross = 110/102 - 1 = 0.078431
    assert abs(returns_df["d1_open_gross"].iloc[0] - (110.0 / 102.0 - 1.0)) < 1e-6
    assert abs(returns_df["d1_open_net"].iloc[0] - (110.0 / 102.0 - 1.0 - 0.0046)) < 1e-6

    # d1_close = 112.0, d2_close = 122.0, d3_close = 132.0
    # Incremental:
    # d1_intraday = 112/110 - 1
    assert abs(returns_df["d1_intraday_return"].iloc[0] - (112.0 / 110.0 - 1.0)) < 1e-6
    # d1_to_d2 = 122/112 - 1
    assert abs(returns_df["d1_to_d2_close_return"].iloc[0] - (122.0 / 112.0 - 1.0)) < 1e-6
    # d2_to_d3 = 132/122 - 1
    assert abs(returns_df["d2_to_d3_close_return"].iloc[0] - (132.0 / 122.0 - 1.0)) < 1e-6

    # MFE / MAE up to D+3
    # Highs: 115, 125, 135 -> max is 135 -> MFE = 135 / 102 - 1
    assert abs(returns_df["d3_mfe"].iloc[0] - (135.0 / 102.0 - 1.0)) < 1e-6
    # Lows: 108, 118, 128 -> min is 108 -> MAE = 108 / 102 - 1
    assert abs(returns_df["d3_mae"].iloc[0] - (108.0 / 102.0 - 1.0)) < 1e-6


def test_calculate_horizon_metrics() -> None:
    returns = np.array([0.01, 0.02, -0.01, 0.03, -0.005])
    m = calculate_horizon_metrics(returns, cost_ratio=0.0046)
    assert m["n"] == 5
    assert m["mean_gross_bp"] > 0
    assert m["win_rate"] >= 0.0
    assert "bootstrap_ci_net_bp" in m
    assert 50 in m["percentiles_bp"]


def test_simulate_multiday_tp_exit() -> None:
    ph = _make_sample_price_history()
    df = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"]),
        "stock_code": ["000660"],
    })
    attached = attach_forward_path(df, ph, horizons=(1, 2, 3))

    # Entry 102.0. TP 5% = 107.1.
    # D+1: open 110.0 (already > 107.1) -> triggers gap open exit on D+1!
    rets, days, reasons = simulate_multiday_tp_exit(attached, take_profit_pct=0.05, max_horizon=3)
    assert days[0] == 1.0
    assert reasons[0] == "d1_tp_gap"
    assert abs(rets[0] - (110.0 / 102.0 - 1.0)) < 1e-6

    # If TP is 20% = 122.4
    # D+1: open 110, high 115 (< 122.4)
    # D+2: open 120, high 125 (>= 122.4) -> triggers touch exit on D+2!
    rets2, days2, reasons2 = simulate_multiday_tp_exit(attached, take_profit_pct=0.20, max_horizon=3)
    assert days2[0] == 2.0
    assert reasons2[0] == "d2_tp_touch"
    assert abs(rets2[0] - 0.20) < 1e-6

    # If no TP (take_profit_pct=None), fixed D+3 MOC exit
    rets3, days3, reasons3 = simulate_multiday_tp_exit(attached, take_profit_pct=None, max_horizon=3, fallback="moc")
    assert days3[0] == 3.0
    assert reasons3[0] == "d3_fallback_moc"
    assert abs(rets3[0] - (132.0 / 102.0 - 1.0)) < 1e-6
