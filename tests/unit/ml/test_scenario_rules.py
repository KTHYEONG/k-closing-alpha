from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.scenario_rules import CANONICAL_SCENARIOS, derive_scenario_labels, scenario_agreement_report


def _price_history_with_setups() -> pd.DataFrame:
    # 300 trading days for one base symbol so 120d/252d windows fill.
    dates = pd.bdate_range("2023-01-02", periods=300)
    rows = []
    base = 10000.0
    for i, d in enumerate(dates):
        px = base * (1.0 + 0.0005 * i)
        rows.append({"date": d, "symbol": "000001", "open": px, "high": px * 1.01,
                     "low": px * 0.99, "close": px, "volume": 100000.0,
                     "trade_value_100m": 300.0, "inst_netbuy": 0.0, "foreign_netbuy": 0.0,
                     "daily_change_pct": 0.005})
    df = pd.DataFrame(rows)
    # symbol 000002: prior day is a ceiling close, entry day is the day after.
    c = []
    for i, d in enumerate(dates):
        px = 5000.0 * (1.0 + 0.001 * i)
        chg = 0.30 if i == 298 else 0.01
        close = px * 1.05 if i == 298 else px
        high = close if i == 298 else px * 1.01
        c.append({"date": d, "symbol": "000002", "open": px, "high": high, "low": px * 0.98,
                  "close": close, "volume": 200000.0, "trade_value_100m": 500.0,
                  "inst_netbuy": 0.0, "foreign_netbuy": 0.0, "daily_change_pct": chg})
    return pd.concat([df, pd.DataFrame(c)], ignore_index=True)


def test_derive_scenario_labels_covers_every_canonical_setup() -> None:
    # Arrange
    ph = _price_history_with_setups()
    last = ph["date"].max()
    prev_ceiling_day = sorted(ph["date"].unique())[299]
    panel = pd.DataFrame(
        {
            "trade_date": [last, prev_ceiling_day, last, last],
            "stock_code": ["000001", "000002", "000001", "000001"],
            # row 0: normal grind -> some non-null canonical label
            # row 1: day after 000002's ceiling -> '상한가 다음날'
            # row 2: forced 상따 candle (close at high, +25%)
            # row 3: forced 상승형 음봉 (green day, close near low)
            "change_rate": [3.0, 2.0, 25.0, 8.0],
            "high_price": [10200.0, 5300.0, 12500.0, 11000.0],
            "low_price": [9900.0, 5100.0, 10100.0, 10000.0],
            "close_price": [10100.0, 5250.0, 12480.0, 10050.0],
        }
    )

    # Act
    out = derive_scenario_labels(panel, ph)

    # Assert
    assert list(out.index) == list(panel.index)
    assert out.isna().sum() == 0
    assert set(out.unique()).issubset(set(CANONICAL_SCENARIOS))
    assert out.iloc[1] == "상한가 다음날"
    assert out.iloc[2] == "상따"
    assert out.iloc[3] == "상승형 음봉"


def test_prev_ceiling_outranks_new_high() -> None:
    # Arrange: 260 days climbing, day -2 is a ceiling, entry = day -1 makes a new high.
    dates = pd.bdate_range("2023-01-02", periods=260)
    rows = []
    for i, d in enumerate(dates):
        px = 1000.0 + 5.0 * i
        ceiling = i == 258
        close = px * 1.30 if ceiling else px
        high = close
        rows.append({"date": d, "symbol": "000009", "open": px, "high": high,
                     "low": px * 0.98, "close": close, "volume": 100000.0,
                     "trade_value_100m": 300.0, "inst_netbuy": 0.0,
                     "foreign_netbuy": 0.0, "daily_change_pct": 0.30 if ceiling else 0.02})
    ph = pd.DataFrame(rows)
    entry_day = dates[259]
    panel = pd.DataFrame({"trade_date": [entry_day], "stock_code": ["000009"],
                          "change_rate": [5.0], "high_price": [2300.0],
                          "low_price": [2250.0], "close_price": [2300.0]})

    # Act
    out = derive_scenario_labels(panel, ph)

    # Assert
    assert out.iloc[0] == "상한가 다음날"


def test_derive_scenario_labels_fail_closed_without_price_history() -> None:
    panel = pd.DataFrame({"trade_date": [pd.Timestamp("2024-01-02")], "stock_code": ["000001"],
                          "change_rate": [3.0], "high_price": [100.0], "low_price": [95.0],
                          "close_price": [99.0]})

    with pytest.raises(ValueError, match="price_history"):
        derive_scenario_labels(panel, None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="price_history"):
        derive_scenario_labels(panel, pd.DataFrame())


def test_scenario_agreement_report_contract() -> None:
    manual = pd.Series(["신고가", "신고가", "거래량 폭증", "상따", "미분류"])
    auto = pd.Series(["신고가", "거래량 폭증", "거래량 폭증", "상따", "신고가"])

    rep = scenario_agreement_report(manual, auto)

    assert rep["n"] == 5
    assert 0.0 <= rep["overall_agreement"] <= 1.0
    assert abs(rep["overall_agreement"] - 0.6) < 1e-9
    assert set(rep["per_class"].keys()) == set(CANONICAL_SCENARIOS)
    assert rep["per_class"]["신고가"]["support"] == 2
    assert abs(rep["per_class"]["신고가"]["recall"] - 0.5) < 1e-9

    with pytest.raises(ValueError):  # noqa: PT011 - contract skeleton asserts type only
        scenario_agreement_report(manual, auto.iloc[:3])
