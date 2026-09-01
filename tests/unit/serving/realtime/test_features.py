"""Live snapshot feature-frame compatibility tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.serving.realtime.features import (
    _ROBUST_Z_COLUMNS,
    _apply_robust_z,
    add_scenario_features,
    build_snapshot_features,
    engineer_features,
)

from tests.unit.serving.realtime.fixtures import (
    daily_snapshot_df,
    snapshot_feature_cols,
)


def test_build_snapshot_features_maps_to_ml_schema() -> None:
    out = build_snapshot_features(
        daily_snapshot_df(), decision_date=pd.Timestamp("2026-08-04")
    )
    assert np.issubdtype(out["trade_date"].dtype, np.datetime64)
    np.testing.assert_allclose(out["buy_price"].to_numpy(), out["close_price"].to_numpy())
    assert out["major_density"].notna().all()
    assert out["prog_dominance"].notna().all()
    # 로드된 번들의 feature_cols 는 스냅샷 컬럼에 포함되어야 합니다.
    assert set(snapshot_feature_cols()).issubset(out.columns)


def test_build_snapshot_features_preserves_display_metadata() -> None:
    out = build_snapshot_features(daily_snapshot_df())
    assert out["종목명"].tolist() == ["AAA", "BBB"]
    assert out["theme_sector"].tolist() == ["테마A", "테마A"]
    assert out["chart_analysis"].tolist() == ["거래량 폭증", "상따"]


def test_build_snapshot_features_buy_price_falls_back_to_close_price() -> None:
    snapshot = daily_snapshot_df()
    implicit = build_snapshot_features(snapshot)
    explicit = snapshot.copy()
    explicit["(매수 가격)"] = explicit["종가"]
    explicit_out = build_snapshot_features(explicit)
    np.testing.assert_allclose(
        implicit["buy_price"].to_numpy(), explicit_out["buy_price"].to_numpy()
    )
    pd.testing.assert_series_equal(
        implicit["buy_price_change_rate"], explicit_out["buy_price_change_rate"]
    )


def test_build_snapshot_features_keeps_explicit_buy_price() -> None:
    snapshot = daily_snapshot_df()
    snapshot["(매수 가격)"] = [9_999.0, 19_999.0]
    out = build_snapshot_features(snapshot)
    np.testing.assert_allclose(out["buy_price"].to_numpy(), [9_999.0, 19_999.0])


def test_build_snapshot_features_requires_close_price_without_buy_price() -> None:
    snapshot = daily_snapshot_df().drop(columns=["종가"])
    with pytest.raises(ValueError, match="close_price is missing"):
        build_snapshot_features(snapshot)


def test_build_snapshot_features_rejects_non_finite_close_price() -> None:
    snapshot = daily_snapshot_df()
    snapshot.loc[0, "종가"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_snapshot_features(snapshot)


def test_build_snapshot_features_rejects_non_positive_close_price() -> None:
    snapshot = daily_snapshot_df()
    snapshot.loc[0, "종가"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        build_snapshot_features(snapshot)


def test_engineer_features_created() -> None:
    cleaned = daily_snapshot_df().rename(columns={"등락률": "change_rate"})
    cleaned["prev_close_price"] = cleaned["종가"] * 0.99
    cleaned["open_price"] = cleaned["시가"]
    cleaned["high_price"] = cleaned["고가"]
    cleaned["low_price"] = cleaned["저가"]
    cleaned["close_price"] = cleaned["종가"]
    cleaned["market_cap_100m"] = cleaned["시가총액"]
    cleaned["trade_value_100m"] = cleaned["거래대금"]
    cleaned["selection_rank"] = cleaned["선정순위"]
    cleaned["inst_net_buy"] = cleaned["기관_순매수"]
    cleaned["foreign_net_buy"] = cleaned["외국인_순매수"]
    cleaned["prog_net_buy"] = cleaned["프로그램_순매수"]
    cleaned["market_type"] = cleaned["시장구분"]
    cleaned["total_candidate_count"] = cleaned["총_종목수"]
    cleaned["avg_trade_value"] = cleaned["평균_거래대금"]
    cleaned["kospi_change"] = cleaned["kospi"]
    cleaned["kosdaq_change"] = cleaned["kosdaq"]
    cleaned["buy_price"] = cleaned["종가"]
    cleaned["trade_date"] = pd.Timestamp("2026-08-04")
    engineered = engineer_features(cleaned)
    expected = {
        "buy_price_change_rate",
        "gap_ratio",
        "intraday_return",
        "major_density",
        "prog_dominance",
        "rank_ratio",
        "relative_change_rate",
        "change_rate_pct_rank",
        "log_market_cap_100m",
    }
    assert expected.issubset(set(engineered.columns))


def test_apply_robust_z_bounds() -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "trade_date": ["2026-08-04"] * 12,
            "change_rate": rng.normal(size=12),
            "major_density": rng.uniform(0, 1, size=12),
        }
    )
    out = _apply_robust_z(df, _ROBUST_Z_COLUMNS)
    for col in _ROBUST_Z_COLUMNS:
        z_col = f"{col}_z"
        if col not in df.columns:
            assert z_col not in out.columns
            continue
        assert z_col in out.columns
        vals = out[z_col].dropna()
        assert vals.between(-5, 5).all()
    assert out["change_rate_z"].notna().sum() > 0


def test_scenario_realtime_sangtta_feature_repair_01() -> None:
    """[SCENARIO_REALTIME_SANGTTA_FEATURE_REPAIR_01] build_snapshot_features generates
    all 11 scenario one-hot and context features accurately matching the active
    61-feature model bundle."""
    expected_one_hot = [
        "scenario_is_sangtta",
        "scenario_is_120_breakout",
        "scenario_is_volume_surge",
        "scenario_is_new_high",
        "scenario_is_near_new_high",
        "scenario_is_limitup_next_day",
        "scenario_is_rising_bearish",
        "scenario_other",
    ]
    expected_context = [
        "scenario_count_for_stock_date",
        "has_sangtta_for_stock_date",
        "is_multi_scenario_stock_date",
    ]
    out = build_snapshot_features(
        daily_snapshot_df(), decision_date=pd.Timestamp("2026-08-04")
    )
    for col in expected_one_hot + expected_context:
        assert col in out.columns, f"Missing scenario feature: {col}"

    sangtta_row = out[out["chart_analysis"] == "상따"].iloc[0]
    assert sangtta_row["scenario_is_sangtta"] == 1.0
    assert sangtta_row["has_sangtta_for_stock_date"] == 1.0


def test_add_scenario_features_generates_one_hot_and_context() -> None:
    """add_scenario_features produces 8 one-hot + 3 context features."""
    df = pd.DataFrame(
        {
            "chart_analysis": ["상따", "거래량 폭증", "상따"],
            "stock_code": ["000020", "000020", "000030"],
            "trade_date": ["2026-08-04", "2026-08-04", "2026-08-04"],
        }
    )
    out = add_scenario_features(df)
    expected_one_hot = [
        "scenario_is_sangtta",
        "scenario_is_120_breakout",
        "scenario_is_volume_surge",
        "scenario_is_new_high",
        "scenario_is_near_new_high",
        "scenario_is_limitup_next_day",
        "scenario_is_rising_bearish",
        "scenario_other",
    ]
    expected_context = [
        "scenario_count_for_stock_date",
        "has_sangtta_for_stock_date",
        "is_multi_scenario_stock_date",
    ]
    for col in expected_one_hot + expected_context:
        assert col in out.columns

    # Row 0: 상따 → scenario_is_sangtta=1
    assert out.iloc[0]["scenario_is_sangtta"] == 1.0
    # Row 1: 거래량 폭증 → scenario_is_volume_surge=1
    assert out.iloc[1]["scenario_is_volume_surge"] == 1.0
    # stock 000020 has 2 scenarios → is_multi_scenario_stock_date=1
    assert out.iloc[0]["is_multi_scenario_stock_date"] == 1.0
    assert out.iloc[0]["scenario_count_for_stock_date"] == 2.0


def test_ml_feature_compatibility_with_auto_theme() -> None:
    from src.serving.realtime.features import build_snapshot_features

    df_snapshot = pd.DataFrame([
        {
            "trade_date": "2026-09-01",
            "stock_code": "452430",
            "close_price": 15000,
            "prev_close_price": 14000,
            "open_price": 14200,
            "high_price": 15500,
            "low_price": 14100,
            "change_rate": 7.14,
            "market_cap_100m": 1000,
            "trade_value_100m": 200,
            "volume": 100000,
            "selection_rank": 1,
            "total_candidate_count": 2,
            "market_type": "KOSDAQ",
            "theme_sector": "반도체",
            "chart_analysis": "거래량 폭증",
            "kospi_change": 0.5,
            "kosdaq_change": 1.0,
            "v_kospi": 15.0,
            "v_kosdaq": 18.0,
        },
        {
            "trade_date": "2026-09-01",
            "stock_code": "005930",
            "close_price": 75000,
            "prev_close_price": 73000,
            "open_price": 73500,
            "high_price": 75500,
            "low_price": 73000,
            "change_rate": 2.74,
            "market_cap_100m": 4500000,
            "trade_value_100m": 5000,
            "volume": 7000000,
            "selection_rank": 2,
            "total_candidate_count": 2,
            "market_type": "KOSPI",
            "theme_sector": "반도체",
            "chart_analysis": "거래량 폭증",
            "kospi_change": 0.5,
            "kosdaq_change": 1.0,
            "v_kospi": 15.0,
            "v_kosdaq": 18.0,
        }
    ])

    features = build_snapshot_features(df_snapshot, decision_date=pd.Timestamp("2026-09-01"))
    assert "sector_relative_change" in features.columns
    expected_mean = (7.14 + 2.74) / 2
    assert np.isclose(features.loc[0, "sector_relative_change"], 7.14 - expected_mean, atol=1e-4)
    assert np.isclose(features.loc[1, "sector_relative_change"], 2.74 - expected_mean, atol=1e-4)
