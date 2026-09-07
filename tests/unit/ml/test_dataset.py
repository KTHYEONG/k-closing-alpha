"""build_ml_dataset: feature_set 분기 및 close_morning_history 가산 통합 계약."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.dataset import _ALLOWED_FEATURE_SETS, _EXCLUDED_FROM_X, build_ml_dataset
from src.ml.history_features import HISTORY_FEATURE_COLUMNS


def _raw_trade_log(n_dates: int = 30, per_day: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for d in pd.bdate_range("2024-01-02", periods=n_dates):
        for j in range(per_day):
            e = rng.normal()
            rows.append(
                {
                    "매수날짜": d.strftime("%Y-%m-%d"),
                    "종목코드": f"{j:06d}",
                    "(시가)": "10000",
                    "(고가)": "10400",
                    "(저가)": "9800",
                    "(종가)": "10200",
                    "(전일종가)": "10000",
                    "(시가총액, 억)": "5000",
                    "(거래대금, 억)": "300",
                    "(등락률)": f"{2 + e:.2f}",
                    "(선정 순위)": str(j + 1),
                    "(기관_순매수)": f"{e * 100:.0f}",
                    "(외국인_순매수)": f"{e * 80:.0f}",
                    "(프로그램_순매수)": f"{e * 50:.0f}",
                    "(체결강도)": "120",
                    "(시장구분)": "KOSPI",
                    "(총 종목 수)": str(per_day),
                    "(평균 거래대금)": "250",
                    "(kospi, %)": "0.3",
                    "(kosdaq, %)": "0.1",
                    "v_kospi": "18",
                    "v_kosdaq": "20",
                    "(거래량)": "100000",
                    "(테마/섹터)": "반도체",
                    "(차트분석)": "거래량 폭증",
                    "(매수 가격)": "10200",
                    "(매도 가격)": f"{10200 * (1 + 0.01 * e):.0f}",
                    "(수익률, %)": f"{e:.2f}",
                }
            )
    return pd.DataFrame(rows)


def _price_history(codes: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2023-09-01", periods=140)
    frames = []
    for code in codes:
        px = 10000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.015, size=140))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": code,
                    "open": px,
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "volume": np.full(140, 1e5),
                    "trade_value_100m": np.full(140, 300.0),
                    "inst_netbuy": rng.normal(0, 50, size=140),
                    "foreign_netbuy": rng.normal(0, 40, size=140),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_close_morning_history_registered_in_allowed_sets() -> None:
    assert "close_morning_history" in _ALLOWED_FEATURE_SETS
    assert "realized_vol" in _EXCLUDED_FROM_X


def test_close_morning_history_requires_price_history() -> None:
    raw = _raw_trade_log()
    with pytest.raises(ValueError, match="price_history"):
        build_ml_dataset(raw, None, feature_set="close_morning_history", price_history_df=None)


def test_close_morning_history_is_superset_and_excludes_realized_vol() -> None:
    raw = _raw_trade_log()
    codes = [f"{j:06d}" for j in range(6)]
    ph = _price_history(codes)

    x61, _t61, _c61, _p61 = build_ml_dataset(raw.copy(), None, feature_set="close_morning61")
    xh, _th, _ch, proc_h = build_ml_dataset(
        raw.copy(), None, feature_set="close_morning_history", price_history_df=ph
    )

    assert set(x61.columns).issubset(set(xh.columns))
    for col in HISTORY_FEATURE_COLUMNS:
        assert col in xh.columns
    assert "realized_vol" not in xh.columns
    # realized_vol survives on the processed frame for downstream sizing and is finite/positive
    assert "realized_vol" in proc_h.columns
    rv = proc_h["realized_vol"].to_numpy(dtype=float)
    assert np.isfinite(rv).all()
    assert (rv > 0.0).all()


def test_close_morning61_unaffected_by_history_branch() -> None:
    raw = _raw_trade_log()
    x1, _t1, c1, p1 = build_ml_dataset(raw.copy(), None, feature_set="close_morning61")
    x2, _t2, c2, p2 = build_ml_dataset(raw.copy(), None, feature_set="close_morning61")
    assert sorted(x1.columns) == sorted(x2.columns)
    assert sorted(c1) == sorted(c2)
    assert "realized_vol" not in x1.columns
    np.testing.assert_allclose(
        p1.sort_index()["target_return"].to_numpy(),
        p2.sort_index()["target_return"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_build_ml_dataset_output_unchanged_after_vectorization() -> None:
    raw = _raw_trade_log()
    x1, t1, c1, p1 = build_ml_dataset(raw.copy(), None, feature_set="close_morning61", panel_mode="scenario_action")
    x2, t2, c2, p2 = build_ml_dataset(raw.copy(), None, feature_set="close_morning61", panel_mode="scenario_action")
    assert sorted(x1.columns) == sorted(x2.columns)
    for col in x1.columns:
        a = x1.sort_index()[col].reset_index(drop=True)
        b = x2.sort_index()[col].reset_index(drop=True)
        try:
            np.testing.assert_allclose(
                a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                rtol=1e-12, atol=1e-15, equal_nan=True,
            )
        except (ValueError, TypeError):
            np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    np.testing.assert_allclose(
        p1.sort_index()["target_return"].to_numpy(), p2.sort_index()["target_return"].to_numpy(),
        rtol=1e-12, atol=1e-15,
    )
    np.testing.assert_array_equal(
        p1.sort_index()["target_rank"].to_numpy(), p2.sort_index()["target_rank"].to_numpy()
    )

def test_label_source_is_excluded_from_model_features() -> None:
    import pandas as pd

    from src.data.candidate_panel import LABEL_SOURCE_COLUMN
    from src.ml.dataset import _EXCLUDED_FROM_X, build_ml_dataset

    # Arrange
    assert LABEL_SOURCE_COLUMN in _EXCLUDED_FROM_X
    trade_log = pd.read_parquet("data/parquet/trade_log.parquet").head(2000)
    theme = pd.read_parquet("data/parquet/theme.parquet")
    tagged = trade_log.copy()
    tagged[LABEL_SOURCE_COLUMN] = "sheet_executed"

    # Act
    x_plain, _, cat_plain, _ = build_ml_dataset(trade_log, theme, feature_set="close_morning61")
    x_tagged, _, cat_tagged, _ = build_ml_dataset(tagged, theme, feature_set="close_morning61")

    # Assert
    assert LABEL_SOURCE_COLUMN not in x_tagged.columns
    assert LABEL_SOURCE_COLUMN not in cat_tagged
    assert list(x_tagged.columns) == list(x_plain.columns)
    assert cat_tagged == cat_plain


def test_create_multi_targets_subtracts_measured_round_trip_cost() -> None:
    import numpy as np

    from src.ml.dataset import LABEL_THRESHOLDS, create_multi_targets
    from src.serving.realtime.inference import ROUND_TRIP_COST_RATIO

    # Given: net_return in percent units, spanning both label thresholds post-cost
    df = pd.DataFrame(
        {
            "trade_date": ["2026-01-05", "2026-01-05", "2026-01-06", "2026-01-06"],
            "net_return": [3.0, -1.0, 0.5, -5.0],
        }
    )

    # When
    out = create_multi_targets(df, clip_lower=-0.10, clip_upper=0.10)

    # Then
    net_of_cost = df["net_return"].to_numpy() / 100.0 - ROUND_TRIP_COST_RATIO
    np.testing.assert_allclose(
        out["target_return"].to_numpy(),
        np.clip(net_of_cost, -0.10, 0.10),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        out["target_good"].to_numpy(),
        (net_of_cost >= LABEL_THRESHOLDS["target_good"]).astype(int),
    )
    np.testing.assert_array_equal(
        out["target_bad"].to_numpy(),
        (net_of_cost <= LABEL_THRESHOLDS["target_bad"]).astype(int),
    )


def test_retarget_with_clip_matches_create_multi_targets_cost_basis() -> None:
    import numpy as np

    from src.ml.dataset import create_multi_targets, retarget_with_clip

    # Given
    df = pd.DataFrame(
        {
            "trade_date": ["2026-02-02", "2026-02-02", "2026-02-03", "2026-02-03", "2026-02-04", "2026-02-04"],
            "net_return": [4.0, -2.0, 1.0, -0.5, 2.5, -3.0],
        }
    )
    base = create_multi_targets(df, clip_lower=-0.10, clip_upper=0.10)

    # When: retarget the same frame to a tighter candidate clip
    tight = retarget_with_clip(base, -0.08, 0.08)
    base_again = retarget_with_clip(base, -0.10, 0.10)

    # Then: identical cost basis -> identical target_return at identical bounds
    np.testing.assert_allclose(
        base_again["target_return"].to_numpy(), base["target_return"].to_numpy(), rtol=0.0, atol=1e-12
    )
    # tighter clip only ever pulls values inward, never shifts the un-clipped middle
    mid = np.abs(base["target_return"].to_numpy()) < 0.08
    np.testing.assert_allclose(
        tight["target_return"].to_numpy()[mid], base["target_return"].to_numpy()[mid], rtol=0.0, atol=1e-12
    )


def _raw_trade_log_scenario_auto(n_dates: int = 40, per_day: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for d in pd.bdate_range("2024-01-02", periods=n_dates):
        for j in range(per_day):
            e = rng.normal()
            rows.append({
                "매수날짜": d.strftime("%Y-%m-%d"), "종목코드": f"{j:06d}",
                "(시가)": "10000", "(고가)": "10400", "(저가)": "9800", "(종가)": "10200",
                "(전일종가)": "10000", "(시가총액, 억)": "5000", "(거래대금, 억)": "300",
                "(등락률)": f"{11 + e:.2f}", "(선정 순위)": str(j + 1),
                "(기관_순매수)": f"{e*100:.0f}", "(외국인_순매수)": f"{e*80:.0f}",
                "(프로그램_순매수)": f"{e*50:.0f}", "(체결강도)": "120",
                "(시장구분)": "KOSPI", "(총 종목 수)": str(per_day), "(평균 거래대금)": "250",
                "(kospi, %)": "0.3", "(kosdaq, %)": "0.1", "v_kospi": "18", "v_kosdaq": "20",
                "(거래량)": "100000", "(테마/섹터)": "반도체", "(차트분석)": "거래량 폭증",
                "(매수 가격)": "10200", "(매도 가격)": f"{10200*(1+0.01*e):.0f}",
                "(수익률, %)": f"{e:.2f}",
            })
    return pd.DataFrame(rows)


def _price_history_scenario_auto(codes: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2023-06-01", periods=200)
    frames = []
    for code in codes:
        px = 10000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.015, size=200))
        frames.append(pd.DataFrame({
            "date": dates, "symbol": code, "open": px, "high": px * 1.02,
            "low": px * 0.98, "close": px, "volume": 1e5, "trade_value_100m": 300.0,
            "inst_netbuy": 0.0, "foreign_netbuy": 0.0, "daily_change_pct": 0.02,
        }))
    return pd.concat(frames, ignore_index=True)


def test_build_ml_dataset_scenario_source_auto_preserves_schema() -> None:
    # Arrange
    raw = _raw_trade_log_scenario_auto()
    # price_history covers only 4 of the 6 daily codes -> the other 2 must fall to '미분류'
    ph = _price_history_scenario_auto(["000000", "000001", "000002", "000003"])

    x_manual, _t_m, cat_m, _p_m = build_ml_dataset(
        raw.copy(), None, feature_set="close_morning61", panel_mode="scenario_action",
        price_history_df=ph,
    )
    x_auto, _t_a, cat_a, proc_a = build_ml_dataset(
        raw.copy(), None, feature_set="close_morning61", panel_mode="scenario_action",
        price_history_df=ph, scenario_source="auto",
    )

    # Assert: identical feature schema, auto ran the derivation, no null scenario key
    assert list(x_auto.columns) == list(x_manual.columns)
    assert cat_a == cat_m
    assert proc_a["chart_analysis"].isna().sum() == 0
    assert proc_a.attrs.get("scenario_source") == "auto"

    with pytest.raises(ValueError, match="price_history"):
        build_ml_dataset(raw.copy(), None, feature_set="close_morning61",
                         panel_mode="scenario_action", price_history_df=None,
                         scenario_source="auto")


def _raw_trade_log_scenario_none(n_dates: int = 30, per_day: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    scen_cycle = ["거래량 폭증", "신고가", "상따", "120 돌파", "미분류", "신고가 근접"]
    for d in pd.bdate_range("2024-01-02", periods=n_dates):
        for j in range(per_day):
            e = rng.normal()
            rows.append({
                "매수날짜": d.strftime("%Y-%m-%d"), "종목코드": f"{j:06d}",
                "(시가)": "10000", "(고가)": "10400", "(저가)": "9800", "(종가)": "10200",
                "(전일종가)": "10000", "(시가총액, 억)": "5000", "(거래대금, 억)": "300",
                "(등락률)": f"{11 + e:.2f}", "(선정 순위)": str(j + 1),
                "(기관_순매수)": f"{e*100:.0f}", "(외국인_순매수)": f"{e*80:.0f}",
                "(프로그램_순매수)": f"{e*50:.0f}", "(체결강도)": "120",
                "(시장구분)": "KOSPI", "(총 종목 수)": str(per_day), "(평균 거래대금)": "250",
                "(kospi, %)": "0.3", "(kosdaq, %)": "0.1", "v_kospi": "18", "v_kosdaq": "20",
                "(거래량)": "100000", "(테마/섹터)": "반도체",
                "(차트분석)": scen_cycle[j % len(scen_cycle)],
                "(매수 가격)": "10200", "(매도 가격)": f"{10200*(1+0.01*e):.0f}",
                "(수익률, %)": f"{e:.2f}",
            })
    return pd.DataFrame(rows)


def test_build_ml_dataset_scenario_source_none_withholds_scenario_features() -> None:
    from src.ml.scenario_panel import SCENARIO_CONTEXT_FEATURES, SCENARIO_ONE_HOT_FEATURES

    raw = _raw_trade_log_scenario_none()

    x_manual, _tm, _cm, _pm = build_ml_dataset(
        raw.copy(), None, feature_set="close_morning61", panel_mode="scenario_action",
    )
    x_none, _tn, _cn, _pn = build_ml_dataset(
        raw.copy(), None, feature_set="close_morning61", panel_mode="scenario_action",
        scenario_source="none",
    )

    banned = set(SCENARIO_ONE_HOT_FEATURES) | set(SCENARIO_CONTEXT_FEATURES)
    assert banned & set(x_manual.columns), "guard: manual mode must expose the scenario block"
    assert not (banned & set(x_none.columns))
    assert not [c for c in x_none.columns if c.startswith("scenario_")]
    # non-scenario features retained
    kept = set(x_manual.columns) - banned
    assert kept.issubset(set(x_none.columns))


def test_build_ml_dataset_scenario_source_rejects_unknown() -> None:
    raw = _raw_trade_log_scenario_auto()
    with pytest.raises(ValueError, match="scenario_source"):
        build_ml_dataset(raw.copy(), None, feature_set="close_morning61", scenario_source="bogus")
