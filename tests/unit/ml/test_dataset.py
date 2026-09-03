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
