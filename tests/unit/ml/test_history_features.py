import pandas as pd
import pytest


def test_load_price_history_missing_column_raises(tmp_path) -> None:
    import pandas as pd
    import pytest

    from src.ml.history_features import load_price_history

    # Given: a parquet missing the 'inst_netbuy' required column
    bad = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=3),
            "symbol": ["000660", "000660", "000660"],
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [10.0, 10.0, 10.0],
            "trade_value_100m": [5.0, 5.0, 5.0],
            "foreign_netbuy": [0.0, 0.0, 0.0],
        }
    )
    path = tmp_path / "ph.parquet"
    bad.to_parquet(path)

    # When / Then
    with pytest.raises(ValueError, match="inst_netbuy"):
        load_price_history(path)


def test_load_price_history_normalizes_symbol_and_sorts(tmp_path) -> None:
    import pandas as pd

    from src.ml.history_features import load_price_history

    raw = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "symbol": [660, 660, 660],
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [3.0, 1.0, 2.0],
            "volume": [1.0, 1.0, 1.0],
            "trade_value_100m": [1.0, 1.0, 1.0],
            "inst_netbuy": [0.0, 0.0, 0.0],
            "foreign_netbuy": [0.0, 0.0, 0.0],
        }
    )
    path = tmp_path / "ph.parquet"
    raw.to_parquet(path)

    out = load_price_history(path)

    assert list(out["symbol"].unique()) == ["000660"]
    assert out["date"].is_monotonic_increasing
    # duplicate (symbol, 2024-01-02) collapsed to one row, keeping last (close == 2.0)
    jan2 = out.loc[out["date"] == pd.Timestamp("2024-01-02")]
    assert len(jan2) == 1
    assert float(jan2["close"].iloc[0]) == 2.0


def test_compute_trailing_frame_momentum_known_series() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.history_features import compute_trailing_frame

    # Given: 30 sessions of exactly +1%/day compounding for one symbol
    n = 30
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100.0 * (1.01 ** np.arange(n))
    ph = pd.DataFrame(
        {
            "symbol": ["000660"] * n,
            "date": dates,
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": np.full(n, 1000.0),
            "trade_value_100m": np.full(n, 50.0),
            "inst_netbuy": np.zeros(n),
            "foreign_netbuy": np.zeros(n),
        }
    )

    trailing = compute_trailing_frame(ph)

    last = trailing.iloc[-1]
    # 5 trailing daily log returns of ln(1.01) each
    assert last["ret_5d"] == pytest.approx(5.0 * np.log(1.01), rel=1e-6)
    assert last["up_day_ratio_10d"] == pytest.approx(1.0)
    assert last["dist_ma5"] > 0.0


def test_attach_history_features_no_lookahead() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.history_features import attach_history_features

    dates = pd.bdate_range("2024-01-01", periods=20)
    # flat at 100 for 19 sessions, then a +50% spike on the last session (== trade_date)
    close = np.concatenate([np.full(19, 100.0), [150.0]])
    ph = pd.DataFrame(
        {
            "symbol": ["000660"] * 20,
            "date": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(20, 1000.0),
            "trade_value_100m": np.full(20, 50.0),
            "inst_netbuy": np.zeros(20),
            "foreign_netbuy": np.zeros(20),
        }
    )
    panel = pd.DataFrame({"trade_date": [dates[-1]], "stock_code": ["000660"]})

    out = attach_history_features(panel, ph, date_col="trade_date", code_col="stock_code")

    # Trailing features use data STRICTLY before 2024-01-26 -> all flat -> zero momentum
    assert out["ret_5d"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["ret_20d"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert abs(out["dist_ma20"].iloc[0]) < 1e-9


def test_attach_history_features_missing_symbol_safe() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.history_features import (
        _DEFAULT_REALIZED_VOL_FALLBACK,
        HISTORY_FEATURE_COLUMNS,
        attach_history_features,
    )

    dates = pd.bdate_range("2024-01-01", periods=20)
    ph = pd.DataFrame(
        {
            "symbol": ["000660"] * 20,
            "date": dates,
            "open": np.full(20, 100.0),
            "high": np.full(20, 100.0),
            "low": np.full(20, 100.0),
            "close": np.full(20, 100.0),
            "volume": np.full(20, 1000.0),
            "trade_value_100m": np.full(20, 50.0),
            "inst_netbuy": np.zeros(20),
            "foreign_netbuy": np.zeros(20),
        }
    )
    panel = pd.DataFrame({"trade_date": [dates[-1]], "stock_code": ["999999"]})

    out = attach_history_features(panel, ph, date_col="trade_date", code_col="stock_code")

    assert len(out) == 1
    for col in HISTORY_FEATURE_COLUMNS:
        assert pd.isna(out[col].iloc[0])
    assert out["realized_vol"].iloc[0] == pytest.approx(_DEFAULT_REALIZED_VOL_FALLBACK)


def test_attach_history_features_preserves_row_order_and_realized_vol_positive() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.history_features import attach_history_features

    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=60)
    frames = []
    for sym in ("000660", "005930", "035720"):
        px = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, size=60))
        frames.append(
            pd.DataFrame(
                {
                    "symbol": sym,
                    "date": dates,
                    "open": px,
                    "high": px * 1.01,
                    "low": px * 0.99,
                    "close": px,
                    "volume": rng.uniform(500, 1500, size=60),
                    "trade_value_100m": rng.uniform(10, 90, size=60),
                    "inst_netbuy": rng.normal(0, 100, size=60),
                    "foreign_netbuy": rng.normal(0, 80, size=60),
                }
            )
        )
    ph = pd.concat(frames, ignore_index=True)
    panel = pd.DataFrame(
        {
            "trade_date": [dates[40], dates[45], dates[50]],
            "stock_code": ["005930", "000660", "035720"],
        },
        index=[11, 7, 3],
    )

    out = attach_history_features(panel, ph, date_col="trade_date", code_col="stock_code")

    assert list(out.index) == [11, 7, 3]
    assert list(out["stock_code"]) == ["005930", "000660", "035720"]
    assert np.isfinite(out["realized_vol"].to_numpy()).all()
    assert (out["realized_vol"].to_numpy() > 0.0).all()


def test_build_ml_dataset_close_morning_history_superset() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.dataset import build_ml_dataset
    from src.ml.history_features import HISTORY_FEATURE_COLUMNS

    rng = np.random.default_rng(5)
    rows = []
    codes = [f"{j:06d}" for j in range(6)]
    dates = pd.bdate_range("2024-01-02", periods=40)
    for d in dates:
        for j, code in enumerate(codes):
            e = rng.normal()
            rows.append(
                {
                    "매수날짜": d.strftime("%Y-%m-%d"),
                    "종목코드": code,
                    "(시가)": "10000", "(고가)": "10400", "(저가)": "9800",
                    "(종가)": "10200", "(전일종가)": "10000",
                    "(시가총액, 억)": "5000", "(거래대금, 억)": "300",
                    "(등락률)": f"{2 + e:.2f}", "(선정 순위)": str(j + 1),
                    "(기관_순매수)": f"{e * 100:.0f}", "(외국인_순매수)": f"{e * 80:.0f}",
                    "(프로그램_순매수)": f"{e * 50:.0f}", "(체결강도)": "120",
                    "(시장구분)": "KOSPI", "(총 종목 수)": "6", "(평균 거래대금)": "250",
                    "(kospi, %)": "0.3", "(kosdaq, %)": "0.1", "v_kospi": "18", "v_kosdaq": "20",
                    "(거래량)": "100000", "(테마/섹터)": "반도체",
                    "(차트분석)": "거래량 폭증", "(매수 가격)": "10200",
                    "(매도 가격)": f"{10200 * (1 + 0.01 * e):.0f}", "(수익률, %)": f"{e:.2f}",
                }
            )
    raw = pd.DataFrame(rows)

    ph_frames = []
    ph_dates = pd.bdate_range("2023-09-01", periods=140)
    for code in codes:
        px = 10000.0 * np.cumprod(1.0 + rng.normal(0.0, 0.015, size=140))
        ph_frames.append(
            pd.DataFrame(
                {
                    "date": ph_dates, "symbol": code, "open": px, "high": px * 1.01,
                    "low": px * 0.99, "close": px, "volume": np.full(140, 1e5),
                    "trade_value_100m": np.full(140, 300.0),
                    "inst_netbuy": rng.normal(0, 50, size=140),
                    "foreign_netbuy": rng.normal(0, 40, size=140),
                }
            )
        )
    price_history_df = pd.concat(ph_frames, ignore_index=True)

    x61, _t61, _c61, _p61 = build_ml_dataset(raw.copy(), None, feature_set="close_morning61")
    xh, _th, _ch, _ph = build_ml_dataset(
        raw.copy(), None, feature_set="close_morning_history", price_history_df=price_history_df
    )

    assert set(x61.columns).issubset(set(xh.columns))
    for col in HISTORY_FEATURE_COLUMNS:
        assert col in xh.columns
    assert "realized_vol" not in xh.columns

    with pytest.raises(ValueError, match="price_history_df"):
        build_ml_dataset(raw.copy(), None, feature_set="close_morning_history", price_history_df=None)


def test_close_morning61_legacy_parity_unaffected() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.dataset import build_ml_dataset

    rng = np.random.default_rng(11)
    rows = []
    for d in pd.bdate_range("2024-01-02", periods=25):
        for j in range(6):
            e = rng.normal()
            rows.append(
                {
                    "매수날짜": d.strftime("%Y-%m-%d"), "종목코드": f"{j:06d}",
                    "(시가)": "10000", "(고가)": "10400", "(저가)": "9800",
                    "(종가)": "10200", "(전일종가)": "10000",
                    "(시가총액, 억)": "5000", "(거래대금, 억)": "300",
                    "(등락률)": f"{2 + e:.2f}", "(선정 순위)": str(j + 1),
                    "(기관_순매수)": f"{e * 100:.0f}", "(외국인_순매수)": f"{e * 80:.0f}",
                    "(프로그램_순매수)": f"{e * 50:.0f}", "(체결강도)": "120",
                    "(시장구분)": "KOSPI", "(총 종목 수)": "6", "(평균 거래대금)": "250",
                    "(kospi, %)": "0.3", "(kosdaq, %)": "0.1", "v_kospi": "18", "v_kosdaq": "20",
                    "(거래량)": "100000", "(테마/섹터)": "반도체",
                    "(차트분석)": "거래량 폭증", "(매수 가격)": "10200",
                    "(매도 가격)": f"{10200 * (1 + 0.01 * e):.0f}", "(수익률, %)": f"{e:.2f}",
                }
            )
    raw = pd.DataFrame(rows)

    x1, _t1, c1, p1 = build_ml_dataset(raw.copy(), None, feature_set="close_morning61")
    x2, _t2, c2, p2 = build_ml_dataset(raw.copy(), None, feature_set="close_morning61")

    assert sorted(x1.columns) == sorted(x2.columns)
    assert sorted(c1) == sorted(c2)
    np.testing.assert_allclose(
        p1.sort_index()["target_return"].to_numpy(),
        p2.sort_index()["target_return"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_price_history_parquet_path_setting() -> None:
    from src import settings

    path = settings.PRICE_HISTORY_PARQUET_PATH
    assert path.name == "price_history.parquet"
    assert path.parent.name == "history"
