import pytest


def test_build_pit_sector_map_no_lookahead() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.sector_features import build_pit_sector_map

    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2022-01-03", "2024-12-31")
    syms = [f"{i:06d}" for i in range(24)]
    frames = []
    for _i, s in enumerate(syms):  # noqa: B007 - spec skeleton
        base = rng.normal(0, 0.01, size=len(dates))
        px = 1000.0 * np.cumprod(1.0 + base)
        frames.append(pd.DataFrame({"date": dates, "symbol": s, "close": px}))
    ph = pd.concat(frames, ignore_index=True)

    m1 = build_pit_sector_map(ph, n_clusters=3, lookback_days=400, min_obs=100)

    # 2024년 구간의 종가를 크게 왜곡해도 2024년 라벨은 불변이어야 한다
    ph2 = ph.copy()
    hit = ph2["date"] >= pd.Timestamp("2024-01-01")
    ph2.loc[hit, "close"] = ph2.loc[hit, "close"] * 5.0
    m2 = build_pit_sector_map(ph2, n_clusters=3, lookback_days=400, min_obs=100)

    a = m1[m1["year"] == 2024].sort_values("symbol").reset_index(drop=True)
    b = m2[m2["year"] == 2024].sort_values("symbol").reset_index(drop=True)
    assert not a.empty
    pd.testing.assert_frame_equal(a, b)


def test_sector_mom_5d_excludes_current_day() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.sector_features import compute_market_sector_returns

    dates = pd.bdate_range("2024-01-01", periods=20)
    syms = ["000001", "000002", "000003"]
    frames = []
    for s in syms:
        px = np.full(len(dates), 1000.0)
        frames.append(pd.DataFrame({"date": dates, "symbol": s, "close": px}))
    ph = pd.concat(frames, ignore_index=True)
    smap = pd.DataFrame({"year": 2024, "symbol": syms, "cluster_id": [0, 0, 0]})

    agg_flat = compute_market_sector_returns(ph, smap)
    last = agg_flat["date"].max()
    mom_flat = float(agg_flat.loc[agg_flat["date"] == last, "sector_mom_5d"].iloc[0])

    # 마지막 날만 +50% 급등시킨다
    ph2 = ph.copy()
    ph2.loc[ph2["date"] == last, "close"] = 1500.0
    agg_spike = compute_market_sector_returns(ph2, smap)
    mom_spike = float(agg_spike.loc[agg_spike["date"] == last, "sector_mom_5d"].iloc[0])
    ret_spike = float(agg_spike.loc[agg_spike["date"] == last, "sector_mkt_ret"].iloc[0])

    assert ret_spike > 40.0  # 당일 수익률에는 반영된다
    assert mom_spike == pytest.approx(mom_flat, abs=1e-9)  # 트레일링에는 반영되지 않는다


def test_attach_sector_features_full_coverage_and_order() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.sector_features import SECTOR_FEATURE_COLUMNS, attach_sector_features

    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2022-01-03", "2024-06-28")
    syms = [f"{i:06d}" for i in range(30)]
    frames = []
    for s in syms:
        px = 1000.0 * np.cumprod(1.0 + rng.normal(0, 0.012, size=len(dates)))
        frames.append(pd.DataFrame({"date": dates, "symbol": s, "close": px}))
    ph = pd.concat(frames, ignore_index=True)

    d = dates[dates >= pd.Timestamp("2024-03-01")][:3]
    panel = pd.DataFrame(
        {
            "trade_date": [d[0], d[1], d[2]],
            "stock_code": ["000005", "000012", "000020"],
            "change_rate": [3.0, -1.5, 0.5],
        },
        index=[9, 4, 7],
    )

    out = attach_sector_features(panel, ph, n_clusters=4)

    assert list(out.index) == [9, 4, 7]
    assert list(out["stock_code"]) == ["000005", "000012", "000020"]
    for col in SECTOR_FEATURE_COLUMNS:
        assert col in out.columns
    assert out["sector_rel_mkt"].notna().all()
    # sector_rel_mkt = change_rate - sector_mkt_ret
    np.testing.assert_allclose(
        out["sector_rel_mkt"].to_numpy(dtype=float),
        out["change_rate"].to_numpy(dtype=float) - out["sector_mkt_ret"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-12,
    )


def test_attach_sector_features_unassigned_symbol_is_nan() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.sector_features import SECTOR_FEATURE_COLUMNS, attach_sector_features

    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2022-01-03", "2024-06-28")
    syms = [f"{i:06d}" for i in range(20)]
    frames = []
    for s in syms:
        px = 1000.0 * np.cumprod(1.0 + rng.normal(0, 0.01, size=len(dates)))
        frames.append(pd.DataFrame({"date": dates, "symbol": s, "close": px}))
    ph = pd.concat(frames, ignore_index=True)

    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-03-04")],
            "stock_code": ["999999"],
            "change_rate": [2.0],
        }
    )

    out = attach_sector_features(panel, ph, n_clusters=3)

    assert len(out) == 1
    for col in SECTOR_FEATURE_COLUMNS:
        assert pd.isna(out[col].iloc[0])


def test_close_morning_sector_replaces_degenerate_feature() -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.dataset import _ALLOWED_FEATURE_SETS, build_ml_dataset
    from src.ml.sector_features import SECTOR_FEATURE_COLUMNS

    assert "close_morning_sector" in _ALLOWED_FEATURE_SETS

    rng = np.random.default_rng(5)
    codes = [f"{j:06d}" for j in range(6)]
    rows = []
    for d in pd.bdate_range("2024-03-01", periods=30):
        for j, code in enumerate(codes):
            e = rng.normal()
            rows.append(
                {
                    "\ub9e4\uc218\ub0a0\uc9dc": d.strftime("%Y-%m-%d"), "\uc885\ubaa9\ucf54\ub4dc": code,
                    "(\uc2dc\uac00)": "10000", "(\uace0\uac00)": "10400", "(\uc800\uac00)": "9800",
                    "(\uc885\uac00)": "10200", "(\uc804\uc77c\uc885\uac00)": "10000",
                    "(\uc2dc\uac00\ucd1d\uc561, \uc5b5)": "5000", "(\uac70\ub798\ub300\uae08, \uc5b5)": "300",
                    "(\ub4f1\ub77d\ub960)": f"{2 + e:.2f}", "(\uc120\uc815 \uc21c\uc704)": str(j + 1),
                    "(\uae30\uad00_\uc21c\ub9e4\uc218)": f"{e * 100:.0f}", "(\uc678\uad6d\uc778_\uc21c\ub9e4\uc218)": f"{e * 80:.0f}",
                    "(\ud504\ub85c\uadf8\ub7a8_\uc21c\ub9e4\uc218)": f"{e * 50:.0f}", "(\uccb4\uacb0\uac15\ub3c4)": "120",
                    "(\uc2dc\uc7a5\uad6c\ubd84)": "KOSPI", "(\ucd1d \uc885\ubaa9 \uc218)": "6", "(\ud3c9\uade0 \uac70\ub798\ub300\uae08)": "250",
                    "(kospi, %)": "0.3", "(kosdaq, %)": "0.1", "v_kospi": "18", "v_kosdaq": "20",
                    "(\uac70\ub798\ub7c9)": "100000", "(\ud14c\ub9c8/\uc139\ud130)": "\ubc18\ub3c4\uccb4",
                    "(\ucc28\ud2b8\ubd84\uc11d)": "\uac70\ub798\ub7c9 \ud3ed\uc99d", "(\ub9e4\uc218 \uac00\uaca9)": "10200",
                    "(\ub9e4\ub3c4 \uac00\uaca9)": f"{10200 * (1 + 0.01 * e):.0f}", "(\uc218\uc775\ub960, %)": f"{e:.2f}",
                }
            )
    raw = pd.DataFrame(rows)

    ph_dates = pd.bdate_range("2022-01-03", "2024-04-30")
    ph = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": ph_dates, "symbol": c,
                    "open": 10000.0, "high": 10400.0, "low": 9800.0,
                    "close": 10000.0 * np.cumprod(1.0 + rng.normal(0, 0.012, size=len(ph_dates))),
                    "volume": 1e5, "trade_value_100m": 300.0,
                    "inst_netbuy": rng.normal(0, 50, size=len(ph_dates)),
                    "foreign_netbuy": rng.normal(0, 40, size=len(ph_dates)),
                }
            )
            for c in codes
        ],
        ignore_index=True,
    )

    xs, _t, _c, _p = build_ml_dataset(
        raw.copy(), None, feature_set="close_morning_sector", price_history_df=ph
    )

    assert "sector_relative_change" not in xs.columns
    for col in SECTOR_FEATURE_COLUMNS:
        assert col in xs.columns
    assert "sector_cluster_id" not in xs.columns

    with pytest.raises(ValueError, match="price_history"):  # noqa: PT011 - spec skeleton expanded for ruff
        build_ml_dataset(raw.copy(), None, feature_set="close_morning_sector", price_history_df=None)
