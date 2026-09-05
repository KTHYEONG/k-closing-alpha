def test_classify_ceiling_entry_flags_limit_up_close_only() -> None:
    import pandas as pd

    from src.ml.buyability import classify_ceiling_entry

    # Given: ceiling close at the high, ceiling close beaten intraday, ordinary strong close, bad prev_close
    df = pd.DataFrame(
        {
            "close_price": [13000.0, 13000.0, 12500.0, 13000.0],
            "prev_close_price": [10000.0, 10000.0, 10000.0, 0.0],
            "high_price": [13000.0, 13500.0, 12800.0, 13000.0],
        }
    )

    # When
    flag = classify_ceiling_entry(df)

    # Then
    assert flag.tolist() == [True, False, False, False]
    assert flag.dtype == bool
    assert not flag.isna().any()


def test_attach_entry_auction_liquidity_marks_missing_partition_as_unmeasured(tmp_path) -> None:
    import pandas as pd

    from src.ml.buyability import attach_entry_auction_liquidity

    # Given: a panel whose entry dates have no intraday partition anywhere under intraday_root
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2019-05-02", "2019-05-03"]),
            "stock_code": ["005930", "000660"],
        }
    )

    # When
    out = attach_entry_auction_liquidity(df, intraday_root=tmp_path)

    # Then: unmeasured is distinguishable from a measured zero
    assert len(out) == len(df)
    assert out["auction_bars_found"].tolist() == [False, False]
    assert out["auction_value_100m"].isna().all()


def test_estimate_fill_ratio_caps_at_one_and_rejects_bad_inputs() -> None:
    import numpy as np
    import pytest

    from src.ml.buyability import estimate_fill_ratio

    # Given: a deep auction, a thin one, a measured zero, and an unmeasured row
    auction = np.array([100.0, 0.24, 0.0, np.nan])

    # When: 10% participation against a 1.0억 target
    ratio = estimate_fill_ratio(auction, target_notional_100m=1.0, participation_cap=0.10)

    # Then
    assert ratio[0] == pytest.approx(1.0)
    assert ratio[1] == pytest.approx(0.024)
    assert ratio[2] == pytest.approx(0.0)
    assert np.isnan(ratio[3])

    with pytest.raises(ValueError, match="target_notional_100m"):
        estimate_fill_ratio(auction, target_notional_100m=0.0)
    with pytest.raises(ValueError, match="participation_cap"):
        estimate_fill_ratio(auction, target_notional_100m=1.0, participation_cap=1.5)


def test_apply_buyability_gate_annotates_without_dropping_rows() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.buyability import apply_buyability_gate

    # Given: a deep-auction row, a thin ceiling row, and an unmeasured row
    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-03-02"] * 3),
            "stock_code": ["005930", "111111", "222222"],
            "close_price": [12500.0, 13000.0, 11000.0],
            "prev_close_price": [10000.0, 10000.0, 10000.0],
            "high_price": [12800.0, 13000.0, 11200.0],
            "auction_value_100m": [50.0, 0.24, np.nan],
            "auction_vol_share": [0.010, 0.0004, np.nan],
            "auction_bars_found": [True, True, False],
        }
    )

    # When
    gated, provenance = apply_buyability_gate(df, target_notional_100m=1.0, min_fill_ratio=1.0)

    # Then
    assert len(gated) == 3
    assert gated["is_buyable"].tolist() == [True, False, True]
    assert gated["is_ceiling_entry"].tolist() == [False, True, False]
    assert provenance["n_rows"] == 3
    assert provenance["n_blocked"] == 1
    assert provenance["n_unmeasured"] == 1

    # And: idempotent
    again, _ = apply_buyability_gate(gated, target_notional_100m=1.0, min_fill_ratio=1.0)
    pd.testing.assert_frame_equal(gated, again[gated.columns])


def test_evaluate_buyability_sleeves_reselects_top1_within_fillable_sleeve() -> None:
    import pandas as pd
    import pytest

    from src.ml.buyability import evaluate_buyability_sleeves

    # Given: 40 days where the top-scored pick is always a thin-auction ceiling row worth +6%,
    # and the runner-up is an ordinary fillable row worth +0.2%
    dates = pd.bdate_range("2026-01-05", periods=40)
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "trade_date": d, "stock_code": f"9{i:05d}", "pred": 0.9,
                "net_return": 6.0, "close_price": 13000.0,
                "prev_close_price": 10000.0, "high_price": 13000.0,
                "auction_value_100m": 0.05, "auction_vol_share": 0.0002,
                "auction_bars_found": True,
            }
        )
        rows.append(
            {
                "trade_date": d, "stock_code": f"1{i:05d}", "pred": 0.5,
                "net_return": 0.2, "close_price": 11000.0,
                "prev_close_price": 10000.0, "high_price": 11200.0,
                "auction_value_100m": 40.0, "auction_vol_share": 0.011,
                "auction_bars_found": True,
            }
        )
    oof = pd.DataFrame(rows)

    # When
    results = evaluate_buyability_sleeves(oof, target_notional_100m=1.0)
    by_sleeve = {r.sleeve: r for r in results}

    # Then: the fillable sleeve re-selects rather than relabels
    assert [r.sleeve for r in results] == ["fillable", "ceiling", "pooled"]
    assert by_sleeve["fillable"].top1_mean == pytest.approx(0.2)
    assert by_sleeve["pooled"].top1_mean == pytest.approx(6.0)
    assert by_sleeve["fillable"].n_days == 40


def test_summarize_buyability_sleeves_reports_measurement_coverage() -> None:
    import numpy as np
    import pandas as pd

    from src.ml.buyability import evaluate_buyability_sleeves, summarize_buyability_sleeves

    # Given: 40 days, half of them with no auction measurement at all
    dates = pd.bdate_range("2026-01-05", periods=40)
    rows = []
    for i, d in enumerate(dates):
        measured = i % 2 == 0
        for j, (pred, ret, close, high) in enumerate(
            [(0.9, 1.0, 13000.0, 13000.0), (0.5, 0.2, 11000.0, 11200.0)]
        ):
            rows.append(
                {
                    "trade_date": d, "stock_code": f"{j}{i:05d}", "pred": pred,
                    "net_return": ret, "close_price": close,
                    "prev_close_price": 10000.0, "high_price": high,
                    "auction_value_100m": 40.0 if measured else np.nan,
                    "auction_vol_share": 0.011 if measured else np.nan,
                    "auction_bars_found": measured,
                }
            )
    oof = pd.DataFrame(rows)

    # When
    summary = summarize_buyability_sleeves(evaluate_buyability_sleeves(oof, target_notional_100m=1.0))

    # Then
    assert summary["n_rows"] == 80
    assert summary["n_measured"] == 40
    assert summary["measured_share"] == 0.5
    assert set(summary["sleeves"]) == {"fillable", "ceiling", "pooled"}


def test_buyability_cover_attach_canonical_and_validations(tmp_path) -> None:
    import numpy as np
    import pandas as pd
    import pytest

    from src.ml.buyability import (
        BuyabilitySleeveResult,
        attach_entry_auction_liquidity,
        apply_buyability_gate,
        classify_ceiling_entry,
        evaluate_buyability_sleeves,
        summarize_buyability_sleeves,
    )

    # classify missing column raises
    with pytest.raises(ValueError, match="close_price"):
        classify_ceiling_entry(pd.DataFrame({"close_price": [1.0]}))
    # attach missing date/code raises
    with pytest.raises(ValueError, match="date_col"):
        attach_entry_auction_liquidity(pd.DataFrame({"x": [1]}))
    # gate bad inputs raise
    with pytest.raises(ValueError, match="target_notional"):
        apply_buyability_gate(pd.DataFrame({"trade_date": [], "stock_code": []}), target_notional_100m=0.0)
    with pytest.raises(ValueError, match="participation_cap"):
        apply_buyability_gate(
            pd.DataFrame({"trade_date": [], "stock_code": []}),
            target_notional_100m=1.0,
            participation_cap=2.0,
        )
    # evaluate validations raise
    oof = pd.DataFrame({"trade_date": [], "stock_code": [], "pred": [], "net_return": []})
    with pytest.raises(ValueError, match="required columns"):
        evaluate_buyability_sleeves(oof.drop(columns=["pred"]), target_notional_100m=1.0)
    with pytest.raises(ValueError, match="target_notional"):
        evaluate_buyability_sleeves(oof, target_notional_100m=-1.0)
    with pytest.raises(ValueError, match="participation_cap"):
        evaluate_buyability_sleeves(oof, target_notional_100m=1.0, participation_cap=0.0)
    with pytest.raises(ValueError, match="alpha"):
        evaluate_buyability_sleeves(oof, target_notional_100m=1.0, alpha=0.9)
    with pytest.raises(ValueError, match="non-empty"):
        summarize_buyability_sleeves(())

    # attach with default root (intraday_root None) degrades to unmeasured
    df = pd.DataFrame(
        {"trade_date": pd.to_datetime(["2019-05-02"]), "stock_code": ["005930"]}
    )
    out = attach_entry_auction_liquidity(df)
    assert out["auction_bars_found"].tolist() == [False]
    assert out["auction_value_100m"].isna().all()

    # canonical partition read path: one date, one symbol, auction + day bars
    snap = "2026-03-02"
    month = snap[:7]
    part_dir = tmp_path / "intraday" / "1m" / "regular" / month
    part_dir.mkdir(parents=True, exist_ok=True)
    bars = pd.DataFrame(
        {
            "snapshot_date": [snap] * 4,
            "symbol": ["005930"] * 4,
            "ts_hms": [145900, 151500, 152100, 152500],
            "open": [10000, 10000, 10000, 10000],
            "high": [10000, 10000, 10000, 10000],
            "low": [10000, 10000, 10000, 10000],
            "close": [10000, 10000, 10000, 10000],
            "volume": [1000, 1000, 500, 500],
            "value_krw": [10_000_000, 10_000_000, 5_000_000, 15_000_000],
            "has_trade": [True] * 4,
            "vendor": ["kis"] * 4,
        }
    )
    bars.to_parquet(part_dir / f"{snap}.parquet")
    panel = pd.DataFrame(
        {"trade_date": pd.to_datetime([snap, snap]), "stock_code": ["005930", "000660"]}
    )
    got = attach_entry_auction_liquidity(panel, intraday_root=tmp_path)
    assert got["auction_bars_found"].tolist() == [True, False]
    assert got.loc[0, "auction_value_100m"] == pytest.approx(0.2)
    assert got.loc[0, "auction_vol_share"] == pytest.approx(1000 / 3000)
    assert np.isnan(got.loc[1, "auction_value_100m"])

    # gate without pre-attached auction cols exercises attach fallback
    raw = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2019-05-02"]),
            "stock_code": ["005930"],
            "close_price": [11000.0],
            "prev_close_price": [10000.0],
            "high_price": [11200.0],
        }
    )
    gated, prov = apply_buyability_gate(raw, target_notional_100m=1.0)
    assert len(gated) == 1 and prov["n_rows"] == 1
    with pytest.raises(ValueError, match="trade_date"):
        apply_buyability_gate(pd.DataFrame({"close_price": [1.0]}), target_notional_100m=1.0)

    # empty fillable/ceiling sleeves + summarize fallback (cache miss)
    dates = pd.bdate_range("2026-02-02", periods=35)
    rows = [
        {
            "trade_date": d,
            "stock_code": f"{i:06d}",
            "pred": 1.0,
            "net_return": 0.5,
            "close_price": 11000.0,
            "prev_close_price": 10000.0,
            "high_price": 11200.0,
            "auction_value_100m": 0.0,
            "auction_vol_share": 0.0,
            "auction_bars_found": True,
        }
        for i, d in enumerate(dates)
    ]
    res = evaluate_buyability_sleeves(pd.DataFrame(rows), target_notional_100m=1.0)
    by = {r.sleeve: r for r in res}
    assert by["fillable"].n_days == 0
    assert by["ceiling"].n_days == 0
    manual = (
        BuyabilitySleeveResult("fillable", 2, 4, 0.1, 0.01, 0.5, 1.0, 0.0),
        BuyabilitySleeveResult("ceiling", 1, 2, 0.2, 0.0, 0.0, 2.0, 0.0),
        BuyabilitySleeveResult("pooled", 2, 6, 0.15, 0.02, 0.3, 1.5, 0.0),
    )
    s = summarize_buyability_sleeves(manual)
    assert s["n_rows"] == 6 and s["n_measured"] == 6


def test_buyability_cover_attach_raw_ls_partition(tmp_path) -> None:
    import pandas as pd

    from src.ml.buyability import attach_entry_auction_liquidity

    snap = "2026-03-03"
    part_dir = tmp_path / "intraday" / "1m" / "regular" / snap[:7]
    part_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(
        {
            "symbol": ["005930"] * 3,
            "time": [145900, 152100, 152500],
            "open": [10000, 10000, 10000],
            "high": [10000, 10000, 10000],
            "low": [10000, 10000, 10000],
            "close": [10000, 10000, 10000],
            "jdiff_vol": [1000, 500, 500],
            "value": [10.0, 5.0, 15.0],
        }
    )
    raw.to_parquet(part_dir / f"{snap}.parquet")
    panel = pd.DataFrame({"trade_date": pd.to_datetime([snap]), "stock_code": ["005930"]})
    got = attach_entry_auction_liquidity(panel, intraday_root=tmp_path)
    assert got["auction_bars_found"].tolist() == [True]
    assert got.loc[0, "auction_value_100m"] > 0


def test_buyability_cover_zero_volume_nosymbol_and_rank_edge(tmp_path) -> None:
    import numpy as np
    import pandas as pd

    from src.ml.buyability import attach_entry_auction_liquidity, evaluate_buyability_sleeves

    # attach over a frame that already carries auction cols exercises overwrite (line 99)
    base = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-03-02"]),
            "stock_code": ["005930"],
            "auction_value_100m": [1.0],
            "auction_vol_share": [0.01],
            "auction_bars_found": [True],
        }
    )
    again = attach_entry_auction_liquidity(base, intraday_root=tmp_path)
    assert again["auction_bars_found"].tolist() == [False]

    # zero day-volume symbol stays unmeasured (canonical day_vol<=0 branch)
    snap = "2026-03-04"
    part_dir = tmp_path / "intraday" / "1m" / "regular" / snap[:7]
    part_dir.mkdir(parents=True, exist_ok=True)
    bars = pd.DataFrame(
        {
            "snapshot_date": [snap] * 3,
            "symbol": ["000660"] * 3,
            "ts_hms": [145900, 152100, 152500],
            "open": [10000] * 3,
            "high": [10000] * 3,
            "low": [10000] * 3,
            "close": [10000] * 3,
            "volume": [0, 0, 0],
            "value_krw": [0, 0, 0],
            "has_trade": [False] * 3,
            "vendor": ["kis"] * 3,
        }
    )
    bars.to_parquet(part_dir / f"{snap}.parquet")
    panel = pd.DataFrame({"trade_date": pd.to_datetime([snap]), "stock_code": ["000660"]})
    got = attach_entry_auction_liquidity(panel, intraday_root=tmp_path)
    assert got["auction_bars_found"].tolist() == [False]
    assert np.isnan(got.loc[0, "auction_value_100m"])

    # raw vendor file without a symbol column exercises the no-symbol branch
    snap2 = "2026-03-05"
    part_dir2 = tmp_path / "intraday" / "1m" / "regular" / snap2[:7]
    part_dir2.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(
        {
            "time": [145900, 152100],
            "open": [10000, 10000],
            "high": [10000, 10000],
            "low": [10000, 10000],
            "close": [10000, 10000],
            "jdiff_vol": [100, 100],
            "value": [1.0, 1.0],
        }
    )
    raw.to_parquet(part_dir2 / f"{snap2}.parquet")
    panel2 = pd.DataFrame({"trade_date": pd.to_datetime([snap2]), "stock_code": ["005930"]})
    got2 = attach_entry_auction_liquidity(panel2, intraday_root=tmp_path)
    assert got2["auction_bars_found"].tolist() == [False]

    # rank_ic with fewer than two finite scores
    oof = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2026-01-05", "2026-01-05"]),
            "stock_code": ["000001", "000002"],
            "pred": [float("nan"), 0.5],
            "net_return": [0.2, 0.3],
            "close_price": [11000.0, 11000.0],
            "prev_close_price": [10000.0, 10000.0],
            "high_price": [11200.0, 11200.0],
            "auction_value_100m": [40.0, 40.0],
            "auction_vol_share": [0.01, 0.01],
            "auction_bars_found": [True, True],
        }
    )
    res = evaluate_buyability_sleeves(oof, target_notional_100m=1.0)
    assert len(res) == 3


def test_buyability_cover_raw_zero_day_volume(tmp_path) -> None:
    import pandas as pd

    from src.ml.buyability import attach_entry_auction_liquidity

    snap = "2026-03-06"
    part_dir = tmp_path / "intraday" / "1m" / "regular" / snap[:7]
    part_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(
        {
            "symbol": ["000660"] * 2,
            "time": [145900, 152100],
            "open": [10000, 10000],
            "high": [10000, 10000],
            "low": [10000, 10000],
            "close": [10000, 10000],
            "jdiff_vol": [0, 0],
            "value": [0.0, 0.0],
        }
    )
    raw.to_parquet(part_dir / f"{snap}.parquet")
    panel = pd.DataFrame({"trade_date": pd.to_datetime([snap]), "stock_code": ["000660"]})
    got = attach_entry_auction_liquidity(panel, intraday_root=tmp_path)
    assert got["auction_bars_found"].tolist() == [False]
