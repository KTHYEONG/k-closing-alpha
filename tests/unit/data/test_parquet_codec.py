from __future__ import annotations


def test_downcast_price_history_frame_preserves_values_across_dtype_change() -> None:
    import pandas as pd

    from src.data.parquet_codec import downcast_price_history_frame

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-01"]),
            "symbol": ["005930", "000660", "005930"],
            "open": [70000.0, 130000.0, 69000.0],
            "high": [70500.0, 131000.0, 69500.0],
            "low": [69800.0, 129500.0, 68900.0],
            "close": [70200.0, 130500.0, 69200.0],
            "prev_close": [69200.0, float("nan"), 68000.0],
            "market_cap_100m": [4190000.5, 950000.25, 4180000.0],
            "trade_value_100m": [12345.678, 5432.1, 11000.0],
            "daily_change_pct": [0.01445086705202312, float("nan"), 0.0176],
            "market": ["KOSPI", "KOSPI", "KOSPI"],
            "volume": [15000000, 8000000, 14000000],
            "foreign_netbuy": [1000.0, -500.0, 900.0],
            "inst_netbuy": [200.0, 100.0, 150.0],
            "program_netbuy": [50000000.0, -20000000.0, 40000000.0],
            "kospi_pct": [0.005, 0.005, 0.004],
            "kosdaq_pct": [0.003, 0.003, 0.002],
            "v_kospi": [18.5, 18.5, 19.0],
            "v_kosdaq": [float("nan"), float("nan"), float("nan")],
        }
    )

    out = downcast_price_history_frame(df)

    assert str(out["open"].dtype) == "Int32"
    assert str(out["prev_close"].dtype) == "Int32"
    assert pd.isna(out.loc[out["symbol"] == "000660", "prev_close"].iloc[0])
    assert out["open"].dropna().astype("int64").tolist() == [69000, 70000, 130000] or set(out["open"].dropna().tolist()) == {69000, 70000, 130000}
    assert str(out["symbol"].dtype) == "category"
    assert str(out["volume"].dtype) in ("int64", "Int64")
    assert str(out["market_cap_100m"].dtype) == "float32"
    for orig, new in zip(df["market_cap_100m"].sort_values(), out["market_cap_100m"].astype("float64").sort_values(), strict=True):  # noqa: B905 - skeleton fidelity
        assert abs(orig - new) / max(abs(orig), 1e-9) < 1e-4
    assert out["symbol"].astype(str).is_monotonic_increasing


def test_downcast_price_history_frame_keeps_return_columns_float64() -> None:
    import pandas as pd

    from src.data.parquet_codec import downcast_price_history_frame

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"]),
            "symbol": ["005930"],
            "open": [70000.0], "high": [70500.0], "low": [69800.0], "close": [70200.0], "prev_close": [69200.0],
            "market_cap_100m": [4190000.5], "trade_value_100m": [12345.678],
            "daily_change_pct": [0.014450867052023121],
            "market": ["KOSPI"], "volume": [15000000],
            "foreign_netbuy": [1000.0], "inst_netbuy": [200.0], "program_netbuy": [50000000.0],
            "kospi_pct": [0.0054321987], "kosdaq_pct": [0.0033219876],
            "v_kospi": [18.5], "v_kosdaq": [float("nan")],
        }
    )

    out = downcast_price_history_frame(df)

    assert str(out["daily_change_pct"].dtype) == "float64"
    assert str(out["kospi_pct"].dtype) == "float64"
    assert str(out["kosdaq_pct"].dtype) == "float64"
    assert out["daily_change_pct"].iloc[0] == df["daily_change_pct"].iloc[0]
    assert out["kospi_pct"].iloc[0] == df["kospi_pct"].iloc[0]


def test_downcast_price_history_frame_rejects_int32_overflow() -> None:
    import pandas as pd
    import pytest

    from src.data.parquet_codec import downcast_price_history_frame

    base = {
        "date": pd.to_datetime(["2026-01-02"]), "symbol": ["005930"],
        "open": [70000.0], "high": [70500.0], "low": [69800.0], "close": [70200.0], "prev_close": [69200.0],
        "market_cap_100m": [1.0], "trade_value_100m": [1.0], "daily_change_pct": [0.01],
        "market": ["KOSPI"], "volume": [2_498_359_659],
        "foreign_netbuy": [1.0], "inst_netbuy": [1.0], "program_netbuy": [1.0],
        "kospi_pct": [0.01], "kosdaq_pct": [0.01], "v_kospi": [18.5], "v_kosdaq": [18.5],
    }

    ok = downcast_price_history_frame(pd.DataFrame(base))
    assert str(ok["volume"].dtype) in ("int64", "Int64")
    assert int(ok["volume"].iloc[0]) == 2_498_359_659

    bad = dict(base)
    bad["close"] = [3_000_000_000.0]
    with pytest.raises(ValueError, match="int32"):
        downcast_price_history_frame(pd.DataFrame(bad))


def test_downcast_altdata_panel_frame_leaves_integer_columns_untouched() -> None:
    import pandas as pd

    from src.data.parquet_codec import downcast_altdata_panel_frame

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-01"]),
            "symbol": ["000660", "005930"],
            "program_net_value": [-193758630, 42251260],
        }
    )

    out = downcast_altdata_panel_frame(df)

    assert str(out["symbol"].dtype) == "category"
    assert out["program_net_value"].dtype == df["program_net_value"].dtype
    assert out["symbol"].tolist() == ["000660", "005930"]
    assert out["program_net_value"].tolist() == [-193758630, 42251260]


def test_downcast_price_history_frame_is_fail_open_on_missing_columns() -> None:
    import pandas as pd

    from src.data.parquet_codec import downcast_price_history_frame

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"]),
            "symbol": ["005930"],
            "open": [70000.0],
            "volume": [1000.0],
            "market_cap_100m": [1.0],
        }
    )

    out = downcast_price_history_frame(df)

    assert str(out["open"].dtype) == "Int32"
    assert str(out["market_cap_100m"].dtype) == "float32"
    assert "prev_close" not in out.columns
    assert len(out) == 1


def test_downcast_price_history_frame_rejects_int64_volume_overflow() -> None:
    import pandas as pd
    import pytest

    from src.data.parquet_codec import downcast_price_history_frame

    df = pd.DataFrame({"symbol": ["005930"], "volume": [float("inf")]})
    with pytest.raises(ValueError, match="int64"):
        downcast_price_history_frame(df)


def test_downcast_price_history_frame_keeps_nullable_int64_volume_with_nan() -> None:
    import pandas as pd

    from src.data.parquet_codec import downcast_price_history_frame

    df = pd.DataFrame({"symbol": ["005930", "000660"], "volume": [1000.0, float("nan")]})
    out = downcast_price_history_frame(df)
    assert str(out["volume"].dtype) == "Int64"
    assert int(out.loc[out["symbol"] == "005930", "volume"].iloc[0]) == 1000
    assert pd.isna(out.loc[out["symbol"] == "000660", "volume"].iloc[0])


def test_downcast_price_history_frame_sorts_by_symbol_then_date() -> None:
    """symbol을 1차 정렬키로 사용해야 zstd 컬럼 압축률이 극대화된다 (date 우선 정렬 대비 실측 열위)."""
    import pandas as pd

    from src.data.parquet_codec import downcast_price_history_frame

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-02", "2026-01-01"]),
            "symbol": ["000660", "005930", "005930", "000660"],
            "volume": [1, 2, 3, 4],
        }
    )

    out = downcast_price_history_frame(df)

    # date 우선 정렬이었다면 [000660(01-01), 005930(01-01), 000660(01-02), 005930(01-02)] 순서가 된다.
    assert out["symbol"].astype(str).tolist() == ["000660", "000660", "005930", "005930"]


def test_downcast_altdata_panel_frame_sorts_by_symbol_then_date() -> None:
    import pandas as pd

    from src.data.parquet_codec import downcast_altdata_panel_frame

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-02", "2026-01-01"]),
            "symbol": ["000660", "005930", "005930", "000660"],
            "value": [1, 2, 3, 4],
        }
    )

    out = downcast_altdata_panel_frame(df)

    assert out["symbol"].astype(str).tolist() == ["000660", "000660", "005930", "005930"]
