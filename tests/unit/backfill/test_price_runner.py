from __future__ import annotations


def test_price_runner_to_parquet_writes_via_shared_codec(tmp_path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from src.backfill.price import runner

    cols = {
        "date": pd.to_datetime(["2026-01-02"]), "symbol": ["005930"],
        "open": [70000.0], "high": [70500.0], "low": [69800.0], "close": [70200.0], "prev_close": [69200.0],
        "market_cap_100m": [1.0], "trade_value_100m": [1.0], "daily_change_pct": [0.01],
        "market": ["KOSPI"], "volume": [1000],
        "foreign_netbuy": [1.0], "inst_netbuy": [1.0], "program_netbuy": [1.0],
        "kospi_pct": [0.01], "kosdaq_pct": [0.01], "v_kospi": [18.5], "v_kosdaq": [18.5],
    }
    path = tmp_path / "price_history.parquet"

    runner._to_parquet(pd.DataFrame(cols), path)

    meta = pq.ParquetFile(path).metadata
    assert meta.row_group(0).column(0).compression.upper() == "ZSTD"
    first = pd.read_parquet(path)
    assert len(first) == 1

    cols2 = dict(cols)
    cols2["date"] = pd.to_datetime(["2026-01-03"])
    runner._to_parquet(pd.DataFrame(cols2), path)

    merged = pd.read_parquet(path)
    assert len(merged) == 2
