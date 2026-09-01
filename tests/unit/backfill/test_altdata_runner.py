import pandas as pd

from src.backfill.altdata.runner import _atomic_write_parquet, _incremental_merge


def test_incremental_merge_and_atomic_write(tmp_path) -> None:
    path = tmp_path / "shorting.parquet"
    first = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "symbol": ["005930"], "short_volume": [10.0]})
    _atomic_write_parquet(first, path)
    assert path.exists() and not (tmp_path / "shorting.parquet.tmp").exists()
    nxt = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "symbol": ["005930", "005930"],
        "short_volume": [99.0, 20.0],
    })
    merged = _incremental_merge(path, nxt, ("date", "symbol"))
    assert len(merged) == 2
    assert merged.sort_values("date").iloc[0]["short_volume"] == 99.0
