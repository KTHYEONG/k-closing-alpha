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


def test_altdata_covered_dates_reads_only_date_column(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.backfill.altdata import runner

    path = tmp_path / "panel.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "symbol": ["005930"], "value": [1]}).to_parquet(path, index=False)

    seen_kwargs = {}
    real_read_parquet = pd.read_parquet

    def spy_read_parquet(p, *args, **kwargs):
        seen_kwargs.update(kwargs)
        return real_read_parquet(p, *args, **kwargs)

    monkeypatch.setattr(runner.pd, "read_parquet", spy_read_parquet)

    result = runner._covered_dates(path)

    assert seen_kwargs.get("columns") == ["date"]
    assert len(result) == 1


def test_altdata_atomic_write_parquet_delegates_to_shared_codec(tmp_path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from src.backfill.altdata import runner

    path = tmp_path / "panel.parquet"
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-01"]), "symbol": ["000660", "005930"], "value": [1, 2]})

    runner._atomic_write_parquet(df, path)

    meta = pq.ParquetFile(path).metadata
    assert meta.row_group(0).column(0).compression.upper() == "ZSTD"
    stored = pd.read_parquet(path)
    assert stored["symbol"].tolist() == ["000660", "005930"]


def test_altdata_package_lazy_reexports() -> None:
    import pytest

    import src.backfill.altdata as pkg

    assert pkg.AltDataFetchConfig.__name__ == "AltDataFetchConfig"
    assert callable(pkg.run_altdata_backfill)
    with pytest.raises(AttributeError, match="no attribute"):
        _ = pkg.no_such_symbol
