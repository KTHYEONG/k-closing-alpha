from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.io_utils import atomic_write_parquet


def test_atomic_write_parquet_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    target = tmp_path / "nested" / "out.parquet"

    atomic_write_parquet(df, target)

    assert target.exists()
    loaded = pd.read_parquet(target)
    pd.testing.assert_frame_equal(loaded.reset_index(drop=True), df.reset_index(drop=True))


def test_atomic_write_parquet_defaults_to_zstd_compression(tmp_path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from src.data.io_utils import atomic_write_parquet

    target = tmp_path / "out.parquet"
    atomic_write_parquet(pd.DataFrame({"a": [1, 2, 3]}), target)

    meta = pq.ParquetFile(target).metadata
    codec = meta.row_group(0).column(0).compression
    assert codec.upper() == "ZSTD"


def test_atomic_write_parquet_snappy_override_still_works(tmp_path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from src.data.io_utils import atomic_write_parquet

    target = tmp_path / "out.parquet"
    atomic_write_parquet(pd.DataFrame({"a": [1, 2, 3]}), target, compression="snappy")

    meta = pq.ParquetFile(target).metadata
    assert meta.row_group(0).column(0).compression.upper() == "SNAPPY"
