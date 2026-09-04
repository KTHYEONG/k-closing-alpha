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
