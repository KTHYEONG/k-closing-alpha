from __future__ import annotations

import stat
from pathlib import Path

import pandas as pd

from src.data.io_utils import atomic_write_parquet


def test_atomic_write_parquet_is_group_and_other_readable(tmp_path: Path) -> None:
    # tempfile.NamedTemporaryFile은 umask와 무관하게 0600으로 생성되고 os.replace가 그 권한을
    # 그대로 승계한다 -- 다른 유저(백업 계정 등)가 결과 파일을 읽을 수 있어야 한다.
    target = tmp_path / "out.parquet"
    atomic_write_parquet(pd.DataFrame({"a": [1, 2, 3]}), target)

    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode & 0o044 == 0o044


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


def test_read_existing_missing_returns_empty(tmp_path: Path) -> None:
    import pandas as pd

    from src.data.io_utils import read_existing_parquet

    target = tmp_path / "absent.parquet"

    result = read_existing_parquet(target)

    assert isinstance(result, pd.DataFrame) and result.empty
    assert not target.exists()


def test_read_existing_corrupt_raises_typed_error(tmp_path: Path) -> None:
    import pytest

    from src.data.io_utils import ExistingStoreUnreadableError, read_existing_parquet

    target = tmp_path / "history.parquet"
    target.write_bytes(b"not a valid parquet file")

    with pytest.raises(ExistingStoreUnreadableError, match="history\\.parquet") as exc_info:
        read_existing_parquet(target)

    assert exc_info.value.__cause__ is not None
    assert target.read_bytes() == b"not a valid parquet file"


def test_read_existing_zero_row_file_is_valid(tmp_path: Path) -> None:
    import pandas as pd

    from src.data.io_utils import read_existing_parquet

    target = tmp_path / "empty.parquet"
    pd.DataFrame({"a": pd.Series(dtype="int64")}).to_parquet(target, index=False)

    result = read_existing_parquet(target)

    assert isinstance(result, pd.DataFrame) and result.empty


def test_read_existing_projects_columns(tmp_path: Path) -> None:
    import pandas as pd

    from src.data.io_utils import read_existing_parquet

    target = tmp_path / "panel.parquet"
    pd.DataFrame({"date": ["2026-09-29"], "symbol": ["005930"]}).to_parquet(target, index=False)

    result = read_existing_parquet(target, columns=["date"])

    assert list(result.columns) == ["date"]
