from __future__ import annotations

import os
import stat
from pathlib import Path

import pandas as pd
import pytest

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


def _with_umask(mask: int):
    import contextlib
    import os

    @contextlib.contextmanager
    def _guard():
        old = os.umask(mask)
        try:
            yield
        finally:
            os.umask(old)

    return _guard()


def test_atomic_bytes_roundtrip_creates_parents(tmp_path: Path) -> None:
    from src.data.io_utils import atomic_write_bytes

    target = tmp_path / "nested" / "deep" / "out.bin"

    atomic_write_bytes(target, b"x", mode=None)

    assert target.read_bytes() == b"x"
    assert target.parent.is_dir()


def test_atomic_explicit_mode_ignores_umask(tmp_path: Path) -> None:
    import stat

    from src.data.io_utils import atomic_write_bytes

    with _with_umask(0o077):
        atomic_write_bytes(tmp_path / "a.bin", b"x", mode=0o644)
    assert stat.S_IMODE((tmp_path / "a.bin").stat().st_mode) == 0o644

    with _with_umask(0o022):
        atomic_write_bytes(tmp_path / "b.bin", b"x", mode=0o600)
    assert stat.S_IMODE((tmp_path / "b.bin").stat().st_mode) == 0o600


def test_atomic_none_mode_follows_umask(tmp_path: Path) -> None:
    import stat

    from src.data.io_utils import atomic_write_text

    with _with_umask(0o027):
        atomic_write_text(tmp_path / "out.txt", "x", mode=None)
        (tmp_path / "reference.txt").write_text("x", encoding="utf-8")
    assert stat.S_IMODE((tmp_path / "out.txt").stat().st_mode) == 0o640
    assert stat.S_IMODE((tmp_path / "out.txt").stat().st_mode) == stat.S_IMODE(
        (tmp_path / "reference.txt").stat().st_mode
    )


def test_atomic_overwrite_does_not_inherit_old_mode(tmp_path: Path) -> None:
    import os
    import stat

    from src.data.io_utils import atomic_write_bytes

    target = tmp_path / "out.bin"
    target.write_bytes(b"old")
    os.chmod(target, 0o600)

    atomic_write_bytes(target, b"new", mode=0o644)

    assert target.read_bytes() == b"new"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_atomic_replace_failure_keeps_destination_and_removes_temp(tmp_path: Path, monkeypatch) -> None:
    import os

    import pytest

    from src.data.io_utils import atomic_write_bytes

    target = tmp_path / "out.bin"
    target.write_bytes(b"old")

    def _boom(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(OSError, match="disk gone"):
        atomic_write_bytes(target, b"new", mode=None)

    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob("stage-*.tmp")) == []


def test_atomic_base_exception_cleanup(tmp_path: Path, monkeypatch) -> None:
    import os

    import pytest

    from src.data.io_utils import atomic_write_bytes

    target = tmp_path / "out.bin"

    def _boom(src, dst):
        raise KeyboardInterrupt()

    monkeypatch.setattr(os, "replace", _boom)
    with pytest.raises(KeyboardInterrupt):
        atomic_write_bytes(target, b"new", mode=None)

    assert list(tmp_path.glob("stage-*.tmp")) == []


def test_atomic_context_body_failure_unpublished(tmp_path: Path) -> None:
    import pytest

    from src.data.io_utils import atomic_output_path

    target = tmp_path / "out.bin"

    def _write_then_fail() -> None:
        with atomic_output_path(target, mode=None) as tmp:
            tmp.write_bytes(b"partial")
            raise ValueError("no publish")

    with pytest.raises(ValueError, match="no publish"):
        _write_then_fail()

    assert not target.exists()
    assert list(tmp_path.glob("stage-*.tmp")) == []


def test_atomic_context_clean_exit_publishes(tmp_path: Path) -> None:
    from src.data.io_utils import atomic_output_path

    target = tmp_path / "out.bin"

    with atomic_output_path(target, mode=None) as tmp:
        yielded = tmp
        tmp.write_bytes(b"abc")

    assert target.read_bytes() == b"abc"
    assert not yielded.exists()


def test_atomic_temp_name_contract(tmp_path: Path, monkeypatch) -> None:
    import os

    from src.data.io_utils import atomic_write_text
    from src.tools.capture_offsite import _is_inflight

    target = tmp_path / "report.json"
    sources: list[str] = []
    real_replace = os.replace

    def _spy(src, dst):
        sources.append(os.path.basename(src))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _spy)
    atomic_write_text(target, "{}", mode=None)
    atomic_write_text(target, "{}", mode=None)

    from src.data.io_utils import ATOMIC_TMP_PREFIX, ATOMIC_TMP_SUFFIX

    assert len(sources) == 2
    assert sources[0] != sources[1]
    for name in sources:
        assert name.startswith(f"{ATOMIC_TMP_PREFIX}{target.name}.")
        assert name.endswith(ATOMIC_TMP_SUFFIX)
        assert _is_inflight(name) is True


def test_atomic_long_basename_truncated(tmp_path: Path, monkeypatch) -> None:
    import os

    from src.data.io_utils import atomic_write_text

    target = tmp_path / ("n" * 250)
    sources: list[str] = []
    real_replace = os.replace

    def _spy(src, dst):
        sources.append(os.path.basename(src))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _spy)
    atomic_write_text(target, "{}", mode=None)

    assert target.read_text(encoding="utf-8") == "{}"
    assert len(sources) == 1
    assert len(sources[0].encode("utf-8")) <= 255


def test_atomic_single_replace_per_publish(tmp_path: Path, monkeypatch) -> None:
    import os

    from src.data.io_utils import atomic_write_text

    target = tmp_path / "out.txt"
    calls = {"n": 0}
    real_replace = os.replace

    def _counting(src, dst):
        calls["n"] += 1
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _counting)
    atomic_write_text(target, "x", mode=None)

    assert calls["n"] == 1


def test_atomic_durable_fsyncs_file_and_directory(tmp_path: Path, monkeypatch) -> None:
    import os
    import stat

    from src.data.io_utils import atomic_write_text

    target = tmp_path / "out.txt"
    synced: list[tuple[int, bool]] = []
    real_fsync = os.fsync

    def _spy(fd):
        synced.append((fd, stat.S_ISDIR(os.fstat(fd).st_mode)))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _spy)
    atomic_write_text(target, "x", mode=None, durable=True)

    assert len(synced) >= 2
    assert any(is_dir for _, is_dir in synced)

    synced.clear()
    atomic_write_text(tmp_path / "plain.txt", "x", mode=None, durable=False)
    assert synced == []


def test_atomic_text_encoding_exact(tmp_path: Path) -> None:
    from src.data.io_utils import atomic_write_text

    target = tmp_path / "out.txt"

    atomic_write_text(target, "한글 ✓", mode=None, encoding="utf-8")

    assert target.read_bytes() == "한글 ✓".encode()


def test_atomic_invalid_mode_rejected(tmp_path: Path) -> None:
    import pytest

    from src.data.io_utils import atomic_write_text

    target = tmp_path / "sub" / "out.txt"

    with pytest.raises(ValueError, match="mode"):
        atomic_write_text(target, "x", mode=0o1777)

    assert not target.exists()


def test_atomic_empty_basename_rejected_without_effect(tmp_path: Path) -> None:
    import pytest

    from src.data.io_utils import atomic_write_bytes

    before = {p.name for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="must name a file"):
        atomic_write_bytes("", b"x", mode=None)

    assert {p.name for p in tmp_path.iterdir()} == before


def test_atomic_write_failure_removes_temp(tmp_path: Path, monkeypatch) -> None:
    from pathlib import Path as _Path

    import pytest

    from src.data.io_utils import atomic_write_bytes

    target = tmp_path / "out.bin"

    def _boom(self, data):
        raise OSError("disk gone")

    monkeypatch.setattr(_Path, "write_bytes", _boom)
    with pytest.raises(OSError, match="disk gone"):
        atomic_write_bytes(target, b"new", mode=None)

    assert not target.exists()
    assert list(tmp_path.glob("stage-*.tmp")) == []


def test_atomic_parquet_mode_under_restrictive_umask(tmp_path: Path) -> None:
    import stat

    import pandas as pd

    from src.data.io_utils import atomic_write_parquet

    target = tmp_path / "out.parquet"
    with _with_umask(0o077):
        atomic_write_parquet(pd.DataFrame({"a": [1, 2, 3]}), target)

    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert list(tmp_path.glob("stage-*.tmp")) == []


def test_atomic_parquet_failure_leaves_no_temp(tmp_path: Path, monkeypatch) -> None:
    import pandas as pd
    import pytest

    from src.data.io_utils import atomic_write_parquet

    target = tmp_path / "out.parquet"

    def _partial(self, path, **kwargs):
        from pathlib import Path as _Path

        _Path(path).write_bytes(b"partial")
        raise RuntimeError("encode boom")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _partial)
    with pytest.raises(RuntimeError, match="encode boom"):
        atomic_write_parquet(pd.DataFrame({"a": [1]}), target)

    assert not target.exists()
    assert list(tmp_path.glob("stage-*.tmp")) == []


def _hold_sidecar(target: Path):
    import contextlib
    import fcntl

    from src.utils.file_lock import sidecar_lock_path

    @contextlib.contextmanager
    def _guard():
        lock_path = sidecar_lock_path(target)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        holder = open(lock_path, "w")  # noqa: PTH123, SIM115 - lock held across the block
        try:
            fcntl.flock(holder, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(holder, fcntl.LOCK_UN)
            holder.close()

    return _guard()


def test_store_lock_timeout_is_typed_and_logged(tmp_path: Path, caplog) -> None:
    import logging

    import pytest

    from src.data import io_utils

    target = tmp_path / "store.parquet"
    target.write_bytes(b"v1")

    with (
        _hold_sidecar(target),
        caplog.at_level(logging.ERROR, logger="src.data.io_utils"),
        pytest.raises(io_utils.StoreLockTimeoutError, match="store\\.parquet") as exc_info,
        io_utils.store_write_lock(target, purpose="t", timeout_seconds=0.0),
    ):
        pass  # pragma: no cover - acquisition never succeeds

    assert isinstance(exc_info.value, TimeoutError)
    assert str(target) in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    timeout_records = [r for r in caplog.records if "stage=store_lock status=TIMEOUT" in r.getMessage()]
    assert len(timeout_records) == 1
    assert timeout_records[0].levelno == logging.ERROR


def test_store_lock_body_timeout_error_not_retyped(tmp_path: Path, caplog) -> None:
    import logging

    import pytest

    from src.data import io_utils
    from src.utils.file_lock import sidecar_lock_path

    target = tmp_path / "store.parquet"
    target.write_bytes(b"v1")
    body_exc = TimeoutError("body")

    with (
        caplog.at_level(logging.ERROR, logger="src.data.io_utils"),
        pytest.raises(TimeoutError) as exc_info,
        io_utils.store_write_lock(target, purpose="t"),
    ):
        raise body_exc

    assert exc_info.value is body_exc
    assert not isinstance(exc_info.value, io_utils.StoreLockTimeoutError)
    assert not any("stage=store_lock status=TIMEOUT" in r.getMessage() for r in caplog.records)
    assert not sidecar_lock_path(target).exists()


def test_store_lock_default_timeout_resolved_at_call(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.data import io_utils

    monkeypatch.setattr(io_utils, "STORE_LOCK_TIMEOUT_SECONDS", 0.0)
    target = tmp_path / "store.parquet"
    target.write_bytes(b"v1")

    with (
        _hold_sidecar(target),
        pytest.raises(io_utils.StoreLockTimeoutError),
        io_utils.store_write_lock(target, purpose="t"),
    ):
        pass  # pragma: no cover - acquisition never succeeds


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permission checks")
def test_store_lock_unopenable_sidecar_raises_typed_timeout(tmp_path: Path) -> None:
    from src.data.io_utils import StoreLockTimeoutError, store_write_lock

    target = tmp_path / "archive.parquet"
    sidecar = tmp_path / "archive.parquet.lock"
    sidecar.touch()
    sidecar.chmod(0o000)
    try:
        with (
            pytest.raises(StoreLockTimeoutError),
            store_write_lock(target, purpose="condition-archive", timeout_seconds=0.05),
        ):
            pytest.fail("body must not run without the lock")
    finally:
        sidecar.chmod(0o644)
