"""Invariant guards for restoring evicted history partitions."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.data.partition_eviction import EvictionFamily, evict_verified_partitions, read_tombstone, tombstone_path
from src.tools.partition_restore import restore_partitions

BASE = "gdrive:x/data/history"
REL = "intraday/1m/regular/2026-09/2026-09-01.parquet"


def _evicted(tmp_path: Path) -> tuple[Path, bytes]:
    path = tmp_path / REL
    path.parent.mkdir(parents=True)
    pd.DataFrame({"symbol": ["005930"], "v": [1]}).to_parquet(path, index=False)
    payload = path.read_bytes()
    md5 = hashlib.md5(payload).hexdigest()  # noqa: S324

    def _listing(cmd, **kwargs):
        out = f"{md5}  {path.name}\n" if cmd[1] == "md5sum" else f"{len(payload)};{path.name}\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")

    report = evict_verified_partitions(
        history_dir=tmp_path, today=date(2026, 11, 5), families=[EvictionFamily("bars", ("intraday/1m",), 10)],
        remote_base=BASE, run_fn=_listing, max_files=5,
    )
    assert report.evicted == (REL,) and not path.exists()
    return path, payload


def _copyto(content: bytes):
    def _run(cmd, **kwargs):
        assert cmd[1] == "copyto" and cmd[2] == f"{BASE}/{REL}"
        Path(cmd[3]).write_bytes(content)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return _run


def test_restore_roundtrip(tmp_path: Path) -> None:
    path, payload = _evicted(tmp_path)

    restored = restore_partitions(history_dir=tmp_path, relpaths=[REL], remote_base=BASE, run_fn=_copyto(payload))

    assert restored == (REL,)
    assert path.read_bytes() == payload
    assert read_tombstone(path) is None
    assert not list(path.parent.glob("*.restore.tmp"))


def test_restore_md5_mismatch_keeps_tombstone(tmp_path: Path) -> None:
    path, payload = _evicted(tmp_path)

    with pytest.raises(RuntimeError):
        restore_partitions(history_dir=tmp_path, relpaths=[REL], remote_base=BASE, run_fn=_copyto(payload[:-1] + b"X"))

    assert not path.exists()
    assert tombstone_path(path).exists()
    assert not list(path.parent.glob("*.restore.tmp"))


def test_restore_dry_run_changes_nothing(tmp_path: Path) -> None:
    path, _ = _evicted(tmp_path)

    def _never(cmd, **kwargs):
        raise AssertionError("no rclone on dry run")

    assert restore_partitions(history_dir=tmp_path, relpaths=[REL], remote_base=BASE, run_fn=_never, dry_run=True) == (REL,)
    assert tombstone_path(path).exists() and not path.exists()


def test_restore_continues_after_failure_then_raises(tmp_path: Path) -> None:
    path, payload = _evicted(tmp_path)

    with pytest.raises(RuntimeError):
        restore_partitions(
            history_dir=tmp_path, relpaths=["intraday/1m/regular/2026-09/missing.parquet", REL],
            remote_base=BASE, run_fn=_copyto(payload),
        )

    assert path.read_bytes() == payload and not tombstone_path(path).exists()


def test_restore_rejects_missing_malformed_and_existing(tmp_path: Path) -> None:
    path, payload = _evicted(tmp_path)
    tombstone_path(path).write_text("{")

    with pytest.raises(RuntimeError, match="failed"):
        restore_partitions(history_dir=tmp_path, relpaths=[REL], remote_base=BASE, run_fn=_copyto(payload))
    assert not path.exists()


def test_restore_refuses_when_partition_already_present(tmp_path: Path) -> None:
    path, payload = _evicted(tmp_path)
    path.write_bytes(b"local")

    with pytest.raises(RuntimeError, match="failed"):
        restore_partitions(history_dir=tmp_path, relpaths=[REL], remote_base=BASE, run_fn=_copyto(payload))
    assert path.read_bytes() == b"local"


def test_restore_rclone_failure_leaves_tombstone(tmp_path: Path) -> None:
    path, _ = _evicted(tmp_path)

    def _fail(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "err")

    with pytest.raises(RuntimeError, match="failed"):
        restore_partitions(history_dir=tmp_path, relpaths=[REL], remote_base=BASE, run_fn=_fail)
    assert tombstone_path(path).exists()


def test_cli_restores_by_path_and_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import settings
    from src.tools import partition_restore

    path, payload = _evicted(tmp_path)
    monkeypatch.setattr(settings, "HISTORY_DIR", tmp_path)

    def _run(cmd, **kwargs):
        assert cmd[2].endswith(f"/data/history/{REL}")
        Path(cmd[3]).write_bytes(payload)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(partition_restore.subprocess, "run", _run)

    assert partition_restore.main(["--family", "bars", "--from", "2026-09-01", "--to", "2026-09-30", "--dry-run"]) == 0
    assert tombstone_path(path).exists()
    assert partition_restore.main(["--path", REL]) == 0
    assert path.exists()


def test_cli_exits_nonzero_on_failure_and_validates_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import settings
    from src.tools import partition_restore

    monkeypatch.setattr(settings, "HISTORY_DIR", tmp_path)
    assert partition_restore.main(["--path", REL]) == 1
    with pytest.raises(SystemExit):
        partition_restore.main([])
    with pytest.raises(SystemExit):
        partition_restore.main(["--family", "bars"])


def test_cli_family_selection_ignores_non_date_tombstones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src import settings
    from src.tools import partition_restore

    path, _ = _evicted(tmp_path)
    (path.parent / "notes.parquet.evicted.json").write_text("{}")
    monkeypatch.setattr(settings, "HISTORY_DIR", tmp_path)

    assert partition_restore.main(["--family", "bars", "--from", "2026-09-01", "--to", "2026-09-30", "--dry-run"]) == 0
