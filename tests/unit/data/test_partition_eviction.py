"""Invariant guards for tombstoned eviction of offsite-verified history partitions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, timedelta, timezone, UTC
from pathlib import Path

import pandas as pd
import pytest

from src.config.collection import CollectionSettings
from src.data import partition_eviction as pe
from src.data.partition_eviction import (
    EvictionFamily,
    evict_verified_partitions,
    read_tombstone,
    stored_symbols,
    tombstone_path,
    validate_retention,
)

TODAY = date(2026, 11, 5)
BASE = "gdrive:x/data/history"
FAMILY = EvictionFamily("bars", ("intraday/1m",), 10)
OLD = "2026-09-01"


def _write(root: Path, name: str, rel: str = "intraday/1m/regular/2026-09", symbols=("005930", "000660")) -> Path:
    path = root / rel / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"symbol": list(symbols), "v": range(len(symbols))}).to_parquet(path, index=False)
    return path


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()  # noqa: S324


class FakeRclone:
    """Serves md5sum/lsf listings from a per-remote-dir mapping name -> (md5, size)."""

    def __init__(self, remote: dict[str, dict[str, tuple[str, int]]], fail_dirs: frozenset[str] = frozenset()) -> None:
        self.remote = remote
        self.fail_dirs = fail_dirs
        self.calls: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(list(cmd))
        directory = cmd[-1]
        if directory in self.fail_dirs:
            return subprocess.CompletedProcess(cmd, 1, "", "boom")
        entries = self.remote.get(directory, {})
        if cmd[1] == "md5sum":
            out = "".join(f"{md5}  {name}\n" for name, (md5, _) in entries.items())
        else:
            out = "".join(f"{size};{name}\n" for name, (_, size) in entries.items())
        return subprocess.CompletedProcess(cmd, 0, out, "")


def _remote_for(*paths: Path, month: str = "intraday/1m/regular/2026-09") -> dict[str, dict[str, tuple[str, int]]]:
    return {f"{BASE}/{month}": {p.name: (_md5(p), p.stat().st_size) for p in paths}}


def _run(root: Path, rclone, **kwargs):
    params = {"history_dir": root, "today": TODAY, "families": [FAMILY], "remote_base": BASE, "run_fn": rclone, "max_files": 100}
    params.update(kwargs)
    return evict_verified_partitions(**params)


def test_evicts_only_md5_and_size_match(tmp_path: Path) -> None:
    good = _write(tmp_path, "2026-09-01")
    bad = _write(tmp_path, "2026-09-02")
    gone = _write(tmp_path, "2026-09-03")
    remote = _remote_for(good, bad)
    remote[f"{BASE}/intraday/1m/regular/2026-09"][bad.name] = ("0" * 32, bad.stat().st_size)

    report = _run(tmp_path, FakeRclone(remote))

    assert report.evicted == ("intraday/1m/regular/2026-09/2026-09-01.parquet",)
    assert not good.exists() and tombstone_path(good).exists()
    assert bad.exists() and gone.exists()
    assert dict(report.kept) == {
        "intraday/1m/regular/2026-09/2026-09-02.parquet": "remote_mismatch",
        "intraday/1m/regular/2026-09/2026-09-03.parquet": "remote_missing",
    }


def test_size_mismatch_is_kept(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-09-01")
    remote = _remote_for(path)
    remote[f"{BASE}/intraday/1m/regular/2026-09"][path.name] = (_md5(path), path.stat().st_size + 1)

    report = _run(tmp_path, FakeRclone(remote))

    assert path.exists() and report.evicted == ()
    assert report.kept[0][1] == "remote_mismatch"


def test_keeps_within_retention(tmp_path: Path) -> None:
    boundary = TODAY - timedelta(days=FAMILY.retention_days)
    path = _write(tmp_path, boundary.isoformat(), rel="intraday/1m/regular/2026-10")
    rclone = FakeRclone(_remote_for(path, month="intraday/1m/regular/2026-10"))

    report = _run(tmp_path, rclone)

    assert path.exists() and report.evicted == () and report.kept == ()
    assert rclone.calls == []


def test_listing_failure_keeps_month(tmp_path: Path) -> None:
    aug = _write(tmp_path, "2026-08-01", rel="intraday/1m/regular/2026-08")
    sep = _write(tmp_path, "2026-09-01")
    remote = _remote_for(sep)
    remote.update(_remote_for(aug, month="intraday/1m/regular/2026-08"))
    rclone = FakeRclone(remote, fail_dirs=frozenset({f"{BASE}/intraday/1m/regular/2026-08"}))

    report = _run(tmp_path, rclone)

    assert aug.exists() and not sep.exists()
    assert report.kept == (("intraday/1m/regular/2026-08/2026-08-01.parquet", "remote_unverifiable"),)


def test_tombstone_before_unlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, OLD)
    rclone = FakeRclone(_remote_for(path))
    real_unlink = Path.unlink

    def _boom(self: Path, *a, **k):
        if self == path:
            raise OSError("disk")
        return real_unlink(self, *a, **k)

    with monkeypatch.context() as ctx:
        ctx.setattr(Path, "unlink", _boom)
        first = _run(tmp_path, rclone)
    assert first.evicted == () and path.exists() and tombstone_path(path).exists()

    calls_before = len(rclone.calls)
    second = _run(tmp_path, rclone)

    assert second.evicted == ("intraday/1m/regular/2026-09/2026-09-01.parquet",)
    assert not path.exists()
    assert len(rclone.calls) == calls_before


def test_tombstone_conflict_kept(tmp_path: Path) -> None:
    path = _write(tmp_path, OLD)
    tombstone_path(path).write_text(
        json.dumps({"partition": "p", "remote_path": "r", "md5": "0" * 32, "size": 1, "rows": 1, "symbols": [], "evicted_at": "t"})
    )

    report = _run(tmp_path, FakeRclone({}))

    assert path.exists()
    assert report.kept[0][1] == "tombstone_conflict"


def test_dry_run_no_mutation(tmp_path: Path) -> None:
    path = _write(tmp_path, OLD)

    report = _run(tmp_path, FakeRclone(_remote_for(path)), dry_run=True)

    assert report.evicted == ("intraday/1m/regular/2026-09/2026-09-01.parquet",)
    assert report.bytes_freed > 0
    assert path.exists() and not tombstone_path(path).exists()


def test_cap_and_deadline(tmp_path: Path) -> None:
    paths = [_write(tmp_path, f"2026-09-0{i}") for i in range(1, 6)]
    remote = _remote_for(*paths)

    capped = _run(tmp_path, FakeRclone(remote), max_files=2)
    assert len(capped.evicted) == 2 and capped.capped and not capped.stopped_by_deadline

    expired = datetime.now(UTC) - timedelta(minutes=1)
    late = _run(tmp_path, FakeRclone(remote), deadline=expired)
    assert late.evicted == () and late.stopped_by_deadline


def test_tombstone_records_symbols(tmp_path: Path) -> None:
    path = _write(tmp_path, OLD, symbols=("000660", "005930", "005930"))
    expected_md5, expected_size = _md5(path), path.stat().st_size

    _run(tmp_path, FakeRclone(_remote_for(path)))

    tomb = read_tombstone(path)
    assert tomb is not None
    assert tomb.symbols == ("000660", "005930") and tomb.rows == 3
    assert tomb.md5 == expected_md5 and tomb.size == expected_size
    assert tomb.remote_path == f"{BASE}/intraday/1m/regular/2026-09/{OLD}.parquet"
    assert stored_symbols(path) == {"000660", "005930"}


def test_stored_symbols_empty_without_parquet_or_tombstone(tmp_path: Path) -> None:
    assert stored_symbols(tmp_path / "none.parquet") == set()


def test_malformed_tombstone_raises(tmp_path: Path) -> None:
    path = tmp_path / "2026-09-01.parquet"
    tombstone_path(path).write_text("{not json")
    with pytest.raises(ValueError, match="Malformed"):
        read_tombstone(path)
    tombstone_path(path).write_text(json.dumps({"partition": "p"}))
    with pytest.raises(ValueError, match="Malformed"):
        read_tombstone(path)


def test_validate_retention_rejects_short_bars() -> None:
    profile = CollectionSettings(HISTORY_EVICTION_BARS_RETENTION_DAYS=CollectionSettings().COLLECTION_KIS_MINUTE_RETENTION_DAYS + 7)
    with pytest.raises(ValueError, match="must exceed"):
        validate_retention(profile)
    validate_retention(CollectionSettings())


def test_never_touches_excluded_trees(tmp_path: Path) -> None:
    capture = _write(tmp_path, OLD, rel="capture/intraday/1m/regular/2026-09")
    offsite = _write(tmp_path, OLD, rel="price_history_offsite/2026-09")
    families = pe.eviction_families(CollectionSettings())

    report = evict_verified_partitions(
        history_dir=tmp_path, today=date(2030, 1, 1), families=families, remote_base=BASE, run_fn=FakeRclone({}), max_files=10
    )

    assert capture.exists() and offsite.exists() and report.evicted == ()


def test_symlinks_and_non_date_names_are_skipped(tmp_path: Path) -> None:
    real = _write(tmp_path, "2026-09-01", rel="elsewhere")
    month = tmp_path / "intraday/1m/regular/2026-09"
    month.mkdir(parents=True)
    (month / "2026-09-01.parquet").symlink_to(real)
    (month / ".stage-abc.parquet").write_bytes(b"x")
    (month / "notes.parquet").write_bytes(b"x")

    report = _run(tmp_path, FakeRclone({}))

    assert report.evicted == () and report.kept == ()


def test_empty_remote_hash_is_unverifiable(tmp_path: Path) -> None:
    path = _write(tmp_path, OLD)
    remote = _remote_for(path)
    remote[f"{BASE}/intraday/1m/regular/2026-09"][path.name] = ("", path.stat().st_size)

    class Blank(FakeRclone):
        def __call__(self, cmd, **kwargs):
            result = super().__call__(cmd, **kwargs)
            if cmd[1] == "md5sum":
                return subprocess.CompletedProcess(cmd, 0, f"{'':34}{path.name}\n", "")
            return result

    report = _run(tmp_path, Blank(remote))

    assert path.exists() and report.kept[0][1] == "remote_unverifiable"


def test_tombstone_with_bad_symbols_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "2026-09-01.parquet"
    payload = {"partition": "p", "remote_path": "r", "md5": "m", "size": 1, "rows": 1, "symbols": "005930", "evicted_at": "t"}
    tombstone_path(path).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Malformed"):
        read_tombstone(path)


def test_stored_symbols_fails_closed_on_unreadable_evidence(tmp_path: Path) -> None:
    corrupt = tmp_path / "2026-09-01.parquet"
    corrupt.write_bytes(b"not parquet")
    with pytest.raises(OSError, match="Cannot read existing partition evidence"):
        stored_symbols(corrupt)
    bad_tomb = tmp_path / "2026-09-02.parquet"
    tombstone_path(bad_tomb).write_text("{")
    with pytest.raises(OSError, match="tombstone"):
        stored_symbols(bad_tomb)


def test_missing_remote_directory_keeps_files_as_missing(tmp_path: Path) -> None:
    path = _write(tmp_path, OLD)

    def _not_found(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 3, "", "directory not found")

    report = _run(tmp_path, _not_found)

    assert path.exists() and report.kept[0][1] == "remote_missing"


def test_rclone_spawn_failure_is_unverifiable(tmp_path: Path) -> None:
    path = _write(tmp_path, OLD)

    def _boom(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, 1)

    report = _run(tmp_path, _boom)

    assert path.exists() and report.kept[0][1] == "remote_unverifiable"


def test_blank_listing_lines_and_impossible_dates_are_ignored(tmp_path: Path) -> None:
    path = _write(tmp_path, OLD)
    _write(tmp_path, "2026-13-45")

    def _padded(cmd, **kwargs):
        inner = FakeRclone(_remote_for(path))(cmd, **kwargs)
        return subprocess.CompletedProcess(cmd, 0, "\n" + inner.stdout, "")

    report = _run(tmp_path, _padded)

    assert report.evicted == ("intraday/1m/regular/2026-09/2026-09-01.parquet",)
    assert (tmp_path / "intraday/1m/regular/2026-09/2026-13-45.parquet").exists()


def test_frame_without_symbol_column_records_empty_symbols(tmp_path: Path) -> None:
    path = tmp_path / "intraday/1m/regular/2026-09" / f"{OLD}.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame({"v": [1, 2]}).to_parquet(path, index=False)

    _run(tmp_path, FakeRclone(_remote_for(path)))

    tomb = read_tombstone(path)
    assert tomb is not None and tomb.symbols == () and tomb.rows == 2


def test_malformed_tombstone_beside_file_is_conflict(tmp_path: Path) -> None:
    path = _write(tmp_path, OLD)
    tombstone_path(path).write_text("{")

    report = _run(tmp_path, FakeRclone({}))

    assert path.exists() and report.kept[0][1] == "tombstone_conflict"


def test_file_changed_before_lock_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, "2026-09-01")
    remote = _remote_for(path)
    real_md5 = pe.md5_file
    calls = {"n": 0}

    def _md5_then_changed(target: Path) -> str:
        calls["n"] += 1
        return real_md5(target) if calls["n"] == 1 else "f" * 32

    monkeypatch.setattr(pe, "md5_file", _md5_then_changed)

    report = _run(tmp_path, FakeRclone(remote))

    assert path.exists() and not tombstone_path(path).exists()
    assert report.evicted == () and report.kept[0][1] == "changed_during_eviction"


def test_lock_timeout_keeps_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write(tmp_path, "2026-09-01")

    def _busy(*_args, **_kwargs):
        raise TimeoutError("busy")

    monkeypatch.setattr(pe, "exclusive_file_lock", _busy)

    report = _run(tmp_path, FakeRclone(_remote_for(path)))

    assert path.exists() and not tombstone_path(path).exists()
    assert report.evicted == () and report.kept[0][1] == "local_unreadable"
