"""Crash-safe exclusive file lock invariant guards."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from pathlib import Path

import pytest

from src.utils.file_lock import (
    LOCK_POLL_INTERVAL_SECONDS,
    exclusive_file_lock,
    sidecar_lock_path,
)


def _hold_lock(path_str: str, hold_seconds: float, ready: object, done: object) -> None:
    """Child entry: hold the lock, signal readiness, block until released."""
    from src.utils.file_lock import exclusive_file_lock as lock

    with lock(Path(path_str), timeout_seconds=30.0, purpose="test"):
        ready.set()  # type: ignore[attr-defined]
        assert done.wait(timeout=30.0)  # type: ignore[attr-defined]
        time.sleep(hold_seconds)


def _counter_worker(path_str: str, counter_str: str, log_str: str, tag: str, rounds: int) -> None:
    """Child entry: read-increment-write a shared counter under the lock."""
    from src.utils.file_lock import exclusive_file_lock as lock

    sidecar = Path(path_str)
    for i in range(rounds):
        with lock(sidecar, timeout_seconds=30.0, purpose="test"):
            with open(log_str, "a", encoding="utf-8") as fh:
                fh.write(f"enter-{tag}-{i}\n")
            with open(counter_str, encoding="utf-8") as fh:
                value = int(fh.read().strip())
            with open(counter_str, "w", encoding="utf-8") as fh:
                fh.write(str(value + 1))
            with open(log_str, "a", encoding="utf-8") as fh:
                fh.write(f"exit-{tag}-{i}\n")


def _spawn(fn: object, *args: object) -> tuple[object, object, object]:
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    done = ctx.Event()
    proc = ctx.Process(target=fn, args=(*args, ready, done))
    proc.start()
    return proc, ready, done


def test_sidecar_lock_path_naming_creates_nothing(tmp_path: Path) -> None:
    """sidecar_lock_path returns <name>.lock without touching disk."""
    target = tmp_path / "a" / "b.parquet"
    assert sidecar_lock_path(target) == tmp_path / "a" / "b.parquet.lock"
    assert not (tmp_path / "a").exists()


def test_normal_release_removes_sidecar(tmp_path: Path) -> None:
    """Completing the block creates parents and removes the sidecar."""
    sidecar = tmp_path / "sub" / "x.parquet.lock"
    with exclusive_file_lock(sidecar, timeout_seconds=5.0, purpose="partition"):
        assert sidecar.exists()
    assert sidecar.parent.is_dir()
    assert not sidecar.exists()


def test_exception_in_block_propagates_and_releases(tmp_path: Path) -> None:
    """A block error propagates unchanged and frees the lock immediately."""
    sidecar = tmp_path / "x.parquet.lock"
    with (
        pytest.raises(ValueError, match="boom"),
        exclusive_file_lock(sidecar, timeout_seconds=5.0, purpose="partition"),
    ):
        raise ValueError("boom")
    assert not sidecar.exists()
    with exclusive_file_lock(sidecar, timeout_seconds=0.0, purpose="partition"):
        pass
    assert not sidecar.exists()


def test_stale_sidecar_does_not_block(tmp_path: Path) -> None:
    """A leftover file from the retired scheme is reused, then removed."""
    sidecar = tmp_path / "x.parquet.lock"
    sidecar.write_text("stale", encoding="utf-8")
    with exclusive_file_lock(sidecar, timeout_seconds=0.0, purpose="partition"):
        pass
    assert not sidecar.exists()


def test_timeout_raises_timeout_error_subclass_of_oserror(tmp_path: Path) -> None:
    """Contention past the budget raises TimeoutError with the uniform message."""
    sidecar = tmp_path / "x.parquet.lock"
    proc, ready, done = _spawn(_hold_lock, str(sidecar), 2.0)
    try:
        assert ready.wait(timeout=30.0)
        started = time.monotonic()
        with (
            pytest.raises(TimeoutError) as exc_info,
            exclusive_file_lock(sidecar, timeout_seconds=0.2, purpose="partition"),
        ):
            pass
        elapsed = time.monotonic() - started
        assert exc_info.value.args == (f"timed out acquiring partition lock: {sidecar}",)
        try:
            raise exc_info.value
        except OSError:
            pass
        else:  # pragma: no cover
            pytest.fail("TimeoutError must be caught by except OSError")
        assert elapsed >= 0.2
        assert elapsed < 2.0
    finally:
        done.set()
        proc.join(timeout=30.0)


def test_zero_timeout_is_a_single_attempt(tmp_path: Path) -> None:
    """timeout_seconds=0.0 fails fast without sleeping through polls."""
    sidecar = tmp_path / "x.parquet.lock"
    proc, ready, done = _spawn(_hold_lock, str(sidecar), 0.5)
    try:
        assert ready.wait(timeout=30.0)
        started = time.monotonic()
        with (
            pytest.raises(TimeoutError, match="timed out acquiring test lock"),
            exclusive_file_lock(sidecar, timeout_seconds=0.0, purpose="test"),
        ):
            pass  # pragma: no cover
        assert time.monotonic() - started < 2 * LOCK_POLL_INTERVAL_SECONDS + 0.5
    finally:
        done.set()
        proc.join(timeout=30.0)


def test_negative_timeout_rejected_without_sidecar(tmp_path: Path) -> None:
    """A negative budget raises ValueError before touching disk."""
    sidecar = tmp_path / "x.parquet.lock"
    with (
        pytest.raises(ValueError, match="timeout_seconds"),
        exclusive_file_lock(sidecar, timeout_seconds=-1.0, purpose="partition"),
    ):
        pass  # pragma: no cover
    assert not sidecar.exists()


def test_empty_purpose_rejected_without_sidecar(tmp_path: Path) -> None:
    """An empty purpose raises ValueError before touching disk."""
    sidecar = tmp_path / "x.parquet.lock"
    with (
        pytest.raises(ValueError, match="purpose"),
        exclusive_file_lock(sidecar, timeout_seconds=5.0, purpose=""),
    ):
        pass  # pragma: no cover
    assert not sidecar.exists()


def test_killed_holder_releases_the_lock(tmp_path: Path) -> None:
    """SIGKILL of the holder frees the lock even though its file remains."""
    sidecar = tmp_path / "x.parquet.lock"
    proc, ready, done = _spawn(_hold_lock, str(sidecar), 30.0)
    try:
        assert ready.wait(timeout=30.0)
        assert sidecar.exists()
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(timeout=30.0)
        assert proc.exitcode == -signal.SIGKILL
        started = time.monotonic()
        with exclusive_file_lock(sidecar, timeout_seconds=5.0, purpose="partition"):
            pass
        assert time.monotonic() - started < 5.0
    finally:
        if proc.is_alive():
            done.set()
            proc.join(timeout=30.0)
    assert not sidecar.exists()


def test_concurrent_writers_serialize_without_lost_updates(tmp_path: Path) -> None:
    """4 processes x 50 read-increment-write cycles yield exactly 200."""
    sidecar = tmp_path / "counter.parquet.lock"
    counter = tmp_path / "counter.txt"
    log = tmp_path / "markers.log"
    counter.write_text("0", encoding="utf-8")
    log.write_text("", encoding="utf-8")
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_counter_worker, args=(str(sidecar), str(counter), str(log), f"p{k}", 50))
        for k in range(4)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120.0)
    assert all(proc.exitcode == 0 for proc in procs)
    assert counter.read_text(encoding="utf-8").strip() == "200"
    lines = log.read_text(encoding="utf-8").split()
    assert len(lines) == 400
    for enter, exit in zip(lines[::2], lines[1::2], strict=True):
        assert enter.startswith("enter-")
        assert exit == enter.replace("enter-", "exit-", 1)


def test_unlink_race_cannot_yield_two_holders(tmp_path: Path) -> None:
    """A waiter holding a pre-release descriptor never overlaps a newcomer."""
    import threading

    sidecar = tmp_path / "x.parquet.lock"
    log = tmp_path / "markers.log"
    log.write_text("", encoding="utf-8")

    def _guarded(tag: str, delay: float) -> None:
        time.sleep(delay)
        with exclusive_file_lock(sidecar, timeout_seconds=30.0, purpose="test"):
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"enter-{tag}\n")
            time.sleep(0.05)
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"exit-{tag}\n")

    with exclusive_file_lock(sidecar, timeout_seconds=5.0, purpose="test"):
        waiter = threading.Thread(target=_guarded, args=("waiter", 0.0))
        waiter.start()
        time.sleep(0.3)
    newcomer = threading.Thread(target=_guarded, args=("newcomer", 0.0))
    newcomer.start()
    waiter.join(timeout=60.0)
    newcomer.join(timeout=60.0)
    lines = log.read_text(encoding="utf-8").split()
    assert sorted(lines) == ["enter-newcomer", "enter-waiter", "exit-newcomer", "exit-waiter"]
    for enter, exit in zip(lines[::2], lines[1::2], strict=True):
        assert exit == enter.replace("enter-", "exit-", 1)


def test_unexpected_errno_propagates_without_descriptor_leak(tmp_path: Path) -> None:
    """A parent that is a regular file raises OSError, never TimeoutError."""
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    sidecar = blocker / "x.parquet.lock"
    with (
        pytest.raises(OSError, match=r"File exists|Not a directory"),
        exclusive_file_lock(sidecar, timeout_seconds=5.0, purpose="partition"),
    ):
        pass  # pragma: no cover


def test_unexpected_flock_errno_propagates_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-contention flock errno is re-raised, never converted to TimeoutError."""
    import errno
    import fcntl as fcntl_module

    def _boom(fd: int, op: int) -> None:
        raise OSError(errno.EIO, "simulated io error")

    monkeypatch.setattr(fcntl_module, "flock", _boom)
    sidecar = tmp_path / "x.parquet.lock"
    with (
        pytest.raises(OSError, match="simulated io error"),
        exclusive_file_lock(sidecar, timeout_seconds=5.0, purpose="partition"),
    ):
        pass  # pragma: no cover


def test_inode_mismatch_relocks_within_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale-inode observation discards the descriptor and acquires afresh."""
    real_stat = os.stat
    other = tmp_path / "other.txt"
    other.write_text("x", encoding="utf-8")
    calls = {"n": 0}

    def _mismatch_once(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if str(path).endswith(".lock") and calls["n"] == 0:
            calls["n"] += 1
            return real_stat(other)
        return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "stat", _mismatch_once)
    sidecar = tmp_path / "x.parquet.lock"
    with exclusive_file_lock(sidecar, timeout_seconds=5.0, purpose="partition"):
        pass
    assert calls["n"] == 1
    assert not sidecar.exists()


def test_persistent_inode_mismatch_times_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identity retries share the acquisition deadline."""
    real_stat = os.stat
    other = tmp_path / "other.txt"
    other.write_text("x", encoding="utf-8")
    def _always_mismatch(path: object, *args: object, **kwargs: object) -> os.stat_result:
        if str(path).endswith(".lock"):
            return real_stat(other)
        return real_stat(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "stat", _always_mismatch)
    sidecar = tmp_path / "x.parquet.lock"
    started = time.monotonic()
    with (
        pytest.raises(TimeoutError, match="timed out acquiring partition lock"),
        exclusive_file_lock(sidecar, timeout_seconds=0.1, purpose="partition"),
    ):
        pass  # pragma: no cover
    assert time.monotonic() - started >= 0.1
