"""Crash-safe cross-process exclusive file locks via kernel flock."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import time
from collections.abc import Iterator
from pathlib import Path

DEFAULT_LOCK_TIMEOUT_SECONDS: float = 30.0
LOCK_POLL_INTERVAL_SECONDS: float = 0.01

_CONTENTION_ERRNOS: frozenset[int] = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})


def sidecar_lock_path(target: Path) -> Path:
    """Return the sidecar lock path guarding ``target``.

    The sidecar sits next to the guarded file (``<name>.lock``) so it shares
    the target's filesystem and is visible to the capture-offsite in-flight
    scan, which skips ``*.lock`` names.

    Args:
        target: File whose publication is being serialized.

    Returns:
        ``target.parent / (target.name + ".lock")``.
    """
    return target.parent / (target.name + ".lock")


def open_lock_descriptor(path: Path) -> int:
    """Open a descriptor suitable for ``flock`` on ``path``, creating the file 0644 when absent.

    flock needs only an open descriptor, not write access: a 0644 lock file left by another uid (a crashed root-run
    manual job) stays lockable through a read-only descriptor, so it must not fail the caller.

    Args:
        path: Lock file; its parent directory must exist.

    Returns:
        An open file descriptor owned by the caller, who must close it.

    Raises:
        PermissionError: The file can be opened neither read-write nor read-only by this uid.
        OSError: Any other ``open`` failure.
    """
    try:
        return os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except PermissionError:
        return os.open(path, os.O_RDONLY)


@contextlib.contextmanager
def exclusive_file_lock(path: Path, *, timeout_seconds: float, purpose: str) -> Iterator[None]:
    """Hold a crash-safe, cross-process exclusive lock on ``path`` for the block.

    The lock is a kernel ``flock`` on an open descriptor, not the existence of
    the file. The kernel releases it when the holding process dies, so a
    SIGKILLed writer (systemd timeout, container kill, OOM) can never leave a
    lock that blocks later writers. A leftover file from a crashed holder or
    from the retired O_EXCL scheme is reused rather than treated as held. The
    sidecar is unlinked on release so no lock files persist on disk. After each
    acquisition the descriptor's inode is checked against the path, which
    rejects the unlink race where a waiter locks an inode that its previous
    holder has already removed.

    The lock is not reentrant. Because flock ownership belongs to the open file
    description, a second acquisition of the same path from the same process
    waits until timeout like any other contender.

    Args:
        path: Sidecar lock file. It must be on a local filesystem shared by all
            contenders; parent directories are created if missing.
        timeout_seconds: Wall-clock budget, measured on the monotonic clock,
            for acquiring the lock. ``0.0`` means exactly one attempt.
        purpose: Short label naming the guarded resource (e.g. ``"partition"``,
            ``"publish"``), used in the timeout message.

    Yields:
        None, while the lock is held.

    A sidecar owned by another uid is opened read-only; one this uid cannot open at all is
    treated as held until the deadline.

    Raises:
        TimeoutError: The lock was not acquired within ``timeout_seconds``.
            The message starts with ``"timed out acquiring {purpose} lock"``.
        OSError: Unexpected ``open``/``flock`` failure (any errno other than
            contention), propagated unchanged.
    """
    if timeout_seconds < 0.0:
        raise ValueError(f"timeout_seconds must be >= 0.0: {timeout_seconds!r}")
    if not isinstance(purpose, str) or not purpose:
        raise ValueError(f"purpose must be a non-empty string: {purpose!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            fd = open_lock_descriptor(path)
        except PermissionError:
            # Neither writable nor readable by this uid (a foreign 0600 leftover or live holder):
            # exclusion cannot be proven, so wait like contention and fail closed at the deadline.
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring {purpose} lock (lock file not accessible): {path}") from None
            time.sleep(LOCK_POLL_INTERVAL_SECONDS)
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            if exc.errno not in _CONTENTION_ERRNOS:
                raise
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring {purpose} lock: {path}") from None
            time.sleep(LOCK_POLL_INTERVAL_SECONDS)
            continue
        try:
            if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (
                os.stat(path).st_dev,
                os.stat(path).st_ino,
            ):
                raise FileNotFoundError(path)
        except FileNotFoundError:
            os.close(fd)
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring {purpose} lock: {path}") from None
            time.sleep(LOCK_POLL_INTERVAL_SECONDS)
            continue
        break
    try:
        yield
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            os.close(fd)
        raise
    else:
        try:
            path.unlink(missing_ok=True)
        finally:
            os.close(fd)
