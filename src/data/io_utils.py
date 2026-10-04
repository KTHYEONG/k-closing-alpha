"""Atomic file publication and append-only store reads shared by every persistent writer."""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

import pandas as pd

from src.utils.file_lock import DEFAULT_LOCK_TIMEOUT_SECONDS, exclusive_file_lock, sidecar_lock_path

logger = logging.getLogger(__name__)

# The stage-*.tmp shape is a cross-module contract: src/tools/capture_offsite.py::_is_inflight
# excludes exactly these names from sealing and makes local prune keep the directory, and .tmp
# is not a joblib/pandas compression-inferring extension.
ATOMIC_TMP_PREFIX: Final[str] = "stage-"
ATOMIC_TMP_SUFFIX: Final[str] = ".tmp"


class ExistingStoreUnreadableError(OSError):
    """An existing history file could not be read; it must not be overwritten."""


STORE_LOCK_TIMEOUT_SECONDS: float = DEFAULT_LOCK_TIMEOUT_SECONDS


class StoreLockTimeoutError(TimeoutError):
    """A shared store's write lock was not acquired in time; nothing was read or written under it."""


@contextlib.contextmanager
def store_write_lock(target: Path, *, purpose: str, timeout_seconds: float | None = None) -> Iterator[None]:
    """Serialize one read-modify-write of a shared store across processes.

    Atomic replace keeps the file whole but cannot stop lost updates: two writers that both read version N
    publish N+a and N+b, and the later one silently drops the other's rows. systemd ordering prevents this
    only for scheduled units; manual reruns bypass it. Hold this lock from before the read until after the
    atomic write.

    The lock is the kernel-flock sidecar ``<target>.lock`` (``exclusive_file_lock``): released on process
    death, unlinked on release, not reentrant. Do not nest two acquisitions of the same target.

    Args:
        target: Store file being rewritten (it need not exist yet).
        purpose: Short store label for logs and the timeout message (e.g. ``"condition-archive"``).
        timeout_seconds: Acquisition budget; ``0.0`` means one non-blocking attempt. ``None`` resolves
            ``STORE_LOCK_TIMEOUT_SECONDS`` at call time.

    Yields:
        None while the lock is held.

    Raises:
        StoreLockTimeoutError: Acquisition did not succeed within the budget (chained from the underlying
            ``TimeoutError``); logged as ``[DATA] stage=store_lock status=TIMEOUT``.
        OSError: Unexpected lock-file failure, propagated unchanged.
    """
    budget = STORE_LOCK_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    acquired = False
    try:
        with exclusive_file_lock(sidecar_lock_path(Path(target)), timeout_seconds=budget, purpose=purpose):
            acquired = True
            yield
    except TimeoutError as exc:
        if not acquired:
            logger.error(
                "[DATA] stage=store_lock status=TIMEOUT purpose=%s path=%s timeout_s=%.1f",
                purpose,
                target,
                budget,
            )
            raise StoreLockTimeoutError(f"timed out acquiring {purpose} store lock: {target}") from exc
        raise


def read_existing_parquet(path: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read an append-only store for a read-modify-write merge.

    Absence is the only legitimate "empty history". Any failure to read a file
    that exists means the history is unknown, and replacing it with today's rows
    would destroy data that exists only here and in the offsite copy.

    Args:
        path: Parquet file path.
        columns: Optional column projection.

    Returns:
        The stored frame, or an empty DataFrame when the path does not exist.

    Raises:
        ExistingStoreUnreadableError: The path exists but reading failed; the
            original exception is chained as __cause__ and the path is in the message.
    """
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(target, columns=columns)
    except Exception as exc:
        raise ExistingStoreUnreadableError(
            f"existing history at {target} could not be read; refusing to overwrite"
        ) from exc


def _checked_target(target: str | os.PathLike[str], mode: int | None) -> Path:
    if mode is not None and (not isinstance(mode, int) or not 0o000 <= mode <= 0o777):
        raise ValueError(f"mode must be None or an int within 0o000..0o777, got {mode!r}")
    path = Path(target)
    if not path.name:
        raise ValueError(f"target must name a file, got {target!r}")
    return path


def _stage_name(basename: str) -> str:
    name = f"{ATOMIC_TMP_PREFIX}{basename}.{uuid.uuid4().hex}{ATOMIC_TMP_SUFFIX}"
    raw = name.encode("utf-8")
    if len(raw) > 255:
        head = basename.encode("utf-8")[: len(basename.encode("utf-8")) - (len(raw) - 255)]
        basename = head.decode("utf-8", errors="ignore")
        name = f"{ATOMIC_TMP_PREFIX}{basename}.{uuid.uuid4().hex}{ATOMIC_TMP_SUFFIX}"
    return name


def _create_temp(parent: Path, basename: str, mode: int | None) -> Path:
    tmp = parent / _stage_name(basename)
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600 if mode is not None else 0o666)
    os.close(fd)
    return tmp


def _remove_quietly(tmp: Path) -> None:
    with contextlib.suppress(OSError):
        tmp.unlink()


def _publish_temp(tmp: Path, target: Path, *, mode: int | None, durable: bool) -> None:
    try:
        if durable:
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, target)
        if durable:
            dir_fd = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    except BaseException:
        _remove_quietly(tmp)
        raise


@contextlib.contextmanager
def atomic_output_path(target: str | os.PathLike[str], *, mode: int | None, durable: bool = False) -> Iterator[Path]:
    """Yield a private temp path whose content is published to ``target`` atomically on clean exit.

    For writers that need a filesystem path (pyarrow, joblib) rather than an in-memory payload. The temp file
    is created exclusively in ``target``'s directory, so ``os.replace`` is a same-filesystem rename: readers see
    either the previous complete file or the new complete file, never a partial one. The writer must write to
    the yielded path in place (truncate/overwrite); it must not unlink or rename it.

    Args:
        target: Destination file. Parent directories are created if missing. Symlinks are not resolved; an
            existing symlink at ``target`` is replaced by the new file, as ``os.replace`` does.
        mode: Exact permission bits of the published file, applied before publication and independent of the
            process umask (e.g. ``0o644`` for files read by the offsite-backup account, ``0o600`` for
            credentials). ``None`` keeps the umask-derived mode a plain ``open(target, "w")`` would create
            (``0o666 & ~umask``). Required keyword so every caller states its permission contract; the
            2026-09-19 offsite-backup failure came from an implicit ``0o600``.
        durable: When True, the temp file's data is fsynced before the rename and the parent directory is
            fsynced after it, so the publication survives power loss. When False, durability is left to the
            kernel's writeback (crash-consistent only).

    Yields:
        The temp path (``<parent>/stage-<target name>.<uuid32 hex>.tmp``), already created and empty.

    Raises:
        OSError: Directory creation, temp creation, chmod, fsync, or rename failed. On any failure before the
            rename (including a ``BaseException`` raised by the caller's block), the temp file is removed and
            ``target`` is untouched. A directory-fsync failure after the rename raises with ``target`` already
            published.
    """
    path = _checked_target(target, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _create_temp(path.parent, path.name, mode)
    try:
        yield tmp
    except BaseException:
        _remove_quietly(tmp)
        raise
    _publish_temp(tmp, path, mode=mode, durable=durable)


def atomic_write_bytes(target: str | os.PathLike[str], data: bytes, *, mode: int | None, durable: bool = False) -> None:
    """Atomically publish ``data`` as the complete content of ``target``.

    Same publication, mode, durability and cleanup contract as ``atomic_output_path``.

    Args:
        target: Destination file.
        data: Complete file content.
        mode: Exact permission bits, or ``None`` for the umask-derived default.
        durable: fsync file and parent directory around the rename.

    Raises:
        OSError: See ``atomic_output_path``; ``target`` is never left partially written.
    """
    path = _checked_target(target, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _create_temp(path.parent, path.name, mode)
    try:
        tmp.write_bytes(data)
    except BaseException:
        _remove_quietly(tmp)
        raise
    _publish_temp(tmp, path, mode=mode, durable=durable)


def atomic_write_text(
    target: str | os.PathLike[str], text: str, *, mode: int | None, encoding: str = "utf-8", durable: bool = False
) -> None:
    """Atomically publish ``text`` (encoded with ``encoding``) as the complete content of ``target``.

    Callers serialize JSON themselves so each keeps its exact ``json.dumps`` options; the encoded bytes are
    what ``Path.write_text(text, encoding=encoding)`` would write (no newline translation).

    Args:
        target: Destination file.
        text: Complete file content.
        mode: Exact permission bits, or ``None`` for the umask-derived default.
        encoding: Text encoding.
        durable: fsync file and parent directory around the rename.

    Raises:
        UnicodeEncodeError: ``text`` is not encodable; nothing is created.
        OSError: See ``atomic_output_path``.
    """
    data = text.encode(encoding)
    atomic_write_bytes(target, data, mode=mode, durable=durable)


def atomic_write_parquet(
    df: pd.DataFrame,
    target_path: Path,
    compression: str = "zstd",
    compression_level: int | None = 6,
) -> None:
    """Atomically publish ``df`` as a parquet file at ``target_path`` with mode 0644.

    Mode is fixed at 0644 because these stores are read by the offsite-backup account (a different user);
    a temp file's private mode would otherwise be inherited through the rename (2026-09-19 incident).

    Args:
        df: Frame to write (index dropped).
        target_path: Destination parquet file.
        compression: Parquet codec.
        compression_level: zstd level; ignored for other codecs or when None.

    Raises:
        OSError: Publication failed; the destination is untouched and no temp file remains.
        Exception: Any serialization error from ``DataFrame.to_parquet``, after the same cleanup.
    """
    target_path = Path(target_path)
    try:
        with atomic_output_path(target_path, mode=0o644) as tmp_path:
            kwargs: dict[str, object] = {"index": False, "compression": compression}
            if compression == "zstd" and compression_level is not None:
                kwargs["compression_level"] = compression_level
            df.to_parquet(tmp_path, **kwargs)
    except Exception as e:
        logger.error("Failed to write parquet atomically to %s: %s", target_path, e)
        raise
