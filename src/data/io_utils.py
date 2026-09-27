"""Parquet 원자적 쓰기 공용 유틸."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ExistingStoreUnreadableError(OSError):
    """An existing history file could not be read; it must not be overwritten."""


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


def atomic_write_parquet(
    df: pd.DataFrame,
    target_path: Path,
    compression: str = "zstd",
    compression_level: int | None = 6,
) -> None:
    """임시파일 작성 후 os.replace로 원자적 교체한다."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = target_path.parent
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".parquet") as tmp:
        tmp_path = Path(tmp.name)
    try:
        kwargs: dict[str, object] = {"index": False, "compression": compression}
        if compression == "zstd" and compression_level is not None:
            kwargs["compression_level"] = compression_level
        df.to_parquet(tmp_path, **kwargs)  # type: ignore[arg-type]
        # NamedTemporaryFile은 umask와 무관하게 항상 0600으로 생성되고 os.replace가 그 권한을
        # 그대로 승계한다 -- 실측: 2026-09-19 오프사이트 백업(다른 유저 실행)이 permission denied로 실패.
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, target_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error("Failed to write parquet atomically to %s: %s", target_path, e)
        raise
