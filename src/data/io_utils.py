"""Parquet 원자적 쓰기 공용 유틸."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


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
        os.replace(tmp_path, target_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error("Failed to write parquet atomically to %s: %s", target_path, e)
        raise
