"""Intraday 분봉 날짜 파티션 저장소 (date-partitioned parquet)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet

logger = logging.getLogger(__name__)

__all__ = ["intraday_partition_path", "read_intraday_range", "write_intraday_partition"]


def intraday_partition_path(bar_interval_minutes: int, snapshot_date: str, session: str) -> Path:
    """data/history/intraday/{interval}m/{session}/{YYYY-MM}/{YYYY-MM-DD}.parquet 경로 산출."""
    month = str(snapshot_date)[:7]
    return (
        Path(settings.HISTORY_DIR)
        / "intraday"
        / f"{int(bar_interval_minutes)}m"
        / str(session)
        / month
        / f"{snapshot_date}.parquet"
    )


def write_intraday_partition(df: pd.DataFrame, bar_interval_minutes: int, snapshot_date: str, session: str) -> int:
    """일자별 파티션 파일 단위로만 원자적 저장한다. 빈 df는 0 반환 no-op."""
    if df is None or df.empty:
        return 0
    target = intraday_partition_path(bar_interval_minutes, snapshot_date, session)
    atomic_write_parquet(df, target)
    logger.info("Wrote intraday partition %s (%d rows)", target, len(df))
    return len(df)


def read_intraday_range(
    bar_interval_minutes: int, start_date: str, end_date: str, session: str = "regular"
) -> pd.DataFrame:
    """날짜 범위에 해당하는 파티션 파일만 글롭하여 concat. 대상 없으면 빈 DataFrame."""
    base = (
        Path(settings.HISTORY_DIR)
        / "intraday"
        / f"{int(bar_interval_minutes)}m"
        / str(session)
    )
    if not base.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(base.rglob("*.parquet")):
        date_str = path.stem
        if start_date <= date_str <= end_date:
            try:
                frames.append(pd.read_parquet(path))
            except Exception as e:
                logger.warning("Failed to read intraday partition %s: %s", path, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
