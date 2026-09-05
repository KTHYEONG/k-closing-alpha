"""Intraday 분봉 날짜 파티션 저장소 (date-partitioned parquet)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import settings
from src.data.intraday_schema import assert_canonical_bars, assert_canonical_ticks
from src.data.io_utils import atomic_write_parquet

logger = logging.getLogger(__name__)

__all__ = ["intraday_partition_path", "merge_partition_frame", "read_intraday_range", "tick_partition_path", "write_intraday_partition", "write_tick_partition"]


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


def merge_partition_frame(new_df: pd.DataFrame, target: Path, key_cols: tuple[str, ...]) -> pd.DataFrame:
    """기존 파티션과 신규 프레임을 키 기준 병합한다 (new wins, 키 정렬)."""
    try:
        existing = pd.read_parquet(target) if target.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning("[DATA] Failed to read existing partition %s; writing new only: %s", target, e)
        existing = pd.DataFrame()
    if existing is None or len(existing) == 0:
        merged = new_df.copy()
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=list(key_cols), keep="last")
    merged = merged.sort_values(list(key_cols), kind="stable").reset_index(drop=True)
    if "symbol" in merged.columns and existing is not None and len(existing) > 0 and "symbol" in existing.columns:
        before = set(existing["symbol"].astype(str).unique().tolist())
        after = set(merged["symbol"].astype(str).unique().tolist())
        if not before.issubset(after):
            raise ValueError(f"Partition write would reduce symbol coverage: lost={sorted(before - after)}")
    return merged


def write_intraday_partition(df: pd.DataFrame, bar_interval_minutes: int, snapshot_date: str, session: str) -> int:
    """정규 바 파티션을 게이트 검증 후 병합 저장한다. 빈 df는 0 반환 no-op."""
    if df is None or df.empty:
        return 0
    assert_canonical_bars(df)
    target = intraday_partition_path(bar_interval_minutes, snapshot_date, session)
    merged = merge_partition_frame(df, target, ("symbol", "ts_hms"))
    atomic_write_parquet(merged, target)
    logger.info("Wrote intraday partition %s (%d rows)", target, len(merged))
    return len(merged)


def write_tick_partition(df: pd.DataFrame, snapshot_date: str, session: str = "regular") -> int:
    """틱 파티션을 게이트 검증 후 병합 저장한다. 빈 df는 0 반환 no-op."""
    if df is None or df.empty:
        return 0
    assert_canonical_ticks(df)
    target = tick_partition_path(snapshot_date, session)
    merged = merge_partition_frame(df, target, ("symbol", "ts_hms", "volume"))
    atomic_write_parquet(merged, target)
    logger.info("Wrote tick partition %s (%d rows)", target, len(merged))
    return len(merged)


def tick_partition_path(snapshot_date: str, session: str = "regular") -> Path:
    """data/history/intraday/ticks/{session}/{YYYY-MM}/{YYYY-MM-DD}.parquet 경로 산출."""
    month = str(snapshot_date)[:7]
    return (
        Path(settings.HISTORY_DIR)
        / "intraday"
        / "ticks"
        / str(session)
        / month
        / f"{snapshot_date}.parquet"
    )


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
