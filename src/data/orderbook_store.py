"""호가 사다리/예상체결 원천 스냅샷 저장소 (KIS output1 verbatim)."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet

logger = logging.getLogger(__name__)

__all__ = ["append_orderbook_snapshots", "build_orderbook_rows", "orderbook_partition_path"]

_NUMERIC_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _coerce_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if text != "" and _NUMERIC_RE.match(text):
            num = pd.to_numeric(text, errors="coerce")
            if num is not None and not pd.isna(num):
                return num
        return value
    return value


def build_orderbook_rows(
    res: dict, symbol: str, venue: str, capture_reason: str, capture_ts: datetime
) -> list[dict]:
    """벤더 output1 페이로드를 키 그대로 복사한 단일 행으로 만든다."""
    if not isinstance(res, dict) or str(res.get("rt_cd", "")) != "0":
        return []
    output1 = res.get("output1")
    if not isinstance(output1, dict):
        return []
    row: dict[str, Any] = {
        "capture_ts": capture_ts,
        "symbol": str(symbol).zfill(6),
        "venue": str(venue),
        "capture_reason": str(capture_reason),
    }
    for key, value in output1.items():
        row[str(key)] = _coerce_value(value)
    return [row]


def orderbook_partition_path(snapshot_date: str) -> Path:
    """data/history/orderbook/{YYYY-MM}/{YYYY-MM-DD}.parquet 경로 산출."""
    month = str(snapshot_date)[:7]
    return Path(settings.HISTORY_DIR) / "orderbook" / month / f"{snapshot_date}.parquet"


def append_orderbook_snapshots(rows: list[dict], snapshot_date: str) -> int:
    """호가 스냅샷 행을 일자 파티션에 병합 추가한다. 빈 입력은 0 반환 no-op."""
    if not rows:
        return 0
    target = orderbook_partition_path(snapshot_date)
    new_df = pd.DataFrame(rows)
    try:
        existing = pd.read_parquet(target) if target.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning("[DATA] Failed to read existing orderbook partition %s; writing new only: %s", target, e)
        existing = pd.DataFrame()
    if existing is None or len(existing) == 0:
        merged = new_df.copy()
    else:
        union_cols = sorted(set(existing.columns.tolist()) | set(new_df.columns.tolist()))
        merged = pd.concat(
            [existing.reindex(columns=union_cols), new_df.reindex(columns=union_cols)],
            ignore_index=True,
        )
        merged = merged.drop_duplicates(subset=["capture_ts", "symbol", "venue"], keep="last")
        merged = merged.sort_values(["capture_ts", "symbol", "venue"], kind="stable").reset_index(drop=True)
    atomic_write_parquet(merged, target)
    logger.info("Wrote orderbook partition %s (%d rows)", target, len(merged))
    return len(merged)
