"""호가 사다리/예상체결 원천 스냅샷 저장소 (KIS output1+output2 verbatim)."""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet, read_existing_parquet

logger = logging.getLogger(__name__)

__all__ = ["append_orderbook_snapshots", "build_orderbook_rows", "orderbook_partition_path"]

_NUMERIC_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def _coerce_value(key: str, value: Any) -> Any:
    """KIS 필드값을 숫자로 캐스팅한다(단, 종목코드 계열 식별자는 제외).

    stck_shrn_iscd/mksc_shrn_iscd 같은 iscd 필드는 자릿수 전부 숫자인 경우도
    있고(예: "005930") 워런트/신주인수권 등에서 문자를 포함하기도 한다
    (예: "0220W0"). 식별자를 숫자로 바꾸면 선행 0이 소실되고, 같은 컬럼 안에
    숫자/문자열이 섞여 parquet 스키마 추론이 실패하므로 애초에 변환 대상에서
    제외한다.
    """
    if key.endswith("iscd"):
        return value
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


def _require_session(value: str) -> str:
    """Require a path-safe orderbook session name (regular or a dated venue session)."""
    if not isinstance(value, str) or not value.strip() or value.strip() != value:
        raise ValueError("orderbook session must be nonempty")
    if "/" in value or "\\" in value or "\x00" in value:
        raise ValueError("orderbook session must be path-safe")
    if value in (".", "..") or ".." in value:
        raise ValueError("orderbook session must be path-safe")
    return value


def build_orderbook_rows(
    res: dict[str, Any],
    symbol: str,
    venue: str,
    capture_reason: str,
    capture_ts: datetime,
    *,
    scheduled_at: datetime | None = None,
    request_started_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """벤더 output1+output2 페이로드를 키 그대로 복사한 단일 행으로 만든다."""
    if not isinstance(res, dict) or str(res.get("rt_cd", "")) != "0":
        return []
    output1 = res.get("output1")
    output2 = res.get("output2")
    if not isinstance(output1, dict) and not isinstance(output2, dict):
        return []
    row: dict[str, Any] = {
        "capture_ts": capture_ts,
        "symbol": str(symbol).zfill(6),
        "venue": str(venue),
        "capture_reason": str(capture_reason),
    }
    if scheduled_at is not None:
        row["scheduled_at"] = scheduled_at
    if request_started_at is not None:
        row["request_started_at"] = request_started_at
    if isinstance(output1, dict):
        for key, value in output1.items():
            row[str(key)] = _coerce_value(str(key), value)
    if isinstance(output2, dict):
        for key, value in output2.items():
            row[str(key)] = _coerce_value(str(key), value)
    return [row]


def orderbook_partition_path(snapshot_date: str, session: str = "regular") -> Path:
    """data/history/orderbook/{YYYY-MM}/{YYYY-MM-DD}.parquet 경로 산출."""
    month = str(snapshot_date)[:7]
    if _require_session(session) == "regular":
        return Path(settings.HISTORY_DIR) / "orderbook" / month / f"{snapshot_date}.parquet"
    return Path(settings.HISTORY_DIR) / "orderbook" / session / month / f"{snapshot_date}.parquet"


def append_orderbook_snapshots(rows: list[dict[str, Any]], snapshot_date: str, *, session: str = "regular") -> int:
    """호가 스냅샷 행을 일자 파티션에 병합 추가한다. 빈 입력은 0 반환 no-op."""
    if not rows:
        return 0
    target = orderbook_partition_path(snapshot_date, session)
    new_df = pd.DataFrame(rows)
    existing = read_existing_parquet(target)
    if len(existing) == 0:
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
