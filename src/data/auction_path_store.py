"""Closing-auction indicative price path archive (DB class-8 rows)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src import settings
from src.config.market_session import (
    AUCTION_PATH_CEIL_HHMMSS,
    AUCTION_PATH_FLOOR_HHMMSS,
    DECISION_WINDOW_END_HHMMSS,
)
from src.data.io_utils import atomic_write_parquet

__all__ = [
    "AUCTION_PATH_COLUMNS",
    "AuctionPathRowKind",
    "auction_path_partition_path",
    "build_auction_path_rows",
    "write_auction_path_partition",
]

SEOUL = ZoneInfo("Asia/Seoul")


class AuctionPathRowKind(StrEnum):
    """Classification of one auction-path row."""

    INDICATIVE = "indicative"
    AUCTION_FILL = "auction_fill"
    UNEXPECTED_TRADE = "unexpected_trade"


AUCTION_PATH_COLUMNS: tuple[str, ...] = (
    "snapshot_date",
    "symbol",
    "seq",
    "ts_hms",
    "price",
    "buy_qty",
    "sell_qty",
    "kind",
    "available_at",
    "vendor",
)

_INT_RE = re.compile(r"^[+-]?\d+$")
_HHMMSS_RE = re.compile(r"^\d{6}$")


def _parse_int(raw: Any, *, symbol: str, field: str) -> int:
    if isinstance(raw, bool):
        raise ValueError(f"auction path {symbol} field {field} is not an integer: {raw!r}")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        if raw.is_integer():
            return int(raw)
        raise ValueError(f"auction path {symbol} field {field} is not an integer: {raw!r}")
    text = str(raw).strip().replace(",", "")
    if not _INT_RE.fullmatch(text):
        raise ValueError(f"auction path {symbol} field {field} is not an integer: {raw!r}")
    return int(text)


def build_auction_path_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    snapshot_date: str,
    symbol: str,
    available_at: datetime,
) -> pd.DataFrame:
    """Classify one symbol's raw DB class-8 rows into chronological auction-path rows."""
    if available_at.tzinfo is None or available_at.utcoffset() is None:
        raise ValueError(f"auction path {symbol} available_at must be timezone-aware")
    expect = datetime(
        int(snapshot_date[0:4]),
        int(snapshot_date[5:7]),
        int(snapshot_date[8:10]),
        15,
        30,
        0,
        tzinfo=SEOUL,
    )
    if available_at.astimezone(SEOUL) < expect:
        raise ValueError(f"auction path {symbol} available_at precedes the close print")
    records: list[dict[str, Any]] = []
    for row in reversed(list(raw_rows)):
        hour = str(row.get("Hour", "") or "").strip()
        if not _HHMMSS_RE.fullmatch(hour):
            raise ValueError(f"auction path {symbol} field Hour is not HHMMSS: {row.get('Hour')!r}")
        if hour < AUCTION_PATH_FLOOR_HHMMSS or hour > AUCTION_PATH_CEIL_HHMMSS:
            continue
        ts_hms = int(hour)
        price = _parse_int(row.get("Prpr"), symbol=symbol, field="Prpr")
        buy_qty = _parse_int(row.get("ShnuCnqn"), symbol=symbol, field="ShnuCnqn")
        sell_qty = _parse_int(row.get("SelnCnqn"), symbol=symbol, field="SelnCnqn")
        if buy_qty < 0 or sell_qty < 0:
            raise ValueError(f"auction path {symbol} negative quantity at {hour}: {buy_qty}/{sell_qty}")
        no_fill = buy_qty == 0 and sell_qty == 0
        if hour < DECISION_WINDOW_END_HHMMSS:
            kind = AuctionPathRowKind.INDICATIVE if no_fill else AuctionPathRowKind.UNEXPECTED_TRADE
        elif not no_fill:
            kind = AuctionPathRowKind.AUCTION_FILL
        else:
            continue
        records.append(
            {
                "snapshot_date": str(snapshot_date),
                "symbol": str(symbol),
                "seq": len(records),
                "ts_hms": ts_hms,
                "price": price,
                "buy_qty": buy_qty,
                "sell_qty": sell_qty,
                "kind": kind.value,
                "available_at": available_at,
                "vendor": "dbsec",
            }
        )
    frame = pd.DataFrame(records, columns=list(AUCTION_PATH_COLUMNS))
    if not frame.empty:
        for column in ("seq", "ts_hms", "price", "buy_qty", "sell_qty"):
            frame[column] = frame[column].astype("int64")
    return frame


def auction_path_partition_path(snapshot_date: str) -> Path:
    """Resolve the auction-path partition for one date."""
    month = str(snapshot_date)[:7]
    return Path(settings.HISTORY_DIR) / "auction_path" / month / f"{snapshot_date}.parquet"


def write_auction_path_partition(frame: pd.DataFrame, snapshot_date: str) -> int:
    """Atomically replace one day's partition with ``frame``."""
    if frame is None or len(frame) == 0:
        return 0
    ordered = frame.reindex(columns=list(AUCTION_PATH_COLUMNS)).copy()
    for column in ("seq", "ts_hms", "price", "buy_qty", "sell_qty"):
        ordered[column] = pd.to_numeric(ordered[column], errors="raise").astype("int64")
    atomic_write_parquet(ordered, auction_path_partition_path(str(snapshot_date)))
    return len(ordered)
