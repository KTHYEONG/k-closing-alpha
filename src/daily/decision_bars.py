"""Decision-session 1m bars from Toss for the 15:20 PIT-CORE path features."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

from src import settings
from src.api.toss.client import TossApiClient
from src.backfill.intraday.toss_regular import (
    TOSS_REGULAR_FIRST_LABEL_HHMMSS,
    _extract_candles,
    _parse_toss_label,
    _redacted_error,
)
from src.config.market_session import DECISION_WINDOW_START_HHMMSS
from src.data.capture_contracts import SEOUL
from src.data.io_utils import atomic_write_parquet
from src.ml.pit_core_features import bar_end_minutes

logger = logging.getLogger(__name__)

DECISION_BARS_DIRNAME: str = "decision_bars"
TOSS_DECISION_PAGE_COUNT: int = 200
TOSS_DECISION_MAX_PAGES: int = 3
DECISION_BARS_COLUMNS: tuple[str, ...] = (
    "symbol", "end_min", "open", "high", "low", "close", "volume", "vendor", "fetched_at",
)  # fmt: skip

_FIRST_PAGE_BEFORE_HMS: str = "15:21:00"
_TOSS_VENDOR: str = "toss"


def _seoul_now() -> datetime:
    return datetime.now(SEOUL)


def _to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=list(DECISION_BARS_COLUMNS))
    return frame.astype(
        {
            "symbol": str,
            "end_min": "int64",
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "vendor": str,
        }
    )


async def _walk_symbol(
    client: Any, session: aiohttp.ClientSession, code: str, decision_date: pd.Timestamp
) -> list[dict[str, Any]]:
    day = decision_date.strftime("%Y-%m-%d")
    before = f"{day}T{_FIRST_PAGE_BEFORE_HMS}+09:00"
    raw: list[dict[str, Any]] = []
    for _ in range(TOSS_DECISION_MAX_PAGES):
        payload = await client.get_candles(
            session, code, interval="1m", count=TOSS_DECISION_PAGE_COUNT, before=before, adjusted=False
        )
        candles = _extract_candles(payload)
        if candles is None:
            raise ValueError("toss candles payload is malformed")
        if not candles:
            break
        labelled = [(c, _parse_toss_label(str(c.get("timestamp", "")))) for c in candles]
        for candle, (_, label_day, hhmmss) in labelled:
            if label_day == day and TOSS_REGULAR_FIRST_LABEL_HHMMSS <= hhmmss <= DECISION_WINDOW_START_HHMMSS:
                raw.append({**candle, "hhmmss": int(hhmmss)})
        if any(label_day < day for _, (_, label_day, _) in labelled):
            break
        oldest_candle, _ = min(labelled, key=lambda item: item[1][0])
        before = str(oldest_candle["timestamp"])
    return raw


def _symbol_frame(code: str, raw: list[dict[str, Any]], fetched_at: datetime) -> pd.DataFrame:
    if not raw:
        return _to_frame([])
    hhmmss = [int(r["hhmmss"]) for r in raw]
    end_min = bar_end_minutes(pd.Series(hhmmss, dtype="int64"), pd.Series([_TOSS_VENDOR] * len(raw)))
    frame = pd.DataFrame(
        {
            "symbol": code,
            "end_min": end_min,
            "open": [float(r["openPrice"]) for r in raw],
            "high": [float(r["highPrice"]) for r in raw],
            "low": [float(r["lowPrice"]) for r in raw],
            "close": [float(r["closePrice"]) for r in raw],
            "volume": [float(r["volume"]) for r in raw],
            "vendor": _TOSS_VENDOR,
            "fetched_at": fetched_at,
        }
    )
    return frame.drop_duplicates(subset="end_min", keep="first").sort_values("end_min").reset_index(drop=True)


async def _fetch_one(
    client: Any,
    session: aiohttp.ClientSession,
    symbol: str,
    decision_date: pd.Timestamp,
    fetched_at: datetime,
) -> pd.DataFrame:
    code = str(symbol).zfill(6)
    try:
        raw = await _walk_symbol(client, session, code, decision_date)
    except Exception as exc:
        logger.warning(
            "[DATA] stage=decision_bars symbol=%s status=FAILED reason=%s", code, _redacted_error(exc)
        )
        return _to_frame([])
    return _symbol_frame(code, raw, fetched_at)


async def fetch_decision_bars(
    client: Any,
    session: aiohttp.ClientSession,
    symbols: Sequence[str],
    decision_date: pd.Timestamp,
    *,
    now_fn: Callable[[], datetime] | None = None,
) -> pd.DataFrame:
    """Return decision-date regular-session 1m bars ending at or before 15:20 for each symbol.

    Raises:
        ValueError: When the clock is not on decision_date or before 15:20:00 (the 15:20 bar is not final).
    """
    now = (now_fn or _seoul_now)()
    if now.date() != pd.Timestamp(decision_date).date():
        raise ValueError(f"clock {now.date()} is not on decision date {pd.Timestamp(decision_date).date()}")
    if now.strftime("%H%M%S") < DECISION_WINDOW_START_HHMMSS:
        raise ValueError("15:20 bar is not final yet")
    frames = await asyncio.gather(*(_fetch_one(client, session, s, decision_date, now) for s in symbols))
    if not frames:
        return _to_frame([])
    return pd.concat(frames, ignore_index=True)


def persist_decision_bars(decision_date: pd.Timestamp, bars: pd.DataFrame, *, root: Path | None = None) -> Path:
    """Write the decision-day bars as an audit parquet, replacing any earlier fetch of the same day."""
    directory = root or settings.HISTORY_DIR / DECISION_BARS_DIRNAME
    path = Path(directory) / f"{pd.Timestamp(decision_date).strftime('%Y-%m-%d')}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(bars, path)
    return path


def build_decision_bars_client() -> Any:
    """Return a Toss client for decision-session bars; raises ValueError when no app key is configured."""
    client = TossApiClient()
    if not client.app_key:
        raise ValueError("Toss app key is not configured")
    return client
