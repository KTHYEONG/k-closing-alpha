"""Shared Toss stand-ins for the NXT overnight backfill tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.data.capture_contracts import SEOUL

KST = "+09:00"


def _ts(day: str, hhmm: str) -> str:
    return f"{day}T{hhmm}:00.000{KST}"


def _labels(first: str, last: str) -> list[str]:
    start = datetime.strptime(f"2000-01-01 {first}", "%Y-%m-%d %H:%M")
    end = datetime.strptime(f"2000-01-01 {last}", "%Y-%m-%d %H:%M")
    out = []
    while start <= end:
        out.append(start.strftime("%H:%M"))
        start += timedelta(minutes=1)
    return out


def candle(day: str, hhmm: str, volume: int, close: int = 1000) -> dict[str, Any]:
    return {
        "timestamp": _ts(day, hhmm), "openPrice": str(close), "highPrice": str(close),
        "lowPrice": str(close), "closePrice": str(close), "volume": str(volume), "currency": "KRW",
    }


REGULAR = ("09:01", "15:30")
EVENING = ("15:41", "20:00")
PREMARKET = ("08:01", "08:50")


def day_series(
    day: str,
    *,
    regular: bool = True,
    evening: Any = None,
    premarket: Any = None,
) -> list[dict[str, Any]]:
    """Dense Toss candles of one day. `evening`/`premarket` are None (absent) or a callable label->volume."""
    out: list[dict[str, Any]] = []
    if premarket is not None:
        out += [candle(day, label, premarket(label), close=1100) for label in _labels(*PREMARKET)]
    if regular:
        out += [candle(day, label, 5) for label in _labels(*REGULAR)]
    if evening is not None:
        out += [candle(day, label, evening(label), close=1200) for label in _labels(*EVENING)]
    return out


def always(volume: int):
    return lambda _label: volume


class FakeToss:
    """Newest-first, inclusive-`before` Toss candles server over per-symbol dense series."""

    def __init__(self, series: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.series = series or {}
        self.calls: list[dict[str, Any]] = []
        self.script: dict[int, Any] = {}
        self.fixed_page: list[dict[str, Any]] | None = None
        self.fail_symbols: set[str] = set()
        self.on_call: Any = None

    async def get_candles(self, session, symbol, *, interval="1m", count=100, before=None, adjusted=None):
        self.calls.append({"symbol": symbol, "interval": interval, "count": count, "before": before, "adjusted": adjusted})
        if self.on_call is not None:
            self.on_call(self)
        if symbol in self.fail_symbols:
            return {"error": {"code": "internal", "message": "boom"}}
        scripted = self.script.get(len(self.calls) - 1)
        if isinstance(scripted, Exception):
            raise scripted
        if scripted is not None:
            return scripted
        if self.fixed_page is not None:
            return {"result": {"candles": list(self.fixed_page)}}
        bars = self.series.get(symbol, [])
        cutoff = datetime.fromisoformat(before) if before else None
        eligible = [b for b in bars if cutoff is None or datetime.fromisoformat(b["timestamp"]) <= cutoff]
        eligible.sort(key=lambda b: b["timestamp"], reverse=True)
        return {"result": {"candles": eligible[: int(count)]}}


def reference_frame(symbol: str, day: str, evening: Any, *, vendor: str = "kiwoom") -> pd.DataFrame:
    """Stored (start-labeled) NXT evening rows that a primary vendor would have written for the same minutes."""
    rows = []
    for label in _labels(*EVENING):
        volume = evening(label)
        if volume <= 0:
            continue
        hour, minute = int(label[:2]), int(label[3:])
        total = hour * 60 + minute - 1
        rows.append({
            "snapshot_date": day, "symbol": symbol, "ts_hms": (total // 60) * 10000 + (total % 60) * 100,
            "open": 1200, "high": 1200, "low": 1200, "close": 1200, "volume": volume,
            "value_krw": 1200 * volume, "has_trade": True, "vendor": vendor,
        })
    return pd.DataFrame(rows)


def kst_now() -> datetime:
    return datetime.now(SEOUL)
