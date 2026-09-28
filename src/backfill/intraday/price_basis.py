"""Price-basis reference for historical intraday backfill.

KIS ``FHKST03010230`` (historical minute chart) returns corporate-action adjusted bars with no raw
option, and its adjustment is not exactly invertible from the panel factor (measured: 40% of adjusted
symbol-days land off the KRX tick grid after inversion). The store is kept on the raw basis instead:
a symbol-day is taken from KIS only when no corporate action happened after it, and otherwise from a
raw-basis vendor. This module answers the first question from the price panel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class PriceReference:
    """Point-in-time adjustment state per (YYYY-MM-DD, 6-digit symbol) from the price panel.

    ``close_raw`` is the exchange's official unadjusted close and never changes; ``close`` is the
    panel's adjusted close, rebased on every later corporate action exactly like KIS historical charts.
    A symbol-day is adjusted when the two differ; when they are equal no corporate action happened
    after that day, so a KIS historical fetch of it is already on the raw basis.
    """

    _adjusted: frozenset[tuple[str, str]] = field(default_factory=frozenset, repr=False)
    _known: frozenset[tuple[str, str]] = field(default_factory=frozenset, repr=False)

    @classmethod
    def from_price_history(cls, frame: pd.DataFrame) -> PriceReference:
        """Build from panel rows with columns date, symbol, close, close_raw.

        Raises:
            ValueError: Missing columns or duplicate (date, symbol) keys.
        """
        required = ("date", "symbol", "close", "close_raw")
        missing = [c for c in required if frame is None or c not in frame.columns]
        if missing:
            raise ValueError(f"Price history missing columns: {missing}")
        days = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d").astype(str).tolist()
        symbols = frame["symbol"].astype(str).str.zfill(6).tolist()
        keys = list(zip(days, symbols))
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate (date, symbol) keys in price history")
        closes = pd.to_numeric(frame["close"], errors="coerce").tolist()
        raws = pd.to_numeric(frame["close_raw"], errors="coerce").tolist()
        adjusted: set[tuple[str, str]] = set()
        known: set[tuple[str, str]] = set()
        for key, close, raw in zip(keys, closes, raws):
            if not (math.isfinite(close) and math.isfinite(raw) and close > 0 and raw > 0):
                continue
            known.add(key)
            if round(close) != round(raw):
                adjusted.add(key)
        return cls(_adjusted=frozenset(adjusted), _known=frozenset(known))

    def is_adjusted(self, snapshot_date: str, symbol: str) -> bool:
        """True when the panel's adjusted close differs from the raw close for that symbol-day."""
        return (str(snapshot_date), str(symbol).zfill(6)) in self._adjusted

    def is_known(self, snapshot_date: str, symbol: str) -> bool:
        """True when the panel has a valid close pair for that symbol-day (the basis is decidable)."""
        return (str(snapshot_date), str(symbol).zfill(6)) in self._known
