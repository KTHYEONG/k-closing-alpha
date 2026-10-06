"""Unified tick-vs-bar volume consistency contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import pandas as pd

from src.config.market_session import (
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_REGULAR,
    KRX_AFTERMARKET_HOUR_CEIL,
    KRX_REGULAR_HOUR_CEIL,
)


class TickBarRelation(str, Enum):  # noqa: UP042 - spec mandates (str, Enum) for the public contract
    """One symbol's stored tick volume relative to its 1m-bar volume."""

    CONSISTENT = "consistent"
    TICK_SHORT = "tick_short"
    TICK_SURPLUS = "tick_surplus"


@dataclass(frozen=True)
class TickBarPolicy:
    """Per-session tolerances as fractions of bar volume."""

    shortfall_tolerance: float
    surplus_tolerance: float | None


REGULAR_POLICY = TickBarPolicy(shortfall_tolerance=0.01, surplus_tolerance=None)
AFTERMARKET_POLICY = TickBarPolicy(shortfall_tolerance=0.0, surplus_tolerance=0.01)

CERTIFIED_SOURCE_DIFF_TOLERANCE: float = 0.05
"""Largest relative shortfall still attributed to LS-vs-Kiwoom aggregation when the tape total is certified.

Observed 1.4-2.4 % on 2026-10-06; chosen with ~2x margin. Certified shortfalls beyond this are abnormal even for a
vendor disagreement and stay warnings.
"""

CERTIFIED_SOURCE_DIFF_MAX_SHARE: float = 0.05
"""Max share of compared symbols that may be certified source diffs before the class is treated as systemic.

Beyond this share the pattern indicates a vendor aggregation change, not per-symbol noise.
"""


class CertifiedTickShortfall(str, Enum):  # noqa: UP042 - public contract like TickBarRelation
    """One TICK_SHORT symbol's certified-source classification."""

    LOST = "lost"
    SOURCE_DIFF = "source_diff"
    SOURCE_DIFF_EXCESS = "source_diff_excess"


SESSION_POLICIES: Mapping[str, TickBarPolicy] = MappingProxyType(
    {
        INTRADAY_SESSION_REGULAR: REGULAR_POLICY,
        INTRADAY_SESSION_KRX_AFTERMARKET: AFTERMARKET_POLICY,
        INTRADAY_SESSION_NXT_AFTERMARKET: AFTERMARKET_POLICY,
    }
)

# The regular 15:30 bar is the end-labelled closing-auction print outside the continuous tick window.
# The KRX-aftermarket ceiling bar is start-labelled and carries the aftermarket close print, not on the tick tape.
# NXT aftermarket bars are start-labelled and end at 19:59 (measured 2026-10: no 20:00 bar), so no cutoff applies.
BAR_VOLUME_CUTOFF_HMS: Mapping[str, int | None] = MappingProxyType(
    {
        INTRADAY_SESSION_REGULAR: int(KRX_REGULAR_HOUR_CEIL),
        INTRADAY_SESSION_KRX_AFTERMARKET: int(KRX_AFTERMARKET_HOUR_CEIL),
        INTRADAY_SESSION_NXT_AFTERMARKET: None,
    }
)


def comparable_bar_volumes(session: str, bars: pd.DataFrame) -> dict[str, float]:
    """Sum stored 1m-bar volume per symbol over the part of the session that ticks can cover.

    Ticks and bars are compared over the same window only: bars stamped at or after the session's cutoff
    (`BAR_VOLUME_CUTOFF_HMS`) hold auction/close prints that never appear on the tick tape, so counting them
    turns every symbol into a false tick shortfall. Every consumer of the tick-bar contract (daily audit,
    tape sweep) must aggregate bars through this function so they agree on identical partitions.

    Args:
        session: One of the keys of `SESSION_POLICIES`.
        bars: Stored 1m bars with columns `symbol` and `volume`; `ts_hms` (HHMMSS int) is optional.
            Rows whose `ts_hms` is not numeric are kept; without a `ts_hms` column no cutoff applies.

    Returns:
        Mapping symbol (str) -> summed bar volume (float, >= 0 for non-negative inputs); non-numeric
        volumes count as 0. Empty mapping for an empty frame.

    Raises:
        ValueError: Unknown session, or `bars` lacks `symbol` or `volume`.
    """
    if session not in SESSION_POLICIES:
        raise ValueError(f"Unknown session: {session!r}")
    if "symbol" not in bars.columns or "volume" not in bars.columns:
        raise ValueError("bars must carry symbol and volume columns")
    if len(bars) == 0:
        return {}
    frame = bars.loc[:, ["symbol", "volume"] + (["ts_hms"] if "ts_hms" in bars.columns else [])].copy()
    cutoff = BAR_VOLUME_CUTOFF_HMS[session]
    if cutoff is not None and "ts_hms" in frame.columns:
        stamps = pd.to_numeric(frame["ts_hms"], errors="coerce")
        frame = frame.loc[stamps.isna() | (stamps < cutoff)]
        if len(frame) == 0:
            return {}
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
    return {str(k): float(v) for k, v in volume.groupby(frame["symbol"].astype(str)).sum().items()}


def summed_tick_volumes(ticks: pd.DataFrame) -> dict[str, float]:
    """Sum stored tick volume per symbol over every stored row.

    The tick side is never window-filtered here: out-of-window rows are a separate completeness defect that the
    sweep flags on its own, and the audit has always compared the full stored tape.

    Args:
        ticks: Stored ticks with columns `symbol` and `volume`.

    Returns:
        Mapping symbol (str) -> summed tick volume (float); non-numeric volumes count as 0.

    Raises:
        ValueError: `ticks` lacks `symbol` or `volume`.
    """
    if "symbol" not in ticks.columns or "volume" not in ticks.columns:
        raise ValueError("ticks must carry symbol and volume columns")
    if len(ticks) == 0:
        return {}
    volume = pd.to_numeric(ticks["volume"], errors="coerce").fillna(0)
    return {str(k): float(v) for k, v in volume.groupby(ticks["symbol"].astype(str)).sum().items()}


def classify_tick_bar_volume(session: str, bar_volume: float, tick_volume: float) -> TickBarRelation:
    """Classify one symbol's stored tick volume against its 1m-bar volume for a session.

    Bars and ticks are collected independently (vendor-aggregated bars versus the exchange tape), so equality is not structural.
    Regular-session residuals are routine and tolerated up to 1% of bar volume; aftermarket volumes have matched exactly in
    practice, so a shortfall there is treated as lost ticks while a sub-1% tick surplus is tolerated as aggregation noise.
    Tick shortfall is the actionable class: the tape sweep recovers it. Tick surplus beyond tolerance indicates bar-side loss and is
    reported but never targeted by the sweep.

    Args:
        session: One of the keys of `SESSION_POLICIES`.
        bar_volume: Summed 1m-bar volume for the symbol over the session window (shares, >= 0).
        tick_volume: Summed tick volume for the symbol over the session window (shares, >= 0).

    Returns:
        `TICK_SHORT` when `bar_volume - tick_volume` exceeds `shortfall_tolerance * bar_volume`;
        `TICK_SURPLUS` when the policy has a surplus tolerance and the surplus exceeds it (a positive tick volume against zero bar
        volume is a surplus); otherwise `CONSISTENT`.

    Raises:
        ValueError: Unknown session, or a negative or non-finite volume.
    """
    policy = SESSION_POLICIES.get(session)
    if policy is None:
        raise ValueError(f"Unknown session: {session!r}")
    for name, value in (("bar_volume", bar_volume), ("tick_volume", tick_volume)):
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite: {value!r}")
        if float(value) < 0:
            raise ValueError(f"{name} must be >= 0: {value!r}")
    bar = float(bar_volume)
    tick = float(tick_volume)
    if bar == 0.0:
        if tick == 0.0:
            return TickBarRelation.CONSISTENT
        if policy.surplus_tolerance is None:
            return TickBarRelation.CONSISTENT
        return TickBarRelation.TICK_SURPLUS

    if tick < bar:
        if (bar - tick) > policy.shortfall_tolerance * bar:
            return TickBarRelation.TICK_SHORT
        return TickBarRelation.CONSISTENT
    if tick > bar:
        if policy.surplus_tolerance is None:
            return TickBarRelation.CONSISTENT
        if (tick - bar) > float(policy.surplus_tolerance) * bar:
            return TickBarRelation.TICK_SURPLUS
        return TickBarRelation.CONSISTENT
    return TickBarRelation.CONSISTENT


def classify_certified_tick_shortfall(
    session: str, bar_volume: float, tick_volume: float, *, tape_certified: bool
) -> CertifiedTickShortfall | None:
    """Classify a symbol that `classify_tick_bar_volume` rates TICK_SHORT.

    Why: a shortfall that the Kiwoom tape's own vendor total certifies as complete cannot be repaired; reporting it as lost ticks
    makes the audit warn forever. Uncertified shortfalls stay actionable.

    Returns:
        None when the symbol is not TICK_SHORT for the session policy; LOST when not tape-certified; SOURCE_DIFF when certified and
        (bar - tick) / bar <= CERTIFIED_SOURCE_DIFF_TOLERANCE; SOURCE_DIFF_EXCESS otherwise.

    Raises:
        ValueError: Unknown session, a negative or non-finite volume (same contract as classify_tick_bar_volume).
    """
    relation = classify_tick_bar_volume(session, bar_volume, tick_volume)
    if relation is not TickBarRelation.TICK_SHORT:
        return None
    if session != INTRADAY_SESSION_REGULAR or not tape_certified:
        return CertifiedTickShortfall.LOST
    bar = float(bar_volume)
    tick = float(tick_volume)
    relative_shortfall = (bar - tick) / bar if bar > 0 else 0.0
    if relative_shortfall <= CERTIFIED_SOURCE_DIFF_TOLERANCE:
        return CertifiedTickShortfall.SOURCE_DIFF
    return CertifiedTickShortfall.SOURCE_DIFF_EXCESS
