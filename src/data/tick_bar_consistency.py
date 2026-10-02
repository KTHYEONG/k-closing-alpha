"""Unified tick-vs-bar volume consistency contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from src.config.market_session import (
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_REGULAR,
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
SESSION_POLICIES: Mapping[str, TickBarPolicy] = MappingProxyType(
    {
        INTRADAY_SESSION_REGULAR: REGULAR_POLICY,
        INTRADAY_SESSION_KRX_AFTERMARKET: AFTERMARKET_POLICY,
        INTRADAY_SESSION_NXT_AFTERMARKET: AFTERMARKET_POLICY,
    }
)


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
