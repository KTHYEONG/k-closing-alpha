"""Invariant guards for the unified tick-vs-bar volume contract."""

from __future__ import annotations

import math

import pytest

from src.backfill.intraday.tape_harvest import TAPE_SESSIONS
from src.config.market_session import (
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_REGULAR,
)
from src.data.tick_bar_consistency import (
    AFTERMARKET_POLICY,
    REGULAR_POLICY,
    SESSION_POLICIES,
    TickBarPolicy,
    TickBarRelation,
    classify_tick_bar_volume,
)


def test_regular_tolerance_boundary() -> None:
    assert classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, 10_000, 9_900) is TickBarRelation.CONSISTENT
    assert classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, 10_000, 9_899) is TickBarRelation.TICK_SHORT


def test_regular_surplus_never_classified() -> None:
    assert classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, 1_000, 5_000) is TickBarRelation.CONSISTENT


def test_aftermarket_strict_shortfall() -> None:
    for session in (INTRADAY_SESSION_KRX_AFTERMARKET, INTRADAY_SESSION_NXT_AFTERMARKET):
        assert classify_tick_bar_volume(session, 1_000_000, 999_999) is TickBarRelation.TICK_SHORT


def test_aftermarket_sub_percent_surplus_tolerated() -> None:
    assert (
        classify_tick_bar_volume(INTRADAY_SESSION_KRX_AFTERMARKET, 1_191_030, 1_191_032)
        is TickBarRelation.CONSISTENT
    )


def test_aftermarket_material_surplus_flagged() -> None:
    assert classify_tick_bar_volume(INTRADAY_SESSION_NXT_AFTERMARKET, 1_000, 1_011) is TickBarRelation.TICK_SURPLUS
    assert classify_tick_bar_volume(INTRADAY_SESSION_NXT_AFTERMARKET, 1_000, 1_010) is TickBarRelation.CONSISTENT


def test_zero_bar_volume() -> None:
    assert classify_tick_bar_volume(INTRADAY_SESSION_KRX_AFTERMARKET, 0, 0) is TickBarRelation.CONSISTENT
    assert classify_tick_bar_volume(INTRADAY_SESSION_KRX_AFTERMARKET, 0, 500) is TickBarRelation.TICK_SURPLUS
    assert classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, 0, 500) is TickBarRelation.CONSISTENT


def test_zero_tick_volume_with_bars() -> None:
    for session in (
        INTRADAY_SESSION_REGULAR,
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
    ):
        assert classify_tick_bar_volume(session, 100, 0) is TickBarRelation.TICK_SHORT


def test_exact_match_is_consistent() -> None:
    assert classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, 1_000, 1_000) is TickBarRelation.CONSISTENT
    assert classify_tick_bar_volume(INTRADAY_SESSION_KRX_AFTERMARKET, 1_000, 1_000) is TickBarRelation.CONSISTENT


def test_invalid_input() -> None:
    with pytest.raises(ValueError, match="Unknown session"):
        classify_tick_bar_volume("bogus", 100, 100)
    with pytest.raises(ValueError, match="bar_volume"):
        classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, -1, 0)
    with pytest.raises(ValueError, match="tick_volume"):
        classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, 100, -5)
    with pytest.raises(ValueError, match="finite"):
        classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, math.nan, 0)
    with pytest.raises(ValueError, match="finite"):
        classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, 100, math.inf)
    with pytest.raises(ValueError, match="finite"):
        classify_tick_bar_volume(INTRADAY_SESSION_REGULAR, "100", 100)  # type: ignore[arg-type]


def test_policy_coverage() -> None:
    assert TickBarPolicy(shortfall_tolerance=0.01, surplus_tolerance=None) == REGULAR_POLICY
    assert TickBarPolicy(shortfall_tolerance=0.0, surplus_tolerance=0.01) == AFTERMARKET_POLICY
    for spec in TAPE_SESSIONS:
        assert spec.session in SESSION_POLICIES
    with pytest.raises(TypeError):
        SESSION_POLICIES["regular"] = REGULAR_POLICY  # type: ignore[index]
