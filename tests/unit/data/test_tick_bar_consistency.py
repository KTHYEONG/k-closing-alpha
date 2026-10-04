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
    BAR_VOLUME_CUTOFF_HMS,
    REGULAR_POLICY,
    SESSION_POLICIES,
    TickBarPolicy,
    TickBarRelation,
    classify_tick_bar_volume,
    comparable_bar_volumes,
    summed_tick_volumes,
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
        classify_tick_bar_volume(INTRADAY_SESSION_KRX_AFTERMARKET, 1_191_030, 1_191_032) is TickBarRelation.CONSISTENT
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


def test_cutoff_map_covers_every_policy_session() -> None:
    from src.config.market_session import KRX_AFTERMARKET_HOUR_CEIL, KRX_REGULAR_HOUR_CEIL

    assert set(BAR_VOLUME_CUTOFF_HMS) == set(SESSION_POLICIES)
    assert BAR_VOLUME_CUTOFF_HMS[INTRADAY_SESSION_REGULAR] == int(KRX_REGULAR_HOUR_CEIL)
    assert BAR_VOLUME_CUTOFF_HMS[INTRADAY_SESSION_KRX_AFTERMARKET] == int(KRX_AFTERMARKET_HOUR_CEIL)
    assert BAR_VOLUME_CUTOFF_HMS[INTRADAY_SESSION_NXT_AFTERMARKET] is None


def _bars(rows: list[tuple[str, object, object]]) -> object:
    import pandas as pd

    return pd.DataFrame([{"symbol": s, "ts_hms": t, "volume": v} for s, t, v in rows])


def test_krx_aftermarket_ceiling_bar_excluded() -> None:
    frame = _bars([("A", 160000, 100), ("A", 161000, 50), ("A", 200000, 7)])
    assert comparable_bar_volumes(INTRADAY_SESSION_KRX_AFTERMARKET, frame) == {"A": 150.0}


def test_regular_closing_auction_bar_excluded() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [{"symbol": "A", "ts_hms": 152000, "volume": 10}, {"symbol": "A", "ts_hms": 153000, "volume": 90}]
    )
    assert comparable_bar_volumes(INTRADAY_SESSION_REGULAR, frame) == {"A": 10.0}


def test_nxt_aftermarket_ceiling_bar_kept() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        [{"symbol": "A", "ts_hms": 195900, "volume": 5}, {"symbol": "A", "ts_hms": 200000, "volume": 7}]
    )
    assert comparable_bar_volumes(INTRADAY_SESSION_NXT_AFTERMARKET, frame) == {"A": 12.0}


def test_non_numeric_stamp_and_missing_ts_hms_keep_rows() -> None:
    import pandas as pd

    kept = pd.DataFrame([{"symbol": "A", "ts_hms": None, "volume": 4}])
    assert comparable_bar_volumes(INTRADAY_SESSION_KRX_AFTERMARKET, kept) == {"A": 4.0}
    no_ts = pd.DataFrame([{"symbol": "A", "volume": 7}])
    assert comparable_bar_volumes(INTRADAY_SESSION_KRX_AFTERMARKET, no_ts) == {"A": 7.0}


def test_non_numeric_volume_counts_as_zero() -> None:
    import pandas as pd

    frame = pd.DataFrame([{"symbol": "A", "volume": "x"}, {"symbol": "A", "volume": 5}])
    assert comparable_bar_volumes(INTRADAY_SESSION_REGULAR, frame) == {"A": 5.0}
    ticks = pd.DataFrame([{"symbol": "A", "volume": "x"}, {"symbol": "A", "volume": 5}])
    assert summed_tick_volumes(ticks) == {"A": 5.0}


def test_empty_frame_yields_empty_mapping() -> None:
    import pandas as pd

    assert comparable_bar_volumes(INTRADAY_SESSION_REGULAR, pd.DataFrame({"symbol": [], "volume": []})) == {}
    assert summed_tick_volumes(pd.DataFrame({"symbol": [], "volume": []})) == {}


def test_unknown_session_and_missing_columns_fail_closed() -> None:
    import pandas as pd

    with pytest.raises(ValueError, match="Unknown session"):
        comparable_bar_volumes("night", pd.DataFrame({"symbol": ["A"], "volume": [1]}))
    with pytest.raises(ValueError, match="symbol"):
        comparable_bar_volumes(INTRADAY_SESSION_REGULAR, pd.DataFrame({"symbol": ["A"]}))
    with pytest.raises(ValueError, match="symbol"):
        summed_tick_volumes(pd.DataFrame({"symbol": ["A"]}))


def test_tick_sums_are_not_window_filtered() -> None:
    import pandas as pd

    ticks = pd.DataFrame(
        [
            {"symbol": "A", "ts_hms": 155900, "volume": 3},
            {"symbol": "A", "ts_hms": 200100, "volume": 4},
        ]
    )
    assert summed_tick_volumes(ticks) == {"A": 7.0}


def test_input_frames_are_not_mutated() -> None:
    import pandas as pd

    bars = pd.DataFrame(
        [{"symbol": "A", "ts_hms": 153000, "volume": 9}, {"symbol": "A", "ts_hms": 152000, "volume": 1}]
    )
    before = bars.copy(deep=True)
    comparable_bar_volumes(INTRADAY_SESSION_REGULAR, bars)
    pd.testing.assert_frame_equal(bars, before)
    ticks = pd.DataFrame([{"symbol": "A", "volume": 2}])
    before_ticks = ticks.copy(deep=True)
    summed_tick_volumes(ticks)
    pd.testing.assert_frame_equal(ticks, before_ticks)
