"""Invariant scenarios for the Toss NXT overnight fetch layer."""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from src.backfill.intraday.toss_overnight import (
    TOSS_CANDLE_MAX_COUNT,
    TOSS_PAIR_MAX_CALLS,
    fetch_toss_overnight_pair,
    fetch_toss_window,
    session_end_label_window,
    toss_session_frame,
)
from src.config.market_session import KRX_AFTERMARKET_START_DATE, NXT_START_DATE
from src.data.capture_store import CaptureStore
from tests.unit.backfill.toss_fakes import FakeToss, always, candle, day_series

T = "2025-06-12"
T1 = "2025-06-13"
SYM = "005930"


def _store(tmp_path) -> CaptureStore:
    return CaptureStore(tmp_path / "capture")


def _dense(evening=None, premarket=None, *, regular_on_t1: bool = True) -> list[dict]:
    return (
        day_series(T, evening=evening if evening is not None else always(10))
        + day_series(T1, regular=regular_on_t1, premarket=premarket if premarket is not None else always(7))
    )


def _pair(toss, tmp_path, entry=T, nxt=T1, run_id="r-pair"):
    return asyncio.run(fetch_toss_overnight_pair(toss, None, SYM, entry, nxt, store=_store(tmp_path), run_id=run_id))


def test_two_calls_cover_evening_and_premarket(tmp_path) -> None:
    toss = FakeToss({SYM: _dense()})
    result = _pair(toss, tmp_path)
    evening, evening_entry = result.evening
    premarket, premarket_entry = result.premarket
    assert result.calls == 2 and len(toss.calls) == 2
    assert result.failed is False
    assert evening_entry.status.value == "COMPLETE" and premarket_entry.status.value == "COMPLETE"
    assert len(evening) == 260 and len(premarket) == 50
    assert evening_entry.venue == "NXT" and evening_entry.session == "nxt_aftermarket"
    assert premarket_entry.session == "nxt_premarket"


def test_labels_shift_one_minute_earlier() -> None:
    frame = toss_session_frame(day_series(T, evening=always(3)), SYM, T, "nxt_aftermarket")
    pre = toss_session_frame(day_series(T1, premarket=always(3)), SYM, T1, "nxt_premarket")
    assert (int(frame["ts_hms"].min()), int(frame["ts_hms"].max())) == (154000, 195900)
    assert (int(pre["ts_hms"].min()), int(pre["ts_hms"].max())) == (80000, 84900)
    assert frame["vendor"].unique().tolist() == ["toss"]
    assert bool(frame["has_trade"].all())


def test_zero_volume_placeholders_are_dropped() -> None:
    frame = toss_session_frame(
        day_series(T, evening=lambda label: 4 if label.endswith(("0", "5")) else 0), SYM, T, "nxt_aftermarket"
    )
    assert len(frame) > 0
    assert bool((frame["volume"] > 0).all())
    assert (frame["value_krw"] == frame["close"] * frame["volume"]).all()
    assert frame["ts_hms"].is_unique


def test_listed_but_never_traded_is_no_trades(tmp_path) -> None:
    toss = FakeToss({SYM: _dense(evening=always(0), premarket=always(0))})
    result = _pair(toss, tmp_path)
    for (frame, entry), session in ((result.evening, "nxt_aftermarket"), (result.premarket, "nxt_premarket")):
        assert entry.status.value == "NO_TRADES" and entry.session == session
        assert frame.empty and len(entry.raw_refs) > 0


def test_not_on_nxt_is_one_call_and_not_applicable(tmp_path) -> None:
    toss = FakeToss({SYM: day_series(T) + day_series(T1, premarket=None)})
    result = _pair(toss, tmp_path)
    assert result.calls == 1
    for frame, entry in (result.evening, result.premarket):
        assert entry.status.value == "NOT_APPLICABLE" and entry.reason == "nxt_not_listed"
        assert frame.empty and len(entry.raw_refs) == 1


def test_listed_only_from_next_day(tmp_path) -> None:
    toss = FakeToss({SYM: day_series(T) + day_series(T1, premarket=always(6))})
    result = _pair(toss, tmp_path)
    assert result.evening[1].status.value == "NOT_APPLICABLE"
    assert result.premarket[1].status.value == "COMPLETE" and len(result.premarket[0]) == 50


def test_empty_successful_envelope_is_not_applicable(tmp_path) -> None:
    toss = FakeToss({})
    result = _pair(toss, tmp_path)
    for _, entry in (result.evening, result.premarket):
        assert entry.status.value == "NOT_APPLICABLE" and entry.reason == "toss_empty"
        assert len(entry.raw_refs) == 1
    assert result.failed is False


def test_error_envelope_is_never_an_empty_page(tmp_path) -> None:
    toss = FakeToss({SYM: _dense()})
    toss.script[0] = {"requestId": "x", "error": {"code": "rate-limit", "message": "slow down"}}
    result = _pair(toss, tmp_path)
    for frame, entry in (result.evening, result.premarket):
        assert entry.status.value == "FAILED" and entry.reason == "toss:vendor_error:rate-limit"
        assert frame.empty and len(entry.raw_refs) == 1
    assert result.failed is True


def test_error_envelope_without_code_is_unknown_vendor_error(tmp_path) -> None:
    toss = FakeToss({SYM: _dense()})
    toss.script[0] = {"error": "boom"}
    result = _pair(toss, tmp_path)
    assert result.evening[1].reason == "toss:vendor_error:unknown"


def test_transport_exception_is_contained(tmp_path) -> None:
    toss = FakeToss({SYM: _dense()})
    toss.script[1] = RuntimeError("socket closed")
    result = _pair(toss, tmp_path)
    assert result.failed is True
    assert result.evening[1].status.value == "FAILED"
    assert result.evening[1].reason == "toss:transport:RuntimeError"
    assert len(result.evening[1].raw_refs) == 2  # page 0 body and the failed page 1 placeholder
    assert result.evening[0].empty and result.premarket[0].empty


def test_unproven_traversal_is_partial_after_call_budget(tmp_path) -> None:
    toss = FakeToss({SYM: _dense()})
    toss.fixed_page = [candle(T1, f"{hour:02d}:{minute:02d}", 3) for hour in range(9, 13) for minute in range(60)][:200]
    result = _pair(toss, tmp_path)
    assert result.calls == TOSS_PAIR_MAX_CALLS
    assert result.evening[1].status.value == "PARTIAL" and result.evening[1].reason == "incomplete_window"
    assert result.premarket[1].status.value == "PARTIAL"
    assert result.failed is False


def test_cursor_overlap_is_deduplicated() -> None:
    toss = FakeToss({SYM: _dense()})
    window = asyncio.run(
        fetch_toss_window(toss, None, SYM, before=f"{T1}T08:50:00.000+09:00", stop_day=T, stop_label="15:41")
    )
    stamps = [c["timestamp"] for c in window.candles]
    assert len(stamps) == len(set(stamps)) and stamps == sorted(stamps)
    assert window.reached_stop is True and window.error is None and window.calls == 2


def test_second_call_requests_only_what_is_missing(tmp_path) -> None:
    toss = FakeToss({SYM: _dense()})
    _pair(toss, tmp_path)
    assert toss.calls[0]["count"] == TOSS_CANDLE_MAX_COUNT
    # 첫 페이지는 프리 50봉 + 저녁 150봉(20:00..17:31): 17:31 -> 15:41 은 110분 + 겹침 1 + 여유 1
    assert toss.calls[1]["count"] == 112


def test_every_call_is_unadjusted_one_minute_and_bounded(tmp_path) -> None:
    toss = FakeToss({SYM: _dense()})
    _pair(toss, tmp_path)
    assert toss.calls and all(
        c["adjusted"] is False and c["interval"] == "1m" and 1 <= c["count"] <= TOSS_CANDLE_MAX_COUNT for c in toss.calls
    )


def test_consolidated_regime_entry_days_are_refused(tmp_path) -> None:
    toss = FakeToss({SYM: _dense()})
    with pytest.raises(ValueError, match="consolidated"):
        _pair(toss, tmp_path, entry=KRX_AFTERMARKET_START_DATE, nxt="2026-09-15")
    assert toss.calls == []


def test_entry_days_before_nxt_start_are_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="consolidated"):
        _pair(FakeToss({}), tmp_path, entry="2025-03-03", nxt="2025-03-04")
    assert NXT_START_DATE == "2025-03-04"


def test_malformed_pair_arguments_are_refused(tmp_path) -> None:
    toss = FakeToss({})
    with pytest.raises(ValueError, match="entry_day"):
        _pair(toss, tmp_path, entry="not-a-date")
    with pytest.raises(ValueError, match="next_day"):
        _pair(toss, tmp_path, nxt="not-a-date")
    with pytest.raises(ValueError, match="after entry_day"):
        _pair(toss, tmp_path, nxt=T)
    with pytest.raises(ValueError, match="run_id"):
        _pair(toss, tmp_path, run_id=" ")


def test_raw_evidence_is_attached_per_call(tmp_path) -> None:
    toss = FakeToss({SYM: _dense()})
    result = _pair(toss, tmp_path)
    assert len(result.evening[1].raw_refs) == 2
    assert result.evening[1].raw_refs == result.premarket[1].raw_refs


def test_session_labels_derive_from_the_constants() -> None:
    assert session_end_label_window("nxt_aftermarket") == ("15:41", "20:00")
    assert session_end_label_window("nxt_premarket") == ("08:01", "08:50")
    with pytest.raises(ValueError, match="Unknown session"):
        session_end_label_window("regular")


def test_session_frame_validates_inputs() -> None:
    with pytest.raises(ValueError, match="Unknown session"):
        toss_session_frame([], SYM, T, "regular")
    with pytest.raises(ValueError, match="Invalid session_date"):
        toss_session_frame([], SYM, "bad", "nxt_aftermarket")
    with pytest.raises(ValueError, match="Missing required Toss fields"):
        toss_session_frame([{"timestamp": f"{T}T16:00:00.000+09:00"}], SYM, T, "nxt_aftermarket")
    with pytest.raises(ValueError, match="Missing required Toss fields"):
        toss_session_frame(["oops"], SYM, T, "nxt_aftermarket")  # type: ignore[list-item]
    bad = candle(T, "16:00", 1)
    bad["volume"] = "many"
    with pytest.raises(ValueError, match="Invalid Toss volume"):
        toss_session_frame([bad], SYM, T, "nxt_aftermarket")
    assert toss_session_frame([], SYM, T, "nxt_aftermarket").empty


def test_session_frame_ignores_other_days_and_out_of_window_bars() -> None:
    other_day = candle("2025-06-11", "16:00", 9)
    outside = candle(T, "15:35", 9)
    kept = candle(T, "16:00", 9)
    frame = toss_session_frame([other_day, outside, kept], SYM, T, "nxt_aftermarket")
    assert frame["ts_hms"].tolist() == [155900]


def test_window_validates_arguments_and_reports_empty_pages() -> None:
    toss = FakeToss({})
    with pytest.raises(ValueError, match="before"):
        asyncio.run(fetch_toss_window(toss, None, SYM, before="nope", stop_day=T, stop_label="15:41"))
    with pytest.raises(ValueError, match="before"):
        asyncio.run(fetch_toss_window(toss, None, SYM, before=f"{T1}T08:50:00", stop_day=T, stop_label="15:41"))
    with pytest.raises(ValueError, match="stop_day"):
        asyncio.run(fetch_toss_window(toss, None, SYM, before=f"{T1}T08:50:00+09:00", stop_day="x", stop_label="15:41"))
    with pytest.raises(ValueError, match="stop_label"):
        asyncio.run(fetch_toss_window(toss, None, SYM, before=f"{T1}T08:50:00+09:00", stop_day=T, stop_label="1541"))
    empty = asyncio.run(
        fetch_toss_window(toss, None, SYM, before=f"{T1}T08:50:00+09:00", stop_day=T, stop_label="15:41")
    )
    assert empty.candles == () and empty.error is None and empty.reached_stop is False


def test_window_observer_sees_every_page_including_errors() -> None:
    seen: list[tuple[object, int]] = []
    toss = FakeToss({SYM: _dense()})
    toss.script[1] = {"error": {"code": "invalid-request"}}
    window = asyncio.run(
        fetch_toss_window(
            toss, None, SYM, before=f"{T1}T08:50:00.000+09:00", stop_day=T, stop_label="15:41",
            on_page=lambda payload, started, received, index: seen.append((payload, index)),
        )
    )
    assert [index for _, index in seen] == [0, 1]
    assert window.error == "vendor_error:invalid-request"
    assert window.reached_stop is False


def test_frame_dtypes_match_the_canonical_schema() -> None:
    from src.data.intraday_schema import CANONICAL_BAR_COLUMNS

    frame = toss_session_frame(day_series(T, evening=always(2)), SYM, T, "nxt_aftermarket")
    assert list(frame.columns) == list(CANONICAL_BAR_COLUMNS)
    assert isinstance(frame, pd.DataFrame)


def test_malformed_candle_timestamps_are_refused() -> None:
    bad = candle(T, "16:00", 3)
    bad["timestamp"] = "garbage"
    with pytest.raises(ValueError, match="Invalid candle timestamp"):
        toss_session_frame([bad], SYM, T, "nxt_aftermarket")


def test_window_stops_when_the_page_crosses_before_the_stop_day() -> None:
    toss = FakeToss({SYM: day_series(T)})
    window = asyncio.run(
        fetch_toss_window(toss, None, SYM, before=f"{T}T15:30:00.000+09:00", stop_day=T1, stop_label="08:01")
    )
    assert window.reached_stop is True and window.calls == 1 and window.error is None


def test_exhausted_vendor_history_ends_the_walk_without_error() -> None:
    toss = FakeToss({SYM: day_series(T)[:50]})
    window = asyncio.run(
        fetch_toss_window(toss, None, SYM, before=f"{T}T15:30:00.000+09:00", stop_day="2025-06-01", stop_label="08:01")
    )
    assert window.reached_stop is False and window.error is None and window.calls == 1 and len(window.candles) == 50


def test_malformed_vendor_bars_are_dropped_at_merge_time(tmp_path) -> None:
    toss = FakeToss({SYM: _dense()})
    good = _dense()[-5:]
    toss.fixed_page = ["garbage", {"foo": 1}, {"timestamp": "not-a-time", "volume": "1"}, *good]
    window = asyncio.run(
        fetch_toss_window(toss, None, SYM, before=f"{T1}T08:50:00.000+09:00", stop_day=T1, stop_label="08:01")
    )
    assert len(window.candles) == len(good) and window.error is None
