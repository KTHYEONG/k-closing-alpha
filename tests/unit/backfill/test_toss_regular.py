from __future__ import annotations

import asyncio
import math
from datetime import datetime

import pandas as pd

from src.backfill.intraday.toss_regular import (
    TOSS_CANDLES_PAGE_MAX,
    TOSS_REGULAR_EXPECTED_BARS,
    acquire_toss_regular_bars,
    probe_toss_retention_floor,
    toss_basis_verdict,
)
from src.config.collection import CollectionSettings
from src.data.capture_contracts import CaptureStatus, SEOUL
from src.data.capture_store import CaptureStore

_DAY = "2024-02-28"
_PREV = "2024-02-27"


def _hhmmss(minute_of_day: int) -> str:
    return f"{minute_of_day // 60:02d}{minute_of_day % 60:02d}00"


def _candle(day: str, hhmmss: str, volume: int = 100, price: str = "70000") -> dict:
    return {
        "timestamp": f"{day}T{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}.000+09:00",
        "openPrice": price,
        "highPrice": price,
        "lowPrice": price,
        "closePrice": price,
        "volume": str(volume),
        "currency": "KRW",
    }


def _regular_grid(day: str, volume: int = 100) -> list[dict]:
    out = [_candle(day, _hhmmss(minute), volume=volume) for minute in range(9 * 60 + 1, 15 * 60 + 31)]
    assert len(out) == 390
    return out


def _profile(tmp_path, routes=None):
    if routes is None:
        routes = {"toss:toss-candles": "KRX"}
    return CollectionSettings(COLLECTION_ROOT=tmp_path / "capture", COLLECTION_VERIFIED_CHART_ROUTES=dict(routes))


def _store(tmp_path) -> CaptureStore:
    return CaptureStore(tmp_path / "capture")


class _GridToss:
    def __init__(self, all_newest_first: list[dict], error=None, explode_on: set[int] | None = None):
        self._all = list(all_newest_first)
        self._error = error
        self._explode = set(explode_on or set())
        self.calls: list[dict] = []

    async def get_candles(self, session, symbol, *, interval="1m", count=200, before=None, adjusted=None):
        idx = len(self.calls)
        self.calls.append({"symbol": symbol, "interval": interval, "count": count, "before": before, "adjusted": adjusted})
        if idx in self._explode:
            raise ConnectionError("boom")
        if self._error is not None:
            return self._error
        try:
            bound = datetime.fromisoformat(str(before)) if before else None
        except Exception:
            bound = None
        out = []
        for candle in self._all:
            try:
                current = datetime.fromisoformat(str(candle.get("timestamp", "")))
            except Exception:
                out.append(candle)
                if len(out) >= int(count):
                    break
                continue
            if bound is None or current <= bound:
                out.append(candle)
                if len(out) >= int(count):
                    break
        return {"result": {"candles": out}}


def _run(coro):
    return asyncio.run(coro)


def test_two_calls_for_complete_session(tmp_path) -> None:
    grid = _regular_grid(_DAY)
    client = _GridToss(list(reversed(grid)))
    total = 390 * 100
    frame, entry = _run(
        acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=float(total),
                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-two")
    )
    assert entry.status == CaptureStatus.COMPLETE
    assert entry.reason == "toss_regular:390"
    assert len(frame) == 390
    assert [c["count"] for c in client.calls] == [200, 190]
    assert len(client.calls) == 2
    labels = frame["ts_hms"].astype(int).tolist()
    assert labels[0] == 90100 and labels[-1] == 153000
    assert labels == sorted(labels)
    assert len(entry.raw_refs) == 2


def test_raw_basis_is_explicit(tmp_path) -> None:
    grid = _regular_grid(_DAY)
    client = _GridToss(list(reversed(grid)))
    _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=39000.0,
                                   profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-raw"))
    assert client.calls and all(c["adjusted"] is False for c in client.calls)


def _newest_first(candles: list[dict]) -> list[dict]:
    return sorted(candles, key=lambda c: str(c.get("timestamp", "")), reverse=True)


def test_premarket_and_other_date_bars_dropped(tmp_path) -> None:
    regular = _regular_grid(_DAY)
    premarket = [_candle(_DAY, f"08{m:02d}00") for m in range(30, 60)]
    prev_day = [_candle(_PREV, _hhmmss(m)) for m in range(14 * 60, 15 * 60 + 31)]
    client = _GridToss(_newest_first(regular + premarket + prev_day))
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=39000.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-drop"))
    assert entry.status == CaptureStatus.COMPLETE
    assert len(frame) == 390
    labels = set(frame["ts_hms"].astype(int).tolist())
    assert not any(80000 <= v < 90100 for v in labels)
    assert (frame["snapshot_date"] == _DAY).all()
    assert (frame["vendor"] == "toss").all()


def test_incomplete_grid_takes_third_page_only(tmp_path) -> None:
    regular = _regular_grid(_DAY)
    trimmed = [c for c in regular if c["timestamp"][11:16] > "09:20"]
    premarket = [_candle(_DAY, f"08{m:02d}00") for m in range(0, 60)]
    prev_day = [_candle(_PREV, _hhmmss(m)) for m in range(9 * 60 + 1, 15 * 60 + 31)]
    client = _GridToss(_newest_first(trimmed + premarket + prev_day))
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY,
                                                  eod_volume=float(sum(int(c["volume"]) for c in trimmed)),
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-third"))
    assert len(client.calls) == 3
    assert entry.status == CaptureStatus.COMPLETE
    assert len(frame) == len(trimmed)


def test_malformed_page_fails_closed(tmp_path) -> None:
    grid = _regular_grid(_DAY)
    dup = list(reversed(grid))
    dup[5] = dict(dup[4])
    client = _GridToss(dup)
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=39000.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-mal"))
    assert frame.empty
    assert entry.status == CaptureStatus.FAILED
    assert entry.reason == "toss_malformed_page"


def test_malformed_seconds_fails_closed(tmp_path) -> None:
    bad = _candle(_DAY, "093000")
    bad["timestamp"] = f"{_DAY}T09:30:30.000+09:00"
    grid = _regular_grid(_DAY)
    client = _GridToss([bad, *list(reversed(grid))[:199]])
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=39000.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-sec"))
    assert frame.empty and entry.reason == "toss_malformed_page"


def test_bad_timestamp_page_fails_closed(tmp_path) -> None:
    grid = _regular_grid(_DAY)
    broken = list(reversed(grid))
    broken[0] = {"timestamp": "not-a-time", "openPrice": "1", "highPrice": "1", "lowPrice": "1", "closePrice": "1", "volume": "1"}
    client = _GridToss(broken)
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=39000.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-bad"))
    assert frame.empty and entry.reason == "toss_malformed_page"


def test_stock_not_found_is_symbol_level(tmp_path) -> None:
    client = _GridToss([], error={"error": {"code": "stock-not-found", "message": "gone"}})
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "000000", _DAY, eod_volume=100.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-delist"))
    assert frame.empty
    assert entry.status == CaptureStatus.NOT_APPLICABLE
    assert entry.reason == "toss_stock_not_found"
    assert len(entry.raw_refs) == 1


def test_vendor_error_maps_to_failure(tmp_path) -> None:
    client = _GridToss([], error={"error": {"code": "invalid-request", "message": "bad"}})
    _, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=100.0,
                                              profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-vendor"))
    assert entry.status == CaptureStatus.FAILED and entry.reason == "vendor_failure:invalid-request"
    blank, entry2 = _run(acquire_toss_regular_bars(_GridToss([], error={"error": {}}), object(), "005930", _DAY,
                                                   eod_volume=100.0, profile=_profile(tmp_path),
                                                   capture_store=_store(tmp_path), run_id="run-vendor2"))
    assert blank.empty and entry2.reason == "vendor_failure:unknown"
    missing, entry3 = _run(acquire_toss_regular_bars(_GridToss([], error={"result": {}}), object(), "005930", _DAY,
                                                     eod_volume=100.0, profile=_profile(tmp_path),
                                                     capture_store=_store(tmp_path), run_id="run-vendor3"))
    assert missing.empty and entry3.reason == "vendor_failure:unknown"


def test_non_dict_payload_fails_closed(tmp_path) -> None:
    class _None:
        async def get_candles(self, *a, **k):
            return None

    frame, entry = _run(acquire_toss_regular_bars(_None(), object(), "005930", _DAY, eod_volume=100.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-none"))
    assert frame.empty and entry.reason == "vendor_failure:unknown"


def test_empty_without_proof(tmp_path) -> None:
    client = _GridToss([])
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=100.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-empty"))
    assert frame.empty and entry.reason == "toss_empty_without_proof"


def test_no_regular_bars_when_only_premarket(tmp_path) -> None:
    premarket = [_candle(_DAY, f"08{m:02d}00") for m in range(0, 60)]
    client = _GridToss(list(reversed(premarket)))
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=100.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-pre"))
    assert frame.empty and entry.reason == "toss_no_regular_bars"


def test_older_date_newest_stops_without_regular(tmp_path) -> None:
    prev_day = [_candle(_PREV, _hhmmss(m)) for m in range(9 * 60 + 1, 15 * 60 + 31)]
    client = _GridToss(list(reversed(prev_day)))
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=100.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-prev"))
    assert frame.empty and entry.reason == "toss_no_regular_bars"
    assert len(client.calls) == 1


def test_consolidated_tape_rejected(tmp_path) -> None:
    grid = _regular_grid(_DAY, volume=150)
    client = _GridToss(list(reversed(grid)))
    total = 390 * 150
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=float(total) / 1.5,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-cons"))
    assert not frame.empty
    assert (frame["vendor"] == "toss").all()
    assert entry.status == CaptureStatus.NOT_APPLICABLE
    assert entry.reason == "toss_consolidated_tape"


def test_consolidated_frame_is_canonical_and_sorted(tmp_path) -> None:
    """A toss_consolidated_tape rejection returns its canonical vendor-toss frame for the consolidated partition."""
    grid = _regular_grid(_DAY, volume=150)
    client = _GridToss(list(reversed(grid)))
    total = 390 * 150
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=float(total) / 1.5,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-cons-frame"))
    assert entry.status == CaptureStatus.NOT_APPLICABLE and entry.reason == "toss_consolidated_tape"
    assert len(frame) == 390
    assert (frame["vendor"] == "toss").all()
    assert (frame["snapshot_date"] == _DAY).all()
    labels = frame["ts_hms"].astype(int).tolist()
    assert labels == sorted(labels)
    assert labels[0] == 90100 and labels[-1] == 153000


def test_other_rejections_stay_empty(tmp_path) -> None:
    """Shortfall, unverifiable, malformed and not-found rejections carry no frame."""
    short = _GridToss(list(reversed(_regular_grid(_DAY, volume=50))))
    frame, entry = _run(acquire_toss_regular_bars(short, object(), "005930", _DAY, eod_volume=39000.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-o-short"))
    assert frame.empty and entry.reason == "toss_volume_shortfall"
    unverifiable = _GridToss(list(reversed(_regular_grid(_DAY))))
    frame, entry = _run(acquire_toss_regular_bars(unverifiable, object(), "005930", _DAY, eod_volume=None,
                                                 profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-o-unv"))
    assert frame.empty and entry.reason == "toss_basis_unverifiable"
    malformed = _GridToss([{"timestamp": "not-a-time", "volume": "1"}])
    frame, entry = _run(acquire_toss_regular_bars(malformed, object(), "005930", _DAY, eod_volume=100.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-o-mal"))
    assert frame.empty and entry.reason == "toss_malformed_page"
    missing = _GridToss([], error={"error": {"code": "stock-not-found", "message": "gone"}})
    frame, entry = _run(acquire_toss_regular_bars(missing, object(), "000000", _DAY, eod_volume=100.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-o-miss"))
    assert frame.empty and entry.reason == "toss_stock_not_found"


def test_volume_shortfall_retryable(tmp_path) -> None:
    grid = _regular_grid(_DAY, volume=50)
    client = _GridToss(list(reversed(grid)))
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=39000.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-short"))
    assert frame.empty and entry.reason == "toss_volume_shortfall"


def test_unverifiable_basis_via_acquire(tmp_path) -> None:
    grid = _regular_grid(_DAY)
    client = _GridToss(list(reversed(grid)))
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=None,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-unv"))
    assert frame.empty and entry.reason == "toss_basis_unverifiable"


def test_basis_boundaries() -> None:
    frame = pd.DataFrame({"volume": [100] * 100})
    assert toss_basis_verdict(frame, 10000.0, ratio_min=1.0, ratio_tolerance=0.0).accepted
    assert toss_basis_verdict(frame, 10000.0 / 1.0, ratio_min=1.0, ratio_tolerance=0.0).accepted
    rejected = toss_basis_verdict(frame, 10000.0 / 1.001, ratio_min=1.0, ratio_tolerance=0.0)
    assert not rejected.accepted and rejected.reason == "toss_consolidated_tape"


def test_unverifiable_basis() -> None:
    frame = pd.DataFrame({"volume": [100] * 10})
    for bad in (None, float("nan"), 0.0, -5.0, float("inf")):
        verdict = toss_basis_verdict(frame, bad, ratio_min=0.9, ratio_tolerance=0.05)
        assert not verdict.accepted and verdict.reason == "toss_basis_unverifiable" and verdict.volume_ratio is None
    assert toss_basis_verdict(frame, "bad", ratio_min=0.9, ratio_tolerance=0.05).reason == "toss_basis_unverifiable"
    assert toss_basis_verdict(frame, object(), ratio_min=0.9, ratio_tolerance=0.05).reason == "toss_basis_unverifiable"
    empty = toss_basis_verdict(pd.DataFrame(), 100.0, ratio_min=0.9, ratio_tolerance=0.05)
    assert empty.reason == "toss_volume_shortfall"
    no_col = toss_basis_verdict(pd.DataFrame({"x": [1]}), 100.0, ratio_min=0.9, ratio_tolerance=0.05)
    assert no_col.reason == "toss_volume_shortfall"
    inf_frame = pd.DataFrame({"volume": [float("inf")]})
    assert toss_basis_verdict(inf_frame, 100.0, ratio_min=0.9, ratio_tolerance=0.05).reason == "toss_basis_unverifiable"


def test_shortfall_and_consolidated_reasons() -> None:
    low = pd.DataFrame({"volume": [10] * 10})
    verdict = toss_basis_verdict(low, 10000.0, ratio_min=0.9, ratio_tolerance=0.05)
    assert verdict.reason == "toss_volume_shortfall"
    high = pd.DataFrame({"volume": [1000] * 100})
    verdict2 = toss_basis_verdict(high, 10000.0, ratio_min=0.9, ratio_tolerance=0.05)
    assert verdict2.reason == "toss_consolidated_tape"


def test_uncertified_venue_fails_closed(tmp_path) -> None:
    grid = _regular_grid(_DAY)
    client = _GridToss(list(reversed(grid)))
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=39000.0,
                                                  profile=_profile(tmp_path, routes={}), capture_store=_store(tmp_path),
                                                  run_id="run-uncert"))
    assert frame.empty
    assert entry.status == CaptureStatus.UNKNOWN
    assert entry.reason == "uncertified_venue"
    assert len(entry.raw_refs) == 3


def test_evidence_retained_on_transport_failure(tmp_path) -> None:
    grid = _regular_grid(_DAY)
    client = _GridToss(list(reversed(grid)), explode_on={1})
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=39000.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-trans"))
    assert frame.empty
    assert entry.status == CaptureStatus.FAILED
    assert entry.reason.startswith("transport:")
    assert len(entry.raw_refs) == 1


def test_future_date_raises(tmp_path) -> None:
    import pytest

    today = datetime.now(SEOUL).date().isoformat()
    with pytest.raises(ValueError, match="strictly past"):
        _run(acquire_toss_regular_bars(_GridToss([]), object(), "005930", today, eod_volume=1.0,
                                       profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-future"))
    with pytest.raises(ValueError, match="run_id"):
        _run(acquire_toss_regular_bars(_GridToss([]), object(), "005930", _DAY, eod_volume=1.0,
                                       profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="  "))


def test_missing_columns_fail_malformed(tmp_path) -> None:
    bad = [{"timestamp": f"{_DAY}T09:30:00.000+09:00", "openPrice": "1"} for _ in range(200)]
    client = _GridToss(list(reversed(bad)))
    frame, entry = _run(acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=100.0,
                                                  profile=_profile(tmp_path), capture_store=_store(tmp_path), run_id="run-cols"))
    assert frame.empty and entry.reason == "toss_malformed_page"


def test_normalized_empty_maps_to_no_regular(tmp_path, monkeypatch) -> None:
    import src.backfill.intraday.toss_regular as mod

    grid = _regular_grid(_DAY)
    client = _GridToss(list(reversed(grid)))
    monkeypatch.setattr(mod, "normalize_bar_frame", lambda *a, **k: pd.DataFrame())
    frame, entry = _run(mod.acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=39000.0,
                                                       profile=_profile(tmp_path), capture_store=_store(tmp_path),
                                                       run_id="run-norm-empty"))
    assert frame.empty and entry.reason == "toss_no_regular_bars"


def test_default_profile_and_store(tmp_path, monkeypatch) -> None:
    import src.backfill.intraday.toss_regular as mod

    grid = _regular_grid(_DAY)
    client = _GridToss(list(reversed(grid)))
    monkeypatch.setattr(mod, "CollectionSettings", lambda: _profile(tmp_path))
    frame, entry = _run(mod.acquire_toss_regular_bars(client, object(), "005930", _DAY, eod_volume=39000.0,
                                                       profile=None, capture_store=None, run_id="run-default"))
    assert entry.status == CaptureStatus.COMPLETE and len(frame) == 390


def _floor_client(served_from: str | None, calls: list, bad_days: set[str] | None = None, error_day: str | None = None):
    bad_days = bad_days or set()

    class _C:
        async def get_candles(self, session, symbol, *, interval="1m", count=200, before=None, adjusted=None):
            day = str(before)[:10]
            calls.append(day)
            assert adjusted is False
            assert str(before) == f"{day}T15:30:00+09:00"
            if error_day is not None and day == error_day:
                return {"error": {"code": "invalid-request", "message": "bad"}} if day != "no-result" else {"weird": 1}
            if served_from is not None and day >= served_from:
                if day in bad_days:
                    return {"result": {"candles": [{"timestamp": "bad", "volume": "1"}]}}
                return {"result": {"candles": [_candle(day, "093000")]}}
            return {"result": {"candles": []}}

    return _C()


def test_floor_bisection(tmp_path) -> None:
    days = [f"2024-01-{d:02d}" for d in range(1, 21)]
    calls: list[str] = []
    floor = _run(probe_toss_retention_floor(_floor_client("2024-01-13", calls), object(),
                                            trading_days=days, reference_symbol="005930"))
    assert floor == "2024-01-13"
    assert len(calls) <= math.ceil(math.log2(len(days))) + 1


def test_floor_all_empty_returns_none() -> None:
    days = ["2024-01-01", "2024-01-02"]
    assert _run(probe_toss_retention_floor(_floor_client(None, []), object(), trading_days=days, reference_symbol="005930")) is None
    assert _run(probe_toss_retention_floor(_floor_client(None, []), object(), trading_days=[], reference_symbol="005930")) is None
    assert _run(probe_toss_retention_floor(_floor_client("2024-01-01", []), object(), trading_days=["2024-01-01"],
                                           reference_symbol="005930")) == "2024-01-01"


def test_floor_vendor_error_propagates() -> None:
    import pytest

    days = ["2024-01-01", "2024-01-02"]
    with pytest.raises(RuntimeError, match="vendor_failure"):
        _run(probe_toss_retention_floor(_floor_client("2024-01-01", [], error_day="2024-01-02"), object(),
                                        trading_days=days, reference_symbol="005930"))


def test_floor_transport_propagates() -> None:
    import pytest

    class _Boom:
        async def get_candles(self, *a, **k):
            raise TimeoutError("down")

    with pytest.raises(TimeoutError, match="down"):
        _run(probe_toss_retention_floor(_Boom(), object(), trading_days=["2024-01-01"], reference_symbol="005930"))


def test_floor_skips_bad_timestamps() -> None:
    days = ["2024-01-01", "2024-01-02"]
    floor = _run(probe_toss_retention_floor(_floor_client("2024-01-01", [], bad_days={"2024-01-01"}), object(),
                                            trading_days=days, reference_symbol="005930"))
    assert floor == "2024-01-02"


def test_expected_constants() -> None:
    assert TOSS_CANDLES_PAGE_MAX == 200
    assert TOSS_REGULAR_EXPECTED_BARS == 390
