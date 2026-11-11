"""Aftermarket order-book capture invariant guards."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

from src.execution.paper_broker import HeldRoster, load_held_roster


def _seoul(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> dt.datetime:
    from src.data.capture_contracts import SEOUL

    return dt.datetime(year, month, day, hour, minute, second, tzinfo=SEOUL)


def _profile(tmp_path: Path, **overrides: Any):
    from src.config.collection import CollectionSettings

    base: dict[str, Any] = {
        "COLLECTION_ROOT": tmp_path / "capture",
        "COLLECTION_AFTERMARKET_BOOK_DENSE_SECONDS": 3600,
        "COLLECTION_AFTERMARKET_BOOK_SPARSE_TIMES": (),
        "COLLECTION_AFTERMARKET_BOOK_FLUSH_ROUNDS": 15,
        "COLLECTION_CONCURRENCY_PER_KEY": 8,
    }
    base.update(overrides)
    return CollectionSettings(**base)


def _store(tmp_path: Path):
    from src.data.capture_store import CaptureStore

    return CaptureStore(tmp_path / "capture")


def _publish_cohort(store: Any, snapshot_date: str, eligible: list[str]) -> Any:
    from src.data.capture_contracts import (
        SEOUL,
        CaptureContext,
        CaptureDataset,
        CaptureManifest,
        CaptureStatus,
        Cohort,
    )

    day = dt.date.fromisoformat(snapshot_date)
    cohort = Cohort(
        trading_date=day,
        cohort_id=f"c-{snapshot_date}",
        eligible_symbols=tuple(eligible),
        scanned_symbols=tuple(eligible),
        eligibility_rule_version="v1",
        rejections={},
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=day,
            run_id="decision-1",
            dataset=CaptureDataset.SCAN,
            vendor="owner-local",
            endpoint="decision-input",
            symbol=None,
            venue="KRX",
            session="regular",
            capture_reason="test",
            cohort_id=cohort.cohort_id,
            scheduled_at=None,
        ),
        cohort=cohort,
        completed_at=dt.datetime(2026, 9, 29, 9, 0, tzinfo=SEOUL),
        entries=(),
        artifacts=(),
        status=CaptureStatus.COMPLETE,
    )
    store.publish_manifest(manifest)
    return cohort


class _Session:
    async def __aenter__(self) -> Any:
        return object()

    async def __aexit__(self, *args: Any) -> bool:
        return False


def _listed_book(hour: str = "170001") -> dict[str, Any]:
    return {"rt_cd": "0", "output1": {"aspr_acpt_hour": hour, "askp1": "1000"}}


class _StubClient:
    def __init__(self, *args: Any, handler: Any = None, calls: list[tuple[str | None, str]] | None = None, clock: list[dt.datetime] | None = None, step_seconds: int = 0, **kwargs: Any) -> None:
        self._handler = handler or (lambda div, code: dict(_listed_book()))
        self.calls: list[tuple[str | None, str]] = calls if calls is not None else []
        self._clock = clock
        self._step = step_seconds

    def create_session(self) -> _Session:
        return _Session()

    async def ensure_token(self, session: Any) -> None:
        return None

    async def get_orderbook_snapshot(self, session: Any, code: str, market_div_code: str | None = None) -> dict[str, Any]:
        self.calls.append((market_div_code, code))
        if self._clock is not None and self._step:
            self._clock[0] = self._clock[0] + dt.timedelta(seconds=self._step)
        return self._handler(market_div_code, code)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


_DEFAULT_SPARSE: tuple[str, ...] = (
    "154500",
    "160500",
    "163000",
    "170000",
    "173000",
    "180000",
    "183000",
    "190000",
    "193000",
    "195000",
    "195800",
)


def test_schedule_spans_evening_without_duplicate_rounds() -> None:
    """Schedule spans the NXT evening without duplicates."""
    from src.daily.aftermarket_book import aftermarket_book_rounds

    rounds = aftermarket_book_rounds(dt.date(2026, 9, 29), dense_seconds=60, sparse_times=_DEFAULT_SPARSE)

    assert rounds[0].scheduled_at == _seoul(2026, 9, 29, 15, 41)
    assert rounds[-1].scheduled_at == _seoul(2026, 9, 29, 19, 59)
    assert all(item.scheduled_at < _seoul(2026, 9, 29, 20, 0) for item in rounds)
    assert [item for item in rounds if item.kind == "sparse"] and all(
        item.scheduled_at.strftime("%H%M%S") in _DEFAULT_SPARSE for item in rounds if item.kind == "sparse"
    )
    sparse_minutes = {(item.scheduled_at.hour, item.scheduled_at.minute) for item in rounds if item.kind == "sparse"}
    assert all(
        (item.scheduled_at.hour, item.scheduled_at.minute) not in sparse_minutes
        for item in rounds
        if item.kind == "dense"
    )
    minutes = [(item.scheduled_at.hour, item.scheduled_at.minute) for item in rounds]
    assert len(set(minutes)) == len(minutes)
    assert tuple(sorted(item.scheduled_at for item in rounds)) == tuple(item.scheduled_at for item in rounds)


def test_schedule_rejects_non_positive_spacing() -> None:
    """Non-positive dense spacing is rejected."""
    import pytest

    from src.daily.aftermarket_book import aftermarket_book_rounds

    with pytest.raises(ValueError, match="positive"):
        aftermarket_book_rounds(dt.date(2026, 9, 29), dense_seconds=0, sparse_times=())


def test_venues_resolve_krx_only_after_floor() -> None:
    """KRX book requested only after 16:00."""
    from src.daily.aftermarket_book import venues_for_round

    assert venues_for_round(_seoul(2026, 9, 29, 15, 45)) == ("NXT",)
    assert venues_for_round(_seoul(2026, 9, 29, 16, 0)) == ("NXT",)
    assert venues_for_round(_seoul(2026, 9, 29, 16, 1)) == ("KRX", "NXT")


def _patch_positions(monkeypatch: Any, symbols: list[str]) -> None:
    import pandas as pd

    class _Ledger:
        def load_open_positions(self) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "entry_order_id": [f"o{i}" for i in range(len(symbols))],
                    "symbol": list(symbols),
                    "qty": [1] * len(symbols),
                    "entry_price": [1000] * len(symbols),
                    "decision_date": ["2026-09-29"] * len(symbols),
                }
            )

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", _Ledger)


def _write_rank_pool(path: Path, rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_dense_universe_combines_rank_pool_and_positions(tmp_path: Path, monkeypatch: Any) -> None:
    """Dense universe is the rank pool plus positions."""
    from src.daily.aftermarket_book import resolve_book_universes

    pool_path = tmp_path / "rank_pool.parquet"
    _write_rank_pool(
        pool_path,
        [
            {"symbol": "000001", "decision_date": "2026-09-29", "pred": 0.9},
            {"symbol": "000002", "decision_date": "2026-09-29", "pred": 0.8},
            {"symbol": "000003", "decision_date": "2026-09-28", "pred": 0.7},
        ],
    )
    _patch_positions(monkeypatch, ["000002", "000004"])
    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-29", ["000005"])
    now = _seoul(2026, 9, 29, 15, 35)

    dense, sparse = resolve_book_universes("2026-09-29", store=store, now=now, held=load_held_roster(), rank_pool_path=pool_path)

    assert dense == ("000001", "000002", "000004")
    assert sparse == ("000001", "000002", "000004", "000005")


def test_missing_rank_pool_falls_back_to_positions_with_warning(tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
    """Missing rank pool degrades loudly."""
    import logging

    from src.daily.aftermarket_book import resolve_book_universes

    _patch_positions(monkeypatch, ["000004"])
    store = _store(tmp_path)
    now = _seoul(2026, 9, 29, 15, 35)

    with caplog.at_level(logging.WARNING, logger="src.daily.aftermarket_book"):
        dense, sparse = resolve_book_universes(
            "2026-09-29", store=store, now=now, rank_pool_path=tmp_path / "absent.parquet", held=load_held_roster()
        )

    assert dense == ("000004",)
    assert sparse == ("000004",)
    assert "rank_pool=MISSING" in caplog.text
    assert "cohort=MISSING" in caplog.text


def test_empty_or_malformed_rank_pool_falls_back_to_positions(tmp_path: Path, monkeypatch: Any) -> None:
    """Empty, column-drifted, stale or blank rank pools degrade to positions."""
    import pandas as pd

    from src.daily.aftermarket_book import resolve_book_universes

    _patch_positions(monkeypatch, ["000004"])
    store = _store(tmp_path)
    now = _seoul(2026, 9, 29, 15, 35)

    empty_path = tmp_path / "empty.parquet"
    pd.DataFrame([{"symbol": "000001"}]).iloc[0:0].to_parquet(empty_path, index=False)
    dense, _ = resolve_book_universes("2026-09-29", store=store, now=now, held=load_held_roster(), rank_pool_path=empty_path)
    assert dense == ("000004",)

    drifted_path = tmp_path / "drifted.parquet"
    pd.DataFrame([{"weird": 1}]).to_parquet(drifted_path, index=False)
    dense, _ = resolve_book_universes("2026-09-29", store=store, now=now, held=load_held_roster(), rank_pool_path=drifted_path)
    assert dense == ("000004",)

    stale_path = tmp_path / "stale.parquet"
    _write_rank_pool(stale_path, [{"symbol": "000001", "decision_date": "2026-09-28", "pred": 0.5}])
    dense, _ = resolve_book_universes("2026-09-29", store=store, now=now, held=load_held_roster(), rank_pool_path=stale_path)
    assert dense == ("000004",)

    blank_path = tmp_path / "blank.parquet"
    _write_rank_pool(blank_path, [{"symbol": "", "decision_date": "2026-09-29", "pred": 0.5}])
    dense, _ = resolve_book_universes("2026-09-29", store=store, now=now, held=load_held_roster(), rank_pool_path=blank_path)
    assert dense == ("000004",)


def test_positions_unavailable_yields_empty_dense(tmp_path: Path, caplog: Any) -> None:
    """Unreadable ledgers degrade to an empty dense universe with an INCOMPLETE warning."""
    import logging

    store = _store(tmp_path)
    now = _seoul(2026, 9, 29, 15, 35)

    from src.daily.aftermarket_book import resolve_book_universes

    with caplog.at_level(logging.WARNING, logger="src.daily.aftermarket_book"):
        dense, sparse = resolve_book_universes(
            "2026-09-29",
            store=store,
            now=now,
            rank_pool_path=tmp_path / "absent.parquet",
            held=HeldRoster((), False, "OSError"),
        )

    assert dense == ()
    assert sparse == ()
    assert "status=INCOMPLETE reason=held_roster_unavailable error=OSError" in caplog.text


def _unlisted_handler(div: str | None, code: str) -> dict[str, Any]:
    if div == "NX" and code in ("000002", "000005"):
        if code == "000005":
            return {"rt_cd": "0"}
        return {"rt_cd": "0", "output1": {"aspr_acpt_hour": None, "askp1": "1000"}}
    if code == "000003":
        return {"rt_cd": "1", "msg1": "busy"}
    if code == "000004":
        raise RuntimeError("boom")
    return dict(_listed_book())


def test_unlisted_nxt_symbols_stop_polling(tmp_path: Path) -> None:
    """Unlisted NXT symbols stop being polled."""
    from src.daily.aftermarket_book import run_aftermarket_book_capture
    from src.data.capture_contracts import CaptureStatus

    profile = _profile(tmp_path)
    store = _store(tmp_path)
    client = _StubClient(handler=_unlisted_handler)
    now = [_seoul(2026, 9, 29, 15, 30)]
    symbols = ("000001", "000002", "000003", "000004", "000005")

    manifests = _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=profile,
            store=store,
            clients=[client],
            dense_symbols=symbols,
            sparse_symbols=symbols,
            now_fn=lambda: now[0],
        )
    )

    nx_calls_b = [call for call in client.calls if call == ("NX", "000002")]
    assert len(nx_calls_b) == 1
    nx_calls_e = [call for call in client.calls if call == ("NX", "000005")]
    assert len(nx_calls_e) == 1
    by_venue = {m.context.session: m for m in manifests}
    nxt_entries = {e.symbol: e for e in by_venue["nxt_aftermarket"].entries}
    assert nxt_entries["000002"].status == CaptureStatus.NOT_APPLICABLE
    assert nxt_entries["000002"].reason == "nxt_not_listed"
    assert len(nxt_entries["000002"].raw_refs) == 1
    assert nxt_entries["000005"].status == CaptureStatus.NOT_APPLICABLE
    assert nxt_entries["000001"].status == CaptureStatus.COMPLETE
    assert nxt_entries["000003"].status == CaptureStatus.PARTIAL
    assert nxt_entries["000003"].reason == "vendor_failure"
    assert len(nxt_entries["000003"].raw_refs) >= 1
    assert nxt_entries["000004"].status == CaptureStatus.PARTIAL
    assert nxt_entries["000004"].reason == "vendor_failure"
    assert nxt_entries["000004"].raw_refs == ()
    krx_entries = {e.symbol: e for e in by_venue["krx_aftermarket"].entries}
    assert krx_entries["000002"].status == CaptureStatus.COMPLETE


def test_capture_rejects_empty_clients_and_bad_dates(tmp_path: Path) -> None:
    """Empty clients or invalid snapshot dates are rejected."""
    import pytest

    from src.daily.aftermarket_book import run_aftermarket_book_capture

    profile = _profile(tmp_path)
    store = _store(tmp_path)
    now = _seoul(2026, 9, 29, 15, 30)

    with pytest.raises(ValueError, match="prewarmed data clients"):
        _run(
            run_aftermarket_book_capture(
                "2026-09-29", profile=profile, store=store, clients=[], dense_symbols=(), sparse_symbols=(), now_fn=lambda: now
            )
        )
    with pytest.raises(ValueError, match="Invalid snapshot_date"):
        _run(
            run_aftermarket_book_capture(
                "not-a-date",
                profile=profile,
                store=store,
                clients=[_StubClient()],
                dense_symbols=(),
                sparse_symbols=(),
                now_fn=lambda: now,
            )
        )


def test_late_requests_dropped_without_drifting_schedule(tmp_path: Path) -> None:
    """Late requests are dropped, not delayed."""
    from src.daily.aftermarket_book import aftermarket_book_rounds, run_aftermarket_book_capture
    from src.data.capture_contracts import CaptureStatus

    profile = _profile(tmp_path)
    store = _store(tmp_path)
    box = [_seoul(2026, 9, 29, 16, 39)]
    client = _StubClient(clock=box, step_seconds=1800)
    symbols = ("000001", "000002")
    expected_rounds = aftermarket_book_rounds(dt.date(2026, 9, 29), dense_seconds=3600, sparse_times=())
    instants = {item.scheduled_at for item in expected_rounds}

    manifests = _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=profile,
            store=store,
            clients=[client],
            dense_symbols=symbols,
            sparse_symbols=symbols,
            now_fn=lambda: box[0],
        )
    )

    entries = [e for m in manifests for e in m.entries]
    assert any(e.status == CaptureStatus.PARTIAL and e.reason == "deadline" for e in entries)
    assert all(e.scheduled_at in instants for e in entries)
    stamps = {
        store.read_artifact(ref)["context"]["scheduled_at"] for e in entries for ref in e.raw_refs
    }
    assert _seoul(2026, 9, 29, 17, 40).isoformat() in stamps


def test_rounds_missed_before_start_recorded_never_executed(tmp_path: Path) -> None:
    """Missed rounds at start are recorded, never executed."""
    from src.daily.aftermarket_book import run_aftermarket_book_capture
    from src.data.capture_contracts import CaptureStatus

    profile = _profile(tmp_path)
    store = _store(tmp_path)
    start = _seoul(2026, 9, 29, 17, 0)
    client = _StubClient()
    symbols = ("000001", "000002")

    manifests = _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=profile,
            store=store,
            clients=[client],
            dense_symbols=symbols,
            sparse_symbols=symbols,
            now_fn=lambda: start,
        )
    )

    assert len(client.calls) == 12
    entries = [e for m in manifests for e in m.entries]
    missed = [e for e in entries if e.reason == "missed_start"]
    assert missed and all(e.scheduled_at == _seoul(2026, 9, 29, 16, 40) for e in missed)
    stamps = {
        store.read_artifact(ref)["context"]["scheduled_at"]
        for e in entries
        for ref in e.raw_refs
    }
    assert stamps and all(stamp is not None and stamp >= start.isoformat() for stamp in stamps)


def test_raw_and_normalized_evidence_per_response(tmp_path: Path, monkeypatch: Any) -> None:
    """Raw and normalized evidence per response."""
    import pandas as pd

    from src.daily.aftermarket_book import run_aftermarket_book_capture
    from src.data import orderbook_store

    monkeypatch.setattr(orderbook_store.settings, "HISTORY_DIR", tmp_path / "history")
    profile = _profile(tmp_path)
    store = _store(tmp_path)
    client = _StubClient()
    now = [_seoul(2026, 9, 29, 15, 30)]
    symbols = ("000001", "000002")

    manifests = _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=profile,
            store=store,
            clients=[client],
            dense_symbols=symbols,
            sparse_symbols=symbols,
            now_fn=lambda: now[0],
        )
    )

    assert sum(len(m.artifacts) for m in manifests) == 16
    target = tmp_path / "history" / "orderbook" / "aftermarket" / "2026-09" / "2026-09-29.parquet"
    frame = pd.read_parquet(target)
    assert len(frame) == 16
    assert {"scheduled_at", "request_started_at", "aspr_acpt_hour"} <= set(frame.columns)
    assert not (tmp_path / "capture" / "normalized").exists()


def test_manifests_published_per_block_and_venue(tmp_path: Path) -> None:
    """Manifests per block and venue."""
    from src.daily.aftermarket_book import run_aftermarket_book_capture
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    profile = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_FLUSH_ROUNDS=2)
    store = _store(tmp_path)
    client = _StubClient(handler=_unlisted_handler)
    now = [_seoul(2026, 9, 29, 15, 30)]
    symbols = ("000001", "000002")

    manifests = _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=profile,
            store=store,
            clients=[client],
            dense_symbols=symbols,
            sparse_symbols=symbols,
            now_fn=lambda: now[0],
        )
    )

    assert len(manifests) == 4
    run_ids = [m.context.run_id for m in manifests]
    assert len(set(run_ids)) == 4
    for manifest in manifests:
        assert manifest.context.capture_reason == "aftermarket-book"
        assert manifest.context.dataset == CaptureDataset.ORDERBOOK
        assert manifest.context.vendor == "kis"
        assert manifest.context.endpoint == "aftermarket-book"
        assert manifest.context.venue == "owner-local"
        assert manifest.context.cohort_id is None
    sessions = sorted(m.context.session for m in manifests)
    assert sessions == ["krx_aftermarket", "krx_aftermarket", "nxt_aftermarket", "nxt_aftermarket"]
    second_nxt = [m for m in manifests if m.context.session == "nxt_aftermarket"][1]
    carried = {e.symbol: e for e in second_nxt.entries}["000002"]
    assert carried.status == CaptureStatus.NOT_APPLICABLE
    assert carried.reason == "nxt_not_listed"


def test_venue_without_rounds_still_flushes_and_heartbeats(tmp_path: Path, caplog: Any) -> None:
    """A block with no KRX rounds still flushes, skips the KRX manifest and heartbeats."""
    import logging

    from src.daily.aftermarket_book import run_aftermarket_book_capture

    profile = _profile(
        tmp_path,
        COLLECTION_AFTERMARKET_BOOK_DENSE_SECONDS=1200,
        COLLECTION_AFTERMARKET_BOOK_FLUSH_ROUNDS=1,
    )
    store = _store(tmp_path)
    client = _StubClient()
    now = [_seoul(2026, 9, 29, 15, 30)]
    symbols = ("000001",)

    with caplog.at_level(logging.INFO, logger="src.daily.aftermarket_book"):
        manifests = _run(
            run_aftermarket_book_capture(
                "2026-09-29",
                profile=profile,
                store=store,
                clients=[client],
                dense_symbols=symbols,
                sparse_symbols=symbols,
                now_fn=lambda: now[0],
            )
        )

    sessions = sorted(m.context.session for m in manifests)
    assert sessions.count("krx_aftermarket") == 11
    assert sessions.count("nxt_aftermarket") == 12
    assert "stage=aftermarket_book block=0" in caplog.text


def test_wait_primitive_sleep_and_skip() -> None:
    """The scheduler waits for future rounds on real clocks and skips waiting under test clocks."""
    from src.daily.aftermarket_book import _wait_until_scheduled

    seen: list[float] = []

    async def _recorder(delay: float) -> None:
        seen.append(delay)

    async def _main() -> None:
        import asyncio

        await _wait_until_scheduled(
            _seoul(2026, 9, 29, 16, 0),
            now_fn=lambda: _seoul(2026, 9, 29, 15, 59),
            sleeper=_recorder,
            injected_clock=False,
        )
        await _wait_until_scheduled(
            _seoul(2026, 9, 29, 16, 0),
            now_fn=lambda: _seoul(2026, 9, 29, 15, 59),
            sleeper=asyncio.sleep,
            injected_clock=True,
        )

    _run(_main())
    assert seen == [60.0]


def test_run_async_returns_none_on_closed_days(tmp_path: Path) -> None:
    """Saturdays resolve CLOSED before any credential is touched."""
    from src.daily.aftermarket_book import _run_async

    assert _run(_run_async("2026-09-26", _profile(tmp_path))) is None


def test_run_async_full_evening_flow(tmp_path: Path, monkeypatch: Any) -> None:
    """The async runner warms tokens, verifies the trading day and captures empty universes."""
    import src.daily.aftermarket_book as book

    profile = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_SLOTS=("2", "3"))
    monkeypatch.setattr("src.api.kis.client.KisApiClient", _StubClient)
    monkeypatch.setenv("KIS_DATA_SLOTS", "2,3")
    monkeypatch.setenv("KIS_DATA_2_APP_KEY", "key-2")
    monkeypatch.setenv("KIS_DATA_2_APP_SECRET", "secret-2")
    monkeypatch.setenv("KIS_DATA_3_APP_KEY", "key-3")
    monkeypatch.setenv("KIS_DATA_3_APP_SECRET", "secret-3")

    async def _trading_day(client: Any, session: Any, snapshot_date: str) -> bool:
        return True

    monkeypatch.setattr(book, "is_kis_trading_day", _trading_day)
    monkeypatch.setattr(book.settings, "PARQUET_DIR", tmp_path / "parquet")

    class _EmptyLedger:
        def load_open_positions(self) -> Any:
            import pandas as pd

            return pd.DataFrame({"symbol": pd.Series(dtype="str")})

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", _EmptyLedger)

    manifests = _run(book._run_async("2026-09-23", profile))

    assert manifests == ()


def test_run_async_skips_non_trading_days(tmp_path: Path, monkeypatch: Any) -> None:
    """A negative KIS trading-day oracle skips without capturing."""
    import src.daily.aftermarket_book as book

    profile = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_SLOTS=("2", "3"))
    monkeypatch.setattr("src.api.kis.client.KisApiClient", _StubClient)
    monkeypatch.setenv("KIS_DATA_SLOTS", "2,3")
    monkeypatch.setenv("KIS_DATA_2_APP_KEY", "key-2")
    monkeypatch.setenv("KIS_DATA_2_APP_SECRET", "secret-2")
    monkeypatch.setenv("KIS_DATA_3_APP_KEY", "key-3")
    monkeypatch.setenv("KIS_DATA_3_APP_SECRET", "secret-3")

    async def _holiday(client: Any, session: Any, snapshot_date: str) -> bool:
        return False

    monkeypatch.setattr(book, "is_kis_trading_day", _holiday)
    monkeypatch.setattr(book.settings, "PARQUET_DIR", tmp_path / "parquet")
    recorded: list[tuple] = []
    monkeypatch.setattr(book, "record_run_outcome", lambda job, outcome, **kw: recorded.append((job, outcome, kw)))

    assert _run(book._run_async("2026-09-23", profile)) is None
    # KIS-oracle skips record nothing at the inner level; main() owns the SKIPPED record.
    assert recorded == []


def test_main_skips_cleanly(tmp_path: Path, monkeypatch: Any) -> None:
    """Non-standard days skip cleanly before any client is built."""
    import src.daily.aftermarket_book as book
    from src.data.capture_contracts import SessionClock

    class _ExplodingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("client must not be constructed on skip paths")

    monkeypatch.setattr("src.api.kis.client.KisApiClient", _ExplodingClient)

    monkeypatch.setattr(book, "CollectionSettings", lambda *args: _profile(tmp_path))

    shifted = SessionClock(
        trading_date=dt.date(2026, 9, 29),
        open_at=_seoul(2026, 9, 29, 9, 0),
        close_at=_seoul(2026, 9, 29, 15, 30),
        close_confirmation_deadline=_seoul(2026, 9, 29, 15, 33),
        provenance="test",
    )
    enabled = _profile(
        tmp_path,
        COLLECTION_AFTERMARKET_BOOK_SLOTS=("2", "3"),
        COLLECTION_SESSION_OVERRIDES={"2026-09-29": shifted},
    )
    monkeypatch.setattr(book, "CollectionSettings", lambda *args: enabled)
    book.main(["--date", "2026-09-29"])
    book.main(["--date", "2026-09-27"])


def test_main_rejects_invalid_dates(tmp_path: Path, monkeypatch: Any) -> None:
    """Invalid --date values fail fast."""
    import pytest

    import src.daily.aftermarket_book as book

    enabled = _profile(tmp_path)
    monkeypatch.setattr(book, "CollectionSettings", lambda *args: enabled)

    with pytest.raises(ValueError, match="Invalid snapshot_date"):
        book.main(["--date", "not-a-date"])


def test_main_skips_closed_sessions(tmp_path: Path, monkeypatch: Any) -> None:
    """CLOSED sessions skip before credential resolution."""
    import src.daily.aftermarket_book as book
    from src.data.session_calendar import SessionDay, SessionKind

    enabled = _profile(tmp_path)
    monkeypatch.setattr(book, "CollectionSettings", lambda *args: enabled)
    monkeypatch.setattr(
        book,
        "resolve_session_day",
        lambda day, **kwargs: SessionDay(trading_date=day, kind=SessionKind.CLOSED, clock=None, provenance="test"),
    )

    book.main(["--date", "2026-09-29"])


def test_main_reports_unverified_trading_days(tmp_path: Path, monkeypatch: Any) -> None:
    """A negative oracle after a STANDARD resolve skips as a non-trading day."""
    import src.daily.aftermarket_book as book
    from src.data.capture_contracts import SessionClock
    from src.data.session_calendar import SessionDay, SessionKind

    day = dt.date(2026, 9, 29)
    enabled = _profile(tmp_path)
    monkeypatch.setattr(book, "CollectionSettings", lambda *args: enabled)
    monkeypatch.setattr(
        book,
        "resolve_session_day",
        lambda trading_date, **kwargs: SessionDay(
            trading_date=trading_date, kind=SessionKind.STANDARD, clock=SessionClock.standard(trading_date), provenance="test"
        ),
    )

    async def _none(snapshot_date: str, profile: Any) -> None:
        return None

    monkeypatch.setattr(book, "_run_async", _none)
    book.main(["--date", day.isoformat()])


def test_main_logs_final_summary(tmp_path: Path, monkeypatch: Any) -> None:
    """A completed evening run logs its manifest count."""
    import src.daily.aftermarket_book as book
    from src.data.capture_contracts import SessionClock
    from src.data.session_calendar import SessionDay, SessionKind

    enabled = _profile(tmp_path)
    monkeypatch.setattr(book, "CollectionSettings", lambda *args: enabled)
    monkeypatch.setattr(
        book,
        "resolve_session_day",
        lambda trading_date, **kwargs: SessionDay(
            trading_date=trading_date, kind=SessionKind.STANDARD, clock=SessionClock.standard(trading_date), provenance="test"
        ),
    )

    async def _empty(snapshot_date: str, profile: Any) -> tuple[Any, ...]:
        return ()

    monkeypatch.setattr(book, "_run_async", _empty)
    book.main(["--date", "2026-09-29"])


def test_nxt_listing_requires_both_empty_acceptance_and_zero_price() -> None:
    from src.daily.aftermarket_book import _is_nxt_listed

    # 실측 비상장 응답: 접수시각·현재가 모두 비어 있음
    assert _is_nxt_listed({"rt_cd": "0", "output1": {"aspr_acpt_hour": None}, "output2": {"stck_prpr": "0"}}) is False
    assert _is_nxt_listed({"rt_cd": "0", "output1": None, "output2": None}) is False
    # 개장 직후 호가가 잠시 비어도 직전가가 있으면 상장으로 유지
    assert _is_nxt_listed({"rt_cd": "0", "output1": {"aspr_acpt_hour": ""}, "output2": {"stck_prpr": "8,390"}}) is True
    assert _is_nxt_listed({"rt_cd": "0", "output1": {"aspr_acpt_hour": "154100"}, "output2": {"stck_prpr": "0"}}) is True
    assert _is_nxt_listed({"rt_cd": "0", "output1": {"aspr_acpt_hour": None}, "output2": {"stck_prpr": "n/a"}}) is False


def test_roster_incomplete_forces_partial_manifests(tmp_path: Path) -> None:
    """A not-ok roster forces every manifest PARTIAL while entries stay untouched."""
    from src.daily.aftermarket_book import run_aftermarket_book_capture
    from src.data.capture_contracts import CaptureStatus

    profile = _profile(tmp_path)
    store = _store(tmp_path)
    client = _StubClient()
    now = [_seoul(2026, 9, 29, 15, 30)]
    symbols = ("000001",)

    manifests = _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=profile,
            store=store,
            clients=[client],
            dense_symbols=symbols,
            sparse_symbols=symbols,
            roster_incomplete=True,
            now_fn=lambda: now[0],
        )
    )

    assert manifests
    assert all(m.status == CaptureStatus.PARTIAL for m in manifests)
    entries = [e for m in manifests for e in m.entries]
    assert entries and all(e.status == CaptureStatus.COMPLETE for e in entries)


def test_roster_complete_keeps_entry_rule(tmp_path: Path) -> None:
    """A fully known roster leaves the entry-driven COMPLETE rule intact."""
    from src.daily.aftermarket_book import run_aftermarket_book_capture
    from src.data.capture_contracts import CaptureStatus

    profile = _profile(tmp_path)
    store = _store(tmp_path)
    client = _StubClient()
    now = [_seoul(2026, 9, 29, 15, 30)]
    symbols = ("000001",)

    manifests = _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=profile,
            store=store,
            clients=[client],
            dense_symbols=symbols,
            sparse_symbols=symbols,
            roster_incomplete=False,
            now_fn=lambda: now[0],
        )
    )

    assert manifests
    assert all(m.status == CaptureStatus.COMPLETE for m in manifests)


def test_run_async_propagates_ledger_failure(tmp_path: Path, monkeypatch: Any, caplog: Any) -> None:
    """_run_async with a broken ledger still publishes, but every manifest is PARTIAL."""
    import logging

    import src.daily.aftermarket_book as book
    from src.data.capture_contracts import CaptureStatus

    profile = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_SLOTS=("2", "3"))
    monkeypatch.setattr("src.api.kis.client.KisApiClient", _StubClient)
    monkeypatch.setenv("KIS_DATA_SLOTS", "2,3")
    monkeypatch.setenv("KIS_DATA_2_APP_KEY", "key-2")
    monkeypatch.setenv("KIS_DATA_2_APP_SECRET", "secret-2")
    monkeypatch.setenv("KIS_DATA_3_APP_KEY", "key-3")
    monkeypatch.setenv("KIS_DATA_3_APP_SECRET", "secret-3")

    async def _trading_day(client: Any, session: Any, snapshot_date: str) -> bool:
        return True

    monkeypatch.setattr(book, "is_kis_trading_day", _trading_day)
    monkeypatch.setattr(book.settings, "PARQUET_DIR", tmp_path / "parquet")

    store = _store(tmp_path)
    _publish_cohort(store, "2026-09-23", ["000001"])
    pool_path = tmp_path / "parquet" / "rank_pool_predictions.parquet"
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    _write_rank_pool(pool_path, [{"symbol": "000001", "decision_date": "2026-09-23", "pred": 0.9}])

    def _boom_ledger(*args: Any, **kwargs: Any) -> Any:
        raise OSError("locked")

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", _boom_ledger)

    with caplog.at_level(logging.WARNING, logger="src.daily.aftermarket_book"):
        manifests = _run(book._run_async("2026-09-23", profile))

    assert manifests
    assert all(m.status == CaptureStatus.PARTIAL for m in manifests)
    assert "reason=held_roster_unavailable" in caplog.text


def test_run_async_continues_with_healthy_slots_when_one_token_fails(tmp_path: Path, monkeypatch: Any) -> None:
    import src.daily.aftermarket_book as book

    class _FlakyClient(_StubClient):
        def __init__(self, app_key: str, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.app_key = app_key

        async def ensure_token(self, session: Any) -> None:
            if self.app_key == "key-2":
                raise RuntimeError("token issuance failed")

    profile = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_SLOTS=("2", "3"))
    monkeypatch.setattr("src.api.kis.client.KisApiClient", _FlakyClient)
    monkeypatch.setenv("KIS_DATA_SLOTS", "2,3")
    for slot in ("2", "3"):
        monkeypatch.setenv(f"KIS_DATA_{slot}_APP_KEY", f"key-{slot}")
        monkeypatch.setenv(f"KIS_DATA_{slot}_APP_SECRET", f"secret-{slot}")

    async def _trading_day(client: Any, session: Any, snapshot_date: str) -> bool:
        return True

    seen: dict[str, Any] = {}

    async def _capture(snapshot_date: str, **kwargs: Any) -> tuple[()]:
        seen.update(kwargs)
        return ()

    monkeypatch.setattr(book, "is_kis_trading_day", _trading_day)
    monkeypatch.setattr(book, "run_aftermarket_book_capture", _capture)
    monkeypatch.setattr(book.settings, "PARQUET_DIR", tmp_path / "parquet")

    class _EmptyLedger:
        def load_open_positions(self) -> Any:
            import pandas as pd

            return pd.DataFrame({"symbol": pd.Series(dtype="str")})

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", _EmptyLedger)

    _run(book._run_async("2026-09-23", profile))

    assert [c.app_key for c in seen["clients"]] == ["key-3"]
    assert seen["degraded_slots"] == ("DATA_2",)


def test_degraded_slot_marks_every_manifest_partial(tmp_path: Path) -> None:
    from src.daily.aftermarket_book import run_aftermarket_book_capture
    from src.data.capture_contracts import CaptureStatus

    now = [_seoul(2026, 9, 29, 15, 30)]
    manifests = _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=_profile(tmp_path),
            store=_store(tmp_path),
            clients=[_StubClient()],
            dense_symbols=("000001",),
            sparse_symbols=("000001",),
            degraded_slots=("DATA_2",),
            now_fn=lambda: now[0],
        )
    )
    assert manifests
    assert {m.status for m in manifests} == {CaptureStatus.PARTIAL}


def _timed_runner(start: dt.datetime) -> tuple[dict[str, dt.datetime], Any]:
    state = {"now": start}

    async def _sleep(seconds: float) -> None:
        state["now"] += dt.timedelta(seconds=seconds)

    return state, _sleep


def test_sparse_rounds_append_research_after_sparse(tmp_path: Path) -> None:
    """Sparse rounds sweep the sorted cohort before research extras; dense rounds never do."""
    from src.daily.aftermarket_book import run_aftermarket_book_capture

    profile = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_SPARSE_TIMES=("170000",))
    store = _store(tmp_path)
    state, sleep = _timed_runner(_seoul(2026, 9, 29, 16, 0))
    seen: list[tuple[str, str]] = []

    def _handler(div: str | None, code: str) -> dict[str, Any]:
        seen.append((state["now"].strftime("%H%M"), code))
        return _listed_book()

    _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=profile,
            store=store,
            clients=[_StubClient(handler=_handler)],
            dense_symbols=("000001",),
            sparse_symbols=("000001", "000002"),
            research_symbols=("000009", "000003"),
            now_fn=lambda: state["now"],
            sleep_fn=sleep,
        )
    )

    sparse_codes = [code for at, code in seen if at == "1700"]
    assert sparse_codes == ["000001", "000002", "000009", "000003"] * 2
    dense_codes = {code for at, code in seen if at != "1700"}
    assert dense_codes == {"000001"}


def test_research_book_manifest_per_block_venue(tmp_path: Path) -> None:
    """Each (block, venue) that swept research symbols publishes one sibling right after its primary."""
    from src.daily.aftermarket_book import run_aftermarket_book_capture
    from src.data.capture_contracts import CaptureDataset

    profile = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_SPARSE_TIMES=("170000",), COLLECTION_AFTERMARKET_BOOK_FLUSH_ROUNDS=2)
    store = _store(tmp_path)
    state, sleep = _timed_runner(_seoul(2026, 9, 29, 16, 0))

    manifests = _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=profile,
            store=store,
            clients=[_StubClient()],
            dense_symbols=("000001",),
            sparse_symbols=("000001", "000002"),
            research_symbols=("000009", "000003"),
            now_fn=lambda: state["now"],
            sleep_fn=sleep,
        )
    )

    assert len(manifests) == 8
    research = [m for m in manifests if m.context.capture_reason == "aftermarket-book-research"]
    assert sorted(m.context.session for m in research) == ["krx_aftermarket", "nxt_aftermarket"]
    for sibling in research:
        index = manifests.index(sibling)
        primary = manifests[index - 1]
        assert primary.context.capture_reason == "aftermarket-book"
        assert primary.context.session == sibling.context.session
        assert sibling.context.run_id == primary.context.run_id + "-research"
        assert sibling.context.dataset == CaptureDataset.ORDERBOOK
        assert {e.symbol for e in sibling.entries} == {"000009", "000003"}
        assert {e.symbol for e in primary.entries} == {"000001", "000002"}


def test_empty_research_symbols_is_identity(tmp_path: Path) -> None:
    """Passing no research symbols produces exactly the pre-extras manifests."""
    from src.daily.aftermarket_book import run_aftermarket_book_capture

    def _signature(manifests: tuple[Any, ...]) -> list[Any]:
        return [
            (
                m.context.session,
                m.context.capture_reason,
                m.status.value,
                [(e.symbol, e.scheduled_at, e.status.value, e.reason, e.rows) for e in m.entries],
            )
            for m in manifests
        ]

    def _capture(root: Path, **kwargs: Any) -> tuple[Any, ...]:
        profile = _profile(root, COLLECTION_AFTERMARKET_BOOK_SPARSE_TIMES=("170000",))
        return _run(
            run_aftermarket_book_capture(
                "2026-09-29",
                profile=profile,
                store=_store(root),
                clients=[_StubClient()],
                dense_symbols=("000001",),
                sparse_symbols=("000001", "000002"),
                now_fn=lambda: _seoul(2026, 9, 29, 16, 0),
                **kwargs,
            )
        )

    baseline = _capture(tmp_path / "a")
    explicit_empty = _capture(tmp_path / "b", research_symbols=())

    assert _signature(explicit_empty) == _signature(baseline)
    assert not [m for m in explicit_empty if m.context.capture_reason.endswith("-research")]


def _failure_messages(caplog: Any) -> list[str]:
    return [record.getMessage() for record in caplog.records if "status=REQUEST_FAILED" in record.getMessage()]


def _venue_of(message: str) -> str:
    return message.split("venue=", 1)[1].split(" ", 1)[0]


def _book_run(
    tmp_path: Path,
    client: Any,
    *,
    symbols: tuple[str, ...] = ("000001", "000002"),
    flush_rounds: int = 15,
    clock: list[dt.datetime] | None = None,
    sleep_fn: Any = None,
    clients: list[Any] | None = None,
) -> tuple[Any, ...]:
    from src.daily.aftermarket_book import run_aftermarket_book_capture

    profile = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_FLUSH_ROUNDS=flush_rounds)
    now = clock if clock is not None else [_seoul(2026, 9, 29, 15, 30)]
    return _run(
        run_aftermarket_book_capture(
            "2026-09-29",
            profile=profile,
            store=_store(tmp_path),
            clients=clients or [client],
            dense_symbols=symbols,
            sparse_symbols=symbols,
            now_fn=lambda: now[0],
            sleep_fn=sleep_fn,
        )
    )


def _raise_boom(div: str | None, code: str) -> dict[str, Any]:
    raise RuntimeError("boom")


def test_exception_failure_logged_once_per_cause_with_traceback(tmp_path: Path, caplog: Any) -> None:
    """A repeated exception cause logs once per venue and key, with its traceback, and stays PARTIAL."""
    import logging

    from src.data.capture_contracts import CaptureStatus

    caplog.set_level(logging.WARNING, logger="src.daily.aftermarket_book")
    client = _StubClient(handler=lambda div, code: _raise_boom(div, code) if code == "000002" else dict(_listed_book()))
    manifests = _book_run(tmp_path, client)

    failures = _failure_messages(caplog)
    assert failures
    assert all("symbol=000002" in line and "error=RuntimeError" in line for line in failures)
    assert all("Traceback (most recent call last)" in line for line in failures)
    venues = [_venue_of(line) for line in failures]
    assert len(venues) == len(set(venues))
    entries = [e for m in manifests for e in m.entries if e.symbol == "000002"]
    assert entries
    assert all(e.status == CaptureStatus.PARTIAL and e.reason == "vendor_failure" for e in entries)


def test_distinct_symbols_same_cause_do_not_multiply_logs(tmp_path: Path, caplog: Any) -> None:
    """Many symbols failing with one cause log once per venue; the block line counts every failure."""
    import logging

    from src.config.market_session import KRX_CLOSE_MARKET_DIV_CODE, NXT_MARKET_DIV_CODE

    caplog.set_level(logging.INFO, logger="src.daily.aftermarket_book")
    client = _StubClient(handler=_raise_boom)
    _book_run(tmp_path, client, symbols=("000001", "000002", "000003", "000004", "000005"))

    failures = _failure_messages(caplog)
    venues = [_venue_of(line) for line in failures]
    assert len(venues) == len(set(venues)) >= 1
    counts: dict[str, int] = {}
    for div, _code in client.calls:
        venue = "KRX" if div == KRX_CLOSE_MARKET_DIV_CODE else "NXT" if div == NXT_MARKET_DIV_CODE else "?"
        counts[venue] = counts.get(venue, 0) + 1
    expected = ",".join(f"{venue}/RuntimeError={n}" for venue, n in sorted(counts.items()))
    block_lines = [r.getMessage() for r in caplog.records if "stage=aftermarket_book block=" in r.getMessage()]
    assert len(block_lines) == 1
    assert block_lines[0].endswith(f"request_errors={expected}")


def test_payload_failure_logged_with_msg_cd(tmp_path: Path, caplog: Any) -> None:
    """A non-zero rt_cd payload logs its vendor code without a traceback and stays vendor_failure."""
    import logging

    from src.data.capture_contracts import CaptureStatus

    caplog.set_level(logging.WARNING, logger="src.daily.aftermarket_book")
    coded = _StubClient(handler=lambda div, code: {"rt_cd": "1", "msg_cd": "EGW00201", "msg1": "busy"})
    manifests = _book_run(tmp_path, coded, symbols=("000001",))
    failures = _failure_messages(caplog)
    assert failures
    assert all("error=msg_cd:EGW00201" in line and "msg_cd=EGW00201" in line for line in failures)
    assert all("Traceback" not in line for line in failures)
    entries = [e for m in manifests for e in m.entries]
    assert all(e.status == CaptureStatus.PARTIAL and e.reason == "vendor_failure" for e in entries)

    caplog.clear()
    bare = _StubClient(handler=lambda div, code: {"rt_cd": "1", "msg1": "busy"})
    _book_run(tmp_path / "bare", bare, symbols=("000001",))
    failures = _failure_messages(caplog)
    assert failures
    assert all("error=msg_cd:none" in line for line in failures)


def test_different_venues_or_keys_log_separately(tmp_path: Path, caplog: Any) -> None:
    """Failures of one cause log once per (venue, key_id) pair."""
    import logging

    from src.api.kis.key_pool import kis_key_id

    caplog.set_level(logging.WARNING, logger="src.daily.aftermarket_book")
    first = _StubClient(handler=_raise_boom)
    first.app_key = "AK-FIRST"
    second = _StubClient(handler=_raise_boom)
    second.app_key = "AK-SECOND"
    _book_run(tmp_path, first, symbols=("000001", "000002"), clients=[first, second])

    failures = _failure_messages(caplog)
    pairs = [(_venue_of(line), line.split("key_id=", 1)[1].split(" ", 1)[0]) for line in failures]
    assert len(pairs) == len(set(pairs))
    assert {key for _venue, key in pairs} == {kis_key_id("AK-FIRST"), kis_key_id("AK-SECOND")}


def test_block_line_appends_request_errors_and_resets(tmp_path: Path, caplog: Any) -> None:
    """Block counts report only that block's failures, then reset for the next block."""
    import logging

    from src.config.market_session import KRX_CLOSE_MARKET_DIV_CODE, NXT_MARKET_DIV_CODE

    caplog.set_level(logging.INFO, logger="src.daily.aftermarket_book")
    clock = [_seoul(2026, 9, 29, 15, 30)]
    failed: list[str] = []

    def handler(div: str | None, code: str) -> dict[str, Any]:
        if clock[0] < _seoul(2026, 9, 29, 17, 0):
            failed.append(div or "")
            raise RuntimeError("boom")
        return dict(_listed_book())

    async def advance(seconds: float) -> None:
        clock[0] = clock[0] + dt.timedelta(seconds=seconds)

    client = _StubClient(handler=handler)
    _book_run(tmp_path, client, symbols=("000001",), flush_rounds=2, clock=clock, sleep_fn=advance)

    names = {KRX_CLOSE_MARKET_DIV_CODE: "KRX", NXT_MARKET_DIV_CODE: "NXT"}
    counts: dict[str, int] = {}
    for div in failed:
        counts[names[div]] = counts.get(names[div], 0) + 1
    expected = ",".join(f"{venue}/RuntimeError={n}" for venue, n in sorted(counts.items()))
    block_lines = [r.getMessage() for r in caplog.records if "stage=aftermarket_book block=" in r.getMessage()]
    assert len(block_lines) >= 2
    assert block_lines[0].startswith("[DATA] stage=aftermarket_book block=0 rounds=2 ")
    assert block_lines[0].endswith(f"request_errors={expected}")
    assert block_lines[1].endswith("request_errors=-")


def test_no_failure_no_extra_log(tmp_path: Path, caplog: Any) -> None:
    """A clean run emits no request-failure record and a `-` block count."""
    import logging

    caplog.set_level(logging.INFO, logger="src.daily.aftermarket_book")
    _book_run(tmp_path, _StubClient())
    assert _failure_messages(caplog) == []
    block_lines = [r.getMessage() for r in caplog.records if "stage=aftermarket_book block=" in r.getMessage()]
    assert block_lines and all(line.endswith("request_errors=-") for line in block_lines)


def test_log_contains_no_secrets(tmp_path: Path, caplog: Any, monkeypatch: Any) -> None:
    """Credentials never reach the failure record; the key appears only as its digest."""
    import logging

    from src.api.kis.key_pool import kis_key_id
    from src.tools import alerts

    caplog.set_level(logging.INFO, logger="src.daily.aftermarket_book")
    monkeypatch.setattr(alerts, "_alert_secret_values", lambda env=None: frozenset({"TOKEN-SECRET-XYZ"}))

    def handler(div: str | None, code: str) -> dict[str, Any]:
        raise RuntimeError("rejected token=TOKEN-SECRET-XYZ")

    client = _StubClient(handler=handler)
    client.app_key = "AK-SECRET-VALUE"
    _book_run(tmp_path, client, symbols=("000001",))

    failures = _failure_messages(caplog)
    assert failures
    assert all(kis_key_id("AK-SECRET-VALUE") in line for line in failures)
    for record in caplog.records:
        assert "AK-SECRET-VALUE" not in record.getMessage()
        assert "TOKEN-SECRET-XYZ" not in record.getMessage()


def test_second_run_in_same_process_logs_again(tmp_path: Path, caplog: Any) -> None:
    """Dedupe state is per run: an identical failure in a second run logs again."""
    import logging

    caplog.set_level(logging.WARNING, logger="src.daily.aftermarket_book")
    _book_run(tmp_path / "one", _StubClient(handler=_raise_boom), symbols=("000001",))
    first = len(_failure_messages(caplog))
    _book_run(tmp_path / "two", _StubClient(handler=_raise_boom), symbols=("000001",))
    second = len(_failure_messages(caplog)) - first
    assert first >= 1
    assert second == first
