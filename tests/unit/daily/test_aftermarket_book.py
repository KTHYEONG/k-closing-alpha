"""Aftermarket order-book capture invariant guards."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any


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

    dense, sparse = resolve_book_universes("2026-09-29", store=store, now=now, rank_pool_path=pool_path)

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
            "2026-09-29", store=store, now=now, rank_pool_path=tmp_path / "absent.parquet"
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
    dense, _ = resolve_book_universes("2026-09-29", store=store, now=now, rank_pool_path=empty_path)
    assert dense == ("000004",)

    drifted_path = tmp_path / "drifted.parquet"
    pd.DataFrame([{"weird": 1}]).to_parquet(drifted_path, index=False)
    dense, _ = resolve_book_universes("2026-09-29", store=store, now=now, rank_pool_path=drifted_path)
    assert dense == ("000004",)

    stale_path = tmp_path / "stale.parquet"
    _write_rank_pool(stale_path, [{"symbol": "000001", "decision_date": "2026-09-28", "pred": 0.5}])
    dense, _ = resolve_book_universes("2026-09-29", store=store, now=now, rank_pool_path=stale_path)
    assert dense == ("000004",)

    blank_path = tmp_path / "blank.parquet"
    _write_rank_pool(blank_path, [{"symbol": "", "decision_date": "2026-09-29", "pred": 0.5}])
    dense, _ = resolve_book_universes("2026-09-29", store=store, now=now, rank_pool_path=blank_path)
    assert dense == ("000004",)


def test_positions_unavailable_yields_empty_dense(tmp_path: Path, monkeypatch: Any) -> None:
    """Unreadable ledgers degrade to an empty dense universe."""

    class _BrokenLedger:
        def load_open_positions(self) -> Any:
            raise OSError("locked")

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", _BrokenLedger)
    store = _store(tmp_path)
    now = _seoul(2026, 9, 29, 15, 35)

    from src.daily.aftermarket_book import resolve_book_universes

    dense, sparse = resolve_book_universes("2026-09-29", store=store, now=now, rank_pool_path=tmp_path / "absent.parquet")

    assert dense == ()
    assert sparse == ()


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

    assert _run(book._run_async("2026-09-23", profile)) is None


def test_main_skips_cleanly(tmp_path: Path, monkeypatch: Any) -> None:
    """Disabled or non-standard days skip cleanly."""
    import src.daily.aftermarket_book as book
    from src.data.capture_contracts import SessionClock

    class _ExplodingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("client must not be constructed on skip paths")

    monkeypatch.setattr("src.api.kis.client.KisApiClient", _ExplodingClient)

    monkeypatch.setattr(book, "CollectionSettings", lambda *args: _profile(tmp_path))
    book.main(["--date", "2026-09-29"])

    shifted = SessionClock(
        trading_date=dt.date(2026, 9, 29),
        open_at=_seoul(2026, 9, 29, 9, 0),
        close_at=_seoul(2026, 9, 29, 15, 30),
        close_confirmation_deadline=_seoul(2026, 9, 29, 15, 33),
        provenance="test",
    )
    enabled = _profile(
        tmp_path,
        COLLECTION_AFTERMARKET_BOOK_ENABLED=True,
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

    enabled = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_ENABLED=True)
    monkeypatch.setattr(book, "CollectionSettings", lambda *args: enabled)

    with pytest.raises(ValueError, match="Invalid snapshot_date"):
        book.main(["--date", "not-a-date"])


def test_main_skips_closed_sessions(tmp_path: Path, monkeypatch: Any) -> None:
    """CLOSED sessions skip before credential resolution."""
    import src.daily.aftermarket_book as book
    from src.data.session_calendar import SessionDay, SessionKind

    enabled = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_ENABLED=True)
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
    enabled = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_ENABLED=True)
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

    enabled = _profile(tmp_path, COLLECTION_AFTERMARKET_BOOK_ENABLED=True)
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
