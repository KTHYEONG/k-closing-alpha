"""Invariant guards for the closing-auction indicative path capture."""

from __future__ import annotations

import ast
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.api.dbsec.client import DbsecTokenThrottled
from src.config.collection import CollectionSettings
from src.daily import auction_path
from src.data import auction_path_store
from src.data.capture_contracts import (
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    build_cohort,
)
from src.data.capture_store import CaptureStore
from src.data.session_calendar import SessionDay, SessionKind

SEOUL = ZoneInfo("Asia/Seoul")
DATE = "2026-10-15"
READY_NOW = datetime(2026, 10, 15, 16, 0, tzinfo=SEOUL)


def _row(hour: str, price: str = "70000", buy: str = "0", sell: str = "0") -> dict[str, str]:
    return {"Hour": hour, "Prpr": price, "ShnuCnqn": buy, "SelnCnqn": sell}


class _FakeClient:
    def __init__(self, behaviors: dict[str, dict[str, Any]]) -> None:
        self.behaviors = behaviors
        self.calls: list[str] = []

    async def get_auction_path(self, session: Any, code: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(code)
        behavior = self.behaviors[code]
        if behavior.get("raise") is not None:
            raise behavior["raise"]
        rows: list[dict[str, str]] = list(behavior.get("rows", []))
        pages = int(behavior.get("pages", 1))
        on_page = kwargs["on_page"]
        chunks = [rows[i::pages] for i in range(pages)] if rows else [[] for _ in range(pages)]
        for index, chunk in enumerate(chunks):
            on_page(
                {"rsp_cd": "00000", "Out": chunk},
                {"cont_yn": "Y" if index + 1 < pages else "N", "cont_key": ""},
                READY_NOW,
                READY_NOW,
                index,
                0,
            )
        return {
            "rt_cd": behavior.get("rt_cd", "0"),
            "msg1": behavior.get("msg1", ""),
            "output2": rows,
            "termination_reason": behavior.get("termination", "crossed_time_floor"),
            "pages_fetched": pages,
        }


def _live_settings(monkeypatch: Any, tmp_path: Path, **overrides: Any) -> None:
    values: dict[str, Any] = {
        "DB_AUCTION_PATH_ENABLED": True,
        "DB_APP_KEY": "test-key",
        "DB_AUCTION_PATH_MAX_PAGES": 60,
        "DB_AUCTION_PATH_DEADLINE_HHMMSS": "190000",
        "DB_AUCTION_PATH_CONCURRENCY": 2,
        "HISTORY_DIR": tmp_path / "history",
    }
    values.update(overrides)
    for key, value in values.items():
        monkeypatch.setattr(auction_path.settings, key, value)


def _standard_day(monkeypatch: Any) -> None:
    def _resolve(trading_date: date, overrides: Any = None) -> SessionDay:
        return SessionDay(trading_date=trading_date, kind=SessionKind.STANDARD, clock=None, provenance="test")

    monkeypatch.setattr(auction_path, "resolve_session_day", _resolve)


def _recorded(monkeypatch: Any) -> list[tuple[str, str, dict[str, Any]]]:
    calls: list[tuple[str, str, dict[str, Any]]] = []
    monkeypatch.setattr(
        auction_path, "record_run_outcome", lambda job, outcome, **kw: calls.append((job, outcome, kw)) or {}
    )
    return calls


def _seed_store(tmp_path: Path, symbols: list[str]) -> CaptureStore:
    store = CaptureStore(tmp_path / "capture")
    trading_day = date.fromisoformat(DATE)
    cohort = build_cohort(
        trading_day,
        scanned_symbols=symbols,
        eligible_symbols=symbols,
        rejections={},
        eligibility_rule_version="v1",
    )
    store.publish_manifest(
        CaptureManifest(
            schema_version=1,
            context=CaptureContext(
                trading_date=trading_day,
                run_id="seed-1",
                dataset=CaptureDataset.SCAN,
                vendor="kis",
                endpoint="seed",
                symbol=None,
                venue="KRX",
                session="regular",
                capture_reason="seed",
                cohort_id=cohort.cohort_id,
                scheduled_at=None,
            ),
            cohort=cohort,
            completed_at=READY_NOW,
            entries=(),
            artifacts=(),
            status=CaptureStatus.COMPLETE,
        )
    )
    return store


def _run(monkeypatch: Any, tmp_path: Path, client: _FakeClient, symbols: list[str], **kw: Any):
    _live_settings(monkeypatch, tmp_path)
    store = _seed_store(tmp_path, symbols)
    manifest = auction_path.run_auction_path_capture(
        snapshot_date=kw.get("snapshot_date", DATE),
        profile=CollectionSettings(),
        client=client,  # type: ignore[arg-type]
        session=object(),
        store=store,
        now=kw.get("now", lambda: READY_NOW),
    )
    import asyncio

    return asyncio.run(manifest)


def test_auction_path_skips_when_disabled_or_without_key(tmp_path, monkeypatch) -> None:
    _live_settings(monkeypatch, tmp_path, DB_AUCTION_PATH_ENABLED=False)
    _standard_day(monkeypatch)
    recorded = _recorded(monkeypatch)
    monkeypatch.setattr(auction_path, "_run_async", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no run")))

    auction_path.main(["--date", DATE])
    assert recorded and recorded[0][1] == "SKIPPED" and recorded[0][2]["reason"] == "disabled"

    _live_settings(monkeypatch, tmp_path, DB_APP_KEY="")
    recorded.clear()
    auction_path.main(["--date", DATE])
    assert recorded and recorded[0][1] == "SKIPPED" and recorded[0][2]["reason"] == "no_credentials"


def test_auction_path_rejects_non_today_date(tmp_path, monkeypatch) -> None:
    _live_settings(monkeypatch, tmp_path)
    _standard_day(monkeypatch)
    _recorded(monkeypatch)
    client = _FakeClient({})
    store = _seed_store(tmp_path, ["000001"])

    async def _fake_async(snapshot_date: str, profile: Any) -> Any:
        return await auction_path.run_auction_path_capture(
            snapshot_date=snapshot_date,
            profile=profile,
            client=client,  # type: ignore[arg-type]
            session=object(),
            store=store,
            now=lambda: READY_NOW,
        )

    monkeypatch.setattr(auction_path, "_run_async", _fake_async)
    with pytest.raises(ValueError, match="today"):
        auction_path.main(["--date", "2026-10-14"])
    assert client.calls == []


def test_auction_path_rejects_run_before_ready_time(tmp_path, monkeypatch) -> None:
    _live_settings(monkeypatch, tmp_path)
    client = _FakeClient({})
    store = _seed_store(tmp_path, ["000001"])
    early = datetime(2026, 10, 15, 15, 40, tzinfo=SEOUL)

    import asyncio

    with pytest.raises(ValueError, match="too_early"):
        asyncio.run(
            auction_path.run_auction_path_capture(
                snapshot_date=DATE,
                profile=CollectionSettings(),
                client=client,  # type: ignore[arg-type]
                session=object(),
                store=store,
                now=lambda: early,
            )
        )
    assert client.calls == []


def test_auction_path_writes_only_complete_symbols(tmp_path, monkeypatch) -> None:
    _live_settings(monkeypatch, tmp_path)
    recorded = _recorded(monkeypatch)
    store = _seed_store(tmp_path, ["000001", "000002"])
    client = _FakeClient(
        {
            "000001": {
                "termination": "crossed_time_floor",
                "rows": [_row("152500"), _row("153001", sell="100")],
            },
            "000002": {"termination": "page_budget", "rows": [_row("152500")]},
        }
    )

    import asyncio

    manifest = asyncio.run(
        auction_path.run_auction_path_capture(
            snapshot_date=DATE,
            profile=CollectionSettings(),
            client=client,  # type: ignore[arg-type]
            session=object(),
            store=store,
            now=lambda: READY_NOW,
        )
    )
    by_symbol = {entry.symbol: entry for entry in manifest.entries}
    assert by_symbol["000001"].status == CaptureStatus.COMPLETE
    assert by_symbol["000002"].status == CaptureStatus.PARTIAL
    assert by_symbol["000002"].reason == "incomplete:page_budget"
    assert manifest.status == CaptureStatus.PARTIAL
    stored = pd.read_parquet(auction_path_store.auction_path_partition_path(DATE))
    assert sorted(stored["symbol"].unique().tolist()) == ["000001"]
    assert recorded and recorded[0][1] == "DEGRADED"


def test_auction_path_halted_symbol_is_unknown_not_complete(tmp_path, monkeypatch) -> None:
    _live_settings(monkeypatch, tmp_path)
    _recorded(monkeypatch)
    store = _seed_store(tmp_path, ["000001"])
    client = _FakeClient({"000001": {"termination": "crossed_time_floor", "rows": [_row("151900", buy="3")]}})

    import asyncio

    manifest = asyncio.run(
        auction_path.run_auction_path_capture(
            snapshot_date=DATE,
            profile=CollectionSettings(),
            client=client,  # type: ignore[arg-type]
            session=object(),
            store=store,
            now=lambda: READY_NOW,
        )
    )
    assert manifest.entries[0].status == CaptureStatus.UNKNOWN
    assert manifest.entries[0].reason == "no_indicative_rows"


def test_auction_path_token_throttle_fails_remaining_without_retry(tmp_path, monkeypatch) -> None:
    _live_settings(monkeypatch, tmp_path)
    _recorded(monkeypatch)
    store = _seed_store(tmp_path, ["000001", "000002"])
    client = _FakeClient(
        {
            "000001": {"raise": DbsecTokenThrottled("throttled")},
            "000002": {"termination": "crossed_time_floor", "rows": [_row("152500")]},
        }
    )

    import asyncio

    manifest = asyncio.run(
        auction_path.run_auction_path_capture(
            snapshot_date=DATE,
            profile=CollectionSettings(),
            client=client,  # type: ignore[arg-type]
            session=object(),
            store=store,
            now=lambda: READY_NOW,
        )
    )
    assert [entry.status for entry in manifest.entries] == [CaptureStatus.FAILED, CaptureStatus.FAILED]
    assert all(entry.reason == "token_throttled" for entry in manifest.entries)
    assert client.calls == ["000001"]


def test_auction_path_raw_pages_persisted_for_every_symbol(tmp_path, monkeypatch) -> None:
    _live_settings(monkeypatch, tmp_path)
    _recorded(monkeypatch)
    store = _seed_store(tmp_path, ["000001", "000002"])
    client = _FakeClient(
        {
            "000001": {"termination": "crossed_time_floor", "pages": 3, "rows": [_row("152500")]},
            "000002": {"termination": "crossed_time_floor", "pages": 3, "rows": [_row("152501")]},
        }
    )

    import asyncio

    manifest = asyncio.run(
        auction_path.run_auction_path_capture(
            snapshot_date=DATE,
            profile=CollectionSettings(),
            client=client,  # type: ignore[arg-type]
            session=object(),
            store=store,
            now=lambda: READY_NOW,
        )
    )
    assert len(manifest.artifacts) == 6
    assert all(len(entry.raw_refs) == 3 for entry in manifest.entries)
    for ref in manifest.artifacts:
        envelope = store.read_artifact(ref)
        assert envelope["context"]["dataset"] == CaptureDataset.AUCTION_PATH.value


def test_decision_modules_never_import_auction_path() -> None:
    root = Path(__file__).resolve().parents[3] / "src" / "daily"
    hits: list[str] = []
    for name in ("collect", "predict", "finalize_close", "order_session", "paper_trade"):
        tree = ast.parse((root / f"{name}.py").read_text(encoding="utf-8"))
        targets = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        targets.extend(
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        )
        hits.extend(
            f"{name}: {target}"
            for target in targets
            if target in ("src.daily.auction_path", "src.data.auction_path_store")
            or target.startswith(("src.daily.auction_path.", "src.data.auction_path_store."))
        )
    assert hits == []


def _run_direct(monkeypatch, tmp_path, client: _FakeClient, symbols: list[str], **kw):
    import asyncio

    _live_settings(monkeypatch, tmp_path)
    _recorded(monkeypatch)
    store = _seed_store(tmp_path, symbols)
    return asyncio.run(
        auction_path.run_auction_path_capture(
            snapshot_date=kw.get("snapshot_date", DATE),
            profile=CollectionSettings(),
            client=client,  # type: ignore[arg-type]
            session=object(),
            store=store,
            now=kw.get("now", lambda: READY_NOW),
        )
    )


def test_auction_path_rejects_invalid_snapshot_date(tmp_path, monkeypatch) -> None:
    import asyncio

    _live_settings(monkeypatch, tmp_path)
    store = _seed_store(tmp_path, ["000001"])
    with pytest.raises(ValueError, match="Invalid snapshot_date"):
        asyncio.run(
            auction_path.run_auction_path_capture(
                snapshot_date="not-a-date",
                profile=CollectionSettings(),
                client=_FakeClient({}),  # type: ignore[arg-type]
                session=object(),
                store=store,
                now=lambda: READY_NOW,
            )
        )


def test_auction_path_vendor_failure_and_exhaustion(tmp_path, monkeypatch) -> None:
    manifest = _run_direct(
        monkeypatch,
        tmp_path,
        _FakeClient(
            {
                "000001": {"raise": RuntimeError("boom")},
                "000002": {"rt_cd": "1", "msg1": "denied", "termination": "vendor_failure", "rows": []},
                "000003": {"termination": "exhausted", "rows": [_row("152900")]},
                "000004": {"termination": "crossed_time_floor", "rows": [_row("152500", price="1.5")]},
            }
        ),
        ["000001", "000002", "000003", "000004"],
    )
    by_symbol = {entry.symbol: entry for entry in manifest.entries}
    assert by_symbol["000001"].status == CaptureStatus.FAILED
    assert by_symbol["000001"].reason == "RuntimeError"
    assert by_symbol["000002"].status == CaptureStatus.FAILED
    assert by_symbol["000003"].status == CaptureStatus.UNKNOWN
    assert by_symbol["000003"].reason == "exhausted_before_floor"
    assert by_symbol["000004"].status == CaptureStatus.FAILED
    assert manifest.status == CaptureStatus.PARTIAL


def test_auction_path_main_rejects_invalid_date(tmp_path, monkeypatch) -> None:
    _live_settings(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="Invalid snapshot_date"):
        auction_path.main(["--date", "bad-date"])


def test_auction_path_main_skips_non_trading_day(tmp_path, monkeypatch) -> None:
    _live_settings(monkeypatch, tmp_path)
    recorded = _recorded(monkeypatch)
    monkeypatch.setattr(
        auction_path,
        "resolve_session_day",
        lambda trading_date, overrides=None: SessionDay(
            trading_date=trading_date, kind=SessionKind.CLOSED, clock=None, provenance="test"
        ),
    )
    monkeypatch.setattr(auction_path, "_run_async", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no run")))

    auction_path.main(["--date", DATE])
    assert recorded and recorded[0][1] == "SKIPPED" and recorded[0][2]["reason"] == "non_trading_day"


def test_auction_path_main_runs_live_bindings(tmp_path, monkeypatch) -> None:
    import asyncio
    from datetime import date as _date

    _live_settings(monkeypatch, tmp_path)
    _standard_day(monkeypatch)
    _recorded(monkeypatch)
    profile = CollectionSettings(COLLECTION_ROOT=tmp_path / "capture")
    monkeypatch.setattr(auction_path, "CollectionSettings", lambda: profile)
    seen: dict[str, Any] = {}

    async def _fake_run(**kwargs: Any) -> Any:
        seen["dataset"] = kwargs["client"].__class__.__name__
        seen["has_session"] = kwargs["session"] is not None
        today = _date.fromisoformat(DATE)

        return CaptureManifest(
            schema_version=1,
            context=CaptureContext(
                trading_date=today,
                run_id="run-1",
                dataset=CaptureDataset.AUCTION_PATH,
                vendor="dbsec",
                endpoint="daily-price-8",
                symbol=None,
                venue="KRX",
                session="regular",
                capture_reason="auction-path",
                cohort_id=None,
                scheduled_at=None,
            ),
            cohort=None,
            completed_at=READY_NOW,
            entries=(),
            artifacts=(),
            status=CaptureStatus.COMPLETE,
        )

    monkeypatch.setattr(auction_path, "run_auction_path_capture", _fake_run)
    auction_path.main(["--date", DATE])
    assert seen["dataset"] == "DbsecApiClient"
    assert seen["has_session"] is True
    assert asyncio.run(auction_path._run_async(DATE, profile)).status == CaptureStatus.COMPLETE


def test_auction_path_overlaps_symbols_up_to_concurrency_bound(tmp_path, monkeypatch) -> None:
    import asyncio

    _live_settings(monkeypatch, tmp_path, DB_AUCTION_PATH_CONCURRENCY=2)
    _recorded(monkeypatch)
    symbols = ["000001", "000002", "000003", "000004"]
    store = _seed_store(tmp_path, symbols)
    state = {"inflight": 0, "peak": 0}

    class _SlowClient:
        async def get_auction_path(self, session: Any, code: str, **kwargs: Any) -> dict[str, Any]:
            state["inflight"] += 1
            state["peak"] = max(state["peak"], state["inflight"])
            await asyncio.sleep(0)
            state["inflight"] -= 1
            return {"rt_cd": "0", "output2": [_row("152500")], "termination_reason": "crossed_time_floor", "pages_fetched": 1}

    manifest = asyncio.run(
        auction_path.run_auction_path_capture(
            snapshot_date=DATE,
            profile=CollectionSettings(),
            client=_SlowClient(),  # type: ignore[arg-type]
            session=object(),
            store=store,
            now=lambda: READY_NOW,
        )
    )
    assert state["peak"] == 2
    assert [entry.symbol for entry in manifest.entries] == symbols
    assert all(entry.status == CaptureStatus.COMPLETE for entry in manifest.entries)
