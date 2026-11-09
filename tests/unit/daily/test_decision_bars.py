from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import src.daily.decision_bars as db
from src.data.capture_contracts import SEOUL

DECISION = pd.Timestamp("2026-09-09")


def _candle(ts: str, price: float = 10_000.0) -> dict[str, Any]:
    return {
        "timestamp": ts,
        "openPrice": price,
        "highPrice": price + 5,
        "lowPrice": price - 5,
        "closePrice": price,
        "volume": 100.0,
    }


class FakeClient:
    def __init__(self, pages: dict[str, list[list[dict[str, Any]]]], fail: set[str] | None = None) -> None:
        self.pages = {k: list(v) for k, v in pages.items()}
        self.fail = fail or set()
        self.calls: list[tuple[str, str]] = []

    async def get_candles(self, session, symbol, *, interval, count, before, adjusted):
        assert interval == "1m" and count == db.TOSS_DECISION_PAGE_COUNT and adjusted is False
        self.calls.append((symbol, before))
        if symbol in self.fail:
            raise RuntimeError("boom")
        queue = self.pages.get(symbol, [])
        candles = queue.pop(0) if queue else []
        return {"result": {"candles": candles}}


def _clock(hh: int, mm: int, ss: int = 0):
    return lambda: datetime(2026, 9, 9, hh, mm, ss, tzinfo=SEOUL)


def _run(client: FakeClient, symbols: list[str], now_fn=None) -> pd.DataFrame:
    clock = now_fn if now_fn is not None else _clock(15, 21)
    return asyncio.run(db.fetch_decision_bars(client, object(), symbols, DECISION, now_fn=clock))


def test_fetch_keeps_only_decision_day_labels_up_to_1520() -> None:
    page = [
        _candle("2026-09-09T15:30:00+09:00"),
        _candle("2026-09-09T15:21:00+09:00"),
        _candle("2026-09-09T15:20:00+09:00"),
        _candle("2026-09-09T15:19:00+09:00"),
        _candle("2026-09-09T09:01:00+09:00"),
        _candle("2026-09-09T09:00:00+09:00"),
        _candle("2026-09-08T15:30:00+09:00"),
    ]
    client = FakeClient({"000001": [page]})

    bars = _run(client, ["000001"])

    assert bars["end_min"].tolist() == [541, 919, 920]
    assert bars["end_min"].max() == 920
    assert set(bars["vendor"]) == {"toss"}
    assert list(bars.columns) == list(db.DECISION_BARS_COLUMNS)


def test_fetch_walks_pages_until_previous_day() -> None:
    page1 = [_candle("2026-09-09T15:20:00+09:00"), _candle("2026-09-09T15:19:00+09:00")]
    page2 = [_candle("2026-09-09T15:18:00+09:00"), _candle("2026-09-08T15:30:00+09:00")]
    client = FakeClient({"000001": [page1, page2]})

    bars = _run(client, ["000001"])

    assert len(client.calls) == 2
    assert client.calls[0][1] == "2026-09-09T15:21:00+09:00"
    assert client.calls[1][1] == "2026-09-09T15:19:00+09:00"
    assert bars["end_min"].tolist() == [918, 919, 920]


def test_fetch_before_1520_raises_without_client_calls() -> None:
    client = FakeClient({"000001": [[_candle("2026-09-09T15:19:00+09:00")]]})

    with pytest.raises(ValueError, match="not final"):
        _run(client, ["000001"], now_fn=_clock(15, 19, 59))
    assert client.calls == []


def test_fetch_symbol_failure_is_isolated(caplog) -> None:
    ok = [_candle("2026-09-09T15:20:00+09:00")]
    client = FakeClient({"000001": [ok], "000002": [ok]}, fail={"000002"})

    with caplog.at_level(logging.WARNING, logger=db.logger.name):
        bars = _run(client, ["000001", "000002"])

    assert bars["symbol"].tolist() == ["000001"]
    assert bars["end_min"].tolist() == [920]
    assert any(
        "[DATA] stage=decision_bars symbol=000002 status=FAILED reason=RuntimeError" in r.getMessage()
        for r in caplog.records
    )


def test_persist_is_atomic_and_overwrites(tmp_path) -> None:
    first = pd.DataFrame({"symbol": ["000001"], "end_min": [920]})
    second = pd.DataFrame({"symbol": ["000001", "000002"], "end_min": [919, 920]})

    db.persist_decision_bars(DECISION, first, root=tmp_path)
    path = db.persist_decision_bars(DECISION, second, root=tmp_path)

    assert path == tmp_path / "2026-09-09.parquet"
    assert pd.read_parquet(path).equals(second)


def test_client_builder_rejects_missing_key(monkeypatch) -> None:
    monkeypatch.setattr(db, "TossApiClient", lambda: SimpleNamespace(app_key=""))

    with pytest.raises(ValueError, match="app key"):
        db.build_decision_bars_client()


def test_fetch_malformed_payload_yields_zero_rows_for_that_symbol(caplog) -> None:
    class MalformedClient(FakeClient):
        async def get_candles(self, session, symbol, **kwargs):
            return {"error": {"code": "stock-not-found"}}

    with caplog.at_level(logging.WARNING, logger=db.logger.name):
        bars = _run(MalformedClient({}), ["000001"])

    assert bars.empty
    assert list(bars.columns) == list(db.DECISION_BARS_COLUMNS)
    assert any("symbol=000001 status=FAILED reason=ValueError" in r.getMessage() for r in caplog.records)


def test_fetch_without_decision_day_labels_yields_zero_rows() -> None:
    client = FakeClient({"000001": [[_candle("2026-09-08T15:20:00+09:00")]]})

    bars = _run(client, ["000001"])

    assert bars.empty


def test_fetch_rejects_clock_on_another_date() -> None:
    client = FakeClient({})

    with pytest.raises(ValueError, match="not on decision date"):
        _run(client, ["000001"], now_fn=lambda: datetime(2026, 9, 10, 15, 30, tzinfo=SEOUL))
    assert client.calls == []


def test_fetch_with_no_symbols_returns_empty_frame() -> None:
    bars = _run(FakeClient({}), [])

    assert bars.empty
    assert list(bars.columns) == list(db.DECISION_BARS_COLUMNS)


def test_default_clock_is_seoul_aware() -> None:
    now = db._seoul_now()

    assert now.utcoffset() == SEOUL.utcoffset(now)


def test_client_builder_returns_client_when_key_present(monkeypatch) -> None:
    client = SimpleNamespace(app_key="configured")
    monkeypatch.setattr(db, "TossApiClient", lambda: client)

    assert db.build_decision_bars_client() is client
