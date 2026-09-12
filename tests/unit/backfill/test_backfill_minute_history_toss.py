from __future__ import annotations


def test_enumerate_toss_gap_targets_filters_to_window(monkeypatch) -> None:
    import src.backfill.intraday.backfill_minute_history_toss as mod

    monkeypatch.setattr(
        mod, "enumerate_backfill_targets",
        lambda as_of=None, lookback_days=365, include_exit_day=True: [
            ("2022-08-31", "000001"),
            ("2022-09-01", "000002"),
            ("2024-01-01", "000003"),
            ("2025-09-01", "000004"),
            ("2025-09-02", "000005"),
        ],
    )

    out = mod.enumerate_toss_gap_targets()

    assert out == [("2022-09-01", "000002"), ("2024-01-01", "000003"), ("2025-09-01", "000004")]


def test_fetch_toss_regular_session_bars_pages_and_filters_to_regular_hours() -> None:
    import asyncio

    from src.backfill.intraday.backfill_minute_history_toss import _fetch_toss_regular_session_bars

    class FakeToss:
        def __init__(self):
            self.calls: list[dict] = []

        async def get_candles(self, session, symbol, *, interval="1m", count=200, before=None, adjusted=None):
            self.calls.append({"symbol": symbol, "before": before})
            if before == "2024-02-28T15:30:00.000+09:00":
                return {"result": {"candles": [
                    {"timestamp": "2024-02-28T15:30:00.000+09:00", "openPrice": "70000", "highPrice": "70100", "lowPrice": "69900", "closePrice": "70000", "volume": "10"},
                    {"timestamp": "2024-02-28T12:11:00.000+09:00", "openPrice": "69000", "highPrice": "69100", "lowPrice": "68900", "closePrice": "69000", "volume": "20"},
                ]}}
            if before == "2024-02-28T12:11:00.000+09:00":
                return {"result": {"candles": [
                    {"timestamp": "2024-02-28T12:11:00.000+09:00", "openPrice": "69000", "highPrice": "69100", "lowPrice": "68900", "closePrice": "69000", "volume": "20"},
                    {"timestamp": "2024-02-28T08:52:00.000+09:00", "openPrice": "68000", "highPrice": "68100", "lowPrice": "67900", "closePrice": "68000", "volume": "5"},
                ]}}
            return {"result": {"candles": []}}

    toss = FakeToss()
    out = asyncio.run(_fetch_toss_regular_session_bars(toss, object(), "000250", "2024-02-28"))

    # Then: the 08:52 pre-market bar is excluded (before the 09:00 regular-session floor).
    # The 12:11 boundary bar is returned by BOTH pages (Toss's `before` cursor is inclusive),
    # but _fetch_toss_regular_session_bars dedupes on (symbol, ts_hms) itself -- relying on
    # write_intraday_partition's merge-time dedup is NOT enough, since merge_partition_frame
    # only dedupes new-vs-existing rows and does nothing for duplicates WITHIN a first write
    # (no pre-existing partition file yet), as verified empirically for this contract.
    assert sorted(out["ts_hms"].tolist()) == [121100, 153000]
    assert toss.calls == [
        {"symbol": "000250", "before": "2024-02-28T15:30:00.000+09:00"},
        {"symbol": "000250", "before": "2024-02-28T12:11:00.000+09:00"},
    ]


def test_run_toss_1m_backfill_writes_partition_and_skips_existing(tmp_path, monkeypatch) -> None:
    import src.backfill.intraday.backfill_minute_history_toss as mod
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(
        mod, "enumerate_backfill_targets",
        lambda as_of=None, lookback_days=365, include_exit_day=True: [("2024-02-28", "000250"), ("2024-02-28", "004840")],
    )

    class FakeToss:
        def __init__(self):
            self.calls: list[str] = []

        async def get_candles(self, session, symbol, *, interval="1m", count=200, before=None, adjusted=None):
            self.calls.append(symbol)
            return {"result": {"candles": [
                {"timestamp": "2024-02-28T15:30:00.000+09:00", "openPrice": "70000", "highPrice": "70100", "lowPrice": "69900", "closePrice": "70000", "volume": "10"},
            ]}}

    toss = FakeToss()
    result = mod.run_toss_1m_backfill(gap_start="2024-01-01", gap_end="2024-12-31", toss=toss)

    # Then: 1 date written, 2 unique (symbol, ts_hms) rows (one per code) -- each code fires
    # 2 raw Toss calls (page1 always, page2 since page1 returned a candle), but the intra-batch
    # dedup in _fetch_toss_regular_session_bars collapses the identical repeated bar to 1 row/code.
    assert result == {"dates": 1, "rows": 2}
    assert sorted(toss.calls) == ["000250", "000250", "004840", "004840"]

    # When: re-run after the partition already has these codes -- must skip re-fetching.
    toss2 = FakeToss()
    result2 = mod.run_toss_1m_backfill(gap_start="2024-01-01", gap_end="2024-12-31", toss=toss2)

    assert result2 == {"dates": 1, "rows": 0}
    assert toss2.calls == []


def test_run_toss_1m_backfill_is_fail_soft_per_symbol(tmp_path, monkeypatch) -> None:
    import src.backfill.intraday.backfill_minute_history_toss as mod
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(
        mod, "enumerate_backfill_targets",
        lambda as_of=None, lookback_days=365, include_exit_day=True: [("2024-02-28", "000250"), ("2024-02-28", "999999")],
    )

    class FlakyToss:
        async def get_candles(self, session, symbol, *, interval="1m", count=200, before=None, adjusted=None):
            if symbol == "999999":
                raise ConnectionError("delisted or network error")
            return {"result": {"candles": [
                {"timestamp": "2024-02-28T15:30:00.000+09:00", "openPrice": "70000", "highPrice": "70100", "lowPrice": "69900", "closePrice": "70000", "volume": "10"},
            ]}}

    # Then: one symbol failing must not abort the date -- the healthy symbol's row is still written.
    result = mod.run_toss_1m_backfill(gap_start="2024-01-01", gap_end="2024-12-31", toss=FlakyToss())

    assert result == {"dates": 1, "rows": 1}


# --- Diff-coverage supplements (not contract scenarios): exercise the
# contract-required early-return / lazy-init / entrypoint branches. ---


def test_toss_backfill_fetch_returns_empty_when_no_candles() -> None:
    import asyncio

    from src.backfill.intraday.backfill_minute_history_toss import _fetch_toss_regular_session_bars

    class EmptyToss:
        async def get_candles(self, session, symbol, *, interval="1m", count=200, before=None, adjusted=None):
            return {"result": {"candles": []}}

    out = asyncio.run(_fetch_toss_regular_session_bars(EmptyToss(), object(), "000250", "2024-02-28"))

    assert out.empty


def test_toss_backfill_no_targets_returns_zero_counts(monkeypatch) -> None:
    import src.backfill.intraday.backfill_minute_history_toss as mod

    monkeypatch.setattr(
        mod, "enumerate_backfill_targets",
        lambda as_of=None, lookback_days=365, include_exit_day=True: [],
    )

    assert mod.run_toss_1m_backfill(gap_start="2024-01-01", gap_end="2024-12-31", toss=object()) == {"dates": 0, "rows": 0}


def test_toss_backfill_lazy_client_init(tmp_path, monkeypatch) -> None:
    import src.api.toss.client as toss_client_mod
    import src.backfill.intraday.backfill_minute_history_toss as mod
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)
    monkeypatch.setattr(
        mod, "enumerate_backfill_targets",
        lambda as_of=None, lookback_days=365, include_exit_day=True: [("2024-02-28", "000250")],
    )

    class FakeToss:
        async def get_candles(self, session, symbol, *, interval="1m", count=200, before=None, adjusted=None):
            return {"result": {"candles": [
                {"timestamp": "2024-02-28T15:30:00.000+09:00", "openPrice": "70000", "highPrice": "70100", "lowPrice": "69900", "closePrice": "70000", "volume": "10"},
            ]}}

    monkeypatch.setattr(toss_client_mod, "TossApiClient", lambda *args, **kwargs: FakeToss())

    result = mod.run_toss_1m_backfill(gap_start="2024-01-01", gap_end="2024-12-31", toss=None)

    assert result == {"dates": 1, "rows": 1}


def test_toss_backfill_main_entrypoint(monkeypatch) -> None:
    import runpy

    import src.backfill.intraday.backfill_minute_history as bh

    monkeypatch.setattr(
        bh, "enumerate_backfill_targets",
        lambda as_of=None, lookback_days=365, include_exit_day=True: [],
    )

    runpy.run_module("src.backfill.intraday.backfill_minute_history_toss", run_name="__main__")

