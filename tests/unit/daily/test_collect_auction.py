from __future__ import annotations



def test_run_auction_capture_polls_watchlist_until_window_close(monkeypatch) -> None:
    from datetime import datetime

    import pandas as pd

    from src.daily import collect_auction

    monkeypatch.setattr(
        collect_auction.archive,
        "fetch_archive_snapshot",
        lambda *a, **kw: pd.DataFrame({"종목코드": ["005930", "000660"]}),
    )

    appended: list[dict] = []
    monkeypatch.setattr(
        collect_auction,
        "append_orderbook_snapshots",
        lambda rows, snapshot_date: appended.extend(rows) or len(rows),
    )

    clock = iter(
        [
            datetime(2026, 9, 4, 15, 20, 0),
            datetime(2026, 9, 4, 15, 20, 0),
            datetime(2026, 9, 4, 15, 20, 10),
            datetime(2026, 9, 4, 15, 20, 10),
            datetime(2026, 9, 4, 15, 30, 1),
        ]
    )

    class _Client:
        async def create_session(self):
            raise AssertionError("session creation must be injectable")

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output1": {"askp1": "70000", "bidp1": "69900", "antc_cnpr": "69950"}}

    total = collect_auction.run_auction_capture(
        snapshot_date="2026-09-04",
        interval_seconds=0,
        client=_Client(),
        now_fn=lambda: next(clock),
    )

    assert total == 4
    assert {r["capture_reason"] for r in appended} == {"auction"}
    assert {r["symbol"] for r in appended} == {"005930", "000660"}


def test_run_auction_capture_empty_watchlist_is_noop(monkeypatch) -> None:
    import pandas as pd

    from src.daily import collect_auction

    monkeypatch.setattr(collect_auction.archive, "fetch_archive_snapshot", lambda *a, **kw: pd.DataFrame())

    assert collect_auction.run_auction_capture(snapshot_date="2026-09-04") == 0


def test_watchlist_codes_returns_empty_on_fetch_failure(monkeypatch) -> None:
    from src.daily import collect_auction

    def _raise(*a, **kw):
        raise RuntimeError("archive unavailable")

    monkeypatch.setattr(collect_auction.archive, "fetch_archive_snapshot", _raise)

    assert collect_auction._watchlist_codes("2026-09-04") == []


def test_run_auction_capture_sweep_continues_after_per_code_failure(monkeypatch) -> None:
    """스윕 중 특정 종목 호출이 실패해도 나머지 종목은 계속 수집한다."""
    from datetime import datetime

    import pandas as pd

    from src.daily import collect_auction

    monkeypatch.setattr(
        collect_auction.archive,
        "fetch_archive_snapshot",
        lambda *a, **kw: pd.DataFrame({"종목코드": ["005930", "000660"]}),
    )
    appended: list[dict] = []
    monkeypatch.setattr(
        collect_auction,
        "append_orderbook_snapshots",
        lambda rows, snapshot_date: appended.extend(rows) or len(rows),
    )

    clock = iter([datetime(2026, 9, 4, 15, 20, 0), datetime(2026, 9, 4, 15, 20, 0), datetime(2026, 9, 4, 15, 30, 1)])

    class _Client:
        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            if code == "005930":
                raise RuntimeError("network error")
            return {"rt_cd": "0", "output1": {"askp1": "70000", "bidp1": "69900"}}

    total = collect_auction.run_auction_capture(
        snapshot_date="2026-09-04",
        interval_seconds=0,
        client=_Client(),
        now_fn=lambda: next(clock),
    )

    assert total == 1
    assert {r["symbol"] for r in appended} == {"000660"}


def test_run_auction_capture_owns_client_when_none_provided(monkeypatch) -> None:
    """client 미지정 시 KisApiClient를 자체 생성/인증하고 세션을 정리한다."""
    from datetime import datetime

    import pandas as pd

    from src.daily import collect_auction

    monkeypatch.setattr(
        collect_auction.archive,
        "fetch_archive_snapshot",
        lambda *a, **kw: pd.DataFrame({"종목코드": ["005930"]}),
    )
    monkeypatch.setattr(collect_auction, "append_orderbook_snapshots", lambda rows, snapshot_date: len(rows))

    calls = {"ensure_token": 0, "closed": False}

    class _FakeSession:
        async def close(self):
            calls["closed"] = True

    class _FakeKisApiClient:
        def create_session(self):
            return _FakeSession()

        async def ensure_token(self, session):
            calls["ensure_token"] += 1

        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output1": {"askp1": "70000"}}

    import src.api.kis.client as kis_client_module

    monkeypatch.setattr(kis_client_module, "KisApiClient", _FakeKisApiClient)

    clock = iter([datetime(2026, 9, 4, 15, 20, 0), datetime(2026, 9, 4, 15, 20, 0), datetime(2026, 9, 4, 15, 30, 1)])

    total = collect_auction.run_auction_capture(
        snapshot_date="2026-09-04", interval_seconds=0, now_fn=lambda: next(clock)
    )

    assert total == 1
    assert calls["ensure_token"] == 1
    assert calls["closed"] is True


def test_run_auction_capture_logs_and_continues_on_persist_failure(monkeypatch) -> None:
    """스냅샷 영속화 자체가 실패해도 다음 sweep 주기를 계속 진행한다."""
    from datetime import datetime

    import pandas as pd

    from src.daily import collect_auction

    monkeypatch.setattr(
        collect_auction.archive,
        "fetch_archive_snapshot",
        lambda *a, **kw: pd.DataFrame({"종목코드": ["005930"]}),
    )

    def _raise(rows, snapshot_date):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(collect_auction, "append_orderbook_snapshots", _raise)
    monkeypatch.setattr(collect_auction.time, "sleep", lambda s: None)

    clock = iter([datetime(2026, 9, 4, 15, 20, 0), datetime(2026, 9, 4, 15, 20, 0), datetime(2026, 9, 4, 15, 30, 1)])

    class _Client:
        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output1": {"askp1": "70000"}}

    total = collect_auction.run_auction_capture(
        snapshot_date="2026-09-04",
        interval_seconds=1,
        client=_Client(),
        now_fn=lambda: next(clock),
    )

    assert total == 0


def test_collect_auction_main_invokes_run_auction_capture(monkeypatch) -> None:
    from src.daily import collect_auction

    captured: dict = {}

    def _fake_run(**kwargs):
        captured.update(kwargs)
        return 3

    monkeypatch.setattr(collect_auction, "run_auction_capture", _fake_run)
    monkeypatch.setattr(
        "sys.argv", ["collect_auction", "--date", "2026-09-04", "--interval", "5", "--start", "1520", "--end", "1530"]
    )

    collect_auction.main()

    assert captured == {
        "snapshot_date": "2026-09-04",
        "interval_seconds": 5,
        "start_hm": "1520",
        "end_hm": "1530",
        "finalize": True,
    }


def test_collect_auction_lazy_import_and_monkeypatch_retarget_together(monkeypatch) -> None:
    """Reproduces the exact mechanism test_collect_auction.py:145-147 relies on:
    monkeypatching KisApiClient on the module a lazy import will resolve against."""
    import src.api.kis.client as kis_client_module

    calls = {"constructed": 0}

    class _FakeKisApiClient:
        def __init__(self) -> None:
            calls["constructed"] += 1

    # Given: the monkeypatch targets the DIRECT module (post-rename), matching
    # what src/daily/collect_auction.py's lazy import now resolves to.
    monkeypatch.setattr(kis_client_module, "KisApiClient", _FakeKisApiClient)

    # When: a fresh lazy import, exactly as collect_auction.py:55 performs it inside
    # its function body, happens AFTER the monkeypatch is applied.
    from src.api.kis.client import KisApiClient

    owned_client = KisApiClient()

    # Then: the fake, not the real client, was resolved and constructed --
    # proving the paired lazy-import/monkeypatch rename works end-to-end.
    assert isinstance(owned_client, _FakeKisApiClient)
    assert calls["constructed"] == 1

def test_run_auction_capture_triggers_close_finalization_after_window(monkeypatch) -> None:
    from datetime import datetime

    import pandas as pd

    from src.daily import collect_auction, finalize_close

    monkeypatch.setattr(
        collect_auction.archive,
        "fetch_archive_snapshot",
        lambda *a, **kw: pd.DataFrame({"종목코드": ["005930"]}),
    )
    monkeypatch.setattr(collect_auction, "append_orderbook_snapshots", lambda rows, snapshot_date: len(rows))

    calls: list[dict] = []

    async def _fake_finalize(snapshot_date=None, **kwargs):
        calls.append({"snapshot_date": snapshot_date, **kwargs})
        return 1

    monkeypatch.setattr(finalize_close, "run_close_finalization", _fake_finalize)

    class _Client:
        async def get_orderbook_snapshot(self, session, code, market_div_code=None):
            return {"rt_cd": "0", "output1": {"askp1": "70000"}, "output2": {"antc_cnpr": "69950"}}

    client = _Client()

    def _clock_factory():
        ticks = iter(
            [
                datetime(2026, 9, 10, 15, 29, 50),
                datetime(2026, 9, 10, 15, 29, 50),
                datetime(2026, 9, 10, 15, 30, 1),
            ]
        )
        return lambda: next(ticks)

    # Given/When: finalize=True -> 스윕 종료 후 확정 패스 1회 호출
    collect_auction.run_auction_capture(
        snapshot_date="2026-09-10",
        interval_seconds=0,
        client=client,
        now_fn=_clock_factory(),
        finalize=True,
    )
    assert len(calls) == 1
    assert calls[0]["snapshot_date"] == "2026-09-10"

    # Then: 기본값(finalize 미지정)은 부작용 없음
    collect_auction.run_auction_capture(
        snapshot_date="2026-09-10",
        interval_seconds=0,
        client=client,
        now_fn=_clock_factory(),
    )
    assert len(calls) == 1


