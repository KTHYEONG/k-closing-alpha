from __future__ import annotations



def test_archive_target_codes_unions_previous_session_watchlist(monkeypatch) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    # Given: 2026-09-04 이전 아카이브 영업일은 금요일이 아닌 2026-09-01 (휴장 가정)
    frames = {
        "2026-09-04": pd.DataFrame({"종목코드": ["005930", "000660"]}),
        "2026-09-01": pd.DataFrame({"종목코드": ["009900", "005930"]}),
    }
    all_rows = pd.DataFrame({"스냅샷_날짜": ["2026-09-01", "2026-09-04"], "종목코드": ["009900", "005930"]})

    def fake_fetch(snapshot_date=None, month=None, all_rows_flag=False, **kwargs):
        if kwargs.get("all_rows") or all_rows_flag:
            return all_rows
        return frames.get(snapshot_date, pd.DataFrame())

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", fake_fetch)

    # When
    codes = archive_intraday._archive_target_codes("2026-09-04")

    # Then
    assert set(codes) == {"005930", "000660", "009900"}
    assert len(codes) == 3
    assert archive_intraday.resolve_previous_archive_date("2026-09-04") == "2026-09-01"
    assert archive_intraday.resolve_previous_archive_date("2026-09-01") is None


def test_today_watchlist_codes_returns_empty_on_fetch_failure(monkeypatch) -> None:
    from src.daily import archive_intraday

    def _raise(*a, **kw):
        raise RuntimeError("archive unavailable")

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", _raise)

    assert archive_intraday._today_watchlist_codes("2026-09-04") == []


def test_today_watchlist_codes_returns_empty_when_column_missing(monkeypatch) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", lambda **kw: pd.DataFrame())

    assert archive_intraday._today_watchlist_codes("2026-09-04") == []


def test_resolve_previous_archive_date_returns_none_on_fetch_failure(monkeypatch) -> None:
    from src.daily import archive_intraday

    def _raise(*a, **kw):
        raise RuntimeError("archive unavailable")

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", _raise)

    assert archive_intraday.resolve_previous_archive_date("2026-09-04") is None


def test_resolve_previous_archive_date_returns_none_when_column_missing(monkeypatch) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", lambda **kw: pd.DataFrame())

    assert archive_intraday.resolve_previous_archive_date("2026-09-04") is None


def test_archive_intraday_main_invokes_run_intraday_archive(monkeypatch) -> None:
    from src.daily import archive_intraday

    captured: dict = {}

    def _fake_run(snapshot_date=None):
        captured["snapshot_date"] = snapshot_date
        return (1, 2, 3)

    monkeypatch.setattr(archive_intraday, "run_intraday_archive", _fake_run)
    monkeypatch.setattr("sys.argv", ["archive_intraday", "2026-09-04"])

    archive_intraday.main()

    assert captured == {"snapshot_date": "2026-09-04"}


def test_archive_target_codes_unions_previous_session_watchlist_regression_unchanged() -> None:
    import pandas as pd

    from src.daily import archive_intraday

    frames = {
        "2026-09-04": pd.DataFrame({"종목코드": ["005930", "000660"]}),
        "2026-09-01": pd.DataFrame({"종목코드": ["009900", "005930"]}),
    }
    all_rows = pd.DataFrame({"스냅샷_날짜": ["2026-09-01", "2026-09-04"], "종목코드": ["009900", "005930"]})

    def fake_fetch(snapshot_date=None, month=None, all_rows_flag=False, **kwargs):
        if kwargs.get("all_rows") or all_rows_flag:
            return all_rows
        return frames.get(snapshot_date, pd.DataFrame())

    orig = archive_intraday.archive.fetch_archive_snapshot
    archive_intraday.archive.fetch_archive_snapshot = fake_fetch
    try:
        codes = archive_intraday._archive_target_codes("2026-09-04")
    finally:
        archive_intraday.archive.fetch_archive_snapshot = orig

    assert set(codes) == {"005930", "000660", "009900"}
    assert len(codes) == 3

def test_run_intraday_archive_wires_kiwoom_client_when_key_present(monkeypatch) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    monkeypatch.setattr(archive_intraday, "_archive_target_codes", lambda snap: ["005930"])
    monkeypatch.setattr(archive_intraday.settings, "KIWOM_APP_KEY", "dummy_key", raising=False)

    async def _is_trading(_c, _s, _d):
        return True

    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _is_trading)

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            assert self is not None
            return False

    class _FakeKisClient:
        def create_session(self):
            return _FakeSession()

        async def ensure_token(self, session):
            assert session is not None
            return "tok"

    captured: dict = {}

    class _FakeKiwoomClient:
        pass

    def _fake_kiwoom_ctor():
        inst = _FakeKiwoomClient()
        captured["kiwoom_instance"] = inst
        return inst

    async def _fake_collect_bars(*a, **kw):
        return pd.DataFrame()

    async def _fake_collect_nxt(*a, **kw):
        return pd.DataFrame()

    async def _fake_collect_ticks(client, session, codes, snap_date, ls_client=None, kiwoom_client=None):
        captured["kiwoom_client_passed"] = kiwoom_client
        return pd.DataFrame()

    monkeypatch.setattr(archive_intraday, "KisApiClient", lambda: _FakeKisClient())
    monkeypatch.setattr(archive_intraday, "LsApiClient", lambda: None)
    monkeypatch.setattr(archive_intraday, "KiwoomApiClient", _fake_kiwoom_ctor)
    monkeypatch.setattr(archive_intraday, "collect_intraday_bars", _fake_collect_bars)
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _fake_collect_nxt)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _fake_collect_ticks)
    monkeypatch.setattr(archive_intraday, "write_intraday_partition", lambda *a, **kw: 0)
    monkeypatch.setattr(archive_intraday, "write_tick_partition", lambda *a, **kw: 0)

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07")

    assert result == (0, 0, 0)
    assert captured["kiwoom_client_passed"] is captured["kiwoom_instance"]


def test_run_intraday_archive_no_kiwoom_client_when_key_absent(monkeypatch) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    monkeypatch.setattr(archive_intraday, "_archive_target_codes", lambda snap: ["005930"])
    monkeypatch.setattr(archive_intraday.settings, "KIWOM_APP_KEY", "", raising=False)

    async def _is_trading(_c, _s, _d):
        return True

    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _is_trading)

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            assert self is not None
            return False

    class _FakeKisClient:
        def create_session(self):
            return _FakeSession()

        async def ensure_token(self, session):
            assert session is not None
            return "tok"

    captured: dict = {}

    def _fake_kiwoom_ctor():
        captured["kiwoom_ctor_called"] = True
        return object()

    async def _fake_collect_bars(*a, **kw):
        return pd.DataFrame()

    async def _fake_collect_nxt(*a, **kw):
        return pd.DataFrame()

    async def _fake_collect_ticks(client, session, codes, snap_date, ls_client=None, kiwoom_client=None):
        captured["kiwoom_client_passed"] = kiwoom_client
        return pd.DataFrame()

    monkeypatch.setattr(archive_intraday, "KisApiClient", lambda: _FakeKisClient())
    monkeypatch.setattr(archive_intraday, "LsApiClient", lambda: None)
    monkeypatch.setattr(archive_intraday, "KiwoomApiClient", _fake_kiwoom_ctor)
    monkeypatch.setattr(archive_intraday, "collect_intraday_bars", _fake_collect_bars)
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _fake_collect_nxt)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _fake_collect_ticks)
    monkeypatch.setattr(archive_intraday, "write_intraday_partition", lambda *a, **kw: 0)
    monkeypatch.setattr(archive_intraday, "write_tick_partition", lambda *a, **kw: 0)

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07")

    assert result == (0, 0, 0)
    assert captured.get("kiwoom_client_passed") is None
    assert "kiwoom_ctor_called" not in captured


def test_run_intraday_archive_passes_kiwoom_to_nxt_collector(monkeypatch) -> None:
    from unittest.mock import AsyncMock, patch
    import pandas as pd
    from src import settings
    from src.daily.archive_intraday import run_intraday_archive

    monkeypatch.setattr(settings, "KIWOM_APP_KEY", "mock_kw_key")
    monkeypatch.setattr(settings, "LS_APP_KEY", "mock_ls_key")

    with (
        patch("src.daily.archive_intraday._archive_target_codes", return_value=["005930"]),
        patch("src.daily.archive_intraday.collect_intraday_bars", new_callable=AsyncMock) as mock_bars,
        patch("src.daily.archive_intraday.collect_nxt_aftermarket_bars", new_callable=AsyncMock) as mock_nxt,
        patch("src.daily.archive_intraday.collect_nxt_premarket_bars", new_callable=AsyncMock) as mock_nxt_pre,
        patch("src.daily.archive_intraday.collect_intraday_trade_ticks", new_callable=AsyncMock) as mock_ticks,
        patch("src.daily.archive_intraday.write_intraday_partition", return_value=5),
        patch("src.daily.archive_intraday.write_tick_partition", return_value=10),
        patch("src.api.kis.client.KisApiClient.ensure_token", new_callable=AsyncMock),
    ):
        mock_bars.return_value = pd.DataFrame()
        mock_nxt.return_value = pd.DataFrame()
        mock_nxt_pre.return_value = pd.DataFrame()
        mock_ticks.return_value = pd.DataFrame()

        res = run_intraday_archive("2026-09-04")
        assert res == (5, 10, 10)
        assert mock_nxt.call_count == 1
        _, kwargs = mock_nxt.call_args
        assert kwargs.get("kiwoom_client") is not None



def test_run_intraday_archive_collects_and_writes_premarket_partition(monkeypatch) -> None:
    from unittest.mock import AsyncMock, patch
    import pandas as pd
    from src import settings
    from src.config.market_session import INTRADAY_SESSION_NXT_PREMARKET
    from src.daily.archive_intraday import run_intraday_archive

    monkeypatch.setattr(settings, "KIWOM_APP_KEY", "mock_kw_key")
    monkeypatch.setattr(settings, "LS_APP_KEY", "mock_ls_key")

    partitions_written = []

    def fake_write_partition(df, interval, date, session):
        partitions_written.append(session)
        return len(df)

    with (
        patch("src.daily.archive_intraday._archive_target_codes", return_value=["005930"]),
        patch("src.daily.archive_intraday.collect_intraday_bars", new_callable=AsyncMock) as mock_bars,
        patch("src.daily.archive_intraday.collect_nxt_aftermarket_bars", new_callable=AsyncMock) as mock_nxt_after,
        patch("src.daily.archive_intraday.collect_nxt_premarket_bars", new_callable=AsyncMock) as mock_nxt_pre,
        patch("src.daily.archive_intraday.collect_intraday_trade_ticks", new_callable=AsyncMock) as mock_ticks,
        patch("src.daily.archive_intraday.write_intraday_partition", side_effect=fake_write_partition),
        patch("src.daily.archive_intraday.write_tick_partition", return_value=10),
        patch("src.api.kis.client.KisApiClient.ensure_token", new_callable=AsyncMock),
    ):
        mock_bars.return_value = pd.DataFrame([{"dummy": 1}] * 5)
        mock_nxt_after.return_value = pd.DataFrame([{"dummy": 1}] * 3)
        mock_nxt_pre.return_value = pd.DataFrame([{"dummy": 1}] * 2)
        mock_ticks.return_value = pd.DataFrame([{"dummy": 1}] * 10)

        res = run_intraday_archive("2026-09-04")
        assert res == (5, 5, 10)
        assert INTRADAY_SESSION_NXT_PREMARKET in partitions_written
        assert mock_nxt_pre.call_count == 1

def test_archive_target_codes_unions_universe_scan_pool_across_sessions(monkeypatch) -> None:
    # Given
    import pandas as pd

    from src.daily import archive_intraday
    from src.daily.universe_scan import UNIVERSE_SCAN_SCENARIO_TAG

    frames = {
        "2026-09-04": pd.DataFrame(
            {"종목코드": ["005930"], "시나리오": [UNIVERSE_SCAN_SCENARIO_TAG]}
        ),
        "2026-09-03": pd.DataFrame(
            {"종목코드": ["000660"], "시나리오": [UNIVERSE_SCAN_SCENARIO_TAG]}
        ),
    }
    all_rows = pd.DataFrame(
        {"스냅샷_날짜": ["2026-09-03", "2026-09-04"], "종목코드": ["000660", "005930"]}
    )

    def fake_fetch(snapshot_date=None, month=None, all_rows_flag=False, **kwargs):
        if kwargs.get("all_rows") or all_rows_flag:
            return all_rows
        return frames.get(snapshot_date, pd.DataFrame())

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", fake_fetch)

    # When
    codes = archive_intraday._archive_target_codes("2026-09-04")

    # Then: 청산일(D+1) 분봉 확보를 위해 전일 풀도 포함
    assert set(codes) == {"005930", "000660"}


def test_run_intraday_archive_instantiates_kiwoom_with_kiwoom_app_key_alias(monkeypatch) -> None:
    from unittest.mock import AsyncMock, MagicMock
    import pandas as pd
    from src.daily import archive_intraday

    monkeypatch.setattr(archive_intraday, "_archive_target_codes", lambda snap: ["005930"])
    monkeypatch.setattr(archive_intraday.settings, "KIWOM_APP_KEY", "", raising=False)
    monkeypatch.setattr(archive_intraday.settings, "KIWOOM_APP_KEY", "test_kiwoom_key", raising=False)

    async def _is_trading(_c, _s, _d):
        return True

    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _is_trading)

    fake_client = MagicMock()
    fake_session = AsyncMock()
    fake_client.create_session.return_value.__aenter__.return_value = fake_session
    fake_client.ensure_token = AsyncMock(return_value="tok")

    captured: dict = {}

    def _fake_kiwoom_ctor():
        captured["kiwoom_ctor_called"] = True
        return object()

    async def _mock_collect_ticks(client, session, codes, snap_date, ls_client=None, kiwoom_client=None):
        captured["kiwoom_client_passed"] = kiwoom_client
        return pd.DataFrame()

    monkeypatch.setattr(archive_intraday, "KisApiClient", lambda: fake_client)
    monkeypatch.setattr(archive_intraday, "LsApiClient", lambda: None)
    monkeypatch.setattr(archive_intraday, "KiwoomApiClient", _fake_kiwoom_ctor)
    monkeypatch.setattr(archive_intraday, "collect_intraday_bars", AsyncMock(return_value=pd.DataFrame()))
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", AsyncMock(return_value=pd.DataFrame()))
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", AsyncMock(return_value=pd.DataFrame()))
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _mock_collect_ticks)
    monkeypatch.setattr(archive_intraday, "write_intraday_partition", lambda *a, **kw: 0)
    monkeypatch.setattr(archive_intraday, "write_tick_partition", lambda *a, **kw: 0)

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07")

    assert result == (0, 0, 0)
    assert captured.get("kiwoom_ctor_called") is True
    assert captured.get("kiwoom_client_passed") is not None


def test_run_intraday_archive_writes_krx_aftermarket_to_its_own_session(monkeypatch) -> None:
    import pandas as pd

    from src.config.market_session import (
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_REGULAR,
    )
    from src.daily import archive_intraday

    monkeypatch.setattr(archive_intraday, "_archive_target_codes", lambda snap: ["005930"])

    async def _is_trading(_c, _s, _d):
        return True

    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _is_trading)

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def create_session(self):
            return _Session()

        async def ensure_token(self, session):
            return None

    monkeypatch.setattr(archive_intraday, "KisApiClient", lambda *a, **kw: _Client())
    monkeypatch.setattr(archive_intraday, "LsApiClient", lambda: None)
    monkeypatch.setattr(archive_intraday, "KiwoomApiClient", lambda: None)

    def _frame(tag: str) -> pd.DataFrame:
        return pd.DataFrame({"tag": [tag]})

    async def _regular(*a, **kw):
        return _frame("regular")

    async def _nxt_after(*a, **kw):
        return _frame("nxt_after")

    async def _nxt_pre(*a, **kw):
        return _frame("nxt_pre")

    async def _krx_after(*a, **kw):
        return _frame("krx_after")

    async def _ticks(*a, **kw):
        return _frame("ticks")

    monkeypatch.setattr(archive_intraday, "collect_intraday_bars", _regular)
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _nxt_after)
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _nxt_pre)
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _krx_after)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _ticks)

    written: list[tuple[str, str]] = []

    def _write_bars(df, interval, snap, session):
        written.append((str(df["tag"].iloc[0]), session))
        return len(df)

    monkeypatch.setattr(archive_intraday, "write_intraday_partition", _write_bars)
    monkeypatch.setattr(archive_intraday, "write_tick_partition", lambda df, snap, session: len(df))

    archive_intraday.run_intraday_archive(snapshot_date="2026-09-14")

    mapping = dict(written)
    assert mapping["krx_after"] == INTRADAY_SESSION_KRX_AFTERMARKET
    assert mapping["regular"] == INTRADAY_SESSION_REGULAR
    assert mapping["krx_after"] != mapping["regular"]


def test_intraday_watchlist_drops_unreachable_scenario_filter(monkeypatch) -> None:
    import inspect

    import pandas as pd

    from src.daily import archive_intraday

    # Then: 도달 불가능하던 고스트 훅이 사라진다
    assert not hasattr(archive_intraday, "EXCLUDED_INTRADAY_SCENARIOS")
    params = inspect.signature(archive_intraday._today_watchlist_codes).parameters
    assert list(params) == ["snapshot_date"]

    # And: 워치리스트 추출 자체는 정상 동작한다(6자리 zero-fill 포함)
    monkeypatch.setattr(
        archive_intraday.archive,
        "fetch_archive_snapshot",
        lambda snapshot_date=None, **kw: pd.DataFrame({"종목코드": ["5930", "035720"]}),
    )
    codes = archive_intraday._today_watchlist_codes("2026-09-09")
    assert codes == ["005930", "035720"]


def test_run_intraday_archive_skips_non_trading_day_without_collecting(monkeypatch) -> None:
    from src.daily import archive_intraday

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

    class _FakeClient:
        def __init__(self, *_a, **_kw):
            self.token = None

        def create_session(self, **_kw):
            return _FakeSession()

        async def ensure_token(self, _session, force_refresh=False):
            self.token = "T"
            return "T"

    async def _never(*_a, **_kw):
        raise AssertionError("non-trading day must not trigger collection")

    async def _not_trading(_client, _session, _date):
        return False

    monkeypatch.setattr(archive_intraday, "_archive_target_codes", lambda _d: ["005930"])
    monkeypatch.setattr(archive_intraday, "KisApiClient", _FakeClient)
    monkeypatch.setattr(archive_intraday, "LsApiClient", lambda *_a, **_kw: None)
    monkeypatch.setattr(archive_intraday, "KiwoomApiClient", lambda *_a, **_kw: None)
    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _not_trading)
    monkeypatch.setattr(archive_intraday, "collect_intraday_bars", _never)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _never)

    # When
    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-05")

    # Then
    assert result == (0, 0, 0)
