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

    def _fake_run(snapshot_date=None, **kwargs):
        captured["snapshot_date"] = snapshot_date
        captured["kwargs"] = kwargs
        return (1, 2, 3)

    monkeypatch.setattr(archive_intraday, "run_intraday_archive", _fake_run)
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--date", "2026-09-04"])

    archive_intraday.main()

    assert captured["snapshot_date"] == "2026-09-04"
    assert captured["kwargs"]["phase"] == "all"


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

    monkeypatch.setattr(archive_intraday, "KisApiClient", lambda *a, **kw: _FakeKisClient())
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

    monkeypatch.setattr(archive_intraday, "KisApiClient", lambda *a, **kw: _FakeKisClient())
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
        patch("src.daily.archive_intraday.is_kis_trading_day", new_callable=AsyncMock) as mock_trading_day,
    ):
        mock_trading_day.return_value = True
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
        patch("src.daily.archive_intraday.is_kis_trading_day", new_callable=AsyncMock) as mock_trading_day,
    ):
        mock_trading_day.return_value = True
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

    monkeypatch.setattr(archive_intraday, "KisApiClient", lambda *a, **kw: fake_client)
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


def _wire_legacy_run_fakes(monkeypatch, calls: list[str]):
    import pandas as pd

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

    def _tracked(name: str, tag: str):
        async def _fn(*a, **kw):
            calls.append(name)
            return _frame(tag)

        return _fn

    monkeypatch.setattr(archive_intraday, "collect_intraday_bars", _tracked("bars", "regular"))
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _tracked("nxt_after", "nxt_after"))
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _tracked("nxt_pre", "nxt_pre"))
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _tracked("krx_after", "krx_after"))
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _tracked("ticks", "ticks"))
    monkeypatch.setattr(archive_intraday, "write_intraday_partition", lambda df, interval, snap, session: len(df))
    monkeypatch.setattr(archive_intraday, "write_tick_partition", lambda df, snap, session: len(df))


def test_run_intraday_archive_phase_regular_only_collects_regular_session(monkeypatch) -> None:
    from src.daily import archive_intraday

    calls: list[str] = []
    _wire_legacy_run_fakes(monkeypatch, calls)

    n_bars, n_nxt, n_ticks = archive_intraday.run_intraday_archive(snapshot_date="2026-09-14", phase="regular")

    assert calls == ["bars", "ticks"]
    assert n_bars == 1 and n_ticks == 1
    assert n_nxt == 0


def test_run_intraday_archive_phase_aftermarket_only_collects_aftermarket_sessions(monkeypatch) -> None:
    from src.daily import archive_intraday

    calls: list[str] = []
    _wire_legacy_run_fakes(monkeypatch, calls)

    n_bars, n_nxt, n_ticks = archive_intraday.run_intraday_archive(snapshot_date="2026-09-14", phase="aftermarket")

    assert calls == ["nxt_after", "nxt_pre", "krx_after"]
    assert n_nxt == 2  # nxt_after(1 row) + nxt_pre(1 row); krx_after is not summed into n_nxt
    assert n_bars == 0 and n_ticks == 0


def test_run_intraday_archive_rejects_unknown_phase(monkeypatch) -> None:
    import pytest

    from src.daily import archive_intraday

    calls: list[str] = []
    _wire_legacy_run_fakes(monkeypatch, calls)

    with pytest.raises(ValueError, match="Invalid phase"):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-14", phase="bogus")

    assert calls == []


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


def test_archive_intraday_builds_client_from_data_account(monkeypatch, tmp_path) -> None:
    from src.daily import archive_intraday

    captured = {}

    class _FakeKisClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def create_session(self):
            raise AssertionError("session should not be opened in this scenario")

    token_file = str(tmp_path / "x.json")
    monkeypatch.setattr(archive_intraday, "KisApiClient", _FakeKisClient)
    monkeypatch.setattr(
        archive_intraday,
        "kis_data_client_kwargs",
        lambda: {"app_key": "data-key", "app_secret": "s", "account_id": "", "hts_id": "h", "token_file": token_file},
        raising=False,
    )

    # When
    archive_intraday.KisApiClient(**archive_intraday.kis_data_client_kwargs())

    # Then
    assert captured["app_key"] == "data-key"


def _raw_profile(tmp_path):
    from src.config.collection import CollectionSettings

    return CollectionSettings(COLLECTION_ROOT=tmp_path / "cap")


def _archive_store(tmp_path):
    from src.data.capture_store import CaptureStore

    return CaptureStore(tmp_path / "cap")


def _publish_cohort(store, snapshot_date, eligible):
    import datetime as _dt

    from src.data.capture_contracts import (
        SEOUL as _SEOUL,
        CaptureContext,
        CaptureDataset,
        CaptureManifest,
        CaptureStatus,
        Cohort,
    )

    trading_day = _dt.date.fromisoformat(snapshot_date)
    cohort = Cohort(
        trading_date=trading_day,
        cohort_id=f"c-{snapshot_date}",
        eligible_symbols=tuple(eligible),
        scanned_symbols=tuple(eligible),
        eligibility_rule_version="v1",
        rejections={},
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=trading_day, run_id="decision-1", dataset=CaptureDataset.SCAN,
            vendor="owner-local", endpoint="decision-input", symbol=None, venue="KRX",
            session="regular", capture_reason="test", cohort_id=cohort.cohort_id, scheduled_at=None,
        ),
        cohort=cohort,
        completed_at=_dt.datetime(2026, 9, 1, 15, 34, tzinfo=_SEOUL),
        entries=(),
        artifacts=(),
        status=CaptureStatus.COMPLETE,
    )
    store.publish_manifest(manifest)
    return cohort


def _canon_bar_frame(snapshot_date, symbol):
    import pandas as pd

    from src.data.intraday_schema import normalize_bar_frame

    raw = pd.DataFrame({"time": ["090300"], "open": [70000], "high": [70100], "low": [69900],
                        "close": [70000], "jdiff_vol": [100], "value": [70]})
    return normalize_bar_frame(raw, "ls", snapshot_date, symbol)


def _empty_bar_frame_for(snapshot_date):
    import pandas as pd

    from src.data.intraday_schema import normalize_bar_frame

    return normalize_bar_frame(pd.DataFrame(), "ls", snapshot_date, "000000")


def _fake_entry(symbol, dataset, session, status, rows=0):
    from src.data.capture_contracts import CaptureDataset as _Dataset
    from src.data.capture_contracts import CaptureStatus as _Status
    from src.data.capture_contracts import CoverageEntry

    _ = _Dataset
    return CoverageEntry(
        symbol=symbol, dataset=dataset, venue="KRX", session=session, scheduled_at=None,
        status=_Status(status), rows=rows, first_event_time=None, last_event_time=None,
        reason="test-fake", raw_refs=(),
    )


def _archive_fakes(entries_map):
    seen = {}

    async def _collect(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        seen.setdefault("codes", []).append(list(codes))
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame, entry = entries_map[code]
            on_symbol(code, frame, entry)
        import pandas as pd

        return pd.DataFrame()

    return _collect, seen


def _seed_panel(tmp_path, symbols_by_date: dict[str, list[str]]) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    panel_path = archive_intraday.settings.PRICE_HISTORY_PARQUET_PATH
    rows: list[dict[str, str]] = [
        {"date": day, "symbol": symbol} for day, symbols in symbols_by_date.items() for symbol in symbols
    ]
    frame = pd.DataFrame(rows, columns=["date", "symbol"])
    frame["date"] = pd.to_datetime(frame["date"])
    panel_file = __import__("pathlib").Path(panel_path)
    panel_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(panel_file, index=False)


def _raw_archive_mocks(monkeypatch, tmp_path, fake_collect):
    from src.daily import archive_intraday

    monkeypatch.setattr(archive_intraday.settings, "HISTORY_DIR", tmp_path, raising=False)
    monkeypatch.setattr(archive_intraday.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "price_history.parquet", raising=False)
    monkeypatch.setattr(archive_intraday.settings, "LS_APP_KEY", "", raising=False)
    monkeypatch.setattr(archive_intraday.settings, "KIWOM_APP_KEY", "", raising=False)
    monkeypatch.setattr(archive_intraday.settings, "KIWOOM_APP_KEY", "", raising=False)

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def create_session(self):
            return _Session()

        async def ensure_token(self, session):
            return "tok"

    async def _trading(_client, _session, _date):
        return True

    monkeypatch.setattr(archive_intraday, "KisApiClient", lambda *a, **kw: _Client())
    monkeypatch.setattr(archive_intraday, "LsApiClient", lambda: None)
    monkeypatch.setattr(archive_intraday, "KiwoomApiClient", lambda: None)
    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _trading)
    monkeypatch.setattr(archive_intraday, "collect_intraday_bars", fake_collect)
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", fake_collect)
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", fake_collect)
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", fake_collect)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", fake_collect)


def test_run_archive_uses_calendar_previous_day_cohort(monkeypatch, tmp_path) -> None:
    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930", "000660"])
    _publish_cohort(store, "2026-09-04", ["009900"])
    entries_map = {
        code: (_empty_bar_frame_for("2026-09-07"), _fake_entry(code, __import__("src.data.capture_contracts", fromlist=["CaptureDataset"]).CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
        for code in ("005930", "000660", "009900")
    }
    fake_collect, seen = _archive_fakes(entries_map)
    _raw_archive_mocks(monkeypatch, tmp_path, fake_collect)
    _seed_panel(tmp_path, {"2026-09-04": ["005930", "000660"], "2026-09-03": ["009900"]})

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))

    assert result == (0, 0, 0)
    assert sorted(seen["codes"][0]) == ["000660", "005930", "009900"]
    manifests = store.read_manifests("2026-09-07")
    assert len([item for item in manifests if item.context.run_id.startswith("archive-")]) == 5


def test_run_archive_same_day_retry_does_not_collide_with_prior_attempt(monkeypatch, tmp_path) -> None:
    """실측 회귀: run_id가 날짜로만 고정돼 있어 같은 날 두 번째 실행(수동 재시도 또는
    실패 후 재기동)이 이전 시도의 불변 매니페스트와 충돌해 ValueError로 즉시 실패하던 버그
    (2026-09-18 실측: 수동 재실행이 'conflicting immutable artifact identity'로 죽음)."""
    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    entries_map = {
        "005930": (_empty_bar_frame_for("2026-09-07"), _fake_entry("005930", __import__("src.data.capture_contracts", fromlist=["CaptureDataset"]).CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN")),
    }
    fake_collect, _ = _archive_fakes(entries_map)
    _raw_archive_mocks(monkeypatch, tmp_path, fake_collect)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})

    # When: 같은 날 두 번 연속 실행
    archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))
    archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))

    # Then: 두 번째 실행도 충돌 없이 자기 몫의 매니페스트를 남긴다
    manifests = store.read_manifests("2026-09-07")
    archive_manifests = [item for item in manifests if item.context.run_id.startswith("archive-")]
    assert len(archive_manifests) == 10
    assert len({item.context.run_id for item in archive_manifests}) == 10


def test_run_archive_ignores_unavailable_external_collector(monkeypatch, tmp_path) -> None:
    import sys

    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    entries_map = {
        "005930": (_empty_bar_frame_for("2026-09-07"), _fake_entry("005930", __import__("src.data.capture_contracts", fromlist=["CaptureDataset"]).CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN")),
    }
    fake_collect, _ = _archive_fakes(entries_map)
    _raw_archive_mocks(monkeypatch, tmp_path, fake_collect)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setitem(sys.modules, "krx_alpha", None)

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))

    assert result == (0, 0, 0)


def test_run_archive_partial_work_reported_degraded(monkeypatch, tmp_path, caplog) -> None:
    import logging

    import pandas as pd

    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930", "000660"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    good_frame = _canon_bar_frame("2026-09-07", "005930")
    entries_map = {
        "005930": (good_frame, _fake_entry("005930", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=len(good_frame))),
        "000660": (_empty_bar_frame_for("2026-09-07"), _fake_entry("000660", CaptureDataset.MINUTE_BARS, "regular", "PARTIAL")),
    }

    async def _fake_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame, entry = entries_map[code]
            on_symbol(code, frame, entry)
        return pd.DataFrame()

    async def _fake_other(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _empty_bar_frame_for(snap_date),
                      _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    async def _fake_ticks(client, session, codes, snap_date, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _empty_bar_frame_for(snap_date),
                      _fake_entry(code, CaptureDataset.TRADE_TICKS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _fake_bars)
    _seed_panel(tmp_path, {"2026-09-04": ["005930", "000660"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _fake_other)
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _fake_other)
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _fake_other)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _fake_ticks)

    with caplog.at_level(logging.WARNING, logger=archive_intraday.logger.name):
        result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))

    assert result[0] == 1
    assert any("DEGRADED" in rec.message for rec in caplog.records)
    manifests = store.read_manifests("2026-09-07")
    bars_manifest = next(item for item in manifests if item.context.run_id.startswith("archive-2026-09-07-regular-bars-"))
    assert bars_manifest.status.value == "PARTIAL"
    assert {item.symbol for item in bars_manifest.entries} == {"005930", "000660"}


def test_repair_cli_stages_evidence_without_applying(monkeypatch, tmp_path) -> None:
    import pandas as pd

    import src.tools.repair_intraday_capture as repair_mod
    from src.data.intraday_schema import normalize_tick_frame

    monkeypatch.setattr(repair_mod.settings, "HISTORY_DIR", tmp_path, raising=False)
    part = tmp_path / "intraday" / "ticks" / "regular" / "2026-09" / "2026-09-03.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame({"time": ["090300"], "close": [70000], "jdiff_vol": [5]})
    normalize_tick_frame(raw, "ls", "2026-09-03", "005930").to_parquet(part, index=False)
    before = part.read_bytes()

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        async def ensure_token(self, session):
            return "tok"

    monkeypatch.setattr(repair_mod, "_open_clients", lambda: (_Client(), _Session(), None, None))

    async def _fake_ticks(client, session, codes, snap_date, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _empty_bar_frame_for(snap_date),
                      _fake_entry(code, __import__("src.data.capture_contracts", fromlist=["CaptureDataset"]).CaptureDataset.TRADE_TICKS, "regular", "PARTIAL"))
        return pd.DataFrame()

    monkeypatch.setattr("src.backfill.intraday.collector.collect_intraday_trade_ticks", _fake_ticks)

    repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "regular_ticks"])

    assert part.read_bytes() == before
    report = tmp_path / "capture" / "staging" / "intraday" / "repair-2026-09-03-2026-09-03.json"
    assert report.exists()
    import json as _json

    payload = _json.loads(report.read_text())
    assert payload["attempts"][0]["unresolved"] == "005930"

    import pytest

    with pytest.raises(ValueError, match="range"):
        repair_mod.main(["--start", "2026-09-05", "--end", "2026-09-03", "--dataset", "regular_ticks"])
    with pytest.raises(ValueError, match="--dataset"):
        repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "bogus"])
    with pytest.raises(ValueError, match="at least one"):
        repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03"])
    with pytest.raises(ValueError, match="--max-pages"):
        repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "regular_ticks", "--max-pages", "0"])
    with pytest.raises(ValueError, match="aware"):
        repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "regular_ticks", "--deadline", "2026-09-04T00:00:00"])
    with pytest.raises(ValueError, match="no symbols"):
        repair_mod.main(["--start", "2026-09-10", "--end", "2026-09-10", "--dataset", "regular_ticks"])


def test_repair_cli_apply_publishes_certified_partition(monkeypatch, tmp_path) -> None:
    import pandas as pd

    import src.tools.repair_intraday_capture as repair_mod
    from src.data.capture_contracts import CaptureDataset
    from src.data.intraday_schema import normalize_tick_frame

    monkeypatch.setattr(repair_mod.settings, "HISTORY_DIR", tmp_path, raising=False)

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        async def ensure_token(self, session):
            return "tok"

    monkeypatch.setattr(repair_mod, "_open_clients", lambda: (_Client(), _Session(), None, None))

    async def _fake_ticks(client, session, codes, snap_date, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            raw = pd.DataFrame({"time": ["090300"], "close": [71000], "jdiff_vol": [7]})
            frame = normalize_tick_frame(raw, "ls", snap_date, code)
            on_symbol(code, frame, _fake_entry(code, CaptureDataset.TRADE_TICKS, "regular", "COMPLETE", rows=len(frame)))
        return pd.DataFrame()

    monkeypatch.setattr("src.backfill.intraday.collector.collect_intraday_trade_ticks", _fake_ticks)

    repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "regular_ticks",
                     "--symbol", "005930", "--apply"])

    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    stored = pd.read_parquet(intraday_store.tick_partition_path("2026-09-03", "regular"))
    assert len(stored) == 1
    assert stored.iloc[0]["price"] == 71000


def test_run_archive_cohort_and_cli_boundaries(monkeypatch, tmp_path) -> None:
    import pytest

    from src.daily import archive_intraday

    assert archive_intraday._previous_trading_day("2026-09-07") == "2026-09-04"
    assert archive_intraday._previous_trading_day("2026-09-08") == "2026-09-07"
    with pytest.raises(ValueError, match="snapshot_date"):
        archive_intraday._previous_trading_day("bogus")
    with pytest.raises(ValueError, match="snapshot_date"):
        archive_intraday.run_intraday_archive(snapshot_date="bogus", profile=_raw_profile(tmp_path))
    with pytest.raises(ValueError, match="bar_interval"):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", bar_interval_minutes=0, profile=_raw_profile(tmp_path))
    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    with pytest.raises(FileNotFoundError, match="no qualifying cohort"):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-08", profile=_raw_profile(tmp_path))
    monkeypatch.setattr(archive_intraday, "_archive_target_codes", lambda _d: [])
    monkeypatch.setattr(archive_intraday, "CollectionSettings", lambda: _raw_profile(tmp_path))
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--date", "bogus-date"])
    with pytest.raises(SystemExit) as exc:
        archive_intraday.main()
    assert exc.value.code == 2
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--date", "2026-09-08"])
    monkeypatch.setattr(archive_intraday, "_archive_target_codes", lambda _d: ["005930"])
    with pytest.raises(SystemExit) as exc:
        archive_intraday.main()
    assert exc.value.code == 1


def test_run_archive_default_root_paper_and_prev_gap(monkeypatch, tmp_path, caplog) -> None:
    import logging

    import pandas as pd

    from src import settings as _settings
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    monkeypatch.setattr(_settings, "HISTORY_DIR", tmp_path, raising=False)
    from src.data.capture_store import CaptureStore as _CaptureStore

    store = _CaptureStore(tmp_path / "capture")
    _publish_cohort(store, "2026-09-07", ["005930"])

    class _Ledger:
        def load_open_positions(self):
            return pd.DataFrame({"symbol": ["099999"]})

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", lambda *a, **k: _Ledger())

    async def _fake(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _empty_bar_frame_for(snap_date),
                      _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _fake)
    monkeypatch.setattr(archive_intraday.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "price_history.parquet", raising=False)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    profile = _raw_profile(tmp_path).model_copy(update={"COLLECTION_ROOT": None})

    with caplog.at_level(logging.WARNING, logger=archive_intraday.logger.name):
        result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=profile)

    assert result == (0, 0, 0)
    assert any("INCOMPLETE" in rec.message for rec in caplog.records)

    def _boom_ledger(*a, **k):
        raise RuntimeError("ledger down")

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", _boom_ledger)
    from src.data.capture_store import CaptureStore as _SecondStore

    _publish_cohort(_SecondStore(tmp_path / "cap2"), "2026-09-07", ["005930"])
    fallback_profile = _raw_profile(tmp_path).model_copy(update={"COLLECTION_ROOT": tmp_path / "cap2"})
    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=fallback_profile)
    assert result == (0, 0, 0)


def test_run_archive_non_trading_day_skips_in_raw_mode(monkeypatch, tmp_path) -> None:
    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])

    async def _never(*args, **kwargs):
        raise AssertionError("must not collect on non-trading day")

    _raw_archive_mocks(monkeypatch, tmp_path, _never)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _never)
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _never)
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _never)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _never)

    async def _not_trading(_client, _session, _date):
        return False

    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _not_trading)

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))

    assert result == (0, 0, 0)


def test_run_archive_full_complete_and_partial_fragments(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.daily import archive_intraday
    from src.data import intraday_store
    from src.data.capture_contracts import CaptureDataset

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930", "000660"])
    _publish_cohort(store, "2026-09-04", ["005930"])

    import datetime as _dt

    from src.data.capture_contracts import SEOUL as _SEOUL, CapturedResponse, CaptureStatus as _CS

    _seed_ctx = __import__("src.data.capture_contracts", fromlist=["CaptureContext"]).CaptureContext(
        trading_date=_dt.date(2026, 9, 7), run_id="seed", dataset=__import__("src.data.capture_contracts", fromlist=["CaptureDataset"]).CaptureDataset.SCAN,
        vendor="owner-local", endpoint="seed", symbol=None, venue="KRX", session="regular",
        capture_reason="seed", cohort_id=None, scheduled_at=None,
    )
    _now = _dt.datetime.now(_SEOUL)
    shared = store.append_response(CapturedResponse(
        context=_seed_ctx, request_started_at=_now, received_at=_now, payload={"seed": True},
        status=_CS.COMPLETE, source_timestamp=None, source_published_at=None,
        page_index=0, attempt_index=0, continuation={}, error_type=None,
    ))

    def _entry(symbol, dataset, session, status, rows=0, refs=()):
        from src.data.capture_contracts import CoverageEntry

        return CoverageEntry(
            symbol=symbol, dataset=dataset, venue="KRX", session=session, scheduled_at=None,
            status=status, rows=rows, first_event_time=None, last_event_time=None,
            reason="test-full", raw_refs=tuple(refs),
        )

    from src.data.capture_contracts import CaptureStatus

    def _tick_frame(symbol):
        raw = pd.DataFrame({"time": ["090300"], "close": [71000], "jdiff_vol": [7]})
        from src.data.intraday_schema import normalize_tick_frame

        return normalize_tick_frame(raw, "ls", "2026-09-07", symbol)

    async def _fake_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            if code == "005930":
                frame = _canon_bar_frame(snap_date, code)
                on_symbol(code, frame, _entry(code, CaptureDataset.MINUTE_BARS, "regular", CaptureStatus.COMPLETE, rows=len(frame), refs=[shared]))
            else:
                on_symbol(code, _canon_bar_frame(snap_date, code),
                          _entry(code, CaptureDataset.MINUTE_BARS, "regular", CaptureStatus.PARTIAL, refs=[shared]))
        return pd.DataFrame()

    def _fake_session_bars_factory(session_tag):
        async def _fake_session_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
            on_symbol = kwargs.get("on_symbol")
            for code in codes:
                frame = _canon_bar_frame(snap_date, code)
                on_symbol(code, frame, _entry(code, CaptureDataset.MINUTE_BARS, session_tag, CaptureStatus.COMPLETE, rows=len(frame), refs=[shared]))
            return pd.DataFrame()

        return _fake_session_bars

    async def _fake_ticks(client, session, codes, snap_date, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            if code == "005930":
                frame = _tick_frame(code)
                on_symbol(code, frame, _entry(code, CaptureDataset.TRADE_TICKS, "regular", CaptureStatus.COMPLETE, rows=len(frame), refs=[shared]))
            else:
                on_symbol(code, _tick_frame(code),
                          _entry(code, CaptureDataset.TRADE_TICKS, "regular", CaptureStatus.PARTIAL, refs=[shared]))
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _fake_bars)
    _seed_panel(tmp_path, {"2026-09-04": ["005930", "000660"], "2026-09-03": ["005930"]})
    from src.config.market_session import (
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
        INTRADAY_SESSION_NXT_PREMARKET,
    )

    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _fake_session_bars_factory(INTRADAY_SESSION_NXT_AFTERMARKET))
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _fake_session_bars_factory(INTRADAY_SESSION_NXT_PREMARKET))
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _fake_session_bars_factory(INTRADAY_SESSION_KRX_AFTERMARKET))
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _fake_ticks)

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))

    assert result == (1, 4, 1)
    manifests = store.read_manifests("2026-09-07")
    def _by_prefix(prefix: str):
        return next(item for item in manifests if item.context.run_id.startswith(prefix))

    assert _by_prefix("archive-2026-09-07-regular-bars-").status.value == "PARTIAL"
    assert _by_prefix("archive-2026-09-07-regular-ticks-").status.value == "PARTIAL"
    assert _by_prefix("archive-2026-09-07-nxt-aftermarket-").status.value == "COMPLETE"
    stored_ticks = pd.read_parquet(intraday_store.tick_partition_path("2026-09-07", "regular"))
    assert len(stored_ticks) == 1


def test_repair_cli_bars_and_deadline_branches(monkeypatch, tmp_path) -> None:
    import pandas as pd

    import src.tools.repair_intraday_capture as repair_mod
    from src.data.intraday_schema import normalize_bar_frame

    monkeypatch.setattr(repair_mod.settings, "HISTORY_DIR", tmp_path, raising=False)
    part = tmp_path / "intraday" / "1m" / "regular" / "2026-09" / "2026-09-03.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame({"time": ["090300"], "open": [70000], "high": [70100], "low": [69900],
                        "close": [70000], "jdiff_vol": [100], "value": [70]})
    normalize_bar_frame(raw, "ls", "2026-09-03", "005930").to_parquet(part, index=False)

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        async def ensure_token(self, session):
            return "tok"

    monkeypatch.setattr(repair_mod, "_open_clients", lambda: (_Client(), _Session(), None, None))

    async def _fake_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _empty_bar_frame_for(snap_date),
                      _fake_entry(code, __import__("src.data.capture_contracts", fromlist=["CaptureDataset"]).CaptureDataset.MINUTE_BARS, "regular", "PARTIAL"))
        return pd.DataFrame()

    monkeypatch.setattr("src.backfill.intraday.collector.collect_intraday_bars", _fake_bars)

    import pytest

    with pytest.raises(ValueError, match="--start"):
        repair_mod.main(["--start", "bogus", "--end", "2026-09-03", "--dataset", "regular_bars"])
    with pytest.raises(ValueError, match="--deadline"):
        repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "regular_bars",
                         "--deadline", "bogus"])
    repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "regular_bars",
                     "--deadline", "2030-01-01T00:00:00+09:00"])
    repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "regular_bars",
                     "--deadline", "2020-01-01T00:00:00+09:00"])
    import json as _json

    payload = _json.loads((tmp_path / "capture" / "staging" / "intraday" / "repair-2026-09-03-2026-09-03.json").read_text())
    assert any(item.get("status") == "deadline-exceeded" for item in payload["attempts"])


def test_repair_open_clients_constructs(monkeypatch, tmp_path) -> None:
    import asyncio

    import src.tools.repair_intraday_capture as repair_mod

    async def _open():
        client, session_ctx, ls_client, kiwoom_client = repair_mod._open_clients()
        async with session_ctx:
            return (client is not None, ls_client, kiwoom_client)

    _, ls_client, _ = asyncio.run(_open())
    assert ls_client is None or hasattr(ls_client, "get_tick_chart")


def test_repair_cli_selection_and_publication_branches(monkeypatch, tmp_path) -> None:
    import pandas as pd
    import pytest

    import src.tools.repair_intraday_capture as repair_mod
    from src.config.collection import CollectionSettings

    monkeypatch.setattr(repair_mod.settings, "HISTORY_DIR", tmp_path, raising=False)
    assert repair_mod._capture_root(CollectionSettings(COLLECTION_ROOT=tmp_path / "cap-x")) == tmp_path / "cap-x"
    real_open_clients = repair_mod._open_clients

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        async def ensure_token(self, session):
            return "tok"

    monkeypatch.setattr(repair_mod, "_open_clients", lambda: (_Client(), _Session(), None, None))

    async def _fake_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame = _canon_bar_frame(snap_date, code)
            on_symbol(code, frame, _fake_entry(code, __import__("src.data.capture_contracts", fromlist=["CaptureDataset"]).CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=len(frame)))
        return pd.DataFrame()

    monkeypatch.setattr("src.backfill.intraday.collector.collect_intraday_bars", _fake_bars)

    with pytest.raises(ValueError, match="span"):
        repair_mod.main(["--start", "2026-01-01", "--end", "2026-03-15", "--dataset", "regular_bars", "--symbol", "005930"])
    many = []
    for idx in range(9):
        many.extend(["--symbol", f"{900000 + idx:06d}"])
    with pytest.raises(ValueError, match="attempts"):
        repair_mod.main(["--start", "2026-09-01", "--end", "2026-09-30", "--dataset", "regular_bars",
                         "--dataset", "regular_ticks", *many])
    repair_mod.main(["--start", "2020-01-01", "--end", "2020-01-01", "--dataset", "regular_bars",
                     "--symbol", "005930"])
    import json as _json

    payload = _json.loads((tmp_path / "capture" / "staging" / "intraday" / "repair-2020-01-01-2020-01-01.json").read_text())
    assert payload["attempts"][0]["status"] == "unsupported-horizon"
    repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "regular_bars",
                     "--symbol", "005930", "--max-pages", "5", "--apply"])
    from src.data import intraday_store

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    stored = pd.read_parquet(intraday_store.intraday_partition_path(1, "2026-09-03", "regular"))
    assert len(stored) == 1

    async def _boom(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        raise OSError("network down")

    monkeypatch.setattr("src.backfill.intraday.collector.collect_intraday_bars", _boom)
    with pytest.raises(RuntimeError, match="Repair publication failed"):
        repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "regular_bars",
                         "--symbol", "005930"])

    async def _fake_ok(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _canon_bar_frame(snap_date, code),
                      _fake_entry(code, __import__("src.data.capture_contracts", fromlist=["CaptureDataset"]).CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))
        return pd.DataFrame()

    monkeypatch.setattr("src.backfill.intraday.collector.collect_intraday_bars", _fake_ok)
    monkeypatch.setattr("src.data.intraday_store.write_intraday_partition", lambda *a, **k: 0)
    with pytest.raises(RuntimeError, match="verification failed"):
        repair_mod.main(["--start", "2026-09-03", "--end", "2026-09-03", "--dataset", "regular_bars",
                         "--symbol", "005930", "--apply"])

    monkeypatch.setattr(repair_mod.settings, "KIWOM_APP_KEY", "", raising=False)
    monkeypatch.setattr(repair_mod.settings, "KIWOOM_APP_KEY", "", raising=False)
    monkeypatch.setattr(repair_mod.settings, "LS_APP_KEY", "", raising=False)

    import asyncio as _asyncio

    async def _use_real_open():
        client, session_ctx, ls_client, kiwoom_client = real_open_clients()
        assert client is not None
        assert ls_client is None
        assert kiwoom_client is None
        async with session_ctx:
            return True

    assert _asyncio.run(_use_real_open())


def test_fixture_cohort_rejected_without_panel_listing(monkeypatch, tmp_path) -> None:
    """Fixture cohort rejected: COMPLETE manifest with unlisted symbols fails closed."""
    import logging

    import pytest

    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    store = _archive_store(tmp_path)
    cohort = _publish_cohort(store, "2026-09-07", ["000001", "000009"])
    entries_map = {
        code: (_empty_bar_frame_for("2026-09-07"), _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
        for code in ("000001", "000009")
    }
    fake_collect, _ = _archive_fakes(entries_map)
    _raw_archive_mocks(monkeypatch, tmp_path, fake_collect)
    _seed_panel(tmp_path, {"2026-09-04": ["005930", "000660"]})

    with pytest.raises(ValueError, match=cohort.cohort_id):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))
    from src.data import intraday_store

    assert not intraday_store.intraday_partition_path(1, "2026-09-07", "regular").exists()
    manifests = store.read_manifests("2026-09-07")
    assert not [m for m in manifests if m.context.run_id.startswith("archive-")]


def test_contaminated_previous_day_cohort_rejected(monkeypatch, tmp_path) -> None:
    """Contaminated previous-day cohort rejected: valid today, unlisted prev fails closed."""
    import pytest

    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["000001"])
    entries_map = {
        "005930": (_empty_bar_frame_for("2026-09-07"), _fake_entry("005930", CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN")),
        "000001": (_empty_bar_frame_for("2026-09-07"), _fake_entry("000001", CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN")),
    }
    fake_collect, _ = _archive_fakes(entries_map)
    _raw_archive_mocks(monkeypatch, tmp_path, fake_collect)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})

    with pytest.raises(ValueError, match="cohort_contamination"):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))
    manifests = store.read_manifests("2026-09-07")
    assert not [m for m in manifests if m.context.run_id.startswith("archive-")]


def test_genuine_cohort_after_holiday_passes(monkeypatch, tmp_path, caplog) -> None:
    """Genuine cohort after holiday passes: Tuesday resolves against preceding Friday panel."""
    import logging

    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-08", ["005930", "000660"])
    entries_map = {
        code: (_empty_bar_frame_for("2026-09-08"), _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
        for code in ("005930", "000660")
    }
    fake_collect, seen = _archive_fakes(entries_map)
    _raw_archive_mocks(monkeypatch, tmp_path, fake_collect)
    _seed_panel(tmp_path, {"2026-09-04": ["005930", "000660"]})

    with caplog.at_level(logging.INFO, logger=archive_intraday.logger.name):
        result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-08", profile=_raw_profile(tmp_path))

    assert result == (0, 0, 0)
    assert seen["codes"][0] == ["005930", "000660"]
    assert any("status=VERIFIED" in rec.message for rec in caplog.records)


def test_paper_follow_symbols_exempt(monkeypatch, tmp_path) -> None:
    """Paper-follow symbols exempt: ledger symbol absent from panel is appended without error."""
    import pandas as pd

    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    entries_map = {
        code: (_empty_bar_frame_for("2026-09-07"), _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
        for code in ("005930", "099999")
    }
    fake_collect, seen = _archive_fakes(entries_map)
    _raw_archive_mocks(monkeypatch, tmp_path, fake_collect)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})

    class _Ledger:
        def load_open_positions(self):
            return pd.DataFrame({"symbol": ["099999"]})

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", lambda *a, **k: _Ledger())

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))

    assert result == (0, 0, 0)
    assert "099999" in seen["codes"][0]


def test_missing_panel_fails_closed(monkeypatch, tmp_path) -> None:
    """Missing panel fails closed: FileNotFoundError propagates before any vendor call."""
    import pytest

    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])

    async def _never(*args, **kwargs):
        raise AssertionError("vendor must not be called without panel")

    _raw_archive_mocks(monkeypatch, tmp_path, _never)
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _never)
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _never)
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _never)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _never)

    with pytest.raises(FileNotFoundError):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))


def test_stale_panel_window_rejected(monkeypatch, tmp_path) -> None:
    """Stale panel window rejected: no rows in the 14d window fails closed."""
    import pytest

    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    fake_collect, _ = _archive_fakes(
        {"005930": (_empty_bar_frame_for("2026-09-07"), _fake_entry("005930", __import__("src.data.capture_contracts", fromlist=["CaptureDataset"]).CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))}
    )
    _raw_archive_mocks(monkeypatch, tmp_path, fake_collect)
    _seed_panel(tmp_path, {"2026-08-01": ["005930"]})

    with pytest.raises(ValueError, match="stale price_history"):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))


def _batched_frame(symbol: str):
    import pandas as pd

    return pd.DataFrame([{"symbol": symbol}])


def test_batched_partition_publisher_defers_write_until_threshold() -> None:
    from src.daily.archive_intraday import _BatchedPartitionPublisher
    from src.data.capture_contracts import CaptureDataset

    calls: list = []
    publisher = _BatchedPartitionPublisher(
        write_fn=lambda df, coverage: calls.append((df, coverage)) or 0, batch_size=3,
    )
    publisher.add("005930", _batched_frame("005930"),
                  _fake_entry("005930", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))
    publisher.add("000660", _batched_frame("000660"),
                  _fake_entry("000660", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))

    assert calls == []


def test_batched_partition_publisher_auto_flushes_at_threshold() -> None:
    from src.daily.archive_intraday import _BatchedPartitionPublisher
    from src.data.capture_contracts import CaptureDataset

    calls: list = []
    publisher = _BatchedPartitionPublisher(
        write_fn=lambda df, coverage: calls.append((df, coverage)) or len(df), batch_size=3,
    )
    codes = ["005930", "000660", "035420"]
    for code in codes:
        publisher.add(code, _batched_frame(code),
                      _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))

    assert len(calls) == 1
    frame, coverage = calls[0]
    assert len(frame) == 3
    assert set(coverage) == set(codes)
    assert publisher.flush() == 0
    assert len(calls) == 1


def test_batched_partition_publisher_flush_writes_partial_batch() -> None:
    from src.daily.archive_intraday import _BatchedPartitionPublisher
    from src.data.capture_contracts import CaptureDataset

    calls: list = []

    def _spy(df, coverage):
        calls.append((df, coverage))
        return 7

    publisher = _BatchedPartitionPublisher(write_fn=_spy, batch_size=10)
    for code in ("005930", "000660"):
        publisher.add(code, _batched_frame(code),
                      _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))

    assert publisher.flush() == 7
    assert len(calls) == 1
    frame, coverage = calls[0]
    assert len(frame) == 2
    assert set(coverage) == {"005930", "000660"}


def test_batched_partition_publisher_flush_without_buffer_is_noop() -> None:
    from src.daily.archive_intraday import _BatchedPartitionPublisher

    calls: list = []
    publisher = _BatchedPartitionPublisher(
        write_fn=lambda df, coverage: calls.append((df, coverage)) or 0, batch_size=3,
    )

    assert publisher.flush() == 0
    assert calls == []


def test_batched_partition_publisher_write_failure_propagates() -> None:
    import pytest

    from src.daily.archive_intraday import _BatchedPartitionPublisher
    from src.data.capture_contracts import CaptureDataset

    def _boom(df, coverage):
        raise ValueError("partition unavailable")

    publisher = _BatchedPartitionPublisher(write_fn=_boom, batch_size=2)
    publisher.add("005930", _batched_frame("005930"),
                  _fake_entry("005930", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))
    with pytest.raises(ValueError, match="partition unavailable"):
        publisher.add("000660", _batched_frame("000660"),
                      _fake_entry("000660", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))


def test_batched_partition_publisher_rejects_nonpositive_batch_size() -> None:
    import pytest

    from src.daily.archive_intraday import _BatchedPartitionPublisher

    with pytest.raises(ValueError, match="batch_size"):
        _BatchedPartitionPublisher(write_fn=lambda df, coverage: 0, batch_size=0)


def test_run_archive_batches_bar_writes_across_symbols(monkeypatch, tmp_path) -> None:
    from src.config.collection import CollectionSettings
    from src.config.market_session import INTRADAY_SESSION_REGULAR
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    codes = ["005930", "000660", "035420"]
    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", codes)
    _publish_cohort(store, "2026-09-04", ["005930"])

    async def _fake_bars(client, session, codes_arg, snap_date, bar_interval_minutes=1, **kwargs):
        import pandas as pd

        on_symbol = kwargs.get("on_symbol")
        for code in codes_arg:
            frame = _canon_bar_frame(snap_date, code)
            on_symbol(code, frame, _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=len(frame)))
        return pd.DataFrame()

    async def _fake_other(client, session, codes_arg, snap_date, bar_interval_minutes=1, **kwargs):
        import pandas as pd

        on_symbol = kwargs.get("on_symbol")
        for code in codes_arg:
            on_symbol(code, _empty_bar_frame_for(snap_date),
                      _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    async def _fake_ticks(client, session, codes_arg, snap_date, **kwargs):
        import pandas as pd

        on_symbol = kwargs.get("on_symbol")
        for code in codes_arg:
            on_symbol(code, _empty_bar_frame_for(snap_date),
                      _fake_entry(code, CaptureDataset.TRADE_TICKS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _fake_bars)
    _seed_panel(tmp_path, {"2026-09-04": codes, "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _fake_other)
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _fake_other)
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _fake_other)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _fake_ticks)

    writes: list = []

    def _spy_write(df, interval, snap_date, session, *, coverage=None, batch_rows=None):
        writes.append((session, len(df), sorted(coverage or {})))
        return len(df)

    monkeypatch.setattr(archive_intraday, "write_intraday_partition", _spy_write)
    monkeypatch.setattr(archive_intraday, "write_tick_partition", lambda *a, **k: 0)

    profile = CollectionSettings(COLLECTION_ROOT=tmp_path / "cap", COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE=2)
    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=profile)

    regular_writes = [item for item in writes if item[0] == INTRADAY_SESSION_REGULAR]
    assert [item[1] for item in regular_writes] == [2, 1]
    assert regular_writes[0][2] == ["000660", "005930"]
    assert regular_writes[1][2] == ["035420"]
    assert len(writes) == 2
    assert result == (3, 0, 0)


def test_collection_archive_symbol_batch_size_defaults_to_25() -> None:
    from src.config.collection import CollectionSettings

    assert CollectionSettings().COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE == 25


def test_collection_archive_symbol_batch_size_rejects_nonpositive() -> None:
    import pytest
    from pydantic import ValidationError

    from src.config.collection import CollectionSettings

    with pytest.raises(ValidationError):
        CollectionSettings(COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE=0)
