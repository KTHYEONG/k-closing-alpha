from __future__ import annotations

import datetime as _dt

import pytest

from src.daily.archive_intraday import _resolve_previous_trading_day as _real_resolve_previous_trading_day


@pytest.fixture(autouse=True)
def _stub_prev_trading_day(monkeypatch):
    """Keep `_run`-driving tests on the weekday calendar while the oracle is unit-stubbed.

    The production `_resolve_previous_trading_day` reaches the KIS oracle through
    `src.daily.collect`; tests that drive `run_intraday_archive` with a stubbed
    `is_kis_trading_day` gate would otherwise hit the live oracle.
    """

    async def _fake_prev_weekday(client, session, snapshot_date):
        day = _dt.date.fromisoformat(str(snapshot_date)) - _dt.timedelta(days=1)
        while day.weekday() >= 5:
            day -= _dt.timedelta(days=1)
        return day.isoformat()

    monkeypatch.setattr(
        "src.daily.archive_intraday._resolve_previous_trading_day", _fake_prev_weekday
    )


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
    from datetime import datetime as _dt

    from src.daily import archive_intraday
    from src.data.capture_contracts import SEOUL as _SEOUL

    captured: dict = {}

    def _fake_run(snapshot_date=None, **kwargs):
        captured["snapshot_date"] = snapshot_date
        captured["kwargs"] = kwargs
        return (1, 2, 3)

    monkeypatch.setattr(archive_intraday, "run_intraday_archive", _fake_run)
    monkeypatch.setattr(archive_intraday, "archive_phase_complete", lambda *a, **k: False)
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--date", "2026-09-04"])
    frozen = _dt(2026, 9, 4, 21, 0, tzinfo=_SEOUL)

    class _FrozenDt(_dt):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is not None:
                return frozen.astimezone(tz)
            return frozen

    monkeypatch.setattr(archive_intraday, "datetime", _FrozenDt)

    archive_intraday.main()

    assert captured["snapshot_date"] == "2026-09-04"
    assert captured["kwargs"]["phase"] == "all"


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
    monkeypatch.setattr(archive_intraday.settings, "KIWOOM_APP_KEY", "")

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

    with pytest.raises(ValueError, match="snapshot_date"):
        archive_intraday.run_intraday_archive(snapshot_date="bogus", profile=_raw_profile(tmp_path))
    with pytest.raises(ValueError, match="bar_interval"):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", bar_interval_minutes=0, profile=_raw_profile(tmp_path))

    async def _never(*args, **kwargs):
        raise AssertionError("must not collect without a cohort")

    # 거래일로 판정된 날의 코호트 부재는 여전히 fail-closed여야 한다(휴장일 SKIP과 구분).
    _raw_archive_mocks(monkeypatch, tmp_path, _never)
    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    with pytest.raises(FileNotFoundError, match="no qualifying cohort"):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-08", profile=_raw_profile(tmp_path))
    monkeypatch.setattr(archive_intraday, "CollectionSettings", lambda: _raw_profile(tmp_path))
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--date", "bogus-date"])
    with pytest.raises(SystemExit) as exc:
        archive_intraday.main()
    assert exc.value.code == 2
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--date", "2026-09-08"])
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


def test_run_archive_non_trading_day_without_cohort_skips_cleanly(monkeypatch, tmp_path) -> None:
    """A weekday holiday has no cohort (collect skips it); the archive must skip, not fail."""
    from src.daily import archive_intraday

    _archive_store(tmp_path)

    async def _never(*args, **kwargs):
        raise AssertionError("must not collect on non-trading day")

    _raw_archive_mocks(monkeypatch, tmp_path, _never)

    async def _not_trading(_client, _session, _date):
        return False

    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _not_trading)

    for phase in ("regular", "aftermarket"):
        result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-24", profile=_raw_profile(tmp_path), phase=phase)
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

    monkeypatch.setattr(repair_mod.settings, "KIWOOM_APP_KEY", "")
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


def test_run_intraday_archive_flags_shifted_day_degraded(monkeypatch, tmp_path) -> None:
    from datetime import date

    from src.config.collection import CollectionSettings
    from src.daily import archive_intraday
    from src.data.session_calendar import SessionDay, SessionKind

    target = date(2026, 11, 19)
    monkeypatch.setattr(
        archive_intraday, "resolve_session_day",
        lambda _d, **_k: SessionDay(trading_date=target, kind=SessionKind.SHIFTED, clock=None, provenance="krx_calendar"),
    )
    monkeypatch.setattr(archive_intraday, "_resolve_cohort_codes", lambda *a, **k: (["005930"], False))

    async def _is_trading(_c, _s, _d):
        return True

    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _is_trading)

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class _FakeKisClient:
        def create_session(self):
            return _FakeSession()

        async def ensure_token(self, session):
            return "tok"

    async def _noop_collector(*a, **k):
        return None

    monkeypatch.setattr(archive_intraday, "KisApiClient", lambda *a, **kw: _FakeKisClient())
    monkeypatch.setattr(archive_intraday, "LsApiClient", lambda: None)
    monkeypatch.setattr(archive_intraday, "KiwoomApiClient", lambda: None)
    monkeypatch.setattr(archive_intraday, "collect_intraday_bars", _noop_collector)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _noop_collector)
    outcomes: list[tuple] = []
    monkeypatch.setattr(
        archive_intraday, "record_run_outcome", lambda *a, **k: outcomes.append((a, k))
    )
    profile = CollectionSettings(
        COLLECTION_ROOT=tmp_path / "capture", _env_file=None
    )

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-11-19", phase="regular", profile=profile)

    assert result == (0, 0, 0)
    assert outcomes == [
        (("archive_intraday", "DEGRADED"), {"run_date": "2026-11-19", "reason": "shifted_session_standard_window"})
    ]


def _publish_evening_manifest(store, target_date, dataset, session, status, run_suffix):
    import datetime as _dt

    from src.data.capture_contracts import (
        SEOUL as _SEOUL,
        CaptureContext,
        CaptureManifest,
        CaptureStatus,
        CoverageEntry,
    )

    entry_status = CaptureStatus(status)
    entry = CoverageEntry(
        symbol="005930", dataset=dataset, venue="KRX", session=session, scheduled_at=None,
        status=entry_status, rows=1 if entry_status == CaptureStatus.COMPLETE else 0,
        first_event_time=None, last_event_time=None, reason="test-evening", raw_refs=(),
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=_dt.date.fromisoformat(target_date), run_id=f"archive-{target_date}-{session}-{run_suffix}",
            dataset=dataset, vendor="kis", endpoint="archive-task", symbol=None, venue="KRX",
            session=session, capture_reason="evening-archive", cohort_id=None, scheduled_at=None,
        ),
        cohort=None,
        completed_at=_dt.datetime(2026, 9, 22, 21, 0, tzinfo=_SEOUL),
        entries=(entry,),
        artifacts=(),
        status=entry_status,
    )
    store.publish_manifest(manifest)


def _freeze_archive_now(monkeypatch, archive_intraday, frozen):
    from datetime import datetime as _dt

    class _FrozenDt(_dt):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is not None:
                return frozen.astimezone(tz)
            return frozen

    monkeypatch.setattr(archive_intraday, "datetime", _FrozenDt)


def test_resolve_archive_target_date_next_morning_catchup() -> None:
    from datetime import datetime as _dt

    from src.daily.archive_intraday import resolve_archive_target_date
    from src.data.capture_contracts import SEOUL as _SEOUL

    now = _dt(2026, 9, 23, 7, 10, tzinfo=_SEOUL)

    assert resolve_archive_target_date(now, "regular") == "2026-09-22"


def test_resolve_archive_target_date_monday_catchup() -> None:
    from datetime import datetime as _dt

    from src.daily.archive_intraday import resolve_archive_target_date
    from src.data.capture_contracts import SEOUL as _SEOUL

    now = _dt(2026, 9, 28, 7, 10, tzinfo=_SEOUL)

    assert resolve_archive_target_date(now, "regular") == "2026-09-25"


def test_resolve_archive_target_date_on_time() -> None:
    from datetime import datetime as _dt

    from src.daily.archive_intraday import resolve_archive_target_date
    from src.data.capture_contracts import SEOUL as _SEOUL

    now = _dt(2026, 9, 23, 15, 40, 0, tzinfo=_SEOUL)

    assert resolve_archive_target_date(now, "regular") == "2026-09-23"


def test_resolve_archive_target_date_aftermarket_ready_time() -> None:
    from datetime import datetime as _dt

    from src.daily.archive_intraday import resolve_archive_target_date
    from src.data.capture_contracts import SEOUL as _SEOUL

    assert resolve_archive_target_date(_dt(2026, 9, 23, 20, 5, 0, tzinfo=_SEOUL), "aftermarket") == "2026-09-23"
    assert resolve_archive_target_date(_dt(2026, 9, 23, 0, 30, tzinfo=_SEOUL), "aftermarket") == "2026-09-22"
    assert resolve_archive_target_date(_dt(2026, 9, 23, 0, 30, tzinfo=_SEOUL), "all") == "2026-09-22"


def test_resolve_archive_target_date_rejects_naive_and_unknown_phase() -> None:
    from datetime import datetime as _dt

    import pytest

    from src.daily.archive_intraday import resolve_archive_target_date
    from src.data.capture_contracts import SEOUL as _SEOUL

    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_archive_target_date(_dt(2026, 9, 23, 7, 10), "regular")
    with pytest.raises(ValueError, match="Invalid phase"):
        resolve_archive_target_date(_dt(2026, 9, 23, 7, 10, tzinfo=_SEOUL), "bogus")


def test_archive_phase_complete_regular_complete_and_partial(tmp_path) -> None:
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    store = _archive_store(tmp_path)
    _publish_evening_manifest(store, "2026-09-22", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", "bars")
    _publish_evening_manifest(store, "2026-09-22", CaptureDataset.TRADE_TICKS, "regular", "COMPLETE", "ticks")

    assert archive_intraday.archive_phase_complete(store, "2026-09-22", "regular") is True

    store2 = _archive_store(tmp_path / "cap2")
    _publish_evening_manifest(store2, "2026-09-22", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", "bars")
    _publish_evening_manifest(store2, "2026-09-22", CaptureDataset.TRADE_TICKS, "regular", "PARTIAL", "ticks")

    assert archive_intraday.archive_phase_complete(store2, "2026-09-22", "regular") is False


def test_archive_phase_complete_ignores_non_evening_and_unreadable(tmp_path) -> None:
    import datetime as _dt

    from src.daily import archive_intraday
    from src.data.capture_contracts import (
        SEOUL as _SEOUL,
        CaptureContext,
        CaptureDataset,
        CaptureManifest,
        CaptureStatus,
        CoverageEntry,
    )

    store = _archive_store(tmp_path)
    entry = CoverageEntry(
        symbol="005930", dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="regular",
        scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1,
        first_event_time=None, last_event_time=None, reason="test", raw_refs=(),
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=_dt.date(2026, 9, 22), run_id="decision-1", dataset=CaptureDataset.SCAN,
            vendor="owner-local", endpoint="decision-input", symbol=None, venue="KRX",
            session="regular", capture_reason="decision-input", cohort_id=None, scheduled_at=None,
        ),
        cohort=None,
        completed_at=_dt.datetime(2026, 9, 22, 16, 0, tzinfo=_SEOUL),
        entries=(entry,),
        artifacts=(),
        status=CaptureStatus.COMPLETE,
    )
    store.publish_manifest(manifest)

    assert archive_intraday.archive_phase_complete(store, "2026-09-22", "regular") is False

    class _BrokenStore:
        def read_manifests(self, _date):
            raise ValueError("unreadable manifest evidence")

    assert archive_intraday.archive_phase_complete(_BrokenStore(), "2026-09-22", "regular") is False


def test_archive_phase_complete_krx_required_only_from_start_date(tmp_path) -> None:
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    store = _archive_store(tmp_path)
    _publish_evening_manifest(store, "2026-09-10", CaptureDataset.MINUTE_BARS, "nxt_premarket", "COMPLETE", "pre")
    _publish_evening_manifest(store, "2026-09-10", CaptureDataset.MINUTE_BARS, "nxt_aftermarket", "COMPLETE", "after")

    assert archive_intraday.archive_phase_complete(store, "2026-09-10", "aftermarket") is True

    store2 = _archive_store(tmp_path / "cap2")
    _publish_evening_manifest(store2, "2026-09-22", CaptureDataset.MINUTE_BARS, "nxt_premarket", "COMPLETE", "pre")
    _publish_evening_manifest(store2, "2026-09-22", CaptureDataset.MINUTE_BARS, "nxt_aftermarket", "COMPLETE", "after")

    assert archive_intraday.archive_phase_complete(store2, "2026-09-22", "aftermarket") is False


def test_archive_main_aftermarket_catchup_refused(monkeypatch, tmp_path) -> None:
    import datetime as _dt

    from src.daily import archive_intraday
    from src.data.capture_contracts import SEOUL as _SEOUL

    _freeze_archive_now(monkeypatch, archive_intraday, _dt.datetime(2026, 9, 23, 0, 30, tzinfo=_SEOUL))
    monkeypatch.setattr(archive_intraday, "CollectionSettings", lambda: _raw_profile(tmp_path))
    calls: list = []
    monkeypatch.setattr(archive_intraday, "run_intraday_archive", lambda **k: calls.append(k) or (0, 0, 0))
    outcomes: list = []
    monkeypatch.setattr(archive_intraday, "record_run_outcome", lambda *a, **k: outcomes.append((a, k)))
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--phase", "aftermarket"])

    archive_intraday.main()

    assert calls == []
    assert outcomes == [(("archive_intraday", "DEGRADED"), {"run_date": "2026-09-22", "reason": "aftermarket_not_replayable"})]


def test_archive_main_already_archived_skipped(monkeypatch, tmp_path, caplog) -> None:
    import datetime as _dt
    import logging

    from src.daily import archive_intraday
    from src.data.capture_contracts import SEOUL as _SEOUL, CaptureDataset

    store = _archive_store(tmp_path)
    _publish_evening_manifest(store, "2026-09-22", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", "bars")
    _publish_evening_manifest(store, "2026-09-22", CaptureDataset.TRADE_TICKS, "regular", "COMPLETE", "ticks")
    _freeze_archive_now(monkeypatch, archive_intraday, _dt.datetime(2026, 9, 22, 21, 0, tzinfo=_SEOUL))
    monkeypatch.setattr(archive_intraday, "CollectionSettings", lambda: _raw_profile(tmp_path))
    calls: list = []
    monkeypatch.setattr(archive_intraday, "run_intraday_archive", lambda **k: calls.append(k) or (0, 0, 0))
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--phase", "regular", "--date", "2026-09-22"])

    with caplog.at_level(logging.INFO, logger=archive_intraday.logger.name):
        archive_intraday.main()

    assert calls == []
    assert any("already_archived" in rec.message for rec in caplog.records)


def test_archive_main_all_past_runs_regular_only(monkeypatch, tmp_path) -> None:
    import datetime as _dt

    from src.daily import archive_intraday
    from src.data.capture_contracts import SEOUL as _SEOUL

    _freeze_archive_now(monkeypatch, archive_intraday, _dt.datetime(2026, 9, 23, 7, 10, tzinfo=_SEOUL))
    monkeypatch.setattr(archive_intraday, "CollectionSettings", lambda: _raw_profile(tmp_path))
    calls: list = []
    monkeypatch.setattr(archive_intraday, "run_intraday_archive", lambda **k: calls.append(k) or (1, 0, 1))
    outcomes: list = []
    monkeypatch.setattr(archive_intraday, "record_run_outcome", lambda *a, **k: outcomes.append((a, k)))
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--phase", "all"])

    archive_intraday.main()

    assert [item["phase"] for item in calls] == ["regular"]
    assert calls[0]["snapshot_date"] == "2026-09-22"
    assert outcomes == [(("archive_intraday", "DEGRADED"), {"run_date": "2026-09-22", "reason": "aftermarket_not_replayable"})]


def test_archive_main_all_past_skips_when_regular_done(monkeypatch, tmp_path, caplog) -> None:
    import datetime as _dt
    import logging

    from src.daily import archive_intraday
    from src.data.capture_contracts import SEOUL as _SEOUL, CaptureDataset

    store = _archive_store(tmp_path)
    _publish_evening_manifest(store, "2026-09-22", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", "bars")
    _publish_evening_manifest(store, "2026-09-22", CaptureDataset.TRADE_TICKS, "regular", "COMPLETE", "ticks")
    _freeze_archive_now(monkeypatch, archive_intraday, _dt.datetime(2026, 9, 23, 7, 10, tzinfo=_SEOUL))
    monkeypatch.setattr(archive_intraday, "CollectionSettings", lambda: _raw_profile(tmp_path))
    calls: list = []
    monkeypatch.setattr(archive_intraday, "run_intraday_archive", lambda **k: calls.append(k) or (0, 0, 0))
    outcomes: list = []
    monkeypatch.setattr(archive_intraday, "record_run_outcome", lambda *a, **k: outcomes.append((a, k)))
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--phase", "all"])

    with caplog.at_level(logging.INFO, logger=archive_intraday.logger.name):
        archive_intraday.main()

    assert calls == []
    assert outcomes == [(("archive_intraday", "DEGRADED"), {"run_date": "2026-09-22", "reason": "aftermarket_not_replayable"})]
    assert any("already_archived" in rec.message for rec in caplog.records)


def test_run_archive_default_profile_archives_with_capture_evidence(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.config.market_session import (
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
        INTRADAY_SESSION_NXT_PREMARKET,
    )
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset, CaptureStatus
    from src.data.capture_store import CaptureStore
    from src.data.intraday_schema import normalize_tick_frame

    store = CaptureStore(tmp_path / "capture")
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])

    def _tick_frame(symbol):
        raw = pd.DataFrame({"time": ["090300"], "close": [71000], "jdiff_vol": [7]})
        return normalize_tick_frame(raw, "ls", "2026-09-07", symbol)

    async def _fake_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _canon_bar_frame(snap_date, code),
                      _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))
        return pd.DataFrame()

    def _fake_session_bars_factory(session_tag):
        async def _fake_session_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
            on_symbol = kwargs.get("on_symbol")
            for code in codes:
                on_symbol(code, _canon_bar_frame(snap_date, code),
                          _fake_entry(code, CaptureDataset.MINUTE_BARS, session_tag, "COMPLETE", rows=1))
            return pd.DataFrame()

        return _fake_session_bars

    async def _fake_ticks(client, session, codes, snap_date, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _tick_frame(code),
                      _fake_entry(code, CaptureDataset.TRADE_TICKS, "regular", "COMPLETE", rows=1))
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _fake_bars)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _fake_session_bars_factory(INTRADAY_SESSION_NXT_AFTERMARKET))
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _fake_session_bars_factory(INTRADAY_SESSION_NXT_PREMARKET))
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _fake_session_bars_factory(INTRADAY_SESSION_KRX_AFTERMARKET))
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _fake_ticks)

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07")

    assert result[0] >= 1
    manifests = store.read_manifests("2026-09-07")
    assert any(item.context.run_id.startswith("archive-2026-09-07-") for item in manifests)


def test_run_archive_phase_gating_on_capture_path(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])

    calls: list[str] = []

    def _tracked(name: str):
        async def _fn(*a, **kw):
            calls.append(name)
            return pd.DataFrame()

        return _fn

    _raw_archive_mocks(monkeypatch, tmp_path, _tracked("bars"))
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _tracked("nxt_after"))
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _tracked("nxt_pre"))
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _tracked("krx_after"))
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _tracked("ticks"))

    assert archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="regular") == (0, 0, 0)
    assert calls == ["bars", "ticks"]
    calls.clear()
    assert archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="aftermarket") == (0, 0, 0)
    assert calls == ["nxt_after", "nxt_pre", "krx_after"]


def test_run_archive_unknown_phase_rejected_before_io(monkeypatch, tmp_path) -> None:
    import pytest

    from src.daily import archive_intraday

    constructed: list = []
    _raw_archive_mocks(monkeypatch, tmp_path, None)

    def _forbidden_client(*a: object, **kw: object) -> object:
        raise AssertionError("must not construct")

    monkeypatch.setattr(archive_intraday, "KisApiClient", _forbidden_client)

    with pytest.raises(ValueError, match="Invalid phase"):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="bogus")


def test_archive_main_skips_legacy_watchlist_read(monkeypatch, tmp_path, caplog) -> None:
    import datetime as _dt
    import logging

    from src.daily import archive_intraday
    from src.data.capture_contracts import SEOUL as _SEOUL, CaptureDataset

    store = _archive_store(tmp_path)
    _publish_evening_manifest(store, "2026-09-22", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", "bars")
    _publish_evening_manifest(store, "2026-09-22", CaptureDataset.TRADE_TICKS, "regular", "COMPLETE", "ticks")
    _freeze_archive_now(monkeypatch, archive_intraday, _dt.datetime(2026, 9, 22, 21, 0, tzinfo=_SEOUL))
    monkeypatch.setattr(archive_intraday, "CollectionSettings", lambda: _raw_profile(tmp_path))
    calls: list = []
    monkeypatch.setattr(archive_intraday, "run_intraday_archive", lambda **k: calls.append(k) or (0, 0, 0))
    monkeypatch.setattr("sys.argv", ["archive_intraday", "--phase", "regular", "--date", "2026-09-22"])

    def _forbidden(**kw):
        raise AssertionError("main must not read the legacy watchlist")

    monkeypatch.setattr(archive_intraday.archive, "fetch_archive_snapshot", _forbidden)

    with caplog.at_level(logging.INFO, logger=archive_intraday.logger.name):
        archive_intraday.main()

    assert calls == []
    assert any("already_archived" in rec.message for rec in caplog.records)


def test_run_archive_kiwoom_client_built_only_with_canonical_key(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])

    built: list = []
    seen: dict = {}

    def _tracked(name: str):
        async def _fn(*a, **kw):
            seen[name] = kw.get("kiwoom_client")
            return pd.DataFrame()

        return _fn

    _raw_archive_mocks(monkeypatch, tmp_path, _tracked("bars"))
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _tracked("nxt_after"))
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _tracked("nxt_pre"))
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _tracked("krx_after"))
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _tracked("ticks"))

    def _kiwoom_factory(*a: object, **kw: object) -> object:
        inst = object()
        built.append(inst)
        return inst

    monkeypatch.setattr(archive_intraday, "KiwoomApiClient", _kiwoom_factory)

    monkeypatch.setattr(archive_intraday.settings, "KIWOOM_APP_KEY", "k")
    assert archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="regular") == (0, 0, 0)
    assert len(built) == 1
    assert seen["ticks"] is built[0]

    built.clear()
    seen.clear()
    monkeypatch.setattr(archive_intraday.settings, "KIWOOM_APP_KEY", "")
    assert archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="regular") == (0, 0, 0)
    assert built == []
    assert seen["ticks"] is None


def test_archive_skips_holiday_for_previous_cohort(monkeypatch, tmp_path) -> None:
    """First run after a weekday holiday unions today's cohort with the real prior session."""
    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-28", ["005930"])
    _publish_cohort(store, "2026-09-23", ["009900"])
    entries_map = {
        code: (_empty_bar_frame_for("2026-09-28"), _fake_entry(code, __import__("src.data.capture_contracts", fromlist=["CaptureDataset"]).CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
        for code in ("005930", "009900")
    }
    fake_collect, seen = _archive_fakes(entries_map)
    _raw_archive_mocks(monkeypatch, tmp_path, fake_collect)
    _seed_panel(tmp_path, {"2026-09-25": ["005930"], "2026-09-22": ["009900"]})

    async def _resolver(client, session, snapshot_date):
        assert snapshot_date == "2026-09-28"
        return "2026-09-23"

    monkeypatch.setattr(archive_intraday, "_resolve_previous_trading_day", _resolver)

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-28", profile=_raw_profile(tmp_path))

    assert result == (0, 0, 0)
    assert sorted(seen["codes"][0]) == ["005930", "009900"]
    codes, incomplete = archive_intraday._resolve_cohort_codes(
        "2026-09-28", _raw_profile(tmp_path), store, previous_trading_day="2026-09-23"
    )
    assert sorted(codes) == ["005930", "009900"]
    assert incomplete is False


def test_unresolved_previous_day_degrades_never_aborts(monkeypatch, tmp_path, caplog) -> None:
    import logging

    from src.daily import archive_intraday

    async def _boom(client, session, snapshot_date):
        raise RuntimeError("oracle down")

    monkeypatch.setattr("src.daily.collect.resolve_prev_trading_day_kis", _boom)

    with caplog.at_level(logging.WARNING):
        resolved = __import__("asyncio").run(
            _real_resolve_previous_trading_day(object(), object(), "2026-09-28")
        )
    assert resolved is None
    assert "reason=previous_day_unresolved" in caplog.text

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-28", ["005930"])
    _seed_panel(tmp_path, {"2026-09-25": ["005930"]})

    codes, incomplete = archive_intraday._resolve_cohort_codes(
        "2026-09-28", _raw_profile(tmp_path), store, previous_trading_day=None
    )
    assert codes == ["005930"]
    assert incomplete is True


def test_resolved_but_missing_previous_cohort_keeps_signal(monkeypatch, tmp_path, caplog) -> None:
    import logging

    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-28", ["005930"])
    _seed_panel(tmp_path, {"2026-09-25": ["005930"]})

    with caplog.at_level(logging.WARNING):
        codes, incomplete = archive_intraday._resolve_cohort_codes(
            "2026-09-28", _raw_profile(tmp_path), store, previous_trading_day="2026-09-23"
        )
    assert codes == ["005930"]
    assert incomplete is True
    assert "missing_previous" in caplog.text


def test_resolve_previous_trading_day_returns_oracle_date(monkeypatch) -> None:
    import asyncio

    import pandas as pd

    async def _oracle(client, session, decision_date, **kwargs):
        return pd.Timestamp("2026-09-23")

    monkeypatch.setattr("src.daily.collect.resolve_prev_trading_day_kis", _oracle)

    assert asyncio.run(_real_resolve_previous_trading_day(object(), object(), "2026-09-28")) == "2026-09-23"


def test_unresolved_previous_day_still_follows_paper_positions(monkeypatch, tmp_path) -> None:
    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-28", ["005930"])
    _seed_panel(tmp_path, {"2026-09-25": ["005930"]})
    monkeypatch.setattr(archive_intraday, "_paper_follow_symbols", lambda: ["009900"])

    codes, incomplete = archive_intraday._resolve_cohort_codes(
        "2026-09-28", _raw_profile(tmp_path), store, previous_trading_day=None
    )

    assert sorted(codes) == ["005930", "009900"]
    assert incomplete is True
