from __future__ import annotations

import datetime as _dt

import pytest

from src.daily.archive_intraday import _resolve_previous_trading_day as _real_resolve_previous_trading_day
from src.execution.paper_broker import HeldRoster


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
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", fake_collect)


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
    assert len([item for item in manifests if item.context.run_id.startswith("archive-")]) == 7


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
    assert len(archive_manifests) == 14
    assert len({item.context.run_id for item in archive_manifests}) == 14


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
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _fake_ticks)

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
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _never)

    async def _not_trading(_client, _session, _date):
        return False

    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _not_trading)
    recorded: list[tuple] = []
    monkeypatch.setattr(
        archive_intraday, "record_run_outcome", lambda job, outcome, **kw: recorded.append((job, outcome, kw))
    )

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))

    assert result == (0, 0, 0)
    assert recorded == [
        (
            "archive_intraday",
            "SKIPPED",
            {"run_date": "2026-09-07", "reason": "non_trading_day", "metrics": {"session": "CLOSED"}},
        )
    ]


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
    recorded: list[tuple] = []
    monkeypatch.setattr(
        archive_intraday, "record_run_outcome", lambda job, outcome, **kw: recorded.append((job, outcome, kw))
    )

    for phase in ("regular", "aftermarket"):
        result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-24", profile=_raw_profile(tmp_path), phase=phase)
        assert result == (0, 0, 0)
    assert len(recorded) == 2
    assert all(r[0] == "archive_intraday" and r[1] == "SKIPPED" for r in recorded)


def test_run_archive_closed_session_skips_before_kis_oracle(monkeypatch, tmp_path) -> None:
    from datetime import date

    from src.daily import archive_intraday
    from src.data.session_calendar import SessionDay, SessionKind

    _archive_store(tmp_path)

    async def _never(*args, **kwargs):
        raise AssertionError("must not collect on non-trading day")

    _raw_archive_mocks(monkeypatch, tmp_path, _never)
    monkeypatch.setattr(
        archive_intraday, "resolve_session_day",
        lambda _d, **_k: SessionDay(trading_date=date(2026, 10, 9), kind=SessionKind.CLOSED, clock=None, provenance="test"),
    )

    async def _boom_oracle(*args, **kwargs):
        raise AssertionError("CLOSED gate must skip before the KIS oracle")

    monkeypatch.setattr(archive_intraday, "is_kis_trading_day", _boom_oracle)
    recorded: list[tuple] = []
    monkeypatch.setattr(
        archive_intraday, "record_run_outcome", lambda job, outcome, **kw: recorded.append((job, outcome, kw))
    )

    assert archive_intraday.run_intraday_archive(snapshot_date="2026-10-09", profile=_raw_profile(tmp_path)) == (0, 0, 0)
    assert recorded == [
        (
            "archive_intraday",
            "SKIPPED",
            {"run_date": "2026-10-09", "reason": "non_trading_day", "metrics": {"session": "CLOSED"}},
        )
    ]


def test_run_archive_full_complete_and_partial_entries(monkeypatch, tmp_path) -> None:
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
                on_symbol(code, _empty_bar_frame_for(snap_date),
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
                on_symbol(code, pd.DataFrame(),
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

    async def _fake_after_ticks(client, session, codes, snap_date, **kwargs):
        from src.data.capture_contracts import CaptureStatus as _TickStatus

        venue = kwargs.get("venue")
        session_tag = INTRADAY_SESSION_KRX_AFTERMARKET if venue == "KRX" else INTRADAY_SESSION_NXT_AFTERMARKET
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame = _tick_frame(code)
            on_symbol(code, frame, _entry(code, CaptureDataset.TRADE_TICKS, session_tag, _TickStatus.COMPLETE, rows=len(frame), refs=[shared]))
        return pd.DataFrame()

    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _fake_after_ticks)

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path))

    assert result == (1, 4, 1)
    manifests = store.read_manifests("2026-09-07")

    def _by_stream(dataset, session: str):
        return next(
            item for item in manifests
            if item.context.endpoint == "archive-task"
            and item.context.dataset is dataset
            and item.context.session == session
        )

    assert _by_stream(CaptureDataset.MINUTE_BARS, "regular").status.value == "PARTIAL"
    assert _by_stream(CaptureDataset.TRADE_TICKS, "regular").status.value == "PARTIAL"
    assert _by_stream(CaptureDataset.MINUTE_BARS, "nxt_aftermarket").status.value == "COMPLETE"
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
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _never)

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


def _batched_frame(symbol: str, nrows: int = 1):
    import pandas as pd

    return pd.DataFrame([{"symbol": symbol}] * int(nrows))


def test_batched_partition_publisher_defers_write_until_threshold() -> None:
    from src.daily.archive_intraday import _BatchedPartitionPublisher
    from src.data.capture_contracts import CaptureDataset

    calls: list = []
    publisher = _BatchedPartitionPublisher(
        write_fn=lambda df, coverage: calls.append((df, coverage)) or 0, max_rows=3,
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
        write_fn=lambda df, coverage: calls.append((df, coverage)) or len(df), max_rows=3,
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

    publisher = _BatchedPartitionPublisher(write_fn=_spy, max_rows=10)
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
        write_fn=lambda df, coverage: calls.append((df, coverage)) or 0, max_rows=3,
    )

    assert publisher.flush() == 0
    assert calls == []


def test_batched_partition_publisher_write_failure_propagates() -> None:
    import pytest

    from src.daily.archive_intraday import _BatchedPartitionPublisher
    from src.data.capture_contracts import CaptureDataset

    def _boom(df, coverage):
        raise ValueError("partition unavailable")

    publisher = _BatchedPartitionPublisher(write_fn=_boom, max_rows=2)
    publisher.add("005930", _batched_frame("005930"),
                  _fake_entry("005930", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))
    with pytest.raises(ValueError, match="partition unavailable"):
        publisher.add("000660", _batched_frame("000660"),
                      _fake_entry("000660", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))


def test_batched_partition_publisher_rejects_nonpositive_batch_size() -> None:
    import pytest

    from src.daily.archive_intraday import _BatchedPartitionPublisher

    with pytest.raises(ValueError, match="max_rows"):
        _BatchedPartitionPublisher(write_fn=lambda df, coverage: 0, max_rows=0)


def test_batched_partition_publisher_rewrite_count_bounded_by_rows() -> None:
    from src.daily.archive_intraday import _BatchedPartitionPublisher
    from src.data.capture_contracts import CaptureDataset

    calls: list = []
    publisher = _BatchedPartitionPublisher(
        write_fn=lambda df, coverage: calls.append((len(df), set(coverage))) or len(df), max_rows=100_000,
    )
    for idx in range(300):
        code = f"{idx:06d}"
        publisher.add(code, _batched_frame(code, 1000),
                      _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1000))
    publisher.flush()

    assert len(calls) == 3
    assert sum(n for n, _ in calls) == 300_000


def test_batched_partition_publisher_small_session_flushes_once() -> None:
    from src.config.collection import CollectionSettings
    from src.daily.archive_intraday import _BatchedPartitionPublisher
    from src.data.capture_contracts import CaptureDataset

    calls: list = []
    publisher = _BatchedPartitionPublisher(
        write_fn=lambda df, coverage: calls.append((len(df), set(coverage))) or len(df),
        max_rows=int(CollectionSettings().COLLECTION_ARCHIVE_PUBLISH_ROWS),
    )
    for idx in range(300):
        code = f"{idx:06d}"
        publisher.add(code, _batched_frame(code, 400),
                      _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=400))
    publisher.flush()

    assert len(calls) == 1
    assert calls[0][0] == 120_000


def test_batched_partition_publisher_preserves_zero_row_coverage() -> None:
    import pandas as pd

    from src.daily.archive_intraday import _BatchedPartitionPublisher
    from src.data.capture_contracts import CaptureDataset

    calls: list = []
    publisher = _BatchedPartitionPublisher(
        write_fn=lambda df, coverage: calls.append((df, coverage)) or 0, max_rows=1_000_000,
    )
    publisher.add("005930", _batched_frame("005930", 2),
                  _fake_entry("005930", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=2))
    from src.data.capture_contracts import ArtifactRef, CoverageEntry
    empty_entry = CoverageEntry(
        symbol="000000", dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="regular",
        scheduled_at=None, status=__import__("src.data.capture_contracts", fromlist=["CaptureStatus"]).CaptureStatus.NO_TRADES,
        rows=0, first_event_time=None, last_event_time=None, reason="verified empty with proof",
        raw_refs=(ArtifactRef(path="raw/000000.json.gz", sha256="a" * 64, bytes=8, rows=0),),
    )
    publisher.add("000000", pd.DataFrame([{"symbol": "000000"}]).iloc[0:0], empty_entry)
    publisher.flush()

    assert len(calls) == 1
    _, coverage = calls[0]
    assert set(coverage) == {"005930", "000000"}


def test_batched_partition_publisher_rejects_duplicate_symbol() -> None:
    import pytest

    from src.daily.archive_intraday import _BatchedPartitionPublisher
    from src.data.capture_contracts import CaptureDataset

    publisher = _BatchedPartitionPublisher(write_fn=lambda df, coverage: 0, max_rows=100)
    publisher.add("005930", _batched_frame("005930"),
                  _fake_entry("005930", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))
    with pytest.raises(ValueError, match="Duplicate"):
        publisher.add("005930", _batched_frame("005930"),
                      _fake_entry("005930", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", rows=1))


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

    def _spy_write(df, interval, snap_date, session, *, coverage=None, batch_rows=None, **kwargs):
        writes.append((session, len(df), sorted(coverage or {})))
        return len(df)

    monkeypatch.setattr(archive_intraday, "write_intraday_partition", _spy_write)
    monkeypatch.setattr(archive_intraday, "write_tick_partition", lambda *a, **k: 0)

    profile = CollectionSettings(COLLECTION_ROOT=tmp_path / "cap", COLLECTION_ARCHIVE_PUBLISH_ROWS=2)
    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=profile)

    regular_writes = [item for item in writes if item[0] == INTRADAY_SESSION_REGULAR]
    assert [item[1] for item in regular_writes] == [2, 1]
    assert regular_writes[0][2] == ["000660", "005930"]
    assert regular_writes[1][2] == ["035420"]
    assert len(writes) == 2
    assert result == (3, 0, 0)


def test_collection_archive_publish_rows_defaults_to_2m() -> None:
    from src.config.collection import CollectionSettings

    assert CollectionSettings().COLLECTION_ARCHIVE_PUBLISH_ROWS == 2_000_000


def test_collection_archive_publish_rows_rejects_nonpositive() -> None:
    import pytest
    from pydantic import ValidationError

    from src.config.collection import CollectionSettings

    with pytest.raises(ValidationError):
        CollectionSettings(COLLECTION_ARCHIVE_PUBLISH_ROWS=0)


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

    async def _fake_after_ticks(client, session, codes, snap_date, **kwargs):
        venue = kwargs.get("venue")
        session_tag = INTRADAY_SESSION_KRX_AFTERMARKET if venue == "KRX" else INTRADAY_SESSION_NXT_AFTERMARKET
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame = _tick_frame(code)
            on_symbol(code, frame, _fake_entry(code, CaptureDataset.TRADE_TICKS, session_tag, "COMPLETE", rows=len(frame)))
        return pd.DataFrame()

    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _fake_after_ticks)

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
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _tracked("after_ticks"))

    assert archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="regular") == (0, 0, 0)
    assert calls == ["bars", "ticks"]
    calls.clear()
    assert archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="aftermarket") == (0, 0, 0)
    assert calls == ["nxt_after", "nxt_pre", "krx_after", "after_ticks", "after_ticks"]


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
        "2026-09-28", _raw_profile(tmp_path), store, previous_trading_day="2026-09-23", held=HeldRoster((), True, None)
    )
    assert sorted(codes) == ["005930", "009900"]
    assert incomplete is False


def test_unresolved_previous_day_degrades_never_aborts(monkeypatch, tmp_path, caplog) -> None:
    import logging

    from src.daily import archive_intraday

    async def _boom(client, session, snapshot_date):
        raise RuntimeError("oracle down")

    monkeypatch.setattr("src.daily.archive_intraday.resolve_prev_trading_day_kis", _boom)

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
        "2026-09-28", _raw_profile(tmp_path), store, previous_trading_day=None, held=HeldRoster((), True, None)
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
            "2026-09-28", _raw_profile(tmp_path), store, previous_trading_day="2026-09-23", held=HeldRoster((), True, None)
        )
    assert codes == ["005930"]
    assert incomplete is True
    assert "missing_previous" in caplog.text


def test_resolve_previous_trading_day_returns_oracle_date(monkeypatch) -> None:
    import asyncio

    import pandas as pd

    async def _oracle(client, session, decision_date, **kwargs):
        return pd.Timestamp("2026-09-23")

    monkeypatch.setattr("src.daily.archive_intraday.resolve_prev_trading_day_kis", _oracle)

    assert asyncio.run(_real_resolve_previous_trading_day(object(), object(), "2026-09-28")) == "2026-09-23"


def test_unresolved_previous_day_still_follows_paper_positions(monkeypatch, tmp_path) -> None:
    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-28", ["005930"])
    _seed_panel(tmp_path, {"2026-09-25": ["005930"]})

    codes, incomplete = archive_intraday._resolve_cohort_codes(
        "2026-09-28", _raw_profile(tmp_path), store, previous_trading_day=None, held=HeldRoster(("009900",), True, None)
    )

    assert sorted(codes) == ["005930", "009900"]
    assert incomplete is True


def test_aftermarket_phase_writes_both_tick_partitions_and_manifests(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.daily import archive_intraday
    from src.data import intraday_store
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930", "000660"])
    _publish_cohort(store, "2026-09-04", ["005930", "000660"])

    def _tick_frame(symbol):
        raw = pd.DataFrame({"time": ["160100"], "close": [71000], "jdiff_vol": [7]})
        from src.data.intraday_schema import normalize_tick_frame

        return normalize_tick_frame(raw, "ls", "2026-09-07", symbol)

    def _tick_entry(symbol, session):
        return CoverageEntry(
            symbol=symbol, dataset=CaptureDataset.TRADE_TICKS, venue="KRX" if session == "krx_aftermarket" else "NXT",
            session=session, scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1,
            first_event_time=None, last_event_time=None, reason="test-after-ticks", raw_refs=(),
        )

    async def _fake_after_ticks(client, session, codes, snap_date, **kwargs):
        venue = kwargs.get("venue")
        session_tag = "krx_aftermarket" if venue == "KRX" else "nxt_aftermarket"
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame = _tick_frame(code)
            on_symbol(code, frame, _tick_entry(code, session_tag))
        return pd.DataFrame()

    async def _fake_session_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame = _canon_bar_frame(snap_date, code)
            on_symbol(code, frame, CoverageEntry(
                symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue="NXT",
                session="nxt_aftermarket", scheduled_at=None, status=CaptureStatus.COMPLETE,
                rows=len(frame), first_event_time=None, last_event_time=None,
                reason="test-nxt-bars", raw_refs=(),
            ))
        return pd.DataFrame()

    async def _noop(client, session, codes, snap_date, *args, **kwargs):
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _noop)
    _seed_panel(tmp_path, {"2026-09-04": ["005930", "000660"], "2026-09-03": ["005930", "000660"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _fake_session_bars)
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _fake_after_ticks)

    archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="aftermarket")

    assert intraday_store.tick_partition_path("2026-09-07", "krx_aftermarket").exists()
    assert intraday_store.tick_partition_path("2026-09-07", "nxt_aftermarket").exists()
    manifests = store.read_manifests("2026-09-07")
    tick_manifests = [
        item for item in manifests
        if item.context.dataset is CaptureDataset.TRADE_TICKS and item.context.endpoint == "archive-task"
    ]
    by_session = {item.context.session: item for item in tick_manifests}
    assert by_session["krx_aftermarket"].status is CaptureStatus.COMPLETE
    assert by_session["nxt_aftermarket"].status is CaptureStatus.COMPLETE


def test_nxt_tick_cohort_is_nxt_certified_bar_subset(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["000001", "000002"])
    _publish_cohort(store, "2026-09-04", ["000001", "000002"])

    import datetime as _dt

    from src.data.capture_contracts import (
        SEOUL as _SEOUL,
        CapturedResponse,
        CaptureContext,
        CoverageEntry as _CoverageEntry,
    )

    _seed_ctx = CaptureContext(
        trading_date=_dt.date(2026, 9, 7), run_id="seed", dataset=CaptureDataset.SCAN,
        vendor="owner-local", endpoint="seed", symbol=None, venue="NXT",
        session="nxt_aftermarket", capture_reason="seed", cohort_id=None, scheduled_at=None,
    )
    _now = _dt.datetime.now(_SEOUL)
    shared = store.append_response(CapturedResponse(
        context=_seed_ctx, request_started_at=_now, received_at=_now, payload={"seed": True},
        status=CaptureStatus.COMPLETE, source_timestamp=None, source_published_at=None,
        page_index=0, attempt_index=0, continuation={}, error_type=None,
    ))

    requested: list[list[str]] = []

    async def _fake_nxt_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            status = CaptureStatus.COMPLETE if code == "000001" else CaptureStatus.NOT_APPLICABLE
            on_symbol(code, _empty_bar_frame_for(snap_date),
                      _CoverageEntry(
                          symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue="NXT",
                          session="nxt_aftermarket", scheduled_at=None, status=status, rows=0,
                          first_event_time=None, last_event_time=None, reason="test-nxt-bars",
                          raw_refs=(shared,),
                      ))
        return pd.DataFrame()

    async def _fake_after_ticks(client, session, codes, snap_date, **kwargs):
        requested.append(list(codes))
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _fake_nxt_bars)
    _seed_panel(tmp_path, {"2026-09-04": ["000001", "000002"], "2026-09-03": ["000001", "000002"]})
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _fake_after_ticks)

    archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="aftermarket")

    assert sorted(requested[0]) == ["000001", "000002"]
    assert requested[1] == ["000001"]


def test_phase_completeness_requires_tick_manifests_from_start_date(tmp_path) -> None:
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    for target_date in ("2026-09-29", "2026-09-28"):
        store = _archive_store(tmp_path)
        for session in ("nxt_premarket", "nxt_aftermarket", "krx_aftermarket"):
            _publish_evening_manifest(store, target_date, CaptureDataset.MINUTE_BARS, session, "COMPLETE", "bars")
        _publish_evening_manifest(store, target_date, CaptureDataset.MINUTE_BARS, "regular", "COMPLETE", "bars")
        _publish_evening_manifest(store, target_date, CaptureDataset.TRADE_TICKS, "regular", "COMPLETE", "ticks")

        if target_date == "2026-09-29":
            assert archive_intraday.archive_phase_complete(store, target_date, "aftermarket") is False
        else:
            assert archive_intraday.archive_phase_complete(store, target_date, "aftermarket") is True


def test_run_archive_aftermarket_uncertified_ticks_recorded_not_written(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.daily import archive_intraday
    from src.data import intraday_store
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    staged: list[tuple[str, str]] = []
    original_publish_frame = store.publish_frame

    def _spy_publish_frame(frame, *, context):
        staged.append((context.symbol, context.session))
        return original_publish_frame(frame, context=context)

    monkeypatch.setattr(store, "publish_frame", _spy_publish_frame)
    monkeypatch.setattr(archive_intraday, "CaptureStore", lambda *_a, **_k: store)

    async def _partial_after_ticks(client, session, codes, snap_date, **kwargs):
        session_tag = "krx_aftermarket" if kwargs.get("venue") == "KRX" else "nxt_aftermarket"
        for code in codes:
            kwargs["on_symbol"](code, pd.DataFrame(), CoverageEntry(
                symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue="UNKNOWN", session=session_tag,
                scheduled_at=None, status=CaptureStatus.PARTIAL, rows=0, first_event_time=None,
                last_event_time=None, reason="unrepaired", raw_refs=(),
            ))
        return pd.DataFrame()

    async def _nxt_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        for code in codes:
            frame = _canon_bar_frame(snap_date, code)
            kwargs["on_symbol"](code, frame, CoverageEntry(
                symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue="NXT", session="nxt_aftermarket",
                scheduled_at=None, status=CaptureStatus.COMPLETE, rows=len(frame), first_event_time=None,
                last_event_time=None, reason="test-nxt-bars", raw_refs=(),
            ))
        return pd.DataFrame()

    async def _noop(client, session, codes, snap_date, *args, **kwargs):
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _noop)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _nxt_bars)
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _partial_after_ticks)

    archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="aftermarket")

    assert not intraday_store.tick_partition_path("2026-09-07", "krx_aftermarket").exists()
    assert not intraday_store.tick_partition_path("2026-09-07", "nxt_aftermarket").exists()
    assert staged == []
    tick_manifests = [
        item for item in store.read_manifests("2026-09-07")
        if item.context.dataset is CaptureDataset.TRADE_TICKS and item.context.endpoint == "archive-task"
    ]
    assert len(tick_manifests) == 2
    assert {item.status for item in tick_manifests} == {CaptureStatus.PARTIAL}
    for item in tick_manifests:
        assert {entry.symbol for entry in item.entries} == {"005930"}
        assert all(entry.status == CaptureStatus.PARTIAL for entry in item.entries)


def test_cohort_codes_include_held_symbols(tmp_path) -> None:
    """Held symbols are appended after cohort codes without forcing incompleteness."""
    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-28", ["005930"])
    _publish_cohort(store, "2026-09-23", ["005930"])
    _seed_panel(tmp_path, {"2026-09-25": ["005930"], "2026-09-22": ["005930"]})

    codes, incomplete = archive_intraday._resolve_cohort_codes(
        "2026-09-28", _raw_profile(tmp_path), store,
        previous_trading_day="2026-09-23", held=HeldRoster(("099999",), True, None),
    )
    assert codes == ["005930", "099999"]
    assert incomplete is False


def test_cohort_codes_not_ok_roster_is_incomplete(tmp_path, caplog) -> None:
    """A not-ok roster makes the cohort incomplete with a held_roster_unavailable warning."""
    import logging

    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-28", ["005930"])
    _publish_cohort(store, "2026-09-23", ["005930"])
    _seed_panel(tmp_path, {"2026-09-25": ["005930"], "2026-09-22": ["005930"]})

    with caplog.at_level(logging.WARNING, logger=archive_intraday.logger.name):
        codes, incomplete = archive_intraday._resolve_cohort_codes(
            "2026-09-28", _raw_profile(tmp_path), store,
            previous_trading_day="2026-09-23", held=HeldRoster((), False, "OSError"),
        )
    assert codes == ["005930"]
    assert incomplete is True
    assert any("reason=held_roster_unavailable error=OSError" in rec.message for rec in caplog.records)


def _healthy_empty_ledger(monkeypatch) -> None:
    import pandas as pd

    class _Ledger:
        def load_open_positions(self):
            return pd.DataFrame()

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", lambda *a, **k: _Ledger())


def _complete_run_setup(monkeypatch, tmp_path):
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    entries_map = {
        "005930": (_empty_bar_frame_for("2026-09-07"), _fake_entry("005930", CaptureDataset.MINUTE_BARS, "regular", "COMPLETE")),
    }
    fake_collect, seen = _archive_fakes(entries_map)
    _raw_archive_mocks(monkeypatch, tmp_path, fake_collect)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    return store


def test_archive_run_ledger_failure_publishes_partial_manifests(monkeypatch, tmp_path) -> None:
    """A broken ledger never aborts the run but every task manifest is PARTIAL."""
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureStatus

    store = _complete_run_setup(monkeypatch, tmp_path)

    def _boom_ledger(*a, **k):
        raise RuntimeError("ledger down")

    monkeypatch.setattr("src.execution.paper_broker.PaperLedger", _boom_ledger)
    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="regular")

    assert result == (0, 0, 0)
    task_manifests = [
        item for item in store.read_manifests("2026-09-07") if item.context.endpoint == "archive-task"
    ]
    assert task_manifests
    assert all(item.status == CaptureStatus.PARTIAL for item in task_manifests)
    assert archive_intraday.archive_phase_complete(store, "2026-09-07", "regular") is False


def test_archive_run_healthy_ledger_publishes_complete_manifests(monkeypatch, tmp_path) -> None:
    """A readable ledger keeps the all-COMPLETE run COMPLETE (regression guard)."""
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureStatus

    store = _complete_run_setup(monkeypatch, tmp_path)
    _healthy_empty_ledger(monkeypatch)
    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="regular")

    assert result == (0, 0, 0)
    task_manifests = [
        item for item in store.read_manifests("2026-09-07") if item.context.endpoint == "archive-task"
    ]
    assert task_manifests
    assert all(item.status == CaptureStatus.COMPLETE for item in task_manifests)
    assert archive_intraday.archive_phase_complete(store, "2026-09-07", "regular") is True


def test_archive_run_reads_ledger_once(monkeypatch, tmp_path) -> None:
    """The held roster is resolved exactly once per archive run."""
    from src.daily import archive_intraday

    store = _complete_run_setup(monkeypatch, tmp_path)
    _healthy_empty_ledger(monkeypatch)
    calls: list[int] = []
    real_load = archive_intraday.load_held_roster

    def _counting() -> HeldRoster:
        calls.append(1)
        return real_load()

    monkeypatch.setattr(archive_intraday, "load_held_roster", _counting)
    archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="all")

    assert len(calls) == 1


def test_unresolved_previous_day_with_not_ok_roster_warns_roster(monkeypatch, tmp_path, caplog) -> None:
    """Unresolved previous day plus a not-ok roster: incomplete with the roster warning."""
    import logging

    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-28", ["005930"])
    _seed_panel(tmp_path, {"2026-09-25": ["005930"]})

    with caplog.at_level(logging.WARNING, logger=archive_intraday.logger.name):
        codes, incomplete = archive_intraday._resolve_cohort_codes(
            "2026-09-28", _raw_profile(tmp_path), store,
            previous_trading_day=None, held=HeldRoster((), False, "OSError"),
        )
    assert codes == ["005930"]
    assert incomplete is True
    assert any("reason=held_roster_unavailable error=OSError" in rec.message for rec in caplog.records)


def test_admit_for_write_truth_table() -> None:
    import pandas as pd

    from src.daily.archive_intraday import _admit_for_write
    from src.data.capture_contracts import ArtifactRef, CaptureDataset, CaptureStatus, CoverageEntry

    ref = ArtifactRef(path="a/b.parquet", sha256="abc", bytes=10)

    def _entry(status):
        needs_evidence = status in (CaptureStatus.NO_TRADES, CaptureStatus.NOT_APPLICABLE)
        return CoverageEntry(
            symbol="005930", dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="regular",
            scheduled_at=None, status=status, rows=0, first_event_time=None, last_event_time=None,
            reason="test-evidence" if needs_evidence else "test",
            raw_refs=(ref,) if needs_evidence else (),
        )

    one_row = _canon_bar_frame("2026-09-07", "005930")
    zero_row = _empty_bar_frame_for("2026-09-07")
    assert len(one_row) >= 1
    assert len(zero_row) == 0
    for status in (CaptureStatus.COMPLETE, CaptureStatus.NO_TRADES):
        assert _admit_for_write("bars", "005930", one_row, _entry(status)) is True
        assert _admit_for_write("bars", "005930", zero_row, _entry(status)) is True
        assert _admit_for_write("bars", "005930", None, _entry(status)) is True
    for status in (
        CaptureStatus.UNKNOWN, CaptureStatus.FAILED, CaptureStatus.PARTIAL,
        CaptureStatus.PENDING, CaptureStatus.NOT_APPLICABLE,
    ):
        assert _admit_for_write("bars", "005930", zero_row, _entry(status)) is False
        assert _admit_for_write("bars", "005930", None, _entry(status)) is False
        assert _admit_for_write("bars", "005930", pd.DataFrame(), _entry(status)) is False


def test_admit_for_write_rejects_not_applicable_with_rows() -> None:
    import pytest

    from src.daily.archive_intraday import _admit_for_write
    from src.data.capture_contracts import ArtifactRef, CaptureDataset, CaptureStatus, CoverageEntry

    ref = ArtifactRef(path="a/b.parquet", sha256="abc", bytes=10)
    entry = CoverageEntry(
        symbol="005930", dataset=CaptureDataset.MINUTE_BARS, venue="NXT", session="nxt_aftermarket",
        scheduled_at=None, status=CaptureStatus.NOT_APPLICABLE, rows=0, first_event_time=None,
        last_event_time=None, reason="test-evidence", raw_refs=(ref,),
    )
    frame = _canon_bar_frame("2026-09-07", "005930")
    with pytest.raises(ValueError, match="collector_contract_violation"):
        _admit_for_write("nxt_after", "005930", frame, entry)


def test_resolve_nxt_tick_targets_partitions_cohort() -> None:
    from src.config.market_session import INTRADAY_SESSION_NXT_AFTERMARKET
    from src.daily.archive_intraday import _resolve_nxt_tick_targets
    from src.data.capture_contracts import ArtifactRef, CaptureDataset, CaptureStatus, CoverageEntry

    ref = ArtifactRef(path="a/b.parquet", sha256="abc", bytes=10)

    def _bar(symbol, status, venue, refs=()):
        needs_evidence = status in (CaptureStatus.NO_TRADES, CaptureStatus.NOT_APPLICABLE)
        return CoverageEntry(
            symbol=symbol, dataset=CaptureDataset.MINUTE_BARS, venue=venue, session="nxt_aftermarket",
            scheduled_at=None, status=status, rows=0, first_event_time=None, last_event_time=None,
            reason="test-evidence" if needs_evidence else "test",
            raw_refs=tuple(refs) if refs else ((ref,) if needs_evidence else ()),
        )

    cohort = ["000001", "000002", "000003", "000004", "000005", "000006"]
    bars = [
        _bar("000001", CaptureStatus.COMPLETE, "NXT"),
        _bar("000002", CaptureStatus.NO_TRADES, "NXT"),
        _bar("000003", CaptureStatus.NOT_APPLICABLE, "NXT"),
        _bar("000004", CaptureStatus.UNKNOWN, "NXT"),
        _bar("000005", CaptureStatus.COMPLETE, "KRX"),
        CoverageEntry(
            symbol=None, dataset=CaptureDataset.MINUTE_BARS, venue="NXT", session="nxt_aftermarket",
            scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1, first_event_time=None,
            last_event_time=None, reason="test-none", raw_refs=(),
        ),
        _bar("999999", CaptureStatus.COMPLETE, "NXT"),
    ]
    targets, skipped = _resolve_nxt_tick_targets(cohort, bars)
    assert targets == ["000001", "000002"]
    by_symbol = {entry.symbol: entry for entry in skipped}
    assert set(by_symbol) == {"000003", "000004", "000005", "000006"}
    assert by_symbol["000003"].status == CaptureStatus.NOT_APPLICABLE
    assert by_symbol["000003"].reason == "skipped:nxt_bars_not_applicable"
    assert by_symbol["000004"].status == CaptureStatus.UNKNOWN
    assert by_symbol["000004"].reason == "skipped:nxt_bars_unresolved"
    assert by_symbol["000005"].status == CaptureStatus.UNKNOWN
    assert by_symbol["000005"].reason == "skipped:nxt_bars_unresolved"
    assert by_symbol["000006"].status == CaptureStatus.UNKNOWN
    assert by_symbol["000006"].reason == "skipped:nxt_bars_missing"
    assert by_symbol["000006"].raw_refs == ()
    assert set(targets) | set(by_symbol) == set(cohort)
    assert not (set(targets) & set(by_symbol))
    assert len(targets) + len(skipped) == len(cohort)
    for entry in skipped:
        assert entry.dataset is CaptureDataset.TRADE_TICKS
        assert entry.session == INTRADAY_SESSION_NXT_AFTERMARKET
        assert entry.rows == 0
        assert entry.scheduled_at is None
        assert entry.first_event_time is None
        assert entry.last_event_time is None


def test_resolve_nxt_tick_targets_rejects_duplicate_symbol() -> None:
    import pytest

    from src.daily.archive_intraday import _resolve_nxt_tick_targets
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry

    def _bar(symbol):
        return CoverageEntry(
            symbol=symbol, dataset=CaptureDataset.MINUTE_BARS, venue="NXT", session="nxt_aftermarket",
            scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1, first_event_time=None,
            last_event_time=None, reason="test", raw_refs=(),
        )

    with pytest.raises(ValueError, match="duplicate"):
        _resolve_nxt_tick_targets(["000001"], [_bar("000001"), _bar("000001")])


def test_publish_task_manifest_requires_expected_coverage(tmp_path, caplog) -> None:
    import logging

    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry

    store = _archive_store(tmp_path)
    import datetime as _dt

    from src.data.capture_contracts import SEOUL as _SEOUL

    trading_day = _dt.date(2026, 9, 7)

    def _good(symbol):
        return CoverageEntry(
            symbol=symbol, dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="regular",
            scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1, first_event_time=None,
            last_event_time=None, reason="test", raw_refs=(),
        )

    def _bad(symbol):
        return CoverageEntry(
            symbol=symbol, dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="regular",
            scheduled_at=None, status=CaptureStatus.PARTIAL, rows=0, first_event_time=None,
            last_event_time=None, reason="test", raw_refs=(),
        )

    manifest = archive_intraday._publish_task_manifest(
        store, trading_day=trading_day, run_id="run-a", dataset=CaptureDataset.MINUTE_BARS,
        vendor="kis", session="regular", entries=[_good("005930"), _good("000660")],
        expected_symbols=["005930", "000660"],
    )
    assert manifest.status == CaptureStatus.COMPLETE
    with caplog.at_level(logging.WARNING, logger=archive_intraday.logger.name):
        manifest = archive_intraday._publish_task_manifest(
            store, trading_day=trading_day, run_id="run-b", dataset=CaptureDataset.MINUTE_BARS,
            vendor="kis", session="regular", entries=[_good("005930")],
            expected_symbols=["005930", "000660"],
        )
    assert manifest.status == CaptureStatus.PARTIAL
    assert any("reason=missing_entries" in rec.getMessage() and "n_missing=1" in rec.getMessage() for rec in caplog.records)
    manifest = archive_intraday._publish_task_manifest(
        store, trading_day=trading_day, run_id="run-c", dataset=CaptureDataset.MINUTE_BARS,
        vendor="kis", session="regular", entries=[], expected_symbols=[],
    )
    assert manifest.status == CaptureStatus.COMPLETE
    manifest = archive_intraday._publish_task_manifest(
        store, trading_day=trading_day, run_id="run-d", dataset=CaptureDataset.MINUTE_BARS,
        vendor="kis", session="regular", entries=[_good("005930"), _bad("000660")],
        expected_symbols=["005930", "000660"],
    )
    assert manifest.status == CaptureStatus.PARTIAL
    manifest = archive_intraday._publish_task_manifest(
        store, trading_day=trading_day, run_id="run-e", dataset=CaptureDataset.MINUTE_BARS,
        vendor="kis", session="regular", entries=[_good("005930"), _good("000660")],
        expected_symbols=["005930", "000660"], roster_incomplete=True,
    )
    assert manifest.status == CaptureStatus.PARTIAL


def test_run_archive_extended_bars_non_complete_recorded_not_written(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.config.market_session import (
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
        INTRADAY_SESSION_NXT_PREMARKET,
    )
    from src.daily import archive_intraday
    from src.data import intraday_store
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930", "000660"])
    _publish_cohort(store, "2026-09-04", ["005930"])

    async def _fake_unknown(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _empty_bar_frame_for(snap_date),
                      _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    async def _fake_failed(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _empty_bar_frame_for(snap_date),
                      _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "FAILED"))
        return pd.DataFrame()

    async def _fake_ticks(client, session, codes, snap_date, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, pd.DataFrame(),
                      _fake_entry(code, CaptureDataset.TRADE_TICKS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    async def _noop(client, session, codes, snap_date, *args, **kwargs):
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _noop)
    _seed_panel(tmp_path, {"2026-09-04": ["005930", "000660"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _fake_unknown)
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _fake_failed)
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _fake_unknown)
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _fake_ticks)

    result = archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="aftermarket")

    assert result[1] == 0
    assert not intraday_store.intraday_partition_path(1, "2026-09-07", INTRADAY_SESSION_NXT_AFTERMARKET).exists()
    assert not intraday_store.intraday_partition_path(1, "2026-09-07", INTRADAY_SESSION_NXT_PREMARKET).exists()
    assert not intraday_store.intraday_partition_path(1, "2026-09-07", INTRADAY_SESSION_KRX_AFTERMARKET).exists()
    manifests = store.read_manifests("2026-09-07")
    by_session = {
        item.context.session: item for item in manifests
        if item.context.endpoint == "archive-task" and item.context.dataset is CaptureDataset.MINUTE_BARS
    }
    for session in (INTRADAY_SESSION_NXT_AFTERMARKET, INTRADAY_SESSION_NXT_PREMARKET, INTRADAY_SESSION_KRX_AFTERMARKET):
        assert by_session[session].status == CaptureStatus.PARTIAL
        assert {entry.symbol for entry in by_session[session].entries} == {"005930", "000660"}


def _nxt_tick_test_store(monkeypatch, tmp_path, nxt_status):
    import datetime as _dt

    from src.daily import archive_intraday
    from src.data.capture_contracts import (
        SEOUL as _SEOUL,
        CapturedResponse,
        CaptureContext,
        CaptureDataset,
        CaptureStatus,
        CoverageEntry,
    )

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-30", ["005930", "000660"])
    _publish_cohort(store, "2026-09-29", ["005930", "000660"])
    seed_ctx = CaptureContext(
        trading_date=_dt.date(2026, 9, 30), run_id="seed", dataset=CaptureDataset.SCAN,
        vendor="owner-local", endpoint="seed", symbol=None, venue="NXT",
        session="nxt_aftermarket", capture_reason="seed", cohort_id=None, scheduled_at=None,
    )
    now = _dt.datetime.now(_SEOUL)
    shared = store.append_response(CapturedResponse(
        context=seed_ctx, request_started_at=now, received_at=now, payload={"seed": True},
        status=CaptureStatus.COMPLETE, source_timestamp=None, source_published_at=None,
        page_index=0, attempt_index=0, continuation={}, error_type=None,
    ))

    async def _fake_nxt_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        import pandas as pd

        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _empty_bar_frame_for(snap_date), CoverageEntry(
                symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue="NXT",
                session="nxt_aftermarket", scheduled_at=None, status=nxt_status, rows=0,
                first_event_time=None, last_event_time=None, reason="test-nxt-bars",
                raw_refs=(shared,),
            ))
        return pd.DataFrame()

    async def _fake_complete_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        import pandas as pd

        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame = _canon_bar_frame(snap_date, code)
            on_symbol(code, frame, CoverageEntry(
                symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue="KRX",
                session="regular", scheduled_at=None, status=CaptureStatus.COMPLETE,
                rows=len(frame), first_event_time=None, last_event_time=None,
                reason="test-bars", raw_refs=(),
            ))
        return pd.DataFrame()

    def _complete_bars_factory(session_tag, venue):
        async def _fn(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
            import pandas as pd

            on_symbol = kwargs.get("on_symbol")
            for code in codes:
                frame = _canon_bar_frame(snap_date, code)
                on_symbol(code, frame, CoverageEntry(
                    symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue=venue,
                    session=session_tag, scheduled_at=None, status=CaptureStatus.COMPLETE,
                    rows=len(frame), first_event_time=None, last_event_time=None,
                    reason="test-bars", raw_refs=(),
                ))
            return pd.DataFrame()
        return _fn

    async def _fake_complete_ticks(client, session, codes, snap_date, **kwargs):
        import pandas as pd

        from src.data.intraday_schema import normalize_tick_frame

        venue = kwargs.get("venue", "KRX")
        session_tag = "krx_aftermarket" if venue == "KRX" else "nxt_aftermarket"
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            raw = pd.DataFrame({"time": ["160100"], "close": [71000], "jdiff_vol": [7]})
            frame = normalize_tick_frame(raw, "ls", snap_date, code)
            on_symbol(code, frame, CoverageEntry(
                symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue=venue,
                session=session_tag, scheduled_at=None, status=CaptureStatus.COMPLETE,
                rows=len(frame), first_event_time=None, last_event_time=None,
                reason="test-ticks", raw_refs=(),
            ))
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _fake_complete_bars)
    _seed_panel(tmp_path, {"2026-09-29": ["005930", "000660"], "2026-09-28": ["005930", "000660"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _fake_nxt_bars)
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _complete_bars_factory("nxt_premarket", "NXT"))
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _complete_bars_factory("krx_aftermarket", "KRX"))
    requested: list[list[str]] = []

    async def _tracking_after_ticks(client, session, codes, snap_date, **kwargs):
        requested.append(list(codes))
        return await _fake_complete_ticks(client, session, codes, snap_date, **kwargs)

    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _tracking_after_ticks)
    return store, requested, shared


def test_run_archive_nxt_ticks_partial_when_nxt_bars_unresolved(monkeypatch, tmp_path) -> None:
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    store, requested, _ = _nxt_tick_test_store(monkeypatch, tmp_path, CaptureStatus.UNKNOWN)
    archive_intraday.run_intraday_archive(snapshot_date="2026-09-30", profile=_raw_profile(tmp_path), phase="aftermarket")
    assert requested[-1] == []
    manifests = store.read_manifests("2026-09-30")
    nxt_ticks = next(
        item for item in manifests
        if item.context.endpoint == "archive-task"
        and item.context.dataset is CaptureDataset.TRADE_TICKS
        and item.context.session == "nxt_aftermarket"
    )
    assert nxt_ticks.status == CaptureStatus.PARTIAL
    assert {entry.symbol for entry in nxt_ticks.entries} == {"005930", "000660"}
    assert all(entry.status == CaptureStatus.UNKNOWN for entry in nxt_ticks.entries)
    assert all(entry.reason == "skipped:nxt_bars_unresolved" for entry in nxt_ticks.entries)
    assert archive_intraday.archive_phase_complete(store, "2026-09-30", "aftermarket") is False


def test_run_archive_nxt_ticks_complete_when_no_symbol_nxt_listed(monkeypatch, tmp_path) -> None:
    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    store, requested, shared = _nxt_tick_test_store(monkeypatch, tmp_path, CaptureStatus.NOT_APPLICABLE)
    archive_intraday.run_intraday_archive(snapshot_date="2026-09-30", profile=_raw_profile(tmp_path), phase="aftermarket")
    assert requested[-1] == []
    manifests = store.read_manifests("2026-09-30")
    nxt_ticks = next(
        item for item in manifests
        if item.context.endpoint == "archive-task"
        and item.context.dataset is CaptureDataset.TRADE_TICKS
        and item.context.session == "nxt_aftermarket"
    )
    assert nxt_ticks.status == CaptureStatus.COMPLETE
    assert {entry.symbol for entry in nxt_ticks.entries} == {"005930", "000660"}
    assert all(entry.status == CaptureStatus.NOT_APPLICABLE for entry in nxt_ticks.entries)
    assert all(entry.reason == "skipped:nxt_bars_not_applicable" for entry in nxt_ticks.entries)
    assert all(entry.raw_refs == (shared,) for entry in nxt_ticks.entries)
    assert shared in nxt_ticks.artifacts


@pytest.mark.parametrize(
    "stream",
    ["bars", "ticks", "nxt_after", "nxt_pre", "krx_after", "krx_after_ticks", "nxt_after_ticks"],
)
def test_run_archive_rejects_uncertified_frame_with_rows(monkeypatch, tmp_path, stream) -> None:
    import pandas as pd
    import pytest

    from src.daily import archive_intraday
    from src.data import intraday_store
    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry
    from src.data.intraday_schema import normalize_tick_frame

    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path, raising=False)
    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])

    def _bar_frame(code):
        return _canon_bar_frame("2026-09-07", code)

    def _tick_frame(code):
        raw = pd.DataFrame({"time": ["090300"], "close": [71000], "jdiff_vol": [7]})
        return normalize_tick_frame(raw, "ls", "2026-09-07", code)

    def _good_bar(code, dataset, session, venue="KRX"):
        frame = _bar_frame(code)
        return frame, CoverageEntry(
            symbol=code, dataset=dataset, venue=venue, session=session, scheduled_at=None,
            status=CaptureStatus.COMPLETE, rows=len(frame), first_event_time=None,
            last_event_time=None, reason="test-good", raw_refs=(),
        )

    def _good_tick(code, dataset, session, venue="KRX"):
        frame = _tick_frame(code)
        return frame, CoverageEntry(
            symbol=code, dataset=dataset, venue=venue, session=session, scheduled_at=None,
            status=CaptureStatus.COMPLETE, rows=len(frame), first_event_time=None,
            last_event_time=None, reason="test-good", raw_refs=(),
        )

    def _bad_bar(code, dataset, session):
        frame = _bar_frame(code)
        return frame, CoverageEntry(
            symbol=code, dataset=dataset, venue="UNKNOWN", session=session, scheduled_at=None,
            status=CaptureStatus.PARTIAL, rows=len(frame), first_event_time=None,
            last_event_time=None, reason="test-bad", raw_refs=(),
        )

    def _bad_tick(code, dataset, session):
        frame = _tick_frame(code)
        return frame, CoverageEntry(
            symbol=code, dataset=dataset, venue="UNKNOWN", session=session, scheduled_at=None,
            status=CaptureStatus.PARTIAL, rows=len(frame), first_event_time=None,
            last_event_time=None, reason="test-bad", raw_refs=(),
        )

    target = stream

    async def _fake_regular_bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            if target == "bars":
                frame, entry = _bad_bar(code, CaptureDataset.MINUTE_BARS, "regular")
            else:
                frame, entry = _good_bar(code, CaptureDataset.MINUTE_BARS, "regular")
            on_symbol(code, frame, entry)
        return pd.DataFrame()

    async def _fake_regular_ticks(client, session, codes, snap_date, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            if target == "ticks":
                frame, entry = _bad_tick(code, CaptureDataset.TRADE_TICKS, "regular")
            else:
                frame, entry = _good_tick(code, CaptureDataset.TRADE_TICKS, "regular")
            on_symbol(code, frame, entry)
        return pd.DataFrame()

    def _session_bars_factory(session_tag, key):
        async def _fn(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
            on_symbol = kwargs.get("on_symbol")
            for code in codes:
                if target == key:
                    frame, entry = _bad_bar(code, CaptureDataset.MINUTE_BARS, session_tag)
                else:
                    frame = _bar_frame(code)
                    entry = CoverageEntry(
                        symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue="NXT" if "nxt" in session_tag else "KRX",
                        session=session_tag, scheduled_at=None, status=CaptureStatus.COMPLETE,
                        rows=len(frame), first_event_time=None, last_event_time=None,
                        reason="test-good", raw_refs=(),
                    )
                on_symbol(code, frame, entry)
            return pd.DataFrame()
        return _fn

    async def _fake_after_ticks(client, session, codes, snap_date, **kwargs):
        venue = kwargs.get("venue")
        session_tag = "krx_aftermarket" if venue == "KRX" else "nxt_aftermarket"
        key = "krx_after_ticks" if venue == "KRX" else "nxt_after_ticks"
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            if target == key:
                frame, entry = _bad_tick(code, CaptureDataset.TRADE_TICKS, session_tag)
            else:
                frame, entry = _good_tick(code, CaptureDataset.TRADE_TICKS, session_tag, venue=venue)
            on_symbol(code, frame, entry)
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _fake_regular_bars)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _fake_regular_ticks)
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _session_bars_factory("nxt_aftermarket", "nxt_after"))
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _session_bars_factory("nxt_premarket", "nxt_pre"))
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _session_bars_factory("krx_aftermarket", "krx_after"))
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _fake_after_ticks)

    with pytest.raises(ValueError, match=f"collector_contract_violation stream={target} symbol=005930"):
        archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="all")
    task_manifests = [item for item in store.read_manifests("2026-09-07") if item.context.endpoint == "archive-task"]
    assert task_manifests == []
    normalized = tmp_path / "cap" / "normalized" / "2026-09-07"
    assert not normalized.exists() or list(normalized.rglob("*.parquet")) == []


def test_stream_table_is_well_formed() -> None:
    from src.daily.archive_intraday import _STREAMS
    from src.data.capture_contracts import CaptureDataset

    assert len(_STREAMS) == 7
    keys = [s.key for s in _STREAMS]
    assert len(set(keys)) == 7
    pairs = [(s.dataset, s.session) for s in _STREAMS]
    assert len(set(pairs)) == 7
    by_key = {s.key: s for s in _STREAMS}
    assert by_key["nxt_after_ticks"].codes_from is not None
    assert by_key["nxt_after_ticks"].codes_from.source == "nxt_after"
    order = {s.key: idx for idx, s in enumerate(_STREAMS)}
    for spec in _STREAMS:
        if spec.codes_from is not None:
            assert order[spec.codes_from.source] < order[spec.key]
            assert by_key[spec.codes_from.source].phase == spec.phase
        assert spec.return_slot in (0, 1, 2, None)
    assert by_key["bars"].return_slot == 0
    assert by_key["nxt_after"].return_slot == 1
    assert by_key["nxt_pre"].return_slot == 1
    assert by_key["ticks"].return_slot == 2
    assert by_key["bars"].dataset is CaptureDataset.MINUTE_BARS
    assert by_key["ticks"].dataset is CaptureDataset.TRADE_TICKS


def test_stream_run_ids_are_prefix_free() -> None:
    from src.daily.archive_intraday import _STREAMS, _stream_run_id

    attempt = "abcd1234"
    run_ids = [_stream_run_id("2026-09-07", spec, attempt) for spec in _STREAMS]
    assert len(set(run_ids)) == 7
    for idx, first in enumerate(run_ids):
        for jdx, second in enumerate(run_ids):
            if idx != jdx:
                assert not second.startswith(first)
    slugs = sorted(
        run_id.removeprefix("archive-2026-09-07-").removesuffix(f"-{attempt}") for run_id in run_ids
    )
    assert slugs == sorted([
        "regular-bars",
        "nxt-aftermarket-bars",
        "nxt-premarket-bars",
        "krx-aftermarket-bars",
        "krx-aftermarket-ticks",
        "nxt-aftermarket-ticks",
        "regular-ticks",
    ])


def _spec16_contract_fakes(session_tags):
    import pandas as pd

    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry

    def _bar_entry(code, session_tag, venue="KRX"):
        return CoverageEntry(
            symbol=code, dataset=CaptureDataset.MINUTE_BARS, venue=venue, session=session_tag,
            scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1,
            first_event_time=None, last_event_time=None, reason="test-spec16", raw_refs=(),
        )

    def _tick_entry(code, session_tag, venue="KRX"):
        return CoverageEntry(
            symbol=code, dataset=CaptureDataset.TRADE_TICKS, venue=venue, session=session_tag,
            scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1,
            first_event_time=None, last_event_time=None, reason="test-spec16", raw_refs=(),
        )

    async def _bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            on_symbol(code, _canon_bar_frame(snap_date, code), _bar_entry(code, "regular"))
        return pd.DataFrame()

    def _session_bars_factory(session_tag, venue="KRX"):
        async def _fn(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
            on_symbol = kwargs.get("on_symbol")
            for code in codes:
                on_symbol(code, _canon_bar_frame(snap_date, code), _bar_entry(code, session_tag, venue))
            return pd.DataFrame()

        return _fn

    async def _ticks(client, session, codes, snap_date, **kwargs):
        import pandas as pd

        from src.data.intraday_schema import normalize_tick_frame

        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            raw = pd.DataFrame({"time": ["090300"], "close": [71000], "jdiff_vol": [7]})
            frame = normalize_tick_frame(raw, "ls", snap_date, code)
            on_symbol(code, frame, _tick_entry(code, "regular"))
        return pd.DataFrame()

    async def _after_ticks(client, session, codes, snap_date, **kwargs):
        import pandas as pd

        from src.data.intraday_schema import normalize_tick_frame

        venue = kwargs.get("venue")
        session_tag = session_tags["KRX"] if venue == "KRX" else session_tags["NXT"]
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            raw = pd.DataFrame({"time": ["160100"], "close": [71000], "jdiff_vol": [7]})
            frame = normalize_tick_frame(raw, "ls", snap_date, code)
            on_symbol(code, frame, _tick_entry(code, session_tag, venue=venue))
        return pd.DataFrame()

    return _bars, _session_bars_factory, _ticks, _after_ticks


def test_run_archive_phase_all_collection_order(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    calls: list[str] = []

    async def _rec(name, venue=None):
        calls.append(f"{name}:{venue}" if venue else name)

    async def _bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        await _rec("bars")
        for code in codes:
            kwargs["on_symbol"](code, _empty_bar_frame_for(snap_date),
                                _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    def _session_bars_factory(name):
        async def _fn(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
            await _rec(name)
            for code in codes:
                kwargs["on_symbol"](code, _empty_bar_frame_for(snap_date),
                                    _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
            return pd.DataFrame()

        return _fn

    async def _after_ticks(client, session, codes, snap_date, **kwargs):
        await _rec("after_ticks", kwargs.get("venue"))
        for code in codes:
            kwargs["on_symbol"](code, pd.DataFrame(),
                                _fake_entry(code, CaptureDataset.TRADE_TICKS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    async def _ticks(client, session, codes, snap_date, **kwargs):
        await _rec("ticks")
        for code in codes:
            kwargs["on_symbol"](code, pd.DataFrame(),
                                _fake_entry(code, CaptureDataset.TRADE_TICKS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _bars)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _session_bars_factory("nxt_after"))
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _session_bars_factory("nxt_pre"))
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _session_bars_factory("krx_after"))
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _after_ticks)
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _ticks)

    archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="all")

    assert calls == ["bars", "nxt_after", "nxt_pre", "krx_after", "after_ticks:KRX", "after_ticks:NXT", "ticks"]


def test_run_archive_publishes_one_manifest_per_stream_in_table_order(monkeypatch, tmp_path) -> None:
    from src.config.market_session import (
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
    )
    from src.daily import archive_intraday

    store_probe: dict = {}
    real_cls = archive_intraday.CaptureStore
    spy: list = []

    def _factory(root):
        inst = real_cls(root)
        store_probe["inst"] = inst
        orig = inst.publish_manifest

        def _wrapped(manifest):
            if manifest.context.endpoint == "archive-task":
                spy.append(manifest)
            return orig(manifest)

        inst.publish_manifest = _wrapped  # type: ignore[method-assign]
        return inst

    monkeypatch.setattr(archive_intraday, "CaptureStore", _factory)
    tmp_store = _archive_store(tmp_path)
    _publish_cohort(tmp_store, "2026-09-07", ["005930"])
    _publish_cohort(tmp_store, "2026-09-04", ["005930"])
    tags = {"KRX": INTRADAY_SESSION_KRX_AFTERMARKET, "NXT": INTRADAY_SESSION_NXT_AFTERMARKET}
    _bars, _sess_factory, _ticks, _after = _spec16_contract_fakes(tags)
    from src.config.market_session import (
        INTRADAY_SESSION_NXT_PREMARKET,
        INTRADAY_SESSION_REGULAR,
    )

    _raw_archive_mocks(monkeypatch, tmp_path, _bars)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(
        archive_intraday, "collect_nxt_aftermarket_bars",
        _sess_factory(INTRADAY_SESSION_NXT_AFTERMARKET, "NXT"),
    )
    monkeypatch.setattr(
        archive_intraday, "collect_nxt_premarket_bars",
        _sess_factory(INTRADAY_SESSION_NXT_PREMARKET, "NXT"),
    )
    monkeypatch.setattr(
        archive_intraday, "collect_krx_aftermarket_bars",
        _sess_factory(INTRADAY_SESSION_KRX_AFTERMARKET, "KRX"),
    )
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _ticks)
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _after)

    archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="all")

    assert len(spy) == 7
    expected_pairs = [(s.dataset, s.session) for s in archive_intraday._STREAMS]
    assert [(m.context.dataset, m.context.session) for m in spy] == expected_pairs
    expected_vendors = [s.manifest_vendor for s in archive_intraday._STREAMS]
    assert [m.context.vendor for m in spy] == expected_vendors
    assert all(m.context.vendor == "owner-local" for m in spy)
    slug_by_pair = {
        (s.dataset, s.session): f"{str(s.session).replace('_', '-')}-{'bars' if s.dataset.name == 'MINUTE_BARS' else 'ticks'}"
        for s in archive_intraday._STREAMS
    }
    for manifest in spy:
        slug = slug_by_pair[(manifest.context.dataset, manifest.context.session)]
        assert manifest.context.run_id.startswith(f"archive-2026-09-07-{slug}-")
    assert INTRADAY_SESSION_REGULAR is not None


def test_run_archive_stream_adapters_pass_historical_arguments(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    seen: dict = {}

    def _recorder(name):
        async def _fn(*args, **kwargs):
            seen[name] = (tuple(args), dict(kwargs))
            on_symbol = kwargs.get("on_symbol")
            codes = args[2] if len(args) > 2 else []
            snap_date = args[3] if len(args) > 3 else "2026-09-07"
            for code in codes:
                on_symbol(code, _empty_bar_frame_for(snap_date),
                          _fake_entry(code, CaptureDataset.MINUTE_BARS, "regular", "UNKNOWN"))
            return pd.DataFrame()

        return _fn

    async def _tick_recorder(*args, **kwargs):
        seen["ticks"] = (tuple(args), dict(kwargs))
        on_symbol = kwargs.get("on_symbol")
        codes = args[2] if len(args) > 2 else []
        snap_date = args[3] if len(args) > 3 else "2026-09-07"
        for code in codes:
            on_symbol(code, pd.DataFrame(),
                      _fake_entry(code, CaptureDataset.TRADE_TICKS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    async def _after_recorder(*args, **kwargs):
        venue = kwargs.get("venue")
        seen[f"after_{venue}"] = (tuple(args), dict(kwargs))
        on_symbol = kwargs.get("on_symbol")
        codes = args[2] if len(args) > 2 else []
        snap_date = args[3] if len(args) > 3 else "2026-09-07"
        for code in codes:
            on_symbol(code, pd.DataFrame(),
                      _fake_entry(code, CaptureDataset.TRADE_TICKS, "regular", "UNKNOWN"))
        return pd.DataFrame()

    _raw_archive_mocks(monkeypatch, tmp_path, _recorder("bars"))
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(archive_intraday, "collect_nxt_aftermarket_bars", _recorder("nxt_after"))
    monkeypatch.setattr(archive_intraday, "collect_nxt_premarket_bars", _recorder("nxt_pre"))
    monkeypatch.setattr(archive_intraday, "collect_krx_aftermarket_bars", _recorder("krx_after"))
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _tick_recorder)
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _after_recorder)

    archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="all")

    bars_args, bars_kw = seen["bars"]
    assert len(bars_args) == 5
    assert set(bars_kw) == {"ls_client", "profile", "capture_store", "run_id", "on_symbol"}
    for name in ("nxt_after", "nxt_pre"):
        pos, kw = seen[name]
        assert len(pos) == 5
        assert set(kw) == {"kiwoom_client", "profile", "capture_store", "run_id", "on_symbol"}
    pos, kw = seen["krx_after"]
    assert len(pos) == 5
    assert set(kw) == {"profile", "capture_store", "run_id", "on_symbol"}
    pos, kw = seen["ticks"]
    assert len(pos) == 4
    assert set(kw) == {"ls_client", "kiwoom_client", "profile", "capture_store", "run_id", "on_symbol"}
    for venue in ("KRX", "NXT"):
        pos, kw = seen[f"after_{venue}"]
        assert len(pos) == 4
        assert set(kw) == {"venue", "profile", "capture_store", "run_id", "on_symbol"}
        assert kw["venue"] == venue


def test_run_archive_routes_rows_to_dataset_writer(monkeypatch, tmp_path) -> None:
    from src.config.market_session import (
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
        INTRADAY_SESSION_NXT_PREMARKET,
        INTRADAY_SESSION_REGULAR,
    )
    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930", "000660"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    tags = {"KRX": INTRADAY_SESSION_KRX_AFTERMARKET, "NXT": INTRADAY_SESSION_NXT_AFTERMARKET}
    _bars, _sess_factory, _ticks, _after = _spec16_contract_fakes(tags)
    _raw_archive_mocks(monkeypatch, tmp_path, _bars)
    _seed_panel(tmp_path, {"2026-09-04": ["005930", "000660"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(
        archive_intraday, "collect_nxt_aftermarket_bars",
        _sess_factory(INTRADAY_SESSION_NXT_AFTERMARKET, "NXT"),
    )
    monkeypatch.setattr(
        archive_intraday, "collect_nxt_premarket_bars",
        _sess_factory(INTRADAY_SESSION_NXT_PREMARKET, "NXT"),
    )
    monkeypatch.setattr(
        archive_intraday, "collect_krx_aftermarket_bars",
        _sess_factory(INTRADAY_SESSION_KRX_AFTERMARKET, "KRX"),
    )
    monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _ticks)
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _after)

    bar_writes: list = []
    tick_writes: list = []

    def _spy_bar(df, interval, snap_date, session, *, coverage=None, batch_rows=None, **kwargs):
        bar_writes.append({"interval": interval, "session": session, "codes": sorted(coverage or {}), "rows": len(df)})
        return len(df)

    def _spy_tick(df, snap_date, session, *, coverage=None, batch_rows=None, **kwargs):
        tick_writes.append({"session": session, "codes": sorted(coverage or {}), "rows": len(df)})
        return len(df)

    monkeypatch.setattr(archive_intraday, "write_intraday_partition", _spy_bar)
    monkeypatch.setattr(archive_intraday, "write_tick_partition", _spy_tick)

    archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="all")

    assert {w["session"] for w in bar_writes} == {
        INTRADAY_SESSION_REGULAR,
        INTRADAY_SESSION_NXT_AFTERMARKET,
        INTRADAY_SESSION_NXT_PREMARKET,
        INTRADAY_SESSION_KRX_AFTERMARKET,
    }
    assert {w["session"] for w in tick_writes} == {
        INTRADAY_SESSION_REGULAR,
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
    }
    assert {w["interval"] for w in bar_writes} == {1}
    for write in [*bar_writes, *tick_writes]:
        assert write["rows"] >= 1
        assert write["codes"] != []


def _spec16_counting_fakes(counts):
    import pandas as pd

    from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry

    def _frame(n, snap_date, code, tick=False):
        if tick:
            from src.data.intraday_schema import normalize_tick_frame

            base = normalize_tick_frame(
                pd.DataFrame({"time": ["090300"], "close": [71000], "jdiff_vol": [7]}),
                "ls", snap_date, code,
            )
        else:
            base = _canon_bar_frame(snap_date, code)
        return pd.concat([base] * n, ignore_index=True) if n > 1 else base

    def _entry(code, dataset, session, venue, n):
        return CoverageEntry(
            symbol=code, dataset=dataset, venue=venue, session=session, scheduled_at=None,
            status=CaptureStatus.COMPLETE, rows=n, first_event_time=None,
            last_event_time=None, reason="test-count", raw_refs=(),
        )

    async def _bars(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame = _frame(counts["bars"], snap_date, code)
            on_symbol(code, frame, _entry(code, CaptureDataset.MINUTE_BARS, "regular", "KRX", len(frame)))
        return pd.DataFrame()

    def _sess_factory(session_tag, key, venue="KRX"):
        async def _fn(client, session, codes, snap_date, bar_interval_minutes=1, **kwargs):
            on_symbol = kwargs.get("on_symbol")
            for code in codes:
                frame = _frame(counts[key], snap_date, code)
                on_symbol(code, frame, _entry(code, CaptureDataset.MINUTE_BARS, session_tag, venue, len(frame)))
            return pd.DataFrame()

        return _fn

    async def _ticks(client, session, codes, snap_date, **kwargs):
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame = _frame(counts["ticks"], snap_date, code, tick=True)
            on_symbol(code, frame, _entry(code, CaptureDataset.TRADE_TICKS, "regular", "KRX", len(frame)))
        return pd.DataFrame()

    async def _after(client, session, codes, snap_date, **kwargs):
        from src.config.market_session import (
            INTRADAY_SESSION_KRX_AFTERMARKET,
            INTRADAY_SESSION_NXT_AFTERMARKET,
        )

        venue = kwargs.get("venue")
        tag = INTRADAY_SESSION_KRX_AFTERMARKET if venue == "KRX" else INTRADAY_SESSION_NXT_AFTERMARKET
        key = "krx_after_ticks" if venue == "KRX" else "nxt_after_ticks"
        on_symbol = kwargs.get("on_symbol")
        for code in codes:
            frame = _frame(counts[key], snap_date, code, tick=True)
            on_symbol(code, frame, _entry(code, CaptureDataset.TRADE_TICKS, tag, venue, len(frame)))
        return pd.DataFrame()

    return _bars, _sess_factory, _ticks, _after


def test_run_archive_return_tuple_counts_reported_streams_only(monkeypatch, tmp_path) -> None:
    from src.config.market_session import (
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
        INTRADAY_SESSION_NXT_PREMARKET,
    )
    from src.daily import archive_intraday

    counts = {
        "bars": 1, "nxt_after": 2, "nxt_pre": 3, "krx_after": 4,
        "krx_after_ticks": 5, "nxt_after_ticks": 6, "ticks": 7,
    }

    def _run_once(phase):
        fresh = tmp_path / f"cap-{phase}"
        fresh.mkdir(exist_ok=True)
        from src.config.collection import CollectionSettings

        profile = CollectionSettings(COLLECTION_ROOT=fresh)
        store = _archive_store.__wrapped__ if hasattr(_archive_store, "__wrapped__") else None
        from src.data.capture_store import CaptureStore

        real_store = CaptureStore(fresh)
        _publish_cohort(real_store, "2026-09-07", ["005930"])
        _publish_cohort(real_store, "2026-09-04", ["005930"])
        _bars, _sess_factory, _ticks, _after = _spec16_counting_fakes(counts)
        _raw_archive_mocks(monkeypatch, tmp_path, _bars)
        _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
        monkeypatch.setattr(
            archive_intraday, "collect_nxt_aftermarket_bars",
            _sess_factory(INTRADAY_SESSION_NXT_AFTERMARKET, "nxt_after", "NXT"),
        )
        monkeypatch.setattr(
            archive_intraday, "collect_nxt_premarket_bars",
            _sess_factory(INTRADAY_SESSION_NXT_PREMARKET, "nxt_pre", "NXT"),
        )
        monkeypatch.setattr(
            archive_intraday, "collect_krx_aftermarket_bars",
            _sess_factory(INTRADAY_SESSION_KRX_AFTERMARKET, "krx_after", "KRX"),
        )
        monkeypatch.setattr(archive_intraday, "collect_intraday_trade_ticks", _ticks)
        monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _after)
        monkeypatch.setattr(archive_intraday.settings, "HISTORY_DIR", tmp_path, raising=False)
        monkeypatch.setattr(archive_intraday.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "price_history.parquet", raising=False)
        return archive_intraday.run_intraday_archive(snapshot_date="2026-09-07", profile=profile, phase=phase)

    assert _run_once("all") == (1, 5, 7)
    assert _run_once("regular") == (1, 0, 7)
    assert _run_once("aftermarket") == (0, 5, 0)


def test_run_archive_logs_one_line_per_executed_stream(monkeypatch, tmp_path, caplog) -> None:
    import logging

    from src.config.market_session import (
        INTRADAY_SESSION_KRX_AFTERMARKET,
        INTRADAY_SESSION_NXT_AFTERMARKET,
        INTRADAY_SESSION_NXT_PREMARKET,
    )
    from src.daily import archive_intraday

    store = _archive_store(tmp_path)
    _publish_cohort(store, "2026-09-07", ["005930"])
    _publish_cohort(store, "2026-09-04", ["005930"])
    tags = {"KRX": INTRADAY_SESSION_KRX_AFTERMARKET, "NXT": INTRADAY_SESSION_NXT_AFTERMARKET}
    _bars, _sess_factory, _ticks, _after = _spec16_contract_fakes(tags)
    _raw_archive_mocks(monkeypatch, tmp_path, _bars)
    _seed_panel(tmp_path, {"2026-09-04": ["005930"], "2026-09-03": ["005930"]})
    monkeypatch.setattr(
        archive_intraday, "collect_nxt_aftermarket_bars",
        _sess_factory(INTRADAY_SESSION_NXT_AFTERMARKET, "NXT"),
    )
    monkeypatch.setattr(
        archive_intraday, "collect_nxt_premarket_bars",
        _sess_factory(INTRADAY_SESSION_NXT_PREMARKET, "NXT"),
    )
    monkeypatch.setattr(
        archive_intraday, "collect_krx_aftermarket_bars",
        _sess_factory(INTRADAY_SESSION_KRX_AFTERMARKET, "KRX"),
    )
    monkeypatch.setattr(archive_intraday, "collect_aftermarket_trade_ticks", _after)

    with caplog.at_level(logging.INFO, logger=archive_intraday.logger.name):
        archive_intraday.run_intraday_archive(
            snapshot_date="2026-09-07", profile=_raw_profile(tmp_path), phase="aftermarket"
        )

    stream_records = [rec for rec in caplog.records if "stage=intraday_archive_stream" in rec.getMessage()]
    assert len(stream_records) == 5
    keys = {rec.getMessage().split("stream=")[1].split()[0] for rec in stream_records}
    assert keys == {"nxt_after", "nxt_pre", "krx_after", "krx_after_ticks", "nxt_after_ticks"}
    for rec in stream_records:
        message = rec.getMessage()
        for field in ("status=", "targets=", "entries=", "rows="):
            assert field in message
    assert not any("stage=krx_aftermarket" in rec.getMessage() for rec in caplog.records)
    assert not any("stage=aftermarket_ticks" in rec.getMessage() for rec in caplog.records)


def test_archive_phase_complete_matches_stream_table(tmp_path) -> None:
    import datetime as _dt

    from src.daily import archive_intraday
    from src.data.capture_contracts import CaptureDataset

    for target_date in ("2026-09-10", "2026-09-20", "2026-10-01"):
        for phase in ("regular", "aftermarket", "all"):
            phases = {"regular", "aftermarket"} if phase == "all" else {phase}
            required = [
                (s.dataset, s.session)
                for s in archive_intraday._STREAMS
                if s.phase in phases
                and (s.required_from is None or str(target_date) >= s.required_from)
            ]
            assert required != []
            store = _archive_store(tmp_path / f"cap-{target_date}-{phase}")
            for idx, (dataset, session) in enumerate(required):
                _publish_evening_manifest(store, target_date, dataset, session, "COMPLETE", f"s{idx}")
            assert archive_intraday.archive_phase_complete(store, target_date, phase) is True
            for drop_idx in range(len(required)):
                pruned = _archive_store(tmp_path / f"cap-{target_date}-{phase}-drop{drop_idx}-{_dt.datetime.now().microsecond}")
                for idx, (dataset, session) in enumerate(required):
                    if idx == drop_idx:
                        continue
                    _publish_evening_manifest(pruned, target_date, dataset, session, "COMPLETE", f"s{idx}")
                assert archive_intraday.archive_phase_complete(pruned, target_date, phase) is False

