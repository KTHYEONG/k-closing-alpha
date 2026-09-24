"""altdata_capture CLI entrypoint invariant guards."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _profile(tmp_path: Path, **overrides: Any):
    from src.config.collection import CollectionSettings

    base: dict[str, Any] = {
        "COLLECTION_ROOT": tmp_path / "capture",
        "COLLECTION_RAW_ENABLED": True,
        "COLLECTION_ALTDATA_ENABLED": True,
    }
    base.update(overrides)
    return CollectionSettings(**base)


def test_main_returns_zero_when_disabled_not_a_failure(tmp_path, monkeypatch, caplog) -> None:
    """실측 회귀: disabled 스킵 경로가 exit code 2(실패)를 반환해 systemd가 이를 진짜
    실패로 취급, OnFailure 얼러트 이메일을 발송하던 버그(2026-09-18 kca-altdata-capture
    실측). 같은 패턴의 auction_capture.main()은 이미 성공(0/None)으로 처리한다."""
    import logging

    from src.daily import altdata_capture

    monkeypatch.setattr(altdata_capture, "CollectionSettings", lambda: _profile(tmp_path, COLLECTION_ALTDATA_ENABLED=False))
    with caplog.at_level(logging.INFO, logger=altdata_capture.logger.name):
        rc = altdata_capture.main([])

    assert rc == 0
    assert any("SKIP" in r.message for r in caplog.records)


def test_main_returns_zero_when_raw_capture_disabled(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from src.daily import altdata_capture

    monkeypatch.setattr(
        altdata_capture, "CollectionSettings",
        lambda: _profile(tmp_path, COLLECTION_RAW_ENABLED=False, COLLECTION_ALTDATA_ENABLED=False),
    )
    with caplog.at_level(logging.INFO, logger=altdata_capture.logger.name):
        rc = altdata_capture.main([])

    assert rc == 0
    assert any("SKIP" in r.message for r in caplog.records)


def test_main_rejects_invalid_date(tmp_path, monkeypatch) -> None:
    import pytest

    from src.daily import altdata_capture

    monkeypatch.setattr(altdata_capture, "CollectionSettings", lambda: _profile(tmp_path))
    with pytest.raises(ValueError, match="Invalid date"):
        altdata_capture.main(["--date", "not-a-date"])


def test_main_runs_configured_capture_and_reports_manifest_status(tmp_path, monkeypatch) -> None:
    from src.daily import altdata_capture
    from src.data.capture_contracts import CaptureStatus

    class _Manifest:
        status = CaptureStatus.COMPLETE

    captured: dict[str, Any] = {}

    def _fake_run(trading_day, *, profile, store, cfg):
        captured["trading_day"] = trading_day
        return _Manifest()

    monkeypatch.setattr(altdata_capture, "CollectionSettings", lambda: _profile(tmp_path))
    monkeypatch.setattr(altdata_capture, "run_altdata_capture", _fake_run)

    rc = altdata_capture.main(["--date", "2026-09-18"], trading_day_fn=lambda _d: True)

    assert rc == 0
    assert captured["trading_day"].isoformat() == "2026-09-18"


def test_krx_listed_universe_returns_none_on_holiday(tmp_path, monkeypatch) -> None:
    """비거래일(양쪽 시장 0행)에는 유니버스를 조작하지 않고 None을 반환한다."""
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture

    monkeypatch.setattr(altdata_capture, "fetch_krx_daily", lambda window_end, cfg: pd.DataFrame(columns=["symbol"]))
    cfg = AltDataFetchConfig(start=pd.Timestamp("2026-09-17"), end=pd.Timestamp("2026-09-18"), out_dir=tmp_path)

    universe = altdata_capture._krx_listed_universe(pd.Timestamp("2026-09-18"), cfg)

    assert universe is None


def test_krx_listed_universe_returns_deduped_frozenset(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture

    monkeypatch.setattr(
        altdata_capture,
        "fetch_krx_daily",
        lambda window_end, cfg: pd.DataFrame({"symbol": ["005930", "000660", "005930", " "]}),
    )
    cfg = AltDataFetchConfig(start=pd.Timestamp("2026-09-17"), end=pd.Timestamp("2026-09-18"), out_dir=tmp_path)

    universe = altdata_capture._krx_listed_universe(pd.Timestamp("2026-09-18"), cfg)

    assert universe == frozenset({"005930", "000660"})


def test_krx_listed_universe_drops_non_6digit_codes(tmp_path, monkeypatch) -> None:
    """실측 회귀: KRX 일별 목록에 ETN(예: '0013V0')이 섞여 있어 그대로 넘기면
    AltDataFetchConfig가 '6-digit string이어야 한다'며 ValueError를 던졌다
    (2026-09-18 실측). 일반주식 6자리 숫자 코드만 유니버스로 채택한다."""
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture

    monkeypatch.setattr(
        altdata_capture,
        "fetch_krx_daily",
        lambda window_end, cfg: pd.DataFrame({"symbol": ["005930", "0013V0", "A05930", "12345"]}),
    )
    cfg = AltDataFetchConfig(start=pd.Timestamp("2026-09-17"), end=pd.Timestamp("2026-09-18"), out_dir=tmp_path)

    universe = altdata_capture._krx_listed_universe(pd.Timestamp("2026-09-18"), cfg)

    assert universe == frozenset({"005930"})


def test_krx_listed_universe_walks_back_past_publication_lag(tmp_path, monkeypatch) -> None:
    """실측 회귀: KRX 일별 API가 당일 데이터를 아직 발행하지 않아(2026-09-18 저녁
    시각에도 0행) window_end 그대로 조회하면 항상 빈 유니버스가 됐다. 발행된
    가장 최근 날짜까지 거슬러 올라가 조회한다."""
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture

    seen_dates: list[pd.Timestamp] = []

    def _fake_fetch(window_end, cfg):
        seen_dates.append(window_end)
        if window_end == pd.Timestamp("2026-09-18"):
            return pd.DataFrame(columns=["symbol"])
        return pd.DataFrame({"symbol": ["005930"]})

    monkeypatch.setattr(altdata_capture, "fetch_krx_daily", _fake_fetch)
    cfg = AltDataFetchConfig(start=pd.Timestamp("2026-09-17"), end=pd.Timestamp("2026-09-18"), out_dir=tmp_path)

    universe = altdata_capture._krx_listed_universe(pd.Timestamp("2026-09-18"), cfg)

    assert universe == frozenset({"005930"})
    assert seen_dates == [pd.Timestamp("2026-09-18"), pd.Timestamp("2026-09-17")]


def test_krx_listed_universe_gives_up_after_lookback_exhausted(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture

    calls: list[pd.Timestamp] = []

    def _always_empty(window_end, cfg):
        calls.append(window_end)
        return pd.DataFrame(columns=["symbol"])

    monkeypatch.setattr(altdata_capture, "fetch_krx_daily", _always_empty)
    cfg = AltDataFetchConfig(start=pd.Timestamp("2026-09-17"), end=pd.Timestamp("2026-09-18"), out_dir=tmp_path)

    universe = altdata_capture._krx_listed_universe(pd.Timestamp("2026-09-18"), cfg)

    assert universe is None
    assert len(calls) == altdata_capture._UNIVERSE_LOOKBACK_DAYS


def test_run_altdata_capture_fills_declared_universe_from_krx_when_absent(tmp_path, monkeypatch) -> None:
    """실측 회귀: universe_symbols 미설정 시 shorting/credit_balance/program_trade_daily
    3개 종목별 패널이 늘 스킵돼 manifest가 영구히 COMPLETE 못 되던 문제(2026-09-19 실측)."""
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture
    from src.data.capture_contracts import CaptureStatus

    captured: dict[str, Any] = {}

    def _fake_backfill(cfg, *, capture_store, run_id, reobserve):
        captured["cfg"] = cfg
        captured["run_id"] = run_id
        return {"capture": "ok"}

    class _Context:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

    class _Manifest:
        def __init__(self, run_id: str) -> None:
            self.status = CaptureStatus.COMPLETE
            self.context = _Context(run_id)

    class _Store:
        def read_manifests(self, date_str: str) -> list[Any]:
            return [_Manifest(captured["run_id"])]

    monkeypatch.setattr(altdata_capture, "run_altdata_backfill", _fake_backfill)
    monkeypatch.setattr(
        altdata_capture,
        "fetch_krx_daily",
        lambda window_end, cfg: pd.DataFrame({"symbol": ["005930", "000660"]}),
    )

    profile = _profile(tmp_path, COLLECTION_ALTDATA_LOOKBACK_DAYS=2)
    window_start, window_end = altdata_capture._rolling_bounds(pd.Timestamp("2026-09-18").date(), 2)
    cfg = AltDataFetchConfig(start=window_start, end=window_end, out_dir=tmp_path, krx_api_key="k")

    manifest = altdata_capture.run_altdata_capture(pd.Timestamp("2026-09-18").date(), profile=profile, store=_Store(), cfg=cfg)

    assert manifest.status == CaptureStatus.COMPLETE
    assert captured["cfg"].universe_symbols == frozenset({"005930", "000660"})


def test_run_altdata_capture_preserves_explicit_universe(tmp_path, monkeypatch) -> None:
    """이미 명시적으로 설정된 universe_symbols는 KRX 재조회로 덮어쓰지 않는다."""
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture
    from src.data.capture_contracts import CaptureStatus

    captured: dict[str, Any] = {}

    def _fake_backfill(cfg, *, capture_store, run_id, reobserve):
        captured["cfg"] = cfg
        captured["run_id"] = run_id
        return {"capture": "ok"}

    def _boom(window_end, cfg):
        raise AssertionError("fetch_krx_daily must not be called when universe already declared")

    class _Context:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

    class _Manifest:
        def __init__(self, run_id: str) -> None:
            self.status = CaptureStatus.COMPLETE
            self.context = _Context(run_id)

    class _Store:
        def read_manifests(self, date_str: str) -> list[Any]:
            return [_Manifest(captured["run_id"])]

    monkeypatch.setattr(altdata_capture, "run_altdata_backfill", _fake_backfill)
    monkeypatch.setattr(altdata_capture, "fetch_krx_daily", _boom)

    profile = _profile(tmp_path, COLLECTION_ALTDATA_LOOKBACK_DAYS=2)
    window_start, window_end = altdata_capture._rolling_bounds(pd.Timestamp("2026-09-18").date(), 2)
    declared = frozenset({"999999"})
    cfg = AltDataFetchConfig(start=window_start, end=window_end, out_dir=tmp_path, krx_api_key="k", universe_symbols=declared)

    altdata_capture.run_altdata_capture(pd.Timestamp("2026-09-18").date(), profile=profile, store=_Store(), cfg=cfg)

    assert captured["cfg"].universe_symbols == declared


def _capture_harness(monkeypatch, captured):
    from src.daily import altdata_capture
    from src.data.capture_contracts import CaptureStatus

    def _fake_backfill(cfg, *, capture_store, run_id, reobserve):
        captured["cfg"] = cfg
        captured["run_id"] = run_id
        return {"capture": "ok"}

    class _Context:
        def __init__(self, run_id: str) -> None:
            self.run_id = run_id

    class _Manifest:
        def __init__(self, run_id: str) -> None:
            self.status = CaptureStatus.COMPLETE
            self.context = _Context(run_id)

    class _Store:
        def read_manifests(self, date_str: str) -> list[Any]:
            return [_Manifest(captured["run_id"])]

    monkeypatch.setattr(altdata_capture, "run_altdata_backfill", _fake_backfill)
    return _Store()


def test_run_altdata_capture_keeps_single_key_without_extra_slots(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture

    captured: dict[str, Any] = {}
    store = _capture_harness(monkeypatch, captured)
    profile = _profile(tmp_path, COLLECTION_ALTDATA_LOOKBACK_DAYS=2)
    assert profile.COLLECTION_ALTDATA_EXTRA_SLOTS == ()
    window_start, window_end = altdata_capture._rolling_bounds(pd.Timestamp("2026-09-18").date(), 2)
    cfg = AltDataFetchConfig(
        start=window_start, end=window_end, out_dir=tmp_path, krx_api_key="k",
        universe_symbols=frozenset({"005930"}),
    )
    altdata_capture.run_altdata_capture(pd.Timestamp("2026-09-18").date(), profile=profile, store=store, cfg=cfg)
    assert captured["cfg"].extra_client_kwargs == ()


def test_run_altdata_capture_injects_declared_slot_credentials(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture

    captured: dict[str, Any] = {}
    store = _capture_harness(monkeypatch, captured)
    monkeypatch.setenv("KIS_DATA_SLOTS", "2,3")
    monkeypatch.setenv("KIS_DATA_2_APP_KEY", "key2")
    monkeypatch.setenv("KIS_DATA_2_APP_SECRET", "sec2")
    monkeypatch.setenv("KIS_DATA_2_HTS_ID", "hts2")
    monkeypatch.setenv("KIS_DATA_3_APP_KEY", "key3")
    monkeypatch.setenv("KIS_DATA_3_APP_SECRET", "sec3")
    monkeypatch.setenv("KIS_DATA_3_HTS_ID", "hts3")
    monkeypatch.delenv("KIS_DECISION_SHARD_SLOTS", raising=False)
    monkeypatch.delenv("KIS_TRADE_APP_KEY", raising=False)
    monkeypatch.delenv("KIS_APP_KEY", raising=False)
    profile = _profile(tmp_path, COLLECTION_ALTDATA_LOOKBACK_DAYS=2, COLLECTION_ALTDATA_EXTRA_SLOTS=("2", "3"))
    window_start, window_end = altdata_capture._rolling_bounds(pd.Timestamp("2026-09-18").date(), 2)
    cfg = AltDataFetchConfig(
        start=window_start, end=window_end, out_dir=tmp_path, krx_api_key="k",
        universe_symbols=frozenset({"005930"}),
    )
    altdata_capture.run_altdata_capture(pd.Timestamp("2026-09-18").date(), profile=profile, store=store, cfg=cfg)
    assert captured["cfg"].extra_client_kwargs == (("key2", "sec2", "hts2"), ("key3", "sec3", "hts3"))


def test_run_altdata_capture_preserves_prefilled_extra_keys(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture

    captured: dict[str, Any] = {}
    store = _capture_harness(monkeypatch, captured)

    def _boom(env, *, slots):
        raise AssertionError("resolve must not be called when extra keys already set")

    monkeypatch.setattr(altdata_capture, "resolve_research_credentials", _boom)
    profile = _profile(tmp_path, COLLECTION_ALTDATA_LOOKBACK_DAYS=2, COLLECTION_ALTDATA_EXTRA_SLOTS=("2",))
    window_start, window_end = altdata_capture._rolling_bounds(pd.Timestamp("2026-09-18").date(), 2)
    prefilled = (("pre", "sec", "hts"),)
    cfg = AltDataFetchConfig(
        start=window_start, end=window_end, out_dir=tmp_path, krx_api_key="k",
        universe_symbols=frozenset({"005930"}), extra_client_kwargs=prefilled,
    )
    altdata_capture.run_altdata_capture(pd.Timestamp("2026-09-18").date(), profile=profile, store=store, cfg=cfg)
    assert captured["cfg"].extra_client_kwargs == prefilled


def test_run_altdata_capture_propagates_shard_overlap_error(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import pytest

    from src.backfill.altdata.config import AltDataFetchConfig
    from src.daily import altdata_capture

    captured: dict[str, Any] = {}
    store = _capture_harness(monkeypatch, captured)
    monkeypatch.setenv("KIS_DATA_SLOTS", "2,3")
    monkeypatch.setenv("KIS_DECISION_SHARD_SLOTS", "2")
    monkeypatch.setenv("KIS_DATA_2_APP_KEY", "key2")
    monkeypatch.setenv("KIS_DATA_2_APP_SECRET", "sec2")
    profile = _profile(tmp_path, COLLECTION_ALTDATA_LOOKBACK_DAYS=2, COLLECTION_ALTDATA_EXTRA_SLOTS=("2",))
    window_start, window_end = altdata_capture._rolling_bounds(pd.Timestamp("2026-09-18").date(), 2)
    cfg = AltDataFetchConfig(
        start=window_start, end=window_end, out_dir=tmp_path, krx_api_key="k",
        universe_symbols=frozenset({"005930"}),
    )
    with pytest.raises(ValueError, match="overlaps KIS_DECISION_SHARD_SLOTS"):
        altdata_capture.run_altdata_capture(pd.Timestamp("2026-09-18").date(), profile=profile, store=store, cfg=cfg)


def test_main_builds_pool_from_settings_in_priority_order(tmp_path, monkeypatch) -> None:
    """main builds the pool from settings in priority order."""
    from src.daily import altdata_capture
    from src.data.capture_contracts import CaptureStatus

    captured: dict[str, Any] = {}

    class _Manifest:
        status = CaptureStatus.COMPLETE

    def _fake_run(trading_day, *, profile, store, cfg):
        captured["cfg"] = cfg
        return _Manifest()

    monkeypatch.setattr(altdata_capture, "CollectionSettings", lambda: _profile(tmp_path))
    monkeypatch.setattr(altdata_capture, "run_altdata_capture", _fake_run)
    monkeypatch.setattr(altdata_capture.settings, "OPENDART_API_KEY", "key-a", raising=False)
    monkeypatch.setattr(altdata_capture.settings, "OPENDART_API_KEY_2", "key-b", raising=False)
    monkeypatch.setattr(altdata_capture.settings, "DART_API_KEY", "key-c", raising=False)
    altdata_capture.main(["--date", "2026-09-18"], trading_day_fn=lambda _d: True)
    pool = captured["cfg"].dart_key_pool
    assert pool.labels == ("KEY_1", "KEY_2", "LEGACY")
    assert captured["cfg"].dart_api_key == ""


def _harness_manifest(status, entries):
    from datetime import date, datetime

    from src.data.capture_contracts import (
        CaptureContext,
        CaptureDataset,
        CaptureManifest,
        SEOUL,
    )

    return CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=date(2026, 9, 18),
            run_id="run-harness",
            dataset=CaptureDataset.SHORTING,
            vendor="owner-local",
            endpoint="altdata-backfill",
            symbol=None,
            venue="KRX",
            session="regular",
            capture_reason="altdata-backfill",
            cohort_id=None,
            scheduled_at=None,
        ),
        cohort=None,
        completed_at=datetime(2026, 9, 18, 21, 40, tzinfo=SEOUL),
        entries=tuple(entries),
        artifacts=(),
        status=status,
    )


def _harness_entry(dataset, status, reason):
    from src.data.capture_contracts import CoverageEntry

    return CoverageEntry(
        symbol=None,
        dataset=dataset,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=status,
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason=reason,
        raw_refs=(),
    )


def _harness_main(monkeypatch, tmp_path, manifest):
    from src.daily import altdata_capture

    calls: list[str] = []

    def _fake_run(trading_day, *, profile, store, cfg):
        calls.append(trading_day.isoformat())
        return manifest

    monkeypatch.setattr(altdata_capture, "CollectionSettings", lambda: _profile(tmp_path))
    monkeypatch.setattr(altdata_capture, "run_altdata_capture", _fake_run)
    return altdata_capture, calls


def test_main_weekend_skips_without_calendar_or_network(tmp_path, monkeypatch) -> None:
    """Weekend skips without calendar or network."""
    from src.daily import altdata_capture

    def _boom_oracle(_d: str) -> bool:
        raise AssertionError("oracle must not be called on weekends")

    def _boom_config(*a: Any, **k: Any) -> Any:
        raise AssertionError("config must not be constructed on weekends")

    def _boom_store(*a: Any, **k: Any) -> Any:
        raise AssertionError("store must not be constructed on weekends")

    monkeypatch.setattr(altdata_capture, "CollectionSettings", lambda: _profile(tmp_path))
    monkeypatch.setattr(altdata_capture, "AltDataFetchConfig", _boom_config)
    monkeypatch.setattr(altdata_capture, "CaptureStore", _boom_store)
    rc = altdata_capture.main(["--date", "2026-09-12"], trading_day_fn=_boom_oracle)
    assert rc == 0


def test_main_weekday_holiday_skips(tmp_path, monkeypatch, caplog) -> None:
    """Weekday holiday skips."""
    import logging

    from src.daily import altdata_capture

    def _boom_run(*a: Any, **k: Any) -> Any:
        raise AssertionError("run must not be invoked on holidays")

    monkeypatch.setattr(altdata_capture, "CollectionSettings", lambda: _profile(tmp_path))
    monkeypatch.setattr(altdata_capture, "run_altdata_capture", _boom_run)
    with caplog.at_level(logging.INFO, logger=altdata_capture.logger.name):
        rc = altdata_capture.main(["--date", "2026-09-24"], trading_day_fn=lambda _d: False)
    assert rc == 0
    assert "reason=non_trading_day" in caplog.text


def test_main_trading_day_runs(tmp_path, monkeypatch) -> None:
    """Trading day runs."""
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    manifest = _harness_manifest(
        CaptureStatus.COMPLETE, (_harness_entry(CaptureDataset.SHORTING, CaptureStatus.COMPLETE, "ok"),)
    )
    altdata_capture, calls = _harness_main(monkeypatch, tmp_path, manifest)
    rc = altdata_capture.main(["--date", "2026-09-18"], trading_day_fn=lambda _d: True)
    assert rc == 0
    assert calls == ["2026-09-18"]


def test_main_oracle_outage_proceeds(tmp_path, monkeypatch, caplog) -> None:
    """Oracle outage proceeds."""
    import logging

    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    manifest = _harness_manifest(
        CaptureStatus.COMPLETE, (_harness_entry(CaptureDataset.SHORTING, CaptureStatus.COMPLETE, "ok"),)
    )
    altdata_capture, calls = _harness_main(monkeypatch, tmp_path, manifest)

    def _outage(_d: str) -> bool:
        raise RuntimeError("KIS trading-day oracle failed")

    with caplog.at_level(logging.WARNING, logger=altdata_capture.logger.name):
        rc = altdata_capture.main(["--date", "2026-09-18"], trading_day_fn=_outage)
    assert rc == 0
    assert calls == ["2026-09-18"]
    assert "calendar_lookup=FAIL" in caplog.text and "proceed=true" in caplog.text


def test_main_unexpected_oracle_error_propagates(tmp_path, monkeypatch) -> None:
    """Unexpected oracle error propagates."""
    import pytest

    from src.daily import altdata_capture

    monkeypatch.setattr(altdata_capture, "CollectionSettings", lambda: _profile(tmp_path))

    def _broken(_d: str) -> bool:
        raise ValueError("oracle contract violated")

    with pytest.raises(ValueError, match="contract violated"):
        altdata_capture.main(["--date", "2026-09-18"], trading_day_fn=_broken)


def test_main_disclosure_quota_exit_is_clean(tmp_path, monkeypatch, caplog) -> None:
    """Disclosure quota exit is clean."""
    import logging

    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    manifest = _harness_manifest(
        CaptureStatus.PARTIAL,
        (
            _harness_entry(CaptureDataset.SHORTING, CaptureStatus.COMPLETE, "ok"),
            _harness_entry(CaptureDataset.DISCLOSURE, CaptureStatus.FAILED, "quota_exceeded"),
        ),
    )
    altdata_capture, _calls = _harness_main(monkeypatch, tmp_path, manifest)
    with caplog.at_level(logging.WARNING, logger=altdata_capture.logger.name):
        rc = altdata_capture.main(["--date", "2026-09-18"], trading_day_fn=lambda _d: True)
    assert rc == 0
    assert "status=DEGRADED" in caplog.text


def test_main_real_source_failure_still_fails(tmp_path, monkeypatch) -> None:
    """Real source failure still fails."""
    from src.data.capture_contracts import CaptureDataset, CaptureStatus

    manifest = _harness_manifest(
        CaptureStatus.PARTIAL,
        (
            _harness_entry(CaptureDataset.DISCLOSURE, CaptureStatus.FAILED, "quota_exceeded"),
            _harness_entry(CaptureDataset.CREDIT_BALANCE, CaptureStatus.FAILED, "vendor_failure"),
        ),
    )
    altdata_capture, _calls = _harness_main(monkeypatch, tmp_path, manifest)
    rc = altdata_capture.main(["--date", "2026-09-18"], trading_day_fn=lambda _d: True)
    assert rc == 1


def test_main_disabled_profile_stays_first(tmp_path, monkeypatch, caplog) -> None:
    """Disabled profile stays first."""
    import logging

    from src.daily import altdata_capture

    def _boom_oracle(_d: str) -> bool:
        raise AssertionError("oracle must not be called when disabled")

    monkeypatch.setattr(altdata_capture, "CollectionSettings", lambda: _profile(tmp_path, COLLECTION_ALTDATA_ENABLED=False))
    with caplog.at_level(logging.INFO, logger=altdata_capture.logger.name):
        rc = altdata_capture.main(["--date", "2026-09-24"], trading_day_fn=_boom_oracle)
    assert rc == 0
    assert "reason=disabled" in caplog.text
