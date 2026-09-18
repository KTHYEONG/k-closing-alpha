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

    rc = altdata_capture.main(["--date", "2026-09-18"])

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
