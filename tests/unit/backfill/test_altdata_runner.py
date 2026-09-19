import pandas as pd

from src.backfill.altdata.runner import _atomic_write_parquet, _incremental_merge


def test_incremental_merge_and_atomic_write(tmp_path) -> None:
    path = tmp_path / "shorting.parquet"
    first = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]), "symbol": ["005930"], "short_volume": [10.0]})
    _atomic_write_parquet(first, path)
    assert path.exists() and not (tmp_path / "shorting.parquet.tmp").exists()
    nxt = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "symbol": ["005930", "005930"],
        "short_volume": [99.0, 20.0],
    })
    merged = _incremental_merge(path, nxt, ("date", "symbol"))
    assert len(merged) == 2
    assert merged.sort_values("date").iloc[0]["short_volume"] == 99.0


def test_altdata_covered_dates_reads_only_date_column(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.backfill.altdata import runner

    path = tmp_path / "panel.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "symbol": ["005930"], "value": [1]}).to_parquet(path, index=False)

    seen_kwargs = {}
    real_read_parquet = pd.read_parquet

    def spy_read_parquet(p, *args, **kwargs):
        seen_kwargs.update(kwargs)
        return real_read_parquet(p, *args, **kwargs)

    monkeypatch.setattr(runner.pd, "read_parquet", spy_read_parquet)

    result = runner._covered_dates(path)

    assert seen_kwargs.get("columns") == ["date"]
    assert len(result) == 1


def test_altdata_atomic_write_parquet_delegates_to_shared_codec(tmp_path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    from src.backfill.altdata import runner

    path = tmp_path / "panel.parquet"
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-01"]), "symbol": ["000660", "005930"], "value": [1, 2]})

    runner._atomic_write_parquet(df, path)

    meta = pq.ParquetFile(path).metadata
    assert meta.row_group(0).column(0).compression.upper() == "ZSTD"
    stored = pd.read_parquet(path)
    assert stored["symbol"].tolist() == ["000660", "005930"]


def test_altdata_package_lazy_reexports() -> None:
    import pytest

    import src.backfill.altdata as pkg

    assert pkg.AltDataFetchConfig.__name__ == "AltDataFetchConfig"
    assert callable(pkg.run_altdata_backfill)
    with pytest.raises(AttributeError, match="no attribute"):
        _ = pkg.no_such_symbol


def _altdata_cfg(tmp_path, **overrides):
    from pathlib import Path

    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig

    base = {
        "start": pd.Timestamp("2026-09-08"),
        "end": pd.Timestamp("2026-09-10"),
        "out_dir": Path(tmp_path) / "altdata",
        "sources": ("shorting",),
    }
    base.update(overrides)
    return AltDataFetchConfig(**base)


def test_run_altdata_backfill_reobserves_present_dates(monkeypatch, tmp_path) -> None:
    """Sources are queried again."""
    import pandas as pd

    from src.backfill.altdata import runner

    cfg = _altdata_cfg(tmp_path)
    days = [pd.Timestamp(d).normalize() for d in pd.bdate_range(cfg.start, cfg.end)]
    panel = cfg.out_dir / "shorting.parquet"
    panel.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": days, "symbol": ["005930"] * len(days),
        "short_volume": [1.0] * len(days), "short_value": [2.0] * len(days),
    }).to_parquet(panel, index=False)
    calls: list[int] = []

    def _collect(cfg_, missing):
        calls.append(len(missing))
        return pd.DataFrame({
            "date": missing, "symbol": ["005930"] * len(missing),
            "short_volume": [9.0] * len(missing),
        })

    monkeypatch.setattr(runner, "collect_shorting", _collect)
    first = runner.run_altdata_backfill(cfg)
    assert first["panels"]["shorting"]["status"] == "up_to_date"
    assert calls == []
    second = runner.run_altdata_backfill(cfg, reobserve=True)
    assert second["panels"]["shorting"]["status"] == "ok"
    assert calls == [len(days)]


def test_run_altdata_backfill_correction_preserves_history(monkeypatch, tmp_path) -> None:
    """Both observations remain."""
    import pandas as pd

    from src.backfill.altdata import runner
    from src.data.capture_store import CaptureStore

    cfg = _altdata_cfg(tmp_path)
    store = CaptureStore(tmp_path / "capture")
    values = iter([3.0, 7.0])

    def _collect(cfg_, missing):
        val = next(values)
        return pd.DataFrame({"date": missing, "symbol": ["005930"] * len(missing), "short_volume": [val] * len(missing)})

    monkeypatch.setattr(runner, "collect_shorting", _collect)
    runner.run_altdata_backfill(cfg, capture_store=store, run_id="run-first", reobserve=True)
    runner.run_altdata_backfill(cfg, capture_store=store, run_id="run-second", reobserve=True)
    frames = sorted((tmp_path / "capture").rglob("SHORTING-nosymbol.parquet"))
    assert len(frames) == 2
    seen = sorted(float(pd.read_parquet(p)["short_volume"].iloc[0]) for p in frames)
    assert seen == [3.0, 7.0]
    assert "capture" in runner.run_altdata_backfill(cfg, capture_store=store, run_id="run-third", reobserve=True)


def test_run_altdata_backfill_missing_credentials_are_explicit(tmp_path) -> None:
    """Source is unavailable."""
    from src.backfill.altdata import runner
    from src.data.capture_store import CaptureStore

    cfg = _altdata_cfg(tmp_path, sources=("disclosure",), dart_api_key="")
    store = CaptureStore(tmp_path / "capture")
    report = runner.run_altdata_backfill(cfg, capture_store=store, run_id="run-nokey", reobserve=True)
    assert report["panels"]["disclosure"]["status"] == "skipped_no_key"
    assert report["capture"]["status"] == "PARTIAL"


def test_run_altdata_backfill_numeric_panel_excludes_provenance_clocks(monkeypatch, tmp_path) -> None:
    """Clocks never enter numeric panel."""
    import pandas as pd

    from src.backfill.altdata import runner
    from src.data.capture_store import CaptureStore

    cfg = _altdata_cfg(tmp_path)
    store = CaptureStore(tmp_path / "capture")

    def _collect(cfg_, missing):
        return pd.DataFrame({
            "date": missing, "symbol": ["005930"] * len(missing),
            "short_volume": [1.0] * len(missing),
            "observed_at": ["2026-09-17T09:00:00+09:00"] * len(missing),
            "capture_ref": ["some/path"] * len(missing),
        })

    monkeypatch.setattr(runner, "collect_shorting", _collect)
    report = runner.run_altdata_backfill(cfg, capture_store=store, run_id="run-clocks", reobserve=True)
    assert report["panels"]["shorting"]["status"] == "ok"
    stored = pd.read_parquet(cfg.out_dir / "shorting.parquet")
    assert "observed_at" not in stored.columns and "capture_ref" not in stored.columns
    assert report["capture"]["status"] == "COMPLETE"


def test_run_altdata_backfill_collector_output_labeled_honestly(monkeypatch, tmp_path) -> None:
    """Artifact is labeled collector output."""
    import pandas as pd

    from src.backfill.altdata import runner
    from src.data.capture_store import CaptureStore

    cfg = _altdata_cfg(tmp_path)
    store = CaptureStore(tmp_path / "capture")
    raw = pd.DataFrame({
        "date": [pd.Timestamp("2026-09-08")], "symbol": ["005930"], "short_volume": [42.0],
    })
    monkeypatch.setattr(runner, "collect_shorting", lambda cfg_, missing: raw)
    runner.run_altdata_backfill(cfg, capture_store=store, run_id="run-honest", reobserve=True)
    frames = list((tmp_path / "capture").rglob("SHORTING-nosymbol.parquet"))
    assert len(frames) == 1
    kept = pd.read_parquet(frames[0])
    assert float(kept["short_volume"].iloc[0]) == 42.0
    assert "http" not in " ".join(kept.columns).lower()


def test_run_altdata_backfill_partial_sources_never_certify_complete(monkeypatch, tmp_path) -> None:
    """Manifest identifies failure and preserves successes."""
    import pandas as pd

    from src.backfill.altdata import runner
    from src.data.capture_store import CaptureStore

    cfg = _altdata_cfg(tmp_path, sources=("shorting", "credit_balance"))
    store = CaptureStore(tmp_path / "capture")
    monkeypatch.setattr(
        runner, "collect_shorting",
        lambda cfg_, missing: pd.DataFrame({"date": missing, "symbol": ["005930"] * len(missing), "short_volume": [1.0] * len(missing)}),
    )

    def _boom(cfg_, missing):
        raise RuntimeError("credit feed down")

    monkeypatch.setattr(runner, "collect_credit_balance", _boom)
    report = runner.run_altdata_backfill(cfg, capture_store=store, run_id="run-partial", reobserve=True)
    assert report["panels"]["shorting"]["status"] == "ok"
    assert report["panels"]["credit_balance"]["status"] == "unavailable"
    assert report["capture"]["status"] == "PARTIAL"
    assert list((tmp_path / "capture").rglob("SHORTING-nosymbol.parquet")) != []


def test_run_altdata_backfill_dart_quota_exceeded_labeled_distinctly(monkeypatch, tmp_path) -> None:
    """DART 계정 한도초과(k-stock-engine 과 키 공유)는 'empty collector result' 로 뭉개지지 않고
    별도 상태(quota_exceeded)와 원인 메시지를 보존해야 진단이 가능하다."""
    import pandas as pd

    from src.backfill.altdata import disclosure as disc_mod
    from src.backfill.altdata import runner
    from src.backfill.altdata.ratelimit import DartNonRetryableError

    cfg = _altdata_cfg(tmp_path, sources=("disclosure",), dart_api_key="k")
    monkeypatch.setattr(disc_mod, "download_corp_code_map", lambda cfg_: pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}))

    def _quota_boom(cfg_, corp_map, **kw):
        raise DartNonRetryableError("DART error status=020 msg=사용한도를 초과하였습니다.")

    monkeypatch.setattr(disc_mod, "collect_disclosures", _quota_boom)
    report = runner.run_altdata_backfill(cfg)
    panel = report["panels"]["disclosure"]
    assert panel["status"] == "quota_exceeded"
    assert "020" in panel["error"] and "사용한도" in panel["error"]


def test_run_altdata_backfill_rejects_inconsistent_capture_context(tmp_path) -> None:
    """Capture configuration is inconsistent."""
    import pytest

    from src.backfill.altdata import runner
    from src.data.capture_store import CaptureStore

    cfg = _altdata_cfg(tmp_path)
    store = CaptureStore(tmp_path / "capture")
    with pytest.raises(ValueError, match="supplied together"):
        runner.run_altdata_backfill(cfg, capture_store=store)
    with pytest.raises(ValueError, match="supplied together"):
        runner.run_altdata_backfill(cfg, run_id="orphan")


def test_run_altdata_backfill_retains_dart_pages(monkeypatch, tmp_path) -> None:
    """Each page has its own identity and clocks."""
    import pandas as pd

    from src.backfill.altdata import disclosure as disc_mod
    from src.backfill.altdata import runner
    from src.data.capture_store import CaptureStore

    def _run_dart_case(reobserve_flag: bool) -> None:
        out = tmp_path / f"alt-{reobserve_flag}"
        cfg = _altdata_cfg(out, sources=("disclosure",), dart_api_key="k")
        store = CaptureStore(out / "capture")
        monkeypatch.setattr(disc_mod, "download_corp_code_map", lambda cfg_: pd.DataFrame({"corp_code": ["00000001"], "stock_code": ["005930"], "corp_name": ["n"]}))
        seen_pages: list[int] = []

        def _fake_collect(cfg_, corp_map, *, on_window=None, covered_dates=None, on_page=None):
            from datetime import datetime

            from src.data.capture_contracts import SEOUL

            assert covered_dates is not None
            if reobserve_flag:
                assert covered_dates == set()
            for page in range(2):
                if on_page is not None:
                    on_page(
                        {"status": "000", "list": [{"rcept_no": f"r{page}"}]},
                        {"endpoint": "dart", "page_no": str(page)},
                        datetime.now(SEOUL), datetime.now(SEOUL), page, 0,
                    )
                    seen_pages.append(page)
            if on_window is not None:
                on_window(pd.DataFrame({"date": [pd.Timestamp("2026-09-08")], "symbol": ["005930"], "n_total": [1], "has_material": [False]}))
            return pd.DataFrame(columns=["date", "symbol"])

        monkeypatch.setattr(disc_mod, "collect_disclosures", _fake_collect)
        report = runner.run_altdata_backfill(cfg, capture_store=store, run_id=f"run-dart-{reobserve_flag}", reobserve=reobserve_flag)
        assert report["panels"]["disclosure"]["status"] == "ok"
        assert seen_pages == [0, 1]
        assert report["capture"]["status"] == "COMPLETE"

    _run_dart_case(False)
    _run_dart_case(True)

    cfg_plain = _altdata_cfg(tmp_path / "plain", sources=("disclosure",), dart_api_key="k")

    def _plain_collect(cfg_, corp_map, **kw):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2026-09-08")], "symbol": ["005930"], "n_total": [1], "has_material": [False],
        })
        if kw.get("on_window") is not None:
            kw["on_window"](df)
        return df

    monkeypatch.setattr(disc_mod, "collect_disclosures", _plain_collect)
    plain = runner.run_altdata_backfill(cfg_plain)
    assert plain["panels"]["disclosure"]["status"] == "ok"
    assert "capture" not in plain


def test_run_altdata_backfill_capture_failure_propagates(monkeypatch, tmp_path) -> None:
    """A page could not be retained."""
    from datetime import datetime

    import pytest

    from src.backfill.altdata import runner
    from src.data.capture_contracts import SEOUL, RawCaptureError
    from src.data.capture_store import CaptureStore

    store = CaptureStore(tmp_path / "capture")

    def _boom(response):
        raise OSError("disk full")

    store.append_response = _boom  # type: ignore[method-assign]
    observer = runner._dart_page_observer(store, __import__("pandas").Timestamp("2026-09-10"), "run-boom", [])
    with pytest.raises(RawCaptureError):
        observer({"status": "000"}, {"endpoint": "dart"}, datetime.now(SEOUL), datetime.now(SEOUL), 0, 0)


def _capture_profile(**overrides):
    from src.config.collection import CollectionSettings

    base = {"COLLECTION_RAW_ENABLED": True, "COLLECTION_ALTDATA_ENABLED": True, "COLLECTION_ALTDATA_LOOKBACK_DAYS": 3}
    base.update(overrides)
    return CollectionSettings(**base, _env_file=None)


def _capture_cfg(tmp_path, trading_day, lookback=3, **overrides):
    import pandas as pd

    from src.backfill.altdata.config import AltDataFetchConfig

    end = pd.Timestamp(trading_day).normalize()
    start = end - pd.Timedelta(days=lookback - 1)
    kw = {"start": start, "end": end, "out_dir": tmp_path / "altdata"}
    kw.update(overrides)
    return AltDataFetchConfig(**kw)


def test_run_altdata_capture_refreshes_declared_window(monkeypatch, tmp_path) -> None:
    """Aggregate manifest reconciling each configured source."""
    import pandas as pd

    import src.daily.altdata_capture as cap
    from src.data.capture_store import CaptureStore

    trading_day = __import__("datetime").date(2026, 9, 10)
    profile = _capture_profile()
    store = CaptureStore(tmp_path / "capture")
    cfg = _capture_cfg(tmp_path, trading_day)
    monkeypatch.setattr(cap, "fetch_krx_daily", lambda window_end, cfg_: pd.DataFrame(columns=["symbol"]))
    seen: dict = {}

    def _fake_backfill(cfg_, *, capture_store=None, run_id=None, reobserve=False):
        from src.data.capture_contracts import (
            CaptureContext,
            CaptureDataset,
            CaptureManifest,
            CaptureStatus,
            CoverageEntry,
        )
        from datetime import datetime

        from src.data.capture_contracts import SEOUL

        seen.update({"reobserve": reobserve, "run_id": run_id, "start": pd.Timestamp(cfg_.start).normalize(), "end": pd.Timestamp(cfg_.end).normalize()})
        assert capture_store is store
        manifest = CaptureManifest(
            schema_version=1,
            context=CaptureContext(
                trading_date=trading_day, run_id=run_id, dataset=CaptureDataset.SHORTING,
                vendor="owner-local", endpoint="altdata-backfill", symbol=None,
                venue="KRX", session="regular", capture_reason="altdata-backfill",
                cohort_id=None, scheduled_at=None,
            ),
            cohort=None,
            completed_at=datetime.now(SEOUL),
            entries=(CoverageEntry(
                symbol=None, dataset=CaptureDataset.SHORTING, venue="KRX", session="regular",
                scheduled_at=None, status=CaptureStatus.COMPLETE, rows=1,
                first_event_time=None, last_event_time=None, reason="ok", raw_refs=(),
            ),),
            artifacts=(),
            status=CaptureStatus.COMPLETE,
        )
        capture_store.publish_manifest(manifest)
        return {"panels": {}, "capture": {"run_id": run_id, "status": "COMPLETE"}}

    monkeypatch.setattr(cap, "run_altdata_backfill", _fake_backfill)
    manifest = cap.run_altdata_capture(trading_day, profile=profile, store=store, cfg=cfg)
    assert manifest.status.value == "COMPLETE"
    assert seen["reobserve"] is True
    assert (seen["start"], seen["end"]) == (pd.Timestamp("2026-09-08"), pd.Timestamp("2026-09-10"))


def test_run_altdata_capture_rejects_disabled_profile_and_mismatched_bounds(monkeypatch, tmp_path) -> None:
    """Disabled profile or mismatched rolling bounds."""
    import pytest

    import src.daily.altdata_capture as cap
    from src.data.capture_store import CaptureStore

    trading_day = __import__("datetime").date(2026, 9, 10)
    store = CaptureStore(tmp_path / "capture")
    cfg = _capture_cfg(tmp_path, trading_day)
    monkeypatch.setattr(cap, "fetch_krx_daily", lambda window_end, cfg_: __import__("pandas").DataFrame(columns=["symbol"]))
    with pytest.raises(ValueError, match="enabled raw and altdata"):
        cap.run_altdata_capture(trading_day, profile=_capture_profile(COLLECTION_ALTDATA_ENABLED=False), store=store, cfg=cfg)
    with pytest.raises(ValueError, match="rolling bounds"):
        cap.run_altdata_capture(trading_day, profile=_capture_profile(), store=store, cfg=_capture_cfg(tmp_path, __import__("datetime").date(2026, 9, 9)))
    monkeypatch.setattr(cap, "run_altdata_backfill", lambda cfg_, **kw: {"panels": {}})
    with pytest.raises(ValueError, match="manifest missing"):
        cap.run_altdata_capture(trading_day, profile=_capture_profile(), store=store, cfg=cfg)


def test_altdata_capture_main_reports_complete_and_incomplete(monkeypatch, tmp_path) -> None:
    """Zero for a complete configured-source run, nonzero for incomplete acquisition."""
    import src.daily.altdata_capture as cap
    from src.data.capture_contracts import CaptureStatus

    class _Manifest:
        def __init__(self, status):
            self.status = status

    monkeypatch.setattr(cap, "CollectionSettings", lambda *a, **k: _capture_profile())
    monkeypatch.setattr(cap, "_capture_root", lambda profile: tmp_path / "cap")
    monkeypatch.setattr(cap.settings, "ALTDATA_DIR", tmp_path / "altdata", raising=False)
    monkeypatch.setattr(cap.settings, "KRX_OPENAPI_KEY", "k", raising=False)
    monkeypatch.setattr(cap, "run_altdata_capture", lambda day, **kw: _Manifest(CaptureStatus.COMPLETE))
    assert cap.main(["--date", "2026-09-10"]) == 0
    monkeypatch.setattr(cap, "run_altdata_capture", lambda day, **kw: _Manifest(CaptureStatus.PARTIAL))
    assert cap.main([]) == 1


def test_altdata_capture_main_skips_when_disabled_and_rejects_bad_date(monkeypatch) -> None:
    """Disabled execution performs no requests."""
    import pytest

    import src.daily.altdata_capture as cap

    monkeypatch.setattr(cap, "CollectionSettings", lambda *a, **k: _capture_profile(COLLECTION_ALTDATA_ENABLED=False))
    called: list[str] = []
    monkeypatch.setattr(cap, "run_altdata_capture", lambda day, **kw: called.append("x") or None)
    # 실측 회귀: disabled는 설정상 정상 상태이지 실패가 아니다 -- 0이 아닌 값을 반환하면
    # systemd가 이를 진짜 실패로 취급해 OnFailure 얼러트를 쏜다(2026-09-18 실측).
    assert cap.main(["--date", "2026-09-10"]) == 0
    assert called == []
    with pytest.raises(ValueError, match="Invalid date"):
        cap.main(["--date", "not-a-date"])


def test_altdata_capture_root_respects_override(tmp_path) -> None:
    """Capture root follows the configured override."""
    import src.daily.altdata_capture as cap

    assert cap._capture_root(_capture_profile(COLLECTION_ROOT=tmp_path / "cap")) == tmp_path / "cap"
    assert cap._capture_root(_capture_profile()).name == "capture"


def test_altdata_capture_main_covers_single_day_window(monkeypatch, tmp_path) -> None:
    """Calendar days are not misrepresented as trading sessions."""
    import src.daily.altdata_capture as cap
    from src.data.capture_contracts import CaptureStatus

    class _Manifest:
        status = CaptureStatus.COMPLETE

    seen: dict = {}
    monkeypatch.setattr(cap, "CollectionSettings", lambda *a, **k: _capture_profile(COLLECTION_ALTDATA_LOOKBACK_DAYS=1))
    monkeypatch.setattr(cap, "_capture_root", lambda profile: tmp_path / "cap1")
    monkeypatch.setattr(cap.settings, "ALTDATA_DIR", tmp_path / "altdata", raising=False)
    monkeypatch.setattr(cap.settings, "KRX_OPENAPI_KEY", "k", raising=False)

    def _fake_run(day, **kw):
        seen.update(kw)
        return _Manifest()

    monkeypatch.setattr(cap, "run_altdata_capture", _fake_run)
    assert cap.main(["--date", "2026-09-10"]) == 0
    assert seen["cfg"].start != seen["cfg"].end
