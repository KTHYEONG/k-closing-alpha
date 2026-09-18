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
