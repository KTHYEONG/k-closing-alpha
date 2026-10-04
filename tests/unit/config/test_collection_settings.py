"""Toss backfill CollectionSettings defaults and validation."""

from __future__ import annotations

import pytest

from src.config.collection import CollectionSettings


def test_toss_backfill_defaults() -> None:
    """Declared Toss backfill defaults load without error."""
    profile = CollectionSettings(_env_file=None)
    assert profile.COLLECTION_TOSS_BACKFILL_CONCURRENCY == 8
    assert profile.COLLECTION_TOSS_BASIS_VOLUME_RATIO_MIN == 0.90
    assert profile.COLLECTION_TOSS_BASIS_VOLUME_TOLERANCE == 1e-9
    assert profile.COLLECTION_TOSS_RETENTION_REFERENCE_SYMBOL == "005930"
    assert profile.COLLECTION_TOSS_OUTAGE_FAILURE_SHARE == 0.5
    assert profile.COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS == ("0850-0940", "1510-1550")


def test_toss_backfill_rejects_bad_thresholds() -> None:
    """Ratio, tolerance, concurrency and outage-share bounds are strict."""
    with pytest.raises(ValueError, match="greater than 0"):
        CollectionSettings(COLLECTION_TOSS_BASIS_VOLUME_RATIO_MIN=0.0, _env_file=None)
    with pytest.raises(ValueError, match="less than 1"):
        CollectionSettings(COLLECTION_TOSS_BASIS_VOLUME_RATIO_MIN=1.0, _env_file=None)
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        CollectionSettings(COLLECTION_TOSS_BASIS_VOLUME_TOLERANCE=-0.1, _env_file=None)
    with pytest.raises(ValueError, match="greater than 0"):
        CollectionSettings(COLLECTION_TOSS_BACKFILL_CONCURRENCY=0, _env_file=None)
    with pytest.raises(ValueError, match="greater than 0"):
        CollectionSettings(COLLECTION_TOSS_OUTAGE_FAILURE_SHARE=0.0, _env_file=None)
    with pytest.raises(ValueError, match="less than or equal to 1"):
        CollectionSettings(COLLECTION_TOSS_OUTAGE_FAILURE_SHARE=1.5, _env_file=None)


def test_toss_backfill_rejects_malformed_blackout_window() -> None:
    """A malformed blackout item raises at settings construction."""
    with pytest.raises(ValueError, match="blackout"):
        CollectionSettings(COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS=("8:50",), _env_file=None)
    with pytest.raises(ValueError, match="blackout"):
        CollectionSettings(COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS=("0850-2460",), _env_file=None)


def test_blackout_windows_parse_comma_and_json_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blackout windows accept the comma spelling shared by every env loader."""
    monkeypatch.setenv("COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS", "0850-0940,1510-1550")
    assert CollectionSettings(_env_file=None).COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS == (
        "0850-0940",
        "1510-1550",
    )
    monkeypatch.setenv("COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS", '["0850-0940"]')
    assert CollectionSettings(_env_file=None).COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS == ("0850-0940",)
    monkeypatch.setenv("COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS", "bogus")
    with pytest.raises(ValueError, match="blackout"):
        CollectionSettings(_env_file=None)
