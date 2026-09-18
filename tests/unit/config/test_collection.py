"""CollectionSettings acquisition-limit invariant guards."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.config import Settings
from src.config.collection import CollectionSettings
from src.data.capture_contracts import SessionClock

SEOUL = ZoneInfo("Asia/Seoul")


def _clock(trading_date: date) -> SessionClock:
    return SessionClock.standard(trading_date)


def test_capture_root_follows_configured_data_directory(tmp_path: Path) -> None:
    """Capture root follows configured data directory."""
    settings = Settings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data", _env_file=None)
    capture_root = settings.HISTORY_DIR / "capture"
    assert str(capture_root).startswith(str(tmp_path))
    assert settings.COLLECTION_ROOT is None


def test_enabled_auctions_require_ownership(tmp_path: Path) -> None:
    """Enabled auctions require ownership."""
    with pytest.raises(ValueError, match="declared ownership and research credentials") as exc:
        CollectionSettings(COLLECTION_AUCTION_ENABLED=True, _env_file=None)
    assert "DATA_5" not in str(exc.value)
    with pytest.raises(ValueError, match="declared ownership and research credentials"):
        CollectionSettings(
            COLLECTION_AUCTION_ENABLED=True,
            COLLECTION_RESEARCH_SLOTS=("1",),
            _env_file=None,
        )
    owned = CollectionSettings(
        COLLECTION_AUCTION_ENABLED=True,
        COLLECTION_RESEARCH_SLOTS=("1", "2"),
        COLLECTION_KEY_OWNERSHIP_PATH=tmp_path / "ownership.json",
        _env_file=None,
    )
    assert owned.COLLECTION_RESEARCH_SLOTS == ("1", "2")


def test_repair_and_normal_limits_agree() -> None:
    """Repair and normal limits agree."""
    with pytest.raises(ValueError, match="at least the normal page budget"):
        CollectionSettings(
            COLLECTION_CHART_MAX_PAGES=30,
            COLLECTION_TICK_REPAIR_MAX_PAGES=10,
            _env_file=None,
        )
    ok = CollectionSettings(
        COLLECTION_CHART_MAX_PAGES=30,
        COLLECTION_TICK_REPAIR_MAX_PAGES=30,
        _env_file=None,
    )
    assert ok.COLLECTION_TICK_REPAIR_MAX_PAGES >= ok.COLLECTION_CHART_MAX_PAGES


def test_existing_exports_remain_stable() -> None:
    """Existing exports remain stable."""
    from src import settings as settings_module

    settings = Settings(_env_file=None)
    assert settings.TARGET_CONDITION_NAME == "종가매매"
    assert settings.KIWOM_TICK_MAX_PAGES == 30
    assert settings.COLLECTION_CHART_MAX_PAGES == 30
    assert settings.COLLECTION_RAW_ENABLED is True
    for name in (
        "COLLECTION_ROOT",
        "COLLECTION_RAW_ENABLED",
        "COLLECTION_AUCTION_ENABLED",
        "COLLECTION_ALTDATA_ENABLED",
        "COLLECTION_RESEARCH_SLOTS",
        "COLLECTION_KEY_OWNERSHIP_PATH",
        "COLLECTION_AUCTION_INTERVAL_SECONDS",
        "COLLECTION_REQUEST_TIMEOUT_SECONDS",
        "COLLECTION_CONCURRENCY_PER_KEY",
        "COLLECTION_CHART_MAX_PAGES",
        "COLLECTION_TICK_REPAIR_MAX_PAGES",
        "COLLECTION_ARROW_BATCH_ROWS",
        "COLLECTION_MAX_RSS_MIB",
        "COLLECTION_ALTDATA_LOOKBACK_DAYS",
        "COLLECTION_VERIFIED_CHART_ROUTES",
        "COLLECTION_OPEN_CONFIRM_SECONDS",
        "COLLECTION_SESSION_OVERRIDES",
    ):
        assert hasattr(settings, name)
        assert name in settings_module.__all__
    assert "CollectionSettings" in settings_module.__all__


def test_acquisition_limits_are_strict() -> None:
    """Acquisition limits are strict."""
    with pytest.raises(ValueError, match="greater than 0"):
        CollectionSettings(COLLECTION_AUCTION_INTERVAL_SECONDS=0, _env_file=None)
    with pytest.raises(ValueError, match="greater than 0"):
        CollectionSettings(COLLECTION_CONCURRENCY_PER_KEY=-1, _env_file=None)
    with pytest.raises(ValueError, match="finite"):
        CollectionSettings(COLLECTION_REQUEST_TIMEOUT_SECONDS=float("inf"), _env_file=None)
    with pytest.raises(ValueError, match="finite"):
        CollectionSettings(COLLECTION_REQUEST_TIMEOUT_SECONDS=float("nan"), _env_file=None)
    with pytest.raises(ValueError, match="greater than 30"):
        CollectionSettings(COLLECTION_OPEN_CONFIRM_SECONDS=30, _env_file=None)
    with pytest.raises(ValueError, match="greater than 0"):
        CollectionSettings(COLLECTION_ARROW_BATCH_ROWS=0, _env_file=None)
    with pytest.raises(ValueError, match="greater than 0"):
        CollectionSettings(COLLECTION_MAX_RSS_MIB=0, _env_file=None)
    with pytest.raises(ValueError, match="greater than 0"):
        CollectionSettings(COLLECTION_ALTDATA_LOOKBACK_DAYS=-2, _env_file=None)
    with pytest.raises(ValueError, match="greater than 0"):
        CollectionSettings(COLLECTION_CHART_MAX_PAGES=0, _env_file=None)
    with pytest.raises(ValueError, match="unique"):
        CollectionSettings(COLLECTION_RESEARCH_SLOTS=("1", "1"), _env_file=None)
    with pytest.raises(ValueError, match="decimal pool identifiers"):
        CollectionSettings(COLLECTION_RESEARCH_SLOTS=("",), _env_file=None)
    with pytest.raises(ValueError, match="decimal pool identifiers"):
        CollectionSettings(COLLECTION_RESEARCH_SLOTS=("DATA_5",), _env_file=None)
    with pytest.raises(ValueError, match=r"chart-to-venue|chart routes"):
        CollectionSettings(COLLECTION_VERIFIED_CHART_ROUTES={"": "KRX"}, _env_file=None)


def test_verified_exceptional_session_preserved() -> None:
    """Verified exceptional session."""
    trading_date = date(2026, 9, 17)
    clock = SessionClock(
        trading_date=trading_date,
        open_at=datetime(2026, 9, 17, 10, 0, tzinfo=SEOUL),
        close_at=datetime(2026, 9, 17, 14, 0, tzinfo=SEOUL),
        close_confirmation_deadline=datetime(2026, 9, 17, 14, 3, tzinfo=SEOUL),
        provenance="verified-notice",
    )
    settings = CollectionSettings(
        COLLECTION_SESSION_OVERRIDES={trading_date.isoformat(): clock},
        _env_file=None,
    )
    kept = settings.COLLECTION_SESSION_OVERRIDES[trading_date.isoformat()]
    assert kept.trading_date == trading_date
    assert kept.open_at == clock.open_at
    assert kept.close_at == clock.close_at
    assert kept.provenance == "verified-notice"


def test_legacy_mode_blocks_independent_publication(tmp_path: Path) -> None:
    profile = CollectionSettings(_env_file=None)
    assert profile.COLLECTION_RAW_ENABLED is True
    with pytest.raises(ValueError, match="legacy operating mode"):
        CollectionSettings(
            COLLECTION_RAW_ENABLED=False,
            COLLECTION_AUCTION_ENABLED=True,
            COLLECTION_RESEARCH_SLOTS=("1",),
            COLLECTION_KEY_OWNERSHIP_PATH=tmp_path / "ownership.json",
            _env_file=None,
        )
    with pytest.raises(ValueError, match="legacy operating mode"):
        CollectionSettings(COLLECTION_RAW_ENABLED=False, COLLECTION_ALTDATA_ENABLED=True, _env_file=None)
    legacy = CollectionSettings(COLLECTION_RAW_ENABLED=False, _env_file=None)
    assert legacy.COLLECTION_RAW_ENABLED is False


def test_session_override_keys_must_match_trading_date() -> None:
    clock = _clock(date(2026, 9, 17))
    with pytest.raises(ValueError, match="ISO dates"):
        CollectionSettings(COLLECTION_SESSION_OVERRIDES={"not-a-date": clock}, _env_file=None)
    with pytest.raises(ValueError, match="must equal its trading_date"):
        CollectionSettings(COLLECTION_SESSION_OVERRIDES={"2026-09-18": clock}, _env_file=None)
    assert _clock(date(2026, 9, 17)).provenance == "standard_profile"
