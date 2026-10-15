from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.dbsec import DbsecSettings
from src.config.secret_fields import SECRET_SETTING_FIELDS


def test_dbsec_settings_default_disabled_without_key(monkeypatch) -> None:
    monkeypatch.delenv("DB_APP_KEY", raising=False)
    monkeypatch.delenv("DB_APP_SECRET", raising=False)
    settings = DbsecSettings()
    assert settings.DB_APP_KEY == ""


def test_dbsec_token_interval_rejects_below_one_minute() -> None:
    with pytest.raises(ValidationError):
        DbsecSettings(DB_TOKEN_ISSUE_MIN_INTERVAL_SECONDS=30)


def test_dbsec_credentials_registered_as_secrets() -> None:
    assert "DB_APP_KEY" in SECRET_SETTING_FIELDS
    assert "DB_APP_SECRET" in SECRET_SETTING_FIELDS


def test_dbsec_auction_path_defaults_disabled_and_rejects_bad_deadline(monkeypatch) -> None:
    monkeypatch.delenv("DB_AUCTION_PATH_ENABLED", raising=False)
    assert DbsecSettings().DB_AUCTION_PATH_ENABLED is False
    with pytest.raises(ValidationError):
        DbsecSettings(DB_AUCTION_PATH_DEADLINE_HHMMSS="19:00")
