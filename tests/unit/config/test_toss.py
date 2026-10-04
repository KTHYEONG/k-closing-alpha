from __future__ import annotations


def test_toss_settings_defaults() -> None:
    from src.config.toss import TossSettings

    settings = TossSettings()

    assert settings.TOSS_BASE_URL == "https://openapi.tossinvest.com"
    assert isinstance(settings.TOSS_APP_KEY, str)
    assert isinstance(settings.TOSS_APP_SECRET, str)


def test_toss_retry_settings_defaults() -> None:
    from src.config.toss import TossSettings

    settings = TossSettings()

    assert settings.TOSS_RATE_LIMIT_MAX_RETRIES == 3
    assert settings.TOSS_RATE_LIMIT_BACKOFF_SECONDS == 1.2


def test_toss_retry_settings_validate_bounds() -> None:
    import pytest
    from pydantic import ValidationError

    from src.config.toss import TossSettings

    with pytest.raises(ValidationError):
        TossSettings(TOSS_RATE_LIMIT_MAX_RETRIES=0)
    with pytest.raises(ValidationError):
        TossSettings(TOSS_RATE_LIMIT_BACKOFF_SECONDS=-1)


def test_global_settings_expose_toss_config() -> None:
    import src.config as config_mod
    from src import settings

    assert settings.TOSS_BASE_URL == "https://openapi.tossinvest.com"
    assert isinstance(settings.TOSS_APP_KEY, str)
    assert config_mod.TOSS_APP_KEY == settings.TOSS_APP_KEY
    assert config_mod.TOSS_APP_SECRET == settings.TOSS_APP_SECRET
    assert config_mod.TOSS_BASE_URL == "https://openapi.tossinvest.com"
