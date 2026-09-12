from __future__ import annotations


def test_toss_settings_defaults() -> None:
    from src.config.toss import TossSettings

    settings = TossSettings()

    assert settings.TOSS_BASE_URL == "https://openapi.tossinvest.com"
    assert isinstance(settings.TOSS_APP_KEY, str)
    assert isinstance(settings.TOSS_APP_SECRET, str)


def test_global_settings_expose_toss_config() -> None:
    import src.config as config_mod
    from src import settings

    assert settings.TOSS_BASE_URL == "https://openapi.tossinvest.com"
    assert isinstance(settings.TOSS_APP_KEY, str)
    assert config_mod.TOSS_APP_KEY == settings.TOSS_APP_KEY
    assert config_mod.TOSS_APP_SECRET == settings.TOSS_APP_SECRET
    assert config_mod.TOSS_BASE_URL == "https://openapi.tossinvest.com"
