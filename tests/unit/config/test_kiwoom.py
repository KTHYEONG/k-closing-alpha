from __future__ import annotations


def test_kiwoom_settings_defaults() -> None:
    from src.config.kiwoom import KiwoomSettings

    settings = KiwoomSettings()
    assert settings.KIWOM_BASE_URL == "https://api.kiwoom.com"
    assert isinstance(settings.KIWOM_APP_KEY, str)
    assert isinstance(settings.KIWOM_SECRET_KEY, str)
    assert settings.KIWOM_TICK_MAX_PAGES == 30


def test_global_settings_expose_kiwoom_config() -> None:
    from src import settings
    import src.config as config_mod

    assert settings.KIWOM_BASE_URL == "https://api.kiwoom.com"
    assert isinstance(settings.KIWOM_APP_KEY, str)
    assert int(settings.KIWOM_TICK_MAX_PAGES) == 30
    assert config_mod.KIWOM_APP_KEY == settings.KIWOM_APP_KEY
    assert config_mod.KIWOM_BASE_URL == "https://api.kiwoom.com"

