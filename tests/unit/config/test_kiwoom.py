from __future__ import annotations


def _hermetic_env(monkeypatch) -> None:
    for name in (
        "KIWOOM_APP_KEY", "KIWOOM_SECRET_KEY", "KIWOOM_BASE_URL",
        "KIWOM_APP_KEY", "KIWOM_SECRET_KEY", "KIWOM_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def test_kiwoom_settings_load_canonical_env_names(monkeypatch) -> None:
    from src.config.kiwoom import KiwoomSettings

    _hermetic_env(monkeypatch)
    monkeypatch.setenv("KIWOOM_APP_KEY", "a")
    monkeypatch.setenv("KIWOOM_SECRET_KEY", "b")

    settings = KiwoomSettings()

    assert settings.KIWOOM_APP_KEY == "a"
    assert settings.KIWOOM_SECRET_KEY == "b"
    assert settings.KIWOOM_BASE_URL == "https://api.kiwoom.com"


def test_kiwoom_settings_accept_legacy_misspelled_env_names(monkeypatch) -> None:
    from src.config.kiwoom import KiwoomSettings

    _hermetic_env(monkeypatch)
    monkeypatch.setenv("KIWOM_APP_KEY", "legacy")

    assert KiwoomSettings().KIWOOM_APP_KEY == "legacy"


def test_kiwoom_settings_new_name_wins_over_legacy(monkeypatch) -> None:
    from src.config.kiwoom import KiwoomSettings

    _hermetic_env(monkeypatch)
    monkeypatch.setenv("KIWOOM_APP_KEY", "new")
    monkeypatch.setenv("KIWOM_APP_KEY", "old")

    assert KiwoomSettings().KIWOOM_APP_KEY == "new"


def test_kiwoom_settings_retired_tick_budget_field_is_gone() -> None:
    from src.config.kiwoom import KiwoomSettings

    assert "KIWOM_TICK_MAX_PAGES" not in KiwoomSettings.model_fields
    assert not any(name.startswith("KIWOM_") for name in KiwoomSettings.model_fields)


def test_global_settings_expose_renamed_kiwoom_fields() -> None:
    import src.config as config_mod
    from src import settings

    assert settings.KIWOOM_BASE_URL == "https://api.kiwoom.com"
    assert isinstance(settings.KIWOOM_APP_KEY, str)
    assert isinstance(settings.KIWOOM_SECRET_KEY, str)
    assert config_mod.KIWOOM_APP_KEY == settings.KIWOOM_APP_KEY
    assert config_mod.KIWOOM_BASE_URL == "https://api.kiwoom.com"
    assert hasattr(config_mod, "KIWOM_APP_KEY") is False
