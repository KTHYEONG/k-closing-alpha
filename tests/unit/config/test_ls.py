from __future__ import annotations

from src.config.ls import LsSettings


def test_ls_settings_defaults() -> None:
    settings = LsSettings()
    assert settings.LS_BASE_URL.startswith("https://")
    assert isinstance(settings.LS_APP_KEY, str)
    assert isinstance(settings.LS_APP_SECRET, str)


def test_global_settings_expose_ls_tick_budget() -> None:
    from src import settings

    assert "LS_TICK_MAX_PAGES" not in LsSettings.model_fields
    assert settings.COLLECTION_CHART_MAX_PAGES == 30
