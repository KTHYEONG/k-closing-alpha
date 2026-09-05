from __future__ import annotations

from src.config.ls import LsSettings


def test_ls_settings_defaults() -> None:
    settings = LsSettings()
    assert settings.LS_BASE_URL.startswith("https://")
    assert isinstance(settings.LS_APP_KEY, str)
    assert isinstance(settings.LS_APP_SECRET, str)


def test_global_settings_expose_tick_budget_and_orderbook_dir() -> None:
    from src import settings

    assert int(settings.LS_TICK_MAX_PAGES) == 100
    assert str(settings.ORDERBOOK_DIR).endswith("orderbook")
    assert settings.ORDERBOOK_DIR.parent == settings.HISTORY_DIR
