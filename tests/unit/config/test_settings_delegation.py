"""Settings access delegation invariant guards."""

from __future__ import annotations

from pathlib import Path

import pytest

EXPECTED_MODEL_FIELDS = (
    "COLLECTION_ROOT",
    "COLLECTION_AUCTION_ENABLED",
    "COLLECTION_ALTDATA_ENABLED",
    "COLLECTION_RESEARCH_SLOTS",
    "COLLECTION_ALTDATA_EXTRA_SLOTS",
    "COLLECTION_AUCTION_INTERVAL_SECONDS",
    "COLLECTION_REQUEST_TIMEOUT_SECONDS",
    "COLLECTION_CONCURRENCY_PER_KEY",
    "COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE",
    "COLLECTION_CHART_MAX_PAGES",
    "COLLECTION_TICK_REPAIR_MAX_PAGES",
    "COLLECTION_ARROW_BATCH_ROWS",
    "COLLECTION_MAX_RSS_MIB",
    "COLLECTION_ALTDATA_LOOKBACK_DAYS",
    "COLLECTION_VERIFIED_CHART_ROUTES",
    "COLLECTION_OPEN_CONFIRM_SECONDS",
    "COLLECTION_SESSION_OVERRIDES",
    "ALERT_WEBHOOK_URL",
    "ALERT_GMAIL_USER",
    "ALERT_GMAIL_APP_PASSWORD",
    "ALERT_GMAIL_TO",
    "ALERT_RETRY_ATTEMPTS",
    "ALERT_RETRY_BACKOFF_SECONDS",
    "ALERT_OUTBOX_MAX_DRAIN",
    "TOSS_APP_KEY",
    "TOSS_APP_SECRET",
    "TOSS_BASE_URL",
    "KIWOOM_APP_KEY",
    "KIWOOM_SECRET_KEY",
    "KIWOOM_BASE_URL",
    "OPENDART_API_KEY",
    "OPENDART_API_KEY_2",
    "DART_API_KEY",
    "KRX_OPENAPI_KEY",
    "KRX_OPENAPI_BASE_URL",
    "KRX_OPENAPI_BASE_URLS",
    "KRX_OPENAPI_ENDPOINTS",
    "API_SEMAPHORE_LIMIT",
    "PAPER_SEED_CAPITAL",
    "PAPER_ENTRY_SIZING_BUFFER_BP",
    "LS_APP_KEY",
    "LS_APP_SECRET",
    "LS_BASE_URL",
    "LS_MIN_INTERVAL_SECONDS",
    "LS_RATE_LIMIT_MAX_RETRIES",
    "LS_RATE_LIMIT_BACKOFF_SECONDS",
    "KIS_APP_KEY",
    "KIS_APP_SECRET",
    "KIS_ACCOUNT_ID",
    "KIS_HTS_ID",
    "KIS_DATA_ROLE",
    "KIS_TOKEN_CACHE_DIR",
    "KIS_BASE_URL",
    "BASE_DIR",
    "DATA_DIR",
)


def test_env_base_preserves_field_order() -> None:
    from src.config import Settings

    assert tuple(Settings.model_fields) == EXPECTED_MODEL_FIELDS


def test_env_base_preserves_config() -> None:
    from src.config import (
        AlertSettings,
        AltDataSettings,
        CollectionSettings,
        KisSettings,
        KiwoomSettings,
        LsSettings,
        PathSettings,
        Settings,
        TossSettings,
        TradingSettings,
    )
    from src.config._env import EnvSettings

    assert EnvSettings.model_fields == {}
    for cls in (
        PathSettings,
        KisSettings,
        LsSettings,
        KiwoomSettings,
        TossSettings,
        AlertSettings,
        AltDataSettings,
        CollectionSettings,
        TradingSettings,
        Settings,
    ):
        assert issubclass(cls, EnvSettings)
        assert cls.model_config["env_file"] == EnvSettings.model_config["env_file"]
        assert cls.model_config["env_file_encoding"] == EnvSettings.model_config["env_file_encoding"]
        assert cls.model_config["extra"] == EnvSettings.model_config["extra"]


def test_standalone_domain_class_loads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.config.collection import CollectionSettings

    monkeypatch.setenv("COLLECTION_CHART_MAX_PAGES", "7")
    assert CollectionSettings(_env_file=None).COLLECTION_CHART_MAX_PAGES == 7


def test_default_roots_unchanged() -> None:
    from src.config import Settings
    from src.config._env import PROJECT_ROOT

    defaults = Settings(_env_file=None)
    assert defaults.BASE_DIR == PROJECT_ROOT
    assert defaults.DATA_DIR == PROJECT_ROOT / "data"


def test_instance_mutation_visible_through_module(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import settings

    monkeypatch.setattr(settings.settings, "KIS_DATA_ROLE", "decision")
    assert settings.KIS_DATA_ROLE == "decision"


def test_computed_paths_follow_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src import settings
    from src.config import Settings as SettingsCls

    monkeypatch.setattr(settings.settings, "DATA_DIR", tmp_path)
    assert tmp_path / "history" == settings.HISTORY_DIR
    assert tmp_path / "history" / "price_history.parquet" == settings.PRICE_HISTORY_PARQUET_PATH
    field_names = set(SettingsCls.model_fields) | set(SettingsCls.model_computed_fields)
    if "ORDERBOOK_DIR" in field_names:
        assert tmp_path / "history" / "orderbook" == settings.ORDERBOOK_DIR


def test_previously_unexported_fields_reachable() -> None:
    from src import settings

    assert settings.LS_MIN_INTERVAL_SECONDS == settings.settings.LS_MIN_INTERVAL_SECONDS
    assert settings.COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE == settings.settings.COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE


def test_unknown_name_fails_loudly() -> None:
    from src import settings

    for name in ("NOT_A_FIELD", "base_dir"):
        with pytest.raises(AttributeError):
            getattr(settings, name)
        assert not hasattr(settings, name)


def test_config_package_delegates() -> None:
    import src.config as config_mod

    assert config_mod.KIS_TOKEN_CACHE_DIR == config_mod.settings.KIS_TOKEN_CACHE_DIR


def test_exports_are_classes_and_instance_only() -> None:
    import src.config as config_mod
    import src.settings as settings_mod
    from src.config._env import EnvSettings

    assert config_mod.__all__ == settings_mod.__all__
    for name in config_mod.__all__:
        value = getattr(config_mod, name)
        assert isinstance(value, EnvSettings) or (isinstance(value, type) and issubclass(value, EnvSettings))
    assert "Settings" in config_mod.__all__
    assert "CollectionSettings" in config_mod.__all__
    assert "AlertSettings" in config_mod.__all__


def test_submodule_imports_survive_delegation() -> None:
    from src.config import collection, market_session

    import types

    assert isinstance(collection, types.ModuleType)
    assert isinstance(market_session, types.ModuleType)


def test_monkeypatch_undo_shadow_removed_by_helper(tmp_path: Path) -> None:
    from src import settings as settings_module
    from tests.settings_isolation import drop_settings_shadows

    tmp_a = tmp_path / "a"
    tmp_b = tmp_path / "b"
    patcher = pytest.MonkeyPatch()
    patcher.setattr(settings_module, "HISTORY_DIR", tmp_a)
    patcher.undo()
    try:
        assert "HISTORY_DIR" in vars(settings_module)
        removed = drop_settings_shadows()
        assert "HISTORY_DIR" in removed
        patcher2 = pytest.MonkeyPatch()
        patcher2.setattr(settings_module.settings, "DATA_DIR", tmp_b)
        try:
            assert tmp_b / "history" == settings_module.HISTORY_DIR
        finally:
            patcher2.undo()
    finally:
        drop_settings_shadows()


def test_helper_leaves_exports_intact() -> None:
    from src import settings as settings_module
    from tests.settings_isolation import drop_settings_shadows

    drop_settings_shadows()
    assert drop_settings_shadows() == ()
    assert settings_module.settings is not None
    assert settings_module.Settings is not None
    for name in settings_module.__all__:
        assert hasattr(settings_module, name)


def test_no_snapshot_globals_at_start() -> None:
    import src.config as config_mod
    from src import settings as settings_module
    from src.config import Settings

    field_names = set(Settings.model_fields) | set(Settings.model_computed_fields)
    assert not (set(vars(settings_module)) & field_names)
    assert not (set(vars(config_mod)) & field_names)


def test_hermetic_data_root() -> None:
    from src import settings
    from src.config._env import PROJECT_ROOT

    assert settings.HISTORY_DIR == settings.settings.DATA_DIR / "history"
    assert settings.settings.DATA_DIR != PROJECT_ROOT / "data"
    assert PROJECT_ROOT not in settings.settings.DATA_DIR.parents
    assert "kca-prod-isolation" in str(settings.settings.DATA_DIR)


def test_hermetic_token_cache() -> None:
    from src import settings

    assert settings.KIS_TOKEN_CACHE_DIR == settings.settings.KIS_TOKEN_CACHE_DIR
    assert "kis_cache" in str(settings.KIS_TOKEN_CACHE_DIR)
