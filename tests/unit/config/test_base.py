"""config/base.py 경로 설정 도메인 단위 테스트."""

from __future__ import annotations

from pathlib import Path

from src.config.base import PathSettings


def test_path_settings_defaults_point_to_project_root() -> None:
    settings = PathSettings()
    assert Path(__file__).resolve().parent.parent.parent.parent == settings.BASE_DIR
    assert settings.DATA_DIR == settings.BASE_DIR / "data"
    assert settings.CONFIGS_DIR == settings.BASE_DIR / "configs"
    assert settings.MODELS_DIR == settings.BASE_DIR / "artifacts" / "models"


def test_path_settings_derived_paths(tmp_path: Path) -> None:
    settings = PathSettings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data", _env_file=None)
    assert tmp_path / "data" / "parquet" == settings.PARQUET_DIR
    assert settings.TRADE_LOG_PARQUET_PATH == settings.PARQUET_DIR / "trade_log.parquet"
    assert settings.THEME_PARQUET_PATH == settings.PARQUET_DIR / "theme.parquet"
    assert tmp_path / "data" / "daily" == settings.DAILY_DIR
    assert settings.HISTORY_PARQUET_PATH == settings.HISTORY_DIR / "archive.parquet"
    assert settings.TOKEN_FILE == settings.CONFIGS_DIR / "kis_token_cache.json"
    assert tmp_path / "data" / "history" == settings.HISTORY_DIR
    assert settings.LABEL_ENCODER_PATH == settings.MODELS_DIR / "best_stock_rg_cat_encoders.json"
    assert settings.MODEL_PATH == settings.MODELS_DIR / "best_stock_rg_cat.joblib"


def test_path_settings_no_longer_defines_stock_db_or_condition_csv(tmp_path: Path) -> None:
    """stock.db 폐기에 따라 STOCK_DB_PATH/CONDITION_PARQUET_PATH computed field가 base 도메인에서 제거되었는지 검증합니다."""
    settings = PathSettings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data", _env_file=None)
    assert not hasattr(settings, "STOCK_DB_PATH")
    assert not hasattr(settings, "CONDITION_PARQUET_PATH")


def test_ls_tick_max_pages_moved_to_ls_settings() -> None:
    from src.config.base import PathSettings
    from src.config.kiwoom import KiwoomSettings
    from src.config.ls import LsSettings
    from src.settings import Settings

    # Then: the vendor budget left the path-settings class.
    assert "LS_TICK_MAX_PAGES" not in PathSettings.model_fields
    # And: it now sits beside its twin.
    assert "LS_TICK_MAX_PAGES" in LsSettings.model_fields
    assert "KIWOM_TICK_MAX_PAGES" in KiwoomSettings.model_fields

    # And: the value and the consumer-facing access path are unchanged.
    settings = Settings()
    assert settings.LS_TICK_MAX_PAGES == 100
    assert settings.KIWOM_TICK_MAX_PAGES == 30
