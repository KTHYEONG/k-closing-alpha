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
    assert tmp_path / "data" / "stock.db" == settings.STOCK_DB_PATH
    assert tmp_path / "data" / "parquet" == settings.PARQUET_DIR
    assert settings.TRADE_LOG_PARQUET_PATH == settings.PARQUET_DIR / "trade_log.parquet"
    assert settings.THEME_PARQUET_PATH == settings.PARQUET_DIR / "theme.parquet"
    assert settings.CONDITION_PARQUET_PATH == settings.PARQUET_DIR / "condition_history.parquet"
    assert settings.TOKEN_FILE == settings.CONFIGS_DIR / "kis_token_cache.json"
    assert tmp_path / "data" / "chart_pass_cache.json" == settings.CHART_PASS_CACHE_FILE
    assert tmp_path / "data" / "history" == settings.HISTORY_DIR
    assert settings.LABEL_ENCODER_PATH == settings.MODELS_DIR / "best_stock_rg_cat_encoders.json"
    assert settings.MODEL_PATH == settings.MODELS_DIR / "best_stock_rg_cat.joblib"


def test_path_settings_spec_alias_backward_compat(tmp_path: Path) -> None:
    settings = PathSettings(BASE_DIR=tmp_path, _env_file=None)
    assert settings.base_dir == settings.BASE_DIR
    assert settings.data_dir == settings.DATA_DIR
    assert settings.artifacts_dir == tmp_path / "artifacts"
    assert settings.models_dir == settings.MODELS_DIR
