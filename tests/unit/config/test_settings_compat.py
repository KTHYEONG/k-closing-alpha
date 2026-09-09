"""Settings 경로 해석 및 환경변수 검증 테스트."""

from __future__ import annotations

from pathlib import Path

from src.settings import Settings


def test_default_paths_point_to_project_root() -> None:
    settings = Settings()
    assert Path(__file__).resolve().parents[3] == settings.BASE_DIR
    assert settings.MODELS_DIR == settings.BASE_DIR / "artifacts" / "models"
    assert settings.DATA_DIR == settings.BASE_DIR / "data"


def test_models_dir_under_artifacts() -> None:
    """모델 아티팩트는 artifacts/models/ 로 이관되어야 합니다."""
    settings = Settings()
    assert str(settings.MODELS_DIR).replace("\\", "/").endswith("artifacts/models")
    assert settings.models_dir == settings.MODELS_DIR


def test_derived_paths_based_on_base_dir(tmp_path: Path) -> None:
    settings = Settings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data")
    assert tmp_path / "data" / "stock.db" == settings.STOCK_DB_PATH
    assert tmp_path / "data" / "daily" / "daily_stocks.csv" == settings.CONDITION_CSV_PATH
    assert settings.MODEL_PATH == settings.MODELS_DIR / "best_stock_rg_cat.joblib"


def test_condition_excel_path_removed(tmp_path: Path) -> None:
    """CONDITION_EXCEL_PATH 레거시 필드는 제거되었습니다."""
    settings = Settings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data", _env_file=None)
    assert not hasattr(settings, "CONDITION_EXCEL_PATH")


def test_kis_config_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KIS_APP_KEY", "test_key")
    monkeypatch.setenv("KIS_APP_SECRET", "test_secret")
    settings = Settings(_env_file=None)
    assert settings.KIS_API_CONFIG["app_key"] == "test_key"
    assert settings.KIS_API_CONFIG["app_secret"] == "test_secret"  # noqa: S105


def test_module_level_backward_compat_reexports() -> None:
    """기존 `settings.XXX` 모듈 레벨 참조가 유지되는지 검증합니다."""
    from src import settings

    assert settings_module_base_dir() == settings.BASE_DIR
    assert settings_module_base_dir() / "artifacts" / "models" == settings.MODELS_DIR


def settings_module_base_dir() -> Path:
    from src import settings

    return settings.BASE_DIR




def test_config_modularity_and_compatibility() -> None:
    """CONFIG_MODULARITY_AND_COMPATIBILITY: Settings \uc2f1\uae00\ud1a4 \ubc0f \ub3c4\uba54\uc778 config \ubaa8\ub4c8\uc774 \uc21c\ud658 \uc784\ud3ec\ud2b8\ub098 \ub204\ub77d \uc18d\uc131 \uc5c6\uc774 \ub85c\ub4dc\ub41c\ub2e4."""
    from src import settings
    from src.config.base import PathSettings
    from src.config.kis import KisSettings
    from src.config.trading import TradingSettings

    assert hasattr(settings, "CONDITION_CSV_PATH")
    assert isinstance(settings.settings, Settings)
    assert isinstance(PathSettings, type)
    assert isinstance(KisSettings, type)
    assert isinstance(TradingSettings, type)
    assert settings.settings.STOCK_DB_PATH == settings.STOCK_DB_PATH
    assert settings.settings.KIS_API_CONFIG["app_key"] == settings.KIS_API_CONFIG["app_key"]



def test_gsheet_settings_removed_from_settings() -> None:
    from src import settings

    for name in (
        "GSPREAD_KEY_PATH_ENV",
        "GSPREAD_SA_JSON",
        "GOOGLE_KEY_PATH",
        "GOOGLE_SHEET_NAME",
        "TRADE_WORKSHEETS",
        "GOTTEN_COLS",
        "gspread_key_path",
        "google_sheet_name",
        "GSheetSettings",
    ):
        assert not hasattr(settings, name), f"{name} should have been removed from settings module"
        assert not hasattr(settings.settings, name), f"{name} should have been removed from Settings instance"

