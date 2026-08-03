"""Settings 경로 해석 및 환경변수 검증 테스트."""

from __future__ import annotations

from pathlib import Path

from src.settings import Settings


def test_default_paths_point_to_project_root() -> None:
    settings = Settings()
    assert Path(__file__).resolve().parent.parent.parent == settings.BASE_DIR
    assert settings.MODELS_DIR == settings.BASE_DIR / "artifacts" / "models"
    assert settings.DATA_DIR == settings.BASE_DIR / "data"


def test_models_dir_under_artifacts() -> None:
    """모델 아티팩트는 artifacts/models/ 로 이관되어야 합니다."""
    settings = Settings()
    assert str(settings.MODELS_DIR).endswith("artifacts/models")
    assert settings.models_dir == settings.MODELS_DIR


def test_derived_paths_based_on_base_dir(tmp_path: Path) -> None:
    settings = Settings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data")
    assert tmp_path / "data" / "stock.db" == settings.STOCK_DB_PATH
    assert tmp_path / "data" / "condition_종가매매.xlsx" == settings.CONDITION_EXCEL_PATH
    assert settings.MODEL_PATH == settings.MODELS_DIR / "best_stock_rg_cat.joblib"


def test_kis_config_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("KIS_APP_KEY", "test_key")
    monkeypatch.setenv("KIS_APP_SECRET", "test_secret")
    settings = Settings(_env_file=None)
    assert settings.KIS_API_CONFIG["app_key"] == "test_key"
    assert settings.KIS_API_CONFIG["app_secret"] == "test_secret"  # noqa: S105


def test_google_key_path_absolute_resolution(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GSPREAD_KEY_PATH", "configs/key.json")
    settings = Settings(BASE_DIR=tmp_path, _env_file=None)
    assert tmp_path / "configs" / "key.json" == settings.GOOGLE_KEY_PATH


def test_module_level_backward_compat_reexports() -> None:
    """기존 `settings.XXX` 모듈 레벨 참조가 유지되는지 검증합니다."""
    from src import settings

    assert settings_module_base_dir() == settings.BASE_DIR
    assert settings_module_base_dir() / "artifacts" / "models" == settings.MODELS_DIR


def settings_module_base_dir() -> Path:
    from src import settings

    return settings.BASE_DIR
