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


def test_derived_paths_based_on_base_dir(tmp_path: Path) -> None:
    settings = Settings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data")
    assert tmp_path / "artifacts" / "models" == settings.MODELS_DIR
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

    assert not hasattr(settings, "CONDITION_CSV_PATH")
    assert isinstance(settings.settings, Settings)
    assert isinstance(PathSettings, type)
    assert isinstance(KisSettings, type)
    assert isinstance(TradingSettings, type)
    assert settings.settings.PARQUET_DIR == settings.PARQUET_DIR
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


def test_models_dir_follows_base_dir_override(tmp_path: Path) -> None:
    from src.settings import Settings

    # Given: only BASE_DIR is overridden; MODELS_DIR is NOT passed.
    settings = Settings(BASE_DIR=tmp_path, DATA_DIR=tmp_path / "data", CONFIGS_DIR=tmp_path / "configs")

    # Then: the artifact paths follow the override instead of pinning to the repo root.
    assert tmp_path / "artifacts" / "models" == settings.MODELS_DIR
    assert settings.MODEL_PATH == settings.MODELS_DIR / "best_stock_rg_cat.joblib"
    assert settings.LABEL_ENCODER_PATH == settings.MODELS_DIR / "best_stock_rg_cat_encoders.json"

    # And: every other derived path follows too, so the artifact tree is not split.
    for name in (
        "PARQUET_DIR", "DAILY_DIR", "HISTORY_DIR", "ORDERBOOK_DIR", "ALTDATA_DIR",
        "PRICE_HISTORY_PARQUET_PATH", "HISTORY_PARQUET_PATH", "TOKEN_FILE",
        "MODELS_DIR", "MODEL_PATH", "LABEL_ENCODER_PATH",
    ):
        assert str(getattr(settings, name)).startswith(str(tmp_path)), f"{name} ignored the BASE_DIR override"


def test_default_settings_paths_are_unchanged() -> None:
    from pathlib import Path

    from src.settings import Settings

    # Given: the default instance, as production constructs it.
    settings = Settings()
    root = Path(__file__).resolve().parents[3]

    # Then: nothing about the default layout moved.
    assert root == settings.BASE_DIR
    assert root / "data" == settings.DATA_DIR
    assert root / "artifacts" / "models" == settings.MODELS_DIR
    assert root / "artifacts" / "models" / "best_stock_rg_cat.joblib" == settings.MODEL_PATH
    assert root / "data" / "history" / "price_history.parquet" == settings.PRICE_HISTORY_PARQUET_PATH


def test_lowercase_setting_aliases_are_removed() -> None:
    from src import settings as settings_module
    from src.settings import Settings

    instance = Settings()
    aliases = (
        "base_dir", "data_dir", "artifacts_dir", "models_dir",
        "kis_app_key", "kis_app_secret", "kis_account_id",
    )

    # Then: the shim is gone from both access paths.
    for name in aliases:
        assert not hasattr(instance, name), f"Settings.{name} alias should be removed"
        assert not hasattr(settings_module, name), f"settings module {name} re-export should be removed"
        assert name not in settings_module.__all__

    # And: the real names still work.
    for name in ("BASE_DIR", "DATA_DIR", "MODELS_DIR", "KIS_APP_KEY", "KIS_APP_SECRET", "KIS_ACCOUNT_ID"):
        assert hasattr(instance, name)
        assert hasattr(settings_module, name)



def test_alert_settings_threaded_into_global_settings_singleton() -> None:
    from src import settings
    from src.config import AlertSettings, Settings

    # Then: AlertSettings 가 Settings 의 믹스인 베이스
    assert issubclass(Settings, AlertSettings)
    # And: 모듈 레벨 재수출이 싱글톤 인스턴스 값과 일치
    assert settings.ALERT_WEBHOOK_URL == settings.settings.ALERT_WEBHOOK_URL
    assert settings.ALERT_GMAIL_USER == settings.settings.ALERT_GMAIL_USER
    assert settings.ALERT_GMAIL_APP_PASSWORD == settings.settings.ALERT_GMAIL_APP_PASSWORD
    assert settings.ALERT_GMAIL_TO == settings.settings.ALERT_GMAIL_TO
    # And: __all__ 에 신규 심볼이 등록됨
    assert "AlertSettings" in settings.__all__
    assert "ALERT_WEBHOOK_URL" in settings.__all__
