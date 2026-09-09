"""config/trading.py 트레이딩 설정 도메인 단위 테스트."""

from __future__ import annotations

from src.config import Settings
from src.config.trading import TradingSettings


def test_trading_settings_defaults() -> None:
    settings = TradingSettings(_env_file=None)
    assert settings.TARGET_CONDITION_NAME == "종가매매"
    assert settings.OVERHEATED_CONDITION_NAME == "단기과열"
    assert settings.API_SEMAPHORE_LIMIT == 8
    assert not hasattr(settings, "API_SLEEP_INTERVAL")
    assert settings.EMA_PERIOD == 20
    assert settings.SMA_PERIOD == 120
    assert settings.CANDLE_BODY_RATIO_THRESHOLD == 0.5
    assert settings.GAP_UP_THRESHOLD == 0.1
    assert "거래량 폭증" in settings.DEFAULT_SCENARIOS
    assert settings.DAY_NAME_MAP[0] == "월요일"


def test_trading_condition_name_drives_aggregate_paths(tmp_path) -> None:
    settings = Settings(
        BASE_DIR=tmp_path,
        DATA_DIR=tmp_path / "data",
        TARGET_CONDITION_NAME="상따",
        _env_file=None,
    )
    assert tmp_path / "data" / "history" == settings.HISTORY_DIR
    assert settings.HISTORY_PARQUET_PATH == settings.HISTORY_DIR / "archive.parquet"





def test_candidate_source_mode_setting_is_removed() -> None:
    from src import settings
    from src.config.trading import TradingSettings

    # Then: the pipeline is automated-only, so the mode switch is gone
    assert "CANDIDATE_SOURCE_MODE" not in TradingSettings.model_fields
    assert not hasattr(settings, "CANDIDATE_SOURCE_MODE")


def test_legacy_sqlite_and_csv_computed_paths_removed() -> None:
    """stock.db/archive.db/daily_stocks.csv 폐기에 따라 관련 computed path가 완전히 제거되었는지 검증합니다."""
    from src import settings
    from src.config import Settings as SettingsClass

    for name in (
        "STOCK_DB_PATH",
        "HISTORY_DB_PATH",
        "HISTORY_CSV_PATH",
        "CONDITION_CSV_PATH",
        "CONDITION_PARQUET_PATH",
    ):
        assert not hasattr(settings, name), f"{name} should have been removed from settings module"
        assert not hasattr(settings.settings, name), f"{name} should have been removed from Settings instance"
        assert name not in SettingsClass.model_fields
