"""config/trading.py 트레이딩 설정 도메인 단위 테스트."""

from __future__ import annotations

from src.config import Settings
from src.config.trading import TradingSettings


def test_trading_settings_defaults() -> None:
    settings = TradingSettings(_env_file=None)
    assert settings.TARGET_CONDITION_NAME == "종가매매"
    assert settings.OVERHEATED_CONDITION_NAME == "단기과열"
    assert settings.API_SEMAPHORE_LIMIT == 4
    assert settings.API_SLEEP_INTERVAL == 0.2
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
    assert tmp_path / "data" / "daily" / "daily_stocks.csv" == settings.CONDITION_CSV_PATH
    assert tmp_path / "data" / "condition_상따.xlsx" == settings.CONDITION_EXCEL_PATH
    assert tmp_path / "data" / "history" == settings.HISTORY_DIR
    assert settings.HISTORY_DB_PATH == settings.HISTORY_DIR / "condition_history_상따.db"
    assert settings.HISTORY_CSV_PATH == settings.HISTORY_DIR / "condition_history_상따.csv"
