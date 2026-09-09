from pathlib import Path
import pandas as pd
import pytest

from src import settings
from src.data.parquet_loader import (
    _atomic_write_parquet,
    load_condition_data_from_parquet,
    load_theme_from_parquet,
    save_theme_to_parquet,
    upsert_condition_parquet,
)


@pytest.fixture
def tmp_parquet_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p_dir = tmp_path / "parquet"
    p_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "PARQUET_DIR", p_dir)
    monkeypatch.setattr(settings, "TRADE_LOG_PARQUET_PATH", p_dir / "trade_log.parquet")
    monkeypatch.setattr(settings, "THEME_PARQUET_PATH", p_dir / "theme.parquet")
    monkeypatch.setattr(settings, "HISTORY_PARQUET_PATH", p_dir / "condition_history.parquet")
    return p_dir


def test_save_and_load_theme_parquet(tmp_parquet_dir: Path) -> None:
    data = {
        "종목코드": [5930, "000660"],
        "테마": ["반도체", "반도체"],
    }
    df_theme = pd.DataFrame(data)
    save_theme_to_parquet(df_theme)

    theme_map = load_theme_from_parquet()
    assert isinstance(theme_map, dict)
    assert theme_map.get("005930") == "반도체"
    assert theme_map.get("000660") == "반도체"


def test_upsert_and_load_condition_parquet(tmp_parquet_dir: Path) -> None:
    data_day1 = {
        "스냅샷_날짜": ["2026-08-01 15:30:00", "2026-08-01 15:30:00"],
        "종목코드": ["005930", "000660"],
        "순위": [1, 2],
    }
    df1 = pd.DataFrame(data_day1)
    upsert_condition_parquet(df1)

    df_loaded = load_condition_data_from_parquet(date="2026-08-01")
    assert len(df_loaded) == 2

    # 중복 업서트 테스트 (동일 날짜/종목코드)
    upsert_condition_parquet(df1)
    df_loaded_dedup = load_condition_data_from_parquet(date="2026-08-01")
    assert len(df_loaded_dedup) == 2


def test_parquet_loader_atomic_write_delegates_and_still_works(tmp_path: Path) -> None:
    df = pd.DataFrame({"종목코드": ["005930"], "종가": [70000]})
    target = tmp_path / "legacy.parquet"

    _atomic_write_parquet(df, target)

    assert target.exists()
    loaded = pd.read_parquet(target)
    assert loaded.loc[0, "종목코드"] == "005930"


def test_parquet_loader_module_no_longer_exposes_trade_log_functions() -> None:
    import src.data.parquet_loader as mod

    assert not hasattr(mod, "save_trade_log_to_parquet")
    assert not hasattr(mod, "load_trade_log_from_parquet")
    assert hasattr(mod, "save_theme_to_parquet")
    assert hasattr(mod, "load_theme_from_parquet")
    assert hasattr(mod, "upsert_condition_parquet")
    assert hasattr(mod, "load_condition_data_from_parquet")

