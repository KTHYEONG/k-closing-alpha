from __future__ import annotations

import importlib
import logging

import pytest


def test_load_theme_returns_empty_and_logs_when_parquet_missing(tmp_path, monkeypatch, caplog) -> None:
    from src.data import data_loader

    monkeypatch.setattr(data_loader.settings, "THEME_PARQUET_PATH", tmp_path / "missing_theme.parquet")

    with caplog.at_level(logging.WARNING, logger="src.data.data_loader"):
        result = data_loader.load_theme()

    assert result == {}
    assert any("Theme parquet missing" in rec.message for rec in caplog.records)
    assert not hasattr(data_loader, "sqlite3")
    assert not hasattr(data_loader, "_get_db_connection")
    assert not hasattr(data_loader, "DB_PATH")
    assert not hasattr(data_loader, "load_theme_from_db")


def test_load_condition_data_removed_as_orphaned() -> None:
    from src.data import data_loader

    assert not hasattr(data_loader, "load_condition_data")
    assert not hasattr(data_loader, "load_condition_data_from_db")


def test_db_loader_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.data.db_loader")


def test_stock_db_file_and_config_surface_fully_retired() -> None:
    from src.config import Settings

    assert "STOCK_DB_PATH" not in Settings.model_fields


def test_load_theme_returns_mapping_for_valid_file(tmp_path, monkeypatch) -> None:
    import pandas as pd

    from src.data import data_loader

    theme_path = tmp_path / "theme.parquet"
    pd.DataFrame({"종목코드": ["005930"], "테마": ["반도체"]}).to_parquet(theme_path)
    monkeypatch.setattr(data_loader.settings, "THEME_PARQUET_PATH", theme_path)

    assert data_loader.load_theme() == {"005930": "반도체"}


def test_load_theme_propagates_unreadable_error(tmp_path, monkeypatch) -> None:
    from src.data import data_loader
    from src.data.parquet_loader import ThemeMapUnreadableError

    theme_path = tmp_path / "theme.parquet"
    theme_path.write_bytes(b"not a parquet file")
    monkeypatch.setattr(data_loader.settings, "THEME_PARQUET_PATH", theme_path)

    with pytest.raises(ThemeMapUnreadableError):
        data_loader.load_theme()


def test_load_theme_empty_file_warns_empty_status(tmp_path, monkeypatch, caplog) -> None:
    import pandas as pd

    from src.data import data_loader

    theme_path = tmp_path / "theme.parquet"
    pd.DataFrame({"종목코드": ["005930"], "테마": ["  "]}).to_parquet(theme_path)
    monkeypatch.setattr(data_loader.settings, "THEME_PARQUET_PATH", theme_path)

    with caplog.at_level(logging.WARNING, logger="src.data.data_loader"):
        assert data_loader.load_theme() == {}

    assert any("THEME_EMPTY" in rec.message and "Theme parquet missing" in rec.message for rec in caplog.records)
