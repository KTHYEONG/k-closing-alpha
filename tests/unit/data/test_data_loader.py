from __future__ import annotations

import importlib

import pytest


def test_theme_loader_module_retired() -> None:
    # Theme is research-only: no runtime module loads it (research tools take an explicit path).
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.data.data_loader")


def test_db_loader_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.data.db_loader")


def test_stock_db_file_and_config_surface_fully_retired() -> None:
    from src.config import Settings

    assert "STOCK_DB_PATH" not in Settings.model_fields
