"""config/gsheet.py Google Sheets 설정 도메인 단위 테스트."""

from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.config.gsheet import GSheetSettings


def test_gsheet_settings_defaults() -> None:
    settings = GSheetSettings(_env_file=None)
    assert settings.GOOGLE_SHEET_NAME == "Stock"
    assert settings.TRADE_WORKSHEETS == ["Trade", "Trade2"]
    assert settings.THEME_WORKSHEET_NAME == "코드_테마_DB"
    assert settings.GOTTEN_COLS["CODE"] == "(종목코드)"


def test_google_key_path_empty_when_env_unset() -> None:
    settings = Settings(_env_file=None)
    assert Path("") == settings.GOOGLE_KEY_PATH


def test_google_key_path_resolves_relative_to_base_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GSPREAD_KEY_PATH", "configs/key.json")
    settings = Settings(BASE_DIR=tmp_path, _env_file=None)
    assert tmp_path / "configs" / "key.json" == settings.GOOGLE_KEY_PATH


def test_google_key_path_preserves_absolute(tmp_path: Path, monkeypatch) -> None:
    key = tmp_path / "abs" / "key.json"
    monkeypatch.setenv("GSPREAD_KEY_PATH", str(key))
    settings = Settings(BASE_DIR=tmp_path, _env_file=None)
    assert key == settings.GOOGLE_KEY_PATH


def test_gsheet_settings_spec_alias_backward_compat(monkeypatch) -> None:
    monkeypatch.setenv("GSPREAD_KEY_PATH", "configs/key.json")
    settings = GSheetSettings(_env_file=None)
    assert settings.gspread_key_path == "configs/key.json"
    assert settings.google_sheet_name == "Stock"
