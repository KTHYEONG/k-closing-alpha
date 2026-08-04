"""SQLite DB 로더 단위 테스트.

`src.data.db_loader` 는 `src.data.data_loader` 로 전달하는 레거시 어댑터이므로,
SQLite 경로/연결은 `data_loader` 의 `DB_PATH`/`_get_db_connection` 을 기준으로 검증합니다.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.data import data_loader, db_loader


@pytest.fixture
def patch_db_path(tmp_path, monkeypatch):
    """data_loader 의 SQLite 경로를 임시 DB 로 교체합니다."""
    db_path = tmp_path / "stock.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE table_trade_log (
                매수날짜 TEXT, 종목코드 TEXT, 종가 REAL, 매수가격 REAL, 매도가격 REAL
            )
            """
        )
        conn.execute("CREATE TABLE table_theme (종목코드 TEXT, 테마 TEXT)")
        conn.execute(
            "CREATE TABLE table_condition (스냅샷_날짜 TEXT, 종목코드 TEXT, 순위 INT)"
        )
        conn.executemany(
            "INSERT INTO table_trade_log VALUES (?, ?, ?, ?, ?)",
            [
                ("2024-01-02", "005930", 70000, 7000, 7200),
                ("2024-01-03", "000660", 30000, 30000, 31000),
            ],
        )
        conn.execute(
            "INSERT INTO table_theme VALUES ('005930', '반도체'), ('000660', 'AI')"
        )
        conn.executemany(
            "INSERT INTO table_condition VALUES (?, ?, ?)",
            [
                ("2024-01-02", "005930", 1),
                ("2024-01-03", "000660", 2),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(data_loader.settings, "STOCK_DB_PATH", db_path)
    monkeypatch.setattr(data_loader, "DB_PATH", str(db_path))
    # Parquet 우선 로드를 건너뛰도록 임시 경로로 분리
    monkeypatch.setattr(data_loader.settings, "TRADE_LOG_PARQUET_PATH", tmp_path / "trade.parquet")
    monkeypatch.setattr(data_loader.settings, "THEME_PARQUET_PATH", tmp_path / "theme.parquet")
    monkeypatch.setattr(data_loader.settings, "HISTORY_PARQUET_PATH", tmp_path / "condition.parquet")
    return db_path


def test_load_trade_log_from_db(patch_db_path) -> None:
    df = db_loader.load_trade_log_from_db()
    assert len(df) == 2
    assert set(df["종목코드"]) == {"005930", "000660"}


def test_load_theme_from_db(patch_db_path) -> None:
    theme_map = db_loader.load_theme_from_db()
    assert theme_map == {"005930": "반도체", "000660": "AI"}


def test_get_db_connection_missing_file(tmp_path, monkeypatch) -> None:
    missing = tmp_path / "nope.db"
    monkeypatch.setattr(data_loader.settings, "STOCK_DB_PATH", missing)
    monkeypatch.setattr(data_loader, "DB_PATH", str(missing))
    with pytest.raises(FileNotFoundError):
        data_loader._get_db_connection()


def test_load_condition_data_from_db_with_date(patch_db_path) -> None:
    df = db_loader.load_condition_data_from_db(date="2024-01-02")
    assert len(df) == 1
    assert (df["스냅샷_날짜"] == "2024-01-02").all()


def test_load_condition_data_from_db_with_limit(patch_db_path) -> None:
    df = db_loader.load_condition_data_from_db(limit=1)
    assert len(df) == 1
