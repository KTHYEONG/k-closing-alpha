"""SQLite DB 로더 단위 테스트."""

from __future__ import annotations

import pytest

from src.data import db_loader


@pytest.fixture
def patch_db_path(tmp_path, monkeypatch):
    """db_loader.DB_PATH를 임시 DB로 교체합니다."""
    import sqlite3

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
    monkeypatch.setattr(db_loader, "DB_PATH", str(db_path))
    return db_path


def test_load_trade_log_from_db(patch_db_path) -> None:
    df = db_loader.load_trade_log_from_db()
    assert len(df) == 2
    assert set(df["종목코드"]) == {"005930", "000660"}


def test_load_theme_from_db(patch_db_path) -> None:
    theme_map = db_loader.load_theme_from_db()
    assert theme_map == {"005930": "반도체", "000660": "AI"}


def test_get_db_connection_missing_file(tmp_path) -> None:
    missing = str(tmp_path / "nope.db")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(db_loader, "DB_PATH", missing)
    with pytest.raises(FileNotFoundError):
        db_loader.get_db_connection()
    monkeypatch.undo()


def test_load_condition_data_from_db_with_date(patch_db_path) -> None:
    df = db_loader.load_condition_data_from_db(date="2024-01-02")
    assert len(df) == 1
    assert (df["스냅샷_날짜"] == "2024-01-02").all()


def test_load_condition_data_from_db_with_limit(patch_db_path) -> None:
    df = db_loader.load_condition_data_from_db(limit=1)
    assert len(df) == 1
