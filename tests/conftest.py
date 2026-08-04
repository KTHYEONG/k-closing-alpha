"""공통 pytest fixture 모듈.

Settings 경로 해석, 임시 SQLite DB, 샘플 매매일지 DataFrame 등을 제공합니다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.settings import Settings


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    """임시 디렉토리를 가리키는 Settings 인스턴스를 반환합니다."""
    return Settings(
        BASE_DIR=tmp_path,
        DATA_DIR=tmp_path / "data",
        CONFIGS_DIR=tmp_path / "configs",
        MODELS_DIR=tmp_path / "artifacts" / "models",
    )


@pytest.fixture
def sample_trade_df() -> pd.DataFrame:
    """스케일 보정 테스트용 샘플 매매일지 DataFrame."""
    return pd.DataFrame(
        {
            "종목코드": ["005930", "000660", "005930"],
            "매수날짜": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "종가": [70_000, 30_000, 70_000],
            "매수가격": [7_000, 30_000, 7_000],  # 005930: 1/10 스케일 오류
            "매도가격": [7_200, 31_000, 7_200],
        }
    )


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """임시 SQLite DB 파일을 생성하고 경로를 반환합니다."""
    db_path = tmp_path / "stock.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE table_trade_log (
                매수날짜 TEXT,
                종목코드 TEXT,
                종가 REAL,
                매수가격 REAL,
                매도가격 REAL,
                수익률 REAL
            )
            """
        )
        conn.execute(
            "CREATE TABLE table_theme (종목코드 TEXT, 테마 TEXT)"
        )
        conn.executemany(
            "INSERT INTO table_trade_log VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("2024-01-02", "005930", 70000, 7000, 7200, 2.86),
                ("2024-01-03", "000660", 30000, 30000, 31000, 3.33),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    return db_path
