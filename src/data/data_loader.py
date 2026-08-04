"""Data loader module for loading trade logs, theme mappings, and condition search datasets.

Provides clean, format-agnostic data loading interfaces, prioritizing Parquet files with fallback to SQLite.
"""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from src import settings
from src.data.parquet_loader import (
    load_condition_data_from_parquet,
    load_theme_from_parquet,
    load_trade_log_from_parquet,
)

logger = logging.getLogger(__name__)
DB_PATH = str(settings.STOCK_DB_PATH)


def _get_db_connection() -> sqlite3.Connection:
    """Return SQLite connection if DB file exists."""
    if not settings.STOCK_DB_PATH.exists():
        raise FileNotFoundError(f"SQLite DB file not found: {DB_PATH}")
    return sqlite3.connect(DB_PATH)


def load_trade_log() -> pd.DataFrame:
    """Load trade log DataFrame (prioritizing Parquet dataset).

    Returns:
        pd.DataFrame: Trade log dataset.
    """
    if settings.TRADE_LOG_PARQUET_PATH.exists():
        df_pq = load_trade_log_from_parquet()
        if not df_pq.empty:
            return df_pq

    try:
        conn = _get_db_connection()
        query = "SELECT * FROM table_trade_log ORDER BY 매수날짜"
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.warning("Failed to load trade log from SQLite fallback: %s", e)
        return pd.DataFrame()


def load_theme() -> dict[str, str]:
    """Load stock theme mapping dictionary (prioritizing Parquet dataset).

    Returns:
        dict[str, str]: Mapping of stock_code to theme string.
    """
    if settings.THEME_PARQUET_PATH.exists():
        theme_map = load_theme_from_parquet()
        if theme_map:
            return theme_map

    try:
        conn = _get_db_connection()
        query = 'SELECT "종목코드", "테마" FROM table_theme'
        df = pd.read_sql(query, conn)
        conn.close()

        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        return dict(zip(df["종목코드"], df["테마"], strict=False))
    except Exception as e:
        logger.warning("Failed to load theme map from SQLite fallback: %s", e)
        return {}


def load_condition_data(date: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load condition search snapshot history (prioritizing Parquet dataset).

    Args:
        date: Optional date string ('YYYY-MM-DD').
        limit: Optional recent N rows limit.

    Returns:
        pd.DataFrame: Condition history DataFrame.
    """
    if settings.HISTORY_PARQUET_PATH.exists():
        df_pq = load_condition_data_from_parquet(date=date, limit=limit)
        if not df_pq.empty:
            return df_pq

    try:
        conn = _get_db_connection()
        params = []
        if date:
            query = "SELECT * FROM table_condition WHERE 스냅샷_날짜 LIKE ?"
            params.append(f"{date}%")
        else:
            query = "SELECT * FROM table_condition"

        if limit:
            query += " ORDER BY 스냅샷_날짜 DESC LIMIT ?"
            params.append(limit)

        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        logger.warning("Failed to load condition data from SQLite fallback: %s", e)
        return pd.DataFrame()


# Backward compatibility aliases
load_trade_log_from_db = load_trade_log
load_theme_from_db = load_theme
load_condition_data_from_db = load_condition_data
