"""One-time migration script to convert existing SQLite stock.db tables to Parquet files."""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from src import settings
from src.data.parquet_loader import (
    save_theme_to_parquet,
    save_trade_log_to_parquet,
    upsert_condition_parquet,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_all() -> None:
    """Migrate SQLite tables to Parquet files."""
    db_path = settings.STOCK_DB_PATH
    if not db_path.exists():
        logger.warning("SQLite DB path does not exist: %s", db_path)
        return

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        tables = [row[0] for row in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        logger.info("Found tables in SQLite: %s", tables)

        if "table_trade_log" in tables:
            df_trade = pd.read_sql("SELECT * FROM table_trade_log", conn)
            save_trade_log_to_parquet(df_trade)
            logger.info("Migrated table_trade_log -> Parquet")

        if "table_theme" in tables:
            df_theme = pd.read_sql("SELECT * FROM table_theme", conn)
            save_theme_to_parquet(df_theme)
            logger.info("Migrated table_theme -> Parquet")

        for cond_table in ["condition_history", "table_condition"]:
            if cond_table in tables:
                df_cond = pd.read_sql(f"SELECT * FROM {cond_table}", conn)  # noqa: S608
                upsert_condition_parquet(df_cond)
                logger.info("Migrated %s -> Parquet", cond_table)

    finally:
        conn.close()
        logger.info("Migration to Parquet completed.")


if __name__ == "__main__":
    migrate_all()
