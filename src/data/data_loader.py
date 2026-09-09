"""Data loader module for loading theme mappings and condition search datasets from Parquet."""

from __future__ import annotations

import logging

from src import settings
from src.data.parquet_loader import load_theme_from_parquet

logger = logging.getLogger(__name__)


def load_theme() -> dict[str, str]:
    """Load stock theme mapping dictionary (prioritizing Parquet dataset).

    Returns:
        dict[str, str]: Mapping of stock_code to theme string.
    """
    if settings.THEME_PARQUET_PATH.exists():
        theme_map = load_theme_from_parquet()
        if theme_map:
            return theme_map

    logger.warning("Theme parquet missing or empty at %s; returning empty theme map", settings.THEME_PARQUET_PATH)
    return {}
