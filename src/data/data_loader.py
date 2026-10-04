"""Data loader module for loading theme mappings and condition search datasets from Parquet."""

from __future__ import annotations

import logging

from src import settings
from src.data.parquet_loader import (  # noqa: F401 - documents the propagated failure type
    ThemeMapUnreadableError,
    load_theme_from_parquet,
)

logger = logging.getLogger(__name__)


def load_theme() -> dict[str, str]:
    """Load the stock theme mapping used for archive enrichment.

    Returns:
        The mapping from ``load_theme_from_parquet``. An absent or row-empty theme file
        yields ``{}`` with a ``[DATA]`` WARNING; callers must then leave theme values
        missing (NaN), never substitute a placeholder.

    Raises:
        ThemeMapUnreadableError: The theme file exists but is unreadable or schema-invalid.
            Propagated unchanged; the caller decides whether theme is essential.
    """
    if not settings.THEME_PARQUET_PATH.exists():
        logger.warning(
            "[DATA] Theme parquet missing stage=theme_load status=THEME_MISSING path=%s",
            settings.THEME_PARQUET_PATH,
        )
        return {}
    theme_map = load_theme_from_parquet()
    if theme_map:
        return theme_map
    logger.warning(
        "[DATA] Theme parquet missing or empty stage=theme_load status=THEME_EMPTY path=%s",
        settings.THEME_PARQUET_PATH,
    )
    return {}
