"""Legacy db_loader adapter module.

Forwarding functions to src.data.data_loader for backward compatibility.
"""

from __future__ import annotations

from src.data.data_loader import (
    load_condition_data,
    load_condition_data_from_db,
    load_theme,
    load_theme_from_db,
    load_trade_log,
    load_trade_log_from_db,
)

__all__ = [
    "load_condition_data",
    "load_condition_data_from_db",
    "load_theme",
    "load_theme_from_db",
    "load_trade_log",
    "load_trade_log_from_db",
]
