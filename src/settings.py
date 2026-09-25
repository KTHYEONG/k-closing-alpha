"""Project global settings (live-delegating bridge to `src.config`)."""

from __future__ import annotations

from typing import Any

import src.config as _config
from src.config import (
    AlertSettings,
    AltDataSettings,
    CollectionSettings,
    KisSettings,
    KiwoomSettings,
    LsSettings,
    PathSettings,
    Settings,
    TossSettings,
    TradingSettings,
    settings,
)

__all__ = [
    "AlertSettings",
    "AltDataSettings",
    "CollectionSettings",
    "KisSettings",
    "KiwoomSettings",
    "LsSettings",
    "PathSettings",
    "Settings",
    "TossSettings",
    "TradingSettings",
    "settings",
]


def __getattr__(name: str) -> Any:
    """Delegate legacy `from src import settings` attribute reads to the live Settings singleton.

    Kept as a thin bridge so consumer modules need no import changes; every field read is resolved
    at access time.

    Raises:
        AttributeError: `name` is not a Settings field or computed field.
    """
    return _config.__getattr__(name)
