"""Domain settings package (Settings singleton)."""

from __future__ import annotations

from typing import Any

from src.config.alerts import AlertSettings
from src.config.altdata import AltDataSettings
from src.config.base import PathSettings
from src.config.collection import CollectionSettings
from src.config.kis import KisSettings
from src.config.kiwoom import KiwoomSettings
from src.config.ls import LsSettings
from src.config.toss import TossSettings
from src.config.trading import TradingSettings


class Settings(PathSettings, KisSettings, LsSettings, TradingSettings, AltDataSettings, KiwoomSettings, TossSettings, AlertSettings, CollectionSettings):
    """Combined application settings singleton."""


settings = Settings()


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
    """Resolve Settings field names against the live singleton at access time.

    Module-level names used to be import-time copies, so runtime mutation of the singleton was
    invisible to module-attribute consumers and fields absent from the copy list were unreachable.
    Delegation keeps `config.X` reads live while preserving the attribute-style API.

    Raises:
        AttributeError: `name` is neither a Settings model field nor a computed field.
    """
    if name in Settings.model_fields or name in Settings.model_computed_fields:
        return getattr(settings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
