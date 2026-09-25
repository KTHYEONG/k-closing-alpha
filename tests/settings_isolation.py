"""Test-support helper restoring live settings delegation."""

from __future__ import annotations

import src.config as _config_mod
from src import settings as _settings_mod
from src.config import Settings


def drop_settings_shadows() -> tuple[str, ...]:
    """Remove settings names that monkeypatch undo re-materialized as real module attributes.

    pytest's undo writes back the value it captured with getattr; under delegation that pins a
    stale value on the module and silently disables live reads for every later test.

    Returns:
        The removed names, sorted.
    """
    names = set(Settings.model_fields) | set(Settings.model_computed_fields)
    removed: set[str] = set()
    for module in (_settings_mod, _config_mod):
        module_dict = vars(module)
        for name in names:
            if name in module_dict:
                del module_dict[name]
                removed.add(name)
    return tuple(sorted(removed))
