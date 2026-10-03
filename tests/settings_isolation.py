"""Test-support helper restoring live settings delegation."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator

import pydantic_settings
from pydantic import AliasChoices

import src.config as _config_mod
from src import settings as _settings_mod
from src.config import Settings

# Credential/runtime namespaces whose members are not all declared as settings fields
# (e.g. the numbered KIS_DATA_<n>_* key pools), so field introspection alone would miss them.
_AMBIENT_NAMESPACES = re.compile(
    r"^(KIS|KIWOOM|LS|TOSS|OPENDART|ALERT|LIVE|COLLECTION|BINANCE|KRX|ECOS|FRED|TIINGO)_|^(USE_KRX_OPENAPI|FUTURES_REDIS_URL)$"
)


def _settings_classes(base: type = pydantic_settings.BaseSettings) -> Iterator[type[pydantic_settings.BaseSettings]]:
    for sub in base.__subclasses__():
        yield sub
        yield from _settings_classes(sub)


def declared_env_names() -> set[str]:
    """Every environment variable name any settings class can read (fields, prefixes, aliases)."""
    names: set[str] = set()
    for cls in _settings_classes():
        prefix = str(cls.model_config.get("env_prefix", ""))
        for field_name, field in cls.model_fields.items():
            alias = field.validation_alias
            if isinstance(alias, AliasChoices):
                names.update(a for a in alias.choices if isinstance(a, str))
            elif isinstance(alias, str):
                names.add(alias)
            names.add(prefix + field_name)
    return names


def ambient_env_names(environ: dict[str, str] | None = None) -> set[str]:
    """Names in ``environ`` that a test must not inherit from the host shell.

    Union of declared settings names (so new fields are covered automatically) and the
    credential namespaces above. Matching is case-insensitive, like pydantic-settings.
    """
    environ = os.environ if environ is None else environ
    declared = {name.upper() for name in declared_env_names()}
    return {key for key in environ if key.upper() in declared or _AMBIENT_NAMESPACES.match(key.upper())}


def scrub_ambient_env() -> tuple[str, ...]:
    """Remove host-shell settings/credentials from ``os.environ`` for the whole process."""
    removed = sorted(ambient_env_names())
    for key in removed:
        os.environ.pop(key, None)
    return tuple(removed)


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
