"""Guards the hermetic-environment helper that keeps host-shell secrets out of the suite."""

from __future__ import annotations

import os

from tests.settings_isolation import ambient_env_names, declared_env_names


def test_declared_names_cover_fields_and_aliases() -> None:
    names = declared_env_names()
    assert "OPENDART_API_KEY_2" in names
    assert "KIWOM_APP_KEY" in names  # AliasChoices member, not a field name


def test_ambient_names_select_only_host_settings() -> None:
    environ = {
        "OPENDART_API_KEY_2": "x",
        "KIS_DATA_3_APP_KEY": "x",  # numbered pool, not a declared field
        "kiwom_app_key": "x",  # case-insensitive alias
        "PATH": "/usr/bin",
        "POLARS_MAX_THREADS": "2",
    }
    assert ambient_env_names(environ) == {"OPENDART_API_KEY_2", "KIS_DATA_3_APP_KEY", "kiwom_app_key"}


def test_suite_starts_without_host_alt_data_keys() -> None:
    assert "OPENDART_API_KEY_2" not in os.environ
    assert "OPENDART_API_KEY" not in os.environ
