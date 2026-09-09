"""SCENARIO: spreadsheet_legacy_removal contract -- removed modules stay removed."""
from __future__ import annotations

import importlib

import pytest

_REMOVED_MODULES = [
    "src.sync.vmarket",
    "src.sync.foreign",
    "src.sync.program",
    "src.sync.volume",
    "src.sync.sheet_helpers",
    "src.data.gsheet_loader",
    "src.data.sync_sheet_db",
    "src.backfill.backfill_sheet",
    "src.backfill.run_sheet_backfill",
    "src.backfill.backfill_condition_history",
    "src.backfill.fix_scale",
    "src.processing.scale_corrector",
    "src.config.gsheet",
    "src.daily.materialize_ml_panel",
    "src.tools.migrate_sqlite_to_parquet",
]


@pytest.mark.parametrize("module_name", _REMOVED_MODULES)
def test_spreadsheet_legacy_module_fully_removed(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)

