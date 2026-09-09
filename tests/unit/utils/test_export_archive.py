"""Unit tests for the export_archive string cleanup (archive.db references removed).

Covers the docstring-only contract changes in ``src/utils/export_archive.py``:
module docstring, ``export_archive_snapshot`` docstring, and CLI description
no longer mention the retired ``archive.db`` / SQLite fallback path.
"""

from __future__ import annotations

import sys

import pytest

from src.utils import export_archive
from src.utils.export_archive import export_archive_snapshot


def test_module_docstring_no_longer_mentions_archive_db() -> None:
    doc = export_archive.__doc__ or ""
    assert "archive.db" not in doc
    assert "archive.parquet" in doc


def test_snapshot_docstring_no_longer_mentions_sqlite_fallback() -> None:
    doc = export_archive_snapshot.__doc__ or ""
    assert "archive.db" not in doc
    assert "SQLite" not in doc
    assert "archive.parquet" in doc


def test_cli_description_no_longer_mentions_archive_db(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["export_archive", "--help"])
    with pytest.raises(SystemExit, match="0"):
        export_archive._parse_args()
    out = capsys.readouterr().out
    assert "archive.db" not in out
    assert "archive.parquet" in out
