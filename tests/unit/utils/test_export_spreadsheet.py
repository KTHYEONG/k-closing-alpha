"""Unit tests for the archive → TSV/CSV export utility (export_archive.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.daily.archive import ARCHIVE_COLUMN_ORDER
from src.utils import export_archive
from src.utils.export_archive import export_archive_snapshot


@pytest.fixture
def sample_archive_df() -> pd.DataFrame:
    """Return a valid snapshot DataFrame populated with mock values."""
    row = {col: f"val_{col}" for col in ARCHIVE_COLUMN_ORDER}
    row["스냅샷_날짜"] = "2026-08-03"
    row["선정순위"] = 1
    return pd.DataFrame([row])


def test_export_archive_snapshot_tsv_format(
    tmp_path: Path, sample_archive_df: pd.DataFrame
) -> None:
    out_file = tmp_path / "output.tsv"
    with patch(
        "src.utils.export_archive.fetch_archive_snapshot",
        return_value=sample_archive_df,
    ):
        result_path = export_archive_snapshot(
            out_file, date="2026-08-03", fmt="tsv"
        )

    assert result_path.exists()
    df_read = pd.read_csv(result_path, sep="\t")
    assert list(df_read.columns) == ARCHIVE_COLUMN_ORDER
    assert len(df_read) == 1
    assert str(df_read["스냅샷_날짜"].iloc[0]) == "2026-08-03"


def test_export_archive_snapshot_csv_format(
    tmp_path: Path, sample_archive_df: pd.DataFrame
) -> None:
    out_file = tmp_path / "output.csv"
    with patch(
        "src.utils.export_archive.fetch_archive_snapshot",
        return_value=sample_archive_df,
    ):
        result_path = export_archive_snapshot(
            out_file, date="2026-08-03", fmt="csv"
        )

    assert result_path.exists()
    df_read = pd.read_csv(result_path, encoding="utf-8-sig")
    assert list(df_read.columns) == ARCHIVE_COLUMN_ORDER
    assert len(df_read) == 1


def test_export_archive_snapshot_empty_raises_valueerror(tmp_path: Path) -> None:
    out_file = tmp_path / "output.tsv"
    empty_df = pd.DataFrame(columns=ARCHIVE_COLUMN_ORDER)
    with (
        patch(
            "src.utils.export_archive.fetch_archive_snapshot",
            return_value=empty_df,
        ),
        pytest.raises(ValueError, match="스냅샷이 아카이브에 없습니다"),
    ):
        export_archive_snapshot(out_file, date="9999-12-31")


def test_cli_main_date_flag(
    tmp_path: Path, sample_archive_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export_archive, "DEFAULT_OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["export_archive.py", "--date", "2026-08-03"])

    with patch(
        "src.utils.export_archive.fetch_archive_snapshot",
        return_value=sample_archive_df,
    ):
        export_archive.main()

    generated = tmp_path / "out" / "archive_2026-08-03.tsv"
    assert generated.exists()


def test_cli_main_out_flag(
    tmp_path: Path, sample_archive_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "custom" / "result.csv"
    monkeypatch.setattr(
        sys, "argv", ["export_archive.py", "--out", str(out_path), "--format", "csv"]
    )

    with patch(
        "src.utils.export_archive.fetch_archive_snapshot",
        return_value=sample_archive_df,
    ):
        export_archive.main()

    assert out_path.exists()


def test_cli_main_no_data_raises_valueerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_df = pd.DataFrame(columns=ARCHIVE_COLUMN_ORDER)
    monkeypatch.setattr(sys, "argv", ["export_archive.py"])

    with (
        patch(
            "src.utils.export_archive.fetch_archive_snapshot",
            return_value=empty_df,
        ),
        pytest.raises(ValueError, match="아카이브에 조회 가능한 스냅샷이 없습니다"),
    ):
        export_archive.main()
