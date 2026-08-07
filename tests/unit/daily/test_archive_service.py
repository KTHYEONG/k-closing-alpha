"""일일 아카이브(archive) 서비스 단위 테스트: 변환·에러·임시 경로."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.daily import archive


def test_archive_condition_prefers_csv_over_xlsx(tmp_path: Path) -> None:
    """아카이브가 condition_*.csv 를 우선 인식하고 스냅샷 날짜를 삽입한다."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    csv_file = data_dir / "condition_종가매매.csv"
    csv_file.write_text(
        "(종목코드),종목명,(차트통과),(시나리오)\n000001,AAA,1,신고가\n",
        encoding="utf-8-sig",
    )

    with (
        patch.object(archive.settings, "DATA_DIR", data_dir),
        patch.object(archive.settings, "HISTORY_DIR", tmp_path / "history"),
        patch.object(
            archive.settings, "HISTORY_DB_PATH", tmp_path / "history.db"
        ),
        patch.object(
            archive.settings, "HISTORY_CSV_PATH", tmp_path / "history.csv"
        ),
        patch.object(archive.settings, "CONDITION_CSV_PATH", csv_file),
        patch.object(archive, "import_csv_history_if_needed"),
        patch.object(archive, "upsert_archive_snapshot") as upsert_mock,
    ):
        archive.main()

    upsert_mock.assert_called_once()
    df = upsert_mock.call_args.args[0]
    assert "스냅샷_날짜" in df.columns
    assert "(종목코드)" in df.columns
    assert df["(종목코드)"].astype(str).str.zfill(6).tolist() == ["000001"]


def test_archive_condition_skips_when_csv_and_xlsx_missing(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    with (
        patch.object(archive.settings, "DATA_DIR", data_dir),
        patch.object(archive.settings, "HISTORY_DIR", tmp_path / "history"),
        patch.object(
            archive.settings, "HISTORY_DB_PATH", tmp_path / "history.db"
        ),
        patch.object(
            archive.settings, "HISTORY_CSV_PATH", tmp_path / "history.csv"
        ),
        patch.object(
            archive.settings, "CONDITION_CSV_PATH", data_dir / "condition_종가매매.csv"
        ),
        patch.object(archive, "import_csv_history_if_needed"),
        patch.object(archive, "upsert_history") as upsert_mock,
    ):
        archive.main()

    upsert_mock.assert_not_called()


def test_upsert_history_stores_and_dedups(tmp_path: Path) -> None:
    db_path = tmp_path / "history.db"
    df = pd.DataFrame(
        {"스냅샷_날짜": ["2026-08-04"], "(종목코드)": ["000001"], "종목명": ["AAA"]}
    )
    with patch("src.data.parquet_loader.upsert_condition_parquet"):
        archive.upsert_history(df, str(db_path))

    rows = archive.fetch_date_rows("2026-08-04", str(db_path))
    assert len(rows) == 1
    assert rows["(종목코드)"].tolist() == ["000001"]


def test_fetch_date_rows_raises_on_missing_db(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        archive.fetch_date_rows("2026-08-04", str(tmp_path / "nope.db"))


def test_import_csv_history_if_needed_migrates_legacy_csv(tmp_path: Path) -> None:
    history_csv = tmp_path / "history.csv"
    history_csv.write_text(
        "스냅샷_날짜,종목코드\n2026-08-01,000002\n2026-08-02,000003\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "history.db"

    with patch("src.data.parquet_loader.upsert_condition_parquet"):
        archive.import_csv_history_if_needed(str(history_csv), str(db_path))
        # 두 번째 호출은 누락 날짜가 없으므로 무시
        archive.import_csv_history_if_needed(str(history_csv), str(db_path))

    rows = archive.fetch_date_rows("2026-08-01", str(db_path))
    assert rows["종목코드"].astype(str).str.zfill(6).tolist() == ["000002"]
