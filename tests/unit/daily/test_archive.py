"""Unit tests for the redesigned archive module (Parquet + SQLite dual persistence).

SCENARIO_ARCHIVE_UPSERT_01: Upsert candidates DataFrame into parquet and db without duplication.
SCENARIO_ARCHIVE_FETCH_02: Fetch latest or specified date snapshot in standard 27 column order aligned with spreadsheet.
SCENARIO_ARCHIVE_EXPORT_03: Export candidate snapshot as TSV string matching spreadsheet layout for copy-pasting.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from src.daily import archive


@pytest.fixture
def tmp_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect archive storage paths to a temp dir for isolated tests."""
    a_dir = tmp_path / "history"
    a_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(archive.settings, "HISTORY_DIR", a_dir)
    monkeypatch.setattr(
        archive.settings, "HISTORY_PARQUET_PATH", a_dir / "archive.parquet"
    )
    monkeypatch.setattr(archive.settings, "HISTORY_DB_PATH", a_dir / "archive.db")
    return a_dir


def _candidate_row(code: str, name: str, rank: int, date: str = "2026-08-04") -> dict:
    return {
        "스냅샷_날짜": date,
        "종목코드": code,
        "종목명": name,
        "시가": 1000,
        "고가": 1100,
        "저가": 900,
        "종가": 1050,
        "전일종가": 1000,
        "시가총액": 5000.0,
        "거래대금": 120.0,
        "등락률": 5.0,
        "선정순위": rank,
        "기관_순매수": 1.0,
        "외국인_순매수": 2.0,
        "프로그램_순매수": 0.5,
        "체결강도": 120.0,
        "시장구분": "KOSPI",
        "총_종목수": 2,
        "평균_거래대금": 100.0,
        "kospi": 0.5,
        "kosdaq": 0.3,
        "v_kospi": 12.5,
        "v_kosdaq": 15.2,
        "거래량": 100000,
        "테마_섹터": "반도체",
        "시나리오": "거래량 폭증",
    }


def test_scenario_archive_upsert_01(tmp_archive: Path) -> None:
    """SCENARIO_ARCHIVE_UPSERT_01: Upsert candidates DataFrame into parquet and db without duplication."""
    df = pd.DataFrame(
        [
            _candidate_row("005930", "삼성전자", 1),
            _candidate_row("000660", "SK하이닉스", 2),
        ]
    )
    assert archive.upsert_archive_snapshot(df, snapshot_date="2026-08-04") == 2

    dup = pd.DataFrame([_candidate_row("005930", "삼성전자", 1)])
    assert archive.upsert_archive_snapshot(dup, snapshot_date="2026-08-04") == 2

    parquet_df = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    assert len(parquet_df) == 2
    assert set(parquet_df["종목코드"].astype(str).str.zfill(6)) == {"000660", "005930"}

    with sqlite3.connect(archive.settings.HISTORY_DB_PATH) as conn:
        db_df = pd.read_sql("SELECT * FROM condition_history", conn)
    assert len(db_df) == 2
    assert set(db_df["종목코드"].astype(str).str.zfill(6)) == {"000660", "005930"}


def test_python_assertion_archive_upsert(tmp_archive: Path) -> None:
    assert (
        archive.upsert_archive_snapshot(
            pd.DataFrame([{"종목코드": "005930", "종목명": "삼성전자"}]),
            snapshot_date="2026-08-04",
        )
        >= 1
    )


def test_scenario_archive_fetch_02(tmp_archive: Path) -> None:
    """SCENARIO_ARCHIVE_FETCH_02: Fetch latest or specified date snapshot in standard 27 column order aligned with spreadsheet."""
    archive.upsert_archive_snapshot(
        pd.DataFrame([_candidate_row("005930", "삼성전자", 1, date="2026-08-03")]),
        snapshot_date="2026-08-03",
    )
    archive.upsert_archive_snapshot(
        pd.DataFrame([_candidate_row("005930", "삼성전자", 1, date="2026-08-04")]),
        snapshot_date="2026-08-04",
    )

    latest = archive.fetch_archive_snapshot()
    assert latest["스냅샷_날짜"].tolist() == ["2026-08-04"]
    assert latest.columns.tolist() == archive.ARCHIVE_COLUMN_ORDER

    specified = archive.fetch_archive_snapshot("2026-08-03")
    assert specified["스냅샷_날짜"].tolist() == ["2026-08-03"]
    assert specified.columns.tolist() == archive.ARCHIVE_COLUMN_ORDER
    assert len(archive.ARCHIVE_COLUMN_ORDER) == 26


def test_scenario_archive_export_03(tmp_archive: Path) -> None:
    """SCENARIO_ARCHIVE_EXPORT_03: Export candidate snapshot as TSV string matching spreadsheet layout for copy-pasting."""
    archive.upsert_archive_snapshot(
        pd.DataFrame([_candidate_row("005930", "삼성전자", 1)]),
        snapshot_date="2026-08-04",
    )
    tsv = archive.export_archive_for_spreadsheet("2026-08-04")
    lines = tsv.splitlines()
    assert lines[0].split("\t") == archive.ARCHIVE_COLUMN_ORDER
    assert lines[1].startswith("2026-08-04\t005930\t삼성전자\t1000\t")
    assert len(lines[1].split("\t")) == 26

    latest_tsv = archive.export_archive_for_spreadsheet()
    assert latest_tsv.splitlines()[0].split("\t") == archive.ARCHIVE_COLUMN_ORDER


def test_upsert_fills_snapshot_date_when_missing(tmp_archive: Path) -> None:
    count = archive.upsert_archive_snapshot(
        pd.DataFrame([{"종목코드": "005930", "종목명": "삼성전자"}])
    )
    assert count >= 1
    df = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    assert df["스냅샷_날짜"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()


def test_upsert_keeps_existing_snapshot_date(tmp_archive: Path) -> None:
    df = pd.DataFrame(
        [{"스냅샷_날짜": "2026-07-01", "종목코드": "005930", "종목명": "삼성전자"}]
    )
    archive.upsert_archive_snapshot(df)
    loaded = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    assert loaded["스냅샷_날짜"].tolist() == ["2026-07-01"]


def test_fetch_falls_back_to_sqlite_when_parquet_missing(tmp_archive: Path) -> None:
    archive.upsert_archive_snapshot(
        pd.DataFrame([_candidate_row("005930", "삼성전자", 1)]),
        snapshot_date="2026-08-04",
    )
    archive.settings.HISTORY_PARQUET_PATH.unlink()
    df = archive.fetch_archive_snapshot("2026-08-04")
    assert not df.empty
    assert df["종목코드"].tolist() == ["005930"]


def test_fetch_empty_archive_returns_standard_columns(tmp_archive: Path) -> None:
    df = archive.fetch_archive_snapshot()
    assert df.empty
    assert df.columns.tolist() == archive.ARCHIVE_COLUMN_ORDER


def test_export_empty_df_returns_header() -> None:
    tsv = archive.export_archive_for_spreadsheet(pd.DataFrame())
    assert tsv.splitlines() == ["\t".join(archive.ARCHIVE_COLUMN_ORDER)]


def test_export_empty_df_without_header_returns_empty() -> None:
    assert archive.export_archive_for_spreadsheet(pd.DataFrame(), include_header=False) == ""


def test_export_df_direct_without_header() -> None:
    df = pd.DataFrame([{"종목코드": "1", "종목명": "테스트"}])
    tsv = archive.export_archive_for_spreadsheet(df, include_header=False)
    lines = tsv.splitlines()
    assert len(lines) == 1
    fields = lines[0].split("\t")
    assert len(fields) == 26
    assert fields[1] == "000001"


def test_upsert_adds_missing_columns_to_existing_table(tmp_archive: Path) -> None:
    with sqlite3.connect(archive.settings.HISTORY_DB_PATH) as conn:
        conn.execute('CREATE TABLE condition_history ("종목코드" TEXT)')
    count = archive.upsert_archive_snapshot(
        pd.DataFrame([{"종목코드": "005930", "종목명": "삼성전자"}]),
        snapshot_date="2026-08-04",
    )
    assert count == 1
    df = archive.fetch_archive_snapshot("2026-08-04")
    assert len(df) == 1
    assert df["종목코드"].tolist() == ["005930"]
