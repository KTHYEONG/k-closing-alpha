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
    """SCENARIO_ARCHIVE_UPSERT_01: Upsert candidates DataFrame into parquet and db replacing previous snapshot for same date."""
    df = pd.DataFrame(
        [
            _candidate_row("005930", "삼성전자", 1),
            _candidate_row("000660", "SK하이닉스", 2),
        ]
    )
    assert archive.upsert_archive_snapshot(df, snapshot_date="2026-08-04") == 2

    dup = pd.DataFrame([_candidate_row("005930", "삼성전자", 1)])
    assert archive.upsert_archive_snapshot(dup, snapshot_date="2026-08-04") == 1

    parquet_df = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    assert len(parquet_df) == 1
    assert set(parquet_df["종목코드"].astype(str).str.zfill(6)) == {"005930"}

    with sqlite3.connect(archive.settings.HISTORY_DB_PATH) as conn:
        db_df = pd.read_sql("SELECT * FROM condition_history", conn)
    assert len(db_df) == 1
    assert set(db_df["종목코드"].astype(str).str.zfill(6)) == {"005930"}


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

    latest_month = archive.fetch_archive_snapshot()
    assert latest_month["스냅샷_날짜"].tolist() == ["2026-08-03", "2026-08-04"]
    assert latest_month.columns.tolist() == archive.ARCHIVE_READ_COLUMN_ORDER

    specified = archive.fetch_archive_snapshot("2026-08-03")
    assert specified["스냅샷_날짜"].tolist() == ["2026-08-03"]
    assert specified.columns.tolist() == archive.ARCHIVE_READ_COLUMN_ORDER
    assert len(archive.ARCHIVE_COLUMN_ORDER) == 37


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
    assert len(lines[1].split("\t")) == len(archive.ARCHIVE_COLUMN_ORDER)

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


def test_upsert_preserves_timezone_aware_timestamps(tmp_archive: Path) -> None:
    """스냅샷 시각이 유실되지 않고 Asia/Seoul timezone-aware 로 보존됩니다."""
    archive.upsert_archive_snapshot(
        pd.DataFrame([_candidate_row("005930", "삼성전자", 1)]),
        snapshot_date="2026-08-04",
    )
    loaded = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    ts = pd.to_datetime(loaded["snapshot_timestamp"])
    assert ts.notna().all()
    assert ts.dt.tz is not None
    # 시각 미기록 데이터는 결정적 15:30 KST 관례를 사용합니다.
    assert ts.dt.hour.iloc[0] == 15
    assert ts.dt.minute.iloc[0] == 30


def test_upsert_dedups_by_snapshot_identity_when_intraday(tmp_archive: Path) -> None:
    """동일 날짜/종목이어도 intraday 캡처 시각이 다르면 스냅샷 정체성으로 보존됩니다."""
    row = _candidate_row("005930", "삼성전자", 1)
    morning = dict(row, snapshot_timestamp=pd.Timestamp("2026-08-04 09:00:00", tz="Asia/Seoul"))
    afternoon = dict(row, snapshot_timestamp=pd.Timestamp("2026-08-04 14:00:00", tz="Asia/Seoul"))
    archive.upsert_archive_snapshot(pd.DataFrame([morning]), snapshot_date="2026-08-04")
    archive.upsert_archive_snapshot(pd.DataFrame([afternoon]), snapshot_date="2026-08-04")
    loaded = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    assert len(loaded) == 2
    assert loaded["snapshot_timestamp"].nunique() == 2

    # 동일 스냅샷 정체성으로 재 upsert 시 해당 행만 교체되어 2건이 유지됩니다.
    updated = dict(morning, 종목명="삼성전자개편")
    archive.upsert_archive_snapshot(pd.DataFrame([updated]), snapshot_date="2026-08-04")
    loaded = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    assert len(loaded) == 2
    morning_ts = pd.Timestamp("2026-08-04 09:00:00", tz="Asia/Seoul")
    assert (
        loaded.loc[pd.to_datetime(loaded["snapshot_timestamp"]) == morning_ts, "종목명"]
        .iloc[0]
        == "삼성전자개편"
    )


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
    assert df.columns.tolist() == archive.ARCHIVE_READ_COLUMN_ORDER


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
    assert len(fields) == len(archive.ARCHIVE_COLUMN_ORDER)
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


def test_main_fetch_target_date_saves_tsv(
    tmp_archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() FETCH_TARGET_DATE 분기는 조회 결과를 archive_{date}.tsv로 저장합니다."""
    archive.upsert_archive_snapshot(
        pd.DataFrame([_candidate_row("005930", "삼성전자", 1)]),
        snapshot_date="2026-08-04",
    )
    monkeypatch.setattr(archive, "FETCH_TARGET_DATE", "2026-08-04")

    archive.main()

    target_file = tmp_archive / "archive_2026-08-04.tsv"
    assert target_file.exists()
    df = pd.read_csv(target_file, sep="\t")
    assert "2026-08-04" in df["스냅샷_날짜"].astype(str).values


def test_upsert_localizes_naive_snapshot_timestamp(tmp_archive: Path) -> None:
    """Naive(시간대 미지정) 스냅샷 시각은 Asia/Seoul 로 로컬라이즈됩니다."""
    row = _candidate_row("005930", "삼성전자", 1)
    row["snapshot_timestamp"] = "2026-08-04 09:00:00"
    archive.upsert_archive_snapshot(pd.DataFrame([row]), snapshot_date="2026-08-04")
    loaded = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    ts = pd.to_datetime(loaded["snapshot_timestamp"])
    assert ts.dt.tz is not None
    assert ts.dt.hour.iloc[0] == 9


def test_upsert_preserves_supplied_decision_timestamp(tmp_archive: Path) -> None:
    """호출자가 decision/feature_available 타임스탬프를 지정하면 보존됩니다."""
    row = _candidate_row("005930", "삼성전자", 1)
    row["feature_available_timestamp"] = pd.Timestamp("2026-08-04 08:00:00", tz="Asia/Seoul")
    row["decision_timestamp"] = pd.Timestamp("2026-08-04 15:30:00", tz="Asia/Seoul")
    archive.upsert_archive_snapshot(pd.DataFrame([row]), snapshot_date="2026-08-04")
    loaded = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    feature_ts = pd.to_datetime(loaded["feature_available_timestamp"])
    decision_ts = pd.to_datetime(loaded["decision_timestamp"])
    assert feature_ts.dt.hour.iloc[0] == 8
    assert decision_ts.dt.hour.iloc[0] == 15


def test_upsert_dedups_multiple_intraday_captures_in_one_batch(tmp_archive: Path) -> None:
    """단일 배치에 동일 날짜/종목의 서로 다른 intraday 시각이 있으면 스냅샷 정체성으로 중복 제거합니다."""
    row = _candidate_row("005930", "삼성전자", 1)
    morning = dict(row, snapshot_timestamp=pd.Timestamp("2026-08-04 09:00:00", tz="Asia/Seoul"))
    noon = dict(row, snapshot_timestamp=pd.Timestamp("2026-08-04 12:00:00", tz="Asia/Seoul"))
    dup = dict(row, snapshot_timestamp=pd.Timestamp("2026-08-04 09:00:00", tz="Asia/Seoul"))
    archive.upsert_archive_snapshot(
        pd.DataFrame([morning, noon, dup]), snapshot_date="2026-08-04"
    )
    loaded = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    assert len(loaded) == 2
    assert loaded["snapshot_timestamp"].nunique() == 2


def test_upsert_handles_null_snapshot_timestamp(tmp_archive: Path) -> None:
    """snapshot_timestamp 가 NaT 인 행은 IS NULL 정체성으로 교체되며 오류 없이 저장됩니다."""
    row = _candidate_row("005930", "삼성전자", 1)
    row["snapshot_timestamp"] = pd.NaT
    archive.upsert_archive_snapshot(pd.DataFrame([row]), snapshot_date="2026-08-04")
    loaded = pd.read_parquet(archive.settings.HISTORY_PARQUET_PATH)
    assert len(loaded) == 1


def test_upsert_sqlite_null_timestamp_identity_replacement(tmp_archive: Path) -> None:
    """NULL snapshot_timestamp 행은 IS NULL 정체성으로 재배치(중복 없이 교체)됩니다."""
    import sqlite3

    db_path = str(archive.settings.HISTORY_DB_PATH)
    row = _candidate_row("005930", "삼성전자", 1)
    with_null = pd.DataFrame([dict(row, snapshot_timestamp=pd.NaT)])
    archive._upsert_sqlite_archive(with_null, db_path)
    archive._upsert_sqlite_archive(with_null, db_path)
    with sqlite3.connect(db_path) as conn:
        db_df = pd.read_sql("SELECT * FROM condition_history", conn)
    assert len(db_df) == 1


def test_archive_main_preserves_real_capture_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_archive: Path
) -> None:
    import os
    from datetime import datetime

    csv_path = tmp_path / "daily_stocks.csv"
    pd.DataFrame({"종목코드": ["005930"], "종목명": ["삼성전자"]}).to_csv(csv_path, index=False, encoding="utf-8-sig")

    captured_ts = datetime(2026, 9, 3, 15, 18, 0)
    mtime_epoch = captured_ts.timestamp()
    os.utime(csv_path, (mtime_epoch, mtime_epoch))

    monkeypatch.setattr(archive.settings, "CONDITION_CSV_PATH", csv_path)

    archive.main()

    stored = archive.fetch_archive_snapshot(snapshot_date="2026-09-03")
    ts = pd.to_datetime(stored["snapshot_timestamp"].iloc[0])
    assert ts.hour == 15
    assert ts.minute == 18


def test_fetch_archive_snapshot_defaults_to_latest_snapshot_per_code(tmp_archive: Path) -> None:
    import pandas as pd

    from src.daily import archive

    row1 = _candidate_row("005930", "삼성전자", 1)
    row1["snapshot_timestamp"] = pd.Timestamp("2026-08-04 15:19:00", tz="Asia/Seoul")
    row1["등락률"] = 5.0
    archive.upsert_archive_snapshot(pd.DataFrame([row1]), snapshot_date="2026-08-04")

    row2 = _candidate_row("005930", "삼성전자", 1)
    row2["snapshot_timestamp"] = pd.Timestamp("2026-08-04 15:25:00", tz="Asia/Seoul")
    row2["등락률"] = 7.0
    archive.upsert_archive_snapshot(pd.DataFrame([row2]), snapshot_date="2026-08-04")

    result = archive.fetch_archive_snapshot(snapshot_date="2026-08-04")

    assert len(result) == 1
    assert result.iloc[0]["등락률"] == 7.0


def test_fetch_archive_snapshot_latest_only_false_preserves_full_history(tmp_archive: Path) -> None:
    import pandas as pd

    from src.daily import archive

    row1 = _candidate_row("005930", "삼성전자", 1)
    row1["snapshot_timestamp"] = pd.Timestamp("2026-08-04 15:19:00", tz="Asia/Seoul")
    archive.upsert_archive_snapshot(pd.DataFrame([row1]), snapshot_date="2026-08-04")

    row2 = _candidate_row("005930", "삼성전자", 1)
    row2["snapshot_timestamp"] = pd.Timestamp("2026-08-04 15:25:00", tz="Asia/Seoul")
    archive.upsert_archive_snapshot(pd.DataFrame([row2]), snapshot_date="2026-08-04")

    result = archive.fetch_archive_snapshot(snapshot_date="2026-08-04", latest_only=False)

    assert len(result) == 2


def test_export_archive_for_spreadsheet_deduplicates_reruns(tmp_archive: Path) -> None:
    import pandas as pd

    from src.daily import archive

    row1 = _candidate_row("005930", "삼성전자", 1)
    row1["snapshot_timestamp"] = pd.Timestamp("2026-08-04 15:19:00", tz="Asia/Seoul")
    archive.upsert_archive_snapshot(pd.DataFrame([row1]), snapshot_date="2026-08-04")

    row2 = _candidate_row("005930", "삼성전자", 1)
    row2["snapshot_timestamp"] = pd.Timestamp("2026-08-04 15:25:00", tz="Asia/Seoul")
    archive.upsert_archive_snapshot(pd.DataFrame([row2]), snapshot_date="2026-08-04")

    tsv = archive.export_archive_for_spreadsheet("2026-08-04")

    data_lines = [ln for ln in tsv.strip().split("\n")][1:]  # noqa: C416
    assert len(data_lines) == 1


def test_upsert_archive_snapshot_logs_rerun_detection(tmp_archive: Path, caplog) -> None:
    import logging

    import pandas as pd

    from src.daily import archive

    row1 = _candidate_row("005930", "삼성전자", 1)
    row1["snapshot_timestamp"] = pd.Timestamp("2026-08-04 15:19:00", tz="Asia/Seoul")
    archive.upsert_archive_snapshot(pd.DataFrame([row1]), snapshot_date="2026-08-04")

    row2 = _candidate_row("005930", "삼성전자", 1)
    row2["snapshot_timestamp"] = pd.Timestamp("2026-08-04 15:25:00", tz="Asia/Seoul")
    with caplog.at_level(logging.INFO, logger="src.daily.archive"):
        archive.upsert_archive_snapshot(pd.DataFrame([row2]), snapshot_date="2026-08-04")

    assert any("rerun" in rec.message for rec in caplog.records)
