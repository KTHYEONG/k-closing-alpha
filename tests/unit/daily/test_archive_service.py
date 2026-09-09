"""일일 아카이브(archive) 서비스 단위 테스트: 변환·에러·임시 경로."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.daily import archive






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


