from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import intraday_store


def test_intraday_store_write_and_range_read_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    df_in = pd.DataFrame({"종목코드": ["005930"], "stck_prpr": [70000]})
    rows_written = intraday_store.write_intraday_partition(df_in, 1, "2026-09-03", "regular")
    assert rows_written == 1

    df_out_of_range = pd.DataFrame({"종목코드": ["000660"], "stck_prpr": [180000]})
    intraday_store.write_intraday_partition(df_out_of_range, 1, "2026-08-01", "regular")

    result = intraday_store.read_intraday_range(1, "2026-09-01", "2026-09-30", session="regular")

    assert len(result) == 1
    assert result.iloc[0]["종목코드"] == "005930"


def test_intraday_store_write_empty_df_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(intraday_store.settings, "HISTORY_DIR", tmp_path)

    rows_written = intraday_store.write_intraday_partition(pd.DataFrame(), 1, "2026-09-03", "regular")

    assert rows_written == 0
