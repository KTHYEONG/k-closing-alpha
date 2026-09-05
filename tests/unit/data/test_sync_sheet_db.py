"""sync_sheet_db: 매매일지 전용 동기화 (테마 시트 폐지 후) 검증."""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.data import sync_sheet_db as mod


def test_filter_valid_rows_drops_blank_code_rows() -> None:
    df = pd.DataFrame({"종목코드": ["005930", "", "  ", "000660"], "종목명": ["삼성전자", "빈행", "빈행2", "SK하이닉스"]})

    out = mod.filter_valid_rows(df)

    assert list(out["종목코드"]) == ["005930", "000660"]


def test_sync_trade_log_persists_to_sqlite_and_parquet(monkeypatch) -> None:
    raw = pd.DataFrame(
        {
            "(매수날짜)": ["2026-09-01"],
            "종목코드": ["5930"],
            "(v-kospi)": [12.3],
            "(v-kosdaq)": [45.6],
        }
    )
    monkeypatch.setattr(mod, "load_and_combine_sheets", lambda *a, **k: raw)

    saved: dict = {}
    monkeypatch.setattr(
        "src.data.parquet_loader.save_trade_log_to_parquet",
        lambda df: saved.__setitem__("df", df),
    )

    conn = sqlite3.connect(":memory:")
    mod.sync_trade_log(conn)

    stored = pd.read_sql("SELECT * FROM table_trade_log", conn)
    assert stored.loc[0, "종목코드"] == "005930"
    assert stored.loc[0, "매수날짜"] == "2026-09-01"
    assert "v_kospi" in stored.columns and "v_kosdaq" in stored.columns
    assert saved["df"]["종목코드"].iloc[0] == "005930"


def test_sync_gsheet_data_only_syncs_trade_log(monkeypatch) -> None:
    """코드_테마_DB 시트 폐지 이후 sync_gsheet_data는 매매일지만 동기화한다."""
    calls: list[str] = []
    monkeypatch.setattr(mod, "sync_trade_log", lambda conn: calls.append("trade_log"))
    monkeypatch.setattr(mod, "DB_PATH", ":memory:")

    real_connect = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda path: real_connect(":memory:"))
    monkeypatch.setattr("os.makedirs", lambda *a, **k: None)

    mod.sync_gsheet_data()

    assert calls == ["trade_log"]
    assert not hasattr(mod, "sync_theme_only")
