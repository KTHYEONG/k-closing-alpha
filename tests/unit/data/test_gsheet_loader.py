"""GSheet 클라이언트 mock 테스트.

실제 Google Sheets API에 의존하지 않고 gspread 객체를 mock으로 대체하여
로더 로직을 검증합니다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.data.gsheet_loader import (
    GSheetClientManager,
    append_stocks_to_gsheet,
    load_and_combine_sheets,
    load_data_from_gsheet,
)


class _FakeWorksheet:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.appended: list[list[Any]] = []
        self.batch_payload: list[Any] = []

    def get_all_records(self) -> list[dict[str, Any]]:
        return self._records

    def get_all_values(self) -> list[list[Any]]:
        if not self._records:
            return []
        headers = list(self._records[0].keys())
        rows = [headers]
        rows.extend([rec.get(h, "") for h in headers] for rec in self._records)
        return rows

    def append_rows(self, rows: list[list[Any]]) -> None:
        self.appended.extend(rows)

    def batch_update(self, payload: list[Any]) -> None:
        self.batch_payload.extend(payload)


class _FakeSpreadsheet:
    def __init__(self, records: dict[str, list[dict[str, Any]]]) -> None:
        self._worksheets = {
            name: _FakeWorksheet(recs) for name, recs in records.items()
        }

    def worksheet(self, name: str) -> _FakeWorksheet:
        return self._worksheets[name]


class _FakeClient:
    def __init__(self, spreadsheet: _FakeSpreadsheet) -> None:
        self._spreadsheet = spreadsheet

    def open(self, name: str) -> _FakeSpreadsheet:
        return self._spreadsheet


@pytest.fixture
def fake_gsheet_client(monkeypatch) -> _FakeSpreadsheet:
    records = {
        "Trade": [
            {"(매수날짜)": "2024-01-02", "(종목코드)": "005930", "종목명": "삼성전자"},
            {"(매수날짜)": "2024-01-03", "(종목코드)": "000660", "종목명": "SK하이닉스"},
        ],
        "Trade2": [],
    }
    spreadsheet = _FakeSpreadsheet(records)
    # append_stocks_to_gsheet는 정확히 "종목코드"/"종목명" 헤더를 찾으므로
    # 추가 테스트용 시트는 일반 헤더를 사용합니다.
    spreadsheet._worksheets["Trade"] = _FakeWorksheet(
        [
            {"매수날짜": "2024-01-02", "종목코드": "005930", "종목명": "삼성전자"},
            {"매수날짜": "2024-01-03", "종목코드": "000660", "종목명": "SK하이닉스"},
        ]
    )
    monkeypatch.setattr(
        GSheetClientManager, "__init__", lambda self, key_path: None
    )
    monkeypatch.setattr(
        GSheetClientManager, "get_spreadsheet", lambda self, name: spreadsheet
    )
    monkeypatch.setattr(
        GSheetClientManager,
        "get_all_records",
        lambda self, sheet, ws_name, *args, **kwargs: spreadsheet.worksheet(ws_name).get_all_records(),
    )
    monkeypatch.setattr(
        GSheetClientManager,
        "get_all_values",
        lambda self, sheet, ws_name: (
            spreadsheet.worksheet(ws_name).get_all_values(),
            spreadsheet.worksheet(ws_name),
        ),
    )
    monkeypatch.setattr(
        GSheetClientManager,
        "append_rows",
        lambda self, ws, rows: ws.append_rows(rows),
    )
    return spreadsheet


def test_load_and_combine_sheets(fake_gsheet_client, tmp_path) -> None:
    key_path = tmp_path / "dummy.json"
    key_path.write_text("{}")
    df = load_and_combine_sheets(str(key_path), "Stock", ["Trade", "Trade2"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_data_from_gsheet_missing_key(monkeypatch, tmp_path) -> None:
    df = load_data_from_gsheet(str(tmp_path / "missing.json"), "Stock", "Trade")
    assert df is None


def test_append_stocks_to_gsheet_dedup(fake_gsheet_client, tmp_path) -> None:
    ws = fake_gsheet_client.worksheet("Trade")
    key_path = tmp_path / "dummy.json"
    key_path.write_text("{}")

    append_stocks_to_gsheet(
        str(key_path),
        "Stock",
        "Trade",
        [
            {"종목코드": "005930", "종목명": "삼성전자"},  # 이미 존재 → 제외
            {"종목코드": "123456", "종목명": "신규종목"},
        ],
    )
    assert ws.appended == [["", "123456", "신규종목"]]
