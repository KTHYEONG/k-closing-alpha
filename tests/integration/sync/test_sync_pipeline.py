"""sync 모듈 통합 흐름 테스트 (mock API).

KisApiClient 및 Google Sheets를 mock으로 대체하여
foreign/program sync 메인의 시트 업데이트 흐름을 검증합니다.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest

from src.sync import foreign, program


class _FakeWs:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self.updated: list[Any] = []

    def get_all_records(self) -> list[dict[str, Any]]:
        return self._records

    def row_values(self, row: int) -> list[str]:
        if not self._records:
            return []
        return list(self._records[0].keys())

    def batch_update(self, payload: list[Any]) -> None:
        self.updated.extend(payload)


class _FakeManager:
    def __init__(self, records: dict[str, list[dict[str, Any]]]) -> None:
        self._ws = {name: _FakeWs(recs) for name, recs in records.items()}

    def get_all_records(self, sheet_name: str, worksheet_name: str) -> list[dict[str, Any]]:
        return self._ws[worksheet_name].get_all_records()

    def get_spreadsheet(self, sheet_name: str) -> Any:
        class _Sh:
            def worksheet(self, name: str) -> _FakeWs:
                return self._ws[name]

        sh = _Sh()
        sh._ws = self._ws
        return sh


@pytest.fixture
def fake_manager() -> _FakeManager:
    return _FakeManager(
        {
            "Trade": [
                {
                    "(매수날짜)": "2024-01-02",
                    "(종목코드)": "005930",
                    "(기관_순매수)": "",
                    "(외국인_순매수)": "",
                }
            ]
        }
    )


def _install_fakes(
    monkeypatch, manager: _FakeManager, client_class_name: str, tmp_path
) -> None:
    import asyncio

    key_path = tmp_path / "fake_key.json"
    key_path.write_text("{}")

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def ensure_token(self, session: Any) -> None:
            return None

        def create_session(self) -> _FakeClient:
            return self

        @property
        def semaphore(self) -> Any:
            return asyncio.Semaphore(4)

    if client_class_name == "foreign":
        monkeypatch.setattr(foreign, "GOOGLE_KEY_PATH", str(key_path))
        monkeypatch.setattr(foreign, "GSheetClientManager", lambda key: manager)
        monkeypatch.setattr(
            foreign,
            "KisApiClient",
            _FakeClient,
        )
        monkeypatch.setattr(
            foreign,
            "get_investor_trade_daily_async",
            async_fetch_investor,
        )
    else:
        monkeypatch.setattr(program, "GOOGLE_KEY_PATH", str(key_path))
        monkeypatch.setattr(program, "GSheetClientManager", lambda key: manager)
        monkeypatch.setattr(program, "KisApiClient", _FakeClient)
        monkeypatch.setattr(program, "get_program_history_async", async_fetch_program)


async def async_fetch_investor(
    session: Any, client: Any, code: str, start_date: str, end_date: str, **kwargs: Any
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "inst_netbuy": [123],
            "foreign_netbuy": [456],
        }
    )


async def async_fetch_program(
    session: Any, client: Any, code: str, start_date: str, end_date: str, **kwargs: Any
) -> dict[str, float]:
    return {"20240102": 789.0}


def test_foreign_sync_updates_sheet(monkeypatch, fake_manager, tmp_path) -> None:
    _install_fakes(monkeypatch, fake_manager, "foreign", tmp_path)
    asyncio.run(foreign.main())
    assert fake_manager._ws["Trade"].updated, "업데이트된 셀이 있어야 합니다."
    ranges = [u["range"] for u in fake_manager._ws["Trade"].updated]
    assert ranges, "batch_update 호출 흔적이 있어야 합니다."


def test_program_sync_updates_sheet(monkeypatch, fake_manager, tmp_path) -> None:
    fake_manager._ws["Trade"]._records[0]["(프로그램_순매수)"] = ""
    _install_fakes(monkeypatch, fake_manager, "program", tmp_path)
    asyncio.run(program.main())
    assert fake_manager._ws["Trade"].updated
