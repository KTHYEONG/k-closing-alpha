"""일일 collect -> archive CSV/Parquet 영속화 및 collect.main() wiring 통합 테스트."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from src.daily import collect


class _FakeKisClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def ensure_token(self, session: object) -> None:
        return None

    async def get_market_index_rate(self, session: object, code: str) -> dict:
        return {"rt_cd": "0", "output1": {"bstp_nmix_prdy_ctrt": "1.00"}}

    async def get_condition_list(self, session: object) -> dict:
        cond_names = [
            collect.settings.TARGET_CONDITION_NAME,
            collect.settings.OVERHEATED_CONDITION_NAME,
            collect.settings.NEW_HIGH_CONDITION_NAME,
            collect.settings.NEAR_NEW_HIGH_CONDITION_NAME,
        ]
        return {
            "rt_cd": "0",
            "output2": [
                {"condition_nm": name, "seq": idx + 1}
                for idx, name in enumerate(cond_names)
            ],
        }

    async def get_condition_result(self, session: object, seq: int) -> dict:
        if seq == 4:  # 신고가 근접 조건: 실패 응답 경로 검증
            return {"rt_cd": "9", "msg1": "조회 실패"}
        return {
            "rt_cd": "0",
            "output2": [
                {"code": "005930", "name": "삼성전자", "price": "1000", "chgrate": "1.00"}
            ],
        }


class _FakeSession:
    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False




def test_collect_main_persists_wide_snapshot_to_store_without_csv(monkeypatch, tmp_path) -> None:
    """collect.main() 은 CSV 없이 wide 스냅샷을 저장소에 직접 기록한다."""
    import asyncio

    from src.daily import collect

    monkeypatch.setattr(collect, "HTS_ID", "TEST")
    monkeypatch.setattr(collect, "KisApiClient", _FakeKisClient)
    monkeypatch.setattr(collect.aiohttp, "ClientSession", lambda **kw: _FakeSession())

    async def _fake_scan(client, session, **kwargs):
        return [
            {"code": "000001", "name": "AAA", "price": "18000", "chgrate": "5.0"},
            {"code": "000004", "name": "DDD", "price": "30000", "chgrate": "5.0"},
        ]

    async def _fake_fetch_all(stock_list, client, session):
        rows = [
            {"종목명": "AAA", "종목코드": "000001", "시장구분": "KOSPI", "시가": 17900.0,
             "고가": 18100.0, "저가": 17800.0, "종가": 18000.0, "전일종가": 17142.86,
             "거래량": 1_000_000, "거래대금": 500.0, "시가총액": 3000.0,
             "기관_순매수": 10.0, "외국인_순매수": 5.0, "등락률": 5.0, "현재가_실패": False},
            {"종목명": "DDD", "종목코드": "000004", "시장구분": "KOSPI", "시가": 29900.0,
             "고가": 30100.0, "저가": 29800.0, "종가": 30000.0, "전일종가": 28571.43,
             "거래량": 800_000, "거래대금": 400.0, "시가총액": 5000.0,
             "기관_순매수": 8.0, "외국인_순매수": 6.0, "등락률": 5.0, "현재가_실패": False},
        ]
        return rows, []

    monkeypatch.setattr(collect, "fetch_candidate_stock_list", _fake_scan)
    monkeypatch.setattr(collect, "fetch_all_stock_data", _fake_fetch_all)

    captured = {}

    def _fake_upsert(df, snapshot_date=None):
        captured["df"] = df.copy()
        return len(df)

    monkeypatch.setattr(collect.archive, "upsert_archive_snapshot", _fake_upsert)

    # When
    asyncio.run(collect.main(force=True))

    # Then: the spreadsheet-era CSV path no longer exists as a settings surface at all
    assert not hasattr(collect.settings, "CONDITION_CSV_PATH")

    # Then: the wide cross-section is stored, rejection recorded as a flag
    stored = captured["df"]
    assert stored["종목코드"].tolist() == ["000001", "000004"]
    assert stored["admitted"].tolist() == [True, False]


def test_collect_main_returns_early_when_scan_empty(monkeypatch, tmp_path, caplog) -> None:
    """collect.main() 은 자동 스캔이 비면 upsert 없이 경고와 함께 조기 반환한다."""
    import asyncio
    import logging

    from src.daily import collect

    monkeypatch.setattr(collect, "HTS_ID", "TEST")
    monkeypatch.setattr(collect, "KisApiClient", _FakeKisClient)
    monkeypatch.setattr(collect.aiohttp, "ClientSession", lambda **kw: _FakeSession())

    # Given: the ranking scan finds nothing inside the band today
    async def _empty_scan(client, session, **kwargs):
        rows: list[dict] = []
        return rows

    monkeypatch.setattr(collect, "fetch_candidate_stock_list", _empty_scan)

    upsert_calls: list[object] = []
    monkeypatch.setattr(
        collect.archive, "upsert_archive_snapshot",
        lambda df, snapshot_date=None: upsert_calls.append(df) or len(df),
    )

    # When
    with caplog.at_level(logging.INFO, logger=collect.logger.name):
        asyncio.run(collect.main(force=True))

    # Then: no persistence attempt, and the emptiness is surfaced
    assert upsert_calls == []
    assert any("자동 스캔 후보가 없습니다" in rec.message for rec in caplog.records)


def test_collect_main_marks_index_failed_and_nans_kospi_kosdaq_on_index_failure(monkeypatch, tmp_path) -> None:
    """collect.main() 은 지수 조회 실패 시에도 계속 진행하되 kospi/kosdaq을 NaN, 지수_실패=True로 기록한다."""
    import asyncio

    from src.daily import collect

    class _FailingIndexClient(_FakeKisClient):
        async def get_market_index_rate(self, session: object, code: str) -> dict:
            return {"rt_cd": "9", "msg1": "index unavailable"}

    monkeypatch.setattr(collect, "HTS_ID", "TEST")
    monkeypatch.setattr(collect, "KisApiClient", _FailingIndexClient)
    monkeypatch.setattr(collect.aiohttp, "ClientSession", lambda **kw: _FakeSession())

    async def _fake_scan(client, session, **kwargs):
        return [{"code": "000001", "name": "AAA", "price": "18000", "chgrate": "5.0"}]

    async def _fake_fetch_all(stock_list, client, session):
        rows = [
            {"종목명": "AAA", "종목코드": "000001", "시장구분": "KOSPI", "시가": 17900.0,
             "고가": 18100.0, "저가": 17800.0, "종가": 18000.0, "전일종가": 17142.86,
             "거래량": 1_000_000, "거래대금": 500.0, "시가총액": 3000.0,
             "기관_순매수": 10.0, "외국인_순매수": 5.0, "등락률": 5.0, "현재가_실패": False},
        ]
        return rows, []

    monkeypatch.setattr(collect, "fetch_candidate_stock_list", _fake_scan)
    monkeypatch.setattr(collect, "fetch_all_stock_data", _fake_fetch_all)

    captured = {}

    def _fake_upsert(df, snapshot_date=None):
        captured["df"] = df.copy()
        return len(df)

    monkeypatch.setattr(collect.archive, "upsert_archive_snapshot", _fake_upsert)

    # When
    asyncio.run(collect.main(force=True))

    # Then: index failure is explicit, not silently coerced to 0.0
    import math

    stored = captured["df"]
    assert math.isnan(stored["kospi"].iloc[0])
    assert math.isnan(stored["kosdaq"].iloc[0])
    assert stored["지수_실패"].iloc[0] == True  # noqa: E712


def test_collect_main_raises_and_skips_persist_when_coverage_gate_fails(monkeypatch, tmp_path) -> None:
    import asyncio

    import pytest

    from src.daily import collect

    monkeypatch.setattr(collect, "HTS_ID", "TEST")
    monkeypatch.setattr(collect, "KisApiClient", _FakeKisClient)
    monkeypatch.setattr(collect.aiohttp, "ClientSession", lambda **kw: _FakeSession())

    async def _fake_scan(client, session, **kwargs):
        return [
            {"code": "000001", "name": "AAA", "price": "18000", "chgrate": "5.0"},
            {"code": "000009", "name": "ETN", "price": "0", "chgrate": "0.0"},
        ]

    async def _fake_fetch_all(stock_list, client, session):
        # Given: 1 healthy row + 1 degenerate all-zero 'success' row (50% degraded,
        # far below the 99% default threshold)
        rows = [
            {"종목명": "AAA", "종목코드": "000001", "시장구분": "KOSPI", "시가": 17900.0,
             "고가": 18100.0, "저가": 17800.0, "종가": 18000.0, "전일종가": 17142.86,
             "거래량": 1_000_000, "거래대금": 500.0, "시가총액": 3000.0,
             "기관_순매수": 10.0, "외국인_순매수": 5.0, "등락률": 5.0, "현재가_실패": False},
            {"종목명": "ETN", "종목코드": "000009", "시장구분": "KOSDAQ", "시가": 0.0,
             "고가": 0.0, "저가": 0.0, "종가": 0.0, "전일종가": 1.0,
             "거래량": 0.0, "거래대금": 0.0, "시가총액": 0.0,
             "기관_순매수": 0.0, "외국인_순매수": 0.0, "등락률": 0.0, "현재가_실패": False},
        ]
        return rows, []

    monkeypatch.setattr(collect, "fetch_candidate_stock_list", _fake_scan)
    monkeypatch.setattr(collect, "fetch_all_stock_data", _fake_fetch_all)

    upsert_calls: list[object] = []
    monkeypatch.setattr(
        collect.archive, "upsert_archive_snapshot",
        lambda df, snapshot_date=None: upsert_calls.append(df) or len(df),
    )

    # When / Then: the coverage gate raises before any persistence is attempted
    with pytest.raises(ValueError, match="real-time collection coverage"):
        asyncio.run(collect.main(force=True))
    assert upsert_calls == []


def test_collect_main_persists_price_anomaly_column_as_all_false_for_healthy_snapshot(monkeypatch, tmp_path) -> None:
    import asyncio

    from src.daily import collect

    monkeypatch.setattr(collect, "HTS_ID", "TEST")
    monkeypatch.setattr(collect, "KisApiClient", _FakeKisClient)
    monkeypatch.setattr(collect.aiohttp, "ClientSession", lambda **kw: _FakeSession())

    async def _fake_scan(client, session, **kwargs):
        return [
            {"code": "000001", "name": "AAA", "price": "18000", "chgrate": "5.0"},
            {"code": "000004", "name": "DDD", "price": "30000", "chgrate": "5.0"},
        ]

    async def _fake_fetch_all(stock_list, client, session):
        rows = [
            {"종목명": "AAA", "종목코드": "000001", "시장구분": "KOSPI", "시가": 17900.0,
             "고가": 18100.0, "저가": 17800.0, "종가": 18000.0, "전일종가": 17142.86,
             "거래량": 1_000_000, "거래대금": 500.0, "시가총액": 3000.0,
             "기관_순매수": 10.0, "외국인_순매수": 5.0, "등락률": 5.0, "현재가_실패": False},
            {"종목명": "DDD", "종목코드": "000004", "시장구분": "KOSPI", "시가": 29900.0,
             "고가": 30100.0, "저가": 29800.0, "종가": 30000.0, "전일종가": 28571.43,
             "거래량": 800_000, "거래대금": 400.0, "시가총액": 5000.0,
             "기관_순매수": 8.0, "외국인_순매수": 6.0, "등락률": 5.0, "현재가_실패": False},
        ]
        return rows, []

    monkeypatch.setattr(collect, "fetch_candidate_stock_list", _fake_scan)
    monkeypatch.setattr(collect, "fetch_all_stock_data", _fake_fetch_all)

    captured = {}

    def _fake_upsert(df, snapshot_date=None):
        captured["df"] = df.copy()
        return len(df)

    monkeypatch.setattr(collect.archive, "upsert_archive_snapshot", _fake_upsert)

    # When
    asyncio.run(collect.main(force=True))

    # Then: the gate passed (no raise), persistence happened, and the new column is present
    stored = captured["df"]
    assert stored["가격_비정상"].tolist() == [False, False]
