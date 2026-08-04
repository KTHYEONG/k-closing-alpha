"""condition_history_종가매매 백필 파이프라인 단위 테스트.

네트워크/API에 의존하지 않도록 페이크 데이터 원천(FakeDataSource)을 주입한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backfill.backfill_condition_history import (
    COLUMN_ORDER,
    ConditionHistoryBackfiller,
    run_condition_history_backfill,
    save_cleaned,
)

MIN_DATE = "2025-12-29"
RESTORE_START = "2026-05-20"


class FakeDataSource:
    """결정적 데이터를 생성하는 네트워크 없는 페이크 데이터 원천."""

    def __init__(self, name_map: dict[str, tuple[str, str]] | None = None) -> None:
        self.name_map = name_map or {}
        self._stock_code = 1000

    def resolve_stock(self, name: str) -> tuple[str, str] | None:
        return self.name_map.get(str(name).strip())

    def fetch_ohlcv(
        self, code: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        dates = pd.bdate_range(start=start, end=end)
        n = len(dates)
        close = 1000.0 + pd.Series(range(n)) * 100.0
        return pd.DataFrame(
            {
                "date": dates,
                "open": close - 50.0,
                "high": close + 150.0,
                "low": close - 100.0,
                "close": close,
                "volume": np.arange(n) * 1000 + 10_000,
                "trade_value_krw": close * (np.arange(n) * 1000 + 10_000),
                "market_cap_krw": close * 1_000_000.0,
            }
        )

    def fetch_investor_flow(self, code: str, dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(dates, format="%Y%m%d"),
                "inst_netbuy_eok": 100.0,
                "foreign_netbuy_eok": -50.0,
            }
        )

    def fetch_program_flow(self, code: str, dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(dates, format="%Y%m%d"),
                "program_netbuy_eok": 200.0,
            }
        )

    def fetch_index_returns(self, dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": pd.to_datetime(dates, format="%Y%m%d"),
                "kospi_pct": 0.5,
                "kosdaq_pct": 1.2,
            }
        )


def _build_excel(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    df = df.reindex(columns=COLUMN_ORDER)
    path = tmp_path / "condition_history_test.xlsx"
    df.to_excel(path, index=False)
    return path


def _row(
    date: str,
    name: str,
    *,
    code: object = np.nan,
    market: object = np.nan,
    v_kospi: object = np.nan,
    v_kosdaq: object = np.nan,
    volume: object = np.nan,
) -> dict:
    return {
        "스냅샷_날짜": date,
        "종목명": name,
        "종목코드": code,
        "시가": np.nan,
        "고가": np.nan,
        "저가": np.nan,
        "종가": np.nan,
        "전일종가": np.nan,
        "등락률": np.nan,
        "체결강도": np.nan,
        "시장구분": market,
        "시가총액(억)": np.nan,
        "거래대금(억)": np.nan,
        "순위": np.nan,
        "기관_순매수(억)": np.nan,
        "외국인_순매수(억)": np.nan,
        "프로그램_순매수(억)": np.nan,
        "전체종목수": np.nan,
        "평균거래대금(억)": np.nan,
        "KOSPI등락률": np.nan,
        "KOSDAQ등락률": np.nan,
        "(v-kospi)": v_kospi,
        "(v-kosdaq)": v_kosdaq,
        "(거래량)": volume,
    }


def test_filter_pre_20251229_rows(tmp_path: Path) -> None:
    """filter_pre_20251229_rows scenario test."""
    rows = [
        _row("2025-12-20", "구식종목", code="000001"),
        _row("2025-12-26", "구식종목2", code="000002"),
        _row("2026-01-05", "보존종목", code="000003"),
        _row(RESTORE_START, "삼성전자", volume=50_000),
    ]
    excel = _build_excel(tmp_path, rows)
    source = FakeDataSource({"삼성전자": ("005930", "KOSPI")})

    result = run_condition_history_backfill(
        str(excel), data_source=source, save_output=False
    )

    dates = pd.to_datetime(result["스냅샷_날짜"], errors="coerce")
    assert (dates >= pd.Timestamp(MIN_DATE)).all()
    assert len(result) == 2
    assert {"2026-01-05", RESTORE_START} == set(
        pd.to_datetime(result["스냅샷_날짜"]).dt.strftime("%Y-%m-%d")
    )


def test_fill_post_20260520_missing_fields(tmp_path: Path) -> None:
    """fill_post_20260520_missing_fields scenario test."""
    rows = [
        _row(RESTORE_START, "삼성전자", v_kospi=36.0, v_kosdaq=32.0, volume=50_000),
        _row(RESTORE_START, "네이버", v_kospi=36.0, v_kosdaq=32.0, volume=30_000),
    ]
    excel = _build_excel(tmp_path, rows)
    source = FakeDataSource(
        {"삼성전자": ("005930", "KOSPI"), "네이버": ("035420", "KOSPI")}
    )

    result = run_condition_history_backfill(
        str(excel), data_source=source, save_output=False
    )

    assert len(result) == 2
    restored_cols = [
        "종목코드",
        "시가",
        "고가",
        "저가",
        "종가",
        "전일종가",
        "등락률",
        "시장구분",
        "시가총액(억)",
        "거래대금(억)",
        "순위",
        "기관_순매수(억)",
        "외국인_순매수(억)",
        "프로그램_순매수(억)",
        "전체종목수",
        "평균거래대금(억)",
        "KOSPI등락률",
        "KOSDAQ등락률",
    ]
    for col in restored_cols:
        assert result[col].notna().all(), f"컬럼이 비어 있습니다: {col}"

    assert set(result["종목코드"]) == {"005930", "035420"}
    assert set(result["시장구분"]) == {"KOSPI"}
    # 체결강도는 과거 복원 불가 -> NaN 유지
    assert result["체결강도"].isna().all()
    # 기존 보존 컬럼은 그대로 유지
    assert result["(v-kospi)"].eq(36.0).all()
    assert result["(거래량)"].tolist() == [50_000.0, 30_000.0]
    # 일자별 집계
    assert (result["전체종목수"] == 2).all()
    assert sorted(result["순위"]) == [1.0, 2.0]
    assert result["평균거래대금(억)"].notna().all()


def test_prev_close_and_change_pct_restored(tmp_path: Path) -> None:
    rows = [
        _row(RESTORE_START, "삼성전자"),
    ]
    excel = _build_excel(tmp_path, rows)
    source = FakeDataSource({"삼성전자": ("005930", "KOSPI")})

    result = run_condition_history_backfill(
        str(excel), data_source=source, save_output=False
    )

    row = result.iloc[0]
    assert pd.notna(row["전일종가"])
    assert row["전일종가"] != row["종가"]
    assert pd.notna(row["등락률"])


def test_unresolved_name_stays_nan(tmp_path: Path) -> None:
    rows = [_row(RESTORE_START, "미상장미스터리")]
    excel = _build_excel(tmp_path, rows)
    source = FakeDataSource({})

    result = run_condition_history_backfill(
        str(excel), data_source=source, save_output=False
    )

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["종목코드"])
    assert pd.isna(result.iloc[0]["시가"])
    assert pd.notna(result.iloc[0]["전체종목수"])


def test_missing_input_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_condition_history_backfill(
            str(tmp_path / "missing.xlsx"), save_output=False
        )


def test_missing_snapshot_date_column_rejected(tmp_path: Path) -> None:
    df = pd.DataFrame({"종목명": ["삼성전자"], "종가": [1000.0]})
    backfiller = ConditionHistoryBackfiller(FakeDataSource({}))
    with pytest.raises(ValueError, match="스냅샷_날짜"):
        backfiller.filter(df)


def test_save_cleaned_writes_csv_and_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "스냅샷_날짜": pd.to_datetime(["2026-05-20"]),
            "종목명": ["삼성전자"],
            "종목코드": ["005930"],
            "시가": [1000.0],
            "시장구분": ["KOSPI"],
        }
    )
    paths = save_cleaned(df, tmp_path)

    csv_path = tmp_path / "condition_history_cleaned.csv"
    parquet_path = tmp_path / "condition_history_cleaned.parquet"
    assert paths == [csv_path, parquet_path]
    assert csv_path.exists()
    assert parquet_path.exists()

    back = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    assert back["종목코드"].tolist() == ["005930"]
    assert back["스냅샷_날짜"].tolist() == ["2026-05-20"]

    pq = pd.read_parquet(parquet_path)
    assert pq["종목코드"].tolist() == ["005930"]
