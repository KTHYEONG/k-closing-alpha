"""일일 collect -> archive CSV/Parquet 영속화 및 collect.main() wiring 통합 테스트."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd

from src.daily import collect
from src.daily.collect import STANDARD_COLUMN_ORDER, save_collected_condition_data


def _standard_columns_df() -> pd.DataFrame:
    """표준 열 이름(순서는 뒤섞임) + 비표준 여분 열을 포함한 샘플 DataFrame."""
    row = {
        "시나리오": ["신고가", "거래량 폭증"],
        "상장일수": [300, 500],
        "종목명": ["AAA", "BBB"],
        "종목코드": [1, 2],
        "시가": [10000, 20000],
        "고가": [11000, 21000],
        "저가": [9000, 19000],
        "종가": [10500, 20500],
        "전일종가": [10000, 20000],
        "시가총액": [5000.0, 8000.0],
        "거래대금": [120.0, 250.0],
        "등락률": [5.0, 2.5],
        "선정순위": [1, 2],
        "기관_순매수": [10.0, 20.0],
        "외국인_순매수": [30.0, 40.0],
        "프로그램_순매수": [5.0, 6.0],
        "체결강도": [120.0, 110.0],
        "시장구분": ["KOSPI", "KOSDAQ"],
        "총_종목수": [2, 2],
        "평균_거래대금": [185.0, 185.0],
        "kospi": [0.5, 0.5],
        "kosdaq": [0.3, 0.3],
        "v_kospi": [12.5, 12.5],
        "v_kosdaq": [15.2, 15.2],
        "거래량": [100000, 200000],
        "extra_col": ["x", "y"],
    }
    return pd.DataFrame(row)


def test_csv_export_order_verification(tmp_path: Path) -> None:
    """CSV_EXPORT_ORDER_VERIFICATION 시나리오: 저장된 CSV 는 STANDARD_COLUMN_ORDER 와 정확히 일치하고 종목코드는 zero-fill 문자열이다."""
    csv_path = tmp_path / "condition_종가매매.csv"

    result = save_collected_condition_data(_standard_columns_df(), csv_path)

    assert result == csv_path
    assert csv_path.exists()

    # utf-8-sig BOM 확인 (Excel/Google Sheets 한글 호환)
    with open(csv_path, "rb") as fh:
        assert fh.read(3) == b"\xef\xbb\xbf"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    assert df.columns.tolist() == list(STANDARD_COLUMN_ORDER)
    assert "extra_col" not in df.columns

    # 파일에 저장된 종목코드는 6자리 zero-fill 문자열
    raw_df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    assert raw_df["종목코드"].tolist() == ["000001", "000002"]
    assert (raw_df["종목코드"].str.len() == 6).all()


def test_save_collected_condition_data_creates_sibling_parquet(tmp_path: Path) -> None:
    csv_path = tmp_path / "condition_종가매매.csv"

    save_collected_condition_data(_standard_columns_df(), csv_path)

    parquet_path = csv_path.with_suffix(".parquet")
    assert parquet_path.exists()
    df = pd.read_parquet(parquet_path)
    assert df.columns.tolist() == list(STANDARD_COLUMN_ORDER)
    assert df["종목코드"].astype(str).str.zfill(6).tolist() == ["000001", "000002"]


def test_save_collected_condition_data_returns_path(tmp_path: Path) -> None:
    csv_path = tmp_path / "nested" / "condition.csv"

    result = save_collected_condition_data(_standard_columns_df(), csv_path)

    assert result == csv_path
    assert csv_path.parent.exists()


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


async def _fake_fetch_all_stock_data(*args, **kwargs) -> tuple[list[dict], list]:
    return (
        [
            {
                "종목명": "삼성전자",
                "종목코드": "005930",
                "시가": 1000,
                "고가": 1100,
                "저가": 900,
                "종가": 1050,
                "전일종가": 1000,
                "시장구분": "KOSPI",
                "시가총액": 5000.0,
                "거래대금": 120.0,
                "체결강도": 120.0,
                "등락률": 5.0,
                "선정순위": 1,
                "기관_순매수": 1.0,
                "외국인_순매수": 2.0,
                "프로그램_순매수": 0.5,
                "시나리오": "거래량 폭증",
                "거래량": 100000,
            }
        ],
        [],
    )


def test_collect_main_saves_csv_without_auto_archive(
    monkeypatch, tmp_path: Path
) -> None:
    """collect.main() 은 자동 아카이브 없이 지정된 CSV/Parquet 에 수집 데이터를 저장한다."""
    history_dir = tmp_path / "history"
    csv_path = tmp_path / "daily" / "daily_stocks.csv"
    monkeypatch.setattr(collect.settings, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(
        collect.settings, "HISTORY_PARQUET_PATH", history_dir / "archive.parquet"
    )
    monkeypatch.setattr(
        collect.settings, "HISTORY_DB_PATH", history_dir / "archive.db"
    )
    monkeypatch.setattr(collect.settings, "CONDITION_CSV_PATH", csv_path)

    monkeypatch.setattr(collect, "HTS_ID", "TEST")
    monkeypatch.setattr(collect, "KisApiClient", _FakeKisClient)
    monkeypatch.setattr(collect, "fetch_all_stock_data", _fake_fetch_all_stock_data)
    monkeypatch.setattr(collect.aiohttp, "ClientSession", lambda **kw: _FakeSession())
    monkeypatch.setattr(collect, "load_theme_from_db", lambda: {})
    monkeypatch.setattr(collect, "append_stocks_to_gsheet", lambda *a, **k: None)

    asyncio.run(collect.main())

    assert csv_path.exists()
    assert not (history_dir / "archive.db").exists()
    assert not (history_dir / "archive.parquet").exists()
