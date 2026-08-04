"""일일 데이터 파이프라인 표준 저장 및 아카이브 CSV 우선 인식 테스트.

CSV_EXPORT 시나리오: 수집된 조건검색 DataFrame 이 STANDARD_COLUMN_ORDER 순서 그대로
utf-8-sig CSV 로 저장되고, 읽어온 종목코드는 6자리 zero-fill 문자열이어야 한다.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

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


# ---------------------------------------------------------
# 표준 CSV 저장 (CSV_EXPORT_ORDER_VERIFICATION 시나리오)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# collect 헬퍼 함수
# ---------------------------------------------------------
def test_safe_float_converts_values() -> None:
    assert collect.safe_float(None) == 0.0
    assert collect.safe_float("1,234.5") == 1234.5
    assert collect.safe_float("abc") == 0.0
    assert collect.safe_float(7) == 7.0


def test_parse_market_index_rate_returns_zero_on_missing() -> None:
    assert collect.parse_market_index_rate(None) == 0.0
    assert collect.parse_market_index_rate({"rt_cd": "1"}) == 0.0
    assert collect.parse_market_index_rate({"rt_cd": "0", "output1": None}) == 0.0


def test_parse_market_index_rate_uses_rate_and_fallback() -> None:
    assert (
        collect.parse_market_index_rate(
            {"rt_cd": "0", "output1": {"bstp_nmix_prdy_ctrt": "1.25"}}
        )
        == 1.25
    )
    fallback = collect.parse_market_index_rate(
        {"rt_cd": "0", "output1": {"bstp_nmix_prpr": "100", "bstp_nmix_prdy_vrss": "2"}}
    )
    assert fallback == pytest.approx(2.04)


def test_validate_hts_id_raises_on_placeholder() -> None:
    with (
        patch.object(collect, "HTS_ID", "여기에 HTS ID를 입력"),
        pytest.raises(RuntimeError),
    ):
        collect._validate_hts_id()
    with patch.object(collect, "HTS_ID", "real-hts"):
        collect._validate_hts_id()  # should not raise


# ---------------------------------------------------------
# fetch_single_stock 시나리오 우선순위
# ---------------------------------------------------------
def _fake_client() -> SimpleNamespace:
    return SimpleNamespace(
        get_current_price=AsyncMock(
            return_value={
                "rt_cd": "0",
                "output": {
                    "stck_prpr": "10500",
                    "stck_oprc": "10000",
                    "stck_hgpr": "11000",
                    "stck_lwpr": "9000",
                    "acml_vol": "100000",
                    "prdy_ctrt": "5.00",
                    "lstn_stcn": "1000000",
                    "rprs_mrkt_kor_name": "KOSPI",
                    "hts_avls": "100000",
                    "acml_tr_pbmn": "12000000000",
                },
            }
        ),
        get_trade_strength=AsyncMock(
            return_value={"rt_cd": "0", "output": [{"tday_rltv": "120"}]}
        ),
        get_investor_trend_estimate=AsyncMock(
            return_value={
                "rt_cd": "0",
                "output2": [{"frgn_fake_ntby_qty": "10000", "orgn_fake_ntby_qty": "5000"}],
            }
        ),
        get_program_net_buy=AsyncMock(
            return_value={
                "rt_cd": "0",
                "output": [{"whol_smtn_ntby_tr_pbmn": "500000000"}],
            }
        ),
    )


async def _run_fetch_single_stock(client, **scenario_sets) -> tuple[dict, list[str]]:
    sem = asyncio.Semaphore(2)
    stock = {"code": "005930", "name": "삼성전자", "price": "10000", "chgrate": "1.0"}
    with patch(
        "src.api.kis_client.calculate_all_moving_averages",
        new=AsyncMock(
            return_value=(
                {5: 10000, 10: 10000, 20: 10000},
                (10000.0, True, 300),
                (10000.0, True),
                (10000.0, True),
            )
        ),
    ):
        return await collect.fetch_single_stock(
            0, stock, 1, sem, client, None, **scenario_sets
        )


def test_fetch_single_stock_sangdda_scenario() -> None:
    result, failed = asyncio.run(
        _run_fetch_single_stock(_fake_client(), upper_limit_stock_codes={"005930"})
    )
    assert failed == []
    assert result["시나리오"] == "상따"
    assert result["종목코드"] == "005930"


def test_fetch_single_stock_default_scenario_volume_surge() -> None:
    result, failed = asyncio.run(_run_fetch_single_stock(_fake_client()))
    assert failed == []
    assert result["시나리오"] == "거래량 폭증"


def test_fetch_all_stock_data_aggregates_and_reports_failures() -> None:
    async def _fake_single_stock(*args, **kwargs) -> tuple[dict, list[str]]:
        return ({"종목명": "AAA", "종목코드": "005930"}, ["체결강도"])

    with patch.object(collect, "fetch_single_stock", side_effect=_fake_single_stock):
        results, failed = asyncio.run(
            collect.fetch_all_stock_data(
                [{"code": "005930", "name": "AAA"}],
                None,
                None,
                set(),
                set(),
                set(),
                set(),
                set(),
            )
        )
    assert len(results) == 1
    assert results[0]["종목명"] == "AAA"
    assert failed == [("AAA", "005930", ["체결강도"])]


# ---------------------------------------------------------
# archive CSV 우선 인식
# ---------------------------------------------------------
def test_archive_condition_prefers_csv_over_xlsx(tmp_path: Path) -> None:
    """아카이브가 condition_*.csv 를 우선 인식하고 스냅샷 날짜를 삽입한다."""
    import src.daily.archive as archive

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    csv_file = data_dir / "condition_종가매매.csv"
    csv_file.write_text(
        "(종목코드),종목명,(차트통과),(시나리오)\n000001,AAA,1,신고가\n",
        encoding="utf-8-sig",
    )

    with (
        patch.object(archive.settings, "DATA_DIR", data_dir),
        patch.object(archive.settings, "HISTORY_DIR", tmp_path / "history"),
        patch.object(
            archive.settings, "HISTORY_DB_PATH", tmp_path / "history.db"
        ),
        patch.object(
            archive.settings, "HISTORY_CSV_PATH", tmp_path / "history.csv"
        ),
        patch.object(archive.settings, "CONDITION_CSV_PATH", csv_file),
        patch.object(archive, "import_csv_history_if_needed"),
        patch.object(archive, "upsert_history") as upsert_mock,
    ):
        archive.main()

    upsert_mock.assert_called_once()
    df = upsert_mock.call_args.args[0]
    assert "스냅샷_날짜" in df.columns
    assert "(종목코드)" in df.columns
    assert df["(종목코드)"].astype(str).str.zfill(6).tolist() == ["000001"]


def test_archive_condition_skips_when_csv_and_xlsx_missing(tmp_path: Path) -> None:
    import src.daily.archive as archive

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)

    with (
        patch.object(archive.settings, "DATA_DIR", data_dir),
        patch.object(archive.settings, "HISTORY_DIR", tmp_path / "history"),
        patch.object(
            archive.settings, "HISTORY_DB_PATH", tmp_path / "history.db"
        ),
        patch.object(
            archive.settings, "HISTORY_CSV_PATH", tmp_path / "history.csv"
        ),
        patch.object(
            archive.settings, "CONDITION_CSV_PATH", data_dir / "condition_종가매매.csv"
        ),
        patch.object(archive, "import_csv_history_if_needed"),
        patch.object(archive, "upsert_history") as upsert_mock,
    ):
        archive.main()

    upsert_mock.assert_not_called()


def test_upsert_history_stores_and_dedups(tmp_path: Path) -> None:
    import src.daily.archive as archive

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
    import src.daily.archive as archive

    with pytest.raises(FileNotFoundError):
        archive.fetch_date_rows("2026-08-04", str(tmp_path / "nope.db"))


def test_import_csv_history_if_needed_migrates_legacy_csv(tmp_path: Path) -> None:
    import src.daily.archive as archive

    history_csv = tmp_path / "history.csv"
    history_csv.write_text(
        "스냅샷_날짜,종목코드\n2026-08-01,000002\n2026-08-02,000003\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "history.db"

    with patch("src.data.parquet_loader.upsert_condition_parquet"):
        archive.import_csv_history_if_needed(str(history_csv), str(db_path))
        # 두 번째 호출은 누락 날짜가 없으므로 무시
        archive.import_csv_history_if_needed(str(history_csv), str(db_path))

    rows = archive.fetch_date_rows("2026-08-01", str(db_path))
    assert rows["종목코드"].astype(str).str.zfill(6).tolist() == ["000002"]


# ---------------------------------------------------------
# collect.main() → upsert_archive_snapshot wiring
# ---------------------------------------------------------
class _FakeKisClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def ensure_token(self, session: object) -> None:
        return None

    async def get_market_index_rate(self, session: object, code: str) -> dict:
        return {"rt_cd": "0", "output1": {"bstp_nmix_prdy_ctrt": "1.00"}}

    async def get_condition_list(self, session: object) -> dict:
        return {
            "rt_cd": "0",
            "output2": [
                {"condition_nm": collect.settings.TARGET_CONDITION_NAME, "seq": 1}
            ],
        }

    async def get_condition_result(self, session: object, seq: int) -> dict:
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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
