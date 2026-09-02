from pathlib import Path

import pandas as pd

from src.backfill.altdata import disclosure
from src.backfill.altdata.config import AltDataFetchConfig


def test_aggregate_rows_categorizes_and_flags_material() -> None:
    rows = [
        {"stock_code": "005930", "report_nm": "단일판매ㆍ공급계약체결", "rcept_dt": "20240102"},
        {"stock_code": "005930", "report_nm": "전환사채권발행결정", "rcept_dt": "20240102"},
        {"stock_code": "000660", "report_nm": "기업설명회(IR)개최", "rcept_dt": "20240102"},
        {"stock_code": "", "report_nm": "기타경영사항", "rcept_dt": "20240102"},
    ]
    out = disclosure._aggregate_rows(rows)
    row = out[out["symbol"] == "005930"].iloc[0]
    assert row["n_supply_contract"] == 1
    assert row["n_cb_bw"] == 1
    assert row["n_total"] == 2
    assert bool(row["has_material"]) is True
    assert bool(out[out["symbol"] == "000660"].iloc[0]["has_material"]) is False
    assert (out["symbol"] == "").sum() == 0


def test_collect_disclosures_uses_pblntf_ty_and_flushes_per_window(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def _fake_window(cfg, pblntf_ty, start_ymd, end_ymd, corp_to_stock):  # noqa: ANN001, ANN202
        calls.append((pblntf_ty, start_ymd, end_ymd))
        return [{"stock_code": "005930", "report_nm": "유상증자결정", "rcept_dt": f"{start_ymd}"}]

    monkeypatch.setattr(disclosure, "_fetch_disclosure_window", _fake_window)
    flushed: list[pd.DataFrame] = []
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-12-31"),
        out_dir=Path("x"), dart_api_key="k", retries=1, retry_sleep_sec=0.0,
    )
    ret = disclosure.collect_disclosures(
        cfg, pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}), on_window=flushed.append
    )
    assert ret.empty  # on_window 모드 → 누적 안 함
    assert {c[0] for c in calls} == {"B", "I"}  # corp_cls 분할 대신 공시유형 필터
    for _ty, bgn, end in calls:
        assert pd.Timestamp(end) - pd.Timestamp(bgn) <= pd.Timedelta(days=92)
    assert len(flushed) >= 4  # 1년 → 창별 flush
    assert "n_rights_offering" in flushed[0].columns


def test_collect_disclosures_skips_fully_covered_windows(monkeypatch) -> None:
    fetched_windows: list[str] = []

    def _fake_window(cfg, pblntf_ty, start_ymd, end_ymd, corp_to_stock):  # noqa: ANN001, ANN202
        fetched_windows.append(start_ymd)
        return []

    monkeypatch.setattr(disclosure, "_fetch_disclosure_window", _fake_window)
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2020-12-31"),
        out_dir=Path("x"), dart_api_key="k", retries=1, retry_sleep_sec=0.0,
    )
    covered = {d.normalize() for d in pd.bdate_range("2020-01-01", "2020-03-21")}
    disclosure.collect_disclosures(
        cfg, pd.DataFrame({"corp_code": [], "stock_code": [], "corp_name": []}),
        on_window=lambda _df: None, covered_dates=covered,
    )
    # 첫 80일 창(2020-01-01~)은 전부 covered → 조회 안 함
    assert "20200101" not in fetched_windows
    assert any(w >= "20200322" for w in fetched_windows)


import io
import zipfile

import pytest


def test_download_corp_code_map_parses_zip(monkeypatch) -> None:
    xml = (
        "<result><list><corp_code>00126380</corp_code><corp_name>삼성전자</corp_name>"
        "<stock_code>005930</stock_code><modify_date>20240101</modify_date></list>"
        "<list><corp_code>00999999</corp_code><corp_name>비상장</corp_name>"
        "<stock_code> </stock_code><modify_date>20240101</modify_date></list></result>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("CORPCODE.xml", xml)
    payload = buf.getvalue()

    class _Resp:
        status_code = 200
        content = payload
        headers = {"content-type": "application/x-msdownload"}

    monkeypatch.setattr(disclosure.requests, "get", lambda *a, **k: _Resp())
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-02-01"),
        out_dir=Path("x"), dart_api_key="k",
    )
    df = disclosure.download_corp_code_map(cfg)
    assert list(df["stock_code"]) == ["005930"]
    with pytest.raises(ValueError, match="DART_API_KEY"):
        disclosure.download_corp_code_map(
            AltDataFetchConfig(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-02-01"), out_dir=Path("x"))
        )
