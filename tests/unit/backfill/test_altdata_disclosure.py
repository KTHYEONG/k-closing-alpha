from pathlib import Path

import pandas as pd

from src.backfill.altdata import disclosure
from src.backfill.altdata.config import AltDataFetchConfig


def test_disclosure_categorizes_and_maps_symbols(monkeypatch) -> None:
    pages = {
        ("Y", 1): {"status": "000", "total_page": 1, "list": [
            {"corp_code": "00126380", "stock_code": "005930", "report_nm": "단일판매ㆍ공급계약체결", "rcept_dt": "20240102"},
            {"corp_code": "00126380", "stock_code": "005930", "report_nm": "전환사채권발행결정", "rcept_dt": "20240102"},
            {"corp_code": "99999999", "stock_code": "", "report_nm": "기타경영사항", "rcept_dt": "20240102"},
        ]},
        ("K", 1): {"status": "013", "message": "nil"},
    }
    monkeypatch.setattr(disclosure, "_dart_get_json", lambda url, params, cfg: pages[(params["corp_cls"], params["page_no"])])
    corp_map = pd.DataFrame({"corp_code": ["99999999"], "stock_code": [""], "corp_name": ["X"]})
    cfg = AltDataFetchConfig(
        start=pd.Timestamp("2024-01-02"), end=pd.Timestamp("2024-01-03"),
        out_dir=Path("x"), dart_api_key="k", retries=1, retry_sleep_sec=0.0,
    )
    out = disclosure.collect_disclosures(cfg, corp_map)
    row = out[out["symbol"] == "005930"].iloc[0]
    assert row["n_supply_contract"] == 1
    assert row["n_cb_bw"] == 1
    assert row["n_total"] == 2
    assert bool(row["has_material"]) is True
    assert (out["symbol"] == "").sum() == 0


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
