"""DART 공시 수집기."""

from __future__ import annotations

import io
import logging
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.ratelimit import wait_for_dart_slot

logger = logging.getLogger(__name__)

_DISCLOSURE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "earnings": ("실적", "영업(잠정)", "매출액또는손익구조"),
    "supply_contract": ("단일판매", "공급계약"),
    "cb_bw": ("전환사채", "신주인수권부사채", "교환사채"),
    "rights_offering": ("유상증자",),
    "bonus_issue": ("무상증자",),
    "treasury": ("자기주식", "자사주"),
    "largest_holder_change": ("최대주주변경", "최대주주등의주식"),
    "capital_reduction": ("감자",),
    "disclosure_inquiry": ("조회공시", "불성실공시"),
}

_MATERIAL_CATEGORIES = {"earnings", "supply_contract", "cb_bw", "rights_offering", "largest_holder_change", "capital_reduction"}


def _dart_get_json(url: str, params: dict[str, object], cfg: AltDataFetchConfig) -> dict[str, object]:
    wait_for_dart_slot(cfg)
    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"DART request failed status={resp.status_code}")
    data = resp.json()
    # Check status field if present
    if isinstance(data, dict) and "status" in data:
        status = str(data.get("status", "")).strip()
        # 013 means no data (nil) - not error for pagination
        if status != "000" and status != "013":
            raise RuntimeError(f"DART error status={status} msg={data.get('message','')}")
    return data  # type: ignore[return-value]


def download_corp_code_map(cfg: AltDataFetchConfig) -> pd.DataFrame:
    """DART corpCode.xml을 다운로드하여 corp_code-stock_code 맵을 반환합니다.

    Args:
        cfg: Alt-data 설정.

    Returns:
        corp_code, stock_code, corp_name 컬럼을 가진 DataFrame.
    """
    if not str(cfg.dart_api_key).strip():
        raise ValueError("DART_API_KEY is required for disclosure backfill")
    wait_for_dart_slot(cfg)
    url = "https://opendart.fss.or.kr/api/corpCode.xml"
    resp = requests.get(url, params={"crtfc_key": cfg.dart_api_key}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"DART corpCode request failed status={resp.status_code}")
    content = resp.content
    # Response is ZIP containing CORPCODE.xml
    try:
        buf = io.BytesIO(content)
        with zipfile.ZipFile(buf) as zf:
            # Find CORPCODE.xml (case-sensitive or not)
            names = zf.namelist()
            target = None
            for n in names:
                if n.lower() == "corpcode.xml":
                    target = n
                    break
            if target is None:
                # take first file
                target = names[0]
            xml_bytes = zf.read(target)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"DART corpCode ZIP parse failed: {exc}") from exc
    # Parse XML
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise RuntimeError(f"DART CORPCODE.xml parse failed: {exc}") from exc
    rows: list[dict[str, str]] = []
    for elem in root.findall("list"):
        corp_code = (elem.findtext("corp_code") or "").strip()
        corp_name = (elem.findtext("corp_name") or "").strip()
        stock_code = (elem.findtext("stock_code") or "").strip()
        if not corp_code:
            continue
        # 8-digit corp_code, 6-digit stock_code
        corp_code = corp_code.zfill(8)
        stock_code = stock_code.strip()
        # Drop unlisted (blank stock_code)
        if not stock_code or not stock_code.strip():
            continue
        stock_code = stock_code.strip().zfill(6)
        rows.append({"corp_code": corp_code, "stock_code": stock_code, "corp_name": corp_name})
    if not rows:
        return pd.DataFrame(columns=["corp_code", "stock_code", "corp_name"])
    return pd.DataFrame(rows, columns=["corp_code", "stock_code", "corp_name"])


def collect_disclosures(cfg: AltDataFetchConfig, corp_map: pd.DataFrame) -> pd.DataFrame:
    """DART 공시 목록을 수집하여 일별 종목 집계 패널로 반환합니다.

    Args:
        cfg: Alt-data 설정.
        corp_map: corp_code-stock_code 맵.

    Returns:
        (date, symbol) 집계 DataFrame.
    """
    if not str(cfg.dart_api_key).strip():
        raise ValueError("DART_API_KEY is required for disclosure backfill")
    # Build corp_code -> stock_code map for fallback
    corp_to_stock: dict[str, str] = {}
    if corp_map is not None and not corp_map.empty:
        for _, r in corp_map.iterrows():
            cc = str(r.get("corp_code", "")).strip().zfill(8)
            sc = str(r.get("stock_code", "")).strip().zfill(6)
            if cc and sc and sc != "000000" and sc.strip():
                corp_to_stock[cc] = sc

    start_ymd = pd.Timestamp(cfg.start).strftime("%Y%m%d")
    end_ymd = pd.Timestamp(cfg.end).strftime("%Y%m%d")
    all_rows: list[dict[str, str]] = []

    for corp_cls in ["Y", "K"]:
        page_no = 1
        while True:
            params: dict[str, object] = {
                "crtfc_key": cfg.dart_api_key,
                "bgn_de": start_ymd,
                "end_de": end_ymd,
                "corp_cls": corp_cls,
                "page_no": page_no,
                "page_count": int(cfg.page_count),
            }
            data = _dart_get_json("https://opendart.fss.or.kr/api/list.json", params, cfg)
            status = str(data.get("status", "")).strip()
            if status == "013":
                break
            lst = data.get("list")
            if not isinstance(lst, list) or not lst:
                break
            for item in lst:
                if not isinstance(item, dict):
                    continue
                corp_code = str(item.get("corp_code", "")).strip().zfill(8)
                stock_code = str(item.get("stock_code", "")).strip()
                if stock_code:
                    stock_code = stock_code.strip().zfill(6)
                else:
                    stock_code = corp_to_stock.get(corp_code, "")
                report_nm = str(item.get("report_nm", "")).strip()
                rcept_dt = str(item.get("rcept_dt", "")).strip()
                all_rows.append(
                    {
                        "corp_code": corp_code,
                        "stock_code": stock_code,
                        "report_nm": report_nm,
                        "rcept_dt": rcept_dt,
                    }
                )
            # Check pagination end
            total_page = data.get("total_page")
            try:
                tp = int(total_page) if total_page is not None else None
            except Exception:
                tp = None
            if tp is not None and page_no >= tp:
                break
            # If returned less than page_count, likely last page
            if len(lst) < int(cfg.page_count):
                # But if total_page not given, we still break
                if tp is None:
                    break
            page_no += 1
            if page_no > 10000:  # safety
                break

    if not all_rows:
        # Return empty with correct columns
        cols = ["date", "symbol"] + [f"n_{k}" for k in _DISCLOSURE_CATEGORIES] + ["n_total", "has_material"]
        return pd.DataFrame(columns=cols)

    # Filter: drop rows with empty stock_code (unmappable)
    filtered = [r for r in all_rows if r["stock_code"] and r["stock_code"].strip() and r["stock_code"] != "000000"]
    if not filtered:
        cols = ["date", "symbol"] + [f"n_{k}" for k in _DISCLOSURE_CATEGORIES] + ["n_total", "has_material"]
        return pd.DataFrame(columns=cols)

    # Aggregate to (date=rcept_dt, symbol)
    # Categorize each row
    agg: dict[tuple[str, str], dict[str, int]] = {}
    for r in filtered:
        d = r["rcept_dt"]
        sym = r["stock_code"].strip().zfill(6)
        if not d or not sym:
            continue
        # Normalize date
        try:
            # rcept_dt is yyyymmdd
            pd.to_datetime(d, format="%Y%m%d")
        except Exception:
            try:
                pd.to_datetime(d)
            except Exception:
                continue
        key = (d, sym)
        if key not in agg:
            agg[key] = {f"n_{k}": 0 for k in _DISCLOSURE_CATEGORIES}
            agg[key]["n_total"] = 0
            # track material
            agg[key]["_has_material"] = 0  # type: ignore[typeddict-item]
        agg[key]["n_total"] += 1
        report_nm = r["report_nm"]
        for cat, patterns in _DISCLOSURE_CATEGORIES.items():
            for pat in patterns:
                if pat in report_nm:
                    agg[key][f"n_{cat}"] += 1
                    break
        # check material
        # has_material true if any of earnings etc matches
        for mat_cat in _MATERIAL_CATEGORIES:
            patterns = _DISCLOSURE_CATEGORIES.get(mat_cat, ())
            for pat in patterns:
                if pat in report_nm:
                    agg[key]["_has_material"] = 1  # type: ignore[typeddict-item]
                    break

    rows_out: list[dict[str, object]] = []
    for (d, sym), counts in agg.items():
        # Convert date
        try:
            dt = pd.to_datetime(d, format="%Y%m%d", errors="coerce")
            if pd.isna(dt):
                dt = pd.to_datetime(d, errors="coerce")
        except Exception:
            dt = pd.to_datetime(d, errors="coerce")
        if pd.isna(dt):
            continue
        out_row: dict[str, object] = {"date": pd.Timestamp(dt).normalize(), "symbol": sym}
        for k in _DISCLOSURE_CATEGORIES:
            out_row[f"n_{k}"] = int(counts.get(f"n_{k}", 0))
        out_row["n_total"] = int(counts.get("n_total", 0))
        out_row["has_material"] = bool(counts.get("_has_material", 0))
        rows_out.append(out_row)

    if not rows_out:
        cols = ["date", "symbol"] + [f"n_{k}" for k in _DISCLOSURE_CATEGORIES] + ["n_total", "has_material"]
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows_out)
    # Sort
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out
