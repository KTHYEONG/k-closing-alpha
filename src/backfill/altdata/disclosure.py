"""DART 공시 수집기 (효율 개편판).

효율 최적화:
- ``pblntf_ty`` 필터(B: 주요사항보고서, I: 거래소공시)로 조회량을 ~2배 축소.
  ``corp_cls`` 를 분할 조회하지 않고 응답에서 Y/K 만 필터.
- 창(≤3개월) 단위 페이지 병렬 조회(rate-limit 은 프로세스 전역 유지).
- 창 단위 집계 후 ``on_window`` 콜백으로 즉시 flush 가능 → 중단 안전 + 메모리 bounded.
"""

from __future__ import annotations

import io
import logging
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable

import pandas as pd
import requests

from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.ratelimit import retry_call, wait_for_dart_slot

logger = logging.getLogger(__name__)

_LIST_URL = "https://opendart.fss.or.kr/api/list.json"

# 수집 대상 공시유형: B(주요사항보고서), I(거래소공시). 아래 카테고리를 모두 포함.
_PBLNTF_TYPES: tuple[str, ...] = ("B", "I")
_KEPT_CORP_CLS: frozenset[str] = frozenset({"Y", "K"})
_PAGE_WORKERS = 4
_WINDOW_DAYS = 80

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
_MATERIAL_CATEGORIES: frozenset[str] = frozenset(
    {"earnings", "supply_contract", "cb_bw", "rights_offering", "largest_holder_change", "capital_reduction"}
)
_OUT_COLS: list[str] = (
    ["date", "symbol"] + [f"n_{k}" for k in _DISCLOSURE_CATEGORIES] + ["n_total", "has_material"]
)


def _dart_get_json(url: str, params: dict[str, object], cfg: AltDataFetchConfig) -> dict[str, object]:
    wait_for_dart_slot(cfg)
    resp = requests.get(url, params=params, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"DART request failed status={resp.status_code}")
    data = resp.json()
    if isinstance(data, dict) and "status" in data:
        status = str(data.get("status", "")).strip()
        # 000 정상, 013 무자료 — 그 외는 오류.
        if status not in ("000", "013"):
            raise RuntimeError(f"DART error status={status} msg={data.get('message', '')}")
    return data  # type: ignore[return-value]


def download_corp_code_map(cfg: AltDataFetchConfig) -> pd.DataFrame:
    """DART corpCode.xml 을 받아 corp_code-stock_code 맵(상장 종목만)을 반환합니다."""
    if not str(cfg.dart_api_key).strip():
        raise ValueError("DART_API_KEY is required for disclosure backfill")
    wait_for_dart_slot(cfg)
    resp = requests.get(
        "https://opendart.fss.or.kr/api/corpCode.xml",
        params={"crtfc_key": cfg.dart_api_key},
        timeout=30,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"DART corpCode request failed status={resp.status_code}")
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            target = next((n for n in names if n.lower() == "corpcode.xml"), names[0])
            xml_bytes = zf.read(target)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"DART corpCode ZIP parse failed: {exc}") from exc
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise RuntimeError(f"DART CORPCODE.xml parse failed: {exc}") from exc

    rows: list[dict[str, str]] = []
    for elem in root.findall("list"):
        corp_code = (elem.findtext("corp_code") or "").strip()
        stock_code = (elem.findtext("stock_code") or "").strip()
        if not corp_code or not stock_code:
            continue
        rows.append(
            {
                "corp_code": corp_code.zfill(8),
                "stock_code": stock_code.zfill(6),
                "corp_name": (elem.findtext("corp_name") or "").strip(),
            }
        )
    return pd.DataFrame(rows, columns=["corp_code", "stock_code", "corp_name"])


def _iter_windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    """[start, end] 를 ``_WINDOW_DAYS`` 이하 창(YYYYMMDD 쌍)으로 분할합니다."""
    out: list[tuple[str, str]] = []
    cur = pd.Timestamp(start).normalize()
    hard_end = pd.Timestamp(end).normalize()
    while cur <= hard_end:
        w_end = min(cur + pd.Timedelta(days=_WINDOW_DAYS), hard_end)
        out.append((cur.strftime("%Y%m%d"), w_end.strftime("%Y%m%d")))
        cur = w_end + pd.Timedelta(days=1)
    return out


def _parse_items(
    lst: list[object], corp_to_stock: dict[str, str]
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in lst:
        if not isinstance(item, dict):
            continue
        if str(item.get("corp_cls", "")).strip() not in _KEPT_CORP_CLS:
            continue
        corp_code = str(item.get("corp_code", "")).strip().zfill(8)
        stock_code = str(item.get("stock_code", "")).strip()
        stock_code = stock_code.zfill(6) if stock_code else corp_to_stock.get(corp_code, "")
        rows.append(
            {
                "stock_code": stock_code,
                "report_nm": str(item.get("report_nm", "")).strip(),
                "rcept_dt": str(item.get("rcept_dt", "")).strip(),
            }
        )
    return rows


def _fetch_list_page(
    cfg: AltDataFetchConfig, base_params: dict[str, object], page_no: int
) -> tuple[list[dict[str, str]], int | None]:
    """단일 페이지 조회 → (parsed rows, total_page). 실패 시 ([], None)."""
    params = {**base_params, "page_no": page_no}
    data = retry_call(lambda: _dart_get_json(_LIST_URL, params, cfg), cfg, label=f"dart list p{page_no}")
    if not isinstance(data, dict) or str(data.get("status", "")).strip() == "013":
        return [], None
    lst = data.get("list")
    if not isinstance(lst, list) or not lst:
        return [], None
    try:
        tp = int(data["total_page"]) if data.get("total_page") is not None else None
    except (TypeError, ValueError, KeyError):
        tp = None
    return lst, tp  # type: ignore[return-value]


def _fetch_disclosure_window(
    cfg: AltDataFetchConfig,
    pblntf_ty: str,
    start_ymd: str,
    end_ymd: str,
    corp_to_stock: dict[str, str],
) -> list[dict[str, str]]:
    """단일 (공시유형, ≤3개월 창) 목록을 1페이지 조회 후 나머지 페이지를 병렬 수집합니다."""
    from concurrent.futures import ThreadPoolExecutor

    base_params: dict[str, object] = {
        "crtfc_key": cfg.dart_api_key,
        "bgn_de": start_ymd,
        "end_de": end_ymd,
        "pblntf_ty": pblntf_ty,
        "page_count": int(cfg.page_count),
    }
    first_lst, total_page = _fetch_list_page(cfg, base_params, 1)
    rows = _parse_items(first_lst, corp_to_stock)
    if not first_lst or not total_page or total_page <= 1:
        return rows

    max_page = min(int(total_page), 10000)
    with ThreadPoolExecutor(max_workers=_PAGE_WORKERS) as pool:
        for lst, _tp in pool.map(
            lambda p: _fetch_list_page(cfg, base_params, p), range(2, max_page + 1)
        ):
            rows.extend(_parse_items(lst, corp_to_stock))
    return rows


def _aggregate_rows(rows: list[dict[str, str]]) -> pd.DataFrame:
    """원시 공시행을 (date, symbol) 카테고리 카운트 패널로 집계합니다."""
    agg: dict[tuple[str, str], dict[str, int]] = {}
    for r in rows:
        sym = r["stock_code"].strip().zfill(6) if r["stock_code"] else ""
        d = r["rcept_dt"].strip()
        if not sym or sym == "000000" or len(d) != 8:
            continue
        key = (d, sym)
        rec = agg.get(key)
        if rec is None:
            rec = dict.fromkeys((f"n_{k}" for k in _DISCLOSURE_CATEGORIES), 0)
            rec["n_total"] = 0
            rec["_material"] = 0
            agg[key] = rec
        rec["n_total"] += 1
        report_nm = r["report_nm"]
        for cat, patterns in _DISCLOSURE_CATEGORIES.items():
            if any(p in report_nm for p in patterns):
                rec[f"n_{cat}"] += 1
                if cat in _MATERIAL_CATEGORIES:
                    rec["_material"] = 1

    if not agg:
        return pd.DataFrame(columns=_OUT_COLS)

    out_rows: list[dict[str, object]] = []
    for (d, sym), rec in agg.items():
        dt = pd.to_datetime(d, format="%Y%m%d", errors="coerce")
        if pd.isna(dt):
            continue
        row: dict[str, object] = {"date": pd.Timestamp(dt).normalize(), "symbol": sym}
        for k in _DISCLOSURE_CATEGORIES:
            row[f"n_{k}"] = int(rec[f"n_{k}"])
        row["n_total"] = int(rec["n_total"])
        row["has_material"] = bool(rec["_material"])
        out_rows.append(row)
    if not out_rows:
        return pd.DataFrame(columns=_OUT_COLS)
    return pd.DataFrame(out_rows, columns=_OUT_COLS).sort_values(["date", "symbol"]).reset_index(drop=True)


def collect_disclosures(
    cfg: AltDataFetchConfig,
    corp_map: pd.DataFrame,
    *,
    on_window: Callable[[pd.DataFrame], None] | None = None,
    covered_dates: set[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """DART 공시 목록을 일별 종목 집계 패널로 수집합니다.

    Args:
        cfg: Alt-data 설정 (``dart_api_key`` 필수).
        corp_map: corp_code-stock_code 맵 (blank stock_code 보완용).
        on_window: 주어지면 각 창의 집계 프레임으로 즉시 호출하고 누적하지 않습니다
            (중단 안전 + 메모리 bounded). 이 경우 반환값은 빈 프레임입니다.
        covered_dates: 이미 수집된 날짜 집합. 창의 모든 영업일이 여기 포함되면
            해당 창을 건너뜁니다 (재개 시 재조회 방지).

    Returns:
        ``on_window`` 미지정 시 전체 기간 (date, symbol) 집계 DataFrame.
    """
    if not str(cfg.dart_api_key).strip():
        raise ValueError("DART_API_KEY is required for disclosure backfill")

    corp_to_stock: dict[str, str] = {}
    if corp_map is not None and not corp_map.empty:
        cc = corp_map["corp_code"].astype(str).str.strip().str.zfill(8)
        sc = corp_map["stock_code"].astype(str).str.strip().str.zfill(6)
        corp_to_stock = {c: s for c, s in zip(cc, sc, strict=True) if s and s != "000000"}

    parts: list[pd.DataFrame] = []
    for start_ymd, end_ymd in _iter_windows(cfg.start, cfg.end):
        if covered_dates:
            win_days = pd.bdate_range(start_ymd, end_ymd)
            if len(win_days) > 0 and all(d.normalize() in covered_dates for d in win_days):
                logger.info("[DATA] stage=altdata_disc window=%s..%s status=SKIP_COVERED", start_ymd, end_ymd)
                continue
        window_rows: list[dict[str, str]] = []
        for pblntf_ty in _PBLNTF_TYPES:
            window_rows.extend(
                _fetch_disclosure_window(cfg, pblntf_ty, start_ymd, end_ymd, corp_to_stock)
            )
        window_df = _aggregate_rows(window_rows)
        logger.info(
            "[DATA] stage=altdata_disc window=%s..%s rows=%d", start_ymd, end_ymd, len(window_df)
        )
        if window_df.empty:
            continue
        if on_window is not None:
            on_window(window_df)
        else:
            parts.append(window_df)

    if on_window is not None:
        return pd.DataFrame(columns=_OUT_COLS)
    if not parts:
        return pd.DataFrame(columns=_OUT_COLS)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
