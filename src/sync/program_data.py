import os
import sys
import time

import pandas as pd
import requests

# src/etc 안에서 바로 kis_common을 import 할 수 있도록 경로 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
for path in [CURRENT_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

from kis_common import APP_KEY, APP_SECRET, URL_BASE, get_access_token


def _clean_num(v):
    if v is None:
        return 0.0
    s = str(v).strip().replace(",", "")
    if s in {"", "-", "--", "None", "nan"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def get_program_history(
    code: str,
    start_date: str,
    end_date: str,
    *,
    target_dates: list[str] | None = None,
    max_calls: int = 120,
    sleep_sec: float = 0.02,
):
    """종목별 프로그램 매매 추이(일별) 조회.

    API:
      GET /uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily
      TR_ID: FHPPG04650201

    start_date, end_date: 'YYYYMMDD'
    """
    token = get_access_token()

    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily"

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHPPG04650201",
        "custtype": "P",
    }

    prog_map = {}
    s = pd.to_datetime(str(start_date).strip(), format="%Y%m%d", errors="coerce")
    e = pd.to_datetime(str(end_date).strip(), format="%Y%m%d", errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return prog_map
    if s > e:
        s, e = e, s
    start = s.strftime("%Y%m%d")
    end = e.strftime("%Y%m%d")

    wanted = None
    if target_dates:
        w = {
            str(d).strip()
            for d in target_dates
            if len(str(d).strip()) == 8 and str(d).strip().isdigit()
        }
        w = {d for d in w if start <= d <= end}
        if w:
            wanted = w

    code = str(code).strip().zfill(6)
    cursor = end
    seen_days = set()
    no_progress = 0

    for _ in range(max(1, int(max_calls))):
        if cursor < start:
            break
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": cursor,
        }
        try:
            res = requests.get(url, headers=headers, params=params, timeout=15)
            data = res.json()
        except Exception:
            dt = pd.to_datetime(cursor, format="%Y%m%d", errors="coerce")
            if pd.isna(dt):
                break
            cursor = (dt - pd.Timedelta(days=1)).strftime("%Y%m%d")
            continue

        if res.status_code != 200 or data.get("rt_cd") != "0":
            dt = pd.to_datetime(cursor, format="%Y%m%d", errors="coerce")
            if pd.isna(dt):
                break
            cursor = (dt - pd.Timedelta(days=1)).strftime("%Y%m%d")
            continue

        rows = data.get("output", [])
        row_days = set()
        for item in rows:
            date = (item.get("stck_bsop_date") or "").strip()[:8]
            if not date or len(date) != 8 or not date.isdigit():
                continue
            row_days.add(date)
            if not (start <= date <= end):
                continue
            net_amt = _clean_num(item.get("whol_smtn_ntby_tr_pbmn", "0"))
            prog_map[date] = float(net_amt)
            seen_days.add(date)

        if wanted and wanted.issubset(seen_days):
            break

        if row_days:
            min_day = min(row_days)
            next_dt = pd.to_datetime(min_day, format="%Y%m%d", errors="coerce")
            if pd.isna(next_dt):
                break
            next_cursor = (next_dt - pd.Timedelta(days=1)).strftime("%Y%m%d")
        else:
            dt = pd.to_datetime(cursor, format="%Y%m%d", errors="coerce")
            if pd.isna(dt):
                break
            next_cursor = (dt - pd.Timedelta(days=30)).strftime("%Y%m%d")

        if next_cursor >= cursor:
            no_progress += 1
            dt = pd.to_datetime(cursor, format="%Y%m%d", errors="coerce")
            if pd.isna(dt):
                break
            next_cursor = (dt - pd.Timedelta(days=30)).strftime("%Y%m%d")
        else:
            no_progress = 0
        cursor = next_cursor
        if no_progress >= 3:
            break
        time.sleep(max(0.0, float(sleep_sec)))

    return prog_map
