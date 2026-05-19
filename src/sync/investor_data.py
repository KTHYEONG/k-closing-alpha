from __future__ import annotations

import os
import sys
import time

import pandas as pd
import requests

# Allow direct execution from src/etc.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
for path in [CURRENT_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

from kis_common import APP_KEY, APP_SECRET, URL_BASE, get_access_token


def _clean_num(v):
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s in {"", "-", "--", "None", "nan"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _request_investor_daily(
    *,
    code: str,
    trade_date: str,
    token: str,
    timeout_sec: int = 15,
) -> requests.Response:
    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHPTJ04160001",
        "custtype": "P",
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": str(code).strip().zfill(6),
        "FID_INPUT_DATE_1": str(trade_date).strip(),
        "FID_ORG_ADJ_PRC": "",
        "FID_ETC_CLS_CODE": "",
    }
    return requests.get(url, headers=headers, params=params, timeout=timeout_sec)


def _collect_rows(body: dict) -> list[dict]:
    rows: list[dict] = []
    for key in ["output2", "output1", "output"]:
        val = body.get(key)
        if isinstance(val, list):
            rows.extend([x for x in val if isinstance(x, dict)])
        elif isinstance(val, dict):
            rows.append(val)
    return rows


def _prev_day_ymd(ymd: str, days: int = 1) -> str:
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        return ymd
    return (dt - pd.Timedelta(days=max(1, int(days)))).strftime("%Y%m%d")


def get_investor_trade_daily(
    code: str,
    start_date: str,
    end_date: str,
    *,
    target_dates: list[str] | None = None,
    max_calls: int = 120,
    sleep_sec: float = 0.02,
) -> pd.DataFrame:
    """Fetch per-day foreign/institution netbuy by stock via KIS.

    API:
      GET /uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily
      TR_ID: FHPTJ04160001
    """
    code = str(code).strip().zfill(6)
    s_dt = pd.to_datetime(str(start_date).strip(), format="%Y%m%d", errors="coerce")
    e_dt = pd.to_datetime(str(end_date).strip(), format="%Y%m%d", errors="coerce")
    if pd.isna(s_dt) or pd.isna(e_dt):
        return pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])
    if s_dt > e_dt:
        s_dt, e_dt = e_dt, s_dt
    start = s_dt.strftime("%Y%m%d")
    end = e_dt.strftime("%Y%m%d")

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

    token = get_access_token()
    all_rows: list[dict] = []
    seen_days = set()
    cursor = end
    no_progress = 0

    for _ in range(max(1, int(max_calls))):
        if cursor < start:
            break
        try:
            resp = _request_investor_daily(code=code, trade_date=cursor, token=token)
            body = resp.json()
        except Exception:
            cursor = _prev_day_ymd(cursor, 1)
            continue

        if resp.status_code != 200:
            cursor = _prev_day_ymd(cursor, 1)
            continue
        if body.get("rt_cd") != "0":
            cursor = _prev_day_ymd(cursor, 1)
            continue

        rows = _collect_rows(body)
        row_days = {
            str(item.get("stck_bsop_date") or "").strip()
            for item in rows
            if isinstance(item, dict)
        }
        row_days = {d for d in row_days if len(d) == 8 and d.isdigit()}
        if rows:
            all_rows.extend(rows)
            seen_days.update({d for d in row_days if start <= d <= end})

        if wanted and wanted.issubset(seen_days):
            break

        if row_days:
            min_day = min(row_days)
            next_cursor = _prev_day_ymd(min_day, 1)
        else:
            next_cursor = _prev_day_ymd(cursor, 30)

        if next_cursor >= cursor:
            no_progress += 1
            next_cursor = _prev_day_ymd(cursor, 30)
        else:
            no_progress = 0
        cursor = next_cursor

        if no_progress >= 3:
            break

        time.sleep(max(0.0, float(sleep_sec)))

    if not all_rows:
        return pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])

    out_rows = []
    for item in all_rows:
        d = str(item.get("stck_bsop_date") or "").strip()
        if len(d) != 8 or not d.isdigit():
            continue
        if not (start <= d <= end):
            continue

        foreign = _clean_num(item.get("frgn_ntby_tr_pbmn"))
        inst = _clean_num(item.get("orgn_ntby_tr_pbmn"))
        if foreign is None:
            foreign = _clean_num(item.get("frgn_ntby_qty"))
        if inst is None:
            inst = _clean_num(item.get("orgn_ntby_qty"))

        out_rows.append(
            {
                "date": pd.to_datetime(d, format="%Y%m%d", errors="coerce"),
                "foreign_netbuy": foreign,
                "inst_netbuy": inst,
            }
        )

    if not out_rows:
        return pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])

    out = pd.DataFrame(out_rows)
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    out["foreign_netbuy"] = pd.to_numeric(out["foreign_netbuy"], errors="coerce")
    out["inst_netbuy"] = pd.to_numeric(out["inst_netbuy"], errors="coerce")
    return out.reset_index(drop=True)
