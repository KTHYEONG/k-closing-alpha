import os
import sys
import requests

# src/etc 안에서 바로 kis_common을 import 할 수 있도록 경로 추가
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
for path in [CURRENT_DIR, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.append(path)

from kis_common import APP_KEY, APP_SECRET, URL_BASE, get_access_token


def get_program_history(code: str, start_date: str, end_date: str):
    """
    종목별 프로그램 매매 추이 (TR_ID: FHPPG04650200)
    start_date, end_date: 'YYYYMMDD'
    """
    token = get_access_token()

    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/program-trade-by-stock"

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {token}",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
        "tr_id": "FHPPG04650200",
        "custtype": "P",
    }

    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": code,
        "FID_INPUT_DATE_1": start_date,
        "FID_INPUT_DATE_2": end_date,
    }

    res = requests.get(url, headers=headers, params=params)
    data = res.json()

    if data.get("rt_cd") != "0":
        raise RuntimeError(f"프로그램 매매 API 오류: {data.get('msg1')}")

    prog_map = {}
    for item in data.get("output", []):
        date = (item.get("stck_bsop_date") or "").strip()[:8]
        if not date or not (start_date <= date <= end_date):
            continue

        net_amt_str = item.get("whol_smtn_ntby_tr_pbmn", "0")
        net_amt = float(net_amt_str.replace(",", ""))

        prog_map[date] = net_amt

    return prog_map
