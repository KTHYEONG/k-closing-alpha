import sys
import os

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import requests
import json
import pandas as pd
import configs.kis_config as kis_config

import asyncio
import aiohttp
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timedelta

from openpyxl import load_workbook
from openpyxl.styles import Alignment

# =========================================================
# [설정] API 접속 정보
# =========================================================
APP_KEY = kis_config.real_investment["app_key"]
APP_SECRET = kis_config.real_investment["app_secret"]
URL_BASE = "https://openapi.koreainvestment.com:9443"

# HTS ID를 config 파일에서 불러옴
HTS_ID = kis_config.real_investment.get("hts_id")
TARGET_CONDITION_NAME = "종가매매"

# 토큰 캐시 파일을 configs 폴더에 저장
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
configs_dir = os.path.join(project_root, "configs")
TOKEN_FILE = os.path.join(configs_dir, "kis_token_cache.json")

print(f"[{datetime.now()}] 조건검색 (엑셀 중앙정렬 기능 추가) 시작...")

if not HTS_ID or "여기에" in HTS_ID:
    print(
        "❌ 오류: configs/kis_config.py 파일의 'hts_id'에 본인의 HTS ID를 입력해주세요!"
    )
    exit()


# ---------------------------------------------------------
# 0. 세션 및 재시도 설정 (연결 안정성 확보)
# ---------------------------------------------------------
session = requests.Session()
retry_strategy = Retry(
    total=3,  # 최대 3번 재시도
    backoff_factor=1,  # 재시도 간격 지수적 증가
    status_forcelist=[429, 500, 502, 503, 504],  # 재시도할 HTTP 상태 코드
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)


# ---------------------------------------------------------
# 1. 접근 토큰 관리
# ---------------------------------------------------------
def get_access_token(force_refresh=False):
    if not force_refresh and os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            if saved_data.get("app_key") == APP_KEY:
                expired_at = datetime.strptime(
                    saved_data["expired_at"], "%Y-%m-%d %H:%M:%S"
                )
                if datetime.now() < expired_at - timedelta(minutes=10):
                    print(f"✅ 기존 토큰 재사용 (만료: {expired_at})")
                    return saved_data["access_token"]
        except Exception as e:
            print(f"⚠️ 토큰 캐시 읽기 실패: {e}")

    print("🚀 새 토큰 발급 요청 중...")
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET,
    }

    res = session.post(
        f"{URL_BASE}/oauth2/tokenP", headers=headers, data=json.dumps(body)
    )
    data = res.json()

    if "access_token" not in data:
        print(f"❌ 토큰 발급 실패: {data}")
        exit()

    new_token = data["access_token"]
    expires_in = data.get("expires_in", 86400)
    expired_at_str = (datetime.now() + timedelta(seconds=expires_in)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "access_token": new_token,
                    "expired_at": expired_at_str,
                    "app_key": APP_KEY,
                },
                f,
            )
    except Exception as e:
        print(f"⚠️ 토큰 캐시 저장 실패: {e}")

    return new_token


TOKEN = get_access_token()


# ---------------------------------------------------------
# 종목별 프로그램 매매 추이 (순매수) 조회 함수
# ---------------------------------------------------------
def get_program_net_buy(code):
    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/program-trade-by-stock"

    tr_id = "FHPPG04650101"

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",
    }

    params = {
        "FID_COND_MRKT_DIV_CODE": "J",  # J: KRX, NX: NXT, UN: 통합
        "FID_INPUT_ISCD": code,
    }

    try:
        res = session.get(url, headers=headers, params=params)

        if res.status_code != 200:
            print(f" ⚠️ [API 호출 오류] ({code}) 상태코드: {res.status_code}")
            return 0.0

        data = res.json()

        # 응답 데이터 파싱
        if data["rt_cd"] == "0" and data.get("output"):
            # whol_smtn_ntby_tr_pbmn: 전체 합계 순매수 거래 대금
            latest_data = data["output"][0]
            net_buy_amt_str = latest_data.get("whol_smtn_ntby_tr_pbmn", "0")
            return float(net_buy_amt_str)

        return 0.0

    except Exception as e:
        print(f" ⚠️ 프로그램 매매 조회 실패 ({code}): {e}")
        return 0.0


# ---------------------------------------------------------
# 시장 지수(KOSPI, KOSDAQ) 등락률 조회 함수
# ---------------------------------------------------------
def get_market_index_rate(market_code):
    headers = {
        "content-type": "application/json; charset=utf-8",
        "authorization": f"Bearer {TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKUP03500100",  # 업종기간별시세
        "custtype": "P",
    }

    now = datetime.now()
    str_today = now.strftime("%Y%m%d")
    str_past = (now - timedelta(days=5)).strftime("%Y%m%d")

    params = {
        "fid_cond_mrkt_div_code": "U",
        "fid_input_iscd": market_code,
        "fid_input_date_1": str_past,
        "fid_input_date_2": str_today,
        "fid_period_div_code": "D",
        "fid_org_adj_prc": "0",
    }

    try:
        res = session.get(
            f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-daily-indexchartprice",
            headers=headers,
            params=params,
        )
        data = res.json()

        if data["rt_cd"] != "0":
            return 0.0

        if data.get("output1"):
            out1 = data["output1"]
            rate_str = out1.get("bstp_nmix_prdy_ctrt") or out1.get("prdy_ctrt")

            if rate_str and float(rate_str) != 0.0:
                return float(rate_str)

            try:
                current_price = float(out1.get("bstp_nmix_prpr", "0"))
                change_amount = float(out1.get("bstp_nmix_prdy_vrss", "0"))
                prev_close = current_price - change_amount
                if prev_close != 0:
                    cal_rate = (change_amount / prev_close) * 100
                    return round(cal_rate, 2)
            except Exception:
                pass

        return 0.0

    except Exception as e:
        print(f" ⚠️ 지수 조회 실패 ({market_code}): {e}")

    return 0.0


# ---------------------------------------------------------
# 외인/기관 추정가집계 API 호출 함수
# ---------------------------------------------------------
def get_investor_trend_estimate(code):
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "HHPTJ04160200",
        "custtype": "P",
    }
    params = {"MKSC_SHRN_ISCD": code}

    try:
        res = session.get(
            f"{URL_BASE}/uapi/domestic-stock/v1/quotations/investor-trend-estimate",
            headers=headers,
            params=params,
        )
        data = res.json()

        if data["rt_cd"] == "0":
            if data["output2"] and len(data["output2"]) > 0:
                latest = data["output2"][0]
                frgn_qty = int(latest["frgn_fake_ntby_qty"])
                orgn_qty = int(latest["orgn_fake_ntby_qty"])
                return frgn_qty, orgn_qty
            else:
                return 0, 0
        else:
            return 0, 0

    except Exception as e:
        print(f" ⚠️ 수급 추정 조회 실패 ({code}): {e}")
        return 0, 0


# ---------------------------------------------------------
# 종목 상세 정보 조회 (현재가, 거래대금 등)
# ---------------------------------------------------------
def get_stock_detail(code):
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST01010100",
        "custtype": "P",
    }
    params = {
        "fid_cond_mrkt_div_code": "J",
        "fid_input_iscd": code,
    }

    try:
        res = session.get(
            f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-price",
            headers=headers,
            params=params,
        )
        data = res.json()
        if data["rt_cd"] == "0":
            return data["output"]
    except Exception as e:
        print(f"   ⚠️ 상세 조회 실패 ({code}): {e}")

    return None


# ---------------------------------------------------------
# 2. 내 조건 목록 가져오기
# ---------------------------------------------------------
def get_condition_list():
    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "HHKST03900300",
        "custtype": "P",
    }
    params = {"user_id": HTS_ID}
    res = session.get(
        f"{URL_BASE}/uapi/domestic-stock/v1/quotations/psearch-title",
        headers=headers,
        params=params,
    )
    data = res.json()
    if data.get("rt_cd") != "0":
        print(f"❌ 조건 목록 조회 실패: {data.get('msg1')}")
        return []
    return data["output2"]


# ---------------------------------------------------------
# [실행] 시장 지수 조회 (KOSPI, KOSDAQ)
# ---------------------------------------------------------
print("📊 시장 지수(KOSPI, KOSDAQ) 조회 중...")
kospi_rate = get_market_index_rate("0001")
kosdaq_rate = get_market_index_rate("1001")
print(f"   👉 KOSPI: {kospi_rate}% | KOSDAQ: {kosdaq_rate}%")


my_conditions = get_condition_list()
if not my_conditions:
    print("⚠ 저장된 조건이 없습니다.")
    exit()

target_cond = None
for c in my_conditions:
    if TARGET_CONDITION_NAME in c["condition_nm"]:
        target_cond = c
        break

if target_cond is None:
    print(f"❌ '{TARGET_CONDITION_NAME}' 조건을 찾을 수 없습니다.")
    exit()

print(f"\n>> '[{target_cond['condition_nm']}]' 검색 시작...")


# ---------------------------------------------------------
# 종목별 체결강도 조회 함수 (TR_ID: FHKST01010300)
# ---------------------------------------------------------
def get_trade_strength(code):
    url = f"{URL_BASE}/uapi/domestic-stock/v1/quotations/inquire-ccnl"

    headers = {
        "content-type": "application/json",
        "authorization": f"Bearer {TOKEN}",
        "appKey": APP_KEY,
        "appSecret": APP_SECRET,
        "tr_id": "FHKST01010300",  # 주식현재가 체결
        "custtype": "P",
    }

    params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": code}

    try:
        res = session.get(url, headers=headers, params=params)
        data = res.json()

        if data["rt_cd"] == "0" and data.get("output"):
            # output 리스트의 첫 번째 항목(가장 최신 체결)을 가져옵니다.
            latest_trade = data["output"][0]

            if "tday_rltv" in latest_trade:
                return float(latest_trade["tday_rltv"])

            # tday_rltv가 없을 경우를 위한 예비 처리
            if "ctrb_strgt" in latest_trade:
                return float(latest_trade["ctrb_strgt"])

        return 0.0

    except Exception as e:
        print(f" ⚠️ 체결강도 조회 실패 ({code}): {e}")
        return 0.0


# ---------------------------------------------------------
# 4. 검색 실행
# ---------------------------------------------------------
headers = {
    "content-type": "application/json",
    "authorization": f"Bearer {TOKEN}",
    "appKey": APP_KEY,
    "appSecret": APP_SECRET,
    "tr_id": "HHKST03900400",
    "custtype": "P",
}
params = {"user_id": HTS_ID, "seq": target_cond["seq"]}
res = session.get(
    f"{URL_BASE}/uapi/domestic-stock/v1/quotations/psearch-result",
    headers=headers,
    params=params,
)
data = res.json()

if data["rt_cd"] != "0":
    print(f"❌ 검색 실패: {data['msg1']}")
    exit()

stock_list = data["output2"]
print(f"✅ 검색 결과: 총 {len(stock_list)} 종목 포착!")


# ---------------------------------------------------------
# 5. 상세 정보 조회 및 데이터 매핑
# ---------------------------------------------------------
async def fetch_single_stock(i, stock, total, sem):
    """개별 종목 데이터를 수집하는 비동기 태스크"""
    async with sem:
        code = stock["code"]
        name = stock["name"]

        try:
            price = int(float(stock.get("price", 0)))
            rate = float(stock.get("chgrate", 0))
        except:
            price, rate = 0, 0.0

        # 여러 API를 동시에 호출 (비동기)
        detail, vol_power, (frgn_qty, orgn_qty), program_amt_won = await asyncio.gather(
            asyncio.to_thread(get_stock_detail, code),
            asyncio.to_thread(get_trade_strength, code),
            asyncio.to_thread(get_investor_trend_estimate, code),
            asyncio.to_thread(get_program_net_buy, code),
        )

        market_name, mkt_cap_eok, trade_amt_eok = "", 0, 0
        if detail:
            try:
                price = int(float(str(detail.get("stck_prpr", price)).replace(",", "")))
                rate = float(str(detail.get("prdy_ctrt", rate)).replace(",", ""))
                shares = float(str(detail.get("lstn_stcn", "0")).replace(",", ""))

                raw_market = str(detail.get("rprs_mrkt_kor_name", "")).upper()
                market_name = (
                    "KOSPI"
                    if "KOSPI" in raw_market or "유가" in raw_market
                    else "KOSDAQ" if "KOSDAQ" in raw_market else raw_market
                )

                raw_mkt_cap = (
                    float(str(detail.get("hts_avls", "0")).replace(",", ""))
                    * 100_000_000
                )
                if raw_mkt_cap == 0:
                    raw_mkt_cap = shares * price

                mkt_cap_eok = round(raw_mkt_cap / 100_000_000, 2)
                trade_amt_eok = round(
                    float(str(detail.get("acml_tr_pbmn", "0")).replace(",", ""))
                    / 100_000_000,
                    2,
                )
            except:
                pass

        frgn_net_eok = round((frgn_qty * price) / 100_000_000, 2)
        orgn_net_eok = round((orgn_qty * price) / 100_000_000, 2)
        program_net_eok = round(program_amt_won / 100_000_000, 2)

        print(f"\r 🔍 데이터 수집 중... ({i+1}/{total})", end="", flush=True)

        return {
            "종목명": name,
            "종목코드": code,
            "시장구분": market_name,
            "시가총액(억)": mkt_cap_eok,
            "거래대금(억)": trade_amt_eok,
            "체결강도": vol_power,
            "등락률": rate,
            "순위": 0,
            "기관_순매수(억)": orgn_net_eok,
            "외국인_순매수(억)": frgn_net_eok,
            "프로그램_순매수(억)": program_net_eok,
            "(차트통과)": 1,
        }


async def fetch_all_stock_data(stock_list):
    # 실전 계좌 TPS 20 제한을 고려하여 동시 실행 종목 수를 5개로 제한
    # (5개 종목 * 종목당 4개 API = 20 TPS)
    sem = asyncio.Semaphore(5)
    total = len(stock_list)
    tasks = [
        fetch_single_stock(i, stock, total, sem) for i, stock in enumerate(stock_list)
    ]
    results = await asyncio.gather(*tasks)

    print("\n✅ 데이터 수집 완료")
    return results


results = asyncio.run(fetch_all_stock_data(stock_list))

if results:
    # 저장 경로를 프로젝트 루트의 'data' 폴더로 설정
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)  # data 폴더가 없으면 생성

    clean_name = target_cond["condition_nm"].replace("/", "_").replace("\\", "_")
    save_path = os.path.join(data_dir, f"condition_{clean_name}.xlsx")

    df = pd.DataFrame(results)
    df = df.sort_values(by="거래대금(억)", ascending=False)
    df["순위"] = range(1, len(df) + 1)

    # --- [개선] 별도 캐시 파일을 이용한 (차트통과) 데이터 영구 보존 로직 ---
    cache_path = os.path.join(data_dir, "chart_pass_cache.json")
    today_str = datetime.now().strftime("%Y-%m-%d")
    daily_memory = {}

    # 1. 기존 캐시 로드 (오늘 날짜인 경우만)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                if cache_data.get("date") == today_str:
                    daily_memory = cache_data.get("data", {})
        except Exception:
            pass

    # 2. [수정] 신규/제외 종목 감지 (엑셀 복원 전 수행하여 정확한 변경사항 포착)
    current_codes = set(df["종목코드"].unique())
    cached_codes = set(daily_memory.keys())

    # 2-1. 신규 종목 포착 (현재 결과에는 있으나 캐시에는 없는 것)
    new_codes = current_codes - cached_codes
    if new_codes:
        new_stocks = df[df["종목코드"].isin(new_codes)]
        print(f"\n✨ [신규 종목 포착] {len(new_stocks)}개 종목이 새로 발견되었습니다:")
        for _, row in new_stocks.iterrows():
            print(f"   - {row['종목명']} ({row['종목코드']})")

        # 신규 종목들을 메모리에 추가 (기본값 1)
        for code in new_codes:
            daily_memory[code] = 1

    # 3. 기존 엑셀 파일에서 사용자가 수정한 최신 값 반영 (수동 수정사항 유지)
    if os.path.exists(save_path):
        try:
            file_mtime = datetime.fromtimestamp(os.path.getmtime(save_path)).date()
            if file_mtime == datetime.now().date():
                old_df = pd.read_excel(save_path, engine="openpyxl")
                if "종목코드" in old_df.columns and "(차트통과)" in old_df.columns:
                    old_df["종목코드"] = old_df["종목코드"].astype(str).str.zfill(6)
                    current_excel_vals = dict(
                        zip(old_df["종목코드"], old_df["(차트통과)"])
                    )
                    daily_memory.update(current_excel_vals)
        except Exception as e:
            print(f"    ⚠️ 캐시 업데이트 중 오류: {e}")

    df["(차트통과)"] = df["종목코드"].map(daily_memory).fillna(1).astype(int)

    # 4. 업데이트된 캐시 저장
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"date": today_str, "data": daily_memory}, f, indent=2)
    except Exception:
        pass

    total_count = len(df)
    avg_trade_amt = round(df["거래대금(억)"].mean(), 2)

    df["전체종목수"] = total_count
    df["평균거래대금(억)"] = avg_trade_amt
    df["KOSPI등락률"] = kospi_rate
    df["KOSDAQ등락률"] = kosdaq_rate

    cols_order = [
        "(차트통과)",
        "종목명",
        "종목코드",
        "시가총액(억)",
        "거래대금(억)",
        "등락률",
        "순위",
        "기관_순매수(억)",
        "외국인_순매수(억)",
        "프로그램_순매수(억)",
        "체결강도",
        "시장구분",
        "전체종목수",
        "평균거래대금(억)",
        "KOSPI등락률",
        "KOSDAQ등락률",
    ]
    df = df[cols_order]

    # 엑셀 저장
    df.to_excel(save_path, index=False)

    # =========================================================
    # 저장된 엑셀 파일을 다시 열어 '중앙 정렬' 적용 -> for 가독성 향상
    # =========================================================
    try:
        wb = load_workbook(save_path)
        ws = wb.active

        center_align = Alignment(horizontal="center", vertical="center")

        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = center_align

        wb.save(save_path)

    except Exception as e:
        print(f"⚠️ 엑셀 서식 적용 실패: {e}")

    print(f"\n📋 [검색 결과 요약 - 총 {total_count}종목]")
    for _, row in df.head(total_count).iterrows():
        pass_status = "통과" if row["(차트통과)"] == 1 else "제외"
        print(
            f"   {int(row['순위']):2d}. {row['종목명']:<10} | {row['등락률']:>6.2f}% | {row['거래대금(억)']:>7.1f}억 | {pass_status}"
        )

    print(f"\n📂 엑셀 저장 완료: {save_path}")
else:
    print("⚠ 검색된 종목이 없습니다.")
