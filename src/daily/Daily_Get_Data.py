import sys
import os

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import json
import pandas as pd

import asyncio
import aiohttp
import time
from datetime import datetime, timedelta

from openpyxl import load_workbook
from openpyxl.styles import Alignment

# 커스텀 모듈 임포트
from src.api.kis_client import KisApiClient
from src.utils.display import Colors

from src import settings

# =========================================================
# [설정] API 접속 정보
# =========================================================
APP_KEY = settings.KIS_API_CONFIG["app_key"]
APP_SECRET = settings.KIS_API_CONFIG["app_secret"]
ACCOUNT_ID = settings.KIS_API_CONFIG.get("account_id", "")
HTS_ID = settings.KIS_API_CONFIG.get("hts_id")

TARGET_CONDITION_NAME = settings.TARGET_CONDITION_NAME
TOKEN_FILE = str(settings.TOKEN_FILE)

print(f"[{datetime.now()}] 조건검색 (엑셀 중앙정렬 기능 추가) 시작...")

if not HTS_ID or "여기에" in HTS_ID:
    print(
        "❌ 오류: configs/kis_config.py 파일의 'hts_id'에 본인의 HTS ID를 입력해주세요!"
    )
    exit()


def safe_float(value, default=0.0):
    """문자열이나 None 값을 안전하게 float로 변환"""
    if value is None:
        return default
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------
# 헬퍼 함수: 시장 지수 등락률 파싱
# ---------------------------------------------------------
def parse_market_index_rate(data):
    if not data or data.get("rt_cd") != "0":
        return 0.0
    out1 = data.get("output1")
    if not out1:
        return 0.0
    rate_str = out1.get("bstp_nmix_prdy_ctrt") or out1.get("prdy_ctrt")
    try:
        if rate_str and float(rate_str) != 0.0:
            return float(rate_str)
        current_price = float(out1.get("bstp_nmix_prpr", "0"))
        change_amount = float(out1.get("bstp_nmix_prdy_vrss", "0"))
        prev_close = current_price - change_amount
        if prev_close != 0:
            return round((change_amount / prev_close) * 100, 2)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------
# 상세 정보 조회 및 데이터 매핑 (비동기)
# ---------------------------------------------------------
async def fetch_single_stock(i, stock, total, sem, client, session):
    """개별 종목 데이터를 수집하는 비동기 태스크"""
    async with sem:
        # 요청 간의 미세한 간격을 두어 TPS 분산 (Staggering)
        await asyncio.sleep(settings.API_SLEEP_INTERVAL)
        code = stock["code"]
        name = stock["name"]

        try:
            price = int(float(stock.get("price", 0)))
            rate = float(stock.get("chgrate", 0))
        except:
            price, rate = 0, 0.0

        # KisApiClient를 사용하여 비동기로 여러 API 동시 호출
        res_detail, res_strength, res_investor, res_program = await asyncio.gather(
            client.get_current_price(session, code),
            client.get_trade_strength(session, code),
            client.get_investor_trend_estimate(session, code),
            client.get_program_net_buy(session, code),
        )

        # 데이터 파싱
        detail = res_detail.get("output") if res_detail.get("rt_cd") == "0" else None

        vol_power = 0.0
        if res_strength.get("rt_cd") == "0" and res_strength.get("output"):
            latest = res_strength["output"][0]
            vol_power = safe_float(latest.get("tday_rltv") or latest.get("ctrb_strgt"))

        frgn_qty, orgn_qty = 0, 0
        if res_investor.get("rt_cd") == "0" and res_investor.get("output2"):
            latest = res_investor["output2"][0]
            frgn_qty = int(safe_float(latest.get("frgn_fake_ntby_qty", 0)))
            orgn_qty = int(safe_float(latest.get("orgn_fake_ntby_qty", 0)))

        program_amt_won = 0.0
        if res_program.get("rt_cd") == "0" and res_program.get("output"):
            program_amt_won = safe_float(
                res_program["output"][0].get("whol_smtn_ntby_tr_pbmn", 0)
            )

        if res_detail.get("rt_cd") != "0":
            print(
                f"\n {Colors.RED}⚠️ [{code}] 상세정보 로드 실패: {res_detail.get('msg1')}{Colors.RESET}"
            )

        market_name, mkt_cap_eok, trade_amt_eok = "", 0, 0
        if detail:
            try:
                price = int(safe_float(detail.get("stck_prpr"), price))
                rate = safe_float(detail.get("prdy_ctrt"), rate)
                shares = safe_float(detail.get("lstn_stcn"), 0)

                raw_market = str(detail.get("rprs_mrkt_kor_name", "")).upper()
                market_name = (
                    "KOSPI"
                    if "KOSPI" in raw_market or "유가" in raw_market
                    else "KOSDAQ" if "KOSDAQ" in raw_market else raw_market
                )

                raw_mkt_cap = safe_float(detail.get("hts_avls")) * 100_000_000
                if raw_mkt_cap == 0:
                    raw_mkt_cap = shares * price

                mkt_cap_eok = round(raw_mkt_cap / 100_000_000, 2)
                trade_amt_eok = round(
                    safe_float(detail.get("acml_tr_pbmn")) / 100_000_000, 2
                )
            except Exception as e:
                print(f"\n {Colors.RED}⚠️ [{name}] 파싱 에러: {e}{Colors.RESET}")

        frgn_net_eok = round((frgn_qty * price) / 100_000_000, 2)
        orgn_net_eok = round((orgn_qty * price) / 100_000_000, 2)
        program_net_eok = round(program_amt_won / 100_000_000, 2)

        print(
            f"\r {Colors.YELLOW}🔍 데이터 수집 중... ({i+1}/{total}){Colors.RESET}",
            end="",
            flush=True,
        )

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


async def fetch_all_stock_data(stock_list, client, session):
    # API 요청 제한 설정
    sem = asyncio.Semaphore(settings.API_SEMAPHORE_LIMIT)
    total = len(stock_list)
    tasks = [
        fetch_single_stock(i, stock, total, sem, client, session)
        for i, stock in enumerate(stock_list)
    ]
    results = await asyncio.gather(*tasks)

    print(f"\n{Colors.GREEN}✅ 데이터 수집 완료{Colors.RESET}")
    return results


async def main():
    async with aiohttp.ClientSession() as session:
        # 1. 클라이언트 초기화 및 토큰 확보
        client = KisApiClient(
            APP_KEY, APP_SECRET, ACCOUNT_ID, HTS_ID, token_file=TOKEN_FILE
        )
        await client.ensure_token(session)

        # 2. 시장 지수 조회
        print(f"{Colors.CYAN}📊 시장 지수(KOSPI, KOSDAQ) 조회 중...{Colors.RESET}")
        res_kospi = await client.get_market_index_rate(session, "0001")
        res_kosdaq = await client.get_market_index_rate(session, "1001")

        kospi_rate = parse_market_index_rate(res_kospi)
        kosdaq_rate = parse_market_index_rate(res_kosdaq)
        print(f"   👉 KOSPI: {kospi_rate}% | KOSDAQ: {kosdaq_rate}%")

        # 3. 조건 목록 및 대상 조건 찾기
        res_cond_list = await client.get_condition_list(session)
        my_conditions = res_cond_list.get("output2", [])

        if not my_conditions:
            print(f"{Colors.YELLOW}⚠ 저장된 조건이 없습니다.{Colors.RESET}")
            return

        target_cond = next(
            (c for c in my_conditions if TARGET_CONDITION_NAME in c["condition_nm"]),
            None,
        )
        if not target_cond:
            print(
                f"{Colors.RED}❌ '{TARGET_CONDITION_NAME}' 조건을 찾을 수 없습니다.{Colors.RESET}"
            )
            return

        print(
            f"\n>> {Colors.BOLD}[{target_cond['condition_nm']}]{Colors.RESET} 검색 시작..."
        )

        # 4. 조건검색 결과 조회
        res_cond_res = await client.get_condition_result(session, target_cond["seq"])
        if res_cond_res.get("rt_cd") != "0":
            print(f"{Colors.RED}❌ 검색 실패: {res_cond_res.get('msg1')}{Colors.RESET}")
            return

        stock_list = res_cond_res.get("output2", [])
        print(
            f"{Colors.GREEN}✅ 검색 결과: 총 {len(stock_list)} 종목 포착!{Colors.RESET}"
        )

        # 5. 상세 데이터 수집
        results = await fetch_all_stock_data(stock_list, client, session)

        if results:
            # 데이터 저장 및 후처리 로직 (기존과 동일)
            clean_name = (
                target_cond["condition_nm"].replace("/", "_").replace("\\", "_")
            )
            save_path = str(settings.DATA_DIR / f"condition_{clean_name}.xlsx")

            df = pd.DataFrame(results)
            df = df.sort_values(by="거래대금(억)", ascending=False)
            df["순위"] = range(1, len(df) + 1)

            # 캐시 로직
            cache_path = str(settings.CHART_PASS_CACHE_FILE)
            today_str = datetime.now().strftime("%Y-%m-%d")
            daily_memory = {}

            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                        if cache_data.get("date") == today_str:
                            daily_memory = cache_data.get("data", {})
                except:
                    pass

            # 신규 종목 감지 및 캐시 업데이트
            current_codes = set(df["종목코드"].unique())
            cached_codes = set(daily_memory.keys())
            new_codes = current_codes - cached_codes

            if new_codes:
                print(
                    f"\n{Colors.MAGENTA}✨ [신규 종목 포착] {len(new_codes)}개 종목이 새로 발견되었습니다.{Colors.RESET}"
                )
                for code in new_codes:
                    daily_memory[code] = 1

            # 기존 엑셀 수정사항 반영
            if os.path.exists(save_path):
                try:
                    if (
                        datetime.fromtimestamp(os.path.getmtime(save_path)).date()
                        == datetime.now().date()
                    ):
                        old_df = pd.read_excel(save_path, engine="openpyxl")
                        if (
                            "종목코드" in old_df.columns
                            and "(차트통과)" in old_df.columns
                        ):
                            old_df["종목코드"] = (
                                old_df["종목코드"].astype(str).str.zfill(6)
                            )
                            daily_memory.update(
                                dict(zip(old_df["종목코드"], old_df["(차트통과)"]))
                            )
                except:
                    pass

            df["(차트통과)"] = df["종목코드"].map(daily_memory).fillna(1).astype(int)

            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"date": today_str, "data": daily_memory}, f, indent=2)

            # 추가 지표 계산
            total_count = len(df)
            df["전체종목수"] = total_count
            df["평균거래대금(억)"] = round(df["거래대금(억)"].mean(), 2)
            df["KOSPI등락률"] = kospi_rate
            df["KOSDAQ등락률"] = kosdaq_rate

            # 컬럼 정렬 및 저장
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
            df[cols_order].to_excel(save_path, index=False)

            # 엑셀 서식 적용
            try:
                wb = load_workbook(save_path)
                ws = wb.active
                center_align = Alignment(horizontal="center", vertical="center")
                for row in ws.iter_rows():
                    for cell in row:
                        cell.alignment = center_align
                wb.save(save_path)
            except:
                pass

            print(
                f"\n{Colors.BOLD}📋 [검색 결과 요약 - 총 {total_count}종목]{Colors.RESET}"
            )
            for _, row in df.iterrows():
                pass_status = (
                    f"{Colors.GREEN}통과{Colors.RESET}"
                    if row["(차트통과)"] == 1
                    else f"{Colors.GRAY}제외{Colors.RESET}"
                )
                print(
                    f"   {int(row['순위']):2d}. {row['종목명']:<10} | {row['등락률']:>6.2f}% | {row['거래대금(억)']:>7.1f}억 | {pass_status}"
                )

            print(f"\n{Colors.GREEN}📂 엑셀 저장 완료: {save_path}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠ 검색된 종목이 없습니다.{Colors.RESET}")


if __name__ == "__main__":
    asyncio.run(main())
