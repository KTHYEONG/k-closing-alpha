import asyncio
import aiohttp
import sys
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import pandas as pd

# 프로젝트 루트 디렉토리를 sys.path에 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.api.kis_client import KisApiClient
from src import settings


async def fetch_index_for_date_range(client, session, code, name, start_date, end_date):
    """특정 기간의 지수 데이터를 가져옵니다."""
    print(f"Fetching {name} ({code}) from {start_date} to {end_date}...")
    
    all_data = []
    curr_start = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")
    
    chunk_count = 0
    while curr_start < end_dt:
        # API는 한 번에 최대 50개 레코드만 반환하므로 50일 단위로 조회 (영업일 기준 약 35일)
        curr_end = curr_start + timedelta(days=50)
        if curr_end > end_dt:
            curr_end = end_dt
            
        s_str = curr_start.strftime("%Y%m%d")
        e_str = curr_end.strftime("%Y%m%d")
        
        chunk_count += 1
        print(f"  Chunk {chunk_count}: {s_str} ~ {e_str}", end="", flush=True)
        
        try:
            resp = await client.get_market_index_history(session, code, s_str, e_str)
            if resp.get('rt_cd') == '0':
                items = resp.get('output2', [])
                all_data.extend(items)
                print(f" ✓ ({len(items)} records)")
            else:
                print(f" ✗ Error: {resp.get('msg1')}")
        except Exception as e:
            print(f" ✗ Exception: {e}")
            
        curr_start = curr_end + timedelta(days=1)
        await asyncio.sleep(0.3)  # Rate limit
        
    print(f"  Total records for {name}: {len(all_data)}")
    return all_data


def build_date_index_map(data_list):
    """API 응답 데이터를 날짜별 딕셔너리로 변환합니다."""
    date_map = {}
    for item in data_list:
        date = item.get('stck_bsop_date')
        if date:
            # 날짜 포맷 변환: YYYYMMDD -> YYYY-MM-DD
            try:
                formatted_date = f"{date[:4]}-{date[4:6]}-{date[6:8]}"
                close_value = float(item.get('bstp_nmix_prpr') or 0)
                date_map[formatted_date] = close_value
            except (ValueError, IndexError):
                continue
    return date_map


def normalize_date(date_str):
    """다양한 날짜 형식을 YYYY-MM-DD로 정규화합니다."""
    if not date_str:
        return None
        
    # 이미 YYYY-MM-DD 형식인 경우
    if len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
        return date_str
    
    # YYYYMMDD 형식
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # YYYY/MM/DD 형식
    if '/' in date_str:
        parts = date_str.split('/')
        if len(parts) == 3:
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
    
    # YYYY.MM.DD 형식
    if '.' in date_str:
        parts = date_str.split('.')
        if len(parts) == 3:
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
    
    return None


def calculate_historical_volatility(data_list):
    """지수 데이터를 받아 역사적 변동성을 계산합니다."""
    # 데이터 정리
    records = []
    for item in data_list:
        date = item.get('stck_bsop_date')
        try:
            close = float(item.get('bstp_nmix_prpr') or 0)
            if date and close > 0:
                records.append({'date': date, 'close': close})
        except:
            continue
            
    df = pd.DataFrame(records)
    df = df.sort_values('date').reset_index(drop=True)
    
    # 1. 로그 수익률 계산: ln(오늘종가 / 어제종가)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. 이동 표준편차 계산 (20일 = 약 1달 영업일)
    window = 20
    df['std'] = df['log_ret'].rolling(window=window).std()
    
    # 3. 연율화 (Annualization): 표준편차 * sqrt(252) * 100(퍼센트)
    df['hv'] = df['std'] * np.sqrt(252) * 100
    
    # 맵 생성 (YYYY-MM-DD -> HV)
    hv_map = {}
    for _, row in df.dropna().iterrows():
        d_str = row['date']
        formatted_date = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}"
        hv_map[formatted_date] = round(row['hv'], 2)
        
    return df, hv_map


def update_gsheet_with_calculated_data(key_path, sheet_name, worksheet_names, kospi200_map, hv_map, vkosdaq_map=None):
    """구글 시트의 (v-kospi), (v-kosdaq) 컬럼에 계산된 역사적 변동성을 업데이트합니다."""
    
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
        client = gspread.authorize(creds)
        sh = client.open(sheet_name)
        
        for ws_name in worksheet_names:
            print(f"\n[INFO] Processing worksheet: {ws_name}")
            ws = sh.worksheet(ws_name)
            
            all_values = ws.get_all_values()
            if not all_values:
                continue
                
            headers = all_values[0]
            
            # 필요한 컬럼 인덱스 찾기
            try:
                date_col_idx = headers.index("(매수날짜)")
            except ValueError:
                print(f"  [Error] '(매수날짜)' 컬럼을 찾을 수 없습니다.")
                continue

            # (v-kospi) 컬럼 찾기
            vkospi_col_idx = -1
            if "(v-kospi)" in headers:
                vkospi_col_idx = headers.index("(v-kospi)")
            
            # (v-kosdaq) 컬럼 찾기
            vkosdaq_col_idx = -1
            if "(v-kosdaq)" in headers:
                vkosdaq_col_idx = headers.index("(v-kosdaq)")
            
            updates = []
            
            for row_idx, row in enumerate(all_values[1:], start=2):
                # 행 길이 맞추기
                while len(row) < len(headers):
                    row.append("")
                    
                date_value = row[date_col_idx] if date_col_idx < len(row) else ""
                
                if date_value:
                    normalized_date = normalize_date(date_value)
                    if normalized_date:
                        # 1. V-KOSPI 업데이트
                        if vkospi_col_idx != -1:
                            hv_val = hv_map.get(normalized_date, "")
                            if hv_val:
                                updates.append({
                                    'range': f'{chr(65 + vkospi_col_idx)}{row_idx}',
                                    'values': [[str(hv_val)]]
                                })
                        
                        # 2. V-KOSDAQ 업데이트
                        if vkosdaq_col_idx != -1 and vkosdaq_map:
                            hv_val_k = vkosdaq_map.get(normalized_date, "")
                            if hv_val_k:
                                updates.append({
                                    'range': f'{chr(65 + vkosdaq_col_idx)}{row_idx}',
                                    'values': [[str(hv_val_k)]]
                                })

            if updates:
                ws.batch_update(updates)
                print(f"  [INFO] {ws_name}에 {len(updates)}개 셀을 업데이트했습니다.")
            else:
                print(f"  [INFO] {ws_name}에 변경사항이 없습니다.")
                
    except Exception as e:
        print(f"[Error] 구글 시트 업데이트 실패: {e}")


async def main():
    client = KisApiClient()
    
    # KOSPI 200 코드 (1028) & KOSDAQ 150 코드 (2203 예상)
    kospi200_code = "1028"
    kosdaq150_code = "2203"
    
    # 데이터 수집 기간
    start_date = "20151201" 
    end_date = datetime.now().strftime("%Y%m%d")
    
    async with aiohttp.ClientSession() as session:
        await client.ensure_token(session)
        
        # 1. KOSPI 200 처리
        print(f"\n[INFO] KOSPI 200 (코드: {kospi200_code}) 데이터 수집...")
        kospi200_data = await fetch_index_for_date_range(
            client, session, kospi200_code, "KOSPI 200", start_date, end_date
        )
        _, hv_map_kospi = calculate_historical_volatility(kospi200_data) if kospi200_data else (None, {})
        kospi200_map = build_date_index_map(kospi200_data) if kospi200_data else {}
        
        # 2. KOSDAQ 150 처리
        print(f"\n[INFO] KOSDAQ 150 (코드: {kosdaq150_code}) 데이터 수집...")
        kosdaq150_data = await fetch_index_for_date_range(
            client, session, kosdaq150_code, "KOSDAQ 150", start_date, end_date
        )
        _, hv_map_kosdaq = calculate_historical_volatility(kosdaq150_data) if kosdaq150_data else (None, {})
        
        print(f"\n[INFO] 변동성 계산 완료: V-KOSPI {len(hv_map_kospi)}개, V-KOSDAQ {len(hv_map_kosdaq)}개")
        
        # 구글 시트 업데이트
        update_gsheet_with_calculated_data(
            str(settings.GOOGLE_KEY_PATH),
            settings.GOOGLE_SHEET_NAME,
            settings.TRADE_WORKSHEETS,
            kospi200_map,
            hv_map_kospi,
            vkosdaq_map=hv_map_kosdaq
        )
        
        print("\n[완료] 데이터 업데이트 완료!")
        print("       - (v-kospi): KOSPI 200 기반 HV")
        print("       - (v-kosdaq): KOSDAQ 150 기반 HV")


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
