import os
import sys
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

from src import settings

load_dotenv(settings.BASE_DIR / ".env")


def load_data_from_gsheet(key_path, sheet_name, worksheet_name):
    """구글 시트에서 데이터를 DataFrame으로 로드하는 함수"""
    if not os.path.exists(key_path):
        print(f"Error: 인증 키 파일({key_path})이 없습니다.")
        sys.exit(1)
    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
        client = gspread.authorize(creds)
        sh = client.open(sheet_name)
        ws = sh.worksheet(worksheet_name)
        records = ws.get_all_records()
        df = pd.DataFrame(records)
        if df.empty:
            print(f"Warning: '{worksheet_name}' 시트에서 가져온 데이터가 없습니다.")
        return df
    except Exception as e:
        print(f"'{worksheet_name}' 시트 로드 실패: {e}")
        return None


def load_and_combine_sheets(*args):
    """여러 워크시트에서 데이터를 로드하고 하나로 합치는 함수"""
    if len(args) == 2:
        sheet_name, worksheet_names = args
        key_path = str(settings.GOOGLE_KEY_PATH)
    elif len(args) == 3:
        key_path, sheet_name, worksheet_names = args
    else:
        raise TypeError(
            "load_and_combine_sheets() takes either 2 arguments "
            "(sheet_name, worksheet_names) or 3 arguments "
            "(key_path, sheet_name, worksheet_names)"
        )

    print("[INFO] 구글 시트에서 데이터 로드 중...")
    if not key_path:
        print("Error: 구글 시트 인증 키 경로가 설정되지 않았습니다.")
        sys.exit(1)
    df_list = []
    for ws_name in worksheet_names:
        df_sheet = load_data_from_gsheet(key_path, sheet_name, ws_name)  # type: ignore
        if df_sheet is not None and not df_sheet.empty:
            df_list.append(df_sheet)

    if not df_list:
        print("Error: 모든 시트에서 데이터를 가져오지 못했습니다.")
        sys.exit(1)

    return pd.concat(df_list, ignore_index=True)


def append_stocks_to_gsheet(key_path, sheet_name, worksheet_name, stocks_to_add):
    """
    구글 시트에 새로운 종목들을 추가하는 함수 (이미 존재하는 종목은 제외)
    stocks_to_add: [{'종목코드': '000000', '종목명': '삼성전자'}, ...] 리스트
    """
    if not stocks_to_add:
        return

    try:
        scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, scope)
        client = gspread.authorize(creds)
        sh = client.open(sheet_name)
        ws = sh.worksheet(worksheet_name)

        # 1. 시트의 전체 데이터와 헤더 로드 (API 호출 최소화)
        all_values = ws.get_all_values()
        if not all_values:
            print("[Warning] 시트가 비어있습니다. 헤더가 필요합니다.")
            return

        headers = all_values[0]
        # 종목코드 컬럼의 인덱스 찾기
        try:
            code_col_idx = headers.index("종목코드")
        except ValueError:
            print("[Error] '종목코드' 컬럼을 찾을 수 없습니다.")
            return

        # 2. 이미 존재하는 종목코드 집합 생성
        existing_codes = {str(row[code_col_idx]).zfill(6) for row in all_values[1:]}

        # 3. 새로운 종목만 필터링하여 추가할 행 생성
        new_rows = []
        for stock in stocks_to_add:
            code = str(stock["종목코드"]).zfill(6)
            if code not in existing_codes:
                row_to_append = []
                for header in headers:
                    if header == "종목코드":
                        row_to_append.append(code)
                    elif header == "종목명":
                        row_to_append.append(stock["종목명"])
                    else:
                        row_to_append.append("")  # 다른 컬럼(테마 등)은 비워둠
                new_rows.append(row_to_append)

        # 4. 데이터 일괄 추가
        if new_rows:
            ws.append_rows(new_rows)
            print(f"[INFO] 구글 시트에 {len(new_rows)}개 종목이 새로 추가되었습니다.")
        else:
            print("[INFO] 추가할 새로운 종목이 없습니다 (이미 시트에 존재).")

    except Exception as e:
        print(f"[Error] 구글 시트 업데이트 실패: {e}")
