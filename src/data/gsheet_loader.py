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
