import os
import sys
import time
from collections.abc import Callable
from typing import Any, Optional, TypeVar, cast

import gspread
import pandas as pd
from dotenv import load_dotenv
from gspread.exceptions import APIError
from oauth2client.service_account import ServiceAccountCredentials

from src import settings

load_dotenv(settings.BASE_DIR / ".env")

F = TypeVar("F", bound=Callable[..., Any])


def retry_on_quota_limit(max_retries: int = 3, base_delay: float = 2.0) -> Callable[[F], F]:
    """API Quota 초과(HTTP 429) 시 지수 백오프 방식으로 대기 후 재시도합니다."""
    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            retries = 0
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except APIError as e:
                    if (
                        hasattr(e, "response")
                        and e.response is not None
                        and e.response.status_code == 429
                        and retries < max_retries
                    ):
                        wait_time = base_delay * (2**retries)
                        print(
                            f"    [Warning] API Quota 초과(429). {wait_time}초 대기 후 재시도... ({retries+1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        retries += 1
                    else:
                        raise e
            raise Exception("Google Sheets API Quota Error: 최대 재시도 횟수 초과")
        return cast(F, wrapper)
    return decorator


class GSheetClientManager:
    """인증 정보와 Spreadsheet 객체를 캐싱하여 불필요한 API 호출을 제거합니다."""

    _instance: Optional["GSheetClientManager"] = None
    _initialized: bool = False
    scope: list[str]
    creds: ServiceAccountCredentials
    client: gspread.Client
    spreadsheets_cache: dict[str, gspread.Spreadsheet]

    def __new__(cls, key_path: str) -> "GSheetClientManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, key_path: str) -> None:
        if self._initialized:
            return

        self.scope = [
            "https://spreadsheets.google.com/feeds",
            "https://www.googleapis.com/auth/drive",
        ]
        self.creds = ServiceAccountCredentials.from_json_keyfile_name(key_path, self.scope)
        self.client = gspread.authorize(self.creds)
        self.spreadsheets_cache = {}  # 시트 이름별 캐싱
        self._initialized = True

    @retry_on_quota_limit()
    def get_spreadsheet(self, sheet_name: str) -> gspread.Spreadsheet:
        if sheet_name not in self.spreadsheets_cache:
            self.spreadsheets_cache[sheet_name] = self.client.open(sheet_name)
        return self.spreadsheets_cache[sheet_name]

    @retry_on_quota_limit()
    def get_all_records(self, sheet_name: str, worksheet_name: str) -> list[dict[str, Any]]:
        sh = self.get_spreadsheet(sheet_name)
        ws = sh.worksheet(worksheet_name)
        return ws.get_all_records()

    @retry_on_quota_limit()
    def get_all_values(
        self, sheet_name: str, worksheet_name: str
    ) -> tuple[list[list[Any]], gspread.Worksheet]:
        sh = self.get_spreadsheet(sheet_name)
        ws = sh.worksheet(worksheet_name)
        return ws.get_all_values(), ws

    @retry_on_quota_limit()
    def append_rows(self, ws: gspread.Worksheet, new_rows: list[list[Any]]) -> None:
        ws.append_rows(new_rows)


def load_data_from_gsheet(
    key_path: str, sheet_name: str, worksheet_name: str
) -> pd.DataFrame | None:
    """구글 시트에서 데이터를 DataFrame으로 로드하는 함수"""
    if not os.path.exists(key_path):
        print(f"Error: 인증 키 파일({key_path})이 없습니다.")
        sys.exit(1)
    try:
        manager = GSheetClientManager(key_path)
        records = manager.get_all_records(sheet_name, worksheet_name)
        df = pd.DataFrame(records)
        if df.empty:
            print(f"Warning: '{worksheet_name}' 시트에서 가져온 데이터가 없습니다.")
        return df
    except Exception as e:
        print(f"'{worksheet_name}' 시트 로드 실패: {e}")
        return None


def load_and_combine_sheets(*args: Any) -> pd.DataFrame:
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
        df_sheet = load_data_from_gsheet(key_path, sheet_name, ws_name)
        if df_sheet is not None and not df_sheet.empty:
            df_list.append(df_sheet)

    if not df_list:
        print("Error: 모든 시트에서 데이터를 가져오지 못했습니다.")
        sys.exit(1)

    return pd.concat(df_list, ignore_index=True)


def append_stocks_to_gsheet(
    key_path: str, sheet_name: str, worksheet_name: str, stocks_to_add: list[dict[str, Any]]
) -> None:
    """
    구글 시트에 새로운 종목들을 추가하는 함수 (이미 존재하는 종목은 제외)
    stocks_to_add: [{'종목코드': '000000', '종목명': '삼성전자'}, ...] 리스트
    """
    if not stocks_to_add:
        return

    try:
        manager = GSheetClientManager(key_path)
        all_values, ws = manager.get_all_values(sheet_name, worksheet_name)

        if not all_values:
            print("[Warning] 시트가 비어있습니다. 헤더가 필요합니다.")
            return

        headers = all_values[0]
        try:
            code_col_idx = headers.index("종목코드")
        except ValueError:
            print("[Error] '종목코드' column을 찾을 수 없습니다.")
            return

        existing_codes = {
            str(row[code_col_idx]).zfill(6) for row in all_values[1:] if len(row) > code_col_idx
        }

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
                        row_to_append.append("")
                new_rows.append(row_to_append)

        if new_rows:
            manager.append_rows(ws, new_rows)
            print(f"[INFO] 구글 시트에 {len(new_rows)}개 종목이 새로 추가되었습니다.")
        else:
            print("[INFO] 추가할 새로운 종목이 없습니다 (이미 시트에 존재).")

    except Exception as e:
        print(f"[Error] 구글 시트 업데이트 실패: {e}")
