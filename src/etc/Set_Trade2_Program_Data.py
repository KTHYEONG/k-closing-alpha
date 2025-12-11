import os
import sys
import time
from datetime import datetime

import gspread
import pandas as pd
from dotenv import load_dotenv
from gspread.utils import rowcol_to_a1
from oauth2client.service_account import ServiceAccountCredentials

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

for path in [PROJECT_ROOT, CURRENT_DIR]:
    if path not in sys.path:
        sys.path.append(path)

from program_data import get_program_history

ENV_FILE_PATH = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(ENV_FILE_PATH)

google_key_env = os.getenv("GSPREAD_KEY_PATH")
if not google_key_env:
    raise EnvironmentError(
        f"'GSPREAD_KEY_PATH' 환경변수가 없습니다 (.env: {ENV_FILE_PATH})"
    )

if not os.path.isabs(google_key_env):
    google_key_env = os.path.join(PROJECT_ROOT, google_key_env)

GOOGLE_KEY_PATH = os.path.normpath(google_key_env)
GOOGLE_SHEET_NAME = "Stock"
WORKSHEET_NAME = "Trade2"

DATE_COL = "(매수날짜)"
CODE_COL = "(종목코드)"
PROGRAM_COL = "(프로그램_순매수)"


def _load_sheet_dataframe():
    """Trade2 시트를 DataFrame으로 로드"""
    if not os.path.exists(GOOGLE_KEY_PATH):
        raise FileNotFoundError(f"인증 키 파일을 찾을 수 없습니다: {GOOGLE_KEY_PATH}")

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name(GOOGLE_KEY_PATH, scope)
    client = gspread.authorize(creds)

    sh = client.open(GOOGLE_SHEET_NAME)
    ws = sh.worksheet(WORKSHEET_NAME)

    records = ws.get_all_records()
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("가져온 데이터가 없습니다.")

    return df, ws


def _save_sheet_dataframe(ws, df):
    """(Deprecated) 전체 시트 덮어쓰기는 수식 손실 위험이 있어 사용하지 않음."""
    raise NotImplementedError(
        "전체 시트 업데이트는 수식 보존을 위해 비활성화되었습니다."
    )


def _normalize_code(code_value: object) -> str:
    """시트에 섞여 있을 수 있는 코드 표기를 6자리 문자열로 통일"""
    code_str = str(code_value).split(".")[0]
    return code_str.zfill(6)


def _fetch_program_for_date(code: str, trade_date: str):
    """기존 fetch_program 로직을 사용해 해당 일자의 순매수 금액을 조회"""
    try:
        history = get_program_history(code, trade_date, trade_date)
    except Exception as exc:
        print(f"프로그램 매매 조회 실패 ({code}, {trade_date}): {exc}")
        return None

    return history.get(trade_date)


def _get_column_index(ws, column_name: str) -> int:
    """헤더 행에서 열 번호(1-based) 확인"""
    headers = ws.row_values(1)
    if column_name not in headers:
        raise KeyError(f"시트에 '{column_name}' 열이 없습니다.")
    return headers.index(column_name) + 1


def _batch_update_cells(ws, updates):
    """
    지정된 셀만 부분 업데이트해 기존 수식을 보존.
    updates: List[Tuple[row_idx, col_idx, value]]
    """
    if not updates:
        return

    data = []
    for row_idx, col_idx, value in updates:
        rng = rowcol_to_a1(row_idx, col_idx)
        # gspread/requests json 직렬화 호환을 위해 numpy 타입 -> 파이썬 기본형 변환
        if pd.isna(value):
            val = ""
        elif hasattr(value, "item"):
            val = value.item()
        else:
            val = value
        data.append({"range": rng, "values": [[val]]})
    ws.batch_update(data)


def fill_program_net_buy():
    """매수날짜/종목코드를 기반으로 (프로그램_순매수) 열을 채움"""
    print("Google Sheets(Trade2)에서 데이터 불러오는 중...")
    df, ws = _load_sheet_dataframe()
    df.replace("", pd.NA, inplace=True)

    for col in [DATE_COL, CODE_COL]:
        if col not in df.columns:
            raise KeyError(f"시트에 '{col}' 열이 없습니다.")

    if PROGRAM_COL not in df.columns:
        df[PROGRAM_COL] = pd.NA

    target_mask = df[PROGRAM_COL].isna() & df[CODE_COL].notna() & df[DATE_COL].notna()

    target_indices = df[target_mask].index
    total_count = len(target_indices)
    print(f"업데이트 대상 행 수: {total_count}개")

    if total_count == 0:
        print("채울 데이터가 없습니다.")
        return

    program_col_idx = _get_column_index(ws, PROGRAM_COL)
    updates = []

    for idx, row in enumerate(target_indices, start=1):
        date_value = df.loc[row, DATE_COL]
        code_value = df.loc[row, CODE_COL]

        try:
            trade_date = pd.to_datetime(date_value).strftime("%Y%m%d")
        except Exception as exc:
            print(f"[{idx}/{total_count}] 잘못된 날짜 '{date_value}': {exc}")
            continue

        code = _normalize_code(code_value)
        program_amt = _fetch_program_for_date(code, trade_date)

        if program_amt is None:
            print(f"[{idx}/{total_count}] {trade_date} {code} -> 데이터 없음/조회 실패")
            continue

        df.at[row, PROGRAM_COL] = program_amt
        updates.append(
            (row + 2, program_col_idx, program_amt)
        ) 
        print(
            f"[{idx}/{total_count}] {trade_date} {code} -> 프로그램 순매수 {program_amt:,.2f}"
        )

        time.sleep(0.1)

    _batch_update_cells(ws, updates)
    print("\n모든 업데이트 완료! Google Sheets(Trade2)에 반영했습니다.")


if __name__ == "__main__":
    fill_program_net_buy()
