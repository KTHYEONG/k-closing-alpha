import os
import time
import pandas as pd
from pykrx import stock

# Google Sheets 연동
import gspread
from gspread.utils import rowcol_to_a1
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

from src import settings

GOOGLE_KEY_PATH = str(settings.GOOGLE_KEY_PATH)
GOOGLE_SHEET_NAME = settings.GOOGLE_SHEET_NAME
WORKSHEET_NAME = settings.TRADE2_WORKSHEET_NAME

INST_COL = settings.GOTTEN_COLS["INST"]
FRGN_COL = settings.GOTTEN_COLS["FOREIGN"]
DATE_COL = settings.GOTTEN_COLS["DATE"]
CODE_COL = settings.GOTTEN_COLS["CODE"]


def _load_sheet_dataframe():
    """Trade2 시트를 DataFrame으로 로드"""
    if not os.path.exists(GOOGLE_KEY_PATH):
        raise FileNotFoundError(f"인증 키 파일이 없습니다: {GOOGLE_KEY_PATH}")

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
    raise NotImplementedError("전체 시트 업데이트는 수식 보존을 위해 비활성화되었습니다.")


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


def fill_missing_stock_data():
    # 1. 시트에서 데이터 로드
    print("Google Sheets(Trade2)에서 데이터를 불러오는 중...")
    df, ws = _load_sheet_dataframe()

    # 빈 문자열을 NaN으로 변환해 결측 탐지
    df.replace("", pd.NA, inplace=True)

    # 필수 컬럼 존재 여부 확인
    required_cols = [INST_COL, FRGN_COL, CODE_COL, DATE_COL]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"시트에 '{col}' 열이 없습니다.")

    # 2. 업데이트할 대상 행 식별
    # (기관_순매수) 또는 (외국인_순매수)가 비어있는(NaN) 행 중, 종목코드가 유효한 행
    # 주의: 순매수가 0인 경우는 정상 값으로 간주.
    target_mask = (df[INST_COL].isna() | df[FRGN_COL].isna()) & df[
        CODE_COL
    ].notna()

    target_indices = df[target_mask].index
    total_count = len(target_indices)
    print(f"업데이트 대상 행 개수: {total_count}개")

    inst_col_idx = _get_column_index(ws, INST_COL)
    frgn_col_idx = _get_column_index(ws, FRGN_COL)
    updates = []

    for idx, i in enumerate(target_indices):
        try:
            # 날짜와 종목코드 추출
            date_value = df.loc[i, DATE_COL]
            code_value = df.loc[i, CODE_COL]

            # 날짜 포맷 (여러 형식을 안전하게 처리)
            date = pd.to_datetime(date_value).strftime("%Y%m%d")

            # 종목코드 포맷 (숫자/문자 모두 처리)
            code_str = str(code_value).split(".")[0]
            code = code_str.zfill(6)

            df_net = stock.get_market_trading_value_by_date(date, date, code)

            if not df_net.empty:
                # 외국인 순매수 (컬럼명이 다를 수 있어 후보군/포함 문자열로 탐색)
                foreign_net = 0
                foreign_candidates = [
                    "외국인",
                    "외국인합계",
                    "외국인계",
                    "외국계",
                    "외국인(합계)",
                ]
                foreign_cols = [c for c in foreign_candidates if c in df_net.columns]
                if foreign_cols:
                    foreign_net = df_net[foreign_cols[0]].iloc[0]
                else:
                    # 후보가 없으면 '외국'이 포함된 모든 컬럼을 합산하여 사용
                    foreign_any = [c for c in df_net.columns if "외국" in c]
                    if foreign_any:
                        foreign_net = df_net[foreign_any].sum(axis=1).iloc[0]

                # 기관 순매수
                inst_cols = [
                    "금융투자",
                    "보험",
                    "투신",
                    "사모",
                    "은행",
                    "기타금융",
                    "연기금",
                    "연기금등",
                ]
                valid_inst_cols = [c for c in inst_cols if c in df_net.columns]

                if "기관합계" in df_net.columns:
                    inst_net = df_net["기관합계"].iloc[0]
                else:
                    inst_net = df_net[valid_inst_cols].sum(axis=1).iloc[0]

                df.at[i, INST_COL] = inst_net
                df.at[i, FRGN_COL] = foreign_net
                updates.append((i + 2, inst_col_idx, inst_net))
                updates.append((i + 2, frgn_col_idx, foreign_net))

                print(
                    f"[{idx+1}/{total_count}] {date} {code} "
                    f"업데이트 완료 -> 기관: {inst_net}, 외국인: {foreign_net}"
                )
            else:
                print(f"[{idx+1}/{total_count}] {date} {code} 데이터 없음")

            time.sleep(0.1)

        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue

    # 4. 결과를 필요한 셀에만 반영 (다른 열의 수식을 보존)
    _batch_update_cells(ws, updates)
    print("\n모든 작업이 완료되었습니다. Google Sheets(Trade2)에 값만 업데이트했습니다.")


if __name__ == "__main__":
    fill_missing_stock_data()
