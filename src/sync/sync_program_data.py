import os
import time
from typing import Any

import pandas as pd
from gspread.utils import rowcol_to_a1

try:
    from src.sync.program_data import get_program_history
except Exception:
    from sync.program_data import get_program_history  # type: ignore[no-redef]
from src import settings
from src.data.gsheet_loader import GSheetClientManager, retry_on_quota_limit

GOOGLE_KEY_PATH = str(settings.GOOGLE_KEY_PATH)
GOOGLE_SHEET_NAME = settings.GOOGLE_SHEET_NAME
WORKSHEET_NAME = "Trade2"

DATE_COL = settings.GOTTEN_COLS["DATE"]
CODE_COL = settings.GOTTEN_COLS["CODE"]
PROGRAM_COL = settings.GOTTEN_COLS["PROGRAM"]


def _load_sheet_dataframe() -> tuple[pd.DataFrame, Any]:
    """Trade2 시트를 DataFrame으로 로드"""
    if not os.path.exists(GOOGLE_KEY_PATH):
        raise FileNotFoundError(f"인증 키 파일을 찾을 수 없습니다: {GOOGLE_KEY_PATH}")

    manager = GSheetClientManager(GOOGLE_KEY_PATH)
    records = manager.get_all_records(GOOGLE_SHEET_NAME, WORKSHEET_NAME)
    
    sh = manager.get_spreadsheet(GOOGLE_SHEET_NAME)
    ws = sh.worksheet(WORKSHEET_NAME)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("가져온 데이터가 없습니다.")

    return df, ws


def _save_sheet_dataframe(ws: Any, df: pd.DataFrame) -> None:
    """(Deprecated) 전체 시트 덮어쓰기는 수식 손실 위험이 있어 사용하지 않음."""
    raise NotImplementedError(
        "전체 시트 업데이트는 수식 보존을 위해 비활성화되었습니다."
    )


def _normalize_code(code_value: object) -> str:
    """시트에 섞여 있을 수 있는 코드 표기를 6자리 문자열로 통일"""
    code_str = str(code_value).split(".")[0]
    return code_str.zfill(6)


def _get_column_index(ws: Any, column_name: str) -> int:
    """헤더 행에서 열 번호(1-based) 확인"""
    headers: list[str] = ws.row_values(1)
    if column_name not in headers:
        raise KeyError(f"시트에 '{column_name}' 열이 없습니다.")
    return headers.index(column_name) + 1


@retry_on_quota_limit()
def _batch_update_cells(ws: Any, updates: list[tuple[int, int, Any]]) -> None:
    """지정된 셀만 부분 업데이트해 기존 수식을 보존.
    updates: list[tuple[row_idx, col_idx, value]]
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


def fill_program_net_buy() -> None:
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

    # 대상 데이터 프레임 복사 및 가공 (인덱스 보존)
    target_df = df[target_mask].copy()
    target_df["standardized_code"] = target_df[CODE_COL].apply(_normalize_code)
    target_df["parsed_date"] = pd.to_datetime(target_df[DATE_COL])

    # 종목별 그룹화 진행
    grouped = target_df.groupby("standardized_code")
    total_groups = len(grouped)
    print(f"업데이트 대상 종목 수: {total_groups}개")

    program_col_idx = _get_column_index(ws, PROGRAM_COL)
    updates = []
    processed_rows = 0

    for g_idx, (code, group) in enumerate(grouped):
        try:
            dates_in_group = group["parsed_date"].dropna()
            if dates_in_group.empty:
                continue

            # 날짜 범위 산출
            min_date_dt = dates_in_group.min()
            max_date_dt = dates_in_group.max()
            min_date_str = min_date_dt.strftime("%Y%m%d")
            max_date_str = max_date_dt.strftime("%Y%m%d")
            target_dates_list = [d.strftime("%Y%m%d") for d in dates_in_group]

            print(
                f"[{g_idx+1}/{total_groups}] 종목 {code} 프로그램 수급 조회 중... "
                f"기간: {min_date_str} ~ {max_date_str} (누락 날짜: {len(target_dates_list)}개)"
            )

            # KIS OpenAPI를 이용한 프로그램 데이터 일괄 조회
            prog_map = {}
            try:
                prog_map = get_program_history(
                    code=code,
                    start_date=min_date_str,
                    end_date=max_date_str,
                    target_dates=target_dates_list,
                )
            except Exception as e:
                print(f"[warn] KIS 프로그램 매매 조회 실패 for {code}: {e}")

            if prog_map:
                # 데이터 채우기
                for i, row in group.iterrows():
                    d_val = row["parsed_date"]
                    d_str = d_val.strftime("%Y%m%d")

                    if d_str in prog_map:
                        program_amt = prog_map[d_str]

                        df.at[i, PROGRAM_COL] = program_amt

                        # gspread row index는 1-based이며 헤더 행 때문에 i + 2
                        updates.append((i + 2, program_col_idx, program_amt))
                        processed_rows += 1

                print(
                    f"-> 종목 {code} 프로그램 매핑 완료: {len(group)}개 행 중 "
                    f"{len([d for d in target_dates_list if d in prog_map])}개 반영 성공"
                )
            else:
                print(f"-> 종목 {code} 데이터 수집 실패 또는 데이터 없음")

            # KIS API 과부하 방지 및 안정성 보장
            time.sleep(0.2)

            # 100개 행(100개 셀)마다 구글 시트 실시간 저장 (중단/API 제한 대비)
            if len(updates) >= 100:
                _batch_update_cells(ws, updates)
                updates.clear()
                print(
                    f"--> [Checkpoint] {processed_rows}/{total_count} "
                    "Google Sheets 실시간 저장 완료."
                )

        except Exception as e:
            print(f"Error processing stock {code}: {e}")
            continue

    # 남은 잔여 업데이트 반영
    if updates:
        _batch_update_cells(ws, updates)

    print("\n모든 업데이트 완료! Google Sheets(Trade2)에 반영했습니다.")


if __name__ == "__main__":
    fill_program_net_buy()
