import os
import time
from typing import Any

import pandas as pd
from gspread.utils import rowcol_to_a1

from src import settings
from src.data.gsheet_loader import GSheetClientManager, retry_on_quota_limit

try:
    from src.sync.investor_data import get_investor_trade_daily
except ImportError:
    from sync.investor_data import get_investor_trade_daily  # type: ignore[no-redef]


GOOGLE_KEY_PATH = str(settings.GOOGLE_KEY_PATH)
GOOGLE_SHEET_NAME = settings.GOOGLE_SHEET_NAME
WORKSHEET_NAME = "Trade2"

INST_COL = settings.GOTTEN_COLS["INST"]
FRGN_COL = settings.GOTTEN_COLS["FOREIGN"]
DATE_COL = settings.GOTTEN_COLS["DATE"]
CODE_COL = settings.GOTTEN_COLS["CODE"]


def _load_sheet_dataframe() -> tuple[pd.DataFrame, Any]:
    """Trade2 시트를 DataFrame으로 로드"""
    if not os.path.exists(GOOGLE_KEY_PATH):
        raise FileNotFoundError(f"인증 키 파일이 없습니다: {GOOGLE_KEY_PATH}")

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
    raise NotImplementedError("전체 시트 업데이트는 수식 보존을 위해 비활성화되었습니다.")


def _get_column_index(ws: Any, column_name: str) -> int:
    """헤더 행에서 열 번호(1-based) 확인"""
    headers: list[str] = ws.row_values(1)
    if column_name not in headers:
        raise KeyError(f"시트에 '{column_name}' 열이 없습니다.")
    return headers.index(column_name) + 1


@retry_on_quota_limit()
def _batch_update_cells(ws: Any, updates: list[tuple[int, int, Any]]) -> None:
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


def fill_missing_stock_data() -> None:
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
    target_mask = (df[INST_COL].isna() | df[FRGN_COL].isna()) & df[
        CODE_COL
    ].notna()

    target_indices = df[target_mask].index
    total_count = len(target_indices)
    print(f"업데이트 대상 행 개수: {total_count}개")

    if total_count == 0:
        print("업데이트할 결측 데이터가 없습니다.")
        return

    # 대상 데이터 프레임 복사 및 가공 (인덱스 보존)
    target_df = df[target_mask].copy()
    target_df["standardized_code"] = target_df[CODE_COL].apply(
        lambda x: str(x).split(".")[0].zfill(6)
    )
    target_df["parsed_date"] = pd.to_datetime(target_df[DATE_COL])

    # 종목별 그룹화 진행
    grouped = target_df.groupby("standardized_code")
    total_groups = len(grouped)
    print(f"업데이트 대상 종목 수: {total_groups}개")

    inst_col_idx = _get_column_index(ws, INST_COL)
    frgn_col_idx = _get_column_index(ws, FRGN_COL)
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
                f"[{g_idx+1}/{total_groups}] 종목 {code} 조회 중... "
                f"기간: {min_date_str} ~ {max_date_str} (누락 날짜: {len(target_dates_list)}개)"
            )

            # KIS OpenAPI를 이용한 수급 데이터 일괄 조회
            df_net = pd.DataFrame()
            try:
                df_net = get_investor_trade_daily(
                    code=code,
                    start_date=min_date_str,
                    end_date=max_date_str,
                    target_dates=target_dates_list,
                )
            except Exception as e:
                print(f"[warn] KIS 수급 조회 실패 for {code}: {e}")

            if df_net is not None and not df_net.empty:
                # 빠른 맵핑을 위해 dict 구조로 가공 (key: YYYYMMDD -> val: (inst_net, foreign_net))
                net_dict = {}
                for _, net_row in df_net.iterrows():
                    d_str = net_row["date"].strftime("%Y%m%d")
                    net_dict[d_str] = (
                        net_row.get("inst_netbuy", 0.0),
                        net_row.get("foreign_netbuy", 0.0),
                    )

                # 데이터 채우기
                for i, row in group.iterrows():
                    d_val = row["parsed_date"]
                    d_str = d_val.strftime("%Y%m%d")

                    if d_str in net_dict:
                        inst_net, foreign_net = net_dict[d_str]

                        if pd.isna(foreign_net):
                            foreign_net = 0.0
                        if pd.isna(inst_net):
                            inst_net = 0.0

                        df.at[i, INST_COL] = inst_net
                        df.at[i, FRGN_COL] = foreign_net

                        # gspread row index는 1-based이며 헤더 행 때문에 i + 2
                        updates.append((i + 2, inst_col_idx, inst_net))
                        updates.append((i + 2, frgn_col_idx, foreign_net))
                        processed_rows += 1

                print(
                    f"-> 종목 {code} 매핑 완료: {len(group)}개 행 중 "
                    f"{len([d for d in target_dates_list if d in net_dict])}개 반영 성공"
                )
            else:
                print(f"-> 종목 {code} 데이터 수집 실패 또는 데이터 없음")

            # KIS API 과부하 방지 및 안정성 보장
            time.sleep(0.2)

            # 100개 행(200개 셀)마다 구글 시트 실시간 저장 (중단/API 제한 대비)
            if len(updates) >= 200:
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

    print("\n모든 작업이 완료되었습니다. Google Sheets(Trade2)에 값만 업데이트했습니다.")


if __name__ == "__main__":
    fill_missing_stock_data()
