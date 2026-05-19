import asyncio
import os
import time
from typing import Any

import aiohttp
import pandas as pd
from gspread.utils import rowcol_to_a1
from tqdm.asyncio import tqdm

from src import settings
from src.api.kis_client import KisApiClient
from src.data.gsheet_loader import GSheetClientManager, retry_on_quota_limit

try:
    from src.sync.program_data import get_program_history_async
except ImportError:
    from sync.program_data import get_program_history_async  # type: ignore[no-redef]


GOOGLE_KEY_PATH = str(settings.GOOGLE_KEY_PATH)
GOOGLE_SHEET_NAME = settings.GOOGLE_SHEET_NAME
WORKSHEET_NAME = "Trade2"

DATE_COL = settings.GOTTEN_COLS["DATE"]
CODE_COL = settings.GOTTEN_COLS["CODE"]
PROGRAM_COL = settings.GOTTEN_COLS["PROGRAM"]


def _load_sheet_dataframe() -> tuple[pd.DataFrame, Any]:
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


def _get_column_index(ws: Any, column_name: str) -> int:
    headers: list[str] = ws.row_values(1)
    if column_name not in headers:
        raise KeyError(f"시트에 '{column_name}' 열이 없습니다.")
    return headers.index(column_name) + 1


@retry_on_quota_limit()
def _batch_update_cells(ws: Any, updates: list[tuple[int, int, Any]]) -> None:
    if not updates:
        return
    data = [
        {
            "range": rowcol_to_a1(row_idx, col_idx),
            "values": [[value.item() if hasattr(value, "item") else ("" if pd.isna(value) else value)]],
        }
        for row_idx, col_idx, value in updates
    ]
    ws.batch_update(data)


async def main() -> None:
    """비동기 메인 로직"""
    print("Google Sheets(Trade2)에서 데이터 불러오는 중...")
    df, ws = _load_sheet_dataframe()
    df.replace("", pd.NA, inplace=True)

    for col in [DATE_COL, CODE_COL]:
        if col not in df.columns:
            raise KeyError(f"시트에 '{col}' 열이 없습니다.")
    if PROGRAM_COL not in df.columns:
        df[PROGRAM_COL] = pd.NA

    target_mask = df[PROGRAM_COL].isna() & df[CODE_COL].notna() & df[DATE_COL].notna()
    if not target_mask.any():
        print("채울 데이터가 없습니다.")
        return
    
    print(f"업데이트 대상 행 수: {target_mask.sum()}개")
    target_df = df[target_mask].copy()
    target_df["standardized_code"] = target_df[CODE_COL].apply(lambda x: str(x).split(".")[0].zfill(6))
    target_df["parsed_date"] = pd.to_datetime(target_df[DATE_COL])

    grouped = target_df.groupby("standardized_code")
    print(f"업데이트 대상 종목 수: {len(grouped)}개")

    program_col_idx = _get_column_index(ws, PROGRAM_COL)
    
    # KIS API 데이터 병렬 수집
    client = KisApiClient()
    async with client.create_session() as session:
        await client.ensure_token(session)
        
        async def sem_task(code, min_date_str, max_date_str, target_dates_list):
            async with client.semaphore:
                try:
                    res = await get_program_history_async(
                        session, client, code, min_date_str, max_date_str, target_dates=target_dates_list
                    )
                    # 각 API 호출 후 미세한 지연을 두어 TPS(20)를 절대 넘지 않도록 함
                    await asyncio.sleep(0.05) 
                    return res
                except Exception as e:
                    print(f"\n[Error] {code} 조회 중 예외 발생: {e}")
                    return {}

        # 모든 종목에 대한 코루틴 생성
        all_coros = []
        for code, group in grouped:
            dates_in_group = group["parsed_date"].dropna()
            if dates_in_group.empty:
                continue
            min_date_str = dates_in_group.min().strftime("%Y%m%d")
            max_date_str = dates_in_group.max().strftime("%Y%m%d")
            target_dates_list = [d.strftime("%Y%m%d") for d in dates_in_group]
            all_coros.append(sem_task(code, min_date_str, max_date_str, target_dates_list))

        print(f"\nKIS API에서 프로그램 수급 데이터 조회 중 (총 {len(all_coros)}개 종목, 청크 단위 병렬 처리)...")
        
        # 10개씩 청크로 나누어 처리하여 DNS 부하 및 TPS 관리
        CHUNK_SIZE = 10
        results = []
        
        with tqdm(total=len(all_coros), desc="프로그램 수급 조회") as pbar:
            for i in range(0, len(all_coros), CHUNK_SIZE):
                chunk = all_coros[i : i + CHUNK_SIZE]
                chunk_results = await asyncio.gather(*chunk)
                results.extend(chunk_results)
                pbar.update(len(chunk))
                # 청크 사이의 명시적 휴식 (TPS 20 제한 준수)
                await asyncio.sleep(0.1) 

    # 결과 처리 및 업데이트 목록 생성
    print("\n조회된 데이터 처리 및 업데이트 목록 생성 중...")
    updates = []
    
    for (code, group), prog_map in zip(grouped, results):
        if not prog_map:
            continue
        
        for i, row in group.iterrows():
            d_str = row["parsed_date"].strftime("%Y%m%d")
            if d_str in prog_map:
                program_amt = prog_map[d_str]
                updates.append((i + 2, program_col_idx, program_amt))
    
    print(f"총 {len(updates)}개 셀에 대한 업데이트 준비 완료.")

    # Google Sheets 일괄 업데이트
    if not updates:
        print("실제로 업데이트할 내용이 없습니다.")
        return
        
    BATCH_SIZE = 5000
    print(f"\nGoogle Sheets에 데이터 일괄 업데이트 시작 (배치 크기: {BATCH_SIZE} 셀)...")
    
    for i in range(0, len(updates), BATCH_SIZE):
        chunk = updates[i : i + BATCH_SIZE]
        try:
            _batch_update_cells(ws, chunk)
            print(f"  - {i + len(chunk)}/{len(updates)}개 셀 업데이트 완료.")
            if i + BATCH_SIZE < len(updates):
                time.sleep(1)
        except Exception as e:
            print(f"!! Google Sheets 업데이트 중 오류 발생 (청크 {i}~{i+BATCH_SIZE}): {e}")
            continue

    print("\n모든 업데이트 완료!")


def fill_program_net_buy() -> None:
    """매수날짜/종목코드를 기반으로 (프로그램_순매수) 열을 채우는 작업의 동기 래퍼"""
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    fill_program_net_buy()
