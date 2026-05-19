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
    # 비동기 함수 임포트
    from src.sync.investor_data import get_investor_trade_daily_async
except ImportError:
    from sync.investor_data import get_investor_trade_daily_async  # type: ignore[no-redef]


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
    print("Google Sheets(Trade2)에서 메타데이터 로드 중...")
    records = manager.get_all_records(GOOGLE_SHEET_NAME, WORKSHEET_NAME)

    sh = manager.get_spreadsheet(GOOGLE_SHEET_NAME)
    ws = sh.worksheet(WORKSHEET_NAME)

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("가져온 데이터가 없습니다.")

    return df, ws


def _get_column_index(ws: Any, column_name: str) -> int:
    """헤더 행에서 열 번호(1-based) 확인"""
    headers: list[str] = ws.row_values(1)
    if column_name not in headers:
        raise KeyError(f"시트에 '{column_name}' 열이 없습니다.")
    return headers.index(column_name) + 1


@retry_on_quota_limit()
def _batch_update_cells(ws: Any, updates: list[tuple[int, int, Any]]) -> None:
    """지정된 셀만 부분 업데이트. updates: List[Tuple[row_idx, col_idx, value]]"""
    if not updates:
        return

    data = []
    for row_idx, col_idx, value in updates:
        rng = rowcol_to_a1(row_idx, col_idx)
        if pd.isna(value):
            val = ""
        elif hasattr(value, "item"):
            val = value.item()
        else:
            val = value
        data.append({"range": rng, "values": [[val]]})

    ws.batch_update(data)


async def main() -> None:
    # 1. 시트에서 데이터 로드
    df, ws = _load_sheet_dataframe()

    # 빈 문자열을 NaN으로 변환해 결측 탐지
    df.replace("", pd.NA, inplace=True)
    required_cols = [INST_COL, FRGN_COL, CODE_COL, DATE_COL]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"시트에 '{col}' 열이 없습니다.")

    # 2. 업데이트할 대상 행 식별
    target_mask = (df[INST_COL].isna() | df[FRGN_COL].isna()) & df[CODE_COL].notna()
    target_indices = df[target_mask].index
    if not target_indices.any():
        print("업데이트할 결측 데이터가 없습니다.")
        return

    print(f"업데이트 대상 행 개수: {len(target_indices)}개")

    target_df = df[target_mask].copy()
    target_df["standardized_code"] = target_df[CODE_COL].apply(
        lambda x: str(x).split(".")[0].zfill(6)
    )
    target_df["parsed_date"] = pd.to_datetime(target_df[DATE_COL])

    grouped = target_df.groupby("standardized_code")
    print(f"업데이트 대상 종목 수: {len(grouped)}개")

    inst_col_idx = _get_column_index(ws, INST_COL)
    frgn_col_idx = _get_column_index(ws, FRGN_COL)
    
    # 3. KIS API 데이터 병렬 수집
    client = KisApiClient()
    async with client.create_session() as session:
        await client.ensure_token(session)
        
        async def sem_task(code, min_date_str, max_date_str, target_dates_list):
            async with client.semaphore:
                try:
                    res = await get_investor_trade_daily_async(
                        session=session,
                        client=client,
                        code=code,
                        start_date=min_date_str,
                        end_date=max_date_str,
                        target_dates=target_dates_list,
                    )
                    # 각 API 호출 후 미세한 지연을 두어 TPS(20)를 절대 넘지 않도록 함
                    await asyncio.sleep(0.05)
                    return res
                except Exception as e:
                    print(f"\n[Error] {code} 조회 중 예외 발생: {e}")
                    return None

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

        print(f"\nKIS API에서 수급 데이터 조회 중 (총 {len(all_coros)}개 종목, 청크 단위 병렬 처리)...")
        
        # 10개씩 청크로 나누어 처리하여 DNS 부하 및 TPS 관리
        CHUNK_SIZE = 10
        results = []
        
        with tqdm(total=len(all_coros), desc="수급 데이터 조회") as pbar:
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
    processed_rows = 0
    
    # 각 종목 그룹에 대해 결과 매핑
    for (code, group), df_net in zip(grouped, results):
        if df_net is None or df_net.empty:
            # print(f"-> 종목 {code} 데이터 수집 실패 또는 데이터 없음")
            continue

        net_dict = {
            net_row["date"].strftime("%Y%m%d"): (
                net_row.get("inst_netbuy", 0.0),
                net_row.get("foreign_netbuy", 0.0),
            )
            for _, net_row in df_net.iterrows()
        }

        # 데이터 채우기
        for i, row in group.iterrows():
            d_str = row["parsed_date"].strftime("%Y%m%d")
            if d_str in net_dict:
                inst_net, foreign_net = net_dict[d_str]

                inst_net = 0.0 if pd.isna(inst_net) else inst_net
                foreign_net = 0.0 if pd.isna(foreign_net) else foreign_net

                # gspread row index는 1-based이며 헤더 행 때문에 i + 2
                updates.append((i + 2, inst_col_idx, inst_net))
                updates.append((i + 2, frgn_col_idx, foreign_net))
                processed_rows += 1
    
    print(f"총 {processed_rows}개 행({len(updates)}개 셀)에 대한 업데이트 준비 완료.")

    # 5. Google Sheets 일괄 업데이트 (대용량 배치)
    if not updates:
        print("실제로 업데이트할 내용이 없습니다.")
        return
        
    BATCH_SIZE = 5000  # 한 번에 업데이트할 셀의 개수 (2,500개 행에 해당)
    print(f"\nGoogle Sheets에 데이터 일괄 업데이트 시작 (배치 크기: {BATCH_SIZE} 셀)...")
    
    for i in range(0, len(updates), BATCH_SIZE):
        chunk = updates[i : i + BATCH_SIZE]
        try:
            _batch_update_cells(ws, chunk)
            print(f"  - {i + len(chunk)}/{len(updates)}개 셀 업데이트 완료.")
            
            # 마지막 청크가 아니면, API Quota를 준수하기 위해 1초 대기
            if i + BATCH_SIZE < len(updates):
                time.sleep(1)

        except Exception as e:
            print(f"!! Google Sheets 업데이트 중 오류 발생 (청크 {i}~{i+BATCH_SIZE}): {e}")
            print("  - 해당 청크는 건너뛰고 다음 작업을 계속합니다.")
            continue

    print("\n모든 작업이 완료되었습니다.")


def fill_missing_stock_data() -> None:
    """비동기 main 함수를 실행하는 동기 래퍼"""
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    fill_missing_stock_data()
