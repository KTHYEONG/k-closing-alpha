"""Google Sheets(Trade/Trade2)의 누락된 거래량(Volume) 데이터를 pykrx를 통해 수집하여 동기화하는 스크립트.

비동기 병렬 처리 및 배치 업데이트 최적화를 통해 성능을 개선했습니다.
pykrx의 동기적 한계를 극복하기 위해 asyncio.to_thread를 사용합니다.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import pandas as pd
from gspread.utils import rowcol_to_a1
from tqdm.asyncio import tqdm

from src import settings
from src.api.kis_client import KisApiClient
from src.data.gsheet_loader import GSheetClientManager, retry_on_quota_limit

try:
    from pykrx import stock
except ImportError:
    stock = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GOOGLE_KEY_PATH: str = str(settings.GOOGLE_KEY_PATH)
GOOGLE_SHEET_NAME: str = settings.GOOGLE_SHEET_NAME

DATE_COL: str = settings.GOTTEN_COLS["DATE"]
CODE_COL: str = settings.GOTTEN_COLS["CODE"]
VOLUME_ALIASES: list[str] = ["(거래량)", "거래량", "volume"]


def _get_volume_col_name(headers: list[str]) -> str:
    for alias in VOLUME_ALIASES:
        if alias in headers:
            return alias
    raise KeyError(f"시트에 {VOLUME_ALIASES} 중 일치하는 열이 없습니다.")


def _find_volume_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if "거래량" in str(col) or "volume" in str(col).lower():
            return str(col)
    return None


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


async def fetch_volume_data_for_stock(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """pykrx 동기 함수를 별도 스레드에서 실행하여 비동기적으로 거래량 데이터를 가져옵니다."""
    if stock is None:
        return pd.DataFrame()
    try:
        # asyncio.to_thread를 사용해 동기 함수를 논블로킹 방식으로 호출
        loop = asyncio.get_running_loop()
        df_ohlcv = await loop.run_in_executor(
            None, stock.get_market_ohlcv_by_date, start_date, end_date, code
        )
        return df_ohlcv
    except Exception as e:
        logger.warning(f"pykrx 거래량 조회 중 예외 발생 for {code}: {e}")
        return pd.DataFrame()


async def main() -> None:
    """비동기 메인 로직"""
    logger.info("Google Sheets에서 거래량 동기화 작업을 시작합니다.")
    if not os.path.exists(GOOGLE_KEY_PATH):
        raise FileNotFoundError(f"인증 키 파일이 존재하지 않습니다: {GOOGLE_KEY_PATH}")

    manager = GSheetClientManager(GOOGLE_KEY_PATH)
    sh = manager.get_spreadsheet(GOOGLE_SHEET_NAME)
    
    all_updates: dict[str, list] = {ws_name: [] for ws_name in settings.TRADE_WORKSHEETS}
    all_ws_objects: dict[str, Any] = {}

    for worksheet_name in settings.TRADE_WORKSHEETS:
        logger.info(f"Worksheet '{worksheet_name}' 처리 중...")
        try:
            records = manager.get_all_records(GOOGLE_SHEET_NAME, worksheet_name)
            ws = sh.worksheet(worksheet_name)
            all_ws_objects[worksheet_name] = ws
        except Exception as e:
            logger.warning(f"시트 '{worksheet_name}' 로드 실패: {e}")
            continue

        df = pd.DataFrame(records)
        if df.empty:
            logger.info(f"시트 '{worksheet_name}'가 비어 있어 건너뜁니다.")
            continue

        df.replace("", pd.NA, inplace=True)
        headers: list[str] = ws.row_values(1)
        try:
            volume_col = _get_volume_col_name(headers)
        except KeyError as e:
            logger.warning(f"시트 '{worksheet_name}'에서 거래량 열 식별 실패: {e}")
            continue

        required_cols = [CODE_COL, DATE_COL]
        if any(col not in df.columns for col in required_cols):
            logger.warning(f"시트 '{worksheet_name}'에 필수 열 {required_cols}이 누락되었습니다.")
            continue

        target_mask = df[volume_col].isna() & df[CODE_COL].notna() & df[DATE_COL].notna()
        if not target_mask.any():
            logger.info(f"시트 '{worksheet_name}'에 업데이트할 데이터가 없습니다.")
            continue
        
        logger.info(f"시트 '{worksheet_name}' 내 결측 거래량 행 수: {target_mask.sum()}개")

        target_df_copy = df[target_mask].copy()
        target_df_copy["standardized_code"] = target_df_copy[CODE_COL].apply(
            lambda x: str(x).split(".")[0].zfill(6)
        )
        target_df_copy["parsed_date"] = pd.to_datetime(target_df_copy[DATE_COL])

        grouped = target_df_copy.groupby("standardized_code")
        logger.info(f"업데이트 대상 종목 수: {len(grouped)}개")

        volume_col_idx = headers.index(volume_col) + 1
        
        # 동시성 제한을 위한 KisApiClient (세마포어 활용)
        client = KisApiClient()
        
        async def sem_fetch_task(code, min_date_str, max_date_str):
            async with client.semaphore:
                try:
                    res = await fetch_volume_data_for_stock(code, min_date_str, max_date_str)
                    # 네트워크 부하 분산을 위한 미세 지연
                    await asyncio.sleep(0.1)
                    return res
                except Exception as e:
                    logger.warning(f"[Error] {code} 조회 중 예외 발생: {e}")
                    return pd.DataFrame()

        # 모든 종목에 대한 코루틴 생성
        all_coros = []
        for code, group in grouped:
            dates_in_group = group["parsed_date"].dropna()
            if dates_in_group.empty:
                continue
            min_date_str = dates_in_group.min().strftime("%Y%m%d")
            max_date_str = dates_in_group.max().strftime("%Y%m%d")
            all_coros.append(sem_fetch_task(code, min_date_str, max_date_str))

        logger.info(f"\npykrx에서 거래량 데이터 조회 중 (총 {len(all_coros)}개 종목, 청크 단위 병렬 처리)...")
        
        # 10개씩 청크로 나누어 처리하여 DNS 부하 관리
        CHUNK_SIZE = 10
        results = []
        
        with tqdm(total=len(all_coros), desc=f"{worksheet_name} 거래량 조회") as pbar:
            for i in range(0, len(all_coros), CHUNK_SIZE):
                chunk = all_coros[i : i + CHUNK_SIZE]
                chunk_results = await asyncio.gather(*chunk)
                results.extend(chunk_results)
                pbar.update(len(chunk))
                # 청크 사이의 명시적 휴식
                await asyncio.sleep(0.2) 

        for (code, group), df_ohlcv in zip(grouped, results):
            if df_ohlcv.empty:
                continue
                
            vol_col_name = _find_volume_col(df_ohlcv)
            if not vol_col_name:
                continue

            vol_dict = {
                pd.to_datetime(dt).strftime("%Y%m%d"): round(float(row[vol_col_name]))
                for dt, row in df_ohlcv.iterrows() if pd.notna(row[vol_col_name])
            }
            
            for idx, row in group.iterrows():
                d_str = row["parsed_date"].strftime("%Y%m%d")
                if d_str in vol_dict:
                    volume_val = vol_dict[d_str]
                    all_updates[worksheet_name].append((int(idx) + 2, volume_col_idx, volume_val))

    # 모든 시트의 모든 업데이트를 한 번에 처리
    logger.info("\n모든 데이터 조회 완료. Google Sheets 업데이트 시작...")
    for ws_name, updates in all_updates.items():
        if not updates:
            logger.info(f"시트 '{ws_name}'에 업데이트할 내용이 없습니다.")
            continue
        
        ws = all_ws_objects.get(ws_name)
        if not ws:
            continue
            
        BATCH_SIZE = 5000
        logger.info(f"시트 '{ws_name}'에 {len(updates)}개 셀 업데이트 중 (배치 크기: {BATCH_SIZE})...")
        
        for i in range(0, len(updates), BATCH_SIZE):
            chunk = updates[i : i + BATCH_SIZE]
            try:
                _batch_update_cells(ws, chunk)
                logger.info(f"  - {i + len(chunk)}/{len(updates)}개 셀 업데이트 완료.")
                if i + BATCH_SIZE < len(updates):
                    time.sleep(1)
            except Exception as e:
                logger.error(f"!! Google Sheets 업데이트 중 오류 발생 (청크 {i}~{i+BATCH_SIZE}): {e}")
                continue

    logger.info("\n모든 시트의 거래량 동기화 작업이 성공적으로 종료되었습니다.")


def sync_missing_volume_data() -> None:
    """비동기 main 함수를 실행하는 동기 래퍼"""
    if stock is None:
        logger.error("pykrx 라이브러리가 설치되지 않았습니다. 'pip install pykrx'로 설치해주세요.")
        return
    import sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"작업 중 오류 발생: {e}", exc_info=True)


if __name__ == "__main__":
    sync_missing_volume_data()
