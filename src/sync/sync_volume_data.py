"""Google Sheets(Trade/Trade2)의 누락된 거래량(Volume) 데이터를 pykrx를 통해 수집하여 동기화하는 스크립트.

기존 수식을 완전히 보존하기 위해 변경된 셀만 배치(Batch) 업데이트 방식으로 최적화하여 작성되었습니다.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd
from gspread.utils import rowcol_to_a1

from src import settings
from src.data.gsheet_loader import GSheetClientManager, retry_on_quota_limit

try:
    from pykrx import stock
except ImportError:
    stock = None

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GOOGLE_KEY_PATH: str = str(settings.GOOGLE_KEY_PATH)
GOOGLE_SHEET_NAME: str = settings.GOOGLE_SHEET_NAME

DATE_COL: str = settings.GOTTEN_COLS["DATE"]
CODE_COL: str = settings.GOTTEN_COLS["CODE"]
VOLUME_ALIASES: list[str] = ["(거래량)", "거래량", "volume"]


def _get_volume_col_name(headers: list[str]) -> str:
    """헤더 목록에서 거래량 열 명칭을 추출합니다.

    Args:
        headers: 시트의 첫 번째 행 헤더 목록.

    Returns:
        매칭된 거래량 열의 이름.

    Raises:
        KeyError: 정의된 거래량 별칭 중 매칭되는 열이 없는 경우.

    """
    for alias in VOLUME_ALIASES:
        if alias in headers:
            return alias
    raise KeyError(f"시트에 {VOLUME_ALIASES} 중 일치하는 열이 없습니다.")


def _find_volume_col(df: pd.DataFrame) -> str | None:
    """pykrx 결과 DataFrame에서 거래량 열 이름을 찾습니다.

    Args:
        df: pykrx 조회 결과 DataFrame.

    Returns:
        찾은 거래량 열 이름 또는 None.

    """
    for col in df.columns:
        if "거래량" in str(col) or "volume" in str(col).lower():
            return str(col)
    return None


@retry_on_quota_limit()
def _batch_update_cells(ws: Any, updates: list[tuple[int, int, Any]]) -> None:
    """구글 시트의 지정된 셀들만 부분 업데이트하여 기존 수식을 보존합니다.

    Args:
        ws: gspread worksheet 객체.
        updates: (행 번호, 열 번호, 값) 튜플의 리스트.

    """
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


def sync_missing_volume_data() -> None:
    """매수날짜 및 종목코드를 기반으로 구글 시트의 누락된 거래량 데이터를 수집 및 채웁니다.

    Time Complexity:
        O(N) - N은 업데이트 대상 종목 그룹 수. 각 종목 그룹별로 단일 pykrx API 조회를 수행합니다.
    Space Complexity:
        O(M) - M은 구글 시트에서 로드한 전체 레코드 수.
    """
    logger.info("Google Sheets에서 거래량 동기화 작업을 시작합니다.")
    
    if not os.path.exists(GOOGLE_KEY_PATH):
        raise FileNotFoundError(f"인증 키 파일이 존재하지 않습니다: {GOOGLE_KEY_PATH}")

    manager = GSheetClientManager(GOOGLE_KEY_PATH)
    sh = manager.get_spreadsheet(GOOGLE_SHEET_NAME)

    for worksheet_name in settings.TRADE_WORKSHEETS:
        logger.info(f"Worksheet '{worksheet_name}' 처리 중...")
        try:
            records = manager.get_all_records(GOOGLE_SHEET_NAME, worksheet_name)
            ws = sh.worksheet(worksheet_name)
        except Exception as e:
            logger.warning(f"시트 '{worksheet_name}' 로드 실패: {e}")
            continue

        df = pd.DataFrame(records)
        if df.empty:
            logger.info(f"시트 '{worksheet_name}'가 비어 있어 건너뜁니다.")
            continue

        # 빈 문자열을 NaN으로 변환하여 결측 식별 가능하도록 가공
        df.replace("", pd.NA, inplace=True)

        headers: list[str] = ws.row_values(1)
        try:
            volume_col = _get_volume_col_name(headers)
        except KeyError as e:
            logger.warning(f"시트 '{worksheet_name}'에서 거래량 열 식별 실패: {e}")
            continue

        # 필수 컬럼 검증
        required_cols = [CODE_COL, DATE_COL]
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            logger.warning(f"시트 '{worksheet_name}'에 필수 열 {missing_required}이 누락되었습니다.")
            continue

        # 업데이트 대상 행 필터링 (거래량이 비어 있고, 종목코드와 날짜가 기입된 행)
        target_mask = df[volume_col].isna() & df[CODE_COL].notna() & df[DATE_COL].notna()
        target_indices = df[target_mask].index
        total_count = len(target_indices)
        logger.info(f"시트 '{worksheet_name}' 내 결측 거래량 행 수: {total_count}개")

        if total_count == 0:
            logger.info(f"시트 '{worksheet_name}'에 업데이트할 데이터가 없습니다.")
            continue

        # 대상 데이터프레임 복사 및 정밀 가공 (종목코드 6자리 zfill 및 날짜 파싱)
        target_df_copy = df[target_mask].copy()
        target_df_copy["standardized_code"] = target_df_copy[CODE_COL].apply(
            lambda x: str(x).split(".")[0].zfill(6)
        )
        target_df_copy["parsed_date"] = pd.to_datetime(target_df_copy[DATE_COL])

        # 종목 단위 그룹화 진행
        grouped = target_df_copy.groupby("standardized_code")
        total_groups = len(grouped)
        logger.info(f"업데이트 대상 종목 수: {total_groups}개")

        volume_col_idx = headers.index(volume_col) + 1
        updates: list[tuple[int, int, Any]] = []
        processed_rows = 0

        for g_idx, (code, group) in enumerate(grouped):
            try:
                dates_in_group = group["parsed_date"].dropna()
                if dates_in_group.empty:
                    continue

                min_date_dt = dates_in_group.min()
                max_date_dt = dates_in_group.max()
                min_date_str = min_date_dt.strftime("%Y%m%d")
                max_date_str = max_date_dt.strftime("%Y%m%d")
                target_dates_list = [d.strftime("%Y%m%d") for d in dates_in_group]

                logger.info(
                    f"[{g_idx+1}/{total_groups}] 종목 {code} 거래량 조회 기간: "
                    f"{min_date_str} ~ {max_date_str} (누락 날짜 수: {len(target_dates_list)}개)"
                )

                df_ohlcv = pd.DataFrame()
                if stock is not None:
                    try:
                        df_ohlcv = stock.get_market_ohlcv_by_date(min_date_str, max_date_str, code)
                    except Exception as e:
                        logger.warning(f"pykrx 거래량 조회 실패 for {code}: {e}")
                else:
                    logger.warning("pykrx 패키지를 불러올 수 없어 조회를 건너뜁니다.")

                if df_ohlcv is not None and not df_ohlcv.empty:
                    vol_col_name = _find_volume_col(df_ohlcv)
                    if not vol_col_name:
                        logger.warning(f"종목 {code} pykrx 결과에서 거래량 컬럼 탐지 실패.")
                        continue

                    # 빠른 검색 매핑을 위한 딕셔너리 변환 (Key: YYYYMMDD -> Val: 거래량)
                    vol_dict: dict[str, int] = {}
                    for dt_val, row_ohlcv in df_ohlcv.iterrows():
                        d_str = pd.to_datetime(dt_val).strftime("%Y%m%d")
                        try:
                            raw_vol = row_ohlcv[vol_col_name]
                            vol_dict[d_str] = round(float(raw_vol)) if pd.notna(raw_vol) else 0
                        except Exception:
                            vol_dict[d_str] = 0

                    # 그룹 소속 행 데이터 업데이트 구성
                    for idx, row in group.iterrows():
                        d_val = row["parsed_date"]
                        d_str = d_val.strftime("%Y%m%d")

                        if d_str in vol_dict:
                            volume_val = vol_dict[d_str]
                            df.at[idx, volume_col] = volume_val

                            # gspread 행 번호는 1-based이며 헤더 행 감안하여 idx + 2
                            updates.append((int(idx) + 2, volume_col_idx, volume_val))
                            processed_rows += 1

                    mapped_count = len([d for d in target_dates_list if d in vol_dict])
                    logger.info(f"-> 종목 {code} 매핑 완료: {len(group)}개 중 {mapped_count}개 반영 완료.")
                else:
                    logger.warning(f"-> 종목 {code}의 pykrx 거래량 데이터가 조회되지 않았습니다.")

                # 과부하 방지용 슬립
                time.sleep(0.2)

                # 구글 시트 일괄 전송 (100개 셀 적재 시 배치 전송)
                if len(updates) >= 100:
                    _batch_update_cells(ws, updates)
                    updates.clear()
                    logger.info(f"--> [Checkpoint] {processed_rows}/{total_count} Google Sheets 부분 동기화 완료.")

            except Exception as e:
                logger.error(f"종목 {code} 처리 중 오류 발생: {e}", exc_info=True)
                continue

        # 미전송 잔여 셀 전송
        if updates:
            _batch_update_cells(ws, updates)
            logger.info(f"시트 '{worksheet_name}'의 최종 업데이트 완료.")

    logger.info("모든 시트의 거래량 동기화 작업이 성공적으로 종료되었습니다.")


if __name__ == "__main__":
    sync_missing_volume_data()
