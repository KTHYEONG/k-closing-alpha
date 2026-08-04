"""sync/* 모듈 간 중복 헬퍼 통합 모듈.

Google Sheets 메타데이터 로드, 컬럼 인덱스 조회, 배치 셀 업데이트 로직을
단일 지점으로 추출하여 sync/ 내 모듈들이 재사용합니다.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from gspread.utils import rowcol_to_a1

from src import settings
from src.data.gsheet_loader import GSheetClientManager, retry_on_quota_limit

logger = logging.getLogger(__name__)

GOOGLE_SHEET_NAME = settings.GOOGLE_SHEET_NAME


def load_sheet_dataframe(
    manager: GSheetClientManager,
    worksheet_name: str,
    *,
    sheet_name: str = GOOGLE_SHEET_NAME,
) -> tuple[pd.DataFrame, Any]:
    """특정 시트를 DataFrame 및 Worksheet 객체와 함께 로드합니다."""
    logger.info("Google Sheets(%s)에서 데이터 로드 중...", worksheet_name)
    records = manager.get_all_records(sheet_name, worksheet_name)
    sh = manager.get_spreadsheet(sheet_name)
    ws = sh.worksheet(worksheet_name)
    df = pd.DataFrame(records)
    return df, ws


def get_column_index(ws: Any, column_name: str) -> int:
    """헤더 행에서 열 번호(1-based)를 확인합니다."""
    headers: list[str] = ws.row_values(1)
    if column_name not in headers:
        raise KeyError(f"시트에 '{column_name}' 열이 없습니다.")
    return headers.index(column_name) + 1


@retry_on_quota_limit()
def batch_update_cells(ws: Any, updates: list[Any]) -> None:
    """지정된 셀만 부분 업데이트합니다.

    Args:
        updates: 두 가지 형식을 지원합니다.
            - List[Tuple[row_idx, col_idx, value]]: 셀 좌표 기반 업데이트
            - List[Dict[str, Any]]: 이미 A1 range로 구성된 update payload
    """
    if not updates:
        return

    if isinstance(updates[0], dict):
        ws.batch_update(updates)
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
