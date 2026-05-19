"""db_to_xlsx.py
-------------
SQLite DB의 condition_history 테이블 데이터를 xlsx 파일로 추출하는 유틸리티.

Usage:
    python src/utils/db_to_xlsx.py [--db DB_PATH] [--out OUTPUT_PATH] [--table TABLE_NAME]

Example:
    python src/utils/db_to_xlsx.py \
        --db "data/history/condition_history_종가매매.db" \
        --out "data/history/condition_history_종가매매.xlsx"

"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 기본 경로 상수 ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = _PROJECT_ROOT / "data" / "history" / "condition_history_종가매매.db"
DEFAULT_TABLE = "condition_history"
DEFAULT_OUT_PATH = _PROJECT_ROOT / "data" / "history" / "condition_history_종가매매.xlsx"


# ── DB 컬럼 추출 순서 정의 ──────────────────────────────────────────────────
# 사용자가 알려준 DB 내 실제 컬럼명과 순서입니다.
# 1~11번은 일반 명칭, 12번부터는 단위/괄호가 포함된 명칭입니다.
COLUMN_ORDER = [
    "스냅샷_날짜", "종목명", "종목코드", "시가", "고가", "저가", "종가", "전일종가",
    "등락률", "체결강도", "시장구분",
    "시가총액(억)", "거래대금(억)", "순위", "기관_순매수(억)", "외국인_순매수(억)",
    "프로그램_순매수(억)", "전체종목수", "평균거래대금(억)", "KOSPI등락률", "KOSDAQ등락률",
    "(v-kospi)", "(v-kosdaq)", "(거래량)", "(ema5)", "(ema10)", "(ema20)"
]


# ── 핵심 함수 ─────────────────────────────────────────────────────────────────
def export_table_to_xlsx(
    db_path: Path | str,
    out_path: Path | str,
    table: str = DEFAULT_TABLE,
    *,
    chunksize: int = 50_000,
) -> Path:
    """SQLite 테이블에서 지정된 컬럼 순서대로 데이터를 추출하여 xlsx로 저장한다."""
    db_path = Path(db_path).resolve()
    out_path = Path(out_path).resolve()

    if not db_path.exists():
        raise FileNotFoundError(f"DB 파일을 찾을 수 없습니다: {db_path}")

    logger.info("DB 연결: %s", db_path)
    conn = sqlite3.connect(db_path)

    try:
        # 1. 실제 DB 컬럼 목록 확인 (대소문자 및 오타 대응용)
        temp_df = pd.read_sql(f"SELECT * FROM [{table}] LIMIT 0", conn)
        actual_db_cols = temp_df.columns.tolist()
        
        # 2. COLUMN_ORDER에 정의된 이름 중 DB에 실제 존재하는 것만 필터링
        valid_cols = []
        for col_name in COLUMN_ORDER:
            if col_name in actual_db_cols:
                valid_cols.append(col_name)
            else:
                # partial match나 오타 가능성 체크
                found = False
                for actual in actual_db_cols:
                    if col_name.replace(" ", "") == actual.replace(" ", ""):
                        valid_cols.append(actual)
                        found = True
                        break
                if not found:
                    logger.warning("DB에 '%s' 컬럼이 존재하지 않아 건너뜁니다.", col_name)

        if not valid_cols:
            logger.error("추출 가능한 컬럼이 없습니다. 전체 컬럼을 대상으로 시도합니다.")
            query = f"SELECT * FROM [{table}]"
        else:
            # SQL에서 컬럼명에 대괄호 처리 ([시가총액(억)] 등 특수문자 대응)
            cols_sql = ", ".join([f"[{c}]" for c in valid_cols])
            query = f"SELECT {cols_sql} FROM [{table}]"

        logger.info("데이터 로드 중...")
        
        # 3. 데이터 로드 (청크 단위)
        chunks = list(pd.read_sql(query, conn, chunksize=chunksize))
        
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    finally:
        conn.close()

    if df.empty:
        logger.warning("데이터가 비어있습니다.")
        return out_path

    # 4. 엑셀 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")

    logger.info("변환 완료: %d rows saved to %s", len(df), out_path)
    return out_path


# ── CLI 진입점 ────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SQLite condition_history 테이블 → xlsx 추출기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="SQLite DB 파일 경로",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUT_PATH),
        help="저장할 xlsx 파일 경로",
    )
    parser.add_argument(
        "--table",
        type=str,
        default=DEFAULT_TABLE,
        help="추출할 테이블 이름",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50_000,
        help="청크 단위 행 수 (메모리 절약)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    saved_path = export_table_to_xlsx(
        db_path=args.db,
        out_path=args.out,
        table=args.table,
        chunksize=args.chunksize,
    )
    print(f"✅ 저장 완료: {saved_path}")


if __name__ == "__main__":
    main()
