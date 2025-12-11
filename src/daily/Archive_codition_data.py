# -*- coding: utf-8 -*-
import os
import sqlite3
from datetime import datetime

import pandas as pd

TARGET_CONDITION_NAME = "종가매매"
TABLE_NAME = "condition_history"
SNAP_DATE_COL = "스냅샷_날짜"
STOCK_CODE_COL = "종목코드"
RANK_COL = "순위"
# 조회하고 싶을 때 YYYY-MM-DD 형식으로 지정.
FETCH_TARGET_DATE = None  # "2025-12-09"


def upsert_history(df: pd.DataFrame, db_path: str) -> None:
    """Append snapshot rows to SQLite and deduplicate by 날짜/종목코드."""
    with sqlite3.connect(db_path) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

        if STOCK_CODE_COL in df.columns:
            # 동일 날짜/종목코드는 최신(rowid)만 유지
            conn.execute(
                f"""
                DELETE FROM {TABLE_NAME}
                WHERE rowid NOT IN (
                    SELECT MAX(rowid)
                    FROM {TABLE_NAME}
                    GROUP BY "{SNAP_DATE_COL}", "{STOCK_CODE_COL}"
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_date_code
                ON {TABLE_NAME} ("{SNAP_DATE_COL}", "{STOCK_CODE_COL}")
                """
            )

        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_date
            ON {TABLE_NAME} ("{SNAP_DATE_COL}")
            """
        )
        conn.commit()


# 기존에 저장해두던 데이터 DB로 마이그레이션
def import_csv_history_if_needed(history_csv: str, history_db: str) -> None:
    """Import legacy CSV history into SQLite if DB is empty or missing dates."""
    if not os.path.exists(history_csv):
        return

    # CSV에 있는 날짜 집합
    try:
        csv_dates = set(
            pd.read_csv(history_csv, usecols=[SNAP_DATE_COL])[SNAP_DATE_COL]
            .dropna()
            .unique()
        )
    except ValueError:
        print(
            f"[warn] CSV에 '{SNAP_DATE_COL}' 컬럼이 없어 import를 건너뜁니다: {history_csv}"
        )
        return

    with sqlite3.connect(history_db) as conn:
        table_exists = (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (TABLE_NAME,),
            ).fetchone()
            is not None
        )
        if not table_exists:
            db_dates = set()
        else:
            db_dates = {
                row[0]
                for row in conn.execute(
                    f'SELECT DISTINCT "{SNAP_DATE_COL}" FROM {TABLE_NAME}'
                ).fetchall()
                if row[0] is not None
            }

    missing_dates = csv_dates - db_dates
    if not table_exists and not csv_dates:
        print(
            f"[warn] CSV에 '{SNAP_DATE_COL}' 데이터가 비어 import를 건너뜁니다: {history_csv}"
        )
        return
    if not missing_dates and table_exists:
        return

    df_csv = pd.read_csv(history_csv)
    if SNAP_DATE_COL not in df_csv.columns:
        print(
            f"[warn] CSV에 '{SNAP_DATE_COL}' 컬럼이 없어 import를 건너뜁니다: {history_csv}"
        )
        return

    # 기존 CSV에 시간 컬럼이 있어도 제거
    if "스냅샷_시간" in df_csv.columns:
        df_csv = df_csv.drop(columns=["스냅샷_시간"])

    upsert_history(df_csv, history_db)
    print(f"[done] 기존 CSV 히스토리를 SQLite로 마이그레이션: {history_db}")


def fetch_date_rows(date_str: str, history_db: str) -> pd.DataFrame:
    """Return rows for a given date, excluding 스냅샷_시간 column if present."""
    if not os.path.exists(history_db):
        raise FileNotFoundError(f"DB not found: {history_db}")

    with sqlite3.connect(history_db) as conn:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_NAME})")]
        if not cols:
            raise RuntimeError(f"Table {TABLE_NAME} not found in {history_db}")
        cols = [c for c in cols if c != "스냅샷_시간"]
        col_clause = ", ".join(f'"{c}"' for c in cols)
        # 순위 컬럼이 있으면 순위 기준 오름차순, 없으면 종목코드로 정렬
        if RANK_COL in cols:
            order_clause = f'"{RANK_COL}"'
        else:
            order_clause = f'"{STOCK_CODE_COL}"'
        query = (
            f"SELECT {col_clause} FROM {TABLE_NAME} "
            f'WHERE "{SNAP_DATE_COL}" = ? '
            f"ORDER BY {order_clause}"
        )
        return pd.read_sql(query, conn, params=[date_str])


def main():
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_dir = os.path.join(project_root, "data")
    history_dir = os.path.join(data_dir, "history")
    os.makedirs(history_dir, exist_ok=True)

    clean_name = TARGET_CONDITION_NAME.replace("/", "_").replace("\\", "_")
    latest_path = os.path.join(data_dir, f"condition_{clean_name}.xlsx")
    history_csv = os.path.join(history_dir, f"condition_history_{clean_name}.csv")
    history_db = os.path.join(history_dir, f"condition_history_{clean_name}.db")

    import_csv_history_if_needed(history_csv, history_db)

    if FETCH_TARGET_DATE:
        df = fetch_date_rows(FETCH_TARGET_DATE, history_db)
        if df.empty:
            print(f"[info] 조회된 데이터가 없습니다: {FETCH_TARGET_DATE}")
        else:
            target_file = os.path.join(
                history_dir, f"condition_{clean_name}_{FETCH_TARGET_DATE}.xlsx"
            )
            df.to_excel(target_file, index=False)
            print(f"[done] 조회 결과를 저장했습니다: {target_file}")
        return

    if not os.path.exists(latest_path):
        print(f"[skip] 최신 파일이 없습니다: {latest_path}")
        return

    # 전날 결과 불러오기
    df = pd.read_excel(latest_path)

    # 파일 수정 시각을 스냅샷 시각으로 사용 (없으면 현재 시각)
    snap_dt = datetime.fromtimestamp(os.path.getmtime(latest_path))
    df.insert(0, SNAP_DATE_COL, snap_dt.strftime("%Y-%m-%d"))

    upsert_history(df, history_db)
    print(f"[done] SQLite 누적 저장: {history_db}")


if __name__ == "__main__":
    main()
