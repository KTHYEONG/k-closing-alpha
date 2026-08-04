import logging
import os
import sqlite3
from datetime import datetime

import pandas as pd

from src import settings

logger = logging.getLogger(__name__)

TARGET_CONDITION_NAME = settings.TARGET_CONDITION_NAME
TABLE_NAME = "condition_history"
SNAP_DATE_COL = "스냅샷_날짜"
STOCK_CODE_COL = "종목코드"
RANK_COL = "순위"
# 조회하고 싶을 때 YYYY-MM-DD 형식으로 지정.
FETCH_TARGET_DATE = None  # "2025-12-09"

# 구글 스프레드시트(매매일지) 26개 열과 1:1 대응하는 표준 아카이브 컬럼 순서
ARCHIVE_COLUMN_ORDER = [
    "스냅샷_날짜",
    "종목코드",
    "종목명",
    "시가",
    "고가",
    "저가",
    "종가",
    "전일종가",
    "시가총액",
    "거래대금",
    "등락률",
    "선정순위",
    "기관_순매수",
    "외국인_순매수",
    "프로그램_순매수",
    "체결강도",
    "시장구분",
    "총_종목수",
    "평균_거래대금",
    "kospi",
    "kosdaq",
    "v_kospi",
    "v_kosdaq",
    "거래량",
    "테마_섹터",
    "시나리오",
]


def upsert_history(df: pd.DataFrame, db_path: str) -> None:
    """Append snapshot rows to SQLite and deduplicate by 날짜/종목코드."""
    with sqlite3.connect(db_path) as conn:
        # 신규 컬럼(예: 차트통과) 대응: 테이블이 이미 존재할 경우 누락된 컬럼을 추가
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (TABLE_NAME,),
        )
        if cursor.fetchone():
            existing_cols = [
                row[1] for row in cursor.execute(f"PRAGMA table_info({TABLE_NAME})")
            ]
            for col in df.columns:
                if col not in existing_cols:
                    # 특수문자가 포함된 컬럼명을 위해 쌍따옴표로 감싸서 ALTER TABLE 실행
                    conn.execute(f'ALTER TABLE {TABLE_NAME} ADD COLUMN "{col}"')

        df.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

        if STOCK_CODE_COL in df.columns:
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

    # Parquet 아카이브 저장
    try:
        from src.data.parquet_loader import upsert_condition_parquet
        upsert_condition_parquet(df)
    except Exception as e:
        logger.error("Parquet 조건검색 아카이브 저장 오류: %s", e)


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
        logger.info(
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
        logger.info(
            f"[warn] CSV에 '{SNAP_DATE_COL}' 데이터가 비어 import를 건너뜁니다: {history_csv}"
        )
        return
    if not missing_dates and table_exists:
        return

    df_csv = pd.read_csv(history_csv)
    if SNAP_DATE_COL not in df_csv.columns:
        logger.info(
            f"[warn] CSV에 '{SNAP_DATE_COL}' 컬럼이 없어 import를 건너뜁니다: {history_csv}"
        )
        return

    # 기존 CSV에 시간 컬럼이 있어도 제거
    if "스냅샷_시간" in df_csv.columns:
        df_csv = df_csv.drop(columns=["스냅샷_시간"])

    upsert_history(df_csv, history_db)
    logger.info(f"[done] 기존 CSV 히스토리를 SQLite로 마이그레이션: {history_db}")


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


def _standardize_archive_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder candidates to ARCHIVE_COLUMN_ORDER with zero-filled stock codes and standardized theme/chart values."""
    out = df.copy()
    if out.empty:
        return pd.DataFrame(columns=ARCHIVE_COLUMN_ORDER)

    if STOCK_CODE_COL in out.columns:
        out[STOCK_CODE_COL] = out[STOCK_CODE_COL].astype(str).str.zfill(6)

    # 1. 테마_섹터 표준화 (theme.parquet / table_theme 에서 공식 load_theme 로 조인)
    if "테마_섹터" not in out.columns and "테마" in out.columns:
        out["테마_섹터"] = out["테마"]

    from src.data.data_loader import load_theme
    theme_map = load_theme()

    if STOCK_CODE_COL in out.columns:
        if "테마_섹터" not in out.columns or out["테마_섹터"].isna().all():
            out["테마_섹터"] = out[STOCK_CODE_COL].map(theme_map).fillna("테마 없음")
        else:
            out["테마_섹터"] = out["테마_섹터"].fillna(out[STOCK_CODE_COL].map(theme_map)).fillna("테마 없음")
    else:
        if "테마_섹터" not in out.columns:
            out["테마_섹터"] = "테마 없음"

    # 2. 시나리오 표준화 (Scenario_Base 호환 및 과거 잔재 _Y / _N 접미사 전면 제거)
    if "시나리오" not in out.columns or out["시나리오"].isna().all():
        if "차트분석" in out.columns and not out["차트분석"].isna().all():
            out["시나리오"] = out["차트분석"].astype(str)
        elif "Scenario_Base" in out.columns and not out["Scenario_Base"].isna().all():
            out["시나리오"] = out["Scenario_Base"].astype(str)
        else:
            out["시나리오"] = "기본 분석"

    # 기존 데이터에 남아있는 _Y, _N 레거시 접미사 일괄 제거
    out["시나리오"] = (
        out["시나리오"].astype(str).str.replace(r"_[YN]$", "", regex=True)
    )

    # 차트분석 컬럼 완전 제거 (26개 표준 컬럼만 엄격 유지)
    if "차트분석" in out.columns:
        out = out.drop(columns=["차트분석"])

    return out.reindex(columns=ARCHIVE_COLUMN_ORDER)


def _upsert_sqlite_archive(df: pd.DataFrame, db_path: str) -> int:
    """Upsert snapshot rows into the SQLite archive, deduplicating by (date, code)."""
    with sqlite3.connect(db_path) as conn:
        df.head(0).to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        existing_cols = [
            row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_NAME})")
        ]
        for col in df.columns:
            if col not in existing_cols:
                conn.execute(f'ALTER TABLE {TABLE_NAME} ADD COLUMN "{col}"')

        total = 0
        for snap_date, group in df.groupby(SNAP_DATE_COL, dropna=False):
            existing = pd.read_sql(
                f'SELECT * FROM "{TABLE_NAME}" WHERE "{SNAP_DATE_COL}" = ?',
                conn,
                params=[snap_date],
            )
            merged = pd.concat([existing, group], ignore_index=True)
            merged = merged.drop_duplicates(
                subset=[SNAP_DATE_COL, STOCK_CODE_COL], keep="last"
            )
            conn.execute(
                f'DELETE FROM "{TABLE_NAME}" WHERE "{SNAP_DATE_COL}" = ?', [snap_date]
            )
            merged.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
            total += len(merged)
        conn.commit()
    return total


def _read_sqlite_archive() -> pd.DataFrame:
    """Load the full SQLite archive table as a DataFrame."""
    if not os.path.exists(settings.HISTORY_DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(settings.HISTORY_DB_PATH) as conn:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_NAME})")]
        if not cols:
            return pd.DataFrame()
        col_clause = ", ".join(f'"{c}"' for c in cols)
        return pd.read_sql(f"SELECT {col_clause} FROM {TABLE_NAME}", conn)


def upsert_archive_snapshot(df: pd.DataFrame, snapshot_date: str | None = None) -> int:
    """Upsert candidate snapshot into both Parquet and SQLite archives.

    The snapshot date is taken from the argument or, when absent, filled with
    today's date. Rows are deduplicated by (스냅샷_날짜, 종목코드) with keep-last
    semantics and stored in the standard 27-column layout in both stores.

    Args:
        df: Candidate snapshot DataFrame.
        snapshot_date: Snapshot date (YYYY-MM-DD) or None to reuse/fill.

    Returns:
        Number of rows stored/updated for the snapshot date.
    """
    out = df.copy()
    if snapshot_date is not None:
        out[SNAP_DATE_COL] = snapshot_date
    elif SNAP_DATE_COL not in out.columns or out[SNAP_DATE_COL].isna().all():
        out[SNAP_DATE_COL] = datetime.now().strftime("%Y-%m-%d")
    else:
        out[SNAP_DATE_COL] = out[SNAP_DATE_COL].fillna(datetime.now().strftime("%Y-%m-%d"))

    out = _standardize_archive_df(out)
    out = out.drop_duplicates(subset=[SNAP_DATE_COL, STOCK_CODE_COL], keep="last")

    settings.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    row_count = _upsert_sqlite_archive(out, str(settings.HISTORY_DB_PATH))

    try:
        from src.data.parquet_loader import upsert_condition_parquet

        upsert_condition_parquet(out)
    except Exception as e:
        logger.error("Parquet 조건검색 아카이브 저장 오류: %s", e)

    logger.info("[DATA] archive upsert date=%s rows=%d", snapshot_date or "latest", row_count)
    return row_count


def fetch_archive_snapshot(snapshot_date: str | None = None) -> pd.DataFrame:
    """Read candidate snapshot from archive in the standard 27-column order.

    The Parquet archive is the primary store; the SQLite archive is used as a
    fallback when Parquet is absent. With no snapshot_date the latest date is returned.

    Args:
        snapshot_date: Target date (YYYY-MM-DD) or None for the latest snapshot.

    Returns:
        DataFrame reordered to ARCHIVE_COLUMN_ORDER.
    """
    if settings.HISTORY_PARQUET_PATH.exists():
        df = pd.read_parquet(settings.HISTORY_PARQUET_PATH)
    else:
        df = _read_sqlite_archive()
    if df.empty or SNAP_DATE_COL not in df.columns:
        return pd.DataFrame(columns=ARCHIVE_COLUMN_ORDER)

    if snapshot_date is None:
        snapshot_date = str(df[SNAP_DATE_COL].astype(str).max())
    df = df[df[SNAP_DATE_COL].astype(str) == snapshot_date]
    df = _standardize_archive_df(df)
    return df.sort_values("선정순위", na_position="last", kind="stable")


def export_archive_for_spreadsheet(
    df_or_date: pd.DataFrame | str | None = None,
    sep: str = "\t",
    include_header: bool = True,
) -> str:
    """Render an archive snapshot as a TSV string ready for spreadsheet copy-paste.

    Args:
        df_or_date: Candidate DataFrame, snapshot date (YYYY-MM-DD), or None for latest.
        sep: Column separator (tab by default).
        include_header: Whether to include the standard 27-column header row.

    Returns:
        TSV string whose columns match the spreadsheet trade-log layout 1:1.
    """
    if isinstance(df_or_date, pd.DataFrame):
        df = _standardize_archive_df(df_or_date)
    elif isinstance(df_or_date, str):
        df = fetch_archive_snapshot(df_or_date)
    else:
        df = fetch_archive_snapshot()

    if df.empty:
        if include_header:
            return sep.join(ARCHIVE_COLUMN_ORDER) + "\n"
        return ""

    df = df.reindex(columns=ARCHIVE_COLUMN_ORDER).fillna("")
    return df.to_csv(sep=sep, index=False, header=include_header, lineterminator="\n")


def main():
    history_dir = settings.HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)

    clean_name = TARGET_CONDITION_NAME.replace("/", "_").replace("\\", "_")
    # 표준 CSV(utf-8-sig) 경로 우선 인식, 레거시 xlsx 폴백
    csv_latest = str(settings.CONDITION_CSV_PATH)
    xlsx_latest = str(settings.DATA_DIR / f"condition_{clean_name}.xlsx")
    latest_path = csv_latest if os.path.exists(csv_latest) else xlsx_latest
    history_csv = str(settings.HISTORY_CSV_PATH)
    history_db = str(settings.HISTORY_DB_PATH)

    import_csv_history_if_needed(history_csv, history_db)

    if FETCH_TARGET_DATE:
        df = fetch_date_rows(FETCH_TARGET_DATE, history_db)
        if df.empty:
            logger.info(f"[info] 조회된 데이터가 없습니다: {FETCH_TARGET_DATE}")
        else:
            target_file = os.path.join(
                history_dir, f"condition_{clean_name}_{FETCH_TARGET_DATE}.xlsx"
            )
            df.to_excel(target_file, index=False)
            logger.info(f"[done] 조회 결과를 저장했습니다: {target_file}")
        return

    if not os.path.exists(latest_path):
        logger.info(f"[skip] 최신 파일이 없습니다: {latest_path}")
        return

    # 전날 결과 불러오기
    if latest_path.endswith(".csv"):
        df = pd.read_csv(latest_path, encoding="utf-8-sig")
    else:
        df = pd.read_excel(latest_path)

    # 파일 수정 시각을 스냅샷 시각으로 사용 (없으면 현재 시각)
    snap_dt = datetime.fromtimestamp(os.path.getmtime(latest_path))
    df.insert(0, SNAP_DATE_COL, snap_dt.strftime("%Y-%m-%d"))

    upsert_history(df, history_db)
    logger.info(f"[done] SQLite 누적 저장: {history_db}")


if __name__ == "__main__":
    main()
