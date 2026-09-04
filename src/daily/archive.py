import logging
import os
import sqlite3
from datetime import datetime

import pandas as pd

from src import settings
from src.processing.schema import ARCHIVE_COLUMN_ORDER

logger = logging.getLogger(__name__)

TABLE_NAME = "condition_history"
SNAP_DATE_COL = "스냅샷_날짜"
STOCK_CODE_COL = "종목코드"
RANK_COL = "순위"
# 조회하고 싶을 때 YYYY-MM-DD 형식으로 지정.
FETCH_TARGET_DATE = None  # "2025-12-09"

# point-in-time 무결성 타임스탬프 (Asia/Seoul timezone-aware)
SNAPSHOT_TIMESTAMP_COL = "snapshot_timestamp"
FEATURE_AVAILABLE_TIMESTAMP_COL = "feature_available_timestamp"
DECISION_TIMESTAMP_COL = "decision_timestamp"
EXECUTION_TIMESTAMP_COL = "execution_timestamp"
TIMESTAMP_COLS = (
    SNAPSHOT_TIMESTAMP_COL,
    FEATURE_AVAILABLE_TIMESTAMP_COL,
    DECISION_TIMESTAMP_COL,
    EXECUTION_TIMESTAMP_COL,
)
KST = "Asia/Seoul"

# 조회(읽기) 시 타임스탬프 컬럼이 26개 표준 컬럼 뒤에 붙은 전체 순서
ARCHIVE_READ_COLUMN_ORDER = [*ARCHIVE_COLUMN_ORDER, *TIMESTAMP_COLS]

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

    df_csv = pd.read_csv(history_csv, dtype={"종목코드": str})
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
    """Reorder candidates to ARCHIVE_COLUMN_ORDER with zero-filled stock codes and standardized theme/chart values.

    Timezone-aware timestamp columns(``snapshot_timestamp`` 등)은 26개 표준 컬럼
    뒤에 보존되어 스냅샷 시각이 유실되지 않습니다.
    """
    out = df.copy()
    if out.empty:
        return pd.DataFrame(columns=ARCHIVE_READ_COLUMN_ORDER)

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

    # reindex 는 타임스탬프 컬럼을 버리므로 보존 후 재부착
    timestamp_series = {col: out[col] for col in TIMESTAMP_COLS if col in out.columns}
    out = out.reindex(columns=ARCHIVE_COLUMN_ORDER)
    for col, series in timestamp_series.items():
        out[col] = series.reindex(out.index)
    for col in TIMESTAMP_COLS:
        if col not in out.columns:
            out[col] = pd.NaT
    return out[ARCHIVE_READ_COLUMN_ORDER]


def _kst_timestamp(snapshot_date: str) -> pd.Timestamp:
    """스냅샷 날짜에서 Asia/Seoul timezone-aware 타임스탬프를 결정적으로 생성합니다.

    스냅샷 시각이 기록되지 않은 과거 데이터는 장 마감 15:30 KST 관례로 간주합니다.
    """
    return pd.Timestamp(snapshot_date, tz=KST) + pd.Timedelta(hours=15, minutes=30)


def _ensure_tz(series: pd.Series, fallback: pd.Series) -> pd.Series:
    """타임스탬프 컬럼을 Asia/Seoul timezone-aware 로 강제하고 결측치를 보정합니다."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(KST)
    else:
        parsed = parsed.dt.tz_convert(KST)
    return parsed.fillna(fallback)


def _upsert_sqlite_archive(df: pd.DataFrame, db_path: str) -> int:
    """Overwrite snapshot rows in the SQLite archive for the given snapshot identities."""
    with sqlite3.connect(db_path) as conn:
        df.head(0).to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        existing_cols = [
            row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_NAME})")
        ]
        for col in df.columns:
            if col not in existing_cols:
                conn.execute(f'ALTER TABLE {TABLE_NAME} ADD COLUMN "{col}"')

        has_timestamp = SNAPSHOT_TIMESTAMP_COL in existing_cols
        total = 0
        group_cols = [SNAP_DATE_COL, SNAPSHOT_TIMESTAMP_COL] if has_timestamp else [SNAP_DATE_COL]
        for keys, group in df.groupby(group_cols, dropna=False):
            snap_date = keys[0] if isinstance(keys, tuple) else keys
            if has_timestamp:
                snap_ts = keys[1] if isinstance(keys, tuple) else None
                if pd.isna(snap_ts):
                    conn.execute(
                        f'DELETE FROM "{TABLE_NAME}" WHERE "{SNAP_DATE_COL}" = ? AND '
                        f'"{SNAPSHOT_TIMESTAMP_COL}" IS NULL',
                        [snap_date],
                    )
                else:
                    conn.execute(
                        f'DELETE FROM "{TABLE_NAME}" WHERE "{SNAP_DATE_COL}" = ? AND '
                        f'"{SNAPSHOT_TIMESTAMP_COL}" = ?',
                        [snap_date, str(snap_ts)],
                    )
            else:
                conn.execute(
                    f'DELETE FROM "{TABLE_NAME}" WHERE "{SNAP_DATE_COL}" = ?', [snap_date]
                )
            group.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
            total += len(group)
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
    today's date. Timezone-aware ``snapshot_timestamp``/``feature_available_timestamp``
    (Asia/Seoul) are preserved per row; when a snapshot time is not recorded, the
    deterministic 15:30 KST close convention is used. Rows are deduplicated by the
    full snapshot identity (snapshot_timestamp, stock_code) when multiple intraday
    captures exist, falling back to (스냅샷_날짜, 종목코드) otherwise. Stored in the
    standard 26-column layout plus timestamp columns in both stores.

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

    # 스냅샷 시각 보존: 미지정 시 날짜 기준 결정적 15:30 KST 관례 적용
    if SNAPSHOT_TIMESTAMP_COL not in out.columns:
        out[SNAPSHOT_TIMESTAMP_COL] = out[SNAP_DATE_COL].map(
            lambda d: _kst_timestamp(str(d))
        )
    else:
        out[SNAPSHOT_TIMESTAMP_COL] = _ensure_tz(
            out[SNAPSHOT_TIMESTAMP_COL], out[SNAP_DATE_COL].map(_kst_timestamp)
        )
    for col in (
        FEATURE_AVAILABLE_TIMESTAMP_COL,
        DECISION_TIMESTAMP_COL,
        EXECUTION_TIMESTAMP_COL,
    ):
        if col not in out.columns:
            out[col] = out[SNAPSHOT_TIMESTAMP_COL]
        else:
            out[col] = _ensure_tz(out[col], out[SNAPSHOT_TIMESTAMP_COL])

    out = _standardize_archive_df(out)
    has_intraday = out[SNAPSHOT_TIMESTAMP_COL].nunique() > 1
    if has_intraday:
        out = out.drop_duplicates(subset=[SNAPSHOT_TIMESTAMP_COL, STOCK_CODE_COL], keep="last")
    else:
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


def fetch_archive_snapshot(
    snapshot_date: str | None = None,
    month: str | None = None,
    all_rows: bool = False,
) -> pd.DataFrame:
    """Read candidate snapshot from archive in standard 27-column order.

    If all_rows is True, returns all historical data. Otherwise filters by snapshot_date,
    month (YYYY-MM), or defaults to the latest available month.

    Args:
        snapshot_date: Target date (YYYY-MM-DD) or None.
        month: Target month (YYYY-MM) or None.
        all_rows: If True, return all rows without filtering by date/month.

    Returns:
        DataFrame reordered to ARCHIVE_COLUMN_ORDER, sorted by date and rank.
    """
    if settings.HISTORY_PARQUET_PATH.exists():
        df = pd.read_parquet(settings.HISTORY_PARQUET_PATH)
    else:
        df = _read_sqlite_archive()
    if df.empty or SNAP_DATE_COL not in df.columns:
        return pd.DataFrame(columns=ARCHIVE_READ_COLUMN_ORDER)

    df_dates = df[SNAP_DATE_COL].astype(str)

    if not all_rows:
        if snapshot_date is not None:
            df = df[df_dates == snapshot_date]
        elif month is not None:
            df = df[df_dates.str.startswith(month)]
        else:
            latest_date = str(df_dates.max())
            latest_month = latest_date[:7]  # YYYY-MM
            df = df[df_dates.str.startswith(latest_month)]

    df = _standardize_archive_df(df)
    return df.sort_values(
        [SNAP_DATE_COL, "선정순위"],
        ascending=[True, True],
        na_position="last",
        kind="stable",
    )


def export_archive_for_spreadsheet(
    df_or_date: pd.DataFrame | str | None = None,
    sep: str = "\t",
    include_header: bool = True,
    month: str | None = None,
) -> str:
    """Render archive snapshot rows as a TSV string ready for spreadsheet copy-paste.

    By default, renders all rows for the latest month (or specified month/date).

    Args:
        df_or_date: Candidate DataFrame, snapshot date (YYYY-MM-DD), or None for month group.
        sep: Column separator (tab by default).
        include_header: Whether to include standard header row.
        month: Target month (YYYY-MM) when df_or_date is None.

    Returns:
        TSV string whose columns match standard 26-column layout.
    """
    if isinstance(df_or_date, pd.DataFrame):
        df = _standardize_archive_df(df_or_date)
    elif isinstance(df_or_date, str):
        if len(df_or_date) == 7:  # YYYY-MM format
            df = fetch_archive_snapshot(month=df_or_date)
        else:
            df = fetch_archive_snapshot(snapshot_date=df_or_date)
    else:
        df = fetch_archive_snapshot(month=month)

    if df.empty:
        if include_header:
            return sep.join(ARCHIVE_COLUMN_ORDER) + "\n"
        return ""

    df = df.reindex(columns=ARCHIVE_COLUMN_ORDER).fillna("")
    return df.to_csv(sep=sep, index=False, header=include_header, lineterminator="\n")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    history_dir = settings.HISTORY_DIR
    history_dir.mkdir(parents=True, exist_ok=True)

    # 표준 CSV(utf-8-sig) 경로 우선 인식
    csv_latest = str(settings.CONDITION_CSV_PATH)
    history_csv = str(settings.HISTORY_CSV_PATH)
    history_db = str(settings.HISTORY_DB_PATH)

    import_csv_history_if_needed(history_csv, history_db)

    if FETCH_TARGET_DATE:
        df = fetch_date_rows(FETCH_TARGET_DATE, history_db)
        if df.empty:
            logger.info(f"[info] 조회된 데이터가 없습니다: {FETCH_TARGET_DATE}")
        else:
            target_file = os.path.join(history_dir, f"archive_{FETCH_TARGET_DATE}.tsv")
            df.to_csv(target_file, sep="\t", index=False)
            logger.info(f"[done] 조회 결과를 저장했습니다: {target_file}")
        return

    if not os.path.exists(csv_latest):
        logger.warning(f"[skip] 아카이브할 최신 조건검색 CSV 파일이 없습니다: {csv_latest}")
        return

    # 최신 결과 불러오기 (종목코드 0 누락 및 지수 표기 방지)
    df = pd.read_csv(csv_latest, encoding="utf-8-sig", dtype={"종목코드": str})

    # 파일 수정 시각을 스냅샷 시각으로 사용 (없으면 현재 시각)
    snap_dt = datetime.fromtimestamp(os.path.getmtime(csv_latest))
    snapshot_date = snap_dt.strftime("%Y-%m-%d")
    df.insert(0, SNAP_DATE_COL, snapshot_date)
    df[SNAPSHOT_TIMESTAMP_COL] = snap_dt

    stored_rows = upsert_archive_snapshot(df, snapshot_date=snapshot_date)
    logger.info(
        f"[SUCCESS] 조건검색 아카이브 완료 (날짜: {snapshot_date}, 저장 종목 수: {stored_rows}건)"
    )


if __name__ == "__main__":
    main()
