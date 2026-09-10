import logging
from datetime import datetime

import pandas as pd

from src import settings
from src.processing.schema import ARCHIVE_COLUMN_ORDER

logger = logging.getLogger(__name__)

SNAP_DATE_COL = "스냅샷_날짜"
STOCK_CODE_COL = "종목코드"

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

SNAPSHOT_TIMESTAMP_SYNTHETIC_COL: str = "snapshot_timestamp_synthetic"

# 조회(읽기) 시 타임스탬프 컬럼이 표준 컬럼 뒤에 붙은 전체 순서
ARCHIVE_READ_COLUMN_ORDER = [*ARCHIVE_COLUMN_ORDER, *TIMESTAMP_COLS, SNAPSHOT_TIMESTAMP_SYNTHETIC_COL]


def _standardize_archive_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder candidates to ARCHIVE_COLUMN_ORDER with zero-filled stock codes and standardized theme/chart values.

    Timezone-aware timestamp columns(``snapshot_timestamp`` 등)은 표준 컬럼
    뒤에 보존되어 스냅샷 시각이 유실되지 않습니다.
    """
    out = df.copy()
    if out.empty:
        return pd.DataFrame(columns=ARCHIVE_READ_COLUMN_ORDER)

    if STOCK_CODE_COL in out.columns:
        out[STOCK_CODE_COL] = out[STOCK_CODE_COL].astype(str).str.zfill(6)

    # 1. 테마_섹터 표준화 (theme.parquet에서 공식 load_theme 로 조인)
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

    # 차트분석 컬럼 완전 제거 (표준 컬럼만 엄격 유지)
    if "차트분석" in out.columns:
        out = out.drop(columns=["차트분석"])

    # reindex 는 타임스탬프/합성 컬럼을 버리므로 보존 후 재부착
    timestamp_series = {col: out[col] for col in TIMESTAMP_COLS if col in out.columns}
    synthetic_series = out[SNAPSHOT_TIMESTAMP_SYNTHETIC_COL] if SNAPSHOT_TIMESTAMP_SYNTHETIC_COL in out.columns else None
    out = out.reindex(columns=ARCHIVE_COLUMN_ORDER)
    for col, series in timestamp_series.items():
        out[col] = series.reindex(out.index)
    for col in TIMESTAMP_COLS:
        if col not in out.columns:
            out[col] = pd.NaT
    if synthetic_series is not None:
        out[SNAPSHOT_TIMESTAMP_SYNTHETIC_COL] = synthetic_series.reindex(out.index)
    elif SNAPSHOT_TIMESTAMP_SYNTHETIC_COL not in out.columns:
        out[SNAPSHOT_TIMESTAMP_SYNTHETIC_COL] = False
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


def _write_condition_parquet_with_retry(out: pd.DataFrame) -> None:
    """Parquet 아카이브 쓰기를 최대 2회 시도하고 실패 시 RuntimeError를 발생시킨다."""
    from src.data import parquet_loader

    try:
        parquet_loader.upsert_condition_parquet(out)
        return
    except Exception:
        try:
            parquet_loader.upsert_condition_parquet(out)
            return
        except Exception as second_exc:
            raise RuntimeError(f"parquet archive write failed after retry: {second_exc}") from second_exc


def upsert_archive_snapshot(df: pd.DataFrame, snapshot_date: str | None = None) -> int:
    """Upsert candidate snapshot into the parquet archive.

    The snapshot date is taken from the argument or, when absent, filled with
    today's date. Timezone-aware ``snapshot_timestamp``/``feature_available_timestamp``
    (Asia/Seoul) are preserved per row; when a snapshot time is not recorded, the
    deterministic 15:30 KST close convention is used. Rows are deduplicated by the
    full snapshot identity (snapshot_timestamp, stock_code) when multiple intraday
    captures exist, falling back to (스냅샷_날짜, 종목코드) otherwise. Stored in the
    standard column layout plus timestamp columns.

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
        out[SNAPSHOT_TIMESTAMP_SYNTHETIC_COL] = True
    else:
        out[SNAPSHOT_TIMESTAMP_COL] = _ensure_tz(
            out[SNAPSHOT_TIMESTAMP_COL], out[SNAP_DATE_COL].map(_kst_timestamp)
        )
        if SNAPSHOT_TIMESTAMP_SYNTHETIC_COL not in out.columns:
            out[SNAPSHOT_TIMESTAMP_SYNTHETIC_COL] = False
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
    _target_date = str(out[SNAP_DATE_COL].iloc[0]) if len(out) else str(snapshot_date or "")
    try:
        if settings.HISTORY_PARQUET_PATH.exists():
            _existing = pd.read_parquet(settings.HISTORY_PARQUET_PATH)
            if _existing is not None and not _existing.empty and SNAP_DATE_COL in _existing.columns:
                _prev = _existing[_existing[SNAP_DATE_COL].astype(str) == _target_date]
                if not _prev.empty:
                    if SNAPSHOT_TIMESTAMP_COL in _prev.columns:
                        _latest = pd.to_datetime(_prev[SNAPSHOT_TIMESTAMP_COL], errors="coerce").max()
                    else:
                        _latest = pd.NaT
                    logger.info(
                        "🔄 [아카이브 갱신] %s 당일 스냅샷 rerun detected (기존 %d행, latest=%s, %d행 덮어쓰기)",
                        _target_date,
                        len(_prev),
                        str(_latest),
                        len(out),
                    )
    except Exception:
        pass
    row_count = len(out)

    _write_condition_parquet_with_retry(out)

    logger.info("💾 [아카이브 저장] %s 스냅샷 적재 완료 (총 %d행)", _target_date or "latest", row_count)
    return row_count


def fetch_archive_snapshot(
    snapshot_date: str | None = None,
    month: str | None = None,
    all_rows: bool = False,
    latest_only: bool = True,
) -> pd.DataFrame:
    """Read candidate snapshot from archive in standard column order.

    If all_rows is True, returns all historical data. Otherwise filters by snapshot_date,
    month (YYYY-MM), or defaults to the latest available month.
    With latest_only=True (default), rerun duplicates are collapsed to the latest
    snapshot per (스냅샷_날짜, 종목코드) by snapshot_timestamp; latest_only=False
    preserves full history.

    Args:
        snapshot_date: Target date (YYYY-MM-DD) or None.
        month: Target month (YYYY-MM) or None.
        all_rows: If True, return all rows without filtering by date/month.
        latest_only: If True, keep only the latest snapshot per date/code.

    Returns:
        DataFrame reordered to ARCHIVE_COLUMN_ORDER, sorted by date and rank.
    """
    if not settings.HISTORY_PARQUET_PATH.exists():
        return pd.DataFrame(columns=ARCHIVE_READ_COLUMN_ORDER)
    df = pd.read_parquet(settings.HISTORY_PARQUET_PATH)
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
    if latest_only and not df.empty and SNAP_DATE_COL in df.columns and STOCK_CODE_COL in df.columns:
        if SNAPSHOT_TIMESTAMP_COL in df.columns:
            df = df.sort_values(SNAPSHOT_TIMESTAMP_COL, ascending=True, na_position="first", kind="stable")
        df = df.drop_duplicates(subset=[SNAP_DATE_COL, STOCK_CODE_COL], keep="last")
    return df.sort_values(
        [SNAP_DATE_COL, STOCK_CODE_COL],
        ascending=[True, True],
        na_position="last",
        kind="stable",
    )
