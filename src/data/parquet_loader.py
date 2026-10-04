"""Parquet I/O operations and dataset manager module.

Provides high-performance columnar data loading, saving, and upsert capabilities
for theme mappings and condition search snapshots using Parquet format.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import settings
from src.data.io_utils import store_write_lock

logger = logging.getLogger(__name__)


class ThemeMapUnreadableError(RuntimeError):
    """An existing theme map file could not be read or lacks its required schema.

    Distinct from absence: an absent file is a legitimate "no theme data" state,
    while an unreadable one means theme knowledge exists but is unknown right now.
    Callers must not translate this into a placeholder value that would be persisted.
    """


def _ensure_dir(path: Path) -> None:
    """Ensure directory exists for given path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_parquet(df: pd.DataFrame, target_path: Path) -> None:
    """Safely write DataFrame to Parquet using a temporary file to guarantee atomic writes.

    Time Complexity: O(N) where N is total rows in df.
    Space Complexity: O(N) for parquet buffer.

    Args:
        df: DataFrame to save.
        target_path: Destination parquet file path.
    """
    from src.data.io_utils import atomic_write_parquet

    atomic_write_parquet(df, target_path)


def _clean_df_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame to ensure PyArrow compatibility with mixed object types from Google Sheets."""
    df_clean = df.copy()
    df_clean.columns = [str(c) for c in df_clean.columns]

    for col in df_clean.columns:
        # object/string 컬럼의 빈 문자열 처리 및 타입 안정화
        if df_clean[col].dtype == "object":
            # 숫자로 일괄 변환 시도해보기 (실패시 object 유지)
            s_numeric = pd.to_numeric(df_clean[col], errors="coerce")
            # 숫자로 변환된 비율이 무의미하지 않은 경우 (문자열 비율 확인)
            non_null_orig = df_clean[col].replace(r"^\s*$", None, regex=True).dropna()
            non_null_num = s_numeric.dropna()
            if len(non_null_orig) > 0 and len(non_null_orig) == len(non_null_num):
                df_clean[col] = s_numeric
            else:
                # 숫자 혼용 실패한 컬럼은 순수 string으로 통일
                df_clean[col] = df_clean[col].astype(str).replace({"nan": None, "None": None, "<NA>": None})

    return df_clean


def load_theme_from_parquet() -> dict[str, str]:
    """Load the stock-code → theme mapping from ``settings.THEME_PARQUET_PATH``.

    Returns:
        Mapping of 6-character stock code to stripped, non-empty theme text. Rows whose
        theme is null or whitespace-only are omitted so a lookup miss is the only way
        "no theme" is expressed. Duplicate codes resolve to the last row. Returns an
        empty dict, with a ``[DATA]`` WARNING, when the file does not exist.

    Raises:
        ThemeMapUnreadableError: The file exists but reading it fails (original exception
            chained as ``__cause__``; path in the message), or the columns ``종목코드`` /
            ``테마`` are missing.
    """
    parquet_path = settings.THEME_PARQUET_PATH
    if not parquet_path.exists():
        logger.warning("[DATA] Theme parquet file not found stage=theme_load status=THEME_MISSING path=%s", parquet_path)
        return {}

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise ThemeMapUnreadableError(f"theme map at {parquet_path} could not be read") from exc
    missing = [col for col in ("종목코드", "테마") if col not in df.columns]
    if missing:
        raise ThemeMapUnreadableError(f"theme map at {parquet_path} is missing columns: {missing}")

    usable = df.loc[df["테마"].notna() & (df["테마"].astype(str).str.strip() != "")]
    codes = usable["종목코드"].astype(str).str.zfill(6)
    themes = usable["테마"].astype(str).str.strip()
    return dict(zip(codes, themes, strict=False))


def upsert_condition_parquet(df: pd.DataFrame) -> None:
    """Overwrite condition snapshot rows in parquet for the target dates.

    The existing-file read, identity-based replacement, dedup and atomic write run under the archive's
    store write lock, so concurrent collect/finalize runs (e.g. a manual rerun overlapping the scheduled
    unit) cannot drop each other's snapshot rows.

    Args:
        df: Condition history snapshot DataFrame.

    Raises:
        StoreLockTimeoutError: The archive lock was not acquired within STORE_LOCK_TIMEOUT_SECONDS; the
            archive is untouched.
    """
    if df is None or df.empty:
        return

    parquet_path = settings.HISTORY_PARQUET_PATH
    with store_write_lock(parquet_path, purpose="condition-archive"):
        if parquet_path.exists():
            df_existing = pd.read_parquet(parquet_path)
            if "스냅샷_날짜" in df.columns and "스냅샷_날짜" in df_existing.columns:
                has_identity = "snapshot_timestamp" in df.columns and "snapshot_timestamp" in df_existing.columns
                if has_identity:
                    # 스냅샷 정체성(날짜, 시각) 단위로만 교체해 intraday 캡처를 보존합니다.
                    existing_key = (
                        df_existing["스냅샷_날짜"].astype(str)
                        + "|"
                        + df_existing["snapshot_timestamp"].astype(str)
                    )
                    new_keys = set(
                        df["스냅샷_날짜"].astype(str)
                        + "|"
                        + df["snapshot_timestamp"].astype(str)
                    )
                    df_existing = df_existing[~existing_key.isin(new_keys)]
                else:
                    target_dates = set(df["스냅샷_날짜"].dropna().astype(str).unique())
                    df_existing = df_existing[~df_existing["스냅샷_날짜"].astype(str).isin(target_dates)]
            df_combined = pd.concat([df_existing, df], ignore_index=True)
        else:
            df_combined = df.copy()

        # 스냅샷 정체성이 있으면 (날짜, 시각, 종목), 없으면 (날짜, 종목) 기준 중복 제거
        if "snapshot_timestamp" in df_combined.columns:
            dedup_cols = ["스냅샷_날짜", "snapshot_timestamp", "종목코드"]
        else:
            dedup_cols = ["스냅샷_날짜", "종목코드"]
        dedup_cols = [col for col in dedup_cols if col in df_combined.columns]
        if dedup_cols:
            df_combined = df_combined.drop_duplicates(subset=dedup_cols, keep="last")

        df_combined = _clean_df_for_parquet(df_combined)
        _atomic_write_parquet(df_combined, parquet_path)
    logger.info("Upserted condition history parquet: %s (%d rows)", parquet_path, len(df_combined))
