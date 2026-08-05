"""Parquet I/O operations and dataset manager module.

Provides high-performance columnar data loading, saving, and upsert capabilities
for trade logs, theme mappings, and condition search snapshots using Parquet format.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

from src import settings

logger = logging.getLogger(__name__)


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
    _ensure_dir(target_path)
    temp_dir = target_path.parent
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".parquet") as tmp:
        tmp_path = Path(tmp.name)

    try:
        df.to_parquet(tmp_path, index=False, compression="snappy")
        os.replace(tmp_path, target_path)
    except Exception as e:
        if tmp_path.exists():
            tmp_path.unlink()
        logger.error("Failed to write parquet atomically to %s: %s", target_path, e)
        raise


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


def save_trade_log_to_parquet(df: pd.DataFrame) -> None:
    """Save trade log DataFrame to parquet format.

    Args:
        df: Trade log DataFrame.
    """
    if df is None or df.empty:
        logger.warning("Trade log DataFrame is empty. Skipping parquet save.")
        return

    df_copy = _clean_df_for_parquet(df)
    if "종목코드" in df_copy.columns:
        df_copy["종목코드"] = df_copy["종목코드"].astype(str).str.zfill(6)

    _atomic_write_parquet(df_copy, settings.TRADE_LOG_PARQUET_PATH)
    logger.info("Saved trade log to parquet: %s (%d rows)", settings.TRADE_LOG_PARQUET_PATH, len(df_copy))


def load_trade_log_from_parquet() -> pd.DataFrame:
    """Load trade log DataFrame from parquet file.

    Returns:
        pd.DataFrame: Trade log DataFrame.
    """
    parquet_path = settings.TRADE_LOG_PARQUET_PATH
    if not parquet_path.exists():
        logger.info("Trade log parquet file not found at %s. Returning empty DataFrame.", parquet_path)
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    if "종목코드" in df.columns:
        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
    return df


def save_theme_to_parquet(df: pd.DataFrame) -> None:
    """Save theme mapping DataFrame to parquet format.

    Args:
        df: Theme DataFrame containing 종목코드 and 테마 columns.
    """
    if df is None or df.empty:
        logger.warning("Theme DataFrame is empty. Skipping parquet save.")
        return

    df_copy = _clean_df_for_parquet(df)
    if "종목코드" in df_copy.columns:
        df_copy["종목코드"] = (
            df_copy["종목코드"]
            .astype(str)
            .str.strip()
            .str.split(".")
            .str[0]
            .str.zfill(6)
        )

    _atomic_write_parquet(df_copy, settings.THEME_PARQUET_PATH)
    logger.info("Saved theme mapping to parquet: %s (%d rows)", settings.THEME_PARQUET_PATH, len(df_copy))


def load_theme_from_parquet() -> dict[str, str]:
    """Load theme mapping dictionary from parquet file.

    Returns:
        dict[str, str]: Mapping of stock_code to theme string.
    """
    parquet_path = settings.THEME_PARQUET_PATH
    if not parquet_path.exists():
        logger.warning("Theme parquet file not found at %s.", parquet_path)
        return {}

    try:
        df = pd.read_parquet(parquet_path)
        if "종목코드" not in df.columns or "테마" not in df.columns:
            logger.warning("Theme parquet missing required columns.")
            return {}

        df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        theme_map: dict[str, str] = dict(zip(df["종목코드"], df["테마"], strict=False))
        return theme_map
    except Exception as e:
        logger.error("Failed to load theme parquet: %s", e)
        return {}


def upsert_condition_parquet(df: pd.DataFrame) -> None:
    """Overwrite condition snapshot rows in parquet for the target dates.

    Time Complexity: O(N + M) where N is existing rows and M is new rows.
    Space Complexity: O(N + M) in memory merge.

    Args:
        df: Condition history snapshot DataFrame.
    """
    if df is None or df.empty:
        return

    parquet_path = settings.HISTORY_PARQUET_PATH
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


def load_condition_data_from_parquet(date: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Load condition search snapshot history from parquet.

    Args:
        date: Filter by date prefix string ('YYYY-MM-DD').
        limit: Limit recent N rows.

    Returns:
        pd.DataFrame: Condition history DataFrame.
    """
    parquet_path = settings.HISTORY_PARQUET_PATH
    if not parquet_path.exists():
        logger.info("Condition history parquet file not found at %s.", parquet_path)
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)

    if date and "스냅샷_날짜" in df.columns:
        df = df[df["스냅샷_날짜"].astype(str).str.startswith(date)]

    if limit and len(df) > limit:
        df = df.tail(limit)

    return df
