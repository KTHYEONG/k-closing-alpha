"""Parquet codec/dtype 단일 정책점 (비-오더북 parquet 쓰기 전용)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.io_utils import atomic_write_parquet

INTRADAY_COMPRESSION: str = "zstd"
PARQUET_COMPRESSION_LEVEL: int = 6
PRICE_HISTORY_FLOAT32_COLUMNS: tuple[str, ...] = (
    "market_cap_100m",
    "trade_value_100m",
    "foreign_netbuy",
    "inst_netbuy",
    "program_netbuy",
    "v_kospi",
    "v_kosdaq",
)
PRICE_HISTORY_FLOAT64_RETAIN_COLUMNS: tuple[str, ...] = (
    "daily_change_pct",
    "kospi_pct",
    "kosdaq_pct",
)
PRICE_HISTORY_NULLABLE_INT32_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "prev_close",
)

_INT32_MIN: int = -2147483648
_INT32_MAX: int = 2147483647
_INT64_MAX: int = 9223372036854775807


def downcast_price_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    """price_history 프레임을 저장용 dtype으로 변환한 복사본을 반환한다."""
    out = df.copy()
    for col in PRICE_HISTORY_NULLABLE_INT32_COLUMNS:
        if col not in out.columns:
            continue
        rounded = out[col].round()
        vals = pd.to_numeric(rounded, errors="coerce").dropna()
        if not vals.empty and (bool((vals < _INT32_MIN).any()) or bool((vals > _INT32_MAX).any())):
            raise ValueError(f"Column {col!r} overflows int32 range [{_INT32_MIN}, {_INT32_MAX}]")
    if "volume" in out.columns:
        vmax = pd.to_numeric(out["volume"], errors="coerce").max(skipna=True)
        if pd.notna(vmax) and float(vmax) > float(_INT64_MAX):
            raise ValueError(f"Column 'volume' overflows int64 range (max={vmax})")
    for col in PRICE_HISTORY_NULLABLE_INT32_COLUMNS:
        if col in out.columns:
            out[col] = out[col].round().astype("Int32")
    for col in PRICE_HISTORY_FLOAT32_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    if "volume" in out.columns:
        if out["volume"].isna().any():
            out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype("Int64")
        else:
            out["volume"] = pd.to_numeric(out["volume"], errors="coerce").astype("int64")
    for col in ("symbol", "market"):
        if col in out.columns:
            out[col] = out[col].astype("category")
    sort_keys = [c for c in ("symbol", "date") if c in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    return out


def downcast_altdata_panel_frame(df: pd.DataFrame) -> pd.DataFrame:
    """altdata 패널 프레임을 저장용 dtype으로 변환한 복사본을 반환한다."""
    out = df.copy()
    if "symbol" in out.columns:
        out["symbol"] = out["symbol"].astype("category")
    sort_keys = [c for c in ("symbol", "date") if c in out.columns]
    if sort_keys:
        out = out.sort_values(sort_keys, kind="stable").reset_index(drop=True)
    return out


def write_price_history_parquet(df: pd.DataFrame, target_path: Path) -> None:
    """다운캐스트 후 원자적 쓰기 (병합 없이 순수 쓰기)."""
    atomic_write_parquet(
        downcast_price_history_frame(df),
        target_path,
        compression=INTRADAY_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )


def write_altdata_panel_parquet(df: pd.DataFrame, target_path: Path) -> None:
    """다운캐스트 후 원자적 쓰기 (병합 없이 순수 쓰기)."""
    atomic_write_parquet(
        downcast_altdata_panel_frame(df),
        target_path,
        compression=INTRADAY_COMPRESSION,
        compression_level=PARQUET_COMPRESSION_LEVEL,
    )
