"""Daily security-classification snapshot alongside price_history."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src import settings
from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.krx_api import (
    KRX_ENDPOINT_KSQ_BASE_INFO,
    KRX_ENDPOINT_STK_BASE_INFO,
    fetch_krx_openapi_day_strict,
)
from src.data.capture_contracts import PageObserver
from src.data.parquet_codec import write_altdata_panel_parquet

logger = logging.getLogger(__name__)

SECURITY_CLASSIFICATION_MARKETS: tuple[tuple[str, str], ...] = (
    (KRX_ENDPOINT_STK_BASE_INFO, "KOSPI"),
    (KRX_ENDPOINT_KSQ_BASE_INFO, "KOSDAQ"),
)
SECURITY_CLASSIFICATION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "ISU_SRT_CD",
    "SECUGRP_NM",
    "KIND_STKCERT_TP_NM",
    "SECT_TP_NM",
)
SECURITY_CLASSIFICATION_COLUMNS: tuple[str, ...] = (
    "date",
    "symbol",
    "security_group",
    "security_kind",
    "section_type",
    "is_screenable",
)
SECURITY_CLASSIFICATION_PARQUET_FILENAME: str = "security_classification.parquet"

_SCREEN_EXCLUDE_SUBSTRINGS: tuple[str, ...] = ("관리종목", "SPAC", "투자주의환기")


def _clean_text(series: pd.Series) -> pd.Series:
    return series.map(lambda v: "" if pd.isna(v) else str(v).strip())


def normalize_security_classification(raw: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    """Map one KRX issue-base-info block to the classification panel schema.

    Args:
        raw: OutBlock rows of stk/ksq_isu_base_info for one date.
        trade_date: Requested basDd.

    Returns:
        Frame with SECURITY_CLASSIFICATION_COLUMNS; every input row retained
        with its screenability verdict.

    Raises:
        ValueError: When a required column is missing or a symbol repeats.
    """
    if raw.empty:
        return pd.DataFrame(columns=list(SECURITY_CLASSIFICATION_COLUMNS))
    missing = [c for c in SECURITY_CLASSIFICATION_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"security classification block missing columns: {missing}")
    day = pd.Timestamp(trade_date).normalize()
    group = _clean_text(raw["SECUGRP_NM"])
    kind = _clean_text(raw["KIND_STKCERT_TP_NM"])
    section = _clean_text(raw["SECT_TP_NM"])
    excluded = pd.Series(False, index=raw.index)
    for marker in _SCREEN_EXCLUDE_SUBSTRINGS:
        excluded = excluded | section.str.contains(marker, regex=False)
    out = pd.DataFrame({
        "date": day,
        "symbol": raw["ISU_SRT_CD"].astype(str).str.strip().to_numpy(),
        "security_group": group.to_numpy(),
        "security_kind": kind.to_numpy(),
        "section_type": section.to_numpy(),
        "is_screenable": ((group == "주권") & (kind == "보통주") & (~excluded)).to_numpy(),
    })
    dup = int(out["symbol"].duplicated().sum())
    if dup:
        raise ValueError(f"security classification block repeats {dup} symbols on {day.date()}")
    return out


def fetch_security_classification(
    trade_date: pd.Timestamp, cfg: AltDataFetchConfig, *, on_page: PageObserver | None = None
) -> pd.DataFrame:
    """Fetch and normalize both markets' classification blocks for one basDd.

    Args:
        trade_date: Requested trading date.
        cfg: KRX fetch configuration.
        on_page: Durable observer forwarded to both market endpoints.

    Returns:
        Concatenated classification rows for both markets.

    Raises:
        RuntimeError: On partial publication across the two markets.
    """
    ymd = pd.Timestamp(trade_date).strftime("%Y%m%d")
    parts = [
        normalize_security_classification(fetch_krx_openapi_day_strict(ep, ymd, cfg, on_page=on_page), trade_date)
        for ep, _ in SECURITY_CLASSIFICATION_MARKETS
    ]
    sizes = [len(p) for p in parts]
    if all(n == 0 for n in sizes):
        return pd.DataFrame(columns=list(SECURITY_CLASSIFICATION_COLUMNS))
    if any(n == 0 for n in sizes):
        raise RuntimeError(f"KRX classification partial publication on {ymd}: rows per market {sizes}")
    return pd.concat(parts, ignore_index=True)


def merge_security_classification(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Upsert new rows into existing keyed on (date, symbol).

    Args:
        existing: Stored panel rows.
        new_rows: Newly fetched rows.

    Returns:
        Merged frame sorted by symbol/date with latest observation kept.
    """
    if existing.empty:
        return new_rows.copy()
    if new_rows.empty:
        return existing.copy()
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "symbol"], keep="last")
    return merged.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def run_security_classification_ingest(
    trade_dates: Sequence[pd.Timestamp],
    *,
    path: str | os.PathLike[str] | None = None,
    krx_cfg: AltDataFetchConfig | None = None,
) -> dict[str, int]:
    """Capture classification snapshots for price-confirmed dates and upsert the panel.

    Args:
        trade_dates: Dates already confirmed published by run_price_ingest.
        path: Panel parquet path; None selects ALTDATA_DIR/security_classification.parquet.
        krx_cfg: KRX config; None builds one from settings.KRX_OPENAPI_KEY.

    Returns:
        Counters describing the newly fetched batch.

    Raises:
        RuntimeError: When a confirmed date yields no classification rows.
    """
    counters = {"n_fetched": 0, "n_written": 0, "n_admin": 0, "n_alert": 0, "n_spac": 0, "n_non_common": 0}
    dates = sorted(pd.Timestamp(d).normalize() for d in trade_dates)
    if not dates:
        return counters
    out_path = Path(path) if path is not None else Path(settings.ALTDATA_DIR) / SECURITY_CLASSIFICATION_PARQUET_FILENAME
    cfg = krx_cfg or AltDataFetchConfig(
        start=min(dates),
        end=max(dates) + pd.Timedelta(days=1),
        out_dir=Path("."),
        krx_api_key=settings.KRX_OPENAPI_KEY,
    )
    frames: list[pd.DataFrame] = []
    for d in dates:
        frame = fetch_security_classification(d, cfg)
        if frame.empty:
            raise RuntimeError(f"KRX classification missing for price-confirmed date {d.date()}")
        frames.append(frame)
    batch = pd.concat(frames, ignore_index=True)
    section = batch["section_type"].astype(str)
    counters["n_fetched"] = int(len(batch))
    counters["n_written"] = int(len(batch))
    counters["n_admin"] = int(section.str.contains("관리종목", regex=False).sum())
    counters["n_alert"] = int(section.str.contains("투자주의환기", regex=False).sum())
    counters["n_spac"] = int(section.str.contains("SPAC", regex=False).sum())
    counters["n_non_common"] = int((batch["security_kind"].astype(str) != "보통주").sum())
    if out_path.exists():
        existing = pd.read_parquet(out_path)
    else:
        existing = pd.DataFrame(columns=list(SECURITY_CLASSIFICATION_COLUMNS))
    merged = merge_security_classification(existing, batch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_altdata_panel_parquet(merged, out_path)
    logger.info(
        "[DATA] stage=security_classification_ingest dates=%s n_fetched=%d n_written=%d n_admin=%d n_alert=%d n_spac=%d n_non_common=%d",
        [d.strftime("%Y-%m-%d") for d in dates],
        counters["n_fetched"],
        counters["n_written"],
        counters["n_admin"],
        counters["n_alert"],
        counters["n_spac"],
        counters["n_non_common"],
    )
    return counters


def load_security_classification(
    decision_date: pd.Timestamp,
    *,
    prev_trading_day: pd.Timestamp,
    path: str | os.PathLike[str] | None = None,
) -> frozenset[str]:
    """Return symbols verdicted screenable as of the previous trading day.

    Args:
        decision_date: Decision date; only strictly-prior rows are read.
        prev_trading_day: Previous trading day, must predate decision_date.
        path: Panel parquet path; None selects ALTDATA_DIR/security_classification.parquet.

    Returns:
        Screenable symbols on prev_trading_day.

    Raises:
        ValueError: When prev_trading_day is not before decision_date, or when
            no rows exist on the previous trading day.
        FileNotFoundError: When the panel parquet does not exist.
    """
    prev = pd.Timestamp(prev_trading_day).normalize()
    if prev >= pd.Timestamp(decision_date).normalize():
        raise ValueError(
            f"prev_trading_day {prev.date()} must be before decision_date {pd.Timestamp(decision_date).date()}"
        )
    src_path = Path(path) if path is not None else Path(settings.ALTDATA_DIR) / SECURITY_CLASSIFICATION_PARQUET_FILENAME
    if not src_path.exists():
        raise FileNotFoundError(f"security_classification not found: {src_path}")
    rows = pd.read_parquet(src_path, filters=[("date", "==", prev)])
    if rows.empty:
        raise ValueError(f"stale security_classification: no rows on prev_trading_day={prev.date()}")
    mask = rows["is_screenable"].astype(bool).to_numpy()
    return frozenset(rows.loc[mask, "symbol"].astype(str).tolist())
