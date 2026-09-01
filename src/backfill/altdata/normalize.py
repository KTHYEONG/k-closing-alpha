"""패널 정규화 유틸리티."""

from __future__ import annotations

import pandas as pd

from src.backfill.altdata.config import _ALTDATA_PANELS, AltDataFetchConfig


def normalize_panel(df: pd.DataFrame, panel: str, cfg: AltDataFetchConfig) -> pd.DataFrame:
    """원시 수집 프레임을 표준 스키마로 정규화합니다.

    Args:
        df: 원시 DataFrame.
        panel: 패널 이름.
        cfg: Alt-data 설정.

    Returns:
        정규화된 DataFrame.
    """
    meta = _ALTDATA_PANELS.get(panel)
    if meta is None:
        raise ValueError(f"unknown panel '{panel}'")
    key_cols: tuple[str, ...] = meta["key_cols"]
    level: str = meta["level"]

    # Define expected columns for empty case
    if df is None or df.empty:
        # Return empty with correct columns (key cols + existing numeric cols if any)
        # For empty, we at least return key_cols
        if df is None or df.empty:
            # Build columns: use key_cols plus any numeric cols that would be expected?
            # Just return empty with key_cols
            empty_cols = list(key_cols)
            # Preserve other columns if df had them
            if df is not None and len(df.columns) > 0:
                for c in df.columns:
                    if c not in empty_cols:
                        empty_cols.append(c)
            return pd.DataFrame(columns=empty_cols)

    work = df.copy()

    # Coerce date
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        # tz-naive midnight-normalized
        # Remove tz if any
        try:
            if hasattr(work["date"].dtype, "tz") and work["date"].dt.tz is not None:
                work["date"] = work["date"].dt.tz_localize(None)
        except Exception:
            pass
        work["date"] = work["date"].dt.normalize()
    else:
        work["date"] = pd.NaT

    # Symbol handling for symbol-level panels
    if level == "symbol":
        if "symbol" not in work.columns:
            work["symbol"] = ""
        work["symbol"] = work["symbol"].astype(str).str.strip().str.zfill(6)
        # Coerce numeric columns (all except date,symbol)
        for c in list(work.columns):
            if c in ("date", "symbol"):
                continue
            work[c] = pd.to_numeric(work[c], errors="coerce")
        # Drop rows with NaT date or blank symbol
        work = work.dropna(subset=["date"])
        # Blank symbol after zfill would be "000000" if originally "" -> treat as blank
        # Original blank "" becomes "000000" after zfill(6)?? Actually "".zfill(6)=="000000"
        # So we need to detect original blank before zfill? But spec says drop blank symbol.
        # We consider "000000" as blank if originally empty.
        # Instead we drop where symbol is "000000" or "" or "nan"
        work = work[work["symbol"] != "000000"]
        work = work[work["symbol"].str.strip() != ""]
        work = work[work["symbol"].str.lower() != "nan"]
        # Universe filter
        if cfg.universe_symbols is not None:
            work = work[work["symbol"].isin(cfg.universe_symbols)]
    else:
        # market-level: only date key
        for c in list(work.columns):
            if c == "date":
                continue
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work = work.dropna(subset=["date"])

    # Drop duplicates and sort
    if work.empty:
        # Return empty with correct columns
        return work.reset_index(drop=True)

    # Sort by key_cols
    sort_cols = [c for c in key_cols if c in work.columns]
    if sort_cols:
        work = work.sort_values(sort_cols)
        work = work.drop_duplicates(subset=list(key_cols), keep="last")
        work = work.sort_values(sort_cols)
    work = work.reset_index(drop=True)
    return work
