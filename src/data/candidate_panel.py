"""조건검색/아카이브 스냅샷에서 가격 백필 유니버스(symbol/market)를 추출하는 모듈."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import settings
from src.daily import archive

logger = logging.getLogger(__name__)


def _default_condition_history_path() -> Path:
    return settings.HISTORY_DIR / "condition_history_cleaned.parquet"

def _resolve_archive_df(archive_df: pd.DataFrame | None) -> pd.DataFrame:
    if archive_df is not None:
        return archive_df
    try:
        df = archive.fetch_archive_snapshot(all_rows=True)
    except Exception as exc:
        logger.warning("[DATA] stage=candidate_panel status=no_archive error=%s", exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    return df

def load_candidate_universe_symbols(
    condition_history_path: Path | None = None,
    archive_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Deduplicated (symbol, market) universe from candidate sources (column-pruned)."""
    frames: list[pd.DataFrame] = []
    path = condition_history_path or _default_condition_history_path()
    try:
        cdf = pd.read_parquet(path, columns=["종목코드", "시장구분"])
    except Exception:
        try:
            cdf = pd.read_parquet(path, columns=["종목코드"])
        except Exception as exc:
            logger.warning("[DATA] stage=candidate_universe status=no_condition_history path=%s error=%s", path, exc)
            cdf = pd.DataFrame()
    if cdf is not None and not cdf.empty and "종목코드" in cdf.columns:
        part = pd.DataFrame({"symbol": cdf["종목코드"]})
        part["market"] = cdf["시장구분"] if "시장구분" in cdf.columns else np.nan
        frames.append(part)

    arch = _resolve_archive_df(archive_df)
    if arch is not None and not arch.empty and "종목코드" in arch.columns:
        part = pd.DataFrame({"symbol": arch["종목코드"]})
        part["market"] = arch["시장구분"] if "시장구분" in arch.columns else np.nan
        frames.append(part)

    if not frames:
        return pd.DataFrame(columns=["symbol", "market"])
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["symbol"] = out["symbol"].astype(str).str.strip().str.zfill(6)
    out["market"] = out["market"].astype(str).fillna("UNKNOWN")
    out.loc[out["market"].isin({"nan", "None", ""}), "market"] = "UNKNOWN"
    out = out[out["symbol"].str.fullmatch(r"\d{6}", na=False)].copy()
    out = out.drop_duplicates(subset=["symbol"], keep="last")
    return out[["symbol", "market"]].reset_index(drop=True)
