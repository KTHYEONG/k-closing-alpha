"""공매도 거래량/잔고 수집기."""

from __future__ import annotations

import logging

import pandas as pd

from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.ratelimit import retry_call, wait_for_pykrx_slot

logger = logging.getLogger(__name__)

try:
    from pykrx import stock
except ImportError:  # pragma: no cover
    stock = None  # type: ignore[assignment]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(map(str, df.columns))
    for c in candidates:
        if c in cols:
            return c
    # defensive: normalized matching
    norm_map = {"".join(ch for ch in str(c).strip().lower() if ch.isalnum()): c for c in cols}
    for c in candidates:
        n = "".join(ch for ch in str(c).strip().lower() if ch.isalnum())
        if n in norm_map:
            return norm_map[n]
    return None


def _to_ymd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def collect_shorting(cfg: AltDataFetchConfig, business_days: list[pd.Timestamp]) -> pd.DataFrame:
    """공매도 패널을 수집합니다.

    Args:
        cfg: Alt-data 설정.
        business_days: 영업일 목록.

    Returns:
        수집된 원시 DataFrame.
    """
    if stock is None:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "short_volume",
                "short_value",
                "day_total_volume",
                "short_volume_ratio",
                "short_balance_qty",
                "short_balance_value",
                "listed_shares",
                "short_balance_ratio",
            ]
        )
    rows: list[pd.DataFrame] = []
    for day in business_days:
        ymd = _to_ymd(day)
        for market in cfg.markets:
            def _vol_call() -> pd.DataFrame:
                wait_for_pykrx_slot(cfg)
                return stock.get_shorting_volume_by_ticker(ymd, market)

            def _bal_call() -> pd.DataFrame:
                wait_for_pykrx_slot(cfg)
                return stock.get_shorting_balance_by_ticker(ymd, market)

            vol_df = retry_call(_vol_call, cfg, label=f"shorting_volume {ymd} {market}")
            bal_df = retry_call(_bal_call, cfg, label=f"shorting_balance {ymd} {market}")

            if (vol_df is None or vol_df.empty) and (bal_df is None or bal_df.empty):
                continue

            # Merge on ticker index
            merged: pd.DataFrame | None = None
            if vol_df is not None and not vol_df.empty:
                v = vol_df.copy()
                # map columns defensively
                vol_col = _find_col(v, ["공매도", "공매도거래량", "매도"])
                val_col = _find_col(v, ["공매도금액", "공매도대금", "시가총액"])
                ratio_col = _find_col(v, ["비중", "공매도비중"])
                total_vol_col = _find_col(v, ["매수", "거래량", "일반거래량"])
                v_mapped = pd.DataFrame(index=v.index)
                if vol_col is not None:
                    v_mapped["short_volume"] = pd.to_numeric(v[vol_col], errors="coerce")
                else:
                    # fallback: first numeric column
                    if len(v.columns) > 0:
                        v_mapped["short_volume"] = pd.to_numeric(v.iloc[:, 0], errors="coerce")
                if val_col is not None:
                    v_mapped["short_value"] = pd.to_numeric(v[val_col], errors="coerce")
                if total_vol_col is not None:
                    v_mapped["day_total_volume"] = pd.to_numeric(v[total_vol_col], errors="coerce")
                if ratio_col is not None:
                    v_mapped["short_volume_ratio"] = pd.to_numeric(v[ratio_col], errors="coerce")
                merged = v_mapped

            if bal_df is not None and not bal_df.empty:
                b = bal_df.copy()
                bal_qty_col = _find_col(b, ["공매도잔고", "공매도잔고수량", "잔고"])
                bal_val_col = _find_col(b, ["공매도금액", "공매도잔고금액", "시가총액"])
                listed_col = _find_col(b, ["상장주식수", "상장주식"])
                bal_ratio_col = _find_col(b, ["비중", "공매도비중", "잔고비중"])
                b_mapped = pd.DataFrame(index=b.index)
                if bal_qty_col is not None:
                    b_mapped["short_balance_qty"] = pd.to_numeric(b[bal_qty_col], errors="coerce")
                if bal_val_col is not None:
                    b_mapped["short_balance_value"] = pd.to_numeric(b[bal_val_col], errors="coerce")
                if listed_col is not None:
                    b_mapped["listed_shares"] = pd.to_numeric(b[listed_col], errors="coerce")
                if bal_ratio_col is not None:
                    b_mapped["short_balance_ratio"] = pd.to_numeric(b[bal_ratio_col], errors="coerce")
                # if vol had no ratio, try balance ratio fallback
                if merged is None:
                    merged = b_mapped
                else:
                    merged = merged.join(b_mapped, how="outer")

            if merged is None or merged.empty:
                continue
            merged = merged.copy()
            merged["date"] = pd.Timestamp(day).normalize()
            merged["symbol"] = merged.index.astype(str).str.strip().str.zfill(6)
            # Ensure all expected columns exist
            for col in [
                "short_volume",
                "short_value",
                "day_total_volume",
                "short_volume_ratio",
                "short_balance_qty",
                "short_balance_value",
                "listed_shares",
                "short_balance_ratio",
            ]:
                if col not in merged.columns:
                    merged[col] = pd.NA
            merged = merged.reset_index(drop=True)
            rows.append(merged)

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "short_volume",
                "short_value",
                "day_total_volume",
                "short_volume_ratio",
                "short_balance_qty",
                "short_balance_value",
                "listed_shares",
                "short_balance_ratio",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    # reorder columns
    cols = [
        "date",
        "symbol",
        "short_volume",
        "short_value",
        "day_total_volume",
        "short_volume_ratio",
        "short_balance_qty",
        "short_balance_value",
        "listed_shares",
        "short_balance_ratio",
    ]
    # Ensure columns exist
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]
