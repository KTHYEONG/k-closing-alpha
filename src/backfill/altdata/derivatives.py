"""KOSPI200 지수-선물 베이시스 수집기."""

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


def _to_ymd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def collect_derivatives_basis(cfg: AltDataFetchConfig, business_days: list[pd.Timestamp]) -> pd.DataFrame:
    """파생 베이시스 패널을 수집합니다.

    Args:
        cfg: Alt-data 설정.
        business_days: 영업일 목록.

    Returns:
        수집된 원시 DataFrame.
    """
    cols = ["date", "kospi200_close", "k200_future_close", "basis", "basis_pct", "future_volume", "future_open_interest"]
    if stock is None or not business_days:
        return pd.DataFrame(columns=cols)

    # Spot: get_index_ohlcv for KOSPI200 (code 1028)
    def _spot_call() -> pd.DataFrame:
        wait_for_pykrx_slot(cfg)
        start_ymd = _to_ymd(business_days[0])
        end_ymd = _to_ymd(business_days[-1])
        return stock.get_index_ohlcv(start_ymd, end_ymd, "1028")

    spot_df = retry_call(_spot_call, cfg, label="derivatives spot 1028")
    # Build map date -> close
    spot_map: dict[pd.Timestamp, float] = {}
    if spot_df is not None and not spot_df.empty:
        work = spot_df.copy()
        # index is date
        work.index = pd.to_datetime(work.index, errors="coerce")
        # Find close column
        close_col = None
        for cand in ["종가", "Close", "close"]:
            if cand in work.columns:
                close_col = cand
                break
        if close_col is None and len(work.columns) > 3:
            close_col = list(work.columns)[3]
        if close_col is not None:
            for idx, val in work[close_col].items():
                d = pd.Timestamp(idx).normalize()
                try:
                    spot_map[d] = float(pd.to_numeric(val, errors="coerce"))
                except Exception:
                    continue

    rows: list[dict[str, object]] = []
    for day in business_days:
        ymd = _to_ymd(day)
        d_norm = pd.Timestamp(day).normalize()
        kospi200_close = spot_map.get(d_norm, float("nan"))

        # Future per day
        def _fut_call() -> pd.DataFrame:
            wait_for_pykrx_slot(cfg)
            return stock.get_future_ohlcv_by_ticker(ymd)

        fut_df = retry_call(_fut_call, cfg, label=f"derivatives future {ymd}")
        future_close = float("nan")
        future_vol = float("nan")
        future_oi = float("nan")
        if fut_df is not None and not fut_df.empty:
            work = fut_df.copy()
            # Try to pick nearest expiry >= ymd
            # If expiry column exists, parse it
            expiry_col = None
            for cand in ["만기일", "expiry", "Expiry", "expire"]:
                if cand in work.columns:
                    expiry_col = cand
                    break
            close_col2 = None
            for cand in ["종가", "Close", "close"]:
                if cand in work.columns:
                    close_col2 = cand
                    break
            if close_col2 is None and len(work.columns) > 0:
                close_col2 = list(work.columns)[0]
            vol_col = None
            for cand in ["거래량", "Volume", "volume"]:
                if cand in work.columns:
                    vol_col = cand
                    break
            oi_col = None
            for cand in ["미결제약정", "open_interest", "OpenInterest"]:
                if cand in work.columns:
                    oi_col = cand
                    break
            # If expiry available, filter
            if expiry_col is not None:
                try:
                    work["_expiry_parsed"] = pd.to_datetime(work[expiry_col], errors="coerce")
                    ymd_ts = pd.Timestamp(ymd)
                    # Keep rows where expiry >= ymd
                    valid = work[work["_expiry_parsed"] >= ymd_ts]
                    if not valid.empty:
                        valid = valid.sort_values("_expiry_parsed")
                        picked = valid.iloc[0]
                    else:
                        # fallback to first row
                        picked = work.iloc[0]
                    if close_col2 is not None:
                        future_close = float(pd.to_numeric(picked[close_col2], errors="coerce"))
                    if vol_col is not None:
                        future_vol = float(pd.to_numeric(picked[vol_col], errors="coerce"))
                    if oi_col is not None:
                        future_oi = float(pd.to_numeric(picked[oi_col], errors="coerce"))
                except Exception:
                    # fallback to first row close
                    if close_col2 is not None:
                        try:
                            future_close = float(pd.to_numeric(work.iloc[0][close_col2], errors="coerce"))
                        except Exception:
                            pass
            else:
                # No expiry column: take first row
                if close_col2 is not None:
                    try:
                        future_close = float(pd.to_numeric(work.iloc[0][close_col2], errors="coerce"))
                    except Exception:
                        pass
                if vol_col is not None:
                    try:
                        future_vol = float(pd.to_numeric(work.iloc[0][vol_col], errors="coerce"))
                    except Exception:
                        pass
                if oi_col is not None:
                    try:
                        future_oi = float(pd.to_numeric(work.iloc[0][oi_col], errors="coerce"))
                    except Exception:
                        pass

        # Compute basis
        try:
            if pd.notna(kospi200_close) and pd.notna(future_close):
                basis = float(future_close) - float(kospi200_close)
                basis_pct = float(basis) / float(kospi200_close) if float(kospi200_close) != 0 else float("nan")
            else:
                basis = float("nan")
                basis_pct = float("nan")
        except Exception:
            basis = float("nan")
            basis_pct = float("nan")

        rows.append(
            {
                "date": d_norm,
                "kospi200_close": kospi200_close,
                "k200_future_close": future_close,
                "basis": basis,
                "basis_pct": basis_pct,
                "future_volume": future_vol,
                "future_open_interest": future_oi,
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    return out
