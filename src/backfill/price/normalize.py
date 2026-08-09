"""컬럼/플로우 정규화 (column and flow normalization)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backfill.price.config import KRW_100M


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(map(str, df.columns))
    for c in candidates:
        if c in cols:
            return c
    return None


def _sum_present_cols(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return None
    part = df[cols].apply(pd.to_numeric, errors="coerce")
    return part.sum(axis=1, min_count=1)


def _subtract_flow_frames(buy: pd.DataFrame, sell: pd.DataFrame) -> pd.DataFrame:
    if buy is None or sell is None or buy.empty or sell.empty:
        return pd.DataFrame()

    b = buy.copy()
    s = sell.copy()
    b.index = pd.to_datetime(b.index, errors="coerce")
    s.index = pd.to_datetime(s.index, errors="coerce")
    b = b[~b.index.isna()].copy()
    s = s[~s.index.isna()].copy()
    if b.empty or s.empty:
        return pd.DataFrame()

    common_idx = b.index.intersection(s.index)
    common_cols = [c for c in b.columns if c in s.columns]
    if len(common_idx) == 0 or not common_cols:
        return pd.DataFrame()

    out = pd.DataFrame(index=common_idx)
    for c in common_cols:
        out[c] = pd.to_numeric(b.loc[common_idx, c], errors="coerce") - pd.to_numeric(
            s.loc[common_idx, c], errors="coerce"
        )
    out = out.sort_index().dropna(how="all")
    return out


def _to_ymd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def _normalize_investor_flow(flow_df: pd.DataFrame) -> pd.DataFrame:
    if flow_df is None or flow_df.empty:
        return pd.DataFrame(columns=["date", "foreign_netbuy", "inst_netbuy"])

    out = flow_df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].copy()
    out["date"] = out.index

    def _pick_by_keywords(keywords: list[str], exclude: set[str] | None = None) -> str | None:
        exclude = exclude or set()
        for c in out.columns:
            name = str(c).strip().lower()
            if c in exclude:
                continue
            if any(k in name for k in keywords):
                return c
        return None

    foreign_total_col = _pick_by_keywords(["외국", "foreign", "foreigner"])
    inst_total_col = _pick_by_keywords(["기관", "institution", "inst"])

    foreign_detail_cols = [
        c
        for c in out.columns
        if c != foreign_total_col
        and any(k in str(c).strip().lower() for k in ["외국", "foreign", "foreigner"])
    ]
    inst_detail_cols = [
        c
        for c in out.columns
        if c != inst_total_col
        and any(
            k in str(c).strip().lower()
            for k in [
                "기관",
                "institution",
                "inst",
                "금융",
                "보험",
                "투신",
                "연기금",
                "사모",
                "은행",
                "기타",
            ]
        )
        and not any(k in str(c).strip().lower() for k in ["외국", "foreign", "foreigner"])
    ]

    norm = pd.DataFrame({"date": out["date"]})
    if foreign_total_col is not None:
        norm["foreign_netbuy"] = pd.to_numeric(out[foreign_total_col], errors="coerce")
    elif foreign_detail_cols:
        norm["foreign_netbuy"] = (
            out[foreign_detail_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        )
    else:
        norm["foreign_netbuy"] = np.nan

    if inst_total_col is not None:
        norm["inst_netbuy"] = pd.to_numeric(out[inst_total_col], errors="coerce")
    elif inst_detail_cols:
        norm["inst_netbuy"] = (
            out[inst_detail_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)
        )
    else:
        norm["inst_netbuy"] = np.nan

    return norm[["date", "foreign_netbuy", "inst_netbuy"]].drop_duplicates(
        subset=["date"],
        keep="last",
    )


def _normalize_symbol_history(
    ohlcv: pd.DataFrame,
    market_cap_df: pd.DataFrame,
    symbol: str,
    market_hint: str,
) -> pd.DataFrame:
    if ohlcv is None or ohlcv.empty:
        return pd.DataFrame()

    out = ohlcv.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].copy()
    out = out.sort_index()
    out["date"] = out.index

    cols = list(out.columns)
    open_col = _find_col(out, ["시가", "Open", "open"]) or (cols[0] if len(cols) > 0 else None)
    high_col = _find_col(out, ["고가", "High", "high"]) or (cols[1] if len(cols) > 1 else None)
    low_col = _find_col(out, ["저가", "Low", "low"]) or (cols[2] if len(cols) > 2 else None)
    close_col = _find_col(out, ["종가", "Close", "close"]) or (cols[3] if len(cols) > 3 else None)
    vol_col = _find_col(out, ["거래량", "Volume", "volume"]) or (cols[4] if len(cols) > 4 else None)
    value_col = _find_col(out, ["거래대금", "거래금액", "Value", "TradingValue", "acml_tr_pbmn", "trade_value_krw"])
    change_col = _find_col(out, ["등락률", "Change", "change", "prdy_ctrt", "daily_change_pct"])
    
    norm = pd.DataFrame(
        {
            "date": out["date"],
            "symbol": symbol,
            "open": pd.to_numeric(out[open_col], errors="coerce"),
            "high": pd.to_numeric(out[high_col], errors="coerce"),
            "low": pd.to_numeric(out[low_col], errors="coerce"),
            "close": pd.to_numeric(out[close_col], errors="coerce"),
            "volume": pd.to_numeric(out[vol_col], errors="coerce") if vol_col else np.nan,
            "trade_value_krw": pd.to_numeric(out[value_col], errors="coerce") if value_col else np.nan,
            "daily_change_pct_raw": pd.to_numeric(out[change_col], errors="coerce") if change_col else np.nan,
            "market": market_hint,
        }
    )

    cap_col = None
    cap_value_col = None
    if market_cap_df is not None and not market_cap_df.empty:
        cap = market_cap_df.copy()
        # If 'date' is in columns, use it; otherwise use index
        if "date" not in cap.columns:
            cap.index = pd.to_datetime(cap.index, errors="coerce")
            cap = cap[~cap.index.isna()].copy().sort_index()
            cap["date"] = cap.index
        else:
            cap["date"] = pd.to_datetime(cap["date"], errors="coerce")
            cap = cap[~cap["date"].isna()].copy().sort_values("date")
        
        cap_col = _find_col(cap, ["시가총액", "MarketCap", "market_cap"])
        cap_value_col = _find_col(cap, ["거래대금", "거래금액", "Value", "TradingValue"])

        if cap_col is not None:
            # Drop index name to avoid ambiguity during merge if it matches column name
            if cap.index.name == "date":
                cap.index.name = None
            cap_part = cap[["date", cap_col]].rename(columns={cap_col: "market_cap_krw"})
            norm = norm.merge(cap_part, on="date", how="left")
        else:
            norm["market_cap_krw"] = np.nan

        if cap_value_col is not None and norm["trade_value_krw"].isna().all():
            cap_value_part = cap[["date", cap_value_col]].rename(columns={cap_value_col: "trade_value_krw_cap"})
            norm = norm.merge(cap_value_part, on="date", how="left")
            norm["trade_value_krw"] = norm["trade_value_krw"].fillna(
                pd.to_numeric(norm["trade_value_krw_cap"], errors="coerce")
            )
            norm = norm.drop(columns=["trade_value_krw_cap"])
    else:
        norm["market_cap_krw"] = np.nan

    raw = pd.to_numeric(norm["daily_change_pct_raw"], errors="coerce")
    if raw.notna().any() and float(raw.abs().median(skipna=True)) > 1.0:
        norm["daily_change_pct"] = raw / 100.0
    else:
        norm["daily_change_pct"] = raw

    norm["trade_value_100m"] = pd.to_numeric(norm["trade_value_krw"], errors="coerce") / KRW_100M
    norm["market_cap_100m"] = pd.to_numeric(norm["market_cap_krw"], errors="coerce") / KRW_100M
    norm["prev_close"] = norm.groupby("symbol", sort=False)["close"].shift(1)

    keep_cols = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "prev_close",
        "market_cap_100m",
        "trade_value_100m",
        "daily_change_pct",
        "market",
        "volume",
    ]
    norm = norm[keep_cols].sort_values(["symbol", "date"]).reset_index(drop=True)
    norm = norm.drop_duplicates(subset=["symbol", "date"], keep="last")
    return norm
