"""KRX Open API 시장 전체 일별매매정보 백필."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.backfill.backfill_regime import (
    MarketFactorFetchConfig,
    _get_env_value,
    _safe_get_krx_openapi_day,
)

ENDPOINTS = (
    "/svc/apis/sto/stk_bydd_trd",
    "/svc/apis/sto/ksq_bydd_trd",
)


def _normalize_rows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.get("BAS_DD"), format="%Y%m%d", errors="coerce"),
            "symbol": raw.get("ISU_CD", "").astype(str).str.extract(r"(\d{6})", expand=False),
            "open": pd.to_numeric(raw.get("TDD_OPNPRC"), errors="coerce"),
            "high": pd.to_numeric(raw.get("TDD_HGPRC"), errors="coerce"),
            "low": pd.to_numeric(raw.get("TDD_LWPRC"), errors="coerce"),
            "close": pd.to_numeric(raw.get("TDD_CLSPRC"), errors="coerce"),
            "prev_close": pd.to_numeric(raw.get("CMPPREVDD_PRC"), errors="coerce"),
            "volume": pd.to_numeric(raw.get("ACC_TRDVOL"), errors="coerce"),
            "trade_value_100m": pd.to_numeric(raw.get("ACC_TRDVAL"), errors="coerce") / 100_000_000.0,
            "market_cap_100m": pd.to_numeric(raw.get("MKTCAP"), errors="coerce") / 100_000_000.0,
            "daily_change_pct": pd.to_numeric(raw.get("FLUC_RT"), errors="coerce"),
            "market": raw.get("MKT_NM", "UNKNOWN").astype(str),
        }
    )
    return out.dropna(subset=["date", "symbol", "close"]).drop_duplicates(["symbol", "date"])


def collect_krx_daily_history(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    parquet_path: Path,
    request_sleep_sec: float = 0.05,
    allowed_symbols: set[str] | None = None,
) -> pd.DataFrame:
    """시장 전체 KRX 일별매매정보로 기존 parquet의 결측을 보강합니다."""
    auth_key = _get_env_value("KRX_OPENAPI_KEY", "")
    if not auth_key:
        raise RuntimeError("KRX_OPENAPI_KEY is not configured.")
    cfg = MarketFactorFetchConfig(retries=3, krx_request_sleep_sec=request_sleep_sec)
    parts: list[pd.DataFrame] = []
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")
    for dt in dates:
        for endpoint in ENDPOINTS:
            raw, unauthorized = _safe_get_krx_openapi_day(
                date_ymd=dt.strftime("%Y%m%d"),
                endpoint=endpoint,
                auth_key=auth_key,
                cfg=cfg,
                base_urls=["https://data-dbg.krx.co.kr"],
            )
            if unauthorized:
                raise RuntimeError(f"KRX Open API unauthorized: {endpoint}")
            norm = _normalize_rows(raw)
            if not norm.empty:
                parts.append(norm)
            time.sleep(max(0.0, request_sleep_sec))
    if not parts:
        return pd.DataFrame()

    krx = pd.concat(parts, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")
    if allowed_symbols is not None:
        allowed = {str(s).strip().zfill(6) for s in allowed_symbols}
        krx = krx[krx["symbol"].isin(allowed)].copy()
    if parquet_path.exists():
        old = pd.read_parquet(parquet_path)
        old["date"] = pd.to_datetime(old["date"], errors="coerce")
        if allowed_symbols is not None:
            allowed = {str(s).strip().zfill(6) for s in allowed_symbols}
            old = old[old["symbol"].astype(str).str.zfill(6).isin(allowed)].copy()
        merged = old.merge(krx, on=["symbol", "date"], how="outer", suffixes=("", "_krx"))
        for col in [
            "open", "high", "low", "close", "prev_close", "volume",
            "trade_value_100m", "market_cap_100m", "daily_change_pct", "market",
        ]:
            krx_col = f"{col}_krx"
            if krx_col in merged.columns:
                merged[col] = merged[col].fillna(merged[krx_col])
                merged = merged.drop(columns=[krx_col])
        merged = merged.drop_duplicates(["symbol", "date"], keep="last")
    else:
        merged = krx
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(parquet_path, index=False)
    return merged
