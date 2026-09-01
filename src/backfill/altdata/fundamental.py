"""밸류에이션 펀더멘털 수집기."""

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


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = list(map(str, df.columns))
    for c in candidates:
        if c in cols:
            return c
    norm_map = {"".join(ch for ch in str(c).strip().lower() if ch.isalnum()): c for c in cols}
    for c in candidates:
        n = "".join(ch for ch in str(c).strip().lower() if ch.isalnum())
        if n in norm_map:
            return norm_map[n]
    return None


def collect_fundamental(cfg: AltDataFetchConfig, business_days: list[pd.Timestamp]) -> pd.DataFrame:
    """펀더멘털 패널을 수집합니다.

    Args:
        cfg: Alt-data 설정.
        business_days: 영업일 목록.

    Returns:
        수집된 원시 DataFrame.
    """
    if stock is None:
        return pd.DataFrame(columns=["date", "symbol", "per", "pbr", "eps", "bps", "div_yield", "dps"])
    rows: list[pd.DataFrame] = []
    for day in business_days:
        ymd = _to_ymd(day)
        for market in cfg.markets:
            def _call() -> pd.DataFrame:
                wait_for_pykrx_slot(cfg)
                return stock.get_market_fundamental_by_ticker(ymd, market)

            df = retry_call(_call, cfg, label=f"fundamental {ymd} {market}")
            if df is None or df.empty:
                continue
            work = df.copy()
            # Map columns
            bps_col = _find_col(work, ["BPS"])
            per_col = _find_col(work, ["PER"])
            pbr_col = _find_col(work, ["PBR"])
            eps_col = _find_col(work, ["EPS"])
            div_col = _find_col(work, ["DIV"])
            dps_col = _find_col(work, ["DPS"])
            mapped = pd.DataFrame(index=work.index)
            mapped["bps"] = pd.to_numeric(work[bps_col], errors="coerce") if bps_col else pd.NA
            mapped["per"] = pd.to_numeric(work[per_col], errors="coerce") if per_col else pd.NA
            mapped["pbr"] = pd.to_numeric(work[pbr_col], errors="coerce") if pbr_col else pd.NA
            mapped["eps"] = pd.to_numeric(work[eps_col], errors="coerce") if eps_col else pd.NA
            mapped["div_yield"] = pd.to_numeric(work[div_col], errors="coerce") if div_col else pd.NA
            mapped["dps"] = pd.to_numeric(work[dps_col], errors="coerce") if dps_col else pd.NA
            mapped["date"] = pd.Timestamp(day).normalize()
            mapped["symbol"] = mapped.index.astype(str).str.strip().str.zfill(6)
            mapped = mapped.reset_index(drop=True)
            rows.append(mapped[["date", "symbol", "per", "pbr", "eps", "bps", "div_yield", "dps"]])

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "per", "pbr", "eps", "bps", "div_yield", "dps"])
    out = pd.concat(rows, ignore_index=True)
    return out[["date", "symbol", "per", "pbr", "eps", "bps", "div_yield", "dps"]]
