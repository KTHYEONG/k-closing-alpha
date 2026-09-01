"""투자자별 수급 상세 수집기."""

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

# Korean -> english suffix mapping
_INVESTOR_TYPES: dict[str, str] = {
    "개인": "individual",
    "외국인": "foreign",
    "기관합계": "institution_total",
    "금융투자": "financial_invest",
    "보험": "insurance",
    "투신": "trust",
    "사모": "private_equity",
    "은행": "bank",
    "연기금": "pension",
    "기타법인": "other_corp",
}


def _to_ymd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def collect_investor_detail(cfg: AltDataFetchConfig, business_days: list[pd.Timestamp]) -> pd.DataFrame:
    """투자자별 순매수 패널을 수집합니다.

    Args:
        cfg: Alt-data 설정.
        business_days: 영업일 목록.

    Returns:
        수집된 원시 DataFrame.
    """
    if stock is None:
        cols = ["date", "symbol"] + [f"net_{v}" for v in _INVESTOR_TYPES.values()]
        return pd.DataFrame(columns=cols)
    # Collect per day per market per investor
    # Returns per-ticker net purchase VALUE for that single day
    cols = ["date", "symbol"] + [f"net_{v}" for v in _INVESTOR_TYPES.values()]
    # Accumulate dict keyed by (date, symbol) -> dict of net values
    data: dict[tuple[pd.Timestamp, str], dict[str, float]] = {}
    for day in business_days:
        ymd = _to_ymd(day)
        for market in cfg.markets:
            for kor, eng in _INVESTOR_TYPES.items():
                def _call(k=kor, m=market) -> pd.DataFrame:  # type: ignore[no-untyped-def]
                    wait_for_pykrx_slot(cfg)
                    return stock.get_market_net_purchases_of_equities_by_ticker(ymd, ymd, m, k)

                df = retry_call(_call, cfg, label=f"investor {ymd} {market} {kor}")
                if df is None or df.empty:
                    continue
                # df index is ticker, columns include some value; take first numeric column
                # Typically column is 순매수 or similar; just pick first column
                work = df.copy()
                # Find value column: take first column
                val_col = list(work.columns)[0] if len(work.columns) > 0 else None
                if val_col is None:
                    continue
                for ticker, val in work[val_col].items():
                    sym = str(ticker).strip().zfill(6)
                    key = (pd.Timestamp(day).normalize(), sym)
                    if key not in data:
                        data[key] = {}
                    try:
                        data[key][f"net_{eng}"] = float(pd.to_numeric(val, errors="coerce"))
                    except Exception:
                        data[key][f"net_{eng}"] = float("nan")

    if not data:
        return pd.DataFrame(columns=cols)
    rows = []
    for (d, sym), vals in data.items():
        row: dict[str, object] = {"date": d, "symbol": sym}
        for eng in _INVESTOR_TYPES.values():
            row[f"net_{eng}"] = vals.get(f"net_{eng}", float("nan"))
        rows.append(row)
    out = pd.DataFrame(rows)
    # Ensure columns
    for c in cols:
        if c not in out.columns:
            out[c] = float("nan")
    return out[cols]
