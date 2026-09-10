"""분할/감자 감지 가드 (가격제한 이탈 종목의 전체이력 재조회 교정)."""

from __future__ import annotations

import logging

import pandas as pd

from src.backfill.price.config import FetchConfig

logger = logging.getLogger(__name__)


def detect_price_limit_breach(panel: pd.DataFrame, price_limit_ratio: float = 0.30) -> list[str]:
    """가격제한(±30%)을 벗어난 일간 등락이 있는 종목을 반환합니다."""
    if panel is None or panel.empty:
        return []
    if not {"date", "symbol", "close"}.issubset(set(map(str, panel.columns))):
        return []
    lower = 1.0 - float(price_limit_ratio)
    upper = 1.0 / (1.0 - float(price_limit_ratio))
    breached: list[str] = []
    for symbol, group in panel.groupby("symbol", sort=False):
        work = group.sort_values("date")
        close = pd.to_numeric(work["close"], errors="coerce")
        ratio = close / close.shift(1)
        hit = ((ratio < lower) | (ratio > upper)).any()
        if bool(hit):
            breached.append(str(symbol))
    return sorted(set(breached))


def heal_corporate_action_breach(
    merged: pd.DataFrame, fetch_cfg: FetchConfig, market_hint: dict[str, str]
) -> pd.DataFrame:
    """이탈 종목의 전체 이력을 재조회한 값으로 완전히 치환합니다."""
    from src.backfill.price.runner import fetch_one_symbol

    breached = detect_price_limit_breach(merged)
    if not breached:
        return merged
    out = merged.copy()
    for symbol in breached:
        try:
            refetched = fetch_one_symbol(
                symbol,
                fetch_cfg.fixed_start_date,
                fetch_cfg.fixed_end_date,
                market_hint.get(symbol, ""),
                fetch_cfg,
            )
        except Exception:
            logger.warning(
                "[DATA] stage=corporate_action_heal symbol=%s status=REFETCH_FAILED",
                symbol,
            )
            continue
        if refetched is None or refetched.empty:
            logger.warning(
                "[DATA] stage=corporate_action_heal symbol=%s status=REFETCH_FAILED",
                symbol,
            )
            continue
        out = out[out["symbol"] != symbol]
        out = pd.concat([out, refetched], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["symbol", "date"]).drop_duplicates(
        subset=["symbol", "date"], keep="last"
    )
    return out
