"""종목별 프로그램매매 일별추이 수집기 (KIS FHPPG04650201 기반)."""

from __future__ import annotations

import asyncio
import logging

import pandas as pd

from src.backfill.altdata.client_pool import fan_out_symbol_calls
from src.backfill.altdata.config import AltDataFetchConfig

logger = logging.getLogger(__name__)

_SCHEMA_COLS = [
    "date",
    "symbol",
    "program_sell_vol",
    "program_buy_vol",
    "program_net_vol",
    "program_sell_value",
    "program_buy_value",
    "program_net_value",
]


def _to_ymd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(columns=_SCHEMA_COLS)


def collect_program_trade_daily(cfg: AltDataFetchConfig, business_days: list[pd.Timestamp]) -> pd.DataFrame:
    """KIS 종목별 프로그램매매 일별추이로 종목-일자 패널을 수집합니다."""
    symbols = sorted(cfg.universe_symbols) if cfg.universe_symbols else []
    if not symbols:
        logger.warning("universe_symbols 없이 program_trade_daily 수집을 건너뜁니다 (종목별 호출 구조라 전체시장 순회 불가)")
        return _empty_panel()
    if not business_days:
        return _empty_panel()
    days = [pd.Timestamp(d).normalize() for d in business_days]
    start_ymd = _to_ymd(min(days))
    end_ymd = _to_ymd(max(days))

    async def _run() -> list[tuple[str, dict]]:
        async def _call(client, session, code):
            return await client.get_program_trade_daily_history(session, code, start_ymd, end_ymd, market_div_code="J")

        def _on_error(code, exc):
            logger.warning("Program trade daily history failed code=%s: %s", code, exc)
            return {"rt_cd": "9", "output": []}

        return await fan_out_symbol_calls(cfg, symbols, _call, _on_error)

    fetched = asyncio.run(_run())
    rows: list[dict] = []
    for code, res in fetched:
        for row in (res.get("output") or res.get("output2") or []):
            day = str(row.get("stck_bsop_date") or "").strip()
            if not day:
                continue
            try:
                dt = pd.Timestamp(pd.to_datetime(day, format="%Y%m%d")).normalize()
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "date": dt,
                    "symbol": str(code).strip().zfill(6),
                    "program_sell_vol": pd.to_numeric(row.get("whol_smtn_seln_vol"), errors="coerce"),
                    "program_buy_vol": pd.to_numeric(row.get("whol_smtn_shnu_vol"), errors="coerce"),
                    "program_net_vol": pd.to_numeric(row.get("whol_smtn_ntby_qty"), errors="coerce"),
                    "program_sell_value": pd.to_numeric(row.get("whol_smtn_seln_tr_pbmn"), errors="coerce"),
                    "program_buy_value": pd.to_numeric(row.get("whol_smtn_shnu_tr_pbmn"), errors="coerce"),
                    "program_net_value": pd.to_numeric(row.get("whol_smtn_ntby_tr_pbmn"), errors="coerce"),
                }
            )
    if not rows:
        return _empty_panel()
    out = pd.DataFrame(rows, columns=_SCHEMA_COLS)
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out
