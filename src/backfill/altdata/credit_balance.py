"""신용잔고 일별추이 수집기 (KIS FHPST04760000 기반)."""

from __future__ import annotations

import asyncio
import logging

import pandas as pd

from src.api.kis.client import KisApiClient
from src.backfill.altdata.config import AltDataFetchConfig

logger = logging.getLogger(__name__)

_SCHEMA_COLS = [
    "date",
    "symbol",
    "loan_new_qty",
    "loan_redemption_qty",
    "loan_balance_qty",
    "loan_balance_amt",
    "loan_balance_rate",
    "stln_balance_qty",
    "stln_balance_amt",
    "stln_balance_rate",
]


def _to_ymd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(columns=_SCHEMA_COLS)


def collect_credit_balance(cfg: AltDataFetchConfig, business_days: list[pd.Timestamp]) -> pd.DataFrame:
    """KIS 신용잔고 일별추이로 종목-일자 패널을 수집합니다."""
    symbols = sorted(cfg.universe_symbols) if cfg.universe_symbols else []
    if not symbols:
        logger.warning("universe_symbols 없이 credit_balance 수집을 건너뜁니다 (종목별 호출 구조라 전체시장 순회 불가)")
        return _empty_panel()
    if not business_days:
        return _empty_panel()
    days = [pd.Timestamp(d).normalize() for d in business_days]
    start_ymd = _to_ymd(min(days))
    end_ymd = _to_ymd(max(days))

    async def _run() -> list[tuple[str, dict]]:
        client = KisApiClient()
        async with client.create_session() as session:
            await client.ensure_token(session)
            sem = asyncio.Semaphore(10)

            async def _one(code: str) -> tuple[str, dict]:
                async with sem:
                    try:
                        res = await client.get_daily_credit_balance_history(
                            session, code, start_ymd, end_ymd, market_div_code="J"
                        )
                    except Exception as e:
                        logger.warning("Daily credit balance history failed code=%s: %s", code, e)
                        return code, {"rt_cd": "9", "output": []}
                    return code, res

            return list(await asyncio.gather(*[_one(c) for c in symbols]))

    fetched = asyncio.run(_run())
    rows: list[dict] = []
    for code, res in fetched:
        for row in (res.get("output") or res.get("output2") or []):
            day = str(row.get("deal_date") or "").strip()
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
                    "loan_new_qty": pd.to_numeric(row.get("whol_loan_new_stcn"), errors="coerce"),
                    "loan_redemption_qty": pd.to_numeric(row.get("whol_loan_rdmp_stcn"), errors="coerce"),
                    "loan_balance_qty": pd.to_numeric(row.get("whol_loan_rmnd_stcn"), errors="coerce"),
                    "loan_balance_amt": pd.to_numeric(row.get("whol_loan_rmnd_amt"), errors="coerce"),
                    "loan_balance_rate": pd.to_numeric(row.get("whol_loan_rmnd_rate"), errors="coerce"),
                    "stln_balance_qty": pd.to_numeric(row.get("whol_stln_rmnd_stcn"), errors="coerce"),
                    "stln_balance_amt": pd.to_numeric(row.get("whol_stln_rmnd_amt"), errors="coerce"),
                    "stln_balance_rate": pd.to_numeric(row.get("whol_stln_rmnd_rate"), errors="coerce"),
                }
            )
    if not rows:
        return _empty_panel()
    out = pd.DataFrame(rows, columns=_SCHEMA_COLS)
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out
