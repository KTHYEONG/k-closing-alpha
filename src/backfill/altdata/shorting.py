"""공매도 거래량 수집기 (KIS 네이티브 TR 기반)."""

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
    "short_volume",
    "short_value",
    "day_total_volume",
    "short_volume_ratio",
    "short_balance_qty",
    "short_balance_value",
    "listed_shares",
    "short_balance_ratio",
]


def _to_ymd(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y%m%d")


def _empty_panel() -> pd.DataFrame:
    return pd.DataFrame(columns=_SCHEMA_COLS)


def collect_shorting(cfg: AltDataFetchConfig, business_days: list[pd.Timestamp]) -> pd.DataFrame:
    """KIS 공매도 일별추이(FHPST04830000)로 종목-일자 패널을 수집합니다.

    거래(체결) 측 4개 컬럼(short_volume/short_value/day_total_volume/
    short_volume_ratio)만 채우고, 잔고 측 4개 컬럼(short_balance_qty/
    short_balance_value/listed_shares/short_balance_ratio)은 대응하는 KIS
    잔고 TR을 이번 조사 범위에서 발견하지 못해 NaN으로 남긴다(부분 커버리지).
    """
    symbols = sorted(cfg.universe_symbols) if cfg.universe_symbols else []
    if not symbols:
        logger.warning("universe_symbols 없이 shorting 수집을 건너뜁니다 (종목별 호출 구조라 전체시장 순회 불가)")
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
                        res = await client.get_daily_short_sale_history(
                            session, code, start_ymd, end_ymd, market_div_code="J"
                        )
                    except Exception as e:
                        logger.warning("Daily short sale history failed code=%s: %s", code, e)
                        return code, {"rt_cd": "9", "output2": []}
                    return code, res

            return list(await asyncio.gather(*[_one(c) for c in symbols]))

    fetched = asyncio.run(_run())
    rows: list[dict] = []
    for code, res in fetched:
        for row in (res.get("output2") or []):
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
                    "short_volume": pd.to_numeric(row.get("ssts_cntg_qty"), errors="coerce"),
                    "short_value": pd.to_numeric(row.get("ssts_tr_pbmn"), errors="coerce"),
                    "day_total_volume": pd.to_numeric(row.get("acml_vol"), errors="coerce"),
                    "short_volume_ratio": pd.to_numeric(row.get("ssts_vol_rlim"), errors="coerce"),
                    "short_balance_qty": float("nan"),
                    "short_balance_value": float("nan"),
                    "listed_shares": float("nan"),
                    "short_balance_ratio": float("nan"),
                }
            )
    if not rows:
        return _empty_panel()
    out = pd.DataFrame(rows, columns=_SCHEMA_COLS)
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out
