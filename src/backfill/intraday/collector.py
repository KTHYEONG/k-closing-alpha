"""당일 정규세션/NXT 애프터마켓 1분봉 수집기 (워치리스트 스코프)."""

from __future__ import annotations

import asyncio
import logging

import pandas as pd

from src.config.market_session import (
    KRX_CLOSE_MARKET_DIV_CODE,
    KRX_REGULAR_HOUR_CEIL,
    KRX_REGULAR_HOUR_FLOOR,
    NXT_AFTERMARKET_HOUR_CEIL,
    NXT_AFTERMARKET_HOUR_FLOOR,
    NXT_MARKET_DIV_CODE,
)

logger = logging.getLogger(__name__)


async def _collect_bars(
    client,
    session,
    stock_codes: list[str],
    snapshot_date: str,
    bar_interval_minutes: int,
    end_hour: str,
    floor_hour: str,
    market_div_code: str,
    historical: bool = False,
) -> pd.DataFrame:
    if not stock_codes:
        return pd.DataFrame()
    sem = asyncio.Semaphore(10)
    target_date = str(snapshot_date).replace("-", "") if historical else ""

    async def _fetch_one(code: str) -> pd.DataFrame:
        async with sem:
            try:
                if historical:
                    res = await client.get_historical_minute_chart(
                        session,
                        code,
                        target_date,
                        bar_interval_minutes=bar_interval_minutes,
                        end_hour=end_hour,
                        floor_hour=floor_hour,
                        market_div_code=market_div_code,
                    )
                else:
                    res = await client.get_intraday_minute_chart(
                        session,
                        code,
                        bar_interval_minutes=bar_interval_minutes,
                        end_hour=end_hour,
                        floor_hour=floor_hour,
                        market_div_code=market_div_code,
                    )
            except Exception as e:
                logger.warning("Intraday bars failed code=%s: %s", code, e)
                return pd.DataFrame()
            if res.get("rt_cd") != "0":
                return pd.DataFrame()
            rows = res.get("output2") or []
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["종목코드"] = str(code).zfill(6)
            df["스냅샷_날짜"] = snapshot_date
            return df

    results = await asyncio.gather(*[_fetch_one(c) for c in stock_codes])
    frames = [d for d in results if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


async def collect_intraday_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1) -> pd.DataFrame:
    """당일 정규세션 1분봉을 세마포어(10) 동시성으로 취합한다."""
    return await _collect_bars(
        client, session, stock_codes, snapshot_date, bar_interval_minutes,
        KRX_REGULAR_HOUR_CEIL, KRX_REGULAR_HOUR_FLOOR, KRX_CLOSE_MARKET_DIV_CODE,
    )


async def collect_nxt_aftermarket_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1) -> pd.DataFrame:
    """NXT 애프터마켓(15:40-20:00) 전체를 1분봉 연속 시계열로 수집한다. 미상장은 조용히 스킵."""
    return await _collect_bars(
        client, session, stock_codes, snapshot_date, bar_interval_minutes,
        NXT_AFTERMARKET_HOUR_CEIL, NXT_AFTERMARKET_HOUR_FLOOR, NXT_MARKET_DIV_CODE,
    )


async def collect_intraday_trade_ticks(client, session, stock_codes: list[str], snapshot_date: str) -> pd.DataFrame:
    """KRX 정규세션(09:00~15:30) 틱 체결을 세마포어(10) 동시성으로 취합한다."""
    if not stock_codes:
        return pd.DataFrame()
    sem = asyncio.Semaphore(10)

    async def _fetch_one(code: str) -> pd.DataFrame:
        async with sem:
            try:
                res = await client.get_intraday_trade_ticks(
                    session,
                    code,
                    floor_hour=KRX_REGULAR_HOUR_FLOOR,
                    end_hour=KRX_REGULAR_HOUR_CEIL,
                    market_div_code=KRX_CLOSE_MARKET_DIV_CODE,
                )
            except Exception as e:
                logger.warning("Intraday trade ticks failed code=%s: %s", code, e)
                return pd.DataFrame()
            if res.get("rt_cd") != "0":
                return pd.DataFrame()
            rows = res.get("output2") or []
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            df["종목코드"] = str(code).zfill(6)
            df["스냅샷_날짜"] = snapshot_date
            return df

    results = await asyncio.gather(*[_fetch_one(c) for c in stock_codes])
    frames = [d for d in results if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


async def backfill_regular_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1) -> pd.DataFrame:
    """특정 과거 날짜(snapshot_date, 'YYYY-MM-DD')의 정규세션 1분봉을 FHKST03010230으로 소급 수집."""
    return await _collect_bars(
        client, session, stock_codes, snapshot_date, bar_interval_minutes,
        KRX_REGULAR_HOUR_CEIL, KRX_REGULAR_HOUR_FLOOR, KRX_CLOSE_MARKET_DIV_CODE,
        historical=True,
    )


async def backfill_nxt_aftermarket_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1) -> pd.DataFrame:
    """특정 과거 날짜의 NXT 애프터마켓 1분봉을 FHKST03010230으로 소급 수집.

    FHKST03010230이 애프터마켓 시간대(15:40-20:00)를 실제로 보관하는지는 실측 미검증
    상태이므로 보관하지 않는 것으로 확인되면 이 함수 호출부를 제거한다. 실패 시 정상
    스킵하며 정규세션 백필(backfill_regular_bars)과 완전히 독립적으로 동작한다.
    """
    return await _collect_bars(
        client, session, stock_codes, snapshot_date, bar_interval_minutes,
        NXT_AFTERMARKET_HOUR_CEIL, NXT_AFTERMARKET_HOUR_FLOOR, NXT_MARKET_DIV_CODE,
        historical=True,
    )
