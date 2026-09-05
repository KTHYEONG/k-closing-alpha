"""당일 정규세션/NXT 애프터마켓 1분봉 수집기 (워치리스트 스코프)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd

from src.api.ls.client import LsApiClient  # noqa: F401 - wiring per spec
from src.config.market_session import (
    KRX_CLOSE_MARKET_DIV_CODE,
    KRX_REGULAR_HOUR_CEIL,
    KRX_REGULAR_HOUR_FLOOR,
    NXT_AFTERMARKET_HOUR_CEIL,
    NXT_AFTERMARKET_HOUR_FLOOR,
    NXT_MARKET_DIV_CODE,
)
from src.data.intraday_schema import normalize_bar_frame, normalize_tick_frame

logger = logging.getLogger(__name__)


def _canonical_kis_bars(rows: list[dict], snapshot_date: str, code: str) -> pd.DataFrame:
    vendor = "kis"
    try:
        return normalize_bar_frame(pd.DataFrame(rows), vendor, snapshot_date, code)
    except Exception as e:
        logger.warning("[DATA] KIS bar normalize failed code=%s: %s", code, e)
        return pd.DataFrame()


def _canonical_kis_ticks(rows: list[dict], snapshot_date: str, code: str) -> pd.DataFrame:
    vendor = "kis"
    prepared = [dict(r) for r in rows]
    if prepared and not any(k in prepared[0] for k in ("cnqn", "cntg_vol")) and "acml_vol" in prepared[0]:
        # acml_vol(누적)만 있는 응답은 정렬 후 1차 차분으로 봉당 체결량(cnqn)을 합성한다.
        # 파싱 실패는 개별 값만 0으로 클램프하고 계속 진행한다 (전체 배치를 버리지 않음).
        ordered = sorted(prepared, key=lambda r: (str(r.get("stck_cntg_hour", "")), str(r.get("acml_vol", "0"))))
        prev = 0
        for rec in ordered:
            try:
                cur = int(str(rec.get("acml_vol", "0")).strip() or "0")
            except ValueError:
                cur = 0
            rec["cnqn"] = str(max(cur - prev, 0))
            prev = cur
        prepared = ordered
    try:
        return normalize_tick_frame(pd.DataFrame(prepared), vendor, snapshot_date, code)
    except Exception as e:
        logger.warning("[DATA] KIS tick normalize failed code=%s: %s", code, e)
        return pd.DataFrame()


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
            return _canonical_kis_bars(rows, snapshot_date, code)

    results = await asyncio.gather(*[_fetch_one(c) for c in stock_codes])
    frames = [d for d in results if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


async def collect_intraday_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1, ls_client: Any | None = None) -> pd.DataFrame:
    """당일 정규세션 1분봉을 세마포어(10) 동시성으로 취합한다."""
    if ls_client is None:
        return await _collect_bars(
            client, session, stock_codes, snapshot_date, bar_interval_minutes,
            KRX_REGULAR_HOUR_CEIL, KRX_REGULAR_HOUR_FLOOR, KRX_CLOSE_MARKET_DIV_CODE,
        )
    if not stock_codes:
        return pd.DataFrame()
    sem = asyncio.Semaphore(10)
    target_date = snapshot_date

    async def _fetch_one(code: str) -> pd.DataFrame:
        async with sem:
            try:
                res = await ls_client.get_minute_chart(session, code, target_date)
            except Exception as e:
                logger.warning("LS minute chart failed code=%s: %s", code, e)
                res = {"rt_cd": "1", "output2": []}
            if res.get("rt_cd") == "0" and (res.get("output2") or []):
                rows = res.get("output2") or []
                logger.info("Fetched %d minute bars for %s via LS", len(rows), code)
                vendor = str(res.get("vendor", "ls") or "ls")
                try:
                    return normalize_bar_frame(pd.DataFrame(rows), vendor, snapshot_date, code)
                except Exception as e:
                    logger.warning("[DATA] LS bar normalize failed code=%s: %s", code, e)
                    return pd.DataFrame()
            else:
                try:
                    res = await client.get_intraday_minute_chart(
                        session, code, bar_interval_minutes=bar_interval_minutes,
                        end_hour=KRX_REGULAR_HOUR_CEIL, floor_hour=KRX_REGULAR_HOUR_FLOOR,
                        market_div_code=KRX_CLOSE_MARKET_DIV_CODE,
                    )
                except Exception as e:
                    logger.warning("Intraday bars failed code=%s: %s", code, e)
                    return pd.DataFrame()
                if res.get("rt_cd") != "0":
                    return pd.DataFrame()
                rows = res.get("output2") or []
                logger.info("Fetched %d minute bars for %s via KIS fallback", len(rows), code)
            if not rows:
                return pd.DataFrame()
            return _canonical_kis_bars(rows, snapshot_date, code)

    results = await asyncio.gather(*[_fetch_one(c) for c in stock_codes])
    frames = [d for d in results if d is not None and not d.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


async def collect_nxt_aftermarket_bars(client, session, stock_codes: list[str], snapshot_date: str, bar_interval_minutes: int = 1) -> pd.DataFrame:
    """NXT 애프터마켓(15:40-20:00) 전체를 1분봉 연속 시계열로 수집한다. 미상장은 조용히 스킵."""
    return await _collect_bars(
        client, session, stock_codes, snapshot_date, bar_interval_minutes,
        NXT_AFTERMARKET_HOUR_CEIL, NXT_AFTERMARKET_HOUR_FLOOR, NXT_MARKET_DIV_CODE,
    )


async def collect_intraday_trade_ticks(client, session, stock_codes: list[str], snapshot_date: str, ls_client: Any | None = None) -> pd.DataFrame:
    """KRX 정규세션(09:00~15:30) 틱 체결을 세마포어(10) 동시성으로 취합한다."""
    if not stock_codes:
        return pd.DataFrame()
    sem = asyncio.Semaphore(10)
    target_date = snapshot_date

    async def _fetch_one(code: str) -> list[dict]:
        async with sem:
            if ls_client is not None:
                try:
                    ls_res = await ls_client.get_tick_chart(session, code, target_date)
                except Exception as e:
                    logger.warning("LS tick chart failed code=%s: %s", code, e)
                    ls_res = {"rt_cd": "1", "output2": []}
                if ls_res.get("rt_cd") == "0" and (ls_res.get("output2") or []):
                    rows = ls_res.get("output2") or []
                    truncated = bool(ls_res.get("truncated", False))
                    vendor = str(ls_res.get("vendor", "ls") or "ls")
                    logger.info("Fetched %d ticks for %s via LS", len(rows), code)
                    try:
                        frame = normalize_tick_frame(pd.DataFrame(rows), vendor, snapshot_date, code, truncated=truncated)
                    except Exception as e:
                        logger.warning("[DATA] LS tick normalize failed code=%s: %s", code, e)
                        return []
                    return frame.to_dict("records") if not frame.empty else []
                if ls_res.get("rt_cd") == "0":
                    return []
            try:
                res = await client.get_intraday_trade_ticks(session, code, floor_hour=KRX_REGULAR_HOUR_FLOOR, end_hour=KRX_REGULAR_HOUR_CEIL, market_div_code=KRX_CLOSE_MARKET_DIV_CODE)
            except Exception as e:
                logger.warning("Intraday trade ticks failed code=%s: %s", code, e)
                return []
            if res.get("rt_cd") != "0":
                return []
            rows = res.get("output2") or []
            if not rows:
                return []
            frame = _canonical_kis_ticks(rows, snapshot_date, code)
            logger.info("Fetched %d ticks for %s via KIS fallback", len(frame), code)
            return frame.to_dict("records") if not frame.empty else []

    results = await asyncio.gather(*[_fetch_one(c) for c in stock_codes])
    all_rows: list[dict] = []
    for rows in results:
        if rows:
            all_rows.extend(rows)
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


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
