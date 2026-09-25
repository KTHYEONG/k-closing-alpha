"""KRX 거래일 판정 (KRX 공식 지수 일별매매정보 기반)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import aiohttp
import pandas as pd

from src import settings
from src.backfill.altdata.config import AltDataFetchConfig
from src.backfill.altdata.krx_api import (
    KRX_ENDPOINT_KOSPI_INDEX_DAILY,
    fetch_krx_openapi_day_strict,
)

logger = logging.getLogger(__name__)

_TRADING_DAY_CACHE: dict[str, bool] = {}

DAY_WEEKEND: str = "weekend"
DAY_HOLIDAY: str = "holiday"
DAY_TRADING: str = "trading"
# 달력 조회 장애: 휴장일로 단정하지 않고 감사를 수행한다(장애 조기 발견 우선)
DAY_UNKNOWN: str = "unknown"


def is_krx_trading_day(date: pd.Timestamp | str, cfg: AltDataFetchConfig | None = None) -> bool:
    """KRX 공식 응답 행수>0을 거래일 권위 판정으로 사용합니다.

    지수 일별매매정보(51행/0.15s)의 행 존재 여부를 판정 근거로 씁니다.
    네트워크/인증 장애는 ``False`` 로 삼키지 않고 그대로 전파합니다.

    Args:
        date: 판정 대상일.
        cfg: Alt-data 설정. ``None`` 이면 기본 설정을 생성합니다.

    Returns:
        거래일이면 ``True``, 휴장일(0행)이면 ``False``.
    """
    key = pd.Timestamp(date).strftime("%Y%m%d")
    if key in _TRADING_DAY_CACHE:
        return _TRADING_DAY_CACHE[key]
    if cfg is None:
        # 이 경로는 cfg의 API 접근 필드(krx_api_key/레이트리밋)만 사용한다.
        # start/end/out_dir는 수집 창 설정이라 여기선 의미가 없지만 필수 인자라 채운다.
        # krx_api_key를 빠뜨리면 strict 페처가 ValueError로 죽으므로 settings에서 주입한다.
        target = pd.Timestamp(date).normalize()
        cfg = AltDataFetchConfig(
            start=target,
            end=target + pd.Timedelta(days=1),
            out_dir=Path("."),
            krx_api_key=settings.KRX_OPENAPI_KEY,
        )
    rows = fetch_krx_openapi_day_strict(KRX_ENDPOINT_KOSPI_INDEX_DAILY, key, cfg)
    result = len(rows) > 0
    _TRADING_DAY_CACHE[key] = result
    return result


async def is_kis_trading_day(client, session, date: pd.Timestamp | str) -> bool:
    """KIS 지수 일별시세(0001) 응답에 요청일이 존재하면 거래일로 판정한다.

    rt_cd != "0"인 장애 응답은 휴장일로 강제하지 않고 RuntimeError로 전파한다.
    """
    ymd = pd.Timestamp(date).strftime("%Y%m%d")
    res = await client.get_market_index_history(session, "0001", ymd, ymd)
    if res.get("rt_cd") != "0":
        raise RuntimeError(f"KIS trading-day oracle failed rt_cd={res.get('rt_cd')} msg={res.get('msg1', '')}")
    rows = res.get("output2") or []
    return any(str(r.get("stck_bsop_date", "")).strip() == ymd for r in rows)


def is_kis_trading_day_sync(snapshot_date: str) -> bool:  # pragma: no cover - live KIS boundary
    """Synchronous KIS trading-day lookup on the data account.

    Wraps ``is_kis_trading_day`` with its own client and session so scheduled
    scripts without an event loop can consult the calendar.

    Args:
        snapshot_date: KST date ``YYYY-MM-DD``.

    Returns:
        True when KIS lists the date as a trading day.

    Raises:
        RuntimeError: The oracle returned a failure response.
        OSError: Transport failure.
    """
    import asyncio

    from src.api.kis.client import KisApiClient, kis_data_client_kwargs

    async def _run() -> bool:
        client = KisApiClient(**kis_data_client_kwargs())  # type: ignore[no-untyped-call]
        async with client.create_session() as session:
            await client.ensure_token(session)
            return await is_kis_trading_day(client, session, snapshot_date)

    return asyncio.run(_run())


def classify_day(snapshot_date: str, trading_day_fn: Callable[[str], bool] | None = None) -> str:
    """Classify a KST date as weekend, holiday, trading day, or unknown.

    The KIS daily index quote is published the same day, so it can answer "is today a holiday"
    before the close; the official KRX index lags by a day and cannot. A lookup failure degrades
    to UNKNOWN instead of raising so a calendar outage never silently suppresses audits.

    Args:
        snapshot_date: KST date `YYYY-MM-DD`.
        trading_day_fn: Trading-day oracle; None uses `is_kis_trading_day_sync`.

    Returns:
        One of DAY_WEEKEND, DAY_HOLIDAY, DAY_TRADING, DAY_UNKNOWN.
    """
    if pd.Timestamp(snapshot_date).weekday() >= 5:
        return DAY_WEEKEND
    oracle = trading_day_fn if trading_day_fn is not None else is_kis_trading_day_sync
    try:
        return DAY_TRADING if oracle(snapshot_date) else DAY_HOLIDAY
    except (RuntimeError, OSError, aiohttp.ClientError) as exc:
        logger.warning("[DATA] stage=daily_audit calendar_lookup=FAIL reason=%s", type(exc).__name__)
        return DAY_UNKNOWN
