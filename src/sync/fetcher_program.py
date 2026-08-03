from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import pandas as pd

from src.api.kis_client import KisApiClient


def _clean_num(v: Any) -> float:
    if v is None:
        return 0.0
    s = str(v).strip().replace(",", "")
    if s in {"", "-", "--", "None", "nan"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


async def get_program_history_async(
    session: aiohttp.ClientSession,
    client: KisApiClient,
    code: str,
    start_date: str,
    end_date: str,
    *,
    target_dates: list[str] | None = None,
    max_calls: int = 120,
) -> dict[str, float]:
    """비동기 버전: 종목별 프로그램 매매 추이(일별)를 조회합니다."""
    url = f"{client.base_url}/uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily"

    prog_map: dict[str, float] = {}
    s = pd.to_datetime(str(start_date).strip(), format="%Y%m%d", errors="coerce")
    e = pd.to_datetime(str(end_date).strip(), format="%Y%m%d", errors="coerce")
    if pd.isna(s) or pd.isna(e):
        return prog_map
    if s > e:
        s, e = e, s
    start = s.strftime("%Y%m%d")
    end = e.strftime("%Y%m%d")

    wanted = {d for d in (target_dates or []) if start <= d <= end}

    code = str(code).strip().zfill(6)
    cursor = end
    seen_days = set()
    no_progress = 0

    for _ in range(max(1, int(max_calls))):
        if cursor < start:
            break

        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": code,
            "FID_INPUT_DATE_1": cursor,
        }

        try:
            data = await client._handle_request(
                session.get, url, headers=client._get_headers("FHPPG04650201"), params=params
            )
        except Exception:
            dt = pd.to_datetime(cursor, format="%Y%m%d", errors="coerce")
            if pd.isna(dt):
                break
            cursor = (dt - pd.Timedelta(days=1)).strftime("%Y%m%d")
            continue

        if data.get("rt_cd") != "0":
            dt = pd.to_datetime(cursor, format="%Y%m%d", errors="coerce")
            if pd.isna(dt):
                break
            cursor = (dt - pd.Timedelta(days=1)).strftime("%Y%m%d")
            continue

        rows = data.get("output", [])
        row_days = set()
        for item in rows:
            date = (item.get("stck_bsop_date") or "").strip()[:8]
            if not date or len(date) != 8 or not date.isdigit():
                continue
            row_days.add(date)

            if not (start <= date <= end):
                continue
            net_amt = _clean_num(item.get("whol_smtn_ntby_tr_pbmn", "0"))
            prog_map[date] = float(net_amt)
            seen_days.add(date)

        if wanted and wanted.issubset(seen_days):
            break

        if row_days:
            min_day = min(row_days)
            next_dt = pd.to_datetime(min_day, format="%Y%m%d", errors="coerce")
            if pd.isna(next_dt):
                break
            next_cursor = (next_dt - pd.Timedelta(days=1)).strftime("%Y%m%d")
        else:
            dt = pd.to_datetime(cursor, format="%Y%m%d", errors="coerce")
            if pd.isna(dt):
                break
            next_cursor = (dt - pd.Timedelta(days=30)).strftime("%Y%m%d")

        if next_cursor >= cursor:
            no_progress += 1
            dt = pd.to_datetime(cursor, format="%Y%m%d", errors="coerce")
            if pd.isna(dt):
                break
            next_cursor = (dt - pd.Timedelta(days=30)).strftime("%Y%m%d")
        else:
            no_progress = 0

        cursor = next_cursor
        if no_progress >= 3:
            break

        await asyncio.sleep(0.01)

    return prog_map


def get_program_history(
    code: str,
    start_date: str,
    end_date: str,
    *,
    target_dates: list[str] | None = None,
    max_calls: int = 120,
    sleep_sec: float = 0.02,
) -> dict[str, float]:
    """종목별 프로그램 매매 추이(일별) 조회 (동기 래퍼).

    kis_common 레거시에 의존하지 않고 KisApiClient 기반 비동기 구현을
    단일 이벤트 루프로 실행합니다.

    API:
      GET /uapi/domestic-stock/v1/quotations/program-trade-by-stock-daily
      TR_ID: FHPPG04650201
    """
    async def _run() -> dict[str, float]:
        client = KisApiClient()
        async with client.create_session() as session:
            await client.ensure_token(session)
            return await get_program_history_async(
                session=session,
                client=client,
                code=code,
                start_date=start_date,
                end_date=end_date,
                target_dates=target_dates,
                max_calls=max_calls,
            )

    return asyncio.run(_run())
