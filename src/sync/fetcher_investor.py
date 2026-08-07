from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import pandas as pd

from src.api.kis_client import KisApiClient

logger = logging.getLogger(__name__)


def _clean_num(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s in {"", "-", "--", "None", "nan"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _collect_rows(body: dict) -> list[dict]:
    rows: list[dict] = []
    for key in ["output2", "output1", "output"]:
        val = body.get(key)
        if isinstance(val, list):
            rows.extend([x for x in val if isinstance(x, dict)])
        elif isinstance(val, dict):
            rows.append(val)
    return rows


def _prev_day_ymd(ymd: str, days: int = 1) -> str:
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    if pd.isna(dt):
        return ymd
    return (dt - pd.Timedelta(days=max(1, int(days)))).strftime("%Y%m%d")


async def _request_investor_daily_async(
    session: aiohttp.ClientSession,
    code: str,
    trade_date: str,
    client: KisApiClient,
    request_slot: Callable[[], Awaitable[None]] | None = None,
) -> dict:
    """비동기 KIS API 호출 헬퍼"""
    url = f"{client.base_url}/uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily"
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": str(code).strip().zfill(6),
        "FID_INPUT_DATE_1": str(trade_date).strip(),
        "FID_ORG_ADJ_PRC": "",
        "FID_ETC_CLS_CODE": "",
    }
    # KisApiClient의 공통 요청 핸들러 사용
    if request_slot is not None:
        await request_slot()
    return await client._handle_request(
        session.get, url, headers=client._get_headers("FHPTJ04160001"), params=params
    )


async def get_investor_trade_daily_async(
    session: aiohttp.ClientSession,
    client: KisApiClient,
    code: str,
    start_date: str,
    end_date: str,
    *,
    target_dates: list[str] | None = None,
    max_calls: int = 120,
    request_slot: Callable[[], Awaitable[None]] | None = None,
    max_consecutive_failures: int = 3,
) -> pd.DataFrame:
    """비동기 버전: KIS API를 통해 종목별 투자자 일별 거래 정보를 가져옵니다."""
    code = str(code).strip().zfill(6)
    s_dt = pd.to_datetime(str(start_date).strip(), format="%Y%m%d", errors="coerce")
    e_dt = pd.to_datetime(str(end_date).strip(), format="%Y%m%d", errors="coerce")
    if pd.isna(s_dt) or pd.isna(e_dt):
        return pd.DataFrame()
    if s_dt > e_dt:
        s_dt, e_dt = e_dt, s_dt
    start = s_dt.strftime("%Y%m%d")
    end = e_dt.strftime("%Y%m%d")

    wanted = {d for d in (target_dates or []) if start <= d <= end}

    all_rows: list[dict] = []
    seen_days = set()
    cursor = end
    no_progress = 0
    failures = 0

    for _ in range(max(1, int(max_calls))):
        if cursor < start:
            break
        try:
            body = await _request_investor_daily_async(session, code, cursor, client, request_slot)
        except Exception as exc:
            failures += 1
            logger.warning(
                "[DATA] stage=investor_flow symbol=%s cursor=%s status=REQUEST_FAIL failures=%d error=%s",
                code, cursor, failures, type(exc).__name__,
            )
            if failures >= max(1, int(max_consecutive_failures)):
                break
            cursor = _prev_day_ymd(cursor, 1)
            continue

        if body.get("rt_cd") != "0":
            failures += 1
            logger.warning(
                "[DATA] stage=investor_flow symbol=%s cursor=%s status=API_FAIL failures=%d msg=%s",
                code, cursor, failures, body.get("msg1", ""),
            )
            if failures >= max(1, int(max_consecutive_failures)):
                break
            cursor = _prev_day_ymd(cursor, 1)
            continue

        failures = 0

        rows = _collect_rows(body)
        row_days = {str(item.get("stck_bsop_date") or "").strip() for item in rows if isinstance(item, dict)}
        row_days = {d for d in row_days if len(d) == 8 and d.isdigit()}

        if rows:
            all_rows.extend(rows)
            seen_days.update({d for d in row_days if start <= d <= end})

        if wanted and wanted.issubset(seen_days):
            break

        if row_days:
            min_day = min(row_days)
            next_cursor = _prev_day_ymd(min_day, 1)
        else:
            # 데이터가 없는 경우, 탐색 범위를 크게 점프
            next_cursor = _prev_day_ymd(cursor, 30)

        if next_cursor >= cursor:
            no_progress += 1
            next_cursor = _prev_day_ymd(cursor, 30)  # 무한 루프 방지
        else:
            no_progress = 0
        cursor = next_cursor

        if no_progress >= 3:
            break

        # 비동기 환경에서는 짧은 sleep이 이벤트 루프에 제어권을 넘겨줌
        await asyncio.sleep(0.01)

    if not all_rows:
        return pd.DataFrame()

    out_rows = []
    for item in all_rows:
        d = str(item.get("stck_bsop_date") or "").strip()
        if not (start <= d <= end):
            continue

        foreign = _clean_num(item.get("frgn_ntby_tr_pbmn"))
        inst = _clean_num(item.get("orgn_ntby_tr_pbmn"))

        out_rows.append(
            {
                "date": pd.to_datetime(d, format="%Y%m%d", errors="coerce"),
                "foreign_netbuy": foreign,
                "inst_netbuy": inst,
            }
        )

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows).dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    out["foreign_netbuy"] = pd.to_numeric(out["foreign_netbuy"], errors="coerce")
    out["inst_netbuy"] = pd.to_numeric(out["inst_netbuy"], errors="coerce")
    return out.reset_index(drop=True)


def get_investor_trade_daily(
    code: str,
    start_date: str,
    end_date: str,
    *,
    target_dates: list[str] | None = None,
    max_calls: int = 120,
    sleep_sec: float = 0.02,
) -> pd.DataFrame:
    """종목별 외국인/기관 순매수 일별 조회 (동기 래퍼).

    kis_common 레거시에 의존하지 않고 KisApiClient 기반 비동기 구현을
    단일 이벤트 루프로 실행합니다.

    API:
      GET /uapi/domestic-stock/v1/quotations/investor-trade-by-stock-daily
      TR_ID: FHPTJ04160001
    """
    async def _run() -> pd.DataFrame:
        client = KisApiClient()
        async with client.create_session() as session:
            await client.ensure_token(session)
            return await get_investor_trade_daily_async(
                session=session,
                client=client,
                code=code,
                start_date=start_date,
                end_date=end_date,
                target_dates=target_dates,
                max_calls=max_calls,
            )

    return asyncio.run(_run())
