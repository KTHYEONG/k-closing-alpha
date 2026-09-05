"""LS증권 OpenAPI 클라이언트 (t8412 분봉 / t8411 틱 우선 라우팅)."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os

from src import settings

logger = logging.getLogger(__name__)

_OAUTH_URL = "https://openapi.ls-sec.co.kr:8080/oauth2/token"
_QUERY_URL = "https://openapi.ls-sec.co.kr:8080/stock/chart"


def _resolve_tick_max_pages(explicit: int | None) -> int:
    if explicit is not None:
        return int(explicit)
    return int(getattr(settings, "LS_TICK_MAX_PAGES", 100) or 100)


class LsApiClient:
    def __init__(self, app_key: str | None = None, app_secret: str | None = None) -> None:
        self.app_key = app_key or getattr(settings, "LS_APP_KEY", "") or os.getenv("LS_APP_KEY", "")
        self.app_secret = app_secret or getattr(settings, "LS_APP_SECRET", "") or os.getenv("LS_APP_SECRET", "")
        self.token: str | None = None
        self._lock: asyncio.Lock | None = None
        self._min_interval: float = 1.05
        self._last_call_time: float = 0.0

    async def ensure_token(self, session) -> str:
        if self.token:
            return self.token
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecretkey": self.app_secret,
            "scope": "oob",
        }
        raw = session.post(_OAUTH_URL, data=payload)
        if inspect.isawaitable(raw):
            raw = await raw
        async with raw as resp:
            body = await resp.json()
        token = str(body.get("access_token", ""))
        if not token:
            raise RuntimeError(f"LS token issuance failed: {body}")
        self.token = token
        return token

    async def _post_tr(
        self,
        session,
        tr_cd: str,
        tr_key: str,
        body: dict,
        tr_cont: str = "N",
        tr_cont_key: str = "",
        max_retries: int = 3,
    ) -> tuple[dict, dict]:
        if not self.token:
            await self.ensure_token(session)
        if self._lock is None:
            self._lock = asyncio.Lock()

        for attempt in range(max_retries):
            async with self._lock:
                loop = asyncio.get_running_loop()
                now = loop.time()
                elapsed = now - self._last_call_time
                if elapsed < self._min_interval:
                    await asyncio.sleep(self._min_interval - elapsed)
                self._last_call_time = loop.time()

            headers = {
                "content-type": "application/json; charset=utf-8",
                "authorization": f"Bearer {self.token}",
                "tr_cd": tr_cd,
                "tr_cont": tr_cont,
                "tr_cont_key": tr_cont_key,
            }
            raw = session.post(_QUERY_URL, json={**body, "tr_cd": tr_cd}, headers=headers)
            if inspect.isawaitable(raw):
                raw = await raw
            async with raw as resp:
                data = await resp.json()
                headers_raw = getattr(resp, "headers", None)
                if isinstance(headers_raw, dict):
                    resp_headers = headers_raw
                elif hasattr(headers_raw, "items") and not type(headers_raw).__name__.endswith("Mock"):
                    try:
                        resp_headers = dict(headers_raw)
                    except Exception:
                        resp_headers = {}
                else:
                    resp_headers = {}

            rsp_cd = str(data.get("rsp_cd", ""))
            if rsp_cd == "IGW00201" and attempt < max_retries - 1:
                logger.warning("LS rate limit hit (IGW00201). Retrying in 1.2s... (attempt %d/%d)", attempt + 1, max_retries)
                await asyncio.sleep(1.2)
                continue
            return data, resp_headers
        return data, resp_headers

    async def get_minute_chart(self, session, code: str, target_date: str) -> dict:
        ymd = str(target_date).replace("-", "")
        try:
            data, _ = await self._post_tr(
                session, "t8412", str(code),
                {"t8412InBlock": {"shcode": str(code), "ncnt": 1, "qrycnt": 500, "nday": "0", "sdate": ymd, "stime": "090000", "edate": ymd, "etime": "153000", "cts_date": "", "cts_time": "", "comp_yn": "N"}},
            )
        except Exception as e:
            logger.warning("LS t8412 failed code=%s: %s", code, e)
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "ls"}
        if str(data.get("rsp_cd", "")) not in ("00000", "0"):
            return {"rt_cd": "1", "msg1": str(data.get("rsp_msg", "")), "output2": [], "vendor": "ls"}
        rows = data.get("t8412OutBlock1") or []
        return {"rt_cd": "0", "output2": [dict(r) for r in rows], "vendor": "ls"}

    async def get_tick_chart(self, session, code: str, target_date: str, max_pages: int | None = None) -> dict:
        ymd = str(target_date).replace("-", "")
        page_budget = _resolve_tick_max_pages(max_pages)
        cts_date, cts_time = "", ""
        tr_cont, tr_cont_key = "N", ""
        all_rows: list[dict] = []
        reached_open = False
        try:
            for _ in range(max(1, int(page_budget))):
                data, resp_headers = await self._post_tr(
                    session, "t8411", str(code),
                    {"t8411InBlock": {"shcode": str(code), "ncnt": 1, "qrycnt": 500, "nday": "0", "sdate": ymd, "stime": "090000", "edate": ymd, "etime": "153000", "cts_date": cts_date, "cts_time": cts_time, "comp_yn": "N"}},
                    tr_cont=tr_cont,
                    tr_cont_key=tr_cont_key,
                )
                if str(data.get("rsp_cd", "")) not in ("00000", "0"):
                    return {"rt_cd": "1", "msg1": str(data.get("rsp_msg", "")), "output2": [], "vendor": "ls", "truncated": False}
                rows = data.get("t8411OutBlock1") or []
                if rows:
                    all_rows = rows + all_rows
                cts = data.get("t8411OutBlock") or {}
                cts_date = str(cts.get("cts_date", "") or "").strip()
                cts_time = str(cts.get("cts_time", "") or "").strip()
                tr_cont = resp_headers.get("tr_cont", "N")
                tr_cont_key = resp_headers.get("tr_cont_key", "")
                if not cts_date and not cts_time and tr_cont != "Y":
                    reached_open = True
                    break
                date_val = str(rows[0].get("date", "") or "").strip() if rows else ""
                if date_val and date_val < ymd:
                    reached_open = True
                    break
                if rows and str(rows[0].get("time", "")) <= "090000":
                    reached_open = True
                    break
        except Exception as e:
            logger.warning("LS t8411 failed code=%s: %s", code, e)
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "ls", "truncated": False}
        filtered = [
            dict(r) for r in all_rows
            if str(r.get("date", ymd)) == ymd and "090000" <= str(r.get("time", "")) <= "153059"
        ]
        truncated = not reached_open
        if truncated:
            earliest = min((str(r.get("time", "")) for r in filtered), default="")
            logger.warning(
                "[DATA] LS tick page budget exhausted code=%s pages=%d earliest=%s",
                code,
                page_budget,
                earliest,
            )
        return {"rt_cd": "0", "output2": filtered, "vendor": "ls", "truncated": truncated}

