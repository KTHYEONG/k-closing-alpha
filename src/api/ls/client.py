"""LS증권 OpenAPI 클라이언트 (t8412 분봉 / t8411 틱 우선 라우팅)."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from src import settings

if TYPE_CHECKING:
    import aiohttp

    from src.data.capture_contracts import BrokerPayload, ChartBudget, PageObserver

logger = logging.getLogger(__name__)

_OAUTH_URL = "https://openapi.ls-sec.co.kr:8080/oauth2/token"
_QUERY_URL = "https://openapi.ls-sec.co.kr:8080/stock/chart"

_SEOUL = ZoneInfo("Asia/Seoul")
_YMD_RE = re.compile(r"^\d{8}$")
_HHMMSS_RE = re.compile(r"^\d{6}$")


def _now_seoul() -> datetime:
    return datetime.now(_SEOUL)


def _validate_target_ymd(target_date: str) -> str:
    ymd = str(target_date).replace("-", "")
    try:
        datetime.strptime(ymd, "%Y%m%d")
    except ValueError:
        raise ValueError(f"invalid target date: {target_date!r}") from None
    return ymd


def _resolve_chart_budget(budget: ChartBudget | None, default_pages: int) -> tuple[int, datetime | None]:
    if budget is None:
        return max(1, int(default_pages)), None
    return max(1, int(budget.max_pages)), budget.deadline


def _deadline_remaining(deadline: datetime | None) -> float | None:
    if deadline is None:
        return None
    return (deadline - _now_seoul()).total_seconds()


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
        self._token_lock: asyncio.Lock | None = None
        # LS_APP_KEY는 krx-alpha와 공유되어 프로세스 내부 페이싱만으로는 키 단위 한도를 보장할 수 없다.
        self._min_interval: float = float(getattr(settings, "LS_MIN_INTERVAL_SECONDS", 1.05) or 1.05)
        self._rate_limit_max_retries: int = int(getattr(settings, "LS_RATE_LIMIT_MAX_RETRIES", 5) or 5)
        self._rate_limit_backoff: float = float(getattr(settings, "LS_RATE_LIMIT_BACKOFF_SECONDS", 1.2) or 1.2)
        self._last_call_time: float = 0.0

    async def ensure_token(self, session) -> str:
        if self.token:
            return self.token
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()
        async with self._token_lock:
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
        max_retries: int | None = None,
    ) -> tuple[dict, dict]:
        if not self.token:
            await self.ensure_token(session)
        if self._lock is None:
            self._lock = asyncio.Lock()
        limit = int(max_retries) if max_retries is not None else self._rate_limit_max_retries

        for attempt in range(limit):
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
            if rsp_cd == "IGW00201" and attempt < limit - 1:
                wait = self._rate_limit_backoff * (2**attempt)
                logger.warning("LS rate limit hit (IGW00201). Retrying in %.1fs... (attempt %d/%d)", wait, attempt + 1, limit)
                await asyncio.sleep(wait)
                self._last_call_time = asyncio.get_running_loop().time()
                continue
            if rsp_cd == "IGW00201":
                logger.warning("[DATA] stage=ls_tr tr_cd=%s status=RATE_LIMITED attempts=%d", tr_cd, limit)
            return data, resp_headers
        return data, resp_headers

    async def get_minute_chart(
        self,
        session: aiohttp.ClientSession,
        code: str,
        target_date: str,
        *,
        budget: ChartBudget | None = None,
        on_page: PageObserver | None = None,
    ) -> BrokerPayload:
        """Read all required chart pages before claiming acquisition completion.

        LS can return late-day and extended-session records despite requested clock
        boundaries. Every raw page must be preserved; the caller classifies venue
        and session only after acquisition.

        Args:
            session: Existing authenticated HTTP session.
            code: Security identifier retained as a string.
            target_date: Requested market date.
            budget: Explicit page/deadline/timeout limits, or configured defaults.
            on_page: Optional synchronous observer of actual raw page evidence.

        Returns:
            Compatible rt_cd/output2/vendor payload plus truncated, termination_reason,
            pages_fetched, and the final permitted continuation state.

        Raises:
            ValueError: Invalid budget or target date.
            OSError: Mandatory raw-page persistence fails.
        """
        ymd = _validate_target_ymd(target_date)
        max_pages, deadline = _resolve_chart_budget(budget, int(getattr(settings, "COLLECTION_CHART_MAX_PAGES", 30) or 30))
        cts_date, cts_time = "", ""
        tr_cont, tr_cont_key = "N", ""
        all_rows: list[dict[str, Any]] = []
        metadata: dict[str, str] = {}
        termination = "exhausted"
        truncated = False
        pages_fetched = 0
        failure_msg = ""
        prev_identity: tuple[Any, ...] | None = None
        stalls = 0
        in_observer = False
        try:
            for page_index in range(max_pages):
                remaining = _deadline_remaining(deadline)
                if remaining is not None and remaining <= 0:
                    termination = "deadline"
                    truncated = True
                    break
                started = _now_seoul()
                call = self._post_tr(
                    session,
                    "t8412",
                    str(code),
                    {"t8412InBlock": {"shcode": str(code), "ncnt": 1, "qrycnt": 500, "nday": "0", "sdate": ymd, "stime": "090000", "edate": ymd, "etime": "153000", "cts_date": cts_date, "cts_time": cts_time, "comp_yn": "N"}},
                    tr_cont=tr_cont,
                    tr_cont_key=tr_cont_key,
                )
                if remaining is None:
                    data, resp_headers = await call
                else:
                    data, resp_headers = await asyncio.wait_for(call, timeout=remaining)
                received = _now_seoul()
                out_block = data.get("t8412OutBlock") or {}
                body_cts_date = str(out_block.get("cts_date", "") or "").strip()
                body_cts_time = str(out_block.get("cts_time", "") or "").strip()
                header_cont = str(resp_headers.get("tr_cont", "N") or "N")
                header_key = str(resp_headers.get("tr_cont_key", "") or "")
                metadata = {
                    "cts_date": body_cts_date,
                    "cts_time": body_cts_time,
                    "tr_cont": header_cont,
                    "tr_cont_key": header_key,
                }
                in_observer = True
                if on_page is not None:
                    on_page(dict(data), metadata, started, received, page_index, 0)
                in_observer = False
                pages_fetched += 1
                if str(data.get("rsp_cd", "")) not in ("00000", "0"):
                    termination = "vendor_failure"
                    truncated = True
                    failure_msg = str(data.get("rsp_msg", ""))
                    break
                rows = data.get("t8412OutBlock1") or []
                if rows:
                    all_rows = [dict(r) for r in rows if isinstance(r, dict)] + all_rows
                identity = (
                    body_cts_date,
                    body_cts_time,
                    header_cont,
                    header_key,
                    tuple((str(r.get("date", "")), str(r.get("time", ""))) for r in rows if isinstance(r, dict)),
                )
                if identity == prev_identity:
                    stalls += 1
                else:
                    stalls = 0
                    prev_identity = identity
                if stalls >= 2:
                    termination = "nonprogress"
                    truncated = True
                    break
                oldest_date = ""
                for r in rows:
                    day = str(r.get("date", "") or "").strip()
                    if _YMD_RE.fullmatch(day) and (not oldest_date or day < oldest_date):
                        oldest_date = day
                if oldest_date and oldest_date < ymd:
                    termination = "crossed_target_date"
                    break
                if not body_cts_date and not body_cts_time and header_cont != "Y":
                    termination = "exhausted"
                    break
                if not body_cts_date and not body_cts_time:
                    termination = "cursor_unknown"
                    truncated = True
                    break
                cts_date, cts_time = body_cts_date, body_cts_time
                tr_cont, tr_cont_key = header_cont, header_key
            else:
                termination = "page_budget"
                truncated = True
        except TimeoutError:
            termination = "deadline"
            truncated = True
        except Exception as e:
            if in_observer:
                raise
            logger.warning("LS t8412 failed code=%s: %s", code, e)
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "ls", "truncated": True, "termination_reason": "vendor_failure", "pages_fetched": pages_fetched, "continuation": metadata}
        if termination == "vendor_failure":
            return {"rt_cd": "1", "msg1": failure_msg, "output2": [], "vendor": "ls", "truncated": True, "termination_reason": termination, "pages_fetched": pages_fetched, "continuation": metadata}
        filtered = [dict(r) for r in all_rows if str(r.get("date", ymd)) == ymd]
        return {"rt_cd": "0", "output2": filtered, "vendor": "ls", "truncated": truncated, "termination_reason": termination, "pages_fetched": pages_fetched, "continuation": metadata}

    async def get_tick_chart(
        self,
        session: aiohttp.ClientSession,
        code: str,
        target_date: str,
        max_pages: int | None = None,
        *,
        budget: ChartBudget | None = None,
        on_page: PageObserver | None = None,
    ) -> BrokerPayload:
        """Keep tick multiplicity and termination evidence across bounded pagination.

        Args:
            session: Existing HTTP session.
            code: Security identifier.
            target_date: Requested market date.
            max_pages: Legacy explicit limit; conflicts with budget are rejected.
            budget: Typed page/deadline/timeout bound.
            on_page: Raw page observer called before filtering or normalization.

        Returns:
            Legacy payload and explicit task termination metadata.

        Raises:
            ValueError: Invalid or conflicting acquisition limits.
            OSError: Mandatory evidence persistence fails.
        """
        ymd = _validate_target_ymd(target_date)
        if max_pages is not None and budget is not None and int(max_pages) != int(budget.max_pages):
            raise ValueError("conflicting tick acquisition limits")
        if max_pages is not None and int(max_pages) <= 0:
            raise ValueError("invalid tick acquisition limits")
        if budget is not None:
            page_budget, deadline = _resolve_chart_budget(budget, _resolve_tick_max_pages(None))
        elif max_pages is not None:
            page_budget, deadline = max(1, int(max_pages)), None
        else:
            page_budget, deadline = _resolve_chart_budget(None, _resolve_tick_max_pages(None))
        cts_date, cts_time = "", ""
        tr_cont, tr_cont_key = "N", ""
        all_rows: list[dict[str, Any]] = []
        metadata: dict[str, str] = {}
        termination = "exhausted"
        truncated = False
        pages_fetched = 0
        failure_msg = ""
        prev_identity: tuple[Any, ...] | None = None
        stalls = 0
        in_observer = False
        try:
            for page_index in range(max(1, int(page_budget))):
                remaining = _deadline_remaining(deadline)
                if remaining is not None and remaining <= 0:
                    termination = "deadline"
                    truncated = True
                    break
                started = _now_seoul()
                call = self._post_tr(
                    session, "t8411", str(code),
                    {"t8411InBlock": {"shcode": str(code), "ncnt": 1, "qrycnt": 500, "nday": "0", "sdate": ymd, "stime": "090000", "edate": ymd, "etime": "153000", "cts_date": cts_date, "cts_time": cts_time, "comp_yn": "N"}},
                    tr_cont=tr_cont,
                    tr_cont_key=tr_cont_key,
                )
                if remaining is None:
                    data, resp_headers = await call
                else:
                    data, resp_headers = await asyncio.wait_for(call, timeout=remaining)
                received = _now_seoul()
                out_block = data.get("t8411OutBlock") or {}
                body_cts_date = str(out_block.get("cts_date", "") or "").strip()
                body_cts_time = str(out_block.get("cts_time", "") or "").strip()
                header_cont = str(resp_headers.get("tr_cont", "N") or "N")
                header_key = str(resp_headers.get("tr_cont_key", "") or "")
                metadata = {
                    "cts_date": body_cts_date,
                    "cts_time": body_cts_time,
                    "tr_cont": header_cont,
                    "tr_cont_key": header_key,
                }
                in_observer = True
                if on_page is not None:
                    on_page(dict(data), metadata, started, received, page_index, 0)
                in_observer = False
                pages_fetched += 1
                if str(data.get("rsp_cd", "")) not in ("00000", "0"):
                    termination = "vendor_failure"
                    truncated = True
                    failure_msg = str(data.get("rsp_msg", ""))
                    break
                rows = data.get("t8411OutBlock1") or []
                if rows:
                    all_rows = [dict(r) for r in rows if isinstance(r, dict)] + all_rows
                identity = (
                    body_cts_date,
                    body_cts_time,
                    header_cont,
                    header_key,
                    tuple((str(r.get("date", "")), str(r.get("time", ""))) for r in rows if isinstance(r, dict)),
                )
                if identity == prev_identity:
                    stalls += 1
                else:
                    stalls = 0
                    prev_identity = identity
                if stalls >= 2:
                    termination = "nonprogress"
                    truncated = True
                    break
                head = rows[0] if rows else {}
                head_date = str(head.get("date", "") or "").strip()
                head_time = str(head.get("time", "") or "").strip()
                if _YMD_RE.fullmatch(head_date) and head_date < ymd:
                    termination = "crossed_target_date"
                    break
                if _YMD_RE.fullmatch(head_date) and head_date == ymd and _HHMMSS_RE.fullmatch(head_time) and head_time <= "090000":
                    termination = "crossed_target_date"
                    break
                if header_cont != "Y" and not body_cts_date and not body_cts_time:
                    termination = "exhausted"
                    break
                if not body_cts_date and not body_cts_time:
                    termination = "cursor_unknown"
                    truncated = True
                    break
                cts_date, cts_time = body_cts_date, body_cts_time
                tr_cont, tr_cont_key = header_cont, header_key
            else:
                termination = "page_budget"
                truncated = True
        except TimeoutError:
            termination = "deadline"
            truncated = True
        except Exception as e:
            if in_observer:
                raise
            logger.warning("LS t8411 failed code=%s: %s", code, e)
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "ls", "truncated": True, "termination_reason": "vendor_failure", "pages_fetched": pages_fetched, "continuation": metadata}
        if termination == "vendor_failure":
            return {"rt_cd": "1", "msg1": failure_msg, "output2": [], "vendor": "ls", "truncated": True, "termination_reason": termination, "pages_fetched": pages_fetched, "continuation": metadata}
        filtered = [
            dict(r) for r in all_rows
            if str(r.get("date", ymd)) == ymd and _HHMMSS_RE.fullmatch(str(r.get("time", ""))) and "090000" <= str(r.get("time", "")) <= "153059"
        ]
        return {"rt_cd": "0", "output2": filtered, "vendor": "ls", "truncated": truncated, "termination_reason": termination, "pages_fetched": pages_fetched, "continuation": metadata}

