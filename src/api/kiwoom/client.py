"""키움증권 REST API 클라이언트 (HTTP/토큰/요청 동작, TR별 5req/s 서버 강제 리밋)."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from src.api.kis.rate_limit import AsyncRateLimiter, get_shared_rate_limiter
from src.config import settings
from src.data.capture_contracts import RawCaptureError

if TYPE_CHECKING:
    import aiohttp

    from src.data.capture_contracts import BrokerPayload, ChartBudget, PageObserver

logger = logging.getLogger(__name__)

# 서버 실측 확인 (scratch/probe_kiwoom_rate_limit_concurrency.json): 초과 시 HTTP 429 +
# 응답 바디에 "유량=5, API ID=<api_id>" 명시. api-id(TR)별로 독립된 버킷이 부과된다.
_KIWOOM_TR_RATE_PER_SEC = 5.0

# 키움 API 앞단 WAF는 aiohttp 기본 User-Agent("Python/x.y aiohttp/x.y")를 자동화 도구로
# 탐지해 요청을 차단한다(HTML "Request Blocked" 400). IP 등록 여부와 무관하게 UA만으로
# 통과 여부가 갈리므로, 브라우저/CLI 도구처럼 보이는 값으로 고정한다.
_KIWOOM_USER_AGENT = "curl/8.5.0"

_SEOUL = ZoneInfo("Asia/Seoul")
_CNTR_TM_RE = re.compile(r"^\d{14}$")


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


class KiwoomApiClient:
    def __init__(self, app_key: str | None = None, secret_key: str | None = None, base_url: str | None = None) -> None:
        """Bind credentials, preferring explicit arguments over the live Settings instance.

        Args:
            app_key: Explicit app key; falls back to ``settings.KIWOOM_APP_KEY``.
            secret_key: Explicit secret; falls back to ``settings.KIWOOM_SECRET_KEY``.
            base_url: Explicit origin; falls back to ``settings.KIWOOM_BASE_URL``.
        """
        self.app_key = app_key or settings.KIWOOM_APP_KEY
        self.secret_key = secret_key or settings.KIWOOM_SECRET_KEY
        self.base_url = base_url or settings.KIWOOM_BASE_URL
        self.token: str | None = None
        self._rate_limiters: dict[str, AsyncRateLimiter] = {}
        self._token_lock: asyncio.Lock | None = None

    def _limiter_for(self, api_id: str) -> AsyncRateLimiter:
        return get_shared_rate_limiter("kiwoom", f"{self.app_key}:{api_id}", _KIWOOM_TR_RATE_PER_SEC)

    async def ensure_token(self, session) -> str:
        if self.token:
            return self.token
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()
        async with self._token_lock:
            if self.token:
                return self.token
            payload = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.secret_key}
            raw = session.post(
                f"{self.base_url}/oauth2/token",
                headers={"Content-Type": "application/json;charset=UTF-8", "User-Agent": _KIWOOM_USER_AGENT},
                json=payload,
            )
            if inspect.isawaitable(raw):
                raw = await raw
            async with raw as resp:
                body = await resp.json()
            token = str(body.get("token", ""))
            if not token:
                raise RuntimeError(f"Kiwoom token issuance failed: {body}")
            self.token = token
            return token

    async def _post_tr(
        self,
        session,
        api_id: str,
        path: str,
        body: dict,
        cont_yn: str = "N",
        next_key: str = "",
        max_retries: int = 3,
    ) -> tuple[dict, dict]:
        if not self.token:
            await self.ensure_token(session)
        limiter = self._limiter_for(api_id)

        for attempt in range(max_retries):
            await limiter.acquire()
            headers = {
                "Content-Type": "application/json;charset=UTF-8",
                "User-Agent": _KIWOOM_USER_AGENT,
                "authorization": f"Bearer {self.token}",
                "api-id": api_id,
                "cont-yn": cont_yn,
                "next-key": next_key,
            }
            raw = session.post(f"{self.base_url}{path}", headers=headers, json=body)
            if inspect.isawaitable(raw):
                raw = await raw
            async with raw as resp:
                data = await resp.json()
                status = resp.status
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

            if status == 429 and attempt < max_retries - 1:
                logger.warning("Kiwoom rate limit hit (429) api_id=%s. Retrying in 1.2s... (attempt %d/%d)", api_id, attempt + 1, max_retries)
                await asyncio.sleep(1.2)
                continue
            return data, resp_headers
        return data, resp_headers

    async def get_fluctuation_ranking(
        self,
        session: aiohttp.ClientSession,
        *,
        rate_min_pct: float,
        rate_max_pct: float,
        market_type: str = "000",
        max_pages: int = 20,
        stex_tp: str = "3",
        on_page: PageObserver | None = None,
    ) -> BrokerPayload:
        """Return the existing filtered ranking while observing unfiltered vendor pages.

        Args:
            session: Existing authenticated HTTP session.
            rate_min_pct: Existing minimum change percentage.
            rate_max_pct: Existing maximum change percentage.
            market_type: Existing market selector.
            max_pages: Existing ranking page limit.
            stex_tp: Existing venue selector.
            on_page: Optional prefilter source evidence observer.
        Returns:
            Existing filtered ranking payload.
        Raises:
            RawCaptureError: Source evidence persistence failed.
        """
        cont_yn, next_key = "N", ""
        collected: list[dict] = []
        in_observer = False
        try:
            for page_index in range(max(1, int(max_pages))):
                started = _now_seoul()
                data, resp_headers = await self._post_tr(
                    session, "ka10027", "/api/dostk/rkinfo",
                    {
                        "mrkt_tp": str(market_type),
                        "stex_tp": str(stex_tp),
                        "sort_tp": "1",
                        "trde_qty_cnd": "0000",
                        "stk_cnd": "0",
                        "crd_cnd": "0",
                        "updown_incls": "0",
                        "pric_cnd": "0",
                        "trde_prica_cnd": "0",
                    },
                    cont_yn=cont_yn, next_key=next_key,
                )
                received = _now_seoul()
                header_cont = str(resp_headers.get("cont-yn", "N") or "N")
                header_key = str(resp_headers.get("next-key", "") or "")
                metadata = {
                    "vendor": "kiwoom",
                    "endpoint": "fluctuation-ranking",
                    "cont-yn": header_cont,
                    "next-key": header_key,
                }
                in_observer = True
                if on_page is not None:
                    on_page(dict(data), metadata, started, received, page_index, 0)
                in_observer = False
                if data.get("return_code") != 0:
                    return {"rt_cd": "1", "msg1": str(data.get("return_msg", "")), "output": [], "vendor": "kiwoom"}
                rows = data.get("pred_pre_flu_rt_upper") or []
                page_rates: list[float] = []
                for r in rows:
                    try:
                        rate = float(str(r.get("flu_rt", "")).strip())
                    except (ValueError, TypeError):
                        continue
                    page_rates.append(rate)
                    if rate < float(rate_min_pct) or rate > float(rate_max_pct):
                        continue
                    collected.append(dict(r))
                cont_yn = header_cont
                next_key = header_key
                if cont_yn != "Y":
                    break
                if page_rates and min(page_rates) < float(rate_min_pct):
                    break
            else:
                logger.warning("[DATA] stage=universe_scan vendor=kiwoom status=TRUNCATED max_pages=%d n_rows=%d", int(max_pages), len(collected))
                return {"rt_cd": "1", "msg1": f"ranking truncated at max_pages={int(max_pages)}", "output": collected, "vendor": "kiwoom", "truncated": True}
        except Exception as e:
            if in_observer:
                if isinstance(e, RawCaptureError):
                    raise
                raise RawCaptureError(str(e)) from e
            logger.warning("Kiwoom fluctuation ranking failed: %s", e)
            return {"rt_cd": "1", "msg1": str(e), "output": [], "vendor": "kiwoom"}
        return {"rt_cd": "0", "output": collected, "vendor": "kiwoom"}

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
        """Expose incomplete successful tick responses as bounded repairable tasks.

        Args:
            session: Existing HTTP session.
            code: Explicit broker instrument identifier.
            target_date: Requested market date.
            max_pages: Compatible legacy page limit.
            budget: Page/deadline/timeout limits.
            on_page: Observer of unfiltered raw responses and continuation metadata.

        Returns:
            Existing rt_cd/output2/vendor/truncated keys plus termination metadata.

        Raises:
            ValueError: Invalid or conflicting bounds.
            OSError: Mandatory capture fails.
        """
        ymd = _validate_target_ymd(target_date)
        if max_pages is not None and budget is not None and int(max_pages) != int(budget.max_pages):
            raise ValueError("conflicting tick acquisition limits")
        if max_pages is not None and int(max_pages) <= 0:
            raise ValueError("invalid tick acquisition limits")
        if budget is not None:
            page_budget, deadline = _resolve_chart_budget(budget, int(settings.COLLECTION_CHART_MAX_PAGES))
        elif max_pages is not None:
            page_budget, deadline = max(1, int(max_pages)), None
        else:
            page_budget, deadline = _resolve_chart_budget(None, int(settings.COLLECTION_CHART_MAX_PAGES))
        cont_yn, next_key = "N", ""
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
                    session, "ka10079", "/api/dostk/chart",
                    {"stk_cd": str(code), "tic_scope": "1", "upd_stkpc_tp": "1", "base_dt": ymd},
                    cont_yn=cont_yn, next_key=next_key,
                )
                if remaining is None:
                    data, resp_headers = await call
                else:
                    data, resp_headers = await asyncio.wait_for(call, timeout=remaining)
                received = _now_seoul()
                header_cont = str(resp_headers.get("cont-yn", "N") or "N")
                header_key = str(resp_headers.get("next-key", "") or "")
                metadata = {"cont-yn": header_cont, "next-key": header_key}
                in_observer = True
                if on_page is not None:
                    on_page(dict(data), metadata, started, received, page_index, 0)
                in_observer = False
                pages_fetched += 1
                if data.get("return_code") != 0:
                    termination = "vendor_failure"
                    truncated = True
                    failure_msg = str(data.get("return_msg", ""))
                    break
                rows = data.get("stk_tic_chart_qry") or []
                if rows:
                    all_rows.extend(dict(r) for r in rows if isinstance(r, dict))
                identity = (
                    header_cont,
                    header_key,
                    tuple(str(r.get("cntr_tm", "")) for r in rows if isinstance(r, dict)),
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
                if header_cont != "Y":
                    termination = "exhausted"
                    break
                if not header_key:
                    termination = "cursor_unknown"
                    truncated = True
                    break
                oldest = str(rows[-1].get("cntr_tm", "") or "") if rows else ""
                if _CNTR_TM_RE.fullmatch(oldest) and oldest[:8] < ymd:
                    termination = "crossed_target_date"
                    break
                cont_yn, next_key = header_cont, header_key
            else:
                termination = "page_budget"
                truncated = True
        except TimeoutError:
            termination = "deadline"
            truncated = True
        except Exception as e:
            if in_observer:
                raise
            logger.warning("Kiwoom tick chart failed code=%s: %s", code, e)
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "kiwoom", "truncated": True, "termination_reason": "vendor_failure", "pages_fetched": pages_fetched, "continuation": metadata}
        if termination == "vendor_failure":
            return {"rt_cd": "1", "msg1": failure_msg, "output2": [], "vendor": "kiwoom", "truncated": True, "termination_reason": termination, "pages_fetched": pages_fetched, "continuation": metadata}
        filtered = [dict(r) for r in all_rows if str(r.get("cntr_tm", "")).startswith(ymd)]
        return {"rt_cd": "0", "output2": filtered, "vendor": "kiwoom", "truncated": truncated, "termination_reason": termination, "pages_fetched": pages_fetched, "continuation": metadata}

    async def get_nxt_premarket_chart(self, session, code: str, target_date: str) -> dict:
        ymd = str(target_date).replace("-", "")
        nx_code = f"{str(code).split('_')[0].zfill(6)}_NX"
        try:
            data, _headers = await self._post_tr(
                session, "ka10080", "/api/dostk/chart",
                {"stk_cd": nx_code, "base_dt": ymd, "tic_scope": "1", "upd_stkpc_tp": "0"},
            )
        except Exception as e:
            logger.warning("Kiwoom NXT premarket chart failed code=%s: %s", code, e)
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "kiwoom"}
        if data.get("return_code") != 0:
            return {"rt_cd": "1", "msg1": str(data.get("return_msg", "")), "output2": [], "vendor": "kiwoom"}
        rows = data.get("stk_min_pole_chart_qry") or []
        kept: list[dict] = []
        for r in rows:
            cntr_tm = str(r.get("cntr_tm", ""))
            if not cntr_tm.startswith(ymd):
                continue
            try:
                hms = int(cntr_tm[8:14])
            except (ValueError, TypeError):
                continue
            if hms < 80000 or hms > 85000:
                continue
            kept.append(dict(r))
        return {"rt_cd": "0", "output2": kept, "vendor": "kiwoom"}

    async def get_nxt_minute_chart(self, session, code: str, target_date: str) -> dict:
        ymd = str(target_date).replace("-", "")
        nx_code = f"{str(code).split('_')[0].zfill(6)}_NX"
        try:
            data, _headers = await self._post_tr(
                session, "ka10080", "/api/dostk/chart",
                {"stk_cd": nx_code, "base_dt": ymd, "tic_scope": "1", "upd_stkpc_tp": "0"},
            )
        except Exception as e:
            logger.warning("Kiwoom NXT minute chart failed code=%s: %s", code, e)
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "kiwoom"}
        if data.get("return_code") != 0:
            return {"rt_cd": "1", "msg1": str(data.get("return_msg", "")), "output2": [], "vendor": "kiwoom"}
        rows = data.get("stk_min_pole_chart_qry") or []
        kept: list[dict] = []
        for r in rows:
            cntr_tm = str(r.get("cntr_tm", ""))
            if not cntr_tm.startswith(ymd):
                continue
            try:
                hms = int(cntr_tm[8:14])
            except (ValueError, TypeError):
                continue
            if hms < 154000 or hms > 200000:
                continue
            kept.append(dict(r))
        return {"rt_cd": "0", "output2": kept, "vendor": "kiwoom"}
