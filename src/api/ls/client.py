"""LS증권 OpenAPI 클라이언트 (t8412 분봉 / t8411 틱 우선 라우팅)."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.api._common import deadline_remaining, now_seoul, parse_expires_in, resolve_chart_budget, validate_target_ymd
from src.api._transport import RetryPolicy, send_with_auth_retry
from src.api.kis.rate_limit import HostPacedRateLimiter, get_host_rate_limiter, host_admission_state_path
from src.api.shared_token import IssuedToken, SharedTokenStore, shared_token_path
from src.config import settings

if TYPE_CHECKING:
    import aiohttp

    from src.data.capture_contracts import BrokerPayload, ChartBudget, PageObserver

logger = logging.getLogger(__name__)

_YMD_RE = re.compile(r"^\d{8}$")
_HHMMSS_RE = re.compile(r"^\d{6}$")

LS_AUTH_REJECTED_RSP_CODES: frozenset[str] = frozenset({"IGW00101", "IGW00102", "IGW00123"})

_LS_TOKEN_EXPIRY_MARGIN_SECONDS: float = 300.0


class LsApiClient:
    def __init__(self, app_key: str | None = None, app_secret: str | None = None) -> None:
        """Bind credentials, origin and pacing from explicit arguments or the live Settings instance.

        Pacing is enforced host-wide through one file-backed bucket per app key
        (shared with every LS REST consumer on the host, including krx-alpha);
        per-instance spacing state was removed so the host bucket is the only
        spacing authority.

        Args:
            app_key: Explicit app key; falls back to ``settings.LS_APP_KEY``.
            app_secret: Explicit secret; falls back to ``settings.LS_APP_SECRET``.
        """
        self.app_key = app_key or settings.LS_APP_KEY
        self.app_secret = app_secret or settings.LS_APP_SECRET
        self.base_url = settings.LS_BASE_URL
        self.token: str | None = None
        self._min_interval: float = float(settings.LS_MIN_INTERVAL_SECONDS)
        self._rate_limit_max_retries: int = int(settings.LS_RATE_LIMIT_MAX_RETRIES)
        self._rate_limit_backoff: float = float(settings.LS_RATE_LIMIT_BACKOFF_SECONDS)

    def _limiter(self) -> HostPacedRateLimiter:
        interval = float(self._min_interval)
        rate = 1.0 / interval if interval > 0 else 1_000_000_000.0
        return get_host_rate_limiter(host_admission_state_path("ls", self.app_key or ""), rate)

    def _token_store(self) -> SharedTokenStore:
        return SharedTokenStore(
            shared_token_path("ls", self.app_key or ""),
            lock_timeout_seconds=float(settings.BROKER_ADMISSION_LOCK_TIMEOUT_SECONDS),
            expiry_margin_seconds=_LS_TOKEN_EXPIRY_MARGIN_SECONDS,
            clock=lambda: datetime.now(UTC),
        )

    async def _issue_token(self, session: aiohttp.ClientSession) -> IssuedToken:
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecretkey": self.app_secret,
            "scope": "oob",
        }
        async with session.post(f"{self.base_url}/oauth2/token", data=payload) as resp:
            body = await resp.json()
        token = str(body.get("access_token", ""))
        if not token:
            raise RuntimeError(f"LS token issuance failed: {body}")
        expires_in = parse_expires_in(body.get("expires_in"))
        return IssuedToken(access_token=token, expires_in_seconds=expires_in)

    async def ensure_token(self, session: aiohttp.ClientSession) -> str:
        record = await self._token_store().get_or_issue(lambda: self._issue_token(session))
        self.token = record.access_token
        return self.token

    @staticmethod
    def _is_auth_rejected(status: int, data: dict[str, Any]) -> bool:
        if status == 401:
            return True
        return str(data.get("rsp_cd", "")) in LS_AUTH_REJECTED_RSP_CODES

    async def _refresh_rejected_token(self, session: aiohttp.ClientSession, rejected_token: str) -> None:
        """Replace the rejected token through the host-shared store (compare-and-swap on the sent token).

        ``SharedTokenStore.replace_rejected`` returns a peer-rotated usable token unchanged, and only issues when
        the stored token is the one that was rejected. Passing the token that was sent, rather than ``self.token``
        at handling time, prevents a second issuance when another task already rotated it.

        Raises:
            TokenStoreLockTimeout: Store lock not acquired within ``BROKER_ADMISSION_LOCK_TIMEOUT_SECONDS``.
            RuntimeError: Issuance returned no token.
        """
        record = await self._token_store().replace_rejected(rejected_token, lambda: self._issue_token(session))
        self.token = record.access_token

    async def _post_tr(
        self,
        session: aiohttp.ClientSession,
        tr_cd: str,
        tr_key: str,
        body: dict[str, Any],
        tr_cont: str = "N",
        tr_cont_key: str = "",
        max_retries: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """POST one LS chart TR and return (json body, response headers).

        LS signals throttling in the body (``rsp_cd == "IGW00201"``), not with HTTP 429. HTTP 429 is therefore not
        retried here; it is returned to the caller. Throttled attempts back off exponentially
        (``LS_RATE_LIMIT_BACKOFF_SECONDS * 2**attempt``). An auth rejection (HTTP 401 or ``rsp_cd`` in
        ``LS_AUTH_REJECTED_RSP_CODES``) triggers one compare-and-swap refresh and an identical replay that does
        not consume an attempt. Every send takes a host admission slot first.

        Args:
            session: Open HTTP session.
            tr_cd: LS TR code (sent as header and injected into the JSON body).
            tr_key: Caller correlation key (symbol); used only for log context.
            body: TR input block.
            tr_cont: Continuation flag header.
            tr_cont_key: Continuation cursor header.
            max_retries: Rate-limit attempts; None means ``LS_RATE_LIMIT_MAX_RETRIES`` captured at construction.

        Returns:
            The body and headers of the last HTTP call. A second consecutive auth rejection, or an exhausted
            ``IGW00201``, is returned as-is.

        Raises:
            TokenStoreLockTimeout, RuntimeError: Refresh failed.
            aiohttp.ClientError: Transport or non-JSON reply.
            ValueError: ``max_retries < 1``.
        """
        if not self.token:
            await self.ensure_token(session)
        limit = int(max_retries) if max_retries is not None else self._rate_limit_max_retries
        backoff = self._rate_limit_backoff

        def open_request(token: str) -> Any:
            return session.post(
                f"{self.base_url}/stock/chart",
                json={**body, "tr_cd": tr_cd},
                headers={
                    "content-type": "application/json; charset=utf-8",
                    "authorization": f"Bearer {token}",
                    "tr_cd": tr_cd,
                    "tr_cont": tr_cont,
                    "tr_cont_key": tr_cont_key,
                },
            )

        policy = RetryPolicy(
            max_attempts=limit,
            rate_limit_wait=lambda attempt, response: backoff * (2**attempt)
            if str(response.body.get("rsp_cd", "")) == "IGW00201"
            else None,
        )
        response = await send_with_auth_retry(open_request, acquire=lambda: self._limiter().acquire(), current_token=lambda: self.token or "", is_auth_rejected=lambda r: self._is_auth_rejected(r.status, r.body), refresh=lambda sent: self._refresh_rejected_token(session, sent), policy=policy, log_stage="ls_tr", log_context=f"tr_cd={tr_cd} tr_key={tr_key}")
        return response.body, response.headers

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
        ymd = validate_target_ymd(target_date)
        max_pages, deadline = resolve_chart_budget(budget, int(settings.COLLECTION_CHART_MAX_PAGES))
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
                remaining = deadline_remaining(deadline)
                if remaining is not None and remaining <= 0:
                    termination = "deadline"
                    truncated = True
                    break
                started = now_seoul()
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
                received = now_seoul()
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
                # 서버가 준 cts_time은 이번 페이지에서 가장 오래된 봉보다 1분 이르지만, 이를 그대로 되보내면 서버는 그 시각
                # "미만"만 돌려줘 경계 봉이 통째로 유실된다(실측 2026-09-29: 하루치가 한 페이지를 넘는 시점부터
                # 종목당 1봉 누락). 가장 오래된 봉 시각을 커서로 쓰면 그 미만(= 미수신 구간)만 정확히 이어 받는다.
                page_times = [str(r.get("time", "") or "") for r in rows if isinstance(r, dict) and str(r.get("date", "") or "").strip() == body_cts_date]
                oldest_time = min((t for t in page_times if t), default="")
                cts_date, cts_time = body_cts_date, oldest_time or body_cts_time
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
        ymd = validate_target_ymd(target_date)
        if max_pages is not None and budget is not None and int(max_pages) != int(budget.max_pages):
            raise ValueError("conflicting tick acquisition limits")
        if max_pages is not None and int(max_pages) <= 0:
            raise ValueError("invalid tick acquisition limits")
        if budget is not None:
            page_budget, deadline = resolve_chart_budget(budget, int(settings.COLLECTION_CHART_MAX_PAGES))
        elif max_pages is not None:
            page_budget, deadline = max(1, int(max_pages)), None
        else:
            page_budget, deadline = resolve_chart_budget(None, int(settings.COLLECTION_CHART_MAX_PAGES))
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
                remaining = deadline_remaining(deadline)
                if remaining is not None and remaining <= 0:
                    termination = "deadline"
                    truncated = True
                    break
                started = now_seoul()
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
                received = now_seoul()
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

