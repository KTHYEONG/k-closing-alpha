"""키움증권 REST API 클라이언트 (HTTP/토큰/요청 동작, TR별 5req/s 서버 강제 리밋)."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from src.api._common import SEOUL_TZ, deadline_remaining, now_seoul, resolve_chart_budget, validate_target_ymd
from src.api._transport import RetryPolicy, send_with_auth_retry
from src.api.kis.rate_limit import HostPacedRateLimiter, get_host_rate_limiter, host_admission_state_path
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

KIWOOM_AUTH_EXPIRED_RETURN_CODE: int = 3
KIWOOM_AUTH_EXPIRED_MSG_CODE: str = "8005"

KIWOOM_REVOKE_PATH: str = "/oauth2/revoke"

_CNTR_TM_RE = re.compile(r"^\d{14}$")


@dataclass(frozen=True)
class TickCursor:
    """Opaque continuation of a newest-first ka10079 pagination.

    Attributes:
        next_key: Vendor cursor header value for the next page.
        rows_received: Rows already consumed before this cursor (all pages of the pagination so far).
        total_ticks: Vendor-implied total (rows_received + remaining) when it could be parsed, else None.
    """

    next_key: str
    rows_received: int
    total_ticks: int | None


@dataclass(frozen=True)
class TapeDayCertificate:
    """Completeness proof for one market date observed on a Kiwoom tick tape walk.

    Attributes:
        day: Market date YYYY-MM-DD.
        received: Rows of that date received by the walk.
        vendor_total: Vendor-implied total for that date (received-so-far + remaining), None when never observed.
        complete: True only when the walk moved past the date and the date is proven whole, either by the vendor
            total (received + 1 == vendor_total) or, when no key ever carried that date, by being bracketed by rows
            of a newer date (or the closed tape head) and an older date inside one unbroken cursor chain.
        basis: "vendor_total", "bracketed" or "none" (the proof used when complete).
    """

    day: str
    received: int
    vendor_total: int | None
    complete: bool
    basis: str = "none"


def _parse_tape_key(next_key: str, base_code: str) -> tuple[str, int] | None:
    """Parse a tape ``next-key`` into its (date, remaining) pair.

    Expected form is ``A<request code><YYYYMMDD><remaining>`` where the request code carries the ``_NX`` suffix
    on the NXT tape; anything else yields
    ``None`` (fail closed: no certificate, never a guess).
    """
    key = str(next_key or "")
    prefix = f"A{base_code}"
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix):]
    if len(rest) < 9 or not rest[:8].isdigit() or not rest[8:].isdigit():
        return None
    ymd = rest[:8]
    try:
        datetime.strptime(ymd, "%Y%m%d")
    except ValueError:
        return None
    return ymd, int(rest[8:])


def _dash_day(ymd: str) -> str:
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"


def _parse_tick_remaining(next_key: str, base_code: str, ymd: str) -> int | None:
    """Parse the vendor remaining count from a ``next-key`` header.

    The documented form is ``A<code><YYYYMMDD><remaining>``; anything else
    yields ``None`` (fail closed: no certificate, never a guess).
    """
    prefix = f"A{base_code}{ymd}"
    key = str(next_key or "")
    tail = key[len(prefix):] if key.startswith(prefix) else ""
    return int(tail) if tail.isdigit() else None


@dataclass(frozen=True)
class KiwoomIssuedToken:
    """Token value plus its vendor-declared expiry (Asia/Seoul, from expires_dt).

    Attributes:
        token: Bearer token; excluded from ``repr``.
        expires_at: Aware Asia/Seoul expiry.
    """

    token: str = field(repr=False)
    expires_at: datetime


def _parse_kiwoom_expires_dt(raw: Any) -> datetime:
    """Parse vendor expires_dt (YYYYMMDDHHMMSS) into an aware Asia/Seoul datetime.

    Raises:
        RuntimeError: Missing or malformed expires_dt.
    """
    text = str(raw or "")
    if not re.fullmatch(r"\d{14}", text):
        raise RuntimeError(f"Kiwoom token expiry unparsable: {text!r}")
    try:
        naive = datetime.strptime(text, "%Y%m%d%H%M%S")
    except ValueError:
        raise RuntimeError(f"Kiwoom token expiry unparsable: {text!r}") from None
    return naive.replace(tzinfo=SEOUL_TZ)


class KiwoomApiClient:
    def __init__(self, app_key: str | None = None, secret_key: str | None = None, base_url: str | None = None) -> None:
        """Bind credentials and retry policy, preferring explicit arguments over the live Settings instance.

        Args:
            app_key: Explicit app key; falls back to ``settings.KIWOOM_APP_KEY``.
            secret_key: Explicit secret; falls back to ``settings.KIWOOM_SECRET_KEY``.
            base_url: Explicit origin; falls back to ``settings.KIWOOM_BASE_URL``.
        """
        self.app_key = app_key or settings.KIWOOM_APP_KEY
        self.secret_key = secret_key or settings.KIWOOM_SECRET_KEY
        self.base_url = base_url or settings.KIWOOM_BASE_URL
        self.token: str | None = None
        self._token_lock: asyncio.Lock | None = None
        self._rate_limit_max_retries: int = int(settings.KIWOOM_RATE_LIMIT_MAX_RETRIES)
        self._rate_limit_backoff: float = float(settings.KIWOOM_RATE_LIMIT_BACKOFF_SECONDS)

    def _limiter_for(self, api_id: str) -> HostPacedRateLimiter:
        return get_host_rate_limiter(host_admission_state_path("kiwoom", self.app_key or "", api_id), _KIWOOM_TR_RATE_PER_SEC)

    def reset_token(self) -> None:
        """Drop the cached token so the next request fetches one; the vendor reuses a live token and expires it ~24h after issuance."""
        self.token = None

    async def issue_token(self, session: aiohttp.ClientSession) -> KiwoomIssuedToken:
        """Issue (or receive the live) token and return it with its vendor-declared expiry.

        Raises:
            RuntimeError: Missing token or unparsable expires_dt in the vendor response.
        """
        payload = {"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.secret_key}
        async with session.post(
            f"{self.base_url}/oauth2/token",
            headers={"Content-Type": "application/json;charset=UTF-8", "User-Agent": _KIWOOM_USER_AGENT},
            json=payload,
        ) as resp:
            body = await resp.json()
        token = str(body.get("token", ""))
        if not token:
            raise RuntimeError(f"Kiwoom token issuance failed: {body}")
        expires_at = _parse_kiwoom_expires_dt(body.get("expires_dt"))
        self.token = token
        return KiwoomIssuedToken(token=token, expires_at=expires_at)

    async def revoke_token(self, session: aiohttp.ClientSession, token: str) -> None:
        """Revoke the given token (Kiwoom au10002) so the next issuance starts a new 24h phase.

        Raises:
            RuntimeError: Vendor reports failure (non-zero return_code) or a non-200 status.
        """
        async with session.post(
            f"{self.base_url}{KIWOOM_REVOKE_PATH}",
            headers={"Content-Type": "application/json;charset=UTF-8", "User-Agent": _KIWOOM_USER_AGENT},
            json={"appkey": self.app_key, "secretkey": self.secret_key, "token": token},
        ) as resp:
            status = resp.status
            body = await resp.json()
        if status != 200:
            raise RuntimeError(f"Kiwoom token revocation failed: status={status}")
        if body.get("return_code") != 0:
            raise RuntimeError(f"Kiwoom token revocation failed: {body.get('return_code')}")

    async def ensure_token(self, session: aiohttp.ClientSession) -> str:
        if self.token:
            return self.token
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()
        async with self._token_lock:
            if self.token:
                return self.token
            issued = await self.issue_token(session)
            return issued.token

    @staticmethod
    def _is_auth_expired(data: Any) -> bool:
        return data.get("return_code") == KIWOOM_AUTH_EXPIRED_RETURN_CODE and KIWOOM_AUTH_EXPIRED_MSG_CODE in str(
            data.get("return_msg", "")
        )

    async def _refresh_rejected_token(self, session: aiohttp.ClientSession, rejected_token: str) -> None:
        """Replace a token the vendor rejected, at most once across concurrent callers (compare-and-swap).

        Under the same ``_token_lock`` that ``ensure_token`` uses: if ``self.token`` is set and differs from
        ``rejected_token``, a concurrent caller already rotated it, so it is adopted with no issuance. Otherwise a
        token is issued through ``issue_token``. ``self.token`` is never cleared. Clearing it would make concurrent
        requests send ``Bearer None`` and start a cascade of re-issuance. A cancellation mid-issuance leaves the
        previous token in place.

        Raises:
            RuntimeError: Token issuance failed (propagated from ``issue_token``).
        """
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()
        async with self._token_lock:
            if self.token and self.token != rejected_token:
                return
            await self.issue_token(session)

    async def _post_tr(
        self,
        session: aiohttp.ClientSession,
        api_id: str,
        path: str,
        body: dict[str, Any],
        cont_yn: str = "N",
        next_key: str = "",
        max_retries: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """POST one Kiwoom TR and return (json body, response headers).

        Kiwoom expires the shared per-key token 24h after issuance and hands the same live token to every issuer,
        so a long run can hold an expired token. An auth rejection (``return_code ==
        KIWOOM_AUTH_EXPIRED_RETURN_CODE`` with ``KIWOOM_AUTH_EXPIRED_MSG_CODE`` in ``return_msg``; HTTP 401 alone is
        not a rejection for this vendor) triggers one compare-and-swap refresh and an identical replay that does
        not consume a rate-limit attempt. HTTP 429 is retried after ``KIWOOM_RATE_LIMIT_BACKOFF_SECONDS`` up to
        ``max_retries`` attempts. Every send, including the replay, first takes a slot from the per-TR host bucket.

        Args:
            session: Open HTTP session.
            api_id: Kiwoom TR id; it also selects the per-TR host admission bucket.
            path: Endpoint path appended to ``base_url``.
            body: JSON request body, sent identically on retries and the replay.
            cont_yn: Continuation flag header.
            next_key: Continuation cursor header.
            max_retries: Rate-limit attempts; None means ``KIWOOM_RATE_LIMIT_MAX_RETRIES`` captured at construction.

        Returns:
            The vendor body and headers of the last HTTP call. A second consecutive auth rejection, or an
            exhausted 429, is returned as-is so callers keep their vendor_failure handling.

        Raises:
            RuntimeError: Token issuance failed during the refresh.
            aiohttp.ClientError: Transport or non-JSON reply (including an HTML 429); the collector owns
                transport retries.
            ValueError: ``max_retries < 1``.
        """
        if not self.token:
            await self.ensure_token(session)
        limiter = self._limiter_for(api_id)
        attempts = int(max_retries) if max_retries is not None else self._rate_limit_max_retries
        backoff = self._rate_limit_backoff

        def open_request(token: str) -> Any:
            return session.post(
                f"{self.base_url}{path}",
                headers={
                    "Content-Type": "application/json;charset=UTF-8",
                    "User-Agent": _KIWOOM_USER_AGENT,
                    "authorization": f"Bearer {token}",
                    "api-id": api_id,
                    "cont-yn": cont_yn,
                    "next-key": next_key,
                },
                json=body,
            )

        policy = RetryPolicy(
            max_attempts=attempts,
            rate_limit_wait=lambda attempt, response: backoff if response.status == 429 else None,
        )
        response = await send_with_auth_retry(open_request, acquire=limiter.acquire, current_token=lambda: self.token or "", is_auth_rejected=lambda r: self._is_auth_expired(r.body), refresh=lambda sent: self._refresh_rejected_token(session, sent), policy=policy, log_stage="kiwoom_tr", log_context=f"api_id={api_id}")
        return response.body, response.headers

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
        collected: list[dict[str, Any]] = []
        in_observer = False
        try:
            for page_index in range(max(1, int(max_pages))):
                started = now_seoul()
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
                received = now_seoul()
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
        venue: Literal["KRX", "NXT"] = "KRX",
        floor_hms: str | None = None,
        resume: TickCursor | None = None,
    ) -> BrokerPayload:
        """Expose incomplete successful tick responses as bounded repairable tasks.

        Args:
            session: Existing HTTP session.
            code: Explicit broker instrument identifier.
            target_date: Requested market date.
            max_pages: Compatible legacy page limit.
            budget: Page/deadline/timeout limits.
            on_page: Observer of unfiltered raw responses and continuation metadata.
            venue: "KRX" sends the plain code (KRX tape); "NXT" sends "{code}_NX" (NXT tape). The two tapes are
                distinct venues and must never be merged.
            floor_hms: Optional HHMMSS lower bound. Pagination stops with termination "crossed_time_floor" once the
                oldest row of a page is on the target date and earlier than floor_hms, proving the window above
                the floor was fully traversed without paging through the regular session.
            resume: Opaque continuation from a previous truncated pagination. The first request sends
                ``cont-yn=Y`` with ``resume.next_key``; totals continue from the cursor so the
                ``rows_received + remaining == total`` invariant spans both calls. Rows of the resumed
                call are returned WITHOUT the earlier prefix; the caller owns concatenation.

        Returns:
            Existing rt_cd/output2/vendor/truncated keys plus termination metadata, ``cursor``,
            ``vendor_total_ticks``, ``rows_received`` and ``complete_by_total``.

        Raises:
            ValueError: Invalid or conflicting bounds.
            OSError: Mandatory capture fails.
        """
        ymd = validate_target_ymd(target_date)
        if venue not in ("KRX", "NXT"):
            raise ValueError(f"unknown tick venue: {venue!r}")
        if floor_hms is not None and (len(str(floor_hms)) != 6 or not str(floor_hms).isdigit()):
            raise ValueError(f"invalid floor_hms: {floor_hms!r}")
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
        base_code = str(code).split("_")[0]
        if resume is not None:
            cont_yn, next_key = "Y", str(resume.next_key)
            rows_received = int(resume.rows_received)
            vendor_total: int | None = resume.total_ticks
        else:
            cont_yn, next_key = "N", ""
            rows_received = 0
            vendor_total = None
        all_rows: list[dict[str, Any]] = []
        metadata: dict[str, str] = {}
        termination = "exhausted"
        truncated = False
        pages_fetched = 0
        failure_msg = ""
        prev_identity: tuple[Any, ...] | None = None
        stalls = 0
        in_observer = False
        last_header_cont = "N"
        last_header_key = ""
        request_code = str(code) if venue == "KRX" else f"{str(code).split('_')[0]}_NX"
        try:
            for page_index in range(max(1, int(page_budget))):
                deadline_left = deadline_remaining(deadline)
                if deadline_left is not None and deadline_left <= 0:
                    termination = "deadline"
                    truncated = True
                    break
                started = now_seoul()
                call = self._post_tr(
                    session, "ka10079", "/api/dostk/chart",
                    {"stk_cd": request_code, "tic_scope": "1", "upd_stkpc_tp": "1", "base_dt": ymd},
                    cont_yn=cont_yn, next_key=next_key,
                )
                if deadline_left is None:
                    data, resp_headers = await call
                else:
                    data, resp_headers = await asyncio.wait_for(call, timeout=deadline_left)
                received = now_seoul()
                header_cont = str(resp_headers.get("cont-yn", "N") or "N")
                header_key = str(resp_headers.get("next-key", "") or "")
                metadata = {"cont-yn": header_cont, "next-key": header_key}
                last_header_cont, last_header_key = header_cont, header_key
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
                dict_rows = [dict(r) for r in rows if isinstance(r, dict)]
                rows_before = rows_received
                if dict_rows:
                    all_rows.extend(dict_rows)
                rows_received += len(dict_rows)
                parsed = _parse_tick_remaining(header_key, base_code, ymd)
                if parsed is not None:
                    candidate = rows_before + len(dict_rows) + parsed
                    if vendor_total is None:
                        vendor_total = candidate
                    elif candidate != vendor_total:
                        termination = "cursor_inconsistent"
                        truncated = True
                        break
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
                if parsed is not None and parsed == 0:
                    termination = "exhausted"
                    break
                if header_cont != "Y":
                    termination = "exhausted"
                    break
                if not header_key:
                    termination = "cursor_unknown"
                    truncated = True
                    break
                oldest = str(dict_rows[-1].get("cntr_tm", "") or "") if dict_rows else ""
                if _CNTR_TM_RE.fullmatch(oldest) and oldest[:8] < ymd:
                    termination = "crossed_target_date"
                    break
                if (
                    floor_hms is not None
                    and _CNTR_TM_RE.fullmatch(oldest)
                    and oldest[:8] == ymd
                    and oldest[8:14] < str(floor_hms)
                ):
                    termination = "crossed_time_floor"
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
            return {"rt_cd": "1", "msg1": str(e), "output2": [], "vendor": "kiwoom", "truncated": True, "termination_reason": "vendor_failure", "pages_fetched": pages_fetched, "continuation": metadata, "cursor": None, "vendor_total_ticks": vendor_total, "rows_received": rows_received, "complete_by_total": False}
        if termination == "vendor_failure":
            return {"rt_cd": "1", "msg1": failure_msg, "output2": [], "vendor": "kiwoom", "truncated": True, "termination_reason": termination, "pages_fetched": pages_fetched, "continuation": metadata, "cursor": None, "vendor_total_ticks": vendor_total, "rows_received": rows_received, "complete_by_total": False}
        filtered = [dict(r) for r in all_rows if str(r.get("cntr_tm", "")).startswith(ymd)]
        complete_by_total = termination == "exhausted" and vendor_total is not None
        cursor: TickCursor | None = None
        if truncated and last_header_cont == "Y" and last_header_key and termination in ("page_budget", "deadline", "crossed_time_floor"):
            cursor = TickCursor(next_key=last_header_key, rows_received=rows_received, total_ticks=vendor_total)
        return {"rt_cd": "0", "output2": filtered, "vendor": "kiwoom", "truncated": truncated, "termination_reason": termination, "pages_fetched": pages_fetched, "continuation": metadata, "cursor": cursor, "vendor_total_ticks": vendor_total, "rows_received": rows_received, "complete_by_total": complete_by_total}

    async def walk_tick_tape(
        self,
        session: aiohttp.ClientSession,
        code: str,
        *,
        venue: Literal["KRX", "NXT"] = "KRX",
        stop_before_day: str,
        budget: ChartBudget | None = None,
        max_pages: int | None = None,
        on_page: PageObserver | None = None,
        on_day_complete: Callable[[str, list[dict[str, Any]], TapeDayCertificate], None] | None = None,
    ) -> BrokerPayload:
        """Walk the newest-first tape back to a stop date, certifying every fully traversed date.

        Args:
            session: Existing HTTP session.
            code: 6-digit symbol (the NXT tape is addressed internally via venue).
            venue: Tape to walk.
            stop_before_day: Oldest date that must be fully traversed (YYYY-MM-DD); the walk ends once a page's oldest row
                belongs to an earlier date or the tape ends.
            budget / max_pages: Page bound; mutually consistent like get_tick_chart.
            on_page: Raw page observer called before filtering (evidence persistence).
            on_day_complete: Called once per date, in newest-to-oldest order, as soon as the date is certified complete,
                with that date's rows (unfiltered by session window) and its certificate. Rows are NOT retained afterwards.

        Returns:
            rt_cd/vendor/termination metadata plus `certificates` (every date seen, complete or not) and `pages_fetched`;
            no row payload (rows are delivered only through on_day_complete to keep memory bounded).
            `termination_reason` is one of `tape_end` / `crossed_stop_day` / `tape_empty` / `page_budget` /
            `deadline` / `nonprogress` / `cursor_inconsistent` / `vendor_failure`; `tape_empty` means the first
            page carried no valid 14-digit `cntr_tm` row with no continuation.

        Raises:
            ValueError: Invalid or conflicting bounds.
            OSError: Mandatory evidence persistence fails.
        """
        stop_ymd = validate_target_ymd(stop_before_day)
        if venue not in ("KRX", "NXT"):
            raise ValueError(f"unknown tick venue: {venue!r}")
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
        base_code = str(code).split("_")[0]
        request_code = str(code) if venue == "KRX" else f"{base_code}_NX"
        cont_yn, next_key = "N", ""
        metadata: dict[str, str] = {}
        termination = "tape_end"
        truncated = False
        pages_fetched = 0
        failure_msg = ""
        prev_identity: tuple[Any, ...] | None = None
        stalls = 0
        in_observer = False
        day_order: list[str] = []
        received: dict[str, int] = {}
        vendor_totals: dict[str, int] = {}
        buffers: dict[str, list[dict[str, Any]]] = {}
        certs: dict[str, TapeDayCertificate] = {}
        certified_order: list[str] = []

        today_ymd = now_seoul().strftime("%Y%m%d")

        def _certify(ymd: str, older_seen: bool) -> None:
            total = vendor_totals.get(ymd)
            count = int(received.get(ymd, 0))
            if total is not None:
                ok = count == total - 1
                basis = "vendor_total" if ok else "none"
            else:
                position = day_order.index(ymd)
                newer_seen = position > 0 or ymd < today_ymd
                ok = newer_seen and older_seen
                basis = "bracketed" if ok else "none"
            cert = TapeDayCertificate(
                day=_dash_day(ymd), received=count, vendor_total=total, complete=bool(ok), basis=basis,
            )
            certs[ymd] = cert
            certified_order.append(ymd)
            rows = buffers.pop(ymd, [])
            if ok and on_day_complete is not None:
                on_day_complete(cert.day, rows, cert)

        try:
            for page_index in range(max(1, int(page_budget))):
                deadline_left = deadline_remaining(deadline)
                if deadline_left is not None and deadline_left <= 0:
                    termination = "deadline"
                    truncated = True
                    break
                started = now_seoul()
                call = self._post_tr(
                    session,
                    "ka10079",
                    "/api/dostk/chart",
                    {"stk_cd": request_code, "tic_scope": "1", "upd_stkpc_tp": "1", "base_dt": stop_ymd},
                    cont_yn=cont_yn,
                    next_key=next_key,
                )
                if deadline_left is None:
                    data, resp_headers = await call
                else:
                    data, resp_headers = await asyncio.wait_for(call, timeout=deadline_left)
                received_at = now_seoul()
                header_cont = str(resp_headers.get("cont-yn", "N") or "N")
                header_key = str(resp_headers.get("next-key", "") or "")
                metadata = {"cont-yn": header_cont, "next-key": header_key}
                in_observer = True
                if on_page is not None:
                    on_page(dict(data), metadata, started, received_at, page_index, 0)
                in_observer = False
                pages_fetched += 1
                if data.get("return_code") != 0:
                    termination = "vendor_failure"
                    truncated = True
                    failure_msg = str(data.get("return_msg", ""))
                    break
                raw_rows = data.get("stk_tic_chart_qry") or []
                valid_rows: list[dict[str, Any]] = []
                malformed = False
                for r in raw_rows:
                    if not isinstance(r, dict):
                        malformed = True
                        continue
                    cntr_tm = str(r.get("cntr_tm", "") or "")
                    if not _CNTR_TM_RE.fullmatch(cntr_tm):
                        malformed = True
                        continue
                    valid_rows.append(dict(r))
                for r in valid_rows:
                    ymd = str(r.get("cntr_tm", ""))[:8]
                    if ymd not in buffers:
                        buffers[ymd] = []
                        day_order.append(ymd)
                        received[ymd] = 0
                    buffers[ymd].append(r)
                    received[ymd] = int(received.get(ymd, 0)) + 1
                if page_index == 0 and not valid_rows and header_cont != "Y":
                    termination = "tape_empty"
                    truncated = False
                    break
                if malformed:
                    termination = "cursor_inconsistent"
                    truncated = True
                    break
                parsed = _parse_tape_key(header_key, request_code) if header_key else None
                if header_key and parsed is None:
                    pass
                elif parsed is not None:
                    key_ymd, remaining = parsed
                    candidate = int(received.get(key_ymd, 0)) + int(remaining)
                    if key_ymd in vendor_totals:
                        if candidate != vendor_totals[key_ymd]:
                            termination = "cursor_inconsistent"
                            truncated = True
                            break
                    else:
                        vendor_totals[key_ymd] = candidate
                identity = (
                    header_cont,
                    header_key,
                    tuple(str(r.get("cntr_tm", "")) for r in valid_rows),
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
                is_tape_end = header_cont != "Y"
                if is_tape_end:
                    unsettled = [d for d in day_order if d not in certs]
                    for ymd in unsettled:
                        _certify(ymd, older_seen=ymd != unsettled[-1])
                    termination = "tape_end"
                    truncated = False
                    break
                if not header_key:
                    termination = "cursor_inconsistent"
                    truncated = True
                    break
                pending = [d for d in day_order if d not in certs]
                if len(pending) >= 2:
                    for ymd in pending[:-1]:
                        _certify(ymd, older_seen=True)
                oldest_ymd: str | None = None
                if valid_rows:
                    oldest_ymd = str(valid_rows[-1].get("cntr_tm", ""))[:8]
                if oldest_ymd is not None and oldest_ymd < stop_ymd:
                    termination = "crossed_stop_day"
                    truncated = False
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
            logger.warning("Kiwoom tape walk failed code=%s: %s", code, e)
            pending_certs = [TapeDayCertificate(day=_dash_day(d), received=int(received.get(d, 0)), vendor_total=vendor_totals.get(d), complete=False) for d in day_order if d not in certs]
            return {"rt_cd": "1", "msg1": str(e), "vendor": "kiwoom", "truncated": True, "termination_reason": "vendor_failure", "pages_fetched": pages_fetched, "continuation": metadata, "certificates": [certs[d] for d in certified_order] + pending_certs}
        if termination == "vendor_failure":
            logger.warning("[DATA] stage=kiwoom_tape_walk status=VENDOR_FAILURE code=%s pages=%d return_msg=%s", code, pages_fetched, failure_msg)
            certificates: list[TapeDayCertificate] = [certs[d] for d in certified_order]
            certificates.extend(TapeDayCertificate(day=_dash_day(d), received=int(received.get(d, 0)), vendor_total=vendor_totals.get(d), complete=False) for d in day_order if d not in certs)
            payload: BrokerPayload = {"rt_cd": "1", "msg1": failure_msg, "vendor": "kiwoom", "truncated": True, "termination_reason": termination, "pages_fetched": pages_fetched, "continuation": metadata, "certificates": certificates}
            return payload
        certificates = [certs[d] for d in certified_order]
        certificates.extend(TapeDayCertificate(day=_dash_day(d), received=int(received.get(d, 0)), vendor_total=vendor_totals.get(d), complete=False) for d in day_order if d not in certs)
        return {"rt_cd": "0", "vendor": "kiwoom", "truncated": truncated, "termination_reason": termination, "pages_fetched": pages_fetched, "continuation": metadata, "certificates": certificates}

    async def get_nxt_premarket_chart(self, session: aiohttp.ClientSession, code: str, target_date: str) -> BrokerPayload:
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
        kept: list[dict[str, Any]] = []
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

    async def get_nxt_minute_chart(self, session: aiohttp.ClientSession, code: str, target_date: str) -> BrokerPayload:
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
        kept: list[dict[str, Any]] = []
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
