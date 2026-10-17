"""DB Securities OpenAPI async client (CHARTMIN minute bars / DAYTRADE auction path)."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.api._common import (
    deadline_remaining,
    now_seoul,
    parse_expires_in,
    resolve_chart_budget,
    validate_target_ymd,
)
from src.api._transport import RetryPolicy, send_with_auth_retry
from src.api.host_broker_contract import TokenIssueThrottled
from src.api.kis.rate_limit import HostPacedRateLimiter, vendor_limiter
from src.api.shared_token import IssuedToken, SharedTokenStore, shared_token_path
from src.config import settings

if TYPE_CHECKING:
    import aiohttp

    from src.data.capture_contracts import BrokerPayload, ChartBudget, PageObserver

logger = logging.getLogger(__name__)

_CODE_RE = re.compile(r"^[0-9A-Z]{6}$")
_HHMMSS_RE = re.compile(r"^\d{6}$")

DB_AUTH_REJECTED_RSP_CODES: frozenset[str] = frozenset({"IGW00121", "IGW00123"})
DB_THROTTLE_RSP_CODE: str = "IGW00201"
DB_AUCTION_PATH_HOUR_CLASS: str = "8"

_CHARTMIN_PATH = "/api/v1/quote/kr-chart/min"
_AUCTION_PATH = "/api/v1/quote/kr-stock/inquiry/daily-price"
_OAUTH_PATH = "/oauth2/token"

_DB_TOKEN_EXPIRY_MARGIN_SECONDS: float = 600.0


class DbsecTokenThrottled(TokenIssueThrottled):  # noqa: N818 - spec-mandated protocol name
    """DB token issuance refused by the 61 s host guard or vendor IGW00201."""


def _rows_of(data: Any) -> list[dict[str, Any]]:
    out = data.get("Out") if isinstance(data, dict) else None
    if isinstance(out, dict):
        return [out]
    if isinstance(out, list):
        return [dict(r) for r in out if isinstance(r, dict)]
    return []


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(f"{k}={row.get(k, '')}" for k in ("Date", "Hour", "Prpr", "CntgVol", "ShnuCnqn", "SelnCnqn"))


class DbsecApiClient:
    def __init__(self, app_key: str | None = None, app_secret: str | None = None) -> None:
        """Bind credentials, origin and pacing from explicit arguments or the live Settings instance."""
        self.app_key = app_key or settings.DB_APP_KEY
        self.app_secret = app_secret or settings.DB_APP_SECRET
        self.base_url = settings.DB_BASE_URL
        self.token: str | None = None
        self._rate_limit_max_retries: int = int(settings.DB_RATE_LIMIT_MAX_RETRIES)
        self._rate_limit_backoff: float = float(settings.DB_RATE_LIMIT_BACKOFF_SECONDS)

    def _token_store(self) -> SharedTokenStore:
        return SharedTokenStore(
            shared_token_path("dbsec", self.app_key or ""),
            vendor="dbsec",
            lock_timeout_seconds=float(settings.BROKER_ADMISSION_LOCK_TIMEOUT_SECONDS),
            expiry_margin_seconds=_DB_TOKEN_EXPIRY_MARGIN_SECONDS,
            issue_throttle_policy="raise",
            clock=lambda: datetime.now(UTC),
        )

    def _scope_limiter(self, scope: str) -> HostPacedRateLimiter:
        return vendor_limiter("dbsec", self.app_key or "", scope)

    def _app_limiter(self) -> HostPacedRateLimiter:
        return vendor_limiter("dbsec", self.app_key or "")

    async def _issue_token(self, session: aiohttp.ClientSession) -> IssuedToken:
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecretkey": self.app_secret,
            "scope": "oob",
        }
        async with session.post(
            f"{self.base_url}{_OAUTH_PATH}",
            data=payload,
            headers={"content-type": "application/x-www-form-urlencoded"},
        ) as resp:
            body = await resp.json()
        token = str(body.get("access_token", "") or "") if isinstance(body, dict) else ""
        if not token:
            code = ""
            if isinstance(body, dict):
                code = str(body.get("error_code", "") or body.get("rsp_cd", "") or "").strip()
            if code == DB_THROTTLE_RSP_CODE:
                raise DbsecTokenThrottled("DB token issuance throttled (1/min)")
            raise RuntimeError(f"DB token issuance failed: {code or 'unknown'}")
        expires_in = parse_expires_in(body.get("expires_in") if isinstance(body, dict) else None)
        return IssuedToken(access_token=token, expires_in_seconds=expires_in)

    async def ensure_token(self, session: aiohttp.ClientSession) -> str:
        """Return a usable host-shared token, issuing at most once host-wide when none is usable."""
        try:
            record = await self._token_store().get_or_issue(lambda: self._issue_token(session))
        except TokenIssueThrottled as exc:
            if isinstance(exc, DbsecTokenThrottled):
                raise
            raise DbsecTokenThrottled("DB token issuance throttled (1/min)") from exc
        self.token = record.access_token
        return self.token

    async def _refresh_rejected_token(self, session: aiohttp.ClientSession, rejected_token: str) -> None:
        try:
            record = await self._token_store().replace_rejected(rejected_token, lambda: self._issue_token(session))
        except TokenIssueThrottled as exc:
            if isinstance(exc, DbsecTokenThrottled):
                raise
            raise DbsecTokenThrottled("DB token issuance throttled (1/min)") from exc
        self.token = record.access_token

    @staticmethod
    def _is_auth_rejected(status: int, data: dict[str, Any]) -> bool:
        if status == 401:
            return True
        if not isinstance(data, dict):
            return False
        return str(data.get("rsp_cd", "")) in DB_AUTH_REJECTED_RSP_CODES

    async def _post(
        self,
        session: aiohttp.ClientSession,
        path: str,
        body: dict[str, Any],
        *,
        cont_key: str = "",
        scope: str,
        code: str = "",
    ) -> tuple[dict[str, Any], dict[str, str]]:
        if not self.token:
            await self.ensure_token(session)
        backoff = self._rate_limit_backoff
        limit = self._rate_limit_max_retries
        scope_limiter = self._scope_limiter(scope)
        app_limiter = self._app_limiter()

        async def _acquire() -> None:
            await scope_limiter.acquire()
            await app_limiter.acquire()

        def open_request(token: str) -> Any:
            return session.post(
                f"{self.base_url}{path}",
                json={"In": body},
                headers={
                    "content-type": "application/json; charset=utf-8",
                    "authorization": f"Bearer {token}",
                    "cont_yn": "Y" if cont_key else "N",
                    "cont_key": cont_key,
                },
            )

        def _wait(attempt: int, response: Any) -> float | None:
            payload = response.body
            if isinstance(payload, dict) and str(payload.get("rsp_cd", "")) == DB_THROTTLE_RSP_CODE:
                return float(backoff * (2**attempt))
            return None

        policy = RetryPolicy(max_attempts=limit, rate_limit_wait=_wait)
        response = await send_with_auth_retry(
            open_request,
            acquire=_acquire,
            current_token=lambda: self.token or "",
            is_auth_rejected=lambda r: self._is_auth_rejected(r.status, r.body if isinstance(r.body, dict) else {}),
            refresh=lambda sent: self._refresh_rejected_token(session, sent),
            policy=policy,
            log_stage="dbsec_tr",
            log_context=f"path={path} code={code}",
        )
        return response.body, response.headers

    @staticmethod
    def _check_code(code: str) -> None:
        if not _CODE_RE.fullmatch(str(code)):
            raise ValueError(f"invalid code: {code!r}")

    async def get_minute_chart(
        self,
        session: aiohttp.ClientSession,
        code: str,
        target_date: str,
        *,
        budget: ChartBudget | None = None,
        on_page: PageObserver | None = None,
    ) -> BrokerPayload:
        """Page 1-minute KRX (J) bars backward until the target date is fully covered; returns rt_cd/output2/vendor payload with termination evidence."""
        self._check_code(code)
        ymd = validate_target_ymd(target_date)
        max_pages, deadline = resolve_chart_budget(budget, int(settings.COLLECTION_CHART_MAX_PAGES))
        body = {
            "dataCnt": "2000",
            "InputCondMrktDivCode": "J",
            "InputIscd1": str(code),
            "InputDate1": ymd,
            "InputDivXtick": "60",
            "InputOrgAdjPrc": "0",
        }
        all_rows: list[dict[str, Any]] = []
        metadata: dict[str, str] = {}
        termination = "exhausted"
        truncated = False
        pages_fetched = 0
        failure_msg = ""
        prev_identity: tuple[Any, ...] | None = None
        stalls = 0
        in_observer = False
        cont_key = ""
        try:
            for page_index in range(max_pages):
                remaining = deadline_remaining(deadline)
                if remaining is not None and remaining <= 0:
                    termination = "deadline"
                    truncated = True
                    break
                started = now_seoul()
                call = self._post(session, _CHARTMIN_PATH, dict(body), cont_key=cont_key, scope="chart", code=str(code))
                if remaining is None:
                    data, resp_headers = await call
                else:
                    data, resp_headers = await asyncio.wait_for(call, timeout=remaining)
                received = now_seoul()
                cont_yn = str(resp_headers.get("cont_yn", "N") or "N")
                resp_key = str(resp_headers.get("cont_key", "") or "")
                metadata = {"cont_yn": cont_yn, "cont_key": resp_key}
                in_observer = True
                if on_page is not None:
                    on_page(dict(data), dict(metadata), started, received, page_index, 0)
                in_observer = False
                pages_fetched += 1
                if not isinstance(data, dict) or str(data.get("rsp_cd", "")) != "00000":
                    termination = "vendor_failure"
                    truncated = True
                    failure_msg = str(data.get("rsp_msg", "") if isinstance(data, dict) else "")
                    break
                rows = _rows_of(data)
                first = _row_key(rows[0]) if rows else ()
                last = _row_key(rows[-1]) if rows else ()
                identity = (resp_key, first, last)
                if identity == prev_identity:
                    stalls += 1
                else:
                    stalls = 0
                    prev_identity = identity
                    # A replayed page would inject phantom events: duplicate rows are real auction updates here.
                    all_rows.extend(rows)
                if stalls >= 2:
                    termination = "nonprogress"
                    truncated = True
                    break
                if any(str(r.get("Date", "")) < ymd for r in rows if str(r.get("Date", "")).strip()):
                    termination = "crossed_target_date"
                    break
                if cont_yn != "Y":
                    termination = "exhausted"
                    break
                cont_key = resp_key
            else:
                termination = "page_budget"
                truncated = True
        except TimeoutError:
            termination = "deadline"
            truncated = True
        except DbsecTokenThrottled:
            raise
        except Exception as e:
            if in_observer:
                raise
            logger.warning("DB chartmin failed code=%s: %s", code, type(e).__name__)
            return {
                "rt_cd": "1",
                "msg1": type(e).__name__,
                "output2": [],
                "vendor": "dbsec",
                "truncated": True,
                "termination_reason": "vendor_failure",
                "pages_fetched": pages_fetched,
                "continuation": metadata,
            }
        if termination == "vendor_failure":
            return {
                "rt_cd": "1",
                "msg1": failure_msg,
                "output2": [],
                "vendor": "dbsec",
                "truncated": True,
                "termination_reason": termination,
                "pages_fetched": pages_fetched,
                "continuation": metadata,
            }
        filtered = [dict(r) for r in all_rows if str(r.get("Date", "")) == ymd]
        return {
            "rt_cd": "0",
            "output2": filtered,
            "vendor": "dbsec",
            "truncated": truncated,
            "termination_reason": termination,
            "pages_fetched": pages_fetched,
            "continuation": metadata,
        }

    async def get_auction_path(
        self,
        session: aiohttp.ClientSession,
        code: str,
        *,
        floor_hhmmss: str,
        budget: ChartBudget | None = None,
        on_page: PageObserver | None = None,
    ) -> BrokerPayload:
        """Page today's DAYTRADE class-8 rows backward until a row older than ``floor_hhmmss``; returns rt_cd/output2/vendor payload with termination evidence."""
        self._check_code(code)
        floor = str(floor_hhmmss)
        if not _HHMMSS_RE.fullmatch(floor):
            raise ValueError(f"invalid floor_hhmmss: {floor_hhmmss!r}")
        max_pages, deadline = resolve_chart_budget(budget, int(settings.COLLECTION_CHART_MAX_PAGES))
        body = {
            "InputCondMrktDivCode": "J",
            "InputIscd1": str(code),
            "InputHourClsCode": DB_AUCTION_PATH_HOUR_CLASS,
        }
        all_rows: list[dict[str, Any]] = []
        metadata: dict[str, str] = {}
        termination = "exhausted"
        truncated = False
        pages_fetched = 0
        failure_msg = ""
        prev_identity: tuple[Any, ...] | None = None
        stalls = 0
        in_observer = False
        cont_key = ""
        try:
            for page_index in range(max_pages):
                remaining = deadline_remaining(deadline)
                if remaining is not None and remaining <= 0:
                    termination = "deadline"
                    truncated = True
                    break
                started = now_seoul()
                call = self._post(session, _AUCTION_PATH, dict(body), cont_key=cont_key, scope="quote", code=str(code))
                if remaining is None:
                    data, resp_headers = await call
                else:
                    data, resp_headers = await asyncio.wait_for(call, timeout=remaining)
                received = now_seoul()
                cont_yn = str(resp_headers.get("cont_yn", "N") or "N")
                resp_key = str(resp_headers.get("cont_key", "") or "")
                metadata = {"cont_yn": cont_yn, "cont_key": resp_key}
                in_observer = True
                if on_page is not None:
                    on_page(dict(data), dict(metadata), started, received, page_index, 0)
                in_observer = False
                pages_fetched += 1
                if not isinstance(data, dict) or str(data.get("rsp_cd", "")) != "00000":
                    termination = "vendor_failure"
                    truncated = True
                    failure_msg = str(data.get("rsp_msg", "") if isinstance(data, dict) else "")
                    break
                rows = _rows_of(data)
                first = _row_key(rows[0]) if rows else ()
                last = _row_key(rows[-1]) if rows else ()
                identity = (resp_key, first, last)
                if identity == prev_identity:
                    stalls += 1
                else:
                    stalls = 0
                    prev_identity = identity
                    # A replayed page would inject phantom events: duplicate rows are real auction updates here.
                    all_rows.extend(rows)
                if stalls >= 2:
                    termination = "nonprogress"
                    truncated = True
                    break
                if any(str(r.get("Hour", "")) < floor for r in rows if str(r.get("Hour", "")).strip()):
                    termination = "crossed_time_floor"
                    break
                if cont_yn != "Y":
                    termination = "exhausted"
                    break
                cont_key = resp_key
            else:
                termination = "page_budget"
                truncated = True
        except TimeoutError:
            termination = "deadline"
            truncated = True
        except DbsecTokenThrottled:
            raise
        except Exception as e:
            if in_observer:
                raise
            logger.warning("DB auction path failed code=%s: %s", code, type(e).__name__)
            return {
                "rt_cd": "1",
                "msg1": type(e).__name__,
                "output2": [],
                "vendor": "dbsec",
                "truncated": True,
                "termination_reason": "vendor_failure",
                "pages_fetched": pages_fetched,
                "continuation": metadata,
            }
        if termination == "vendor_failure":
            return {
                "rt_cd": "1",
                "msg1": failure_msg,
                "output2": [],
                "vendor": "dbsec",
                "truncated": True,
                "termination_reason": termination,
                "pages_fetched": pages_fetched,
                "continuation": metadata,
            }
        kept = [dict(r) for r in all_rows if str(r.get("Hour", "")) >= floor]
        return {
            "rt_cd": "0",
            "output2": kept,
            "vendor": "dbsec",
            "truncated": truncated,
            "termination_reason": termination,
            "pages_fetched": pages_fetched,
            "continuation": metadata,
        }
