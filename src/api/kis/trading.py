"""KIS order/account transport bound to one trading environment."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field, replace
from datetime import date
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlsplit

import aiohttp

from src.api._transport import RetryPolicy, TransportResponse, send_with_auth_retry
from src.api.kis.client import KIS_REST_TPS_PER_APP_KEY, KisApiClient
from src.api.kis.key_pool import token_cache_path
from src.api.kis.rate_limit import HostPacedRateLimiter
from src.config.market_session import KRX_CLOSE_MARKET_DIV_CODE
from src.execution.order_contract import (
    BrokerOrder,
    BrokerPosition,
    OrderIntent,
    OrderSide,
    OrderType,
    PriceLimits,
    SubmitResult,
    SubmitVerdict,
    classify_order_response,
    normalize_order_no,
)
from src.utils.redaction import redact_secrets

if TYPE_CHECKING:
    from src.config import Settings

logger = logging.getLogger(__name__)

KIS_LIVE_HOST: str = "openapi.koreainvestment.com"
KIS_MOCK_HOST: str = "openapivts.koreainvestment.com"
TR_SUFFIX_BUY: str = "TTC0012U"
TR_SUFFIX_SELL: str = "TTC0011U"
TR_SUFFIX_REVISE_CANCEL: str = "TTC0013U"
TR_SUFFIX_BALANCE: str = "TTC8434R"
TR_SUFFIX_DAILY_ORDERS: str = "TTC0081R"
TR_SUFFIX_BUYING_POWER: str = "TTC8908R"
ORDER_EXCHANGE_CODE: str = "KRX"
MAX_CONTINUATION_PAGES: int = 10

_ALLOWED_SUFFIXES: frozenset[str] = frozenset(
    {
        TR_SUFFIX_BUY,
        TR_SUFFIX_SELL,
        TR_SUFFIX_REVISE_CANCEL,
        TR_SUFFIX_BALANCE,
        TR_SUFFIX_DAILY_ORDERS,
        TR_SUFFIX_BUYING_POWER,
    }
)
_AUTH_REJECTED_CODES: frozenset[str] = frozenset({"EGW00121", "EGW00123"})
_ACCOUNT_RE = re.compile(r"^\d{8}$")
_PRODUCT_RE = re.compile(r"^\d{2}$")


class TradingDisabledError(RuntimeError):
    """KIS_TRADING_ENV is "off"."""


class TradingConfigError(RuntimeError):
    """Trading environment, endpoint, credentials or confirmation are inconsistent."""


class BrokerReadError(RuntimeError):
    """An account read failed or returned data that cannot be trusted."""


@dataclass(frozen=True)
class TradingBinding:
    """Immutable binding of one trading environment to its endpoint, credentials and TR prefix."""

    env: Literal["mock", "live"]
    base_url: str
    app_key: str = field(repr=False)
    app_secret: str = field(repr=False)
    account_no: str = field(repr=False)
    product_code: str
    hts_id: str
    rest_tps: float

    def tr_id(self, suffix: str) -> str:
        """Return the environment-prefixed TR id ("V" + suffix for mock, "T" + suffix for live)."""
        if suffix not in _ALLOWED_SUFFIXES:
            raise ValueError(f"unknown TR suffix: {suffix}")
        return ("V" if self.env == "mock" else "T") + suffix


def resolve_trading_binding(cfg: Settings | None = None) -> TradingBinding:
    """Resolve and cross-validate the active trading binding; raises TradingDisabledError or TradingConfigError."""
    if cfg is None:
        from src.config import settings as live

        cfg = live
    env = cfg.KIS_TRADING_ENV
    if env == "off":
        raise TradingDisabledError("KIS_TRADING_ENV is off")
    if env == "mock":
        _validate_mock(cfg)
        base_url = cfg.KIS_MOCK_BASE_URL
        app_key, app_secret = cfg.KIS_MOCK_APP_KEY, cfg.KIS_MOCK_APP_SECRET
        account_no, product_code = cfg.KIS_MOCK_ACCOUNT_NO, cfg.KIS_MOCK_ACCOUNT_PRODUCT_CODE
        rest_tps = cfg.KIS_MOCK_REST_TPS
    elif env == "live":
        _validate_live(cfg)
        base_url = cfg.KIS_BASE_URL
        app_key, app_secret = cfg.KIS_APP_KEY, cfg.KIS_APP_SECRET
        account_no, product_code = cfg.KIS_ACCOUNT_ID, cfg.KIS_ACCOUNT_PRODUCT_CODE
        rest_tps = KIS_REST_TPS_PER_APP_KEY
    else:
        raise TradingConfigError("unknown trading env")
    return TradingBinding(env=env, base_url=base_url, app_key=app_key, app_secret=app_secret,
                          account_no=account_no, product_code=product_code, hts_id=cfg.KIS_HTS_ID, rest_tps=rest_tps)


def _validate_endpoint(base_url: str, host: str, env: str) -> None:
    try:
        parts = urlsplit(base_url)
        valid = parts.scheme == "https" and parts.hostname == host and parts.username is None and parts.password is None and not parts.query and not parts.fragment and parts.path in ("", "/")
        port = parts.port
    except ValueError:
        raise TradingConfigError(f"{env}: invalid endpoint") from None
    if not valid or port == 0:
        raise TradingConfigError(f"{env}: endpoint must use the https {env} host")


def _validate_mock(cfg: Settings) -> None:
    _validate_endpoint(cfg.KIS_MOCK_BASE_URL, KIS_MOCK_HOST, "mock")
    if not cfg.KIS_MOCK_APP_KEY.strip() or not cfg.KIS_MOCK_APP_SECRET.strip():
        raise TradingConfigError("mock: KIS_MOCK_APP_KEY and KIS_MOCK_APP_SECRET must be non-blank")
    if cfg.KIS_MOCK_APP_KEY == cfg.KIS_APP_KEY:
        raise TradingConfigError("mock: KIS_MOCK_APP_KEY must differ from the primary key")
    if cfg.KIS_ACCOUNT_ID.strip() and cfg.KIS_MOCK_ACCOUNT_NO == cfg.KIS_ACCOUNT_ID:
        raise TradingConfigError("mock: KIS_MOCK_ACCOUNT_NO must differ from the live account")
    if not _ACCOUNT_RE.fullmatch(cfg.KIS_MOCK_ACCOUNT_NO):
        raise TradingConfigError("mock: KIS_MOCK_ACCOUNT_NO must be 8 digits")
    if not _PRODUCT_RE.fullmatch(cfg.KIS_MOCK_ACCOUNT_PRODUCT_CODE):
        raise TradingConfigError("mock: KIS_MOCK_ACCOUNT_PRODUCT_CODE must be 2 digits")


def _validate_live(cfg: Settings) -> None:
    _validate_endpoint(cfg.KIS_BASE_URL, KIS_LIVE_HOST, "live")
    if not cfg.KIS_APP_KEY.strip() or not cfg.KIS_APP_SECRET.strip() or not cfg.KIS_ACCOUNT_ID.strip():
        raise TradingConfigError("live: KIS_APP_KEY, KIS_APP_SECRET and KIS_ACCOUNT_ID must be non-blank")
    if cfg.KIS_MOCK_APP_KEY.strip() and cfg.KIS_APP_KEY == cfg.KIS_MOCK_APP_KEY:
        raise TradingConfigError("live: KIS_APP_KEY must differ from the mock key")
    if not _ACCOUNT_RE.fullmatch(cfg.KIS_ACCOUNT_ID):
        raise TradingConfigError("live: KIS_ACCOUNT_ID must be 8 digits")
    if not _PRODUCT_RE.fullmatch(cfg.KIS_ACCOUNT_PRODUCT_CODE):
        raise TradingConfigError("live: KIS_ACCOUNT_PRODUCT_CODE must be 2 digits")
    expected_confirm = "LIVE-" + cfg.KIS_ACCOUNT_ID[-4:]
    if expected_confirm != cfg.KIS_LIVE_TRADING_CONFIRM:
        raise TradingConfigError("live: KIS_LIVE_TRADING_CONFIRM must match the account")


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _context_value(body: dict[str, Any], *names: str) -> str:
    for name in names:
        raw = body.get(name)
        if isinstance(raw, str):
            return str(raw).strip()
    raise BrokerReadError("continuation: required context is missing")


class KisTradingClient:
    def __init__(
        self, binding: TradingBinding, *, http_timeout_seconds: float, api_client: KisApiClient | None = None
    ) -> None:
        """Bind a trading client to one environment."""
        self.binding = binding
        self._timeout = float(http_timeout_seconds)
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise TradingConfigError(f"{binding.env}: HTTP timeout must be finite and positive")
        if api_client is None:
            from src.config import settings as live

            api_client = KisApiClient(
                app_key=binding.app_key,
                app_secret=binding.app_secret,
                account_id="",
                hts_id=binding.hts_id,
                base_url=binding.base_url,
                token_file=token_cache_path(binding.app_key, live.KIS_TOKEN_CACHE_DIR),
                rest_tps=binding.rest_tps,
            )
        if api_client.app_key != binding.app_key or api_client.app_secret != binding.app_secret or api_client.base_url.rstrip("/") != binding.base_url.rstrip("/"):
            raise TradingConfigError(f"{binding.env}: API client does not match the trading binding")
        api_client.account_id = ""
        self.api_client = api_client

    @property
    def rate_limiter(self) -> HostPacedRateLimiter:
        return self.api_client.rate_limiter

    @rate_limiter.setter
    def rate_limiter(self, value: HostPacedRateLimiter) -> None:
        self.api_client.rate_limiter = value

    def create_session(self) -> aiohttp.ClientSession:
        """Return a session whose total timeout is http_timeout_seconds."""
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))

    async def ensure_token(self, session: aiohttp.ClientSession, *, rejected_token: str | None = None) -> None:
        """Load or issue the binding's access token; raises on issuance failure."""
        if rejected_token is None:
            await self.api_client.ensure_token(session)
        else:
            await self.api_client.ensure_token(session, rejected_token=rejected_token)

    async def submit_order(self, session: aiohttp.ClientSession, intent: OrderIntent) -> SubmitResult:
        """Send exactly one order POST and classify it; never raises for transport or vendor errors."""
        suffix = TR_SUFFIX_BUY if intent.side is OrderSide.BUY else TR_SUFFIX_SELL
        body: dict[str, str] = {
            "CANO": self.binding.account_no,
            "ACNT_PRDT_CD": self.binding.product_code,
            "PDNO": intent.symbol,
            "ORD_DVSN": "00" if intent.order_type is OrderType.LIMIT else "01",
            "ORD_QTY": str(intent.qty),
            "ORD_UNPR": str(intent.limit_price) if intent.order_type is OrderType.LIMIT else "0",
            "EXCG_ID_DVSN_CD": ORDER_EXCHANGE_CODE,
        }
        if intent.side is OrderSide.SELL:
            body["SLL_TYPE"] = "01"
        return await self._send_order(
            session,
            path="/uapi/domestic-stock/v1/trading/order-cash",
            tr_id=self.binding.tr_id(suffix),
            body=body,
            intent_id=intent.intent_id,
        )

    async def cancel_order(self, session: aiohttp.ClientSession, *, org_no: str, order_no: str) -> SubmitResult:
        """Send exactly one full-quantity cancel POST and classify it; never raises for transport or vendor errors."""
        body: dict[str, str] = {
            "CANO": self.binding.account_no,
            "ACNT_PRDT_CD": self.binding.product_code,
            "KRX_FWDG_ORD_ORGNO": org_no,
            "ORGN_ODNO": order_no,
            "ORD_DVSN": "00",
            "RVSE_CNCL_DVSN_CD": "02",
            "ORD_QTY": "0",
            "ORD_UNPR": "0",
            "QTY_ALL_ORD_YN": "Y",
            "EXCG_ID_DVSN_CD": ORDER_EXCHANGE_CODE,
        }
        return await self._send_order(
            session,
            path="/uapi/domestic-stock/v1/trading/order-rvsecncl",
            tr_id=self.binding.tr_id(TR_SUFFIX_REVISE_CANCEL),
            body=body,
            intent_id=f"cancel:{order_no}",
        )

    async def _send_order(
        self, session: aiohttp.ClientSession, *, path: str, tr_id: str, body: dict[str, str], intent_id: str
    ) -> SubmitResult:
        url = f"{self.binding.base_url.rstrip('/')}{path}"
        await self.rate_limiter.acquire()
        token = self.api_client.token
        sent_token = token if isinstance(token, str) else ""
        started = monotonic()
        status: int | None = None
        try:
            async with session.post(url, headers=self.api_client._get_headers(tr_id), json=body) as resp:
                status = int(resp.status)
                try:
                    payload: Any = await resp.json()
                except Exception:
                    latency_ms = (monotonic() - started) * 1000.0
                    return self._finish_send(intent_id, tr_id, SubmitVerdict.AMBIGUOUS, None, None, "", "", status, latency_ms)
        except aiohttp.ClientConnectorError:
            latency_ms = (monotonic() - started) * 1000.0
            return self._finish_send(intent_id, tr_id, SubmitVerdict.NOT_SENT, None, None, "", "", None, latency_ms)
        except (aiohttp.ClientError, TimeoutError):
            latency_ms = (monotonic() - started) * 1000.0
            return self._finish_send(intent_id, tr_id, SubmitVerdict.AMBIGUOUS, None, None, "", "", status, latency_ms)
        latency_ms = (monotonic() - started) * 1000.0
        verdict = classify_order_response(status, payload) if isinstance(payload, dict) and status is not None else SubmitVerdict.AMBIGUOUS
        msg_cd = str(payload.get("msg_cd", "")) if isinstance(payload, dict) else ""
        raw_msg = str(payload.get("msg1", "")) if isinstance(payload, dict) else ""
        if verdict is SubmitVerdict.AUTH_REJECTED:
            try:
                await self.ensure_token(session, rejected_token=sent_token)
            except Exception:
                logger.warning("[EXEC] stage=kis_order status=TOKEN_REFRESH_FAILED env=%s tr_id=%s", self.binding.env, tr_id)
        order_no: str | None = None
        org_no: str | None = None
        if verdict is SubmitVerdict.ACCEPTED and isinstance(payload, dict):
            output = payload.get("output")
            if isinstance(output, dict):
                order_no = normalize_order_no(output.get("ODNO")) or None
                raw_org = output.get("KRX_FWDG_ORD_ORGNO")
                org_no = str(raw_org).strip() or None if raw_org is not None else None
        secrets = (self.binding.app_key, self.binding.app_secret, self.binding.account_no, sent_token)
        return self._finish_send(intent_id, tr_id, verdict, order_no, org_no, redact_secrets(msg_cd, secrets), redact_secrets(raw_msg, secrets), status, latency_ms)

    def _finish_send(
        self,
        intent_id: str,
        tr_id: str,
        verdict: SubmitVerdict,
        order_no: str | None,
        org_no: str | None,
        msg_cd: str,
        msg: str,
        status: int | None,
        latency_ms: float,
    ) -> SubmitResult:
        logger.info(
            "[EXEC] stage=kis_order env=%s intent_id=%s tr_id=%s verdict=%s http=%s msg_cd=%s latency_ms=%.3f",
            self.binding.env,
            intent_id,
            tr_id,
            verdict.value,
            status if status is not None else "",
            msg_cd,
            latency_ms,
        )
        return SubmitResult(
            verdict=verdict,
            broker_order_no=order_no,
            broker_org_no=org_no,
            http_status=status,
            msg_cd=msg_cd,
            msg=msg,
            latency_ms=latency_ms,
        )

    async def fetch_daily_orders(self, session: aiohttp.ClientSession, trading_date: str) -> list[BrokerOrder]:
        """Return every broker order of trading_date ("YYYY-MM-DD"); raises BrokerReadError."""
        try:
            parsed_date = date.fromisoformat(trading_date)
            if parsed_date.isoformat() != trading_date:
                raise ValueError("non-canonical date")
        except (TypeError, ValueError):
            raise BrokerReadError("daily orders: trading_date must be YYYY-MM-DD") from None
        ymd = parsed_date.strftime("%Y%m%d")
        params: dict[str, str] = {
            "CANO": self.binding.account_no,
            "ACNT_PRDT_CD": self.binding.product_code,
            "INQR_STRT_DT": ymd,
            "INQR_END_DT": ymd,
            "SLL_BUY_DVSN_CD": "00",
            "INQR_DVSN": "00",
            "PDNO": "",
            "CCLD_DVSN": "00",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "INQR_DVSN_3": "00",
            "INQR_DVSN_1": "",
            "EXCG_ID_DVSN_CD": ORDER_EXCHANGE_CODE,
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        collected: list[BrokerOrder] = []
        headers: dict[str, str] = {}
        page_index = 0
        while True:
            response = await self._get_page(
                session, path="/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
                tr_id=self.binding.tr_id(TR_SUFFIX_DAILY_ORDERS), params=params, extra_headers=headers,
            )
            body = self._checked_body(response, "daily orders")
            rows = body.get("output1")
            if rows is None:
                raise BrokerReadError("daily orders: output1 is missing")
            if not isinstance(rows, list):
                raise BrokerReadError("daily orders: output1 is not a list")
            collected.extend(self._parse_order_row(row) for row in rows)
            if next((v for k, v in response.headers.items() if k.lower() == "tr_cont"), "") not in ("F", "M"):
                return _fold_cancel_rows(collected)
            page_index += 1
            if page_index >= MAX_CONTINUATION_PAGES:
                raise BrokerReadError("daily orders: continuation exceeded the page budget")
            params["CTX_AREA_FK100"] = _context_value(body, "ctx_area_fk100", "CTX_AREA_FK100")
            params["CTX_AREA_NK100"] = _context_value(body, "ctx_area_nk100", "CTX_AREA_NK100")
            headers["tr_cont"] = "N"

    async def fetch_positions(self, session: aiohttp.ClientSession) -> list[BrokerPosition]:
        """Return holdings with holding_qty > 0; raises BrokerReadError."""
        params: dict[str, str] = {
            "CANO": self.binding.account_no,
            "ACNT_PRDT_CD": self.binding.product_code,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }
        collected: list[BrokerPosition] = []
        headers: dict[str, str] = {}
        page_index = 0
        while True:
            response = await self._get_page(
                session, path="/uapi/domestic-stock/v1/trading/inquire-balance",
                tr_id=self.binding.tr_id(TR_SUFFIX_BALANCE), params=params, extra_headers=headers,
            )
            body = self._checked_body(response, "balance")
            rows = body.get("output1")
            if rows is None:
                raise BrokerReadError("balance: output1 is missing")
            if not isinstance(rows, list):
                raise BrokerReadError("balance: output1 is not a list")
            for row in rows:
                position = self._parse_position_row(row)
                if position is None:
                    continue
                collected.append(position)
            if next((v for k, v in response.headers.items() if k.lower() == "tr_cont"), "") not in ("F", "M"):
                return collected
            page_index += 1
            if page_index >= MAX_CONTINUATION_PAGES:
                raise BrokerReadError("balance: continuation exceeded the page budget")
            params["CTX_AREA_FK100"] = _context_value(body, "ctx_area_fk100", "CTX_AREA_FK100")
            params["CTX_AREA_NK100"] = _context_value(body, "ctx_area_nk100", "CTX_AREA_NK100")
            headers["tr_cont"] = "N"

    async def fetch_orderable_cash(self, session: aiohttp.ClientSession, *, symbol: str, limit_price: int) -> int:
        """Return leverage-free orderable cash in KRW; raises BrokerReadError."""
        params: dict[str, str] = {
            "CANO": self.binding.account_no,
            "ACNT_PRDT_CD": self.binding.product_code,
            "PDNO": symbol,
            "ORD_UNPR": str(limit_price),
            "ORD_DVSN": "00",
            "CMA_EVLU_AMT_ICLD_YN": "N",
            "OVRS_ICLD_YN": "N",
        }
        response = await self._get_page(
            session, path="/uapi/domestic-stock/v1/trading/inquire-psbl-order",
            tr_id=self.binding.tr_id(TR_SUFFIX_BUYING_POWER), params=params, extra_headers={},
        )
        body = self._checked_body(response, "buying power")
        output = body.get("output")
        if not isinstance(output, dict):
            raise BrokerReadError("buying power: output is missing")
        candidates: list[int] = []
        for key in ("ord_psbl_cash", "nrcv_buy_amt"):
            parsed = _parse_int(output.get(key))
            if parsed is not None and parsed >= 0:
                candidates.append(parsed)
        if not candidates:
            raise BrokerReadError("buying power: no parseable cash figure")
        return min(candidates)

    async def fetch_price_limits(self, session: aiohttp.ClientSession, symbol: str) -> PriceLimits:
        """Return the symbol's last price and same-day price band; raises BrokerReadError."""
        try:
            res = await self.api_client.get_current_price(
                session, symbol, market_div_code=KRX_CLOSE_MARKET_DIV_CODE, allow_market_div_fallback=False
            )
        except Exception as exc:
            raise BrokerReadError("price limits: quote request failed") from exc
        if not isinstance(res, dict) or res.get("rt_cd") != "0":
            raise BrokerReadError("price limits: quote was rejected")
        output = res.get("output")
        if not isinstance(output, dict):
            raise BrokerReadError("price limits: output is missing")
        last = _parse_int(output.get("stck_prpr"))
        upper = _parse_int(output.get("stck_mxpr"))
        lower = _parse_int(output.get("stck_llam"))
        if last is None or upper is None or lower is None or last <= 0 or upper <= 0 or lower <= 0:
            raise BrokerReadError("price limits: quote fields are not positive integers")
        if not lower <= last <= upper:
            raise BrokerReadError("price limits: last price is outside the band")
        return PriceLimits(symbol=symbol, last=last, upper=upper, lower=lower)

    @staticmethod
    def _parse_order_row(row: Any) -> BrokerOrder:
        if not isinstance(row, dict):
            raise BrokerReadError("daily orders: row is not an object")
        side_code = str(row.get("sll_buy_dvsn_cd", "")).strip()
        if side_code == "02":
            side = OrderSide.BUY
        elif side_code == "01":
            side = OrderSide.SELL
        else:
            raise BrokerReadError("daily orders: unknown side code")
        order_qty = _parse_int(row.get("ord_qty"))
        order_price = _parse_int(row.get("ord_unpr"))
        filled_qty = _parse_int(row.get("tot_ccld_qty"))
        filled_amount = _parse_int(row.get("tot_ccld_amt"))
        remaining = _parse_int(row.get("rmn_qty"))
        rejected = _parse_int(row.get("rjct_qty"))
        if order_qty is None or order_price is None or filled_qty is None:
            raise BrokerReadError("daily orders: quantity fields are unreadable")
        if filled_amount is None or remaining is None or rejected is None:
            raise BrokerReadError("daily orders: quantity fields are unreadable")
        if min(order_qty, order_price, filled_qty, filled_amount, remaining, rejected) < 0:
            raise BrokerReadError("daily orders: negative quantity or amount")
        order_no = normalize_order_no(row.get("odno"))
        orig_order_no = normalize_order_no(row.get("orgn_odno"))
        org_no = str(row.get("ord_gno_brno") or "").strip()
        symbol = str(row.get("pdno") or "").strip()
        if not order_no or not org_no or not symbol:
            raise BrokerReadError("daily orders: identifiers are missing")
        if row.get("cncl_yn") not in ("Y", "N"):
            raise BrokerReadError("daily orders: cancellation status is unreadable")
        return BrokerOrder(
            order_no=order_no,
            orig_order_no=orig_order_no,
            org_no=org_no,
            symbol=symbol,
            side=side,
            order_qty=order_qty,
            order_price=order_price,
            filled_qty=filled_qty,
            filled_amount_krw=filled_amount,
            remaining_qty=remaining,
            cancelled=str(row.get("cncl_yn", "")).strip() == "Y",
            rejected_qty=rejected,
        )

    @staticmethod
    def _parse_position_row(row: Any) -> BrokerPosition | None:
        if not isinstance(row, dict):
            raise BrokerReadError("balance: row is not an object")
        symbol = str(row.get("pdno") or "").strip()
        holding = _parse_int(row.get("hldg_qty"))
        sellable = _parse_int(row.get("ord_psbl_qty"))
        if not symbol or holding is None or sellable is None or holding < 0 or sellable < 0:
            raise BrokerReadError("balance: row fields are unreadable")
        if holding == 0:
            return None
        return BrokerPosition(symbol=symbol, holding_qty=holding, sellable_qty=sellable)

    async def _get_page(
        self,
        session: aiohttp.ClientSession,
        *,
        path: str,
        tr_id: str,
        params: dict[str, str],
        extra_headers: dict[str, str],
    ) -> TransportResponse:
        from src.config import settings as live

        url = f"{self.binding.base_url.rstrip('/')}{path}"
        rest_tps = self.binding.rest_tps

        def _rate_limited(_attempt: int, response: TransportResponse) -> float | None:
            body = response.body
            if isinstance(body, dict) and str(body.get("msg_cd", "")) == "EGW00201":
                return 1.0 / rest_tps
            return None

        def _current_token() -> str:
            token = self.api_client.token
            return token if isinstance(token, str) else ""

        def _is_auth_rejected(response: TransportResponse) -> bool:
            body = response.body
            return isinstance(body, dict) and str(body.get("msg_cd", "")) in _AUTH_REJECTED_CODES

        async def _refresh(sent: str) -> None:
            await self.ensure_token(session, rejected_token=sent)

        def _open(token: str) -> Any:
            headers = self.api_client._get_headers(tr_id)
            headers["authorization"] = f"Bearer {token}"
            headers.update(extra_headers)
            return session.get(url, headers=headers, params=dict(params))

        try:
            return await send_with_auth_retry(
                _open,
                acquire=self.rate_limiter.acquire,
                current_token=_current_token,
                is_auth_rejected=_is_auth_rejected,
                refresh=_refresh,
                policy=RetryPolicy(max_attempts=live.ORDER_SEND_MAX_ATTEMPTS, rate_limit_wait=_rate_limited),
                log_stage="kis_order",
                log_context=f"env={self.binding.env} tr_id={tr_id}",
            )
        except Exception as exc:
            raise BrokerReadError("account read failed on transport") from exc

    @staticmethod
    def _checked_body(response: TransportResponse, stage: str) -> dict[str, Any]:
        body = response.body
        if not isinstance(body, dict):
            raise BrokerReadError(f"{stage}: body is not an object")
        if str(body.get("msg_cd", "")) == "EGW00201":
            raise BrokerReadError(f"{stage}: rate limit was not absorbed")
        if body.get("rt_cd") != "0":
            raise BrokerReadError(f"{stage}: vendor rejected the read")
        return body


def _fold_cancel_rows(rows: list[BrokerOrder]) -> list[BrokerOrder]:
    """Fold cancel child rows into their parents; the broker reports a cancel as a separate row (cncl_yn=Y, orgn_odno=parent)."""
    def is_cancel_child(r: BrokerOrder) -> bool:
        return r.cancelled and bool(r.orig_order_no) and r.orig_order_no != r.order_no

    cancelled_parents = {r.orig_order_no for r in rows if is_cancel_child(r)}
    return [
        replace(r, cancelled=True) if r.order_no in cancelled_parents else r
        for r in rows
        if not is_cancel_child(r)
    ]
