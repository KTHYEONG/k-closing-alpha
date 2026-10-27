"""Pure order domain contracts (no I/O, no settings, no clock)."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

import numpy as np

from src.api.auth_rejection import AUTH_REJECTION_CODES
from src.execution.cost_model import krx_tick_size
from src.execution.paper_broker import side_fee_krw

__all__ = [
    "BrokerOrder",
    "BrokerPosition",
    "EntryPick",
    "OrderIntent",
    "OrderPhase",
    "OrderSide",
    "OrderType",
    "PriceLimits",
    "SubmitResult",
    "SubmitVerdict",
    "classify_order_response",
    "entry_intent_id",
    "entry_limit_price",
    "exit_intent_id",
    "normalize_order_no",
    "size_entry_orders",
]

_SYMBOL_RE = re.compile(r"^[0-9A-Z]{6}$")


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    LIMIT = "limit"
    MARKET = "market"


class OrderPhase(StrEnum):
    ENTRY = "entry"
    EXIT = "exit"


class SubmitVerdict(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    THROTTLED = "throttled"
    AUTH_REJECTED = "auth_rejected"
    NOT_SENT = "not_sent"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class OrderIntent:
    """One order the strategy wants at the broker; identity is intent_id."""

    intent_id: str
    trading_date: str
    phase: OrderPhase
    symbol: str
    side: OrderSide
    order_type: OrderType
    qty: int
    limit_price: int | None
    decision_price: int | None
    parent_intent_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not _SYMBOL_RE.fullmatch(self.symbol):
            raise ValueError(f"symbol must be 6 digits or uppercase letters, got {self.symbol!r}")
        if type(self.qty) is not int or self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")
        try:
            parsed = datetime.strptime(self.trading_date, "%Y-%m-%d")
        except (TypeError, ValueError):
            raise ValueError(f"trading_date must be YYYY-MM-DD, got {self.trading_date!r}") from None
        if parsed.strftime("%Y-%m-%d") != self.trading_date:
            raise ValueError(f"trading_date must be YYYY-MM-DD, got {self.trading_date!r}")
        if not isinstance(self.order_type, OrderType) or not isinstance(self.phase, OrderPhase):
            raise ValueError("order_type and phase must be valid enums")
        if self.order_type is OrderType.LIMIT:
            if type(self.limit_price) is not int or self.limit_price <= 0:
                raise ValueError("LIMIT intent requires limit_price > 0")
        else:
            if self.limit_price is not None:
                raise ValueError("MARKET intent requires limit_price None")
        if self.phase is OrderPhase.ENTRY:
            if self.side is not OrderSide.BUY:
                raise ValueError("ENTRY intent requires side BUY")
            if self.parent_intent_id is not None:
                raise ValueError("ENTRY intent requires parent_intent_id None")
            if type(self.decision_price) is not int or self.decision_price <= 0:
                raise ValueError("ENTRY intent requires decision_price > 0")
        else:
            if self.side is not OrderSide.SELL:
                raise ValueError("EXIT intent requires side SELL")
            if not isinstance(self.parent_intent_id, str) or not self.parent_intent_id.strip():
                raise ValueError("EXIT intent requires non-empty parent_intent_id")


@dataclass(frozen=True)
class SubmitResult:
    """Outcome of one order POST attempt."""

    verdict: SubmitVerdict
    broker_order_no: str | None
    broker_org_no: str | None
    http_status: int | None
    msg_cd: str
    msg: str
    latency_ms: float


@dataclass(frozen=True)
class BrokerOrder:
    """One row of the broker's same-day order list."""

    order_no: str
    orig_order_no: str
    org_no: str
    symbol: str
    side: OrderSide
    order_qty: int
    order_price: int
    filled_qty: int
    filled_amount_krw: int
    remaining_qty: int
    cancelled: bool
    rejected_qty: int


@dataclass(frozen=True)
class BrokerPosition:
    """Broker-side holding of one symbol."""

    symbol: str
    holding_qty: int
    sellable_qty: int


@dataclass(frozen=True)
class PriceLimits:
    """Same-day KRX price band of one symbol."""

    symbol: str
    last: int
    upper: int
    lower: int


@dataclass(frozen=True)
class EntryPick:
    """Sizing input of one decision pick."""

    symbol: str
    allocation: float
    pred: float
    decision_price: int
    limit_price: int


def entry_intent_id(trading_date: str, symbol: str) -> str:
    """Return "<trading_date>:<symbol>:entry"."""
    return f"{trading_date}:{symbol}:entry"


def exit_intent_id(parent_intent_id: str, trading_date: str) -> str:
    """Return "<parent_intent_id>:exit:<trading_date>"."""
    return f"{parent_intent_id}:exit:{trading_date}"


def normalize_order_no(raw: object) -> str:
    """Return the broker order number without surrounding whitespace or leading zeros; "" when empty."""
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    return text.lstrip("0") or ""


def classify_order_response(http_status: int, body: object) -> SubmitVerdict:
    """Classify a decoded order POST response; never raises."""
    try:
        if not isinstance(body, dict):
            return SubmitVerdict.AMBIGUOUS
        msg_cd = str(body.get("msg_cd", ""))
        if msg_cd == "EGW00201":
            return SubmitVerdict.THROTTLED
        if msg_cd in AUTH_REJECTION_CODES["kis"]:
            return SubmitVerdict.AUTH_REJECTED
        if msg_cd == "EGW02006":
            return SubmitVerdict.REJECTED
        rt_cd = body.get("rt_cd")
        if rt_cd == "0":
            output = body.get("output")
            odno = output.get("ODNO") if isinstance(output, dict) else None
            if isinstance(odno, str) and odno.strip():
                return SubmitVerdict.ACCEPTED
            return SubmitVerdict.AMBIGUOUS
        if http_status == 200 and isinstance(rt_cd, str) and rt_cd != "" and rt_cd != "0":
            return SubmitVerdict.REJECTED
        return SubmitVerdict.AMBIGUOUS
    except Exception:
        return SubmitVerdict.AMBIGUOUS


def entry_limit_price(decision_price: int, buffer_bp: float, upper_limit_price: int) -> int:
    """Return the tick-aligned buy limit capped at the upper price band."""
    if type(decision_price) is not int or decision_price <= 0:
        raise ValueError(f"decision_price must be positive, got {decision_price}")
    if not math.isfinite(buffer_bp) or buffer_bp < 0:
        raise ValueError(f"buffer_bp must be finite non-negative, got {buffer_bp}")
    if type(upper_limit_price) is not int or upper_limit_price <= 0:
        raise ValueError(f"upper_limit_price must be positive, got {upper_limit_price}")
    if upper_limit_price < decision_price:
        raise ValueError("upper_limit_price must not be below decision_price")
    raw = Decimal(decision_price) * (1 + Decimal(str(buffer_bp)) / 10_000)
    if not math.isfinite(float(raw)):
        raise ValueError("buffered decision_price must be finite")
    tick = int(krx_tick_size(np.array([float(raw)], dtype=np.float64))[0])
    floored = math.floor(raw / tick) * tick
    result = min(floored, upper_limit_price)
    if result < decision_price:
        raise ValueError("decision_price is inconsistent with the tick grid")
    return result


def size_entry_orders(
    picks: Sequence[EntryPick], capital_krw: int, max_notional_per_order_krw: int
) -> dict[str, int]:
    """Return symbol -> integer buy quantity bounded by allocation, cap and cash."""
    if type(capital_krw) is not int or capital_krw < 0:
        raise ValueError(f"capital_krw must be non-negative, got {capital_krw}")
    if type(max_notional_per_order_krw) is not int or max_notional_per_order_krw <= 0:
        raise ValueError("max_notional_per_order_krw must be positive")
    seen: set[str] = set()
    for pick in picks:
        if pick.symbol in seen:
            raise ValueError(f"duplicate symbol {pick.symbol!r}")
        seen.add(pick.symbol)
        if not math.isfinite(pick.allocation) or pick.allocation <= 0:
            raise ValueError(f"allocation must be positive, got {pick.allocation}")
        if type(pick.limit_price) is not int or pick.limit_price <= 0:
            raise ValueError(f"limit_price must be positive, got {pick.limit_price}")

    def _rank_key(pick: EntryPick) -> tuple[int, float, str]:
        if math.isnan(pick.pred):
            return (1, 0.0, pick.symbol)
        return (0, -float(pick.pred), pick.symbol)

    ordered = sorted(picks, key=_rank_key)
    result: dict[str, int] = {}
    remaining = int(capital_krw)
    for pick in ordered:
        budget = min(math.floor(Decimal(capital_krw) * Decimal(str(pick.allocation))), max_notional_per_order_krw)
        hi = budget // int(pick.limit_price)
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            notional = mid * int(pick.limit_price)
            if notional + side_fee_krw(notional) <= remaining:
                lo = mid
            else:
                hi = mid - 1
        qty = lo
        result[pick.symbol] = qty
        notional = qty * int(pick.limit_price)
        remaining -= notional + side_fee_krw(notional)
    return result
