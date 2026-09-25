"""Paper-trading execution ledger and fill oracle (no real order transmission).

실주문 전송 TR을 어떤 형태로도 참조하지 않는다. 체결 판정은 관측된 실체결
프린트만을 오라클로 사용하며, 체결가에 수수료/세금/스프레드를 가산하지
않는다(거래비용은 src/execution/cost_model.py가 별도로 부과한다).
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet
from src.execution.cost_model import BROKERAGE_FEE_BP, statutory_bp_asof

logger = logging.getLogger(__name__)

ORDER_STATUS_FILLED: str = "FILLED"
ORDER_STATUS_UNCONFIRMED: str = "UNCONFIRMED"
ORDER_STATUS_NO_SNAPSHOT_ROW: str = "NO_SNAPSHOT_ROW"
ORDER_STATUS_ZERO_QTY: str = "ZERO_QTY"
ORDER_STATUS_UNFILLED: str = "UNFILLED"
ORDER_STATUS_MISSED_AUCTION: str = "MISSED_AUCTION"
ORDER_STATUS_INSUFFICIENT_CASH: str = "INSUFFICIENT_CASH"
ORDER_STATUSES: tuple[str, ...] = (
    ORDER_STATUS_FILLED,
    ORDER_STATUS_UNCONFIRMED,
    ORDER_STATUS_NO_SNAPSHOT_ROW,
    ORDER_STATUS_ZERO_QTY,
    ORDER_STATUS_UNFILLED,
    ORDER_STATUS_MISSED_AUCTION,
    ORDER_STATUS_INSUFFICIENT_CASH,
)

# 왕복 수수료를 매수/매도 편도로 나눈다(체결가에는 스프레드가 이미 반영돼 명시비용만 부과)
PAPER_BROKERAGE_SIDE_BP: float = BROKERAGE_FEE_BP / 2.0

ROUND_TRIP_COLUMNS: tuple[str, ...] = (
    "entry_order_id",
    "exit_order_id",
    "symbol",
    "decision_date",
    "entry_filled_at",
    "exit_filled_at",
    "qty",
    "entry_price",
    "exit_price",
    "exit_trigger",
    "gross_pnl",
    "buy_fee",
    "sell_fee",
    "sell_tax",
    "cost",
    "net_pnl",
    "gross_ret",
    "net_ret",
)

NAV_COLUMNS: tuple[str, ...] = (
    "as_of_date",
    "seed_capital",
    "cash",
    "open_cost_basis",
    "open_buy_fees",
    "realized_net_pnl",
    "cumulative_cost",
    "nav",
    "n_open_positions",
    "n_closed_trades",
    "recorded_at",
)

LEDGER_KINDS: tuple[str, ...] = ("orders", "fills", "decisions", "trades", "nav", "corrections")

CORRECTION_ACTION_VOID_FILL: str = "VOID_FILL"
CORRECTION_ACTION_NOTE: str = "NOTE"
CORRECTION_ACTIONS: tuple[str, ...] = (CORRECTION_ACTION_VOID_FILL, CORRECTION_ACTION_NOTE)
CORRECTION_COLUMNS: tuple[str, ...] = (
    "correction_id",
    "action",
    "target_order_ids",
    "reason",
    "evidence",
    "operator",
    "as_of_date",
    "recorded_at",
)

PAPER_LEDGER_LOCK_FILENAME: str = ".ledger.lock"
PAPER_LEDGER_LOCK_TIMEOUT_SECONDS: float = 300.0

_LEDGER_DEDUP_KEYS: dict[str, list[str]] = {
    "orders": ["order_id"],
    "fills": ["order_id"],
    "decisions": ["decision_date", "symbol"],
    "trades": ["exit_order_id"],
    "nav": ["as_of_date"],
    "corrections": ["correction_id"],
}

# fills와 corrections는 append-only 증거물이다. 키 충돌은 갱신이 아니라 계약 위반이다.
_APPEND_ONLY_KINDS: frozenset[str] = frozenset({"fills", "corrections"})


@dataclass(frozen=True)
class PaperOrder:
    order_id: str
    decision_date: str
    symbol: str
    side: str
    qty: int
    limit_price: int | None
    placed_at: pd.Timestamp
    reason: str
    entry_order_id: str | None = None


@dataclass(frozen=True)
class PaperFill:
    order_id: str
    symbol: str
    side: str
    qty: int
    fill_price: int
    filled_at: pd.Timestamp
    trigger: str


def size_order_qty(seed_capital: int, allocation: float, price: int) -> int:
    """정수 주식수 = floor(seed_capital * allocation / price).

    KRX는 소수점 주식이 없으므로 내림한다. 산출이 0주가 되면 0을 반환하고
    호출부가 미체결로 기록한다.
    """
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    if allocation <= 0:
        raise ValueError(f"allocation must be positive, got {allocation}")
    return math.floor(seed_capital * allocation / price)


def investable_capital(cash: int, seed_capital: int) -> int:
    """Compute capital available for entry sizing from available cash.

    Args:
        cash: Available cash in KRW from the latest NAV snapshot.
        seed_capital: Strategy seed capital in KRW used as a sizing cap.

    Returns:
        Investable capital in KRW reserved for buy fees and capped at seed.
    """
    # 매수 수수료까지 현금 안에서 치르도록 편도 수수료분을 남기고, 누적 수익이 있어도 사이징은 시드 기준으로 고정한다
    return max(0, min(int(seed_capital), math.floor(int(cash) / (1.0 + PAPER_BROKERAGE_SIDE_BP / 10_000))))


def sizing_price(decision_price: int, buffer_bp: float) -> int:
    """Price used to fix an entry quantity before the closing auction.

    A live market-on-close order must carry its quantity before 15:30, so the
    only admissible price is the one observed at decision time, optionally
    inflated by a configured buffer so that a higher close still settles in cash.

    Args:
        decision_price: Decision-time snapshot close in KRW (> 0).
        buffer_bp: Non-negative buffer in basis points.

    Returns:
        ceil(decision_price * (1 + buffer_bp / 10_000)) in KRW.

    Raises:
        ValueError: decision_price <= 0 or buffer_bp < 0.
    """
    if decision_price <= 0:
        raise ValueError(f"decision_price must be positive, got {decision_price}")
    if buffer_bp < 0:
        raise ValueError(f"buffer_bp must be non-negative, got {buffer_bp}")
    return math.ceil(decision_price * (1.0 + buffer_bp / 10_000))


def decide_fill(order: PaperOrder, print_price: int, print_ts: pd.Timestamp) -> PaperFill | None:
    """실체결 프린트만을 오라클로 체결 여부를 판정한다.

    체결가는 관측된 원시 프린트가 그대로이며 수수료/스프레드를 절대 가산하지
    않는다. 주문 이전 시각의 프린트로는 체결을 선언할 수 없다(룩어헤드 금지).
    """
    if print_ts < order.placed_at:
        raise ValueError(f"print_ts {print_ts} precedes placed_at {order.placed_at} (lookahead forbidden)")
    trigger = "market" if order.limit_price is None else "limit"
    if order.limit_price is None:
        return PaperFill(order.order_id, order.symbol, order.side, order.qty, print_price, print_ts, trigger)
    if order.side == "buy":
        if print_price <= order.limit_price:
            return PaperFill(order.order_id, order.symbol, order.side, order.qty, print_price, print_ts, trigger)
        return None
    if print_price >= order.limit_price:
        return PaperFill(order.order_id, order.symbol, order.side, order.qty, print_price, print_ts, trigger)
    return None


def build_auction_fill(order: PaperOrder, decision_row: dict[str, Any]) -> PaperFill | None:
    """동시호가(단일가) 진입 체결 판정. 체결가는 관측된 확정 종가 그대로이다.

    종가_확정이 True가 아니면 None을 반환한다(미확정 종목 진입 보류).
    확정시각이 주문시각보다 이르면 룩어헤드 금지 위반으로 ValueError를 던진다.
    """
    # fetch_archive_snapshot은 legacy NaN 혼재로 종가_확정을 float64로 반환한다
    # (True/False -> 1.0/0.0). `not float('nan')`은 False라 미확정(NaN)이 확정으로
    # 오판되는 함정이 있어, "정확히 참"만 확정으로 인정한다(경제적 의미: 확정여부
    # 불명은 미확정과 동일하게 취급).
    confirmed = decision_row.get("종가_확정")
    if not (pd.notna(confirmed) and bool(confirmed)):
        return None
    close = decision_row["종가"]
    if close <= 0:
        raise ValueError(f"종가 must be positive, got {close}")
    execution_timestamp = decision_row["execution_timestamp"]
    if execution_timestamp < order.placed_at:
        raise ValueError(
            f"execution_timestamp {execution_timestamp} precedes placed_at {order.placed_at} (lookahead forbidden)"
        )
    return PaperFill(
        order.order_id, order.symbol, order.side, order.qty, close, execution_timestamp, "auction_close"
    )


def build_open_auction_fill(order: PaperOrder, open_price: int, observed_at: pd.Timestamp) -> PaperFill | None:
    """KRX 시가단일가 시장가 매도 체결 판정. 체결가는 관측된 시가 그대로이다.

    Args:
        order: 시가단일가 시장가 매도 주문.
        open_price: KRX 시가단일가 체결가.
        observed_at: 시가를 관측한 시각.

    Returns:
        시가가 형성됐으면 PaperFill, 미형성(0 이하)이면 None.

    Raises:
        ValueError: 시가 관측이 주문시각보다 이르면(룩어헤드 금지 위반).
    """
    if observed_at < order.placed_at:
        raise ValueError(f"observed_at {observed_at} precedes placed_at {order.placed_at} (lookahead forbidden)")
    if int(open_price) <= 0:
        return None
    return PaperFill(
        order.order_id, order.symbol, order.side, order.qty, int(open_price), order.placed_at, "auction_open"
    )


def order_record(order: PaperOrder, status: str) -> dict[str, Any]:
    """Build a terminal order ledger row for a paper order.

    Args:
        order: Paper order to record.
        status: One of ORDER_STATUSES.

    Returns:
        Ledger row dict with recorded_at timestamp.

    Raises:
        ValueError: When status is not a known order status.
    """
    if status not in ORDER_STATUSES:
        raise ValueError(f"unknown order status {status!r}")
    return {
        "order_id": order.order_id,
        "decision_date": order.decision_date,
        "symbol": order.symbol,
        "side": order.side,
        "qty": order.qty,
        "limit_price": order.limit_price,
        "placed_at": order.placed_at,
        "reason": order.reason,
        "status": status,
        "entry_order_id": order.entry_order_id,
        "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
    }


def build_round_trips(fills: pd.DataFrame) -> pd.DataFrame:
    """Pair buy/sell fills into entry-linked round trips with explicit costs.

    Args:
        fills: Fill ledger rows in record order.

    Returns:
        Round-trip frame with ROUND_TRIP_COLUMNS schema.

    Raises:
        ValueError: On sell without entry_order_id, unknown or closed entry,
            symbol mismatch, qty mismatch, missing sell filled_at,
            or unknown fill side.
    """
    if fills is None or fills.empty:
        return pd.DataFrame(columns=list(ROUND_TRIP_COLUMNS))
    records = fills.to_dict("records")
    open_buys: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for rec in records:
        side = rec.get("side")
        symbol = str(rec.get("symbol"))
        if side == "buy":
            open_buys[str(rec["order_id"])] = rec
        elif side == "sell":
            link = rec.get("entry_order_id")
            if link is None or pd.isna(link):
                raise ValueError(f"sell fill {rec.get('order_id')} has no entry_order_id")
            buy = open_buys.pop(str(link), None)
            if buy is None:
                raise ValueError(f"sell fill {rec.get('order_id')} references unknown or closed entry {link}")
            if str(buy["symbol"]) != symbol:
                raise ValueError(f"sell fill {rec.get('order_id')} symbol {symbol} != entry symbol {buy['symbol']}")
            qty = int(rec["qty"])
            buy_qty = int(buy["qty"])
            if qty != buy_qty:
                raise ValueError(f"sell qty {qty} != buy qty {buy_qty} for {symbol}")
            sell_filled_at = rec.get("filled_at")
            if sell_filled_at is None or pd.isna(sell_filled_at):
                raise ValueError(f"sell fill {rec.get('order_id')} has no filled_at")
            entry_price = int(buy["fill_price"])
            exit_price = int(rec["fill_price"])
            entry_notional = entry_price * qty
            exit_notional = exit_price * qty
            buy_fee = math.floor(entry_notional * PAPER_BROKERAGE_SIDE_BP / 10_000)
            sell_fee = math.floor(exit_notional * PAPER_BROKERAGE_SIDE_BP / 10_000)
            tax_bp = float(
                statutory_bp_asof(
                    np.array([np.datetime64(pd.Timestamp(sell_filled_at).strftime("%Y-%m-%d"))])
                )[0]
            )
            sell_tax = math.floor(exit_notional * tax_bp / 10_000)
            cost = buy_fee + sell_fee + sell_tax
            gross_pnl = exit_notional - entry_notional
            net_pnl = gross_pnl - cost
            rows.append(
                {
                    "entry_order_id": buy.get("order_id"),
                    "exit_order_id": rec.get("order_id"),
                    "symbol": symbol,
                    "decision_date": str(buy.get("decision_date")),
                    "entry_filled_at": buy.get("filled_at"),
                    "exit_filled_at": rec.get("filled_at"),
                    "qty": qty,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_trigger": rec.get("trigger"),
                    "gross_pnl": gross_pnl,
                    "buy_fee": buy_fee,
                    "sell_fee": sell_fee,
                    "sell_tax": sell_tax,
                    "cost": cost,
                    "net_pnl": net_pnl,
                    "gross_ret": exit_price / entry_price - 1.0,
                    "net_ret": net_pnl / entry_notional,
                }
            )
        else:
            raise ValueError(f"unknown fill side {side!r}")
    if not rows:
        return pd.DataFrame(columns=list(ROUND_TRIP_COLUMNS))
    return pd.DataFrame(rows, columns=list(ROUND_TRIP_COLUMNS))


def build_nav_snapshot(fills: pd.DataFrame, seed_capital: int, as_of_date: str) -> pd.DataFrame:
    """Build a single-row NAV snapshot from the fill ledger.

    Args:
        fills: Fill ledger rows.
        seed_capital: Starting capital in KRW.
        as_of_date: Snapshot date string.

    Returns:
        One-row frame with NAV_COLUMNS schema.
    """
    if fills is None or fills.empty:
        return pd.DataFrame(
            [
                {
                    "as_of_date": as_of_date,
                    "seed_capital": int(seed_capital),
                    "cash": int(seed_capital),
                    "open_cost_basis": 0,
                    "open_buy_fees": 0,
                    "realized_net_pnl": 0,
                    "cumulative_cost": 0,
                    "nav": int(seed_capital),
                    "n_open_positions": 0,
                    "n_closed_trades": 0,
                    "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
                }
            ],
            columns=list(NAV_COLUMNS),
        )
    trips = build_round_trips(fills)
    records = fills.to_dict("records")
    buy_outflow = 0
    total_buy_fees = 0
    for rec in records:
        if str(rec.get("side")) == "buy":
            notional = int(rec.get("fill_price")) * int(rec.get("qty"))
            fee = math.floor(notional * PAPER_BROKERAGE_SIDE_BP / 10_000)
            buy_outflow += notional + fee
            total_buy_fees += fee
    closed_ids = set(trips["entry_order_id"].tolist()) if not trips.empty else set()
    open_cost_basis = 0
    open_buy_fees = 0
    n_open = 0
    for rec in records:
        if str(rec.get("side")) == "buy" and rec.get("order_id") not in closed_ids:
            notional = int(rec.get("fill_price")) * int(rec.get("qty"))
            fee = math.floor(notional * PAPER_BROKERAGE_SIDE_BP / 10_000)
            open_cost_basis += notional
            open_buy_fees += fee
            n_open += 1
    exit_inflow = 0
    realized_net_pnl = 0
    sell_costs = 0
    if not trips.empty:
        for _, t in trips.iterrows():
            exit_inflow += int(t["exit_price"]) * int(t["qty"]) - int(t["sell_fee"]) - int(t["sell_tax"])
            sell_costs += int(t["sell_fee"]) + int(t["sell_tax"])
        realized_net_pnl = int(trips["net_pnl"].sum())
    cash = int(seed_capital) - buy_outflow + exit_inflow
    cumulative_cost = total_buy_fees + sell_costs
    nav = cash + open_cost_basis
    return pd.DataFrame(
        [
            {
                "as_of_date": as_of_date,
                "seed_capital": int(seed_capital),
                "cash": int(cash),
                "open_cost_basis": int(open_cost_basis),
                "open_buy_fees": int(open_buy_fees),
                "realized_net_pnl": int(realized_net_pnl),
                "cumulative_cost": int(cumulative_cost),
                "nav": int(nav),
                "n_open_positions": int(n_open),
                "n_closed_trades": len(trips),
                "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
            }
        ],
        columns=list(NAV_COLUMNS),
    )


def refresh_trade_ledgers(ledger: PaperLedger, seed_capital: int, as_of_date: str) -> int:
    """Refresh derived trade and NAV ledgers from effective fills.

    Args:
        ledger: Paper ledger to read fills from and write to.
        seed_capital: Starting capital in KRW.
        as_of_date: Snapshot date string.

    Returns:
        Number of closed round trips.
    """
    fills = ledger.load_effective_fills()
    trips = build_round_trips(fills)
    if trips.empty:
        trips = pd.DataFrame(columns=list(ROUND_TRIP_COLUMNS))
    ledger.rewrite_derived(trips, "trades")
    nav_df = build_nav_snapshot(fills, seed_capital, as_of_date)
    ledger.record(nav_df.to_dict("records"), kind="nav")
    nav_row = nav_df.iloc[0]
    logger.info(
        "[PORTFOLIO] stage=paper_ledger_refresh as_of=%s n_trades=%d nav=%d cash=%d n_open=%d",
        as_of_date,
        len(trips),
        int(nav_row["nav"]),
        int(nav_row["cash"]),
        int(nav_row["n_open_positions"]),
    )
    return len(trips)


class PaperLedger:
    """온디스크 페이퍼 원장. 매 상태전이마다 즉시 flush한다(WSL 재기동 내성)."""

    def __init__(self, root: Path | None = None, *, lock_timeout_seconds: float = PAPER_LEDGER_LOCK_TIMEOUT_SECONDS) -> None:
        if lock_timeout_seconds <= 0:
            raise ValueError(f"lock_timeout_seconds must be positive, got {lock_timeout_seconds}")
        self._root = Path(root) if root is not None else Path(settings.PAPER_DIR)
        self._lock_timeout_seconds = float(lock_timeout_seconds)
        self._lock_fd: int | None = None

    def _store(self, kind: str) -> Path:
        return self._root / f"{kind}.parquet"

    def _lock_path(self) -> Path:
        return self._root / PAPER_LEDGER_LOCK_FILENAME

    @contextlib.contextmanager
    def exclusive(self) -> Iterator[None]:
        """Hold the host-wide paper ledger lock for one read-modify-write session.

        Entry and exit run in separate containers that share the ledger directory
        through a bind mount; flock on a file inside that directory serializes them
        on the host kernel. Every caller that reads ledger state and writes a
        decision derived from it (cash sizing, open-lot selection, corrections)
        must hold this lock for the whole read-decide-write span, not per write.

        Raises:
            TimeoutError: The lock was not acquired within the ledger's lock timeout.
            OSError: The lock file cannot be created or opened.
        """
        if self._lock_fd is not None:
            raise RuntimeError("PaperLedger.exclusive is not re-entrant")
        self._root.mkdir(parents=True, exist_ok=True)
        # 가장 오래 합법적으로 잡는 쪽은 exit 시가 조회 루프(6회 x 10초 + lot당 TR 2회)로
        # 300초에 한참 못 미친다. 그 이상 대기는 wedged 홀더이므로 줄 세우지 말고 크게 실패한다.
        lock_path = self._lock_path()
        fd = os.open(str(lock_path), os.O_RDONLY | os.O_CREAT, 0o666)
        start = time.monotonic()
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except (BlockingIOError, OSError) as exc:
                    import errno as _errno

                    if isinstance(exc, OSError) and exc.errno not in (_errno.EACCES, _errno.EAGAIN, _errno.EWOULDBLOCK):
                        raise  # pragma: no cover - unexpected flock errno
                    if time.monotonic() - start >= self._lock_timeout_seconds:
                        raise TimeoutError(
                            f"paper ledger lock not acquired within {self._lock_timeout_seconds:.1f}s: {lock_path}"
                        ) from exc
                    time.sleep(0.05)
            self._lock_fd = fd
            waited = time.monotonic() - start
            logger.info("[PORTFOLIO] stage=paper_ledger_lock status=ACQUIRED waited_s=%.1f", waited)
            yield
        finally:
            if self._lock_fd is not None:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    self._lock_fd = None
                    os.close(fd)
            else:
                os.close(fd)

    def _append(self, rows: list[dict[str, Any]], kind: str, dedup_keys: list[str]) -> int:
        target = self._store(kind)
        new_df = pd.DataFrame(rows)
        existing = pd.read_parquet(target) if target.exists() else pd.DataFrame()
        if kind in _APPEND_ONLY_KINDS:
            key = dedup_keys[0]
            if not new_df.empty:
                batch_keys = new_df[key].astype(str).tolist() if key in new_df.columns else []
                if len(set(batch_keys)) != len(batch_keys):
                    raise ValueError(f"duplicate {key} within append-only {kind} batch")
                if not existing.empty and key in existing.columns:
                    stored = set(existing[key].astype(str).tolist())
                    for k in batch_keys:
                        if k in stored:
                            raise ValueError(f"append-only {kind} key {k!r} already stored")
            if existing.empty:
                merged = new_df.copy()
            else:
                union_cols = sorted(set(existing.columns.tolist()) | set(new_df.columns.tolist()))
                merged = pd.concat(
                    [existing.reindex(columns=union_cols), new_df.reindex(columns=union_cols)],
                    ignore_index=True,
                )
            atomic_write_parquet(merged, target)
            return len(merged)
        if existing.empty:
            merged = new_df.copy()
        else:
            union_cols = sorted(set(existing.columns.tolist()) | set(new_df.columns.tolist()))
            merged = pd.concat(
                [existing.reindex(columns=union_cols), new_df.reindex(columns=union_cols)],
                ignore_index=True,
            )
            merged = merged.drop_duplicates(subset=dedup_keys, keep="last")
        atomic_write_parquet(merged, target)
        return len(merged)

    def load(self, kind: str) -> pd.DataFrame:
        """Load a ledger parquet store.

        Args:
            kind: Ledger kind in LEDGER_KINDS.

        Returns:
            Stored frame, or an empty frame when the file is absent.

        Raises:
            ValueError: When kind is unknown.
        """
        if kind not in LEDGER_KINDS:
            raise ValueError(f"unknown ledger kind {kind!r}")
        target = self._store(kind)
        if not target.exists():
            return pd.DataFrame()
        return pd.read_parquet(target)

    def record(self, rows: list[dict[str, Any]], kind: str) -> int:
        """Append rows to a ledger kind atomically.

        fills and corrections are append-only evidence: a row whose key already
        exists is a contract violation, never an update. Other kinds keep
        last-write-wins merge semantics because they carry status updates
        (orders) or per-date snapshots (nav, decisions).

        Args:
            rows: Ledger rows to append.
            kind: One of LEDGER_KINDS.

        Returns:
            Row count of the stored ledger after the write.

        Raises:
            ValueError: Unknown kind, or an append-only key collides with a stored or
                in-batch row.
        """
        if kind not in LEDGER_KINDS:
            raise ValueError(f"unknown ledger kind {kind!r}")
        return self._append(rows, kind, _LEDGER_DEDUP_KEYS[kind])

    def record_no_decision(self, decision_date: str, reason: str) -> int:
        """결정 0건인 날도 '결정 없음' 행으로 명시 기록한다(무기록 금지)."""
        row = {
            "decision_date": decision_date,
            "symbol": "",
            "reason": reason,
            "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
        }
        return self._append([row], "decisions", ["decision_date", "symbol"])

    def load_open_positions(self) -> pd.DataFrame:
        """fills 중 청산 체결이 참조하지 않은 진입 로트만 반환한다."""
        fills = self.load_effective_fills()
        if fills.empty:
            return pd.DataFrame(
                {
                    "entry_order_id": pd.Series(dtype="str"),
                    "symbol": pd.Series(dtype="str"),
                    "qty": pd.Series(dtype="int64"),
                    "entry_price": pd.Series(dtype="int64"),
                    "decision_date": pd.Series(dtype="str"),
                }
            )
        # 청산 체결이 참조한 진입 로트만 닫는다(같은 종목 복수 로트를 독립적으로 추적)
        sides = fills["side"].astype(str)
        sells = fills[sides == "sell"]
        if len(sells) > 0:
            if "entry_order_id" not in fills.columns or sells["entry_order_id"].isna().any():
                raise ValueError("sell fill without entry_order_id violates position link")
            closed = set(sells["entry_order_id"].astype(str))
        else:
            closed = set()
        buys = fills[sides == "buy"]
        open_buys = buys[~buys["order_id"].astype(str).isin(closed)]
        return pd.DataFrame(
            {
                "entry_order_id": open_buys["order_id"].astype(str).to_numpy(),
                "symbol": open_buys["symbol"].astype(str).to_numpy(),
                "qty": pd.to_numeric(open_buys["qty"], errors="coerce").fillna(0).astype("int64").to_numpy(),
                "entry_price": pd.to_numeric(open_buys["fill_price"], errors="coerce").fillna(0).astype("int64").to_numpy(),
                "decision_date": open_buys["decision_date"].astype(str).to_numpy(),
            }
        )

    def _voided_order_ids(self) -> set[str]:
        corrections = self.load("corrections")
        if corrections.empty:
            return set()
        voided: set[str] = set()
        mask = corrections["action"].astype(str) == CORRECTION_ACTION_VOID_FILL
        for _, row in corrections[mask].iterrows():
            targets = row.get("target_order_ids", "")
            if targets is None or (not isinstance(targets, str) and pd.isna(targets)):
                continue
            for part in str(targets).split(","):
                part = part.strip()
                if part:
                    voided.add(part)
        return voided

    def load_effective_fills(self) -> pd.DataFrame:
        """Return fills minus every fill voided by a VOID_FILL correction.

        The raw fills store is never rewritten; economic state (cash, open lots,
        round trips, NAV) is always derived from this effective view so a
        correction is an auditable event instead of a silent edit.

        Returns:
            Fill rows in stored order excluding voided order_ids; an empty frame when
            no fills exist.
        """
        fills = self.load("fills")
        if fills.empty:
            return fills
        voided = self._voided_order_ids()
        if not voided:
            return fills
        return fills[~fills["order_id"].astype(str).isin(voided)].reset_index(drop=True)

    def void_fill(
        self, order_id: str, *, reason: str, evidence: str, operator: str, as_of_date: str
    ) -> dict[str, Any]:
        """Void one recorded fill through an append-only correction event.

        Args:
            order_id: Fill order_id to void.
            reason: Short machine-readable reason (e.g. holiday_phantom_exit).
            evidence: Free text or JSON describing how the error was established.
            operator: Human operator identity recorded for accountability.
            as_of_date: KST date (YYYY-MM-DD) whose NAV the correction restates.

        Returns:
            The stored correction row.

        Raises:
            ValueError: Unknown or already-voided order_id, empty reason/operator, or
                voiding a buy fill still referenced by a non-voided sell fill.
        """
        if not str(reason).strip():
            raise ValueError("reason must not be empty")
        if not str(operator).strip():
            raise ValueError("operator must not be empty")
        fills = self.load("fills")
        if fills.empty or order_id not in set(fills["order_id"].astype(str).tolist()):
            raise ValueError(f"unknown fill order_id {order_id!r}")
        if order_id in self._voided_order_ids():
            raise ValueError(f"fill order_id {order_id!r} already voided")
        target_side = str(fills.loc[fills["order_id"].astype(str) == order_id, "side"].iloc[0])
        if target_side == "buy":
            effective = self.load_effective_fills()
            eff_sides = effective["side"].astype(str)
            eff_sells = effective[eff_sides == "sell"]
            if not eff_sells.empty and "entry_order_id" in eff_sells.columns:
                refs = set(eff_sells["entry_order_id"].astype(str).tolist())
                if order_id in refs:
                    raise ValueError(f"buy fill {order_id!r} still referenced by an effective sell fill")
        row: dict[str, Any] = {
            "correction_id": f"{as_of_date}:{CORRECTION_ACTION_VOID_FILL}:{order_id}",
            "action": CORRECTION_ACTION_VOID_FILL,
            "target_order_ids": order_id,
            "reason": reason,
            "evidence": evidence,
            "operator": operator,
            "as_of_date": as_of_date,
            "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
        }
        self.record([row], "corrections")
        return row

    def record_note(
        self,
        *,
        target_order_ids: tuple[str, ...],
        reason: str,
        evidence: str,
        operator: str,
        as_of_date: str,
    ) -> dict[str, Any]:
        """Record a correction that changes no economic state.

        Used for retroactive documentation of repairs performed before correction
        events existed, and for operator annotations of anomalies.

        Returns:
            The stored correction row.

        Raises:
            ValueError: Empty reason/operator or duplicate correction_id.
        """
        if not str(reason).strip():
            raise ValueError("reason must not be empty")
        if not str(operator).strip():
            raise ValueError("operator must not be empty")
        row: dict[str, Any] = {
            "correction_id": f"{as_of_date}:{CORRECTION_ACTION_NOTE}:{reason}",
            "action": CORRECTION_ACTION_NOTE,
            "target_order_ids": ",".join(target_order_ids),
            "reason": reason,
            "evidence": evidence,
            "operator": operator,
            "as_of_date": as_of_date,
            "recorded_at": pd.Timestamp.now(tz="Asia/Seoul"),
        }
        self.record([row], "corrections")
        return row

    def rewrite_derived(self, frame: pd.DataFrame, kind: str) -> int:
        """Atomically replace a derived ledger with a full recomputation.

        Returns:
            Stored row count.

        Raises:
            ValueError: kind is not "trades".
        """
        if kind != "trades":
            raise ValueError(f"rewrite_derived supports only 'trades', got {kind!r}")
        target = self._store(kind)
        out = frame.copy() if frame is not None else pd.DataFrame(columns=list(ROUND_TRIP_COLUMNS))
        atomic_write_parquet(out, target)
        return len(out)
