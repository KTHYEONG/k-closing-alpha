"""페이퍼 트레이딩 일간 세션 (실주문 없이 실행경로 리허설).

주문 전송 TR을 어떤 형태로도 참조하지 않는다. 진입(entry)은 15:21에 영속된
top-k 결정을 소비해 아카이브 종가를 조인해 시장가 등가 주문을 만들고, 청산(exit)은 미청산
포지션을 D+1 KRX 시가단일가(현재가 API stck_oprc, 데이터 계좌)로 시장가 청산한다.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

import aiohttp
import pandas as pd

from src import settings
from src.api.kis.client import KisApiClient, kis_data_client_kwargs
from src.config.market_session import (
    DECISION_WINDOW_END_HHMMSS,
    KRX_CLOSE_MARKET_DIV_CODE,
    PAPER_ENTRY_CATCHUP_DEADLINE_HHMMSS,
    PAPER_ENTRY_EARLIEST_HHMMSS,
    PAPER_ENTRY_HHMMSS,
    PAPER_EXIT_MAX_PRESTART_WAIT_SECONDS,
    PAPER_EXIT_OPEN_AUCTION_HHMMSS,
    PAPER_EXIT_OPEN_QUOTE_EARLIEST_HHMMSS,
    PAPER_EXIT_WINDOW_END_HHMMSS,
)
from src.daily.archive import fetch_archive_snapshot
from src.daily.collect import safe_float
from src.daily.predict import load_topk_decision
from src.data.session_calendar import SessionDay, SessionKind, resolve_session_day, trading_session_gate
from src.data.trading_calendar import is_kis_trading_day
from src.execution.paper_broker import (
    ORDER_STATUS_FILLED,
    ORDER_STATUS_INSUFFICIENT_CASH,
    ORDER_STATUS_MISSED_AUCTION,
    ORDER_STATUS_NO_SNAPSHOT_ROW,
    ORDER_STATUS_UNCONFIRMED,
    ORDER_STATUS_UNFILLED,
    ORDER_STATUS_ZERO_QTY,
    PAPER_BROKERAGE_SIDE_BP,
    PaperLedger,
    PaperOrder,
    build_auction_fill,
    build_nav_snapshot,
    build_open_auction_fill,
    investable_capital,
    order_record,
    refresh_trade_ledgers,
    size_order_qty,
    sizing_price,
)
from src.tools.run_outcome import RUN_OUTCOME_DEGRADED, RUN_OUTCOME_NO_DECISION, record_run_outcome

logger = logging.getLogger(__name__)

# 시가 형성 직후 stck_oprc 반영 지연에 대비한 재조회 한도: 6회 x 10초
PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS: int = 6
PAPER_EXIT_OPEN_QUOTE_RETRY_SECONDS: float = 10.0
# 휴장 연휴(추석 등 최대 ~10 달력일)에도 직전 세션 행이 포함되도록 하는 조회 창 — 빈 응답(장애)과 휴장을 구분한다
PAPER_EXIT_DATED_QUOTE_LOOKBACK_DAYS: int = 14


def _placed_at(date_str: str, hhmmss: str) -> pd.Timestamp:
    return pd.Timestamp(
        f"{date_str} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]}", tz="Asia/Seoul"
    )


def _resolve_session_day(
    decision_date: pd.Timestamp, session_day_fn: Callable[[date], SessionDay] | None
) -> SessionDay:
    resolve = session_day_fn if session_day_fn is not None else resolve_session_day
    return resolve(decision_date.date())


class WindowState(StrEnum):
    """Wall-clock admissibility of a paper phase run relative to its session."""

    EARLY_SKIP = "EARLY_SKIP"
    WAIT = "WAIT"
    OPEN = "OPEN"
    EXPIRED = "EXPIRED"


def exit_window_state(decision_date: pd.Timestamp, now: pd.Timestamp) -> WindowState:
    """Classify an exit run against the open-auction booking window.

    A live book can only sell at the open auction while the open is current;
    booking the open hours later would record an execution no live account
    could have obtained.

    Args:
        decision_date: Exit date (KST, normalized).
        now: Aware KST wall clock.

    Returns:
        EARLY_SKIP when the wait to PAPER_EXIT_OPEN_QUOTE_EARLIEST_HHMMSS exceeds
        PAPER_EXIT_MAX_PRESTART_WAIT_SECONDS; WAIT when a shorter wait is needed;
        OPEN within [earliest, PAPER_EXIT_WINDOW_END_HHMMSS]; EXPIRED after it.

    Raises:
        ValueError: now is naive or not on decision_date.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be tz-aware, got {now!r}")
    now_kst = now.tz_convert("Asia/Seoul")
    if now_kst.date() != decision_date.date():
        raise ValueError(
            f"paper exit open quote is same-day only: decision_date={decision_date.date()} now={now_kst.date()}"
        )
    date_str = decision_date.strftime("%Y-%m-%d")
    earliest = _placed_at(date_str, PAPER_EXIT_OPEN_QUOTE_EARLIEST_HHMMSS)
    if now_kst < earliest:
        wait_s = (earliest - now_kst).total_seconds()
        if wait_s > PAPER_EXIT_MAX_PRESTART_WAIT_SECONDS:
            return WindowState.EARLY_SKIP
        return WindowState.WAIT
    if now_kst <= _placed_at(date_str, PAPER_EXIT_WINDOW_END_HHMMSS):
        return WindowState.OPEN
    return WindowState.EXPIRED


def resolve_entry_decision_date(now: pd.Timestamp) -> pd.Timestamp:
    """Resolve which decision date an entry run is booking.

    A run at or after PAPER_ENTRY_EARLIEST_HHMMSS books today's decision; an
    earlier run is a catch-up of the previous weekday's missed booking and must
    never write records dated today.

    Args:
        now: Aware KST wall clock.

    Returns:
        Normalized naive date: today, or the previous weekday (Mon -> Fri).
    """
    now_kst = now.tz_convert("Asia/Seoul")
    today = now_kst.normalize().tz_localize(None)
    if now_kst >= _placed_at(now_kst.strftime("%Y-%m-%d"), PAPER_ENTRY_EARLIEST_HHMMSS):
        return today
    prev = today - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev


def entry_window_state(decision_date: pd.Timestamp, now: pd.Timestamp) -> WindowState:
    """Classify an entry booking against [D 15:30:00, D+1 08:30:00).

    Returns:
        WAIT before the earliest bound, OPEN inside, EXPIRED at/after the deadline.
        Never EARLY_SKIP.

    Raises:
        ValueError: now is naive.
    """
    if now.tzinfo is None:
        raise ValueError(f"now must be tz-aware, got {now!r}")
    now_kst = now.tz_convert("Asia/Seoul")
    date_str = decision_date.strftime("%Y-%m-%d")
    earliest = _placed_at(date_str, PAPER_ENTRY_EARLIEST_HHMMSS)
    deadline = _placed_at(
        (decision_date + timedelta(days=1)).strftime("%Y-%m-%d"), PAPER_ENTRY_CATCHUP_DEADLINE_HHMMSS
    )
    if now_kst < earliest:
        return WindowState.WAIT
    if now_kst >= deadline:
        return WindowState.EXPIRED
    return WindowState.OPEN


def build_entry_orders(
    picks: pd.DataFrame, decision_date: str, seed_capital: int, placed_at: pd.Timestamp
) -> list[PaperOrder]:
    """픽별 정수 주식수로 시장가 등가 진입 주문을 만든다. 0주 종목은 제외한다."""
    if picks is None or len(picks) == 0:
        return []
    orders: list[PaperOrder] = []
    for _, row in picks.iterrows():
        sizing = int(row["price"])
        if sizing <= 0:
            logger.warning("[DATA] stage=paper_entry status=ZERO_QTY symbol=%s", row["symbol"])
            continue
        qty = size_order_qty(seed_capital, float(row["allocation"]), sizing)
        if qty == 0:
            logger.warning("[DATA] stage=paper_entry status=ZERO_QTY symbol=%s", row["symbol"])
            continue
        orders.append(
            PaperOrder(
                order_id=f"{decision_date}:{row['symbol']}:entry",
                decision_date=decision_date,
                symbol=str(row["symbol"]),
                side="buy",
                qty=qty,
                limit_price=None,
                placed_at=placed_at,
                reason="entry",
            )
        )
    return orders


def build_exit_orders(
    open_positions: pd.DataFrame, decision_date: str, placed_at: pd.Timestamp
) -> list[PaperOrder]:
    """미청산 포지션을 D+1 KRX 시가단일가로 청산하는 시장가 매도 주문을 만든다."""
    # Opening cohort includes symbols with outstanding paper positions via the
    # existing PaperLedger/positions contract; no order is sent or amended.
    if open_positions is None or len(open_positions) == 0:
        return []
    orders: list[PaperOrder] = []
    for _, row in open_positions.iterrows():
        orders.append(
            PaperOrder(
                order_id=f"{row['entry_order_id']}:exit:{decision_date}",
                decision_date=decision_date,
                symbol=str(row["symbol"]),
                side="sell",
                qty=int(row["qty"]),
                limit_price=None,
                placed_at=placed_at,
                reason="open_exit",
                entry_order_id=str(row["entry_order_id"]),
            )
        )
    return orders


@dataclass(frozen=True)
class DatedOpenQuote:
    """Open-auction price with the business date of the session it belongs to.

    Attributes:
        symbol: 6-digit code.
        business_date: KST session date (YYYY-MM-DD) of the attesting daily-chart
            row, or "" when no row was returned.
        open_price: KRX open-auction price in KRW; 0 when unavailable or when the
            two sources disagree.
    """

    symbol: str
    business_date: str
    open_price: int


async def fetch_krx_dated_open_quote(client: Any, session: Any, code: str, decision_date: str) -> DatedOpenQuote:
    """Read today's KRX open with an independent business-date attestation.

    The inquire-price TR (FHKST01010100) answers on holidays with the previous
    session's stck_oprc and carries no date, so its price alone cannot prove
    the session opened today. The daily chart TR (FHKST03010100) over a
    trailing window ending at decision_date supplies the date of the latest
    session; the quote is attested only when that row is dated decision_date
    and its stck_oprc equals the inquire-price open.

    Args:
        client: KisApiClient on the data account.
        session: aiohttp session.
        code: 6-digit symbol.
        decision_date: Exit date (YYYY-MM-DD, KST).

    Returns:
        DatedOpenQuote; open_price is 0 when either TR failed, the chart row is
        missing, or the two opens differ.
    """
    try:
        res = await client.get_current_price(
            session, code, market_div_code=KRX_CLOSE_MARKET_DIV_CODE, allow_market_div_fallback=False
        )
        output = res.get("output") if isinstance(res, dict) and res.get("rt_cd") == "0" else None
        inquire_open = int(safe_float(output.get("stck_oprc"), 0.0)) if isinstance(output, dict) else 0
    except Exception:
        inquire_open = 0
    start = (pd.Timestamp(decision_date) - timedelta(days=PAPER_EXIT_DATED_QUOTE_LOOKBACK_DAYS)).strftime("%Y%m%d")
    end = pd.Timestamp(decision_date).strftime("%Y%m%d")
    business_date = ""
    chart_open = 0
    try:
        chart = await client.get_stock_ohlcv_history(
            session,
            code,
            start_date=start,
            end_date=end,
            period_code="D",
            adj_price="1",
            market_div_code=KRX_CLOSE_MARKET_DIV_CODE,
        )
        rows = chart.get("output2") if isinstance(chart, dict) and chart.get("rt_cd") == "0" else None
        if isinstance(rows, list):
            best_date = ""
            for row in rows:
                if not isinstance(row, dict):
                    continue
                raw_date = str(row.get("stck_bsop_date", ""))
                row_open = int(safe_float(row.get("stck_oprc"), 0.0))
                if row_open > 0 and raw_date > best_date:
                    best_date = raw_date
                    chart_open = row_open
            if best_date:
                business_date = f"{best_date[:4]}-{best_date[4:6]}-{best_date[6:8]}"
    except Exception:
        business_date = ""
        chart_open = 0
    logger.debug(
        "[EXEC] stage=paper_exit_quote symbol=%s business_date=%s inquire_open=%d chart_open=%d",
        code,
        business_date,
        inquire_open,
        chart_open,
    )
    if inquire_open > 0 and business_date == decision_date and inquire_open == chart_open:
        return DatedOpenQuote(symbol=code, business_date=business_date, open_price=inquire_open)
    return DatedOpenQuote(symbol=code, business_date=business_date, open_price=0)


async def run_paper_session(
    decision_date: pd.Timestamp,
    phase: str,
    ledger: PaperLedger | None = None,
    quote_fn: Callable[[str], Awaitable[DatedOpenQuote]] | None = None,
    session: aiohttp.ClientSession | None = None,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    trading_day_fn: Callable[[str], Awaitable[bool]] | None = None,
    record_fn: Callable[..., Any] | None = None,
    session_day_fn: Callable[[date], SessionDay] | None = None,
) -> int:
    """페이퍼 세션을 실행하고 체결 건수를 반환한다. 체결은 원장에 즉시 flush한다. 청산은 미청산 포지션을 D+1 KRX 시가단일가로 시장가 청산한다.

    Exit on a KRX holiday records nothing and keeps every lot open, so the lots
    exit at the next real open auction. ``trading_day_fn`` defaults to the KIS
    trading-day oracle on the data account.

    Exit fills only lots whose open quote is attested to decision_date's
    session; the calendar oracle is a pre-filter, not the proof. When every lot
    returns a quote dated before decision_date the session did not open today:
    nothing is recorded and a DEGRADED outcome ("session_not_opened") is emitted
    through record_fn.

    Raises:
        ValueError: unknown phase or non-same-day exit.
        RuntimeError: the trading-day oracle fails (fail-closed, no fills).
    """
    if phase not in ("entry", "exit"):
        raise ValueError(f"unknown phase {phase!r}")
    ledger = ledger or PaperLedger()
    date_str = decision_date.strftime("%Y-%m-%d")
    sleep_fn = sleep_fn or asyncio.sleep
    now = now_fn() if now_fn is not None else pd.Timestamp.now(tz="Asia/Seoul")
    if phase == "entry":
        # kca-paper-entry는 kca-finalize-close의 ExecStopPost 체인과 독립 백스톱
        # 타이머(15:34 KST) 양쪽에서 매일 트리거된다. 이미 오늘자 entry를 기록한
        # 뒤 재실행되면 cash가 첫 실행분만큼 줄어든 상태로 재사이징해 포지션이
        # 실제보다 작게 재체결되고 orders 감사기록도 덮어써진다(실측: 2026-09-21
        # 017900이 301주에서 50주로 축소, 402340/009150 orders가 ZERO_QTY로
        # 오기록). 오늘자 entry 시도 흔적이 있으면 두 번째 트리거는 조용히 스킵한다.
        # entry와 exit는 바인드 마운트를 공유하는 별도 컨테이너에서 동시에 fire할 수
        # 있으므로 사이징-기록 전 구간을 호스트 전역 락 안에서 수행한다.
        # 진입 기록은 [당일 15:30, 익일 08:30) 창 안에서만 허용된다. 창 이전 기동은
        # 대기하지 않고 스킵하고, 마감 이후 기동은 미기록 알림 후 종료한다.
        state = entry_window_state(decision_date, now)
        if state is WindowState.WAIT:
            logger.info("[DATA] stage=paper_entry status=SKIP reason=before_window date=%s", date_str)
            return 0
        if state is WindowState.EXPIRED:
            logger.info("[DATA] stage=paper_entry status=SKIP reason=entry_window_expired date=%s", date_str)
            if record_fn is not None:
                expired_picks = load_topk_decision(decision_date)
                if not expired_picks.empty:
                    recorded = ledger.load("orders")
                    if recorded.empty or not (
                        (recorded["decision_date"] == date_str) & (recorded["reason"] == "entry")
                    ).any():
                        record_fn(
                            RUN_OUTCOME_NO_DECISION,
                            run_date=date_str,
                            reason="entry_window_expired",
                            metrics={"n_picks": len(expired_picks)},
                        )
            return 0
        session_day = _resolve_session_day(decision_date, session_day_fn)
        session_gate = trading_session_gate(session_day)
        if session_day.kind is SessionKind.CLOSED:
            logger.info("[DATA] stage=paper_entry status=SKIP reason=non_trading_day date=%s", date_str)
            return 0
        with ledger.exclusive():
            existing_orders = ledger.load("orders")
            if not existing_orders.empty and (
                (existing_orders["decision_date"] == date_str) & (existing_orders["reason"] == "entry")
            ).any():
                logger.info("[DATA] stage=paper_entry status=SKIP reason=already_recorded date=%s", date_str)
                return 0
            picks = load_topk_decision(decision_date)
            if picks.empty:
                if trading_day_fn is None:
                    async with aiohttp.ClientSession() as entry_session:
                        entry_client = KisApiClient(**kis_data_client_kwargs())
                        trading_open = await is_kis_trading_day(entry_client, entry_session, date_str)
                else:
                    trading_open = await trading_day_fn(date_str)
                if not trading_open:
                    logger.info("[DATA] stage=paper_entry status=SKIP reason=non_trading_day date=%s", date_str)
                    return 0
                ledger.record_no_decision(date_str, reason="no_persisted_decision")
                return 0
            snap = fetch_archive_snapshot(date_str)
            picks = picks.copy()
            buffer_bp = float(settings.PAPER_ENTRY_SIZING_BUFFER_BP)
            sizing_prices: list[int] = []
            for _, sizing_row in picks.iterrows():
                raw_close = sizing_row.get("close", float("nan"))
                try:
                    decision_price = int(raw_close) if pd.notna(raw_close) else 0
                except (TypeError, ValueError):
                    decision_price = 0
                if decision_price <= 0:
                    sizing_prices.append(0)
                else:
                    sizing_prices.append(sizing_price(decision_price, buffer_bp))
            picks["price"] = sizing_prices
            auction_close = _placed_at(date_str, DECISION_WINDOW_END_HHMMSS)
            missed_auction = False
            if "decided_at" in picks.columns:
                for raw_ts in picks["decided_at"].tolist():
                    try:
                        decided = pd.Timestamp(raw_ts)
                    except (TypeError, ValueError):
                        missed_auction = True
                        break
                    if pd.isna(decided) or decided.tzinfo is None or decided >= auction_close:
                        missed_auction = True
                        break
            else:
                missed_auction = True
            if session_day.kind in (SessionKind.SHIFTED, SessionKind.UNKNOWN):
                missed_auction = True
            placed_at = _placed_at(date_str, PAPER_ENTRY_HHMMSS)
            cash = int(build_nav_snapshot(ledger.load_effective_fills(), settings.PAPER_SEED_CAPITAL, date_str).iloc[0]["cash"])
            capital = investable_capital(cash, settings.PAPER_SEED_CAPITAL)
            if missed_auction:
                missed_orders = build_entry_orders(picks, date_str, seed_capital=capital, placed_at=placed_at)
                missed_rows: list[dict] = [
                    order_record(o, ORDER_STATUS_MISSED_AUCTION) for o in missed_orders
                ]
                missed_symbols = {o.symbol for o in missed_orders}
                for _, pick_row in picks.iterrows():
                    sym = str(pick_row["symbol"])
                    if sym not in missed_symbols:
                        missed_rows.append(
                            order_record(
                                PaperOrder(
                                    order_id=f"{date_str}:{sym}:entry",
                                    decision_date=date_str,
                                    symbol=sym,
                                    side="buy",
                                    qty=0,
                                    limit_price=None,
                                    placed_at=placed_at,
                                    reason="entry",
                                ),
                                ORDER_STATUS_MISSED_AUCTION,
                            )
                        )
                if missed_rows:
                    ledger.record(missed_rows, kind="orders")
                if record_fn is not None:
                    record_fn(
                        RUN_OUTCOME_NO_DECISION,
                        run_date=date_str,
                        reason=session_gate or "decided_after_auction_close",
                    )
                return 0
            orders = build_entry_orders(
                picks, date_str, seed_capital=capital, placed_at=placed_at
            )
            order_rows: list[dict] = []
            order_symbols = {o.symbol for o in orders}
            for _, pick_row in picks.iterrows():
                sym = str(pick_row["symbol"])
                if sym not in order_symbols:
                    if int(pick_row["price"]) == 0:
                        logger.warning(
                            "[DATA] stage=paper_entry status=ZERO_QTY reason=no_decision_price symbol=%s",
                            sym,
                        )
                    order_rows.append(
                        order_record(
                            PaperOrder(
                                order_id=f"{date_str}:{sym}:entry",
                                decision_date=date_str,
                                symbol=sym,
                                side="buy",
                                qty=0,
                                limit_price=None,
                                placed_at=placed_at,
                                reason="entry",
                            ),
                            ORDER_STATUS_ZERO_QTY,
                        )
                    )
            pred_by_symbol: dict[str, float] = {}
            if "pred" in picks.columns:
                for _, pred_row in picks.iterrows():
                    sym = str(pred_row["symbol"])
                    try:
                        pred_value = float(pred_row["pred"])
                    except (TypeError, ValueError):
                        pred_value = float("-inf")
                    if pd.isna(pred_value):
                        pred_value = float("-inf")
                    pred_by_symbol[sym] = pred_value
            ranked_orders = sorted(
                orders, key=lambda o: (-pred_by_symbol.get(o.symbol, float("-inf")), o.symbol)
            )
            by_code = snap.set_index("종목코드").to_dict("index") if not snap.empty else {}
            fills: list[dict] = []
            remaining_cash = cash
            n_insufficient = 0
            for order in ranked_orders:
                row = by_code.get(order.symbol)
                if row is None:
                    logger.warning("[DATA] stage=paper_entry symbol=%s status=NO_SNAPSHOT_ROW", order.symbol)
                    order_rows.append(order_record(order, ORDER_STATUS_NO_SNAPSHOT_ROW))
                    continue
                fill = build_auction_fill(order, row)
                if fill is None:
                    logger.warning("[DATA] stage=paper_entry symbol=%s status=UNCONFIRMED", order.symbol)
                    order_rows.append(order_record(order, ORDER_STATUS_UNCONFIRMED))
                    continue
                fill_cost = fill.fill_price * fill.qty + math.floor(
                    fill.fill_price * fill.qty * PAPER_BROKERAGE_SIDE_BP / 10_000
                )
                if fill_cost > remaining_cash:
                    logger.warning("[DATA] stage=paper_entry symbol=%s status=INSUFFICIENT_CASH", order.symbol)
                    order_rows.append(order_record(order, ORDER_STATUS_INSUFFICIENT_CASH))
                    n_insufficient += 1
                    continue
                remaining_cash -= fill_cost
                fills.append(
                    {
                        "order_id": fill.order_id,
                        "symbol": fill.symbol,
                        "side": fill.side,
                        "qty": fill.qty,
                        "fill_price": fill.fill_price,
                        "filled_at": fill.filled_at,
                        "decision_date": order.decision_date,
                        "trigger": fill.trigger,
                    }
                )
                order_rows.append(order_record(order, ORDER_STATUS_FILLED))
            logger.info(
                "[PORTFOLIO] stage=paper_entry_sizing date=%s n_orders=%d sizing_buffer_bp=%s n_insufficient=%d",
                date_str,
                len(orders),
                buffer_bp,
                n_insufficient,
            )
            if fills:
                ledger.record(fills, kind="fills")
            if order_rows:
                ledger.record(order_rows, kind="orders")
            refresh_trade_ledgers(ledger, settings.PAPER_SEED_CAPITAL, date_str)
            return len(fills)
    placed_at = _placed_at(date_str, PAPER_EXIT_OPEN_AUCTION_HHMMSS)
    owned_session: aiohttp.ClientSession | None = None
    try:
        if quote_fn is None or trading_day_fn is None:
            if session is None:  # pragma: no cover - live KIS boundary
                owned_session = aiohttp.ClientSession()
                session = owned_session
            client = KisApiClient(**kis_data_client_kwargs())
            if trading_day_fn is None:

                async def trading_day_fn(day: str) -> bool:
                    return await is_kis_trading_day(client, session, day)

            if quote_fn is None:

                async def quote_fn(code: str) -> DatedOpenQuote:
                    return await fetch_krx_dated_open_quote(client, session, code, date_str)

        # 휴장일에도 현재가 API는 직전 거래일 stck_oprc를 정상 응답한다. 판정 없이 조회하면
        # 존재하지 않는 시가로 가상 청산이 기록된다(실측: 2026-09-24 추석 연휴 3건).
        # 락 전에 가볍게 읽어 포지션이 없으면 종료하고, 창 판정과 휴장일 판정, 시가 신뢰대기도
        # 락 밖에서 수행한다. 락 안에서는 미청산 로트를 다시 읽어 권위 있게 청산한다.
        open_probe = ledger.load_open_positions()
        if len(open_probe) == 0:
            logger.info("[DATA] stage=paper_exit status=SKIP reason=no_open_positions date=%s", date_str)
            return 0
        exit_state = exit_window_state(decision_date, now)

        def _skip_expired() -> int:
            logger.info(
                "[EXEC] stage=paper_exit status=SKIP reason=exit_window_expired date=%s n_open=%d",
                date_str,
                len(open_probe),
            )
            if record_fn is not None:
                record_fn(
                    RUN_OUTCOME_DEGRADED,
                    run_date=date_str,
                    reason="exit_window_expired",
                    metrics={"n_open": len(open_probe)},
                )
            return 0

        if exit_state is WindowState.EARLY_SKIP:
            logger.info(
                "[EXEC] stage=paper_exit status=SKIP reason=before_window date=%s n_open=%d",
                date_str,
                len(open_probe),
            )
            return 0
        if exit_state is WindowState.EXPIRED:
            return _skip_expired()
        session_day = _resolve_session_day(decision_date, session_day_fn)
        if session_day.kind is SessionKind.SHIFTED:
            logger.info(
                "[EXEC] stage=paper_exit status=HOLD reason=shifted_session_hold date=%s n_open=%d",
                date_str,
                len(open_probe),
            )
            if record_fn is not None:
                record_fn(
                    RUN_OUTCOME_DEGRADED,
                    run_date=date_str,
                    reason="shifted_session_hold",
                    metrics={"n_open": len(open_probe)},
                )
            return 0
        if not await trading_day_fn(date_str):
            if session_day.kind is SessionKind.STANDARD:
                if record_fn is not None:
                    record_fn(
                        RUN_OUTCOME_DEGRADED,
                        run_date=date_str,
                        reason="calendar_disagreement",
                        metrics={"n_open": len(open_probe)},
                    )
                return 0
            logger.info("[EXEC] stage=paper_exit status=SKIP reason=non_trading_day date=%s", date_str)
            return 0
        if session_day.kind is SessionKind.CLOSED:
            if record_fn is not None:
                record_fn(
                    RUN_OUTCOME_DEGRADED,
                    run_date=date_str,
                    reason="calendar_disagreement",
                    metrics={"n_open": len(open_probe)},
                )
            return 0
        earliest = _placed_at(date_str, PAPER_EXIT_OPEN_QUOTE_EARLIEST_HHMMSS)
        observed_at = max(now, earliest)
        if exit_state is WindowState.WAIT:
            await sleep_fn((earliest - now).total_seconds())
            now = now_fn() if now_fn is not None else pd.Timestamp.now(tz="Asia/Seoul")
            exit_state = exit_window_state(decision_date, now)
            if exit_state is WindowState.EXPIRED:
                return _skip_expired()
        with ledger.exclusive():
            orders = build_exit_orders(ledger.load_open_positions(), date_str, placed_at)
            if not orders:
                logger.info("[DATA] stage=paper_exit status=SKIP reason=no_open_positions date=%s", date_str)
                return 0
            open_prices: dict[str, int] = {}
            seen_quotes: dict[str, DatedOpenQuote] = {}
            pending = sorted({o.symbol for o in orders})
            for attempt in range(PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS):
                for code in pending:
                    quote = await quote_fn(code)
                    seen_quotes[code] = quote
                    if quote.open_price > 0 and quote.business_date == date_str:
                        open_prices[code] = quote.open_price
                pending = [c for c in pending if c not in open_prices]
                if not pending:
                    break
                if attempt < PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS - 1:
                    await sleep_fn(PAPER_EXIT_OPEN_QUOTE_RETRY_SECONDS)
            fills = []
            filled_ids: set[str] = set()
            window_end = _placed_at(date_str, PAPER_EXIT_WINDOW_END_HHMMSS)
            late_now = now_fn() if now_fn is not None else pd.Timestamp.now(tz="Asia/Seoul")
            if late_now > window_end:
                open_prices = {}
            for order in orders:
                fill = build_open_auction_fill(order, open_prices.get(order.symbol, 0), observed_at)
                if fill is not None:
                    filled_ids.add(order.order_id)
                    fills.append(
                        {
                            "order_id": fill.order_id,
                            "symbol": fill.symbol,
                            "side": fill.side,
                            "qty": fill.qty,
                            "fill_price": fill.fill_price,
                            "filled_at": fill.filled_at,
                            "decision_date": order.decision_date,
                            "trigger": fill.trigger,
                            "entry_order_id": order.entry_order_id,
                        }
                    )
            if not filled_ids and seen_quotes and all(
                q.business_date and q.business_date < date_str for q in seen_quotes.values()
            ):
                logger.info(
                    "[EXEC] stage=paper_exit status=SKIP reason=session_not_opened date=%s n_open=%d",
                    date_str,
                    len(orders),
                )
                if record_fn is not None:
                    record_fn(
                        RUN_OUTCOME_DEGRADED,
                        run_date=date_str,
                        reason="session_not_opened",
                        metrics={"n_open": len(orders)},
                    )
                return 0
            if fills:
                ledger.record(fills, kind="fills")
            ledger.record(
                [
                    order_record(o, ORDER_STATUS_FILLED if o.order_id in filled_ids else ORDER_STATUS_UNFILLED)
                    for o in orders
                ],
                kind="orders",
            )
            if len(filled_ids) < len(orders):
                unfilled_symbols = sorted({o.symbol for o in orders if o.order_id not in filled_ids})
                logger.warning(
                    "[EXEC] stage=paper_exit status=OPEN_UNAVAILABLE date=%s symbols=%s", date_str, unfilled_symbols
                )
            refresh_trade_ledgers(ledger, settings.PAPER_SEED_CAPITAL, date_str)
            return len(fills)
    finally:
        if owned_session is not None:
            await owned_session.close()  # pragma: no cover - live KIS boundary


def main() -> None:  # pragma: no cover - CLI entry; logic covered via run_paper_session scenarios
    parser = argparse.ArgumentParser(description="Paper-trading daily session (no real orders)")
    parser.add_argument("--phase", choices=["entry", "exit"], required=True)
    parser.add_argument("--date", default=None, help="Decision date YYYY-MM-DD (default today)")
    args = parser.parse_args()
    now = pd.Timestamp.now(tz="Asia/Seoul")
    if args.date:
        decision_date = pd.Timestamp(args.date)
    elif args.phase == "entry":
        decision_date = resolve_entry_decision_date(now)
    else:
        decision_date = now.normalize().tz_localize(None)
    asyncio.run(
        run_paper_session(
            decision_date, phase=args.phase, record_fn=functools.partial(record_run_outcome, f"paper_{args.phase}")
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
