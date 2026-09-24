"""페이퍼 트레이딩 일간 세션 (실주문 없이 실행경로 리허설).

주문 전송 TR을 어떤 형태로도 참조하지 않는다. 진입(entry)은 15:21에 영속된
top-k 결정을 소비해 아카이브 종가를 조인해 시장가 등가 주문을 만들고, 청산(exit)은 미청산
포지션을 D+1 KRX 시가단일가(현재가 API stck_oprc, 데이터 계좌)로 시장가 청산한다.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
import pandas as pd

from src import settings
from src.api.kis.client import KisApiClient, kis_data_client_kwargs
from src.config.market_session import (
    KRX_CLOSE_MARKET_DIV_CODE,
    PAPER_ENTRY_HHMMSS,
    PAPER_EXIT_OPEN_AUCTION_HHMMSS,
    PAPER_EXIT_OPEN_QUOTE_EARLIEST_HHMMSS,
)
from src.daily.archive import fetch_archive_snapshot
from src.daily.collect import safe_float
from src.daily.predict import load_topk_decision
from src.data.trading_calendar import is_kis_trading_day
from src.execution.paper_broker import (
    ORDER_STATUS_FILLED,
    ORDER_STATUS_NO_SNAPSHOT_ROW,
    ORDER_STATUS_UNCONFIRMED,
    ORDER_STATUS_UNFILLED,
    ORDER_STATUS_ZERO_QTY,
    PaperLedger,
    PaperOrder,
    build_auction_fill,
    build_nav_snapshot,
    build_open_auction_fill,
    investable_capital,
    order_record,
    refresh_trade_ledgers,
    size_order_qty,
)

logger = logging.getLogger(__name__)

# 시가 형성 직후 stck_oprc 반영 지연에 대비한 재조회 한도: 6회 x 10초
PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS: int = 6
PAPER_EXIT_OPEN_QUOTE_RETRY_SECONDS: float = 10.0


def _placed_at(date_str: str, hhmmss: str) -> pd.Timestamp:
    return pd.Timestamp(
        f"{date_str} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]}", tz="Asia/Seoul"
    )


def build_entry_orders(
    picks: pd.DataFrame, decision_date: str, seed_capital: int, placed_at: pd.Timestamp
) -> list[PaperOrder]:
    """픽별 정수 주식수로 시장가 등가 진입 주문을 만든다. 0주 종목은 제외한다."""
    if picks is None or len(picks) == 0:
        return []
    orders: list[PaperOrder] = []
    for _, row in picks.iterrows():
        qty = size_order_qty(seed_capital, float(row["allocation"]), int(row["price"]))
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


async def fetch_krx_open_quote(client: Any, session: Any, code: str) -> int:
    """KRX 시가단일가 체결가를 현재가 API stck_oprc로 조회한다."""
    res = await client.get_current_price(
        session, code, market_div_code=KRX_CLOSE_MARKET_DIV_CODE, allow_market_div_fallback=False
    )
    output = res.get("output") if isinstance(res, dict) and res.get("rt_cd") == "0" else None
    if isinstance(output, dict):
        return int(safe_float(output.get("stck_oprc"), 0.0))
    return 0


async def run_paper_session(
    decision_date: pd.Timestamp,
    phase: str,
    ledger: PaperLedger | None = None,
    quote_fn: Callable[[str], Awaitable[int]] | None = None,
    session: aiohttp.ClientSession | None = None,
    now_fn: Callable[[], pd.Timestamp] | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    trading_day_fn: Callable[[str], Awaitable[bool]] | None = None,
) -> int:
    """페이퍼 세션을 실행하고 체결 건수를 반환한다. 체결은 원장에 즉시 flush한다. 청산은 미청산 포지션을 D+1 KRX 시가단일가로 시장가 청산한다.

    Exit on a KRX holiday records nothing and keeps every lot open, so the lots
    exit at the next real open auction. ``trading_day_fn`` defaults to the KIS
    trading-day oracle on the data account.

    Raises:
        ValueError: unknown phase or non-same-day exit.
        RuntimeError: the trading-day oracle fails (fail-closed, no fills).
    """
    if phase not in ("entry", "exit"):
        raise ValueError(f"unknown phase {phase!r}")
    ledger = ledger or PaperLedger()
    date_str = decision_date.strftime("%Y-%m-%d")
    sleep_fn = sleep_fn or asyncio.sleep
    if phase == "entry":
        # kca-paper-entry는 kca-finalize-close의 ExecStopPost 체인과 독립 백스톱
        # 타이머(15:34 KST) 양쪽에서 매일 트리거된다. 이미 오늘자 entry를 기록한
        # 뒤 재실행되면 cash가 첫 실행분만큼 줄어든 상태로 재사이징해 포지션이
        # 실제보다 작게 재체결되고 orders 감사기록도 덮어써진다(실측: 2026-09-21
        # 017900이 301주에서 50주로 축소, 402340/009150 orders가 ZERO_QTY로
        # 오기록). 오늘자 entry 시도 흔적이 있으면 두 번째 트리거는 조용히 스킵한다.
        existing_orders = ledger.load("orders")
        if not existing_orders.empty and (
            (existing_orders["decision_date"] == date_str) & (existing_orders["reason"] == "entry")
        ).any():
            logger.info("[DATA] stage=paper_entry status=SKIP reason=already_recorded date=%s", date_str)
            return 0
        picks = load_topk_decision(decision_date)
        if picks.empty:
            ledger.record_no_decision(date_str, reason="no_persisted_decision")
            return 0
        snap = fetch_archive_snapshot(date_str)
        archive_prices = (
            snap.drop_duplicates(subset=["종목코드"]).set_index("종목코드")["종가"]
            if {"종목코드", "종가"}.issubset(set(snap.columns))
            else pd.Series(dtype="float64")
        )
        overlay = pd.to_numeric(picks["symbol"].map(archive_prices), errors="coerce")
        base = (
            pd.to_numeric(picks["price"], errors="coerce")
            if "price" in picks.columns
            else pd.Series(float("nan"), index=picks.index)
        )
        picks = picks.copy()
        picks["price"] = overlay.fillna(base).fillna(0).astype("int64")
        placed_at = _placed_at(date_str, PAPER_ENTRY_HHMMSS)
        cash = int(build_nav_snapshot(ledger.load("fills"), settings.PAPER_SEED_CAPITAL, date_str).iloc[0]["cash"])
        capital = investable_capital(cash, settings.PAPER_SEED_CAPITAL)
        orders = build_entry_orders(
            picks, date_str, seed_capital=capital, placed_at=placed_at
        )
        order_rows: list[dict] = []
        order_symbols = {o.symbol for o in orders}
        for _, pick_row in picks.iterrows():
            sym = str(pick_row["symbol"])
            if sym not in order_symbols:
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
        by_code = snap.set_index("종목코드").to_dict("index") if not snap.empty else {}
        fills: list[dict] = []
        for order in orders:
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
        if fills:
            ledger.record(fills, kind="fills")
        if order_rows:
            ledger.record(order_rows, kind="orders")
        refresh_trade_ledgers(ledger, settings.PAPER_SEED_CAPITAL, date_str)
        return len(fills)
    placed_at = _placed_at(date_str, PAPER_EXIT_OPEN_AUCTION_HHMMSS)
    orders = build_exit_orders(ledger.load_open_positions(), date_str, placed_at)
    if not orders:
        logger.info("[DATA] stage=paper_exit status=SKIP reason=no_open_positions date=%s", date_str)
        return 0
    now = now_fn() if now_fn is not None else pd.Timestamp.now(tz="Asia/Seoul")
    if now.strftime("%Y-%m-%d") != date_str:
        raise ValueError(f"paper exit open quote is same-day only: decision_date={date_str} now={now.date()}")
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

                async def quote_fn(code: str) -> int:
                    return await fetch_krx_open_quote(client, session, code)

        # 휴장일에도 현재가 API는 직전 거래일 stck_oprc를 정상 응답한다. 판정 없이 조회하면
        # 존재하지 않는 시가로 가상 청산이 기록된다(실측: 2026-09-24 추석 연휴 3건).
        if not await trading_day_fn(date_str):
            logger.info("[EXEC] stage=paper_exit status=SKIP reason=non_trading_day date=%s n_open=%d", date_str, len(orders))
            return 0
        earliest = _placed_at(date_str, PAPER_EXIT_OPEN_QUOTE_EARLIEST_HHMMSS)
        if now < earliest:
            await sleep_fn((earliest - now).total_seconds())
        observed_at = max(now, earliest)
        open_prices: dict[str, int] = {}
        pending = sorted({o.symbol for o in orders})
        for attempt in range(PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS):
            for code in pending:
                price = await quote_fn(code)
                if price > 0:
                    open_prices[code] = price
            pending = [c for c in pending if c not in open_prices]
            if not pending:
                break
            if attempt < PAPER_EXIT_OPEN_QUOTE_MAX_ATTEMPTS - 1:
                await sleep_fn(PAPER_EXIT_OPEN_QUOTE_RETRY_SECONDS)
        fills: list[dict] = []
        filled_ids: set[str] = set()
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
    finally:
        if owned_session is not None:
            await owned_session.close()  # pragma: no cover - live KIS boundary
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


def main() -> None:  # pragma: no cover - CLI entry; logic covered via run_paper_session scenarios
    parser = argparse.ArgumentParser(description="Paper-trading daily session (no real orders)")
    parser.add_argument("--phase", choices=["entry", "exit"], required=True)
    parser.add_argument("--date", default=None, help="Decision date YYYY-MM-DD (default today)")
    args = parser.parse_args()
    decision_date = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    asyncio.run(run_paper_session(decision_date, phase=args.phase))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
