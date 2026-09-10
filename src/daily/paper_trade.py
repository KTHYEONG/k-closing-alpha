"""페이퍼 트레이딩 일간 세션 (실주문 없이 실시간 체결틱으로 실행경로 리허설).

주문 전송 TR을 어떤 형태로도 참조하지 않는다. 진입(entry)은 슬리브 픽에
아카이브 종가를 조인해 시장가 등가 주문을 만들고, 청산(exit)은 미청산
포지션에 익절 지정가 주문을 만든 뒤 H0STCNT0 실체결 프린트로 체결을 판정한다.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math

import aiohttp
import pandas as pd

from src import settings
from src.api.kis.ws_client import KisWebSocketClient, issue_approval_key
from src.config.market_session import PAPER_ENTRY_HHMMSS, PAPER_EXIT_MOC_HHMMSS, PAPER_EXIT_SESSION_START_HHMMSS
from src.daily.archive import fetch_archive_snapshot
from src.daily.predict import run_topk_ranker_sleeve
from src.execution.paper_broker import PAPER_TAKE_PROFIT_RATIO, PaperLedger, PaperOrder, decide_fill, size_order_qty

logger = logging.getLogger(__name__)


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
    open_positions: pd.DataFrame, decision_date: str, placed_at: pd.Timestamp, moc: bool
) -> list[PaperOrder]:
    """미청산 포지션에 익절 지정가(moc=False) 또는 MOC 대체 청산(moc=True) 주문을 만든다."""
    if open_positions is None or len(open_positions) == 0:
        return []
    orders: list[PaperOrder] = []
    for _, row in open_positions.iterrows():
        limit_price = None if moc else math.ceil(int(row["entry_price"]) * (1 + PAPER_TAKE_PROFIT_RATIO))
        orders.append(
            PaperOrder(
                order_id=f"{decision_date}:{row['symbol']}:exit",
                decision_date=decision_date,
                symbol=str(row["symbol"]),
                side="sell",
                qty=int(row["qty"]),
                limit_price=limit_price,
                placed_at=placed_at,
                reason="moc_exit" if moc else "take_profit",
            )
        )
    return orders


async def run_paper_session(
    decision_date: pd.Timestamp,
    phase: str,
    ledger: PaperLedger | None = None,
    ws_client: KisWebSocketClient | None = None,
    session: aiohttp.ClientSession | None = None,
) -> int:
    """페이퍼 세션을 실행하고 체결 건수를 반환한다. 체결은 원장에 즉시 flush한다."""
    if phase not in ("entry", "exit"):
        raise ValueError(f"unknown phase {phase!r}")
    ledger = ledger or PaperLedger()
    date_str = decision_date.strftime("%Y-%m-%d")
    if phase == "entry":
        picks = run_topk_ranker_sleeve(decision_date)
        if picks.empty:
            ledger.record_no_decision(date_str, reason="admitted_below_top_k")
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
        orders = build_entry_orders(
            picks, date_str, seed_capital=settings.PAPER_SEED_CAPITAL, placed_at=placed_at
        )
    else:
        placed_at = _placed_at(date_str, PAPER_EXIT_SESSION_START_HHMMSS)
        positions = ledger.load_open_positions()
        orders = build_exit_orders(positions, date_str, placed_at, moc=False)
    if ws_client is None:  # pragma: no cover - live KIS boundary, probe-verified
        owned = session or aiohttp.ClientSession()
        key = await issue_approval_key(owned, settings.KIS_APP_KEY, settings.KIS_APP_SECRET)
        ws_client = KisWebSocketClient(approval_key=key)
        session = owned
    codes = sorted({o.symbol for o in orders})
    fills: list[dict] = []
    filled_ids: set[str] = set()
    async for symbol, hhmmss, price in ws_client.stream(session, codes):
        print_ts = _placed_at(date_str, hhmmss)
        if print_ts >= placed_at:
            if phase == "exit" and hhmmss >= PAPER_EXIT_MOC_HHMMSS:
                pending = {o.symbol for o in orders if o.order_id not in filled_ids and o.limit_price is not None}
                moc_orders = build_exit_orders(
                    positions[positions["symbol"].isin(pending)], date_str, placed_at, moc=True
                )
                moc_by_symbol = {o.symbol: o for o in moc_orders}
                orders = [moc_by_symbol.get(o.symbol, o) for o in orders]
            for order in [o for o in orders if o.symbol == symbol and o.order_id not in filled_ids]:
                fill = decide_fill(order, price, print_ts)
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
                        }
                    )
    if fills:
        ledger.record(fills, kind="fills")
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
