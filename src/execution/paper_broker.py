"""Paper-trading execution ledger and fill oracle (no real order transmission).

실주문 전송 TR을 어떤 형태로도 참조하지 않는다. 체결 판정은 관측된 실체결
프린트만을 오라클로 사용하며, 체결가에 수수료/세금/스프레드를 가산하지
않는다(거래비용은 src/execution/cost_model.py가 별도로 부과한다).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src import settings
from src.data.io_utils import atomic_write_parquet

logger = logging.getLogger(__name__)

# exit-timing 레버 실측(TP 5% 지정가 + MOC 폴백)에서 유래한 익절 폭.
PAPER_TAKE_PROFIT_RATIO: float = 0.05


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


class PaperLedger:
    """온디스크 페이퍼 원장. 매 상태전이마다 즉시 flush한다(WSL 재기동 내성)."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root) if root is not None else Path(settings.PAPER_DIR)

    def _store(self, kind: str) -> Path:
        return self._root / f"{kind}.parquet"

    def _append(self, rows: list[dict[str, Any]], kind: str, dedup_keys: list[str]) -> int:
        target = self._store(kind)
        new_df = pd.DataFrame(rows)
        existing = pd.read_parquet(target) if target.exists() else pd.DataFrame()
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

    def record(self, rows: list[dict[str, Any]], kind: str) -> int:
        """kind별 parquet에 원자적으로 append-merge한다."""
        if kind not in ("orders", "fills", "decisions"):
            raise ValueError(f"unknown ledger kind {kind!r}")
        keys = ["order_id"] if kind in ("orders", "fills") else ["decision_date", "symbol"]
        return self._append(rows, kind, keys)

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
        """fills 중 같은 symbol의 후속 sell 체결이 없는 buy 체결만 반환한다."""
        target = self._store("fills")
        if not target.exists():
            return pd.DataFrame(
                {
                    "symbol": pd.Series(dtype="str"),
                    "qty": pd.Series(dtype="int64"),
                    "entry_price": pd.Series(dtype="int64"),
                    "decision_date": pd.Series(dtype="str"),
                }
            )
        fills = pd.read_parquet(target)
        # 종목 단위 'sell 존재 여부'가 아니라 기록순 매수/매도 쌍으로 상계한다.
        # 같은 종목 재진입이 흔하므로(과거 청산 이력만으로 신규 포지션을 지우면 영구 유실)
        # 종목별 미상계 매수 = 매수건수 - 매도건수이며, 그 수만큼 최근 매수를 남긴다.
        sides = fills["side"].astype(str)
        symbols = fills["symbol"].astype(str)
        buys = fills[sides == "buy"]
        n_sells = symbols[sides == "sell"].value_counts()
        keep = pd.Series(False, index=buys.index)
        for symbol, idx in buys.groupby(symbols[sides == "buy"]).groups.items():
            open_count = len(idx) - int(n_sells.get(symbol, 0))
            if open_count > 0:
                keep.loc[list(idx)[-open_count:]] = True
        open_buys = buys[keep]
        return pd.DataFrame(
            {
                "symbol": open_buys["symbol"].astype(str).to_numpy(),
                "qty": pd.to_numeric(open_buys["qty"], errors="coerce").fillna(0).astype("int64").to_numpy(),
                "entry_price": pd.to_numeric(open_buys["fill_price"], errors="coerce").fillna(0).astype("int64").to_numpy(),
                "decision_date": open_buys["decision_date"].astype(str).to_numpy(),
            }
        )
