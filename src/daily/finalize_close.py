"""Closing-price finalization pass: decision rows -> confirmed EOD in-place update."""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.api.kis.client import KisApiClient
from src.config.market_session import (
    CLOSING_AUCTION_CONFIRM_EARLIEST_HHMMSS,
    CLOSING_AUCTION_CONFIRMED_MKOP_CODE,
    CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS,
    KRX_CLOSE_MARKET_DIV_CODE,
)
from src.daily import archive
from src.daily.collect import safe_float
from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL

logger = logging.getLogger(__name__)

# 벤더 prdy_ctrt 소수 2자리(%) 반올림 오차(최대 5e-5)의 4배 여유
CLOSE_RATE_CONSISTENCY_ATOL: float = 2e-4


def is_close_confirmed(
    price_output: Mapping[str, Any], book_output2: Mapping[str, Any], now: datetime
) -> bool:
    """3중 fail-closed 확정 게이트 (시계 + mkop 코드 + 두 TR 가격 일치)."""
    if now.strftime("%H%M%S") < CLOSING_AUCTION_CONFIRM_EARLIEST_HHMMSS:
        return False
    if str((book_output2 or {}).get("antc_mkop_cls_code", "")).strip() != CLOSING_AUCTION_CONFIRMED_MKOP_CODE:
        return False
    price = safe_float((price_output or {}).get("stck_prpr"), 0.0)
    book = safe_float((book_output2 or {}).get("stck_prpr"), 0.0)
    if price <= 0.0:
        return False
    return price == book


def build_finalized_row(
    decision_row: Mapping[str, Any], price_output: Mapping[str, Any], confirmed_ts: datetime
) -> dict[str, Any]:
    """확정 현재가 응답을 EOD 갱신 딕셔너리로 변환한다 (허용 키로 한정)."""
    close = int(safe_float(price_output.get("stck_prpr"), 0.0))
    prev = int(safe_float(price_output.get("stck_sdpr"), 0.0)) or int(safe_float(decision_row.get("전일종가"), 0.0))
    open_price = int(safe_float(price_output.get("stck_oprc"), 0.0))
    high_price = int(safe_float(price_output.get("stck_hgpr"), 0.0))
    low_price = int(safe_float(price_output.get("stck_lwpr"), 0.0))
    vol = int(safe_float(price_output.get("acml_vol"), 0.0))
    trade_value = round(safe_float(price_output.get("acml_tr_pbmn"), 0.0) / 100_000_000, 2)
    rate = safe_float(price_output.get("prdy_ctrt"), 0.0)
    decision_vol = int(safe_float(decision_row.get("거래량"), 0.0))
    if close <= 0 or prev <= 0 or vol < decision_vol:
        raise ValueError(f"invalid confirmed quote close={close} prev={prev} vol={vol} decision_vol={decision_vol}")
    if abs(close / prev - 1.0 - rate / 100.0) > CLOSE_RATE_CONSISTENCY_ATOL:
        raise ValueError(f"price/rate mismatch close={close} prev={prev} rate={rate}")
    kept = safe_float(decision_row.get(DECISION_CLOSE_COL), 0.0)
    decision_keep = int(kept) if kept > 0.0 else decision_row.get("종가")
    out: dict[str, Any] = {
        "시가": open_price,
        "고가": high_price,
        "저가": low_price,
        "종가": close,
        "전일종가": prev,
        "거래량": vol,
        "거래대금": trade_value,
        "등락률": rate,
        DECISION_CLOSE_COL: decision_keep,
        CLOSE_CONFIRMED_COL: True,
        "execution_timestamp": confirmed_ts,
    }
    if safe_float(price_output.get("hts_avls"), 0.0) > 0.0:
        out["시가총액"] = round(safe_float(price_output.get("hts_avls"), 0.0), 2)
    return out


async def fetch_confirmed_quote(client: Any, session: Any, code: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """현재가(output)와 호가/예상체결(output2)을 동시 조회한다 (실패 블록은 빈 dict)."""
    price_res, book_res = await asyncio.gather(
        client.get_current_price(session, code, market_div_code=KRX_CLOSE_MARKET_DIV_CODE),
        client.get_orderbook_snapshot(session, code, market_div_code=KRX_CLOSE_MARKET_DIV_CODE),
    )
    price_output = price_res.get("output") if isinstance(price_res, dict) and price_res.get("rt_cd") == "0" and isinstance(price_res.get("output"), dict) else {}
    book_output2 = book_res.get("output2") if isinstance(book_res, dict) and book_res.get("rt_cd") == "0" and isinstance(book_res.get("output2"), dict) else {}
    return price_output, book_output2


async def run_close_finalization(
    snapshot_date: str | None = None,
    *,
    client: Any | None = None,
    session: Any | None = None,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], Any] | None = None,
    retry_interval_seconds: float = 30.0,
) -> int:
    """당일 아카이브 행을 확정값으로 in-place 갱신하고 확정 행 수를 반환한다."""
    now_fn = now_fn or (lambda: datetime.now(ZoneInfo("Asia/Seoul")))
    sleep_fn = sleep_fn or asyncio.sleep
    snap = snapshot_date or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    df = archive.fetch_archive_snapshot(snapshot_date=snap)
    pending = df.index[~df[CLOSE_CONFIRMED_COL].fillna(False).astype(bool)].tolist()
    n_finalized = 0
    while True:
        now = now_fn()
        if now.strftime("%H%M%S") > CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS:
            break
        for idx in list(pending):
            tick = now_fn()
            code = str(df.at[idx, "종목코드"])
            price_output, book_output2 = await fetch_confirmed_quote(client, session, code)
            if not is_close_confirmed(price_output, book_output2, tick):
                continue
            try:
                update = build_finalized_row(df.loc[idx].to_dict(), price_output, tick)
            except ValueError as exc:
                # 불변식 위반 종목은 행을 건드리지 않고 미확정으로 남긴다 (동결가 승격 금지)
                logger.warning(
                    "[DATA] stage=close_finalization code=%s status=REJECTED reason=%s", code, exc
                )
                continue
            for key, value in update.items():
                if key not in df.columns:
                    df[key] = pd.NA
                df.at[idx, key] = value
            pending.remove(idx)
            n_finalized += 1
        if not pending:
            break
        await sleep_fn(retry_interval_seconds)
    if n_finalized >= 1:
        archive.upsert_archive_snapshot(df, snapshot_date=snap)
    unconfirmed = [str(df.at[i, "종목코드"]) for i in pending]
    logger.info(
        "[DATA] stage=close_finalization date=%s n_finalized=%d n_unconfirmed=%d unconfirmed=%s",
        snap,
        n_finalized,
        len(unconfirmed),
        unconfirmed[:10],
    )
    return n_finalized


async def _amain(args) -> int:
    """단일 이벤트 루프 안에서 세션 생성/토큰/확정/종료를 모두 수행한다."""
    owned_client = KisApiClient()
    session = owned_client.create_session()
    try:
        await owned_client.ensure_token(session)
        n = await run_close_finalization(
            snapshot_date=args.date,
            client=owned_client,
            session=session,
            retry_interval_seconds=args.retry_interval,
        )
    finally:
        await session.close()
    logger.info("[DATA] stage=close_finalization rows=%d", n)
    return n


def main() -> None:  # pragma: no cover - CLI entry; logic covered via run_close_finalization scenarios
    """CLI 진입점: 단일 asyncio.run으로 확정 패스를 실행한다."""
    parser = argparse.ArgumentParser(description="Closing-price finalization (in-place EOD update)")
    parser.add_argument("--date", default=None)
    parser.add_argument("--retry-interval", type=float, default=30.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(_amain(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
