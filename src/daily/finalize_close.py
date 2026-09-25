"""Closing-price finalization pass: decision rows -> confirmed EOD in-place update."""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src import settings as settings  # noqa: F401 - test seam: tests patch finalize_close.settings
from src.api.kis.client import KisApiClient, kis_data_client_kwargs
from src.config.market_session import (
    CLOSING_AUCTION_CONFIRM_EARLIEST_HHMMSS,
    CLOSING_AUCTION_CONFIRMED_MKOP_CODE,
    CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS,
    KRX_CLOSE_MARKET_DIV_CODE,
)
from src.daily import archive
from src.daily.collect import safe_float
from src.daily.predict import load_topk_decision
from src.data.capture_contracts import (
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureStatus,
)
from src.data.capture_store import CaptureStore
from src.data.session_calendar import SessionKind, resolve_session_day, trading_session_gate
from src.processing.schema import CLOSE_CONFIRMED_COL, DECISION_CLOSE_COL
from src.tools.run_outcome import RUN_OUTCOME_DEGRADED, RUN_OUTCOME_OK, record_run_outcome
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)

# 벤더 prdy_ctrt 소수 2자리(%) 반올림 오차(최대 5e-5)의 4배 여유
CLOSE_RATE_CONSISTENCY_ATOL: float = 2e-4
# 행당 2콜(현재가+호가). 데이터 계좌 앱키 리미터(초당 18콜)가 실제 상한이므로 4행(8콜) 동시면 리미터 안에서 순차 대비 처리량을 확보한다(collect는 15:30 이전 종료, After= 순서 보장).
FINALIZE_CONCURRENCY: int = 4


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


def is_quote_unresolved(price_output: Mapping[str, Any]) -> bool:
    """KIS 미인식 코드의 빈 종목코드+0가 시세 블록을 판정한다."""
    # KIS는 인식하지 못한 코드(Q접두어 없는 ETN 등)에 rt_cd=0과 전 필드 0, 종목코드 공란을 준다 — 재조회해도 바뀌지 않는다.
    return bool(price_output) and not str(price_output.get("stck_shrn_iscd") or "").strip() and safe_float(price_output.get("stck_prpr"), 0.0) <= 0.0


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


def order_pending_by_priority(df: pd.DataFrame, pending: list[Any], pick_codes: frozenset[str]) -> list[Any]:
    """Picks first, then admitted rows, then the rest, preserving original order within each tier (stable sort)."""
    admitted = df["admitted"].fillna(False).astype(bool) if "admitted" in df.columns else pd.Series(False, index=df.index)

    def _tier(idx: Any) -> int:
        if str(df.at[idx, "종목코드"]) in pick_codes:
            return 0
        return 1 if bool(admitted.at[idx]) else 2

    return sorted(pending, key=_tier)


def classify_finalize_outcome(
    n_rows: int,
    n_finalized: int,
    n_unconfirmed: int,
    unconfirmed_picks: list[str],
    *,
    is_trading_day: bool,
) -> tuple[str, str]:
    """Classify the finalize run outcome from confirmation counts.

    An empty archive on a trading day means the decision snapshot was never
    persisted (collect failed or refused a degraded snapshot); reporting it as
    OK hid the 15:20 failure from the outcome log.

    Args:
        n_rows: Archive rows for the date.
        n_finalized: Number of confirmed rows.
        n_unconfirmed: Number of rows left unconfirmed.
        unconfirmed_picks: Unconfirmed pick codes.
        is_trading_day: KIS trading-day verdict for the date.

    Returns:
        (outcome, reason) using the RUN_OUTCOME vocabulary; empty archive yields
        (DEGRADED, "empty_archive") on a trading day and (OK, "non_trading_day")
        otherwise.
    """
    if n_rows == 0:
        if is_trading_day:
            return RUN_OUTCOME_DEGRADED, "empty_archive"
        return RUN_OUTCOME_OK, "non_trading_day"
    if unconfirmed_picks:
        return RUN_OUTCOME_DEGRADED, "picks_unconfirmed"
    if n_finalized == 0 and n_unconfirmed > 0:
        return RUN_OUTCOME_DEGRADED, "zero_confirmed"
    return RUN_OUTCOME_OK, ""


async def fetch_confirmed_quote(client: Any, session: Any, code: str, *, capture_store: CaptureStore | None = None, run_id: str | None = None, cohort_id: str | None = None, poll_round: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain full close-confirmation responses while preserving the existing gate.

    Args:
        client: Existing explicit KRX data client.
        session: Existing HTTP session.
        code: Security identifier.
        capture_store: Owner-local confirmation evidence store.
        run_id: Confirmation task identity.
        cohort_id: Original decision population reference.
        poll_round: Zero-based polling-round index within run_close_finalization's
            while-loop. A symbol that is still pending gets re-fetched on later
            rounds, and each round's response legitimately differs (price settles
            toward confirmation) -- recorded as attempt_index so the immutable
            artifact store sees distinct identities instead of raising a false
            conflict when round N+1's bytes differ from round N's.

    Returns:
        Existing price and orderbook output2 blocks used by the confirmation gate.

    Raises:
        ValueError: Inconsistent context or date.
        RuntimeError: Propagated acquisition failure under existing caller contracts.
    """
    capture_on = not (capture_store is None and run_id is None and cohort_id is None)
    if capture_on and (capture_store is None or not run_id or not cohort_id):
        raise ValueError("inconsistent close-confirmation capture context")
    started = datetime.now(ZoneInfo("Asia/Seoul"))
    price_res, book_res = await asyncio.gather(
        client.get_current_price(session, code, market_div_code=KRX_CLOSE_MARKET_DIV_CODE, allow_market_div_fallback=False),
        client.get_orderbook_snapshot(session, code, market_div_code=KRX_CLOSE_MARKET_DIV_CODE),
    )
    received = datetime.now(ZoneInfo("Asia/Seoul"))
    if capture_on:
        assert capture_store is not None
        assert run_id is not None
        assert cohort_id is not None
        try:
            trading_day = received.date()
            for dataset, payload, endpoint in (
                (CaptureDataset.PRICE, price_res, "inquire-price"),
                (CaptureDataset.ORDERBOOK, book_res, "inquire-asking-price"),
            ):
                body = dict(payload) if isinstance(payload, dict) else None
                capture_store.append_response(
                    CapturedResponse(
                        context=CaptureContext(
                            trading_date=trading_day,
                            run_id=run_id,
                            dataset=dataset,
                            vendor="kis",
                            endpoint=endpoint,
                            symbol=str(code),
                            venue="KRX",
                            session="regular",
                            capture_reason="close_confirmation",
                            cohort_id=cohort_id,
                            scheduled_at=None,
                        ),
                        request_started_at=started,
                        received_at=received,
                        payload=body,
                        status=CaptureStatus.COMPLETE if isinstance(body, dict) else CaptureStatus.FAILED,
                        source_timestamp=None,
                        source_published_at=None,
                        page_index=0,
                        attempt_index=poll_round,
                        continuation={},
                        error_type=None,
                    )
                )
        except (OSError, ValueError) as exc:
            # ValueError는 capture_store의 immutable identity 충돌(예: 동일 run_id/attempt에 대한
            # 재확인 시세가 이전 값과 달라짐)도 포함한다 -- 실측: 2026-09-18 finalize-close가
            # 이 예외로 전체 확정 루프를 중단시켜 파이프 하위 paper-entry가 UNCONFIRMED로 넘어감.
            # 원본 증거 보존은 부가 기능이므로 실패해도 확정 로직 자체는 계속 진행한다.
            logger.warning("[DATA] stage=close_confirmation code=%s status=DEGRADED reason=%s", code, type(exc).__name__)
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
    pick_codes: frozenset[str] = frozenset(),
    on_outcome: Callable[..., Any] | None = None,
    capture_store: CaptureStore | None = None,
    run_id: str | None = None,
    cohort_id: str | None = None,
    trading_day_fn: Callable[[str], Awaitable[bool]] | None = None,
) -> int:
    """당일 아카이브 행을 확정값으로 in-place 갱신하고 확정 행 수를 반환한다 (우선순위, bounded concurrency, on_outcome)."""
    capture_on = not (capture_store is None and run_id is None and cohort_id is None)
    if capture_on and (capture_store is None or not run_id or not cohort_id):
        raise ValueError("inconsistent close-confirmation capture context")
    now_fn = now_fn or (lambda: datetime.now(ZoneInfo("Asia/Seoul")))
    sleep_fn = sleep_fn or asyncio.sleep
    snap = snapshot_date or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    df = archive.fetch_archive_snapshot(snapshot_date=snap)
    # 아카이브 저장소가 종가_확정을 float64(0.0/1.0/NaN)로 돌려주므로 bool 대입 전에 nullable boolean으로 맞춘다
    df[CLOSE_CONFIRMED_COL] = df[CLOSE_CONFIRMED_COL].astype("boolean")
    pending = df.index[~df[CLOSE_CONFIRMED_COL].fillna(False).astype(bool)].tolist()
    n_rows = len(df)
    pending = order_pending_by_priority(df, pending, pick_codes)
    n_finalized = 0
    unresolved: list[Any] = []
    poll_round = 0
    while True:
        now = now_fn()
        # 현재가 TR은 조회 시점 종가만 주므로 과거 스냅샷에 쓰면 다른 날 종가로 덮어쓴다
        if now.strftime("%Y-%m-%d") != snap:
            raise ValueError(f"close finalization is same-day only: snapshot_date={snap} now={now.strftime('%Y-%m-%d')}")
        if now.strftime("%H%M%S") > CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS:
            break
        confirmed: set[Any] = set()
        dropped: set[Any] = set()
        for start in range(0, len(pending), FINALIZE_CONCURRENCY):
            batch = pending[start : start + FINALIZE_CONCURRENCY]
            tick = now_fn()
            if tick.strftime("%H%M%S") > CLOSING_AUCTION_FINALIZE_DEADLINE_HHMMSS:
                break  # 행 단위 데드라인 — 데드라인 이후 배치는 시작하지 않는다
            quotes = await asyncio.gather(
                *(
                    fetch_confirmed_quote(
                        client, session, str(df.at[idx, "종목코드"]),
                        capture_store=capture_store, run_id=run_id, cohort_id=cohort_id, poll_round=poll_round,
                    )
                    for idx in batch
                )
            )
            for idx, (price_output, book_output2) in zip(batch, quotes, strict=True):
                code = str(df.at[idx, "종목코드"])
                if is_quote_unresolved(price_output):
                    dropped.add(idx)
                    unresolved.append(idx)
                    logger.warning("[DATA] stage=close_finalization code=%s status=UNRESOLVED reason=vendor_unrecognized_code", code)
                    continue
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
                confirmed.add(idx)
                n_finalized += 1
        pending = [i for i in pending if i not in confirmed and i not in dropped]
        if not pending:
            break
        poll_round += 1
        await sleep_fn(retry_interval_seconds)
    if n_finalized >= 1:
        archive.upsert_archive_snapshot(df, snapshot_date=snap)
    if capture_on:
        assert capture_store is not None
        assert run_id is not None
        assert cohort_id is not None
        try:
            confirmed_frame = df[df[CLOSE_CONFIRMED_COL].fillna(False).astype(bool)].copy()
            if len(confirmed_frame) > 0:
                capture_store.publish_frame(
                    confirmed_frame,
                    context=CaptureContext(
                        trading_date=date.fromisoformat(snap),
                        run_id=run_id,
                        dataset=CaptureDataset.DAILY_BARS,
                        vendor="owner-local",
                        endpoint="close-confirmation",
                        symbol=None,
                        venue="KRX",
                        session="regular",
                        capture_reason="close_confirmation",
                        cohort_id=cohort_id,
                        scheduled_at=None,
                    ),
                )
        except OSError as exc:
            logger.warning("[DATA] stage=close_confirmation status=DEGRADED reason=%s", type(exc).__name__)
    unconfirmed = [str(df.at[i, "종목코드"]) for i in pending]
    unresolved_codes = [str(df.at[i, "종목코드"]) for i in unresolved]
    unconfirmed_picks = sorted(c for c in [*unconfirmed, *unresolved_codes] if c in pick_codes)
    if n_rows == 0:
        try:
            if trading_day_fn is not None:
                is_trading_day = bool(await trading_day_fn(snap))
            else:
                from src.data.trading_calendar import is_kis_trading_day

                is_trading_day = bool(await is_kis_trading_day(client, session, snap))
        except Exception:
            outcome, reason = RUN_OUTCOME_DEGRADED, "calendar_unavailable"
        else:
            outcome, reason = classify_finalize_outcome(
                n_rows, n_finalized, len(unconfirmed), unconfirmed_picks, is_trading_day=is_trading_day
            )
    else:
        outcome, reason = classify_finalize_outcome(
            n_rows, n_finalized, len(unconfirmed), unconfirmed_picks, is_trading_day=False
        )
    logger.info(
        "[DATA] stage=close_finalization date=%s n_finalized=%d n_unconfirmed=%d unconfirmed=%s outcome=%s reason=%s n_unresolved=%d",
        snap,
        n_finalized,
        len(unconfirmed),
        unconfirmed[:10],
        outcome,
        reason,
        len(unresolved_codes),
    )
    if on_outcome is not None:
        on_outcome(
            outcome,
            run_date=snap,
            reason=reason,
            metrics={
                "n_rows": n_rows,
                "n_finalized": n_finalized,
                "n_unconfirmed": len(unconfirmed),
                "n_unresolved": len(unresolved_codes),
                "unconfirmed_picks": unconfirmed_picks,
            },
        )
    return n_finalized


def load_pick_codes(snapshot_date: str) -> frozenset[str]:
    """Persisted top-k decision symbols as zero-filled codes."""
    picks = load_topk_decision(pd.Timestamp(snapshot_date))
    if picks.empty:
        return frozenset()
    return frozenset(picks["symbol"].astype(str).str.zfill(6))


async def _amain(args) -> int:
    """단일 이벤트 루프 안에서 세션 생성/토큰/확정/종료를 모두 수행한다. 종가 확정은 읽기 전용 시세 조회라 데이터 계좌 키로 수행하고 체결 계좌 키는 실주문 전용으로 둔다."""
    snap = args.date or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    session_day = resolve_session_day(date.fromisoformat(snap))
    if session_day.kind in (SessionKind.SHIFTED, SessionKind.UNKNOWN):
        gate = trading_session_gate(session_day)
        assert gate is not None
        record_run_outcome("finalize_close", RUN_OUTCOME_OK, run_date=snap, reason=gate)
        return 0
    owned_client = KisApiClient(**kis_data_client_kwargs())
    session = owned_client.create_session()
    try:
        await owned_client.ensure_token(session)
        capture_store = None
        run_id = None
        cohort_id = None
        try:
            from src.data.capture_store import CaptureStore as _Store
            from src.data.capture_store import resolve_capture_root

            _root = resolve_capture_root()
            _store = _Store(_root)
            _cohort = _store.read_cohort(snap, available_by=datetime.now(ZoneInfo("Asia/Seoul")))
            capture_store, run_id, cohort_id = _store, f"close-{snap}", _cohort.cohort_id
        except (FileNotFoundError, ValueError, OSError):
            capture_store, run_id, cohort_id = None, None, None
        n = await run_close_finalization(
            snapshot_date=snap,
            client=owned_client,
            session=session,
            retry_interval_seconds=args.retry_interval,
            pick_codes=load_pick_codes(snap),
            on_outcome=functools.partial(record_run_outcome, "finalize_close"),
            capture_store=capture_store,
            run_id=run_id,
            cohort_id=cohort_id,
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
    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    asyncio.run(_amain(args))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
