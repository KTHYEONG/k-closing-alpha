"""KCA-owned aftermarket order-book observation (dense rank pool, sparse full cohort)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from src import settings
from src.api.kis.key_pool import resolve_research_credentials, token_cache_path
from src.config.collection import CollectionSettings
from src.config.market_session import (
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    KRX_AFTERMARKET_HOUR_FLOOR,
    NXT_AFTERMARKET_HOUR_CEIL,
    NXT_AFTERMARKET_HOUR_FLOOR,
)
from src.data.capture_contracts import (
    GOOD_ENTRY_STATES,
    SEOUL,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.orderbook_store import append_orderbook_snapshots, build_orderbook_rows
from src.data.session_calendar import SessionKind, resolve_session_day
from src.data.trading_calendar import is_kis_trading_day
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

_VENUE_SESSION: dict[str, str] = {"KRX": INTRADAY_SESSION_KRX_AFTERMARKET, "NXT": INTRADAY_SESSION_NXT_AFTERMARKET}
_VENUE_DIV_CODE: dict[str, str] = {"KRX": "J", "NXT": "NX"}
_DENSE_REASON = "aftermarket-dense"
_SPARSE_REASON = "aftermarket-sparse"


@dataclass(frozen=True)
class BookRound:
    """One evening observation instant and the universe it sweeps."""

    scheduled_at: datetime
    kind: Literal["dense", "sparse"]


def _at_hhmmss(trading_day: date, hhmmss: str) -> datetime:
    return datetime(
        trading_day.year,
        trading_day.month,
        trading_day.day,
        int(hhmmss[0:2]),
        int(hhmmss[2:4]),
        int(hhmmss[4:6]),
        tzinfo=SEOUL,
    )


def aftermarket_book_rounds(
    trading_date: date, *, dense_seconds: int, sparse_times: Sequence[str]
) -> tuple[BookRound, ...]:
    """Build the evening observation schedule for aftermarket order books.

    Dense rounds run every dense_seconds from the NXT aftermarket open (exclusive) to its close
    (exclusive). A sparse instant replaces the dense round of the same minute, because the sparse sweep
    already covers the dense universe.

    Args:
        trading_date: KST trading date (STANDARD session).
        dense_seconds: Dense spacing in seconds.
        sparse_times: HHMMSS instants of full-cohort sweeps.

    Returns:
        Rounds sorted by scheduled_at.
    """
    if int(dense_seconds) <= 0:
        raise ValueError("dense_seconds must be positive")
    floor = _at_hhmmss(trading_date, NXT_AFTERMARKET_HOUR_FLOOR)
    ceil = _at_hhmmss(trading_date, NXT_AFTERMARKET_HOUR_CEIL)
    sparse_insts = sorted({_at_hhmmss(trading_date, str(item)) for item in sparse_times})
    sparse_minutes = {(item.hour, item.minute) for item in sparse_insts}
    rounds: list[BookRound] = [BookRound(scheduled_at=item, kind="sparse") for item in sparse_insts]
    moment = floor + timedelta(seconds=int(dense_seconds))
    while moment < ceil:
        if (moment.hour, moment.minute) not in sparse_minutes:
            rounds.append(BookRound(scheduled_at=moment, kind="dense"))
        moment += timedelta(seconds=int(dense_seconds))
    rounds.sort(key=lambda item: item.scheduled_at)
    return tuple(rounds)


def venues_for_round(scheduled_at: datetime) -> tuple[Literal["KRX", "NXT"], ...]:
    """Resolve which venues serve one observation instant.

    The KRX aftermarket book is only guaranteed live strictly after
    ``KRX_AFTERMARKET_HOUR_FLOOR``; the instant exactly at the floor still
    requests NXT only.
    """
    moment = scheduled_at.astimezone(SEOUL)
    floor = _at_hhmmss(moment.date(), KRX_AFTERMARKET_HOUR_FLOOR)
    if moment <= floor:
        return ("NXT",)
    return ("KRX", "NXT")


def _rank_pool_symbols(snapshot_date: str, rank_pool_path: Path | None) -> tuple[str, ...] | None:
    from src.config.base import RANK_POOL_PARQUET_NAME

    target = Path(rank_pool_path) if rank_pool_path is not None else Path(settings.PARQUET_DIR) / RANK_POOL_PARQUET_NAME
    if not target.exists():
        return None
    import pandas as pd

    frame = pd.read_parquet(target)
    if frame.empty:
        return None
    symbol_col = "symbol" if "symbol" in frame.columns else ("종목코드" if "종목코드" in frame.columns else None)
    if symbol_col is None or "decision_date" not in frame.columns:
        return None
    today = frame[frame["decision_date"].astype(str).str[:10] == snapshot_date]
    if today.empty:
        return None
    codes = sorted({str(item).strip().zfill(6) for item in today[symbol_col].astype(str).tolist() if str(item).strip()})
    return tuple(codes) if codes else None


def _position_symbols() -> tuple[str, ...]:
    try:
        from src.execution.paper_broker import PaperLedger

        frame = PaperLedger().load_open_positions()
    except Exception as exc:
        logger.warning("[DATA] stage=aftermarket_book status=paper_unavailable reason=%s", type(exc).__name__)
        return ()
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return ()
    return tuple(sorted({str(item).strip().zfill(6) for item in frame["symbol"].astype(str).tolist() if str(item).strip()}))


def resolve_book_universes(
    snapshot_date: str, *, store: CaptureStore, now: datetime, rank_pool_path: Path | None = None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve the dense (rank pool plus positions) and sparse (cohort plus dense) universes.

    Args:
        snapshot_date: KST trading date being observed.
        store: Capture store holding the day's declared cohort.
        now: Aware point-in-time cutoff for cohort visibility.
        rank_pool_path: Rank-pool parquet override (tests); None uses the production path.

    Returns:
        Sorted unique ``(dense_symbols, sparse_symbols)`` 6-char code tuples.
    """
    positions = _position_symbols()
    pool = _rank_pool_symbols(snapshot_date, rank_pool_path)
    if pool is None:
        logger.warning("[DATA] stage=aftermarket_book rank_pool=MISSING date=%s", snapshot_date)
        dense = positions
    else:
        dense = tuple(sorted(set(pool) | set(positions)))
    try:
        cohort = store.read_cohort(snapshot_date, available_by=now)
    except FileNotFoundError:
        logger.warning("[DATA] stage=aftermarket_book cohort=MISSING date=%s", snapshot_date)
        return dense, dense
    sparse = tuple(sorted({str(item).strip().zfill(6) for item in cohort.eligible_symbols if str(item).strip()} | set(dense)))
    return dense, sparse


async def _wait_until_scheduled(
    scheduled_at: datetime,
    *,
    now_fn: Callable[[], datetime],
    sleeper: Callable[[float], Awaitable[None]],
    injected_clock: bool,
) -> None:
    wait_seconds = (scheduled_at - now_fn()).total_seconds()
    if wait_seconds <= 0 or (injected_clock and sleeper is asyncio.sleep):
        return
    await sleeper(wait_seconds)


def _is_nxt_listed(payload: dict[str, Any]) -> bool:
    """Return True when an NX book response carries a live exchange acceptance time."""
    output1 = payload.get("output1")
    if not isinstance(output1, dict):
        return False
    return bool(str(output1.get("aspr_acpt_hour") or "").strip())


async def run_aftermarket_book_capture(
    snapshot_date: str,
    *,
    profile: CollectionSettings,
    store: CaptureStore,
    clients: Sequence[Any],
    dense_symbols: Sequence[str],
    sparse_symbols: Sequence[str],
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
) -> tuple[CaptureManifest, ...]:
    """Observe KRX and NXT aftermarket order books on a fixed evening schedule without placing orders.

    Every response is retained verbatim as raw evidence and as a normalized row (exchange acceptance time
    output1.aspr_acpt_hour is the point-in-time key for research, not the request time). NXT listing is
    learned from the first NX response per symbol: a null acceptance time marks the symbol NOT_APPLICABLE
    for NX for the rest of the evening. A request not started before the next round's instant is recorded
    as a deadline miss instead of drifting the schedule.

    Args:
        snapshot_date: KST trading date being observed.
        profile: Schedule, flush and concurrency limits.
        store: Capture store for raw responses and manifests.
        clients: Token-ready KIS clients bound to the aftermarket book slots.
        dense_symbols: Dense universe (rank pool and open positions).
        sparse_symbols: Sparse universe (full cohort and dense universe).
        now_fn: Aware clock; None uses Asia/Seoul now.
        sleep_fn: Wait primitive; None uses asyncio.sleep.

    Returns:
        Published manifests, one per (flush block, venue).

    Raises:
        ValueError: Empty clients or invalid snapshot_date.
        OSError: Raw, partition or manifest persistence fails.
    """
    now_clock = now_fn or (lambda: datetime.now(SEOUL))
    sleeper = sleep_fn or asyncio.sleep
    injected_clock = now_fn is not None
    try:
        trading_day = date.fromisoformat(snapshot_date)
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    if not clients:
        raise ValueError("aftermarket book capture requires prewarmed data clients")
    dense = tuple(sorted({str(item).strip().zfill(6) for item in dense_symbols if str(item).strip()}))
    sparse = tuple(sorted({str(item).strip().zfill(6) for item in sparse_symbols if str(item).strip()}))
    rounds = aftermarket_book_rounds(
        trading_day,
        dense_seconds=int(profile.COLLECTION_AFTERMARKET_BOOK_DENSE_SECONDS),
        sparse_times=tuple(profile.COLLECTION_AFTERMARKET_BOOK_SPARSE_TIMES),
    )
    flush_rounds = int(profile.COLLECTION_AFTERMARKET_BOOK_FLUSH_ROUNDS)
    ceil = _at_hhmmss(trading_day, NXT_AFTERMARKET_HOUR_CEIL)
    start_now = now_clock()
    missed: set[datetime] = {item.scheduled_at for item in rounds if item.scheduled_at < start_now}
    nx_unlisted: set[str] = set()
    nx_listing_evidence: dict[str, tuple[Any, datetime | None, datetime | None]] = {}
    manifests: list[CaptureManifest] = []
    buffer: list[dict] = []
    rows_total = 0
    sem = asyncio.Semaphore(int(profile.COLLECTION_CONCURRENCY_PER_KEY) * len(clients))

    async def _send(
        broker_session: Any,
        client: Any,
        symbol: str,
        venue: str,
        round_kind: Literal["dense", "sparse"],
        scheduled_at: datetime,
        deadline: datetime,
        round_index: int,
        run_id: str,
    ) -> tuple[CoverageEntry, list[dict]]:
        started = now_clock()
        if symbol in nx_unlisted and venue == "NXT":
            ref, first, last = nx_listing_evidence[symbol]
            return CoverageEntry(
                symbol=symbol,
                dataset=CaptureDataset.ORDERBOOK,
                venue=venue,
                session=_VENUE_SESSION[venue],
                scheduled_at=scheduled_at,
                status=CaptureStatus.NOT_APPLICABLE,
                rows=0,
                first_event_time=first,
                last_event_time=last,
                reason="nxt_not_listed",
                raw_refs=(ref,),
            ), []
        if started >= deadline:
            return CoverageEntry(
                symbol=symbol,
                dataset=CaptureDataset.ORDERBOOK,
                venue=venue,
                session=_VENUE_SESSION[venue],
                scheduled_at=scheduled_at,
                status=CaptureStatus.PARTIAL,
                rows=0,
                first_event_time=None,
                last_event_time=None,
                reason="deadline",
                raw_refs=(),
            ), []
        reason = _DENSE_REASON if round_kind == "dense" else _SPARSE_REASON
        async with sem:
            try:
                payload = await client.get_orderbook_snapshot(
                    broker_session, symbol, market_div_code=_VENUE_DIV_CODE[venue]
                )
            except Exception:
                received = now_clock()
                return CoverageEntry(
                    symbol=symbol,
                    dataset=CaptureDataset.ORDERBOOK,
                    venue=venue,
                    session=_VENUE_SESSION[venue],
                    scheduled_at=scheduled_at,
                    status=CaptureStatus.PARTIAL,
                    rows=0,
                    first_event_time=started,
                    last_event_time=received,
                    reason="vendor_failure",
                    raw_refs=(),
                ), []
            received = now_clock()
        body = dict(payload) if isinstance(payload, dict) else None
        ok = isinstance(body, dict) and body.get("rt_cd") == "0"
        if ok and venue == "NXT" and not _is_nxt_listed(body):
            nx_unlisted.add(symbol)
        context = CaptureContext(
            trading_date=trading_day,
            run_id=run_id,
            dataset=CaptureDataset.ORDERBOOK,
            vendor="kis",
            endpoint="inquire-asking-price-exp-ccn",
            symbol=symbol,
            venue=venue,
            session=_VENUE_SESSION[venue],
            capture_reason=reason,
            cohort_id=None,
            scheduled_at=scheduled_at,
        )
        ref = store.append_response(
            CapturedResponse(
                context=context,
                request_started_at=started,
                received_at=received,
                payload=body,
                status=CaptureStatus.COMPLETE if ok else CaptureStatus.FAILED,
                source_timestamp=None,
                source_published_at=None,
                page_index=round_index,
                attempt_index=0,
                continuation={},
                error_type=None if ok else "vendor_failure",
            )
        )
        if ok and venue == "NXT" and symbol in nx_unlisted and symbol not in nx_listing_evidence:
            nx_listing_evidence[symbol] = (ref, started, received)
            return CoverageEntry(
                symbol=symbol,
                dataset=CaptureDataset.ORDERBOOK,
                venue=venue,
                session=_VENUE_SESSION[venue],
                scheduled_at=scheduled_at,
                status=CaptureStatus.NOT_APPLICABLE,
                rows=0,
                first_event_time=started,
                last_event_time=received,
                reason="nxt_not_listed",
                raw_refs=(ref,),
            ), []
        rows = (
            build_orderbook_rows(body or {}, symbol, venue, reason, received, scheduled_at=scheduled_at, request_started_at=started)
            if ok
            else []
        )
        return CoverageEntry(
            symbol=symbol,
            dataset=CaptureDataset.ORDERBOOK,
            venue=venue,
            session=_VENUE_SESSION[venue],
            scheduled_at=scheduled_at,
            status=CaptureStatus.COMPLETE if ok else CaptureStatus.PARTIAL,
            rows=len(rows),
            first_event_time=started,
            last_event_time=received,
            reason=reason if ok else "vendor_failure",
            raw_refs=(ref,),
        ), rows

    async with clients[0].create_session() as broker_session:
        for block, block_start in enumerate(range(0, len(rounds), max(flush_rounds, 1))):
            block_rounds = rounds[block_start : block_start + max(flush_rounds, 1)]
            block_run_ids = {
                venue: f"aftermarket-book-{snapshot_date}-b{block}-{venue.lower()}-{uuid.uuid4().hex[:8]}"
                for venue in ("KRX", "NXT")
            }
            per_venue_entries: dict[str, dict[str, list[CoverageEntry]]] = {"KRX": {}, "NXT": {}}
            for offset, book_round in enumerate(block_rounds):
                round_index = block_start + offset
                universe = sparse if book_round.kind == "sparse" else dense
                venues = venues_for_round(book_round.scheduled_at)
                if book_round.scheduled_at in missed:
                    for venue in venues:
                        for symbol in universe:
                            per_venue_entries[venue].setdefault(symbol, []).append(
                                CoverageEntry(
                                    symbol=symbol,
                                    dataset=CaptureDataset.ORDERBOOK,
                                    venue=venue,
                                    session=_VENUE_SESSION[venue],
                                    scheduled_at=book_round.scheduled_at,
                                    status=CaptureStatus.PARTIAL,
                                    rows=0,
                                    first_event_time=None,
                                    last_event_time=None,
                                    reason="missed_start",
                                    raw_refs=(),
                                )
                            )
                    continue
                await _wait_until_scheduled(
                    book_round.scheduled_at, now_fn=now_clock, sleeper=sleeper, injected_clock=injected_clock
                )
                if offset + 1 < len(block_rounds):
                    deadline = block_rounds[offset + 1].scheduled_at
                elif block_start + len(block_rounds) < len(rounds):
                    deadline = rounds[block_start + len(block_rounds)].scheduled_at
                else:
                    deadline = ceil
                coros: list[Awaitable[tuple[CoverageEntry, list[dict]]]] = []
                keys: list[tuple[str, str]] = []
                position = 0
                for venue in venues:
                    for symbol in universe:
                        client = clients[position % len(clients)]
                        position += 1
                        keys.append((venue, symbol))
                        coros.append(
                            _send(
                                broker_session,
                                client,
                                symbol,
                                venue,
                                book_round.kind,
                                book_round.scheduled_at,
                                deadline,
                                round_index,
                                block_run_ids[venue],
                            )
                        )
                for (venue, symbol), (entry, rows) in zip(keys, await asyncio.gather(*coros)):
                    per_venue_entries[venue].setdefault(symbol, []).append(entry)
                    buffer.extend(rows)
            append_orderbook_snapshots(buffer, snapshot_date, session="aftermarket")
            rows_total += len(buffer)
            buffer = []
            completed_at = now_clock()
            ok = late = failed = unlisted = 0
            for venue in ("KRX", "NXT"):
                by_symbol = per_venue_entries[venue]
                if not by_symbol:
                    continue
                entries: list[CoverageEntry] = []
                for symbol in sorted(by_symbol):
                    parts = by_symbol[symbol]
                    if len(parts) == 1:
                        merged = parts[0]
                    elif any(item.status == CaptureStatus.NOT_APPLICABLE for item in parts):
                        listed = next(item for item in parts if item.status == CaptureStatus.NOT_APPLICABLE)
                        merged = listed
                    elif any(item.reason == "missed_start" for item in parts if item.status == CaptureStatus.PARTIAL):
                        merged = _merge_block_entries(symbol, venue, parts, "missed_start")
                    elif any(item.reason == "deadline" for item in parts if item.status == CaptureStatus.PARTIAL):
                        merged = _merge_block_entries(symbol, venue, parts, "deadline")
                    elif any(item.status == CaptureStatus.PARTIAL for item in parts):
                        merged = _merge_block_entries(symbol, venue, parts, "vendor_failure")
                    else:
                        merged = _merge_block_entries(symbol, venue, parts, "aftermarket-book")
                    entries.append(merged)
                    if merged.status == CaptureStatus.COMPLETE:
                        ok += 1
                    elif merged.status == CaptureStatus.NOT_APPLICABLE:
                        unlisted += 1
                    elif merged.reason == "deadline":
                        late += 1
                    else:
                        failed += 1
                status = (
                    CaptureStatus.COMPLETE
                    if all(item.status in GOOD_ENTRY_STATES for item in entries)
                    else CaptureStatus.PARTIAL
                )
                manifest = CaptureManifest(
                    schema_version=1,
                    context=CaptureContext(
                        trading_date=trading_day,
                        run_id=block_run_ids[venue],
                        dataset=CaptureDataset.ORDERBOOK,
                        vendor="kis",
                        endpoint="aftermarket-book",
                        symbol=None,
                        venue="owner-local",
                        session=_VENUE_SESSION[venue],
                        capture_reason="aftermarket-book",
                        cohort_id=None,
                        scheduled_at=None,
                    ),
                    cohort=None,
                    completed_at=completed_at,
                    entries=tuple(entries),
                    artifacts=tuple(ref for item in entries for ref in item.raw_refs),
                    status=status,
                )
                store.publish_manifest(manifest)
                manifests.append(manifest)
            logger.info(
                "[DATA] stage=aftermarket_book block=%d rounds=%d ok=%d late=%d failed=%d nx_unlisted=%d rows_total=%d",
                block,
                len(block_rounds),
                ok,
                late,
                failed,
                unlisted,
                rows_total,
            )
    append_orderbook_snapshots(buffer, snapshot_date, session="aftermarket")
    return tuple(manifests)


def _merge_block_entries(
    symbol: str, venue: str, parts: Sequence[CoverageEntry], reason: str
) -> CoverageEntry:
    """Fold one symbol's per-request outcomes in a flush block into a single entry."""
    rows = sum(item.rows for item in parts)
    refs = tuple(ref for item in parts for ref in item.raw_refs)
    starts = [item.first_event_time for item in parts if item.first_event_time is not None]
    ends = [item.last_event_time for item in parts if item.last_event_time is not None]
    stamp = parts[0].scheduled_at
    complete = reason == "aftermarket-book"
    return CoverageEntry(
        symbol=symbol,
        dataset=CaptureDataset.ORDERBOOK,
        venue=venue,
        session=_VENUE_SESSION[venue],
        scheduled_at=stamp,
        status=CaptureStatus.COMPLETE if complete else CaptureStatus.PARTIAL,
        rows=rows,
        first_event_time=min(starts) if starts else None,
        last_event_time=max(ends) if ends else None,
        reason=reason,
        raw_refs=refs,
    )


async def _run_async(snapshot_date: str, profile: CollectionSettings) -> tuple[CaptureManifest, ...] | None:
    """Run the evening capture; ``None`` when KIS reports ``snapshot_date`` as a market holiday."""
    from src.api.kis.client import KisApiClient

    trading_day = date.fromisoformat(snapshot_date)
    session_day = resolve_session_day(trading_day, overrides=profile.COLLECTION_SESSION_OVERRIDES)
    if session_day.kind is SessionKind.CLOSED:
        logger.info("[DATA] stage=aftermarket_book status=SKIP reason=non_trading_day date=%s", snapshot_date)
        return None
    env = dict(os.environ)
    creds = resolve_research_credentials(env, slots=tuple(profile.COLLECTION_AFTERMARKET_BOOK_SLOTS))
    clients = [
        KisApiClient(
            cred.app_key,
            cred.app_secret,
            "",
            cred.hts_id,
            token_file=str(token_cache_path(cred.app_key, settings.KIS_TOKEN_CACHE_DIR)),
        )
        for cred in creds
    ]
    store = CaptureStore(_capture_root(profile))
    async with clients[0].create_session() as broker_session:
        for client in clients:
            await client.ensure_token(broker_session)
        if not await is_kis_trading_day(clients[0], broker_session, snapshot_date):
            return None
    dense_symbols, sparse_symbols = resolve_book_universes(snapshot_date, store=store, now=datetime.now(SEOUL))
    return await run_aftermarket_book_capture(
        snapshot_date,
        profile=profile,
        store=store,
        clients=clients,
        dense_symbols=dense_symbols,
        sparse_symbols=sparse_symbols,
    )


def main(argv: list[str] | None = None) -> None:
    """Observe aftermarket order books without transmitting orders.

    Args:
        argv: Optional --date ISO-date argument.

    Raises:
        ValueError: Invalid date or uncertified configuration.
        OSError: Required evidence persistence fails.
    """
    parser = argparse.ArgumentParser(description="KCA aftermarket order-book capture (research, no orders)")
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)
    profile = CollectionSettings()
    if not profile.COLLECTION_AFTERMARKET_BOOK_ENABLED:
        logger.info("[DATA] stage=aftermarket_book status=SKIP reason=disabled")
        return
    snapshot_date = args.date or datetime.now(SEOUL).date().isoformat()
    try:
        trading_day = date.fromisoformat(snapshot_date)
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    if trading_day.weekday() >= 5:
        logger.info("[DATA] stage=aftermarket_book status=SKIP reason=holiday date=%s", snapshot_date)
        return
    session_day = resolve_session_day(trading_day, overrides=profile.COLLECTION_SESSION_OVERRIDES)
    if session_day.kind is SessionKind.CLOSED:
        logger.info("[DATA] stage=aftermarket_book status=SKIP reason=non_trading_day date=%s", snapshot_date)
        return
    if session_day.kind is not SessionKind.STANDARD:
        logger.info("[DATA] stage=aftermarket_book status=SKIP reason=non_standard_session date=%s", snapshot_date)
        return
    manifests = asyncio.run(_run_async(snapshot_date, profile))
    if manifests is None:
        logger.info("[DATA] stage=aftermarket_book status=SKIP reason=non_trading_day date=%s", snapshot_date)
        return
    logger.info(
        "[DATA] stage=aftermarket_book status=COMPLETE date=%s manifests=%d",
        snapshot_date,
        len(manifests),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
