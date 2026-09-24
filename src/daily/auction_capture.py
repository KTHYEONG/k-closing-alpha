"""KCA-owned closing/opening auction observation sweeps."""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src import settings
from src.api.kis.key_pool import resolve_research_credentials, token_cache_path
from src.config.collection import CollectionSettings
from src.data.capture_contracts import (
    SEOUL,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
    SessionClock,
)
from src.data.capture_store import CaptureStore
from src.data.trading_calendar import is_kis_trading_day

logger = logging.getLogger(__name__)

_OPEN_OFFSET_MINUTES: tuple[int, ...] = (-20, -10, -5, -2, -1)
_PROGRAM_OFFSET_MINUTES: tuple[int, ...] = (-8, -4)
_OPEN_TRUST_FLOOR_SECONDS: int = 30

_GOOD_ENTRY_STATES = frozenset({CaptureStatus.COMPLETE, CaptureStatus.NO_TRADES, CaptureStatus.NOT_APPLICABLE})


def _capture_root(profile: CollectionSettings) -> Path:
    if profile.COLLECTION_ROOT is not None:
        return Path(profile.COLLECTION_ROOT)
    return Path(settings.HISTORY_DIR) / "capture"


def _previous_trading_day(snapshot_date: str) -> str:
    try:
        day = date.fromisoformat(snapshot_date)
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.isoformat()


def _open_position_symbols() -> list[str]:
    try:
        from src.execution.paper_broker import PaperLedger

        frame = PaperLedger().load_open_positions()
    except Exception as exc:
        logger.warning("[DATA] stage=auction_cohort status=paper_unavailable reason=%s", type(exc).__name__)
        return []
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return []
    return sorted({str(item) for item in frame["symbol"].astype(str).tolist() if str(item).strip()})


def _close_rounds(clock: SessionClock, interval_seconds: int) -> list[datetime]:
    start = clock.close_at - timedelta(minutes=9)
    rounds: list[datetime] = []
    moment = start
    while moment < clock.close_at:
        rounds.append(moment)
        moment += timedelta(seconds=int(interval_seconds))
    return rounds


def _open_rounds(clock: SessionClock) -> list[datetime]:
    return [clock.open_at + timedelta(minutes=offset) for offset in _OPEN_OFFSET_MINUTES]


def _program_rounds(clock: SessionClock) -> list[datetime]:
    return [clock.close_at + timedelta(minutes=offset) for offset in _PROGRAM_OFFSET_MINUTES]


def _resolve_roster(
    snapshot_date: str,
    phase: str,
    store: CaptureStore,
    now: datetime,
    previous_trading_day: str | None = None,
) -> tuple[list[str], str | None, bool]:
    if phase == "close":
        try:
            cohort = store.read_cohort(snapshot_date, available_by=now)
        except FileNotFoundError as exc:
            raise RuntimeError(f"auction cohort cannot be verified: {snapshot_date!r}") from exc
        return [str(item) for item in cohort.eligible_symbols], cohort.cohort_id, False
    prev_day = previous_trading_day or _previous_trading_day(snapshot_date)
    try:
        prev_cohort = store.read_cohort(prev_day, available_by=now)
    except FileNotFoundError:
        logger.warning("[DATA] stage=auction_cohort status=INCOMPLETE reason=missing_previous date=%s prev=%s", snapshot_date, prev_day)
        positions = _open_position_symbols()
        return positions, None, True
    codes: list[str] = [str(item) for item in prev_cohort.eligible_symbols]
    for item in _open_position_symbols():
        if item not in codes:
            codes.append(item)
    return codes, prev_cohort.cohort_id, False


def _extract_open_price(payload: dict[str, Any] | None) -> int:
    if not isinstance(payload, dict):
        return 0
    output = payload.get("output")
    if not isinstance(output, dict):
        return 0
    try:
        return int(float(str(output.get("stck_oprc", "0") or "0").replace(",", "")))
    except (ValueError, TypeError):
        return 0


def _fragment_frame(symbol: str, scheduled_at: datetime, received_at: datetime, payload: dict[str, Any] | None) -> pd.DataFrame:
    import json

    def _encode(value: Any) -> str | None:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)

    output1: str | None = _encode(payload.get("output1")) if isinstance(payload, dict) and "output1" in payload else None
    output2: str | None = _encode(payload.get("output2")) if isinstance(payload, dict) and "output2" in payload else None
    return pd.DataFrame([{"symbol": symbol, "scheduled_at": scheduled_at, "received_at": received_at, "output1": output1, "output2": output2}])


async def _wait_until_scheduled(
    scheduled_at: datetime,
    *,
    now_fn: Callable[[], datetime],
    sleeper: Callable[[float], Awaitable[None]],
    injected_clock: bool,
) -> None:
    """Wait for a scheduled observation, leaving injected clocks under test control."""
    wait_seconds = (scheduled_at - now_fn()).total_seconds()
    if wait_seconds <= 0 or (injected_clock and sleeper is asyncio.sleep):
        return
    await sleeper(wait_seconds)


async def _capture_program_rounds(
    *,
    session: Any,
    rounds: Sequence[datetime],
    roster: Sequence[str],
    clients: Sequence[Any],
    semaphore: asyncio.Semaphore,
    store: CaptureStore,
    trading_day: date,
    run_id: str,
    cohort_id: str | None,
    reason_tag: str,
    close_at: datetime,
    now_fn: Callable[[], datetime],
    sleeper: Callable[[float], Awaitable[None]],
    injected_clock: bool,
) -> list[CoverageEntry]:
    """Capture program observations at their declared pre-close timestamps."""
    entries: list[CoverageEntry] = []
    for prog_index, scheduled_at in enumerate(rounds):
        await _wait_until_scheduled(
            scheduled_at, now_fn=now_fn, sleeper=sleeper, injected_clock=injected_clock
        )
        deadline = rounds[prog_index + 1] if prog_index + 1 < len(rounds) else close_at
        for position, symbol in enumerate(roster):
            client = clients[position % len(clients)]
            started = now_fn()
            if started >= deadline:
                entries.append(
                    CoverageEntry(
                        symbol=symbol,
                        dataset=CaptureDataset.PROGRAM,
                        venue="KRX",
                        session="regular",
                        scheduled_at=scheduled_at,
                        status=CaptureStatus.PARTIAL,
                        rows=0,
                        first_event_time=None,
                        last_event_time=None,
                        reason="deadline",
                        raw_refs=(),
                    )
                )
                continue
            async with semaphore:
                try:
                    payload = await client.get_program_net_buy(session, symbol, market_div_code="J")
                except Exception as exc:
                    received = now_fn()
                    entries.append(
                        CoverageEntry(
                            symbol=symbol,
                            dataset=CaptureDataset.PROGRAM,
                            venue="KRX",
                            session="regular",
                            scheduled_at=scheduled_at,
                            status=CaptureStatus.FAILED,
                            rows=0,
                            first_event_time=started,
                            last_event_time=received,
                            reason=type(exc).__name__,
                            raw_refs=(),
                        )
                    )
                    continue
                received = now_fn()
            body = dict(payload) if isinstance(payload, dict) else None
            ok = isinstance(body, dict) and body.get("rt_cd") == "0"
            context = CaptureContext(
                trading_date=trading_day,
                run_id=run_id,
                dataset=CaptureDataset.PROGRAM,
                vendor="kis",
                endpoint="program-trade-by-stock",
                symbol=symbol,
                venue="KRX",
                session="regular",
                capture_reason=reason_tag,
                cohort_id=cohort_id,
                scheduled_at=scheduled_at,
            )
            try:
                ref = store.append_response(
                    CapturedResponse(
                        context=context,
                        request_started_at=started,
                        received_at=received,
                        payload=body,
                        status=CaptureStatus.COMPLETE if ok else CaptureStatus.FAILED,
                        source_timestamp=None,
                        source_published_at=None,
                        page_index=1000 + prog_index,
                        attempt_index=0,
                        continuation={},
                        error_type=None if ok else "vendor_failure",
                    )
                )
            except OSError:
                entries.append(
                    CoverageEntry(
                        symbol=symbol,
                        dataset=CaptureDataset.PROGRAM,
                        venue="KRX",
                        session="regular",
                        scheduled_at=scheduled_at,
                        status=CaptureStatus.FAILED,
                        rows=0,
                        first_event_time=started,
                        last_event_time=received,
                        reason="persistence",
                        raw_refs=(),
                    )
                )
                continue
            frame_status = CaptureStatus.COMPLETE if ok else CaptureStatus.FAILED
            try:
                frame_context = context.model_copy(update={"run_id": f"{run_id}-g{prog_index}-{position}"})
                store.publish_frame(_fragment_frame(symbol, scheduled_at, received, body), context=frame_context)
            except OSError:
                frame_status = CaptureStatus.PARTIAL
            entries.append(
                CoverageEntry(
                    symbol=symbol,
                    dataset=CaptureDataset.PROGRAM,
                    venue="KRX",
                    session="regular",
                    scheduled_at=scheduled_at,
                    status=frame_status,
                    rows=1 if ok else 0,
                    first_event_time=started,
                    last_event_time=received,
                    reason=reason_tag if frame_status == CaptureStatus.COMPLETE else ("persistence" if ok else "vendor_failure"),
                    raw_refs=(ref,),
                )
            )
    return entries


async def run_auction_capture(
    snapshot_date: str,
    *,
    phase: Literal["close", "open"],
    profile: CollectionSettings,
    store: CaptureStore,
    clients: Sequence[Any],
    session_clock: SessionClock,
    now_fn: Callable[[], datetime] | None = None,
    sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    previous_trading_day: str | None = None,
) -> CaptureManifest:
    """Capture the complete project cohort through independently timed auctions.

    Sweeps are observations across time, not simultaneous market cross sections.
    Opening follow-up uses the previous actual trading day's population even
    when a stock is absent from today's candidates or another collector's feed.

    Args:
        snapshot_date: Actual date of the observed session.
        phase: Closing or opening observation phase.
        profile: Explicit key/call/deadline configuration.
        store: First-party raw and task evidence storage.
        clients: Exclusively qualified, prewarmed data clients.
        session_clock: Verified standard or exceptional dated market clock.
        now_fn: Actual aware clock; None uses Asia/Seoul now.
        sleep_fn: Bounded scheduler wait; None uses asyncio.sleep.

    Returns:
        Manifest retaining all expected entries and observed task states.

    Raises:
        ValueError: Wrong date/phase, clock, or profile.
        OSError: Required raw/coverage persistence fails.
        RuntimeError: Calendar or acquisition infrastructure cannot be verified.
    """
    now_clock = now_fn or (lambda: datetime.now(SEOUL))
    sleeper = sleep_fn or asyncio.sleep
    injected_clock = now_fn is not None
    try:
        trading_day = date.fromisoformat(snapshot_date)
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    if phase not in ("close", "open"):
        raise ValueError(f"unknown phase {phase!r}")
    if session_clock.trading_date != trading_day:
        raise ValueError("session clock trading_date must equal snapshot_date")
    if not profile.COLLECTION_RAW_ENABLED or not profile.COLLECTION_AUCTION_ENABLED:
        raise ValueError("auction capture requires enabled raw and auction collection")
    if not profile.COLLECTION_RESEARCH_SLOTS:
        raise ValueError("auction capture requires declared research slots")
    if not clients:
        raise ValueError("auction capture requires prewarmed data clients")
    now = now_clock()
    roster, cohort_id, cohort_incomplete = _resolve_roster(snapshot_date, phase, store, now, previous_trading_day)
    if phase == "close":
        rounds = _close_rounds(session_clock, int(profile.COLLECTION_AUCTION_INTERVAL_SECONDS))
        phase_end = session_clock.close_at
    else:
        rounds = _open_rounds(session_clock)
        phase_end = session_clock.open_at + timedelta(seconds=int(profile.COLLECTION_OPEN_CONFIRM_SECONDS))
    run_id = f"auction-{phase}-{snapshot_date}-{uuid.uuid4().hex[:8]}"
    reason_tag = f"auction-{phase}"
    sem = asyncio.Semaphore(int(profile.COLLECTION_CONCURRENCY_PER_KEY))
    entries: list[CoverageEntry] = []
    incomplete = bool(cohort_incomplete)
    first_client: Any = clients[0]

    async with first_client.create_session() as broker_session:
        program_task: asyncio.Task[list[CoverageEntry]] | None = None
        if phase == "close":
            program_task = asyncio.create_task(
                _capture_program_rounds(
                    session=broker_session,
                    rounds=_program_rounds(session_clock),
                    roster=roster,
                    clients=clients,
                    semaphore=sem,
                    store=store,
                    trading_day=trading_day,
                    run_id=run_id,
                    cohort_id=cohort_id,
                    reason_tag=reason_tag,
                    close_at=session_clock.close_at,
                    now_fn=now_clock,
                    sleeper=sleeper,
                    injected_clock=injected_clock,
                )
            )
        for index, scheduled_at in enumerate(rounds):
            await _wait_until_scheduled(
                scheduled_at, now_fn=now_clock, sleeper=sleeper, injected_clock=injected_clock
            )
            deadline = rounds[index + 1] if index + 1 < len(rounds) else phase_end
            if now_clock() >= deadline:
                logger.warning(
                    "[DATA] stage=auction_capture status=DEADLINE_EXCEEDED phase=%s round=%s n_missing=%d",
                    phase,
                    scheduled_at.isoformat(),
                    len(roster),
                )
                incomplete = True
                entries.extend(
                    CoverageEntry(
                        symbol=symbol,
                        dataset=CaptureDataset.ORDERBOOK,
                        venue="KRX",
                        session="regular",
                        scheduled_at=scheduled_at,
                        status=CaptureStatus.PARTIAL,
                        rows=0,
                        first_event_time=None,
                        last_event_time=None,
                        reason="deadline",
                        raw_refs=(),
                    )
                    for symbol in roster
                )
                continue
            for position, symbol in enumerate(roster):
                client: Any = clients[position % len(clients)]
                started = now_clock()
                remaining = (deadline - started).total_seconds()
                if remaining <= 0:
                    incomplete = True
                    entries.append(
                        CoverageEntry(
                            symbol=symbol,
                            dataset=CaptureDataset.ORDERBOOK,
                            venue="KRX",
                            session="regular",
                            scheduled_at=scheduled_at,
                            status=CaptureStatus.PARTIAL,
                            rows=0,
                            first_event_time=None,
                            last_event_time=None,
                            reason="deadline",
                            raw_refs=(),
                        )
                    )
                    continue
                async with sem:
                    try:
                        payload = await client.get_orderbook_snapshot(broker_session, symbol, market_div_code="J")
                    except Exception as exc:
                        received = now_clock()
                        incomplete = True
                        entries.append(
                            CoverageEntry(
                                symbol=symbol,
                                dataset=CaptureDataset.ORDERBOOK,
                                venue="KRX",
                                session="regular",
                                scheduled_at=scheduled_at,
                                status=CaptureStatus.FAILED,
                                rows=0,
                                first_event_time=started,
                                last_event_time=received,
                                reason=type(exc).__name__,
                                raw_refs=(),
                            )
                        )
                        continue
                    received = now_clock()
                body = dict(payload) if isinstance(payload, dict) else None
                ok = isinstance(body, dict) and body.get("rt_cd") == "0"
                context = CaptureContext(
                    trading_date=trading_day,
                    run_id=run_id,
                    dataset=CaptureDataset.ORDERBOOK,
                    vendor="kis",
                    endpoint="inquire-asking-price-exp-ccn",
                    symbol=symbol,
                    venue="KRX",
                    session="regular",
                    capture_reason=reason_tag,
                    cohort_id=cohort_id,
                    scheduled_at=scheduled_at,
                )
                try:
                    ref = store.append_response(
                        CapturedResponse(
                            context=context,
                            request_started_at=started,
                            received_at=received,
                            payload=body,
                            status=CaptureStatus.COMPLETE if ok else CaptureStatus.FAILED,
                            source_timestamp=None,
                            source_published_at=None,
                            page_index=index,
                            attempt_index=0,
                            continuation={},
                            error_type=None if ok else "vendor_failure",
                        )
                    )
                except OSError:
                    incomplete = True
                    entries.append(
                        CoverageEntry(
                            symbol=symbol,
                            dataset=CaptureDataset.ORDERBOOK,
                            venue="KRX",
                            session="regular",
                            scheduled_at=scheduled_at,
                            status=CaptureStatus.FAILED,
                            rows=0,
                            first_event_time=started,
                            last_event_time=received,
                            reason="persistence",
                            raw_refs=(),
                        )
                    )
                    continue
                late = received > deadline
                if late:
                    incomplete = True
                    logger.warning(
                        "[DATA] stage=auction_capture status=LATE phase=%s symbol=%s scheduled=%s received=%s",
                        phase,
                        symbol,
                        scheduled_at.isoformat(),
                        received.isoformat(),
                    )
                try:
                    frame_context = context.model_copy(update={"run_id": f"{run_id}-r{index}-{position}"})
                    store.publish_frame(_fragment_frame(symbol, scheduled_at, received, body), context=frame_context)
                except OSError:
                    incomplete = True
                entries.append(
                    CoverageEntry(
                        symbol=symbol,
                        dataset=CaptureDataset.ORDERBOOK,
                        venue="KRX",
                        session="regular",
                        scheduled_at=scheduled_at,
                        status=CaptureStatus.COMPLETE if ok and not late else CaptureStatus.PARTIAL,
                        rows=1 if ok else 0,
                        first_event_time=started,
                        last_event_time=received,
                        reason=reason_tag if ok and not late else ("deadline" if late else "vendor_failure"),
                        raw_refs=(ref,),
                    )
                )
        if phase == "close":
            if program_task is not None:
                program_entries = await program_task
                entries.extend(program_entries)
                if any(entry.status not in _GOOD_ENTRY_STATES for entry in program_entries):
                    incomplete = True
        else:
            floor = session_clock.open_at + timedelta(seconds=_OPEN_TRUST_FLOOR_SECONDS)
            confirm_end = session_clock.open_at + timedelta(seconds=int(profile.COLLECTION_OPEN_CONFIRM_SECONDS))
            for position, symbol in enumerate(roster):
                client = clients[position % len(clients)]
                wait = (floor - now_clock()).total_seconds()
                if wait > 0:
                    await sleeper(wait)
                observed: dict[str, Any] | None = None
                observed_at: datetime | None = None
                poll_started = now_clock()
                attempt = 0
                while now_clock() <= confirm_end:
                    async with sem:
                        payload = await client.get_current_price(broker_session, symbol, market_div_code="J")
                    received = now_clock()
                    body = dict(payload) if isinstance(payload, dict) else None
                    price = _extract_open_price(body)
                    context = CaptureContext(
                        trading_date=trading_day,
                        run_id=run_id,
                        dataset=CaptureDataset.PRICE,
                        vendor="kis",
                        endpoint="inquire-price",
                        symbol=symbol,
                        venue="KRX",
                        session="regular",
                        capture_reason=reason_tag,
                        cohort_id=cohort_id,
                        scheduled_at=floor,
                    )
                    try:
                        ref = store.append_response(
                            CapturedResponse(
                                context=context,
                                request_started_at=poll_started,
                                received_at=received,
                                payload=body,
                                status=CaptureStatus.COMPLETE if price > 0 else CaptureStatus.FAILED,
                                source_timestamp=None,
                                source_published_at=None,
                                page_index=position,
                                attempt_index=attempt,
                                continuation={},
                                error_type=None if price > 0 else "open_unresolved",
                            )
                        )
                    except OSError:
                        incomplete = True
                        entries.append(
                            CoverageEntry(
                                symbol=symbol,
                                dataset=CaptureDataset.PRICE,
                                venue="KRX",
                                session="regular",
                                scheduled_at=floor,
                                status=CaptureStatus.FAILED,
                                rows=0,
                                first_event_time=poll_started,
                                last_event_time=received,
                                reason="persistence",
                                raw_refs=(),
                            )
                        )
                        observed = None
                        observed_at = None
                        break
                    try:
                        frame_context = context.model_copy(update={"run_id": f"{run_id}-o{position}-{attempt}"})
                        store.publish_frame(_fragment_frame(symbol, floor, received, body), context=frame_context)
                    except OSError:
                        incomplete = True
                    if price > 0:
                        observed = body
                        observed_at = received
                        entries.append(
                            CoverageEntry(
                                symbol=symbol,
                                dataset=CaptureDataset.PRICE,
                                venue="KRX",
                                session="regular",
                                scheduled_at=floor,
                                status=CaptureStatus.COMPLETE,
                                rows=1,
                                first_event_time=poll_started,
                                last_event_time=received,
                                reason=reason_tag,
                                raw_refs=(ref,),
                            )
                        )
                        break
                    if received >= confirm_end:
                        incomplete = True
                        entries.append(
                            CoverageEntry(
                                symbol=symbol,
                                dataset=CaptureDataset.PRICE,
                                venue="KRX",
                                session="regular",
                                scheduled_at=floor,
                                status=CaptureStatus.PARTIAL,
                                rows=0,
                                first_event_time=poll_started,
                                last_event_time=received,
                                reason="open_unresolved",
                                raw_refs=(ref,),
                            )
                        )
                        observed = None
                        observed_at = None
                        break
                    attempt += 1
                    await sleeper(1.0)
                if observed is None and observed_at is None and not any(
                    item.symbol == symbol and item.dataset == CaptureDataset.PRICE for item in entries
                ):
                    incomplete = True
                    moment = now_clock()
                    entries.append(
                        CoverageEntry(
                            symbol=symbol,
                            dataset=CaptureDataset.PRICE,
                            venue="KRX",
                            session="regular",
                            scheduled_at=floor,
                            status=CaptureStatus.PARTIAL,
                            rows=0,
                            first_event_time=poll_started,
                            last_event_time=moment,
                            reason="open_unresolved",
                            raw_refs=(),
                        )
                    )
    status = CaptureStatus.COMPLETE
    for item in entries:
        if item.status not in _GOOD_ENTRY_STATES:
            status = CaptureStatus.PARTIAL
            break
    if incomplete and status == CaptureStatus.COMPLETE:
        status = CaptureStatus.PARTIAL
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=trading_day,
            run_id=run_id,
            dataset=CaptureDataset.ORDERBOOK if phase == "close" else CaptureDataset.PRICE,
            vendor="kis",
            endpoint="auction-capture",
            symbol=None,
            venue="KRX",
            session="regular",
            capture_reason=reason_tag,
            cohort_id=cohort_id,
            scheduled_at=None,
        ),
        cohort=None,
        completed_at=now_clock(),
        entries=tuple(entries),
        artifacts=tuple(ref for item in entries for ref in item.raw_refs),
        status=status,
    )
    try:
        store.publish_manifest(manifest)
    except OSError as exc:
        raise OSError("required auction coverage persistence fails") from exc
    return manifest


async def _run_async(snapshot_date: str, phase: str, profile: CollectionSettings) -> CaptureManifest | None:
    """Run one capture phase; ``None`` when KIS reports ``snapshot_date`` as a market holiday."""
    import os

    from src.api.kis.client import KisApiClient

    env = dict(os.environ)
    creds = resolve_research_credentials(env, slots=tuple(profile.COLLECTION_RESEARCH_SLOTS))
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
    trading_day = date.fromisoformat(snapshot_date)
    clock = profile.COLLECTION_SESSION_OVERRIDES.get(snapshot_date, SessionClock.standard(trading_day))
    previous_trading_day: str | None = None
    async with clients[0].create_session() as broker_session:
        for client in clients:
            await client.ensure_token(broker_session)
        # 평일 공휴일(명절 등)은 주말 판정으로 걸러지지 않는다. collect 와 같은 KIS 거래일 오라클로 판정한다.
        if not await is_kis_trading_day(clients[0], broker_session, snapshot_date):
            return None
        if phase == "open":
            from src.daily.collect import resolve_prev_trading_day_kis

            # 연휴 직후 개장의 모집단은 직전 '실제' 거래일 코호트여야 한다(주말만 건너뛰면 휴장일을 가리킨다).
            prev = await resolve_prev_trading_day_kis(clients[0], broker_session, pd.Timestamp(snapshot_date))
            previous_trading_day = prev.strftime("%Y-%m-%d")
    return await run_auction_capture(
        snapshot_date,
        phase=phase,  # type: ignore[arg-type]
        profile=profile,
        store=store,
        clients=clients,
        session_clock=clock,
        previous_trading_day=previous_trading_day,
    )


def main(argv: list[str] | None = None) -> None:
    """Run an explicitly configured research phase without transmitting orders.

    Args:
        argv: Optional --phase {close,open} and --date ISO-date arguments.

    Raises:
        ValueError: Invalid or uncertified configuration.
        RuntimeError: Required calendar, cohort, or evidence cannot be verified.
    """
    parser = argparse.ArgumentParser(description="KCA auction capture (research, no orders)")
    parser.add_argument("--phase", choices=["close", "open"], required=True)
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)
    profile = CollectionSettings()
    if not profile.COLLECTION_AUCTION_ENABLED:
        logger.info("[DATA] stage=auction_capture status=SKIP reason=disabled")
        return
    snapshot_date = args.date or datetime.now(SEOUL).date().isoformat()
    try:
        trading_day = date.fromisoformat(snapshot_date)
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    if trading_day.weekday() >= 5:
        logger.info("[DATA] stage=auction_capture status=SKIP reason=holiday date=%s", snapshot_date)
        return
    manifest = asyncio.run(_run_async(snapshot_date, args.phase, profile))
    if manifest is None:
        # 휴장일은 장애가 아니다: 정상 종료해 OnFailure 오탐 알림과 무의미한 캡처를 막는다.
        logger.info("[DATA] stage=auction_capture status=SKIP reason=non_trading_day date=%s", snapshot_date)
        return
    logger.info(
        "[DATA] stage=auction_capture status=%s phase=%s date=%s entries=%d",
        manifest.status.value,
        args.phase,
        snapshot_date,
        len(manifest.entries),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    # 실측: __main__ 가드가 없어 `python -m`으로 실행해도 main()이 전혀 호출되지 않고
    # 조용히 성공 종료(exit 0)했다 -- kca-auction-open/kca-auction-close가 매번
    # "성공"으로 보이면서 실제로는 아무 것도 수집하지 않았다(2026-09-21 daily-audit이
    # collection:auction_open:1:missing_manifest / auction_close:*:missing_entries로 포착).
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
