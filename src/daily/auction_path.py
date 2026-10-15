"""Post-close capture of the closing-auction indicative price path (research only)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

from src import settings
from src.api.dbsec.client import DbsecApiClient, DbsecTokenThrottled
from src.config.collection import CollectionSettings
from src.config.market_session import AUCTION_PATH_FLOOR_HHMMSS, AUCTION_PATH_READY_HHMMSS
from src.data.auction_path_store import (
    AuctionPathRowKind,
    build_auction_path_rows,
    write_auction_path_partition,
)
from src.data.capture_contracts import (
    GOOD_ENTRY_STATES,
    SEOUL,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureManifest,
    CaptureStatus,
    ChartBudget,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.session_calendar import SessionKind, resolve_session_day
from src.tools.alerts import redact_for_egress
from src.tools.run_outcome import (
    RUN_OUTCOME_DEGRADED,
    RUN_OUTCOME_OK,
    RUN_OUTCOME_SKIPPED,
    record_run_outcome,
)
from src.utils.cli_logging import configure_cli_logging

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)

_ENDPOINT = "daily-price-8"
_REASON = "auction-path"


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


def _redacted_reason(exc: BaseException) -> str:
    text = redact_for_egress(type(exc).__name__)
    return text or "vendor_failure"


async def run_auction_path_capture(
    *,
    snapshot_date: str,
    profile: CollectionSettings,
    client: DbsecApiClient,
    session: aiohttp.ClientSession,
    store: CaptureStore,
    now: Callable[[], datetime],
) -> CaptureManifest:
    """Capture today's closing-auction indicative path for the close cohort."""
    from src.daily.auction_capture import _resolve_roster

    try:
        trading_day = date.fromisoformat(snapshot_date)
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    observed = now()
    if trading_day != observed.astimezone(SEOUL).date():
        raise ValueError(f"auction path captures only today's session: {snapshot_date!r}")
    if observed < _at_hhmmss(trading_day, AUCTION_PATH_READY_HHMMSS):
        raise ValueError(f"auction path not observable before {AUCTION_PATH_READY_HHMMSS}: too_early")
    roster, cohort_id, _ = _resolve_roster(snapshot_date, "close", store, observed)
    deadline = _at_hhmmss(trading_day, str(settings.DB_AUCTION_PATH_DEADLINE_HHMMSS))
    run_id = f"auction-path-{snapshot_date}-{uuid.uuid4().hex[:8]}"
    bound = asyncio.Semaphore(max(1, int(settings.DB_AUCTION_PATH_CONCURRENCY)))
    throttled: list[str] = []

    def _entry(
        symbol: str, status: CaptureStatus, reason: str, rows: int, refs: list[Any], started: datetime
    ) -> CoverageEntry:
        return CoverageEntry(
            symbol=symbol,
            dataset=CaptureDataset.AUCTION_PATH,
            venue="KRX",
            session="regular",
            scheduled_at=None,
            status=status,
            rows=rows,
            first_event_time=started,
            last_event_time=now(),
            reason=reason,
            raw_refs=tuple(refs),
        )

    async def _capture_symbol(symbol: str) -> tuple[CoverageEntry, pd.DataFrame | None, int]:
        refs: list[Any] = []

        def _observe(
            data: dict[str, Any] | None,
            metadata: Any,
            page_started: datetime,
            received_at: datetime,
            page_index: int,
            attempt_index: int,
        ) -> None:
            context = CaptureContext(
                trading_date=trading_day,
                run_id=run_id,
                dataset=CaptureDataset.AUCTION_PATH,
                vendor="dbsec",
                endpoint=_ENDPOINT,
                symbol=symbol,
                venue="KRX",
                session="regular",
                capture_reason=_REASON,
                cohort_id=cohort_id,
                scheduled_at=None,
            )
            refs.append(
                store.append_response(
                    CapturedResponse(
                        context=context,
                        request_started_at=page_started,
                        received_at=received_at,
                        payload=dict(data) if isinstance(data, dict) else None,
                        status=CaptureStatus.COMPLETE,
                        source_timestamp=None,
                        source_published_at=None,
                        page_index=page_index,
                        attempt_index=attempt_index,
                        continuation=dict(metadata) if isinstance(metadata, dict) else {},
                        error_type=None,
                    )
                )
            )

        async with bound:
            started = now()
            if throttled:
                logger.debug("[DATA] stage=auction_path symbol=%s status=FAILED reason=token_throttled", symbol)
                return _entry(symbol, CaptureStatus.FAILED, "token_throttled", 0, refs, started), None, 0
            try:
                result = await client.get_auction_path(
                    session,
                    symbol,
                    floor_hhmmss=AUCTION_PATH_FLOOR_HHMMSS,
                    budget=ChartBudget(
                        max_pages=int(settings.DB_AUCTION_PATH_MAX_PAGES),
                        deadline=deadline,
                        request_timeout_seconds=float(profile.COLLECTION_REQUEST_TIMEOUT_SECONDS),
                    ),
                    on_page=_observe,
                )
            except DbsecTokenThrottled:
                throttled.append(symbol)
                logger.debug("[DATA] stage=auction_path symbol=%s status=FAILED reason=token_throttled", symbol)
                return _entry(symbol, CaptureStatus.FAILED, "token_throttled", 0, refs, started), None, 0
            except Exception as exc:
                logger.debug("[DATA] stage=auction_path symbol=%s status=FAILED reason=%s", symbol, type(exc).__name__)
                return _entry(symbol, CaptureStatus.FAILED, _redacted_reason(exc), 0, refs, started), None, 0
        received = now()
        pages = int(result.get("pages_fetched", 0) or 0)
        if str(result.get("rt_cd", "")) != "0":
            reason = redact_for_egress(str(result.get("msg1") or "vendor_failure")) or "vendor_failure"
            logger.debug("[DATA] stage=auction_path symbol=%s status=FAILED reason=vendor_failure", symbol)
            return _entry(symbol, CaptureStatus.FAILED, reason, 0, refs, started), None, pages
        try:
            frame = build_auction_path_rows(
                result.get("output2") or [],
                snapshot_date=snapshot_date,
                symbol=symbol,
                available_at=received,
            )
        except Exception as exc:
            logger.debug("[DATA] stage=auction_path symbol=%s status=FAILED reason=%s", symbol, type(exc).__name__)
            return _entry(symbol, CaptureStatus.FAILED, _redacted_reason(exc), 0, refs, started), None, pages
        terminal = str(result.get("termination_reason", "") or "")
        n_indicative = int((frame["kind"] == AuctionPathRowKind.INDICATIVE.value).sum()) if len(frame) else 0
        kept: pd.DataFrame | None = None
        if terminal == "crossed_time_floor" and n_indicative >= 1:
            entry = _entry(symbol, CaptureStatus.COMPLETE, _REASON, len(frame), refs, started)
            kept = frame
        elif terminal == "crossed_time_floor":
            entry = _entry(symbol, CaptureStatus.UNKNOWN, "no_indicative_rows", 0, refs, started)
        elif terminal == "exhausted":
            entry = _entry(symbol, CaptureStatus.UNKNOWN, "exhausted_before_floor", 0, refs, started)
        else:
            entry = _entry(symbol, CaptureStatus.PARTIAL, f"incomplete:{terminal or 'unknown'}", 0, refs, started)
        logger.debug(
            "[DATA] stage=auction_path symbol=%s status=%s terminal=%s indicative=%d",
            symbol,
            entry.status.value,
            terminal,
            n_indicative,
        )
        return entry, kept, pages

    outcomes = await asyncio.gather(*(_capture_symbol(symbol) for symbol in roster))
    entries: list[CoverageEntry] = [entry for entry, _, _ in outcomes]
    complete_frames: list[pd.DataFrame] = [frame for _, frame, _ in outcomes if frame is not None]
    total_pages = sum(pages for _, _, pages in outcomes)
    if complete_frames:
        rows_written = write_auction_path_partition(
            pd.concat(complete_frames, ignore_index=True), snapshot_date
        )
    else:
        rows_written = write_auction_path_partition(pd.DataFrame(), snapshot_date)
    status = (
        CaptureStatus.COMPLETE
        if all(item.status in GOOD_ENTRY_STATES for item in entries)
        else CaptureStatus.PARTIAL
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=trading_day,
            run_id=run_id,
            dataset=CaptureDataset.AUCTION_PATH,
            vendor="dbsec",
            endpoint=_ENDPOINT,
            symbol=None,
            venue="KRX",
            session="regular",
            capture_reason=_REASON,
            cohort_id=cohort_id,
            scheduled_at=None,
        ),
        cohort=None,
        completed_at=now(),
        entries=tuple(entries),
        artifacts=tuple(ref for item in entries for ref in item.raw_refs),
        status=status,
    )
    store.publish_manifest(manifest)
    n_complete = sum(1 for item in entries if item.status == CaptureStatus.COMPLETE)
    n_partial = sum(1 for item in entries if item.status == CaptureStatus.PARTIAL)
    n_failed = sum(1 for item in entries if item.status == CaptureStatus.FAILED)
    logger.info(
        "[DATA] stage=auction_path status=%s symbols=%d complete=%d partial=%d failed=%d rows=%d pages=%d",
        status.value,
        len(entries),
        n_complete,
        n_partial,
        n_failed,
        rows_written,
        total_pages,
    )
    record_run_outcome(
        "auction_path",
        RUN_OUTCOME_OK if status == CaptureStatus.COMPLETE else RUN_OUTCOME_DEGRADED,
        run_date=snapshot_date,
        metrics={
            "symbols": len(entries),
            "complete": n_complete,
            "partial": n_partial,
            "failed": n_failed,
            "unknown": sum(1 for item in entries if item.status == CaptureStatus.UNKNOWN),
            "rows": rows_written,
            "pages": total_pages,
        },
    )
    return manifest


async def _run_async(snapshot_date: str, profile: CollectionSettings) -> CaptureManifest:
    """Run the capture with live vendor, session and store bindings."""
    import aiohttp

    client = DbsecApiClient()
    store = CaptureStore(_capture_root(profile))
    async with aiohttp.ClientSession() as http_session:
        return await run_auction_path_capture(
            snapshot_date=snapshot_date,
            profile=profile,
            client=client,
            session=http_session,
            store=store,
            now=lambda: datetime.now(SEOUL),
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entry: ``python -m src.daily.auction_path [--date YYYY-MM-DD]``."""
    parser = argparse.ArgumentParser(description="KCA closing-auction indicative path capture (research, no orders)")
    parser.add_argument("--date", default=None)
    args = parser.parse_args(argv)
    profile = CollectionSettings()
    snapshot_date = args.date or datetime.now(SEOUL).date().isoformat()
    try:
        trading_day = date.fromisoformat(snapshot_date)
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    if not settings.DB_AUCTION_PATH_ENABLED:
        record_run_outcome("auction_path", RUN_OUTCOME_SKIPPED, run_date=snapshot_date, reason="disabled")
        logger.info("[DATA] stage=auction_path status=SKIP reason=disabled date=%s", snapshot_date)
        return
    if not settings.DB_APP_KEY:
        record_run_outcome("auction_path", RUN_OUTCOME_SKIPPED, run_date=snapshot_date, reason="no_credentials")
        logger.info("[DATA] stage=auction_path status=SKIP reason=no_credentials date=%s", snapshot_date)
        return
    session_day = resolve_session_day(trading_day, overrides=profile.COLLECTION_SESSION_OVERRIDES)
    if session_day.kind is SessionKind.CLOSED:
        record_run_outcome(
            "auction_path",
            RUN_OUTCOME_SKIPPED,
            run_date=snapshot_date,
            reason="non_trading_day",
            metrics={"session": "CLOSED"},
        )
        logger.info("[DATA] stage=auction_path status=SKIP reason=non_trading_day date=%s", snapshot_date)
        return
    manifest = asyncio.run(_run_async(snapshot_date, profile))
    logger.info(
        "[DATA] stage=auction_path status=%s date=%s entries=%d",
        manifest.status.value,
        snapshot_date,
        len(manifest.entries),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
