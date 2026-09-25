"""저녁 1회 실행: 당일 워치리스트 정규세션+NXT 애프터마켓 1분봉 아카이브."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src import settings
from src.api.kis.client import KisApiClient, kis_data_client_kwargs
from src.api.kiwoom.client import KiwoomApiClient
from src.api.ls.client import LsApiClient
from src.backfill.intraday.collector import (
    collect_intraday_bars,
    collect_intraday_trade_ticks,
    collect_krx_aftermarket_bars,
    collect_nxt_aftermarket_bars,
    collect_nxt_premarket_bars,
)
from src.config.collection import CollectionSettings
from src.config.market_session import (
    ARCHIVE_AFTERMARKET_READY_HHMMSS,
    ARCHIVE_REGULAR_READY_HHMMSS,
    DEFAULT_BAR_INTERVAL_MINUTES,
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_NXT_PREMARKET,
    INTRADAY_SESSION_REGULAR,
    KRX_AFTERMARKET_START_DATE,
)
from src.daily import archive
from src.data.capture_contracts import (
    GOOD_ENTRY_STATES,
    SEOUL,
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    Cohort,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.intraday_store import write_intraday_partition, write_tick_partition
from src.data.session_calendar import SessionKind, resolve_session_day
from src.data.trading_calendar import is_kis_trading_day
from src.tools.run_outcome import RUN_OUTCOME_DEGRADED, record_run_outcome
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)

_VALID_PHASES = ("regular", "aftermarket", "all")


def _validate_phase(phase: str) -> None:
    if phase not in _VALID_PHASES:
        raise ValueError(f"Invalid phase: {phase!r} (expected one of {', '.join(_VALID_PHASES)})")


def resolve_archive_target_date(now: datetime, phase: str) -> str:
    """Resolve which session date an archive run must cover.

    A run before the phase's ready time on day T is a catch-up of the previous
    weekday's missed run (Persistent timers fire late after downtime); it must
    never archive T, whose session has not finished.

    Args:
        now: Aware KST wall clock.
        phase: "regular", "aftermarket" or "all" ("all" uses the aftermarket ready time).

    Returns:
        ISO date: today when now is at/after the ready time, else the previous weekday.

    Raises:
        ValueError: naive now or unknown phase.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    _validate_phase(phase)
    current = now.astimezone(SEOUL)
    today = current.date()
    ready = ARCHIVE_AFTERMARKET_READY_HHMMSS if phase in ("aftermarket", "all") else ARCHIVE_REGULAR_READY_HHMMSS
    if current.strftime("%H%M%S") >= ready:
        return today.isoformat()
    prev_day = today - timedelta(days=1)
    while prev_day.weekday() >= 5:
        prev_day -= timedelta(days=1)
    return prev_day.isoformat()


def archive_phase_complete(store: CaptureStore, target_date: str, phase: str) -> bool:
    """Return True when every session manifest of the phase is COMPLETE for the date.

    Args:
        store: Capture store holding evening-archive manifests.
        target_date: ISO date.
        phase: "regular" (regular MINUTE_BARS + TRADE_TICKS) or "aftermarket"
            (nxt_premarket, nxt_aftermarket and, from KRX_AFTERMARKET_START_DATE,
            krx_aftermarket MINUTE_BARS); "all" requires both.

    Returns:
        True only if, for each required (dataset, session), the latest
        evening-archive manifest is COMPLETE.
    """
    _validate_phase(phase)
    try:
        manifests = store.read_manifests(str(target_date))
    except (ValueError, OSError):
        return False
    required: list[tuple[CaptureDataset, str]] = []
    if phase in ("regular", "all"):
        required.append((CaptureDataset.MINUTE_BARS, INTRADAY_SESSION_REGULAR))
        required.append((CaptureDataset.TRADE_TICKS, INTRADAY_SESSION_REGULAR))
    if phase in ("aftermarket", "all"):
        required.append((CaptureDataset.MINUTE_BARS, INTRADAY_SESSION_NXT_PREMARKET))
        required.append((CaptureDataset.MINUTE_BARS, INTRADAY_SESSION_NXT_AFTERMARKET))
        if str(target_date) >= KRX_AFTERMARKET_START_DATE:
            required.append((CaptureDataset.MINUTE_BARS, INTRADAY_SESSION_KRX_AFTERMARKET))
    for dataset, session in required:
        candidates = [
            item
            for item in manifests
            if item.context.capture_reason == "evening-archive"
            and item.context.endpoint == "archive-task"
            and item.context.dataset == dataset
            and item.context.session == session
        ]
        if not candidates:
            return False
        latest = max(candidates, key=lambda item: (item.completed_at, item.context.run_id))
        if latest.status != CaptureStatus.COMPLETE:
            return False
    return True


def resolve_previous_archive_date(snapshot_date: str) -> str | None:
    """아카이브 날짜 인덱스에서 snapshot_date 직전 영업일을 찾는다."""
    try:
        df = archive.fetch_archive_snapshot(all_rows=True)
    except Exception as e:
        logger.warning("[DATA] Previous archive date lookup failed date=%s: %s", snapshot_date, e)
        return None
    if df is None or df.empty or "스냅샷_날짜" not in df.columns:
        return None
    dates = sorted({str(d) for d in df["스냅샷_날짜"].astype(str).tolist() if str(d) < str(snapshot_date)})
    return dates[-1] if dates else None


async def _resolve_previous_trading_day(client: Any, session: Any, snapshot_date: str) -> str | None:
    """Resolve the actual previous KRX trading day through the KIS oracle (KRX fallback).

    Returns None instead of raising when both oracles fail or no trading day exists within the
    lookback bound, so a calendar outage degrades the archive to today's cohort (INCOMPLETE)
    rather than losing the whole day's intraday capture.

    Args:
        client: Authenticated KIS data client.
        session: Open HTTP session of that client.
        snapshot_date: KST archive date `YYYY-MM-DD`.

    Returns:
        Previous trading day `YYYY-MM-DD`, or None when unresolved.
    """
    from src.daily.collect import resolve_prev_trading_day_kis

    try:
        prev = await resolve_prev_trading_day_kis(client, session, pd.Timestamp(snapshot_date))
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "[DATA] stage=cohort status=INCOMPLETE reason=previous_day_unresolved date=%s error=%s",
            snapshot_date,
            type(exc).__name__,
        )
        return None
    return prev.strftime("%Y-%m-%d")


def _paper_follow_symbols() -> list[str]:
    try:
        from src.execution.paper_broker import PaperLedger

        open_positions = PaperLedger().load_open_positions()
    except Exception as e:
        logger.warning("[DATA] stage=cohort status=paper_unavailable reason=%s", type(e).__name__)
        return []
    if open_positions is None or open_positions.empty or "symbol" not in open_positions.columns:
        return []
    return sorted({str(item) for item in open_positions["symbol"].astype(str).tolist() if str(item).strip()})


def _panel_listed_before(cohort_date: str) -> frozenset[str]:
    """Return symbols listed on the latest panel date strictly before cohort_date."""
    panel_path = Path(settings.PRICE_HISTORY_PARQUET_PATH)
    if not panel_path.exists():
        raise FileNotFoundError(f"price_history not found: {panel_path}")
    cohort_day = date.fromisoformat(str(cohort_date))
    # 코호트 적격성 규칙(collect)과 동일: 코호트일 직전 최신 패널일의 상장 종목. 공휴일 연휴를 덮는 14일 창.
    rows = pd.read_parquet(
        panel_path,
        columns=["date", "symbol"],
        filters=[("date", ">=", pd.Timestamp(cohort_day) - pd.Timedelta(days=14)), ("date", "<", pd.Timestamp(cohort_day))],
    )
    rows = rows.assign(_d=pd.to_datetime(rows["date"]).dt.normalize())
    if rows.empty:
        raise ValueError(f"stale price_history: no rows in 14d window before {cohort_day.isoformat()}")
    latest = rows["_d"].max()
    listed = rows.loc[rows["_d"] == latest, "symbol"].astype(str)
    return frozenset(listed.tolist())


def _verify_cohort_against_panel(cohort: Cohort) -> None:
    cohort_date = cohort.trading_date.isoformat()
    listed = _panel_listed_before(cohort_date)
    offending = [str(item) for item in cohort.eligible_symbols if str(item) not in listed]
    if offending:
        raise ValueError(
            f"cohort_contamination cohort_id={cohort.cohort_id} date={cohort_date} "
            f"n_offending={len(offending)} samples={offending[:5]}"
        )


def _resolve_cohort_codes(
    snapshot_date: str,
    profile: CollectionSettings,
    store: CaptureStore,
    *,
    previous_trading_day: str | None,
) -> tuple[list[str], bool]:
    """Resolve the archive cohort as today's verified codes plus the prior session's carryover.

    `previous_trading_day` is the oracle-resolved prior session; None means it could not be
    resolved and the previous cohort is treated as missing.

    Args:
        snapshot_date: KST archive date `YYYY-MM-DD`.
        profile: Bounded acquisition profile (unused; kept for call-site symmetry).
        store: Capture store holding verified cohorts.
        previous_trading_day: Oracle-resolved prior session; None degrades to today's cohort.

    Returns:
        (codes, incomplete) with today's eligible codes plus the previous cohort's eligible
        codes and paper-follow symbols; incomplete True when any prior coverage is missing.
    """
    now = datetime.now(SEOUL)
    today_cohort = store.read_cohort(str(snapshot_date), available_by=now)
    _verify_cohort_against_panel(today_cohort)
    codes: list[str] = [str(item) for item in today_cohort.eligible_symbols]
    if previous_trading_day is None:
        for item in _paper_follow_symbols():
            if item not in codes:
                codes.append(item)
        logger.info(
            "[DATA] stage=cohort status=VERIFIED date=%s n_today=%d n_prev=%d",
            snapshot_date, len(today_cohort.eligible_symbols), 0,
        )
        return codes, True
    incomplete = False
    try:
        prev_cohort = store.read_cohort(previous_trading_day, available_by=now)
    except FileNotFoundError:
        logger.warning("[DATA] stage=cohort status=INCOMPLETE reason=missing_previous date=%s prev=%s", snapshot_date, previous_trading_day)
        prev_cohort = None
        incomplete = True
    if prev_cohort is not None:
        _verify_cohort_against_panel(prev_cohort)
        for item in prev_cohort.eligible_symbols:
            if str(item) not in codes:
                codes.append(str(item))
    for item in _paper_follow_symbols():
        if item not in codes:
            codes.append(item)
    n_prev = len(prev_cohort.eligible_symbols) if prev_cohort is not None else 0
    logger.info(
        "[DATA] stage=cohort status=VERIFIED date=%s n_today=%d n_prev=%d",
        snapshot_date, len(today_cohort.eligible_symbols), n_prev,
    )
    return codes, incomplete


def _publish_task_manifest(
    store: CaptureStore,
    *,
    trading_day: date,
    run_id: str,
    dataset: CaptureDataset,
    vendor: str,
    session: str,
    entries: list[CoverageEntry],
) -> CaptureManifest:
    refs: list[Any] = []
    for entry in entries:
        for ref in entry.raw_refs:
            if ref not in refs:
                refs.append(ref)
    status = CaptureStatus.COMPLETE if all(item.status in GOOD_ENTRY_STATES for item in entries) else CaptureStatus.PARTIAL
    manifest = CaptureManifest(
        schema_version=1,
        context=CaptureContext(
            trading_date=trading_day,
            run_id=run_id,
            dataset=dataset,
            vendor=vendor,
            endpoint="archive-task",
            symbol=None,
            venue="owner-local",
            session=session,
            capture_reason="evening-archive",
            cohort_id=None,
            scheduled_at=None,
        ),
        cohort=None,
        completed_at=datetime.now(SEOUL),
        entries=tuple(entries),
        artifacts=tuple(refs),
        status=status,
    )
    store.publish_manifest(manifest)
    return manifest


class _BatchedPartitionPublisher:
    """Buffer certified per-symbol results and flush them as one partition write."""

    def __init__(
        self, write_fn: Callable[[pd.DataFrame, dict[str, CoverageEntry]], int], batch_size: int
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size!r}")
        self._write_fn = write_fn
        self._batch_size = batch_size
        self._frames: list[pd.DataFrame] = []
        self._coverage: dict[str, CoverageEntry] = {}

    def add(self, symbol: str, frame: pd.DataFrame, entry: CoverageEntry) -> None:
        """Buffer one write-eligible symbol; auto-flush once the batch fills."""
        self._frames.append(frame)
        self._coverage[symbol] = entry
        if len(self._coverage) >= self._batch_size:
            self.flush()

    def flush(self) -> int:
        """Write every buffered symbol in one call; no-op returning 0 when empty."""
        if not self._coverage:
            return 0
        combined = pd.concat(self._frames, ignore_index=True)
        written = self._write_fn(combined, dict(self._coverage))
        self._frames = []
        self._coverage = {}
        return written


def run_intraday_archive(snapshot_date: str | None = None, bar_interval_minutes: int = DEFAULT_BAR_INTERVAL_MINUTES, *, profile: CollectionSettings | None = None, phase: str = "all") -> tuple[int, int, int]:
    """Archive the project's dated candidate cohort independently of other collectors.

    Args:
        snapshot_date: Exact trading date, default current Asia/Seoul date.
        bar_interval_minutes: Existing bar interval.
        profile: Validated bounded acquisition profile; None loads CollectionSettings()
            from the environment. Capture evidence is always written.
        phase: Which session group to collect. "regular" acquires KIS/LS/Kiwoom
            regular-session (09:00-15:30) 1m bars and trade ticks only -- both are
            fully settled by 15:30 KST close, so this phase is meant to run right
            after close (e.g. 15:40 KST) independently of the aftermarket phase.
            "aftermarket" acquires NXT premarket, NXT aftermarket, and KRX
            aftermarket 1m bars only -- these sessions do not close until 20:00
            KST, so this phase cannot run meaningfully before then. "all" (the
            default) runs every session, preserving the pre-split behavior for
            ad-hoc backfills and existing callers that pass no phase.

    Returns:
        (regular-bar rows, NXT-bar rows, regular-tick rows) written this call.
        A count is exactly 0 for any session group `phase` did not collect.

    Raises:
        FileNotFoundError: Expected owner-local cohort evidence is absent.
        ValueError: Invalid date, certification, profile, or unrecognized phase.
        OSError: Acquisition evidence or verified publication fails.
    """
    prof = profile if profile is not None else CollectionSettings()
    snap_date = snapshot_date or datetime.now(SEOUL).date().isoformat()
    try:
        trading_day = date.fromisoformat(str(snap_date))
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snap_date!r}") from None
    if int(bar_interval_minutes) <= 0:
        raise ValueError(f"Invalid bar_interval_minutes: {bar_interval_minutes!r}")
    _validate_phase(phase)
    session_day = resolve_session_day(trading_day)
    if session_day.kind is SessionKind.CLOSED:
        logger.info("[DATA] stage=intraday_archive status=SKIP reason=non_trading_day date=%s", snap_date)
        return (0, 0, 0)
    store = CaptureStore(_capture_root(prof))
    do_regular = phase in ("regular", "all")
    do_aftermarket = phase in ("aftermarket", "all")

    async def _run() -> tuple[int, int, int]:
        client = KisApiClient(**kis_data_client_kwargs())
        ls_client = LsApiClient() if settings.LS_APP_KEY else None
        kiwoom_client = KiwoomApiClient() if settings.KIWOOM_APP_KEY else None
        async with client.create_session() as session:
            await client.ensure_token(session)
            if not await is_kis_trading_day(client, session, str(snap_date)):
                logger.info("[DATA] stage=intraday_archive status=SKIP reason=non_trading_day date=%s", snap_date)
                return (0, 0, 0)
            # 휴장일엔 collect가 코호트를 발행하지 않으므로, 코호트 조회는 거래일 판정 뒤에 해야 오탐 실패가 없다.
            prev_trading_day = await _resolve_previous_trading_day(client, session, str(snap_date))
            codes, prev_incomplete = _resolve_cohort_codes(str(snap_date), prof, store, previous_trading_day=prev_trading_day)
            interval = int(bar_interval_minutes)
            batch_rows = int(prof.COLLECTION_ARROW_BATCH_ROWS)
            batch_size = int(prof.COLLECTION_ARCHIVE_SYMBOL_BATCH_SIZE)
            bars_publisher = _BatchedPartitionPublisher(
                lambda df, coverage: write_intraday_partition(
                    df, interval, str(snap_date), INTRADAY_SESSION_REGULAR,
                    coverage=coverage, batch_rows=batch_rows,
                ),
                batch_size,
            )
            ticks_publisher = _BatchedPartitionPublisher(
                lambda df, coverage: write_tick_partition(
                    df, str(snap_date), INTRADAY_SESSION_REGULAR,
                    coverage=coverage, batch_rows=batch_rows,
                ),
                batch_size,
            )
            nxt_after_publisher = _BatchedPartitionPublisher(
                lambda df, coverage: write_intraday_partition(
                    df, interval, str(snap_date), INTRADAY_SESSION_NXT_AFTERMARKET,
                    coverage=coverage, batch_rows=batch_rows,
                ),
                batch_size,
            )
            nxt_pre_publisher = _BatchedPartitionPublisher(
                lambda df, coverage: write_intraday_partition(
                    df, interval, str(snap_date), INTRADAY_SESSION_NXT_PREMARKET,
                    coverage=coverage, batch_rows=batch_rows,
                ),
                batch_size,
            )
            krx_after_publisher = _BatchedPartitionPublisher(
                lambda df, coverage: write_intraday_partition(
                    df, interval, str(snap_date), INTRADAY_SESSION_KRX_AFTERMARKET,
                    coverage=coverage, batch_rows=batch_rows,
                ),
                batch_size,
            )
            bar_entries: list[CoverageEntry] = []
            tick_entries: list[CoverageEntry] = []
            nxt_after_entries: list[CoverageEntry] = []
            nxt_pre_entries: list[CoverageEntry] = []
            krx_after_entries: list[CoverageEntry] = []
            counts = {"bars": 0, "nxt_after": 0, "nxt_pre": 0, "krx_after": 0, "ticks": 0}

            def publish_bars(symbol: str, frame: pd.DataFrame, entry: CoverageEntry) -> None:
                bar_entries.append(entry)
                if entry.status == CaptureStatus.COMPLETE and not frame.empty:
                    bars_publisher.add(symbol, frame, entry)
                    counts["bars"] += len(frame)
                elif not frame.empty:
                    store.publish_frame(frame, context=_fragment_context(trading_day, symbol, CaptureDataset.MINUTE_BARS))

            def publish_ticks(symbol: str, frame: pd.DataFrame, entry: CoverageEntry) -> None:
                tick_entries.append(entry)
                if entry.status == CaptureStatus.COMPLETE and not frame.empty:
                    ticks_publisher.add(symbol, frame, entry)
                    counts["ticks"] += len(frame)
                elif not frame.empty:
                    store.publish_frame(frame, context=_fragment_context(trading_day, symbol, CaptureDataset.TRADE_TICKS))

            def publish_nxt_after(symbol: str, frame: pd.DataFrame, entry: CoverageEntry) -> None:
                nxt_after_entries.append(entry)
                if entry.status == CaptureStatus.COMPLETE and not frame.empty:
                    nxt_after_publisher.add(symbol, frame, entry)
                    counts["nxt_after"] += len(frame)

            def publish_nxt_pre(symbol: str, frame: pd.DataFrame, entry: CoverageEntry) -> None:
                nxt_pre_entries.append(entry)
                if entry.status == CaptureStatus.COMPLETE and not frame.empty:
                    nxt_pre_publisher.add(symbol, frame, entry)
                    counts["nxt_pre"] += len(frame)

            def publish_krx_after(symbol: str, frame: pd.DataFrame, entry: CoverageEntry) -> None:
                krx_after_entries.append(entry)
                if entry.status == CaptureStatus.COMPLETE and not frame.empty:
                    krx_after_publisher.add(symbol, frame, entry)
                    counts["krx_after"] += len(frame)

            # 같은 날 재시도(수동 재실행 또는 실패 후 재기동)가 이전 시도의 불변 매니페스트와
            # 충돌하지 않도록 시도별 고유 접미사를 붙인다(실측: 2026-09-18 수동 재실행이
            # 고정 run_id 때문에 "conflicting immutable artifact identity"로 즉시 실패).
            attempt = uuid.uuid4().hex[:8]
            bars_run = f"archive-{snap_date}-regular-bars-{attempt}"
            after_run = f"archive-{snap_date}-nxt-aftermarket-{attempt}"
            pre_run = f"archive-{snap_date}-nxt-premarket-{attempt}"
            krx_run = f"archive-{snap_date}-krx-aftermarket-{attempt}"
            ticks_run = f"archive-{snap_date}-regular-ticks-{attempt}"
            if do_regular:
                await collect_intraday_bars(client, session, codes, str(snap_date), interval, ls_client=ls_client,
                                            profile=prof, capture_store=store, run_id=bars_run, on_symbol=publish_bars)
                bars_publisher.flush()
            if do_aftermarket:
                await collect_nxt_aftermarket_bars(client, session, codes, str(snap_date), interval, kiwoom_client=kiwoom_client,
                                                   profile=prof, capture_store=store, run_id=after_run, on_symbol=publish_nxt_after)
                nxt_after_publisher.flush()
                await collect_nxt_premarket_bars(client, session, codes, str(snap_date), interval, kiwoom_client=kiwoom_client,
                                                 profile=prof, capture_store=store, run_id=pre_run, on_symbol=publish_nxt_pre)
                nxt_pre_publisher.flush()
                await collect_krx_aftermarket_bars(client, session, codes, str(snap_date), interval,
                                                   profile=prof, capture_store=store, run_id=krx_run, on_symbol=publish_krx_after)
                krx_after_publisher.flush()
                logger.info("[DATA] stage=krx_aftermarket date=%s rows=%d", snap_date, counts["krx_after"])
            if do_regular:
                await collect_intraday_trade_ticks(client, session, codes, str(snap_date), ls_client=ls_client,
                                                   kiwoom_client=kiwoom_client, profile=prof, capture_store=store,
                                                   run_id=ticks_run, on_symbol=publish_ticks)
                ticks_publisher.flush()
            if do_regular:
                _publish_task_manifest(store, trading_day=trading_day, run_id=bars_run,
                                       dataset=CaptureDataset.MINUTE_BARS, vendor="kis",
                                       session=INTRADAY_SESSION_REGULAR, entries=bar_entries)
            if do_aftermarket:
                _publish_task_manifest(store, trading_day=trading_day, run_id=after_run,
                                       dataset=CaptureDataset.MINUTE_BARS, vendor="kiwoom",
                                       session=INTRADAY_SESSION_NXT_AFTERMARKET, entries=nxt_after_entries)
                _publish_task_manifest(store, trading_day=trading_day, run_id=pre_run,
                                       dataset=CaptureDataset.MINUTE_BARS, vendor="kiwoom",
                                       session=INTRADAY_SESSION_NXT_PREMARKET, entries=nxt_pre_entries)
                _publish_task_manifest(store, trading_day=trading_day, run_id=krx_run,
                                       dataset=CaptureDataset.MINUTE_BARS, vendor="kis",
                                       session=INTRADAY_SESSION_KRX_AFTERMARKET, entries=krx_after_entries)
            if do_regular:
                _publish_task_manifest(store, trading_day=trading_day, run_id=ticks_run,
                                       dataset=CaptureDataset.TRADE_TICKS, vendor="kis",
                                       session=INTRADAY_SESSION_REGULAR, entries=tick_entries)
            collected_entries: list[CoverageEntry] = []
            if do_regular:
                collected_entries.extend(bar_entries)
                collected_entries.extend(tick_entries)
            if do_aftermarket:
                collected_entries.extend(nxt_after_entries)
                collected_entries.extend(nxt_pre_entries)
                collected_entries.extend(krx_after_entries)
            incomplete = prev_incomplete or any(
                item.status not in GOOD_ENTRY_STATES
                for item in collected_entries
            )
            if incomplete:
                logger.warning("[DATA] stage=intraday_archive status=DEGRADED date=%s", snap_date)
            n_bars = counts["bars"] if do_regular else 0
            n_nxt = (counts["nxt_after"] + counts["nxt_pre"]) if do_aftermarket else 0
            n_ticks = counts["ticks"] if do_regular else 0
            return (n_bars, n_nxt, n_ticks)

    n_bars, n_nxt, n_ticks = asyncio.run(_run())
    if session_day.kind is SessionKind.SHIFTED:
        record_run_outcome(
            "archive_intraday", RUN_OUTCOME_DEGRADED, run_date=str(snap_date), reason="shifted_session_standard_window"
        )
    return (n_bars, n_nxt, n_ticks)


def _fragment_context(trading_day: date, symbol: str, dataset: CaptureDataset) -> CaptureContext:
    import uuid

    session = INTRADAY_SESSION_REGULAR
    return CaptureContext(
        trading_date=trading_day,
        run_id=f"archive-{trading_day.isoformat()}-fragments-{uuid.uuid4().hex[:6]}",
        dataset=dataset,
        vendor="owner-local",
        endpoint="staged-fragment",
        symbol=symbol,
        venue="UNKNOWN",
        session=session,
        capture_reason="evening-archive",
        cohort_id=None,
        scheduled_at=None,
    )


def main() -> None:
    import argparse

    from src.utils.display import Colors

    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    parser = argparse.ArgumentParser(description="Intraday archive session split")
    parser.add_argument("--phase", choices=["regular", "aftermarket", "all"], default="all")
    parser.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default today)")
    args = parser.parse_args()
    profile = CollectionSettings()
    target_date = args.date or resolve_archive_target_date(datetime.now(SEOUL), args.phase)
    today_str = datetime.now(SEOUL).date().isoformat()
    catchup = target_date != today_str
    logger.info("[DATA] stage=intraday_archive target_date=%s phase=%s catchup=%s", target_date, args.phase, catchup)
    store = CaptureStore(_capture_root(profile))
    if archive_phase_complete(store, target_date, args.phase):
        logger.info("[DATA] stage=intraday_archive status=SKIP reason=already_archived date=%s phase=%s", target_date, args.phase)
        return
    effective_phase = args.phase
    if args.phase in ("aftermarket", "all") and target_date < today_str:
        record_run_outcome("archive_intraday", RUN_OUTCOME_DEGRADED, run_date=target_date, reason="aftermarket_not_replayable")
        if args.phase == "aftermarket":
            logger.info(
                "🚀 [Intraday 아카이브 시작] 대상일: %s, 저장소: %s, phase=%s",
                target_date,
                settings.HISTORY_DIR,
                args.phase,
            )
            logger.info("[DATA] stage=intraday_archive status=SKIP reason=aftermarket_not_replayable date=%s", target_date)
            return
        if archive_phase_complete(store, target_date, "regular"):
            logger.info("[DATA] stage=intraday_archive status=SKIP reason=already_archived date=%s phase=%s", target_date, "regular")
            return
        effective_phase = "regular"
    logger.info(
        "🚀 [Intraday 아카이브 시작] 대상일: %s, 저장소: %s, phase=%s",
        target_date,
        settings.HISTORY_DIR,
        args.phase,
    )
    try:
        bars_rows, nxt_rows, tick_rows = run_intraday_archive(snapshot_date=target_date, profile=profile, phase=effective_phase)
    except ValueError as e:
        logger.error("[DATA] stage=intraday_archive status=ERROR reason=%s", e)
        raise SystemExit(2) from e
    except OSError as e:
        logger.error("[DATA] stage=intraday_archive status=ERROR reason=%s", e)
        raise SystemExit(1) from e

    box_top = "━" * 60
    divider = "─" * 60
    logger.info(f"\n{Colors.BOLD}{box_top}{Colors.RESET}")
    logger.info(f" {Colors.GREEN}{Colors.BOLD}📦 [Intraday 분봉/틱 아카이브 완료]{Colors.RESET} (기준일: {target_date}, phase={args.phase})")
    logger.info(f"{Colors.BOLD}{divider}{Colors.RESET}")
    logger.info(f"   • 정규 세션   : {Colors.GREEN}{bars_rows:>5,}{Colors.RESET} 행 (1분봉)")
    logger.info(f"   • NXT 세션    : {Colors.GREEN}{nxt_rows:>5,}{Colors.RESET} 행 (프리/애프터마켓)")
    logger.info(f"   • 체결 틱     : {Colors.GREEN}{tick_rows:>5,}{Colors.RESET} 행 (정규장 틱 데이터)")
    logger.info(f"   • 저장 경로   : {settings.HISTORY_DIR}")
    logger.info(f"{Colors.BOLD}{box_top}{Colors.RESET}")


if __name__ == "__main__":
    main()
