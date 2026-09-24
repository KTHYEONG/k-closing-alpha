"""저녁 1회 실행: 당일 워치리스트 정규세션+NXT 애프터마켓 1분봉 아카이브."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from datetime import date, datetime
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
    DEFAULT_BAR_INTERVAL_MINUTES,
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_NXT_PREMARKET,
    INTRADAY_SESSION_REGULAR,
)
from src.daily import archive
from src.data.capture_contracts import (
    SEOUL,
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    Cohort,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.intraday_store import write_intraday_partition, write_tick_partition
from src.data.trading_calendar import is_kis_trading_day

logger = logging.getLogger(__name__)

_GOOD_ENTRY_STATES = frozenset({CaptureStatus.COMPLETE, CaptureStatus.NO_TRADES, CaptureStatus.NOT_APPLICABLE})

_VALID_PHASES = ("regular", "aftermarket", "all")


def _validate_phase(phase: str) -> None:
    if phase not in _VALID_PHASES:
        raise ValueError(f"Invalid phase: {phase!r} (expected one of {', '.join(_VALID_PHASES)})")


def _today_watchlist_codes(snapshot_date: str) -> list[str]:
    try:
        df = archive.fetch_archive_snapshot(snapshot_date=snapshot_date)
    except Exception as e:
        logger.warning("Watchlist fetch failed date=%s: %s", snapshot_date, e)
        return []
    if df is None or df.empty or "종목코드" not in df.columns:
        return []
    return df["종목코드"].astype(str).str.zfill(6).dropna().unique().tolist()


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


def _archive_target_codes(snapshot_date: str) -> list[str]:
    """당일 + 직전 아카이브 영업일 워치리스트의 중복 제거 합집합.

    _today_watchlist_codes/resolve_previous_archive_date는 자체적으로 조회 실패를
    흡수해 빈 결과를 반환하므로(never raise), 여기서는 별도 예외 처리가 필요 없다.
    """
    today = _today_watchlist_codes(snapshot_date)
    prev_date = resolve_previous_archive_date(snapshot_date)
    codes: list[str] = list(today)
    if prev_date is not None:
        for code in _today_watchlist_codes(prev_date):
            if code not in codes:
                codes.append(code)
    return codes


def _previous_trading_day(snapshot_date: str) -> str:
    """달력 기준 직전 영업일(토/일 제외)을 확정한다."""
    try:
        day = date.fromisoformat(str(snapshot_date))
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snapshot_date!r}") from None
    day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day.isoformat()


def _capture_root(profile: CollectionSettings) -> Path:
    if profile.COLLECTION_ROOT is not None:
        return Path(profile.COLLECTION_ROOT)
    return Path(settings.HISTORY_DIR) / "capture"


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


def _resolve_cohort_codes(snapshot_date: str, profile: CollectionSettings, store: CaptureStore) -> tuple[list[str], bool]:
    now = datetime.now(SEOUL)
    today_cohort = store.read_cohort(str(snapshot_date), available_by=now)
    _verify_cohort_against_panel(today_cohort)
    codes: list[str] = [str(item) for item in today_cohort.eligible_symbols]
    prev_day = _previous_trading_day(str(snapshot_date))
    incomplete = False
    try:
        prev_cohort = store.read_cohort(prev_day, available_by=now)
    except FileNotFoundError:
        logger.warning("[DATA] stage=cohort status=INCOMPLETE reason=missing_previous date=%s prev=%s", snapshot_date, prev_day)
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
    status = CaptureStatus.COMPLETE if all(item.status in _GOOD_ENTRY_STATES for item in entries) else CaptureStatus.PARTIAL
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


def _legacy_run(snapshot_date: str | None, bar_interval_minutes: int, *, phase: str = "all") -> tuple[int, int, int]:
    _validate_phase(phase)
    snap_date = snapshot_date or datetime.now().strftime("%Y-%m-%d")
    codes = _archive_target_codes(snap_date)
    if not codes:
        return (0, 0, 0)

    do_regular = phase in ("regular", "all")
    do_aftermarket = phase in ("aftermarket", "all")

    async def _run() -> tuple[int, int, int]:
        client = KisApiClient(**kis_data_client_kwargs())
        ls_client = LsApiClient() if getattr(settings, 'LS_APP_KEY', None) else None
        kiwoom_client = KiwoomApiClient() if (getattr(settings, 'KIWOM_APP_KEY', None) or getattr(settings, 'KIWOOM_APP_KEY', None)) else None
        async with client.create_session() as session:
            await client.ensure_token(session)
            if not await is_kis_trading_day(client, session, snap_date):
                logger.info("[DATA] stage=intraday_archive status=SKIP reason=non_trading_day date=%s", snap_date)
                return (0, 0, 0)
            bars = None
            nxt_after = None
            nxt_pre = None
            krx_after = None
            ticks = None
            if do_regular:
                bars = await collect_intraday_bars(client, session, codes, snap_date, bar_interval_minutes, ls_client=ls_client)
            if do_aftermarket:
                nxt_after = await collect_nxt_aftermarket_bars(client, session, codes, snap_date, bar_interval_minutes, kiwoom_client=kiwoom_client)
                nxt_pre = await collect_nxt_premarket_bars(client, session, codes, snap_date, bar_interval_minutes, kiwoom_client=kiwoom_client)
                krx_after = await collect_krx_aftermarket_bars(client, session, codes, snap_date, bar_interval_minutes)
            if do_regular:
                assert bars is not None
                n_bars = write_intraday_partition(bars, bar_interval_minutes, snap_date, INTRADAY_SESSION_REGULAR)
            else:
                n_bars = 0
            if do_aftermarket:
                assert nxt_after is not None
                assert nxt_pre is not None
                assert krx_after is not None
                n_nxt_after = write_intraday_partition(nxt_after, bar_interval_minutes, snap_date, INTRADAY_SESSION_NXT_AFTERMARKET)
                n_nxt_pre = write_intraday_partition(nxt_pre, bar_interval_minutes, snap_date, INTRADAY_SESSION_NXT_PREMARKET)
                n_krx_after = write_intraday_partition(krx_after, bar_interval_minutes, snap_date, INTRADAY_SESSION_KRX_AFTERMARKET)
                logger.info("[DATA] stage=krx_aftermarket date=%s rows=%d", snap_date, n_krx_after)
                n_nxt = n_nxt_after + n_nxt_pre
            else:
                n_nxt = 0
            if do_regular:
                ticks = await collect_intraday_trade_ticks(client, session, codes, snap_date, ls_client=ls_client, kiwoom_client=kiwoom_client)
                assert ticks is not None
                n_ticks = write_tick_partition(ticks, snap_date, INTRADAY_SESSION_REGULAR)
            else:
                n_ticks = 0
            return (n_bars, n_nxt, n_ticks)

    return asyncio.run(_run())


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
        profile: Validated bounded acquisition profile.
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
    prof = profile if profile is not None else CollectionSettings(COLLECTION_RAW_ENABLED=False)
    snap_date = snapshot_date or datetime.now(SEOUL).date().isoformat()
    try:
        trading_day = date.fromisoformat(str(snap_date))
    except ValueError:
        raise ValueError(f"Invalid snapshot_date: {snap_date!r}") from None
    if int(bar_interval_minutes) <= 0:
        raise ValueError(f"Invalid bar_interval_minutes: {bar_interval_minutes!r}")
    _validate_phase(phase)
    if not prof.COLLECTION_RAW_ENABLED:
        return _legacy_run(str(snap_date), int(bar_interval_minutes), phase=phase)
    store = CaptureStore(_capture_root(prof))
    do_regular = phase in ("regular", "all")
    do_aftermarket = phase in ("aftermarket", "all")

    async def _run() -> tuple[int, int, int]:
        client = KisApiClient(**kis_data_client_kwargs())
        ls_client = LsApiClient() if getattr(settings, 'LS_APP_KEY', None) else None
        kiwoom_client = KiwoomApiClient() if (getattr(settings, 'KIWOM_APP_KEY', None) or getattr(settings, 'KIWOOM_APP_KEY', None)) else None
        async with client.create_session() as session:
            await client.ensure_token(session)
            if not await is_kis_trading_day(client, session, str(snap_date)):
                logger.info("[DATA] stage=intraday_archive status=SKIP reason=non_trading_day date=%s", snap_date)
                return (0, 0, 0)
            # 휴장일엔 collect가 코호트를 발행하지 않으므로, 코호트 조회는 거래일 판정 뒤에 해야 오탐 실패가 없다.
            codes, prev_incomplete = _resolve_cohort_codes(str(snap_date), prof, store)
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
                item.status not in _GOOD_ENTRY_STATES
                for item in collected_entries
            )
            if incomplete:
                logger.warning("[DATA] stage=intraday_archive status=DEGRADED date=%s", snap_date)
            n_bars = counts["bars"] if do_regular else 0
            n_nxt = (counts["nxt_after"] + counts["nxt_pre"]) if do_aftermarket else 0
            n_ticks = counts["ticks"] if do_regular else 0
            return (n_bars, n_nxt, n_ticks)

    return asyncio.run(_run())


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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Intraday archive session split")
    parser.add_argument("--phase", choices=["regular", "aftermarket", "all"], default="all")
    parser.add_argument("--date", default=None, help="Snapshot date YYYY-MM-DD (default today)")
    args = parser.parse_args()
    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    target_codes = _archive_target_codes(target_date)
    logger.info(
        "🚀 [Intraday 아카이브 시작] 대상일: %s, 대상 종목: %d개, 저장소: %s, phase=%s",
        target_date,
        len(target_codes),
        settings.HISTORY_DIR,
        args.phase,
    )
    try:
        bars_rows, nxt_rows, tick_rows = run_intraday_archive(snapshot_date=target_date, profile=CollectionSettings(), phase=args.phase)
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
    logger.info(f"   • 대상 종목수 : {Colors.CYAN}{len(target_codes):>5}{Colors.RESET} 종목 (당일 + 직전 영업일 워치리스트)")
    logger.info(f"   • 정규 세션   : {Colors.GREEN}{bars_rows:>5,}{Colors.RESET} 행 (1분봉)")
    logger.info(f"   • NXT 세션    : {Colors.GREEN}{nxt_rows:>5,}{Colors.RESET} 행 (프리/애프터마켓)")
    logger.info(f"   • 체결 틱     : {Colors.GREEN}{tick_rows:>5,}{Colors.RESET} 행 (정규장 틱 데이터)")
    logger.info(f"   • 저장 경로   : {settings.HISTORY_DIR}")
    logger.info(f"{Colors.BOLD}{box_top}{Colors.RESET}")


if __name__ == "__main__":
    main()
