"""Incremental NXT calibration collection: Toss-consolidated vs KIS-exact pairs.

Each stored vendor-`kis` (KRX-only) symbol-day of an NXT-listed symbol is resolved at
most once: a paired day lands in the calibration table; a day that is not a
consolidated tape or violates the EOD identity is recorded NOT_APPLICABLE in the
calibration ledger and never refetched; transient failures stay FAILED and retry on
later runs up to the ledger's attempt cap, then EXHAUSTED.

Operator-run only, deliberately not scheduled: reconstruction covers the fixed
historical gap (2025-03-04 to the KIS window start) while every new day gets exact
KRX bars from the daily KIS collection, and a refit retires the previous ADOPT by
config digest until recertified. Shares the Toss run lock so collection can never
overlap the Toss backfill (shared token causes TOKEN_REPLACED).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import logging
import re
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, AsyncIterator

import pandas as pd

from src import settings as app_settings
from src.backfill.intraday.blackout import blackout_end, parse_blackout_windows, wait_for_blackout
from src.backfill.intraday.extended_session_backfill import ExtendedBackfillLedger
from src.backfill.intraday.nxt_calibration_pairs import (
    CALIBRATION_TABLE_COLUMNS,
    CalibrationSample,
    _is_full_session_kis,
    calibration_row_from_pair,
)
from src.backfill.intraday.price_basis import PriceReference
from src.backfill.intraday.toss_regular_backfill import _OUTAGE_REASON_PREFIXES, toss_run_lock_path
from src.config.collection import CollectionSettings
from src.config.market_session import INTRADAY_SESSION_REGULAR, NXT_START_DATE
from src.config.nxt_reconstruction import NxtReconstructionSettings
from src.data.capture_contracts import SEOUL, CaptureStatus, CoverageEntry
from src.data.capture_store import CaptureStore
from src.data.intraday_store import intraday_partition_path
from src.data.io_utils import atomic_write_parquet, read_existing_parquet
from src.data.nxt_decomposition import (
    DECOMPOSITION_FIT_REPORT_FILENAME,
    fit_decomposition,
    save_decomposition_config,
    save_fit_diagnostics,
)
from src.data.panel_integrity import REQUIRED_SOURCE_COLUMNS, prepare_price_panel
from src.utils.cli_logging import configure_cli_logging
from src.utils.file_lock import exclusive_file_lock

logger = logging.getLogger(__name__)

CALIBRATION_SESSION: str = "nxt_calibration"
CALIBRATION_LEDGER_FILENAME: str = "nxt_calibration.parquet"
CALIBRATION_TABLE_FILENAME: str = "nxt_calibration_table.parquet"

_CONSOLIDATED_REASON: str = "toss_consolidated_tape"
_IDENTITY_REASON: str = "nxt_identity_excluded"
_KRX_ONLY_REASON: str = "nxt_krx_only"
_CALLS_PER_SYMBOL_DAY: int = 2


def default_calibration_ledger_path() -> Path:
    """Dedicated calibration ledger location (never the Toss regular ledger)."""
    return Path(app_settings.HISTORY_DIR) / "intraday" / "backfill_ledger" / CALIBRATION_LEDGER_FILENAME


def default_calibration_table_path() -> Path:
    """Calibration table location under settings.HISTORY_DIR."""
    return Path(app_settings.HISTORY_DIR) / CALIBRATION_TABLE_FILENAME


@dataclass(frozen=True)
class CalibrationRunSummary:
    """Outcome counts of one bounded calibration collection run."""

    dates_done: int
    dates_remaining: int
    paired: int
    not_applicable: int
    failed: int
    outage_aborted: bool
    stopped_by_deadline: bool
    table_rows: int


def _is_outage_failure(reason: str) -> bool:
    return str(reason).startswith(_OUTAGE_REASON_PREFIXES)


def _rekey_session(entry: CoverageEntry, *, status: CaptureStatus | None = None, reason: str | None = None) -> CoverageEntry:
    """Re-key one acquisition outcome to the calibration session without touching its evidence."""
    return entry.model_copy(
        update={
            "session": CALIBRATION_SESSION,
            "status": status if status is not None else entry.status,
            "reason": reason if reason is not None else entry.reason,
        }
    )


def _upsert_table_rows(table_path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    """Upsert one date's rows into the calibration table, atomically, keyed by (date, symbol).

    Raises:
        OSError: The existing table is unreadable or the merge cannot be persisted
            (no partial file is ever visible).
    """
    frame = pd.DataFrame([dict(row) for row in rows], columns=list(CALIBRATION_TABLE_COLUMNS))
    existing = read_existing_parquet(table_path)
    if len(existing):
        missing = [c for c in CALIBRATION_TABLE_COLUMNS if c not in existing.columns]
        if missing:
            raise OSError(f"Calibration table at {table_path} lacks columns: {missing}")
        merged = pd.concat([existing[list(CALIBRATION_TABLE_COLUMNS)], frame], ignore_index=True)
    else:
        merged = frame
    merged = (
        merged.drop_duplicates(subset=["date", "symbol"], keep="last")
        .sort_values(["date", "symbol"], kind="stable")
        .reset_index(drop=True)
    )
    atomic_write_parquet(merged[list(CALIBRATION_TABLE_COLUMNS)], Path(table_path))
    return len(merged)


def _table_row_count(table_path: Path) -> int:
    existing = read_existing_parquet(Path(table_path))
    return 0 if existing.empty else len(existing)


def _eligible_samples(samples: Sequence[CalibrationSample], terminal: frozenset[str]) -> list[CalibrationSample]:
    seen: set[str] = set()
    candidates = []
    for sample in samples:
        symbol = str(sample.symbol).zfill(6)
        if symbol in seen or symbol in terminal:
            continue
        seen.add(symbol)
        if sample.is_raw_basis and _is_full_session_kis(sample.kis_bars):
            candidates.append(sample)
    return candidates


async def run_calibration_collection(
    *,
    samples_by_date: Mapping[str, Sequence[CalibrationSample]],
    client: Any,
    session: Any,
    profile: CollectionSettings,
    settings: NxtReconstructionSettings,
    ledger: ExtendedBackfillLedger,
    table_path: Path,
    evidence_store: CaptureStore,
    run_id: str,
    clock: Callable[[], datetime],
    stop_at: datetime | None = None,
) -> CalibrationRunSummary:
    """Incrementally extend the calibration table with Toss-consolidated vs KIS-exact pairs.

    Resolves each candidate symbol-day at most once: a paired day lands in the table; a day that is not a consolidated
    tape or violates the EOD identity is recorded NOT_APPLICABLE in the calibration ledger and never refetched; transient
    failures are recorded FAILED and retried on later runs up to the ledger's attempt cap, then EXHAUSTED.

    Returns:
        CalibrationRunSummary(dates_done, dates_remaining, paired, not_applicable, failed, outage_aborted,
        stopped_by_deadline, table_rows).

    Raises:
        ValueError: Empty run_id, missing client/session/ledger/evidence/table, or a naive stop_at.
        TimeoutError: The shared Toss run lock is held by another instance (nothing read or written).
        OSError: The table or ledger cannot be persisted (no partial file is ever visible).
    """
    if not str(run_id).strip():
        raise ValueError("run_id must be nonempty")
    if client is None or session is None:
        raise ValueError("toss client and session must be provided")
    if ledger is None or evidence_store is None or table_path is None or clock is None:
        raise ValueError("ledger, evidence_store, table_path and clock must be provided")
    if stop_at is not None and (stop_at.tzinfo is None or stop_at.utcoffset() is None):
        raise ValueError("stop_at must be timezone-aware")
    prof = profile if profile is not None else CollectionSettings()
    recon = settings if settings is not None else NxtReconstructionSettings()
    try:
        with exclusive_file_lock(toss_run_lock_path(), timeout_seconds=0.0, purpose="toss-backfill"):
            return await _run_collection(
                samples_by_date=samples_by_date,
                client=client,
                session=session,
                profile=prof,
                recon=recon,
                ledger=ledger,
                table_path=Path(table_path),
                evidence_store=evidence_store,
                run_id=str(run_id),
                clock=clock,
                stop_at=stop_at,
            )
    except TimeoutError:
        logger.error(
            "[DATA] stage=nxt_calibration_run status=LOCKED reason=another_instance_holds_the_ledger_lock"
        )
        raise


async def _run_collection(
    *,
    samples_by_date: Mapping[str, Sequence[CalibrationSample]],
    client: Any,
    session: Any,
    profile: CollectionSettings,
    recon: NxtReconstructionSettings,
    ledger: ExtendedBackfillLedger,
    table_path: Path,
    evidence_store: CaptureStore,
    run_id: str,
    clock: Callable[[], datetime],
    stop_at: datetime | None,
) -> CalibrationRunSummary:
    from src.backfill.intraday.toss_regular import acquire_toss_regular_bars

    ordered = sorted(str(day) for day in samples_by_date)
    windows = parse_blackout_windows(tuple(profile.COLLECTION_TOSS_BACKFILL_BLACKOUT_WINDOWS))
    concurrency = max(int(recon.NXT_RECON_CALIBRATION_CONCURRENCY), 1)
    tolerance = float(recon.NXT_RECON_IDENTITY_TOLERANCE)
    outage_share = float(profile.COLLECTION_TOSS_OUTAGE_FAILURE_SHARE)
    outage_min_sample = int(profile.COLLECTION_TOSS_OUTAGE_MIN_SAMPLE)
    total = len(ordered)
    done = 0
    paired = 0
    not_applicable = 0
    failed = 0
    outage_aborted = False
    stopped = False
    for day in ordered:
        if stop_at is not None and clock() >= stop_at:
            stopped = True
            break
        if windows:
            end = blackout_end(clock(), windows, weekdays_only=True)
            if end is not None:
                if stop_at is not None and end > stop_at:
                    stopped = True
                    break
                await wait_for_blackout(windows, now_fn=clock, weekdays_only=True)
                if stop_at is not None and clock() >= stop_at:
                    stopped = True
                    break
        started = clock()
        terminal = ledger.terminal_symbols(day, CALIBRATION_SESSION)
        candidates = _eligible_samples(samples_by_date[day], terminal)
        if not candidates:
            done += 1
            continue
        semaphore = asyncio.Semaphore(concurrency)

        async def _one(sample: CalibrationSample) -> tuple[dict[str, Any] | None, CoverageEntry]:
            async with semaphore:
                frame, entry = await acquire_toss_regular_bars(
                    client,
                    session,
                    str(sample.symbol).zfill(6),
                    day,
                    eod_volume=sample.eod_volume,
                    profile=profile,
                    capture_store=evidence_store,
                    run_id=run_id,
                )
                row = None
                if entry.status == CaptureStatus.NOT_APPLICABLE and entry.reason == _CONSOLIDATED_REASON and not frame.empty:
                    row = calibration_row_from_pair({
                        "date": day, "symbol": str(sample.symbol).zfill(6),
                        "kis_bars": sample.kis_bars, "toss_bars": frame,
                        "eod_volume": float(sample.eod_volume),
                    })
                return row, entry

        fetched = await asyncio.gather(*(_one(sample) for sample in candidates))
        del candidates
        attempted = [entry for _, entry in fetched]
        if len(attempted) >= outage_min_sample and (
            sum(1 for entry in attempted if _is_outage_failure(entry.reason)) / len(attempted)
        ) >= outage_share:
            logger.error(
                "[DATA] stage=nxt_calibration_run status=OUTAGE_ABORT date=%s outage_failures=%d attempted=%d",
                day,
                sum(1 for entry in attempted if _is_outage_failure(entry.reason)),
                len(attempted),
            )
            del fetched
            outage_aborted = True
            break
        new_rows: list[dict[str, Any]] = []
        entries: list[CoverageEntry] = []
        for row, entry in fetched:
            if row is not None:
                denom = float(row["eod_volume"])
                residual = float(row["identity_residual"])
                rel = abs(residual) / denom if denom > 0.0 else float("inf")
                if rel > tolerance:
                    entries.append(
                        _rekey_session(entry, status=CaptureStatus.NOT_APPLICABLE, reason=_IDENTITY_REASON)
                    )
                    not_applicable += 1
                else:
                    new_rows.append(row)
                    entries.append(_rekey_session(entry, status=CaptureStatus.COMPLETE, reason="nxt_calibration_pair"))
                    paired += 1
            elif entry.status == CaptureStatus.COMPLETE:
                entries.append(
                    _rekey_session(entry, status=CaptureStatus.NOT_APPLICABLE, reason=_KRX_ONLY_REASON)
                )
                not_applicable += 1
            elif entry.status == CaptureStatus.NOT_APPLICABLE:
                entries.append(_rekey_session(entry))
                not_applicable += 1
            else:
                entries.append(_rekey_session(entry))
                failed += 1
        del fetched
        if new_rows:
            _upsert_table_rows(table_path, new_rows)
        del new_rows
        if entries:
            ledger.record(
                day,
                CALIBRATION_SESSION,
                entries,
                run_id=run_id,
                attempted_at=clock(),
                vendor="toss",
            )
        del entries
        done += 1
        elapsed = (clock() - started).total_seconds()
        logger.info(
            "[DATA] stage=nxt_calibration_run date=%s paired=%d na=%d failed=%d elapsed_s=%.1f",
            day,
            paired,
            not_applicable,
            failed,
            elapsed,
        )
    table_rows = _table_row_count(table_path)
    logger.info(
        "[DATA] stage=nxt_calibration_run status=%s dates_done=%d dates_remaining=%d paired=%d not_applicable=%d failed=%d table_rows=%d stopped_by_deadline=%s",
        "OUTAGE_ABORT" if outage_aborted else "DONE",
        done,
        total - done,
        paired,
        not_applicable,
        failed,
        table_rows,
        stopped,
    )
    return CalibrationRunSummary(
        dates_done=done,
        dates_remaining=total - done,
        paired=paired,
        not_applicable=not_applicable,
        failed=failed,
        outage_aborted=outage_aborted,
        stopped_by_deadline=stopped,
        table_rows=table_rows,
    )


@asynccontextmanager
async def _http_session(client: Any) -> AsyncIterator[Any]:
    """Yield a request session for any client shape (real, session-factory, or bare fake)."""
    factory = getattr(client, "create_session", None)
    if factory is None:
        import aiohttp

        async with aiohttp.ClientSession() as http_session:
            yield http_session
        return
    produced = factory()
    if hasattr(produced, "__aenter__"):
        async with produced as entered:
            yield entered
    else:
        yield produced


def _open_toss_client() -> Any:
    from src.api.toss.client import TossApiClient

    if not (app_settings.TOSS_APP_KEY and app_settings.TOSS_APP_SECRET):
        raise RuntimeError("Toss credentials are not configured")
    return TossApiClient()


def _load_history(as_of: date) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    wide = pd.read_parquet(
        app_settings.PRICE_HISTORY_PARQUET_PATH,
        columns=sorted(REQUIRED_SOURCE_COLUMNS | {"close_raw", "market"}),
    )
    days = pd.to_datetime(wide["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    wide_history = wide[(days <= as_of.isoformat())].copy()
    prepared, _provenance = prepare_price_panel(wide_history)
    calendar = sorted(
        {str(item) for item in pd.to_datetime(wide_history["date"]).dt.strftime("%Y-%m-%d").tolist()}
    )
    return wide_history, prepared, calendar


def _eod_volumes(prepared_panel: pd.DataFrame) -> dict[tuple[str, str], float]:
    volumes = pd.to_numeric(prepared_panel["volume"], errors="coerce")
    days = pd.to_datetime(prepared_panel["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    symbols = prepared_panel["symbol"].astype(str).str.zfill(6)
    return {
        (str(day), str(symbol)): float(volume)
        for day, symbol, volume in zip(days.tolist(), symbols.tolist(), volumes.tolist())
    }


class _CalibrationSamplesByDate(Mapping[str, Sequence[CalibrationSample]]):
    """Load one partition on demand without retaining any date's bar frames."""

    def __init__(self, paths: Mapping[str, Path], reference: PriceReference, volumes: Mapping[tuple[str, str], float]) -> None:
        self._paths = dict(paths)
        self._reference = reference
        self._volumes = volumes

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, day: str) -> Sequence[CalibrationSample]:
        frame = pd.read_parquet(self._paths[day])
        frame = frame[frame["vendor"].astype(str) == "kis"]
        samples = []
        for symbol, group in frame.groupby(frame["symbol"].astype(str)):
            code = str(symbol).zfill(6)
            volume = self._volumes.get((day, code))
            if volume is None or not volume > 0:
                continue
            raw = self._reference.is_known(day, code) and not self._reference.is_adjusted(day, code)
            samples.append(CalibrationSample(day, code, group.reset_index(drop=True), float(volume), raw))
        return samples


def _build_samples(
    *,
    as_of: date,
    start: str | None,
    end: str | None,
    wide_history: pd.DataFrame,
    prepared: pd.DataFrame,
    calendar: Sequence[str],
) -> Mapping[str, Sequence[CalibrationSample]]:
    reference = PriceReference.from_price_history(wide_history)
    eod = _eod_volumes(prepared)
    paths: dict[str, Path] = {}
    for day in sorted(calendar):
        if day < NXT_START_DATE or day >= as_of.isoformat():
            continue
        if start is not None and day < str(start):
            continue
        if end is not None and day > str(end):
            continue
        target = intraday_partition_path(1, day, INTRADAY_SESSION_REGULAR)
        if not target.exists():
            continue
        paths[day] = target
    return _CalibrationSamplesByDate(paths, reference, eod)


def _print_plan_estimate(samples: Mapping[str, Sequence[CalibrationSample]], ledger: ExtendedBackfillLedger) -> None:
    symbol_days = 0
    for day in samples:
        terminal = ledger.terminal_symbols(str(day), CALIBRATION_SESSION)
        symbol_days += len(_eligible_samples(samples[day], terminal))
    print(f"planned_dates={len(samples)}")
    print(f"planned_symbol_days={symbol_days}")
    print(f"estimated_calls={_CALLS_PER_SYMBOL_DAY * symbol_days}")


def _run_fit(table_path: Path, recon: NxtReconstructionSettings) -> int:
    from src.data.pit1520_panel import default_decomposition_config_path

    snapshot = Path(table_path).read_bytes()
    table = pd.read_parquet(io.BytesIO(snapshot))
    digest = hashlib.sha256(snapshot).hexdigest()
    result = fit_decomposition(
        table,
        alphas=tuple(float(a) for a in recon.NXT_RECON_ALPHAS),
        structures=[(int(m), int(g)) for m, g in recon.NXT_RECON_STRUCTURES],
        identity_tolerance=float(recon.NXT_RECON_IDENTITY_TOLERANCE),
        holdout_fraction=float(recon.NXT_RECON_HOLDOUT_FRACTION),
        min_holdout_days=int(recon.NXT_RECON_MIN_HOLDOUT_DAYS),
        max_selection_rel_err_p90=float(recon.NXT_RECON_MAX_SELECTION_REL_ERR_P90),
        table_sha256=digest,
    )
    config_path = default_decomposition_config_path()
    diagnostics_path = config_path.parent / DECOMPOSITION_FIT_REPORT_FILENAME
    save_fit_diagnostics(result.diagnostics, diagnostics_path)
    save_decomposition_config(result.config, config_path)
    logger.info(
        "[ALGO] stage=nxt_fit status=FITTED alpha=%.3f structure=(%d,%d) holdout_p90=%.4f coverage=%.4f alpha_at_boundary=%s",
        float(result.config.ewma_alpha),
        int(result.config.min_prior_days),
        int(result.config.max_gap_days),
        float(result.diagnostics.holdout_rel_err_p90),
        float(result.diagnostics.holdout_coverage),
        bool(result.diagnostics.alpha_at_boundary),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the incremental NXT calibration collection (exit 0 on completion or deadline stop)."""
    parser = argparse.ArgumentParser(description="NXT calibration collection (Toss-consolidated vs KIS-exact).")
    parser.add_argument("--as-of", default=None, help="Run date YYYY-MM-DD (default KST today).")
    parser.add_argument("--start", default=None, help="Inclusive plan start YYYY-MM-DD.")
    parser.add_argument("--end", default=None, help="Inclusive plan end YYYY-MM-DD.")
    parser.add_argument("--stop-at", default=None, help="HHMMSS KST after which no new date starts (default none).")
    parser.add_argument("--plan-only", action="store_true", help="Print the candidate count and estimated Toss calls without writing anything.")
    parser.add_argument("--fit", action="store_true", help="Fit the decomposition config from the table after collection.")
    args = parser.parse_args(argv)
    profile = CollectionSettings()
    recon = NxtReconstructionSettings()
    now = datetime.now(SEOUL)
    as_of = date.fromisoformat(str(args.as_of)) if args.as_of else now.date()
    if args.start is not None:
        date.fromisoformat(str(args.start))
    if args.end is not None:
        date.fromisoformat(str(args.end))
    if args.start is not None and args.end is not None and str(args.end) < str(args.start):
        raise ValueError(f"Invalid plan range: {args.start!r}..{args.end!r}")
    stop_at: datetime | None = None
    if args.stop_at is not None:
        stop_hhmmss = str(args.stop_at)
        if not re.fullmatch(r"\d{6}", stop_hhmmss):
            raise ValueError(f"Invalid --stop-at HHMMSS: {args.stop_at!r}")
        stop_at = datetime(
            now.year, now.month, now.day,
            int(stop_hhmmss[0:2]), int(stop_hhmmss[2:4]), int(stop_hhmmss[4:6]),
            tzinfo=SEOUL,
        )
        if stop_at <= now:
            stop_at += timedelta(days=1)

    ledger = ExtendedBackfillLedger(default_calibration_ledger_path())
    table_path = default_calibration_table_path()

    if args.plan_only:
        wide_history, prepared, calendar = _load_history(as_of)
        samples = _build_samples(
            as_of=as_of, start=args.start, end=args.end,
            wide_history=wide_history, prepared=prepared, calendar=calendar,
        )
        _print_plan_estimate(samples, ledger)
        return 0

    async def _run_locked() -> int:
        _wide_history, _prepared, _calendar = _load_history(as_of)
        samples = _build_samples(
            as_of=as_of, start=args.start, end=args.end,
            wide_history=_wide_history, prepared=_prepared, calendar=_calendar,
        )
        client = _open_toss_client()
        with tempfile.TemporaryDirectory(prefix="nxt_cal_") as tmp:
            evidence_store = CaptureStore(Path(tmp))
            async with _http_session(client) as http_session:
                await client.ensure_token(http_session)
                run_id = f"nxt-calibration-{as_of.isoformat()}-{uuid.uuid4().hex[:8]}"
                summary = await _run_collection(
                        samples_by_date=samples,
                        client=client,
                        session=http_session,
                        profile=profile,
                        recon=recon,
                        ledger=ledger,
                        table_path=table_path,
                        evidence_store=evidence_store,
                        run_id=run_id,
                        clock=lambda: datetime.now(SEOUL),
                        stop_at=stop_at,
                    )
        if summary.outage_aborted:
            return 1
        if args.fit and not summary.stopped_by_deadline:
            try:
                return _run_fit(table_path, recon)
            except (ValueError, OSError) as exc:
                logger.error("[ALGO] stage=nxt_fit status=FAILED reason=%s", type(exc).__name__)
                return 1
        return 0

    acquired = False
    try:
        with exclusive_file_lock(toss_run_lock_path(), timeout_seconds=0.0, purpose="toss-backfill"):
            acquired = True
            return asyncio.run(_run_locked())
    except TimeoutError:
        if acquired:
            raise
        logger.error("[DATA] stage=nxt_calibration_run status=LOCKED reason=another_instance_holds_the_ledger_lock")
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry; logic covered via runner scenarios
    configure_cli_logging()
    raise SystemExit(main())
