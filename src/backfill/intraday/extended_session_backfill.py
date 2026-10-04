"""Nightly extended-session and regular-session 1m backfill (NXT aftermarket/premarket, KRX aftermarket, KRX regular).

KIS keeps minute history for a rolling ~1 year, so one trading day of NXT evening
history is lost permanently every trading day. This job replays the retained window
overnight (23:05-06:50 KST, when no KIS REST user is active) through the certified
KIS historical route, resumable across nights via a durable per-symbol ledger.
The regular stream retains full KRX regular-session 1m bars (09:00-15:30, closing-auction
print included) for past symbol-days in the EOD fetch superset, so a 15:20 decision-time
panel can be reconstructed before KIS minute-history expiry.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src import settings
from src.backfill.intraday.collector import backfill_extended_session_bars, backfill_kiwoom_raw_bars
from src.backfill.intraday.price_basis import PriceReference
from src.config.collection import CollectionSettings
from src.config.market_session import (
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_NXT_PREMARKET,
    INTRADAY_SESSION_REGULAR,
    KRX_AFTERMARKET_START_DATE,
    MAX_PREV_TRADING_DAY_LOOKBACK,
)
from src.data.eod_superset import EodSupersetScreen, eod_superset_mask
from src.data.panel_integrity import REQUIRED_SOURCE_COLUMNS, prepare_price_panel
from src.data.capture_contracts import (
    SEOUL,
    GOOD_ENTRY_STATES,
    ArtifactRef,
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
)
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.intraday_store import intraday_partition_path, remove_intraday_symbols, write_intraday_partition
from src.data.io_utils import atomic_write_parquet, read_existing_parquet
from src.utils.cli_logging import configure_cli_logging
from src.utils.file_lock import DEFAULT_LOCK_TIMEOUT_SECONDS, exclusive_file_lock, sidecar_lock_path

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger(__name__)

_LEDGER_COLUMNS: tuple[str, ...] = (
    "snapshot_date",
    "session",
    "symbol",
    "status",
    "rows",
    "reason",
    "run_id",
    "attempted_at",
    "vendor",
    "price_basis",
    "attempts",
)
_LEDGER_KEYS: tuple[str, ...] = ("snapshot_date", "session", "symbol")
EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS: int = 3
_TERMINAL_LEDGER_STATES: frozenset[str] = frozenset({"COMPLETE", "NO_TRADES", "NOT_APPLICABLE", "EXHAUSTED"})
# 원주가 소스(Kiwoom ka10080 _NX)가 있는 세션. KRX 애프터는 Kiwoom 과거 원주가 경로가 없어 fail-closed.
_RAW_SOURCE_SESSIONS: frozenset[str] = frozenset({INTRADAY_SESSION_NXT_AFTERMARKET, INTRADAY_SESSION_NXT_PREMARKET})


def default_ledger_path() -> Path:
    """Default durable ledger location (one file total for the offsite loose path)."""
    return Path(settings.HISTORY_DIR) / "intraday" / "backfill_ledger" / "extended_sessions.parquet"


@dataclass(frozen=True)
class ExtendedBackfillTask:
    """One date/session backfill unit with its point-in-time symbol universe."""

    snapshot_date: str
    session: str
    symbols: tuple[str, ...]


def _normalize_day_column(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%d")


def _entry_day_universes(
    price_history: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
    min_change_ratio: float,
) -> tuple[list[str], dict[str, set[str]]]:
    """Return the ascending trading-day calendar and per-day screen/recorded universes."""
    if price_history is None or len(price_history) == 0:
        calendar: list[str] = []
        screen: dict[str, set[str]] = {}
    else:
        dated = price_history.copy()
        dated["_day"] = _normalize_day_column(dated["date"])
        dated = dated[dated["_day"].notna()]
        dated["_sym"] = dated["symbol"].astype(str).str.zfill(6)
        close = pd.to_numeric(dated["close"], errors="coerce").to_numpy(dtype="float64")
        prev = pd.to_numeric(dated["prev_close"], errors="coerce").to_numpy(dtype="float64")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = close / prev - 1.0
        dated["_pass"] = (prev > 0) & np.isfinite(ratio) & (ratio >= float(min_change_ratio))
        calendar = sorted(set(dated["_day"].astype(str).tolist()))
        screen = (
            dated.loc[dated["_pass"]]
            .groupby("_day")["_sym"]
            .agg(lambda items: set(str(item) for item in items.tolist()))
            .to_dict()
        )
    recorded: dict[str, set[str]] = {}
    if candidate_pairs is not None and len(candidate_pairs) > 0:
        pairs = candidate_pairs.copy()
        pairs["_day"] = _normalize_day_column(pairs["snapshot_date"])
        pairs = pairs[pairs["_day"].notna()]
        pairs["_sym"] = pairs["symbol"].astype(str).str.zfill(6)
        recorded = (
            pairs.groupby("_day")["_sym"]
            .agg(lambda items: set(str(item) for item in items.tolist()))
            .to_dict()
        )
    universes: dict[str, set[str]] = {}
    for day in set(calendar) | set(recorded):
        universe = set(screen.get(day, set())) | set(recorded.get(day, set()))
        if universe:
            universes[str(day)] = universe
    return calendar, universes


def enumerate_extended_session_tasks(
    *,
    as_of: date,
    retention_days: int,
    min_change_ratio: float,
    price_history: pd.DataFrame,
    candidate_pairs: pd.DataFrame,
) -> list[ExtendedBackfillTask]:
    """Enumerate point-in-time extended-session backfill tasks for KIS-retained past dates.

    The entry-day population U(T) is the union of recorded candidates on T and every symbol whose
    close/prev_close - 1 on T is at least min_change_ratio (observable at T's close, so no lookahead).
    For each entry day T the evening (NXT aftermarket, and KRX aftermarket from its start date) of T and
    the NXT premarket of the next trading day are the overnight path of a close-bought position.

    Args:
        as_of: KST date of the run; only dates strictly before it are emitted (live archive owns as_of).
        retention_days: KIS minute retention in calendar days; older dates are never emitted.
        min_change_ratio: Reconstruction threshold on the entry day's close-to-close return.
        price_history: Columns date, symbol, close, prev_close; its dates are the trading calendar.
        candidate_pairs: Columns snapshot_date (YYYY-MM-DD), symbol.

    Returns:
        Tasks ordered by ascending snapshot_date (closest to expiry first), then session name.
    """
    as_of_str = as_of.isoformat()
    earliest = (as_of - timedelta(days=int(retention_days))).isoformat()

    def _in_bounds(day: str) -> bool:
        return earliest <= day < as_of_str

    calendar, universes = _entry_day_universes(price_history, candidate_pairs, float(min_change_ratio))
    entry_days = sorted(day for day in universes if _in_bounds(str(day)))
    cal_index = {day: pos for pos, day in enumerate(calendar)}
    merged: dict[tuple[str, str], set[str]] = {}
    for entry_day in entry_days:
        universe = set(universes.get(entry_day, set()))
        if not universe:
            continue
        merged.setdefault((entry_day, INTRADAY_SESSION_NXT_AFTERMARKET), set()).update(universe)
        if entry_day >= KRX_AFTERMARKET_START_DATE:
            merged.setdefault((entry_day, INTRADAY_SESSION_KRX_AFTERMARKET), set()).update(universe)
        pos = cal_index.get(entry_day)
        if pos is not None and pos + 1 < len(calendar):
            next_day = calendar[pos + 1]
            if _in_bounds(next_day):
                merged.setdefault((next_day, INTRADAY_SESSION_NXT_PREMARKET), set()).update(universe)
    tasks = [
        ExtendedBackfillTask(snapshot_date=day, session=session, symbols=tuple(sorted(symbols)))
        for (day, session), symbols in merged.items()
        if symbols
    ]
    tasks.sort(key=lambda task: (task.snapshot_date, task.session))
    return tasks


def regular_superset_symbols_by_day(
    *,
    as_of: date,
    earliest: str,
    prepared_panel: pd.DataFrame,
    screen: EodSupersetScreen,
) -> dict[str, tuple[str, ...]]:
    """Zero-padded, de-duplicated EOD fetch-superset symbols per trading date in `[earliest, as_of)`.

    This is a fetch filter only (T's EOD is final at backfill time) and is never a training or decision screen.
    """
    as_of_str = as_of.isoformat()
    if prepared_panel is None or len(prepared_panel) == 0:
        eod_superset_mask(prepared_panel if prepared_panel is not None else pd.DataFrame(), screen)
        return {}
    if "date" not in prepared_panel.columns:
        eod_superset_mask(prepared_panel, screen)
        return {}
    days = _normalize_day_column(prepared_panel["date"])
    in_bounds = (days >= str(earliest)) & (days < as_of_str)
    scoped = prepared_panel.loc[in_bounds].copy()
    if len(scoped) == 0:
        return {}
    mask = eod_superset_mask(scoped, screen)
    scoped = scoped.copy()
    scoped["_day"] = _normalize_day_column(scoped["date"]).astype(str).tolist()
    scoped["_pass"] = np.asarray(mask, dtype=bool)
    passing = scoped.loc[scoped["_pass"]]
    grouped: dict[str, set[str]] = {}
    if len(passing):
        syms = passing["symbol"].astype(str).str.zfill(6).tolist()
        for day, symbol in zip(passing["_day"].astype(str).tolist(), syms):
            grouped.setdefault(str(day), set()).add(str(symbol))
    return {day: tuple(sorted(symbols)) for day, symbols in grouped.items()}


@dataclass(frozen=True)
class RegularBackfillPlan:
    """Regular-session tasks plus the symbol-days deliberately not fetched.

    Attributes:
        tasks: One task per past trading date with a non-empty fetchable universe, ascending by date.
        skipped_adjusted: Superset symbol-days whose panel close differs from close_raw (no raw-basis
            regular source exists; KIS history is adjusted).
        skipped_unknown_basis: Superset symbol-days without a decidable price basis.
    """

    tasks: tuple[ExtendedBackfillTask, ...]
    skipped_adjusted: int
    skipped_unknown_basis: int


def enumerate_regular_session_tasks(
    *,
    as_of: date,
    retention_days: int,
    prepared_panel: pd.DataFrame,
    screen: EodSupersetScreen,
    price_reference: PriceReference,
) -> RegularBackfillPlan:
    """Enumerate regular-session backfill tasks for KIS-retained past dates from the EOD superset.

    Day T's universe is every symbol whose EOD row on T passes the fetch superset. T's EOD values are
    final at backfill time (T < as_of), so selecting with them is not lookahead for acquisition; the
    superset is a fetch filter only and is never consumed as a training or decision screen. Stored and
    ledger-terminal symbols are filtered later by the runner (live-archive output is never refetched).

    Args:
        as_of: KST run date; only dates strictly before it are emitted (the live archive owns as_of).
        retention_days: KIS minute retention in calendar days; dates before as_of - retention_days are
            never emitted.
        prepared_panel: prepare_price_panel output covering the retention window (date, symbol, chg_ratio,
            tv_clean, mc_clean, volume).
        screen: EOD fetch superset.
        price_reference: Raw/adjusted state per symbol-day; adjusted or unknown symbol-days are skipped.

    Returns:
        Plan with tasks ordered by ascending snapshot_date (closest to expiry first) and skip counts.

    Raises:
        ValueError: retention_days < 1, or propagated from eod_superset_mask on missing columns.
    """
    if int(retention_days) < 1:
        raise ValueError(f"retention_days must be >= 1, got {retention_days!r}")
    earliest = (as_of - timedelta(days=int(retention_days))).isoformat()
    symbols_by_day = regular_superset_symbols_by_day(
        as_of=as_of, earliest=earliest, prepared_panel=prepared_panel, screen=screen
    )
    tasks: list[ExtendedBackfillTask] = []
    skipped_adjusted = 0
    skipped_unknown_basis = 0
    for day in sorted(symbols_by_day):
        fetchable: list[str] = []
        for symbol in symbols_by_day[day]:
            if price_reference.is_adjusted(day, symbol):
                skipped_adjusted += 1
            elif not price_reference.is_known(day, symbol):
                skipped_unknown_basis += 1
            else:
                fetchable.append(symbol)
        if fetchable:
            tasks.append(
                ExtendedBackfillTask(
                    snapshot_date=str(day),
                    session=INTRADAY_SESSION_REGULAR,
                    symbols=tuple(sorted(set(fetchable))),
                )
            )
    tasks.sort(key=lambda task: task.snapshot_date)
    return RegularBackfillPlan(
        tasks=tuple(tasks),
        skipped_adjusted=int(skipped_adjusted),
        skipped_unknown_basis=int(skipped_unknown_basis),
    )


def _with_legacy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill columns added after the first ledger files were written (vendor, price_basis, attempts)."""
    out = frame
    if "vendor" not in out.columns:
        out = out.copy()
        out["vendor"] = "kis"
    if "price_basis" not in out.columns:
        out = out.copy()
        out["price_basis"] = ""
    if "attempts" not in out.columns:
        out = out.copy()
        if "status" in out.columns and len(out):
            out["attempts"] = np.where(out["status"].astype(str) == "FAILED", 1, 0)
        else:
            out["attempts"] = 0
    return out


def _latest_attempts(frame: pd.DataFrame, snapshot_date: str, session: str) -> dict[str, tuple[str, int]]:
    """Latest (status, attempts) per symbol for one date/session from a legacy-filled frame."""
    sub = frame[
        (frame["snapshot_date"].astype(str) == str(snapshot_date))
        & (frame["session"].astype(str) == str(session))
    ]
    if sub.empty:
        return {}
    latest = sub.drop_duplicates(subset=["symbol"], keep="last")
    return {
        str(symbol): (str(status), int(attempts))
        for symbol, status, attempts in zip(
            latest["symbol"].astype(str),
            latest["status"].astype(str),
            latest["attempts"],
        )
    }


class ExtendedBackfillLedger:
    """Durable per-symbol outcome log; the latest record per key wins."""

    def __init__(self, path: Path | None = None) -> None:
        """Point the ledger at one parquet file (default the history-tree location)."""
        self._path = Path(path) if path is not None else default_ledger_path()

    @property
    def path(self) -> Path:
        """Ledger file location."""
        return self._path

    def _read_all(self) -> pd.DataFrame:
        return read_existing_parquet(self._path)

    def terminal_symbols(self, snapshot_date: str, session: str) -> frozenset[str]:
        """Symbols already resolved terminally for one date/session (never refetched)."""
        frame = self._read_all()
        if frame.empty:
            return frozenset()
        sub = frame[
            (frame["snapshot_date"].astype(str) == str(snapshot_date))
            & (frame["session"].astype(str) == str(session))
        ]
        if sub.empty:
            return frozenset()
        latest = sub.drop_duplicates(subset=["symbol"], keep="last")
        return frozenset(
            str(item)
            for item in latest.loc[
                latest["status"].astype(str).isin(_TERMINAL_LEDGER_STATES), "symbol"
            ].tolist()
        )

    def unverified_complete_symbols(self, snapshot_date: str, session: str) -> frozenset[str]:
        """Symbols whose latest record is a COMPLETE KIS fetch with no recorded price basis.

        Rows written before price-basis routing may hold adjusted bars; the runner refetches the ones whose
        panel day is adjusted. An empty price_basis on a COMPLETE KIS row is the only marker.
        """
        frame = self._read_all()
        if frame.empty:
            return frozenset()
        frame = _with_legacy_columns(frame)
        sub = frame[
            (frame["snapshot_date"].astype(str) == str(snapshot_date))
            & (frame["session"].astype(str) == str(session))
        ]
        if sub.empty:
            return frozenset()
        latest = sub.drop_duplicates(subset=["symbol"], keep="last")
        basis = latest["price_basis"].fillna("").astype(str)
        mask = (
            (latest["status"].astype(str) == "COMPLETE")
            & (latest["vendor"].astype(str) == "kis")
            & (basis == "")
        )
        return frozenset(str(item) for item in latest.loc[mask, "symbol"].tolist())

    def record(
        self,
        snapshot_date: str,
        session: str,
        entries: Sequence[CoverageEntry],
        *,
        run_id: str,
        attempted_at: datetime,
        vendor: str = "kis",
        price_bases: Mapping[str, str] | None = None,
    ) -> None:
        """Append per-symbol outcomes; failures stay retryable until the attempt cap.

        A key that has failed EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS consecutive runs is recorded as EXHAUSTED (terminal):
        vendor history for an old date does not reappear, and retrying it forever rewrites old evidence directories
        nightly (unprunable locally, one offsite segment per date). Any non-FAILED outcome resets the count.

        Args:
            snapshot_date: Trading date of the session partition.
            session: Session tag of the entries.
            entries: Coverage outcomes to append.
            run_id: Acquisition identity for these rows.
            attempted_at: Aware attempt instant.
            vendor: Data source of these entries for audit; it never changes terminal semantics.
            price_bases: Raw-basis source per COMPLETE symbol ("kis_raw" or "kiwoom_raw"); others empty.
        """
        bases = dict(price_bases) if price_bases is not None else {}
        pending = [
            (
                entry,
                {
                    "snapshot_date": str(snapshot_date),
                    "session": str(session),
                    "symbol": str(entry.symbol),
                    "status": entry.status.value,
                    "rows": int(entry.rows),
                    "reason": str(entry.reason),
                    "run_id": str(run_id),
                    "attempted_at": attempted_at.isoformat(),
                    "vendor": str(vendor),
                    "price_basis": str(bases.get(str(entry.symbol), "")),
                },
            )
            for entry in entries
            if entry.symbol is not None
        ]
        if not pending:
            return
        with exclusive_file_lock(
            sidecar_lock_path(self._path),
            timeout_seconds=DEFAULT_LOCK_TIMEOUT_SECONDS,
            purpose="backfill-ledger",
        ):
            existing = self._read_all()
            if not existing.empty:
                existing = _with_legacy_columns(existing)
                prev = _latest_attempts(existing, str(snapshot_date), str(session))
            else:
                prev = {}
            rows = []
            for entry, row in pending:
                if entry.status == CaptureStatus.FAILED:
                    before = prev.get(str(entry.symbol))
                    attempts = int(before[1]) + 1 if before is not None and before[0] == "FAILED" else 1
                    if attempts >= EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS:
                        row["status"] = "EXHAUSTED"
                        row["reason"] = f"exhausted:{row['reason']}"
                    row["attempts"] = attempts
                else:
                    row["attempts"] = 0
                rows.append(row)
            incoming = pd.DataFrame(rows, columns=list(_LEDGER_COLUMNS))
            combined = (
                pd.concat([existing, incoming], ignore_index=True)
                if not existing.empty
                else incoming
            )
            combined = combined.drop_duplicates(subset=list(_LEDGER_KEYS), keep="last")
            atomic_write_parquet(combined[list(_LEDGER_COLUMNS)], self._path)

    def record_cached_absent(
        self,
        snapshot_date: str,
        session: str,
        symbols: Sequence[str],
        *,
        reason: str,
        run_id: str,
        attempted_at: datetime,
        vendor: str = "toss",
    ) -> None:
        """Append symbol-level cached absences without new evidence or requests.

        A symbol proven unavailable at symbol level (e.g. Toss `stock-not-found`, which never
        varies by date) stays absent for every remaining date without another vendor call; the
        proving response lives under the original run's manifest and is cited through `reason`.
        Rows are NOT_APPLICABLE and therefore terminal, like any other rejection.
        """
        names = sorted({str(symbol) for symbol in symbols if str(symbol)})
        if not names:
            return
        if not str(reason).strip():
            raise ValueError("cached absence requires a nonempty reason")
        with exclusive_file_lock(
            sidecar_lock_path(self._path),
            timeout_seconds=DEFAULT_LOCK_TIMEOUT_SECONDS,
            purpose="backfill-ledger",
        ):
            existing = self._read_all()
            rows = [
                {
                    "snapshot_date": str(snapshot_date),
                    "session": str(session),
                    "symbol": name,
                    "status": "NOT_APPLICABLE",
                    "rows": 0,
                    "reason": str(reason),
                    "run_id": str(run_id),
                    "attempted_at": attempted_at.isoformat(),
                    "vendor": str(vendor),
                    "price_basis": "",
                    "attempts": 0,
                }
                for name in names
            ]
            incoming = pd.DataFrame(rows, columns=list(_LEDGER_COLUMNS))
            combined = (
                pd.concat([existing, incoming], ignore_index=True)
                if not existing.empty
                else incoming
            )
            combined = combined.drop_duplicates(subset=list(_LEDGER_KEYS), keep="last")
            atomic_write_parquet(combined[list(_LEDGER_COLUMNS)], self._path)


@dataclass(frozen=True)
class ExtendedBackfillSummary:
    """Outcome counts of one bounded backfill run."""

    tasks_done: int
    tasks_remaining: int
    complete: int
    no_trades: int
    not_listed: int
    failed: int
    stopped_by_deadline: bool
    exhausted: int = 0


def _stored_partition_symbols(snapshot_date: str, session: str) -> set[str]:
    target = intraday_partition_path(1, str(snapshot_date), str(session))
    if not target.exists():
        return set()
    try:
        existing = pd.read_parquet(target, columns=["symbol"])
    except Exception as exc:
        raise OSError(f"Cannot read existing partition evidence: {target}") from exc
    if existing.empty or "symbol" not in existing.columns:
        return set()
    return {str(item) for item in existing["symbol"].astype(str).tolist()}


@dataclass(frozen=True)
class RegularBackfillCoverage:
    """Resolution state of the regular stream over its planned symbol-days."""

    days_total: int
    days_resolved: int
    symbol_days_total: int
    symbol_days_resolved: int
    symbol_days_pending: int
    oldest_pending_date: str
    oldest_pending_expiry_days: int


def summarize_regular_backfill_coverage(
    plan: RegularBackfillPlan,
    ledger: ExtendedBackfillLedger,
    *,
    as_of: date,
    retention_days: int,
    stored_symbols: Callable[[str, str], set[str]] = _stored_partition_symbols,
) -> RegularBackfillCoverage:
    """Count planned regular symbol-days already resolved (ledger-terminal or stored) and the expiry horizon.

    Args:
        plan: Output of enumerate_regular_session_tasks.
        ledger: Durable outcome ledger.
        as_of: KST run date.
        retention_days: KIS minute retention in calendar days.
        stored_symbols: Stored partition symbols per (snapshot_date, session); injectable for tests.

    Returns:
        Coverage counts; oldest_pending_date is "" and oldest_pending_expiry_days is -1 when nothing is pending.

    Raises:
        OSError: A stored partition is unreadable (propagated; never treated as empty).
    """
    days_total = len(plan.tasks)
    symbol_days_total = sum(len(task.symbols) for task in plan.tasks)
    symbol_days_resolved = 0
    days_resolved = 0
    oldest_pending = ""
    for task in plan.tasks:
        terminal = ledger.terminal_symbols(task.snapshot_date, INTRADAY_SESSION_REGULAR)
        stored = stored_symbols(task.snapshot_date, INTRADAY_SESSION_REGULAR)
        resolved = {symbol for symbol in task.symbols if symbol in terminal or symbol in stored}
        symbol_days_resolved += len(resolved)
        if len(resolved) == len(task.symbols):
            days_resolved += 1
        elif not oldest_pending or task.snapshot_date < oldest_pending:
            oldest_pending = str(task.snapshot_date)
    symbol_days_pending = int(symbol_days_total - symbol_days_resolved)
    if not oldest_pending:
        return RegularBackfillCoverage(
            days_total=int(days_total),
            days_resolved=int(days_resolved),
            symbol_days_total=int(symbol_days_total),
            symbol_days_resolved=int(symbol_days_resolved),
            symbol_days_pending=int(symbol_days_pending),
            oldest_pending_date="",
            oldest_pending_expiry_days=-1,
        )
    expiry = int(retention_days) - (as_of - date.fromisoformat(oldest_pending)).days
    return RegularBackfillCoverage(
        days_total=int(days_total),
        days_resolved=int(days_resolved),
        symbol_days_total=int(symbol_days_total),
        symbol_days_resolved=int(symbol_days_resolved),
        symbol_days_pending=int(symbol_days_pending),
        oldest_pending_date=str(oldest_pending),
        oldest_pending_expiry_days=int(expiry),
    )


def _select_backfill_tasks(
    kis_tasks: Sequence[ExtendedBackfillTask],
    regular_plan: RegularBackfillPlan | None,
    *,
    run_extended: bool,
    run_regular: bool,
) -> list[ExtendedBackfillTask]:
    """Concatenate the enabled streams; the runner orders the union oldest-first.

    Args:
        kis_tasks: Extended-stream tasks.
        regular_plan: Regular-stream plan, or None when the stream is skipped.
        run_extended: Include the extended stream.
        run_regular: Include the regular stream.

    Returns:
        Task list with extended tasks first; the runner sorts by (snapshot_date, session).
    """
    selected = list(kis_tasks) if run_extended else []
    if run_regular and regular_plan is not None:
        selected.extend(regular_plan.tasks)
    return selected


@asynccontextmanager
async def _http_session(client: Any) -> AsyncIterator[aiohttp.ClientSession | None]:
    """Yield a request session for any client shape (real, session-factory, or bare fake)."""
    factory = getattr(client, "create_session", None)
    if factory is None:
        yield None
        return
    produced = factory()
    if hasattr(produced, "__aenter__"):
        async with produced as entered:
            yield entered
    else:
        yield produced


async def run_extended_session_backfill(
    *,
    as_of: date,
    stop_at: datetime,
    profile: CollectionSettings,
    clients: Sequence[Any],
    store: CaptureStore,
    ledger: ExtendedBackfillLedger,
    tasks: Sequence[ExtendedBackfillTask],
    price_reference: PriceReference,
    raw_client: Any | None = None,
    now_fn: Callable[[], datetime] | None = None,
) -> ExtendedBackfillSummary:
    """Execute backfill tasks oldest-first until done or the stop time, resumable across nights.

    Pending symbols of a task exclude ledger-terminal symbols and symbols already present in the stored
    partition (live archive output is never re-fetched or overwritten), plus repair symbols whose
    basis-less COMPLETE row covers an adjusted panel day. The store is raw-basis only: a KIS fetch is kept
    only when the panel shows no later corporate action; an adjusted symbol-day is replaced by Kiwoom raw
    bars, and without a raw source it fails closed (a repair symbol's stored adjusted rows are removed).
    Symbols are spread round-robin over the backfill credentials; each task publishes certified symbols,
    a task manifest and ledger rows before the next task starts, so an interruption loses at most the
    in-flight task.

    Args:
        as_of: KST run date (used for logging and run ids).
        stop_at: Aware KST instant after which no new task starts.
        profile: Collection limits.
        clients: Token-ready KIS clients, one per backfill credential.
        store: Capture store for raw evidence and manifests.
        ledger: Durable per-symbol outcome ledger.
        tasks: Output of enumerate_extended_session_tasks.
        price_reference: Panel adjustment state deciding whether a KIS historical fetch is raw.
        raw_client: Kiwoom client supplying raw bars for adjusted NXT symbol-days; None fails them closed.
        now_fn: Aware clock; None uses Asia/Seoul now.

    Returns:
        Summary counts and whether the deadline stopped the run.

    Side effects:
        A task with no pending symbols (every task symbol is terminal, already stored, and not a repair
        candidate) is a no-op: it performs no network fetch, writes no partition, publishes no manifest,
        creates no run directory and appends no ledger rows. A task with at least one pending symbol
        publishes exactly one manifest covering its attempted entries, whatever their statuses.

    Raises:
        ValueError: Empty clients or naive stop_at.
        OSError: Partition, manifest or ledger persistence fails (fail loud; the task is not marked terminal).
    """
    if not clients:
        raise ValueError("backfill clients must be nonempty")
    if stop_at.tzinfo is None or stop_at.utcoffset() is None:
        raise ValueError("stop_at must be timezone-aware")
    clock = now_fn if now_fn is not None else (lambda: datetime.now(SEOUL))
    ordered = sorted(tasks, key=lambda task: (task.snapshot_date, task.session))
    max_rows = int(profile.COLLECTION_ARCHIVE_PUBLISH_ROWS)
    done = 0
    complete = 0
    no_trades = 0
    not_listed = 0
    failed = 0
    exhausted = 0
    stopped = False
    unreadable: list[str] = []
    async with AsyncExitStack() as stack:
        http_sessions = [await stack.enter_async_context(_http_session(client)) for client in clients]
        for task in ordered:
            if clock() >= stop_at:
                stopped = True
                break
            started = clock()
            terminal = ledger.terminal_symbols(task.snapshot_date, task.session)
            try:
                stored = _stored_partition_symbols(task.snapshot_date, task.session)
            except OSError as exc:
                # One unreadable partition must not starve every later task of the night (oldest-first order
                # would hit it first each run): finish the readable tasks, then fail loud below.
                logger.error(
                    "[DATA] stage=extended_backfill status=TASK_SKIPPED reason=unreadable_partition date=%s session=%s error=%s",
                    task.snapshot_date, task.session, type(exc).__name__,
                )
                unreadable.append(f"{task.snapshot_date}/{task.session}")
                continue
            repair_candidates = {
                symbol
                for symbol in ledger.unverified_complete_symbols(task.snapshot_date, task.session)
                if symbol in task.symbols and price_reference.is_adjusted(task.snapshot_date, symbol)
            }
            pending = [
                symbol
                for symbol in task.symbols
                if (symbol not in terminal and symbol not in stored) or symbol in repair_candidates
            ]
            repaired = sum(1 for symbol in pending if symbol in repair_candidates)
            run_id = f"extended-backfill-{task.snapshot_date}-{task.session}-{uuid.uuid4().hex[:8]}"
            collected: dict[str, tuple[pd.DataFrame, CoverageEntry]] = {}
            symbol_routes: dict[str, tuple[Any, Any]] = {}

            async def _fetch_bucket(
                client: Any, http_session: Any, symbols: list[str]
            ) -> dict[str, tuple[pd.DataFrame, CoverageEntry]]:
                bucket: dict[str, tuple[pd.DataFrame, CoverageEntry]] = {}

                def _observe(symbol: str, frame: pd.DataFrame, entry: CoverageEntry) -> None:
                    bucket[symbol] = (frame, entry)

                await backfill_extended_session_bars(
                    client,
                    http_session,
                    symbols,
                    task.snapshot_date,
                    session_tag=task.session,
                    bar_interval_minutes=1,
                    profile=profile,
                    capture_store=store,
                    run_id=run_id,
                    on_symbol=_observe,
                )
                return bucket

            if pending:
                buckets: list[list[str]] = [[] for _ in clients]
                for index, symbol in enumerate(pending):
                    buckets[index % len(clients)].append(symbol)
                for client, http_session, bucket in zip(clients, http_sessions, buckets):
                    for symbol in bucket:
                        symbol_routes[symbol] = (client, http_session)
                fetched = await asyncio.gather(
                    *(
                        _fetch_bucket(client, http_session, bucket)
                        for client, http_session, bucket in zip(clients, http_sessions, buckets)
                        if bucket
                    )
                )
                for bucket_result in fetched:
                    collected.update(bucket_result)

            def _basis_failed_entry(entry: CoverageEntry, reason: str, refs: tuple[ArtifactRef, ...] = ()) -> CoverageEntry:
                return CoverageEntry(
                    symbol=entry.symbol,
                    dataset=entry.dataset,
                    venue=entry.venue,
                    session=entry.session,
                    scheduled_at=entry.scheduled_at,
                    status=CaptureStatus.FAILED,
                    rows=0,
                    first_event_time=None,
                    last_event_time=None,
                    reason=reason,
                    raw_refs=tuple(entry.raw_refs) + tuple(refs),
                )

            raw_sem = asyncio.Semaphore(max(int(profile.COLLECTION_CONCURRENCY_PER_KEY), 1))
            price_bases: dict[str, str] = {}

            async def _raw_one(symbol: str) -> tuple[str, pd.DataFrame, CoverageEntry, str | None]:
                frame, entry = collected[symbol]
                if not price_reference.is_known(task.snapshot_date, symbol):
                    return symbol, frame.iloc[0:0], _basis_failed_entry(entry, "price_basis_unknown"), None
                if not price_reference.is_adjusted(task.snapshot_date, symbol):
                    return symbol, frame, entry, "kis_raw"
                if raw_client is None or task.session not in _RAW_SOURCE_SESSIONS:
                    return symbol, frame.iloc[0:0], _basis_failed_entry(entry, "price_basis_no_raw_source"), None
                _client, http_session = symbol_routes[symbol]
                async with raw_sem:
                    raw_frame, raw_entry = await backfill_kiwoom_raw_bars(
                        raw_client, http_session, symbol, task.snapshot_date,
                        session_tag=task.session, profile=profile, capture_store=store, run_id=run_id,
                    )
                if raw_entry.status == CaptureStatus.COMPLETE:
                    return symbol, raw_frame, raw_entry, "kiwoom_raw"
                return symbol, frame.iloc[0:0], _basis_failed_entry(
                    entry, f"price_basis_{raw_entry.reason}", raw_entry.raw_refs
                ), None

            basis_targets = [
                symbol
                for symbol in pending
                if symbol in collected
                and collected[symbol][1].status == CaptureStatus.COMPLETE
                and not collected[symbol][0].empty
            ]
            if basis_targets:
                for symbol, frame, entry, basis in await asyncio.gather(*(_raw_one(s) for s in basis_targets)):
                    collected[symbol] = (frame, entry)
                    if basis is not None:
                        price_bases[symbol] = basis
            adjusted = sum(1 for basis in price_bases.values() if basis == "kiwoom_raw")
            entries = [collected[symbol][1] for symbol in pending]
            basis_failed = sum(1 for entry in entries if str(entry.reason).startswith("price_basis_"))
            good = [entry for entry in entries if entry.status in (CaptureStatus.COMPLETE, CaptureStatus.NO_TRADES)]
            comp_chunks = [
                (collected[str(entry.symbol)][0], entry)
                for entry in good
                if entry.status == CaptureStatus.COMPLETE
                and entry.symbol is not None
                and str(entry.symbol) in collected
                and not collected[str(entry.symbol)][0].empty
            ]
            buffer_frames: list[pd.DataFrame] = []
            buffer_coverage: dict[str, CoverageEntry] = {}
            buffered_rows = 0

            def _flush_buffer() -> None:
                nonlocal buffer_frames, buffer_coverage, buffered_rows
                if not buffer_coverage:
                    return
                chunk_frame = pd.concat(buffer_frames, ignore_index=True)
                write_intraday_partition(
                    chunk_frame, 1, task.snapshot_date, task.session, coverage=dict(buffer_coverage)
                )
                buffer_frames = []
                buffer_coverage = {}
                buffered_rows = 0

            for frame, entry in comp_chunks:
                key = str(entry.symbol)
                if key in buffer_coverage:
                    raise ValueError(f"Duplicate symbol buffered: {key!r}")
                buffer_coverage[key] = entry
                buffer_frames.append(frame)
                buffered_rows += int(len(frame))
                if buffered_rows >= max_rows:
                    _flush_buffer()
            _flush_buffer()
            stale = {
                symbol
                for symbol in pending
                if symbol in repair_candidates and collected[symbol][1].status != CaptureStatus.COMPLETE
            }
            if stale:
                remove_intraday_symbols(1, task.snapshot_date, task.session, stale)
            if pending:
                manifest_status = (
                    CaptureStatus.COMPLETE
                    if all(entry.status in GOOD_ENTRY_STATES for entry in entries)
                    else CaptureStatus.PARTIAL
                )
                manifest = CaptureManifest(
                    schema_version=1,
                    context=CaptureContext(
                        trading_date=date.fromisoformat(task.snapshot_date),
                        run_id=run_id,
                        dataset=CaptureDataset.MINUTE_BARS,
                        vendor="kis",
                        endpoint="backfill-task",
                        symbol=None,
                        venue="owner-local",
                        session=task.session,
                        capture_reason="extended-backfill",
                        cohort_id=None,
                        scheduled_at=None,
                    ),
                    cohort=None,
                    completed_at=clock(),
                    entries=tuple(entries),
                    artifacts=tuple(dict.fromkeys(ref for entry in entries for ref in entry.raw_refs)),
                    status=manifest_status,
                )
                store.publish_manifest(manifest)
                if entries:
                    prev_frame = ledger._read_all()
                    prev_map = (
                        _latest_attempts(
                            _with_legacy_columns(prev_frame), task.snapshot_date, task.session
                        )
                        if not prev_frame.empty
                        else {}
                    )
                    ledger.record(
                        task.snapshot_date,
                        task.session,
                        entries,
                        run_id=run_id,
                        attempted_at=clock(),
                        price_bases=price_bases,
                    )
                    exhausted += sum(
                        1
                        for entry in entries
                        if entry.status == CaptureStatus.FAILED
                        and (before := prev_map.get(str(entry.symbol))) is not None
                        and before[0] == "FAILED"
                        and int(before[1]) + 1 >= EXTENDED_BACKFILL_MAX_FAILED_ATTEMPTS
                    )
            task_complete = sum(1 for entry in entries if entry.status == CaptureStatus.COMPLETE)
            task_no_trades = sum(1 for entry in entries if entry.status == CaptureStatus.NO_TRADES)
            task_not_listed = sum(1 for entry in entries if entry.status == CaptureStatus.NOT_APPLICABLE)
            task_failed = len(entries) - task_complete - task_no_trades - task_not_listed
            complete += task_complete
            no_trades += task_no_trades
            not_listed += task_not_listed
            failed += task_failed
            done += 1
            elapsed = (clock() - started).total_seconds()
            logger.info(
                "[DATA] stage=extended_backfill date=%s session=%s pending=%d complete=%d no_trades=%d not_listed=%d failed=%d repaired=%d adjusted=%d basis_failed=%d elapsed_s=%.1f",
                task.snapshot_date,
                task.session,
                len(pending),
                task_complete,
                task_no_trades,
                task_not_listed,
                task_failed,
                repaired,
                adjusted,
                basis_failed,
                elapsed,
            )
    if unreadable:
        raise OSError(f"Cannot read existing partition evidence for tasks {unreadable}")
    return ExtendedBackfillSummary(
        tasks_done=done,
        tasks_remaining=len(ordered) - done,
        complete=complete,
        no_trades=no_trades,
        not_listed=not_listed,
        failed=failed,
        stopped_by_deadline=stopped,
        exhausted=exhausted,
    )


def _coverage_or_none(
    plan: RegularBackfillPlan | None, ledger: ExtendedBackfillLedger, *, as_of: date, retention_days: int
) -> RegularBackfillCoverage | None:
    """Coverage telemetry that never blocks the run; an unreadable partition is reported by the runner instead."""
    if plan is None:
        return None
    try:
        return summarize_regular_backfill_coverage(plan, ledger, as_of=as_of, retention_days=retention_days)
    except OSError as exc:
        logger.error("[DATA] stage=regular_backfill_coverage status=UNAVAILABLE error=%s", type(exc).__name__)
        return None


def main() -> None:  # pragma: no cover - CLI entry; credential/client wiring, logic covered via run_extended_session_backfill scenarios
    """Run the nightly extended-session backfill until the stop time (exit 0 on deadline stop)."""
    import re

    from src.api.kis.key_pool import resolve_research_credentials
    from src.backfill.intraday import backfill_minute_history

    parser = argparse.ArgumentParser(description="Nightly extended-session 1m backfill (NXT/KRX).")
    parser.add_argument("--as-of", default=None, help="Run date YYYY-MM-DD (default KST today).")
    parser.add_argument(
        "--stop-at",
        default=None,
        help="HHMMSS KST after which no new task starts (default COLLECTION_BACKFILL_STOP_HHMMSS).",
    )
    parser.add_argument("--sessions", choices=("all", "extended", "regular"), default="all", help="Backfill streams to run.")
    args = parser.parse_args()
    profile = CollectionSettings()
    now = datetime.now(SEOUL)
    as_of = date.fromisoformat(str(args.as_of)) if args.as_of else now.date()
    stop_hhmmss = str(args.stop_at) if args.stop_at else str(profile.COLLECTION_BACKFILL_STOP_HHMMSS)
    if not re.fullmatch(r"\d{6}", stop_hhmmss):
        raise ValueError(f"Invalid --stop-at HHMMSS: {stop_hhmmss!r}")
    stop_at = datetime(
        now.year,
        now.month,
        now.day,
        int(stop_hhmmss[0:2]),
        int(stop_hhmmss[2:4]),
        int(stop_hhmmss[4:6]),
        tzinfo=SEOUL,
    )
    if stop_at <= now:
        stop_at += timedelta(days=1)
    slots = tuple(profile.COLLECTION_BACKFILL_SLOTS)
    if not slots:
        logger.info("[DATA] stage=extended_backfill status=SKIP reason=disabled")
        return
    env = dict(os.environ)
    creds = resolve_research_credentials(env, slots=slots)
    if not creds:
        raise ValueError("backfill slots resolved to no credentials")

    async def _run() -> ExtendedBackfillSummary:
        from src.api.kis.client import KisApiClient
        from src.api.kis.key_pool import token_cache_path

        from src import settings as app_settings

        clients = [
            KisApiClient(
                cred.app_key,
                cred.app_secret,
                "",
                cred.hts_id,
                token_file=str(token_cache_path(cred.app_key, app_settings.KIS_TOKEN_CACHE_DIR)),
            )
            for cred in creds
        ]
        store = CaptureStore(_capture_root(profile))
        ledger = ExtendedBackfillLedger()
        retention = int(profile.COLLECTION_KIS_MINUTE_RETENTION_DAYS)
        earliest = (as_of - timedelta(days=retention)).isoformat()
        load_start = earliest
        price_history = pd.read_parquet(
            app_settings.PRICE_HISTORY_PARQUET_PATH,
            columns=["date", "symbol", "close", "prev_close", "close_raw"],
        )
        window_days = _normalize_day_column(price_history["date"])
        price_history = price_history[
            (window_days >= load_start) & (window_days <= as_of.isoformat())
        ].copy()
        pair_frames: list[pd.DataFrame] = []
        for source in backfill_minute_history._load_condition_history_sources():
            if source.empty or "스냅샷_날짜" not in source.columns or "종목코드" not in source.columns:
                continue
            normalized = pd.DataFrame(
                {
                    "snapshot_date": _normalize_day_column(source["스냅샷_날짜"]),
                    "symbol": source["종목코드"].astype(str).str.zfill(6),
                }
            ).dropna()
            pair_frames.append(normalized.drop_duplicates())
        candidate_pairs = (
            pd.concat(pair_frames, ignore_index=True).drop_duplicates()
            if pair_frames
            else pd.DataFrame(columns=["snapshot_date", "symbol"])
        )
        kis_tasks = enumerate_extended_session_tasks(
            as_of=as_of,
            retention_days=retention,
            min_change_ratio=float(profile.COLLECTION_BACKFILL_MIN_CHANGE_RATIO),
            price_history=price_history,
            candidate_pairs=candidate_pairs,
        )
        run_extended = str(args.sessions) in ("all", "extended")
        run_regular = str(args.sessions) in ("all", "regular")
        regular_plan: RegularBackfillPlan | None = None
        if not profile.COLLECTION_PIT_BACKFILL_ENABLED:
            logger.info("[DATA] stage=regular_backfill status=SKIP reason=disabled")
            run_regular = False
        if run_regular:
            wide_columns = sorted(REQUIRED_SOURCE_COLUMNS | {"close_raw", "market"})
            wide_start = (as_of - timedelta(days=retention + int(MAX_PREV_TRADING_DAY_LOOKBACK))).isoformat()
            wide_history = pd.read_parquet(
                app_settings.PRICE_HISTORY_PARQUET_PATH,
                columns=wide_columns,
            )
            wide_days = _normalize_day_column(wide_history["date"])
            wide_history = wide_history[
                (wide_days >= wide_start) & (wide_days <= as_of.isoformat())
            ].copy()
            prepared, _provenance = prepare_price_panel(wide_history)
            price_reference = PriceReference.from_price_history(wide_history)
            regular_plan = enumerate_regular_session_tasks(
                as_of=as_of,
                retention_days=retention,
                prepared_panel=prepared,
                screen=EodSupersetScreen.from_profile(profile),
                price_reference=price_reference,
            )
            plan_days = [task.snapshot_date for task in regular_plan.tasks]
            logger.info(
                "[DATA] stage=regular_backfill_plan days=%d symbol_days=%d skipped_adjusted=%d skipped_unknown_basis=%d oldest=%s newest=%s",
                len(regular_plan.tasks),
                sum(len(task.symbols) for task in regular_plan.tasks),
                regular_plan.skipped_adjusted,
                regular_plan.skipped_unknown_basis,
                plan_days[0] if plan_days else "",
                plan_days[-1] if plan_days else "",
            )
        if clients:
            async with clients[0].create_session() as broker_session:
                for client in clients:
                    await client.ensure_token(broker_session)
        price_reference = PriceReference.from_price_history(price_history)
        from src.api.kiwoom.client import KiwoomApiClient

        raw_client = KiwoomApiClient() if app_settings.KIWOOM_APP_KEY else None
        if raw_client is None:
            logger.warning("[DATA] stage=extended_backfill kiwoom=unconfigured adjusted_days=fail_closed")
        tasks = _select_backfill_tasks(kis_tasks, regular_plan, run_extended=run_extended, run_regular=run_regular)
        start_coverage = _coverage_or_none(regular_plan, ledger, as_of=as_of, retention_days=retention)
        if start_coverage is not None:
            logger.info(
                "[DATA] stage=regular_backfill_coverage phase=%s days_total=%d days_resolved=%d symbol_days_total=%d symbol_days_resolved=%d symbol_days_pending=%d oldest_pending=%s oldest_pending_expiry_days=%d",
                "start",
                start_coverage.days_total,
                start_coverage.days_resolved,
                start_coverage.symbol_days_total,
                start_coverage.symbol_days_resolved,
                start_coverage.symbol_days_pending,
                start_coverage.oldest_pending_date,
                start_coverage.oldest_pending_expiry_days,
            )
        summary = await run_extended_session_backfill(
            as_of=as_of,
            stop_at=stop_at,
            profile=profile,
            clients=clients,
            store=store,
            ledger=ledger,
            tasks=tasks,
            price_reference=price_reference,
            raw_client=raw_client,
        )
        end_coverage = _coverage_or_none(regular_plan, ledger, as_of=as_of, retention_days=retention)
        if end_coverage is not None:
            logger.info(
                "[DATA] stage=regular_backfill_coverage phase=%s days_total=%d days_resolved=%d symbol_days_total=%d symbol_days_resolved=%d symbol_days_pending=%d oldest_pending=%s oldest_pending_expiry_days=%d",
                "end",
                end_coverage.days_total,
                end_coverage.days_resolved,
                end_coverage.symbol_days_total,
                end_coverage.symbol_days_resolved,
                end_coverage.symbol_days_pending,
                end_coverage.oldest_pending_date,
                end_coverage.oldest_pending_expiry_days,
            )
        return summary

    kis_summary = asyncio.run(_run())
    logger.info(
        "[DATA] stage=extended_backfill status=DONE tasks_done=%d tasks_remaining=%d complete=%d no_trades=%d not_listed=%d failed=%d exhausted=%d stopped_by_deadline=%s",
        kis_summary.tasks_done,
        kis_summary.tasks_remaining,
        kis_summary.complete,
        kis_summary.no_trades,
        kis_summary.not_listed,
        kis_summary.failed,
        kis_summary.exhausted,
        kis_summary.stopped_by_deadline,
    )


if __name__ == "__main__":  # pragma: no cover
    configure_cli_logging()
    main()
