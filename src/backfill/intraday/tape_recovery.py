"""Kiwoom tape-recovery engine shared by the manual backfill CLI and the daily self-healing sweep. Needs follow the
ADR-008 tick-bar contract: a session needs recovery only on a tick shortfall, missing/truncated/out-of-window ticks, judged on the
same bar window as the daily audit."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import re
import shutil
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src import settings
from src.backfill.intraday import blackout as _blackout
from src.backfill.intraday.tape_disposition import TapeDisposition, decide_tape_disposition
from src.backfill.intraday.tape_harvest import (
    TAPE_SESSIONS,
    TapeSession,
    TickTapePublisher,
    harvest_symbol_tape,
    is_session_closed,
)
from src.config.collection import CollectionSettings
from src.data.capture_contracts import SEOUL, CaptureStatus
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.intraday_store import _partition_row_count, intraday_partition_path, tick_partition_path
from src.data.session_calendar import SessionKind, resolve_session_day
from src.data.tick_bar_consistency import (
    TickBarRelation,
    classify_tick_bar_volume,
    comparable_bar_volumes,
    summed_tick_volumes,
)
from src.strategy.contract import DEFAULT_UNIVERSE

logger = logging.getLogger(__name__)

_MIN_START_MARGIN_SECONDS = 180
_MAX_VENDOR_FAILURE_STREAK = 3

SETTLED_TAPE_STATUSES: frozenset[str] = frozenset({CaptureStatus.COMPLETE.value, CaptureStatus.NO_TRADES.value})

UNRECOVERABLE_STATUS: str = "UNRECOVERABLE"
TERMINAL_TAPE_STATUSES: frozenset[str] = SETTLED_TAPE_STATUSES | {UNRECOVERABLE_STATUS}


@dataclass(frozen=True)
class TapeAttemptHistory:
    """Consecutive trailing non-settled ledger outcomes for one tape key."""

    attempts: int
    latest_reason: str
    latest_status: str


@dataclass(frozen=True)
class Need:
    """One (symbol, day, session) requiring tape recovery."""

    symbol: str
    day: str
    session: str
    venue: Literal["KRX", "NXT"]


@dataclass(frozen=True)
class WalkTask:
    """One tape walk covering all needed days of a symbol on one venue."""

    symbol: str
    venue: Literal["KRX", "NXT"]
    days: tuple[str, ...]
    sessions: tuple[TapeSession, ...]
    pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TapeRunSummary:
    """Outcome of one engine pass over walk tasks."""

    pages: int
    rows: int
    unresolved: int
    remaining: tuple[str, ...]
    stopped: bool
    stopped_reason: str
    skipped_unclosed: int = 0


def parse_walk_deadline(value: str | None) -> datetime | None:
    """Parse an aware ISO `--deadline` into KST; None passes through.

    Raises:
        ValueError: Unparseable or timezone-naive timestamp.
    """
    if value is None:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Invalid --deadline timestamp: {value!r}") from None
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"--deadline must be timezone-aware: {value!r}")
    return moment.astimezone(SEOUL)


def _now() -> datetime:
    return datetime.now(SEOUL)


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(max(0.0, float(seconds)))


def _in_blackout(now_minute: int, window: tuple[int, int]) -> bool:
    return _blackout._minute_in_window(int(now_minute), window)


def _blackout_end(now: datetime, windows: list[tuple[int, int]]) -> datetime | None:
    return _blackout.blackout_end(now, windows, weekdays_only=False)


async def _wait_for_blackout(windows: list[tuple[int, int]]) -> None:
    await _blackout.wait_for_blackout(windows, now_fn=_now, sleep_fn=_sleep, weekdays_only=False)


def _read_symbols(path: Path) -> list[str]:
    try:
        if not path.exists():
            return []
        frame = pd.read_parquet(path, columns=["symbol"])
    except (OSError, ValueError) as exc:
        logger.warning(
            "[DATA] stage=tape_backfill status=DEGRADED reason=unreadable_partition path=%s error=%s",
            path,
            type(exc).__name__,
        )
        return []
    if "symbol" not in frame.columns:  # pragma: no cover - column-filtered reads always carry symbol
        return []
    ordered: list[str] = []
    for item in frame["symbol"].astype(str).tolist():
        if item not in ordered:
            ordered.append(item)
    return ordered


def _read_cohort_symbols(store: CaptureStore, day: str) -> list[str]:
    try:
        cohort = store.read_cohort(day, available_by=_now())
    except (FileNotFoundError, ValueError, OSError):
        return []
    return [str(item) for item in cohort.eligible_symbols]


def _price_history_path() -> Path:
    return Path(settings.PRICE_HISTORY_PARQUET_PATH)


def _ingested_trading_days() -> set[str]:
    path = _price_history_path()
    try:
        if not path.exists():
            return set()
        frame = pd.read_parquet(path, columns=["date"])
    except (OSError, ValueError) as exc:
        logger.warning(
            "[DATA] stage=tape_backfill status=DEGRADED reason=price_history_unreadable error=%s", type(exc).__name__
        )
        return set()
    return {str(item) for item in pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").unique()}


def _is_trading_day(day: str, ingested: set[str]) -> bool:
    """Whether a KST day is a candidate for tape recovery.

    The verified session calendar is authoritative; price history only excludes
    in-range holidays. Never calls the network; invalid ISO dates raise ValueError.
    """
    moment = date.fromisoformat(day)
    if moment.weekday() >= 5:
        return False
    kind = resolve_session_day(moment).kind
    if kind is SessionKind.CLOSED:
        return False
    if kind is SessionKind.UNKNOWN:
        return day in ingested
    if not ingested:
        return True
    if min(ingested) <= day <= max(ingested):
        return day in ingested
    return True


POOL_UNIVERSE_START: str = "2026-09-11"
"""First day whose candidate pool follows the current universe definition (earlier archive pools are >=10% movers)."""

RECONSTRUCTED_TOP_TRADE_VALUE: int = 100
"""Top-N by trade value merged with the change band when a day's pool must be rebuilt from price_history."""

_ARCHIVE_DAY_COLUMN = "스냅샷_날짜"
_ARCHIVE_SYMBOL_COLUMN = "종목코드"


def _history_archive_path() -> Path:
    return Path(settings.HISTORY_PARQUET_PATH)


@functools.lru_cache(maxsize=4)
def _load_archive_pools(path: str, mtime_ns: int) -> dict[str, tuple[str, ...]]:
    del mtime_ns  # cache key only: a rewritten archive must not serve stale pools
    frame = pd.read_parquet(path, columns=[_ARCHIVE_DAY_COLUMN, _ARCHIVE_SYMBOL_COLUMN])
    days = pd.to_datetime(frame[_ARCHIVE_DAY_COLUMN]).dt.strftime("%Y-%m-%d")
    codes = frame[_ARCHIVE_SYMBOL_COLUMN].astype(str).str.zfill(6)
    pools: dict[str, tuple[str, ...]] = {}
    for day, group in codes.groupby(days):
        pools[str(day)] = tuple(dict.fromkeys(group.tolist()))
    return pools


def _archive_pool(day: str) -> list[str]:
    path = _history_archive_path()
    try:
        if not path.exists():
            return []
        pools = _load_archive_pools(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            "[DATA] stage=tape_backfill status=DEGRADED reason=archive_unreadable error=%s", type(exc).__name__
        )
        return []
    return list(pools.get(day, ()))


@functools.lru_cache(maxsize=4)
def _load_price_rows(path: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    frame = pd.read_parquet(
        path,
        columns=["date", "symbol", "close", "prev_close", "trade_value_100m"],
        filters=[("date", ">=", pd.Timestamp(POOL_UNIVERSE_START))],
    )
    frame = frame.assign(day=pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d"), symbol=frame["symbol"].astype(str))
    return frame.drop(columns=["date"])


def _reconstructed_pool(day: str) -> list[str]:
    """Rebuild one day's candidate pool from end-of-day prices (tick-target selection only, never decision-time).

    Applies the current universe's change band plus the top trade-value names; end-of-day inputs make the result
    look-ahead relative to the live 15:20 scan, so it must never feed strategy evaluation.
    """
    path = _price_history_path()
    try:
        if not path.exists():
            return []
        rows = _load_price_rows(str(path), path.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(
            "[DATA] stage=tape_backfill status=DEGRADED reason=price_history_unreadable error=%s", type(exc).__name__
        )
        return []
    today = rows[rows["day"] == day]
    if today.empty:
        return []
    change = today["close"] / today["prev_close"] - 1.0
    band = today[(change >= DEFAULT_UNIVERSE.chg_min) & (change < DEFAULT_UNIVERSE.chg_max)]["symbol"]
    top = today.nlargest(RECONSTRUCTED_TOP_TRADE_VALUE, "trade_value_100m")["symbol"]
    return sorted(set(band.tolist()) | set(top.tolist()))


def _pool_symbols(store: CaptureStore, day: str) -> tuple[list[str], str]:
    """One day's candidate pool and its provenance: cohort, archive_pool, reconstructed, or none.

    Archive and reconstructed pools apply only from POOL_UNIVERSE_START; earlier archive rows follow a different
    (>=10% mover) universe and are deliberately not used.
    """
    cohort = _read_cohort_symbols(store, day)
    if cohort:
        return cohort, "cohort"
    if day >= POOL_UNIVERSE_START:
        archived = _archive_pool(day)
        if archived:
            return archived, "archive_pool"
        rebuilt = _reconstructed_pool(day)
        if rebuilt:
            return rebuilt, "reconstructed"
    return [], "none"


def _day_universe(day: str, store: CaptureStore) -> list[str]:
    codes: list[str] = []
    own, own_source = _pool_symbols(store, day)
    for item in own:
        if item not in codes:
            codes.append(item)
    base = date.fromisoformat(day)
    prev_source, prev_day = "none", ""
    for offset in range(1, 11):
        prev = (base - timedelta(days=offset)).isoformat()
        prev_codes, prev_source = _pool_symbols(store, prev)
        if prev_codes:
            prev_day = prev
            for item in prev_codes:
                if item not in codes:
                    codes.append(item)
            break
    logger.info(
        "[DATA] stage=tape_universe day=%s own=%s:%d prev=%s:%s total_pool=%d",
        day,
        own_source,
        len(own),
        prev_day or "none",
        prev_source,
        len(codes),
    )
    for session in tuple(s.session for s in TAPE_SESSIONS):
        for item in _read_symbols(tick_partition_path(day, session)):
            if item not in codes:
                codes.append(item)
    if not codes:
        for session in ("nxt_premarket", "nxt_aftermarket"):
            for item in _read_symbols(intraday_partition_path(1, day, session)):
                if item not in codes:
                    codes.append(item)
    return codes


@dataclass(frozen=True)
class _TickStats:
    """Per-symbol summary of one stored tick partition."""

    rows: int
    truncated: bool
    out_of_window: bool
    volume: float


def _partition_columns(path: Path, wanted: tuple[str, ...]) -> list[str]:
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema_arrow.names)
    return [c for c in wanted if c in names]


class PartitionIndex:
    """Per-run cache of per-symbol partition summaries so each partition is read once, not once per symbol."""

    def __init__(self) -> None:
        self._ticks: dict[tuple[str, str], dict[str, _TickStats] | None] = {}
        self._bars: dict[tuple[str, str], dict[str, float] | None] = {}

    def tick_stats(self, day: str, spec: TapeSession) -> dict[str, _TickStats] | None:
        key = (day, spec.session)
        if key not in self._ticks:
            self._ticks[key] = self._load_ticks(day, spec)
        return self._ticks[key]

    def bar_volumes(self, day: str, session: str) -> dict[str, float] | None:
        key = (day, session)
        if key not in self._bars:
            self._bars[key] = self._load_bars(day, session)
        return self._bars[key]

    @staticmethod
    def _load_ticks(day: str, spec: TapeSession) -> dict[str, _TickStats] | None:
        path = tick_partition_path(day, spec.session)
        try:
            if not path.exists():
                return None
            frame = pd.read_parquet(path, columns=_partition_columns(path, ("symbol", "ts_hms", "volume", "truncated")))
        except (OSError, ValueError) as exc:
            logger.warning(
                "[DATA] stage=tape_backfill status=DEGRADED reason=unreadable_partition path=%s error=%s",
                path,
                type(exc).__name__,
            )
            return None
        if "symbol" not in frame.columns:
            return None
        frame = frame.assign(symbol=frame["symbol"].astype(str))
        floor, ceil = int(spec.floor), int(spec.ceil)
        stamps = (
            pd.to_numeric(frame["ts_hms"], errors="coerce")
            if "ts_hms" in frame.columns
            else pd.Series(floor, index=frame.index)
        )
        frame["_outside"] = (stamps < floor) | (stamps > ceil)
        frame["_trunc"] = frame["truncated"].fillna(False).astype(bool) if "truncated" in frame.columns else False
        if "volume" in frame.columns:
            volumes = summed_tick_volumes(frame.loc[:, ["symbol", "volume"]])
        else:
            volumes = {str(sym): 0.0 for sym in frame["symbol"].astype(str).unique().tolist()}
        grouped = frame.groupby("symbol").agg(
            rows=("symbol", "size"),
            truncated=("_trunc", "any"),
            outside=("_outside", "any"),
        )
        return {
            str(sym): _TickStats(
                rows=int(r.rows),
                truncated=bool(r.truncated),
                out_of_window=bool(r.outside),
                volume=float(volumes.get(str(sym), 0.0)),
            )
            for sym, r in grouped.iterrows()
        }

    @staticmethod
    def _load_bars(day: str, session: str) -> dict[str, float] | None:
        path = intraday_partition_path(1, day, session)
        try:
            if not path.exists():
                return None
            frame = pd.read_parquet(path, columns=_partition_columns(path, ("symbol", "ts_hms", "volume")))
        except (OSError, ValueError) as exc:
            logger.warning(
                "[DATA] stage=tape_backfill status=DEGRADED reason=unreadable_partition path=%s error=%s",
                path,
                type(exc).__name__,
            )
            return None
        if frame.empty or "symbol" not in frame.columns or "volume" not in frame.columns:
            return None
        return comparable_bar_volumes(session, frame)


def _session_need(symbol: str, day: str, spec: TapeSession, index: PartitionIndex) -> bool:
    stats = index.tick_stats(day, spec)
    if stats is None:
        return True
    row = stats.get(str(symbol))
    if row is None or row.rows == 0 or row.truncated or row.out_of_window:
        return True
    bars = index.bar_volumes(day, spec.session)
    bar_volume = None if bars is None else bars.get(str(symbol))
    return (
        bar_volume is not None
        and classify_tick_bar_volume(spec.session, bar_volume, row.volume) is TickBarRelation.TICK_SHORT
    )


def _session_closed(day: str, session: str, now: datetime) -> bool:
    return is_session_closed(day, session, now)


def collect_tape_needs(
    days: list[str],
    venues: list[Literal["KRX", "NXT"]],
    store: CaptureStore,
    settled: Mapping[tuple[str, str, str], str],
    force: bool,
    index: PartitionIndex | None = None,
) -> list[Need]:
    """Select (symbol, day, session) keys whose stored ticks need tape recovery.

    Only closed trading days are considered; keys already settled in the ledger are skipped unless forced. A session
    needs recovery when its tick partition or symbol row is absent, empty, truncated or out of window, or when the
    tick volume is short of `comparable_bar_volumes` under the ADR-008 policy (tick surplus is never a need).

    Args:
        days: Candidate KST days (YYYY-MM-DD).
        venues: Tape venues to evaluate.
        store: Capture store used for cohort-based universes.
        settled: Ledger status per (symbol, day, session).
        force: Re-select keys regardless of ledger status and volume relation.
        index: Optional partition cache shared across calls; a fresh one is built when None.

    Returns:
        Needs in (day, venue, session, universe) iteration order.
    """
    now = _now()
    cache = index if index is not None else PartitionIndex()
    needs: list[Need] = []
    ingested = _ingested_trading_days()
    for day in days:
        if not _is_trading_day(day, ingested):
            continue
        universe = _day_universe(day, store)
        for venue in venues:
            for spec in [s for s in TAPE_SESSIONS if s.venue == venue]:
                if not _session_closed(day, spec.session, now):
                    continue
                for symbol in universe:
                    if not force and settled.get((symbol, day, spec.session)) in TERMINAL_TAPE_STATUSES:
                        continue
                    if force or _session_need(symbol, day, spec, cache):
                        needs.append(Need(symbol=symbol, day=day, session=spec.session, venue=venue))
    return needs


def order_walk_tasks(needs: list[Need]) -> list[WalkTask]:
    """Group needs into one tape walk per (symbol, venue), ordered by earliest day, symbol, venue."""
    grouped: dict[tuple[str, Literal["KRX", "NXT"]], dict[str, Any]] = {}
    for need in needs:
        key = (need.symbol, need.venue)
        slot = grouped.setdefault(key, {"days": set(), "sessions": set(), "pairs": set()})
        slot["days"].add(need.day)
        slot["sessions"].add(need.session)
        slot["pairs"].add((need.day, need.session))
    tasks: list[WalkTask] = []
    for (symbol, venue), slot in grouped.items():
        by_name = {s.session: s for s in TAPE_SESSIONS if s.venue == venue}
        sessions = tuple(by_name[name] for name in sorted(slot["sessions"]) if name in by_name)
        tasks.append(
            WalkTask(
                symbol=symbol,
                venue=venue,
                days=tuple(sorted(slot["days"])),
                sessions=sessions,
                pairs=tuple(sorted(slot["pairs"])),
            )
        )
    tasks.sort(key=lambda t: (min(t.days), t.symbol, t.venue))
    return tasks


def tape_ledger_path(override: str | None, profile: CollectionSettings) -> Path:
    """Ledger location: the override when given, else `<capture root>/staging/tape_backfill/ledger.jsonl`."""
    if override:
        return Path(str(override))
    return _capture_root(profile) / "staging" / "tape_backfill" / "ledger.jsonl"


def read_settled_ledger(path: Path) -> dict[tuple[str, str, str], str]:
    """Latest ledger status per (symbol, day, session); empty when the ledger is absent.

    Raises:
        RuntimeError: The ledger exists but cannot be read.
    """
    settled: dict[tuple[str, str, str], str] = {}
    try:
        if not path.exists():
            return settled
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Tape backfill ledger unreadable: {path}: {exc}") from exc
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        settled[(str(record["symbol"]), str(record["day"]), str(record["session"]))] = str(record.get("status", ""))
    return settled


_LEGACY_RUN_ID = re.compile(r"^tape-(\d{4}-\d{2}-\d{2})-\d+$")


def _legacy_run_date(run_id: str) -> str:
    match = _LEGACY_RUN_ID.match(run_id)
    return match.group(1) if match else ""


def read_tape_attempts(path: Path) -> dict[tuple[str, str, str], TapeAttemptHistory]:
    """Per (symbol, day, session): count of consecutive trailing non-settled records on distinct `run_date`s and the latest reason/status. Legacy UNKNOWN records without a reason are interpreted as reason `day_not_on_tape` (the only UNKNOWN producer when the venue is certified; the share guard in `terminalize_unrecoverable` protects against venue-wide misconfiguration). Legacy PARTIAL records without a reason stay retryable. Raises RuntimeError when the ledger exists but is unreadable."""
    try:
        if not path.exists():
            return {}
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Tape backfill ledger unreadable: {path}: {exc}") from exc
    ordered: dict[tuple[str, str, str], list[tuple[str, str, str]]] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        try:
            key = (str(record["symbol"]), str(record["day"]), str(record["session"]))
        except KeyError:
            continue
        status = str(record.get("status", ""))
        reason = str(record.get("reason", "") or "")
        if not reason and status == CaptureStatus.UNKNOWN.value:
            reason = "day_not_on_tape"
        run_date = str(record.get("run_date", "") or "") or _legacy_run_date(str(record.get("run_id", "")))
        ordered.setdefault(key, []).append((status, reason, run_date))
    history: dict[tuple[str, str, str], TapeAttemptHistory] = {}
    for key, records in ordered.items():
        latest_status = records[-1][0]
        latest_reason = records[-1][1]
        seen_dates: set[str] = set()
        for index, (status, _reason, run_date) in enumerate(reversed(records)):
            if status in SETTLED_TAPE_STATUSES:
                break
            if run_date and run_date == key[1]:
                continue
            marker = run_date if run_date else f"#legacy-{len(records) - 1 - index}"
            seen_dates.add(marker)
        history[key] = TapeAttemptHistory(
            attempts=len(seen_dates), latest_reason=latest_reason, latest_status=latest_status
        )
    return history


def terminalize_unrecoverable(
    keys: Collection[tuple[str, str, str]],
    *,
    history: Mapping[tuple[str, str, str], TapeAttemptHistory],
    attempted: int,
    min_attempts: int,
    max_share: float,
    ledger: Path,
    run_id: str,
    run_date: str,
) -> tuple[tuple[str, str, str], ...]:
    """Appends one `UNRECOVERABLE` ledger record (with reason, attempts, run_id, run_date) per key whose history decides UNRECOVERABLE, unless the terminal share of `attempted` exceeds `max_share`, in which case nothing is written and a warning `[DATA] stage=tape_sweep status=TERMINALIZE_BLOCKED` is logged. Returns the keys terminalized."""
    ordered = list(keys)
    candidates = [
        key
        for key in ordered
        if (entry := history.get(key)) is not None
        and entry.latest_status not in TERMINAL_TAPE_STATUSES
        and decide_tape_disposition(
            attempts=entry.attempts, latest_reason=entry.latest_reason, min_attempts=min_attempts
        )
        is TapeDisposition.UNRECOVERABLE
    ]
    if not candidates:
        return ()
    if int(attempted) > 0 and len(candidates) / int(attempted) > float(max_share):
        logger.warning(
            "[DATA] stage=tape_sweep status=TERMINALIZE_BLOCKED candidates=%d attempted=%d",
            len(candidates),
            int(attempted),
        )
        return ()
    records = [
        {
            "symbol": key[0],
            "day": key[1],
            "session": key[2],
            "status": UNRECOVERABLE_STATUS,
            "reason": history[key].latest_reason,
            "attempts": history[key].attempts,
            "run_id": run_id,
            "run_date": run_date,
        }
        for key in candidates
    ]
    _append_ledger(ledger, records)
    return tuple(candidates)


def _append_ledger(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
    except OSError as exc:
        raise RuntimeError(f"Tape backfill ledger write failed: {path}: {exc}") from exc


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _heartbeat(done: int, total: int, pages: int, rows: int, unresolved: int, started: float) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    eta = elapsed / done * (total - done) / 60.0 if done > 0 and total > done else 0.0
    logger.info(
        "[DATA] stage=tape_backfill done=%d/%d pages=%d rows=%d unresolved=%d eta_min=%.1f",
        done,
        total,
        pages,
        rows,
        unresolved,
        eta,
    )


def _verify_groups(expected: dict[tuple[str, str], int]) -> None:
    for (day, session), minimum in expected.items():
        try:
            total = _partition_row_count(tick_partition_path(day, session))
        except OSError as exc:
            raise RuntimeError(f"Tape backfill publication failed day={day} session={session}: {exc}") from exc
        if total < minimum:
            raise RuntimeError(
                f"Tape backfill publication verification failed day={day} session={session}: total={total} expected={minimum}"
            )


def _next_blackout_start(now: datetime, windows: list[tuple[int, int]]) -> datetime | None:
    """Earliest upcoming blackout start strictly after `now` (today only), None when none remains."""
    now_minute = now.hour * 60 + now.minute
    starts = [start for start, _end in windows if start > now_minute]
    if not starts:
        return None
    start = min(starts)
    return now.replace(hour=start // 60, minute=start % 60, second=0, microsecond=0)


def _disk_ok(root: Path, profile: CollectionSettings) -> bool:
    """True when free disk is above the floor, after one expired-snapshot prune when it is not."""
    floor = int(profile.COLLECTION_TAPE_MIN_FREE_GIB) * 1024**3
    if _free_bytes(root) >= floor:
        return True
    from src.tools.backup_prune import prune_local_intraday_backups

    purged = prune_local_intraday_backups(today=pd.Timestamp(_now().date()))
    logger.warning("[DATA] stage=tape_backfill status=DISK_GUARD purged=%d", len(purged))
    return _free_bytes(root) >= floor


async def run_walk_tasks(
    tasks: list[WalkTask],
    *,
    client: Any,
    http_session: Any,
    store: CaptureStore,
    profile: CollectionSettings,
    apply: bool,
    ledger: Path,
    deadline: datetime | None,
    blackouts: list[tuple[int, int]],
    run_date: str,
) -> TapeRunSummary:
    """Walk the Kiwoom tapes for each task, publishing and ledgering certified days when `apply` is set.

    Blackout windows and the optional deadline keep walks out of slots reserved for live units, because the tape key
    shares the 5 req/s vendor quota with them. Publication is verified per (day, session) before the ledger is appended,
    so a ledger row never claims ticks that are not on disk.

    Returns:
        TapeRunSummary; `remaining` lists `symbol/venue` of tasks not started when stopping on deadline, disk guard or
        vendor-failure streak (`stopped_reason` in {"", "deadline", "disk_guard", "vendor_failure"}).

    Raises:
        RuntimeError: Evidence, publication or ledger failure.
    """
    publisher = (
        TickTapePublisher(
            store=store, profile=profile, flush_rows=int(profile.COLLECTION_TAPE_FLUSH_ROWS), auto_flush=False
        )
        if apply
        else None
    )
    root = _capture_root(profile)
    expected: dict[tuple[str, str], int] = {}
    pending_ledger: list[dict[str, Any]] = []
    pages = rows = unresolved = skipped_unclosed = failure_streak = 0
    remaining: list[str] = []
    started = time.monotonic()
    stopped_reason = ""

    def _commit() -> None:
        assert publisher is not None
        publisher.flush()
        _verify_groups(expected)
        _append_ledger(ledger, pending_ledger)
        expected.clear()
        pending_ledger.clear()

    for index, task in enumerate(tasks):
        await _wait_for_blackout(blackouts)
        upcoming = _next_blackout_start(_now(), blackouts)
        if upcoming is not None and (upcoming - _now()).total_seconds() < _MIN_START_MARGIN_SECONDS:
            await _sleep((upcoming - _now()).total_seconds() + 1)
            await _wait_for_blackout(blackouts)
            upcoming = _next_blackout_start(_now(), blackouts)
        if deadline is not None and _now() > deadline:
            remaining = [f"{t.symbol}/{t.venue}" for t in tasks[index:]]
            stopped_reason = "deadline"
            logger.info("[DATA] stage=tape_backfill status=DEADLINE remaining=%s", remaining)
            break
        if apply and not _disk_ok(root, profile):
            remaining = [f"{t.symbol}/{t.venue}" for t in tasks[index:]]
            stopped_reason = "disk_guard"
            break
        run_id = f"tape-{run_date}-{index:04d}"
        results: list[Any] = []
        try:
            outcome = await harvest_symbol_tape(
                client,
                http_session,
                task.symbol,
                list(task.days),
                venue=task.venue,
                sessions=list(task.sessions),
                store=store,
                run_id=run_id,
                profile=profile,
                on_result=results.append,
                walk_deadline=upcoming,
                needed=task.pairs,
            )
        except OSError as exc:
            raise RuntimeError(f"Tape backfill evidence failed symbol={task.symbol}: {exc}") from exc
        pages += int(outcome.pages_fetched)
        skipped_unclosed += len(outcome.skipped_unclosed)
        if outcome.unresolved_days:
            logger.info(
                "[DATA] stage=tape_backfill symbol=%s venue=%s status=GUARD_STOP remaining_days=%d",
                task.symbol,
                task.venue,
                len(outcome.unresolved_days),
            )
        if apply:
            assert publisher is not None
            for result in results:
                try:
                    publisher.add(result)
                except ValueError as exc:
                    raise ValueError(
                        f"{exc} task={index} symbol={result.symbol!r} day={result.day!r}"
                        f" session={result.session!r}"
                    ) from exc
                if result.entry.status == CaptureStatus.COMPLETE:
                    key = (result.day, result.session)
                    expected[key] = expected.get(key, 0) + len(result.frame)
                pending_ledger.append(
                    {
                        "symbol": result.symbol,
                        "day": result.day,
                        "session": result.session,
                        "status": result.entry.status.value,
                        "reason": result.entry.reason,
                        "run_id": run_id,
                        "run_date": run_date,
                    }
                )
            if publisher.should_flush():
                _commit()
        else:
            rows += sum(len(r.frame) for r in results if r.entry.status == CaptureStatus.COMPLETE)
        unresolved += len(outcome.unresolved_days)
        if outcome.termination_reason == "vendor_failure":
            failure_streak += 1
            client.reset_token()
            if failure_streak >= _MAX_VENDOR_FAILURE_STREAK:
                remaining = [f"{t.symbol}/{t.venue}" for t in tasks[index + 1 :]]
                stopped_reason = "vendor_failure"
                logger.warning(
                    "[DATA] stage=tape_backfill status=VENDOR_FAILURE_STREAK streak=%d remaining=%d",
                    failure_streak,
                    len(remaining),
                )
                break
        else:
            failure_streak = 0
        done = index + 1
        if done % 50 == 0:
            _heartbeat(done, len(tasks), pages, rows, unresolved, started)
    if apply:
        _commit()
    _heartbeat(len(tasks) - len(remaining), len(tasks), pages, rows, unresolved, started)
    return TapeRunSummary(
        pages=pages,
        rows=rows,
        unresolved=unresolved,
        remaining=tuple(remaining),
        stopped=bool(remaining),
        stopped_reason=stopped_reason,
        skipped_unclosed=skipped_unclosed,
    )
