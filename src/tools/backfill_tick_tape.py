"""One-shot recovery of missing tick days from the Kiwoom tapes.

Tape needs share the tick-bar consistency contract: a session needs recovery only on tick
shortfall. Aftermarket needs now include any tick shortfall (previously > 1%); sessions come
from TAPE_SESSIONS, all of which have a policy.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src import settings
from src.backfill.intraday.tape_harvest import (
    TAPE_SESSIONS,
    TapeSession,
    TickTapePublisher,
    harvest_symbol_tape,
)
from src.config.collection import CollectionSettings
from src.data.capture_contracts import SEOUL, CaptureStatus
from src.data.capture_store import CaptureStore
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.intraday_store import _partition_row_count, intraday_partition_path, tick_partition_path
from src.data.tick_bar_consistency import TickBarRelation, classify_tick_bar_volume
from src.strategy.contract import DEFAULT_UNIVERSE
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)

_MAX_BACKFILL_SPAN_DAYS = 31
_MIN_START_MARGIN_SECONDS = 180
_EST_PAGES_PER_DAY = 30
_EST_BYTES_PER_PAGE = 50_000
_CLOSE_AUCTION_TS = 153000
_REGULAR_READY_HHMMSS = "154000"
_TICK_SESSIONS = ("regular", "krx_aftermarket", "nxt_aftermarket")
# Live Kiwoom units share this key's 5 req/s limit; a slowed 15:20 collect also delays auction-close, which needs its cohort.
# Windows cover collect/predict (15:20), regular archive (15:40-16:30), aftermarket archive (20:05-21:05), price-ingest, extended backfill.
_DEFAULT_BLACKOUTS = ("08:25-08:45", "11:25-11:45", "15:15-17:00", "20:00-21:10", "21:25-21:45", "23:00-23:20")
_MAX_VENDOR_FAILURE_STREAK = 3


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


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover missing tick days from the Kiwoom tapes.")
    parser.add_argument("--start", required=True, help="Inclusive backfill start (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Inclusive backfill end (YYYY-MM-DD).")
    parser.add_argument("--venue", default="all", help="Tape venue in {krx,nxt,all}.")
    parser.add_argument("--apply", action="store_true", help="Publish certified days through partition writers.")
    parser.add_argument("--deadline", default=None, help="Aware ISO timestamp after which no new walk starts.")
    parser.add_argument("--symbols-limit", type=int, default=None, help="Maximum distinct symbols to walk.")
    parser.add_argument("--blackout", action="append", default=[], help="Repeatable blackout window HH:MM-HH:MM (KST).")
    parser.add_argument("--ledger", default=None, help="Ledger JSONL path (default under the capture root).")
    parser.add_argument("--force", action="store_true", help="Re-recover certified days.")
    return parser.parse_args(argv)


def _parse_day(value: str, field: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Invalid {field} date: {value!r}") from None


def _parse_deadline(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"Invalid --deadline timestamp: {value!r}") from None
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(f"--deadline must be timezone-aware: {value!r}")
    return moment.astimezone(SEOUL)


def _parse_blackout(spec: str) -> tuple[int, int]:
    try:
        start_raw, end_raw = str(spec).split("-", 1)
        start_h, start_m = start_raw.split(":", 1)
        end_h, end_m = end_raw.split(":", 1)
        start = int(start_h) * 60 + int(start_m)
        end = int(end_h) * 60 + int(end_m)
    except ValueError:
        raise ValueError(f"Invalid --blackout window: {spec!r} (expected HH:MM-HH:MM)") from None
    if not (0 <= start < 24 * 60 and 0 <= end <= 24 * 60):
        raise ValueError(f"Invalid --blackout window: {spec!r} (expected HH:MM-HH:MM)")
    return start, end


def _now() -> datetime:
    return datetime.now(SEOUL)


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(max(0.0, float(seconds)))


def _in_blackout(now_minute: int, window: tuple[int, int]) -> bool:
    start, end = window
    if end <= start:
        return now_minute >= start or now_minute < end
    return start <= now_minute < end


def _blackout_end(now: datetime, windows: list[tuple[int, int]]) -> datetime | None:
    now_minute = now.hour * 60 + now.minute
    ends: list[datetime] = []
    for start, end in windows:
        if not _in_blackout(now_minute, (start, end)):
            continue
        if end <= start:
            target = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            target += timedelta(minutes=end)
            if now_minute < end:
                target = now.replace(hour=end // 60, minute=end % 60, second=0, microsecond=0)
        elif end >= 24 * 60:
            target = now.replace(hour=23, minute=59, second=59, microsecond=0) + timedelta(seconds=1)
        else:
            target = now.replace(hour=end // 60, minute=end % 60, second=0, microsecond=0)
        ends.append(target)
    return max(ends) if ends else None


async def _wait_for_blackout(windows: list[tuple[int, int]]) -> None:
    while True:
        end = _blackout_end(_now(), windows)
        if end is None:
            return
        await _sleep((end - _now()).total_seconds())


def _open_kiwoom() -> tuple[Any, Any]:
    import aiohttp

    from src.api.kiwoom.client import KiwoomApiClient

    if not settings.KIWOOM_APP_KEY:
        raise RuntimeError("Kiwoom credentials are not configured")
    return KiwoomApiClient(), aiohttp.ClientSession()


def _read_symbols(path: Path) -> list[str]:
    try:
        if not path.exists():
            return []
        frame = pd.read_parquet(path, columns=["symbol"])
    except (OSError, ValueError) as exc:
        logger.warning("[DATA] stage=tape_backfill status=DEGRADED reason=unreadable_partition path=%s error=%s", path, type(exc).__name__)
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
        logger.warning("[DATA] stage=tape_backfill status=DEGRADED reason=price_history_unreadable error=%s", type(exc).__name__)
        return set()
    return {str(item) for item in pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").unique()}


def _is_trading_day(day: str, ingested: set[str]) -> bool:
    """Weekday check, tightened by price_history inside its ingested range (holidays carry no rows)."""
    if date.fromisoformat(day).weekday() >= 5:
        return False
    if not ingested or day < min(ingested) or day > max(ingested):
        return True
    return day in ingested


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
        logger.warning("[DATA] stage=tape_backfill status=DEGRADED reason=archive_unreadable error=%s", type(exc).__name__)
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
        logger.warning("[DATA] stage=tape_backfill status=DEGRADED reason=price_history_unreadable error=%s", type(exc).__name__)
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
        day, own_source, len(own), prev_day or "none", prev_source, len(codes),
    )
    for session in _TICK_SESSIONS:
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
            logger.warning("[DATA] stage=tape_backfill status=DEGRADED reason=unreadable_partition path=%s error=%s", path, type(exc).__name__)
            return None
        if "symbol" not in frame.columns:
            return None
        frame = frame.assign(symbol=frame["symbol"].astype(str))
        floor, ceil = int(spec.floor), int(spec.ceil)
        stamps = pd.to_numeric(frame["ts_hms"], errors="coerce") if "ts_hms" in frame.columns else pd.Series(floor, index=frame.index)
        frame["_outside"] = (stamps < floor) | (stamps > ceil)
        frame["_vol"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0) if "volume" in frame.columns else 0.0
        frame["_trunc"] = frame["truncated"].fillna(False).astype(bool) if "truncated" in frame.columns else False
        grouped = frame.groupby("symbol").agg(
            rows=("symbol", "size"), truncated=("_trunc", "any"), outside=("_outside", "any"), volume=("_vol", "sum"),
        )
        return {
            str(sym): _TickStats(rows=int(r.rows), truncated=bool(r.truncated), out_of_window=bool(r.outside), volume=float(r.volume))
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
            logger.warning("[DATA] stage=tape_backfill status=DEGRADED reason=unreadable_partition path=%s error=%s", path, type(exc).__name__)
            return None
        if frame.empty or "symbol" not in frame.columns or "volume" not in frame.columns:
            return None
        if session == "regular" and "ts_hms" in frame.columns:
            stamps = pd.to_numeric(frame["ts_hms"], errors="coerce")
            frame = frame.loc[stamps.isna() | (stamps < _CLOSE_AUCTION_TS)]
        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0)
        return {str(k): float(v) for k, v in volume.groupby(frame["symbol"].astype(str)).sum().items()}


def _session_need(symbol: str, day: str, spec: TapeSession, index: PartitionIndex) -> bool:
    stats = index.tick_stats(day, spec)
    if stats is None:
        return True
    row = stats.get(str(symbol))
    if row is None or row.rows == 0 or row.truncated or row.out_of_window:
        return True
    bars = index.bar_volumes(day, spec.session)
    bar_volume = None if bars is None else bars.get(str(symbol))
    return bar_volume is not None and classify_tick_bar_volume(spec.session, bar_volume, row.volume) is TickBarRelation.TICK_SHORT


def _session_closed(day: str, session: str, now: datetime) -> bool:
    today = now.date().isoformat()
    if session == "regular":
        return day < today or (day == today and now.strftime("%H%M%S") >= _REGULAR_READY_HHMMSS)
    return day < today


def _collect_needs(
    days: list[str],
    venues: list[Literal["KRX", "NXT"]],
    store: CaptureStore,
    settled: dict[tuple[str, str, str], str],
    force: bool,
    index: PartitionIndex | None = None,
) -> list[Need]:
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
                    if not force and settled.get((symbol, day, spec.session)) in ("COMPLETE", "NO_TRADES"):
                        continue
                    if force or _session_need(symbol, day, spec, cache):
                        needs.append(Need(symbol=symbol, day=day, session=spec.session, venue=venue))
    return needs


def _order_tasks(needs: list[Need]) -> list[WalkTask]:
    grouped: dict[tuple[str, Literal["KRX", "NXT"]], dict[str, Any]] = {}
    for need in needs:
        key = (need.symbol, need.venue)
        slot = grouped.setdefault(key, {"days": set(), "sessions": set()})
        slot["days"].add(need.day)
        slot["sessions"].add(need.session)
    tasks: list[WalkTask] = []
    for (symbol, venue), slot in grouped.items():
        by_name = {s.session: s for s in TAPE_SESSIONS if s.venue == venue}
        sessions = tuple(by_name[name] for name in sorted(slot["sessions"]) if name in by_name)
        tasks.append(WalkTask(symbol=symbol, venue=venue, days=tuple(sorted(slot["days"])), sessions=sessions))
    tasks.sort(key=lambda t: (min(t.days), t.symbol, t.venue))
    return tasks


def _ledger_path(override: str | None, profile: CollectionSettings) -> Path:
    if override:
        return Path(str(override))
    return _capture_root(profile) / "staging" / "tape_backfill" / "ledger.jsonl"


def _read_settled(path: Path) -> dict[tuple[str, str, str], str]:
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


def _project_evidence(tasks: list[WalkTask], profile: CollectionSettings) -> tuple[int, int]:
    guard = int(profile.COLLECTION_TICK_REPAIR_MAX_PAGES)
    pages = sum(min(guard, _EST_PAGES_PER_DAY * len(task.days)) for task in tasks)
    return pages, pages * _EST_BYTES_PER_PAGE


def _free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def _heartbeat(done: int, total: int, pages: int, rows: int, unresolved: int, started: float) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    eta = elapsed / done * (total - done) / 60.0 if done > 0 and total > done else 0.0
    logger.info(
        "[DATA] stage=tape_backfill done=%d/%d pages=%d rows=%d unresolved=%d eta_min=%.1f",
        done, total, pages, rows, unresolved, eta,
    )


def _verify_groups(expected: dict[tuple[str, str], int]) -> None:
    for (day, session), minimum in expected.items():
        try:
            total = _partition_row_count(tick_partition_path(day, session))
        except OSError as exc:
            raise RuntimeError(f"Tape backfill publication failed day={day} session={session}: {exc}") from exc
        if total < minimum:
            raise RuntimeError(f"Tape backfill publication verification failed day={day} session={session}: total={total} expected={minimum}")


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


async def _run_tasks(
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
) -> dict[str, Any]:
    publisher = (
        TickTapePublisher(store=store, profile=profile, flush_rows=int(profile.COLLECTION_TAPE_FLUSH_ROWS), auto_flush=False)
        if apply else None
    )
    root = _capture_root(profile)
    expected: dict[tuple[str, str], int] = {}
    pending_ledger: list[dict[str, Any]] = []
    pages = rows = unresolved = failure_streak = 0
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
                client, http_session, task.symbol, list(task.days), venue=task.venue,
                sessions=list(task.sessions), store=store, run_id=run_id,
                profile=profile, on_result=results.append, walk_deadline=upcoming,
            )
        except OSError as exc:
            raise RuntimeError(f"Tape backfill evidence failed symbol={task.symbol}: {exc}") from exc
        pages += int(outcome.pages_fetched)
        if outcome.unresolved_days:
            logger.info(
                "[DATA] stage=tape_backfill symbol=%s venue=%s status=GUARD_STOP remaining_days=%d",
                task.symbol, task.venue, len(outcome.unresolved_days),
            )
        if apply:
            assert publisher is not None
            for result in results:
                publisher.add(result)
                if result.entry.status == CaptureStatus.COMPLETE:
                    key = (result.day, result.session)
                    expected[key] = expected.get(key, 0) + len(result.frame)
                pending_ledger.append(
                    {"symbol": result.symbol, "day": result.day, "session": result.session, "status": result.entry.status.value, "run_id": run_id}
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
                logger.warning("[DATA] stage=tape_backfill status=VENDOR_FAILURE_STREAK streak=%d remaining=%d", failure_streak, len(remaining))
                break
        else:
            failure_streak = 0
        done = index + 1
        if done % 50 == 0:
            _heartbeat(done, len(tasks), pages, rows, unresolved, started)
    if apply:
        _commit()
    _heartbeat(len(tasks) - len(remaining), len(tasks), pages, rows, unresolved, started)
    return {
        "pages": pages, "rows": rows, "unresolved": unresolved, "remaining": remaining,
        "stopped": bool(remaining), "stopped_reason": stopped_reason,
    }


def main(argv: list[str] | None = None) -> None:
    """Recover missing/incomplete tick days from the Kiwoom tapes.

    Arguments: --start/--end (inclusive YYYY-MM-DD closed days), --venue {krx,nxt,all}, --apply, --deadline (aware ISO),
    --symbols-limit, --blackout HH:MM-HH:MM (repeatable), --ledger PATH, --force (re-recover certified days).

    Raises:
        ValueError: Open-ended or oversized selection (span > 31 days), unknown venue, naive deadline.
    """
    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    args = _parse_args(argv)
    start = _parse_day(args.start, "--start")
    end = _parse_day(args.end, "--end")
    if end < start:
        raise ValueError(f"Invalid backfill range: {args.start!r}..{args.end!r}")
    span = (end - start).days + 1
    if span > _MAX_BACKFILL_SPAN_DAYS:
        raise ValueError(f"Unbounded backfill selection: span={span} exceeds {_MAX_BACKFILL_SPAN_DAYS} days")
    venue_arg = str(args.venue).lower()
    venues: list[Literal["KRX", "NXT"]]
    if venue_arg == "all":
        venues = ["KRX", "NXT"]
    elif venue_arg == "krx":
        venues = ["KRX"]
    elif venue_arg == "nxt":
        venues = ["NXT"]
    else:
        raise ValueError(f"Invalid --venue: {args.venue!r} (expected one of krx, nxt, all)")
    deadline = _parse_deadline(args.deadline)
    blackouts = [_parse_blackout(spec) for spec in (args.blackout or [])] or [_parse_blackout(spec) for spec in _DEFAULT_BLACKOUTS]
    if args.symbols_limit is not None and int(args.symbols_limit) <= 0:
        raise ValueError(f"Invalid --symbols-limit: {args.symbols_limit!r}")
    profile = CollectionSettings()
    store = CaptureStore(_capture_root(profile))
    ledger = _ledger_path(args.ledger, profile)
    settled = {} if args.force else _read_settled(ledger)
    days = [(start + timedelta(days=offset)).isoformat() for offset in range(span)]
    needs = _collect_needs(days, venues, store, settled, bool(args.force))
    tasks = _order_tasks(needs)
    if args.symbols_limit is not None:
        kept: list[WalkTask] = []
        seen_symbols: set[str] = set()
        for task in tasks:
            if task.symbol in seen_symbols or len(seen_symbols) < int(args.symbols_limit):
                kept.append(task)
                seen_symbols.add(task.symbol)
        tasks = kept
    projected_pages, projected_bytes = _project_evidence(tasks, profile)
    if not args.apply:
        logger.info(
            "[DATA] stage=tape_backfill_dry_run needs=%d tasks=%d expected_pages=%d expected_bytes=%d",
            len(needs), len(tasks), projected_pages, projected_bytes,
        )
    try:
        free = _free_bytes(_capture_root(profile))
    except OSError as exc:
        raise RuntimeError(f"Tape backfill storage check failed: {exc}") from exc
    reserve = int(profile.COLLECTION_TAPE_MIN_FREE_GIB) * 1024**3
    if projected_bytes > max(0, free - reserve):
        raise RuntimeError(
            f"Tape backfill storage budget exceeded: projected_bytes={projected_bytes} free={free} reserve={reserve}"
        )
    if not tasks:
        logger.info("[DATA] stage=tape_backfill status=NOOP needs=0")
        return

    async def _run() -> dict[str, Any]:
        client, session_ctx = _open_kiwoom()
        async with session_ctx as http_session:
            return await _run_tasks(
                tasks, client=client, http_session=http_session, store=store, profile=profile,
                apply=bool(args.apply), ledger=ledger, deadline=deadline,
                blackouts=blackouts, run_date=_now().date().isoformat(),
            )

    try:
        summary = asyncio.run(_run())
    except OSError as exc:
        raise RuntimeError(f"Tape backfill infrastructure failed: {exc}") from exc
    logger.info(
        "[DATA] stage=tape_backfill_report needs=%d pages=%d rows=%d unresolved=%d remaining=%s",
        len(needs), summary["pages"], summary["rows"], summary["unresolved"], summary["remaining"],
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry, exercised via `python -m`
    main()
