"""Tombstoned eviction of offsite-verified history partitions.

An evicted partition keeps a sidecar tombstone so writers cannot recreate a partial file (the
nightly loose copy would otherwise push it over the full remote copy) and planners still see
which symbols were stored.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config.base import INTRADAY_KIS_ADJUSTED_1M_DIR_NAME
from src.config.collection import CollectionSettings
from src.tools.offsite_common import DATED_DIR_RE, md5_file, resolve_rclone_bin
from src.utils.file_lock import exclusive_file_lock, sidecar_lock_path

logger = logging.getLogger(__name__)

TOMBSTONE_SUFFIX: str = ".evicted.json"
RCLONE_TIMEOUT_SEC: int = 600
# rclone documented exit code 3: directory not found (no remote copy of that month exists).
RCLONE_EXIT_DIRECTORY_NOT_FOUND: int = 3
# Writer horizons (kis minute window, tape lookback) must stay clear of evicted dates by this margin.
RETENTION_MARGIN_DAYS: int = 7

REASON_REMOTE_MISSING = "remote_missing"
REASON_REMOTE_MISMATCH = "remote_mismatch"
REASON_REMOTE_UNVERIFIABLE = "remote_unverifiable"
REASON_TOMBSTONE_CONFLICT = "tombstone_conflict"
REASON_LOCAL_UNREADABLE = "local_unreadable"
REASON_CHANGED = "changed_during_eviction"
# Writers hold the partition sidecar lock for a whole publish; eviction never waits long behind one.
PARTITION_LOCK_TIMEOUT_SEC: float = 5.0

_MD5_LINE_RE = re.compile(r"^([0-9a-fA-F]{32})\s+(.+)$")
_TOMBSTONE_KEYS = ("partition", "remote_path", "md5", "size", "rows", "symbols", "evicted_at")

RunFn = Callable[..., "subprocess.CompletedProcess[str]"]


class EvictedPartitionError(RuntimeError):
    """A writer targeted a partition that was evicted to offsite storage."""


@dataclass(frozen=True)
class Tombstone:
    partition: str
    remote_path: str
    md5: str
    size: int
    rows: int
    symbols: tuple[str, ...]
    evicted_at: str


@dataclass(frozen=True)
class EvictionFamily:
    name: str
    relroots: tuple[str, ...]
    retention_days: int


@dataclass(frozen=True)
class EvictionReport:
    evicted: tuple[str, ...]
    kept: tuple[tuple[str, str], ...]
    bytes_freed: int
    stopped_by_deadline: bool
    capped: bool


def tombstone_path(partition: Path) -> Path:
    return partition.with_name(partition.name + TOMBSTONE_SUFFIX)


def read_tombstone(partition: Path) -> Tombstone | None:
    """Return the partition's tombstone, None when absent.

    Raises:
        ValueError: The tombstone exists but is malformed.
    """
    path = tombstone_path(partition)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or any(key not in data for key in _TOMBSTONE_KEYS):
            raise ValueError("missing fields")
        symbols = data["symbols"]
        if not isinstance(symbols, list) or any(not isinstance(item, str) for item in symbols):
            raise ValueError("symbols must be a list of strings")
        return Tombstone(
            partition=str(data["partition"]),
            remote_path=str(data["remote_path"]),
            md5=str(data["md5"]),
            size=int(data["size"]),
            rows=int(data["rows"]),
            symbols=tuple(symbols),
            evicted_at=str(data["evicted_at"]),
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Malformed partition tombstone: {path}") from exc


def require_writable(partition: Path) -> None:
    """Raise EvictedPartitionError when the partition has a tombstone."""
    if tombstone_path(partition).exists():
        raise EvictedPartitionError(
            f"Partition is evicted: {partition}; restore with python -m src.tools.partition_restore --path <relpath>"
        )


def is_evicted(partition: Path) -> bool:
    return tombstone_path(partition).exists()


def stored_symbols(partition: Path) -> set[str]:
    """Symbols stored in the partition, falling back to its tombstone when evicted.

    Raises:
        OSError: The parquet or the tombstone is unreadable.
    """
    if partition.exists():
        try:
            existing = pd.read_parquet(partition, columns=["symbol"])
        except Exception as exc:
            raise OSError(f"Cannot read existing partition evidence: {partition}") from exc
        if existing.empty or "symbol" not in existing.columns:
            return set()
        return {str(item) for item in existing["symbol"].astype(str).tolist()}
    try:
        tomb = read_tombstone(partition)
    except ValueError as exc:
        raise OSError(f"Cannot read partition tombstone: {partition}") from exc
    return set(tomb.symbols) if tomb is not None else set()


def eviction_families(profile: CollectionSettings) -> tuple[EvictionFamily, ...]:
    return (
        EvictionFamily(
            "bars", ("intraday/1m", f"intraday/{INTRADAY_KIS_ADJUSTED_1M_DIR_NAME}"), profile.HISTORY_EVICTION_BARS_RETENTION_DAYS
        ),
        EvictionFamily("ticks", ("intraday/ticks",), profile.HISTORY_EVICTION_TICKS_RETENTION_DAYS),
        EvictionFamily(
            "sameday", ("orderbook", "auction_path", "research"), profile.HISTORY_EVICTION_SAMEDAY_RETENTION_DAYS
        ),
    )


def validate_retention(profile: CollectionSettings) -> None:
    """Reject retentions that let a live writer horizon reach an evicted date.

    Raises:
        ValueError: A family retention does not exceed its writer horizon plus the margin.
    """
    floors = {
        "HISTORY_EVICTION_BARS_RETENTION_DAYS": (
            profile.HISTORY_EVICTION_BARS_RETENTION_DAYS,
            profile.COLLECTION_KIS_MINUTE_RETENTION_DAYS + RETENTION_MARGIN_DAYS,
        ),
        "HISTORY_EVICTION_TICKS_RETENTION_DAYS": (
            profile.HISTORY_EVICTION_TICKS_RETENTION_DAYS,
            profile.COLLECTION_TAPE_LOOKBACK_DAYS + RETENTION_MARGIN_DAYS,
        ),
        "HISTORY_EVICTION_SAMEDAY_RETENTION_DAYS": (
            profile.HISTORY_EVICTION_SAMEDAY_RETENTION_DAYS,
            profile.COLLECTION_TAPE_LOOKBACK_DAYS + RETENTION_MARGIN_DAYS,
        ),
    }
    for name, (value, floor) in floors.items():
        if value <= floor:
            raise ValueError(f"{name}={value} must exceed writer horizon + margin ({floor})")


@dataclass(frozen=True)
class _RemoteMonth:
    md5: dict[str, str]
    size: dict[str, int]


def _parse_md5sum(stdout: str) -> dict[str, str]:
    """Map remote name to MD5; a line without a hash (Drive could not compute one) maps to ''."""
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _MD5_LINE_RE.match(stripped)
        if match:
            out[match.group(2)] = match.group(1).lower()
        else:
            out[stripped] = ""
    return out


def _parse_lsf(stdout: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in stdout.splitlines():
        size, sep, name = line.strip().partition(";")
        if sep and size.isdigit():
            out[name] = int(size)
    return out


def _list_remote_month(remote_dir: str, run_fn: RunFn) -> _RemoteMonth | None:
    """Remote md5/size maps for one month directory; None when unverifiable."""
    rclone = resolve_rclone_bin()
    results = []
    for args in (["md5sum", remote_dir], ["lsf", "--format", "sp", remote_dir]):
        try:
            results.append(
                run_fn([rclone, *args], capture_output=True, text=True, timeout=RCLONE_TIMEOUT_SEC, check=False)
            )
        except (OSError, subprocess.SubprocessError):
            return None
    if any(r.returncode == RCLONE_EXIT_DIRECTORY_NOT_FOUND for r in results):
        return _RemoteMonth({}, {})
    if any(r.returncode != 0 for r in results):
        return None
    return _RemoteMonth(_parse_md5sum(results[0].stdout or ""), _parse_lsf(results[1].stdout or ""))


def _write_tombstone(partition: Path, tomb: Tombstone) -> None:
    final = tombstone_path(partition)
    tmp = final.with_name(final.name + ".tmp")
    payload = {key: (list(getattr(tomb, key)) if key == "symbols" else getattr(tomb, key)) for key in _TOMBSTONE_KEYS}
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, final)
    dir_fd = os.open(final.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _read_rows_and_symbols(path: Path) -> tuple[int, tuple[str, ...]]:
    handle = pq.ParquetFile(path)
    rows = int(handle.metadata.num_rows)
    if "symbol" not in handle.schema_arrow.names:
        return rows, ()
    column = pq.read_table(path, columns=["symbol"]).column("symbol").to_pylist()
    return rows, tuple(sorted({str(item) for item in column if item is not None}))


def _candidates(root: Path, cutoff: date) -> list[Path]:
    found: list[Path] = []
    for path in sorted(root.rglob("*.parquet")):
        if path.is_symlink() or not path.is_file() or not DATED_DIR_RE.match(path.stem):
            continue
        try:
            partition_date = date.fromisoformat(path.stem)
        except ValueError:
            continue
        if partition_date < cutoff:
            found.append(path)
    return found


def _evict_locked(path: Path, rel: str, tomb: Tombstone | None, md5: str, size: int, remote_base: str) -> None:
    """Tombstone then unlink one partition; the caller holds its sidecar lock."""
    if tomb is None:
        rows, symbols = _read_rows_and_symbols(path)
        _write_tombstone(
            path,
            Tombstone(
                partition=rel,
                remote_path=f"{remote_base}/{rel}",
                md5=md5,
                size=size,
                rows=rows,
                symbols=symbols,
                evicted_at=datetime.now(UTC).isoformat(),
            ),
        )
    path.unlink()


def evict_verified_partitions(
    *,
    history_dir: Path,
    today: date,
    families: Sequence[EvictionFamily],
    remote_base: str,
    run_fn: RunFn = subprocess.run,
    dry_run: bool = False,
    deadline: datetime | None = None,
    max_files: int,
) -> EvictionReport:
    """Evict expired partitions whose remote MD5 and size match, leaving tombstones.

    Per-month listing failures keep that month; the run never raises for them.
    """
    evicted: list[str] = []
    kept: list[tuple[str, str]] = []
    per_family: Counter[str] = Counter()
    bytes_freed = 0
    stopped = False
    capped = False
    months: dict[Path, _RemoteMonth | None] = {}

    def _keep(rel: str, reason: str) -> None:
        kept.append((rel, reason))
        logger.debug("[SYS] stage=history_eviction kept=%s reason=%s", rel, reason)

    done = False
    for family in families:
        cutoff = today - timedelta(days=family.retention_days)
        for relroot in family.relroots:
            root = history_dir / relroot
            if done or not root.is_dir():
                continue
            for path in _candidates(root, cutoff):
                if deadline is not None and datetime.now(deadline.tzinfo) >= deadline:
                    stopped = True
                    done = True
                    break
                if len(evicted) >= max_files:
                    capped = True
                    done = True
                    break
                rel = path.relative_to(history_dir).as_posix()
                try:
                    local_md5 = md5_file(path)
                    size = path.stat().st_size
                    tomb = read_tombstone(path)
                except (OSError, ValueError):
                    _keep(rel, REASON_TOMBSTONE_CONFLICT if tombstone_path(path).exists() else REASON_LOCAL_UNREADABLE)
                    continue
                if tomb is not None:
                    if tomb.md5 != local_md5:
                        _keep(rel, REASON_TOMBSTONE_CONFLICT)
                        continue
                else:
                    if path.parent not in months:
                        remote_dir = f"{remote_base}/{path.parent.relative_to(history_dir).as_posix()}"
                        months[path.parent] = _list_remote_month(remote_dir, run_fn)
                    month = months[path.parent]
                    if month is None:
                        _keep(rel, REASON_REMOTE_UNVERIFIABLE)
                        continue
                    if path.name not in month.md5 or path.name not in month.size:
                        _keep(rel, REASON_REMOTE_MISSING)
                        continue
                    if not month.md5[path.name]:
                        _keep(rel, REASON_REMOTE_UNVERIFIABLE)
                        continue
                    if month.md5[path.name] != local_md5 or month.size[path.name] != size:
                        _keep(rel, REASON_REMOTE_MISMATCH)
                        continue
                if not dry_run:
                    try:
                        with exclusive_file_lock(
                            sidecar_lock_path(path), timeout_seconds=PARTITION_LOCK_TIMEOUT_SEC, purpose="eviction"
                        ):
                            if md5_file(path) != local_md5 or path.stat().st_size != size:
                                _keep(rel, REASON_CHANGED)
                                continue
                            _evict_locked(path, rel, tomb, local_md5, size, remote_base)
                    except (OSError, TimeoutError):
                        logger.exception("[SYS] stage=history_eviction file=%s status=failed", rel)
                        _keep(rel, REASON_LOCAL_UNREADABLE)
                        continue
                evicted.append(rel)
                per_family[family.name] += 1
                bytes_freed += size
                logger.debug("[SYS] stage=history_eviction evicted=%s bytes=%d dry_run=%s", rel, size, dry_run)
    logger.info(
        "[SYS] stage=history_eviction dry_run=%s evicted=%d by_family=%s bytes=%d kept=%s stopped_by_deadline=%s capped=%s",
        dry_run, len(evicted), dict(per_family), bytes_freed, dict(Counter(reason for _, reason in kept)), stopped, capped,
    )
    return EvictionReport(tuple(evicted), tuple(kept), bytes_freed, stopped, capped)
