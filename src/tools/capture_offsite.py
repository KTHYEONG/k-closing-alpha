"""Capture offsite segment packing for Drive object-count bound.

Google Drive sustains only a few object creations per second, so the many
small capture files are sealed into append-only tar.zst segments per
(tier, trading_date). Late writes into past dates form new segments while
sealed segments are never rewritten.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa

from src.config.collection import CollectionSettings
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.io_utils import atomic_write_text
from src.tools.offsite_common import (
    DATED_DIR_RE as _DATE_RE,
)
from src.tools.offsite_common import (
    OFFSITE_REMOTE_BASE,
)
from src.tools.offsite_common import (
    resolve_rclone_bin as _resolve_rclone_bin,
)
from src.tools.offsite_common import (
    sha256_file as _sha256_file,
)
from src.utils.cli_logging import configure_cli_logging
from src.utils.file_lock import open_lock_descriptor

logger = logging.getLogger(__name__)

RUN_DIR_DEPTH: Mapping[str, int] = {"raw": 5, "normalized": 2, "manifests": 2}

_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class OffsiteConfig:
    """Typed contract for capture offsite packing.

    Attributes:
        remote_root: rclone path under which segments are stored as
            <remote_root>/<tier>/<YYYY-MM>/<YYYY-MM-DD>/<segment_name>.
        tiers: Capture tiers packed into segments; every other capture subtree
            stays on the loose-copy path.
        max_segment_member_bytes: Upper bound on the summed uncompressed member
            size of one segment, bounding staging disk, retry cost and restore
            granularity.
        recent_window_days: Trading dates within this many calendar days of
            today are always fully rescanned, because a run may still be
            appending into its existing run directory.
        full_scan_weekday: Weekday (Mon=0) on which every trading date is
            fully rescanned as a reconciliation safety net.
        rclone_timeout_sec: Per rclone subprocess timeout.
        seal_workers: Concurrent segment uploads during sealing; bounded to
            protect Drive API quota and host CPU/disk.
    """

    remote_root: str = OFFSITE_REMOTE_BASE + "/capture_sealed"
    tiers: tuple[str, ...] = ("raw", "normalized", "manifests")
    max_segment_member_bytes: int = 1_000_000_000
    recent_window_days: int = 3
    full_scan_weekday: int = 4
    rclone_timeout_sec: int = 3600
    seal_workers: int = 4

    def __post_init__(self) -> None:
        if not 1 <= int(self.seal_workers) <= 8:
            raise ValueError(f"seal_workers must be within 1..8: {self.seal_workers!r}")


@dataclass(frozen=True)
class SegmentMember:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class LedgerEntry:
    tier: str
    trading_date: str
    segment_name: str
    remote_path: str
    members: tuple[SegmentMember, ...]
    archive_bytes: int
    archive_md5: str
    committed_at: str


@dataclass(frozen=True)
class SealReport:
    dates_scanned: int
    segments_committed: int
    members_committed: int
    archive_bytes: int
    missing_sealed_members: int
    deferred_dates: int = 0
    oldest_deferred_date: str = ""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def segment_name(members: Sequence[SegmentMember]) -> str:
    """Derive a deterministic segment identity from its member set.

    The name depends only on the sorted (path, size) pairs, so re-sealing the
    same pending members after a crash targets the same remote object instead
    of creating a duplicate.

    Returns:
        "seg-<first 16 hex of sha256>" ; ".tar.zst" is appended by callers.

    Raises:
        ValueError: Empty member set.
    """
    items = list(members)
    if not items:
        raise ValueError("empty member set")
    digest = hashlib.sha256()
    for member in sorted(items, key=lambda entry: entry.path):
        digest.update(member.path.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(member.size).encode("utf-8"))
        digest.update(b"\n")
    return f"seg-{digest.hexdigest()[:16]}"


def _ledger_path(capture_root: Path, tier: str, trading_date: str) -> Path:
    return capture_root / "offsite" / "ledger" / tier / f"{trading_date}.jsonl"


def _entry_to_dict(entry: LedgerEntry) -> dict[str, object]:
    return {
        "tier": entry.tier,
        "trading_date": entry.trading_date,
        "segment_name": entry.segment_name,
        "remote_path": entry.remote_path,
        "members": [{"path": m.path, "size": m.size, "sha256": m.sha256} for m in entry.members],
        "archive_bytes": entry.archive_bytes,
        "archive_md5": entry.archive_md5,
        "committed_at": entry.committed_at,
    }


def _entry_from_dict(raw: object) -> LedgerEntry:
    if not isinstance(raw, dict):
        raise ValueError("malformed ledger line")
    try:
        tier = raw["tier"]
        trading_date = raw["trading_date"]
        segment_name_value = raw["segment_name"]
        remote_path = raw["remote_path"]
        members_raw = raw["members"]
        archive_bytes = raw["archive_bytes"]
        archive_md5 = raw["archive_md5"]
        committed_at = raw["committed_at"]
    except KeyError as exc:
        raise ValueError(f"malformed ledger line: missing {exc}") from None
    if not isinstance(tier, str) or not isinstance(trading_date, str):
        raise ValueError("malformed ledger line")
    if not isinstance(segment_name_value, str) or not isinstance(remote_path, str):
        raise ValueError("malformed ledger line")
    if not isinstance(members_raw, list) or not members_raw:
        raise ValueError("malformed ledger line")
    if not isinstance(archive_bytes, int) or not isinstance(archive_md5, str):
        raise ValueError("malformed ledger line")
    if not isinstance(committed_at, str):
        raise ValueError("malformed ledger line")
    members: list[SegmentMember] = []
    for item in members_raw:
        if not isinstance(item, dict):
            raise ValueError("malformed ledger member")
        path = item.get("path")
        size = item.get("size")
        sha256 = item.get("sha256")
        if not isinstance(path, str) or not isinstance(size, int) or not isinstance(sha256, str):
            raise ValueError("malformed ledger member")
        if not path or ".." in Path(path).parts or os.path.isabs(path):
            raise ValueError(f"malformed ledger member path: {path!r}")
        members.append(SegmentMember(path=path, size=size, sha256=sha256))
    return LedgerEntry(
        tier=tier,
        trading_date=trading_date,
        segment_name=segment_name_value,
        remote_path=remote_path,
        members=tuple(members),
        archive_bytes=archive_bytes,
        archive_md5=archive_md5,
        committed_at=committed_at,
    )


def read_ledger(capture_root: Path, tier: str, trading_date: str) -> list[LedgerEntry]:
    """Load committed segments for one (tier, trading_date).

    The ledger lives at <capture_root>/offsite/ledger/<tier>/<YYYY-MM-DD>.jsonl
    and is itself carried offsite by the loose-copy tier, so it doubles as the
    restore index.

    Returns:
        Entries in commit order; empty when no ledger file exists.

    Raises:
        ValueError: Malformed line or duplicate member path across entries.
    """
    path = _ledger_path(capture_root, tier, trading_date)
    if not path.exists():
        return []
    entries: list[LedgerEntry] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            raw: object = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"malformed ledger line: {exc}") from None
        entry = _entry_from_dict(raw)
        if entry.tier != tier or entry.trading_date != trading_date:
            raise ValueError("ledger entry tier/date mismatch")
        for member in entry.members:
            if member.path in seen:
                raise ValueError(f"duplicate member path across entries: {member.path!r}")
            seen.add(member.path)
        entries.append(entry)
    return entries


def _is_inflight(name: str) -> bool:
    return name.endswith(".lock") or (name.startswith("stage-") and name.endswith(".tmp"))


def _local_md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - content checksum, not security
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def plan_segments(
    capture_root: Path,
    tier: str,
    trading_date: str,
    sealed: Mapping[str, int],
    config: OffsiteConfig,
) -> list[tuple[SegmentMember, ...]]:
    """Partition the unsealed members of one date directory into segments.

    Args:
        capture_root: Capture store root.
        tier: One of config.tiers.
        trading_date: YYYY-MM-DD directory name under the tier.
        sealed: Already-committed member path -> recorded size.
        config: Packing contract.

    Returns:
        Ordered segments whose summed member size respects
        config.max_segment_member_bytes (a single oversized file forms its own
        segment); empty when nothing is pending.

    Raises:
        ValueError: A sealed member's current size differs from the ledger
            (immutability violation).
    """
    date_dir = capture_root / tier / trading_date
    if not date_dir.exists():
        return []
    pending: list[SegmentMember] = []
    for root, _dirs, files in os.walk(date_dir):
        for name in files:
            if _is_inflight(name):
                continue
            full = Path(root) / name
            if full.is_symlink() or not full.is_file():
                continue
            rel = full.relative_to(capture_root).as_posix()
            size = full.stat().st_size
            recorded = sealed.get(rel)
            if recorded is not None:
                if size != recorded:
                    raise ValueError(f"sealed member size changed: {rel!r}")
                continue
            pending.append(SegmentMember(path=rel, size=size, sha256=_sha256_file(full)))
    pending.sort(key=lambda entry: entry.path)
    segments: list[tuple[SegmentMember, ...]] = []
    current: list[SegmentMember] = []
    current_bytes = 0
    for member in pending:
        if not current:
            current = [member]
            current_bytes = member.size
            if member.size >= config.max_segment_member_bytes:
                segments.append(tuple(current))
                current = []
                current_bytes = 0
            continue
        if current_bytes + member.size > config.max_segment_member_bytes:
            segments.append(tuple(current))
            current = [member]
            current_bytes = member.size
            if member.size >= config.max_segment_member_bytes:
                segments.append(tuple(current))
                current = []
                current_bytes = 0
            continue
        current.append(member)
        current_bytes += member.size
    if current:
        segments.append(tuple(current))
    return segments


def _is_valid_date(name: str) -> bool:
    if not _DATE_RE.match(name):
        return False
    try:
        date.fromisoformat(name)
    except ValueError:
        return False
    return True


def _structural_max_mtime_ns(tier_root: Path, trading_date: str, depth: int) -> int:
    date_dir = tier_root / trading_date
    peak = date_dir.stat().st_mtime_ns
    stack: list[Path] = [date_dir]
    while stack:
        current = stack.pop()
        for entry in current.iterdir():
            if not entry.is_dir():
                continue
            child_depth = len(entry.relative_to(tier_root).parts)
            if child_depth >= depth:
                continue
            mtime_ns = entry.stat().st_mtime_ns
            if mtime_ns > peak:
                peak = mtime_ns
            stack.append(entry)
    return peak


def _load_scan_state(capture_root: Path) -> dict[str, int]:
    path = capture_root / "offsite" / "scan_state.json"
    if not path.exists():
        return {}
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return {str(key): int(value) for key, value in raw.items()}


def _save_scan_state(capture_root: Path, state: Mapping[str, int]) -> None:
    path = capture_root / "offsite" / "scan_state.json"
    atomic_write_text(path, json.dumps(dict(state), sort_keys=True), mode=None)


def _remote_path(config: OffsiteConfig, tier: str, trading_date: str, seg: str) -> str:
    return f"{config.remote_root}/{tier}/{trading_date[:7]}/{trading_date}/{seg}.tar.zst"


def _parse_remote_md5(stdout: str) -> str | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.split()[0].lower()
    return None


def _is_missing_remote_dir(stderr: str | None) -> bool:
    text = (stderr or "").lower()
    if not text.strip():
        return True
    markers = (
        "directory not found",
        "not found",
        "doesn't exist",
        "does not exist",
        "no such file or directory",
        "not exist",
        "couldn't find",
        "could not find",
    )
    return any(marker in text for marker in markers)


def _remote_existing_segments(
    rclone: str,
    config: OffsiteConfig,
    tier: str,
    trading_date: str,
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[str]],
) -> frozenset[str]:
    """Names of segment objects already present in the remote date directory, from a single directory listing. Replaces one md5 pre-check per segment: only names found here need a pre-upload MD5 comparison."""
    remote_dir = f"{config.remote_root}/{tier}/{trading_date[:7]}/{trading_date}"
    try:
        result = run_fn(
            [rclone, "lsf", remote_dir],
            capture_output=True,
            text=True,
            timeout=config.rclone_timeout_sec,
            check=False,
        )
    except subprocess.CalledProcessError as exc:
        if _is_missing_remote_dir(exc.stderr):
            return frozenset()
        raise
    if result.returncode != 0:
        if _is_missing_remote_dir(result.stderr):
            return frozenset()
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    names = {line.strip().split("/")[-1] for line in result.stdout.splitlines() if line.strip()}
    names.discard("")
    return frozenset(names)


def _seal_one_segment(
    *,
    capture_root: Path,
    tier: str,
    trading_date: str,
    seg_members: Sequence[SegmentMember],
    segment: str,
    staging_path: Path,
    rclone: str,
    config: OffsiteConfig,
    run_fn: Callable[..., subprocess.CompletedProcess[str]],
    already_remote: bool,
) -> tuple[int, str]:
    try:
        _build_archive(capture_root, seg_members, staging_path)
        _verify_archive_members(staging_path, seg_members)
        local_md5 = _local_md5(staging_path)
        local_bytes = staging_path.stat().st_size
        remote = _remote_path(config, tier, trading_date, segment)
        if already_remote:
            pre = run_fn(
                [rclone, "md5sum", remote],
                capture_output=True,
                text=True,
                timeout=config.rclone_timeout_sec,
                check=False,
            )
            if pre.returncode != 0:
                raise subprocess.CalledProcessError(pre.returncode, pre.args, pre.stdout, pre.stderr)
            if _parse_remote_md5(pre.stdout) != local_md5:
                raise ValueError(f"remote MD5 mismatch for {remote}")
            return local_bytes, local_md5
        uploaded = run_fn(
            [rclone, "copyto", str(staging_path), remote, "--immutable"],
            capture_output=True,
            text=True,
            timeout=config.rclone_timeout_sec,
            check=True,
        )
        if uploaded.returncode != 0:
            raise subprocess.CalledProcessError(
                uploaded.returncode, uploaded.args, uploaded.stdout, uploaded.stderr
            )
        post = run_fn(
            [rclone, "md5sum", remote],
            capture_output=True,
            text=True,
            timeout=config.rclone_timeout_sec,
            check=True,
        )
        if post.returncode != 0:
            raise subprocess.CalledProcessError(post.returncode, post.args, post.stdout, post.stderr)
        if _parse_remote_md5(post.stdout) != local_md5:
            raise ValueError(f"remote MD5 mismatch for {remote}")
        return local_bytes, local_md5
    finally:
        with contextlib.suppress(OSError):
            staging_path.unlink(missing_ok=True)


class _HashingReader:
    def __init__(self, path: Path) -> None:
        self._handle = open(path, "rb")  # noqa: SIM115 - handle lifetime managed by close()
        self._digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size)
        if data:
            self._digest.update(data)
        return data

    def close(self) -> None:
        self._handle.close()

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


def _build_archive(capture_root: Path, members: Sequence[SegmentMember], staging_path: Path) -> None:
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    if staging_path.exists():
        staging_path.unlink()
    with pa.OSFile(str(staging_path), "wb") as raw, pa.CompressedOutputStream(raw, "zstd") as compressed, tarfile.open(
        fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
    ) as tar:
        for member in members:
            full = capture_root / member.path
            current_size = full.stat().st_size
            if current_size != member.size:
                raise ValueError(f"member size changed during seal: {member.path!r}")
            info = tarfile.TarInfo(name=member.path)
            info.size = member.size
            info.mtime = 0
            info.mode = 0o644
            info.type = tarfile.REGTYPE
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            reader = _HashingReader(full)
            try:
                tar.addfile(info, reader)
                digest = reader.hexdigest()
            finally:
                reader.close()
            if digest != member.sha256:
                raise ValueError(f"member hash changed during seal: {member.path!r}")


def _verify_archive_members(staging_path: Path, members: Sequence[SegmentMember]) -> None:
    expected = sorted(member.path for member in members)
    with pa.OSFile(str(staging_path), "rb") as raw, pa.CompressedInputStream(raw, "zstd") as compressed, tarfile.open(
        fileobj=compressed, mode="r|"
    ) as tar:
        names = [info.name for info in tar]
    if sorted(names) != expected:
        raise ValueError("archive round-trip member mismatch")


@dataclass(frozen=True)
class _VerifiedSegment:
    index: int
    local_bytes: int
    local_md5: str


@dataclass(frozen=True)
class _GroupOutcome:
    tier: str
    trading_date: str
    scan_peak: int
    verified: tuple[_VerifiedSegment, ...]
    complete: bool
    failure: BaseException | None


def _seal_group(
    *,
    capture_root: Path,
    tier: str,
    trading_date: str,
    scan_peak: int,
    segments: Sequence[tuple[SegmentMember, ...]],
    staging_dir: Path,
    rclone: str,
    config: OffsiteConfig,
    run_fn: Callable[..., subprocess.CompletedProcess[str]],
    deadline: datetime | None,
    now_fn: Callable[[], datetime],
) -> _GroupOutcome:
    """Build, upload and remote-verify the planned segments of one (tier, date) group in plan order, on a worker thread.

    One remote directory listing per group replaces a per-segment MD5 pre-check; only segment names already present remotely are
    compared by MD5 before skipping their upload. The group is the unit of concurrency because the nightly load is mostly groups
    with a single small segment, whose cost is network latency rather than bytes. The function never touches the ledger or scan
    state: the coordinating thread commits `verified` after the fact so ledger writes stay single-threaded and ordered.

    Args: as typed above; `segments` are the group's planned member sets in plan order.

    Returns:
        `_GroupOutcome` whose `verified` is the contiguous prefix of segments whose remote MD5 equals the local MD5 (uploaded
        now or already present identically); `complete` is true only when every planned segment is verified; `failure` carries the
        first exception raised while handling the group (the contiguous verified prefix before it is still returned).

    The function does not raise for rclone, MD5 or archive failures; it reports them in `failure` so already-verified work is never lost.
    """
    try:
        existing = _remote_existing_segments(rclone, config, tier, trading_date, run_fn=run_fn)
    except BaseException as exc:
        return _GroupOutcome(
            tier=tier,
            trading_date=trading_date,
            scan_peak=scan_peak,
            verified=(),
            complete=False,
            failure=exc,
        )
    verified: list[_VerifiedSegment] = []
    for index, seg_members in enumerate(segments):
        if deadline is not None and now_fn() >= deadline:
            return _GroupOutcome(
                tier=tier,
                trading_date=trading_date,
                scan_peak=scan_peak,
                verified=tuple(verified),
                complete=False,
                failure=None,
            )
        seg = segment_name(tuple(seg_members))
        staging_path = staging_dir / f"{tier}-{trading_date}-{index:04d}-{seg}.tar.zst"
        try:
            local_bytes, local_md5 = _seal_one_segment(
                capture_root=capture_root,
                tier=tier,
                trading_date=trading_date,
                seg_members=seg_members,
                segment=seg,
                staging_path=staging_path,
                rclone=rclone,
                config=config,
                run_fn=run_fn,
                already_remote=f"{seg}.tar.zst" in existing,
            )
        except BaseException as exc:
            return _GroupOutcome(
                tier=tier,
                trading_date=trading_date,
                scan_peak=scan_peak,
                verified=tuple(verified),
                complete=False,
                failure=exc,
            )
        verified.append(_VerifiedSegment(index=index, local_bytes=local_bytes, local_md5=local_md5))
    return _GroupOutcome(
        tier=tier,
        trading_date=trading_date,
        scan_peak=scan_peak,
        verified=tuple(verified),
        complete=True,
        failure=None,
    )


def _append_ledger_entry(ledger_path: Path, entry: LedgerEntry) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_entry_to_dict(entry), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class SealLockHeldError(RuntimeError):
    """Another process holds the local seal lock (capture_root/offsite/.seal.lock)."""


class SealLockUnavailableError(SealLockHeldError):
    """The local seal lock file exists but this uid can open it neither read-write nor read-only."""


@contextlib.contextmanager
def _hold_seal_lock(capture_root: Path) -> Iterator[None]:
    """Hold the local seal lock for one seal or prune pass (single non-blocking attempt).

    The lock file is created 0644 under ``capture_root/offsite/`` and never unlinked. Any flock failure
    is treated as "held" and fails fast, because both holders are long batch jobs that must not queue.
    A lock file owned by another uid is opened read-only (flock does not need write access).

    Raises:
        SealLockHeldError: The lock is held elsewhere; message ``"another seal run holds the local seal
            lock"``, chained from the flock ``OSError``.
        SealLockUnavailableError: The lock file cannot be opened at all (foreign uid, mode without read
            permission); mutual exclusion cannot be established, so the pass must not run.
    """
    lock_path = Path(capture_root) / "offsite" / ".seal.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = open_lock_descriptor(lock_path)
    except PermissionError as exc:
        raise SealLockUnavailableError(f"local seal lock is not accessible to this uid: {lock_path.name}") from exc
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SealLockHeldError("another seal run holds the local seal lock") from exc
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def seal_and_upload(
    capture_root: Path,
    *,
    today: date,
    full_scan: bool,
    deadline: datetime | None = None,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now_fn: Callable[[], datetime] = _utcnow,
    config: OffsiteConfig = OffsiteConfig(),  # noqa: B008
) -> SealReport:
    """Seal pending capture members into verified offsite segments, newest dates first, until the deadline.

    The shared Drive lock is held by this run, so it must end in bounded time; work not started before `deadline`
    is deferred to the next run. Deferred dates keep their previous scan watermark, so the next run re-selects them.
    Dates are processed newest-first (the recent window before older backfill-touched dates) so a deferral never
    delays today's evidence.

    Segments are uploaded before they are recorded, and recorded only after
    the remote MD5 equals the local archive MD5, so the ledger never claims an
    object that is absent or corrupt offsite.

    Raises:
        ValueError: deadline is naive, unexpected layout, or an immutability violation.
        RuntimeError: Another seal run holds the local seal lock.
        subprocess.CalledProcessError: rclone upload/hash failure.
        OSError: Staging or ledger write failure.
    """
    if deadline is not None and (deadline.tzinfo is None or deadline.utcoffset() is None):
        raise ValueError(f"deadline must be timezone-aware: {deadline!r}")
    capture_root = Path(capture_root)
    with _hold_seal_lock(capture_root):
        return _seal_locked(capture_root, today=today, full_scan=full_scan, deadline=deadline, run_fn=run_fn, now_fn=now_fn, config=config)


def _seal_locked(
    capture_root: Path,
    *,
    today: date,
    full_scan: bool,
    deadline: datetime | None,
    run_fn: Callable[..., subprocess.CompletedProcess[str]],
    now_fn: Callable[[], datetime],
    config: OffsiteConfig,
) -> SealReport:
    rclone = _resolve_rclone_bin()
    state = _load_scan_state(capture_root)
    staging_dir = capture_root / "staging" / "offsite"
    staging_dir.mkdir(parents=True, exist_ok=True)
    dates_scanned = 0
    segments_committed = 0
    members_committed = 0
    archive_bytes_total = 0
    missing_sealed_members = 0
    deferred_dates = 0
    deferred_names: list[str] = []
    selected: list[tuple[str, str, int]] = []
    for tier in config.tiers:
        tier_root = capture_root / tier
        if not tier_root.exists():
            continue
        if not tier_root.is_dir() or tier_root.is_symlink():
            raise ValueError(f"unexpected tier layout: {tier!r}")
        date_names: list[str] = []
        for child in sorted(tier_root.iterdir(), key=lambda entry: entry.name):
            if not _is_valid_date(child.name):
                raise ValueError(f"non-date directory under tier {tier!r}: {child.name!r}")
            if child.is_symlink() or not child.is_dir():
                raise ValueError(f"non-date directory under tier {tier!r}: {child.name!r}")
            date_names.append(child.name)
        depth = RUN_DIR_DEPTH[tier]
        for date_name in date_names:
            parsed = date.fromisoformat(date_name)
            if full_scan or (today - parsed).days <= config.recent_window_days:
                is_selected = True
            else:
                key = f"{tier}/{date_name}"
                stored = state.get(key)
                if stored is None:
                    is_selected = True
                else:
                    current_peak = _structural_max_mtime_ns(tier_root, date_name, depth)
                    is_selected = current_peak > stored
            if not is_selected:
                continue
            dates_scanned += 1
            # 계획 전에 워터마크를 잡아야 스캔 도중 생긴 run 디렉터리가 다음 실행에서 재탐지된다
            scan_peak = _structural_max_mtime_ns(tier_root, date_name, depth)
            selected.append((tier, date_name, scan_peak))
    # 최신 날짜부터 처리해야 연기 시 오늘 증거가 밀리지 않는다 (동일 날짜는 config.tiers 순서)
    tier_order = {name: pos for pos, name in enumerate(config.tiers)}
    selected.sort(key=lambda item: (-date.fromisoformat(item[1]).toordinal(), tier_order.get(item[0], 0)))
    jobs: list[tuple[str, str, int, list[tuple[SegmentMember, ...]]]] = []
    for tier, date_name, scan_peak in selected:
        entries = read_ledger(capture_root, tier, date_name)
        sealed: dict[str, int] = {}
        for ledger_entry in entries:
            for member in ledger_entry.members:
                sealed[member.path] = member.size
        for sealed_path in sealed:
            full = capture_root / sealed_path
            if full.is_symlink() or not full.is_file():
                missing_sealed_members += 1
        segments = plan_segments(capture_root, tier, date_name, sealed, config)
        if not segments:
            state[f"{tier}/{date_name}"] = scan_peak
            continue
        jobs.append((tier, date_name, scan_peak, segments))
    if jobs:
        workers = int(config.seal_workers)
        futures: list[Future[_GroupOutcome] | None] = [None] * len(jobs)
        pending: set[Future[_GroupOutcome]] = set()
        index_of: dict[Future[_GroupOutcome], int] = {}
        outcomes: dict[int, _GroupOutcome] = {}
        first_failure: BaseException | None = None
        next_submit = 0
        next_commit = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="offsite-seal") as pool:
            def _fill() -> None:
                nonlocal next_submit
                while next_submit < len(jobs) and len(pending) < workers and first_failure is None:
                    if deadline is not None and now_fn() >= deadline:
                        break
                    tier_s, date_s, peak_s, segs_s = jobs[next_submit]
                    future = pool.submit(
                        _seal_group,
                        capture_root=capture_root,
                        tier=tier_s,
                        trading_date=date_s,
                        scan_peak=peak_s,
                        segments=segs_s,
                        staging_dir=staging_dir,
                        rclone=rclone,
                        config=config,
                        run_fn=run_fn,
                        deadline=deadline,
                        now_fn=now_fn,
                    )
                    futures[next_submit] = future
                    pending.add(future)
                    index_of[future] = next_submit
                    next_submit += 1

            def _collect(future: Future[_GroupOutcome], job_index: int) -> None:
                nonlocal first_failure
                outcomes[job_index] = future.result()
                if outcomes[job_index].failure is not None and first_failure is None:
                    first_failure = outcomes[job_index].failure

            _fill()
            while next_commit < len(jobs):
                if futures[next_commit] is None:
                    for rest in range(next_commit, len(jobs)):
                        if futures[rest] is None:
                            deferred_dates += 1
                            deferred_names.append(jobs[rest][1])
                    break
                if next_commit not in outcomes:
                    done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                    for finished in done:
                        pending.discard(finished)
                        _collect(finished, index_of[finished])
                    _fill()
                    if next_commit not in outcomes:
                        continue
                outcome = outcomes.pop(next_commit)
                tier_c, date_c, peak_c, segs_c = jobs[next_commit]
                for verified_seg in outcome.verified:
                    seg_members = segs_c[verified_seg.index]
                    seg = segment_name(seg_members)
                    committed_at = now_fn().isoformat()
                    ledger_entry = LedgerEntry(
                        tier=tier_c,
                        trading_date=date_c,
                        segment_name=seg,
                        remote_path=_remote_path(config, tier_c, date_c, seg),
                        members=tuple(seg_members),
                        archive_bytes=verified_seg.local_bytes,
                        archive_md5=verified_seg.local_md5,
                        committed_at=committed_at,
                    )
                    _append_ledger_entry(_ledger_path(capture_root, tier_c, date_c), ledger_entry)
                    segments_committed += 1
                    members_committed += len(seg_members)
                    archive_bytes_total += verified_seg.local_bytes
                    logger.info(
                        "[SYS] stage=capture_offsite tier=%s date=%s segment=%s.tar.zst members=%d bytes=%d",
                        tier_c,
                        date_c,
                        seg,
                        len(seg_members),
                        verified_seg.local_bytes,
                    )
                if outcome.complete and outcome.failure is None:
                    state[f"{tier_c}/{date_c}"] = peak_c
                else:
                    deferred_dates += 1
                    deferred_names.append(date_c)
                next_commit += 1
                _fill()
        if first_failure is not None:
            raise first_failure
    _save_scan_state(capture_root, state)
    logger.info(
        "[SYS] stage=capture_offsite dates=%d segments=%d members=%d bytes=%d missing=%d deferred=%d",
        dates_scanned,
        segments_committed,
        members_committed,
        archive_bytes_total,
        missing_sealed_members,
        deferred_dates,
    )
    return SealReport(
        dates_scanned=dates_scanned,
        segments_committed=segments_committed,
        members_committed=members_committed,
        archive_bytes=archive_bytes_total,
        missing_sealed_members=missing_sealed_members,
        deferred_dates=deferred_dates,
        oldest_deferred_date=min(deferred_names) if deferred_names else "",
    )


LOCAL_SEALED_RETENTION_DAYS: int = 30


def _validate_manifest_retention(retention_days: int) -> None:
    """Fail fast when local manifest retention cannot cover the tape reader horizon.

    Manifest readers (`read_manifests` tape window) reach back
    ``COLLECTION_TAPE_LOOKBACK_DAYS``; local sealed manifests must outlive that
    window or a widened tape lookback would read dates already pruned locally.

    Raises:
        ValueError: retention_days <= tape lookback + 1.
    """
    lookback = int(CollectionSettings().COLLECTION_TAPE_LOOKBACK_DAYS)
    if not retention_days > lookback + 1:
        raise ValueError(
            f"LOCAL_SEALED_RETENTION_DAYS={retention_days} must exceed "
            f"COLLECTION_TAPE_LOOKBACK_DAYS={lookback} + 1"
        )



@dataclass(frozen=True)
class LocalRetentionReport:
    """Outcome of one local sealed-capture prune pass.

    Attributes:
        removed: Date directories removed (or that would be, in dry-run).
        kept: Expired date directories kept, with the reason.
        bytes_removed: Bytes reclaimed (or reclaimable, in dry-run).
        skipped_reason: None for a completed pass; otherwise why the pass did not run (removed/kept are
            then empty): ``"seal_lock_held"`` (a seal holds the lock), ``"seal_lock_unavailable"`` (the lock
            file is not openable by this uid) or ``"capture_root_missing"`` (nothing to prune).
        stopped_by_deadline: True when the pass stopped early at its time budget; the remaining expired
            directories are picked up by the next run.
    """

    removed: tuple[str, ...]
    kept: tuple[tuple[str, str], ...]
    bytes_removed: int
    skipped_reason: str | None = None
    stopped_by_deadline: bool = False


def prune_local_sealed_capture(
    capture_root: Path,
    *,
    today: date,
    retention_days: int = LOCAL_SEALED_RETENTION_DAYS,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    config: OffsiteConfig = OffsiteConfig(),  # noqa: B008
    dry_run: bool = False,
    deadline: datetime | None = None,
) -> LocalRetentionReport:
    """Remove local capture date directories whose every file is sealed offsite and verified.

    Raw, normalized, and manifests capture tiers are append-only evidence whose durable
    copy is the sealed tar.zst segment set (ledger + remote MD5). Past the retention
    window, the local copy only costs disk and seal-scan time; production readers need
    at most the previous trading day (manifest readers need the tape lookback window,
    which retention is validated to exceed). A directory is removed whole or not at
    all, because the sealer counts ledger members missing from a directory it still
    scans.

    The whole scan-verify-delete pass holds the local seal lock, the same lock seal_and_upload holds.
    A seal scanning a directory that disappears mid-walk fails or miscounts missing sealed members.
    systemd serializes the scheduled units through the shared Drive lock, but a manual prune does not.

    Args:
        capture_root: Capture store root (holds the tiers and offsite/ledger).
        today: KST calendar date of the run.
        retention_days: Date directories strictly older than today - retention_days are candidates.
        run_fn: Subprocess runner for ``rclone md5sum`` (test injection).
        config: Offsite contract (tiers, remote_root, rclone timeout).
        dry_run: List targets without deleting (still holds the seal lock).
        deadline: Aware instant after which no further directory is started. The pass holds the shared Drive
            lock for its whole duration, so a large backlog (one remote MD5 per segment) would otherwise starve
            every other Drive writer on the host; directories not reached are handled by the next run.

    Returns:
        Removed directories, expired directories kept with a reason, and bytes reclaimed; or an empty
        report with ``skipped_reason`` set when the pass could not run (see LocalRetentionReport).

    Raises:
        OSError: Deleting an eligible directory failed; partial removal is never silenced.
        ValueError: Retention window shorter than the append window or the tape
            reader horizon (COLLECTION_TAPE_LOOKBACK_DAYS + 1).
    """
    if retention_days < config.recent_window_days + 1:
        raise ValueError(
            f"retention_days={retention_days} shorter than append window "
            f"(recent_window_days={config.recent_window_days})"
        )
    _validate_manifest_retention(retention_days)
    if not Path(capture_root).is_dir():
        # Nothing to prune; taking the lock would create offsite/ under a missing (unmounted) root.
        logger.warning("[SYS] stage=capture_prune status=SKIPPED reason=capture_root_missing")
        return LocalRetentionReport(removed=(), kept=(), bytes_removed=0, skipped_reason="capture_root_missing")
    try:
        with _hold_seal_lock(capture_root):
            capture_root = Path(capture_root)
            cutoff = today - timedelta(days=retention_days)
            rclone: str | None = None
            removed: list[str] = []
            kept: list[tuple[str, str]] = []
            bytes_removed = 0
            stopped = False
            for tier in sorted(config.tiers):
                if stopped:
                    break
                tier_root = capture_root / tier
                if not tier_root.exists():
                    continue
                if tier_root.is_symlink() or not tier_root.is_dir():
                    continue
                for child in sorted(tier_root.iterdir(), key=lambda entry: entry.name):
                    if deadline is not None and datetime.now(deadline.tzinfo) >= deadline:
                        stopped = True
                        break
                    name = child.name
                    if not _is_valid_date(name):
                        continue
                    parsed = date.fromisoformat(name)
                    if parsed >= cutoff:
                        continue
                    label = f"{tier}/{name}"
                    if child.is_symlink():
                        kept.append((label, "symlink"))
                        continue
                    if not child.is_dir():
                        continue
                    try:
                        entries = read_ledger(capture_root, tier, name)
                    except ValueError:
                        kept.append((label, "ledger_invalid"))
                        continue
                    if not entries:
                        kept.append((label, "no_ledger"))
                        continue
                    sealed_sizes: dict[str, int] = {}
                    for ledger_entry in entries:
                        for member in ledger_entry.members:
                            sealed_sizes[member.path] = member.size
                    found_inflight = False
                    found_unsealed = False
                    found_size_mismatch = False
                    found_symlink = False
                    regular_sizes = 0
                    for root, dirs, files in os.walk(child):
                        for dirname in dirs:
                            if (Path(root) / dirname).is_symlink():
                                found_symlink = True
                        for filename in files:
                            if _is_inflight(filename):
                                found_inflight = True
                            full = Path(root) / filename
                            if full.is_symlink():
                                found_symlink = True
                                continue
                            if not full.is_file():
                                continue
                            rel = full.relative_to(capture_root).as_posix()
                            recorded = sealed_sizes.get(rel)
                            if recorded is None:
                                found_unsealed = True
                            elif full.stat().st_size != recorded:
                                found_size_mismatch = True
                            regular_sizes += full.stat().st_size
                    if found_inflight:
                        kept.append((label, "inflight"))
                        continue
                    if found_unsealed:
                        kept.append((label, "unsealed_file"))
                        continue
                    if found_size_mismatch:
                        kept.append((label, "size_mismatch"))
                        continue
                    if found_symlink:
                        kept.append((label, "symlink_member"))
                        continue
                    if rclone is None:
                        rclone = _resolve_rclone_bin()
                    verified = True
                    for ledger_entry in entries:
                        try:
                            result = run_fn(
                                [rclone, "md5sum", ledger_entry.remote_path],
                                capture_output=True,
                                text=True,
                                timeout=config.rclone_timeout_sec,
                                check=False,
                            )
                        except Exception:  # noqa: BLE001 - any runner failure keeps the directory
                            verified = False
                            break
                        if result.returncode != 0:
                            verified = False
                            break
                        if _parse_remote_md5(result.stdout) != ledger_entry.archive_md5:
                            verified = False
                            break
                    if not verified:
                        kept.append((label, "remote_unverified"))
                        continue
                    if dry_run:
                        removed.append(label)
                        bytes_removed += regular_sizes
                        continue
                    shutil.rmtree(child)
                    removed.append(label)
                    bytes_removed += regular_sizes
            removed_sorted = tuple(sorted(removed))
            kept_sorted = tuple(sorted(kept))
            if stopped:
                logger.info("[SYS] stage=capture_prune status=DEADLINE removed=%d", len(removed_sorted))
            return LocalRetentionReport(
                removed=removed_sorted, kept=kept_sorted, bytes_removed=bytes_removed, stopped_by_deadline=stopped
            )
    except SealLockUnavailableError:
        logger.error("[SYS] stage=capture_prune status=SKIPPED reason=seal_lock_unavailable")
        return LocalRetentionReport(removed=(), kept=(), bytes_removed=0, skipped_reason="seal_lock_unavailable")
    except SealLockHeldError:
        logger.warning("[SYS] stage=capture_prune status=SKIPPED reason=seal_lock_held")
        return LocalRetentionReport(removed=(), kept=(), bytes_removed=0, skipped_reason="seal_lock_held")


def restore_date(
    capture_root: Path,
    tier: str,
    trading_date: str,
    dest_root: Path,
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    config: OffsiteConfig = OffsiteConfig(),  # noqa: B008
) -> list[str]:
    """Restore every committed segment of one date into dest_root.

    Used for disaster recovery and the periodic restore drill. Ledger
    (capture_root) and destination may differ.

    Returns:
        Sorted relative paths written or verified-identical under dest_root.

    Raises:
        ValueError: Archive MD5 or member sha256 mismatch, unsafe member path,
            or an existing destination file with different content.
        subprocess.CalledProcessError: rclone download failure.
    """
    capture_root = Path(capture_root)
    dest_root = Path(dest_root)
    entries = read_ledger(capture_root, tier, trading_date)
    if not entries:
        return []
    rclone = _resolve_rclone_bin()
    staging_dir = capture_root / "staging" / "offsite"
    staging_dir.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    for ledger_entry in entries:
        staging_dl = staging_dir / f"restore-{ledger_entry.segment_name}.tar.zst.tmp"
        member_tmp = staging_dir / f"restore-{ledger_entry.segment_name}.member.tmp"
        try:
            downloaded = run_fn(
                [rclone, "copyto", ledger_entry.remote_path, str(staging_dl)],
                capture_output=True,
                text=True,
                timeout=config.rclone_timeout_sec,
                check=True,
            )
            if downloaded.returncode != 0:
                raise subprocess.CalledProcessError(
                    downloaded.returncode, downloaded.args, downloaded.stdout, downloaded.stderr
                )
            if _local_md5(staging_dl) != ledger_entry.archive_md5:
                raise ValueError(f"archive MD5 mismatch: {ledger_entry.segment_name}")
            expected = {member.path: member for member in ledger_entry.members}
            seen: set[str] = set()
            with pa.OSFile(str(staging_dl), "rb") as raw, pa.CompressedInputStream(raw, "zstd") as compressed, tarfile.open(
                fileobj=compressed, mode="r|"
            ) as tar:
                for info in tar:
                    name = info.name
                    if not name or os.path.isabs(name) or ".." in Path(name).parts:
                        raise ValueError(f"unsafe member path: {name!r}")
                    member = expected.get(name)
                    if member is None:
                        raise ValueError(f"member not listed in ledger: {name!r}")
                    if info.issym() or info.islnk() or not info.isfile():
                        raise ValueError(f"unsafe member type: {name!r}")
                    if info.size != member.size:
                        raise ValueError(f"member size mismatch: {name!r}")
                    reader = tar.extractfile(info)
                    assert reader is not None
                    digest = hashlib.sha256()
                    with open(member_tmp, "wb") as out:
                        while True:
                            chunk = reader.read(_CHUNK)
                            if not chunk:
                                break
                            digest.update(chunk)
                            out.write(chunk)
                    if digest.hexdigest() != member.sha256:
                        with contextlib.suppress(OSError):
                            member_tmp.unlink(missing_ok=True)
                        raise ValueError(f"member sha256 mismatch: {name!r}")
                    seen.add(name)
                    dest_path = dest_root / name
                    if dest_path.is_symlink():
                        raise ValueError(f"destination is a link: {name!r}")
                    if dest_path.exists():
                        if not dest_path.is_file():
                            raise ValueError(f"destination is not a file: {name!r}")
                        if _sha256_file(dest_path) == member.sha256:
                            restored.append(name)
                            with contextlib.suppress(OSError):
                                member_tmp.unlink(missing_ok=True)
                            continue
                        with contextlib.suppress(OSError):
                            member_tmp.unlink(missing_ok=True)
                        raise ValueError(f"conflicting existing file: {name!r}")
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(member_tmp, dest_path)
                    restored.append(name)
            if seen != set(expected):
                raise ValueError(f"archive missing ledger members: {ledger_entry.segment_name}")
        finally:
            for tmp_path in (staging_dl, member_tmp):
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)
    return sorted(restored)


@dataclass(frozen=True)
class RemoteVerifyReport:
    """Remote integrity of every ledgered segment.

    Attributes:
        checked: Number of ledger entries compared.
        missing: Remote paths listed in a ledger but absent on the remote.
        mismatched: Remote paths whose MD5 differs from the ledger archive_md5.
    """

    checked: int
    missing: tuple[str, ...]
    mismatched: tuple[str, ...]


def verify_remote_segments(
    capture_root: Path,
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    config: OffsiteConfig = OffsiteConfig(),  # noqa: B008
) -> RemoteVerifyReport:
    """Compare every local ledger entry with the remote MD5 listing.

    The ledger (capture_root/offsite/ledger) outlives local date directories, so it
    is the authority for what must exist remotely. One `rclone md5sum` call per
    (tier, YYYY-MM) directory keeps the cost proportional to months, not segments.

    Raises:
        RuntimeError: An rclone listing call failed (unverifiable is not healthy).
    """
    capture_root = Path(capture_root)
    rclone = _resolve_rclone_bin()
    grouped: dict[tuple[str, str], list[LedgerEntry]] = {}
    for tier in config.tiers:
        ledger_dir = capture_root / "offsite" / "ledger" / tier
        if not ledger_dir.is_dir() or ledger_dir.is_symlink():
            continue
        for ledger_path in sorted(ledger_dir.glob("*.jsonl")):
            if not _is_valid_date(ledger_path.stem):
                continue
            for entry in read_ledger(capture_root, tier, ledger_path.stem):
                grouped.setdefault((tier, ledger_path.stem[:7]), []).append(entry)
    checked = 0
    missing: list[str] = []
    mismatched: list[str] = []
    for tier, month in sorted(grouped):
        remote_dir = f"{config.remote_root}/{tier}/{month}"
        result = run_fn(
            [rclone, "md5sum", remote_dir],
            capture_output=True,
            text=True,
            timeout=config.rclone_timeout_sec,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"offsite verify listing failed: {remote_dir}")
        listed: dict[str, str] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            listed[parts[1].lower()] = parts[0].lower()
        for entry in grouped[(tier, month)]:
            checked += 1
            actual = listed.get(f"{entry.trading_date}/{entry.segment_name}.tar.zst".lower())
            if actual is None:
                missing.append(entry.remote_path)
            elif actual != entry.archive_md5.lower():
                mismatched.append(entry.remote_path)
    return RemoteVerifyReport(checked=checked, missing=tuple(sorted(missing)), mismatched=tuple(sorted(mismatched)))


@dataclass(frozen=True)
class RestoreDrillReport:
    """Result of one end-to-end restore rehearsal.

    Attributes:
        tier: Capture tier drilled.
        trading_date: Date restored.
        members_verified: Member files restored and sha256-verified.
    """

    tier: str
    trading_date: str
    members_verified: int


def select_drill_dates(
    capture_root: Path, *, today: date, config: OffsiteConfig = OffsiteConfig()  # noqa: B008
) -> dict[str, str]:
    """Pick one ledgered date per tier to rehearse, rotating deterministically.

    Candidates are ledger dates older than today - config.recent_window_days
    (still-appending dates are excluded). The pick is candidates[iso_week(today) % len],
    so successive weeks walk the whole history without persisted state.

    Returns:
        {tier: trading_date}; tiers without candidates are omitted.
    """
    capture_root = Path(capture_root)
    cutoff = today - timedelta(days=config.recent_window_days)
    picks: dict[str, str] = {}
    for tier in config.tiers:
        ledger_dir = capture_root / "offsite" / "ledger" / tier
        if not ledger_dir.is_dir() or ledger_dir.is_symlink():
            continue
        candidates = sorted(
            ledger_path.stem
            for ledger_path in ledger_dir.glob("*.jsonl")
            if _is_valid_date(ledger_path.stem) and date.fromisoformat(ledger_path.stem) < cutoff
        )
        if not candidates:
            continue
        picks[tier] = candidates[today.isocalendar().week % len(candidates)]
    return picks


def run_restore_drill(
    capture_root: Path,
    *,
    today: date,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    config: OffsiteConfig = OffsiteConfig(),  # noqa: B008
) -> list[RestoreDrillReport]:
    """Restore the selected dates into a throwaway directory and discard it.

    Reuses restore_date (which enforces archive MD5 and member sha256) with
    dest_root under capture_root/staging/offsite/drill-<uuid>, removed in all
    outcomes. Never writes into the live capture tiers.

    Raises:
        ValueError, subprocess.CalledProcessError: propagated from restore_date.
    """
    capture_root = Path(capture_root)
    picks = select_drill_dates(capture_root, today=today, config=config)
    reports: list[RestoreDrillReport] = []
    for tier in sorted(picks):
        trading_date = picks[tier]
        drill_dir = capture_root / "staging" / "offsite" / f"drill-{uuid.uuid4().hex}"
        try:
            restored = restore_date(capture_root, tier, trading_date, drill_dir, run_fn=run_fn, config=config)
        except Exception:  # noqa: BLE001 - drill failure is reported then re-raised to the unit
            logger.info("[SYS] stage=restore_drill tier=%s date=%s members=0 status=FAILED", tier, trading_date)
            raise
        else:
            reports.append(RestoreDrillReport(tier=tier, trading_date=trading_date, members_verified=len(restored)))
            logger.info(
                "[SYS] stage=restore_drill tier=%s date=%s members=%d status=OK", tier, trading_date, len(restored)
            )
        finally:
            shutil.rmtree(drill_dir, ignore_errors=True)
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    """CLI.

    `verify` : verify_remote_segments + run_restore_drill; exit 1 on any missing,
               mismatched, or drill failure (after logging every finding).
    `restore --tier {raw,normalized,manifests} --date YYYY-MM-DD --dest PATH` : disaster
               recovery into PATH (must not be inside the live capture tiers).
    """
    import argparse
    from datetime import datetime as _datetime
    from zoneinfo import ZoneInfo as _ZoneInfo

    parser = argparse.ArgumentParser(description="Offsite verification and disaster recovery")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="re-verify remote MD5s and rehearse a rotating restore")
    restore_cmd = sub.add_parser("restore", help="restore one tier date into PATH")
    restore_cmd.add_argument("--tier", required=True, choices=list(OffsiteConfig().tiers))
    restore_cmd.add_argument("--date", required=True)
    restore_cmd.add_argument("--dest", required=True)
    args = parser.parse_args(argv)
    capture_root = _capture_root()
    if args.command == "restore":
        dest = Path(args.dest)
        resolved = dest.resolve()
        for tier in OffsiteConfig().tiers:
            tier_root = (capture_root / tier).resolve()
            if resolved == tier_root or tier_root in resolved.parents:
                raise ValueError(f"restore dest must not be inside live tier: {dest}")
        restored = restore_date(capture_root, args.tier, args.date, dest)
        logger.info("[SYS] stage=restore tier=%s date=%s members=%d dest=%s", args.tier, args.date, len(restored), dest)
        return 0
    report = verify_remote_segments(capture_root)
    logger.info(
        "[SYS] stage=offsite_verify checked=%d missing=%d mismatched=%d",
        report.checked,
        len(report.missing),
        len(report.mismatched),
    )
    for remote_path in sorted(report.missing):
        logger.info("[SYS] stage=offsite_verify finding=missing remote_path=%s", remote_path)
    for remote_path in sorted(report.mismatched):
        logger.info("[SYS] stage=offsite_verify finding=mismatched remote_path=%s", remote_path)
    today = _datetime.now(_ZoneInfo("Asia/Seoul")).date()
    try:
        run_restore_drill(capture_root, today=today)
    except Exception:  # noqa: BLE001 - drill failure already logged per tier; unit still exits 1
        return 1
    return 1 if (report.missing or report.mismatched) else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    raise SystemExit(main())
