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
import re
import shutil
import subprocess
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa

from src.tools.backup_prune import _resolve_rclone_bin

logger = logging.getLogger(__name__)

RUN_DIR_DEPTH: Mapping[str, int] = {"raw": 5, "normalized": 2}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
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
    """

    remote_root: str = "gdrive:quant-lake/live/k-closing-alpha/capture_sealed"
    tiers: tuple[str, ...] = ("raw", "normalized")
    max_segment_member_bytes: int = 1_000_000_000
    recent_window_days: int = 3
    full_scan_weekday: int = 4
    rclone_timeout_sec: int = 3600


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(state), sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _remote_path(config: OffsiteConfig, tier: str, trading_date: str, seg: str) -> str:
    return f"{config.remote_root}/{tier}/{trading_date[:7]}/{trading_date}/{seg}.tar.zst"


def _parse_remote_md5(stdout: str) -> str | None:
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.split()[0].lower()
    return None


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


def _append_ledger_entry(ledger_path: Path, entry: LedgerEntry) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_entry_to_dict(entry), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def seal_and_upload(
    capture_root: Path,
    *,
    today: date,
    full_scan: bool,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    now_fn: Callable[[], datetime] = _utcnow,
    config: OffsiteConfig = OffsiteConfig(),  # noqa: B008
) -> SealReport:
    """Seal every pending capture member into verified offsite segments.

    Segments are uploaded before they are recorded, and recorded only after
    the remote MD5 equals the local archive MD5, so the ledger never claims an
    object that is absent or corrupt offsite.

    Raises:
        ValueError: Unexpected layout (non-date directory under a tier) or an
            immutability violation.
        RuntimeError: Another seal run holds the local seal lock.
        subprocess.CalledProcessError: rclone upload/hash failure.
        OSError: Staging or ledger write failure.
    """
    capture_root = Path(capture_root)
    lock_path = capture_root / "offsite" / ".seal.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another seal run holds the local seal lock") from exc
        try:
            return _seal_locked(capture_root, today=today, full_scan=full_scan, run_fn=run_fn, now_fn=now_fn, config=config)
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _seal_locked(
    capture_root: Path,
    *,
    today: date,
    full_scan: bool,
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
                selected = True
            else:
                key = f"{tier}/{date_name}"
                stored = state.get(key)
                if stored is None:
                    selected = True
                else:
                    current_peak = _structural_max_mtime_ns(tier_root, date_name, depth)
                    selected = current_peak > stored
            if not selected:
                continue
            dates_scanned += 1
            # 계획 전에 워터마크를 잡아야 스캔 도중 생긴 run 디렉터리가 다음 실행에서 재탐지된다
            scan_peak = _structural_max_mtime_ns(tier_root, date_name, depth)
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
            for seg_members in segments:
                seg = segment_name(seg_members)
                staging_path = staging_dir / f"{seg}.tar.zst"
                try:
                    _build_archive(capture_root, seg_members, staging_path)
                    _verify_archive_members(staging_path, seg_members)
                    local_md5 = _local_md5(staging_path)
                    local_bytes = staging_path.stat().st_size
                    remote = _remote_path(config, tier, date_name, seg)
                    pre = run_fn(
                        [rclone, "md5sum", remote],
                        capture_output=True,
                        text=True,
                        timeout=config.rclone_timeout_sec,
                        check=False,
                    )
                    remote_pre = _parse_remote_md5(pre.stdout) if pre.returncode == 0 else None
                    if remote_pre != local_md5:
                        uploaded = run_fn(
                            [rclone, "copyto", str(staging_path), remote],
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
                        remote_post = _parse_remote_md5(post.stdout)
                        if remote_post != local_md5:
                            raise ValueError(f"remote MD5 mismatch for {remote}")
                    committed_at = now_fn().isoformat()
                    ledger_entry = LedgerEntry(
                        tier=tier,
                        trading_date=date_name,
                        segment_name=seg,
                        remote_path=remote,
                        members=tuple(seg_members),
                        archive_bytes=local_bytes,
                        archive_md5=local_md5,
                        committed_at=committed_at,
                    )
                    _append_ledger_entry(_ledger_path(capture_root, tier, date_name), ledger_entry)
                    segments_committed += 1
                    members_committed += len(seg_members)
                    archive_bytes_total += local_bytes
                    logger.info(
                        "[SYS] stage=capture_offsite tier=%s date=%s segment=%s.tar.zst members=%d bytes=%d",
                        tier,
                        date_name,
                        seg,
                        len(seg_members),
                        local_bytes,
                    )
                finally:
                    with contextlib.suppress(OSError):
                        staging_path.unlink(missing_ok=True)
            state[f"{tier}/{date_name}"] = scan_peak
    _save_scan_state(capture_root, state)
    logger.info(
        "[SYS] stage=capture_offsite dates=%d segments=%d members=%d bytes=%d missing=%d",
        dates_scanned,
        segments_committed,
        members_committed,
        archive_bytes_total,
        missing_sealed_members,
    )
    return SealReport(
        dates_scanned=dates_scanned,
        segments_committed=segments_committed,
        members_committed=members_committed,
        archive_bytes=archive_bytes_total,
        missing_sealed_members=missing_sealed_members,
    )


LOCAL_SEALED_RETENTION_DAYS: int = 30


@dataclass(frozen=True)
class LocalRetentionReport:
    removed: tuple[str, ...]
    kept: tuple[tuple[str, str], ...]
    bytes_removed: int


def prune_local_sealed_capture(
    capture_root: Path,
    *,
    today: date,
    retention_days: int = LOCAL_SEALED_RETENTION_DAYS,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    config: OffsiteConfig = OffsiteConfig(),  # noqa: B008
) -> LocalRetentionReport:
    """Remove local capture date directories whose every file is sealed offsite and verified.

    Raw and normalized capture tiers are append-only evidence whose durable copy is the
    sealed tar.zst segment set (ledger + remote MD5). Past the retention window, the local
    copy only costs disk and seal-scan time; production readers need at most the previous
    trading day. A directory is removed whole or not at all, because the sealer counts
    ledger members missing from a directory it still scans.

    Args:
        capture_root: Capture store root (holds the tiers and offsite/ledger).
        today: KST calendar date of the run.
        retention_days: Date directories strictly older than today - retention_days are candidates.
        run_fn: Subprocess runner for ``rclone md5sum`` (test injection).
        config: Offsite contract (tiers, remote_root, rclone timeout).

    Returns:
        Removed directories, expired directories kept with a reason, and bytes reclaimed.

    Raises:
        OSError: Deleting an eligible directory failed; partial removal is never silenced.
        ValueError: Retention window shorter than the append window.
    """
    if retention_days < config.recent_window_days + 1:
        raise ValueError(
            f"retention_days={retention_days} shorter than append window "
            f"(recent_window_days={config.recent_window_days})"
        )
    capture_root = Path(capture_root)
    cutoff = today - timedelta(days=retention_days)
    rclone: str | None = None
    removed: list[str] = []
    kept: list[tuple[str, str]] = []
    bytes_removed = 0
    for tier in sorted(config.tiers):
        tier_root = capture_root / tier
        if not tier_root.exists():
            continue
        if tier_root.is_symlink() or not tier_root.is_dir():
            continue
        for child in sorted(tier_root.iterdir(), key=lambda entry: entry.name):
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
            shutil.rmtree(child)
            removed.append(label)
            bytes_removed += regular_sizes
    removed_sorted = tuple(sorted(removed))
    kept_sorted = tuple(sorted(kept))
    return LocalRetentionReport(removed=removed_sorted, kept=kept_sorted, bytes_removed=bytes_removed)


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
