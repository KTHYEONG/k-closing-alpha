"""One-time removal of pre-segment loose capture objects and accidental rollback-snapshot copies.

Drive object creation is rate-bound. Loose small files were superseded by
sealed tar.zst segments whose ledger + remote MD5 prove equal-or-better
coverage. Dry-run is the default. The tool must run on or-vps, because the
ledger lives in the capture store there.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.data.intraday_store import _capture_root
from src.tools.backup_prune import _resolve_rclone_bin
from src.tools.capture_offsite import LedgerEntry, OffsiteConfig, SegmentMember, read_ledger, restore_date
from src.tools.offsite_backup import BACKUP_REMOTE_BASE

logger = logging.getLogger(__name__)

LOOSE_REMOTE_ROOT: str = f"{BACKUP_REMOTE_BASE}/data/history/capture"
LOOSE_TIERS: tuple[str, ...] = OffsiteConfig().tiers
ROLLBACK_SNAPSHOT_SUBTREE: str = "backups"
DRIVE_LOCK_WAIT_SEC: float = 7200.0
# 수만 개 객체를 파일당 rclone 프로세스로 지우면 공용 Drive 락을 반나절 점유하므로 목록 1회 일괄 삭제에 상한을 둔다
BATCH_DELETE_TIMEOUT_SEC: int = 4 * 3600


@dataclass(frozen=True)
class LegacyCandidate:
    rule: Literal["sealed_loose_member", "rollback_snapshot_copy"]
    remote_path: str
    size: int
    evidence: str


@dataclass(frozen=True)
class LegacyPlan:
    candidates: tuple[LegacyCandidate, ...]
    kept: tuple[tuple[str, str], ...]


def _member_index(
    ledgers: Mapping[tuple[str, str], Sequence[LedgerEntry]],
) -> dict[str, list[tuple[LedgerEntry, SegmentMember]]]:
    index: dict[str, list[tuple[LedgerEntry, SegmentMember]]] = {}
    for entries in ledgers.values():
        for entry in entries:
            for member in entry.members:
                index.setdefault(member.path, []).append((entry, member))
    return index


def build_legacy_plan(
    loose_objects: Sequence[tuple[str, int]],
    ledgers: Mapping[tuple[str, str], Sequence[LedgerEntry]],
    verified_segments: AbstractSet[str],
) -> LegacyPlan:
    """Classify loose remote capture objects into deletion candidates.

    Args:
        loose_objects: (path relative to LOOSE_REMOTE_ROOT, size) for every remote file under it.
        ledgers: Committed ledger entries keyed by (tier, trading_date).
        verified_segments: Segment remote paths whose current remote MD5 equals their ledger archive_md5.

    Returns:
        sealed_loose_member candidates for "<tier>/<date>/..." objects matched by a member of a
        verified segment with equal size; rollback_snapshot_copy candidates for every object under
        "backups/"; everything else kept with a reason ("unsealed", "size_mismatch",
        "segment_unverified", "out_of_scope").
    """
    index = _member_index(ledgers)
    candidates: list[LegacyCandidate] = []
    kept: list[tuple[str, str]] = []
    for rel, size in loose_objects:
        if rel == ROLLBACK_SNAPSHOT_SUBTREE or rel.startswith(ROLLBACK_SNAPSHOT_SUBTREE + "/"):
            candidates.append(
                LegacyCandidate(
                    rule="rollback_snapshot_copy",
                    remote_path=rel,
                    size=size,
                    evidence="rollback_snapshot_copy",
                )
            )
            continue
        parts = rel.split("/")
        if len(parts) < 3 or parts[0] not in LOOSE_TIERS:
            kept.append((rel, "out_of_scope"))
            continue
        tier, trading_date = parts[0], parts[1]
        entries = ledgers.get((tier, trading_date))
        if not entries:
            kept.append((rel, "unsealed"))
            continue
        matches = index.get(rel, [])
        matches = [item for item in matches if item[0].tier == tier and item[0].trading_date == trading_date]
        if not matches:
            kept.append((rel, "unsealed"))
            continue
        verified_equal = [item for item in matches if item[0].remote_path in verified_segments and item[1].size == size]
        if verified_equal:
            chosen = verified_equal[0][0]
            candidates.append(
                LegacyCandidate(
                    rule="sealed_loose_member",
                    remote_path=rel,
                    size=size,
                    evidence=f"segment={chosen.remote_path} md5={chosen.archive_md5}",
                )
            )
            continue
        if any(item[0].remote_path not in verified_segments for item in matches):
            only_unverified = all(item[0].remote_path not in verified_segments for item in matches)
            if only_unverified:
                kept.append((rel, "segment_unverified"))
                continue
        kept.append((rel, "size_mismatch"))
    return LegacyPlan(candidates=tuple(candidates), kept=tuple(kept))


def verify_segments(
    ledgers: Mapping[tuple[str, str], Sequence[LedgerEntry]],
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> frozenset[str]:
    """Return the remote paths of segments whose ``rclone md5sum`` equals the ledger archive_md5; errors count as unverified (fail-closed)."""
    rclone = _resolve_rclone_bin()
    timeout_sec = OffsiteConfig().rclone_timeout_sec
    want: dict[str, str] = {}
    for entries in ledgers.values():
        for entry in entries:
            want.setdefault(entry.remote_path, entry.archive_md5)
    verified: set[str] = set()
    for remote, archive_md5 in want.items():
        try:
            result = run_fn(
                [rclone, "md5sum", remote],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except Exception as exc:
            logger.debug("[SYS] stage=offsite_legacy_cleanup step=verify_segments remote=%s error=%s", remote, exc)
            continue
        if result.returncode != 0:
            continue
        token: str | None = None
        for line in (result.stdout or "").splitlines():
            stripped = line.strip()
            if stripped:
                token = stripped.split()[0].lower()
                break
        if token is not None and token == archive_md5.lower():
            verified.add(remote)
    return frozenset(verified)


def _drive_lock_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return Path(runtime) / "quant-gdrive.lock"


@contextlib.contextmanager
def _held_drive_lock(timeout_sec: float = DRIVE_LOCK_WAIT_SEC) -> Iterator[Path]:
    path = _drive_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        deadline = time.monotonic() + float(timeout_sec)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring drive lock: {path}") from None
                time.sleep(0.05)
        try:
            yield path
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _list_loose_objects(
    run_fn: Callable[..., subprocess.CompletedProcess[str]],
    *,
    timeout_sec: int,
) -> list[tuple[str, int]]:
    rclone = _resolve_rclone_bin()
    result = run_fn(
        [rclone, "lsjson", "--files-only", "--recursive", LOOSE_REMOTE_ROOT],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    try:
        raw: object = json.loads(result.stdout or "[]")
    except ValueError as exc:
        raise ValueError(f"unreadable loose listing: {exc}") from None
    if not isinstance(raw, list):
        raise ValueError("unreadable loose listing")
    objects: list[tuple[str, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("unreadable loose listing")
        rel = item.get("Path")
        size = item.get("Size")
        if not isinstance(rel, str) or not isinstance(size, int):
            raise ValueError("unreadable loose listing")
        objects.append((rel.replace("\\", "/"), size))
    return objects


def _load_ledgers(capture_root: Path, loose_objects: Sequence[tuple[str, int]]) -> dict[tuple[str, str], list[LedgerEntry]]:
    wanted: set[tuple[str, str]] = set()
    for rel, _size in loose_objects:
        parts = rel.split("/")
        if len(parts) >= 3 and parts[0] in LOOSE_TIERS:
            wanted.add((parts[0], parts[1]))
    ledgers: dict[tuple[str, str], list[LedgerEntry]] = {}
    for key in sorted(wanted):
        ledgers[key] = read_ledger(capture_root, key[0], key[1])
    return ledgers


def _log_plan(plan: LegacyPlan) -> None:
    kept_total = len(plan.kept)
    for rule in ("sealed_loose_member", "rollback_snapshot_copy"):
        selected = [item for item in plan.candidates if item.rule == rule]
        logger.info(
            "[SYS] stage=offsite_legacy_cleanup rule=%s candidates=%d bytes=%d kept=%d",
            rule,
            len(selected),
            sum(item.size for item in selected),
            kept_total,
        )


def _restore_drill_dates(plan: LegacyPlan) -> dict[str, str]:
    latest: dict[str, str] = {}
    for item in plan.candidates:
        if item.rule != "sealed_loose_member":
            continue
        tier, trading_date = item.remote_path.split("/")[:2]
        if trading_date > latest.get(tier, ""):
            latest[tier] = trading_date
    return latest


def main(argv: Sequence[str] | None = None) -> int:
    """``uv run python -m src.tools.offsite_legacy_cleanup [--apply]`` under the shared Drive flock."""
    parser = argparse.ArgumentParser(description="Remove superseded loose capture objects from Drive offsite")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    capture_root = _capture_root()
    timeout_sec = OffsiteConfig().rclone_timeout_sec
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run

    if not args.apply:
        try:
            loose = _list_loose_objects(runner, timeout_sec=timeout_sec)
            ledgers = _load_ledgers(capture_root, loose)
            verified = verify_segments(ledgers, run_fn=runner)
            plan = build_legacy_plan(loose, ledgers, verified)
        except Exception as exc:
            logger.error("[SYS] stage=offsite_legacy_cleanup status=failed error=%s: %s", type(exc).__name__, exc, exc_info=True)
            return 1
        _log_plan(plan)
        return 0

    try:
        with _held_drive_lock(timeout_sec=DRIVE_LOCK_WAIT_SEC):
            loose = _list_loose_objects(runner, timeout_sec=timeout_sec)
            ledgers = _load_ledgers(capture_root, loose)
            verified = verify_segments(ledgers, run_fn=runner)
            plan = build_legacy_plan(loose, ledgers, verified)
            drill_dates = _restore_drill_dates(plan)
            with tempfile.TemporaryDirectory(prefix="legacy-cleanup-drill-") as tmpdir:
                for tier in sorted(drill_dates):
                    trading_date = drill_dates[tier]
                    try:
                        restored = restore_date(capture_root, tier, trading_date, Path(tmpdir), run_fn=runner)
                    except Exception:
                        logger.error(
                            "[SYS] stage=offsite_legacy_cleanup step=restore_drill tier=%s date=%s members=%d status=failed",
                            tier,
                            trading_date,
                            0,
                            exc_info=True,
                        )
                        return 1
                    logger.info(
                        "[SYS] stage=offsite_legacy_cleanup step=restore_drill tier=%s date=%s members=%d status=ok",
                        tier,
                        trading_date,
                        len(restored),
                    )
            rclone = _resolve_rclone_bin()
            if plan.candidates:
                with tempfile.TemporaryDirectory(prefix="legacy-cleanup-delete-") as listdir:
                    files_from = Path(listdir) / "candidates.txt"
                    files_from.write_text("".join(f"{item.remote_path}\n" for item in plan.candidates), encoding="utf-8")
                    runner(
                        [rclone, "delete", LOOSE_REMOTE_ROOT, "--files-from", str(files_from), "--fast-list", "--checkers", "8"],
                        capture_output=True,
                        text=True,
                        timeout=BATCH_DELETE_TIMEOUT_SEC,
                        check=True,
                    )
            runner(
                [rclone, "rmdirs", "--leave-root", LOOSE_REMOTE_ROOT],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=True,
            )
    except Exception as exc:
        logger.error("[SYS] stage=offsite_legacy_cleanup status=failed error=%s: %s", type(exc).__name__, exc, exc_info=True)
        return 1
    _log_plan(plan)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
