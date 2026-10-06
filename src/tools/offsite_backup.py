"""Nightly offsite backup runner carrying sealed segments and loose tiers."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.io_utils import atomic_write_text
from src.tools.capture_offsite import OffsiteConfig, SealReport, seal_and_upload
from src.tools.offsite_common import OFFSITE_REMOTE_BASE
from src.tools.offsite_common import resolve_rclone_bin as _resolve_rclone_bin
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")

BACKUP_REMOTE_BASE: str = OFFSITE_REMOTE_BASE
LOOSE_SUBTREES: tuple[str, ...] = ("data", "artifacts")
LOOSE_EXCLUDES: tuple[str, ...] = (
    "/history/capture/raw/**",
    "/history/capture/normalized/**",
    "/history/capture/manifests/**",
    "/history/capture/backups/**",
    "/history/capture/staging/**",
)
LOOSE_RCLONE_FLAGS: tuple[str, ...] = ("--transfers", "4", "--checkers", "8")
BACKUP_SLOT_KST: time = time(22, 15)
BACKUP_SLOT_WEEKDAYS: frozenset[int] = frozenset({0, 1, 2, 3, 4})
REPORT_RELPATH: str = "offsite/last_run.json"
BACKUP_PROGRESS_RELPATH: str = "offsite/in_progress.json"
BACKUP_SEAL_BUDGET: timedelta = timedelta(minutes=90)
BACKUP_TOTAL_BUDGET: timedelta = timedelta(minutes=150)
BACKUP_PROGRESS_GRACE: timedelta = timedelta(minutes=30)
BACKUP_INFO_ISSUES: frozenset[str] = frozenset({"offsite_backup:running", "offsite_backup:draining"})
BACKUP_DEFERRED_HISTORY_RELPATH: str = "offsite/deferred_history.json"
BACKUP_DEFERRED_MAX_STALLED_RUNS: int = 3
BACKUP_DEFERRED_MAX_AGE_DAYS: int = 21
RCLONE_DURATION_EXCEEDED_EXIT: int = 10
# 개별 복사의 하드 서브프로세스 타임아웃 — 유닛 백스톱(TimeoutStartSec=3h)보다 길어 먼저 끊기지 않는다
LOOSE_RCLONE_TIMEOUT_SEC: int = 4 * 3600


@dataclass(frozen=True)
class BackupRunReport:
    started_at: str
    finished_at: str
    status: str
    steps: dict[str, dict[str, Any]]
    core_panels: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class BackupProgress:
    started_at: str
    deadline_at: str
    pid: int
    host: str


@dataclass(frozen=True)
class DeferredProgress:
    run_started_at: str
    deferred_dates: int
    deferred_loose: tuple[str, ...]
    oldest_deferred_date: str


def _deferred_history_default_path(report_path: Path) -> Path:
    return Path(report_path).parent / Path(BACKUP_DEFERRED_HISTORY_RELPATH).name


def append_deferred_progress(capture_root: Path, entry: DeferredProgress, *, keep: int = 14) -> Path:
    """Append one deferred-progress entry, keeping a rolling window."""
    path = Path(capture_root) / BACKUP_DEFERRED_HISTORY_RELPATH
    entries: list[dict[str, Any]] = []
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            entries.extend(item for item in raw if isinstance(item, dict))
    except (OSError, ValueError):
        entries = []
    entries.append(
        {
            "run_started_at": entry.run_started_at,
            "deferred_dates": entry.deferred_dates,
            "deferred_loose": list(entry.deferred_loose),
            "oldest_deferred_date": entry.oldest_deferred_date,
        }
    )
    entries = entries[-max(1, keep):]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(entries, sort_keys=True), mode=None)
    return path


def read_deferred_progress(path: Path) -> tuple[DeferredProgress, ...]:
    """Return history oldest->newest; absent/unreadable/malformed -> empty tuple."""
    try:
        raw: object = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    if not isinstance(raw, list):
        return ()
    entries: list[DeferredProgress] = []
    for item in raw:
        if not isinstance(item, dict):
            return ()
        try:
            started = item.get("run_started_at")
            count = item.get("deferred_dates")
            loose = item.get("deferred_loose", [])
            oldest = item.get("oldest_deferred_date", "")
            if not isinstance(started, str) or not isinstance(count, int) or isinstance(count, bool):
                return ()
            if not isinstance(loose, list) or not all(isinstance(sub, str) for sub in loose):
                return ()
            if not isinstance(oldest, str):
                return ()
            datetime.fromisoformat(started)
            if oldest:
                date.fromisoformat(oldest)
        except (ValueError, TypeError):
            return ()
        entries.append(
            DeferredProgress(
                run_started_at=started,
                deferred_dates=count,
                deferred_loose=tuple(loose),
                oldest_deferred_date=oldest,
            )
        )
    return tuple(entries)


def _deferred_state_from_steps(steps: Mapping[str, Any]) -> tuple[int, tuple[str, ...], str]:
    count = 0
    seal = steps.get("capture_seal")
    if isinstance(seal, dict):
        raw_count = seal.get("deferred_dates", 0)
        if isinstance(raw_count, int) and not isinstance(raw_count, bool):
            count = raw_count
    loose = tuple(sorted(sub for sub in LOOSE_SUBTREES if isinstance(steps.get(sub), dict) and steps[sub].get("status") == "deferred"))
    oldest = ""
    if isinstance(seal, dict):
        raw_oldest = seal.get("oldest_deferred_date", "")
        if isinstance(raw_oldest, str):
            oldest = raw_oldest
    return count, loose, oldest


def _is_deferred(count: int, loose: tuple[str, ...]) -> bool:
    return count > 0 or bool(loose)


def _classify_deferred(
    effective: tuple[DeferredProgress, ...], audit_at: datetime
) -> str:
    """Return draining (informational) or deferred (warning) for a deferred run."""
    if len(effective) <= 1:
        return "offsite_backup:draining"
    current = effective[-1]
    if current.oldest_deferred_date:
        try:
            age = (_kst_now(audit_at).date() - date.fromisoformat(current.oldest_deferred_date)).days
        except ValueError:
            age = 0
        if age > BACKUP_DEFERRED_MAX_AGE_DAYS:
            return "offsite_backup:deferred"
    trailing = 1
    for pos in range(len(effective) - 1, 0, -1):
        if effective[pos].deferred_dates >= effective[pos - 1].deferred_dates:
            trailing += 1
        else:
            break
    if trailing >= BACKUP_DEFERRED_MAX_STALLED_RUNS:
        return "offsite_backup:deferred"
    if len(effective) >= BACKUP_DEFERRED_MAX_STALLED_RUNS:
        for subtree in LOOSE_SUBTREES:
            if all(subtree in entry.deferred_loose for entry in effective[-BACKUP_DEFERRED_MAX_STALLED_RUNS:]):
                return "offsite_backup:deferred"
    return "offsite_backup:draining"


def backup_backlog_info(
    report_path: Path, audit_at: datetime, *, history_path: Path | None = None
) -> str | None:
    """Render the draining/stalled backlog line with counts, or None when not deferred."""
    try:
        path = Path(report_path)
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("status") != "ok":
            return None
        steps = raw.get("steps")
        if not isinstance(steps, dict):
            return None
        count, _loose, _oldest = _deferred_state_from_steps(steps)
        if not _is_deferred(count, _loose):
            return None
        history_file = Path(history_path) if history_path is not None else _deferred_history_default_path(path)
        history = read_deferred_progress(history_file)
        started_raw = raw.get("started_at")
        current = DeferredProgress(
            run_started_at=str(started_raw) if isinstance(started_raw, str) else "",
            deferred_dates=count,
            deferred_loose=_loose,
            oldest_deferred_date=_oldest,
        )
        effective = (*history, current) if not history or history[-1].run_started_at != current.run_started_at else history
        if len(effective) >= 2:
            previous = effective[-2].deferred_dates
        else:
            return f"backup backlog draining: {count} -> {count} dates"
        verdict = _classify_deferred(effective, audit_at)
        if verdict == "offsite_backup:draining":
            return f"backup backlog draining: {previous} -> {count} dates"
        oldest_suffix = f" oldest={current.oldest_deferred_date}" if current.oldest_deferred_date else ""
        return f"backup backlog stalled: {previous} -> {count} dates{oldest_suffix}"
    except (OSError, ValueError):
        return None


def write_backup_progress(capture_root: Path, progress: BackupProgress) -> Path:
    """Atomically publish the in-flight backup marker."""
    path = Path(capture_root) / BACKUP_PROGRESS_RELPATH
    payload = {
        "started_at": progress.started_at,
        "deadline_at": progress.deadline_at,
        "pid": progress.pid,
        "host": progress.host,
    }
    atomic_write_text(path, json.dumps(payload, sort_keys=True), mode=None)
    return path


def read_backup_progress(path: Path) -> BackupProgress | None:
    """Return the in-flight marker, or None when it must not mask staleness."""
    try:
        raw: object = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    started_raw = raw.get("started_at")
    deadline_raw = raw.get("deadline_at")
    pid_raw = raw.get("pid")
    host_raw = raw.get("host")
    if not isinstance(started_raw, str) or not isinstance(deadline_raw, str):
        return None
    if not isinstance(pid_raw, int) or isinstance(pid_raw, bool) or not isinstance(host_raw, str):
        return None
    try:
        started_at = datetime.fromisoformat(started_raw)
        deadline_at = datetime.fromisoformat(deadline_raw)
    except ValueError:
        return None
    if started_at.tzinfo is None or started_at.utcoffset() is None:
        return None
    if deadline_at.tzinfo is None or deadline_at.utcoffset() is None:
        return None
    return BackupProgress(started_at=started_raw, deadline_at=deadline_raw, pid=pid_raw, host=host_raw)


def clear_backup_progress(capture_root: Path) -> None:
    """Remove the in-flight marker; absent file is not an error."""
    with contextlib.suppress(FileNotFoundError):
        (Path(capture_root) / BACKUP_PROGRESS_RELPATH).unlink()


def loose_copy_command(
    rclone: str, project_root: Path, subtree: str, snapshot_day: str, extra_excludes: Sequence[str] = (),
    max_duration_s: int | None = None,
) -> list[str]:
    """Build the loose-tier rclone copy for one project subtree.

    Copy (never sync) so local deletions never propagate; overwritten remote
    files move to _deleted/<subtree>/<snapshot_day> for backup_prune retention.
    Capture segment tiers and local-only snapshots are excluded (only the
    "data" subtree carries excludes). A max_duration_s bound stops the copy
    softly at the shared-lock budget instead of failing it.
    """
    dest = f"{BACKUP_REMOTE_BASE}/{subtree}"
    cmd = [
        rclone,
        "copy",
        str(project_root / subtree),
        dest,
        "--backup-dir",
        f"{BACKUP_REMOTE_BASE}/_deleted/{subtree}/{snapshot_day}",
        *LOOSE_RCLONE_FLAGS,
    ]
    if subtree == "data":
        for pattern in (*LOOSE_EXCLUDES, *extra_excludes):
            cmd += ["--exclude", pattern]
    elif extra_excludes:
        for pattern in extra_excludes:
            cmd += ["--exclude", pattern]
    if max_duration_s is not None:
        cmd += ["--max-duration", f"{int(max_duration_s)}s", "--cutoff-mode", "soft"]
    return cmd


def _load_previous_core_panels(capture_root: Path) -> list[dict[str, Any]]:
    path = capture_root / REPORT_RELPATH
    if not path.exists():
        return []
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(raw, dict):
        return []
    panels = raw.get("core_panels")
    if isinstance(panels, list):
        return [item for item in panels if isinstance(item, dict)]
    return []


def _excludes_for_subtree(issues: Sequence[str], subtree: str) -> list[str]:
    excludes: list[str] = []
    prefix = f"{subtree}/"
    for issue in issues:
        parts = issue.split(":")
        if len(parts) < 3:
            continue
        relpath = ":".join(parts[1:-1])
        if relpath.startswith(prefix):
            excludes.append("/" + relpath[len(prefix):])
    return sorted(set(excludes))


def _kst_now(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=KST)
    return now.astimezone(KST)


def _write_report(capture_root: Path, report: BackupRunReport) -> None:
    path = capture_root / REPORT_RELPATH
    payload: dict[str, Any] = {
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "status": report.status,
        "steps": report.steps,
    }
    if report.core_panels is not None:
        payload["core_panels"] = report.core_panels
    atomic_write_text(path, json.dumps(payload, sort_keys=True), mode=None)


def run_offsite_backup(
    project_root: Path,
    capture_root: Path,
    *,
    now: datetime,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    seal_fn: Callable[..., SealReport] = seal_and_upload,
    accepted_missing: frozenset[str] = frozenset(),
) -> BackupRunReport:
    """Run one nightly offsite backup: seal capture segments, then loose copies.

    Steps run in order capture_seal -> data -> artifacts so the offsite
    ledger committed by sealing is carried by the same night's data copy.
    A failing step does not skip later steps (partial offsite progress beats
    none); the report is persisted before the overall outcome is raised.

    Raises:
        RuntimeError: Any step failed (after the report is written), so the
            systemd OnFailure alert fires.
    """
    kst = _kst_now(now)
    today = kst.date()
    snapshot_day = today.isoformat()
    full_scan = kst.weekday() == OffsiteConfig().full_scan_weekday
    started_at = kst.astimezone(UTC).isoformat()
    try:
        progress = BackupProgress(
            started_at=started_at,
            deadline_at=(kst + BACKUP_TOTAL_BUDGET + BACKUP_PROGRESS_GRACE).astimezone(UTC).isoformat(),
            pid=os.getpid(),
            host=socket.gethostname(),
        )
        write_backup_progress(capture_root, progress)
    except OSError as exc:
        logger.warning("[SYS] stage=offsite_backup progress_marker=UNWRITABLE reason=%s: %s", type(exc).__name__, exc)
    report_written = False
    try:
        try:
            run_report = _run_offsite_backup_inner(
                project_root,
                capture_root,
                kst=kst,
                today=today,
                snapshot_day=snapshot_day,
                full_scan=full_scan,
                started_at=started_at,
                run_fn=run_fn,
                seal_fn=seal_fn,
                accepted_missing=accepted_missing,
            )
        except Exception as exc:
            run_report = BackupRunReport(
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
                status="failed",
                steps={"backup_run": {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}},
                core_panels=_load_previous_core_panels(capture_root) or None,
            )
        _write_report(capture_root, run_report)
        report_written = True
        try:
            count, loose, oldest = _deferred_state_from_steps(run_report.steps)
            append_deferred_progress(
                capture_root,
                DeferredProgress(
                    run_started_at=run_report.started_at,
                    deferred_dates=count,
                    deferred_loose=loose,
                    oldest_deferred_date=oldest,
                ),
            )
        except OSError as exc:
            logger.warning("[SYS] stage=offsite_backup deferred_history=UNWRITABLE reason=%s: %s", type(exc).__name__, exc)
        if run_report.status != "ok":
            failed = sorted(name for name, step in run_report.steps.items() if step.get("status") == "failed")
            raise RuntimeError(f"offsite backup failed: {','.join(failed)}")
        return run_report
    finally:
        # Preserve interruption evidence if report publication failed or the process was interrupted.
        if report_written:
            try:
                clear_backup_progress(capture_root)
            except OSError as exc:
                logger.warning("[SYS] stage=offsite_backup progress_marker=UNCLEARABLE reason=%s: %s", type(exc).__name__, exc)


def _run_offsite_backup_inner(
    project_root: Path,
    capture_root: Path,
    *,
    kst: datetime,
    today: date,
    snapshot_day: str,
    full_scan: bool,
    started_at: str,
    run_fn: Callable[..., subprocess.CompletedProcess[str]],
    seal_fn: Callable[..., SealReport],
    accepted_missing: frozenset[str],
) -> BackupRunReport:
    rclone = _resolve_rclone_bin()
    steps: dict[str, dict[str, Any]] = {}

    from src.tools.core_snapshot import CorePanelStat, collect_core_stats, validate_core_panels

    current_stats = collect_core_stats(project_root)
    raw_previous = _load_previous_core_panels(capture_root)
    previous_stats = [
        CorePanelStat(
            relpath=str(item.get("relpath", "")),
            sha256=str(item.get("sha256", "")),
            bytes=int(item.get("bytes", 0)),
            rows=None if item.get("rows") is None else int(item["rows"]),
            max_date=str(item.get("max_date", "")),
        )
        for item in raw_previous
        if isinstance(item.get("relpath"), str)
    ]
    core_issues = validate_core_panels(current_stats, previous_stats, accepted_missing=accepted_missing)
    if core_issues:
        steps["core_panels"] = {"status": "failed", "issues": core_issues}
    else:
        steps["core_panels"] = {"status": "ok", "issues": []}

    try:
        report = seal_fn(capture_root, today=today, full_scan=full_scan, deadline=kst + BACKUP_SEAL_BUDGET)
        steps["capture_seal"] = {
            "status": "ok",
            "dates_scanned": report.dates_scanned,
            "segments_committed": report.segments_committed,
            "members_committed": report.members_committed,
            "archive_bytes": report.archive_bytes,
            "missing_sealed_members": report.missing_sealed_members,
            "deferred_dates": report.deferred_dates,
            "oldest_deferred_date": getattr(report, "oldest_deferred_date", ""),
        }
    except Exception as exc:
        steps["capture_seal"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    for subtree in LOOSE_SUBTREES:
        remaining_s = max(60, int(((kst + BACKUP_TOTAL_BUDGET) - datetime.now(KST)).total_seconds()))
        cmd = loose_copy_command(
            rclone, project_root, subtree, snapshot_day, _excludes_for_subtree(core_issues, subtree),
            max_duration_s=remaining_s,
        )
        try:
            result = run_fn(cmd, capture_output=True, text=True, timeout=LOOSE_RCLONE_TIMEOUT_SEC, check=False)
        except Exception as exc:
            steps[subtree] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            continue
        if result.returncode == 0:
            if core_issues and _excludes_for_subtree(core_issues, subtree):
                steps[subtree] = {"status": "failed", "issues": core_issues}
            else:
                steps[subtree] = {"status": "ok"}
        elif result.returncode == RCLONE_DURATION_EXCEEDED_EXIT:
            steps[subtree] = {"status": "deferred", "returncode": result.returncode}
        else:
            payload: dict[str, Any] = {
                "status": "failed",
                "returncode": result.returncode,
                "stderr": (result.stderr or "")[-2000:],
            }
            if core_issues:
                payload["issues"] = core_issues
            steps[subtree] = payload

    status = "ok" if all(step.get("status") in ("ok", "deferred") for step in steps.values()) else "failed"
    finished_at = datetime.now(UTC).isoformat()
    if core_issues:
        persisted_panels = [{"relpath": e.relpath, "sha256": e.sha256, "bytes": e.bytes, "rows": e.rows, "max_date": e.max_date} for e in previous_stats] if raw_previous else None
        if persisted_panels is None:
            persisted_panels = [
                {"relpath": e.relpath, "sha256": e.sha256, "bytes": e.bytes, "rows": e.rows, "max_date": e.max_date}
                for e in current_stats
            ]
    else:
        persisted_panels = [
            {"relpath": e.relpath, "sha256": e.sha256, "bytes": e.bytes, "rows": e.rows, "max_date": e.max_date}
            for e in current_stats
        ]
    run_report = BackupRunReport(
        started_at=started_at, finished_at=finished_at, status=status, steps=steps, core_panels=persisted_panels
    )
    return run_report


def expected_backup_slot(audit_at: datetime) -> datetime:
    """Latest scheduled backup start (Mon-Fri 22:15 KST) strictly before audit_at."""
    at = _kst_now(audit_at)
    day = at.date()
    while True:
        candidate = datetime(day.year, day.month, day.day, BACKUP_SLOT_KST.hour, BACKUP_SLOT_KST.minute, tzinfo=KST)
        if candidate < at and candidate.weekday() in BACKUP_SLOT_WEEKDAYS:
            return candidate
        day = day.fromordinal(day.toordinal() - 1)


def backup_staleness_issues(
    report_path: Path, audit_at: datetime, *, progress_path: Path | None = None, history_path: Path | None = None
) -> list[str]:
    """Classify the last offsite run against the latest scheduled slot.

    Returns:
        [] when the report is ok and started at/after the expected slot;
        otherwise one of ["offsite_backup:missing"], ["offsite_backup:failed"],
        ["offsite_backup:stale"]. Unreadable report -> ["offsite_backup:unreadable"].
        An ok, fresh report whose seal deferred dates or whose loose copy hit the
        duration budget yields ["offsite_backup:draining"] while the backlog
        shrinks (informational, in BACKUP_INFO_ISSUES) and keeps
        ["offsite_backup:deferred"] once it stalls or ages past the bound
        (digest warning, not failure).
        A missing or stale report covered by a fresh in-flight marker yields
        ["offsite_backup:running"] before its deadline (informational, in
        BACKUP_INFO_ISSUES) or ["offsite_backup:interrupted"] after it (warning).
        A failed report is never downgraded to running.
    """
    path = Path(report_path)
    marker_path = Path(progress_path) if progress_path is not None else path.parent / Path(BACKUP_PROGRESS_RELPATH).name
    expected_slot = expected_backup_slot(audit_at)
    # Writers publish the report before clearing the marker; read in the opposite order.
    progress = read_backup_progress(marker_path)

    def _fresh_marker_outcome() -> list[str] | None:
        if progress is None:
            return None
        marker_started = datetime.fromisoformat(progress.started_at)
        marker_deadline = datetime.fromisoformat(progress.deadline_at)
        if marker_started < expected_slot:
            return None
        at = _kst_now(audit_at)
        if at < marker_deadline:
            return ["offsite_backup:running"]
        return ["offsite_backup:interrupted"]

    if not path.exists():
        return _fresh_marker_outcome() or ["offsite_backup:missing"]
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["offsite_backup:unreadable"]
    if not isinstance(raw, dict):
        return ["offsite_backup:unreadable"]
    status = raw.get("status")
    started_raw = raw.get("started_at")
    if not isinstance(status, str) or not isinstance(started_raw, str):
        return ["offsite_backup:unreadable"]
    try:
        started_at = datetime.fromisoformat(started_raw)
    except ValueError:
        return ["offsite_backup:unreadable"]
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    if status != "ok":
        return ["offsite_backup:failed"]
    if started_at < expected_slot:
        return _fresh_marker_outcome() or ["offsite_backup:stale"]
    steps = raw.get("steps")
    if isinstance(steps, dict):
        count, loose, oldest = _deferred_state_from_steps(steps)
        if _is_deferred(count, loose):
            history_file = Path(history_path) if history_path is not None else _deferred_history_default_path(path)
            history = read_deferred_progress(history_file)
            current = DeferredProgress(
                run_started_at=started_raw if isinstance(started_raw, str) else "",
                deferred_dates=count,
                deferred_loose=loose,
                oldest_deferred_date=oldest,
            )
            if history and history[-1].run_started_at == current.run_started_at:
                effective = history
            else:
                effective = (*history, current)
            return [_classify_deferred(effective, audit_at)]
    return []


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry; logic covered via run_offsite_backup scenarios
    import argparse
    import time as _time

    parser = argparse.ArgumentParser(description="Nightly offsite backup: seal segments then loose copy")
    parser.add_argument("--accept-missing", action="append", default=[], metavar="RELPATH")
    args = parser.parse_args(argv)
    project_root = Path.cwd()
    capture_root = _capture_root()
    now = datetime.now(KST)
    started = _time.monotonic()
    try:
        report = run_offsite_backup(project_root, capture_root, now=now, accepted_missing=frozenset(args.accept_missing))
    except RuntimeError as exc:
        logger.info("[SYS] stage=offsite_backup status=failed duration_s=%.0f error=%s", _time.monotonic() - started, exc)
        sys.exit(1)
    seal_step = report.steps.get("capture_seal", {})
    logger.info(
        "[SYS] stage=offsite_backup status=%s duration_s=%.0f segments=%s members=%s archive_bytes=%s",
        report.status,
        _time.monotonic() - started,
        seal_step.get("segments_committed", 0),
        seal_step.get("members_committed", 0),
        seal_step.get("archive_bytes", 0),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
