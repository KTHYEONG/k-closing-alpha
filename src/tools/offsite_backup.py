"""Nightly offsite backup runner carrying sealed segments and loose tiers."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src.data.intraday_store import _capture_root
from src.tools.backup_prune import _resolve_rclone_bin
from src.tools.capture_offsite import OffsiteConfig, SealReport, seal_and_upload

logger = logging.getLogger(__name__)

KST = ZoneInfo("Asia/Seoul")

BACKUP_REMOTE_BASE: str = "gdrive:quant-lake/live/k-closing-alpha"
LOOSE_SUBTREES: tuple[str, ...] = ("data", "artifacts")
LOOSE_EXCLUDES: tuple[str, ...] = (
    "/history/capture/raw/**",
    "/history/capture/normalized/**",
    "/history/capture/backups/**",
    "/history/capture/staging/**",
)
LOOSE_RCLONE_FLAGS: tuple[str, ...] = ("--transfers", "4", "--checkers", "8")
BACKUP_SLOT_KST: time = time(22, 15)
BACKUP_SLOT_WEEKDAYS: frozenset[int] = frozenset({0, 1, 2, 3, 4})
REPORT_RELPATH: str = "offsite/last_run.json"
# systemd TimeoutStartSec(4h)와 동일한 상한 — 개별 복사가 유닛 전체 예산을 넘지 못하게 한다
LOOSE_RCLONE_TIMEOUT_SEC: int = 4 * 3600


@dataclass(frozen=True)
class BackupRunReport:
    started_at: str
    finished_at: str
    status: str
    steps: dict[str, dict[str, Any]]
    core_panels: list[dict[str, Any]] | None = None


def loose_copy_command(
    rclone: str, project_root: Path, subtree: str, snapshot_day: str, extra_excludes: Sequence[str] = ()
) -> list[str]:
    """Build the loose-tier rclone copy for one project subtree.

    Copy (never sync) so local deletions never propagate; overwritten remote
    files move to _deleted/<subtree>/<snapshot_day> for backup_prune retention.
    Capture segment tiers and local-only snapshots are excluded (only the
    "data" subtree carries excludes).
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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload: dict[str, Any] = {
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "status": report.status,
        "steps": report.steps,
    }
    if report.core_panels is not None:
        payload["core_panels"] = report.core_panels
    tmp.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def run_offsite_backup(
    project_root: Path,
    capture_root: Path,
    *,
    now: datetime,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    seal_fn: Callable[..., SealReport] = seal_and_upload,
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
    core_issues = validate_core_panels(current_stats, previous_stats)
    if core_issues:
        steps["core_panels"] = {"status": "failed", "issues": core_issues}
    else:
        steps["core_panels"] = {"status": "ok", "issues": []}

    try:
        report = seal_fn(capture_root, today=today, full_scan=full_scan)
        steps["capture_seal"] = {
            "status": "ok",
            "dates_scanned": report.dates_scanned,
            "segments_committed": report.segments_committed,
            "members_committed": report.members_committed,
            "archive_bytes": report.archive_bytes,
            "missing_sealed_members": report.missing_sealed_members,
        }
    except Exception as exc:
        steps["capture_seal"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    for subtree in LOOSE_SUBTREES:
        cmd = loose_copy_command(rclone, project_root, subtree, snapshot_day, _excludes_for_subtree(core_issues, subtree))
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
        else:
            payload: dict[str, Any] = {
                "status": "failed",
                "returncode": result.returncode,
                "stderr": (result.stderr or "")[-2000:],
            }
            if core_issues:
                payload["issues"] = core_issues
            steps[subtree] = payload

    status = "ok" if all(step.get("status") == "ok" for step in steps.values()) else "failed"
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
    _write_report(capture_root, run_report)
    if status != "ok":
        failed = sorted(name for name, step in steps.items() if step.get("status") != "ok")
        raise RuntimeError(f"offsite backup failed: {','.join(failed)}")
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


def backup_staleness_issues(report_path: Path, audit_at: datetime) -> list[str]:
    """Classify the last offsite run against the latest scheduled slot.

    Returns:
        [] when the report is ok and started at/after the expected slot;
        otherwise one of ["offsite_backup:missing"], ["offsite_backup:failed"],
        ["offsite_backup:stale"]. Unreadable report -> ["offsite_backup:unreadable"].
    """
    path = Path(report_path)
    if not path.exists():
        return ["offsite_backup:missing"]
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
    if started_at < expected_backup_slot(audit_at):
        return ["offsite_backup:stale"]
    return []


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry; logic covered via run_offsite_backup scenarios
    import argparse
    import time as _time

    parser = argparse.ArgumentParser(description="Nightly offsite backup: seal segments then loose copy")
    parser.parse_args(argv)
    project_root = Path.cwd()
    capture_root = _capture_root()
    now = datetime.now(KST)
    started = _time.monotonic()
    try:
        report = run_offsite_backup(project_root, capture_root, now=now)
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
