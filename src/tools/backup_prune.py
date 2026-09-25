"""GDrive 오프사이트 백업 _deleted 스냅샷의 보존기간 정리.

kca-backup 은 덮어쓴 파일을 _deleted/<subtree>/<YYYY-MM-DD>/ 로 옮긴다. rclone
--min-age 는 파일 수정시각 기준이라, 오래전에 수정된 파일은 옮겨진 다음 정리
때 곧바로 삭제되어 보존기간이 지켜지지 않는다. 옮겨진 날짜가 곧 디렉터리
이름이므로 디렉터리 날짜로 보존기간을 판정한다.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.data.capture_store import resolve_capture_root as _capture_root
from src.tools.capture_offsite import prune_local_sealed_capture
from src.tools.offsite_common import DATED_DIR_RE, OFFSITE_REMOTE_BASE, resolve_rclone_bin
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

BACKUP_REMOTE_ROOT: str = OFFSITE_REMOTE_BASE + "/_deleted"
BACKUP_SUBTREES: tuple[str, ...] = ("data", "artifacts")
BACKUP_RETENTION_DAYS: int = 30
BACKUP_MAX_PURGE_DIRS_PER_SUBTREE: int = 7
LOCAL_INTRADAY_BACKUP_RETENTION_DAYS: int = 3
RCLONE_TIMEOUT_SEC: int = 600
# rclone 문서화된 종료코드: 3 = directory not found (아직 한 번도 옮겨진 파일이 없는 하위 트리)
RCLONE_EXIT_DIRECTORY_NOT_FOUND: int = 3
_DATED_DIR = DATED_DIR_RE


_resolve_rclone_bin = resolve_rclone_bin


def expired_snapshot_dirs(
    dir_names: list[str], today: pd.Timestamp, retention_days: int = BACKUP_RETENTION_DAYS
) -> list[str]:
    """Return dated snapshot directory names older than the retention window.

    Args:
        dir_names: Directory names directly under one _deleted subtree.
        today: Reference date (KST calendar day).
        retention_days: Days a snapshot directory is kept after its date.

    Returns:
        Sorted YYYY-MM-DD names strictly older than today - retention_days;
        non-date names and impossible dates are ignored.
    """
    cutoff = pd.Timestamp(today).normalize() - pd.Timedelta(days=int(retention_days))
    expired = []
    for name in dir_names:
        if not _DATED_DIR.match(name):
            continue
        snapshot_day = pd.to_datetime(name, format="%Y-%m-%d", errors="coerce")
        if pd.notna(snapshot_day) and snapshot_day < cutoff:
            expired.append(name)
    return sorted(expired)


def prune_backups(
    *,
    today: pd.Timestamp,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    remote_root: str = BACKUP_REMOTE_ROOT,
    retention_days: int = BACKUP_RETENTION_DAYS,
    dry_run: bool = False,
    max_purge_per_subtree: int = BACKUP_MAX_PURGE_DIRS_PER_SUBTREE,
) -> list[str]:
    """Purge expired dated snapshot directories under every backup subtree.

    Args:
        today: Reference date (KST calendar day).
        run_fn: subprocess.run-compatible runner (tests substitute a fake).
        remote_root: rclone path of the _deleted root.
        retention_days: Days a snapshot directory is kept.
        dry_run: List targets without purging.
        max_purge_per_subtree: Per-subtree purge cap.

    Returns:
        Remote paths that were (or, for dry_run, would be) purged.

    Raises:
        RuntimeError: One subtree's expired set exceeds the purge cap.
        subprocess.CalledProcessError: Listing failed for a reason other than a
            missing subtree, or a purge failed.
    """
    rclone = _resolve_rclone_bin()
    expired_by_subtree: dict[str, list[str]] = {}
    for subtree in BACKUP_SUBTREES:
        base = f"{remote_root}/{subtree}"
        listing = run_fn(
            [rclone, "lsf", "--dirs-only", base], capture_output=True, text=True, timeout=RCLONE_TIMEOUT_SEC, check=False
        )
        if listing.returncode == RCLONE_EXIT_DIRECTORY_NOT_FOUND:
            continue
        if listing.returncode != 0:
            raise subprocess.CalledProcessError(listing.returncode, listing.args, listing.stdout, listing.stderr)
        names = [line.strip().rstrip("/") for line in listing.stdout.splitlines() if line.strip()]
        expired_by_subtree[subtree] = expired_snapshot_dirs(names, today, retention_days)
    for subtree, expired in expired_by_subtree.items():
        # 정상 운영에서는 평일마다 하루치(연휴 직후 최대 수일치)만 만료되므로 상한 초과는 시계/파싱 이상 신호다.
        if len(expired) > max_purge_per_subtree:
            raise RuntimeError(f"backup prune cap exceeded in {subtree}: {len(expired)} expired dirs")
    targets = [f"{remote_root}/{subtree}/{name}" for subtree, expired in expired_by_subtree.items() for name in expired]
    if dry_run:
        return sorted(targets)
    purged: list[str] = []
    for target in sorted(targets):
        run_fn([rclone, "purge", target], capture_output=True, text=True, timeout=RCLONE_TIMEOUT_SEC, check=True)
        purged.append(target)
    return purged


def prune_local_intraday_backups(
    *,
    today: pd.Timestamp,
    backups_root: Path | None = None,
    retention_days: int = LOCAL_INTRADAY_BACKUP_RETENTION_DAYS,
) -> list[str]:
    """intraday 파티션 교체-직전 로컬 스냅샷 중 보존기간이 지난 것을 삭제한다.

    src.data.intraday_store._retain_backup_ref가 파티션을 교체할 때마다
    <backups_root>/<session>/<snapshot_date>/ 아래 하드링크 스냅샷을 남기지만
    회수 로직이 없어 무한 누적된다(실측 3GB/일). daily_audit가 당일 저녁
    이상을 감지하므로 retention_days 경과 후엔 복구 목적의 가치가 없다.

    Args:
        today: 기준일(KST, tz 미보유 자정 정규화 Timestamp) — expired_snapshot_dirs와
            동일한 규약.
        backups_root: <capture_root>/backups/intraday 루트. None이면
            src.data.intraday_store._capture_root() 기준으로 해석한다(테스트 주입용).
        retention_days: 보존일수.

    Returns:
        삭제된 `<session>/<date>` 디렉터리 경로 문자열 목록(정렬됨). 루트가
        아직 존재하지 않으면(백업이 한 번도 생성된 적 없음) 빈 리스트.

    Raises:
        OSError: 디렉터리 삭제 실패(권한 등) — 부분 삭제 상태를 침묵하지 않는다.
    """
    root = backups_root if backups_root is not None else _capture_root() / "backups" / "intraday"
    if not root.exists():
        return []
    purged: list[str] = []
    for session_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        names = [p.name for p in session_dir.iterdir() if p.is_dir()]
        for name in expired_snapshot_dirs(names, today, retention_days):
            shutil.rmtree(session_dir / name)
            purged.append(f"{session_dir.name}/{name}")
    return sorted(purged)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry; logic covered via prune_backups scenarios
    parser = argparse.ArgumentParser(description="Purge _deleted backup snapshots older than the retention window")
    parser.add_argument("--dry-run", action="store_true", help="list purge targets without deleting")
    args = parser.parse_args(argv)
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    purged = prune_backups(today=today, dry_run=True) if args.dry_run else prune_backups(today=today)
    local_purged = prune_local_intraday_backups(today=today)
    sealed_report = prune_local_sealed_capture(_capture_root(), today=today.date())
    logger.info(
        "[SYS] stage=backup_prune purged=%d targets=%s local_purged=%d local_targets=%s sealed_removed=%d sealed_bytes=%d sealed_kept=%d",
        len(purged), purged, len(local_purged), local_purged,
        len(sealed_report.removed), sealed_report.bytes_removed, len(sealed_report.kept),
    )
    logger.debug("[SYS] stage=backup_prune sealed_kept=%s", sealed_report.kept)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
