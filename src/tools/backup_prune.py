"""GDrive 오프사이트 백업 _deleted 스냅샷의 보존기간 정리.

kca-backup 은 덮어쓴 파일을 _deleted/<subtree>/<YYYY-MM-DD>/ 로 옮긴다. rclone
--min-age 는 파일 수정시각 기준이라, 오래전에 수정된 파일은 옮겨진 다음 정리
때 곧바로 삭제되어 보존기간이 지켜지지 않는다. 옮겨진 날짜가 곧 디렉터리
이름이므로 디렉터리 날짜로 보존기간을 판정한다.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.data.intraday_store import _capture_root

logger = logging.getLogger(__name__)

BACKUP_REMOTE_ROOT: str = "gdrive:quant-lake/live/k-closing-alpha/_deleted"
BACKUP_SUBTREES: tuple[str, ...] = ("data", "artifacts")
BACKUP_RETENTION_DAYS: int = 30
LOCAL_INTRADAY_BACKUP_RETENTION_DAYS: int = 3
RCLONE_TIMEOUT_SEC: int = 600
# rclone 문서화된 종료코드: 3 = directory not found (아직 한 번도 옮겨진 파일이 없는 하위 트리)
RCLONE_EXIT_DIRECTORY_NOT_FOUND: int = 3
_DATED_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _resolve_rclone_bin() -> str:
    """Resolve the rclone executable; systemd user PATH does not include ~/.local/bin."""
    return shutil.which("rclone") or str(Path.home() / ".local" / "bin" / "rclone")


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
) -> list[str]:
    """Purge expired dated snapshot directories under every backup subtree.

    Args:
        today: Reference date (KST calendar day).
        run_fn: subprocess.run-compatible runner (tests substitute a fake).
        remote_root: rclone path of the _deleted root.
        retention_days: Days a snapshot directory is kept.

    Returns:
        Remote paths that were purged.

    Raises:
        subprocess.CalledProcessError: Listing failed for a reason other than a
            missing subtree, or a purge failed.
    """
    rclone = _resolve_rclone_bin()
    purged: list[str] = []
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
        for name in expired_snapshot_dirs(names, today, retention_days):
            target = f"{base}/{name}"
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
    parser.parse_args(argv)
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    purged = prune_backups(today=today)
    local_purged = prune_local_intraday_backups(today=today)
    logger.info(
        "[SYS] stage=backup_prune purged=%d targets=%s local_purged=%d local_targets=%s",
        len(purged), purged, len(local_purged), local_purged,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
