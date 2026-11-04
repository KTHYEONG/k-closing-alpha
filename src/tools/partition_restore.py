"""Restore evicted history partitions from the offsite copy and drop their tombstones.

Run under the same flock as the Drive writers when invoked by an operator; this is documented, not enforced.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

from src.data.partition_eviction import eviction_families, read_tombstone, tombstone_path
from src.tools.offsite_common import OFFSITE_REMOTE_BASE, md5_file, resolve_rclone_bin
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

RCLONE_TIMEOUT_SEC: int = 600
RESTORE_TMP_SUFFIX: str = ".restore.tmp"
HISTORY_REMOTE_SUBPATH: str = "data/history"


def restore_partitions(
    *,
    history_dir: Path,
    relpaths: Sequence[str],
    remote_base: str,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    dry_run: bool = False,
) -> tuple[str, ...]:
    """Restore each tombstoned partition, verifying MD5 and size against its tombstone.

    Returns:
        Relative paths restored (would-be restored on dry_run).

    Raises:
        RuntimeError: One or more partitions failed verification or download, raised after all were processed.
    """
    rclone = resolve_rclone_bin()
    restored: list[str] = []
    failed: list[str] = []
    for rel in relpaths:
        partition = history_dir / rel
        try:
            tomb = read_tombstone(partition)
        except ValueError:
            logger.error("[SYS] stage=partition_restore path=%s status=malformed_tombstone", rel)
            failed.append(rel)
            continue
        if tomb is None or partition.exists():
            logger.error("[SYS] stage=partition_restore path=%s status=not_restorable", rel)
            failed.append(rel)
            continue
        if dry_run:
            restored.append(rel)
            continue
        tmp = partition.with_name(partition.name + RESTORE_TMP_SUFFIX)
        try:
            result = run_fn(
                [rclone, "copyto", f"{remote_base}/{tomb.partition}", str(tmp)],
                capture_output=True, text=True, timeout=RCLONE_TIMEOUT_SEC, check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(f"rclone copyto exit {result.returncode}")
            if not tmp.is_file() or tmp.stat().st_size != tomb.size or md5_file(tmp) != tomb.md5:
                raise RuntimeError("size or md5 mismatch")
            os.replace(tmp, partition)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            logger.error("[SYS] stage=partition_restore path=%s status=failed error=%s", rel, type(exc).__name__)
            failed.append(rel)
            continue
        finally:
            tmp.unlink(missing_ok=True)
        tombstone_path(partition).unlink()
        restored.append(rel)
        logger.info("[SYS] stage=partition_restore path=%s status=restored", rel)
    if failed:
        raise RuntimeError(f"Partition restore failed for {failed}")
    return tuple(restored)


def _paths_for_family(history_dir: Path, family: str, start: date, end: date) -> list[str]:
    from src.config.collection import CollectionSettings

    selected = next(item for item in eviction_families(CollectionSettings()) if item.name == family)
    found: list[str] = []
    for relroot in selected.relroots:
        root = history_dir / relroot
        if not root.is_dir():
            continue
        for tomb_file in sorted(root.rglob("*.parquet.evicted.json")):
            stem = tomb_file.name.removesuffix(".parquet.evicted.json")
            try:
                day = date.fromisoformat(stem)
            except ValueError:
                continue
            if start <= day <= end:
                found.append(tomb_file.relative_to(history_dir).as_posix().removesuffix(".evicted.json"))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Restore evicted history partitions from offsite. Run under the same flock as the Drive writers "
            "(not enforced here)."
        )
    )
    parser.add_argument("--family", choices=("bars", "ticks", "sameday"))
    parser.add_argument("--from", dest="start", type=date.fromisoformat)
    parser.add_argument("--to", dest="end", type=date.fromisoformat)
    parser.add_argument("--path", action="append", default=[], help="partition path relative to HISTORY_DIR (repeatable)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from src import settings

    history_dir = Path(settings.HISTORY_DIR)
    relpaths = list(args.path)
    if args.family:
        if args.start is None or args.end is None:
            parser.error("--family requires --from and --to")
        relpaths += _paths_for_family(history_dir, args.family, args.start, args.end)
    if not relpaths:
        parser.error("nothing selected: pass --path or --family with --from/--to")
    try:
        done = restore_partitions(
            history_dir=history_dir, relpaths=relpaths,
            remote_base=f"{OFFSITE_REMOTE_BASE}/{HISTORY_REMOTE_SUBPATH}", run_fn=subprocess.run, dry_run=args.dry_run,
        )
    except RuntimeError:
        logger.exception("[SYS] stage=partition_restore status=failed")
        return 1
    logger.info("[SYS] stage=partition_restore restored=%d dry_run=%s", len(done), args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    raise SystemExit(main())
