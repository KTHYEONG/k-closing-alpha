"""Immutable weekly/monthly snapshots of rebuild-irreplaceable core panels."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import pyarrow.parquet as pq

from src.config.base import RANK_POOL_PARQUET_NAME, TOPK_DECISIONS_PARQUET_NAME
from src.tools.offsite_common import (
    DATED_DIR_RE as _DATED_DIR_RE,
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

logger = logging.getLogger(__name__)

CORE_SNAPSHOT_REMOTE_ROOT: str = OFFSITE_REMOTE_BASE + "/snapshots"
CORE_SNAPSHOT_KEEP_WEEKLY: int = 8
CORE_SNAPSHOT_KEEP_MONTHLY: int = 12
CORE_ROW_SHRINK_TOLERANCE: float = 0.0

RCLONE_TIMEOUT_SEC: int = 600

FIXED_CORE_REL_PATHS: tuple[str, ...] = (
    "data/history/price_history.parquet",
    "data/history/archive.parquet",
    "data/parquet/" + TOPK_DECISIONS_PARQUET_NAME,
    "data/parquet/" + RANK_POOL_PARQUET_NAME,
)
PAPER_GLOB_DIR: str = "data/paper"
BUNDLE_DIR: str = "artifacts/models/topk_ranker"

_DATE_COLUMNS: tuple[str, ...] = ("date", "decision_date", "trading_date")


@dataclass(frozen=True)
class CorePanelStat:
    """Integrity fingerprint of one core file.

    Attributes:
        relpath: Path relative to the project root.
        sha256: Content hash.
        bytes: File size.
        rows: Parquet row count, or None for non-parquet files.
        max_date: Max of the date column for dated panels ("" otherwise).
    """

    relpath: str
    sha256: str
    bytes: int
    rows: int | None
    max_date: str


def core_panel_paths(project_root: Path) -> tuple[Path, ...]:
    """Files whose loss or corruption cannot be rebuilt from vendor APIs alone.

    price_history.parquet, the decision archive (HISTORY_PARQUET_PATH),
    paper ledgers (PAPER_DIR/*.parquet), topk_decisions and
    rank_pool_predictions (PARQUET_DIR), and the live bundle directory with its
    retrain registry.
    """
    root = Path(project_root)
    paths: list[Path] = [root / rel for rel in FIXED_CORE_REL_PATHS]
    paper_dir = root / PAPER_GLOB_DIR
    if paper_dir.is_dir() and not paper_dir.is_symlink():
        for child in sorted(paper_dir.glob("*.parquet")):
            if child.is_symlink() or not child.is_file():
                continue
            if child not in paths:
                paths.append(child)
    bundle_dir = root / BUNDLE_DIR
    if bundle_dir.is_dir() and not bundle_dir.is_symlink():
        for child in sorted(bundle_dir.rglob("*")):
            if child.is_symlink() or not child.is_file():
                continue
            if child not in paths:
                paths.append(child)
    return tuple(paths)


def _stat_one(abs_path: Path, rel: str) -> CorePanelStat | None:
    if not abs_path.exists() or abs_path.is_symlink() or not abs_path.is_file():
        return None
    try:
        size = abs_path.stat().st_size
        sha = _sha256_file(abs_path)
    except OSError:  # pragma: no cover - transient stat race
        return CorePanelStat(relpath=rel, sha256="", bytes=0, rows=None, max_date="")
    if abs_path.suffix != ".parquet":
        return CorePanelStat(relpath=rel, sha256=sha, bytes=size, rows=None, max_date="")
    try:
        rows = int(pq.ParquetFile(str(abs_path)).metadata.num_rows)
    except Exception:
        return CorePanelStat(relpath=rel, sha256="", bytes=size, rows=None, max_date="")
    max_date = ""
    for column in _DATE_COLUMNS:
        try:
            import pandas as pd

            values = pd.read_parquet(abs_path, columns=[column])[column]
            if len(values) == 0:
                max_date = ""
                break
            peak = pd.Timestamp(values.max()).date().isoformat()
            max_date = str(peak)
            break
        except Exception as exc:  # noqa: BLE001 - try next candidate date column
            logger.debug("[SYS] stage=core_snapshot date_column=%s skipped: %s", column, exc)
            continue
    return CorePanelStat(relpath=rel, sha256=sha, bytes=size, rows=rows, max_date=max_date)


def collect_core_stats(project_root: Path) -> list[CorePanelStat]:
    """Stat every core panel; missing files are omitted for missing detection."""
    root = Path(project_root)
    stats: list[CorePanelStat] = []
    for abs_path in core_panel_paths(root):
        try:
            rel = abs_path.relative_to(root).as_posix()
        except ValueError:  # pragma: no cover - defensive absolute-path guard
            continue
        entry = _stat_one(abs_path, rel)
        if entry is None:
            continue
        stats.append(entry)
    return stats


def validate_core_panels(
    stats: Sequence[CorePanelStat], previous: Sequence[CorePanelStat]
) -> list[str]:
    """Detect corruption signatures before a copy can overwrite a good remote version.

    Append-only/growing panels must stay readable, must not lose rows beyond
    CORE_ROW_SHRINK_TOLERANCE and must not move max_date backwards relative to
    the previous verified snapshot manifest.

    Returns:
        Issue strings `core_panel:<relpath>:<reason>` (unreadable, rows_shrank,
        max_date_regressed, missing); empty when valid.
    """
    current = {entry.relpath: entry for entry in stats}
    prev = {entry.relpath: entry for entry in previous}
    issues: list[str] = [f"core_panel:{relpath}:missing" for relpath in sorted(set(prev) - set(current))]
    for relpath in sorted(set(current)):
        entry = current[relpath]
        if not entry.sha256:
            issues.append(f"core_panel:{relpath}:unreadable")
            continue
        old = prev.get(relpath)
        if old is None:
            continue
        if (
            entry.rows is not None
            and old.rows is not None
            and float(entry.rows) < float(old.rows) - CORE_ROW_SHRINK_TOLERANCE
        ):
            issues.append(f"core_panel:{relpath}:rows_shrank")
        if entry.max_date and old.max_date and entry.max_date < old.max_date:
            issues.append(f"core_panel:{relpath}:max_date_regressed")
    return sorted(issues)


def _list_remote_snapshot_dirs(rclone: str, run_fn: Callable[..., subprocess.CompletedProcess[str]]) -> list[str]:
    listing = run_fn(
        [rclone, "lsf", "--dirs-only", CORE_SNAPSHOT_REMOTE_ROOT],
        capture_output=True,
        text=True,
        timeout=RCLONE_TIMEOUT_SEC,
        check=False,
    )
    if listing.returncode != 0:
        if "directory not found" in (listing.stderr or "").lower() or listing.returncode == 3:
            return []
        raise subprocess.CalledProcessError(listing.returncode, listing.args, listing.stdout, listing.stderr)
    names = [line.strip().rstrip("/") for line in listing.stdout.splitlines() if line.strip()]
    return sorted(name for name in names if _DATED_DIR_RE.match(name))


def select_snapshots_to_prune(snapshot_names: Sequence[str]) -> list[str]:
    """Apply keep-last-N weekly plus first-of-month retention."""
    names = sorted({name for name in snapshot_names if _DATED_DIR_RE.match(name)})
    if len(names) < CORE_SNAPSHOT_KEEP_WEEKLY:
        return []
    keep: set[str] = set(names[-CORE_SNAPSHOT_KEEP_WEEKLY:])
    months = sorted({name[:7] for name in names}, reverse=True)[:CORE_SNAPSHOT_KEEP_MONTHLY]
    for month in months:
        first = min(name for name in names if name.startswith(month))
        keep.add(first)
    return sorted(name for name in names if name not in keep)


def _read_remote_manifest(
    rclone: str, snapshot: str, run_fn: Callable[..., subprocess.CompletedProcess[str]]
) -> list[CorePanelStat]:
    remote = f"{CORE_SNAPSHOT_REMOTE_ROOT}/{snapshot}/manifest.json"
    result = run_fn(
        [rclone, "cat", remote],
        capture_output=True,
        text=True,
        timeout=RCLONE_TIMEOUT_SEC,
        check=False,
    )
    if result.returncode != 0:
        return []
    try:
        raw: object = json.loads(result.stdout)
    except ValueError:
        return []
    if not isinstance(raw, list):
        return []
    previous: list[CorePanelStat] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            previous.append(
                CorePanelStat(
                    relpath=str(item["relpath"]),
                    sha256=str(item["sha256"]),
                    bytes=int(item["bytes"]),
                    rows=None if item.get("rows") is None else int(item["rows"]),
                    max_date=str(item.get("max_date", "")),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return previous


def run_core_snapshot(
    project_root: Path,
    *,
    today: date,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    """Copy validated core files to an immutable dated snapshot and apply retention.

    Snapshots live outside `_deleted`, are written with `rclone copy --immutable`
    under snapshots/<YYYY-MM-DD>/ with a manifest.json of CorePanelStat, and are
    pruned only by keep-last-N weekly + first-snapshot-of-month rules, never by age.

    Returns:
        Remote snapshot paths pruned by retention.

    Raises:
        RuntimeError: Validation issues (nothing is uploaded) or an rclone failure.
    """
    root = Path(project_root)
    rclone = _resolve_rclone_bin()
    stats = collect_core_stats(root)
    names = _list_remote_snapshot_dirs(rclone, run_fn)
    previous: list[CorePanelStat] = []
    if names:
        previous = _read_remote_manifest(rclone, names[-1], run_fn)
    issues = validate_core_panels(stats, previous)
    if issues:
        raise RuntimeError(f"core snapshot validation failed: {'; '.join(issues)}")
    day = today.isoformat()
    with tempfile.TemporaryDirectory(prefix="core-snapshot-") as staging_name:
        staging = Path(staging_name)
        for entry in stats:
            src = root / entry.relpath
            dest = staging / entry.relpath
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
        manifest = staging / "manifest.json"
        manifest.write_text(
            json.dumps([asdict(entry) for entry in sorted(stats, key=lambda e: e.relpath)], sort_keys=True),
            encoding="utf-8",
        )
        dest_remote = f"{CORE_SNAPSHOT_REMOTE_ROOT}/{day}/"
        uploaded = run_fn(
            [rclone, "copy", str(staging), dest_remote, "--immutable"],
            capture_output=True,
            text=True,
            timeout=RCLONE_TIMEOUT_SEC,
            check=False,
        )
        if uploaded.returncode != 0:
            raise RuntimeError(f"core snapshot upload failed: {(uploaded.stderr or '').strip()}")
    all_names = sorted(set(names) | {day})
    pruned: list[str] = []
    for name in select_snapshots_to_prune(all_names):
        target = f"{CORE_SNAPSHOT_REMOTE_ROOT}/{name}"
        purged = run_fn(
            [rclone, "purge", target],
            capture_output=True,
            text=True,
            timeout=RCLONE_TIMEOUT_SEC,
            check=False,
        )
        if purged.returncode != 0:
            raise RuntimeError(f"core snapshot prune failed: {target}")
        pruned.append(target)
    logger.info("[SYS] stage=core_snapshot date=%s files=%d pruned=%d", day, len(stats), len(pruned))
    return pruned


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry; logic covered via run_core_snapshot scenarios
    """CLI entry for the weekly core snapshot systemd unit."""
    import argparse
    import time as _time
    from zoneinfo import ZoneInfo

    parser = argparse.ArgumentParser(description="Immutable snapshot of rebuild-irreplaceable core panels")
    parser.parse_args(argv)
    project_root = Path.cwd()
    today = date.today().isoformat()
    import datetime as _dt

    kst = _dt.datetime.now(ZoneInfo("Asia/Seoul")).date()
    started = _time.monotonic()
    try:
        pruned = run_core_snapshot(project_root, today=kst)
    except RuntimeError as exc:
        logger.info("[SYS] stage=core_snapshot date=%s status=failed duration_s=%.0f error=%s", today, _time.monotonic() - started, exc)
        raise SystemExit(1) from exc
    logger.info("[SYS] stage=core_snapshot date=%s status=ok duration_s=%.0f pruned=%s", today, _time.monotonic() - started, pruned)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
