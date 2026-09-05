"""altdata 패널 일회성 저장소 마이그레이션 (R11, R13)."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src import settings
from src.backfill.altdata.config import _ALTDATA_PANELS
from src.data.parquet_codec import downcast_altdata_panel_frame, write_altdata_panel_parquet

logger = logging.getLogger(__name__)


def _stored_codec(path: Path) -> str:
    try:
        meta = pq.ParquetFile(path).metadata
        if meta.num_row_groups == 0:
            return ""
        return str(meta.row_group(0).column(0).compression or "").upper()
    except Exception:
        return ""


def _verify_altdata_panel(original: pd.DataFrame, downcast: pd.DataFrame) -> str | None:
    if len(downcast) != len(original):
        return f"row count changed: {len(original)} -> {len(downcast)}"
    keys = [c for c in ("date", "symbol") if c in original.columns and c in downcast.columns]
    if keys:
        o = original.sort_values(keys, kind="stable").reset_index(drop=True)
        n = downcast.sort_values(keys, kind="stable").reset_index(drop=True)
    else:
        o, n = original.reset_index(drop=True), downcast.reset_index(drop=True)
    if list(o.columns) != list(n.columns):
        return f"column set changed: {list(o.columns)} -> {list(n.columns)}"
    for col in o.columns:
        ov = o[col]
        nv = n[col]
        if col == "symbol":
            ov = ov.astype(str)
            nv = nv.astype(str)
        both_na = ov.isna() & nv.isna()
        eq = (ov == nv) | both_na
        if not bool(eq.fillna(False).all()):
            return f"column {col!r} drifted after round-trip"
    return None


def migrate_altdata_panel_file(source_path: Path, dry_run: bool = False, backup: bool = True) -> dict[str, Any]:
    """altdata 패널 파일을 zstd+category(symbol)로 마이그레이션한다."""
    source_path = Path(source_path)
    if not source_path.exists():
        return {"status": "missing", "source": str(source_path)}
    original_size = source_path.stat().st_size
    if _stored_codec(source_path) == "ZSTD":
        return {"status": "already_migrated", "original_size_bytes": original_size, "new_size_bytes": original_size}
    original = pd.read_parquet(source_path)
    downcast = downcast_altdata_panel_frame(original)
    failure = _verify_altdata_panel(original, downcast)
    if failure is not None:
        logger.warning("[DATA] file=%s status=verification_failed reason=%s", source_path.name, failure)
        return {"status": "verification_failed", "reason": failure, "original_size_bytes": original_size}
    if dry_run:
        with tempfile.TemporaryDirectory() as tmpdir:
            probe = Path(tmpdir) / "probe.parquet"
            write_altdata_panel_parquet(downcast, probe)
            new_size = probe.stat().st_size
        return {
            "status": "would_migrate",
            "original_size_bytes": original_size,
            "new_size_bytes": new_size,
            "rows": len(downcast),
        }
    if backup:
        bak = source_path.with_suffix(".parquet.bak")
        if not bak.exists():
            bak.write_bytes(source_path.read_bytes())
    write_altdata_panel_parquet(downcast, source_path)
    new_size = source_path.stat().st_size
    return {
        "status": "migrated",
        "original_size_bytes": original_size,
        "new_size_bytes": new_size,
        "rows": len(downcast),
    }


def _print_data_line(path: Path, result: dict[str, Any]) -> None:
    orig_mb = float(result.get("original_size_bytes", 0)) / 1024 / 1024
    new_mb = float(result.get("new_size_bytes", result.get("original_size_bytes", 0))) / 1024 / 1024
    print(  # noqa: T201 - R13 mandates stdout [DATA] summary lines
        f"[DATA] file={path.name} status={result.get('status')} "
        f"orig_mb={orig_mb:.1f} new_mb={new_mb:.1f} rows={result.get('rows', 0)}"
    )


def main() -> None:
    """CLI 진입점 (패널별 순회, 수동 실행)."""
    parser = argparse.ArgumentParser(description="Migrate altdata panels to zstd + category(symbol)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    for meta in _ALTDATA_PANELS.values():
        panel_path = settings.ALTDATA_DIR / str(meta["filename"])
        if not panel_path.exists():
            print(f"[DATA] file={panel_path.name} status=missing orig_mb=0.0 new_mb=0.0 rows=0")  # noqa: T201
            continue
        result = migrate_altdata_panel_file(panel_path, dry_run=args.dry_run, backup=not args.no_backup)
        _print_data_line(panel_path, result)


if __name__ == "__main__":  # pragma: no cover - CLI entry, exercised via `python -m`
    main()
