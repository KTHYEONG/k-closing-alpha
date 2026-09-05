"""price_history.parquet 일회성 저장소 마이그레이션 (R10, R13)."""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src import settings
from src.data.parquet_codec import (
    PRICE_HISTORY_FLOAT32_COLUMNS,
    PRICE_HISTORY_FLOAT64_RETAIN_COLUMNS,
    PRICE_HISTORY_NULLABLE_INT32_COLUMNS,
    downcast_price_history_frame,
    write_price_history_parquet,
)

logger = logging.getLogger(__name__)


def _stored_codec(path: Path) -> str:
    try:
        meta = pq.ParquetFile(path).metadata
        if meta.num_row_groups == 0:
            return ""
        return str(meta.row_group(0).column(0).compression or "").upper()
    except Exception:
        return ""


def _frames_aligned(original: pd.DataFrame, downcast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = [c for c in ("date", "symbol") if c in original.columns and c in downcast.columns]
    if keys:
        o = original.sort_values(keys, kind="stable").reset_index(drop=True)
        n = downcast.sort_values(keys, kind="stable").reset_index(drop=True)
        return o, n
    return original.reset_index(drop=True), downcast.reset_index(drop=True)


def _verify_price_history(original: pd.DataFrame, downcast: pd.DataFrame) -> str | None:
    if len(downcast) != len(original):
        return f"row count changed: {len(original)} -> {len(downcast)}"
    o, n = _frames_aligned(original, downcast)
    for col in PRICE_HISTORY_NULLABLE_INT32_COLUMNS:
        if col not in o.columns or col not in n.columns:
            continue
        orig_r = pd.to_numeric(o[col], errors="coerce").round()
        new_f = pd.to_numeric(n[col].astype("float64"), errors="coerce")
        both_na = orig_r.isna() & new_f.isna()
        eq = (orig_r == new_f) | both_na
        if not bool(eq.fillna(False).all()):
            return f"Int32 column {col!r} drifted after round-trip"
    for col in PRICE_HISTORY_FLOAT32_COLUMNS:
        if col not in o.columns or col not in n.columns:
            continue
        ov = pd.to_numeric(o[col], errors="coerce").astype("float64")
        nv = pd.to_numeric(n[col], errors="coerce").astype("float64")
        if bool((ov.isna() != nv.isna()).any()):
            return f"float32 column {col!r} NaN mask changed"
        both_na = ov.isna() & nv.isna()
        denom = ov.abs().where(ov.abs() > 1e-9, 1e-9)
        err = ((ov - nv).abs() / denom).where(~both_na, 0.0)
        if bool((err > 1e-4).any()):
            return f"float32 column {col!r} relative error exceeds 1e-4"
    for col in PRICE_HISTORY_FLOAT64_RETAIN_COLUMNS:
        if col not in o.columns or col not in n.columns:
            continue
        ov = o[col]
        nv = n[col]
        both_na = ov.isna() & nv.isna()
        eq = (ov == nv) | both_na
        if not bool(eq.fillna(False).all()):
            return f"retain column {col!r} was modified"
    return None


def migrate_price_history_file(source_path: Path, dry_run: bool = False, backup: bool = True) -> dict[str, Any]:
    """price_history 파일을 다운캐스트+zstd로 마이그레이션한다."""
    source_path = Path(source_path)
    if not source_path.exists():
        return {"status": "missing", "source": str(source_path)}
    original_size = source_path.stat().st_size
    if _stored_codec(source_path) == "ZSTD":
        return {"status": "already_migrated", "original_size_bytes": original_size, "new_size_bytes": original_size}
    original = pd.read_parquet(source_path)
    downcast = downcast_price_history_frame(original)
    failure = _verify_price_history(original, downcast)
    if failure is not None:
        logger.warning("[DATA] file=%s status=verification_failed reason=%s", source_path.name, failure)
        return {"status": "verification_failed", "reason": failure, "original_size_bytes": original_size}
    if dry_run:
        with tempfile.TemporaryDirectory() as tmpdir:
            probe = Path(tmpdir) / "probe.parquet"
            write_price_history_parquet(downcast, probe)
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
    write_price_history_parquet(downcast, source_path)
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
    """CLI 진입점 (단일 호출, 수동 실행)."""
    parser = argparse.ArgumentParser(description="Migrate price_history.parquet to zstd + downcast dtypes")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    target = settings.PRICE_HISTORY_PARQUET_PATH
    result = migrate_price_history_file(target, dry_run=args.dry_run, backup=not args.no_backup)
    _print_data_line(target, result)


if __name__ == "__main__":  # pragma: no cover - CLI entry, exercised via `python -m`
    main()
