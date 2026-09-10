"""인트라데이 레거시 파티션의 정규 스키마 일회성 마이그레이션.

파티션(날짜) 단위 순차 처리로 수행하고 전체 패널을 동시 적재하지 않는다.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src import settings
from src.data.intraday_schema import (
    CANONICAL_BAR_COLUMNS,
    CANONICAL_TICK_COLUMNS,
    extract_vendor_business_dates,
    normalize_bar_frame,
    normalize_tick_frame,
)
from src.data.io_utils import atomic_write_parquet

logger = logging.getLogger(__name__)

LEGACY_KIS_BAR_MARKER: str = "stck_cntg_hour"
LEGACY_LS_BAR_MARKER: str = "jdiff_vol"


def _kind_vendor(kind: str) -> str:
    return "ls" if kind == "legacy_ls" else "kis"


def _dedup_key_cols(is_tick: bool) -> list[str]:
    """중복 판정 키. 틱은 같은 초에 복수 체결이 정상이므로 volume까지 포함한다."""
    return ["symbol", "ts_hms", "volume"] if is_tick else ["symbol", "ts_hms"]


def count_stale_business_date_rows(raw: pd.DataFrame, kind: str, snapshot_date: str) -> int:
    """영업일 게이트가 제거할 것으로 기대되는 행수를 원천에서 독립 산출한다.

    migrate_frame 결과 행수에서 역산하면 항등식이 되어 검증이 무력화되므로,
    정규화 손실과 게이트 제거를 구분하기 위해 반드시 원천에서 따로 센다.
    """
    dates = extract_vendor_business_dates(raw, _kind_vendor(kind))
    if dates is None:
        return 0
    target = str(snapshot_date).replace("-", "")[:8]
    return int((dates.astype(str) != target).sum())


def classify_partition(columns: list[str], *, is_tick: bool) -> str:
    """컬럼 목록으로 파티션 종류를 판별해 canonical/legacy_ls/legacy_kis/unknown을 반환한다."""
    required = list(CANONICAL_TICK_COLUMNS) if is_tick else list(CANONICAL_BAR_COLUMNS)
    if all(c in columns for c in required):
        return "canonical"
    if LEGACY_LS_BAR_MARKER in columns:
        return "legacy_ls"
    if LEGACY_KIS_BAR_MARKER in columns:
        return "legacy_kis"
    return "unknown"


def migrate_frame(raw: pd.DataFrame, kind: str, snapshot_date: str, *, is_tick: bool) -> pd.DataFrame:
    """레거시 프레임을 종목별로 그룹핑해 정규화한 뒤 concat한다."""
    if kind not in ("canonical", "legacy_ls", "legacy_kis"):
        raise ValueError(f"Unknown partition kind: {kind!r}")
    cols = list(CANONICAL_TICK_COLUMNS) if is_tick else list(CANONICAL_BAR_COLUMNS)
    sym_col = "symbol" if "symbol" in raw.columns else ("종목코드" if "종목코드" in raw.columns else None)
    if sym_col is None:
        raise ValueError("Missing symbol column: expected one of ['symbol', '종목코드']")
    vendor = _kind_vendor(kind)
    norm = normalize_tick_frame if is_tick else normalize_bar_frame
    symbol_groups: list[pd.DataFrame] = [raw] if kind == "canonical" else [group for _, group in raw.groupby(sym_col, sort=False)]
    out_parts: list[pd.DataFrame] = []
    for part in symbol_groups:
        code = str(part[sym_col].iloc[0]) if kind != "canonical" else ""
        out_parts.append(part.reindex(columns=cols) if kind == "canonical" else norm(part, vendor, snapshot_date, code))
    return pd.concat(out_parts, ignore_index=True)


def migrate_partition_file(path: Path, *, dry_run: bool = True, backup: bool = True) -> dict[str, Any]:
    """단일 파티션 파일을 정규 스키마로 변환한다."""
    path = Path(path)
    raw = pd.read_parquet(path)
    n_before = len(raw)
    is_tick = "/ticks/" in str(path)
    kind = classify_partition(list(raw.columns), is_tick=is_tick)
    if kind == "canonical":
        return {"path": str(path), "kind": kind, "n_before": n_before, "n_after": n_before, "n_date_dropped": 0, "migrated": False, "reason": "already_canonical"}
    snapshot_date = path.stem
    n_date_dropped = count_stale_business_date_rows(raw, kind, snapshot_date)
    migrated = migrate_frame(raw, kind, snapshot_date, is_tick=is_tick)
    n_after = len(migrated)
    has_dup = bool(migrated.duplicated(subset=_dedup_key_cols(is_tick)).any()) if len(migrated) > 0 else False
    reason: str | None = "duplicate_key" if has_dup else ("row_count_mismatch" if n_after + n_date_dropped != n_before else None)
    if dry_run or reason is not None:
        return {"path": str(path), "kind": kind, "n_before": n_before, "n_after": n_after, "n_date_dropped": n_date_dropped, "migrated": False, "reason": reason}
    if backup:
        shutil.copy(path, Path(str(path) + ".legacy.bak"))
    atomic_write_parquet(migrated, path)
    return {"path": str(path), "kind": kind, "n_before": n_before, "n_after": n_after, "n_date_dropped": n_date_dropped, "migrated": True, "reason": None}


def run_migration(root: Path | None = None, *, dry_run: bool = True, backup: bool = True) -> dict[str, Any]:
    """data/history/intraday 하위 모든 parquet 파티션을 스캔해 마이그레이션을 적용하고 집계한다."""
    base = Path(root) if root is not None else Path(settings.HISTORY_DIR) / "intraday"
    files = sorted(base.rglob("*.parquet"))
    n_files = len(files)
    n_migrated = 0
    n_already_canonical = 0
    n_failed = 0
    n_rows_before = 0
    n_rows_after = 0
    n_date_dropped = 0
    failures: list[dict[str, Any]] = []
    for p in files:
        try:
            res = migrate_partition_file(p, dry_run=dry_run, backup=backup)
        except Exception as e:
            res = {"path": str(p), "kind": "unknown", "n_before": 0, "n_after": 0, "n_date_dropped": 0, "migrated": False, "reason": str(e)}
        n_rows_before += int(res.get("n_before", 0))
        n_rows_after += int(res.get("n_after", 0))
        n_date_dropped += int(res.get("n_date_dropped", 0))
        n_migrated += 1 if res.get("migrated") else 0
        n_already_canonical += 1 if (not res.get("migrated") and res.get("reason") == "already_canonical") else 0
        is_failure = not res.get("migrated") and res.get("reason") != "already_canonical"
        n_failed += 1 if is_failure else 0
        if is_failure:
            failures.append({"path": str(res.get("path", p)), "reason": str(res.get("reason"))})
        logger.info("[DATA] stage=migration file=%s kind=%s migrated=%s reason=%s", p, res.get("kind"), res.get("migrated"), res.get("reason"))
    return {"n_files": n_files, "n_migrated": n_migrated, "n_already_canonical": n_already_canonical, "n_failed": n_failed, "n_rows_before": n_rows_before, "n_rows_after": n_rows_after, "n_date_dropped": n_date_dropped, "failures": failures}


def main() -> None:  # pragma: no cover - CLI entry, manual one-shot migration via `python -m`
    """CLI 진입점. --apply(기본 dry-run), --no-backup, --root 인자를 파싱해 run_migration을 실행한다."""
    parser = argparse.ArgumentParser(description="Migrate legacy intraday partitions to the canonical schema")
    parser.add_argument("--apply", action="store_true", help="Write migrated partitions (default is dry-run)")
    parser.add_argument("--no-backup", action="store_true", help="Skip .legacy.bak backup before overwrite")
    parser.add_argument("--root", default=None, help="Intraday history root to scan (default: settings HISTORY_DIR/intraday)")
    args = parser.parse_args()
    result = run_migration(root=args.root, dry_run=not args.apply, backup=not args.no_backup)
    logger.info("[DATA] stage=migration_summary result=%s", result)


if __name__ == "__main__":  # pragma: no cover - CLI entry, exercised via `python -m`
    main()
