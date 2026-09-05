"""intraday 코퍼스 일회성 정준 스키마 마이그레이션 (R12, R13)."""

from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src import settings
from src.config.market_session import INTRADAY_SESSION_NXT_AFTERMARKET, INTRADAY_SESSION_REGULAR
from src.data import intraday_store
from src.data.intraday_schema import normalize_bar_frame, normalize_tick_frame

logger = logging.getLogger(__name__)

_LS_BAR_SIGNATURE: frozenset[str] = frozenset({"date", "time", "open", "high", "low", "close", "jdiff_vol", "value"})
_LS_TICK_SIGNATURE: frozenset[str] = frozenset({"time", "close", "jdiff_vol"})
_KNOWN_SESSIONS: frozenset[str] = frozenset({INTRADAY_SESSION_REGULAR, INTRADAY_SESSION_NXT_AFTERMARKET})


def detect_intraday_vendor(df: pd.DataFrame) -> str:
    """컬럼 시그니처로 벤더를 판별한다 (추측 금지)."""
    cols = set(df.columns)
    if _LS_BAR_SIGNATURE.issubset(cols):
        return "ls"
    if "stck_cntg_hour" in cols:
        return "kis"
    if _LS_TICK_SIGNATURE.issubset(cols):
        return "ls"
    return _raise_unknown_vendor(cols)


def _raise_unknown_vendor(cols: set[str]) -> str:
    raise ValueError(f"Unknown intraday vendor column set: {sorted(cols)}")


def _stored_codec(path: Path) -> str:
    try:
        meta = pq.ParquetFile(path).metadata
        if meta.num_row_groups == 0:
            return ""
        return str(meta.row_group(0).column(0).compression or "").upper()
    except Exception:
        return ""


def _infer_bar_interval_minutes(source_path: Path) -> int:
    for part in source_path.parts:
        match = re.fullmatch(r"(\d+)m", part)
        if match:
            return int(match.group(1))
    return 1


def _infer_session(source_path: Path) -> str:
    """경로의 {session} 세그먼트(regular/nxt_aftermarket)를 추출한다.

    경로에 알려진 세션 디렉토리가 있으면 반드시 그것을 사용한다 (하드코딩 금지 --
    과거 버그: nxt_aftermarket 파일도 무조건 'regular'로 써서 원본은 그대로 두고
    같은 날짜의 regular 파티션을 오염시켰다). 세션 세그먼트 자체가 없는 최소 경로
    (플랫 테스트 픽스처 등)만 'regular' 기본값으로 관대하게 처리한다.
    """
    parts = source_path.parts
    for i, part in enumerate(parts):
        is_session_root = re.fullmatch(r"\d+m", part) or part == "ticks"
        if is_session_root and i + 1 < len(parts) and parts[i + 1] in _KNOWN_SESSIONS:
            return parts[i + 1]
    return INTRADAY_SESSION_REGULAR


def _fail(reason: str, source_path: Path) -> dict[str, Any]:
    logger.warning("[DATA] file=%s status=verification_failed reason=%s", source_path.name, reason)
    return {
        "status": "verification_failed",
        "reason": reason,
        "original_size_bytes": source_path.stat().st_size if source_path.exists() else 0,
    }


def migrate_intraday_partition_file(source_path: Path, kind: str, dry_run: bool = False, backup: bool = True) -> dict[str, Any]:
    """레거시 intraday 파티션 파일을 정준 스키마로 마이그레이션한다."""
    source_path = Path(source_path)
    if kind not in ("bar", "tick"):
        raise ValueError(f"Unknown intraday partition kind: {kind!r} (expected 'bar' or 'tick')")
    if not source_path.exists():
        return {"status": "missing", "source": str(source_path)}
    original_size = source_path.stat().st_size
    if _stored_codec(source_path) == "ZSTD":
        return {"status": "already_migrated", "original_size_bytes": original_size, "new_size_bytes": original_size}
    df = pd.read_parquet(source_path)
    if df is None or df.empty:
        return _fail("empty source frame", source_path)
    if "종목코드" not in df.columns or "스냅샷_날짜" not in df.columns:
        return _fail("missing partition keys (종목코드/스냅샷_날짜)", source_path)
    snapshot_dates = df["스냅샷_날짜"].astype(str).unique().tolist()
    if len(snapshot_dates) != 1:
        return _fail(f"file spans multiple snapshot dates: {sorted(snapshot_dates)}", source_path)
    snapshot_date = str(snapshot_dates[0])
    frames: list[pd.DataFrame] = []
    audit: dict[str, dict[str, int]] = {}
    try:
        for symbol, group in df.groupby("종목코드", sort=True):
            code = str(symbol)
            vendor = detect_intraday_vendor(group)
            if kind == "bar":
                norm = normalize_bar_frame(group, vendor, snapshot_date, code)
            else:
                norm = normalize_tick_frame(group, vendor, snapshot_date, code, truncated=False)
            frames.append(norm)
            if kind == "tick":
                audit[code] = {"earliest_ts_hms": int(norm["ts_hms"].min())}
    except ValueError as exc:
        return _fail(str(exc), source_path)
    session = _infer_session(source_path)
    merged = pd.concat(frames, ignore_index=True)
    if dry_run:
        return {
            "status": "would_migrate",
            "original_size_bytes": original_size,
            "new_size_bytes": original_size,
            "rows": len(merged),
        }
    if kind == "bar":
        interval = _infer_bar_interval_minutes(source_path)
        target = intraday_store.intraday_partition_path(interval, snapshot_date, session)
    else:
        interval = 1
        target = intraday_store.tick_partition_path(snapshot_date, session)
    if backup:
        bak = source_path.with_suffix(".parquet.bak")
        if not bak.exists():
            bak.write_bytes(source_path.read_bytes())
    if target.resolve() == source_path.resolve():
        # In-place migration: drop the legacy payload (a backup was taken
        # above unless --no-backup) so the merge-safe writer below starts
        # from canonical rows only instead of re-ingesting legacy columns.
        target.unlink()
    if kind == "bar":
        intraday_store.write_intraday_partition(merged, interval, snapshot_date, session)
    else:
        intraday_store.write_tick_partition(merged, snapshot_date, session)
        audit_path = source_path.with_suffix(source_path.suffix + ".truncation_audit.json")
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    new_size = target.stat().st_size if target.exists() else original_size
    return {
        "status": "migrated",
        "original_size_bytes": original_size,
        "new_size_bytes": new_size,
        "rows": len(merged),
    }


def _print_data_line(path: Path, result: dict[str, Any]) -> None:
    orig_mb = float(result.get("original_size_bytes", 0)) / 1024 / 1024
    new_mb = float(result.get("new_size_bytes", result.get("original_size_bytes", 0))) / 1024 / 1024
    print(  # noqa: T201 - R13 mandates stdout [DATA] summary lines
        f"[DATA] file={path.name} status={result.get('status')} "
        f"orig_mb={orig_mb:.1f} new_mb={new_mb:.1f} rows={result.get('rows', 0)}"
    )


def main() -> None:
    """CLI 진입점 (코퍼스 순회 + 매니페스트 기록, 수동 실행)."""
    parser = argparse.ArgumentParser(description="Migrate legacy intraday corpus to canonical schema")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    base = settings.HISTORY_DIR / "intraday"
    targets: list[tuple[Path, str]] = []
    for sub, kind in (("1m", "bar"), ("ticks", "tick")):
        root = base / sub
        if not root.exists():
            continue
        targets.extend((path, kind) for path in sorted(root.rglob("*.parquet")))
    manifest_entries: list[dict[str, Any]] = []
    for path, kind in targets:
        result = migrate_intraday_partition_file(path, kind, dry_run=args.dry_run, backup=not args.no_backup)
        manifest_entries.append({"file": str(path), "kind": kind, **result})
        _print_data_line(path, result)
    if not args.dry_run:
        base.mkdir(parents=True, exist_ok=True)
        manifest = {"migrated_at": datetime.now(UTC).isoformat(), "files": manifest_entries}
        (base / "_migration_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry, exercised via `python -m`
    main()
