"""Base + patch offsite form of the price_history monolith.

The monolith is rewritten in full by every ingest, so a loose-tier copy parks a
full ~235 MiB version in `_deleted/` each night. Month partitions alone do not
help: a corporate-action back-adjustment rewrites one symbol's whole history,
which touches every month. This form keeps a rarely rebuilt month-partitioned
base plus a small patch (rows added or changed since the base) and the deleted
keys. File names embed their content md5, so an unchanged file is never
rewritten and a new manifest never references a file that an upload is still
replacing. Local readers keep using the monolith; nothing here mutates it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.io_utils import atomic_output_path, atomic_write_text
from src.data.parquet_codec import (
    INTRADAY_COMPRESSION,
    PARQUET_COMPRESSION_LEVEL,
    downcast_price_history_frame,
    write_price_history_parquet,
)
from src.tools.offsite_common import sha256_file
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

PRICE_HISTORY_RELPATH: str = "data/history/price_history.parquet"
OFFSITE_FORM_RELPATH: str = "data/history/price_history_offsite"
MANIFEST_NAME: str = "_manifest.json"
MANIFEST_SCHEMA_VERSION: int = 1
KIND_BASE, KIND_PATCH, KIND_DELETES = "base", "patch", "deletes"
KEY_COLUMNS: tuple[str, ...] = ("symbol", "date")
DATE_COLUMN: str = "date"
CATEGORY_COLUMNS: tuple[str, ...] = ("symbol", "market")
NAME_DIGEST_CHARS: int = 16
# Rebase once the patch exceeds this share of the panel: about one base rebuild per month at the observed
# ~2.7k new plus ~2.6k back-adjusted rows per trading day (2026-10), instead of one full copy per night.
REBASE_PATCH_ROW_RATIO: float = 0.02


@dataclass(frozen=True)
class OffsiteFile:
    """One file of the offsite form; md5 equals the Drive checksum, rows the parquet row count."""

    relpath: str
    kind: str
    md5: str
    bytes: int
    rows: int


@dataclass(frozen=True)
class ExportReport:
    """Outcome of one export."""

    files: int
    written: tuple[str, ...]
    removed: tuple[str, ...]
    rebased: bool
    patch_rows: int
    delete_rows: int
    rows: int
    source_sha256: str


def _md5(payload: bytes) -> str:
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def _plain(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in CATEGORY_COLUMNS:
        if column in out.columns and isinstance(out[column].dtype, pd.CategoricalDtype):
            out[column] = out[column].astype(out[column].cat.categories.dtype)
    return out


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    sink = io.BytesIO()
    table = pa.Table.from_pandas(frame.reset_index(drop=True), preserve_index=False)
    pq.write_table(table, sink, compression=INTRADAY_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL)
    return sink.getvalue()


def _entry(stem: str, kind: str, payload: bytes, rows: int) -> tuple[OffsiteFile, bytes]:
    digest = _md5(payload)
    relpath = f"{stem}-{digest[:NAME_DIGEST_CHARS]}.parquet"
    return OffsiteFile(relpath, kind, digest, len(payload), rows), payload


def base_files(frame: pd.DataFrame) -> list[tuple[OffsiteFile, bytes]]:
    """Serialize a canonical frame into deterministic calendar-month base files.

    Category columns are written as plain strings so a file's bytes depend only
    on its own rows, not on the global symbol dictionary; rows keep the
    canonical (symbol, date) order.

    Raises:
        ValueError: The frame has no date column or null dates.
    """
    if DATE_COLUMN not in frame.columns:
        raise ValueError("price_history frame has no date column")
    dates = pd.to_datetime(frame[DATE_COLUMN])
    if dates.isna().any():
        raise ValueError("price_history frame has null dates")
    plain = _plain(frame)
    months = (dates.dt.year * 100 + dates.dt.month).to_numpy()
    out: list[tuple[OffsiteFile, bytes]] = []
    for month in sorted(set(months.tolist())):
        part = plain.loc[months == month]
        stem = f"{KIND_BASE}/year={month // 100:04d}/month={month % 100:02d}"
        out.append(_entry(stem, KIND_BASE, _parquet_bytes(part), len(part)))
    return out


def _hashes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Categorical hashing is value-based, so a categorical frame and its plain-string files hash alike.
    keys = pd.util.hash_pandas_object(frame[list(KEY_COLUMNS)], index=False).to_numpy()
    rows = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    return keys, rows


def _publish(out_dir: Path, entry: OffsiteFile, payload: bytes) -> bool:
    path = Path(out_dir) / entry.relpath
    if path.exists() and path.stat().st_size == entry.bytes and _md5(path.read_bytes()) == entry.md5:
        return False
    with atomic_output_path(path, mode=0o644) as tmp_path:
        tmp_path.write_bytes(payload)
    return True


@dataclass(frozen=True)
class OffsiteManifest:
    """Parsed manifest: the monolith it was built from and the files that rebuild it."""

    source_sha256: str
    rows: int
    columns: tuple[str, ...]
    files: tuple[OffsiteFile, ...]


def parse_manifest(text: str) -> OffsiteManifest:
    """Parse and validate manifest JSON text.

    Raises:
        ValueError: The manifest is malformed, has an unknown schema version, or
            does not list base files plus exactly one patch and one deletes file.
    """
    raw: object = json.loads(text)
    if not isinstance(raw, Mapping) or raw.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("price_history offsite manifest has an unknown schema")
    entries, columns = raw.get("files"), raw.get("columns")
    if not isinstance(entries, list) or not isinstance(columns, list):
        raise ValueError("price_history offsite manifest has no file or column list")
    try:
        files = tuple(OffsiteFile(**entry) for entry in entries)
        manifest = OffsiteManifest(str(raw["source_sha256"]), int(raw["rows"]), tuple(str(c) for c in columns), files)
    except (TypeError, KeyError) as exc:
        raise ValueError(f"malformed manifest: {exc}") from exc
    if any(f.kind not in (KIND_BASE, KIND_PATCH, KIND_DELETES) for f in files) or sum(f.kind != KIND_BASE for f in files) != 2:
        raise ValueError("price_history offsite manifest must list base files plus one patch and one deletes file")
    return manifest


def read_manifest(out_dir: Path) -> OffsiteManifest:
    """Read and validate `<out_dir>/_manifest.json`.

    Raises:
        OSError: The manifest cannot be read.
        ValueError: As parse_manifest.
    """
    return parse_manifest((Path(out_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def _load(out_dir: Path, entry: OffsiteFile) -> pd.DataFrame:
    payload = (Path(out_dir) / entry.relpath).read_bytes()
    if _md5(payload) != entry.md5:
        raise ValueError(f"md5 mismatch: {entry.relpath}")
    frame = pq.read_table(io.BytesIO(payload)).to_pandas()
    if len(frame) != entry.rows:
        raise ValueError(f"row count mismatch: {entry.relpath}")
    return frame


def rebuild_frame(out_dir: Path) -> pd.DataFrame:
    """Rebuild the canonical monolith frame: base minus patched or deleted keys, plus the patch.

    Raises:
        OSError: A listed file is missing.
        ValueError: A file's md5 or row count disagrees with the manifest, or the
            manifest's total row count or column order is not reproduced.
    """
    manifest = read_manifest(out_dir)
    files = manifest.files
    base_parts = [_load(out_dir, f) for f in files if f.kind == KIND_BASE]
    patch = next(_load(out_dir, f) for f in files if f.kind == KIND_PATCH)
    deletes = next(_load(out_dir, f) for f in files if f.kind == KIND_DELETES)
    base = pd.concat(base_parts, ignore_index=True) if base_parts else patch.iloc[0:0]
    drop = pd.MultiIndex.from_frame(pd.concat([patch[list(KEY_COLUMNS)], deletes[list(KEY_COLUMNS)]], ignore_index=True))
    keep = ~pd.MultiIndex.from_frame(base[list(KEY_COLUMNS)]).isin(drop)
    frame = pd.concat([base.loc[keep], patch], ignore_index=True)
    if tuple(frame.columns) != manifest.columns or len(frame) != manifest.rows:
        raise ValueError("rebuilt frame does not match the manifest columns/rows")
    for column in CATEGORY_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("category")
    return downcast_price_history_frame(frame)


def _usable_base(out_dir: Path, columns: list[str]) -> list[OffsiteFile] | None:
    try:
        manifest = read_manifest(out_dir)
    except (OSError, ValueError) as exc:
        logger.info("[DATA] stage=price_history_offsite base=UNUSABLE reason=%s: %s", type(exc).__name__, exc)
        return None
    return [f for f in manifest.files if f.kind == KIND_BASE] if list(manifest.columns) == columns else None


def _diff(
    current: pd.DataFrame, cur_index: pd.Index, cur_rows: np.ndarray, out_dir: Path, base: list[OffsiteFile]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows of current new or changed vs the on-disk base, plus base keys absent from current.

    Base files are streamed one at a time so the base is never materialized
    whole. A key-hash collision can only hide a change; the caller's rebuild
    check catches it and forces a rebase.

    Raises:
        OSError, ValueError: A base file is missing or fails its manifest md5/rows.
    """
    changed = np.ones(len(current), dtype=bool)
    gone: list[pd.DataFrame] = []
    for entry in base:
        part = _load(out_dir, entry)
        keys, rows = _hashes(part)
        pos = cur_index.get_indexer(keys)
        hit = pos >= 0
        same = hit & (cur_rows[np.where(hit, pos, 0)] == rows)
        changed[pos[same]] = False
        gone.append(part.loc[~hit, list(KEY_COLUMNS)])
    deletes = pd.concat(gone, ignore_index=True) if gone else current.iloc[0:0][list(KEY_COLUMNS)]
    return current.loc[changed], deletes


def _rebuilds_rows(out_dir: Path, cur_rows: np.ndarray) -> bool:
    """Whether the listed files rebuild exactly the current rows (equal multiset of full-row hashes).

    Streams one file at a time to bound memory; the exact frame round trip is
    pinned by unit tests and re-checked by `rebuild_frame` at restore time.
    """
    files = read_manifest(out_dir).files
    patch = next(_load(out_dir, f) for f in files if f.kind == KIND_PATCH)
    deletes = next(_load(out_dir, f) for f in files if f.kind == KIND_DELETES)
    patch_keys, patch_rows = _hashes(patch)
    dropped = pd.Index(np.concatenate([patch_keys, _hashes(deletes)[0]]))
    kept = [patch_rows]
    for entry in files:
        if entry.kind == KIND_BASE:
            keys, rows = _hashes(_load(out_dir, entry))
            kept.append(rows[~pd.Index(keys).isin(dropped)])
    rebuilt = np.sort(np.concatenate(kept))
    return len(rebuilt) == len(cur_rows) and bool(np.array_equal(rebuilt, np.sort(cur_rows)))


def _remove_unlisted(out_dir: Path, listed: set[str]) -> tuple[str, ...]:
    removed: list[str] = []
    for path in sorted(Path(out_dir).rglob("*.parquet")):
        rel = path.relative_to(out_dir).as_posix()
        if rel not in listed:
            path.unlink()
            removed.append(rel)
    for directory in sorted((d for d in Path(out_dir).rglob("*") if d.is_dir()), reverse=True):
        if not any(directory.iterdir()):
            directory.rmdir()
    return tuple(removed)


def export_offsite_form(source: Path, out_dir: Path, *, rebase_ratio: float = REBASE_PATCH_ROW_RATIO) -> ExportReport:
    """Refresh the base + patch mirror of the monolith and prove the files rebuild its rows.

    The base is rebuilt when absent or unreadable, when its columns differ, when
    current keys repeat, when the patch would exceed rebase_ratio of the panel,
    or when the patch form fails the rebuild check. New files are published
    before the manifest; files the new manifest no longer lists are removed
    afterwards.

    Raises:
        OSError: Reading the source or publishing a file failed.
        ValueError: The source is invalid, or even a fresh base fails the rebuild check.
    """
    source, out_dir = Path(source), Path(out_dir)
    source_sha = sha256_file(source)
    current = pd.read_parquet(source)
    columns = [str(column) for column in current.columns]
    cur_keys, cur_rows = _hashes(current)
    cur_index = pd.Index(cur_keys)
    base = _usable_base(out_dir, columns) if cur_index.is_unique else None
    written: list[str] = []
    for force_rebase in (False, True):
        rebased = force_rebase or base is None
        patch, deletes = current.iloc[0:0], current.iloc[0:0][list(KEY_COLUMNS)]
        if not rebased:
            assert base is not None
            try:
                patch, deletes = _diff(current, cur_index, cur_rows, out_dir, base)
                rebased = len(patch) + len(deletes) > rebase_ratio * max(len(current), 1)
            except (OSError, ValueError) as exc:
                logger.warning("[DATA] stage=price_history_offsite base=UNREADABLE action=rebase reason=%s: %s", type(exc).__name__, exc)
                rebased = True
        if rebased:
            patch, deletes = current.iloc[0:0], current.iloc[0:0][list(KEY_COLUMNS)]
            built = base_files(current)
            written += [entry.relpath for entry, payload in built if _publish(out_dir, entry, payload)]
            base = [entry for entry, _payload in built]
            del built
        extra = [_entry(KIND_PATCH, KIND_PATCH, _parquet_bytes(_plain(patch)), len(patch)),
                 _entry(KIND_DELETES, KIND_DELETES, _parquet_bytes(_plain(deletes)), len(deletes))]
        written += [entry.relpath for entry, payload in extra if _publish(out_dir, entry, payload)]
        assert base is not None
        listed = [*base, *(entry for entry, _payload in extra)]
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source_relpath": PRICE_HISTORY_RELPATH,
            "source_sha256": source_sha,
            "rows": len(current),
            "columns": columns,
            "files": [asdict(entry) for entry in listed],
        }
        atomic_write_text(out_dir / MANIFEST_NAME, json.dumps(manifest, sort_keys=True, indent=1), mode=0o644)
        if _rebuilds_rows(out_dir, cur_rows):
            removed = _remove_unlisted(out_dir, {entry.relpath for entry in listed})
            return ExportReport(
                files=len(listed),
                written=tuple(sorted(set(written))),
                removed=removed,
                rebased=rebased,
                patch_rows=len(patch),
                delete_rows=len(deletes),
                rows=len(current),
                source_sha256=source_sha,
            )
        if rebased:
            raise ValueError("price_history offsite form does not rebuild the monolith rows")
        logger.warning("[DATA] stage=price_history_offsite verify=MISMATCH action=rebase")
    raise AssertionError("unreachable: the rebase attempt either returns or raises")  # pragma: no cover


def restore_monolith(out_dir: Path, target: Path) -> int:
    """Write the rebuilt monolith to target through the canonical price_history writer.

    Returns:
        Rows written.
    """
    frame = rebuild_frame(out_dir)
    write_price_history_parquet(frame, Path(target))
    return len(frame)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI entry; logic covered by unit tests
    parser = argparse.ArgumentParser(description="Base + patch offsite form of price_history")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="refresh the local offsite mirror from the monolith")
    export.add_argument("--source", type=Path, default=Path(PRICE_HISTORY_RELPATH))
    export.add_argument("--out-dir", type=Path, default=Path(OFFSITE_FORM_RELPATH))
    restore = sub.add_parser("restore", help="rebuild a monolith from a downloaded offsite mirror")
    restore.add_argument("--from-dir", type=Path, required=True)
    restore.add_argument("--target", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        report = export_offsite_form(args.source, args.out_dir)
        logger.info(
            "[DATA] stage=price_history_offsite step=export files=%d written=%d removed=%d rebased=%s patch_rows=%d delete_rows=%d rows=%d",
            report.files, len(report.written), len(report.removed), report.rebased, report.patch_rows, report.delete_rows, report.rows,
        )
        return
    if args.target.exists():
        logger.error("[DATA] stage=price_history_offsite step=restore status=REFUSED reason=target_exists target=%s", args.target)
        sys.exit(2)
    rows = restore_monolith(args.from_dir, args.target)
    logger.info("[DATA] stage=price_history_offsite step=restore status=OK rows=%d target=%s", rows, args.target)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
