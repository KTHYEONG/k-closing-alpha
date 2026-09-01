"""Alt-data 백필 오케스트레이터."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Collectors
from src.backfill.altdata import derivatives, fundamental, investor_detail, shorting
from src.backfill.altdata.config import _ALTDATA_PANELS, AltDataFetchConfig
from src.backfill.altdata.normalize import normalize_panel

# Re-export collectors for test monkeypatching
collect_shorting = shorting.collect_shorting
collect_fundamental = fundamental.collect_fundamental
collect_investor_detail = investor_detail.collect_investor_detail
collect_derivatives_basis = derivatives.collect_derivatives_basis

logger = logging.getLogger(__name__)


def _covered_dates(panel_path: Path) -> set[pd.Timestamp]:
    """패널에 이미 존재하는 날짜 집합을 반환합니다.

    Args:
        panel_path: 패널 parquet 경로.

    Returns:
        날짜 집합.
    """
    if not panel_path.exists():
        return set()
    try:
        df = pd.read_parquet(panel_path)
    except Exception:
        return set()
    if df is None or df.empty or "date" not in df.columns:
        return set()
    try:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna().dt.normalize()
        return {pd.Timestamp(d).normalize() for d in dates.unique()}
    except Exception:
        return set()


def _incremental_merge(existing_path: Path, new_df: pd.DataFrame, key_cols: tuple[str, ...]) -> pd.DataFrame:
    """기존 패널과 신규 데이터를 병합합니다.

    Args:
        existing_path: 기존 parquet 경로.
        new_df: 신규 DataFrame.
        key_cols: 키 컬럼 튜플.

    Returns:
        병합된 DataFrame.
    """
    if existing_path.exists():
        try:
            existing = pd.read_parquet(existing_path)
        except Exception:
            existing = pd.DataFrame()
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df.copy()
    else:
        combined = new_df.copy()
    if combined.empty:
        return combined
    # Ensure date normalization for sorting
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.normalize()
    combined = combined.drop_duplicates(subset=list(key_cols), keep="last")
    sort_cols = [c for c in key_cols if c in combined.columns]
    if sort_cols:
        combined = combined.sort_values(sort_cols)
    return combined.reset_index(drop=True)


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """DataFrame을 원자적으로 parquet로 저장합니다.

    Args:
        df: 저장할 DataFrame.
        path: 대상 경로.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    # Ensure suffix handling: with_suffix replaces last suffix; for .parquet -> .tmp
    # But we want path.parquet.tmp ; use with_suffix('.parquet.tmp') logic: instead construct
    # Use path + ".tmp" style
    tmp = Path(str(path) + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _write_manifest(out_dir: Path, entries: dict[str, dict[str, Any]]) -> None:
    """매니페스트를 기록합니다.

    Args:
        out_dir: 출력 디렉토리.
        entries: 패널별 엔트리.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "panels": entries,
    }
    path = out_dir / "_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str, ensure_ascii=False)


def run_altdata_backfill(cfg: AltDataFetchConfig) -> dict[str, Any]:
    """Alt-data 백필을 실행합니다.

    Args:
        cfg: Alt-data 설정.

    Returns:
        매니페스트 딕셔너리.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    business_days = pd.bdate_range(cfg.start, cfg.end).tolist()
    # Ensure Timestamp
    business_days = [pd.Timestamp(d).normalize() for d in business_days]

    entries: dict[str, dict[str, Any]] = {}

    for source in cfg.sources:
        meta = _ALTDATA_PANELS.get(source)
        if meta is None:
            entries[source] = {
                "status": "unavailable",
                "source": source,
                "availability_rule": "eod_release_next_decision",
                "rows": 0,
                "first_date": None,
                "last_date": None,
                "updated_at": datetime.now(UTC).isoformat(),
                "error": f"unknown source {source}",
            }
            continue
        filename: str = meta["filename"]
        key_cols: tuple[str, ...] = meta["key_cols"]
        availability_rule: str = meta["availability_rule"]
        panel_path = cfg.out_dir / filename

        # Determine covered dates
        covered = _covered_dates(panel_path)
        missing = [d for d in business_days if d not in covered]

        if not missing:
            # Already up to date
            # Read existing to report rows/dates
            try:
                existing = pd.read_parquet(panel_path) if panel_path.exists() else pd.DataFrame()
                rows = len(existing) if existing is not None and not existing.empty else 0
                if rows > 0 and "date" in existing.columns:
                    first = pd.to_datetime(existing["date"], errors="coerce").min()
                    last = pd.to_datetime(existing["date"], errors="coerce").max()
                    first_s = str(pd.Timestamp(first).date()) if pd.notna(first) else None
                    last_s = str(pd.Timestamp(last).date()) if pd.notna(last) else None
                else:
                    first_s = None
                    last_s = None
            except Exception:
                rows = 0
                first_s = None
                last_s = None
            entries[source] = {
                "status": "up_to_date",
                "source": source,
                "availability_rule": availability_rule,
                "rows": rows,
                "first_date": first_s,
                "last_date": last_s,
                "updated_at": datetime.now(UTC).isoformat(),
            }
            continue

        # Collect
        try:
            if source == "shorting":
                raw = collect_shorting(cfg, missing)
            elif source == "fundamental":
                raw = collect_fundamental(cfg, missing)
            elif source == "investor_detail":
                raw = collect_investor_detail(cfg, missing)
            elif source == "derivatives_basis":
                raw = collect_derivatives_basis(cfg, missing)
            elif source == "disclosure":
                # disclosure needs corp map handling
                from src.backfill.altdata import disclosure as disc_mod

                if not str(cfg.dart_api_key).strip():
                    raise ValueError("DART_API_KEY is required for disclosure backfill")
                # Cache corp map
                corp_map_path = cfg.out_dir / "corp_code_map.parquet"
                corp_map: pd.DataFrame | None = None
                if corp_map_path.exists():
                    try:
                        corp_map = pd.read_parquet(corp_map_path)
                    except Exception:
                        corp_map = None
                if corp_map is None or corp_map.empty:
                    corp_map = disc_mod.download_corp_code_map(cfg)
                    try:
                        # cache
                        corp_map_path.parent.mkdir(parents=True, exist_ok=True)
                        corp_map.to_parquet(corp_map_path, index=False)
                    except Exception:
                        pass
                raw = disc_mod.collect_disclosures(cfg, corp_map)
                # For disclosure, need to filter to missing dates? The raw already covers full range but we incremental merge later.
                # No extra filtering needed, but normalize will handle.
            else:
                raw = pd.DataFrame()

            # Empty result => unavailable
            if raw is None or raw.empty:
                entries[source] = {
                    "status": "unavailable",
                    "source": source,
                    "availability_rule": availability_rule,
                    "rows": 0,
                    "first_date": None,
                    "last_date": None,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "error": "empty collector result",
                }
                continue

            normalized = normalize_panel(raw, source, cfg)
            if normalized is None or normalized.empty:
                entries[source] = {
                    "status": "unavailable",
                    "source": source,
                    "availability_rule": availability_rule,
                    "rows": 0,
                    "first_date": None,
                    "last_date": None,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "error": "empty after normalization",
                }
                continue

            merged = _incremental_merge(panel_path, normalized, key_cols)
            _atomic_write_parquet(merged, panel_path)
            # Compute coverage
            if not merged.empty and "date" in merged.columns:
                first = pd.to_datetime(merged["date"], errors="coerce").min()
                last = pd.to_datetime(merged["date"], errors="coerce").max()
                first_s = str(pd.Timestamp(first).date()) if pd.notna(first) else None
                last_s = str(pd.Timestamp(last).date()) if pd.notna(last) else None
                rows = len(merged)
            else:
                first_s = None
                last_s = None
                rows = len(merged)
            entries[source] = {
                "status": "ok",
                "source": source,
                "availability_rule": availability_rule,
                "rows": rows,
                "first_date": first_s,
                "last_date": last_s,
                "updated_at": datetime.now(UTC).isoformat(),
            }
        except ValueError as ve:
            # Check for DART key missing -> skipped_no_key
            msg = str(ve)
            if "DART_API_KEY" in msg:
                entries[source] = {
                    "status": "skipped_no_key",
                    "source": source,
                    "availability_rule": availability_rule,
                    "rows": 0,
                    "first_date": None,
                    "last_date": None,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "error": msg,
                }
            else:
                entries[source] = {
                    "status": "unavailable",
                    "source": source,
                    "availability_rule": availability_rule,
                    "rows": 0,
                    "first_date": None,
                    "last_date": None,
                    "updated_at": datetime.now(UTC).isoformat(),
                    "error": repr(ve),
                }
        except Exception as exc:
            entries[source] = {
                "status": "unavailable",
                "source": source,
                "availability_rule": availability_rule,
                "rows": 0,
                "first_date": None,
                "last_date": None,
                "updated_at": datetime.now(UTC).isoformat(),
                "error": repr(exc),
            }

    manifest = {"generated_at": datetime.now(UTC).isoformat(), "panels": entries}
    _write_manifest(cfg.out_dir, entries)
    return manifest
