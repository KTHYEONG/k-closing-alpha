"""Alt-data 백필 오케스트레이터."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Collectors
from src.backfill.altdata import credit_balance, derivatives, program_trade_daily, shorting
from src.backfill.altdata.config import _ALTDATA_PANELS, AltDataFetchConfig, dart_pool_for
from src.backfill.altdata.normalize import normalize_panel
from src.backfill.altdata.ratelimit import DartNonRetryableError, DartQuotaExhaustedError
from src.data.capture_contracts import (
    BrokerPayload,
    CaptureContext,
    CaptureDataset,
    CapturedResponse,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
    PageObserver,
    RawCaptureError,
    SEOUL,
)
from src.data.capture_store import CaptureStore
from src.data.parquet_codec import write_altdata_panel_parquet

# Re-export collectors for test monkeypatching
collect_shorting = shorting.collect_shorting
collect_derivatives_basis = derivatives.collect_derivatives_basis
collect_credit_balance = credit_balance.collect_credit_balance
collect_program_trade_daily = program_trade_daily.collect_program_trade_daily

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
        df = pd.read_parquet(panel_path, columns=["date"])
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
    """DataFrame을 원자적으로 parquet로 저장합니다 (공유 코덱 위임).

    Args:
        df: 저장할 DataFrame.
        path: 대상 경로.
    """
    write_altdata_panel_parquet(df, path)


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


_SOURCE_DATASETS: dict[str, CaptureDataset] = {
    "shorting": CaptureDataset.SHORTING,
    "derivatives_basis": CaptureDataset.DERIVATIVES_BASIS,
    "disclosure": CaptureDataset.DISCLOSURE,
    "credit_balance": CaptureDataset.CREDIT_BALANCE,
    "program_trade_daily": CaptureDataset.PROGRAM_DAILY,
}

_OK_ENTRY_STATES: frozenset[str] = frozenset({"ok", "up_to_date"})

_CLOCK_COLUMNS: frozenset[str] = frozenset({
    "observed_at",
    "received_at",
    "request_started_at",
    "source_timestamp",
    "source_published_at",
    "capture_ref",
    "artifact_ref",
    "manifest_ref",
})


def _altdata_capture_context(trading_day: pd.Timestamp, run_id: str, source: str) -> CaptureContext:
    return CaptureContext(
        trading_date=pd.Timestamp(trading_day).normalize().date(),
        run_id=run_id,
        dataset=_SOURCE_DATASETS.get(source, CaptureDataset.SHORTING),
        vendor="owner-local",
        endpoint=f"{source}-collector",
        symbol=None,
        venue="KRX",
        session="regular",
        capture_reason="altdata-backfill",
        cohort_id=None,
        scheduled_at=None,
    )


def _dart_page_observer(store: CaptureStore, trading_day: pd.Timestamp, run_id: str, sink: list[Any]) -> PageObserver:
    def _on_page(
        payload: BrokerPayload | None,
        meta: Mapping[str, str],
        started: datetime,
        received: datetime,
        page_index: int,
        attempt_index: int,
    ) -> None:
        try:
            ref = store.append_response(
                CapturedResponse(
                    context=_altdata_capture_context(trading_day, run_id, "disclosure"),
                    request_started_at=started,
                    received_at=received,
                    payload=dict(payload) if isinstance(payload, dict) else None,
                    source_timestamp=None,
                    source_published_at=None,
                    status=CaptureStatus.COMPLETE if isinstance(payload, dict) else CaptureStatus.FAILED,
                    page_index=int(page_index),
                    attempt_index=int(attempt_index),
                    continuation={k: str(v) for k, v in dict(meta).items()},
                    error_type=None if isinstance(payload, dict) else "transport",
                )
            )
        except OSError as exc:
            raise RawCaptureError(str(exc)) from exc
        sink.append(ref)

    return _on_page


def _capture_collector_output(
    store: CaptureStore, run_id: str, trading_day: pd.Timestamp, source: str, raw: pd.DataFrame,
) -> Any:
    return store.publish_frame(raw.copy(), context=_altdata_capture_context(trading_day, run_id, source))


def run_altdata_backfill(cfg: AltDataFetchConfig, *, capture_store: CaptureStore | None = None, run_id: str | None = None, reobserve: bool = False) -> dict[str, Any]:
    """Retain source observations and refresh a bounded slow-data window independently.

    Args:
        cfg: Existing source/date configuration.
        capture_store: Optional owner-local immutable provenance store.
        run_id: Required unique run identity when capture_store is supplied.
        reobserve: Re-fetch the configured window despite prior date presence.
    Returns:
        Existing source report with separate capture manifest references.
    Raises:
        ValueError: Capture configuration is inconsistent.
        RawCaptureError: Required observation persistence failed.
    """
    if (capture_store is None) != (run_id is None):
        raise ValueError("capture_store and run_id must be supplied together")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    business_days = pd.bdate_range(cfg.start, cfg.end).tolist()
    # Ensure Timestamp
    business_days = [pd.Timestamp(d).normalize() for d in business_days]

    entries: dict[str, dict[str, Any]] = {}
    capture_refs: dict[str, list[Any]] = {}
    anchor_day = pd.Timestamp(cfg.end).normalize()

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
        missing = list(business_days) if reobserve else [d for d in business_days if d not in covered]

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
            elif source == "derivatives_basis":
                raw = collect_derivatives_basis(cfg, missing)
            elif source == "credit_balance":
                raw = collect_credit_balance(cfg, missing)
            elif source == "program_trade_daily":
                raw = collect_program_trade_daily(cfg, missing)
            elif source == "disclosure":
                # disclosure needs corp map handling
                from src.backfill.altdata import disclosure as disc_mod

                if dart_pool_for(cfg).is_empty():
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
                # 창 단위 즉시 flush: 중단되어도 이미 받은 구간은 보존.
                def _flush_window(window_df: pd.DataFrame, _pp: Path = panel_path, _kc: tuple[str, ...] = key_cols) -> None:
                    norm = normalize_panel(window_df, "disclosure", cfg)
                    if norm is None or norm.empty:
                        return
                    _atomic_write_parquet(_incremental_merge(_pp, norm, _kc), _pp)

                disc_page_refs: list[Any] = []
                disc_mod.collect_disclosures(
                    cfg,
                    corp_map,
                    on_window=_flush_window,
                    covered_dates=set() if reobserve else set(covered),
                    on_page=_dart_page_observer(capture_store, anchor_day, run_id, disc_page_refs) if capture_store is not None and run_id is not None else None,
                )
                if capture_store is not None and run_id is not None:
                    capture_refs[source] = list(disc_page_refs)
                raw = pd.read_parquet(panel_path) if panel_path.exists() else pd.DataFrame()
            else:
                raw = pd.DataFrame()

            if capture_store is not None and run_id is not None and source != "disclosure" and raw is not None and not raw.empty:
                capture_refs[source] = [_capture_collector_output(capture_store, run_id, anchor_day, source, raw)]

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

            feature_raw = raw.drop(columns=[c for c in raw.columns if c in _CLOCK_COLUMNS])
            normalized = normalize_panel(feature_raw, source, cfg)
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
        except DartQuotaExhaustedError as exc:
            # Every pooled key is exhausted by daily quota; the rolling window
            # re-fetches the missed days on the next successful run.
            entries[source] = {
                "status": "quota_exceeded",
                "source": source,
                "availability_rule": availability_rule,
                "rows": 0,
                "first_date": None,
                "last_date": None,
                "updated_at": datetime.now(UTC).isoformat(),
                "error": repr(exc),
            }
        except DartNonRetryableError as exc:
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
    if capture_store is not None and run_id is not None:
        coverage: list[CoverageEntry] = []
        for source in cfg.sources:
            state = str(entries.get(source, {}).get("status", "unavailable"))
            status = CaptureStatus.COMPLETE if state in _OK_ENTRY_STATES else CaptureStatus.FAILED
            rows = int(entries.get(source, {}).get("rows", 0) or 0)
            coverage.append(
                CoverageEntry(
                    symbol=None,
                    dataset=_SOURCE_DATASETS.get(source, CaptureDataset.SHORTING),
                    venue="KRX",
                    session="regular",
                    scheduled_at=None,
                    status=status,
                    rows=rows,
                    first_event_time=None,
                    last_event_time=None,
                    reason=state,
                    raw_refs=tuple(capture_refs.get(source, ())),
                )
            )
        overall = CaptureStatus.COMPLETE if all(e.status == CaptureStatus.COMPLETE for e in coverage) else CaptureStatus.PARTIAL
        first_dataset = _SOURCE_DATASETS.get(cfg.sources[0], CaptureDataset.SHORTING)
        capture_manifest = CaptureManifest(
            schema_version=1,
            context=CaptureContext(
                trading_date=anchor_day.date(),
                run_id=run_id,
                dataset=first_dataset,
                vendor="owner-local",
                endpoint="altdata-backfill",
                symbol=None,
                venue="KRX",
                session="regular",
                capture_reason="altdata-backfill",
                cohort_id=None,
                scheduled_at=None,
            ),
            cohort=None,
            completed_at=datetime.now(SEOUL),
            entries=tuple(coverage),
            artifacts=tuple(r for refs in capture_refs.values() for r in refs),
            status=overall,
        )
        manifest_ref = capture_store.publish_manifest(capture_manifest)
        manifest["capture"] = {"run_id": run_id, "manifest": manifest_ref.path, "status": overall.value}
    _write_manifest(cfg.out_dir, entries)
    return manifest
