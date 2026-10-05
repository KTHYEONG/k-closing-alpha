"""Intraday 분봉 날짜 파티션 저장소 (date-partitioned parquet)."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import settings
from src.config.market_session import INTRADAY_SESSION_REGULAR_CONSOLIDATED, INTRADAY_VERIFIED_SESSIONS
from src.data.capture_contracts import CaptureStatus, CoverageEntry
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.intraday_schema import CANONICAL_BAR_COLUMNS, assert_canonical_bars, assert_canonical_ticks
from src.utils.file_lock import DEFAULT_LOCK_TIMEOUT_SECONDS, exclusive_file_lock, sidecar_lock_path

logger = logging.getLogger(__name__)

__all__ = ["intraday_partition_path", "log_session_coverage_outliers", "remove_intraday_symbols", "tick_partition_path", "write_intraday_partition", "write_tick_partition"]

_LOCK_TIMEOUT_SECONDS = DEFAULT_LOCK_TIMEOUT_SECONDS

# Mirrors the Toss basis gate verdict; the consolidated partition holds only these rows.
_CONSOLIDATED_TAPE_REASON: str = "toss_consolidated_tape"


def _require_known_session(session: str) -> str:
    name = str(session)
    if name not in INTRADAY_VERIFIED_SESSIONS:
        raise ValueError(f"Unknown intraday session: {session!r}")
    return name


def intraday_partition_path(bar_interval_minutes: int, snapshot_date: str, session: str) -> Path:
    """data/history/intraday/{interval}m/{session}/{YYYY-MM}/{YYYY-MM-DD}.parquet 경로 산출."""
    _require_known_session(session)
    month = str(snapshot_date)[:7]
    return (
        Path(settings.HISTORY_DIR)
        / "intraday"
        / f"{int(bar_interval_minutes)}m"
        / str(session)
        / month
        / f"{snapshot_date}.parquet"
    )


def log_session_coverage_outliers(
    merged: pd.DataFrame,
    bar_interval_minutes: int,
    snapshot_date: str,
    session: str,
    *,
    min_peer_ratio: float = 0.8,
    max_boundary_gap_hhmmss: int = 500,
) -> dict[str, int]:
    """파티션 병합 후 종목별 봉 수/세션 경계를 peer와 비교해 저조·절단 종목을 로그로 남긴다.

    두 가지 독립 신호를 낸다:
    1) n_low_coverage: 종목의 총 봉수가 peer 최댓값의 min_peer_ratio 미만(기존 신호).
    2) n_truncated: 종목의 첫/마지막 봉이 배치 전체의 세션 floor/ceiling에서
       max_boundary_gap_hhmmss 이상 벗어남 -- /probe(intraday_gap_retroactive_backfill_policy)
       실측 결과, peer 봉수비율만으로는 '정상적으로 희소한 거래'(세션 전체범위는
       채워지되 내부에 산발적 공백만 있는 경우, 실측 217건 중 199건, 92%)와
       '진짜 부분수집 결함'(세션 시작/끝이 실제로 잘려나간 경우)을 구분하지 못했다
       (false positive). 절단 여부가 더 정밀한 결함 신호임을 실측으로 확인했다.

    벤더 응답이 성공(예외 없음)이었어도 특정 종목만 세션 일부만 수집된 경우 현재는
    어떤 진단도 남기지 않는다. 자동 재수집/차단은 하지 않는다 -- 희소유동성 종목의
    정상적으로 낮은 봉수와 실제 수집 실패를 완전히 구분할 수는 없으므로, 조치는
    로그를 본 사람의 판단에 맡긴다.

    Args:
        merged: write_intraday_partition이 병합해 실제로 쓰는 최종 프레임.
        bar_interval_minutes: 파티션의 봉 간격(로그 컨텍스트용).
        snapshot_date: 파티션 날짜(로그 컨텍스트용).
        session: 세션 태그(로그 컨텍스트용).
        min_peer_ratio: peer 최댓값 대비 이 비율 미만이면 저조로 표식(strict less-than).
        max_boundary_gap_hhmmss: ts_hms(HHMMSS 정수) 기준 이 값 이상 floor/ceiling에서
            벗어나면 절단으로 표식. HHMMSS는 선형 시간이 아니므로(시 경계에서 비선형
            점프) 여유 있게 잡은 기본값이다.

    Returns:
        {"n_symbols": 전체 종목수, "n_low_coverage": 저조 종목수, "n_truncated": 절단 종목수}.
    """
    if merged.empty or "symbol" not in merged.columns:
        return {"n_symbols": 0, "n_low_coverage": 0, "n_truncated": 0}
    counts = merged.groupby("symbol").size()
    peer_max = int(counts.max())
    low = counts[counts < peer_max * float(min_peer_ratio)]
    if len(low):
        logger.warning(
            "[DATA] stage=session_coverage bar_interval=%dm date=%s session=%s peer_max=%d n_low=%d symbols=%s",
            bar_interval_minutes, snapshot_date, session, peer_max, len(low), sorted(low.index.tolist())[:20],
        )
    truncated: pd.Index = pd.Index([], dtype=object)
    if "ts_hms" in merged.columns and len(counts) > 1:
        session_floor = int(merged["ts_hms"].min())
        session_ceil = int(merged["ts_hms"].max())
        first_ts = merged.groupby("symbol")["ts_hms"].min()
        last_ts = merged.groupby("symbol")["ts_hms"].max()
        starts_late = first_ts[first_ts > session_floor + max_boundary_gap_hhmmss].index
        ends_early = last_ts[last_ts < session_ceil - max_boundary_gap_hhmmss].index
        truncated = starts_late.union(ends_early)
        if len(truncated):
            logger.warning(
                "[DATA] stage=session_truncation bar_interval=%dm date=%s session=%s floor=%d ceil=%d n_truncated=%d symbols=%s",
                bar_interval_minutes, snapshot_date, session, session_floor, session_ceil, len(truncated), sorted(truncated.tolist())[:20],
            )
    return {"n_symbols": len(counts), "n_low_coverage": len(low), "n_truncated": len(truncated)}


def _batch_rows_or_default(batch_rows: int | None) -> int:
    if batch_rows is None:
        return int(settings.COLLECTION_ARROW_BATCH_ROWS)
    if int(batch_rows) <= 0:
        raise ValueError(f"batch_rows must be positive: {batch_rows!r}")
    return int(batch_rows)


def _is_valid_hhmmss(value: int) -> bool:
    if value < 0 or value > 235959:
        return False
    hour = value // 10000
    minute = (value // 100) % 100
    second = value % 100
    return 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59


def _validate_tick_frame(
    df: pd.DataFrame, snapshot_date: str, session: str, coverage: Mapping[str, CoverageEntry] | None
) -> list[str]:
    assert_canonical_ticks(df)
    dates = df["snapshot_date"].astype(str)
    if bool((dates != str(snapshot_date)).any()):
        raise ValueError(f"Tick frame carries non-requested snapshot_date for {snapshot_date!r}")
    ts_values = pd.to_numeric(df["ts_hms"], errors="coerce")
    if bool(ts_values.isna().any()):
        raise ValueError("Tick frame carries non-numeric ts_hms")
    for raw in ts_values.astype(int).tolist():
        if not _is_valid_hhmmss(int(raw)):
            raise ValueError(f"Tick frame carries invalid HHMMSS: {raw!r}")
    volumes = pd.to_numeric(df["volume"], errors="coerce")
    if bool(volumes.isna().any()) or bool((volumes < 0).any()):
        raise ValueError("Tick frame carries invalid negative volume")
    symbols = sorted({str(item) for item in df["symbol"].astype(str).tolist()})
    if coverage is not None:
        missing = [item for item in symbols if item not in coverage]
        if missing:
            raise ValueError(f"Tick frame symbols lack coverage certification: {missing}")
        for item in symbols:
            entry = coverage[item]
            if entry.session != str(session):
                raise ValueError(f"Tick coverage session mismatch for {item!r}")
    return symbols


def _validate_bar_frame(
    df: pd.DataFrame, snapshot_date: str, session: str, coverage: Mapping[str, CoverageEntry] | None
) -> list[str]:
    assert_canonical_bars(df)
    dates = df["snapshot_date"].astype(str)
    if bool((dates != str(snapshot_date)).any()):
        raise ValueError(f"Bar frame carries non-requested snapshot_date for {snapshot_date!r}")
    ts_values = pd.to_numeric(df["ts_hms"], errors="coerce")
    if bool(ts_values.isna().any()):
        raise ValueError("Bar frame carries non-numeric ts_hms")
    for raw in ts_values.astype(int).tolist():
        if not _is_valid_hhmmss(int(raw)):
            raise ValueError(f"Bar frame carries invalid HHMMSS: {raw!r}")
    volumes = pd.to_numeric(df["volume"], errors="coerce")
    if bool(volumes.isna().any()) or bool((volumes < 0).any()):
        raise ValueError("Bar frame carries invalid negative volume")
    symbols = sorted({str(item) for item in df["symbol"].astype(str).tolist()})
    if coverage is not None:
        missing = [item for item in symbols if item not in coverage]
        if missing:
            raise ValueError(f"Bar frame symbols lack coverage certification: {missing}")
        for item in symbols:
            entry = coverage[item]
            if entry.session != str(session):
                raise ValueError(f"Bar coverage session mismatch for {item!r}")
    return symbols


def _require_certified(symbols: list[str], coverage: Mapping[str, CoverageEntry], session: str) -> set[str]:
    replaced: set[str] = set()
    for symbol in symbols:
        entry = coverage[symbol]
        if entry.venue == "UNKNOWN":
            raise ValueError(f"UNKNOWN venue cannot certify {symbol!r}")
        if str(session) == INTRADAY_SESSION_REGULAR_CONSOLIDATED:
            if entry.status == CaptureStatus.NOT_APPLICABLE and str(entry.reason) == _CONSOLIDATED_TAPE_REASON:
                replaced.add(symbol)
                continue
            _preserve_staged_attempt(symbol, entry)
            raise ValueError(f"Only consolidated-tape attempts belong in the consolidated partition: {symbol!r} status={entry.status.value}")
        if entry.status in (CaptureStatus.PARTIAL, CaptureStatus.FAILED, CaptureStatus.UNKNOWN):
            _preserve_staged_attempt(symbol, entry)
            raise ValueError(f"Non-certified attempt cannot replace authoritative partition: {symbol!r} status={entry.status.value}")
        if entry.status in (CaptureStatus.NO_TRADES, CaptureStatus.COMPLETE):
            replaced.add(symbol)
        else:
            _preserve_staged_attempt(symbol, entry)
            raise ValueError(f"Non-certified attempt cannot replace authoritative partition: {symbol!r} status={entry.status.value}")
    for symbol, entry in coverage.items():
        if (
            entry.status == CaptureStatus.NO_TRADES
            and symbol not in replaced
            and len(entry.raw_refs) > 0
            and entry.session == str(session)
            and entry.venue != "UNKNOWN"
        ):
            replaced.add(str(symbol))
    return replaced


def _preserve_staged_attempt(symbol: str, entry: CoverageEntry) -> None:
    root = _capture_root()
    staged = root / "staging" / "intraday" / f"{symbol}-{entry.status.value.lower()}.parquet"
    staged.parent.mkdir(parents=True, exist_ok=True)
    marker = staged.parent / f"{symbol}-{entry.status.value.lower()}.manifest.json"
    marker.write_text(f'{{"symbol": "{symbol}", "status": "{entry.status.value}"}}', encoding="utf-8")


def _partition_row_count(target: Path) -> int:
    if not target.exists():
        return 0
    try:
        handle = pq.ParquetFile(target)
    except Exception as e:
        raise OSError(f"Cannot read existing partition evidence: {target}") from e
    return int(handle.metadata.num_rows)


def _stored_symbol_counts(target: Path, symbols: set[str], batch_rows: int) -> dict[str, int]:
    counts: dict[str, int] = {str(item): 0 for item in symbols}
    if not symbols or not target.exists():
        return counts
    try:
        handle = pq.ParquetFile(target)
    except Exception as e:
        raise OSError(f"Cannot read existing partition evidence: {target}") from e
    try:
        for batch in handle.iter_batches(batch_size=batch_rows, columns=["symbol"]):
            frame = batch.to_pandas()
            if "symbol" not in frame.columns:
                raise ValueError("Legacy partition missing key columns: ['symbol']")
            for symbol, n in frame["symbol"].astype(str).value_counts().items():
                key = str(symbol)
                if key in counts:
                    counts[key] += int(n)
    except ValueError:
        raise
    except Exception as e:
        raise OSError(f"Cannot read existing partition evidence: {target}") from e
    return counts


def _quarantine_shrink_rows(frame: pd.DataFrame, snapshot_date: str, session: str, symbol: str) -> str:
    if len(frame) == 0:
        return "none"
    root = _capture_root()
    qdir = root / "quarantine" / "shrink" / str(snapshot_date) / str(session)
    qdir.mkdir(parents=True, exist_ok=True)
    qpath = qdir / f"{symbol}-{uuid.uuid4().hex}.parquet"
    frame.to_parquet(qpath, index=False)
    return str(qpath)


def _apply_shrink_guard(
    target: Path,
    incoming: pd.DataFrame,
    replaced: set[str],
    batch_rows: int,
    snapshot_date: str,
    session: str,
    allow_shrink_symbols: frozenset[str],
) -> tuple[pd.DataFrame, set[str]]:
    allowed = frozenset(str(item) for item in allow_shrink_symbols)
    if not replaced:
        return incoming, set()
    stored_counts = _stored_symbol_counts(target, set(replaced), batch_rows)
    incoming_counts: dict[str, int] = {str(item): 0 for item in replaced}
    if len(incoming) > 0 and "symbol" in incoming.columns:
        for symbol, n in incoming["symbol"].astype(str).value_counts().items():
            key = str(symbol)
            if key in incoming_counts:
                incoming_counts[key] = int(n)
    rejected: set[str] = set()
    for symbol in replaced:
        key = str(symbol)
        before = int(stored_counts.get(key, 0))
        after = int(incoming_counts.get(key, 0))
        if after < before and key not in allowed:
            rejected.add(key)
    for symbol in sorted(rejected):
        before = int(stored_counts.get(symbol, 0))
        after = int(incoming_counts.get(symbol, 0))
        if len(incoming) > 0 and "symbol" in incoming.columns:
            rows = incoming[incoming["symbol"].astype(str) == symbol].copy()
        else:
            rows = incoming.iloc[0:0]
        quarantine = _quarantine_shrink_rows(rows, snapshot_date, session, symbol)
        logger.warning(
            "[DATA] stage=intraday_replace status=SHRINK_REJECTED date=%s session=%s symbol=%s before=%d after=%d quarantine=%s",
            snapshot_date, session, symbol, before, after, quarantine,
        )
    for symbol in sorted(replaced - rejected):
        before = int(stored_counts.get(str(symbol), 0))
        after = int(incoming_counts.get(str(symbol), 0))
        if after < before and str(symbol) in allowed:
            logger.info(
                "[DATA] stage=intraday_replace status=SHRINK_ALLOWED date=%s session=%s symbol=%s before=%d after=%d",
                snapshot_date, session, symbol, before, after,
            )
    if rejected:
        replaced = {str(item) for item in replaced if str(item) not in rejected}
        if len(incoming) > 0 and "symbol" in incoming.columns:
            incoming = incoming[~incoming["symbol"].astype(str).isin(rejected)].copy()
    return incoming, replaced


def _retain_backup_ref(
    target: Path,
    snapshot_date: str,
    session: str,
    *,
    now_fn: Callable[[], datetime] = lambda: datetime.now(ZoneInfo("Asia/Seoul")),
) -> str:
    root = _capture_root()
    retained_on = now_fn().date().isoformat()
    backup_dir = root / "backups" / "intraday" / str(session) / retained_on
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{target.stem}-{snapshot_date}-pre-{uuid.uuid4().hex}.parquet"
    os.link(target, backup)
    return str(backup)


def _deduplicate_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse exact duplicate bar rows and reject contradictory slots.

    One pass over the frame (not one pandas call per bar slot): a full-session partition holds tens of thousands
    of slots, and a per-slot loop made the commit of a 100-symbol day take ~30 s. Rows with a missing symbol or
    timestamp key carry no slot identity and are dropped, as the former group-by did.

    Raises:
        ValueError: Two different rows share one (symbol, ts_hms) slot; the earliest such slot is named.
    """
    key_cols = ["symbol", "ts_hms"]
    keyed = df.dropna(subset=key_cols)
    distinct = keyed.drop_duplicates(ignore_index=True)
    contradictory = distinct.duplicated(subset=key_cols, keep=False)
    if bool(contradictory.any()):
        first = distinct.loc[contradictory].iloc[0]
        raise ValueError(f"Contradictory bar slot for {first['symbol']!r} ts={first['ts_hms']!r}")
    reconciled = distinct if len(distinct) else df.copy()
    reconciled = reconciled.sort_values(key_cols, kind="stable").reset_index(drop=True)
    return reconciled[list(CANONICAL_BAR_COLUMNS)]


def _collect_legacy_overlap(target: Path, symbols: set[str], batch_rows: int) -> dict[str, pd.DataFrame]:
    collected: dict[str, list[pd.DataFrame]] = {item: [] for item in symbols}
    try:
        handle = pq.ParquetFile(target)
    except Exception as e:
        raise OSError(f"Cannot read existing partition evidence: {target}") from e
    for batch in handle.iter_batches(batch_size=batch_rows):
        frame = batch.to_pandas()
        if "symbol" not in frame.columns:
            raise ValueError("Legacy partition missing key columns: ['symbol']")
        for symbol, group in frame.groupby("symbol"):
            key = str(symbol)
            if key in collected:
                collected[key].append(group.copy())
    return {key: (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()) for key, parts in collected.items()}


def _frames_equal(old: pd.DataFrame, new: pd.DataFrame) -> bool:
    if len(old) != len(new):
        return False
    order = sorted(old.columns)
    left = old[order].sort_values(order, kind="stable").reset_index(drop=True)
    right = new[order].sort_values(order, kind="stable").reset_index(drop=True)

    def _rows(frame: pd.DataFrame) -> list[tuple[str, ...]]:
        return sorted(tuple("∅" if pd.isna(value) else str(value) for value in row) for row in frame.itertuples(index=False, name=None))

    return _rows(left) == _rows(right)


def _check_legacy_unchanged(target: Path, incoming: pd.DataFrame, symbols: set[str], batch_rows: int) -> None:
    overlap = _collect_legacy_overlap(target, symbols, batch_rows)
    for symbol in symbols:
        old = overlap.get(symbol, pd.DataFrame())
        new = incoming[incoming["symbol"].astype(str) == symbol].copy()
        if len(old) == 0:
            continue
        if not _frames_equal(old, new):
            raise ValueError(f"Changed uncertified attempt requires certification: {symbol!r}")


def _promote_null_fields(existing: pa.Schema, incoming: pa.Schema | None) -> pa.Schema:
    """Give all-null legacy columns the incoming concrete type so repaired symbols can be rewritten into them.

    A legacy partition whose optional column was entirely null is stored with the null type, which cannot
    hold the concrete values of a certified re-collection; the column keeps every existing null unchanged.
    """
    if incoming is None:
        return existing
    fields = []
    for field in existing:
        position = incoming.get_field_index(field.name)
        if pa.types.is_null(field.type) and position >= 0 and not pa.types.is_null(incoming.field(position).type):
            field = field.with_type(incoming.field(position).type)
        fields.append(field)
    return pa.schema(fields, metadata=existing.metadata)


def _bounded_symbol_replace(
    target: Path,
    incoming: pd.DataFrame,
    replaced: set[str],
    batch_rows: int,
    snapshot_date: str,
    session: str,
    *,
    sort_output: bool,
    allow_shrink_symbols: frozenset[str] = frozenset(),
) -> int:
    # 지연 임포트: parquet_codec -> panel_integrity -> cost_model -> intraday_store로
    # 되돌아오는 순환 임포트를 모듈 로드 시점에 막기 위해 호출 시점에만 가져온다.
    from src.data.parquet_codec import INTRADAY_COMPRESSION, PARQUET_COMPRESSION_LEVEL

    with exclusive_file_lock(sidecar_lock_path(target), timeout_seconds=_LOCK_TIMEOUT_SECONDS, purpose="partition"):
        before_count = _partition_row_count(target)
        incoming, replaced = _apply_shrink_guard(
            target, incoming, set(replaced), batch_rows, str(snapshot_date), str(session), frozenset(allow_shrink_symbols),
        )
        if not replaced:
            return before_count
        before_symbols: set[str] = set()
        backup_ref = ""
        if target.exists():
            backup_ref = _retain_backup_ref(target, snapshot_date, session)
        staging = target.parent / f".stage-{uuid.uuid4().hex}.parquet"
        kept_rows = 0
        writer: pq.ParquetWriter | None = None
        schema: pa.Schema | None = None
        incoming_table: pa.Table | None = None
        if len(incoming) > 0:
            ordered_incoming = incoming
            if sort_output:
                ordered_incoming = incoming.sort_values(["symbol", "ts_hms"], kind="stable").reset_index(drop=True)
            incoming_table = pa.Table.from_pandas(ordered_incoming, preserve_index=False)
        try:
            if target.exists():
                handle = pq.ParquetFile(target)
                for batch in handle.iter_batches(batch_size=batch_rows):
                    frame = batch.to_pandas()
                    if "symbol" in frame.columns:
                        before_symbols.update({str(item) for item in frame["symbol"].astype(str).tolist()})
                    keep = frame[~frame["symbol"].astype(str).isin(replaced)] if "symbol" in frame.columns else frame
                    if keep.empty:
                        continue
                    table = pa.Table.from_pandas(keep, preserve_index=False)
                    if writer is None:
                        schema = _promote_null_fields(table.schema, incoming_table.schema if incoming_table is not None else None)
                        staging.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(
                            staging, schema, compression=INTRADAY_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL
                        )
                    kept_rows += len(keep)
                    writer.write_table(table.cast(schema))
            if incoming_table is not None:
                table = incoming_table
                if schema is None:
                    schema = table.schema
                if writer is None:
                    staging.parent.mkdir(parents=True, exist_ok=True)
                    writer = pq.ParquetWriter(
                        staging, schema, compression=INTRADAY_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL
                    )
                writer.write_table(table.cast(schema))
            if writer is None:
                if replaced and target.exists() and before_symbols and before_symbols <= replaced:
                    target.unlink()
                    logger.info(
                        "[DATA] stage=intraday_replace date=%s session=%s before_rows=%d after_rows=%d replaced=%s backup=%s",
                        snapshot_date, session, before_count, 0, sorted(replaced), backup_ref,
                    )
                    return 0
                return before_count
            writer.close()
            writer = None
            staged_count = _partition_row_count(staging)
            expected = kept_rows + len(incoming)
            if staged_count != expected:
                raise OSError(f"Staged partition verification failed: expected={expected} staged={staged_count}")
            try:
                os.replace(staging, target)
            except OSError as e:
                raise OSError(f"Partition publication failed: {target}") from e
            after_symbols = (before_symbols - replaced) | set(incoming["symbol"].astype(str).tolist()) if len(incoming) else (before_symbols - replaced)
            logger.info(
                "[DATA] stage=intraday_replace date=%s session=%s before_rows=%d after_rows=%d replaced=%s backup=%s",
                snapshot_date, session, before_count, staged_count, sorted(replaced), backup_ref,
            )
            _ = after_symbols
            return staged_count
        finally:
            if writer is not None:
                writer.close()
            if staging.exists():
                staging.unlink()
        return before_count


def remove_intraday_symbols(bar_interval_minutes: int, snapshot_date: str, session: str, symbols: set[str]) -> int:
    """Drop every bar of the given symbols from one partition (backup retained like any replace).

    Used when stored bars are known to be unusable (e.g. adjusted-basis backfill) and no certified
    replacement exists, so the symbol reverts to "not stored" and is fetched again later.

    Returns:
        Rows remaining in the partition.

    Raises:
        OSError: Existing evidence or staging/publication fails.
    """
    target = intraday_partition_path(bar_interval_minutes, snapshot_date, session)
    if not symbols or not target.exists():
        return _partition_row_count(target)
    empty = pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_BAR_COLUMNS})
    explicit = {str(s) for s in symbols}
    return _bounded_symbol_replace(
        target, empty, explicit, _batch_rows_or_default(None), str(snapshot_date), str(session),
        sort_output=True, allow_shrink_symbols=frozenset(explicit),
    )


def write_intraday_partition(
    df: pd.DataFrame,
    bar_interval_minutes: int,
    snapshot_date: str,
    session: str = "regular",
    *,
    coverage: Mapping[str, CoverageEntry] | None = None,
    batch_rows: int | None = None,
    allow_shrink_symbols: frozenset[str] = frozenset(),
) -> int:
    """Publish complete bar attempts without retaining contaminated earlier ranges.

    Args:
        df: Canonical bars from whole-symbol certified attempts.
        bar_interval_minutes: Declared bar interval.
        snapshot_date: Requested market date.
        session: Verified session partition.
        coverage: Expected membership and bounded task certification.
        batch_rows: Arrow rewrite batch bound.
        allow_shrink_symbols: Symbols an operator explicitly authorises to end with fewer
            stored rows than before. Every other replaced symbol whose incoming row count is
            below its stored count is rejected: its stored rows are kept, its incoming rows are
            quarantined, and a SHRINK_REJECTED event is logged. Certified evidence must never
            lose observations through an automated path.

    Returns:
        Total published rows.

    Raises:
        ValueError: Invalid replacement or contradictory duplicate bars.
        OSError: Existing evidence or staging/publication fails.
    """
    bound = _batch_rows_or_default(batch_rows)
    allowed = frozenset(str(item) for item in allow_shrink_symbols)
    target = intraday_partition_path(bar_interval_minutes, snapshot_date, session)
    if df is None or len(df) == 0:
        if coverage is None:
            return _partition_row_count(target)
        no_trades = {
            str(symbol)
            for symbol, entry in coverage.items()
            if entry.status == CaptureStatus.NO_TRADES
            and entry.session == str(session)
            and entry.venue != "UNKNOWN"
            and len(entry.raw_refs) > 0
        }
        if not no_trades:
            return _partition_row_count(target)
        return _bounded_symbol_replace(target, df, no_trades, bound, str(snapshot_date), str(session), sort_output=True, allow_shrink_symbols=allowed)
    symbols = _validate_bar_frame(df, str(snapshot_date), str(session), coverage)
    reconciled = _deduplicate_bars(df)
    if coverage is None:
        replaced = set(symbols)
        if target.exists():
            _check_legacy_unchanged(target, reconciled, replaced, bound)
        total = _bounded_symbol_replace(target, reconciled, replaced, bound, str(snapshot_date), str(session), sort_output=True, allow_shrink_symbols=allowed)
        log_session_coverage_outliers(reconciled, bar_interval_minutes, str(snapshot_date), str(session))
        logger.info("Wrote intraday partition %s (%d rows)", target, total)
        return total
    replaced = _require_certified(symbols, coverage, str(session))
    total = _bounded_symbol_replace(target, reconciled, replaced, bound, str(snapshot_date), str(session), sort_output=True, allow_shrink_symbols=allowed)
    log_session_coverage_outliers(reconciled, bar_interval_minutes, str(snapshot_date), str(session))
    logger.info("Wrote intraday partition %s (%d rows)", target, total)
    return total


def write_tick_partition(
    df: pd.DataFrame,
    snapshot_date: str,
    session: str = "regular",
    *,
    coverage: Mapping[str, CoverageEntry] | None = None,
    batch_rows: int | None = None,
    allow_shrink_symbols: frozenset[str] = frozenset(),
) -> int:
    """Publish verified symbol attempts without inventing trade identities.

    Same-second same-size events, including identical trades, are distinct
    observations. Safe replacement requires a whole-symbol acquisition contract;
    merging events on lossy market fields destroys multiplicity.

    Args:
        df: Canonical records of complete symbol attempts.
        snapshot_date: Exact market date.
        session: Explicit verified session partition.
        coverage: Per-symbol acquisition and venue/session certification.
        batch_rows: Configured bounded Arrow rewrite batch size.
        allow_shrink_symbols: Symbols an operator explicitly authorises to end with fewer
            stored rows than before. Every other replaced symbol whose incoming row count is
            below its stored count is rejected: its stored rows are kept, its incoming rows are
            quarantined, and a SHRINK_REJECTED event is logged. Certified evidence must never
            lose observations through an automated path.

    Returns:
        Total rows in the newly published partition.

    Raises:
        ValueError: Unsafe replacement, incomplete certification, or bad schema.
        OSError: Existing evidence cannot be read or publication fails.
    """
    bound = _batch_rows_or_default(batch_rows)
    allowed = frozenset(str(item) for item in allow_shrink_symbols)
    target = tick_partition_path(snapshot_date, session)
    if df is None or len(df) == 0:
        if coverage is None:
            return _partition_row_count(target)
        no_trades = {
            str(symbol)
            for symbol, entry in coverage.items()
            if entry.status == CaptureStatus.NO_TRADES
            and entry.session == str(session)
            and entry.venue != "UNKNOWN"
            and len(entry.raw_refs) > 0
        }
        if not no_trades:
            return _partition_row_count(target)
        return _bounded_symbol_replace(target, df, no_trades, bound, str(snapshot_date), str(session), sort_output=False, allow_shrink_symbols=allowed)
    symbols = _validate_tick_frame(df, str(snapshot_date), str(session), coverage)
    if coverage is None:
        replaced = set(symbols)
        if target.exists():
            _check_legacy_unchanged(target, df, replaced, bound)
        total = _bounded_symbol_replace(target, df, replaced, bound, str(snapshot_date), str(session), sort_output=False, allow_shrink_symbols=allowed)
        logger.info("Wrote tick partition %s (%d rows)", target, total)
        return total
    replaced = _require_certified(symbols, coverage, str(session))
    total = _bounded_symbol_replace(target, df, replaced, bound, str(snapshot_date), str(session), sort_output=False, allow_shrink_symbols=allowed)
    logger.info("Wrote tick partition %s (%d rows)", target, total)
    return total


def tick_partition_path(snapshot_date: str, session: str = "regular") -> Path:
    """data/history/intraday/ticks/{session}/{YYYY-MM}/{YYYY-MM-DD}.parquet 경로 산출."""
    _require_known_session(session)
    month = str(snapshot_date)[:7]
    return (
        Path(settings.HISTORY_DIR)
        / "intraday"
        / "ticks"
        / str(session)
        / month
        / f"{snapshot_date}.parquet"
    )

