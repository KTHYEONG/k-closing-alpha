"""Intraday 분봉 날짜 파티션 저장소 (date-partitioned parquet)."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import settings
from src.data.capture_contracts import CaptureStatus, CoverageEntry
from src.data.capture_store import resolve_capture_root as _capture_root
from src.data.intraday_schema import CANONICAL_BAR_COLUMNS, assert_canonical_bars, assert_canonical_ticks
from src.utils.file_lock import DEFAULT_LOCK_TIMEOUT_SECONDS, exclusive_file_lock, sidecar_lock_path

logger = logging.getLogger(__name__)

__all__ = ["intraday_partition_path", "log_session_coverage_outliers", "tick_partition_path", "write_intraday_partition", "write_tick_partition"]

_LOCK_TIMEOUT_SECONDS = DEFAULT_LOCK_TIMEOUT_SECONDS


def intraday_partition_path(bar_interval_minutes: int, snapshot_date: str, session: str) -> Path:
    """data/history/intraday/{interval}m/{session}/{YYYY-MM}/{YYYY-MM-DD}.parquet 경로 산출."""
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


def _retain_backup_ref(target: Path, snapshot_date: str, session: str) -> str:
    root = _capture_root()
    backup_dir = root / "backups" / "intraday" / str(session) / str(snapshot_date)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{target.stem}-pre-{uuid.uuid4().hex}.parquet"
    os.link(target, backup)
    return str(backup)


def _deduplicate_bars(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["symbol", "ts_hms"]
    grouped = df.groupby(key_cols, sort=False)
    rows: list[pd.DataFrame] = []
    for _, group in grouped:
        distinct = group.drop_duplicates(ignore_index=True)
        if len(distinct) > 1:
            raise ValueError(f"Contradictory bar slot for {group.iloc[0]['symbol']!r} ts={group.iloc[0]['ts_hms']!r}")
        rows.append(distinct.iloc[[0]])
    reconciled = pd.concat(rows, ignore_index=True) if rows else df.copy()
    reconciled = reconciled.sort_values(["symbol", "ts_hms"], kind="stable").reset_index(drop=True)
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


def _bounded_symbol_replace(
    target: Path,
    incoming: pd.DataFrame,
    replaced: set[str],
    batch_rows: int,
    snapshot_date: str,
    session: str,
    *,
    sort_output: bool,
) -> int:
    # 지연 임포트: parquet_codec -> panel_integrity -> cost_model -> intraday_store로
    # 되돌아오는 순환 임포트를 모듈 로드 시점에 막기 위해 호출 시점에만 가져온다.
    from src.data.parquet_codec import INTRADAY_COMPRESSION, PARQUET_COMPRESSION_LEVEL

    with exclusive_file_lock(sidecar_lock_path(target), timeout_seconds=_LOCK_TIMEOUT_SECONDS, purpose="partition"):
        before_count = _partition_row_count(target)
        before_symbols: set[str] = set()
        backup_ref = ""
        if target.exists():
            backup_ref = _retain_backup_ref(target, snapshot_date, session)
        staging = target.parent / f".stage-{uuid.uuid4().hex}.parquet"
        kept_rows = 0
        writer: pq.ParquetWriter | None = None
        schema: pa.Schema | None = None
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
                        schema = table.schema
                        staging.parent.mkdir(parents=True, exist_ok=True)
                        writer = pq.ParquetWriter(
                            staging, schema, compression=INTRADAY_COMPRESSION, compression_level=PARQUET_COMPRESSION_LEVEL
                        )
                    kept_rows += len(keep)
                    writer.write_table(table.cast(schema))
            if len(incoming) > 0:
                ordered = incoming
                if sort_output:
                    ordered = incoming.sort_values(["symbol", "ts_hms"], kind="stable").reset_index(drop=True)
                table = pa.Table.from_pandas(ordered, preserve_index=False)
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


def write_intraday_partition(
    df: pd.DataFrame,
    bar_interval_minutes: int,
    snapshot_date: str,
    session: str = "regular",
    *,
    coverage: Mapping[str, CoverageEntry] | None = None,
    batch_rows: int | None = None,
) -> int:
    """Publish complete bar attempts without retaining contaminated earlier ranges.

    Args:
        df: Canonical bars from whole-symbol certified attempts.
        bar_interval_minutes: Declared bar interval.
        snapshot_date: Requested market date.
        session: Verified session partition.
        coverage: Expected membership and bounded task certification.
        batch_rows: Arrow rewrite batch bound.

    Returns:
        Total published rows.

    Raises:
        ValueError: Invalid replacement or contradictory duplicate bars.
        OSError: Existing evidence or staging/publication fails.
    """
    bound = _batch_rows_or_default(batch_rows)
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
        return _bounded_symbol_replace(target, df, no_trades, bound, str(snapshot_date), str(session), sort_output=True)
    symbols = _validate_bar_frame(df, str(snapshot_date), str(session), coverage)
    reconciled = _deduplicate_bars(df)
    if coverage is None:
        replaced = set(symbols)
        if target.exists():
            _check_legacy_unchanged(target, reconciled, replaced, bound)
        total = _bounded_symbol_replace(target, reconciled, replaced, bound, str(snapshot_date), str(session), sort_output=True)
        log_session_coverage_outliers(reconciled, bar_interval_minutes, str(snapshot_date), str(session))
        logger.info("Wrote intraday partition %s (%d rows)", target, total)
        return total
    replaced = _require_certified(symbols, coverage, str(session))
    total = _bounded_symbol_replace(target, reconciled, replaced, bound, str(snapshot_date), str(session), sort_output=True)
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

    Returns:
        Total rows in the newly published partition.

    Raises:
        ValueError: Unsafe replacement, incomplete certification, or bad schema.
        OSError: Existing evidence cannot be read or publication fails.
    """
    bound = _batch_rows_or_default(batch_rows)
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
        return _bounded_symbol_replace(target, df, no_trades, bound, str(snapshot_date), str(session), sort_output=False)
    symbols = _validate_tick_frame(df, str(snapshot_date), str(session), coverage)
    if coverage is None:
        replaced = set(symbols)
        if target.exists():
            _check_legacy_unchanged(target, df, replaced, bound)
        total = _bounded_symbol_replace(target, df, replaced, bound, str(snapshot_date), str(session), sort_output=False)
        logger.info("Wrote tick partition %s (%d rows)", target, total)
        return total
    replaced = _require_certified(symbols, coverage, str(session))
    total = _bounded_symbol_replace(target, df, replaced, bound, str(snapshot_date), str(session), sort_output=False)
    logger.info("Wrote tick partition %s (%d rows)", target, total)
    return total


def tick_partition_path(snapshot_date: str, session: str = "regular") -> Path:
    """data/history/intraday/ticks/{session}/{YYYY-MM}/{YYYY-MM-DD}.parquet 경로 산출."""
    month = str(snapshot_date)[:7]
    return (
        Path(settings.HISTORY_DIR)
        / "intraday"
        / "ticks"
        / str(session)
        / month
        / f"{snapshot_date}.parquet"
    )

