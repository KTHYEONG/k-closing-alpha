"""Intraday 분봉 날짜 파티션 저장소 (date-partitioned parquet)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import settings
from src.data.intraday_schema import assert_canonical_bars, assert_canonical_ticks
from src.data.io_utils import atomic_write_parquet

logger = logging.getLogger(__name__)

__all__ = ["intraday_partition_path", "log_session_coverage_outliers", "merge_partition_frame", "read_intraday_range", "tick_partition_path", "write_intraday_partition", "write_tick_partition"]


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


def merge_partition_frame(new_df: pd.DataFrame, target: Path, key_cols: tuple[str, ...]) -> pd.DataFrame:
    """기존 파티션과 신규 프레임을 키 기준 병합한다 (new wins, 키 정렬)."""
    try:
        existing = pd.read_parquet(target) if target.exists() else pd.DataFrame()
    except Exception as e:
        logger.warning("[DATA] Failed to read existing partition %s; writing new only: %s", target, e)
        existing = pd.DataFrame()
    if existing is None or len(existing) == 0:
        merged = new_df.copy()
    else:
        missing_keys = [k for k in key_cols if k not in existing.columns]
        if missing_keys:
            raise ValueError(f"Legacy partition missing key columns: {missing_keys}")
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=list(key_cols), keep="last")
    merged = merged.sort_values(list(key_cols), kind="stable").reset_index(drop=True)
    if "symbol" in merged.columns and existing is not None and len(existing) > 0 and "symbol" in existing.columns:
        before = set(existing["symbol"].astype(str).unique().tolist())
        after = set(merged["symbol"].astype(str).unique().tolist())
        if not before.issubset(after):
            raise ValueError(f"Partition write would reduce symbol coverage: lost={sorted(before - after)}")
    return merged


def log_session_coverage_outliers(
    merged: pd.DataFrame, bar_interval_minutes: int, snapshot_date: str, session: str, *, min_peer_ratio: float = 0.8
) -> dict[str, int]:
    """파티션 병합 후 종목별 봉 수를 같은 배치의 peer 최댓값과 비교해 저조한 종목을 로그로 남긴다.

    벤더 응답이 성공(예외 없음)이었어도 특정 종목만 세션 일부만 수집된 경우
    (부분 수집, 네트워크 중단, 혹은 실제로 희소하게 거래되는 종목) 현재는 어떤
    진단도 남기지 않는다. peer_max 대비 min_peer_ratio 미만인 종목을 WARNING
    으로 남겨 가시성만 확보한다 -- 자동 재수집/차단은 하지 않는다. 희소유동성
    종목의 정상적으로 낮은 봉수와 실제 수집 실패를 이 함수만으로는 구분할 수
    없으므로(둘 다 같은 신호를 낸다), 조치는 로그를 본 사람의 판단에 맡긴다.

    Args:
        merged: write_intraday_partition이 병합해 실제로 쓰는 최종 프레임.
        bar_interval_minutes: 파티션의 봉 간격(로그 컨텍스트용).
        snapshot_date: 파티션 날짜(로그 컨텍스트용).
        session: 세션 태그(로그 컨텍스트용).
        min_peer_ratio: peer 최댓값 대비 이 비율 미만이면 저조로 표식(strict less-than).

    Returns:
        {"n_symbols": 전체 종목수, "n_low_coverage": 저조 종목수}.
    """
    if merged.empty or "symbol" not in merged.columns:
        return {"n_symbols": 0, "n_low_coverage": 0}
    counts = merged.groupby("symbol").size()
    peer_max = int(counts.max())
    low = counts[counts < peer_max * float(min_peer_ratio)]
    if len(low):
        logger.warning(
            "[DATA] stage=session_coverage bar_interval=%dm date=%s session=%s peer_max=%d n_low=%d symbols=%s",
            bar_interval_minutes, snapshot_date, session, peer_max, len(low), sorted(low.index.tolist())[:20],
        )
    return {"n_symbols": len(counts), "n_low_coverage": len(low)}


def write_intraday_partition(df: pd.DataFrame, bar_interval_minutes: int, snapshot_date: str, session: str) -> int:
    """정규 바 파티션을 게이트 검증 후 병합 저장한다. 빈 df는 0 반환 no-op."""
    if df is None or df.empty:
        return 0
    assert_canonical_bars(df)
    target = intraday_partition_path(bar_interval_minutes, snapshot_date, session)
    merged = merge_partition_frame(df, target, ("symbol", "ts_hms"))
    atomic_write_parquet(merged, target)
    log_session_coverage_outliers(merged, bar_interval_minutes, snapshot_date, session)
    logger.info("Wrote intraday partition %s (%d rows)", target, len(merged))
    return len(merged)


def write_tick_partition(df: pd.DataFrame, snapshot_date: str, session: str = "regular") -> int:
    """틱 파티션을 게이트 검증 후 병합 저장한다. 빈 df는 0 반환 no-op."""
    if df is None or df.empty:
        return 0
    assert_canonical_ticks(df)
    target = tick_partition_path(snapshot_date, session)
    merged = merge_partition_frame(df, target, ("symbol", "ts_hms", "volume"))
    atomic_write_parquet(merged, target)
    logger.info("Wrote tick partition %s (%d rows)", target, len(merged))
    return len(merged)


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


def read_intraday_range(
    bar_interval_minutes: int, start_date: str, end_date: str, session: str = "regular"
) -> pd.DataFrame:
    """날짜 범위에 해당하는 파티션 파일만 글롭하여 concat. 대상 없으면 빈 DataFrame."""
    base = (
        Path(settings.HISTORY_DIR)
        / "intraday"
        / f"{int(bar_interval_minutes)}m"
        / str(session)
    )
    if not base.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(base.rglob("*.parquet")):
        date_str = path.stem
        if start_date <= date_str <= end_date:
            try:
                frames.append(pd.read_parquet(path))
            except Exception as e:
                logger.warning("Failed to read intraday partition %s: %s", path, e)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
