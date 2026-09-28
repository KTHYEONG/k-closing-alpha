"""Repair stored KIS per-bar traded values with the normalizer consistency rule."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from src.config.market_session import (
    INTRADAY_SESSION_KRX_AFTERMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_NXT_PREMARKET,
    INTRADAY_SESSION_REGULAR,
)
from src.data.capture_contracts import CaptureDataset, CaptureStatus, CoverageEntry
from src.data.intraday_schema import assert_canonical_bars
from src.data.intraday_store import intraday_partition_path, write_intraday_partition
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

_SESSIONS: tuple[str, ...] = (
    INTRADAY_SESSION_REGULAR,
    INTRADAY_SESSION_NXT_PREMARKET,
    INTRADAY_SESSION_NXT_AFTERMARKET,
    INTRADAY_SESSION_KRX_AFTERMARKET,
)


@dataclass(frozen=True)
class BarValueRepairReport:
    """Outcome of one stored-partition bar-value repair."""

    snapshot_date: str
    session: str
    symbols_repaired: tuple[str, ...]
    rows_repaired: int
    written: bool


def _violation_mask(frame: pd.DataFrame) -> pd.Series:
    vendor = frame["vendor"].astype(str)
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    value = pd.to_numeric(frame["value_krw"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    is_kis = vendor == "kis"
    zero_bad = is_kis & (volume == 0) & (value != 0)
    band_bad = is_kis & (volume > 0) & ((value < low * volume) | (value > high * volume))
    return zero_bad | band_bad


def repair_bar_value_partition(
    snapshot_date: str, session: str, *, bar_interval_minutes: int = 1, apply: bool = False
) -> BarValueRepairReport:
    """Re-apply the KIS per-bar value consistency rule to an already stored 1m partition.

    Partitions written before the normalizer fix hold cumulative carry-in values (e.g. a whole regular
    session's value on the 16:00 KRX aftermarket bar). Raw payloads may already be pruned locally, so the
    repair works on stored rows and applies exactly the normalizer's rule to vendor == "kis" rows only.

    Args:
        snapshot_date: Partition date (YYYY-MM-DD).
        session: Intraday session partition name (e.g. "krx_aftermarket", "regular").
        bar_interval_minutes: Partition interval.
        apply: False reports without writing; True rewrites only the affected symbols.

    Returns:
        Report of repaired symbols/rows and whether a write happened.

    Raises:
        FileNotFoundError: The partition does not exist.
        ValueError: Partition schema is not canonical.
    """
    target = intraday_partition_path(int(bar_interval_minutes), str(snapshot_date), str(session))
    if not target.exists():
        raise FileNotFoundError(f"Bar value repair partition does not exist: {target}")
    frame = pd.read_parquet(target)
    assert_canonical_bars(frame)
    if frame.empty:
        return BarValueRepairReport(str(snapshot_date), str(session), (), 0, False)
    bad = _violation_mask(frame)
    if not bool(bad.any()):
        return BarValueRepairReport(str(snapshot_date), str(session), (), 0, False)
    symbols = tuple(sorted({str(item) for item in frame.loc[bad, "symbol"].astype(str).tolist()}))
    rows = int(bad.sum())
    if not apply:
        return BarValueRepairReport(str(snapshot_date), str(session), symbols, rows, False)
    repaired = frame.copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    repaired.loc[bad, "value_krw"] = (close.loc[bad] * volume.loc[bad]).astype("int64").tolist()
    venue = "KRX" if str(session) in (INTRADAY_SESSION_REGULAR, INTRADAY_SESSION_KRX_AFTERMARKET) else "NXT"
    subset = repaired[repaired["symbol"].astype(str).isin(symbols)].copy()
    coverage = {
        symbol: CoverageEntry(
            symbol=symbol,
            dataset=CaptureDataset.MINUTE_BARS,
            venue=venue,
            session=str(session),
            scheduled_at=None,
            status=CaptureStatus.COMPLETE,
            rows=int((subset["symbol"].astype(str) == symbol).sum()),
            first_event_time=None,
            last_event_time=None,
            reason="repair:bar_value",
            raw_refs=(),
        )
        for symbol in symbols
    }
    write_intraday_partition(
        subset, int(bar_interval_minutes), str(snapshot_date), str(session), coverage=coverage
    )
    return BarValueRepairReport(str(snapshot_date), str(session), symbols, rows, True)


def main() -> None:
    """Repair stored KIS bar values across a date range (dry run unless --apply)."""
    parser = argparse.ArgumentParser(description="Repair stored KIS per-bar traded values.")
    parser.add_argument("--session", required=True, choices=list(_SESSIONS))
    parser.add_argument("--from", dest="from_date", required=True, help="Inclusive repair start (YYYY-MM-DD).")
    parser.add_argument("--to", dest="to_date", required=True, help="Inclusive repair end (YYYY-MM-DD).")
    parser.add_argument("--apply", action="store_true", help="Rewrite affected symbols (default dry run).")
    args = parser.parse_args()
    try:
        start = date.fromisoformat(str(args.from_date))
        end = date.fromisoformat(str(args.to_date))
    except ValueError:
        raise ValueError(f"Invalid --from/--to date: {args.from_date!r}..{args.to_date!r}") from None
    if end < start:
        raise ValueError(f"Invalid date range: {args.from_date!r}..{args.to_date!r}")
    day = start
    while day <= end:
        day_str = day.isoformat()
        try:
            report = repair_bar_value_partition(day_str, str(args.session), apply=bool(args.apply))
        except FileNotFoundError:
            logger.info(
                "[DATA] stage=bar_value_repair date=%s session=%s status=SKIP reason=missing_partition symbols=0 rows=0 written=False",
                day_str,
                args.session,
            )
        else:
            logger.info(
                "[DATA] stage=bar_value_repair date=%s session=%s symbols=%d rows=%d written=%s",
                day_str,
                args.session,
                len(report.symbols_repaired),
                report.rows_repaired,
                report.written,
            )
        day += timedelta(days=1)


if __name__ == "__main__":  # pragma: no cover
    configure_cli_logging()
    main()
