"""Coverage summary of the Toss ledger by year and outcome.

Groups the latest ledger row per (snapshot_date, session, symbol) key by
calendar year and outcome: COMPLETE, the consolidated-tape rejection, the
stock-not-found absence (delisted names are not served), or FAILED/EXHAUSTED
with its reason. Survivorship stays visible instead of folding into
`not_fetched`: delisted symbols appear under `toss_stock_not_found`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.backfill.intraday.extended_session_backfill import ExtendedBackfillLedger
from src.backfill.intraday.toss_regular_backfill import default_toss_ledger_path
from src.data.io_utils import read_existing_parquet

logger = logging.getLogger(__name__)

LEDGER_REPORT_COLUMNS: tuple[str, ...] = ("year", "outcome", "n")

_STOCK_NOT_FOUND_OUTCOME: str = "toss_stock_not_found"
_CONSOLIDATED_OUTCOME: str = "toss_consolidated_tape"


def _classify_outcome(status: object, reason: object) -> str:
    status_text = str(status)
    reason_text = str(reason)
    if status_text == "COMPLETE":
        return "COMPLETE"
    if reason_text == _CONSOLIDATED_OUTCOME:
        return _CONSOLIDATED_OUTCOME
    if reason_text in ("toss_stock_not_found", "toss_stock_not_found_cached"):
        return _STOCK_NOT_FOUND_OUTCOME
    return f"{status_text}:{reason_text}"


def _empty_report() -> pd.DataFrame:
    out = pd.DataFrame({
        "year": pd.Series(dtype="str"),
        "outcome": pd.Series(dtype="str"),
        "n": pd.Series(dtype="int64"),
    })
    return out[list(LEDGER_REPORT_COLUMNS)]


def summarize_toss_ledger(ledger: ExtendedBackfillLedger, *, calendar_years: bool = True) -> pd.DataFrame:
    """Summarize the Toss ledger by year and outcome.

    Args:
        ledger: Durable Toss outcome log; read-only here.
        calendar_years: Group by calendar year; otherwise by snapshot date.

    Returns:
        Frame with columns year, outcome, n; empty with declared columns when
        the ledger file is absent.
    """
    frame = read_existing_parquet(ledger.path)
    if frame.empty or "snapshot_date" not in frame.columns:
        return _empty_report()
    work = frame.copy()
    for col in ("session", "symbol", "status", "reason"):
        if col not in work.columns:
            work[col] = ""
    latest = work.drop_duplicates(subset=["snapshot_date", "session", "symbol"], keep="last")
    days = latest["snapshot_date"].astype(str)
    key = days.str[:4] if calendar_years else days
    outcomes = [
        _classify_outcome(status, reason)
        for status, reason in zip(
            latest["status"].astype(str).tolist(), latest["reason"].astype(str).tolist(), strict=True
        )
    ]
    grouped = (
        pd.DataFrame({"year": key.tolist(), "outcome": outcomes})
        .groupby(["year", "outcome"], sort=True)
        .size()
        .reset_index(name="n")
    )
    grouped["n"] = grouped["n"].astype("int64")
    return grouped[list(LEDGER_REPORT_COLUMNS)].sort_values(["year", "outcome"], kind="stable").reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the Toss ledger coverage table; --out writes parquet/CSV under scratch/."""
    parser = argparse.ArgumentParser(description="Summarize the Toss backfill ledger by year and outcome.")
    parser.add_argument("--ledger", default=None, help="Ledger parquet path (default the history-tree location).")
    parser.add_argument("--out", default=None, help="Output path (.parquet or .csv); default prints only.")
    args = parser.parse_args(argv)
    ledger = ExtendedBackfillLedger(Path(args.ledger) if args.ledger else default_toss_ledger_path())
    report = summarize_toss_ledger(ledger)
    print(report.to_string(index=False))
    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".csv":
            report.to_csv(target, index=False)
        else:
            report.to_parquet(target, index=False)
        logger.info("[DATA] stage=toss_ledger_report status=DONE rows=%d out=%s", len(report), target)
    else:
        logger.info("[DATA] stage=toss_ledger_report status=DONE rows=%d", len(report))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry; logic covered via unit scenarios
    from src.utils.cli_logging import configure_cli_logging

    configure_cli_logging()
    raise SystemExit(main())
