"""Invariant guards for the Toss ledger coverage report."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.backfill.intraday.extended_session_backfill import ExtendedBackfillLedger
from src.backfill.intraday.toss_ledger_report import LEDGER_REPORT_COLUMNS, summarize_toss_ledger
from src.data.capture_contracts import SEOUL, ArtifactRef, CaptureDataset, CaptureStatus, CoverageEntry

_ATTEMPTED = datetime(2026, 9, 29, 3, 0, tzinfo=SEOUL)
_REF = ArtifactRef(path="capture/evidence.parquet", sha256="abc", bytes=10)


def _entry(symbol: str, status: CaptureStatus, reason: str) -> CoverageEntry:
    needs_evidence = status in (CaptureStatus.NO_TRADES, CaptureStatus.NOT_APPLICABLE)
    return CoverageEntry(
        symbol=symbol, dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="regular",
        scheduled_at=None, status=status, rows=0,
        first_event_time=None, last_event_time=None, reason=reason,
        raw_refs=(_REF,) if needs_evidence else (),
    )


def _ledger_with_rows(tmp_path) -> ExtendedBackfillLedger:
    ledger = ExtendedBackfillLedger(tmp_path / "toss_regular.parquet")
    ledger.record(
        "2024-02-28", "regular",
        [_entry("005930", CaptureStatus.COMPLETE, "toss_regular:390")],
        run_id="run-1", attempted_at=_ATTEMPTED, vendor="toss",
        price_bases={"005930": "toss_raw"},
    )
    ledger.record(
        "2024-02-28", "regular",
        [_entry("000660", CaptureStatus.NOT_APPLICABLE, "toss_consolidated_tape")],
        run_id="run-1", attempted_at=_ATTEMPTED, vendor="toss",
    )
    ledger.record(
        "2025-03-04", "regular",
        [
            _entry("000660", CaptureStatus.NOT_APPLICABLE, "toss_consolidated_tape"),
            _entry("999999", CaptureStatus.NOT_APPLICABLE, "toss_stock_not_found"),
            _entry("005380", CaptureStatus.FAILED, "transport:boom"),
        ],
        run_id="run-2", attempted_at=_ATTEMPTED, vendor="toss",
    )
    return ledger


def test_per_year_outcome_counts_reconcile_to_distinct_keys(tmp_path) -> None:
    ledger = _ledger_with_rows(tmp_path)
    report = summarize_toss_ledger(ledger)
    assert list(report.columns) == list(LEDGER_REPORT_COLUMNS)
    stored = pd.read_parquet(tmp_path / "toss_regular.parquet")
    latest = stored.drop_duplicates(subset=["snapshot_date", "session", "symbol"], keep="last")
    latest["year"] = latest["snapshot_date"].astype(str).str[:4]
    for year, group in latest.groupby("year"):
        assert int(report[report["year"] == year]["n"].sum()) == len(group)


def test_missing_ledger_yields_empty_frame_with_declared_columns(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "absent.parquet")
    report = summarize_toss_ledger(ledger)
    assert list(report.columns) == list(LEDGER_REPORT_COLUMNS)
    assert len(report) == 0


def test_survivorship_stays_visible_under_stock_not_found(tmp_path) -> None:
    ledger = _ledger_with_rows(tmp_path)
    ledger.record_cached_absent(
        "2025-03-05", "regular", ["999999"], reason="toss_stock_not_found_cached",
        run_id="run-3", attempted_at=_ATTEMPTED, vendor="toss",
    )
    report = summarize_toss_ledger(ledger)
    not_found = report[report["outcome"] == "toss_stock_not_found"]["n"].sum()
    assert int(not_found) == 2
    assert "toss_stock_not_found_cached" not in set(report["outcome"].tolist())


def test_main_prints_table_and_writes_out(tmp_path, capsys) -> None:
    from src.backfill.intraday.toss_ledger_report import main

    ledger = _ledger_with_rows(tmp_path)
    out = tmp_path / "scratch" / "ledger_report.parquet"
    assert main(["--ledger", str(tmp_path / "toss_regular.parquet"), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "COMPLETE" in printed
    assert len(pd.read_parquet(out)) == len(summarize_toss_ledger(ledger))
