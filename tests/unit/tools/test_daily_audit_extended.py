"""Structural exhaustion remains visible without raising audit issues."""

from datetime import date, datetime

from src.backfill.intraday.extended_session_backfill import ExtendedBackfillLedger
from src.config.collection import CollectionSettings
from src.data.capture_contracts import SEOUL, CaptureDataset, CaptureStatus, CoverageEntry
from src.tools.daily_audit import audit_extended_exhausted_detail


def test_structural_exhaustion_remains_in_audit_info(tmp_path) -> None:
    ledger = ExtendedBackfillLedger(tmp_path / "ledger.parquet")
    entry = CoverageEntry(
        symbol="117670", dataset=CaptureDataset.MINUTE_BARS, venue="KRX", session="krx_aftermarket",
        scheduled_at=None, status=CaptureStatus.FAILED, rows=0,
        first_event_time=None, last_event_time=None, reason="price_basis_no_raw_source", raw_refs=(),
    )
    ledger.record(
        "2026-09-17", "krx_aftermarket", [entry], run_id="structural",
        attempted_at=datetime(2026, 10, 22, 23, 6, tzinfo=SEOUL),
    )
    audit = audit_extended_exhausted_detail(
        date(2026, 10, 23), ledger_path=ledger.path, profile=CollectionSettings(),
    )
    assert "intraday:extended_exhausted_reasons:price_basis_no_raw_source=1" in audit.info_lines
    assert "intraday:extended_exhausted_new:1" in audit.info_lines
    assert audit.issues == ()
