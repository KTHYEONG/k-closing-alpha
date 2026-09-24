"""AltdataVerdict invariant guards."""

from __future__ import annotations

from datetime import date, datetime

from src.data.altdata_health import AltdataVerdict, TOLERATED_DEGRADATIONS, altdata_verdict
from src.data.capture_contracts import (
    CaptureContext,
    CaptureDataset,
    CaptureManifest,
    CaptureStatus,
    CoverageEntry,
    SEOUL,
)


def _context(run_id: str = "run-1") -> CaptureContext:
    return CaptureContext(
        trading_date=date(2026, 9, 18),
        run_id=run_id,
        dataset=CaptureDataset.SHORTING,
        vendor="owner-local",
        endpoint="altdata-backfill",
        symbol=None,
        venue="KRX",
        session="regular",
        capture_reason="altdata-backfill",
        cohort_id=None,
        scheduled_at=None,
    )


def _entry(dataset: CaptureDataset, status: CaptureStatus, reason: str) -> CoverageEntry:
    return CoverageEntry(
        symbol=None,
        dataset=dataset,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=status,
        rows=1 if status == CaptureStatus.COMPLETE else 0,
        first_event_time=None,
        last_event_time=None,
        reason=reason,
        raw_refs=(),
    )


def _manifest(status: CaptureStatus, entries: tuple[CoverageEntry, ...]) -> CaptureManifest:
    return CaptureManifest(
        schema_version=1,
        context=_context(),
        cohort=None,
        completed_at=datetime(2026, 9, 18, 21, 40, tzinfo=SEOUL),
        entries=entries,
        artifacts=(),
        status=status,
    )


def test_altdata_verdict_complete_manifest_is_complete() -> None:
    """Complete manifest is complete."""
    manifest = _manifest(CaptureStatus.COMPLETE, (_entry(CaptureDataset.SHORTING, CaptureStatus.COMPLETE, "ok"),))
    assert altdata_verdict(manifest) == AltdataVerdict.COMPLETE


def test_altdata_verdict_disclosure_quota_alone_is_degraded() -> None:
    """Disclosure quota alone is degraded."""
    assert frozenset({(CaptureDataset.DISCLOSURE, "quota_exceeded")}) == TOLERATED_DEGRADATIONS
    manifest = _manifest(
        CaptureStatus.PARTIAL,
        (
            _entry(CaptureDataset.SHORTING, CaptureStatus.COMPLETE, "ok"),
            _entry(CaptureDataset.DISCLOSURE, CaptureStatus.FAILED, "quota_exceeded"),
        ),
    )
    assert altdata_verdict(manifest) == AltdataVerdict.DEGRADED


def test_altdata_verdict_other_failed_source_stays_failed() -> None:
    """Other failed source stays failed."""
    manifest = _manifest(
        CaptureStatus.PARTIAL,
        (
            _entry(CaptureDataset.DISCLOSURE, CaptureStatus.FAILED, "quota_exceeded"),
            _entry(CaptureDataset.CREDIT_BALANCE, CaptureStatus.FAILED, "vendor_failure"),
        ),
    )
    assert altdata_verdict(manifest) == AltdataVerdict.FAILED


def test_altdata_verdict_disclosure_other_reason_stays_failed() -> None:
    """Disclosure with another reason stays failed."""
    manifest = _manifest(
        CaptureStatus.PARTIAL, (_entry(CaptureDataset.DISCLOSURE, CaptureStatus.FAILED, "unavailable"),)
    )
    assert altdata_verdict(manifest) == AltdataVerdict.FAILED


def test_altdata_verdict_unexplained_partial_is_failed() -> None:
    """Unexplained partial is failed."""
    assert altdata_verdict(_manifest(CaptureStatus.PARTIAL, ())) == AltdataVerdict.FAILED


def test_altdata_verdict_failed_overall_is_failed() -> None:
    """Failed overall manifest is failed."""
    manifest = _manifest(
        CaptureStatus.FAILED,
        (_entry(CaptureDataset.DISCLOSURE, CaptureStatus.FAILED, "quota_exceeded"),),
    )
    assert altdata_verdict(manifest) == AltdataVerdict.FAILED


def test_altdata_verdict_input_is_not_mutated() -> None:
    """Input is not mutated."""
    manifest = _manifest(
        CaptureStatus.PARTIAL,
        (_entry(CaptureDataset.DISCLOSURE, CaptureStatus.FAILED, "quota_exceeded"),),
    )
    snapshot = manifest.model_dump()
    altdata_verdict(manifest)
    assert manifest.model_dump() == snapshot
