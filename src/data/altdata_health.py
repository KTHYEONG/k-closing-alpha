"""Altdata capture verdict for exit-code and audit tolerance."""

from __future__ import annotations

from enum import StrEnum

from src.data.capture_contracts import CaptureDataset, CaptureManifest, CaptureStatus


class AltdataVerdict(StrEnum):
    """Terminal outcome class of a finished altdata capture manifest."""

    COMPLETE = "COMPLETE"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


TOLERATED_DEGRADATIONS: frozenset[tuple[CaptureDataset, str]] = frozenset(
    {(CaptureDataset.DISCLOSURE, "quota_exceeded")}
)


def altdata_verdict(manifest: CaptureManifest) -> AltdataVerdict:
    """Classify a finished altdata capture manifest for exit-code and audit purposes.

    ``COMPLETE`` when the manifest is complete. ``DEGRADED`` when the manifest is
    partial and every non-complete coverage entry matches a declared tolerated
    degradation (a source whose failure is an external, non-actionable quota
    condition and whose data no decision path consumes). Everything else is
    ``FAILED``, including a partial manifest with no entries, so an unexplained
    partial result can never be tolerated.

    Args:
        manifest: Terminal (non-PENDING) altdata capture manifest.

    Returns:
        The verdict; the manifest is not modified.
    """
    if manifest.status == CaptureStatus.COMPLETE:
        return AltdataVerdict.COMPLETE
    if manifest.status != CaptureStatus.PARTIAL:
        return AltdataVerdict.FAILED
    pending = [e for e in manifest.entries if e.status != CaptureStatus.COMPLETE]
    if not pending:
        return AltdataVerdict.FAILED
    if all((e.dataset, e.reason) in TOLERATED_DEGRADATIONS for e in pending):
        return AltdataVerdict.DEGRADED
    return AltdataVerdict.FAILED
