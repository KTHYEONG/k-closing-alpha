"""Terminal disposition of uncertifiable tape needs."""

from __future__ import annotations

from enum import StrEnum


class TapeDisposition(StrEnum):
    """Retryable needs stay loud; unrecoverable ones are informational facts."""

    RETRYABLE = "RETRYABLE"
    UNRECOVERABLE = "UNRECOVERABLE"


def _reason_prefix(reason: str) -> str:
    return reason.split(":")[0].strip()


def _has_exact_total_none(reason: str) -> bool:
    return any(field.strip() == "total=None" for field in reason.split(":")[1:])


def is_uncertifiable_reason(reason: str) -> bool:
    """True when a tape outcome reason proves the vendor cannot certify the day however often it is retried: `day_not_on_tape`, or `tape_total_mismatch` whose `total=None` (no vendor total to compare). A `tape_total_mismatch` with a numeric total, `uncertified_venue`, empty or unknown reasons are retryable (transient page loss or configuration)."""
    if not reason or not reason.strip():
        return False
    prefix = _reason_prefix(reason)
    if prefix == "day_not_on_tape":
        return True
    if prefix == "tape_total_mismatch":
        return _has_exact_total_none(reason)
    return False


def decide_tape_disposition(*, attempts: int, latest_reason: str, min_attempts: int) -> TapeDisposition:
    """UNRECOVERABLE only when `attempts >= min_attempts` and the latest reason is uncertifiable. `attempts` counts consecutive trailing non-settled ledger records for the key. Raises ValueError for `min_attempts < 1`."""
    if int(min_attempts) < 1:
        raise ValueError(f"min_attempts must be >= 1: {min_attempts!r}")
    if int(attempts) >= int(min_attempts) and is_uncertifiable_reason(latest_reason):
        return TapeDisposition.UNRECOVERABLE
    return TapeDisposition.RETRYABLE
