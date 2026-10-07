"""Invariant guards for terminal tape disposition (pure classification)."""

from __future__ import annotations

import pytest

from src.backfill.intraday.tape_disposition import (
    TapeDisposition,
    decide_tape_disposition,
    is_uncertifiable_reason,
)


def test_uncertifiable_reasons() -> None:
    assert is_uncertifiable_reason("day_not_on_tape") is True
    assert is_uncertifiable_reason("tape_total_mismatch:received=23:total=None") is True


def test_numeric_total_mismatch_stays_retryable() -> None:
    assert is_uncertifiable_reason("tape_total_mismatch:received=23:total=24") is False
    assert is_uncertifiable_reason("uncertified_venue") is False


def test_substring_trap() -> None:
    assert is_uncertifiable_reason("tape_total_mismatch:received=23:note=xtotal=Noney") is False
    assert is_uncertifiable_reason("walk_incomplete:total=None") is False


def test_attempt_threshold() -> None:
    assert (
        decide_tape_disposition(attempts=2, latest_reason="day_not_on_tape", min_attempts=3)
        is TapeDisposition.RETRYABLE
    )
    assert (
        decide_tape_disposition(attempts=3, latest_reason="day_not_on_tape", min_attempts=3)
        is TapeDisposition.UNRECOVERABLE
    )


def test_unknown_reason_never_terminal() -> None:
    assert decide_tape_disposition(attempts=99, latest_reason="", min_attempts=3) is TapeDisposition.RETRYABLE
    assert (
        decide_tape_disposition(attempts=99, latest_reason="walk_incomplete:tape_end", min_attempts=3)
        is TapeDisposition.RETRYABLE
    )


def test_invalid_min_attempts() -> None:
    with pytest.raises(ValueError, match="min_attempts"):
        decide_tape_disposition(attempts=5, latest_reason="day_not_on_tape", min_attempts=0)
