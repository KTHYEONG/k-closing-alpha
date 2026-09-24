"""DartKeyPool invariant guards."""

from __future__ import annotations

import logging

import pytest

from src.backfill.altdata.dart_keys import (
    LEGACY_LABEL,
    PRIMARY_LABEL,
    SECONDARY_LABEL,
    DartCredential,
    DartKeyPool,
    resolve_dart_key_pool,
)
from src.backfill.altdata.ratelimit import DartKeysUnusableError, DartQuotaExhaustedError


def test_pool_priority_order_is_slot_order() -> None:
    """Priority order is slot 1, slot 2, legacy."""
    pool = resolve_dart_key_pool(primary="a", secondary="b", legacy="c")
    assert pool.labels == ("KEY_1", "KEY_2", "LEGACY")
    assert [c.label for c in pool.credentials] == ["KEY_1", "KEY_2", "LEGACY"]
    assert pool.current().label == "KEY_1"


def test_pool_blank_and_duplicate_keys_collapse() -> None:
    """Blank and duplicate keys collapse."""
    pool = resolve_dart_key_pool(primary="a", secondary="", legacy="a")
    assert pool.labels == ("KEY_1",)


def test_pool_empty_is_detectable() -> None:
    """Empty pool is detectable."""
    pool = resolve_dart_key_pool(primary="", secondary="", legacy="")
    assert pool.is_empty() is True
    with pytest.raises(ValueError, match="empty"):
        pool.current()


def test_pool_exhausting_head_promotes_next() -> None:
    """Exhausting the head key promotes the next."""
    pool = DartKeyPool([DartCredential(label="KEY_1", key="a"), DartCredential(label="KEY_2", key="b")])
    pool.mark_exhausted(DartCredential(label="KEY_1", key="a"))
    assert pool.current().label == "KEY_2"


def test_pool_all_exhausted_raises_quota_error() -> None:
    """All exhausted raises quota error."""
    pool = DartKeyPool([DartCredential(label="KEY_1", key="SECRET-A-VALUE"), DartCredential(label="KEY_2", key="SECRET-B-VALUE")])
    pool.mark_exhausted(DartCredential(label="KEY_1", key="SECRET-A-VALUE"))
    pool.mark_exhausted(DartCredential(label="KEY_2", key="SECRET-B-VALUE"))
    with pytest.raises(DartQuotaExhaustedError) as exc_info:
        pool.current()
    text = str(exc_info.value)
    assert "KEY_1" in text and "KEY_2" in text
    assert "SECRET-A-VALUE" not in text and "SECRET-B-VALUE" not in text


def test_pool_rejected_key_blocks_tolerance() -> None:
    """A rejected key blocks tolerance."""
    pool = DartKeyPool([DartCredential(label="KEY_1", key="a"), DartCredential(label="KEY_2", key="b")])
    pool.mark_rejected(DartCredential(label="KEY_1", key="a"), "010")
    pool.mark_exhausted(DartCredential(label="KEY_2", key="b"))
    with pytest.raises(DartKeysUnusableError):
        pool.current()
    with pytest.raises(DartKeysUnusableError) as exc_info:
        pool.current()
    assert "KEY_1=010" in str(exc_info.value)


def test_pool_marks_idempotent_and_thread_safe(caplog) -> None:
    """Marks are idempotent and thread-safe."""
    import threading

    pool = DartKeyPool([DartCredential(label="KEY_1", key="a"), DartCredential(label="KEY_2", key="b")])
    cred = DartCredential(label="KEY_1", key="a")
    with caplog.at_level(logging.WARNING, logger="src.backfill.altdata.dart_keys"):
        threads = [threading.Thread(target=pool.mark_exhausted, args=(cred,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    lines = [r for r in caplog.records if "stage=dart_key" in r.getMessage()]
    assert len(lines) == 1
    assert pool.current().label == "KEY_2"


def test_pool_credential_repr_hides_secret(caplog) -> None:
    """Credential repr hides the secret."""
    cred = DartCredential(label="KEY_1", key="SECRETVALUE")
    assert "SECRETVALUE" not in repr(cred)
    pool = DartKeyPool([cred, DartCredential(label="KEY_2", key="b")])
    with caplog.at_level(logging.WARNING, logger="src.backfill.altdata.dart_keys"):
        pool.mark_exhausted(cred)
    assert "SECRETVALUE" not in caplog.text
    assert PRIMARY_LABEL == "KEY_1" and SECONDARY_LABEL == "KEY_2" and LEGACY_LABEL == "LEGACY"


def test_pool_unknown_credential_marks_are_noop(caplog) -> None:
    """Unknown credential marks are ignored."""
    pool = DartKeyPool([DartCredential(label="KEY_1", key="a")])
    pool.mark_exhausted(DartCredential(label="GHOST", key="ghost-value"))
    pool.mark_rejected(DartCredential(label="GHOST", key="ghost-value"), "010")
    assert pool.current().label == "KEY_1"
    assert "GHOST" not in caplog.text
