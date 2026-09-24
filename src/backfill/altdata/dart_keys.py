"""Ordered OpenDART key pool with per-process failover state."""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.backfill.altdata.ratelimit import DartKeysUnusableError, DartQuotaExhaustedError

logger = logging.getLogger(__name__)

PRIMARY_LABEL: str = "KEY_1"
SECONDARY_LABEL: str = "KEY_2"
LEGACY_LABEL: str = "LEGACY"


@dataclass(frozen=True)
class DartCredential:
    """One OpenDART API key with a log-safe label.

    Attributes:
        label: Stable identifier used in logs and manifests instead of the key.
        key: The secret; excluded from ``repr`` so it cannot leak through logging.
    """

    label: str
    key: str = field(repr=False)


class DartKeyPool:
    """Ordered, thread-safe set of OpenDART keys with per-process usability state.

    Page fetching runs on worker threads, so state transitions are serialized.
    A key becomes unusable for the remainder of the process once the API reports
    its daily quota exhausted or rejects it as a configuration error; quota
    resets daily and every scheduled run is a fresh process.

    Args:
        credentials: Keys in priority order. Blank keys are dropped and repeated
            key values collapse to the first occurrence.
    """

    def __init__(self, credentials: Sequence[DartCredential]) -> None:
        seen_values: set[str] = set()
        ordered: list[DartCredential] = []
        for cred in credentials:
            if not cred.key.strip():
                continue
            if cred.key in seen_values:
                continue
            seen_values.add(cred.key)
            ordered.append(cred)
        self._credentials: tuple[DartCredential, ...] = tuple(ordered)
        self._lock = threading.Lock()
        self._exhausted: set[str] = set()
        self._rejected: dict[str, str] = {}

    @property
    def credentials(self) -> tuple[DartCredential, ...]:
        """Keys in priority order."""
        return self._credentials

    @property
    def labels(self) -> tuple[str, ...]:
        """Labels in priority order."""
        return tuple(c.label for c in self._credentials)

    def is_empty(self) -> bool:
        """Return True when the pool holds no non-blank key."""
        return len(self._credentials) == 0

    def _usable_labels(self) -> list[str]:
        return [c.label for c in self._credentials if c.label not in self._exhausted and c.label not in self._rejected]

    def current(self) -> DartCredential:
        """Return the highest-priority usable credential.

        Raises:
            DartQuotaExhaustedError: Every key is exhausted by quota.
            DartKeysUnusableError: No key is usable and at least one was rejected.
            ValueError: The pool is empty.
        """
        with self._lock:
            if not self._credentials:
                raise ValueError("DART key pool is empty")
            for cred in self._credentials:
                if cred.label not in self._exhausted and cred.label not in self._rejected:
                    return cred
            if self._rejected:
                pairs = ", ".join(f"{label}={self._rejected[label]}" for label, _ in self._ordered_rejected())
                raise DartKeysUnusableError(f"DART keys unusable: {pairs}")
            labels = ", ".join(c.label for c in self._credentials)
            raise DartQuotaExhaustedError(f"DART quota exhausted for keys: {labels}")

    def _ordered_rejected(self) -> list[tuple[str, str]]:
        return [(c.label, self._rejected[c.label]) for c in self._credentials if c.label in self._rejected]

    def _find_label(self, credential: DartCredential) -> str | None:
        for cred in self._credentials:
            if cred == credential or cred.label == credential.label:
                return cred.label
        return None

    def mark_exhausted(self, credential: DartCredential) -> None:
        """Record a daily-quota response for ``credential`` (idempotent)."""
        with self._lock:
            label = self._find_label(credential)
            if label is None or label in self._exhausted or label in self._rejected:
                return
            self._exhausted.add(label)
            remaining = len(self._usable_labels())
            logger.warning("[DATA] stage=dart_key label=%s status=EXHAUSTED reason=020 remaining=%d", label, remaining)

    def mark_rejected(self, credential: DartCredential, status: str) -> None:
        """Record a configuration rejection (DART status) for ``credential`` (idempotent)."""
        with self._lock:
            label = self._find_label(credential)
            if label is None or label in self._exhausted or label in self._rejected:
                return
            self._rejected[label] = status
            remaining = len(self._usable_labels())
            logger.error("[DATA] stage=dart_key label=%s status=REJECTED reason=%s remaining=%d", label, status, remaining)


def resolve_dart_key_pool(*, primary: str, secondary: str = "", legacy: str = "") -> DartKeyPool:
    """Build the slot-ordered pool: primary, secondary, legacy.

    Args:
        primary: Slot-1 key (``OPENDART_API_KEY``).
        secondary: Slot-2 key (``OPENDART_API_KEY_2``), blank when not provisioned.
        legacy: Legacy ``DART_API_KEY`` value.

    Returns:
        Pool with blank values omitted and duplicate key values collapsed to the
        highest-priority label. Empty when no key is configured.
    """
    return DartKeyPool(
        [
            DartCredential(label=PRIMARY_LABEL, key=primary),
            DartCredential(label=SECONDARY_LABEL, key=secondary),
            DartCredential(label=LEGACY_LABEL, key=legacy),
        ]
    )
