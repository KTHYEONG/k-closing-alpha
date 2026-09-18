"""Host-side evidence gate for optional acquisition jobs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

from src.config.collection import CollectionSettings


def _consumer_digest(path: Path) -> str | None:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def check_collection_activation(
    profile: CollectionSettings,
    *,
    ownership_path: Path,
    consumer_configs: Mapping[str, Path],
    measured_peak_rss_mib: int,
    measured_max_sweep_seconds: float,
) -> tuple[str, ...]:
    """Check host-side ownership and measured budgets before optional job activation.

    Args:
        profile: Declared acquisition settings.
        ownership_path: Host-local research-key ownership attestation.
        consumer_configs: Exact consumer-name to host config files covered by attestation.
        measured_peak_rss_mib: Observed representative worker peak.
        measured_max_sweep_seconds: Observed full-population sweep duration.
    Returns:
        Empty tuple only when all declared activation evidence is consistent.
    Raises:
        ValueError: Invalid or missing measured evidence.
    """
    if measured_peak_rss_mib <= 0 or not math.isfinite(measured_max_sweep_seconds) or measured_max_sweep_seconds <= 0:
        raise ValueError("measured evidence must be positive and finite")
    if int(measured_peak_rss_mib) > int(profile.COLLECTION_MAX_RSS_MIB):
        return ("worker memory budget exceeded",)
    if float(measured_max_sweep_seconds) > float(profile.COLLECTION_AUCTION_INTERVAL_SECONDS):
        return ("sweep duration exceeds declared interval",)
    if not profile.COLLECTION_RESEARCH_SLOTS:
        return ("research slots are not declared",)
    try:
        raw = Path(ownership_path).read_text(encoding="utf-8")
        document: object = json.loads(raw)
    except (OSError, ValueError):
        return ("ownership attestation unreadable: fresh verification required",)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or not isinstance(document.get("host_id"), str)
        or not str(document.get("host_id")).strip()
        or not isinstance(document.get("verified_at"), str)
        or not isinstance(document.get("credential_owners"), list)
    ):
        return ("ownership attestation invalid: fresh verification required",)
    try:
        moment = datetime.fromisoformat(str(document.get("verified_at")))
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("naive timestamp")
    except ValueError:
        return ("ownership attestation invalid: fresh verification required",)
    by_slot: dict[str, dict[str, object]] = {}
    assert isinstance(document, dict)
    owners = document.get("credential_owners")
    assert isinstance(owners, list)
    for item in owners:
        if isinstance(item, dict) and isinstance(item.get("slot"), str) and str(item.get("slot")) not in by_slot:
            by_slot[str(item.get("slot"))] = item
    for token in profile.COLLECTION_RESEARCH_SLOTS:
        entry = by_slot.get(f"DATA_{token}")
        if entry is None or entry.get("owner") != "k-closing-alpha" or entry.get("purpose") != "research" or entry.get("exclusive") is not True:
            return (f"unverified ownership for slot DATA_{token}",)
    attested: dict[str, str] = {}
    for entry in by_slot.values():
        evidence = entry.get("verified_consumer_config_sha256")
        if not isinstance(evidence, dict) or not evidence:
            return ("consumer evidence incomplete: fresh verification required",)
        for name, digest in evidence.items():
            if name in attested and attested[name] != digest:
                return (f"conflicting ownership for consumer {name}",)
            attested[name] = digest
    if set(attested) != set(consumer_configs):
        return ("consumer inventory mismatch: fresh verification required",)
    for name, config_path in consumer_configs.items():
        digest = _consumer_digest(config_path)
        if digest is None:
            return (f"consumer config unreadable for {name}",)
        if digest != attested[name]:
            return (f"consumer config changed for {name}: fresh attestation required",)
    return ()
