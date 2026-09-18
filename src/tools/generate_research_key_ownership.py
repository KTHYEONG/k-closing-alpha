"""Host-side generator for the research-key ownership attestation JSON.

resolve_research_credentials() refuses to hand a KIS_DATA_<slot> credential to
research consumers (auction_capture, altdata_capture) without this document:
it is the only mechanism proving a data-key slot is exclusively reserved for
research and never silently double-booked with live decision/batch trading.
Re-run this whenever COLLECTION_RESEARCH_SLOTS or a pinned consumer module
changes — a stale digest fails closed rather than trusting an old key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from src.api.kis.key_pool import kis_key_id, load_kis_env

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONSUMER_CONFIGS: dict[str, Path] = {
    "auction_capture": _PROJECT_ROOT / "src" / "daily" / "auction_capture.py",
}


def build_ownership_document(
    *,
    host_id: str,
    slots: Sequence[str],
    env: Mapping[str, str],
    consumer_configs: Mapping[str, Path],
    allowed_rest_roles: Sequence[str] = ("batch",),
) -> dict[str, object]:
    """Build the attestation object consumed by resolve_research_credentials.

    Args:
        host_id: Stable identifier for the issuing host.
        slots: Research slot tokens already reserved via COLLECTION_RESEARCH_SLOTS.
        env: Source of KIS_DATA_<slot>_APP_KEY values to fingerprint (never stored raw).
        consumer_configs: Consumer name to source file whose sha256 pins this attestation.
    Returns:
        JSON-serializable ownership document.
    Raises:
        ValueError: Missing app_key for a declared slot or no consumer configs given.
    """
    if not consumer_configs:
        raise ValueError("at least one consumer config must be pinned")
    evidence = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in consumer_configs.items()}
    owners: list[dict[str, object]] = []
    for token in slots:
        app_key = (env.get(f"KIS_DATA_{token}_APP_KEY") or "").strip()
        if not app_key:
            raise ValueError(f"missing KIS_DATA_{token}_APP_KEY for ownership attestation")
        owners.append(
            {
                "slot": f"DATA_{token}",
                "owner": "k-closing-alpha",
                "purpose": "research",
                "exclusive": True,
                "key_id": kis_key_id(app_key),
                "allowed_rest_roles": list(allowed_rest_roles),
                "verified_consumer_config_sha256": evidence,
            }
        )
    return {
        "schema_version": 1,
        "host_id": host_id,
        "verified_at": datetime.now(UTC).isoformat(),
        "credential_owners": owners,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write a fresh research-key ownership attestation to disk.

    Args:
        argv: CLI arguments; --host-id, one or more --slot, --env-file, --out.
    Returns:
        Zero on success.
    """
    parser = argparse.ArgumentParser(description="Generate research-key ownership attestation")
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--slot", action="append", required=True, dest="slots")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    env = load_kis_env(args.env_file)
    document = build_ownership_document(host_id=args.host_id, slots=tuple(args.slots), env=env, consumer_configs=_CONSUMER_CONFIGS)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("[SYS] stage=research_key_ownership status=WRITTEN slots=%s out=%s", args.slots, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
