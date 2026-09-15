"""Append-only JSONL audit trail of weekly retrain attempts.

Every live bundle and every rejection is recorded so results stay attributable
after the fact. Retrain containers have no .git checkout, so the code commit
comes from the KCA_CODE_COMMIT build arg baked into the image.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

RETRAIN_REGISTRY_FILENAME: str = "retrain_registry.jsonl"
RETRAIN_OUTCOME_PROMOTED: str = "PROMOTED"
RETRAIN_OUTCOME_PROMOTED_UNGATED: str = "PROMOTED_UNGATED"
RETRAIN_OUTCOME_REJECTED: str = "REJECTED"
RETRAIN_OUTCOMES: tuple[str, ...] = (RETRAIN_OUTCOME_PROMOTED, RETRAIN_OUTCOME_PROMOTED_UNGATED, RETRAIN_OUTCOME_REJECTED)
BUNDLE_SHA_PREFIX_LEN: int = 12
CODE_COMMIT_ENV_VAR: str = "KCA_CODE_COMMIT"


def bundle_sha256_prefix(path: str | Path) -> str:
    """Hash a bundle file and return the leading hex digest prefix.

    Args:
        path: Bundle file to hash.

    Returns:
        First BUNDLE_SHA_PREFIX_LEN hex chars of the file's sha256.

    Raises:
        FileNotFoundError: When the file does not exist.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:BUNDLE_SHA_PREFIX_LEN]


def resolve_code_commit_env(environ: Mapping[str, str] | None = None) -> str:
    """Resolve the code commit from the build-arg environment.

    Args:
        environ: Environment mapping; defaults to os.environ.

    Returns:
        Stripped KCA_CODE_COMMIT value, or 'UNKNOWN' when absent or blank.
    """
    env = os.environ if environ is None else environ
    value = str(env.get(CODE_COMMIT_ENV_VAR, "")).strip()
    return value or "UNKNOWN"


def build_retrain_record(
    *,
    outcome: str,
    bundle: Mapping[str, Any],
    bundle_path: str | Path,
    agreement: float | None,
    reasons: Sequence[str],
    attempted_at: pd.Timestamp,
    code_commit: str,
) -> dict[str, Any]:
    """Build one registry row for a retrain attempt.

    Args:
        outcome: One of RETRAIN_OUTCOMES.
        bundle: Trained bundle metadata mapping.
        bundle_path: Where the candidate bundle was saved.
        agreement: Gate agreement score, or None when ungated.
        reasons: Gate reason strings.
        attempted_at: Attempt timestamp.
        code_commit: Code commit the retrain image was built from.

    Returns:
        Ordered record dict ready for JSONL append.

    Raises:
        ValueError: When outcome is not a known retrain outcome.
    """
    if outcome not in RETRAIN_OUTCOMES:
        raise ValueError(f"unknown retrain outcome {outcome!r}; expected one of {RETRAIN_OUTCOMES}")
    return {
        "attempted_at": pd.Timestamp(attempted_at).isoformat(timespec="seconds"),
        "outcome": outcome,
        "strategy_id": str(bundle.get("strategy_id", "UNKNOWN")),
        "training_cutoff": str(bundle.get("training_cutoff", "UNKNOWN")),
        "train_start": str(bundle.get("train_start", "UNKNOWN")),
        "trained_at": str(bundle.get("trained_at", "UNKNOWN")),
        "n_features": len(list(bundle.get("feature_cols", []))),
        "agreement": None if agreement is None else float(agreement),
        "reasons": [str(r) for r in reasons],
        "bundle_path": str(bundle_path),
        "bundle_sha": bundle_sha256_prefix(bundle_path),
        "code_commit": code_commit,
    }


def append_retrain_record(registry_path: Path, record: Mapping[str, Any]) -> None:
    """Append one record as a single JSON line, creating parent dirs.

    Args:
        registry_path: Registry JSONL file.
        record: Record mapping to append.
    """
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")
