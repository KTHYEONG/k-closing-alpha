"""Read-only sizing-pipeline bundle contract for the live daily serving path.

The daily runtime consumes an already-published, immutable model bundle. This
module only deserializes and structurally validates that bundle; it never
trains, calibrates, evaluates, uploads, or overwrites model artifacts.
"""

from __future__ import annotations

import os
from typing import Any

from joblib import load

_BUNDLE_FILENAME = "sizing_pipeline_bundle.joblib"
_MODEL_BUNDLE_KEYS = ("rank_model", "quantile_models", "calibrators")


def load_model_bundle(import_dir: str = "artifacts/models") -> dict[str, Any]:
    """Read-only joblib load of the published sizing-pipeline bundle.

    Requires a non-empty ``feature_cols`` and the ``rank_model`` /
    ``quantile_models`` / ``calibrators`` model keys. A missing directory or
    bundle file raises ``FileNotFoundError``; a malformed or schema-incompatible
    bundle raises ``ValueError``. This loader never retrains or writes artifacts.
    """
    path = os.path.join(import_dir, _BUNDLE_FILENAME)
    if not os.path.isdir(import_dir) or not os.path.isfile(path):
        raise FileNotFoundError(
            f"model artifact bundle not found at {path!r}; run training to save artifacts first"
        )
    bundle: dict[str, Any] = load(path)
    feature_cols = list(bundle.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("bundle feature_cols must be non-empty; refusing to serve inference")
    missing = [key for key in _MODEL_BUNDLE_KEYS if key not in bundle]
    if missing:
        raise ValueError(
            f"bundle is missing required model keys: {missing}; refusing to serve inference"
        )
    return bundle
