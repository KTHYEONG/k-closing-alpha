"""Deploy preflight: refuse an image incompatible with the live bundle."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from typing import Any

from src.ml.topk_ranker_research import (
    RANKER_FEATURE_COLS,
    TOPK_RANKER_BUNDLE_DIR,
    assert_bundle_screen_parity,
)
from src.serving.realtime.artifacts import load_model_bundle
from src.strategy.contract import MIN_TOP_K
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)


def check_bundle_serving_compat(bundle: Mapping[str, Any]) -> list[str]:
    """List every incompatibility between a live bundle and this code's serving path.

    A bundle certified under a different live screen or feature contract makes
    predict return NO_DECISION at 15:21; detecting it when the image is
    deployed moves the failure hours earlier and keeps the previous image live.

    Args:
        bundle: Loaded production bundle.

    Returns:
        Human-readable issues; empty when compatible. Checks: screen parity with
        COST_AWARE_UNIVERSE, feature_cols non-empty and a subset of
        RANKER_FEATURE_COLS, and bundle top_k equal to MIN_TOP_K.
    """
    issues: list[str] = []
    try:
        assert_bundle_screen_parity(dict(bundle))
    except ValueError as exc:
        issues.append(f"bundle screen drift: {exc}")
    feature_cols = list(bundle.get("feature_cols", []))
    if not feature_cols:
        issues.append("bundle feature_cols is empty; refusing to serve inference")
    else:
        unknown = [col for col in feature_cols if col not in RANKER_FEATURE_COLS]
        if unknown:
            issues.append(f"bundle feature_cols unknown to serving contract: {unknown}")
    if bundle.get("top_k") != MIN_TOP_K:
        issues.append(f"bundle top_k {bundle.get('top_k')!r} is not the certified MIN_TOP_K {MIN_TOP_K}")
    return issues


def main(argv: list[str] | None = None) -> None:
    """Deploy preflight entry point: load the live bundle and fail on incompatibility.

    Args:
        argv: Optional --bundle-dir (default TOPK_RANKER_BUNDLE_DIR).

    Raises:
        SystemExit: exit 1 with one `[SYS] stage=deploy_preflight status=FAIL issue=...`
            line per issue; exit 1 when the bundle is missing or unloadable.
    """
    from src.daily.predict import bundle_model_version

    parser = argparse.ArgumentParser(description="Deploy preflight: bundle/serving compatibility gate")
    parser.add_argument("--bundle-dir", default=TOPK_RANKER_BUNDLE_DIR)
    args = parser.parse_args(argv)
    try:
        bundle = load_model_bundle(import_dir=args.bundle_dir)
    except Exception as exc:
        logger.error("[SYS] stage=deploy_preflight status=FAIL issue=%s", f"bundle unloadable: {exc}")
        raise SystemExit(1) from exc
    issues = check_bundle_serving_compat(bundle)
    if issues:
        for issue in issues:
            logger.error("[SYS] stage=deploy_preflight status=FAIL issue=%s", issue)
        raise SystemExit(1)
    logger.info(
        "[SYS] stage=deploy_preflight status=OK bundle=%s model_version=%s",
        args.bundle_dir,
        bundle_model_version(bundle),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    main()
