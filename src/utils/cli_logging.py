"""Single entry point for CLI root-logging configuration."""

from __future__ import annotations

import logging

CLI_LOG_FORMAT_PLAIN: str = "%(message)s"
CLI_LOG_FORMAT_TIMESTAMPED: str = "%(asctime)s [%(levelname)s] %(message)s"


def configure_cli_logging(fmt: str = CLI_LOG_FORMAT_PLAIN, *, level: int = logging.INFO) -> None:
    """Configure root logging for a CLI entry point.

    Must be called only from `main()` / `__main__` paths, never at import time, so importing a job
    module (tests, other jobs) never mutates global logging.

    Args:
        fmt: `logging` format string; callers pass their existing format to keep output identical.
        level: Root level.
    """
    logging.basicConfig(level=level, format=fmt)
