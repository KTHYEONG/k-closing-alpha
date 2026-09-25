"""CLI logging single-entry-point invariant guards."""

from __future__ import annotations

import ast
import logging
from pathlib import Path


def test_format_applied_to_fresh_root(monkeypatch) -> None:
    from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

    root = logging.getLogger()
    monkeypatch.setattr(root, "handlers", [])

    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    handler = root.handlers[-1]
    try:
        record = logging.LogRecord("probe", logging.INFO, __file__, 1, "x", None, None)
        assert handler.format(record).endswith("[INFO] x")
    finally:
        root.removeHandler(handler)


def test_existing_handlers_respected(monkeypatch) -> None:
    from src.utils.cli_logging import configure_cli_logging

    root = logging.getLogger()
    sentinel = logging.NullHandler()
    monkeypatch.setattr(root, "handlers", [sentinel])

    configure_cli_logging()

    assert root.handlers == [sentinel]


def _is_logging_config_call(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id == "logging" and func.attr == "basicConfig"
    return isinstance(func, ast.Name) and func.id == "configure_cli_logging"


def test_no_import_time_logging_configuration() -> None:
    offenders: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        offenders.extend(
            f"{path}:{node.lineno}"
            for node in tree.body
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and _is_logging_config_call(node.value)
        )
    assert offenders == []


def test_single_configuration_entry_point() -> None:
    hits: list[str] = []
    for path in sorted(Path("src").rglob("*.py")):
        if path == Path("src/utils/cli_logging.py"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "logging.basicConfig(" in text:
            hits.append(str(path))
    assert hits == []
