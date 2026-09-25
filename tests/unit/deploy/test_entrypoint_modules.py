"""Deployed entrypoint modules must always resolve to importable code."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]

_SCAN_FILES = [
    *_ROOT.joinpath("deploy/systemd").glob("*.service"),
    _ROOT / "deploy" / "install_systemd.sh",
    _ROOT / ".github" / "workflows" / "deploy.yml",
    _ROOT / "Dockerfile",
]

_ENTRYPOINT_RE = re.compile(r"python\s+-m\s+(src\.[A-Za-z0-9_\.]+)")


def _scan_entrypoint_modules() -> set[str]:
    found: set[str] = set()
    for path in _SCAN_FILES:
        if not path.is_file():
            continue
        found.update(_ENTRYPOINT_RE.findall(path.read_text(encoding="utf-8")))
    return found


def test_every_deployed_module_entrypoint_resolves() -> None:
    """Every python -m src.<dotted> occurrence in deploy surfaces must import."""
    modules = _scan_entrypoint_modules()
    assert modules, "no deployed entrypoints found; the scan regex may have regressed"
    missing = [name for name in sorted(modules) if importlib.util.find_spec(name) is None]
    assert missing == [], f"deployed entrypoints do not resolve: {missing}"


def test_entrypoint_scan_is_not_vacuous() -> None:
    """The scan must match at least one src.daily.* and one src.tools.* module."""
    modules = _scan_entrypoint_modules()
    assert any(name.startswith("src.daily.") for name in modules), modules
    assert any(name.startswith("src.tools.") for name in modules), modules
