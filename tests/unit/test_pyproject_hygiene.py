"""Repository tooling configuration must describe only files that exist."""

from __future__ import annotations

import fnmatch
import glob
import re
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_mypy_exclude_patterns_are_root_anchored() -> None:
    """Every [tool.mypy].exclude pattern must start with ^."""
    exclude = _pyproject()["tool"]["mypy"]["exclude"]
    assert exclude, "mypy exclude list must not be empty"
    unanchored = [p for p in exclude if not p.startswith("^")]
    assert unanchored == [], f"unanchored mypy exclude patterns: {unanchored}"


def test_mypy_exclude_never_drops_src_modules() -> None:
    """No [tool.mypy].exclude regex may match a path under src/."""
    exclude = _pyproject()["tool"]["mypy"]["exclude"]
    patterns = [re.compile(p) for p in exclude]
    dropped = [
        str(p)
        for p in sorted((_ROOT / "src").rglob("*.py"))
        if any(rx.search(p.relative_to(_ROOT).as_posix()) for rx in patterns)
    ]
    assert dropped == [], f"mypy exclude drops src modules: {dropped}"


def test_ruff_excludes_keep_src_and_tests_data_packages() -> None:
    """No [tool.ruff].exclude entry may be the bare token data/logs/results/legacy."""
    exclude = _pyproject()["tool"]["ruff"]["exclude"]
    bare = {"data", "logs", "results", "legacy"}
    offenders = [e for e in exclude if e in bare]
    assert offenders == [], f"unanchored ruff exclude entries: {offenders}"


def test_ruff_per_file_ignore_keys_resolve_to_real_files() -> None:
    """Every per-file-ignores key must glob to at least one existing file."""
    ignores = _pyproject()["tool"]["ruff"]["lint"]["per-file-ignores"]
    stale = [
        key
        for key in ignores
        if not glob.glob(str(_ROOT / key), recursive=True)
    ]
    assert stale == [], f"per-file-ignores keys match no file: {stale}"


def _dotted_modules() -> list[str]:
    names: list[str] = []
    for base in ("src", "tests"):
        for path in sorted((_ROOT / base).rglob("*.py")):
            rel = path.relative_to(_ROOT).with_suffix("")
            names.append(".".join(rel.parts))
    return names


def test_mypy_override_module_patterns_resolve() -> None:
    """Every [[tool.mypy.overrides]] module pattern must match a real module."""
    overrides = _pyproject()["tool"]["mypy"].get("overrides", [])
    modules = _dotted_modules()
    assert modules, "no modules found under src/ and tests/"
    unresolved = [
        pattern
        for override in overrides
        for pattern in override.get("module", [])
        if not any(fnmatch.fnmatchcase(name, pattern) for name in modules)
    ]
    assert unresolved == [], f"mypy override patterns match no module: {unresolved}"


def test_setuptools_discovery_covers_every_src_package() -> None:
    """Every src package directory must be matched by packages.find include."""
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    packages = [
        ".".join(init.parent.relative_to(_ROOT).parts)
        for init in sorted((_ROOT / "src").rglob("__init__.py"))
    ]
    assert packages, "no packages found under src/"
    uncovered = [
        pkg
        for pkg in packages
        if not any(fnmatch.fnmatchcase(pkg, pattern) for pattern in include)
    ]
    assert uncovered == [], f"packages not covered by setuptools include: {uncovered}"
