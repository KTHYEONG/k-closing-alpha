"""AST guards locking the settings access contract."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
BRIDGE_MODULES = ("src/config/__init__.py", "src/settings.py")


def _iter_src_files() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if p.is_file())


def _settings_bound_names(tree: ast.Module) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in {"src", "src.config"} and any(a.name == "settings" for a in node.names):
                bound.update(a.asname or a.name for a in node.names if a.name == "settings")
        elif isinstance(node, ast.Import):
            bound.update(a.asname for a in node.names if a.name == "src.settings" and a.asname)
    return bound


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _getattr_hits(path: Path) -> list[str]:
    tree = _parse(path)
    bound = _settings_bound_names(tree)
    if not bound:
        return []
    rel = path.relative_to(REPO_ROOT)
    return [
        f"{rel}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id in bound
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ]


def test_no_getattr_indirection() -> None:
    violations: list[str] = []
    violations.extend(hit for path in _iter_src_files() for hit in _getattr_hits(path))
    assert violations == [], f"getattr settings indirection: {violations}"


def _module_level_attr_reads(tree: ast.Module, bound: set[str]) -> list[int]:
    return [
        node.lineno
        for stmt in tree.body
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
        for node in ast.walk(stmt)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in bound
    ]


def _default_arg_attr_reads(tree: ast.Module, bound: set[str]) -> list[int]:
    return [
        sub.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for default in [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
        for sub in ast.walk(default)
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id in bound
    ]


def _class_body_attr_reads(tree: ast.Module, bound: set[str]) -> list[int]:
    return [
        sub.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        for stmt in node.body
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
        for sub in ast.walk(stmt)
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id in bound
    ]


def _import_time_hits(path: Path) -> list[str]:
    if path.relative_to(REPO_ROOT).as_posix() in BRIDGE_MODULES:
        return []
    tree = _parse(path)
    bound = _settings_bound_names(tree)
    if not bound:
        return []
    bad = _module_level_attr_reads(tree, bound) + _default_arg_attr_reads(tree, bound) + _class_body_attr_reads(tree, bound)
    rel = path.relative_to(REPO_ROOT)
    return [f"{rel}:{lineno}" for lineno in sorted(bad)]


def test_no_import_time_settings_bindings() -> None:
    violations: list[str] = []
    violations.extend(hit for path in _iter_src_files() for hit in _import_time_hits(path))
    assert violations == [], f"import-time settings bindings: {violations}"


def _star_import_hits(path: Path) -> list[str]:
    tree = _parse(path)
    rel = path.relative_to(REPO_ROOT)
    return [
        f"{rel}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module in {"src.config", "src.settings"}
        and any(a.name == "*" for a in node.names)
    ]


def test_no_star_import_of_settings_modules() -> None:
    violations: list[str] = []
    violations.extend(hit for path in _iter_src_files() for hit in _star_import_hits(path))
    assert violations == [], f"star imports: {violations}"


def test_no_snapshot_names_in_bridge_modules() -> None:
    from src.config import Settings

    field_names = set(Settings.model_fields) | set(Settings.model_computed_fields)
    violations: list[str] = []
    violations.extend(
        f"{rel}:{stmt.lineno}:{name}"
        for rel in BRIDGE_MODULES
        for stmt in _parse(REPO_ROOT / rel).body
        for target in ([*stmt.targets] if isinstance(stmt, ast.Assign) else [stmt.target] if isinstance(stmt, ast.AnnAssign) else [])
        for name in [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]
        if name in field_names
    )
    assert violations == [], f"snapshot names: {violations}"
