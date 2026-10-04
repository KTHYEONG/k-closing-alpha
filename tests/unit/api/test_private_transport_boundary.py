"""AST guard locking the broker private-transport boundary.

Production modules outside ``src/api/`` must never reach into broker-client
private transport (``_handle_request``, ``_get_headers``, ``_post_tr``, ``_get``).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"

_PRIVATE_ATTRS = frozenset({"_handle_request", "_get_headers", "_post_tr", "_get"})


def _iter_guarded_files() -> list[Path]:
    return sorted(
        p
        for p in SRC_ROOT.rglob("*.py")
        if p.is_file() and p.relative_to(SRC_ROOT).parts[0] != "api"
    )


def _hits_in_source(source: str) -> list[int]:
    tree = ast.parse(source)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in _PRIVATE_ATTRS
        and not (isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"})
    ]


def _file_hits(path: Path) -> list[str]:
    rel = path.relative_to(REPO_ROOT)
    return [f"{rel}:{lineno}" for lineno in _hits_in_source(path.read_text(encoding="utf-8"))]


def test_no_private_broker_transport_outside_owner() -> None:
    violations: list[str] = []
    violations.extend(hit for path in _iter_guarded_files() for hit in _file_hits(path))
    assert violations == [], f"private broker transport outside src/api/: {violations}"


def test_guard_detects_violations() -> None:
    assert len(_hits_in_source("async def f(kis, s):\n    await kis._handle_request(s.get, 'u')\n")) == 1


def test_self_access_exempt() -> None:
    assert _hits_in_source("class C:\n    async def g(self):\n        return await self._get('x')\n") == []
