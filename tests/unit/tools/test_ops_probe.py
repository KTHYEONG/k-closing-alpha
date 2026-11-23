from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "ops" / "probes" / "kca_probe.py"
WINDOWS = ("M", "O", "E", "N", "W")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed interpreter and repo script path
        [sys.executable, str(PROBE), *args],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO,
    )


@pytest.mark.parametrize("args", [(), ("X",), ("m",), ("M-extra",)])
def test_unknown_window_exits_2(args: tuple[str, ...]) -> None:
    proc = _run(*args)
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert proc.stderr.strip() == "usage: kca_probe.py <M|O|E|N|W> [YYYY-MM-DD]"


def test_dispatch_table_has_all_windows() -> None:
    tree = ast.parse(PROBE.read_text(encoding="utf-8"))
    branches = set()
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left, comp = node.test.left, node.test.comparators[0]
            if isinstance(left, ast.Name) and left.id == "W" and isinstance(comp, ast.Constant):
                branches.add(comp.value)
    assert branches == set(WINDOWS)


def test_probe_compiles() -> None:
    compile(PROBE.read_text(encoding="utf-8"), str(PROBE), "exec")


def test_deploy_ships_probe_as_runtime_change() -> None:
    # deploy.yml resets the VPS checkout to the commit; a commit touching this path
    # must not be classified non-runtime by runtime_changed.sh, or it never reaches the host.
    script = (REPO / ".github" / "scripts" / "runtime_changed.sh").read_text(encoding="utf-8")
    match = re.search(r"^NON_RUNTIME='(.+)'$", script, re.MULTILINE)
    assert match is not None
    assert re.search(match.group(1), "ops/probes/kca_probe.py") is None
