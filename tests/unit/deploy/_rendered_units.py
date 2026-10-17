"""Resolve wrapper-based units to their legacy-equivalent docker ExecStart lines."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WRAPPER = _REPO_ROOT / "deploy" / "host" / "kca-run"
_BASH = shutil.which("bash") or "/usr/bin/bash"
_WRAPPER_PREFIX = "%h/.local/lib/kca/kca-run "
_SENTINEL_HOME = "/nonexistent/kca-home"


def effective_execstart(unit_path: Path) -> str | None:
    """Return the legacy-equivalent docker ExecStart value for a unit."""
    text = Path(unit_path).read_text(encoding="utf-8")
    exec_line: str | None = None
    for line in text.splitlines():
        if line.startswith("ExecStart="):
            exec_line = line
            break
    if exec_line is None:
        return None
    value = exec_line.split("=", 1)[1]
    if value.startswith("/usr/bin/docker run"):
        return value
    if not value.startswith(_WRAPPER_PREFIX):
        return None
    args = shlex.split(value[len(_WRAPPER_PREFIX):])
    env = {"HOME": _SENTINEL_HOME, "PATH": os.environ["PATH"]}
    proc = subprocess.run(  # noqa: S603
        [_BASH, str(_WRAPPER), "--print-argv", *args],
        capture_output=True,
        text=False,
        env=env,
        timeout=60,
    )
    if proc.returncode != 0 or proc.stderr:
        raise AssertionError(
            f"kca-run failed for {unit_path}: exit={proc.returncode} stderr={proc.stderr!r}"
        )
    raw = proc.stdout.split(b"\0")
    if raw and raw[-1] == b"":
        raw = raw[:-1]
    user_group = f"{os.getuid()}:{os.getgid()}"
    rendered = [
        token.replace(_SENTINEL_HOME, "%h").replace(user_group, "%U:%G")
        for token in (t.decode("utf-8") for t in raw)
    ]
    return " ".join(rendered)


def rendered_unit_text(unit_path: Path) -> str:
    """Return the unit text with a wrapper ExecStart line replaced by its rendering."""
    path = Path(unit_path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("ExecStart=") and _WRAPPER_PREFIX in line:
            lines[index] = f"ExecStart={effective_execstart(path)}\n"
            break
    return "".join(lines)
