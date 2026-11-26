"""Guards for the host-unit idle wait that holds deploy converge (deploy/host/wait-host-units-idle.sh)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "deploy" / "host" / "wait-host-units-idle.sh"

STUB_SYSTEMCTL = """#!/usr/bin/env bash
echo "$*" >> "$STUB_LOG"
if [ -n "${STUB_ALWAYS:-}" ]; then
  echo "$STUB_ALWAYS"
  exit 0
fi
state=$(head -n1 "$STUB_STATES" 2>/dev/null || true)
if [ -n "$state" ]; then
  sed -i '1d' "$STUB_STATES"
  echo "$state"
else
  echo inactive
fi
"""


def _checkout(tmp_path: Path) -> Path:
    systemd = tmp_path / "checkout" / "deploy" / "systemd"
    systemd.mkdir(parents=True)
    (systemd / "kca-backup.service").write_text("[Service]\nExecStart=/usr/bin/python -m x\n", encoding="utf-8")
    (systemd / "kca-collect.service").write_text("[Service]\nExecStart=/usr/bin/kca-run collect\n", encoding="utf-8")
    (systemd / "kca-alert@.service").write_text("[Service]\nExecStart=/usr/bin/python -m alert\n", encoding="utf-8")
    return tmp_path / "checkout"


def _run(tmp_path: Path, *, timeout: int, poll: int, always: str = "", states: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "systemctl"
    stub.write_text(STUB_SYSTEMCTL, encoding="utf-8")
    stub.chmod(0o755)
    states_file = tmp_path / "states"
    states_file.write_text("".join(f"{s}\n" for s in (states or [])), encoding="utf-8")
    log = tmp_path / "systemctl.log"
    env = {
        "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
        "STUB_LOG": str(log),
        "STUB_STATES": str(states_file),
        "STUB_ALWAYS": always,
    }
    return subprocess.run(  # noqa: S603
        ["bash", str(SCRIPT), str(_checkout(tmp_path)), str(timeout), str(poll)],  # noqa: S607
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def test_unit_list_excludes_docker_and_templates(tmp_path: Path) -> None:
    proc = _run(tmp_path, timeout=60, poll=0, always="inactive")

    assert proc.returncode == 0, proc.stdout + proc.stderr
    queried = (tmp_path / "systemctl.log").read_text(encoding="utf-8").splitlines()
    assert queried == ["--user show -p ActiveState --value kca-backup"]


def test_waits_while_activating_then_clears(tmp_path: Path) -> None:
    proc = _run(tmp_path, timeout=60, poll=0, states=["activating", "activating", "inactive"])

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.count("busy=kca-backup") == 2
    assert "status=clear" in proc.stdout
    assert (tmp_path / "systemctl.log").read_text(encoding="utf-8").count("show") == 3


@pytest.mark.parametrize("busy_state", ["active", "activating", "deactivating", "reloading"])
def test_busy_states_block_until_timeout(tmp_path: Path, busy_state: str) -> None:
    proc = _run(tmp_path, timeout=0, poll=0, always=busy_state)

    assert proc.returncode != 0
    assert "status=timeout busy=kca-backup" in proc.stderr
    log = (tmp_path / "systemctl.log").read_text(encoding="utf-8")
    assert " stop" not in log
    assert " kill" not in log
