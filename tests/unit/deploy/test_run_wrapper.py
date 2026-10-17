"""Guards for the systemd container run wrapper (deploy/host/kca-run)."""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from tests.unit.deploy._rendered_units import effective_execstart, rendered_unit_text

ROOT = Path(__file__).resolve().parents[3]
SYSTEMD = ROOT / "deploy" / "systemd"
WRAPPER = ROOT / "deploy" / "host" / "kca-run"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "legacy_docker_execstart.json"
SENTINEL_HOME = "/nonexistent/kca-home"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _git_show(rev_path: str) -> bytes:
    proc = subprocess.run(  # noqa: S603
        ["git", "show", rev_path], capture_output=True, cwd=ROOT, timeout=60, check=True  # noqa: S607
    )
    return proc.stdout


def _print_argv(args: list[str], home: str | None = SENTINEL_HOME) -> subprocess.CompletedProcess[bytes]:
    env = {"PATH": os.environ["PATH"]}
    if home is not None:
        env["HOME"] = home
    return subprocess.run(  # noqa: S603
        ["bash", str(WRAPPER), "--print-argv", *args],  # noqa: S607
        capture_output=True,
        env=env,
        timeout=60,
    )


def _argv(args: list[str], home: str | None = SENTINEL_HOME) -> list[str]:
    proc = _print_argv(args, home)
    assert proc.returncode == 0, proc.stderr
    parts = proc.stdout.split(b"\0")
    if parts and parts[-1] == b"":
        parts = parts[:-1]
    return [part.decode("utf-8") for part in parts]


def _wrapper_units() -> list[Path]:
    return sorted(
        path
        for path in SYSTEMD.glob("kca-*.service")
        if "ExecStart=%h/.local/lib/kca/kca-run " in path.read_text(encoding="utf-8")
    )


def test_legacy_argv_parity_all_units() -> None:
    fixture = _fixture()
    assert {path.name for path in _wrapper_units()} == set(fixture["execstart"])
    for name, legacy in fixture["execstart"].items():
        assert effective_execstart(SYSTEMD / name) == legacy, name


def test_reaper_name_matches_wrapper_name() -> None:
    fixture = _fixture()
    for name in fixture["execstart"]:
        stem = name.removeprefix("kca-").removesuffix(".service")
        text = rendered_unit_text(SYSTEMD / name)
        assert f"--name kca-{stem}" in text, name
        assert f"ExecStopPost=-/usr/bin/docker rm -f kca-{stem}" in text, name


def test_only_execstart_changed() -> None:
    # Pins the wrapper migration itself (pre-migration sha vs migration commit), so later
    # intentional unit/timer edits do not masquerade as migration regressions.
    sha = _fixture()["sha"]
    migrated = _fixture()["migration_sha"]
    for name in _fixture()["execstart"]:
        old = _git_show(f"{sha}:deploy/systemd/{name}").decode("utf-8")
        new = _git_show(f"{migrated}:deploy/systemd/{name}").decode("utf-8")
        old_body = [line for line in old.splitlines() if not line.startswith("ExecStart=")]
        new_body = [line for line in new.splitlines() if not line.startswith("ExecStart=")]
        assert old_body == new_body, name
    timers = _git_show(f"{migrated}:deploy/systemd").decode("utf-8").split()
    migrated_timers = sorted(name for name in timers if name.startswith("kca-") and name.endswith(".timer"))
    assert migrated_timers
    for timer in migrated_timers:
        assert _git_show(f"{sha}:deploy/systemd/{timer}") == _git_show(f"{migrated}:deploy/systemd/{timer}"), timer


def test_exec_replaces_shell() -> None:
    lines = WRAPPER.read_text(encoding="utf-8").splitlines()
    exec_lines = [line for line in lines if line.startswith("exec /usr/bin/docker")]
    assert len(exec_lines) == 1
    assert "|" not in exec_lines[0]
    assert "&" not in exec_lines[0]
    assert ";" not in exec_lines[0]
    assert not any(line.strip().startswith("trap") for line in lines)


_USAGE_CASES: list[tuple[list[str], str | None]] = [
    (["Collect", "--admission", "critical", "--", "uv", "run", "x"], SENTINEL_HOME),
    (["collect", "--bogus", "--", "uv", "run", "x"], SENTINEL_HOME),
    (["collect", "--admission", "critical", "--admission", "bulk", "--", "uv", "run", "x"], SENTINEL_HOME),
    (["collect", "--kis-data", "--kis-data", "--admission", "critical", "--", "uv", "run", "x"], SENTINEL_HOME),
    (["collect", "--admission", "turbo", "--", "uv", "run", "x"], SENTINEL_HOME),
    (["collect", "--admission", "critical", "--role", "turbo", "--", "uv", "run", "x"], SENTINEL_HOME),
    (["collect", "--admission", "critical", "--role", "decision", "--", "uv", "run", "x"], SENTINEL_HOME),
    (["collect", "--kis-data", "--", "uv", "run", "x"], SENTINEL_HOME),
    (["collect", "--resource", "--privileged", "--", "uv", "run", "x"], SENTINEL_HOME),
    (["collect", "--resource", "--memory=6G", "--", "uv", "run", "x"], SENTINEL_HOME),
    (["collect", "--admission", "critical"], SENTINEL_HOME),
    (["collect", "--admission", "critical", "--"], SENTINEL_HOME),
    (["collect", "--admission", "critical", "--", "uv", "run", "x"], None),
    (["collect", "--admission", "critical", "--", "uv", "run", "x"], "relative/path"),
]


@pytest.mark.parametrize(("args", "home"), _USAGE_CASES)
def test_usage_errors_exit_64(args: list[str], home: str | None) -> None:
    proc = _print_argv(args, home)
    assert proc.returncode == 64, proc.stderr
    assert proc.stdout == b""
    lines = proc.stderr.decode("utf-8").splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("[SYS] stage=kca_run status=USAGE_ERROR reason=")
    assert len(lines[0].split("reason=")[1].split()) == 1


def test_print_argv_never_executes(tmp_path: Path) -> None:
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "docker-ran"
    stub = bindir / "docker"
    stub.write_text(f"#!/usr/bin/env bash\ntouch {marker}\n")
    stub.chmod(0o755)
    env = {"HOME": SENTINEL_HOME, "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}"}
    proc = subprocess.run(  # noqa: S603
        ["bash", str(WRAPPER), "--print-argv", "collect", "--admission", "critical", "--", "uv", "x"],  # noqa: S607
        capture_output=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert not marker.exists()
    assert proc.stdout.split(b"\0")[0] == b"/usr/bin/docker"


def test_resource_flags_precede_env_files() -> None:
    tokens = _argv(["retrain", "--resource", "--memory=6g", "--resource", "--cpus=1.8", "--", "uv", "x"])
    assert tokens[0] == "/usr/bin/docker"
    cache_idx = tokens.index("UV_CACHE_DIR=/tmp/uv-cache")
    first_env_idx = tokens.index("--env-file")
    assert [tokens[cache_idx + 1], tokens[cache_idx + 2]] == ["--memory=6g", "--cpus=1.8"]
    assert cache_idx < first_env_idx


def test_resource_flags_keep_given_order() -> None:
    tokens = _argv(
        ["retrain", "--resource", "--cpus=1.8", "--resource", "--memory-swap=8g", "--", "uv", "x"]
    )
    assert tokens[tokens.index("UV_CACHE_DIR=/tmp/uv-cache") + 1 :][:2] == ["--cpus=1.8", "--memory-swap=8g"]


def test_command_tokens_verbatim() -> None:
    tokens = _argv(
        [
            "order-reconcile-entry",
            "--admission",
            "critical",
            "--",
            "uv",
            "run",
            "python",
            "-m",
            "src.daily.order_session",
            "--phase",
            "reconcile-entry",
        ]
    )
    assert tokens[-7:] == ["uv", "run", "python", "-m", "src.daily.order_session", "--phase", "reconcile-entry"]


def test_no_unit_uses_print_argv() -> None:
    offenders = [
        path.name
        for path in sorted(SYSTEMD.glob("kca-*.service"))
        if "--print-argv" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_wrapper_is_executable_in_git() -> None:
    proc = subprocess.run(  # noqa: S603
        ["git", "ls-files", "-s", "deploy/host/kca-run"],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=60,
        check=True,
    )
    assert proc.stdout.startswith("100755"), proc.stdout


def test_install_script_installs_wrapper_before_units() -> None:
    text = (ROOT / "deploy" / "install_systemd.sh").read_text(encoding="utf-8")
    assert ".local/lib/kca" in text
    assert "kca-run" in text
    assert "mv -f" in text
    assert text.index("kca-run") < text.index("kca-*.service")


def test_code_sync_installs_wrapper_before_units_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    import subprocess

    from src.tools import code_sync

    repo = tmp_path / "repo"
    (repo / "deploy" / "host").mkdir(parents=True)
    (repo / "deploy" / "systemd").mkdir(parents=True)
    (repo / "deploy" / "host" / "kca-run").write_bytes(b"#!/usr/bin/env bash\nexec docker\n")
    (repo / "deploy" / "host" / "kca-run").chmod(0o755)
    (repo / "deploy" / "systemd" / "kca-collect.service").write_text("x\n")
    lib = tmp_path / "lib"
    dest = tmp_path / "user"
    dest.mkdir()
    calls: list[list[str]] = []

    def fake_run(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs["check"] is True
        calls.append(list(cmd))  # type: ignore[arg-type]
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    events: list[str] = []
    real_copy = shutil.copyfile

    def spy_copy(src: str, dst: str, **kwargs: object) -> str:
        events.append(f"copy:{Path(dst).name}")
        return real_copy(src, dst, **kwargs)

    real_install = code_sync.install_kca_run_wrapper

    def spy_install(*args: object, **kwargs: object) -> bool:
        events.append("wrapper")
        return real_install(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(shutil, "copyfile", spy_copy)
    monkeypatch.setattr(code_sync, "install_kca_run_wrapper", spy_install)
    result = code_sync.install_systemd_units(str(repo), dest_dir=dest, lib_dir=lib, run_fn=fake_run)

    wrapper = lib / "kca-run"
    assert events[0] == "wrapper"
    assert wrapper.read_bytes() == b"#!/usr/bin/env bash\nexec docker\n"
    assert stat.S_IMODE(wrapper.stat().st_mode) == 0o755
    assert stat.S_IMODE(lib.stat().st_mode) == 0o700
    assert (dest / "kca-collect.service").exists()
    assert list(lib.glob(".kca-run.*")) == []
    assert result.changed == ("kca-collect.service",)


def test_code_sync_wrapper_install_failure_keeps_old_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.tools import code_sync

    repo = tmp_path / "repo"
    (repo / "deploy" / "host").mkdir(parents=True)
    (repo / "deploy" / "host" / "kca-run").write_bytes(b"new\n")
    lib = tmp_path / "lib"
    lib.mkdir()
    target = lib / "kca-run"
    target.write_bytes(b"old\n")
    target.chmod(0o755)

    def boom(*args: object, **kwargs: object) -> object:
        raise OSError("injected replace failure")

    monkeypatch.setattr(code_sync.os, "replace", boom)
    with pytest.raises(OSError, match="injected replace failure"):
        code_sync.install_kca_run_wrapper(str(repo), lib_dir=lib)
    assert target.read_bytes() == b"old\n"
    assert list(lib.glob(".kca-run.*")) == []


def test_code_sync_wrapper_install_is_idempotent(tmp_path: Path) -> None:
    from src.tools import code_sync

    repo = tmp_path / "repo"
    (repo / "deploy" / "host").mkdir(parents=True)
    (repo / "deploy" / "host" / "kca-run").write_bytes(b"#!/usr/bin/env bash\n")
    lib = tmp_path / "lib"

    assert code_sync.install_kca_run_wrapper(str(repo), lib_dir=lib) is True
    assert code_sync.install_kca_run_wrapper(str(repo), lib_dir=lib) is False


def test_code_sync_wrapper_defaults_to_user_lib_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tools import code_sync

    repo = tmp_path / "repo"
    (repo / "deploy" / "host").mkdir(parents=True)
    (repo / "deploy" / "host" / "kca-run").write_bytes(b"#!/usr/bin/env bash\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert code_sync.install_kca_run_wrapper(str(repo)) is True
    assert (tmp_path / ".local" / "lib" / "kca" / "kca-run").read_bytes() == b"#!/usr/bin/env bash\n"
