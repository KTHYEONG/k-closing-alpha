def test_sync_repo_returns_up_to_date_when_no_new_commits() -> None:
    from src.tools.code_sync import SyncResult, UnitInstallResult, sync_repo

    def fake_git(args: list[str], cwd: str) -> str:
        if args[:2] == ["rev-parse", "HEAD"]:
            return "a" * 40
        if args[0] == "fetch":
            return ""
        if args[:2] == ["rev-parse", "FETCH_HEAD"]:
            return "a" * 40
        raise AssertionError(f"unexpected git call: {args}")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not be called when already up to date")

    units = UnitInstallResult(changed=("kca-backup.service",), removed=(), enabled=())
    order: list[str] = []

    # When
    result = sync_repo(
        "/repo", git_fn=fake_git, fast_forward_fn=fail_if_called,
        test_gate_fn=fail_if_called, uv_sync_fn=fail_if_called,
        install_units_fn=lambda repo_dir: order.append("install") or units,
        secure_fn=lambda repo_dir: order.append("secure") or False,
    )

    # Then
    assert result == SyncResult(updated=False, from_sha="a" * 40, to_sha="a" * 40, reason="up_to_date", units=units)
    assert order == ["secure", "install"]


def test_sync_repo_alerts_and_skips_on_non_fast_forward(monkeypatch) -> None:
    from src.tools.code_sync import SyncResult, sync_repo

    calls: list[list[str]] = []

    def fake_git(args: list[str], cwd: str) -> str:
        calls.append(list(args))
        if args[:2] == ["rev-parse", "HEAD"]:
            return "a" * 40
        if args[0] == "fetch":
            return ""
        if args[:2] == ["rev-parse", "FETCH_HEAD"]:
            return "b" * 40
        raise AssertionError(f"unexpected git call: {args}")

    captured: dict = {}
    monkeypatch.setattr(
        "src.tools.alerts.dispatch_failure_alert",
        lambda unit, *, detail="": captured.update(unit=unit, detail=detail) or {"webhook": True, "email": True},
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not be called on a non-fast-forward")

    # When
    result = sync_repo(
        "/repo", git_fn=fake_git, fast_forward_fn=lambda repo_dir, f, t: False,
        test_gate_fn=fail_if_called, uv_sync_fn=fail_if_called,
    )

    # Then
    assert result == SyncResult(updated=False, from_sha="a" * 40, to_sha="b" * 40, reason="not_fast_forward")
    assert not any(c[:2] == ["reset", "--hard"] for c in calls)
    assert captured["unit"] == "kca-code-sync.service"
    assert "aaaaaaaa" in captured["detail"] and "bbbbbbbb" in captured["detail"]


def test_sync_repo_rolls_back_and_alerts_on_test_gate_failure(monkeypatch) -> None:
    from src.tools.code_sync import SyncResult, sync_repo

    reset_calls: list[str] = []

    def fake_git(args: list[str], cwd: str) -> str:
        if args[:2] == ["rev-parse", "HEAD"]:
            return "a" * 40
        if args[0] == "fetch":
            return ""
        if args[:2] == ["rev-parse", "FETCH_HEAD"]:
            return "b" * 40
        if args[:2] == ["reset", "--hard"]:
            reset_calls.append(args[2])
            return ""
        raise AssertionError(f"unexpected git call: {args}")

    captured: dict = {}
    monkeypatch.setattr(
        "src.tools.alerts.dispatch_failure_alert",
        lambda unit, *, detail="": captured.update(unit=unit, detail=detail) or {"webhook": True, "email": True},
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("uv_sync must not run when the test gate fails")

    # When
    result = sync_repo(
        "/repo", git_fn=fake_git, fast_forward_fn=lambda repo_dir, f, t: True,
        test_gate_fn=lambda repo_dir: (False, "1 failed, 99 passed"), uv_sync_fn=fail_if_called,
    )

    # Then: forward-reset then rollback-reset, in that order
    assert reset_calls == ["b" * 40, "a" * 40]
    assert result == SyncResult(updated=False, from_sha="a" * 40, to_sha="b" * 40, reason="test_gate_failed")
    assert captured["unit"] == "kca-code-sync.service"
    assert "1 failed, 99 passed" in captured["detail"]


def test_sync_repo_fast_forwards_and_syncs_deps_on_success() -> None:
    from src.tools.code_sync import SyncResult, UnitInstallResult, sync_repo

    order: list[str] = []

    def fake_git(args: list[str], cwd: str) -> str:
        if args[:2] == ["rev-parse", "HEAD"]:
            return "a" * 40
        if args[0] == "fetch":
            return ""
        if args[:2] == ["rev-parse", "FETCH_HEAD"]:
            return "b" * 40
        if args[:2] == ["reset", "--hard"]:
            return ""
        raise AssertionError(f"unexpected git call: {args}")

    units = UnitInstallResult(changed=(), removed=(), enabled=())

    # When
    result = sync_repo(
        "/repo", git_fn=fake_git, fast_forward_fn=lambda repo_dir, f, t: True,
        test_gate_fn=lambda repo_dir: (True, ""), uv_sync_fn=lambda repo_dir: order.append("uv_sync"),
        install_units_fn=lambda repo_dir: order.append("install") or units,
        secure_fn=lambda repo_dir: order.append("secure") or True,
    )

    # Then
    assert result == SyncResult(updated=True, from_sha="a" * 40, to_sha="b" * 40, reason="fast_forwarded", units=units)
    assert order == ["uv_sync", "secure", "install"]


def test_git_helper_reads_head_and_raises_on_bad_command(tmp_path) -> None:
    import subprocess

    import pytest

    from src.tools.code_sync import _git

    # Given: a real local git repo
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)  # noqa: S607
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)  # noqa: S607
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)  # noqa: S607
    (repo / "f.txt").write_text("1")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)  # noqa: S607
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)  # noqa: S607

    # When: rev-parse HEAD succeeds
    head = _git(["rev-parse", "HEAD"], str(repo))

    # Then
    assert len(head) == 40

    # And: a bad git subcommand raises, never swallowed
    with pytest.raises(subprocess.CalledProcessError):
        _git(["not-a-real-subcommand"], str(repo))


def test_is_fast_forward_detects_ancestor_and_divergence(tmp_path) -> None:
    import subprocess

    from src.tools.code_sync import _git, _is_fast_forward

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)  # noqa: S607
    subprocess.run(["git", "config", "user.email", "t@t.com"], cwd=repo, check=True)  # noqa: S607
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)  # noqa: S607
    (repo / "f.txt").write_text("1")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)  # noqa: S607
    subprocess.run(["git", "commit", "-q", "-m", "c1"], cwd=repo, check=True)  # noqa: S607
    c1 = _git(["rev-parse", "HEAD"], str(repo))
    (repo / "f.txt").write_text("2")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)  # noqa: S607
    subprocess.run(["git", "commit", "-q", "-m", "c2"], cwd=repo, check=True)  # noqa: S607
    c2 = _git(["rev-parse", "HEAD"], str(repo))

    # Then: c1 -> c2 is a clean fast-forward
    assert _is_fast_forward(str(repo), c1, c2) is True
    # And: c2 -> c1 would be a rewind, not a fast-forward
    assert _is_fast_forward(str(repo), c2, c1) is False


def test_run_test_gate_reports_pass_and_bounds_output_tail(monkeypatch, tmp_path) -> None:
    import subprocess

    from src.tools import code_sync

    class _FakeResult:
        returncode = 0
        stdout = "x" * 3000
        stderr = ""

    captured: dict = {}

    def fake_run(cmd, cwd, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_run)

    # When
    passed, tail = code_sync._run_test_gate(str(tmp_path))

    # Then
    assert passed is True
    assert captured["cmd"] == [code_sync._resolve_uv_bin(), "run", "pytest", "-q"]
    assert captured["cmd"][0] != "uv"
    assert captured["cwd"] == str(tmp_path)
    assert len(tail) == code_sync.ALERT_DETAIL_TAIL_CHARS


def test_uv_sync_invokes_uv_sync_command(monkeypatch, tmp_path) -> None:
    import subprocess

    from src.tools import code_sync
    from src.tools.code_sync import _uv_sync

    captured: dict = {}

    def fake_run(cmd, cwd, check, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["check"] = check

    monkeypatch.setattr(subprocess, "run", fake_run)

    # When
    _uv_sync(str(tmp_path))

    # Then
    assert captured["cmd"] == [code_sync._resolve_uv_bin(), "sync"]
    assert captured["cmd"][0] != "uv"
    assert captured["cwd"] == str(tmp_path)
    assert captured["check"] is True


def test_code_sync_main_invokes_sync_repo_with_settings_base_dir(monkeypatch) -> None:
    from src import settings
    from src.tools import code_sync

    captured: dict = {}

    def fake_sync_repo(repo_dir, *, remote, branch):
        captured.update(repo_dir=repo_dir, remote=remote, branch=branch)
        return code_sync.SyncResult(updated=True, from_sha="a" * 40, to_sha="b" * 40, reason="fast_forwarded")

    monkeypatch.setattr(code_sync, "sync_repo", fake_sync_repo)

    # When
    code_sync.main([])

    # Then
    assert captured == {"repo_dir": str(settings.BASE_DIR), "remote": "origin", "branch": "main"}


def test_resolve_uv_bin_prefers_path_when_available(monkeypatch) -> None:
    from src.tools import code_sync

    monkeypatch.setattr(code_sync.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    assert code_sync._resolve_uv_bin() == "/usr/bin/uv"


def test_resolve_uv_bin_falls_back_to_local_bin_when_not_on_path(monkeypatch) -> None:
    from pathlib import Path

    from src.tools import code_sync

    monkeypatch.setattr(code_sync.shutil, "which", lambda name: None)

    resolved = code_sync._resolve_uv_bin()

    assert resolved == str(Path.home() / ".local" / "bin" / "uv")
    assert resolved != "uv"


def test_install_systemd_units_copies_changed_and_new_units_reloads_and_enables_new_timer(tmp_path) -> None:
    import subprocess

    from src.tools import code_sync

    repo = tmp_path / "repo"
    src = repo / "deploy" / "systemd"
    src.mkdir(parents=True)
    dest = tmp_path / "user"
    dest.mkdir()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    (src / "kca-backup.service").write_text("[Service]\nExecStart=new\n")
    (dest / "kca-backup.service").write_text("[Service]\nExecStart=old\n")
    (src / "kca-new.timer").write_text("[Timer]\nOnCalendar=daily\n")
    (src / "kca-same.service").write_text("same\n")
    (dest / "kca-same.service").write_text("same\n")

    # When
    result = code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=fake_run)

    # Then
    assert result.changed == ("kca-backup.service", "kca-new.timer")
    assert result.removed == ()
    assert result.enabled == ("kca-new.timer",)
    assert (dest / "kca-backup.service").read_text() == "[Service]\nExecStart=new\n"
    assert (dest / "kca-new.timer").exists()
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "kca-new.timer"],
    ]


def test_install_systemd_units_is_noop_when_installed_copies_match(tmp_path) -> None:
    import subprocess

    from src.tools import code_sync

    repo = tmp_path / "repo"
    src = repo / "deploy" / "systemd"
    src.mkdir(parents=True)
    dest = tmp_path / "user"
    dest.mkdir()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    (src / "kca-collect.timer").write_text("t\n")
    (dest / "kca-collect.timer").write_text("t\n")

    # When
    result = code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=fake_run)

    # Then
    assert result == code_sync.UnitInstallResult(changed=(), removed=(), enabled=())
    assert calls == []


def test_install_systemd_units_disables_and_removes_units_deleted_from_repo(tmp_path) -> None:
    import subprocess

    from src.tools import code_sync

    repo = tmp_path / "repo"
    src = repo / "deploy" / "systemd"
    src.mkdir(parents=True)
    dest = tmp_path / "user"
    dest.mkdir()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
    (dest / "kca-old.timer").write_text("t\n")
    (dest / "kca-old.service").write_text("s\n")
    (dest / "quant-lake-backup.timer").write_text("other project\n")

    # When
    result = code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=fake_run)

    # Then
    assert result.removed == ("kca-old.service", "kca-old.timer")
    assert not (dest / "kca-old.timer").exists()
    assert not (dest / "kca-old.service").exists()
    assert (dest / "quant-lake-backup.timer").exists()
    assert calls == [
        ["systemctl", "--user", "disable", "--now", "kca-old.timer"],
        ["systemctl", "--user", "daemon-reload"],
    ]


def test_install_systemd_units_propagates_systemctl_failure(tmp_path) -> None:
    import subprocess

    import pytest

    from src.tools import code_sync

    repo = tmp_path / "repo"
    (repo / "deploy" / "systemd").mkdir(parents=True)
    (repo / "deploy" / "systemd" / "kca-predict.service").write_text("s\n")
    dest = tmp_path / "user"

    def failing_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=cmd)

    # When / Then
    with pytest.raises(subprocess.CalledProcessError):
        code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=failing_run)


def test_ensure_secret_permissions_tightens_group_readable_env(tmp_path) -> None:
    from src.tools import code_sync

    # Given: 운영 VPS 실측과 같은 664 권한
    env = tmp_path / ".env"
    env.write_text("KIS_APP_SECRET=x\n")
    env.chmod(0o664)

    # When / Then
    assert code_sync.ensure_secret_permissions(str(tmp_path)) is True
    assert env.stat().st_mode & 0o777 == 0o600
    assert code_sync.ensure_secret_permissions(str(tmp_path)) is False
    assert code_sync.ensure_secret_permissions(str(tmp_path / "absent")) is False


def test_sync_repo_does_not_install_units_when_test_gate_fails(monkeypatch) -> None:
    from src.tools.code_sync import sync_repo

    def fake_git(args: list[str], cwd: str) -> str:
        if args[:2] == ["rev-parse", "HEAD"]:
            return "a" * 40
        if args[:2] == ["rev-parse", "FETCH_HEAD"]:
            return "b" * 40
        return ""

    monkeypatch.setattr("src.tools.alerts.dispatch_failure_alert", lambda unit, *, detail="": {})

    def fail_if_called(*args, **kwargs):
        raise AssertionError("units/secrets must not be touched after a rollback")

    # When
    result = sync_repo(
        "/repo", git_fn=fake_git, fast_forward_fn=lambda repo_dir, f, t: True,
        test_gate_fn=lambda repo_dir: (False, "1 failed"), uv_sync_fn=fail_if_called,
        install_units_fn=fail_if_called, secure_fn=fail_if_called,
    )

    # Then
    assert result.reason == "test_gate_failed"
    assert result.units is None


