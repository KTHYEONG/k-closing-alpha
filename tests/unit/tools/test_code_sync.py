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


def test_sync_units_fetches_resets_and_converges_in_order(monkeypatch) -> None:
    from src.tools.code_sync import UnitInstallResult, sync_units

    order: list[str] = []
    git_calls: list[list[str]] = []
    sha = "b" * 40

    def fake_git(args: list[str], cwd: str) -> str:
        git_calls.append(list(args))
        return ""

    units = UnitInstallResult(changed=("kca-daily-audit.service",), removed=(), enabled=())

    # When
    result = sync_units(
        "/repo", sha=sha, git_fn=fake_git,
        uv_sync_fn=lambda repo_dir: order.append("uv_sync"),
        secure_fn=lambda repo_dir: order.append("secure") or True,
        install_units_fn=lambda repo_dir: order.append("install") or units,
    )

    # Then
    assert git_calls == [["fetch", "origin", sha, "--quiet"], ["reset", "--hard", sha]]
    assert order == ["uv_sync", "secure", "install"]
    assert result == units


def test_sync_units_propagates_git_failure(monkeypatch) -> None:
    import subprocess

    from src.tools.code_sync import sync_units

    def fail_if_called(*args, **kwargs):
        raise AssertionError("uv/units must not run when the checkout fails")

    def failing_git(args: list[str], cwd: str) -> str:
        raise subprocess.CalledProcessError(returncode=1, cmd=["git", *args])

    import pytest

    with pytest.raises(subprocess.CalledProcessError):
        sync_units(
            "/repo", sha="c" * 40, git_fn=failing_git,
            uv_sync_fn=fail_if_called, secure_fn=fail_if_called, install_units_fn=fail_if_called,
        )


def test_code_sync_main_invokes_sync_units_with_settings_base_dir_and_sha(monkeypatch) -> None:
    from src import settings
    from src.tools import code_sync

    captured: dict = {}

    def fake_sync_units(repo_dir, *, sha, remote):
        captured.update(repo_dir=repo_dir, sha=sha, remote=remote)
        return code_sync.UnitInstallResult(changed=(), removed=(), enabled=())

    monkeypatch.setattr(code_sync, "sync_units", fake_sync_units)

    # When
    code_sync.main(["--sha", "d" * 40])

    # Then
    assert captured == {"repo_dir": str(settings.BASE_DIR), "sha": "d" * 40, "remote": "origin"}


def _write_units(directory, names_to_text: dict) -> None:
    for name, text in names_to_text.items():
        (directory / name).write_text(text)


def _fake_systemctl(calls: list):
    import subprocess

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    return fake_run


def test_install_systemd_units_leaves_new_optional_timers_disabled(tmp_path) -> None:
    from src.tools import code_sync

    repo = tmp_path / "repo"
    src = repo / "deploy" / "systemd"
    src.mkdir(parents=True)
    dest = tmp_path / "user"
    dest.mkdir()
    calls: list = []
    _write_units(src, {
        "kca-auction-close.timer": "close\n",
        "kca-auction-open.timer": "open\n",
        "kca-altdata-capture.timer": "alt\n",
    })

    result = code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=_fake_systemctl(calls))

    assert result.changed == ("kca-altdata-capture.timer", "kca-auction-close.timer", "kca-auction-open.timer")
    assert result.enabled == ()
    assert calls == [["systemctl", "--user", "daemon-reload"]]


def test_install_systemd_units_retains_previously_enabled_optional_timer(tmp_path) -> None:
    from src.tools import code_sync

    repo = tmp_path / "repo"
    src = repo / "deploy" / "systemd"
    src.mkdir(parents=True)
    dest = tmp_path / "user"
    dest.mkdir()
    calls: list = []
    _write_units(src, {"kca-auction-close.timer": "v2\n"})
    _write_units(dest, {"kca-auction-close.timer": "v1\n"})

    result = code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=_fake_systemctl(calls))

    assert result.changed == ("kca-auction-close.timer",)
    assert result.enabled == ()
    assert (dest / "kca-auction-close.timer").read_text() == "v2\n"
    assert calls == [["systemctl", "--user", "daemon-reload"]]


def test_install_systemd_units_still_auto_enables_ordinary_timer(tmp_path) -> None:
    from src.tools import code_sync

    repo = tmp_path / "repo"
    src = repo / "deploy" / "systemd"
    src.mkdir(parents=True)
    dest = tmp_path / "user"
    dest.mkdir()
    calls: list = []
    _write_units(src, {"kca-collect.timer": "t\n"})

    result = code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=_fake_systemctl(calls))

    assert result.enabled == ("kca-collect.timer",)
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "kca-collect.timer"],
    ]


def test_install_systemd_units_removes_deleted_optional_unit(tmp_path) -> None:
    from src.tools import code_sync

    repo = tmp_path / "repo"
    src = repo / "deploy" / "systemd"
    src.mkdir(parents=True)
    dest = tmp_path / "user"
    dest.mkdir()
    calls: list = []
    _write_units(dest, {"kca-auction-open.timer": "old\n", "kca-auction-open.service": "old\n"})

    result = code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=_fake_systemctl(calls))

    assert result.removed == ("kca-auction-open.service", "kca-auction-open.timer")
    assert not (dest / "kca-auction-open.timer").exists()
    assert calls == [
        ["systemctl", "--user", "disable", "--now", "kca-auction-open.timer"],
        ["systemctl", "--user", "daemon-reload"],
    ]


def test_install_systemd_units_keeps_existing_trading_schedule_intact(tmp_path) -> None:
    from src.tools import code_sync

    repo = tmp_path / "repo"
    src = repo / "deploy" / "systemd"
    src.mkdir(parents=True)
    dest = tmp_path / "user"
    dest.mkdir()
    calls: list = []
    trading = "[Timer]\nOnCalendar=Mon..Fri 15:20:00 Asia/Seoul\n"
    _write_units(src, {"kca-collect.timer": trading, "kca-auction-close.timer": "close\n"})
    _write_units(dest, {"kca-collect.timer": trading})

    result = code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=_fake_systemctl(calls))

    assert result.changed == ("kca-auction-close.timer",)
    assert result.enabled == ()
    assert (dest / "kca-collect.timer").read_text() == trading


def test_install_systemd_units_second_run_is_idempotent(tmp_path) -> None:
    from src.tools import code_sync

    repo = tmp_path / "repo"
    src = repo / "deploy" / "systemd"
    src.mkdir(parents=True)
    dest = tmp_path / "user"
    dest.mkdir()
    _write_units(src, {"kca-auction-close.timer": "close\n", "kca-collect.timer": "t\n"})
    _write_units(dest, {"kca-collect.timer": "t\n"})
    calls: list = []
    first = code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=_fake_systemctl(calls))
    assert first.changed == ("kca-auction-close.timer",)

    second_calls: list = []
    second = code_sync.install_systemd_units(str(repo), dest_dir=dest, run_fn=_fake_systemctl(second_calls))

    assert second == code_sync.UnitInstallResult(changed=(), removed=(), enabled=())
    assert second_calls == []




