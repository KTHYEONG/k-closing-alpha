def test_sync_repo_returns_up_to_date_when_no_new_commits() -> None:
    from src.tools.code_sync import SyncResult, sync_repo

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

    # When
    result = sync_repo(
        "/repo", git_fn=fake_git, fast_forward_fn=fail_if_called,
        test_gate_fn=fail_if_called, uv_sync_fn=fail_if_called,
    )

    # Then
    assert result == SyncResult(updated=False, from_sha="a" * 40, to_sha="a" * 40, reason="up_to_date")


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
    from src.tools.code_sync import SyncResult, sync_repo

    sync_calls: list[str] = []

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

    # When
    result = sync_repo(
        "/repo", git_fn=fake_git, fast_forward_fn=lambda repo_dir, f, t: True,
        test_gate_fn=lambda repo_dir: (True, ""), uv_sync_fn=lambda repo_dir: sync_calls.append(repo_dir),
    )

    # Then
    assert result == SyncResult(updated=True, from_sha="a" * 40, to_sha="b" * 40, reason="fast_forwarded")
    assert sync_calls == ["/repo"]


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
    assert captured["cmd"] == ["uv", "run", "pytest", "-q"]
    assert captured["cwd"] == str(tmp_path)
    assert len(tail) == code_sync.ALERT_DETAIL_TAIL_CHARS


def test_uv_sync_invokes_uv_sync_command(monkeypatch, tmp_path) -> None:
    import subprocess

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
    assert captured["cmd"] == ["uv", "sync"]
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


