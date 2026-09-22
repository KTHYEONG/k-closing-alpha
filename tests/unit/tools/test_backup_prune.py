def test_expired_snapshot_dirs_uses_directory_date_not_file_age() -> None:
    import pandas as pd

    from src.tools import backup_prune

    names = ["2026-08-31", "2026-09-01", "2026-09-30", "notes", "2026-13-40"]

    # When: 기준일 2026-10-01, 보존 30일 -> 컷오프 2026-09-01
    expired = backup_prune.expired_snapshot_dirs(names, pd.Timestamp("2026-10-01"))

    # Then: 경계일(09-01)은 보존, 그 이전만 만료
    assert expired == ["2026-08-31"]



def test_prune_backups_purges_expired_dirs_and_skips_missing_subtree(monkeypatch) -> None:
    import subprocess

    import pandas as pd

    from src.tools import backup_prune

    monkeypatch.setattr(backup_prune, "_resolve_rclone_bin", lambda: "rclone")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "lsf" and cmd[-1].endswith("/data"):
            return subprocess.CompletedProcess(cmd, 0, stdout="2026-08-01/\n2026-09-30/\n", stderr="")
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, backup_prune.RCLONE_EXIT_DIRECTORY_NOT_FOUND, stdout="", stderr="directory not found")
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    # When
    purged = backup_prune.prune_backups(today=pd.Timestamp("2026-10-01"), run_fn=fake_run, remote_root="gdrive:x/_deleted")

    # Then
    assert purged == ["gdrive:x/_deleted/data/2026-08-01"]
    assert ["rclone", "purge", "gdrive:x/_deleted/data/2026-08-01"] in calls
    assert not any(c[1] == "purge" and "artifacts" in c[-1] for c in calls)



def test_prune_backups_raises_on_listing_failure_other_than_missing_subtree(monkeypatch) -> None:
    import subprocess

    import pandas as pd
    import pytest

    from src.tools import backup_prune

    monkeypatch.setattr(backup_prune, "_resolve_rclone_bin", lambda: "rclone")

    def failing_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="couldn't fetch token")

    # When / Then
    with pytest.raises(subprocess.CalledProcessError):
        backup_prune.prune_backups(today=pd.Timestamp("2026-10-01"), run_fn=failing_run)



def test_resolve_rclone_bin_prefers_path_then_local_bin(monkeypatch) -> None:
    from pathlib import Path

    from src.tools import backup_prune

    monkeypatch.setattr(backup_prune.shutil, "which", lambda name: "/usr/bin/rclone" if name == "rclone" else None)
    assert backup_prune._resolve_rclone_bin() == "/usr/bin/rclone"

    monkeypatch.setattr(backup_prune.shutil, "which", lambda name: None)
    assert backup_prune._resolve_rclone_bin() == str(Path.home() / ".local" / "bin" / "rclone")


def test_prune_local_intraday_backups_removes_only_expired_date_dirs(tmp_path) -> None:
    import pandas as pd

    from src.tools import backup_prune

    # Given: 만료분(컷오프 엄격 이전)과 보존분이 공존 (today=2026-09-21, retention=3 -> 컷오프 2026-09-18)
    root = tmp_path
    expired_dir = root / "regular" / "2026-09-17"
    kept_dir = root / "regular" / "2026-09-20"
    expired_dir.mkdir(parents=True)
    kept_dir.mkdir(parents=True)
    (expired_dir / "2026-09-17-pre-abc.parquet").write_bytes(b"dummy")
    (kept_dir / "2026-09-20-pre-def.parquet").write_bytes(b"dummy")

    # When
    purged = backup_prune.prune_local_intraday_backups(
        today=pd.Timestamp("2026-09-21"), backups_root=root, retention_days=3
    )

    # Then
    assert purged == ["regular/2026-09-17"]
    assert not expired_dir.exists()
    assert kept_dir.exists()


def test_prune_local_intraday_backups_covers_every_session(tmp_path) -> None:
    import pandas as pd

    from src.tools import backup_prune

    # Given
    root = tmp_path
    for session in ("regular", "krx_aftermarket"):
        target = root / session / "2026-09-17"
        target.mkdir(parents=True)
        (target / "snap-pre-abc.parquet").write_bytes(b"dummy")

    # When
    purged = backup_prune.prune_local_intraday_backups(
        today=pd.Timestamp("2026-09-21"), backups_root=root, retention_days=3
    )

    # Then
    assert purged == ["krx_aftermarket/2026-09-17", "regular/2026-09-17"]
    assert not (root / "regular" / "2026-09-17").exists()
    assert not (root / "krx_aftermarket" / "2026-09-17").exists()


def test_prune_local_intraday_backups_returns_empty_when_root_missing(tmp_path) -> None:
    import pandas as pd

    from src.tools import backup_prune

    # Given
    missing = tmp_path / "never-created"

    # When
    purged = backup_prune.prune_local_intraday_backups(
        today=pd.Timestamp("2026-09-21"), backups_root=missing, retention_days=3
    )

    # Then
    assert purged == []


def test_prune_local_intraday_backups_ignores_non_date_dirs(tmp_path) -> None:
    import pandas as pd

    from src.tools import backup_prune

    # Given
    root = tmp_path
    notes = root / "regular" / "notes"
    expired_dir = root / "regular" / "2026-09-17"
    notes.mkdir(parents=True)
    expired_dir.mkdir(parents=True)
    (notes / "memo.txt").write_text("keep", encoding="utf-8")

    # When
    purged = backup_prune.prune_local_intraday_backups(
        today=pd.Timestamp("2026-09-21"), backups_root=root, retention_days=3
    )

    # Then
    assert purged == ["regular/2026-09-17"]
    assert notes.exists()
    assert not expired_dir.exists()


def test_main_purges_remote_and_local_with_single_summary_log(monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    from src.tools import backup_prune

    remote_calls: list = []
    local_calls: list = []

    def _fake_remote(*, today):
        remote_calls.append(today)
        return ["gdrive:x/_deleted/data/2026-08-01"]

    def _fake_local(*, today):
        local_calls.append(today)
        return ["regular/2026-09-17"]

    monkeypatch.setattr(backup_prune, "prune_backups", _fake_remote)
    monkeypatch.setattr(backup_prune, "prune_local_intraday_backups", _fake_local)

    # When
    with caplog.at_level(logging.INFO, logger="src.tools.backup_prune"):
        backup_prune.main([])

    # Then
    assert len(remote_calls) == 1
    assert len(local_calls) == 1
    assert isinstance(remote_calls[0], pd.Timestamp)
    assert isinstance(local_calls[0], pd.Timestamp)
    summary = [r for r in caplog.records if "purged=" in r.getMessage() and "local_purged=" in r.getMessage()]
    assert len(summary) == 1
    assert "local_targets=" in summary[0].getMessage()

