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

