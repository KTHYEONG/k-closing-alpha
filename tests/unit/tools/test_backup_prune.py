import pytest


@pytest.fixture(autouse=True)
def _isolated_runtime_dir(monkeypatch, tmp_path) -> None:
    # The remediation step joins $XDG_RUNTIME_DIR/kca-audit.lock; keep it off the real runtime dir.
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))


@pytest.fixture(autouse=True)
def _stub_ledger_compact(monkeypatch) -> None:
    from src.tools import backup_prune
    from src.tools.capture_offsite import LedgerCompactionReport

    monkeypatch.setattr(
        backup_prune,
        "compact_old_ledgers",
        lambda *a, **kw: LedgerCompactionReport(compacted=(), kept=(), bytes_before=0, bytes_after=0),
    )


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


def _touch(path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_prune_local_ops_logs_removes_only_day_files_past_retention(tmp_path) -> None:
    import pandas as pd

    from src.tools import backup_prune

    # Given today=2026-11-23, retention=60 -> cutoff 2026-09-24 (strictly older is expired)
    data = tmp_path
    expired = [
        data / "logs/audit_digests/2026-09-23.txt",
        data / "logs/alerts/digest/2026-09-01.jsonl",
        data / "logs/alerts/silent/2026-08-31.jsonl",
        data / "logs/events/2026-08/2026-08-01.jsonl",
        data / "logs/events/2026-09/2026-09-23.jsonl",
    ]
    kept = [
        data / "logs/audit_digests/2026-09-24.txt",
        data / "logs/alerts/digest/2026-11-23.jsonl",
        data / "logs/audit_digests/README.txt",
        data / "logs/alerts/incidents.json",
        data / "logs/heartbeat/daily_audit.json",
        data / "logs/remediation/ledger.jsonl",
        data / "logs/events/2026-09/2026-09-24.jsonl",
        data / "logs/events/2026-09/notes.txt",
    ]
    for path in (*expired, *kept):
        _touch(path)

    # When
    purged = backup_prune.prune_local_ops_logs(today=pd.Timestamp("2026-11-23"), data_dir=data, retention_days=60)

    # Then
    assert purged == [
        "logs/alerts/digest/2026-09-01.jsonl",
        "logs/alerts/silent/2026-08-31.jsonl",
        "logs/audit_digests/2026-09-23.txt",
        "logs/events/2026-08/2026-08-01.jsonl",
        "logs/events/2026-09/2026-09-23.jsonl",
    ]
    assert not any(path.exists() for path in expired)
    assert all(path.exists() for path in kept)
    # An emptied month dir goes; a month dir that still holds files stays
    assert not (data / "logs/events/2026-08").exists()
    assert (data / "logs/events/2026-09").is_dir()


def test_prune_local_ops_logs_removes_orphan_sentinel_once(tmp_path, caplog) -> None:
    import logging

    import pandas as pd

    from src.tools import backup_prune

    # Given the orphan left by removed alert code
    orphan = tmp_path / "logs/alerts/failure_sent.json"
    _touch(orphan, "{}")

    # When pruning twice
    with caplog.at_level(logging.INFO, logger="src.tools.backup_prune"):
        first = backup_prune.prune_local_ops_logs(today=pd.Timestamp("2026-11-23"), data_dir=tmp_path)
    second = backup_prune.prune_local_ops_logs(today=pd.Timestamp("2026-11-23"), data_dir=tmp_path)

    # Then it is removed and logged once, and the second run is a no-op
    assert first == ["logs/alerts/failure_sent.json"]
    assert not orphan.exists()
    assert second == []
    assert any("orphan_removed=logs/alerts/failure_sent.json" in r.getMessage() for r in caplog.records)


def test_daily_audit_reads_outcomes_for_a_day_inside_the_window_after_prune() -> None:
    import pandas as pd

    from src.tools import backup_prune
    from src.tools.run_outcome import RUN_OUTCOME_OK, load_run_outcomes, record_run_outcome

    # Given outcomes recorded 60 days (kept) and 61 days (expired) before today
    today = pd.Timestamp("2026-11-23")
    inside = (today - pd.Timedelta(days=60)).date().isoformat()
    outside = (today - pd.Timedelta(days=61)).date().isoformat()
    for day in (inside, outside):
        record_run_outcome("predict", RUN_OUTCOME_OK, run_date=day)

    # When the nightly prune runs on DATA_DIR
    backup_prune.prune_local_ops_logs(today=today)

    # Then a --date rerun of daily_audit inside the window still sees its outcomes
    assert load_run_outcomes(inside) == {"predict": RUN_OUTCOME_OK}
    assert load_run_outcomes(outside) == {}


def test_prune_local_ops_logs_dry_run_and_missing_dirs(tmp_path) -> None:
    import pandas as pd

    from src.tools import backup_prune

    # Given one expired file and no other ops-log dirs
    target = tmp_path / "logs/alerts/silent/2026-01-02.jsonl"
    _touch(target)

    # When
    listed = backup_prune.prune_local_ops_logs(today=pd.Timestamp("2026-11-23"), data_dir=tmp_path, dry_run=True)

    # Then dry run lists without deleting, and the default window is the typed 60-day constant
    assert listed == ["logs/alerts/silent/2026-01-02.jsonl"]
    assert target.exists()
    assert backup_prune.OPS_LOG_RETENTION_DAYS == 60
    assert backup_prune.prune_local_ops_logs(today=pd.Timestamp("2026-11-23"), data_dir=tmp_path / "absent") == []


def test_main_runs_ops_log_prune_on_data_dir(monkeypatch, caplog) -> None:
    import logging
    from pathlib import Path

    from src import settings
    from src.tools import backup_prune
    from src.tools.capture_offsite import LocalRetentionReport

    # Given an expired digest under the (test-isolated) DATA_DIR and stubbed heavy steps
    old = Path(settings.DATA_DIR) / "logs/audit_digests/2020-01-01.txt"
    _touch(old)
    monkeypatch.setattr(backup_prune, "prune_backups", lambda **kw: [])
    monkeypatch.setattr(backup_prune, "prune_local_intraday_backups", lambda **kw: [])
    monkeypatch.setattr(
        backup_prune, "prune_local_sealed_capture", lambda *a, **kw: LocalRetentionReport(removed=(), kept=(), bytes_removed=0)
    )
    monkeypatch.setenv("HISTORY_EVICTION_ENABLED", "false")

    # When
    with caplog.at_level(logging.INFO, logger="src.tools.backup_prune"):
        backup_prune.main([])

    # Then
    assert not old.exists()
    assert any("ops_logs_removed=1" in r.getMessage() for r in caplog.records)


def test_main_purges_remote_and_local_with_single_summary_log(monkeypatch, caplog) -> None:
    import logging

    import pandas as pd

    from src.tools import backup_prune

    remote_calls: list = []
    local_calls: list = []

    def _fake_remote(*, today, **kwargs):
        remote_calls.append(today)
        return ["gdrive:x/_deleted/data/2026-08-01"]

    def _fake_local(*, today, **kwargs):
        local_calls.append(today)
        return ["regular/2026-09-17"]

    monkeypatch.setattr(backup_prune, "prune_backups", _fake_remote)
    monkeypatch.setattr(backup_prune, "prune_local_intraday_backups", _fake_local)

    import src.tools.capture_offsite as capture_offsite

    def _fake_sealed(capture_root, *, today, **kwargs):
        return capture_offsite.LocalRetentionReport(
            removed=("raw/2026-08-01", "raw/2026-08-02"),
            kept=(("raw/2026-08-03", "remote_unverified"),),
            bytes_removed=99,
        )

    monkeypatch.setattr(backup_prune, "prune_local_sealed_capture", _fake_sealed)

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
    assert "sealed_removed=2" in summary[0].getMessage()
    assert "sealed_kept=1" in summary[0].getMessage()



def test_prune_cap_aborts_before_any_purge(monkeypatch) -> None:
    import subprocess

    import pandas as pd
    import pytest

    from src.tools import backup_prune

    monkeypatch.setattr(backup_prune, "_resolve_rclone_bin", lambda: "rclone")
    calls: list[list[str]] = []
    expired = "\n".join(f"2026-07-{day:02d}/" for day in range(1, 13)) + "\n"

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "lsf":
            if cmd[-1].endswith("/data"):
                return subprocess.CompletedProcess(cmd, 0, stdout=expired, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        raise AssertionError("purge must not run when cap exceeded")

    with pytest.raises(RuntimeError, match="cap exceeded"):
        backup_prune.prune_backups(today=pd.Timestamp("2026-10-01"), run_fn=fake_run, remote_root="gdrive:x/_deleted")
    assert not any(c[1] == "purge" for c in calls)


def test_prune_dry_run_lists_without_purging(monkeypatch) -> None:
    import subprocess

    import pandas as pd

    from src.tools import backup_prune

    monkeypatch.setattr(backup_prune, "_resolve_rclone_bin", lambda: "rclone")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1] == "lsf" and cmd[-1].endswith("/data"):
            return subprocess.CompletedProcess(cmd, 0, stdout="2026-08-01/\n2026-08-02/\n", stderr="")
        if cmd[1] == "lsf":
            return subprocess.CompletedProcess(cmd, backup_prune.RCLONE_EXIT_DIRECTORY_NOT_FOUND, stdout="", stderr="directory not found")
        raise AssertionError("purge must not run in dry-run")

    targets = backup_prune.prune_backups(
        today=pd.Timestamp("2026-10-01"), run_fn=fake_run, remote_root="gdrive:x/_deleted", dry_run=True
    )
    assert targets == ["gdrive:x/_deleted/data/2026-08-01", "gdrive:x/_deleted/data/2026-08-02"]
    assert not any(c[1] == "purge" for c in calls)


def test_prune_local_intraday_backups_dry_run_deletes_nothing(tmp_path) -> None:
    import pandas as pd

    from src.tools import backup_prune

    root = tmp_path
    expired_dir = root / "regular" / "2026-09-17"
    expired_dir.mkdir(parents=True)
    (expired_dir / "2026-09-17-pre-abc.parquet").write_bytes(b"dummy")

    purged = backup_prune.prune_local_intraday_backups(
        today=pd.Timestamp("2026-09-21"), backups_root=root, retention_days=3, dry_run=True
    )

    assert purged == ["regular/2026-09-17"]
    assert expired_dir.exists()


def test_backup_prune_main_runs_local_steps_after_remote_failure(monkeypatch) -> None:
    import pytest

    from src.tools import backup_prune

    calls: list[str] = []

    def _boom_remote(*, today, **kwargs):
        raise RuntimeError("remote down")

    def _ok_local(*, today, **kwargs):
        calls.append("local_intraday")
        return []

    import src.tools.capture_offsite as capture_offsite

    def _ok_sealed(capture_root, *, today, **kwargs):
        calls.append("local_sealed")
        return capture_offsite.LocalRetentionReport(removed=(), kept=(), bytes_removed=0)

    monkeypatch.setattr(backup_prune, "prune_backups", _boom_remote)
    monkeypatch.setattr(backup_prune, "prune_local_intraday_backups", _ok_local)
    monkeypatch.setattr(backup_prune, "prune_local_sealed_capture", _ok_sealed)

    with pytest.raises(SystemExit) as exc:
        backup_prune.main([])
    assert exc.value.code == 1
    assert calls == ["local_intraday", "local_sealed"]


def test_backup_prune_max_purge_override_is_forwarded(monkeypatch) -> None:
    from src.tools import backup_prune

    seen: dict[str, object] = {}

    def _fake_remote(*, today, dry_run=False, max_purge_per_subtree=7):
        seen["max_purge_per_subtree"] = max_purge_per_subtree
        seen["dry_run"] = dry_run
        return []

    def _fake_local(*, today, dry_run=False, **kwargs):
        return []

    import src.tools.capture_offsite as capture_offsite

    def _fake_sealed(capture_root, *, today, dry_run=False, **kwargs):
        seen["sealed_dry_run"] = dry_run
        return capture_offsite.LocalRetentionReport(removed=(), kept=(), bytes_removed=0)

    monkeypatch.setattr(backup_prune, "prune_backups", _fake_remote)
    monkeypatch.setattr(backup_prune, "prune_local_intraday_backups", _fake_local)
    monkeypatch.setattr(backup_prune, "prune_local_sealed_capture", _fake_sealed)

    backup_prune.main(["--max-purge-per-subtree", "30"])
    assert seen["max_purge_per_subtree"] == 30

    backup_prune.main(["--dry-run"])
    assert seen["dry_run"] is True
    assert seen["sealed_dry_run"] is True


def test_skipped_sealed_prune_is_not_a_failure(monkeypatch, caplog) -> None:
    import logging

    from src.tools import backup_prune

    def _fake_remote(*, today, **kwargs):
        return []

    def _fake_local(*, today, **kwargs):
        return []

    import src.tools.capture_offsite as capture_offsite

    def _fake_sealed(capture_root, *, today, **kwargs):
        return capture_offsite.LocalRetentionReport(
            removed=(), kept=(), bytes_removed=0, skipped_reason="seal_lock_held"
        )

    monkeypatch.setattr(backup_prune, "prune_backups", _fake_remote)
    monkeypatch.setattr(backup_prune, "prune_local_intraday_backups", _fake_local)
    monkeypatch.setattr(backup_prune, "prune_local_sealed_capture", _fake_sealed)

    with caplog.at_level(logging.INFO, logger="src.tools.backup_prune"):
        backup_prune.main([])

    summary = [r for r in caplog.records if "stage=backup_prune" in r.getMessage() and "dry_run=" in r.getMessage()]
    assert len(summary) == 1
    assert "sealed_skipped=seal_lock_held" in summary[0].getMessage()


def _stub_other_steps(monkeypatch) -> None:
    from src.tools import backup_prune
    import src.tools.capture_offsite as capture_offsite

    monkeypatch.setattr(backup_prune, "prune_backups", lambda **kw: [])
    monkeypatch.setattr(backup_prune, "prune_local_intraday_backups", lambda **kw: [])
    monkeypatch.setattr(
        backup_prune, "prune_local_sealed_capture",
        lambda *a, **kw: capture_offsite.LocalRetentionReport(removed=(), kept=(), bytes_removed=0),
    )


def test_prune_history_step_disabled_by_env(monkeypatch, caplog) -> None:
    import logging

    from src.tools import backup_prune

    _stub_other_steps(monkeypatch)
    monkeypatch.setenv("HISTORY_EVICTION_ENABLED", "false")

    def _never(**kwargs):
        raise AssertionError("eviction must not run when disabled")

    monkeypatch.setattr(backup_prune, "evict_verified_partitions", _never)
    with caplog.at_level(logging.INFO, logger="src.tools.backup_prune"):
        backup_prune.main([])

    assert any("step=local_history status=disabled" in r.getMessage() for r in caplog.records)
    assert any("history_evicted=0" in r.getMessage() for r in caplog.records)


def test_prune_history_step_runs_when_enabled(monkeypatch, tmp_path) -> None:
    from src import settings
    from src.data.partition_eviction import EvictionReport
    from src.tools import backup_prune

    _stub_other_steps(monkeypatch)
    monkeypatch.setenv("HISTORY_EVICTION_ENABLED", "true")
    monkeypatch.setattr(settings, "HISTORY_DIR", tmp_path)
    seen: dict = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        return EvictionReport(("a",), (), 5, False, False)

    monkeypatch.setattr(backup_prune, "evict_verified_partitions", _fake)
    backup_prune.main(["--dry-run"])

    assert seen["dry_run"] is True and seen["max_files"] == 400
    assert seen["remote_base"].endswith("/data/history")
    assert [f.name for f in seen["families"]] == ["bars", "ticks", "sameday"]


def test_prune_history_step_failure_isolated(monkeypatch) -> None:
    import pytest

    from src.tools import backup_prune

    ran: list[str] = []
    _stub_other_steps(monkeypatch)
    monkeypatch.setattr(backup_prune, "prune_backups", lambda **kw: ran.append("remote") or [])
    monkeypatch.setenv("HISTORY_EVICTION_ENABLED", "true")

    def _boom(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(backup_prune, "evict_verified_partitions", _boom)
    with pytest.raises(SystemExit) as excinfo:
        backup_prune.main([])

    assert excinfo.value.code == 1 and ran == ["remote"]


def test_ledger_compact_runs_after_sealed_and_reports_savings(monkeypatch, caplog) -> None:
    import logging

    from src.tools import backup_prune
    from src.tools.capture_offsite import LedgerCompactionReport, LocalRetentionReport

    calls: list[str] = []
    _stub_other_steps(monkeypatch)
    monkeypatch.setenv("HISTORY_EVICTION_ENABLED", "false")
    monkeypatch.setattr(
        backup_prune,
        "prune_local_sealed_capture",
        lambda *a, **kw: calls.append("sealed") or LocalRetentionReport(removed=(), kept=(), bytes_removed=0),
    )
    monkeypatch.setattr(
        backup_prune,
        "compact_old_ledgers",
        lambda *a, **kw: calls.append("ledger")
        or LedgerCompactionReport(compacted=("raw/2026-08-01",), kept=(), bytes_before=100, bytes_after=40),
    )

    with caplog.at_level(logging.INFO, logger=backup_prune.logger.name):
        backup_prune.main([])

    assert calls == ["sealed", "ledger"]
    assert any("ledger_compacted=1 ledger_bytes_saved=60" in rec.message for rec in caplog.records)


def test_remediation_compact_skips_when_writers_hold_the_lock(tmp_path, caplog) -> None:
    import fcntl
    import logging
    import os
    from datetime import timedelta

    import pandas as pd

    from src.tools import backup_prune

    # Given a writer holding the audit lock (as flock(1) does) and an expired ledger row
    lock = tmp_path / "kca-audit.lock"
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"ts": "2026-01-01T00:00:00+09:00", "issue_key": "k", "action": "rerun_unit", "unit": "u", "attempt": 1, "outcome": "resolved", "detail": ""}\n', encoding="utf-8")
    before = ledger.read_bytes()
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        # When
        with caplog.at_level(logging.WARNING, logger="src.tools.backup_prune"):
            report = backup_prune.compact_local_remediation_ledger(
                today=pd.Timestamp("2026-11-23"), lock_path=lock, ledger_path=ledger, lock_timeout=timedelta(seconds=0)
            )
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    # Then nothing is rewritten and the lock file (shared inode with flock(1)) survives
    assert report is None
    assert ledger.read_bytes() == before
    assert lock.exists()
    assert any("LOCK_BUSY" in r.getMessage() for r in caplog.records)

    # When the lock is free
    report = backup_prune.compact_local_remediation_ledger(today=pd.Timestamp("2026-11-23"), lock_path=lock, ledger_path=ledger)

    # Then the expired row is dropped and the lock file is kept
    assert report is not None and report.dropped == 1
    assert ledger.read_text(encoding="utf-8") == ""
    assert lock.exists()


def test_audit_writer_lock_path_requires_runtime_dir(monkeypatch) -> None:
    from src.tools import backup_prune

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    with pytest.raises(RuntimeError):
        backup_prune.audit_writer_lock_path()
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    assert str(backup_prune.audit_writer_lock_path()) == "/run/user/1000/kca-audit.lock"


def test_remediation_compact_waits_for_a_short_writer_then_runs(tmp_path, monkeypatch) -> None:
    import fcntl
    import os
    import threading
    from datetime import timedelta

    import pandas as pd

    from src.tools import backup_prune

    # Given a writer that releases the audit lock shortly after the compaction starts waiting
    monkeypatch.setattr(backup_prune, "AUDIT_WRITER_LOCK_POLL_SEC", 0.01)
    lock = tmp_path / "kca-audit.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)

    def _release() -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    timer = threading.Timer(0.1, _release)
    timer.start()
    try:
        # When
        report = backup_prune.compact_local_remediation_ledger(
            today=pd.Timestamp("2026-11-23"), lock_path=lock, ledger_path=tmp_path / "absent.jsonl", lock_timeout=timedelta(seconds=10)
        )
    finally:
        timer.join()

    # Then it acquired the lock instead of skipping
    assert report is not None and report.kept == 0
