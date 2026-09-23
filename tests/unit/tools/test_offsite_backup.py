from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
UTC = ZoneInfo("UTC")


def _now_kst(day: str, clock: str = "23:00:00") -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}+09:00")


def _write_report(path: Path, *, status: str, started_at: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"started_at": started_at, "finished_at": started_at, "status": status, "steps": {}}), encoding="utf-8")


def test_loose_copy_data_excludes_segment_tiers_and_local_snapshots() -> None:
    from src.tools.offsite_backup import BACKUP_REMOTE_BASE, LOOSE_EXCLUDES, loose_copy_command

    # Given subtree data
    cmd = loose_copy_command("rclone", Path("/proj"), "data", "2026-09-18")

    # When building command / Then invariants
    assert cmd[:3] == ["rclone", "copy", "/proj/data"]
    assert cmd[3] == f"{BACKUP_REMOTE_BASE}/data"
    assert "--backup-dir" in cmd
    assert cmd[cmd.index("--backup-dir") + 1] == f"{BACKUP_REMOTE_BASE}/_deleted/data/2026-09-18"
    for pattern in LOOSE_EXCLUDES:
        assert pattern in cmd
    assert len(LOOSE_EXCLUDES) == 4
    assert "--transfers" in cmd and "4" in cmd
    assert "sync" not in cmd
    assert "--fast-list" not in cmd
    assert not any(part.startswith("--delete") for part in cmd)


def test_loose_copy_artifacts_has_no_capture_excludes() -> None:
    from src.tools.offsite_backup import BACKUP_REMOTE_BASE, loose_copy_command

    # Given subtree artifacts / Then no excludes
    cmd = loose_copy_command("rclone", Path("/proj"), "artifacts", "2026-09-18")
    assert cmd[3] == f"{BACKUP_REMOTE_BASE}/artifacts"
    assert cmd[cmd.index("--backup-dir") + 1] == f"{BACKUP_REMOTE_BASE}/_deleted/artifacts/2026-09-18"
    assert "--exclude" not in cmd


def test_run_executes_seal_before_loose_copies(tmp_path: Path, monkeypatch) -> None:
    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import REPORT_RELPATH, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    order: list[str] = []

    def _seal(capture_root: Path, *, today, full_scan) -> SealReport:
        order.append("capture_seal")
        return SealReport(dates_scanned=1, segments_committed=2, members_committed=3, archive_bytes=4, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        if "data" in cmd[2]:
            order.append("data")
        elif "artifacts" in cmd[2]:
            order.append("artifacts")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    # When running
    report = run_offsite_backup(tmp_path, tmp_path / "capture", now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)

    # Then order and persistence
    assert order == ["capture_seal", "data", "artifacts"]
    assert report.status == "ok"
    persisted = json.loads((tmp_path / "capture" / REPORT_RELPATH).read_text(encoding="utf-8"))
    assert persisted["status"] == "ok"
    assert persisted["steps"]["capture_seal"]["segments_committed"] == 2


def test_run_continues_after_seal_failure_and_raises_after_report(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.offsite_backup import REPORT_RELPATH, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    ran: list[str] = []

    def _boom(capture_root: Path, *, today, full_scan):
        raise RuntimeError("seal down")

    def _run(cmd, **kwargs):
        ran.append(cmd[2])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    # When seal fails / Then both copies still run, report failed, RuntimeError after report
    with pytest.raises(RuntimeError, match=r"offsite backup failed"):
        run_offsite_backup(tmp_path, tmp_path / "capture", now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_boom)
    assert len(ran) == 2
    persisted = json.loads((tmp_path / "capture" / REPORT_RELPATH).read_text(encoding="utf-8"))
    assert persisted["status"] == "failed"
    assert persisted["steps"]["capture_seal"]["status"] == "failed"
    assert "seal down" in persisted["steps"]["capture_seal"]["error"]


def test_run_requests_full_scan_only_on_configured_weekday(tmp_path: Path, monkeypatch) -> None:
    from src.tools.offsite_backup import run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    seen: dict[str, object] = {}

    def _seal(capture_root: Path, *, today, full_scan):
        seen["today"] = today
        seen["full_scan"] = full_scan
        from src.tools.capture_offsite import SealReport

        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    # Given Friday KST -> full scan
    run_offsite_backup(tmp_path, tmp_path / "c1", now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)
    assert seen["full_scan"] is True
    assert str(seen["today"]) == "2026-09-18"

    # Given Tuesday KST -> no full scan
    run_offsite_backup(tmp_path, tmp_path / "c2", now=_now_kst("2026-09-15"), run_fn=_run, seal_fn=_seal)
    assert seen["full_scan"] is False

    # And KST date wins when UTC date differs (15:30 UTC 09-15 == 00:30 KST 09-16)
    utc_now = datetime(2026, 9, 15, 15, 30, tzinfo=UTC)
    run_offsite_backup(tmp_path, tmp_path / "c3", now=utc_now, run_fn=_run, seal_fn=_seal)
    assert str(seen["today"]) == "2026-09-16"
    assert seen["full_scan"] is False


def test_expected_slot_skips_weekend() -> None:
    from src.tools.offsite_backup import expected_backup_slot

    # Given Monday 20:15 audit -> previous Friday 22:15 slot
    slot = expected_backup_slot(datetime.fromisoformat("2026-09-21T20:15:00+09:00"))
    assert (slot.year, slot.month, slot.day, slot.hour, slot.minute) == (2026, 9, 18, 22, 15)

    # Given Wednesday 20:15 -> Tuesday 22:15
    slot2 = expected_backup_slot(datetime.fromisoformat("2026-09-23T20:15:00+09:00"))
    assert (slot2.year, slot2.month, slot2.day, slot2.hour, slot2.minute) == (2026, 9, 22, 22, 15)


def test_staleness_classifies_missing_failed_stale_ok(tmp_path: Path) -> None:
    from src.tools.offsite_backup import backup_staleness_issues

    audit_at = datetime.fromisoformat("2026-09-21T20:15:00+09:00")

    # Given no report -> missing
    assert backup_staleness_issues(tmp_path / "missing.json", audit_at) == ["offsite_backup:missing"]

    # Given failed report -> failed (precedence over stale)
    failed = tmp_path / "failed.json"
    _write_report(failed, status="failed", started_at="2026-09-18T13:15:00+00:00")
    assert backup_staleness_issues(failed, audit_at) == ["offsite_backup:failed"]

    # Given ok report started before slot -> stale
    stale = tmp_path / "stale.json"
    _write_report(stale, status="ok", started_at="2026-09-17T13:15:00+00:00")
    assert backup_staleness_issues(stale, audit_at) == ["offsite_backup:stale"]

    # Given ok report after slot -> []
    ok = tmp_path / "ok.json"
    _write_report(ok, status="ok", started_at="2026-09-18T13:30:00+00:00")
    assert backup_staleness_issues(ok, audit_at) == []

    # And unreadable report
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert backup_staleness_issues(broken, audit_at) == ["offsite_backup:unreadable"]

    # And non-dict / missing keys / bad timestamp variants
    non_dict = tmp_path / "non_dict.json"
    non_dict.write_text("[1, 2]", encoding="utf-8")
    assert backup_staleness_issues(non_dict, audit_at) == ["offsite_backup:unreadable"]
    missing_keys = tmp_path / "missing_keys.json"
    missing_keys.write_text("{}", encoding="utf-8")
    assert backup_staleness_issues(missing_keys, audit_at) == ["offsite_backup:unreadable"]
    bad_ts = tmp_path / "bad_ts.json"
    bad_ts.write_text(json.dumps({"status": "ok", "started_at": "not-a-date"}), encoding="utf-8")
    assert backup_staleness_issues(bad_ts, audit_at) == ["offsite_backup:unreadable"]
    naive_ok = tmp_path / "naive_ok.json"
    naive_ok.write_text(json.dumps({"status": "ok", "started_at": "2026-09-18T22:30:00"}), encoding="utf-8")
    assert backup_staleness_issues(naive_ok, audit_at) == []


def test_run_handles_naive_now_and_loose_failures(tmp_path: Path, monkeypatch) -> None:
    import pytest
    from datetime import datetime as _dt

    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")

    def _seal(capture_root: Path, *, today, full_scan) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _ok(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    # Given naive now assumed KST Friday -> full scan path covered
    seen: dict[str, object] = {}

    def _recording_seal(capture_root: Path, *, today, full_scan):
        seen["today"] = today
        seen["full_scan"] = full_scan
        return _seal(capture_root, today=today, full_scan=full_scan)

    report = run_offsite_backup(tmp_path, tmp_path / "cnaive", now=_dt(2026, 9, 18, 22, 0), run_fn=_ok, seal_fn=_recording_seal)
    assert report.status == "ok"
    assert str(seen["today"]) == "2026-09-18"

    # Given loose copy raising -> step failed but later steps continue, report raised
    def _raising(cmd, **kwargs):
        if "data" in cmd[2]:
            raise OSError("drive down")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match=r"offsite backup failed"):
        run_offsite_backup(tmp_path, tmp_path / "craise", now=_now_kst("2026-09-18"), run_fn=_raising, seal_fn=_seal)

    # Given loose copy non-zero -> stderr tail kept and overall failure
    def _failing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="x" * 3000)

    with pytest.raises(RuntimeError, match=r"offsite backup failed"):
        run_offsite_backup(tmp_path, tmp_path / "cfail", now=_now_kst("2026-09-18"), run_fn=_failing, seal_fn=_seal)
    persisted = json.loads((tmp_path / "cfail" / "offsite" / "last_run.json").read_text(encoding="utf-8"))
    assert persisted["steps"]["data"]["status"] == "failed"
    assert len(persisted["steps"]["data"]["stderr"]) <= 2000


def test_backup_slot_constant_matches_timer_schedule() -> None:
    import re
    from datetime import time
    from pathlib import Path as _Path

    from src.tools.offsite_backup import BACKUP_SLOT_KST, BACKUP_SLOT_WEEKDAYS

    # Given kca-backup.timer
    text = (_Path("deploy/systemd/kca-backup.timer")).read_text(encoding="utf-8")
    match = re.search(r"OnCalendar=(\w+)\.\.(\w+) (\d{2}):(\d{2}):\d{2}", text)
    assert match is not None
    day_index = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
    start, end = day_index[match.group(1)], day_index[match.group(2)]
    assert frozenset(range(start, end + 1)) == BACKUP_SLOT_WEEKDAYS
    assert time(int(match.group(3)), int(match.group(4))) == BACKUP_SLOT_KST
