from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

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
    assert len(LOOSE_EXCLUDES) == 5
    assert "/history/capture/manifests/**" in LOOSE_EXCLUDES
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

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
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


def test_report_written_without_temp(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import REPORT_RELPATH, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=1, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    run_offsite_backup(tmp_path, tmp_path / "capture", now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)

    report_path = tmp_path / "capture" / REPORT_RELPATH
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "ok"
    assert list(report_path.parent.glob("*.tmp")) == []


def test_run_continues_after_seal_failure_and_raises_after_report(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from src.tools.offsite_backup import REPORT_RELPATH, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    ran: list[str] = []

    def _boom(capture_root: Path, *, today, full_scan, deadline=None):
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

    def _seal(capture_root: Path, *, today, full_scan, deadline=None):
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

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _ok(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    # Given naive now assumed KST Friday -> full scan path covered
    seen: dict[str, object] = {}

    def _recording_seal(capture_root: Path, *, today, full_scan, deadline=None):
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


def _write_core_parquet(path, rows: int, max_date: str) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": [max_date] * rows, "value": range(rows)}).to_parquet(path, index=False)


def test_corrupt_core_panel_excluded_from_nightly_copy(tmp_path: Path, monkeypatch) -> None:
    import json
    import subprocess

    import pytest

    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import REPORT_RELPATH, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    project = tmp_path / "proj"
    capture = tmp_path / "capture"
    _write_core_parquet(project / "data/history/price_history.parquet", 900, "2026-09-23")
    capture.mkdir(parents=True, exist_ok=True)
    (capture / "offsite").mkdir(parents=True, exist_ok=True)
    (capture / REPORT_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    (capture / REPORT_RELPATH).write_text(
        json.dumps(
            {
                "started_at": "2026-09-17T13:15:00+00:00",
                "finished_at": "2026-09-17T13:15:00+00:00",
                "status": "ok",
                "steps": {},
                "core_panels": [
                    {
                        "relpath": "data/history/price_history.parquet",
                        "sha256": "a" * 64,
                        "bytes": 10,
                        "rows": 1000,
                        "max_date": "2026-09-23",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    seen: list[list[str]] = []

    def _run(cmd, **kwargs):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="offsite backup failed"):
        run_offsite_backup(project, capture, now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)
    data_cmds = [c for c in seen if len(c) > 2 and c[2].endswith("/data")]
    assert len(data_cmds) == 1
    assert "--exclude" in data_cmds[0]
    assert "/history/price_history.parquet" in data_cmds[0]
    persisted = json.loads((capture / REPORT_RELPATH).read_text(encoding="utf-8"))
    assert persisted["steps"]["core_panels"]["status"] == "failed"
    assert any("rows_shrank" in issue for issue in persisted["steps"]["core_panels"]["issues"])
    assert persisted["steps"]["data"]["status"] == "failed"


def test_healthy_panels_leave_copy_unchanged(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import loose_copy_command, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    project = tmp_path / "proj"
    capture = tmp_path / "capture"

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    seen: list[list[str]] = []

    def _run(cmd, **kwargs):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    report = run_offsite_backup(project, capture, now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)
    assert report.status == "ok"
    data_cmds = [c for c in seen if len(c) > 2 and c[2].endswith("/data")]
    assert len(data_cmds) == 1
    expected = loose_copy_command("rclone", project, "data", "2026-09-18")
    assert data_cmds[0][: len(expected)] == expected
    assert "--max-duration" in data_cmds[0] and "--cutoff-mode" in data_cmds[0]


def test_offsite_helpers_cover_error_branches(tmp_path: Path) -> None:
    import json

    from src.tools.offsite_backup import _excludes_for_subtree, _load_previous_core_panels, loose_copy_command

    assert _excludes_for_subtree(["bad-issue"], "data") == []
    assert _excludes_for_subtree(["core_panel:data/history/a.parquet:missing"], "artifacts") == []
    cmd = loose_copy_command("rclone", tmp_path, "artifacts", "2026-09-18", ["/x.parquet"])
    assert "/x.parquet" in cmd
    from src.tools.offsite_backup import REPORT_RELPATH as _RR

    def _cap_with(content: str) -> Path:
        cap = tmp_path / f"cap_{abs(hash(content)) % 100000}"
        target = cap / _RR
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return cap

    assert _load_previous_core_panels(tmp_path / "nope") == []
    assert _load_previous_core_panels(_cap_with("{bad")) == []
    assert _load_previous_core_panels(_cap_with("[1]")) == []
    assert _load_previous_core_panels(_cap_with(json.dumps({"a": 1}))) == []
    assert _load_previous_core_panels(_cap_with(json.dumps({"core_panels": "nope"}))) == []


def test_offsite_failed_copy_keeps_core_issues(tmp_path: Path, monkeypatch) -> None:
    import json
    import subprocess

    import pytest

    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import REPORT_RELPATH, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    project = tmp_path / "proj"
    capture = tmp_path / "capture2"
    _write_core_parquet(project / "data/history/price_history.parquet", 900, "2026-09-23")
    (capture / REPORT_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    (capture / REPORT_RELPATH).write_text(
        json.dumps(
            {
                "started_at": "2026-09-17T13:15:00+00:00",
                "finished_at": "2026-09-17T13:15:00+00:00",
                "status": "ok",
                "steps": {},
                "core_panels": [
                    {"relpath": "data/history/price_history.parquet", "sha256": "a" * 64, "bytes": 1, "rows": 1000, "max_date": "2026-09-23"}
                ],
            }
        ),
        encoding="utf-8",
    )

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="drive down")

    with pytest.raises(RuntimeError, match="offsite backup failed"):
        run_offsite_backup(project, capture, now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)
    persisted = json.loads((capture / REPORT_RELPATH).read_text(encoding="utf-8"))
    assert persisted["steps"]["data"]["issues"][0].endswith("rows_shrank")


def test_offsite_first_run_unreadable_persists_current(tmp_path: Path, monkeypatch) -> None:
    import json
    import subprocess

    import pytest

    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import REPORT_RELPATH, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    project = tmp_path / "proj_first"
    capture = tmp_path / "cap_first"
    target = project / "data/history/price_history.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"truncated")

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with pytest.raises(RuntimeError, match="offsite backup failed"):
        run_offsite_backup(project, capture, now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)
    persisted = json.loads((capture / REPORT_RELPATH).read_text(encoding="utf-8"))
    assert persisted["steps"]["core_panels"]["status"] == "failed"
    assert persisted["core_panels"]


def test_offsite_backup_accepted_missing_persists_current_baseline(tmp_path: Path, monkeypatch) -> None:
    import json
    import subprocess

    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import REPORT_RELPATH, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    project = tmp_path / "proj_acc"
    capture = tmp_path / "cap_acc"
    _write_core_parquet(project / "data/history/price_history.parquet", 10, "2026-09-23")
    (capture / REPORT_RELPATH).parent.mkdir(parents=True, exist_ok=True)
    (capture / REPORT_RELPATH).write_text(
        json.dumps(
            {
                "started_at": "2026-09-17T13:15:00+00:00",
                "finished_at": "2026-09-17T13:15:00+00:00",
                "status": "ok",
                "steps": {},
                "core_panels": [
                    {
                        "relpath": "data/history/price_history.parquet",
                        "sha256": "a" * 64,
                        "bytes": 10,
                        "rows": 10,
                        "max_date": "2026-09-23",
                    },
                    {
                        "relpath": "data/paper/x.parquet",
                        "sha256": "b" * 64,
                        "bytes": 5,
                        "rows": 5,
                        "max_date": "2026-09-20",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    report = run_offsite_backup(
        project, capture, now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal, accepted_missing=frozenset({"data/paper/x.parquet"})
    )
    assert report.status == "ok"
    assert report.steps["core_panels"]["status"] == "ok"
    persisted = json.loads((capture / REPORT_RELPATH).read_text(encoding="utf-8"))
    rels = {item["relpath"] for item in persisted["core_panels"]}
    assert "data/paper/x.parquet" not in rels
    assert "data/history/price_history.parquet" in rels


def test_loose_copy_carries_remaining_total_budget(tmp_path: Path, monkeypatch) -> None:
    import subprocess
    from datetime import datetime as _datetime

    from src.tools import offsite_backup as _ob
    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    real_datetime = _datetime

    class _FakeDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime(2026, 9, 18, 23, 55, tzinfo=KST)

    monkeypatch.setattr(_ob, "datetime", _FakeDatetime)

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    seen: list[list[str]] = []

    def _run(cmd, **kwargs):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    report = run_offsite_backup(tmp_path, tmp_path / "cbudget", now=_now_kst("2026-09-18", "22:15:00"), run_fn=_run, seal_fn=_seal)
    assert report.status == "ok"
    data_cmd = next(c for c in seen if len(c) > 2 and c[2].endswith("/data"))
    assert "--max-duration" in data_cmd and "3000s" in data_cmd
    assert "--cutoff-mode" in data_cmd and "soft" in data_cmd


def test_seal_deadline_is_run_start_plus_seal_budget(tmp_path: Path, monkeypatch) -> None:
    import subprocess
    from datetime import timedelta

    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import BACKUP_SEAL_BUDGET, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    seen: dict[str, object] = {}

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        seen["deadline"] = deadline
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    start = _now_kst("2026-09-18")
    run_offsite_backup(tmp_path, tmp_path / "cdeadline", now=start, run_fn=_run, seal_fn=_seal)
    assert seen["deadline"] == start + BACKUP_SEAL_BUDGET


def test_duration_exceeded_loose_copy_is_deferred_not_failed(tmp_path: Path, monkeypatch) -> None:
    import json
    import subprocess

    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import REPORT_RELPATH, RCLONE_DURATION_EXCEEDED_EXIT, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, RCLONE_DURATION_EXCEEDED_EXIT, stdout="", stderr="max duration exceeded")

    report = run_offsite_backup(tmp_path, tmp_path / "cdefer", now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)
    assert report.status == "ok"
    assert report.steps["data"]["status"] == "deferred"
    persisted = json.loads((tmp_path / "cdefer" / REPORT_RELPATH).read_text(encoding="utf-8"))
    assert persisted["status"] == "ok"
    assert persisted["steps"]["data"]["status"] == "deferred"


def test_deferred_report_yields_deferred_staleness_issue(tmp_path: Path) -> None:
    import json

    from src.tools.offsite_backup import backup_staleness_issues

    audit_at = _now_kst("2026-09-21", "20:15:00")

    deferred_seal = tmp_path / "deferred_seal.json"
    deferred_seal.write_text(json.dumps({
        "started_at": "2026-09-18T13:30:00+00:00", "finished_at": "2026-09-18T13:30:00+00:00",
        "status": "ok",
        "steps": {"capture_seal": {"status": "ok", "deferred_dates": 3}},
    }), encoding="utf-8")
    assert backup_staleness_issues(deferred_seal, audit_at) == ["offsite_backup:deferred"]

    deferred_loose = tmp_path / "deferred_loose.json"
    deferred_loose.write_text(json.dumps({
        "started_at": "2026-09-18T13:30:00+00:00", "finished_at": "2026-09-18T13:30:00+00:00",
        "status": "ok",
        "steps": {
            "capture_seal": {"status": "ok", "deferred_dates": 0},
            "data": {"status": "deferred", "returncode": 10},
        },
    }), encoding="utf-8")
    assert backup_staleness_issues(deferred_loose, audit_at) == ["offsite_backup:deferred"]


def test_failure_dominates_deferral_in_report_and_staleness(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import backup_staleness_issues, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0,
                          missing_sealed_members=0, deferred_dates=2)

    def _run(cmd, **kwargs):
        if cmd[2].endswith("/data"):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="drive down")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    import pytest

    with pytest.raises(RuntimeError, match="offsite backup failed"):
        run_offsite_backup(tmp_path, tmp_path / "cfaildef", now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)
    from src.tools.offsite_backup import REPORT_RELPATH

    persisted = tmp_path / "cfaildef" / REPORT_RELPATH
    assert backup_staleness_issues(persisted, _now_kst("2026-09-21", "20:15:00")) == ["offsite_backup:failed"]


def _write_old_ok_report(path: Path) -> None:
    _write_report(path, status="ok", started_at="2026-10-01T13:15:00+00:00")


def _write_marker(path: Path, *, started_at: str, deadline_at: str) -> None:
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps({"started_at": started_at, "deadline_at": deadline_at, "pid": 1234, "host": "test-host"}),
        encoding="utf-8",
    )


def test_running_backup_is_not_stale(tmp_path: Path) -> None:
    from src.tools.offsite_backup import backup_staleness_issues

    report = tmp_path / "offsite" / "last_run.json"
    _write_old_ok_report(report)
    marker = tmp_path / "offsite" / "in_progress.json"
    _write_marker(marker, started_at="2026-10-02T13:20:00+00:00", deadline_at="2026-10-02T16:20:00+00:00")
    audit_at = datetime.fromisoformat("2026-10-02T23:39:00+09:00")
    assert backup_staleness_issues(report, audit_at) == ["offsite_backup:running"]


def test_interrupted_backup_is_warning(tmp_path: Path) -> None:
    from src.tools.offsite_backup import backup_staleness_issues

    report = tmp_path / "offsite" / "last_run.json"
    _write_old_ok_report(report)
    marker = tmp_path / "offsite" / "in_progress.json"
    _write_marker(marker, started_at="2026-10-02T13:20:00+00:00", deadline_at="2026-10-02T16:20:00+00:00")
    audit_at = datetime.fromisoformat("2026-10-03T02:00:00+09:00")
    assert backup_staleness_issues(report, audit_at) == ["offsite_backup:interrupted"]


def test_marker_deadline_boundary_with_explicit_path(tmp_path: Path) -> None:
    from datetime import timedelta

    from src.tools.offsite_backup import backup_staleness_issues

    report = tmp_path / "last_run.json"
    _write_old_ok_report(report)
    marker = tmp_path / "custom_progress.json"
    deadline = datetime.fromisoformat("2026-10-02T16:20:00+00:00")
    _write_marker(marker, started_at="2026-10-02T13:15:00+00:00", deadline_at=deadline.isoformat())
    assert backup_staleness_issues(report, deadline - timedelta(microseconds=1), progress_path=marker) == [
        "offsite_backup:running"
    ]
    assert backup_staleness_issues(report, deadline, progress_path=marker) == ["offsite_backup:interrupted"]


def test_old_marker_cannot_mask_staleness(tmp_path: Path) -> None:
    from src.tools.offsite_backup import backup_staleness_issues

    report = tmp_path / "offsite" / "last_run.json"
    _write_old_ok_report(report)
    marker = tmp_path / "offsite" / "in_progress.json"
    _write_marker(marker, started_at="2026-09-30T13:15:00+00:00", deadline_at="2026-09-30T16:15:00+00:00")
    audit_at = datetime.fromisoformat("2026-10-02T23:39:00+09:00")
    assert backup_staleness_issues(report, audit_at) == ["offsite_backup:stale"]


def test_corrupt_or_naive_marker_is_ignored(tmp_path: Path) -> None:
    import json as _json

    from src.tools.offsite_backup import backup_staleness_issues, read_backup_progress

    report = tmp_path / "offsite" / "last_run.json"
    _write_old_ok_report(report)
    audit_at = datetime.fromisoformat("2026-10-02T23:39:00+09:00")
    corrupt = tmp_path / "offsite" / "in_progress.json"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{not json", encoding="utf-8")
    assert read_backup_progress(corrupt) is None
    assert backup_staleness_issues(report, audit_at) == ["offsite_backup:stale"]
    _write_marker(corrupt, started_at="2026-10-02T22:20:00", deadline_at="2026-10-03T01:20:00")
    assert read_backup_progress(corrupt) is None
    assert backup_staleness_issues(report, audit_at) == ["offsite_backup:stale"]
    for bad in (
        "[1, 2]",
        "{}",
        _json.dumps({"started_at": 1, "deadline_at": "2026-10-02T16:20:00+00:00", "pid": 1, "host": "h"}),
        _json.dumps({"started_at": "2026-10-02T13:20:00+00:00", "deadline_at": "2026-10-02T16:20:00+00:00", "pid": True, "host": "h"}),
        _json.dumps({"started_at": "not-a-date", "deadline_at": "2026-10-02T16:20:00+00:00", "pid": 1, "host": "h"}),
        _json.dumps({"started_at": "2026-10-02T13:20:00+00:00", "deadline_at": "2026-10-02T16:20:00", "pid": 1, "host": "h"}),
    ):
        corrupt.write_text(bad, encoding="utf-8")
        assert read_backup_progress(corrupt) is None
    assert backup_staleness_issues(report, audit_at) == ["offsite_backup:stale"]


def test_fresh_report_wins_over_marker(tmp_path: Path) -> None:
    from src.tools.offsite_backup import backup_staleness_issues

    report = tmp_path / "offsite" / "last_run.json"
    _write_report(report, status="ok", started_at="2026-10-02T13:30:00+00:00")
    marker = tmp_path / "offsite" / "in_progress.json"
    _write_marker(marker, started_at="2026-10-02T13:20:00+00:00", deadline_at="2026-10-02T16:20:00+00:00")
    audit_at = datetime.fromisoformat("2026-10-02T23:39:00+09:00")
    assert backup_staleness_issues(report, audit_at) == []


def test_failed_report_not_hidden_by_marker(tmp_path: Path) -> None:
    from src.tools.offsite_backup import backup_staleness_issues

    report = tmp_path / "offsite" / "last_run.json"
    _write_report(report, status="failed", started_at="2026-10-02T13:30:00+00:00")
    marker = tmp_path / "offsite" / "in_progress.json"
    _write_marker(marker, started_at="2026-10-02T13:20:00+00:00", deadline_at="2026-10-02T16:20:00+00:00")
    audit_at = datetime.fromisoformat("2026-10-02T23:39:00+09:00")
    assert backup_staleness_issues(report, audit_at) == ["offsite_backup:failed"]


def test_missing_report_with_marker_is_running(tmp_path: Path) -> None:
    from src.tools.offsite_backup import backup_staleness_issues

    report = tmp_path / "offsite" / "last_run.json"
    marker = tmp_path / "offsite" / "in_progress.json"
    _write_marker(marker, started_at="2026-10-02T13:20:00+00:00", deadline_at="2026-10-02T16:20:00+00:00")
    audit_at = datetime.fromisoformat("2026-10-02T23:39:00+09:00")
    assert backup_staleness_issues(report, audit_at) == ["offsite_backup:running"]


def test_marker_lifecycle_success_and_failure(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    import pytest

    from src.tools import offsite_backup as _ob
    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import BACKUP_PROGRESS_RELPATH, REPORT_RELPATH, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")
    seen_during: dict[str, bool] = {}
    cleared_with_report: dict[str, bool] = {}
    real_clear = _ob.clear_backup_progress

    def _checked_clear(capture_root: Path) -> None:
        cleared_with_report["report_exists"] = (Path(capture_root) / REPORT_RELPATH).exists()
        real_clear(capture_root)

    monkeypatch.setattr(_ob, "clear_backup_progress", _checked_clear)

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        seen_during["marker_exists"] = (Path(capture_root) / BACKUP_PROGRESS_RELPATH).exists()
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    capture = tmp_path / "cap_ok"
    run_offsite_backup(tmp_path, capture, now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)
    assert seen_during["marker_exists"] is True
    assert cleared_with_report["report_exists"] is True
    assert not (capture / BACKUP_PROGRESS_RELPATH).exists()
    assert (capture / REPORT_RELPATH).exists()

    def _failing(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="down")

    capture2 = tmp_path / "cap_fail"
    with pytest.raises(RuntimeError, match="offsite backup failed"):
        run_offsite_backup(tmp_path, capture2, now=_now_kst("2026-09-18"), run_fn=_failing, seal_fn=_seal)
    assert not (capture2 / BACKUP_PROGRESS_RELPATH).exists()
    assert (capture2 / REPORT_RELPATH).exists()


def test_marker_write_failure_does_not_stop_backup(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging
    import subprocess

    from src.tools import offsite_backup as _ob
    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")

    def _boom(capture_root: Path, progress) -> Path:
        raise OSError("read-only")

    monkeypatch.setattr(_ob, "write_backup_progress", _boom)

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    with caplog.at_level(logging.WARNING):
        report = run_offsite_backup(tmp_path, tmp_path / "cap_unwritable", now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)
    assert report.status == "ok"
    assert any("progress_marker=UNWRITABLE" in rec.message for rec in caplog.records)


def test_marker_clear_failure_does_not_mask_report(tmp_path: Path, monkeypatch, caplog) -> None:
    import logging
    import subprocess

    from src.tools import offsite_backup as _ob
    from src.tools.capture_offsite import SealReport
    from src.tools.offsite_backup import REPORT_RELPATH, run_offsite_backup

    monkeypatch.setattr("src.tools.offsite_backup._resolve_rclone_bin", lambda: "rclone")

    def _unclearable(capture_root: Path) -> None:
        raise OSError("lock busy")

    monkeypatch.setattr(_ob, "clear_backup_progress", _unclearable)

    def _seal(capture_root: Path, *, today, full_scan, deadline=None) -> SealReport:
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    capture = tmp_path / "cap_unclearable"
    with caplog.at_level(logging.WARNING):
        report = run_offsite_backup(tmp_path, capture, now=_now_kst("2026-09-18"), run_fn=_run, seal_fn=_seal)
    assert report.status == "ok"
    assert (capture / REPORT_RELPATH).exists()
    assert any("progress_marker=UNCLEARABLE" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("target", [
    "src.tools.offsite_backup._resolve_rclone_bin",
    "src.tools.core_snapshot.collect_core_stats",
    "src.tools.core_snapshot.validate_core_panels",
])
def test_initial_failure_publishes_report_before_clearing_marker(tmp_path: Path, monkeypatch, target: str) -> None:
    from src.tools import offsite_backup as ob

    capture = tmp_path / "capture"
    report_path = capture / ob.REPORT_RELPATH
    _write_old_ok_report(report_path)
    old = json.loads(report_path.read_text())
    baseline = [{"relpath": "data/history/price_history.parquet", "sha256": "a", "bytes": 1, "rows": 1}]
    old["core_panels"] = baseline
    report_path.write_text(json.dumps(old))
    monkeypatch.setattr(ob, "_resolve_rclone_bin", lambda: "rclone")
    real_clear = ob.clear_backup_progress
    start = _now_kst("2026-10-02", "22:20:00")

    def fail(*args, **kwargs):
        assert (capture / ob.BACKUP_PROGRESS_RELPATH).exists()
        raise ValueError("initialization failed")

    def checked_clear(root: Path) -> None:
        persisted = json.loads(report_path.read_text())
        assert persisted["started_at"] == start.astimezone(UTC).isoformat()
        assert persisted["status"] == "failed"
        assert persisted["core_panels"] == baseline
        real_clear(root)

    monkeypatch.setattr(target, fail)
    monkeypatch.setattr(ob, "clear_backup_progress", checked_clear)
    with pytest.raises(RuntimeError, match="offsite backup failed: backup_run"):
        ob.run_offsite_backup(tmp_path, capture, now=start)
    assert not (capture / ob.BACKUP_PROGRESS_RELPATH).exists()
    assert "ValueError: initialization failed" in report_path.read_text()


@pytest.mark.parametrize("failed_run", [False, True])
def test_report_publication_failure_preserves_marker(tmp_path: Path, monkeypatch, failed_run: bool) -> None:
    from datetime import timedelta

    from src.tools import offsite_backup as ob
    from src.tools.capture_offsite import SealReport

    capture = tmp_path / "capture"
    report_path = capture / ob.REPORT_RELPATH
    _write_old_ok_report(report_path)
    previous = report_path.read_bytes()
    monkeypatch.setattr(ob, "_resolve_rclone_bin", lambda: "rclone")

    def seal(*args, **kwargs):
        return SealReport(dates_scanned=0, segments_committed=0, members_committed=0, archive_bytes=0, missing_sealed_members=0)

    def run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, int(failed_run), stdout="", stderr="copy failed")

    def fail_write(root, report):
        assert report.status == ("failed" if failed_run else "ok")
        raise OSError("disk full")

    monkeypatch.setattr(ob, "_write_report", fail_write)
    start = _now_kst("2026-10-02", "22:20:00")
    with pytest.raises(OSError, match="disk full"):
        ob.run_offsite_backup(tmp_path, capture, now=start, seal_fn=seal, run_fn=run)
    progress = ob.read_backup_progress(capture / ob.BACKUP_PROGRESS_RELPATH)
    assert progress is not None
    assert progress.started_at == start.astimezone(UTC).isoformat()
    assert report_path.read_bytes() == previous
    deadline = start + ob.BACKUP_TOTAL_BUDGET + ob.BACKUP_PROGRESS_GRACE
    assert ob.backup_staleness_issues(report_path, deadline + timedelta(seconds=1)) == ["offsite_backup:interrupted"]


def test_process_interruption_preserves_marker(tmp_path: Path, monkeypatch) -> None:
    from src.tools import offsite_backup as ob

    capture = tmp_path / "capture"
    monkeypatch.setattr(ob, "_resolve_rclone_bin", lambda: "rclone")

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("src.tools.core_snapshot.collect_core_stats", interrupt)
    with pytest.raises(KeyboardInterrupt):
        ob.run_offsite_backup(tmp_path, capture, now=_now_kst("2026-10-02", "22:20:00"))
    assert ob.read_backup_progress(capture / ob.BACKUP_PROGRESS_RELPATH) is not None
    assert not (capture / ob.REPORT_RELPATH).exists()


@pytest.mark.parametrize("previous_report", [False, True])
@pytest.mark.parametrize("completion_phase", ["before_marker_read", "after_marker_read", "after_report_read"])
def test_audit_during_completion_never_reports_stale_or_missing(
    tmp_path: Path, monkeypatch, previous_report: bool, completion_phase: str,
) -> None:
    from src.tools import offsite_backup as ob

    capture = tmp_path / "capture"
    report_path = capture / ob.REPORT_RELPATH
    marker_path = capture / ob.BACKUP_PROGRESS_RELPATH
    if previous_report:
        _write_old_ok_report(report_path)
    progress = ob.BackupProgress("2026-10-02T13:20:00+00:00", "2026-10-02T16:20:00+00:00", 1234, "host")
    ob.write_backup_progress(capture, progress)
    real_read = Path.read_text
    real_exists = Path.exists

    def complete() -> None:
        ob._write_report(capture, ob.BackupRunReport(progress.started_at, "2026-10-02T14:39:00+00:00", "ok", {}))
        ob.clear_backup_progress(capture)

    def concurrent_read(path: Path, *args, **kwargs):
        if path == marker_path and completion_phase == "before_marker_read":
            complete()
        contents = real_read(path, *args, **kwargs)
        if (path == marker_path and completion_phase == "after_marker_read") or (
            path == report_path and completion_phase == "after_report_read"
        ):
            complete()
        return contents

    def concurrent_exists(path: Path) -> bool:
        exists = real_exists(path)
        if path == report_path and not exists and completion_phase == "after_report_read":
            complete()
        return exists

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "read_text", concurrent_read)
        scoped.setattr(Path, "exists", concurrent_exists)
        result = ob.backup_staleness_issues(report_path, _now_kst("2026-10-02", "23:39:00"))
    assert result == (["offsite_backup:running"] if completion_phase == "after_report_read" else [])
    assert ob.backup_staleness_issues(report_path, _now_kst("2026-10-02", "23:39:00")) == []
