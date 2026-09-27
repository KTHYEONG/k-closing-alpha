from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

_SEOUL = ZoneInfo("Asia/Seoul")


def _probe_time(day: str, clock: str = "07:40:00") -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}+09:00")


def _write_heartbeat(path: Path, snapshot_date: str, day_kind: str = "trading") -> Path:
    import json

    payload = {
        "snapshot_date": snapshot_date,
        "day_kind": day_kind,
        "subject": f"[kca] {snapshot_date}",
        "undelivered_alerts": 0,
        "finished_at": f"{snapshot_date}T20:15:00+09:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _evaluate(
    day: str,
    heartbeat: Path | None,
    outbox: Path,
    *,
    failed: tuple[str, ...] = (),
    timers: tuple[str, ...] = (),
):
    from src.tools.watchdog_probe import evaluate_watchdog

    target = heartbeat if heartbeat is not None else outbox / "daily_audit.json"
    return evaluate_watchdog(
        _probe_time(day),
        heartbeat_path=target,
        outbox_dir=outbox,
        failed_units=failed,
        inactive_timers=timers,
    )


def test_expected_audit_date_skips_weekend() -> None:
    from src.tools.watchdog_probe import expected_audit_date

    assert expected_audit_date(_probe_time("2026-09-28")) == "2026-09-25"


def test_expected_audit_date_regular_weekday() -> None:
    from src.tools.watchdog_probe import expected_audit_date

    assert expected_audit_date(_probe_time("2026-09-30")) == "2026-09-29"


def test_healthy_when_fresh_heartbeat_and_empty_outbox(tmp_path) -> None:
    heartbeat = _write_heartbeat(tmp_path / "daily_audit.json", "2026-09-29")
    outbox = tmp_path / "outbox"
    outbox.mkdir()

    verdict = _evaluate("2026-09-30", heartbeat, outbox)

    assert verdict.problems == ()
    assert verdict.expected_snapshot_date == "2026-09-29"


def test_holiday_heartbeat_counts_as_alive(tmp_path) -> None:
    heartbeat = _write_heartbeat(tmp_path / "daily_audit.json", "2026-09-25", day_kind="holiday")
    outbox = tmp_path / "outbox"
    outbox.mkdir()

    verdict = _evaluate("2026-09-28", heartbeat, outbox)

    assert verdict.problems == ()


def test_stale_heartbeat_flagged(tmp_path) -> None:
    heartbeat = _write_heartbeat(tmp_path / "daily_audit.json", "2026-09-28")
    outbox = tmp_path / "outbox"
    outbox.mkdir()

    verdict = _evaluate("2026-09-30", heartbeat, outbox)

    assert "heartbeat_stale:2026-09-28" in verdict.problems


def test_missing_and_corrupt_heartbeat_flagged(tmp_path) -> None:
    outbox = tmp_path / "outbox"
    outbox.mkdir()

    assert "heartbeat_missing" in _evaluate("2026-09-30", None, outbox).problems

    corrupt = tmp_path / "daily_audit.json"
    corrupt.write_text("not json {{{", encoding="utf-8")

    assert "heartbeat_unreadable" in _evaluate("2026-09-30", corrupt, outbox).problems

    corrupt.write_text('{"snapshot_date": 20260929}', encoding="utf-8")

    assert "heartbeat_unreadable" in _evaluate("2026-09-30", corrupt, outbox).problems


def test_undelivered_outbox_flagged_as_dead_channel(tmp_path) -> None:
    heartbeat = _write_heartbeat(tmp_path / "daily_audit.json", "2026-09-29")
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    for index in range(2):
        (outbox / f"20260930T07400{index}_deadbeef.json").write_text("{}", encoding="utf-8")

    verdict = _evaluate("2026-09-30", heartbeat, outbox)

    assert "alert_outbox:2" in verdict.problems


def test_failed_units_and_inactive_timers_flagged(tmp_path) -> None:
    heartbeat = _write_heartbeat(tmp_path / "daily_audit.json", "2026-09-29")
    outbox = tmp_path / "outbox"
    outbox.mkdir()

    verdict = _evaluate(
        "2026-09-30",
        heartbeat,
        outbox,
        failed=("kca-predict.service", "kca-collect.service"),
        timers=("kca-predict.timer", "kca-collect.timer"),
    )

    assert "failed_units:kca-collect.service,kca-predict.service" in verdict.problems
    assert "timers_inactive:kca-collect.timer,kca-predict.timer" in verdict.problems


def test_inactive_timers_reports_none_declared_for_empty_dir(tmp_path) -> None:
    from src.tools.watchdog_probe import _inactive_timers

    assert _inactive_timers(tmp_path) == ["none_declared"]


def test_inactive_timers_classifies_by_is_active(monkeypatch, tmp_path) -> None:
    import subprocess
    from types import SimpleNamespace

    from src.tools import watchdog_probe

    (tmp_path / "kca-a.timer").write_text("", encoding="utf-8")
    (tmp_path / "kca-b.timer").write_text("", encoding="utf-8")

    def _run(cmd, **_kwargs):
        if cmd[-1] == "kca-a.timer":
            return SimpleNamespace(returncode=0)
        raise OSError("no systemctl")

    monkeypatch.setattr(subprocess, "run", _run)

    assert watchdog_probe._inactive_timers(tmp_path) == ["kca-b.timer"]


def test_main_exit_code_reflects_verdict(monkeypatch, caplog) -> None:
    import logging

    from src.tools import watchdog_probe
    from src.tools.watchdog_probe import WatchdogVerdict

    monkeypatch.setattr(watchdog_probe, "list_failed_kca_units", lambda: [])
    monkeypatch.setattr(watchdog_probe, "_inactive_timers", lambda: [])

    monkeypatch.setattr(
        watchdog_probe,
        "evaluate_watchdog",
        lambda *args, **kwargs: WatchdogVerdict(problems=(), expected_snapshot_date="2026-09-29"),
    )
    with caplog.at_level(logging.INFO):
        assert watchdog_probe.main([]) == 0

    monkeypatch.setattr(
        watchdog_probe,
        "evaluate_watchdog",
        lambda *args, **kwargs: WatchdogVerdict(
            problems=("heartbeat_stale:2026-09-28",), expected_snapshot_date="2026-09-29"
        ),
    )
    with caplog.at_level(logging.INFO):
        assert watchdog_probe.main([]) == 1

    watchdog_lines = [rec.message for rec in caplog.records if "stage=watchdog" in rec.message]
    assert len(watchdog_lines) == 2
    assert "status=OK" in watchdog_lines[0] and "status=PROBLEM" in watchdog_lines[1]
