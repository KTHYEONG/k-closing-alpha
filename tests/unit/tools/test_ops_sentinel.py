from __future__ import annotations

import subprocess
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_KST = ZoneInfo("Asia/Seoul")
_UTC = ZoneInfo("UTC")


def _kst(day: str, clock: str) -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}+09:00")


def _schedule(
    unit: str = "kca-probe.service",
    clock: str = "07:10:00",
    weekdays: frozenset[int] = frozenset({0, 1, 2, 3, 4}),
    max_runtime: timedelta = timedelta(minutes=15),
) -> object:
    from src.tools.ops_sentinel import JobSchedule, ScheduleSlot

    h, m, s = (int(v) for v in clock.split(":"))
    return JobSchedule(
        unit=unit,
        timer=unit.replace(".service", ".timer"),
        slots=(ScheduleSlot(weekdays=weekdays, time_of_day=time(h, m, s)),),
        max_runtime=max_runtime,
    )


def _state(active: str = "inactive", result: str = "success", start: datetime | None = None):
    from src.tools.ops_sentinel import UnitRunState

    return UnitRunState(active_state=active, result=result, last_start=start, last_exit=start)


def test_every_repository_timer_parses() -> None:
    from src.tools.ops_sentinel import load_job_schedules

    schedules = load_job_schedules()
    assert schedules
    root = Path(__file__).resolve().parents[3] / "deploy" / "systemd"
    for schedule in schedules:
        assert schedule.slots
        assert (root / schedule.timer).exists()
        assert (root / schedule.unit).exists()


def test_weekday_range_and_saturday_never_due() -> None:
    from src.tools.ops_sentinel import latest_due_slot, parse_timer_schedule

    slots = parse_timer_schedule("OnCalendar=Mon..Fri 21:20:00 Asia/Seoul\n")
    assert slots[0].weekdays == frozenset({0, 1, 2, 3, 4})
    assert slots[0].time_of_day == time(21, 20)
    schedule = _schedule(clock="21:20:00")  # type: ignore[arg-type]
    saturday = _kst("2026-10-10", "12:00:00")
    due = latest_due_slot(schedule, saturday)  # type: ignore[arg-type]
    assert due is not None and due.weekday() == 4 and due.hour == 21


def test_multiple_oncalendar_lines() -> None:
    from src.tools.ops_sentinel import parse_timer_schedule

    text = (
        "OnCalendar=Mon..Fri 08:30:00 Asia/Seoul\n"
        "OnCalendar=Mon..Fri 11:30:00 Asia/Seoul\n"
        "OnCalendar=Mon..Fri 21:30:00 Asia/Seoul\n"
    )
    assert len(parse_timer_schedule(text)) == 3


def test_unsupported_grammar_rejected() -> None:
    import pytest

    from src.tools.ops_sentinel import parse_timer_schedule

    with pytest.raises(ValueError, match="OnCalendar"):
        parse_timer_schedule("OnCalendar=hourly\n")


def test_timeout_parsing_and_defaults(tmp_path: Path) -> None:
    from src.tools.ops_sentinel import SENTINEL_DEFAULT_MAX_RUNTIME, load_job_schedules

    def _write(stem: str, timeout_line: str | None) -> None:
        (tmp_path / f"{stem}.timer").write_text(
            "OnCalendar=Mon..Fri 07:10:00 Asia/Seoul\nUnit=kca-x.service\n", encoding="utf-8"
        )
        lines = ["[Service]", "Type=oneshot"]
        if timeout_line is not None:
            lines.append(timeout_line)
        (tmp_path / "kca-x.service").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write("kca-a", "TimeoutStartSec=15min")
    schedules = load_job_schedules(tmp_path)
    assert schedules[0].max_runtime == timedelta(minutes=15)

    for name in ("kca-a.timer", "kca-x.service"):
        (tmp_path / name).unlink()
    _write("kca-a", None)
    assert load_job_schedules(tmp_path)[0].max_runtime == SENTINEL_DEFAULT_MAX_RUNTIME

    for name in ("kca-a.timer", "kca-x.service"):
        (tmp_path / name).unlink()
    _write("kca-a", "TimeoutStartSec=infinity")
    assert load_job_schedules(tmp_path)[0].max_runtime == SENTINEL_DEFAULT_MAX_RUNTIME


def test_late_when_never_started() -> None:
    from src.tools.ops_sentinel import JobVerdict, evaluate_job

    verdict = evaluate_job(_schedule(), None, _kst("2026-10-05", "07:20:00"))  # type: ignore[arg-type]
    assert verdict == JobVerdict.LATE


def test_within_grace_is_pending() -> None:
    from src.tools.ops_sentinel import JobVerdict, evaluate_job

    old = _kst("2026-10-02", "07:10:00").astimezone(_UTC)
    verdict = evaluate_job(_schedule(), _state(start=old), _kst("2026-10-05", "07:14:00"))  # type: ignore[arg-type]
    assert verdict == JobVerdict.PENDING


def test_started_after_slot_is_ok_even_when_failed() -> None:
    from src.tools.ops_sentinel import JobVerdict, evaluate_job

    slot = _kst("2026-10-05", "07:10:00")
    start = slot + timedelta(seconds=2)
    state = _state(active="inactive", result="failed", start=start)
    assert evaluate_job(_schedule(), state, _kst("2026-10-05", "07:20:00")) == JobVerdict.OK  # type: ignore[arg-type]


def test_running_vs_overrun() -> None:
    from src.tools.ops_sentinel import JobVerdict, evaluate_job

    schedule = _schedule()  # type: ignore[assignment]
    slot = _kst("2026-10-05", "07:10:00")
    state = _state(active="active", start=slot)
    assert evaluate_job(schedule, state, slot + timedelta(minutes=10)) == JobVerdict.RUNNING  # type: ignore[arg-type]
    assert evaluate_job(schedule, state, slot + timedelta(minutes=20)) == JobVerdict.OVERRUN  # type: ignore[arg-type]


def test_catchup_after_reboot_is_ok() -> None:
    from src.tools.ops_sentinel import JobVerdict, evaluate_job

    slot = _kst("2026-10-05", "07:10:00")
    state = _state(start=slot + timedelta(minutes=30))
    assert evaluate_job(_schedule(), state, slot + timedelta(hours=2)) == JobVerdict.OK  # type: ignore[arg-type]


def test_weekend_uses_friday_slot() -> None:
    from src.tools.ops_sentinel import JobVerdict, evaluate_job, latest_due_slot

    schedule = _schedule()  # type: ignore[assignment]
    saturday = _kst("2026-10-10", "12:00:00")
    due = latest_due_slot(schedule, saturday)  # type: ignore[arg-type]
    assert due is not None and due.date().isoformat() == "2026-10-09" and due.hour == 7
    friday_slot = _kst("2026-10-09", "07:10:00")
    assert evaluate_job(schedule, _state(start=friday_slot), saturday) == JobVerdict.OK  # type: ignore[arg-type]
    stale = _kst("2026-10-08", "07:10:00")
    assert evaluate_job(schedule, _state(start=stale), saturday) == JobVerdict.LATE  # type: ignore[arg-type]


def test_systemd_unreachable_is_one_issue() -> None:
    from src.tools.ops_sentinel import measure_ops

    def _boom(_units: object) -> dict:
        raise OSError("no systemd")

    issues = measure_ops(
        _kst("2026-10-05", "07:20:00"),
        schedules=(_schedule(),),  # type: ignore[arg-type]
        states_fn=_boom,  # type: ignore[arg-type]
        inactive_timers_fn=lambda: [],
        disk_fn=lambda: (90, 100),
    )
    keys = [issue.key for issue in issues]
    assert keys == ["ops_measure_unavailable:units"]


def test_disk_guard() -> None:
    from src.tools.ops_sentinel import measure_ops

    low = measure_ops(
        _kst("2026-10-05", "07:20:00"),
        schedules=(),
        states_fn=lambda _u: {},
        inactive_timers_fn=lambda: [],
        disk_fn=lambda: (8, 100),
    )
    assert [i.key for i in low] == ["host_disk_low:8"]
    ok = measure_ops(
        _kst("2026-10-05", "07:20:00"),
        schedules=(),
        states_fn=lambda _u: {},
        inactive_timers_fn=lambda: [],
        disk_fn=lambda: (12, 100),
    )
    assert ok == ()


def test_timezone_straddling_midnight() -> None:
    from src.tools.ops_sentinel import JobVerdict, evaluate_job

    schedule = _schedule(clock="00:10:00")  # type: ignore[assignment]
    now = _kst("2026-10-05", "00:20:00")
    fresh_utc = datetime(2026, 10, 4, 15, 11, 0, tzinfo=_UTC)
    assert evaluate_job(schedule, _state(start=fresh_utc), now) == JobVerdict.OK  # type: ignore[arg-type]
    stale_utc = datetime(2026, 10, 4, 15, 0, 0, tzinfo=_UTC)
    assert evaluate_job(schedule, _state(start=stale_utc), now) == JobVerdict.LATE  # type: ignore[arg-type]


def test_parse_rejections_cover_branches() -> None:
    import pytest

    from src.tools.ops_sentinel import parse_timer_schedule

    with pytest.raises(ValueError, match="range"):
        parse_timer_schedule("OnCalendar=Foo..Bar 07:10:00 Asia/Seoul\n")
    with pytest.raises(ValueError, match="range"):
        parse_timer_schedule("OnCalendar=Fri..Mon 07:10:00 Asia/Seoul\n")
    with pytest.raises(ValueError, match="selector"):
        parse_timer_schedule("OnCalendar=Hourly 07:10:00 Asia/Seoul\n")
    with pytest.raises(ValueError, match="timezone"):
        parse_timer_schedule("OnCalendar=Mon..Fri 07:10:00 UTC\n")
    with pytest.raises(ValueError, match="time"):
        parse_timer_schedule("OnCalendar=Mon..Fri 25:00:00 Asia/Seoul\n")
    with pytest.raises(ValueError, match="OnCalendar"):
        parse_timer_schedule("[Timer]\nPersistent=true\n")


def test_timeout_rejections_and_bare_seconds(tmp_path: Path) -> None:
    import pytest

    from src.tools.ops_sentinel import _parse_timeout_value

    assert _parse_timeout_value("3600") == timedelta(seconds=3600)
    with pytest.raises(ValueError, match="TimeoutStartSec"):
        _parse_timeout_value("0")
    with pytest.raises(ValueError, match="TimeoutStartSec"):
        _parse_timeout_value("foo")
    with pytest.raises(ValueError, match="TimeoutStartSec"):
        _parse_timeout_value("0s")


def test_loader_template_skip_stem_fallback_and_missing_service(tmp_path: Path) -> None:
    import pytest

    from src.tools.ops_sentinel import load_job_schedules

    (tmp_path / "kca-alert@.timer").write_text("OnCalendar=Mon..Fri 07:10:00 Asia/Seoul\n", encoding="utf-8")
    (tmp_path / "kca-nounit.timer").write_text("OnCalendar=Mon..Fri 07:10:00 Asia/Seoul\n", encoding="utf-8")
    (tmp_path / "kca-nounit.service").write_text("[Service]\nType=oneshot\n", encoding="utf-8")
    schedules = load_job_schedules(tmp_path)
    assert [s.unit for s in schedules] == ["kca-nounit.service"]
    (tmp_path / "kca-orphan.timer").write_text(
        "OnCalendar=Mon..Fri 07:10:00 Asia/Seoul\nUnit=kca-missing.service\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        load_job_schedules(tmp_path)


def test_read_unit_states_parsing_branches() -> None:
    from src.tools.ops_sentinel import read_unit_states

    assert read_unit_states([]) == {}
    stdout = (
        "Id=kca-a.service\nActiveState=inactive\nResult=success\n"
        "ExecMainStartTimestamp=Mon 2026-10-05 22:10:00 UTC\nExecMainExitTimestamp=n/a\n"
        "\n"
        "Id=kca-b.service\nActiveState=active\nResult=success\n"
        "ExecMainStartTimestamp=\nExecMainExitTimestamp=\n"
        "\n"
        "GARBAGE-LINE\nId=kca-other.service\nActiveState=inactive\n"
    )

    def _run(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("env", {}).get("TZ") == "UTC"
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    states = read_unit_states(["kca-a.service", "kca-b.service"], run_fn=_run)  # type: ignore[arg-type]
    assert states["kca-a.service"].last_start is not None
    assert states["kca-a.service"].last_exit is None
    assert states["kca-b.service"].last_start is None
    assert "kca-other.service" not in states

    def _local_tz(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            cmd, 0, "Id=kca-a.service\nActiveState=inactive\nExecMainStartTimestamp=Mon 2026-10-05 22:10:00 KST\n", ""
        )

    with pytest.raises(ValueError, match=r"does not match|not in UTC"):
        read_unit_states(["kca-a.service"], run_fn=_local_tz)  # type: ignore[arg-type]


def test_naive_now_and_empty_schedule() -> None:
    from datetime import datetime as _dt

    from src.tools.ops_sentinel import JobSchedule, JobVerdict, evaluate_job, latest_due_slot

    schedule = _schedule()  # type: ignore[assignment]
    naive = _dt(2026, 10, 6, 7, 20, 0)
    due = latest_due_slot(schedule, naive)  # type: ignore[arg-type]
    assert due is not None and due.tzinfo is not None
    empty = JobSchedule(unit="kca-e.service", timer="kca-e.timer", slots=(), max_runtime=timedelta(minutes=5))
    assert evaluate_job(empty, None, _kst("2026-10-05", "07:20:00")) == JobVerdict.PENDING


def test_measure_ops_aggregates_and_defaults(monkeypatch: object, tmp_path: Path) -> None:
    from src.tools import ops_sentinel as sentinel
    from src.tools.ops_sentinel import JobVerdict, measure_ops

    late = _schedule(unit="kca-late.service")  # type: ignore[assignment]
    overrun = _schedule(unit="kca-over.service")  # type: ignore[assignment]
    now = _kst("2026-10-05", "07:20:00")
    slot = _kst("2026-10-05", "07:10:00")
    states = {
        "kca-over.service": _state(active="active", start=slot - timedelta(hours=3)),
    }
    issues = measure_ops(
        now,
        schedules=(late, overrun),  # type: ignore[arg-type]
        states_fn=lambda _u: states,
        inactive_timers_fn=lambda: ["kca-z.timer"],
        disk_fn=lambda: (8, 100),
    )
    assert [i.key for i in issues] == [
        "host_disk_low:8",
        "job_late:kca-late.service",
        "job_overrun:kca-over.service",
        "timer_inactive:kca-z.timer",
    ]

    monkeypatch.setattr(sentinel, "load_job_schedules", lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))  # type: ignore[attr-defined]
    assert measure_ops(now)[0].key == "ops_measure_unavailable:schedules"

    monkeypatch.setattr(sentinel, "load_job_schedules", lambda *a, **k: (late,))  # type: ignore[attr-defined]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        sentinel, "read_unit_states", lambda _u: (_ for _ in ()).throw(OSError("down"))
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        sentinel.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, f"Id={late.timer}\nActiveState=inactive\n", ""),
    )
    from types import SimpleNamespace as _NS

    monkeypatch.setattr(sentinel.shutil, "disk_usage", lambda _p: _NS(free=90, total=100))  # type: ignore[attr-defined]
    keys = [i.key for i in measure_ops(now)]
    assert "ops_measure_unavailable:units" in keys
    assert f"timer_inactive:{late.timer}" in keys

    monkeypatch.setattr(sentinel, "read_unit_states", lambda _u: {"kca-late.service": _state(active="inactive", start=slot)})  # type: ignore[attr-defined]
    monkeypatch.setattr(sentinel, "evaluate_job", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("clk")))  # type: ignore[attr-defined]
    keys = [i.key for i in measure_ops(now)]
    assert "ops_measure_unavailable:units" in keys
    monkeypatch.setattr(sentinel, "evaluate_job", lambda *a, **k: JobVerdict.OK)  # type: ignore[attr-defined]

    def _boom_timers() -> list[str]:
        raise OSError("timers down")

    assert measure_ops(now, schedules=(), states_fn=lambda _u: {}, inactive_timers_fn=_boom_timers)[0].key == (
        "ops_measure_unavailable:timers"
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        sentinel.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("down"))
    )
    assert "ops_measure_unavailable:timers" in [i.key for i in measure_ops(now)]

    def _boom_disk() -> tuple[int, int]:
        raise OSError("disk down")

    assert measure_ops(
        now, schedules=(), states_fn=lambda _u: {}, inactive_timers_fn=lambda: [], disk_fn=_boom_disk
    )[0].key == ("ops_measure_unavailable:disk")
    assert (
        measure_ops(now, schedules=(), states_fn=lambda _u: {}, inactive_timers_fn=lambda: [], disk_fn=lambda: (0, 0))
        == ()
    )


def test_default_disk_and_activating_without_start(monkeypatch: object, tmp_path: Path) -> None:
    from src import settings as _settings
    from src.tools import ops_sentinel as sentinel
    from src.tools.ops_sentinel import JobVerdict, evaluate_job

    monkeypatch.setattr(_settings, "DATA_DIR", str(tmp_path), raising=False)
    assert sentinel._default_disk_usage()[1] > 0
    schedule = _schedule()  # type: ignore[assignment]
    now = _kst("2026-10-05", "07:20:00")
    assert evaluate_job(schedule, _state(active="activating", start=None), now) == JobVerdict.RUNNING  # type: ignore[arg-type]


def test_read_only_commands() -> None:
    import src.tools.ops_sentinel as sentinel

    calls: list[list[str]] = []

    def _spy(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert isinstance(cmd, list)
        calls.append([str(c) for c in cmd])
        if "show" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "Id=kca-x.timer\nActiveState=active\n", "")
        raise AssertionError(f"unexpected systemctl verb: {cmd}")

    sentinel.read_unit_states(["kca-x.service"], run_fn=_spy)  # type: ignore[arg-type]
    sentinel._default_inactive_timers(["kca-x.timer"], run_fn=_spy)  # type: ignore[arg-type]
    assert len(calls) == 2
    assert {c[c.index("systemctl") + 2] for c in calls} == {"show"}


def test_inactive_timers_from_show_state() -> None:
    import src.tools.ops_sentinel as sentinel

    stdout = (
        "Id=kca-daily-audit.timer\nActiveState=active\n\n"
        "Id=kca-predict.timer\nActiveState=inactive\n"
    )

    def _run(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout, "")

    declared = ["kca-daily-audit.timer", "kca-predict.timer", "kca-ghost.timer"]
    assert sentinel._default_inactive_timers(declared, run_fn=_run) == ["kca-ghost.timer", "kca-predict.timer"]  # type: ignore[arg-type]
    assert sentinel._default_inactive_timers([]) == []

    def _no_bus(cmd: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, "", "Failed to connect to bus")

    with pytest.raises(RuntimeError):
        sentinel._default_inactive_timers(declared, run_fn=_no_bus)  # type: ignore[arg-type]


def test_empty_systemd_output_is_one_unavailable_issue(tmp_path) -> None:
    from src.tools.ops_sentinel import load_job_schedules, measure_ops

    schedules = load_job_schedules()
    issues = measure_ops(
        datetime(2026, 10, 6, 23, 0, tzinfo=ZoneInfo("Asia/Seoul")),
        schedules=schedules,
        states_fn=lambda units: {},
        inactive_timers_fn=list,
        disk_fn=lambda: (50, 100),
    )
    assert [issue.key for issue in issues] == ["ops_measure_unavailable:units"]
