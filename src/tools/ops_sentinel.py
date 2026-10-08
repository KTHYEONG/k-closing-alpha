"""Scheduled-job sentinel measurement for systemd timers."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from zoneinfo import ZoneInfo

from src.tools.daily_audit import SYSTEMCTL_TIMEOUT_SEC

KST = ZoneInfo("Asia/Seoul")
_UTC = UTC

SENTINEL_GRACE: timedelta = timedelta(minutes=5)
SENTINEL_DEFAULT_MAX_RUNTIME: timedelta = timedelta(hours=2)
HOST_DISK_FREE_MIN_FRACTION: float = 0.10

_WEEKDAY_BY_NAME: dict[str, int] = {
    "Mon": 0,
    "Tue": 1,
    "Wed": 2,
    "Thu": 3,
    "Fri": 4,
    "Sat": 5,
    "Sun": 6,
}

_ONCALENDAR_RE = re.compile(r"^(\S+)\s+(\d{2}):(\d{2})(?::(\d{2}))?\s+(\S+)\s*$")
_TIMEOUT_RE = re.compile(r"(\d+)(h|min|s)")
_SYSTEMD_TS_FORMAT = "%a %Y-%m-%d %H:%M:%S %Z"


@dataclass(frozen=True)
class ScheduleSlot:
    weekdays: frozenset[int]
    time_of_day: time


@dataclass(frozen=True)
class JobSchedule:
    unit: str
    timer: str
    slots: tuple[ScheduleSlot, ...]
    max_runtime: timedelta


@dataclass(frozen=True)
class UnitRunState:
    active_state: str
    result: str
    last_start: datetime | None
    last_exit: datetime | None


class JobVerdict(StrEnum):
    PENDING = "pending"
    OK = "ok"
    RUNNING = "running"
    LATE = "late"
    OVERRUN = "overrun"


@dataclass(frozen=True)
class MeasuredIssue:
    key: str
    text: str


def _parse_weekdays(raw: str) -> frozenset[int]:
    if raw == "*-*-*":
        return frozenset(range(7))
    if ".." in raw:
        start_raw, _, end_raw = raw.partition("..")
        if start_raw not in _WEEKDAY_BY_NAME or end_raw not in _WEEKDAY_BY_NAME:
            raise ValueError(f"Unsupported OnCalendar weekday range: {raw!r}")
        start = _WEEKDAY_BY_NAME[start_raw]
        end = _WEEKDAY_BY_NAME[end_raw]
        if end < start:
            raise ValueError(f"Unsupported OnCalendar weekday range: {raw!r}")
        return frozenset(range(start, end + 1))
    if raw not in _WEEKDAY_BY_NAME:
        raise ValueError(f"Unsupported OnCalendar date selector: {raw!r}")
    return frozenset({_WEEKDAY_BY_NAME[raw]})


def parse_timer_schedule(timer_text: str) -> tuple[ScheduleSlot, ...]:
    """Parse `OnCalendar=` lines of a timer unit into slots. Supports weekday selectors `Mon..Fri`, a single weekday name, `*-*-*`, and `HH:MM[:SS]` followed by `Asia/Seoul`; every other grammar raises ValueError so an unsupported timer fails a repository test instead of silently escaping monitoring. Times are KST wall-clock."""
    slots: list[ScheduleSlot] = []
    found = False
    for line in timer_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("OnCalendar="):
            continue
        found = True
        value = stripped.split("=", 1)[1].strip()
        match = _ONCALENDAR_RE.match(value)
        if match is None:
            raise ValueError(f"Unsupported OnCalendar value: {value!r}")
        date_part, hour_raw, minute_raw, second_raw, tz_part = match.groups()
        if tz_part != "Asia/Seoul":
            raise ValueError(f"Unsupported OnCalendar timezone: {value!r}")
        weekdays = _parse_weekdays(date_part)
        hour, minute, second = int(hour_raw), int(minute_raw), int(second_raw or "0")
        if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60):
            raise ValueError(f"Unsupported OnCalendar time: {value!r}")
        slots.append(ScheduleSlot(weekdays=weekdays, time_of_day=time(hour, minute, second)))
    if not found:
        raise ValueError("Timer has no OnCalendar= lines")
    return tuple(slots)


def _parse_timeout_value(raw: str) -> timedelta:
    value = raw.strip()
    if value == "" or value == "infinity":
        return SENTINEL_DEFAULT_MAX_RUNTIME
    if value.isdigit():
        total = int(value)
        if total <= 0:
            raise ValueError(f"Unsupported TimeoutStartSec: {raw!r}")
        return timedelta(seconds=total)
    if _TIMEOUT_RE.search(value) is None or re.fullmatch(r"(?:\d+(?:h|min|s))+", value) is None:
        raise ValueError(f"Unsupported TimeoutStartSec: {raw!r}")
    total = 0
    for amount, unit in _TIMEOUT_RE.findall(value):
        total += int(amount) * {"h": 3600, "min": 60, "s": 1}[unit]
    if total <= 0:
        raise ValueError(f"Unsupported TimeoutStartSec: {raw!r}")
    return timedelta(seconds=total)


def _default_unit_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "deploy" / "systemd"


def load_job_schedules(unit_dir: Path | None = None) -> tuple[JobSchedule, ...]:
    """Load timer schedules with expected-runtime, timeout, or default runtime limits.

    Raise ValueError for invalid schedules or runtimes and FileNotFoundError for absent services.
    """
    directory = Path(unit_dir) if unit_dir is not None else _default_unit_dir()
    schedules: list[JobSchedule] = []
    for timer_path in sorted(directory.glob("kca-*.timer")):
        if timer_path.name.startswith("kca-alert@"):
            continue
        timer_text = timer_path.read_text(encoding="utf-8")
        slots = parse_timer_schedule(timer_text)
        unit_name: str | None = None
        for line in timer_text.splitlines():
            if line.strip().startswith("Unit="):
                unit_name = line.strip().split("=", 1)[1].strip()
                break
        if not unit_name:
            unit_name = timer_path.stem + ".service"
        service_path = directory / unit_name
        if not service_path.exists():
            raise FileNotFoundError(f"Service file absent for timer {timer_path.name}: {unit_name}")
        service_text = service_path.read_text(encoding="utf-8")
        timeout_raw: str | None = None
        expected_raw: str | None = None
        in_service = False
        for line in service_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("["):
                in_service = stripped == "[Service]"
                continue
            if not in_service:
                continue
            if stripped.startswith("TimeoutStartSec=") and timeout_raw is None:
                timeout_raw = stripped.split("=", 1)[1]
            elif stripped.startswith("X-ExpectedRuntimeSec=") and expected_raw is None:
                expected_raw = stripped.split("=", 1)[1]
        finite_timeout: timedelta | None = None
        max_runtime = SENTINEL_DEFAULT_MAX_RUNTIME
        if timeout_raw is not None and timeout_raw.strip() not in ("", "infinity"):
            finite_timeout = _parse_timeout_value(timeout_raw)
            max_runtime = finite_timeout
        if expected_raw is not None:
            raw = expected_raw.strip()
            if re.fullmatch(r"[0-9]+", raw) is None:
                raise ValueError(f"Unsupported X-ExpectedRuntimeSec in {service_path.name}: {expected_raw!r}")
            try:
                expected = timedelta(seconds=int(raw))
            except (ValueError, OverflowError) as exc:
                raise ValueError(
                    f"Unsupported X-ExpectedRuntimeSec in {service_path.name}: {expected_raw!r}"
                ) from exc
            if expected < timedelta(seconds=60):
                raise ValueError(f"Unsupported X-ExpectedRuntimeSec in {service_path.name}: {expected_raw!r}")
            if finite_timeout is not None and expected > finite_timeout:
                raise ValueError(f"Unsupported X-ExpectedRuntimeSec in {service_path.name}: {expected_raw!r}")
            max_runtime = expected
        schedules.append(
            JobSchedule(unit=unit_name, timer=timer_path.name, slots=slots, max_runtime=max_runtime)
        )
    return tuple(sorted(schedules, key=lambda s: s.unit))


def _parse_systemd_timestamp(raw: str) -> datetime | None:
    value = raw.strip()
    if value in ("", "n/a"):
        return None
    parsed = datetime.strptime(value, _SYSTEMD_TS_FORMAT)
    if parsed.tzname() not in (None, "UTC") or not value.endswith(" UTC"):
        raise ValueError(f"systemd timestamp not in UTC: {value!r}")
    return parsed.replace(tzinfo=_UTC)


def _systemctl_show(
    names: Sequence[str],
    properties: str,
    run_fn: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, dict[str, str]]:
    env = dict(os.environ)
    env["TZ"] = "UTC"
    result = run_fn(
        ["systemctl", "--user", "show", f"--property={properties}", *names],
        capture_output=True,
        text=True,
        timeout=SYSTEMCTL_TIMEOUT_SEC,
        check=False,
        env=env,
    )
    blocks: dict[str, dict[str, str]] = {}
    for block in result.stdout.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            fields[key.strip()] = val.strip()
        unit_id = fields.get("Id", "")
        if unit_id and unit_id in names:
            blocks[unit_id] = fields
    return blocks


def read_unit_states(
    units: Sequence[str],
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, UnitRunState]:
    """One `systemctl --user show` call for all units with `TZ=UTC` environment, parsing `ActiveState`, `Result`, `ExecMainStartTimestamp`, `ExecMainExitTimestamp` (`%a %Y-%m-%d %H:%M:%S UTC`; empty means never). Returns aware UTC datetimes. A unit absent from systemd output maps to no entry. Raises OSError/TimeoutExpired from the runner unchanged (callers keep the previous measurement)."""
    names = list(units)
    if not names:
        return {}
    states: dict[str, UnitRunState] = {}
    for unit_id, fields in _systemctl_show(
        names, "Id,ActiveState,Result,ExecMainStartTimestamp,ExecMainExitTimestamp", run_fn
    ).items():
        states[unit_id] = UnitRunState(
            active_state=fields.get("ActiveState", ""),
            result=fields.get("Result", ""),
            last_start=_parse_systemd_timestamp(fields.get("ExecMainStartTimestamp", "")),
            last_exit=_parse_systemd_timestamp(fields.get("ExecMainExitTimestamp", "")),
        )
    return states


def _ensure_kst(now: datetime) -> datetime:
    if now.tzinfo is None:
        return now.replace(tzinfo=KST)
    return now.astimezone(KST)


def latest_due_slot(schedule: JobSchedule, now: datetime) -> datetime | None:
    """Latest slot time (KST, aware) at or before `now` over the schedule's slots, looking back up to 8 days; None when the schedule has no slot in that range."""
    current = _ensure_kst(now)
    best: datetime | None = None
    base_date = current.date()
    for offset in range(8):
        day = base_date - timedelta(days=offset)
        weekday = day.weekday()
        for slot in schedule.slots:
            if weekday not in slot.weekdays:
                continue
            candidate = datetime(
                day.year,
                day.month,
                day.day,
                slot.time_of_day.hour,
                slot.time_of_day.minute,
                slot.time_of_day.second,
                tzinfo=KST,
            )
            if candidate <= current and (best is None or candidate > best):
                best = candidate
    return best


def evaluate_job(
    schedule: JobSchedule,
    state: UnitRunState | None,
    now: datetime,
    *,
    grace: timedelta = SENTINEL_GRACE,
) -> JobVerdict:
    """Judge the latest due slot: PENDING while `now < slot + grace`; RUNNING while the unit is active/activating within `max_runtime` of its last start; OVERRUN when active longer than `max_runtime`; OK when `last_start >= slot - 60s` and the unit is not active; otherwise LATE. A unit with no state (never started) is LATE once past the grace. A non-success `Result` with a start at or after the slot is OK for this function (failure is reported by the existing failed-unit measurement, not duplicated)."""
    current = _ensure_kst(now) if now.tzinfo is None else now
    slot = latest_due_slot(schedule, current)
    if slot is None:
        return JobVerdict.PENDING
    if current < slot + grace:
        return JobVerdict.PENDING
    if state is None:
        return JobVerdict.LATE
    if state.active_state in ("active", "activating"):
        if state.last_start is not None and current - state.last_start > schedule.max_runtime:
            return JobVerdict.OVERRUN
        return JobVerdict.RUNNING
    if state.last_start is not None and state.last_start >= slot - timedelta(seconds=60):
        return JobVerdict.OK
    return JobVerdict.LATE


def _default_inactive_timers(
    declared: Sequence[str],
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> list[str]:
    names = list(declared)
    if not names:
        return []
    blocks = _systemctl_show(names, "Id,ActiveState", run_fn if run_fn is not None else subprocess.run)
    if not blocks:
        raise RuntimeError("systemctl returned no timer state")
    return sorted(name for name in names if blocks.get(name, {}).get("ActiveState") != "active")


def _default_disk_usage() -> tuple[int, int]:
    from src import settings

    path = Path(settings.DATA_DIR)
    target = path if path.exists() else Path("/")
    usage = shutil.disk_usage(target)
    return (usage.free, usage.total)


def measure_ops(
    now: datetime,
    *,
    schedules: Sequence[JobSchedule] | None = None,
    states_fn: Callable[[Sequence[str]], dict[str, UnitRunState]] | None = None,
    inactive_timers_fn: Callable[[], list[str]] | None = None,
    disk_fn: Callable[[], tuple[int, int]] | None = None,
) -> tuple[MeasuredIssue, ...]:
    """Aggregate issue keys: `job_late:<unit>`, `job_overrun:<unit>`, `timer_inactive:<timer>`, `host_disk_low:<free_percent>`. Each is self-healing (cleared by a later start, restart or cleanup). Returns issues sorted by key. Raises nothing for a missing systemd; a measurement failure of one class is reported as `ops_measure_unavailable:<class>` rather than silencing the class."""
    issues: list[MeasuredIssue] = []
    jobs: Sequence[JobSchedule]
    if schedules is None:
        try:
            jobs = load_job_schedules()
        except Exception as exc:  # noqa: BLE001 - measurement must not raise
            return (MeasuredIssue(key="ops_measure_unavailable:schedules", text=f"schedules unavailable: {exc}"),)
    else:
        jobs = schedules

    states: dict[str, UnitRunState] = {}
    units_ok = True
    if states_fn is None:
        try:
            states = read_unit_states([s.unit for s in jobs])
        except Exception as exc:  # noqa: BLE001 - one class failure must not silence others
            issues.append(MeasuredIssue(key="ops_measure_unavailable:units", text=f"unit states unavailable: {exc}"))
            units_ok = False
    else:
        try:
            states = dict(states_fn([s.unit for s in jobs]))
        except Exception as exc:  # noqa: BLE001 - injected fakes raise OSError in tests
            issues.append(MeasuredIssue(key="ops_measure_unavailable:units", text=f"unit states unavailable: {exc}"))
            units_ok = False

    if units_ok and jobs and not states:
        issues.append(MeasuredIssue(key="ops_measure_unavailable:units", text="systemd returned no unit state"))
        units_ok = False
    if units_ok:
        for schedule in jobs:
            try:
                verdict = evaluate_job(schedule, states.get(schedule.unit), now)
            except Exception as exc:  # noqa: BLE001 - a single bad clock must not drop other jobs
                issues.append(
                    MeasuredIssue(key="ops_measure_unavailable:units", text=f"unit states unavailable: {exc}")
                )
                break
            if verdict == JobVerdict.LATE:
                issues.append(MeasuredIssue(key=f"job_late:{schedule.unit}", text=f"{schedule.unit} missed its slot"))
            elif verdict == JobVerdict.OVERRUN:
                issues.append(
                    MeasuredIssue(key=f"job_overrun:{schedule.unit}", text=f"{schedule.unit} exceeded max runtime")
                )

    if inactive_timers_fn is None:
        declared = [s.timer for s in jobs]
        try:
            inactive = _default_inactive_timers(declared)
            issues.extend(MeasuredIssue(key=f"timer_inactive:{name}", text=f"{name} is not active") for name in inactive)
        except Exception as exc:  # noqa: BLE001 - measurement must not raise
            issues.append(MeasuredIssue(key="ops_measure_unavailable:timers", text=f"timers unavailable: {exc}"))
    else:
        try:
            inactive = inactive_timers_fn()
            issues.extend(MeasuredIssue(key=f"timer_inactive:{name}", text=f"{name} is not active") for name in inactive)
        except Exception as exc:  # noqa: BLE001 - injected fakes raise OSError in tests
            issues.append(MeasuredIssue(key="ops_measure_unavailable:timers", text=f"timers unavailable: {exc}"))

    disk_callable = disk_fn if disk_fn is not None else _default_disk_usage
    try:
        free_bytes, total_bytes = disk_callable()
        fraction = (free_bytes / total_bytes) if total_bytes > 0 else 1.0
        if fraction < HOST_DISK_FREE_MIN_FRACTION:
            percent = int(free_bytes * 100 / total_bytes) if total_bytes > 0 else 0
            issues.append(
                MeasuredIssue(key=f"host_disk_low:{percent}", text=f"data volume free {percent}% below threshold")
            )
    except Exception as exc:  # noqa: BLE001 - measurement must not raise
        issues.append(MeasuredIssue(key="ops_measure_unavailable:disk", text=f"disk unavailable: {exc}"))

    issues.sort(key=lambda issue: issue.key)
    return tuple(issues)
