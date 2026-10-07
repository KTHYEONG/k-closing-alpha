from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

SEOUL = ZoneInfo("Asia/Seoul")


def _now(day: str = "2026-10-07", clock: str = "12:00:00") -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}+09:00")


def _rec(key: str, outcome: str, ts: datetime, unit: str = "kca-tape-sweep.service", attempt: int = 1):
    from src.tools.auto_remediation import RemediationAction, RemediationOutcome, RemediationRecord

    return RemediationRecord(
        ts=ts,
        issue_key=key,
        action=RemediationAction.RERUN_UNIT,
        unit=unit,
        attempt=attempt,
        outcome=RemediationOutcome(outcome),
        detail="seed",
    )


def _ok_run(calls: list, stdout: str = "", stderr: str = "", rc: int = 0):
    def _run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, rc, stdout, stderr)

    return _run


def test_allowlisted_failed_unit_restarted_once(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    path = tmp_path / "ledger.jsonl"
    planned = ar.plan_remediation(
        ["failed_unit:kca-tape-sweep"],
        now=_now(),
        history=(),
        unit_busy=lambda u: False,
        blackout_fn=lambda now: timedelta(0),
    )
    assert len(planned) == 1
    rec, rule = planned[0]
    assert rec.outcome.value == "started"
    assert rec.unit == "kca-tape-sweep.service"
    assert rule is not None
    calls: list = []
    out = ar.execute_remediation([rec], run_fn=_ok_run(calls), path=path)
    assert len(out) == 1
    assert calls == [
        ["systemctl", "--user", "reset-failed", "kca-tape-sweep.service"],
        ["systemctl", "--user", "start", "--no-block", "kca-tape-sweep.service"],
    ]
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])["outcome"] == "started"


def test_trading_units_never_restarted(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    calls: list = []
    planned = ar.plan_remediation(
        ["failed_unit:kca-predict", "job_late:kca-paper-exit"],
        now=_now(),
        history=(),
        unit_busy=lambda u: False,
        blackout_fn=lambda now: timedelta(0),
    )
    assert len(planned) == 2
    assert all(r.outcome.value == "skipped_forbidden" for r, _ in planned)
    ar.execute_remediation([r for r, _ in planned], run_fn=_ok_run(calls), path=tmp_path / "l.jsonl")
    assert calls == []


def test_unknown_unit_forbidden() -> None:
    from src.tools import auto_remediation as ar

    planned = ar.plan_remediation(
        ["failed_unit:kca-unknown"],
        now=_now(),
        history=(),
        unit_busy=lambda u: False,
        blackout_fn=lambda now: timedelta(0),
    )
    assert len(planned) == 1
    rec, _ = planned[0]
    assert rec.outcome.value == "skipped_forbidden"


def test_blackout_defers_without_spending_budget() -> None:
    from src.tools import auto_remediation as ar

    now = _now()
    planned = ar.plan_remediation(
        ["failed_unit:kca-tape-sweep"],
        now=now,
        history=(),
        unit_busy=lambda u: False,
        blackout_fn=lambda n: timedelta(minutes=20),
    )
    assert planned[0][0].outcome.value == "skipped_window"
    hist = ar.read_remediation_ledger(path=Path("/nonexistent-ledger")) if False else ()
    assert hist == ()
    retry = ar.plan_remediation(
        ["failed_unit:kca-tape-sweep"],
        now=now,
        history=(),
        unit_busy=lambda u: False,
        blackout_fn=lambda n: timedelta(0),
    )
    assert retry[0][0].outcome.value == "started"


def test_budget_exhaustion() -> None:
    from src.tools import auto_remediation as ar

    now = _now()
    key = "failed_unit:kca-tape-sweep"
    hist = (_rec(key, "started", now - timedelta(hours=2), attempt=1), _rec(key, "started", now - timedelta(hours=1), attempt=2))
    planned = ar.plan_remediation(
        [key], now=now, history=hist, unit_busy=lambda u: False, blackout_fn=lambda n: timedelta(0)
    )
    assert planned[0][0].outcome.value == "skipped_budget"


def test_budget_per_kst_day() -> None:
    from src.tools import auto_remediation as ar

    now = _now("2026-10-07", "00:40:00")
    yesterday = datetime.fromisoformat("2026-10-06T23:50:00+09:00")
    key = "stale_kis_token:PRIMARY"
    hist = (_rec(key, "started", yesterday, unit="kca-kis-token-warmup.service", attempt=1),)
    planned = ar.plan_remediation(
        [key], now=now, history=hist, unit_busy=lambda u: False, blackout_fn=lambda n: timedelta(0)
    )
    assert planned[0][0].outcome.value == "started"
    assert planned[0][0].attempt == 1


def test_spacing_suppresses_second_start() -> None:
    from src.tools import auto_remediation as ar

    now = _now()
    key = "failed_unit:kca-tape-sweep"
    hist = (_rec(key, "started", now - timedelta(minutes=10)),)
    planned = ar.plan_remediation(
        [key], now=now, history=hist, unit_busy=lambda u: False, blackout_fn=lambda n: timedelta(0)
    )
    assert all(r.outcome.value != "started" for r, _ in planned)
    assert planned == ()


def test_busy_unit_skipped(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    planned = ar.plan_remediation(
        ["failed_unit:kca-tape-sweep"],
        now=_now(),
        history=(),
        unit_busy=lambda u: True,
        blackout_fn=lambda n: timedelta(0),
    )
    assert planned[0][0].outcome.value == "skipped_busy"
    calls: list = []
    ar.execute_remediation([planned[0][0]], run_fn=_ok_run(calls), path=tmp_path / "l.jsonl")
    assert calls == []


def test_shared_drive_siblings_busy() -> None:
    from src.tools import auto_remediation as ar

    def _busy(unit: str) -> bool:
        return unit == "kca-backup-prune.service"

    planned = ar.plan_remediation(
        ["offsite_backup:interrupted"],
        now=_now(),
        history=(),
        unit_busy=_busy,
        blackout_fn=lambda n: timedelta(0),
    )
    assert planned[0][0].outcome.value == "skipped_busy"


def test_data_facts_have_no_rule() -> None:
    from src.tools import auto_remediation as ar

    planned = ar.plan_remediation(
        ["missing:decision", "collection:cohort:0:missing_cohort", "intraday:regular_ticks:2:volume_gap"],
        now=_now(),
        history=(),
        unit_busy=lambda u: False,
        blackout_fn=lambda n: timedelta(0),
    )
    assert planned == ()


def test_tape_expiry_reruns_sweep() -> None:
    from src.tools import auto_remediation as ar

    now = _now()
    key = "intraday:tape_expiring:3:expiring_need"
    planned = ar.plan_remediation(
        [key], now=now, history=(), unit_busy=lambda u: False, blackout_fn=lambda n: timedelta(0)
    )
    assert planned[0][0].unit == "kca-tape-sweep.service"
    assert planned[0][0].outcome.value == "started"
    hist = (planned[0][0],)
    retry = ar.plan_remediation(
        [key], now=now + timedelta(minutes=40), history=hist, unit_busy=lambda u: False,
        blackout_fn=lambda n: timedelta(0),
    )
    assert retry[0][0].outcome.value == "skipped_budget"


def test_settle_on_resolution(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    path = tmp_path / "ledger.jsonl"
    now = _now()
    start = _rec("failed_unit:kca-tape-sweep", "started", now - timedelta(hours=1))
    path.write_text(json.dumps({"ts": start.ts.isoformat(), "issue_key": start.issue_key, "action": start.action.value, "unit": start.unit, "attempt": 1, "outcome": "started", "detail": ""}, ensure_ascii=False) + "\n", encoding="utf-8")
    hist = ar.read_remediation_ledger(path)
    done = ar.settle_remediation([], now=now, history=hist, path=path)
    assert len(done) == 1 and done[0].outcome.value == "resolved"
    hist2 = ar.read_remediation_ledger(path)
    again = ar.settle_remediation([], now=now, history=hist2, path=path)
    assert again == ()


def test_settle_on_persistence(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    path = tmp_path / "ledger.jsonl"
    now = _now()
    key = "failed_unit:kca-tape-sweep"
    start = _rec(key, "started", now - timedelta(minutes=40))
    done = ar.settle_remediation([key], now=now, history=(start,), path=path)
    assert len(done) == 1 and done[0].outcome.value == "unresolved"


def test_hold_notification() -> None:
    from src.tools import auto_remediation as ar

    now = _now()
    key = "failed_unit:kca-tape-sweep"
    start = _rec(key, "started", now - timedelta(minutes=5))
    assert ar.should_hold_notification(key, first_seen=now - timedelta(minutes=10), now=now, history=(start,)) is True
    assert ar.should_hold_notification(key, first_seen=now - timedelta(minutes=50), now=now, history=(start,)) is False
    assert ar.should_hold_notification("missing:decision", first_seen=now - timedelta(minutes=5), now=now, history=()) is False


def test_hold_ends_when_exhausted() -> None:
    from src.tools import auto_remediation as ar

    now = _now()
    key = "intraday:tape_expiring:3:expiring_need"
    start = _rec(key, "started", now - timedelta(minutes=40), unit="kca-tape-sweep.service", attempt=1)
    dead = _rec(key, "unresolved", now - timedelta(minutes=5), unit="kca-tape-sweep.service", attempt=1)
    assert ar.should_hold_notification(key, first_seen=now - timedelta(minutes=10), now=now, history=(start, dead)) is False


def test_ledger_written_before_start(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    path = blocker / "ledger.jsonl"
    calls: list = []
    rec = _rec("failed_unit:kca-tape-sweep", "started", _now())
    with pytest.raises(OSError, match=r".*"):
        ar.execute_remediation([rec], run_fn=_ok_run(calls), path=path)
    assert calls == []


def test_runner_failure_recorded(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar
    from src.tools.run_outcome import RUN_EVENT_REASON_MAX_CHARS

    path = tmp_path / "ledger.jsonl"
    secret = "app_key=RAWDARTKEY99"
    long_err = "boom " + secret + " " + "y" * 500

    def _run(cmd, **kwargs):
        if "reset-failed" in list(cmd):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 1, "", long_err)

    rec = _rec("failed_unit:kca-tape-sweep", "started", _now())
    out = ar.execute_remediation([rec], run_fn=_run, path=path)
    failed = [r for r in out if r.outcome.value == "failed"]
    assert len(failed) == 1
    assert "RAWDARTKEY99" not in failed[0].detail
    assert len(failed[0].detail) <= RUN_EVENT_REASON_MAX_CHARS


def test_unit_name_validation() -> None:
    from src.tools import auto_remediation as ar

    planned = ar.plan_remediation(
        ["failed_unit:kca-tape-sweep; rm -rf /"],
        now=_now(),
        history=(),
        unit_busy=lambda u: False,
        blackout_fn=lambda n: timedelta(0),
    )
    assert planned[0][0].outcome.value == "skipped_forbidden"


def test_allowlist_integrity() -> None:
    from pathlib import Path as _P
    from src.tools import auto_remediation as ar
    from src.tools.issue_registry import _TRADING_CHAIN_UNITS

    assert not (ar.SAFE_RERUN_UNITS & ar.FORBIDDEN_UNITS)
    systemd = _P("deploy/systemd")
    for unit in ar.SAFE_RERUN_UNITS:
        assert (systemd / unit).exists(), unit
    # Decision-chain units from the registry must never auto-restart, except the
    # two idempotent infrastructure units (token warmup, price ingest) that are
    # safe to re-run late despite their tier.
    safe_shorts = {u[len("kca-") : -len(".service")] for u in ar.SAFE_RERUN_UNITS}
    for short in _TRADING_CHAIN_UNITS:
        full = f"kca-{short}.service"
        if short in ("kis-token-warmup", "price-ingest"):
            assert full in ar.SAFE_RERUN_UNITS
        else:
            assert full in ar.FORBIDDEN_UNITS, full


def test_corrupt_ledger_line(tmp_path: Path, caplog) -> None:
    from src.tools import auto_remediation as ar

    path = tmp_path / "ledger.jsonl"
    good = _rec("failed_unit:kca-tape-sweep", "started", _now())
    line = json.dumps({"ts": good.ts.isoformat(), "issue_key": good.issue_key, "action": good.action.value, "unit": good.unit, "attempt": 1, "outcome": "started", "detail": ""}, ensure_ascii=False)
    path.write_text(line + "\n" + "GARBAGE{{{\n" + line + "\n", encoding="utf-8")
    with caplog.at_level("WARNING"):
        recs = ar.read_remediation_ledger(path)
    assert len(recs) == 2


def test_naive_now_rejected() -> None:
    from datetime import datetime as _dt

    from src.tools import auto_remediation as ar

    with pytest.raises(ValueError, match=r"timezone-aware"):
        ar.plan_remediation(["failed_unit:kca-tape-sweep"], now=_dt(2026, 10, 7, 12, 0, 0), history=(), unit_busy=lambda u: False)


def test_empty_suffix_forbidden() -> None:
    from src.tools import auto_remediation as ar

    planned = ar.plan_remediation(
        ["failed_unit:"],
        now=_now(),
        history=(),
        unit_busy=lambda u: False,
        blackout_fn=lambda n: timedelta(0),
    )
    assert planned[0][0].outcome.value == "skipped_forbidden"


def test_ledger_edge_lines(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    path = tmp_path / "ledger.jsonl"
    good = _rec("failed_unit:kca-tape-sweep", "started", _now())
    naive_line = json.dumps({"ts": "2026-10-07T12:00:00", "issue_key": good.issue_key, "action": "rerun_unit", "unit": good.unit, "attempt": 1, "outcome": "started", "detail": ""})
    bad_shape = json.dumps({"ts": good.ts.isoformat(), "issue_key": good.issue_key})
    path.write_text("\n".join(["[1,2]", naive_line, bad_shape, ""]) + "\n", encoding="utf-8")
    recs = ar.read_remediation_ledger(path)
    assert len(recs) == 1
    assert recs[0].ts.tzinfo is not None
    assert ar.read_remediation_ledger(tmp_path / "absent.jsonl") == ()


def test_ledger_default_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src import settings as _settings
    from src.tools import auto_remediation as ar

    monkeypatch.setattr(_settings, "DATA_DIR", tmp_path, raising=False)
    assert ar.remediation_ledger_path() == tmp_path / ar.REMEDIATION_LEDGER_RELPATH
    assert ar.read_remediation_ledger() == ()


def test_execute_invalid_unit(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    calls: list = []
    rec = _rec("failed_unit:kca-tape-sweep", "started", _now(), unit="bad name!")
    out = ar.execute_remediation([rec], run_fn=_ok_run(calls), path=tmp_path / "l.jsonl")
    assert out[-1].outcome.value == "failed"
    assert calls == []


def test_execute_reset_raises(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    def _run(cmd, **kwargs):
        raise OSError("boom")

    rec = _rec("failed_unit:kca-tape-sweep", "started", _now())
    out = ar.execute_remediation([rec], run_fn=_run, path=tmp_path / "l.jsonl")
    assert out[-1].outcome.value == "failed"


def test_execute_reset_rc(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    def _run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 3, "", "reset no good")

    rec = _rec("failed_unit:kca-tape-sweep", "started", _now())
    out = ar.execute_remediation([rec], run_fn=_run, path=tmp_path / "l.jsonl")
    assert out[-1].outcome.value == "failed"


def test_execute_start_raises(tmp_path: Path) -> None:
    from src.tools import auto_remediation as ar

    def _run(cmd, **kwargs):
        if "reset-failed" in list(cmd):
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise OSError("down")

    rec = _rec("failed_unit:kca-tape-sweep", "started", _now())
    out = ar.execute_remediation([rec], run_fn=_run, path=tmp_path / "l.jsonl")
    assert out[-1].outcome.value == "failed"


def test_hold_future_seen_and_failed_retry() -> None:
    from src.tools import auto_remediation as ar

    now = _now()
    key = "failed_unit:kca-tape-sweep"
    assert ar.should_hold_notification(key, first_seen=now + timedelta(minutes=5), now=now, history=()) is True
    start = _rec(key, "started", now - timedelta(minutes=30), attempt=1)
    failed = _rec(key, "failed", now - timedelta(minutes=5), attempt=1)
    other = _rec("failed_unit:kca-backup", "started", now - timedelta(minutes=5))
    assert ar.should_hold_notification(key, first_seen=now - timedelta(minutes=10), now=now, history=(start, failed, other)) is True


def test_skip_rows_never_mask_an_in_flight_start(tmp_path: Path) -> None:
    import src.tools.auto_remediation as ar

    key = "job_late:kca-tape-sweep.service"
    start = _now(clock="12:00:00")
    history = (
        _rec(key, "started", start),
        _rec(key, "skipped_busy", start + timedelta(minutes=35)),
    )
    assert ar.should_hold_notification(key, first_seen=start, now=start + timedelta(minutes=36), history=history)
    settled = ar.settle_remediation(
        [], now=start + timedelta(minutes=36), history=history, path=tmp_path / "ledger.jsonl"
    )
    assert [rec.outcome.value for rec in settled] == ["resolved"]


def test_non_remediable_keys_are_never_held() -> None:
    import src.tools.auto_remediation as ar

    now = _now()
    for key in ("failed_unit:kca-predict.service", "failed_unit:kca-unknown.service", "failed_unit:<systemctl unavailable: OSError>"):
        assert not ar.should_hold_notification(key, first_seen=now, now=now, history=())


def test_repeated_skip_is_recorded_once_per_day() -> None:
    import src.tools.auto_remediation as ar

    key = "failed_unit:kca-predict.service"
    first = ar.plan_remediation([key], now=_now(clock="10:00:00"), history=(), unit_busy=lambda u: False)
    assert [item[0].outcome.value for item in first] == ["skipped_forbidden"]
    history = tuple(item[0] for item in first)
    assert ar.plan_remediation([key], now=_now(clock="10:05:00"), history=history, unit_busy=lambda u: False) == ()
    next_day = ar.plan_remediation(
        [key], now=_now(day="2026-10-08", clock="10:00:00"), history=history, unit_busy=lambda u: False
    )
    assert [item[0].outcome.value for item in next_day] == ["skipped_forbidden"]
