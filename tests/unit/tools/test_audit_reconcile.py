from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

SEOUL = ZoneInfo("Asia/Seoul")


def _now(day: str = "2026-10-05", clock: str = "12:30:00") -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}+09:00")


def _seed(tmp_path: Path, *, snapshot: str = "2026-10-02", open_keys: dict, heartbeat_extra: dict | None = None):
    from src.tools.daily_audit import load_audit_alert_state  # noqa: F401

    state_path = tmp_path / "alert_state.json"
    hb_path = tmp_path / "heartbeat.json"
    now = _now()
    open_entries = {}
    for key, meta in open_keys.items():
        open_entries[key] = {
            "transient": meta.get("transient", True),
            "text": meta.get("text", key),
            "first_seen": meta.get("first_seen", "2026-10-02T21:20:00+09:00"),
            "last_notified": meta.get("last_notified", "2026-10-02T21:20:00+09:00"),
        }
    state_path.write_text(
        json.dumps(
            {"schema_version": 1, "snapshot_date": snapshot, "updated_at": now.isoformat(), "open": open_entries},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    hb = {
        "snapshot_date": snapshot,
        "day_kind": "trading",
        "subject": "[kca] 🚨 2026-10-02 일일점검 경고: 백업이상 offsite_backup:stale",
        "undelivered_alerts": 0,
        "finished_at": "2026-10-02T21:20:00+09:00",
        "schema_version": 2,
        "severity": "WARNING" if open_entries else "OK",
        "open_issues": [
            {"key": k, "transient": v["transient"], "text": v["text"]} for k, v in open_entries.items()
        ],
        "provisional_reasons": [],
        "audit_kind": "scheduled",
        "reconciled_at": None,
    }
    if heartbeat_extra:
        hb.update(heartbeat_extra)
    hb_path.write_text(json.dumps(hb, ensure_ascii=False), encoding="utf-8")
    return state_path, hb_path


def _clean_fns(*, backup: list[str] | Exception | None = None, failed: list[str] | Exception | None = None, tokens: list[str] | Exception | None = None, outbox: int | Exception = 0):
    _backup = [] if backup is None else backup
    _failed_list = [] if failed is None else failed
    _tokens = [] if tokens is None else tokens

    def _failed():
        if isinstance(_failed_list, Exception):
            raise _failed_list
        return list(_failed_list)

    def _tokens_fn(_snapshot: str):
        if isinstance(_tokens, Exception):
            raise _tokens
        return list(_tokens)

    def _backup_fn(_now: datetime):
        if isinstance(_backup, Exception):
            raise _backup
        return list(_backup)

    def _outbox():
        if isinstance(outbox, Exception):
            raise outbox
        return int(outbox)

    return _failed, _tokens_fn, _backup_fn, _outbox


def test_friday_false_positive_heals(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "백업 이상: offsite_backup:stale"}})
    sent: list[tuple[str, str]] = []
    f, t, b, o = _clean_fns(backup=[])
    result = run_audit_reconcile(
        now=_now("2026-10-05"),
        failed_units_fn=f,
        stale_tokens_fn=t,
        backup_issues_fn=b,
        outbox_fn=o,
        dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True, "email": True},
        state_path=sp,
        heartbeat_path=hp,
    )
    assert result.action == "RESOLVED"
    assert result.resolved == ("offsite_backup:stale",)
    assert len(sent) == 1 and "경고" not in sent[0][0]
    hb = json.loads(hp.read_text(encoding="utf-8"))
    assert hb["severity"] == "OK" and hb["open_issues"] == []
    assert hb["audit_kind"] == "reconcile" and hb["reconciled_at"] is not None
    state = json.loads(sp.read_text(encoding="utf-8"))
    assert state["open"] == {}


def test_no_spam_within_reminder_window(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "백업 이상"}})
    f, t, b, o = _clean_fns(backup=["offsite_backup:stale"])
    stamp = "2026-10-05T12:00:00+09:00"
    sp.write_text(
        json.dumps({"schema_version": 1, "snapshot_date": "2026-10-02", "updated_at": stamp,
                    "open": {"offsite_backup:stale": {"transient": True, "text": "백업 이상",
                             "first_seen": stamp, "last_notified": stamp}}}),
        encoding="utf-8",
    )
    sent: list = []
    r1 = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                             outbox_fn=o, dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                             state_path=sp, heartbeat_path=hp)
    r2 = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                             outbox_fn=o, dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                             state_path=sp, heartbeat_path=hp)
    assert r1.action == "NOOP" and r2.action == "NOOP"
    assert sent == []


def test_reminder_after_24h(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    old = "2026-10-03T12:00:00+09:00"
    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "백업 이상",
                                                                 "first_seen": old, "last_notified": old}})
    sent: list[tuple[str, str]] = []
    f, t, b, o = _clean_fns(backup=["offsite_backup:stale"])
    result = run_audit_reconcile(now=_now("2026-10-05"), failed_units_fn=f, stale_tokens_fn=t,
                                 backup_issues_fn=b, outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "REMINDED" and result.notified
    assert len(sent) == 1 and "재알림" in sent[0][0]
    state = json.loads(sp.read_text(encoding="utf-8"))
    assert state["open"]["offsite_backup:stale"]["last_notified"] > old


def test_new_issue_notifies_once(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={})
    sent: list[tuple[str, str]] = []
    f, t, b, o = _clean_fns(failed=["kca-collect.service"])
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "OPENED" and result.opened == ("failed_unit:kca-collect.service",)
    assert len(sent) == 1 and "kca-collect.service" in sent[0][0]
    hb = json.loads(hp.read_text(encoding="utf-8"))
    assert hb["severity"] == "WARNING" and len(hb["open_issues"]) == 1
    sent.clear()
    f2, t2, b2, o2 = _clean_fns(failed=["kca-collect.service"])
    r2 = run_audit_reconcile(now=_now(), failed_units_fn=f2, stale_tokens_fn=t2, backup_issues_fn=b2,
                             outbox_fn=o2,
                             dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                             state_path=sp, heartbeat_path=hp)
    assert r2.action == "NOOP" and sent == []


def test_persistent_issues_survive(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={
        "missing:intraday_complete": {"transient": False, "text": "누락 단계: intraday_complete"},
        "offsite_backup:stale": {"transient": True, "text": "백업 이상: offsite_backup:stale"},
    })
    sent: list[tuple[str, str]] = []
    f, t, b, o = _clean_fns(backup=[])
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "CHANGED"
    assert result.resolved == ("offsite_backup:stale",)
    assert "missing:intraday_complete" in result.still_open
    hb = json.loads(hp.read_text(encoding="utf-8"))
    assert hb["severity"] == "WARNING"
    assert len(sent) == 1 and "경고" in sent[0][0]


def test_measurement_failure_never_resolves(tmp_path, caplog) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    recent = "2026-10-05T12:00:00+09:00"
    sp, hp = _seed(tmp_path, open_keys={"failed_unit:kca-x.service": {"transient": True, "text": "실패 유닛: kca-x.service",
                                                                      "first_seen": recent, "last_notified": recent}})
    sent: list = []
    f, t, b, o = _clean_fns(failed=RuntimeError("boom"))
    with caplog.at_level("WARNING"):
        result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                     outbox_fn=o,
                                     dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                     state_path=sp, heartbeat_path=hp)
    assert result.action == "NOOP" and sent == []
    assert any("[SYS]" in rec.message for rec in caplog.records)
    state = json.loads(sp.read_text(encoding="utf-8"))
    assert "failed_unit:kca-x.service" in state["open"]


def test_running_backup_is_provisional(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={})
    sent: list = []
    f, t, b, o = _clean_fns(backup=["offsite_backup:running"])
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "NOOP" and sent == []
    hb = json.loads(hp.read_text(encoding="utf-8"))
    assert "offsite_backup:running" in hb["provisional_reasons"]
    assert hb["open_issues"] == []


def test_interrupted_backup_opens_warning(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={})
    sent: list[tuple[str, str]] = []
    f, t, b, o = _clean_fns(backup=["offsite_backup:interrupted"])
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "OPENED" and result.opened == ("offsite_backup:interrupted",)
    assert len(sent) == 1 and "경고" in sent[0][0]


def test_dispatch_failure_retries(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={})
    before = sp.read_text(encoding="utf-8")
    f, t, b, o = _clean_fns(failed=["kca-y.service"])
    r1 = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                             outbox_fn=o,
                             dispatch_fn=lambda s, body: {"webhook": False, "email": False},
                             state_path=sp, heartbeat_path=hp)
    assert r1.action == "OPENED" and not r1.notified
    assert sp.read_text(encoding="utf-8") == before
    sent: list = []
    f2, t2, b2, o2 = _clean_fns(failed=["kca-y.service"])
    r2 = run_audit_reconcile(now=_now(), failed_units_fn=f2, stale_tokens_fn=t2, backup_issues_fn=b2,
                             outbox_fn=o2,
                             dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                             state_path=sp, heartbeat_path=hp)
    assert r2.notified and len(sent) == 1


def test_crash_ordering_raises_oserror(tmp_path, monkeypatch) -> None:
    import pytest

    from src.tools import audit_reconcile
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "백업 이상"}})
    f, t, b, o = _clean_fns(backup=[])

    def _boom(*args, **kwargs):
        raise OSError("disk full: heartbeat")

    monkeypatch.setattr(audit_reconcile, "write_audit_heartbeat", _boom)
    with pytest.raises(OSError, match="disk full"):
        run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                            outbox_fn=o,
                            dispatch_fn=lambda s, body: {"webhook": True},
                            state_path=sp, heartbeat_path=hp)


def test_weekend_no_state_is_noop(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp = tmp_path / "missing_state.json"
    hp = tmp_path / "missing_hb.json"
    f, t, b, o = _clean_fns(backup=[])
    result = run_audit_reconcile(now=datetime.fromisoformat("2026-10-04T12:30:00+09:00"),
                                 failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "NOOP"
    assert not sp.exists() and not hp.exists()


def test_weekend_open_issue_resolves(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "백업 이상"}})
    sent: list = []
    f, t, b, o = _clean_fns(backup=[])
    result = run_audit_reconcile(now=datetime.fromisoformat("2026-10-04T12:30:00+09:00"),
                                 failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "RESOLVED" and len(sent) == 1


def test_heartbeat_v2_backward_compatible(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "백업 이상"}})
    f, t, b, o = _clean_fns(backup=["offsite_backup:stale"])
    run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                        outbox_fn=o, dispatch_fn=lambda s, body: {"webhook": True},
                        state_path=sp, heartbeat_path=hp)
    hb = json.loads(hp.read_text(encoding="utf-8"))
    assert isinstance(hb["snapshot_date"], str) and isinstance(hb["subject"], str)
    assert isinstance(hb["undelivered_alerts"], int) and isinstance(hb["finished_at"], str)
    assert hb["schema_version"] == 2
    assert ("경고" in hb["subject"]) == (len(hb["open_issues"]) > 0)


def test_full_audit_replaces_state_retaining_history(tmp_path) -> None:
    from src.tools import daily_audit

    now = _now()
    prev_path = tmp_path / "alert_state.json"
    prev_path.write_text(json.dumps({
        "schema_version": 1, "snapshot_date": "2026-10-01", "updated_at": "2026-10-01T21:20:00+09:00",
        "open": {
            "failed_unit:kca-a.service": {"transient": True, "text": "실패 유닛: kca-a.service",
                                          "first_seen": "2026-10-01T21:20:00+09:00",
                                          "last_notified": "2026-10-01T21:20:00+09:00"},
            "offsite_backup:stale": {"transient": True, "text": "백업 이상",
                                     "first_seen": "2026-10-01T21:20:00+09:00",
                                     "last_notified": "2026-10-01T21:20:00+09:00"},
        }}), encoding="utf-8")
    digest = daily_audit.build_digest(
        "2026-10-02", daily_audit.DAY_TRADING,
        dict.fromkeys(daily_audit.AUDIT_STEPS, True),
        ["kca-a.service"], [],
        backup_issues=["offsite_backup:stale"],
    )
    assert any(i.key == "failed_unit:kca-a.service" for i in digest.issues)
    daily_audit.sync_audit_alert_state_from_digest(digest, "2026-10-02", now=now, path=prev_path)
    state = json.loads(prev_path.read_text(encoding="utf-8"))
    assert state["open"]["failed_unit:kca-a.service"]["first_seen"] == "2026-10-01T21:20:00+09:00"
    assert state["snapshot_date"] == "2026-10-02"


def test_digest_issues_drive_subject() -> None:
    from src.tools import daily_audit

    digest = daily_audit.build_digest(
        "2026-10-02", daily_audit.DAY_TRADING,
        dict.fromkeys(daily_audit.AUDIT_STEPS, True),
        [], [],
        backup_issues=["offsite_backup:stale"],
    )
    assert digest.severity is daily_audit.DigestSeverity.WARNING
    assert [i.key for i in digest.issues] == ["offsite_backup:stale"]
    assert "offsite_backup:stale" in digest.subject


def test_lock_serialization_documented() -> None:
    daily = Path("deploy/systemd/kca-daily-audit.service").read_text(encoding="utf-8")
    reconcile = Path("deploy/systemd/kca-audit-reconcile.service").read_text(encoding="utf-8")
    assert "flock -w 600 %t/kca-audit.lock" in daily
    assert "flock -w 600 %t/kca-audit.lock" in reconcile


def test_transient_key_classification() -> None:
    from src.tools.daily_audit import is_transient_issue_key

    assert is_transient_issue_key("undelivered_alerts")
    assert is_transient_issue_key("offsite_backup:stale")
    assert is_transient_issue_key("failed_unit:kca-a.service")
    assert is_transient_issue_key("stale_kis_token:DATA_3")
    assert not is_transient_issue_key("missing:archive")
    assert not is_transient_issue_key("collection:charts:1:missing_entries")
    assert not is_transient_issue_key("intraday:regular:1:missing_partition")
    assert not is_transient_issue_key("calendar_disagreement")
    assert not is_transient_issue_key("expiry:KRX-calendar")


def test_heartbeat_severity_inference(tmp_path) -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from src.tools.daily_audit import write_audit_heartbeat

    finished = datetime(2026, 10, 2, 21, 20, tzinfo=ZoneInfo("Asia/Seoul"))
    warn = write_audit_heartbeat("2026-10-02", day_kind="trading", subject="[kca] 일일점검 경고: x",
                                 undelivered_alerts=0, finished_at=finished, path=tmp_path / "w.json")
    assert json.loads(warn.read_text(encoding="utf-8"))["severity"] == "WARNING"
    skip = write_audit_heartbeat("2026-10-02", day_kind="holiday", subject="[kca] ⏸️ 2026-10-02 휴장일 SKIP",
                                 undelivered_alerts=0, finished_at=finished, path=tmp_path / "s.json")
    assert json.loads(skip.read_text(encoding="utf-8"))["severity"] == "HOLIDAY_SKIP"
    holiday_kind = write_audit_heartbeat("2026-10-02", day_kind="holiday", subject="[kca] plain",
                                         undelivered_alerts=0, finished_at=finished, path=tmp_path / "h.json")
    assert json.loads(holiday_kind.read_text(encoding="utf-8"))["severity"] == "HOLIDAY_SKIP"
    ok = write_audit_heartbeat("2026-10-02", day_kind="trading", subject="[kca] 🟢 정상",
                               undelivered_alerts=0, finished_at=finished, path=tmp_path / "o.json")
    assert json.loads(ok.read_text(encoding="utf-8"))["severity"] == "OK"


def test_alert_state_unreadable_fails_closed(tmp_path) -> None:
    import pytest
    from src.tools.daily_audit import load_audit_alert_state

    corrupt = tmp_path / "state.json"
    corrupt.write_text("{not json", encoding="utf-8")
    with pytest.raises(OSError, match="Unreadable audit alert state"):
        load_audit_alert_state(corrupt)
    non_dict = tmp_path / "state2.json"
    non_dict.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(OSError, match="Invalid audit alert state"):
        load_audit_alert_state(non_dict)
    assert load_audit_alert_state(tmp_path / "absent.json") is None


def test_parse_time_branches() -> None:
    from datetime import datetime

    from src.tools.audit_reconcile import _parse_time

    fallback = _now()
    assert _parse_time("not-a-time", fallback) is fallback
    assert _parse_time(None, fallback) is fallback
    naive = _parse_time("2026-10-05T12:00:00", fallback)
    assert naive.tzinfo is not None
    aware = _parse_time("2026-10-05T12:00:00+09:00", fallback)
    assert aware.isoformat() == "2026-10-05T12:00:00+09:00"


def test_default_outbox_count(tmp_path, monkeypatch) -> None:
    from src.tools import audit_reconcile
    from src.tools.audit_reconcile import _default_outbox_count

    monkeypatch.setattr(audit_reconcile, "alert_outbox_dir", lambda: tmp_path / "absent")
    assert _default_outbox_count() == 0
    box = tmp_path / "box"
    box.mkdir()
    (box / "a.json").write_text("{}", encoding="utf-8")
    (box / "b.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(audit_reconcile, "alert_outbox_dir", lambda: box)
    assert _default_outbox_count() == 2


def test_stale_tokens_failure_keeps_previous(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    recent = "2026-10-05T12:00:00+09:00"
    sp, hp = _seed(tmp_path, open_keys={"stale_kis_token:DATA_3": {"transient": True, "text": "KIS 토큰 누락: DATA_3",
                                                                  "first_seen": recent, "last_notified": recent}})
    sent: list = []
    f, _t, b, o = _clean_fns()
    result = run_audit_reconcile(now=_now(), failed_units_fn=f,
                                 stale_tokens_fn=lambda _s: (_ for _ in ()).throw(RuntimeError("kis down")),
                                 backup_issues_fn=b, outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "NOOP" and sent == []
    assert "stale_kis_token:DATA_3" in json.loads(sp.read_text(encoding="utf-8"))["open"]


def test_backup_and_outbox_failure_keep_previous(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    recent = "2026-10-05T12:00:00+09:00"
    sp, hp = _seed(tmp_path, open_keys={
        "offsite_backup:stale": {"transient": True, "text": "t", "first_seen": recent, "last_notified": recent},
        "undelivered_alerts": {"transient": True, "text": "t", "first_seen": recent, "last_notified": recent},
    })
    sent: list = []
    f, t, _b, _o = _clean_fns()
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t,
                                 backup_issues_fn=lambda _n: (_ for _ in ()).throw(OSError("backup down")),
                                 outbox_fn=lambda: (_ for _ in ()).throw(OSError("outbox down")),
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "NOOP" and sent == []
    state = json.loads(sp.read_text(encoding="utf-8"))
    assert "offsite_backup:stale" in state["open"] and "undelivered_alerts" in state["open"]


def test_naive_now_rejected(tmp_path) -> None:
    import pytest

    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={})
    f, t, b, o = _clean_fns()
    with pytest.raises(ValueError, match="timezone-aware"):
        run_audit_reconcile(now=datetime(2026, 10, 5, 12, 30), failed_units_fn=f, stale_tokens_fn=t,
                            backup_issues_fn=b, outbox_fn=o,
                            dispatch_fn=lambda s, body: {"webhook": True},
                            state_path=sp, heartbeat_path=hp)


def test_dry_run_writes_nothing(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={})
    before_state, before_hb = sp.read_text(encoding="utf-8"), hp.read_text(encoding="utf-8")
    f, t, b, o = _clean_fns(failed=["kca-dry.service"])
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o, dispatch_fn=lambda s, body: {"webhook": True},
                                 state_path=sp, heartbeat_path=hp, dry_run=True)
    assert result.action == "OPENED" and not result.notified
    assert sp.read_text(encoding="utf-8") == before_state
    assert hp.read_text(encoding="utf-8") == before_hb


def test_dry_run_resolved_and_reminder(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "t"}})
    f, t, b, o = _clean_fns(backup=[])
    r = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                            outbox_fn=o, dispatch_fn=lambda s, body: {"webhook": True},
                            state_path=sp, heartbeat_path=hp, dry_run=True)
    assert r.action == "RESOLVED" and not r.notified
    old = "2026-10-03T12:00:00+09:00"
    sp2, hp2 = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "t",
                                                                   "first_seen": old, "last_notified": old}})
    f2, t2, b2, o2 = _clean_fns(backup=["offsite_backup:stale"])
    r2 = run_audit_reconcile(now=_now(), failed_units_fn=f2, stale_tokens_fn=t2, backup_issues_fn=b2,
                             outbox_fn=o2, dispatch_fn=lambda s, body: {"webhook": True},
                             state_path=sp2, heartbeat_path=hp2, dry_run=True)
    assert r2.action == "REMINDED" and not r2.notified


def test_dry_run_noop_clean_and_changed(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={})
    f, t, b, o = _clean_fns()
    r = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                            outbox_fn=o, dispatch_fn=lambda s, body: {"webhook": True},
                            state_path=sp, heartbeat_path=hp, dry_run=True)
    assert r.action == "NOOP"
    sp2, hp2 = _seed(tmp_path, open_keys={
        "missing:intraday_complete": {"transient": False, "text": "t"},
        "offsite_backup:stale": {"transient": True, "text": "t"},
    })
    f2, t2, b2, o2 = _clean_fns()
    r2 = run_audit_reconcile(now=_now(), failed_units_fn=f2, stale_tokens_fn=t2, backup_issues_fn=b2,
                             outbox_fn=o2, dispatch_fn=lambda s, body: {"webhook": True},
                             state_path=sp2, heartbeat_path=hp2, dry_run=True)
    assert r2.action == "CHANGED" and not r2.notified


def test_reconcile_without_heartbeat_file(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "t"}})
    hp.unlink()
    sent: list = []
    f, t, b, o = _clean_fns(backup=[])
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "RESOLVED" and hp.exists()


def test_holiday_skip_preserved_on_clean_noop(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={}, heartbeat_extra={
        "severity": "HOLIDAY_SKIP", "subject": "[kca] ⏸️ 2026-10-02 휴장일 SKIP", "day_kind": "holiday"})
    f, t, b, o = _clean_fns()
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o, dispatch_fn=lambda s, body: {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "NOOP"
    assert json.loads(hp.read_text(encoding="utf-8"))["severity"] == "HOLIDAY_SKIP"


def test_sync_rejects_non_dict_open(tmp_path) -> None:
    import pytest
    from src.tools import daily_audit

    path = tmp_path / "state.json"
    path.write_text(json.dumps({"schema_version": 1, "snapshot_date": "2026-10-01",
                                "updated_at": "2026-10-01T21:20:00+09:00", "open": [1, 2]}), encoding="utf-8")
    digest = daily_audit.build_digest("2026-10-02", daily_audit.DAY_TRADING,
                                      dict.fromkeys(daily_audit.AUDIT_STEPS, True), [], [])
    assert digest.severity is daily_audit.DigestSeverity.OK
    before = path.read_text(encoding="utf-8")
    with pytest.raises(OSError, match="Invalid audit alert state"):
        daily_audit.sync_audit_alert_state_from_digest(digest, "2026-10-02", now=_now(), path=path)
    assert path.read_text(encoding="utf-8") == before


def test_stale_tokens_open_warning(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={})
    sent: list = []
    f, _t, b, o = _clean_fns()
    result = run_audit_reconcile(now=_now(), failed_units_fn=f,
                                 stale_tokens_fn=lambda _s: ["DATA_3"],
                                 backup_issues_fn=b, outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "OPENED" and result.opened == ("stale_kis_token:DATA_3",)
    assert sent and "DATA_3" in sent[0][0]


def test_outbox_backlog_opens_warning(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={})
    sent: list = []
    f, t, b, _o = _clean_fns()
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=lambda: 3,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "OPENED" and result.opened == ("undelivered_alerts",)
    assert sent and "미전송" in sent[0][1]


def test_missing_snapshot_is_noop(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp = tmp_path / "state.json"
    hp = tmp_path / "hb.json"
    sp.write_text(json.dumps({"schema_version": 1, "updated_at": _now().isoformat(), "open": {}}), encoding="utf-8")
    hp.write_text(json.dumps({"day_kind": "trading"}), encoding="utf-8")
    f, t, b, o = _clean_fns()
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o, dispatch_fn=lambda s, body: {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "NOOP"


def test_non_dict_state_entries_fail_closed(tmp_path) -> None:
    import pytest
    from src.tools.audit_reconcile import run_audit_reconcile

    sp = tmp_path / "state.json"
    hp = tmp_path / "hb.json"
    sp.write_text(json.dumps({"schema_version": 1, "snapshot_date": "2026-10-02",
                              "updated_at": _now().isoformat(), "open": {"junk": "not-a-dict"}}), encoding="utf-8")
    hp.write_text(json.dumps({"snapshot_date": "2026-10-02", "day_kind": "trading",
                              "subject": "x", "finished_at": _now().isoformat()}), encoding="utf-8")
    f, t, b, o = _clean_fns()
    sent: list = []
    with pytest.raises(OSError, match="Invalid audit alert state entry"):
        run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert sent == []


def test_heartbeat_bad_finished_at_and_undelivered(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={}, heartbeat_extra={
        "finished_at": "garbage", "undelivered_alerts": "bad"})
    calls = {"n": 0}

    def _flaky_outbox():
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("outbox down on refresh")
        return 0

    f, t, b, _o = _clean_fns()
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=_flaky_outbox,
                                 dispatch_fn=lambda s, body: {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "NOOP"
    hb = json.loads(hp.read_text(encoding="utf-8"))
    assert hb["undelivered_alerts"] == 0


def test_heartbeat_naive_finished_at(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={}, heartbeat_extra={"finished_at": "2026-10-02T21:20:00"})
    f, t, b, o = _clean_fns()
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o, dispatch_fn=lambda s, body: {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "NOOP"
    hb = json.loads(hp.read_text(encoding="utf-8"))
    assert hb["finished_at"] == "2026-10-02T21:20:00+09:00"


def test_persist_without_prior_notification_stamp(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={
        "offsite_backup:stale": {"transient": True, "text": "t"},
    })
    state = json.loads(sp.read_text(encoding="utf-8"))
    del state["open"]["offsite_backup:stale"]["last_notified"]
    del state["open"]["offsite_backup:stale"]["first_seen"]
    sp.write_text(json.dumps(state), encoding="utf-8")
    sent: list = []
    f, t, b, o = _clean_fns(backup=["offsite_backup:stale"], failed=["kca-new.service"])
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: sent.append((s, body)) or {"webhook": True},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "OPENED" and result.notified
    kept = json.loads(sp.read_text(encoding="utf-8"))["open"]["offsite_backup:stale"]
    assert kept["last_notified"] and kept["first_seen"]


def test_reminder_dispatch_failure_retries(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    old = "2026-10-03T12:00:00+09:00"
    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "t",
                                                                 "first_seen": old, "last_notified": old}})
    before = sp.read_text(encoding="utf-8")
    f, t, b, o = _clean_fns(backup=["offsite_backup:stale"])
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: {"webhook": False},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "REMINDED" and not result.notified
    assert sp.read_text(encoding="utf-8") == before


def test_resolved_dispatch_failure_retries(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True, "text": "t"}})
    before = sp.read_text(encoding="utf-8")
    f, t, b, o = _clean_fns(backup=[])
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: {"webhook": False},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "RESOLVED" and not result.notified
    assert sp.read_text(encoding="utf-8") == before


def test_changed_dispatch_failure_retries(tmp_path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={
        "missing:intraday_complete": {"transient": False, "text": "t"},
        "offsite_backup:stale": {"transient": True, "text": "t"},
    })
    before = sp.read_text(encoding="utf-8")
    f, t, b, o = _clean_fns(backup=[])
    result = run_audit_reconcile(now=_now(), failed_units_fn=f, stale_tokens_fn=t, backup_issues_fn=b,
                                 outbox_fn=o,
                                 dispatch_fn=lambda s, body: {"email": False},
                                 state_path=sp, heartbeat_path=hp)
    assert result.action == "CHANGED" and not result.notified
    assert sp.read_text(encoding="utf-8") == before


def _review_run(sp, hp, sent, *, backup=(), tokens_fn=None, dry_run=False):
    from src.tools.audit_reconcile import run_audit_reconcile

    return run_audit_reconcile(
        now=_now(), failed_units_fn=lambda: [],
        stale_tokens_fn=tokens_fn or (lambda _: []),
        backup_issues_fn=lambda _: list(backup), outbox_fn=lambda: 0,
        dispatch_fn=lambda s, b: sent.append((s, b)) or {"email": True},
        state_path=sp, heartbeat_path=hp, dry_run=dry_run,
        reminder_after=timedelta(days=10),
    )


def test_missing_state_recovers_persistent_heartbeat_issues(tmp_path) -> None:
    sp, hp = _seed(tmp_path, open_keys={
        "missing:intraday_complete": {"transient": False, "text": "누락 단계"},
    })
    sp.unlink()
    sent = []
    result = _review_run(sp, hp, sent)
    assert result.action == "NOOP" and sent == []
    assert "missing:intraday_complete" in json.loads(sp.read_text())["open"]
    hb = json.loads(hp.read_text())
    assert hb["severity"] == "WARNING" and "경고" in hb["subject"]
    assert hb["finished_at"] == "2026-10-02T21:20:00+09:00"


def test_corrupt_state_preserves_heartbeat_and_dispatches_nothing(tmp_path) -> None:
    import pytest

    sp, hp = _seed(tmp_path, open_keys={"missing:archive": {"transient": False}})
    sp.write_text("{broken")
    before = hp.read_text()
    sent = []
    with pytest.raises(OSError, match="Unreadable audit alert state"):
        _review_run(sp, hp, sent)
    assert hp.read_text() == before and sent == []


def test_heartbeat_failure_recovered_without_duplicate_resolution(tmp_path, monkeypatch) -> None:
    import pytest
    from src.tools import audit_reconcile

    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True}})
    writer = audit_reconcile.write_audit_heartbeat
    sent = []

    def _fail(*args, **kwargs):
        raise OSError("heartbeat disk failure")

    monkeypatch.setattr(audit_reconcile, "write_audit_heartbeat", _fail)
    with pytest.raises(OSError, match="heartbeat disk failure"):
        _review_run(sp, hp, sent)
    assert json.loads(sp.read_text())["open"] == {} and len(sent) == 1
    monkeypatch.setattr(audit_reconcile, "write_audit_heartbeat", writer)
    result = _review_run(sp, hp, sent)
    hb = json.loads(hp.read_text())
    assert result.action == "NOOP" and len(sent) == 1
    assert hb["severity"] == "OK" and hb["open_issues"] == []
    assert "경고" not in hb["subject"]


def test_running_suspends_old_warning_until_completion(tmp_path) -> None:
    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True}})
    sent = []
    for _ in range(2):
        result = _review_run(sp, hp, sent, backup=["offsite_backup:running"])
        hb = json.loads(hp.read_text())
        assert result.action == "NOOP" and sent == []
        assert hb["open_issues"] == [] and hb["provisional_reasons"] == ["offsite_backup:running"]
        assert "경고" not in hb["subject"]
        assert json.loads(sp.read_text())["pending_resolutions"]["offsite_backup:stale"]
    result = _review_run(sp, hp, sent)
    assert result.action == "RESOLVED" and len(sent) == 1
    assert "경고" not in sent[0][0]
    assert "pending_resolutions" not in json.loads(sp.read_text())
    _review_run(sp, hp, sent)
    assert len(sent) == 1


def test_running_to_interrupted_opens_once(tmp_path) -> None:
    sp, hp = _seed(tmp_path, open_keys={"offsite_backup:stale": {"transient": True}})
    sent = []
    _review_run(sp, hp, sent, backup=["offsite_backup:running"])
    result = _review_run(sp, hp, sent, backup=["offsite_backup:interrupted"])
    assert result.opened == ("offsite_backup:interrupted",) and len(sent) == 1
    assert json.loads(hp.read_text())["severity"] == "WARNING"
    _review_run(sp, hp, sent, backup=["offsite_backup:interrupted"])
    assert len(sent) == 1


def test_legacy_warning_without_state_fails_closed(tmp_path) -> None:
    import pytest

    sp, hp = _seed(tmp_path, open_keys={})
    sp.unlink()
    hb = json.loads(hp.read_text())
    del hb["open_issues"]
    hp.write_text(json.dumps(hb))
    sent = []
    with pytest.raises(OSError, match="Cannot recover legacy warning"):
        _review_run(sp, hp, sent)
    assert sent == [] and not sp.exists()


def test_corrupt_heartbeat_fails_closed(tmp_path) -> None:
    import pytest
    from src.tools.daily_audit import read_audit_heartbeat

    path = tmp_path / "hb.json"
    path.write_text("{broken")
    with pytest.raises(OSError, match="Unreadable audit heartbeat"):
        read_audit_heartbeat(path)
    path.write_text("[]")
    with pytest.raises(OSError, match="Invalid audit heartbeat"):
        read_audit_heartbeat(path)


def test_newer_token_issuance_heals_historical_warning(tmp_path) -> None:
    from src.api.kis.key_pool import token_cache_path
    from src.tools.daily_audit import list_stale_kis_tokens

    env = {"KIS_DATA_SLOTS": "1,2", "KIS_HOST_DATA_SLOTS": "1,2",
           "KIS_DATA_1_APP_KEY": "review-key-1", "KIS_DATA_1_APP_SECRET": "review-secret-1",
           "KIS_DATA_2_APP_KEY": "review-key-2", "KIS_DATA_2_APP_SECRET": "review-secret-2",
           "KIS_APP_KEY": "review-primary", "KIS_APP_SECRET": "review-primary-secret"}
    token_cache_path("review-primary", tmp_path).write_text(json.dumps({"issued_at": "2026-10-02T07:05:00+09:00"}))
    token_cache_path("review-key-1", tmp_path).write_text(json.dumps({"issued_at": "2026-10-05T07:05:00+09:00"}))
    token_cache_path("review-key-2", tmp_path).write_text(json.dumps({"issued_at": "2026-10-01T07:05:00+09:00"}))
    assert list_stale_kis_tokens("2026-10-02", env=env, cache_dir=tmp_path) == ["DATA_1", "DATA_2"]
    assert list_stale_kis_tokens("2026-10-02", env=env, cache_dir=tmp_path, allow_newer=True) == ["DATA_2"]
    sp, hp = _seed(tmp_path, open_keys={"stale_kis_token:DATA_1": {"transient": True}})
    token_cache_path("review-key-2", tmp_path).write_text(json.dumps({"issued_at": "2026-10-02T07:05:00+09:00"}))
    sent = []
    result = _review_run(sp, hp, sent, tokens_fn=lambda d: list_stale_kis_tokens(d, env=env, cache_dir=tmp_path, allow_newer=True))
    assert result.action == "RESOLVED" and len(sent) == 1
    assert json.loads(hp.read_text())["open_issues"] == []


def test_reconcile_ignores_draining_without_notify(tmp_path: Path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    dispatches: list[tuple[str, str]] = []
    state_path, hb_path = _seed(tmp_path, snapshot="2026-10-05", open_keys={})
    result = run_audit_reconcile(
        now=_now(),
        failed_units_fn=lambda: [],
        stale_tokens_fn=lambda _s: [],
        backup_issues_fn=lambda _n: ["offsite_backup:draining"],
        outbox_fn=lambda: 0,
        dispatch_fn=lambda s, b: dispatches.append((s, b)) or {"mail": True},
        state_path=state_path,
        heartbeat_path=hb_path,
    )
    assert result.action == "NOOP"
    assert result.opened == () and result.resolved == ()
    assert dispatches == []


def test_reconcile_resolves_deferred_into_draining(tmp_path: Path) -> None:
    from src.tools.audit_reconcile import run_audit_reconcile

    dispatches: list[tuple[str, str]] = []
    state_path, hb_path = _seed(
        tmp_path,
        snapshot="2026-10-05",
        open_keys={"offsite_backup:deferred": {"transient": True, "text": "백업 이상: offsite_backup:deferred"}},
    )
    result = run_audit_reconcile(
        now=_now(),
        failed_units_fn=lambda: [],
        stale_tokens_fn=lambda _s: [],
        backup_issues_fn=lambda _n: ["offsite_backup:draining"],
        outbox_fn=lambda: 0,
        dispatch_fn=lambda s, b: dispatches.append((s, b)) or {"mail": True},
        state_path=state_path,
        heartbeat_path=hb_path,
    )
    assert result.action == "RESOLVED"
    assert result.resolved == ("offsite_backup:deferred",)
    assert result.still_open == ()
    assert len(dispatches) == 1
    assert "offsite_backup:deferred -> draining" in dispatches[0][0]
