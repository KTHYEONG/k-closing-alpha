"""Cross-path incident pager: dedup, reminders, resolution and the reconcile/audit paging paths."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.tools import incident_pager
from src.tools.alert_policy import IncidentKind, SourceScope
from src.tools.incident_pager import IncidentPager, PageDecision, page_issue_keys

KST = ZoneInfo("Asia/Seoul")


def _kst(hh: int, mm: int = 0, day: int = 3) -> datetime:
    return datetime(2026, 11, day, hh, mm, tzinfo=KST)


def _pager(tmp_path: Path, sent: list[tuple[str, str]]) -> IncidentPager:
    def _send(subject: str, body: str) -> dict[str, bool]:
        sent.append((subject, body))
        return {"webhook": False, "email": True}

    return IncidentPager(ledger_path=tmp_path / "incidents.json", send_fn=_send)


def test_same_kind_pages_once_per_day_across_sources(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    pager = _pager(tmp_path, sent)

    first = pager.page(IncidentKind.PREDICT, "failed_unit:kca-predict.service", scope=SourceScope.DAY, now=_kst(15, 22))
    second = pager.page(IncidentKind.PREDICT, "run:predict", scope=SourceScope.DAY, now=_kst(15, 23))

    assert (first, second) == (PageDecision.SENT, PageDecision.DEDUPED)
    assert len(sent) == 1


def test_day_sources_expire_silently_and_open_sources_remind_after_reminder_hour(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    pager = _pager(tmp_path, sent)
    pager.page(IncidentKind.ORDER_EXIT, "run:order_exit", scope=SourceScope.DAY, now=_kst(8, 45))
    pager.page(IncidentKind.BACKUP_FAILING, "offsite_backup:failed", scope=SourceScope.OPEN, now=_kst(22, 30))

    assert pager.page(IncidentKind.BACKUP_FAILING, "offsite_backup:failed", scope=SourceScope.OPEN, now=_kst(0, 10, day=4)) is PageDecision.REMINDER_WAIT
    assert pager.page(IncidentKind.BACKUP_FAILING, "offsite_backup:failed", scope=SourceScope.OPEN, now=_kst(8, 0, day=4)) is PageDecision.SENT
    assert pager.page(IncidentKind.BACKUP_FAILING, "offsite_backup:failed", scope=SourceScope.OPEN, now=_kst(9, 0, day=4)) is PageDecision.DEDUPED

    ledger = json.loads((tmp_path / "incidents.json").read_text(encoding="utf-8"))
    assert "run:order_exit" not in ledger["sources"]
    assert IncidentKind.ORDER_EXIT.value not in ledger["pages"]
    assert pager.resolve(["run:order_exit"], now=_kst(9, 5, day=4)) == ()
    assert [s for s, _ in sent].count("[긴급][kca] 오프사이트 백업 이틀 연속 실패") == 2


def test_resolve_mails_once_only_for_paged_kind_with_no_remaining_source(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    pager = _pager(tmp_path, sent)
    pager.page(IncidentKind.KIS_TOKEN_STALE, "stale_kis_token:a", scope=SourceScope.OPEN, now=_kst(13, 30))
    pager.observe(IncidentKind.KIS_TOKEN_STALE, "stale_kis_token:b", scope=SourceScope.OPEN, now=_kst(13, 30))
    pager.observe(IncidentKind.BACKUP_FAILING, "offsite_backup:failed", scope=SourceScope.OPEN, now=_kst(13, 30))
    sent.clear()

    assert pager.resolve(["stale_kis_token:a"], now=_kst(13, 40)) == ()
    assert pager.resolve(["stale_kis_token:b", "offsite_backup:failed"], now=_kst(13, 50)) == (IncidentKind.KIS_TOKEN_STALE,)
    assert pager.resolve(["stale_kis_token:b"], now=_kst(14, 0)) == ()
    assert pager.resolve([], now=_kst(14, 0)) == ()

    assert sent == [("[해결됨][kca] 결정창 직전 KIS 토큰 누락", "해결됨: 결정창 직전 KIS 토큰 누락 (13:50 KST 정상 확인, 조치 불필요)")]


def test_resolve_absent_only_touches_measured_open_sources(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    pager = _pager(tmp_path, sent)
    pager.page(IncidentKind.KIS_TOKEN_WARMUP, "failed_unit:kca-kis-token-warmup.service", scope=SourceScope.OPEN, now=_kst(8, 10))
    pager.page(IncidentKind.KRX_DEFERRED_RECREATE, "failed_unit:krx-deferred-recreate.service", scope=SourceScope.DAY, now=_kst(8, 10))

    resolved = pager.resolve_absent(["undelivered_alerts"], measured_prefixes=("failed_unit:kca-",), now=_kst(9, 0))

    assert resolved == (IncidentKind.KIS_TOKEN_WARMUP,)
    assert pager.first_seen("failed_unit:krx-deferred-recreate.service") == _kst(8, 10)


def test_unreadable_ledger_fails_toward_paging(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    sent: list[tuple[str, str]] = []
    pager = _pager(tmp_path, sent)
    (tmp_path / "incidents.json").write_text("[1, 2]", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=incident_pager.logger.name):
        decision = pager.page(IncidentKind.PREDICT, "run:predict", scope=SourceScope.DAY, now=_kst(15, 22))
        seen = pager.observe(IncidentKind.PREDICT, "run:predict", scope=SourceScope.DAY, now=_kst(15, 22))
        resolved = pager.resolve(["run:predict"], now=_kst(15, 30))
        absent = pager.resolve_absent([], measured_prefixes=("run:",), now=_kst(15, 30))

    assert decision is PageDecision.SENT and len(sent) == 1
    assert seen == _kst(15, 22)
    assert resolved == () and absent == ()
    assert pager.first_seen("run:predict") is None
    assert "LEDGER_FAILED" in caplog.text


def test_first_seen_tolerates_corrupt_entry(tmp_path: Path) -> None:
    path = tmp_path / "incidents.json"
    path.write_text(json.dumps({"sources": {"run:x": {"kind": "predict"}, "bad": 3}, "pages": []}), encoding="utf-8")
    pager = IncidentPager(ledger_path=path, send_fn=lambda s, b: {})
    assert pager.first_seen("run:x") is None
    assert pager.first_seen("missing") is None


def test_default_pager_uses_data_dir_and_dispatch_page(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.tools import alerts

    sent: list[str] = []
    monkeypatch.setattr(alerts, "dispatch_page", lambda s, b: sent.append(s) or {"email": True})
    pager = IncidentPager()

    pager.page(IncidentKind.COLLECT, "run:collect", scope=SourceScope.DAY, now=_kst(15, 25))

    assert sent == ["[긴급][kca] 결정 데이터 수집 실패"]
    assert incident_pager.incident_ledger_path().exists()


def test_record_silent_survives_unwritable_root(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("x", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger=incident_pager.logger.name):
        incident_pager.record_silent("failed_unit:x.service", reason="r", now=_kst(9), root=blocked)
    assert "SILENT_RECORD_FAILED" in caplog.text


def test_page_issue_keys_gates_silent_event_due_and_held_keys(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    pager = _pager(tmp_path, sent)
    keys = [
        "failed_unit:kca-predict.service",
        "failed_unit:kca-price-ingest.service",
        "offsite_backup:failed",
        "stale_kis_token:slot1",
        "failed_unit:kca-kis-token-warmup.service",
        "undelivered_alerts",
    ]

    statuses = page_issue_keys(pager, keys, now=_kst(13, 30), pending=lambda k, f, n: k.startswith("failed_unit:"))

    assert statuses == {
        "failed_unit:kca-kis-token-warmup.service": "HELD",
        "offsite_backup:failed": "NOT_DUE",
        "stale_kis_token:slot1": "SENT",
    }
    assert [s for s, _ in sent] == ["[긴급][kca] 결정창 직전 KIS 토큰 누락"]

    later = page_issue_keys(
        pager, keys, now=_kst(13, 30) + timedelta(hours=24), auto_action=lambda k: f"action {k}"
    )
    assert later["offsite_backup:failed"] == "SENT"
    assert later["failed_unit:kca-kis-token-warmup.service"] == "SENT"
    assert "자동 조치: action offsite_backup:failed" in sent[-1][1] or "자동 조치: action offsite_backup:failed" in sent[-2][1]


def test_page_reconcile_incidents_pages_and_resolves_only_when_measured(tmp_path: Path) -> None:
    from src.tools.audit_reconcile import ReconcileResult, page_reconcile_incidents

    sent: list[tuple[str, str]] = []
    pager = _pager(tmp_path, sent)

    assert page_reconcile_incidents(
        ReconcileResult(action="NOOP", measured=False), now=_kst(13, 30), pager=pager
    ) == {}
    opened = ReconcileResult(action="OPENED", still_open=("stale_kis_token:slot1", "job_late:kca-tape-sweep.service"))
    assert page_reconcile_incidents(opened, now=_kst(13, 30), pager=pager) == {"stale_kis_token:slot1": "SENT"}
    healed = ReconcileResult(action="RESOLVED", still_open=("job_late:kca-tape-sweep.service",))
    page_reconcile_incidents(healed, now=_kst(13, 40), pager=pager)

    assert [s for s, _ in sent] == ["[긴급][kca] 결정창 직전 KIS 토큰 누락", "[해결됨][kca] 결정창 직전 KIS 토큰 누락"]


def test_page_audit_issues_reads_remediation_ledger(tmp_path: Path) -> None:
    from src.tools.daily_audit import page_audit_issues

    sent: list[tuple[str, str]] = []
    statuses = page_audit_issues(
        ["failed_unit:kca-kis-token-warmup.service", "missing:predict"], _kst(21, 20), pager=_pager(tmp_path, sent)
    )

    assert statuses == {"failed_unit:kca-kis-token-warmup.service": "HELD"}
    assert sent == []


def test_record_silent_old_call_shape_unchanged(tmp_path: Path) -> None:
    incident_pager.record_silent("failed_unit:x.service", reason="r", now=_kst(9), root=tmp_path)

    rows = [json.loads(line) for line in (tmp_path / "2026-11-03.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert list(rows[0]) == ["ts", "status", "source", "reason"]


def test_record_silent_writes_location_and_context_when_given(tmp_path: Path) -> None:
    incident_pager.record_silent("failed_unit:x.service", reason="r", now=_kst(9), root=tmp_path, context="tail line")

    row = json.loads((tmp_path / "2026-11-03.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "location" not in row
    assert row["context"] == "tail line"

    incident_pager.record_silent(
        "failed_unit:x.service", reason="r", now=_kst(9), root=tmp_path, location="src/a.py:3 in f", context="c"
    )
    row = json.loads((tmp_path / "2026-11-03.jsonl").read_text(encoding="utf-8").splitlines()[1])
    assert row["location"] == "src/a.py:3 in f"
    assert row["context"] == "c"
