"""Paging policy: CRITICAL vs SILENT classification, timing gates and message format."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from src.tools import alert_policy as policy
from src.tools.alert_policy import IncidentKind, SourceScope

KST = ZoneInfo("Asia/Seoul")


def _kst(hh: int, mm: int = 0, day: int = 3) -> datetime:
    return datetime(2026, 11, day, hh, mm, tzinfo=KST)


@pytest.mark.parametrize(
    ("unit", "kind"),
    [
        ("kca-order-entry.service", IncidentKind.ORDER_ENTRY),
        ("kca-order-exit.service", IncidentKind.ORDER_EXIT),
        ("kca-order-reconcile-entry.service", IncidentKind.ORDER_RECONCILE_ENTRY),
        ("kca-order-reconcile-exit.service", IncidentKind.ORDER_RECONCILE_EXIT),
        ("kca-predict.service", IncidentKind.PREDICT),
        ("kca-collect.service", IncidentKind.COLLECT),
        ("kca-finalize-close.service", IncidentKind.FINALIZE_CLOSE),
        ("kca-kis-token-warmup.service", IncidentKind.KIS_TOKEN_WARMUP),
        ("krx-deferred-recreate.service", IncidentKind.KRX_DEFERRED_RECREATE),
        ("kse-alert@kse-morning.service.service", IncidentKind.KSE_PAGE_UNDELIVERED),
        ("kse-alert@kse-evening.service", IncidentKind.KSE_PAGE_UNDELIVERED),
        ("kca-predict", IncidentKind.PREDICT),
    ],
)
def test_critical_units(unit: str, kind: IncidentKind) -> None:
    assert policy.classify_unit(unit) is kind
    assert policy.alert_class_of_unit(unit) is policy.AlertClass.CRITICAL


@pytest.mark.parametrize(
    "unit",
    [
        "kca-price-ingest.service",
        "kca-auction-close.service",
        "kca-auction-open.service",
        "kca-auction-path.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-aftermarket-book.service",
        "kca-altdata-capture.service",
        "kca-archive-intraday.service",
        "kca-extended-backfill.service",
        "kca-tape-sweep.service",
        "kca-backup.service",
        "kca-retrain.service",
        "kca-daily-audit.service",
        "kca-audit-reconcile.service",
        "kca-api-notice-watch.service",
        "kca-kiwoom-token-rotate.service",
        "krx-host-backup.service",
        "vps-image-gc.service",
        "crypto-pilot-backup.service",
    ],
)
def test_silent_units(unit: str) -> None:
    assert policy.classify_unit(unit) is None
    assert policy.alert_class_of_unit(unit) is policy.AlertClass.SILENT


@pytest.mark.parametrize(
    ("job", "outcome", "reason", "kind"),
    [
        ("order_entry", "DEGRADED", "entry_window_expired", IncidentKind.ORDER_ENTRY),
        ("order_entry", "DEGRADED", "orderable_cash_unreadable", IncidentKind.ORDER_ENTRY),
        ("order_exit", "DEGRADED", "positions_unreadable", IncidentKind.ORDER_EXIT),
        ("order_exit", "DEGRADED", "shifted_session_hold", IncidentKind.ORDER_EXIT),
        ("order_reconcile_entry", "DEGRADED", "unknown_exposure", IncidentKind.ORDER_RECONCILE_ENTRY),
        ("order_reconcile_entry", "DEGRADED", "partial_or_unfilled", None),
        ("order_reconcile_exit", "DEGRADED", "partial_or_unfilled", IncidentKind.ORDER_RECONCILE_EXIT),
        ("predict", "NO_DECISION", "calendar_disagreement", IncidentKind.PREDICT),
        ("collect", "NO_DECISION", "calendar_disagreement", IncidentKind.COLLECT),
        ("finalize_close", "DEGRADED", "picks_unconfirmed", IncidentKind.FINALIZE_CLOSE),
        ("finalize_close", "DEGRADED", "zero_confirmed", IncidentKind.FINALIZE_CLOSE),
        ("finalize_close", "DEGRADED", "empty_archive", IncidentKind.FINALIZE_CLOSE),
        ("finalize_close", "DEGRADED", "calendar_unavailable", None),
        ("price_ingest", "DEGRADED", "stale_panel", IncidentKind.PRICE_PANEL_STALE),
        ("price_ingest", "DEGRADED", "flow_coverage_below_min", None),
        ("price_ingest", "OK", "stale_panel", None),
        ("predict", "OK", "", None),
        ("altdata_capture", "DEGRADED", "partial", None),
        ("extended_backfill", "DEGRADED", "failed", None),
        ("order_cancel", "DEGRADED", "x", None),
    ],
)
def test_run_outcome_classification(job: str, outcome: str, reason: str, kind: IncidentKind | None) -> None:
    assert policy.classify_run_outcome(job, outcome, reason) is kind


def test_order_job_names_match_order_session() -> None:
    from src.daily import order_session

    jobs = {
        order_session.ORDER_JOB_ENTRY,
        order_session.ORDER_JOB_EXIT,
        order_session.ORDER_JOB_RECONCILE_ENTRY,
        order_session.ORDER_JOB_RECONCILE_EXIT,
    }
    assert jobs <= set(policy.CRITICAL_RUN_OUTCOMES)


@pytest.mark.parametrize(
    ("key", "kind"),
    [
        ("failed_unit:kca-predict.service", IncidentKind.PREDICT),
        ("failed_unit:kca-price-ingest.service", None),
        ("stale_kis_token:slot1", IncidentKind.KIS_TOKEN_STALE),
        ("offsite_backup:failed", IncidentKind.BACKUP_FAILING),
        ("offsite_backup:interrupted", IncidentKind.BACKUP_FAILING),
        ("offsite_backup:deferred", None),
        ("offsite_backup:draining", None),
        ("undelivered_alerts", None),
        ("job_late:kca-predict.service", None),
        ("missing:predict", None),
        ("audit:advisory_state:1:unwritable", None),
    ],
)
def test_issue_key_classification(key: str, kind: IncidentKind | None) -> None:
    assert policy.classify_issue_key(key) is kind


def test_backup_pages_only_after_persisting_a_day() -> None:
    first = _kst(22, 30, day=2)
    assert not policy.issue_page_due(IncidentKind.BACKUP_FAILING, first_seen=first, now=first + timedelta(hours=23))
    assert policy.issue_page_due(IncidentKind.BACKUP_FAILING, first_seen=first, now=first + timedelta(hours=24))


def test_stale_token_pages_only_within_lead_before_decision_window() -> None:
    def due(now: datetime) -> bool:
        return policy.issue_page_due(IncidentKind.KIS_TOKEN_STALE, first_seen=_kst(7), now=now)

    assert not due(_kst(13, 19))
    assert due(_kst(13, 20))
    assert due(_kst(15, 19))
    assert not due(_kst(15, 20))
    assert policy.issue_page_due(IncidentKind.PREDICT, first_seen=_kst(7), now=_kst(7))


def test_scope_and_text_table_cover_every_kind() -> None:
    assert set(policy.INCIDENT_TEXTS) == set(IncidentKind)
    assert policy.incident_scope(IncidentKind.BACKUP_FAILING) is SourceScope.OPEN
    assert policy.incident_scope(IncidentKind.ORDER_ENTRY) is SourceScope.DAY


@pytest.mark.parametrize("kind", list(IncidentKind))
def test_page_format_is_short_korean_three_lines(kind: IncidentKind) -> None:
    subject, body = policy.format_page(kind)

    assert subject.startswith("[긴급][kca] ")
    lines = body.splitlines()
    assert [line.split(": ", 1)[0] for line in lines] == ["무슨 일", "자동 조치", "지금 할 일"]
    assert all(line.split(": ", 1)[1] for line in lines)
    assert "/" not in body and "run_id" not in body


def test_page_format_override_and_resolution_line() -> None:
    _, body = policy.format_page(IncidentKind.PREDICT, auto_action="재실행 2회 실패")
    assert body.splitlines()[1] == "자동 조치: 재실행 2회 실패"

    subject, body = policy.format_resolved(IncidentKind.PREDICT, at=_kst(16, 5))
    assert subject == "[해결됨][kca] 오늘 매수 결정 생성 실패"
    assert body.splitlines() == ["해결됨: 오늘 매수 결정 생성 실패 (16:05 KST 정상 확인, 조치 불필요)"]
