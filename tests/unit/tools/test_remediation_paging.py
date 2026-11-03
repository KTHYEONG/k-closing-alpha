"""Remediation-aware paging hold: a CRITICAL page waits only while auto-remediation can still help."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.tools.auto_remediation import (
    RemediationAction,
    RemediationOutcome,
    RemediationRecord,
    execute_remediation,
    load_paging_view,
    remediation_pending,
)

KEY = "failed_unit:kca-kis-token-warmup.service"
UNIT = "kca-kis-token-warmup.service"


def _at(clock: str) -> datetime:
    return datetime.fromisoformat(f"2026-11-03T{clock}").replace(tzinfo=ZoneInfo("Asia/Seoul"))


def _rec(outcome: str, clock: str, attempt: int = 1) -> RemediationRecord:
    return RemediationRecord(
        ts=_at(clock), issue_key=KEY, action=RemediationAction.RERUN_UNIT, unit=UNIT,
        attempt=attempt, outcome=RemediationOutcome(outcome), detail="seed",
    )


def test_remediation_pending_holds_until_budget_spent_or_max_hold() -> None:
    seen = _at("07:25:00")
    now = _at("07:40:00")

    assert remediation_pending(KEY, first_seen=seen, now=now, history=())
    assert remediation_pending(KEY, first_seen=seen, now=now, history=(_rec("started", "07:35:00"),))
    spent = (
        _rec("started", "07:35:00"),
        _rec("unresolved", "08:05:00"),
        _rec("started", "08:05:00", attempt=2),
        _rec("unresolved", "08:35:00", attempt=2),
    )
    assert not remediation_pending(KEY, first_seen=seen, now=_at("08:36:00"), history=spent)
    assert not remediation_pending(KEY, first_seen=seen.replace(tzinfo=None), now=_at("09:30:00"), history=())
    assert not remediation_pending("failed_unit:kca-predict.service", first_seen=seen, now=now, history=())


def test_paging_view_auto_action_and_unreadable_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    view = load_paging_view(_at("12:00:00"), path=ledger)

    assert view.history == ()
    assert view.auto_action("failed_unit:kca-predict.service") is None
    assert "시작하지 못했습니다" in str(view.auto_action(KEY))
    assert view.pending(KEY, _at("12:00:00"), _at("12:00:00"))

    def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    execute_remediation((_rec("started", "11:00:00"),), run_fn=_run, path=ledger)
    assert load_paging_view(_at("12:00:00"), path=ledger).auto_action(KEY) == "자동 재실행 1회를 시도했지만 해결되지 않았습니다."
    assert load_paging_view(_at("12:00:00"), path=tmp_path).history == ()
