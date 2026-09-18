"""Per-job run outcome records appended to a daily JSONL event log.

DEGRADED and NO_DECISION outcomes are alerted even when the process exits 0,
while hard failures stay with systemd OnFailure.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src import settings
from src.tools.alerts import dispatch_digest

logger = logging.getLogger(__name__)

RUN_OUTCOME_OK: str = "OK"
RUN_OUTCOME_DEGRADED: str = "DEGRADED"
RUN_OUTCOME_NO_DECISION: str = "NO_DECISION"
RUN_OUTCOMES: tuple[str, ...] = (RUN_OUTCOME_OK, RUN_OUTCOME_DEGRADED, RUN_OUTCOME_NO_DECISION)
RUN_OUTCOMES_ALERTED: frozenset[str] = frozenset({RUN_OUTCOME_DEGRADED, RUN_OUTCOME_NO_DECISION})
# 알림/로그 한 줄 길이 상한(예외 메시지 전체 덤프 방지).
RUN_EVENT_REASON_MAX_CHARS: int = 300


def run_events_path(run_date: str) -> Path:
    """Return the daily JSONL event log path for a run date."""
    return Path(settings.DATA_DIR) / "logs" / "events" / run_date[:7] / f"{run_date}.jsonl"


def record_run_outcome(
    job: str,
    outcome: str,
    *,
    run_date: str,
    reason: str = "",
    metrics: Mapping[str, Any] | None = None,
    path: Path | None = None,
    alert_fn: Callable[[str, str], dict[str, bool]] | None = None,
) -> dict[str, Any]:
    """Record one per-job run outcome to the daily JSONL event log.

    Args:
        job: Job name (e.g. predict, finalize_close).
        outcome: One of RUN_OUTCOMES.
        run_date: Run date (YYYY-MM-DD).
        reason: Short reason string (truncated to RUN_EVENT_REASON_MAX_CHARS).
        metrics: Optional metrics mapping serialized with full float precision.
        path: Explicit event log path override (tests).
        alert_fn: Alert dispatcher override (tests).

    Returns:
        The recorded event dict.

    Raises:
        ValueError: When outcome is not in RUN_OUTCOMES.
    """
    if outcome not in RUN_OUTCOMES:
        raise ValueError(f"unknown run outcome {outcome!r}; expected one of {RUN_OUTCOMES}")
    record = {
        "ts": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "job": job,
        "run_date": run_date,
        "outcome": outcome,
        "reason": reason[:RUN_EVENT_REASON_MAX_CHARS],
        "metrics": dict(metrics or {}),
    }
    target = path if path is not None else run_events_path(run_date)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.warning(
            "[SYS] stage=run_outcome job=%s status=EVENT_WRITE_FAILED path=%s reason=%s",
            job,
            target,
            type(exc).__name__,
        )
    logger.log(
        logging.INFO if outcome == RUN_OUTCOME_OK else logging.WARNING,
        "[SYS] stage=run_outcome job=%s run_date=%s outcome=%s reason=%s",
        job,
        run_date,
        outcome,
        record["reason"],
    )
    if outcome in RUN_OUTCOMES_ALERTED:
        sender = alert_fn if alert_fn is not None else dispatch_digest
        metrics_text = json.dumps(record["metrics"], ensure_ascii=False, default=str)
        outcome_label = "경고(DEGRADED)" if outcome == RUN_OUTCOME_DEGRADED else "미결정(NO_DECISION)"
        body_lines = [
            f"[⚠️ 프로세스 상태 저하: {outcome_label}]",
            f"• 작업: {job} ({run_date})",
            f"• 사유: {record['reason'] or '(사유 없음)'}",
            "",
            "[상세 내역]",
            f"job={job}",
            f"run_date={run_date}",
            f"outcome={outcome}",
            f"reason={record['reason']}",
            f"metrics={metrics_text}",
            f"event_log={target}",
        ]
        sender(
            f"[KCA] {job} {outcome} {run_date}",
            "\n".join(body_lines),
        )
    return record


def load_run_outcomes(run_date: str, *, path: Path | None = None) -> dict[str, str]:
    """Load the last recorded outcome per job for a run date.

    Args:
        run_date: Run date (YYYY-MM-DD).
        path: Explicit event log path override (tests).

    Returns:
        Mapping of job name to last outcome for the run date.
    """
    target = path if path is not None else run_events_path(run_date)
    if not target.exists():
        return {}
    outcomes: dict[str, str] = {}
    n_corrupt = 0
    for line in target.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            n_corrupt += 1
            continue
        if isinstance(rec, dict) and rec.get("run_date") == run_date and rec.get("outcome") in RUN_OUTCOMES:
            outcomes[str(rec.get("job"))] = str(rec["outcome"])
    if n_corrupt:
        logger.warning(
            "[SYS] stage=run_outcome status=CORRUPT_LINES path=%s n_corrupt=%d",
            target,
            n_corrupt,
        )
    return outcomes
