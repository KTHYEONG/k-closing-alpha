"""Per-job run outcome records appended to a daily JSONL event log.

CRITICAL DEGRADED/NO_DECISION outcomes page even when the process exits 0, while hard
failures stay with systemd OnFailure; the paging policy lives in `src.tools.alert_policy`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from src import settings
from src.tools.alert_policy import RUN_SOURCE_PREFIX, classify_run_outcome, incident_scope
from src.tools.alerts import redact_for_egress
from src.tools.incident_pager import IncidentPager

logger = logging.getLogger(__name__)

RUN_OUTCOME_OK: str = "OK"
RUN_OUTCOME_DEGRADED: str = "DEGRADED"
RUN_OUTCOME_NO_DECISION: str = "NO_DECISION"
RUN_OUTCOME_SKIPPED: str = "SKIPPED"
RUN_OUTCOMES: tuple[str, ...] = (RUN_OUTCOME_OK, RUN_OUTCOME_DEGRADED, RUN_OUTCOME_NO_DECISION, RUN_OUTCOME_SKIPPED)
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
    pager: IncidentPager | None = None,
) -> dict[str, Any]:
    """Record one per-job run outcome to the daily JSONL event log.

    Args:
        job: Job name (e.g. predict, finalize_close).
        outcome: One of RUN_OUTCOMES. SKIPPED is a deliberate no-op (e.g.
            non_trading_day, session_closed); it is never alerted and never
            counts as success -- consumers needing output compare to OK.
        run_date: Run date (YYYY-MM-DD).
        reason: Short reason string; credential-masked, then truncated to
            RUN_EVENT_REASON_MAX_CHARS, before it reaches the event log, logger or digest.
            Required (non-blank) for SKIPPED so the cause stays machine-readable.
        metrics: Optional metrics mapping serialized with full float precision.
        path: Explicit event log path override (tests).
        pager: Incident pager override (tests). DEGRADED/NO_DECISION page only when
            `alert_policy.classify_run_outcome` marks them CRITICAL; OK resolves a paged
            incident of the same job; everything else stays in the event log (SILENT).

    Returns:
        The recorded event dict.

    Raises:
        ValueError: When outcome is not in RUN_OUTCOMES, or when SKIPPED
            has an empty or whitespace-only reason.
    """
    if outcome not in RUN_OUTCOMES:
        raise ValueError(f"unknown run outcome {outcome!r}; expected one of {RUN_OUTCOMES}")
    if outcome == RUN_OUTCOME_SKIPPED and not reason.strip():
        raise ValueError("SKIPPED run outcome requires a non-blank reason")
    record = {
        "ts": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
        "job": job,
        "run_date": run_date,
        "outcome": outcome,
        "reason": redact_for_egress(reason)[:RUN_EVENT_REASON_MAX_CHARS] if reason else "",
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
        logging.INFO if outcome in (RUN_OUTCOME_OK, RUN_OUTCOME_SKIPPED) else logging.WARNING,
        "[SYS] stage=run_outcome job=%s run_date=%s outcome=%s reason=%s",
        job,
        run_date,
        outcome,
        record["reason"],
    )
    if outcome in RUN_OUTCOMES_ALERTED or outcome == RUN_OUTCOME_OK:
        _route_outcome(job, outcome, str(record["reason"]), pager=pager)
    return record


def _route_outcome(job: str, outcome: str, reason: str, *, pager: IncidentPager | None) -> None:
    """Page a CRITICAL outcome, resolve the job's paged incident on OK, and leave the rest SILENT.

    The event log already holds the full record, so a routing failure only logs.
    """
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    source = f"{RUN_SOURCE_PREFIX}{job}"
    active = pager if pager is not None else IncidentPager()
    try:
        if outcome == RUN_OUTCOME_OK:
            active.resolve((source,), now=now)
            return
        kind = classify_run_outcome(job, outcome, reason)
        if kind is None:
            logger.info("[SYS] stage=alert status=SILENT job=%s outcome=%s reason=%s", job, outcome, reason)
            return
        active.page(kind, source, scope=incident_scope(kind), now=now)
    except Exception as exc:
        logger.warning("[SYS] stage=alert status=ROUTE_FAILED job=%s reason=%s", job, type(exc).__name__)


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
