"""Bounded auto-remediation for known idempotent ops conditions.

The host fixes itself (restart the unit, re-warm the token, re-run the sweep)
for allowlisted non-trading conditions; the human is notified only when the
fix did not hold. Wiring into the reconcile loop is Part 5.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path

from src import settings
from src.data.capture_contracts import SEOUL
from src.tools.alerts import redact_for_egress
from src.tools.deploy_window import blackout_remaining
from src.tools.run_outcome import RUN_EVENT_REASON_MAX_CHARS

logger = logging.getLogger(__name__)

REMEDIATION_LEDGER_RELPATH: str = "logs/remediation/ledger.jsonl"
REMEDIATION_MIN_SPACING: timedelta = timedelta(minutes=30)
REMEDIATION_NOTIFY_GRACE: timedelta = timedelta(minutes=45)
REMEDIATION_SYSTEMCTL_TIMEOUT_SEC: int = 30

_UNIT_NAME_RE = re.compile(r"^kca-[a-z0-9-]+\.service$")


class RemediationAction(StrEnum):
    """Action the host may take by itself."""

    RERUN_UNIT = "rerun_unit"


class RemediationOutcome(StrEnum):
    """Ledger outcome of one remediation record."""

    STARTED = "started"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    SKIPPED_WINDOW = "skipped_window"
    SKIPPED_BUDGET = "skipped_budget"
    SKIPPED_BUSY = "skipped_busy"
    SKIPPED_FORBIDDEN = "skipped_forbidden"
    FAILED = "failed"


@dataclass(frozen=True)
class RemediationRule:
    """Prefix-routed remediation policy for one issue class."""

    key_prefix: str
    action: RemediationAction
    target_unit: str | None
    max_attempts_per_day: int


@dataclass(frozen=True)
class RemediationRecord:
    """One ledger row describing a planned, skipped, or settled attempt."""

    ts: datetime
    issue_key: str
    action: RemediationAction
    unit: str
    attempt: int
    outcome: RemediationOutcome
    detail: str


# Idempotent or lock-protected and time-insensitive; a late re-run is safe.
SAFE_RERUN_UNITS: frozenset[str] = frozenset(
    {
        "kca-tape-sweep.service",
        "kca-price-ingest.service",
        "kca-altdata-capture.service",
        "kca-archive-intraday.service",
        "kca-archive-intraday-regular.service",
        "kca-extended-backfill.service",
        "kca-audit-reconcile.service",
        "kca-backup.service",
        "kca-backup-prune.service",
        "kca-core-snapshot.service",
        "kca-offsite-verify.service",
        "kca-kis-token-warmup.service",
    }
)

# A re-run would act on stale market time or duplicate irreversible effects.
FORBIDDEN_UNITS: frozenset[str] = frozenset(
    {
        "kca-collect.service",
        "kca-predict.service",
        "kca-auction-open.service",
        "kca-auction-close.service",
        "kca-finalize-close.service",
        "kca-paper-entry.service",
        "kca-paper-exit.service",
        "kca-aftermarket-book.service",
        "kca-kiwoom-token-rotate.service",
        "kca-retrain.service",
        "kca-daily-audit.service",
    }
)

# Siblings sharing one Drive lock; the units serialize via `flock -w` themselves.
DRIVE_LOCK_SIBLINGS: frozenset[str] = frozenset(
    {
        "kca-backup.service",
        "kca-backup-prune.service",
        "kca-core-snapshot.service",
        "kca-offsite-verify.service",
    }
)

REMEDIATION_RULES: tuple[RemediationRule, ...] = (
    RemediationRule("failed_unit:", RemediationAction.RERUN_UNIT, None, 2),
    RemediationRule("job_late:", RemediationAction.RERUN_UNIT, None, 1),
    RemediationRule(
        "stale_kis_token:", RemediationAction.RERUN_UNIT, "kca-kis-token-warmup.service", 2
    ),
    RemediationRule(
        "intraday:tape_expiring:",
        RemediationAction.RERUN_UNIT,
        "kca-tape-sweep.service",
        1,
    ),
    RemediationRule(
        "offsite_backup:interrupted",
        RemediationAction.RERUN_UNIT,
        "kca-backup.service",
        1,
    ),
)

# SKIPPED_* rows explain a non-action; they must never mask an in-flight STARTED.
_LIFECYCLE_OUTCOMES: frozenset[RemediationOutcome] = frozenset(
    {
        RemediationOutcome.STARTED,
        RemediationOutcome.RESOLVED,
        RemediationOutcome.UNRESOLVED,
        RemediationOutcome.FAILED,
    }
)


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


def _kst_date(value: datetime) -> date:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=SEOUL)
    return aware.astimezone(SEOUL).date()


def _normalize_unit(raw: str) -> str | None:
    name = raw.strip()
    if not name:
        return None
    if ":" in name or "/" in name or " " in name:
        return None
    if not name.endswith(".service"):
        name = f"{name}.service"
    return name


def _match_rule(key: str) -> RemediationRule | None:
    for rule in REMEDIATION_RULES:
        if key.startswith(rule.key_prefix):
            return rule
    return None


def _derive_unit(rule: RemediationRule, key: str) -> str | None:
    if rule.target_unit is not None:
        return rule.target_unit
    return _normalize_unit(key[len(rule.key_prefix) :])


def _record_to_json(rec: RemediationRecord) -> str:
    return json.dumps(
        {
            "ts": rec.ts.isoformat(),
            "issue_key": rec.issue_key,
            "action": rec.action.value,
            "unit": rec.unit,
            "attempt": rec.attempt,
            "outcome": rec.outcome.value,
            "detail": rec.detail,
        },
        ensure_ascii=False,
    )


def _record_from_json(raw: object) -> RemediationRecord | None:
    if not isinstance(raw, dict):
        return None
    try:
        ts = datetime.fromisoformat(str(raw["ts"]))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=SEOUL)
        return RemediationRecord(
            ts=ts,
            issue_key=str(raw["issue_key"]),
            action=RemediationAction(str(raw["action"])),
            unit=str(raw["unit"]),
            attempt=int(raw["attempt"]),
            outcome=RemediationOutcome(str(raw["outcome"])),
            detail=str(raw.get("detail", "")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def remediation_ledger_path() -> Path:
    """Return the JSONL ledger path under DATA_DIR."""
    return Path(settings.DATA_DIR) / REMEDIATION_LEDGER_RELPATH


def read_remediation_ledger(path: Path | None = None) -> tuple[RemediationRecord, ...]:
    """Read ledger records, skipping corrupt lines with a warning.

    Returns an empty tuple when the file is absent. Corrupt lines never raise;
    I/O errors other than absence propagate to the caller.
    """
    target = Path(path) if path is not None else remediation_ledger_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    records: list[RemediationRecord] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except ValueError:
            logger.warning("[SYS] stage=auto_remediation status=CORRUPT_LEDGER path=%s", target)
            continue
        rec = _record_from_json(raw)
        if rec is None:
            logger.warning("[SYS] stage=auto_remediation status=CORRUPT_LEDGER path=%s", target)
            continue
        records.append(rec)
    return tuple(records)


def _attempts_today(
    history: Sequence[RemediationRecord], key: str, now: datetime
) -> tuple[int, datetime | None]:
    today = _kst_date(_ensure_aware(now))
    count = 0
    latest: datetime | None = None
    for rec in history:
        if rec.issue_key != key:
            continue
        if rec.outcome is RemediationOutcome.STARTED:
            if _kst_date(rec.ts) == today:
                count += 1
            if latest is None or rec.ts > latest:
                latest = rec.ts
        elif rec.outcome is RemediationOutcome.FAILED:
            if latest is None or rec.ts > latest:
                latest = rec.ts
    return count, latest


def _latest_record(
    history: Sequence[RemediationRecord], key: str, *, lifecycle_only: bool = False
) -> RemediationRecord | None:
    best: RemediationRecord | None = None
    for rec in history:
        if rec.issue_key != key:
            continue
        if lifecycle_only and rec.outcome not in _LIFECYCLE_OUTCOMES:
            continue
        if best is None or rec.ts >= best.ts:
            best = rec
    return best


def _remediable_unit(key: str) -> tuple[RemediationRule, str] | None:
    rule = _match_rule(key)
    if rule is None:
        return None
    unit = _derive_unit(rule, key)
    if unit is None or not _UNIT_NAME_RE.fullmatch(unit) or unit not in SAFE_RERUN_UNITS:
        return None
    return rule, unit


def plan_remediation(
    open_keys: Collection[str],
    *,
    now: datetime,
    history: Sequence[RemediationRecord],
    unit_busy: Callable[[str], bool],
    blackout_fn: Callable[[datetime], timedelta] = blackout_remaining,
) -> tuple[tuple[RemediationRecord, RemediationRule | None], ...]:
    """Decide, per open issue key matched by a rule, the next record: STARTED-candidate, or a SKIPPED_* record explaining why not. A rule's unit is skipped when forbidden, when `blackout_fn(now) > 0`, when `unit_busy(unit)`, when attempts today (KST date, same issue key) reached the rule budget, or when the previous attempt for the key is younger than REMEDIATION_MIN_SPACING. Keys matching no rule produce no record. Pure: no I/O besides the injected callables."""
    current = _ensure_aware(now)
    planned: list[tuple[RemediationRecord, RemediationRule | None]] = []
    for key in sorted(set(open_keys)):
        rule = _match_rule(key)
        if rule is None:
            continue
        unit = _derive_unit(rule, key)
        attempts, latest_attempt = _attempts_today(history, key, current)
        attempt_no = attempts + 1
        if unit is None or not _UNIT_NAME_RE.fullmatch(unit) or unit not in SAFE_RERUN_UNITS:
            planned.append(
                (
                    RemediationRecord(
                        ts=current,
                        issue_key=key,
                        action=rule.action,
                        unit=unit or key[len(rule.key_prefix) :].strip() or "unknown",
                        attempt=attempt_no,
                        outcome=RemediationOutcome.SKIPPED_FORBIDDEN,
                        detail=f"forbidden unit for {rule.key_prefix}",
                    ),
                    None,
                )
            )
            continue
        if latest_attempt is not None and current - latest_attempt < REMEDIATION_MIN_SPACING:
            continue
        if blackout_fn(current) > timedelta(0):
            planned.append(
                (
                    RemediationRecord(
                        ts=current,
                        issue_key=key,
                        action=rule.action,
                        unit=unit,
                        attempt=attempt_no,
                        outcome=RemediationOutcome.SKIPPED_WINDOW,
                        detail="deploy blackout in effect",
                    ),
                    rule,
                )
            )
            continue
        busy = False
        if unit in DRIVE_LOCK_SIBLINGS:
            for candidate in sorted(DRIVE_LOCK_SIBLINGS):
                if unit_busy(candidate):
                    busy = True
                    break
        elif unit_busy(unit):
            busy = True
        if busy:
            planned.append(
                (
                    RemediationRecord(
                        ts=current,
                        issue_key=key,
                        action=rule.action,
                        unit=unit,
                        attempt=attempt_no,
                        outcome=RemediationOutcome.SKIPPED_BUSY,
                        detail="unit busy or Drive sibling active",
                    ),
                    rule,
                )
            )
            continue
        if attempts >= rule.max_attempts_per_day:
            planned.append(
                (
                    RemediationRecord(
                        ts=current,
                        issue_key=key,
                        action=rule.action,
                        unit=unit,
                        attempt=attempt_no,
                        outcome=RemediationOutcome.SKIPPED_BUDGET,
                        detail="per-day attempt budget exhausted",
                    ),
                    rule,
                )
            )
            continue
        planned.append(
            (
                RemediationRecord(
                    ts=current,
                    issue_key=key,
                    action=rule.action,
                    unit=unit,
                    attempt=attempt_no,
                    outcome=RemediationOutcome.STARTED,
                    detail=f"rule={rule.key_prefix}",
                ),
                rule,
            )
        )
    return tuple(item for item in planned if not _repeats_today(item[0], history))


def _repeats_today(rec: RemediationRecord, history: Sequence[RemediationRecord]) -> bool:
    """A SKIPPED_* identical to the key's latest row today adds no information; bounds ledger growth per tick."""
    if rec.outcome in _LIFECYCLE_OUTCOMES:
        return False
    latest = _latest_record(history, rec.issue_key)
    return latest is not None and latest.outcome is rec.outcome and _kst_date(latest.ts) == _kst_date(rec.ts)


def _append_records(records: Sequence[RemediationRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(_record_to_json(rec) + "\n")


def _systemctl_step(run_fn: Callable[..., subprocess.CompletedProcess[str]], args: list[str]) -> str | None:
    """Run one `systemctl --user` verb; return a redacted bounded failure detail, or None on success."""
    verb = args[0]
    try:
        result = run_fn(
            ["systemctl", "--user", *args],
            capture_output=True,
            text=True,
            timeout=REMEDIATION_SYSTEMCTL_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{verb} {type(exc).__name__}"
    if result.returncode == 0:
        return None
    stderr = str(result.stderr or "")
    return redact_for_egress(stderr)[-RUN_EVENT_REASON_MAX_CHARS:] or f"{verb} rc={result.returncode}"


def execute_remediation(
    planned: Sequence[RemediationRecord],
    *,
    run_fn: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    path: Path | None = None,
) -> tuple[RemediationRecord, ...]:
    """Append every record to the JSONL ledger first, then for STARTED records run `systemctl --user reset-failed <unit>` followed by `systemctl --user start --no-block <unit>` and append FAILED with the stderr tail (credential-redacted via `redact_for_egress`, <= RUN_EVENT_REASON_MAX_CHARS) when either command fails. Never waits for the unit to finish. Raises OSError only when the ledger cannot be written; the unit is then NOT started."""
    target = Path(path) if path is not None else remediation_ledger_path()
    records = list(planned)
    _append_records(records, target)
    written: list[RemediationRecord] = list(records)
    for rec in records:
        if rec.outcome is not RemediationOutcome.STARTED:
            continue
        if not _UNIT_NAME_RE.fullmatch(rec.unit):
            failed = RemediationRecord(
                ts=datetime.now(SEOUL),
                issue_key=rec.issue_key,
                action=rec.action,
                unit=rec.unit,
                attempt=rec.attempt,
                outcome=RemediationOutcome.FAILED,
                detail="invalid unit name",
            )
            _append_records((failed,), target)
            written.append(failed)
            logger.warning(
                "[SYS] stage=auto_remediation unit=%s status=INVALID_UNIT key=%s",
                rec.unit,
                rec.issue_key,
            )
            continue
        failure = _systemctl_step(run_fn, ["reset-failed", rec.unit]) or _systemctl_step(
            run_fn, ["start", "--no-block", rec.unit]
        )
        if failure is not None:
            failed = RemediationRecord(
                ts=datetime.now(SEOUL),
                issue_key=rec.issue_key,
                action=rec.action,
                unit=rec.unit,
                attempt=rec.attempt,
                outcome=RemediationOutcome.FAILED,
                detail=failure,
            )
            _append_records((failed,), target)
            written.append(failed)
            logger.warning(
                "[SYS] stage=auto_remediation unit=%s status=FAILED key=%s detail=%s", rec.unit, rec.issue_key, failure
            )
            continue
        logger.info(
            "[SYS] stage=auto_remediation unit=%s status=STARTED key=%s attempt=%d",
            rec.unit,
            rec.issue_key,
            rec.attempt,
        )
    return tuple(written)


def settle_remediation(
    open_keys: Collection[str],
    *,
    now: datetime,
    history: Sequence[RemediationRecord],
    path: Path | None = None,
) -> tuple[RemediationRecord, ...]:
    """For each issue key whose latest ledger outcome is STARTED, append RESOLVED when the key is no longer open, or UNRESOLVED when it is still open at least REMEDIATION_MIN_SPACING after the start. Idempotent per start."""
    current = _ensure_aware(now)
    open_set = set(open_keys)
    settled: list[RemediationRecord] = []
    for key in sorted({rec.issue_key for rec in history}):
        latest = _latest_record(history, key, lifecycle_only=True)
        if latest is None or latest.outcome is not RemediationOutcome.STARTED:
            continue
        if key not in open_set:
            settled.append(
                RemediationRecord(
                    ts=current,
                    issue_key=key,
                    action=latest.action,
                    unit=latest.unit,
                    attempt=latest.attempt,
                    outcome=RemediationOutcome.RESOLVED,
                    detail="issue no longer open",
                )
            )
        elif current - latest.ts >= REMEDIATION_MIN_SPACING:
            settled.append(
                RemediationRecord(
                    ts=current,
                    issue_key=key,
                    action=latest.action,
                    unit=latest.unit,
                    attempt=latest.attempt,
                    outcome=RemediationOutcome.UNRESOLVED,
                    detail="issue still open after spacing",
                )
            )
    if not settled:
        return ()
    target = Path(path) if path is not None else remediation_ledger_path()
    _append_records(settled, target)
    return tuple(settled)


def should_hold_notification(
    issue_key: str,
    *,
    first_seen: datetime,
    now: datetime,
    history: Sequence[RemediationRecord],
) -> bool:
    """True while a rule exists for the key, the key's first sighting is younger than REMEDIATION_NOTIFY_GRACE, and the day's attempt budget is not exhausted or an attempt is STARTED and unsettled. After the grace or once every budgeted attempt is UNRESOLVED/FAILED/SKIPPED_FORBIDDEN, False so the existing warning path mails. Keys with no rule are never held."""
    current = _ensure_aware(now)
    seen = first_seen if first_seen.tzinfo is not None else first_seen.replace(tzinfo=SEOUL)
    remediable = _remediable_unit(issue_key)
    if remediable is None:
        return False
    rule, _unit = remediable
    if current - seen >= REMEDIATION_NOTIFY_GRACE:
        return False
    latest = _latest_record(history, issue_key, lifecycle_only=True)
    if latest is not None and latest.outcome is RemediationOutcome.STARTED:
        return True
    attempts, _ = _attempts_today(history, issue_key, current)
    return attempts < rule.max_attempts_per_day
