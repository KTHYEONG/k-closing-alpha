"""Re-measure self-healing audit conditions between full weekday audits.

The full audit is a once-per-weekday snapshot; between audits a warning can
heal (backup finished, unit restarted) or appear (backup interrupted). This
reconciler re-measures only the transient classes and publishes the state
change, never the full audit.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.data.capture_contracts import SEOUL
from src.tools.alerts import alert_outbox_dir, dispatch_digest
from src.tools.daily_audit import (
    SYSTEMCTL_TIMEOUT_SEC,
    _default_backup_issues,
    is_transient_issue_key,
    list_failed_kca_units,
    list_stale_kis_tokens,
    load_audit_alert_state,
    read_audit_heartbeat,
    write_audit_alert_state,
    write_audit_heartbeat,
)
from src.utils.cli_logging import configure_cli_logging

logger = logging.getLogger(__name__)

REMINDER_AFTER: timedelta = timedelta(hours=24)

ACTION_NOOP: str = "NOOP"
ACTION_RESOLVED: str = "RESOLVED"
ACTION_OPENED: str = "OPENED"
ACTION_CHANGED: str = "CHANGED"
ACTION_REMINDED: str = "REMINDED"


@dataclass(frozen=True)
class Remediation:
    """Injected remediation boundary for the reconcile tick."""

    plan_fn: Callable[[Collection[str], datetime, Sequence[Any]], tuple[Any, ...]]
    execute_fn: Callable[[Sequence[Any]], tuple[Any, ...]]
    settle_fn: Callable[[Collection[str], datetime, Sequence[Any]], tuple[Any, ...]]
    history_fn: Callable[[], Sequence[Any]]


@dataclass(frozen=True)
class ReconcileResult:
    """Outcome of one reconcile run."""

    action: str
    opened: tuple[str, ...] = ()
    resolved: tuple[str, ...] = ()
    still_open: tuple[str, ...] = ()
    notified: bool = False
    held: tuple[str, ...] = ()
    remediated: tuple[str, ...] = ()


_OPS_KEY_PREFIXES: tuple[str, ...] = (
    "job_late:",
    "job_overrun:",
    "timer_inactive:",
    "host_disk_low:",
    "ops_measure_unavailable:",
)

_NON_REMEDIABLE_PREFIXES: tuple[str, ...] = ("missing:", "collection:", "intraday:", "expiry:")

_AUTO_REMEDIATED_NOTES_LIMIT: int = 10


def _is_ops_key(key: str) -> bool:
    return key.startswith(_OPS_KEY_PREFIXES)


def _is_remediation_excluded(key: str) -> bool:
    if key.startswith("intraday:tape_expiring:"):
        return False
    return key.startswith(_NON_REMEDIABLE_PREFIXES)


def _all_ops_keys(keys: Sequence[str]) -> bool:
    return bool(keys) and all(_is_ops_key(key) for key in keys)


def _parse_time(value: Any, fallback: datetime) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=SEOUL)
    return parsed


def _default_outbox_count() -> int:
    box = alert_outbox_dir()
    if not box.is_dir():
        return 0
    return len(list(box.glob("*.json")))


def _measure_transient(
    *,
    now: datetime,
    snapshot_date: str,
    prev_open: Mapping[str, Any],
    failed_units_fn: Callable[[], list[str]],
    stale_tokens_fn: Callable[[str], list[str]],
    backup_issues_fn: Callable[[datetime], list[str]],
    outbox_fn: Callable[[], int],
    ops_issues_fn: Callable[[datetime], Sequence[Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str], bool]:
    """Re-measure transient classes; a failing class keeps its previous keys."""
    measured: dict[str, dict[str, Any]] = {}

    def _prev_keys(prefix: str) -> dict[str, dict[str, Any]]:
        kept: dict[str, dict[str, Any]] = {}
        for key, entry in prev_open.items():
            if isinstance(entry, dict) and (key == prefix or key.startswith(prefix)):
                kept[key] = {
                    "transient": True,
                    "text": str(entry.get("text", key)),
                }
        return kept

    try:
        units = failed_units_fn()
    except Exception as exc:
        logger.warning("[SYS] stage=audit_reconcile measure=failed_units status=KEEP_PREVIOUS reason=%s", type(exc).__name__)
        measured.update(_prev_keys("failed_unit:"))
    else:
        for unit in units:
            key = f"failed_unit:{unit}"
            measured[key] = {"transient": True, "text": f"실패 유닛: {unit}"}

    try:
        tokens = stale_tokens_fn(snapshot_date)
    except Exception as exc:
        logger.warning("[SYS] stage=audit_reconcile measure=stale_tokens status=KEEP_PREVIOUS reason=%s", type(exc).__name__)
        measured.update(_prev_keys("stale_kis_token:"))
    else:
        for token in tokens:
            key = f"stale_kis_token:{token}"
            measured[key] = {"transient": True, "text": f"KIS 토큰 누락: {token}"}

    provisional: list[str] = []
    draining = False
    try:
        backup_raw = backup_issues_fn(now)
    except Exception as exc:
        logger.warning("[SYS] stage=audit_reconcile measure=backup status=KEEP_PREVIOUS reason=%s", type(exc).__name__)
        measured.update(_prev_keys("offsite_backup:"))
    else:
        draining = "offsite_backup:draining" in backup_raw
        running = [item for item in backup_raw if item in ("offsite_backup:running",)]
        warnings = [item for item in backup_raw if item not in ("offsite_backup:running", "offsite_backup:draining")]
        if running:
            provisional = list(running)
        else:
            for raw in warnings:
                measured[raw] = {"transient": True, "text": f"백업 이상: {raw}"}

    try:
        pending = outbox_fn()
    except Exception as exc:
        logger.warning("[SYS] stage=audit_reconcile measure=outbox status=KEEP_PREVIOUS reason=%s", type(exc).__name__)
        if "undelivered_alerts" in prev_open and isinstance(prev_open["undelivered_alerts"], dict):
            measured["undelivered_alerts"] = {
                "transient": True,
                "text": str(prev_open["undelivered_alerts"].get("text", "undelivered_alerts")),
            }
    else:
        if int(pending) > 0:
            measured["undelivered_alerts"] = {
                "transient": True,
                "text": f"미전송 알림: {int(pending)}건 (outbox 적체)",
            }

    if ops_issues_fn is not None:
        try:
            ops_measured = tuple(ops_issues_fn(now))
        except Exception as exc:
            logger.warning("[SYS] stage=audit_reconcile measure=ops status=KEEP_PREVIOUS reason=%s", type(exc).__name__)
            for prefix in ("job_late:", "job_overrun:", "timer_inactive:", "host_disk_low:"):
                measured.update(_prev_keys(prefix))
        else:
            for item in ops_measured:
                key = str(item.key if hasattr(item, "key") else item[0])
                text = str(item.text if hasattr(item, "text") else item[1])
                measured[key] = {"transient": True, "text": text}

    return measured, provisional, draining


def _warning_subject(snapshot_date: str, keys: Sequence[str]) -> str:
    return f"[kca] 🚨 {snapshot_date} 일일점검 경고: {', '.join(keys)}"


def _resolution_subject(snapshot_date: str, resolved: Sequence[str]) -> str:
    return f"[kca] ✅ {snapshot_date} 점검 정상화: {', '.join(resolved)}"


def _ops_warning_subject(now: datetime, keys: Sequence[str]) -> str:
    label = now.astimezone(SEOUL).date().isoformat() if now.tzinfo is not None else str(now.date())
    return f"[kca] 🚨 {label} 운영감시 경고: {', '.join(keys)}"


def _ops_resolution_subject(now: datetime, resolved: Sequence[str]) -> str:
    label = now.astimezone(SEOUL).date().isoformat() if now.tzinfo is not None else str(now.date())
    return f"[kca] ✅ {label} 운영감시 정상화: {', '.join(resolved)}"


def _pick_warning_subject(snapshot_date: str, now: datetime, keys: Sequence[str]) -> str:
    if _all_ops_keys(keys):
        return _ops_warning_subject(now, keys)
    return _warning_subject(snapshot_date, keys)


def _pick_resolution_subject(snapshot_date: str, now: datetime, resolved: Sequence[str]) -> str:
    if _all_ops_keys(resolved):
        return _ops_resolution_subject(now, resolved)
    return _resolution_subject(snapshot_date, resolved)


def run_audit_reconcile(
    *,
    now: datetime,
    failed_units_fn: Callable[[], list[str]],
    stale_tokens_fn: Callable[[str], list[str]],
    backup_issues_fn: Callable[[datetime], list[str]],
    outbox_fn: Callable[[], int],
    dispatch_fn: Callable[[str, str], dict[str, bool]],
    reminder_after: timedelta = REMINDER_AFTER,
    state_path: Path | None = None,
    heartbeat_path: Path | None = None,
    dry_run: bool = False,
    ops_issues_fn: Callable[[datetime], Sequence[Any]] | None = None,
    remediation: Remediation | None = None,
) -> ReconcileResult:
    """Re-measure the self-healing audit conditions and publish the state change, never the full audit.

    Why: the full audit is a once-per-weekday snapshot; between audits a warning can heal (backup finished, unit
    restarted) or appear (backup interrupted). Without a re-measurement the heartbeat that the dashboard shows stays
    frozen at the last audit's verdict.

    Returns:
        ReconcileResult(action, opened, resolved, still_open, notified, held, remediated).

    Raises:
        OSError: The state or heartbeat cannot be persisted (the unit fails so OnFailure fires).
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    state = load_audit_alert_state(state_path)
    heartbeat = read_audit_heartbeat(heartbeat_path)
    if state is None and heartbeat is None:
        return ReconcileResult(action=ACTION_NOOP)

    prev_open: dict[str, Any] = {}
    if isinstance(state, dict):
        raw_open = state.get("open", {})
        if isinstance(raw_open, dict):
            prev_open = dict(raw_open)
        prev_open.update(state.get("pending_resolutions", {}))
    elif heartbeat is not None:
        if "open_issues" not in heartbeat and "경고" in str(heartbeat.get("subject", "")):
            raise OSError("Cannot recover legacy warning without audit alert state")
        for issue in heartbeat.get("open_issues", []):
            prev_open[issue["key"]] = {
                "transient": issue["transient"], "text": issue["text"],
                "first_seen": heartbeat["finished_at"], "last_notified": heartbeat["finished_at"],
            }
    snapshot_date = str((state or {}).get("snapshot_date") or (heartbeat or {}).get("snapshot_date") or "")
    if not snapshot_date:
        return ReconcileResult(action=ACTION_NOOP)

    persistent: dict[str, dict[str, Any]] = {}
    for key, entry in prev_open.items():
        transient = bool(entry.get("transient", is_transient_issue_key(key)))
        if not transient:
            persistent[key] = {"transient": False, "text": str(entry.get("text", key))}

    measured, provisional, draining = _measure_transient(
        now=now,
        snapshot_date=snapshot_date,
        prev_open=prev_open,
        failed_units_fn=failed_units_fn,
        stale_tokens_fn=stale_tokens_fn,
        backup_issues_fn=backup_issues_fn,
        outbox_fn=outbox_fn,
        ops_issues_fn=None if dry_run or remediation is None else ops_issues_fn,
    )
    new_open: dict[str, dict[str, Any]] = {**persistent, **measured}
    prev_keys = set(prev_open)
    new_keys = set(new_open)
    opened = tuple(sorted(new_keys - prev_keys))
    pending_resolutions = {
        key: entry for key, entry in prev_open.items()
        if provisional and key.startswith("offsite_backup:")
    }
    resolved = tuple(sorted(prev_keys - new_keys - set(pending_resolutions)))
    still_open = tuple(sorted(new_keys))

    from src.tools.auto_remediation import should_hold_notification as _should_hold

    def _first_seen_at(key: str) -> datetime:
        prev = prev_open.get(key)
        if isinstance(prev, dict) and prev.get("first_seen"):
            return _parse_time(prev.get("first_seen"), now)
        return now

    remediation_enabled = remediation is not None
    history: Sequence[Any] = ()
    remediation_failed = False
    settled: tuple[Any, ...] = ()
    started_keys: tuple[str, ...] = ()
    if remediation is not None:
        try:
            history = tuple(remediation.history_fn())
        except Exception as exc:
            logger.warning("[SYS] stage=audit_reconcile remediation=FAILED reason=%s", type(exc).__name__)
            remediation_failed = True
            history = ()
        else:
            if not dry_run:
                try:
                    open_transient = sorted(
                        key for key in new_keys
                        if bool(new_open[key].get("transient", is_transient_issue_key(key)))
                        and not _is_remediation_excluded(key)
                    )
                    settled = tuple(remediation.settle_fn(open_transient, now, history))
                    combined = (*history, *settled)
                    planned = tuple(remediation.plan_fn(open_transient, now, combined))
                    executed: tuple[Any, ...] = tuple(remediation.execute_fn(planned)) if planned else ()
                    failed_keys = {str(rec.issue_key) for rec in executed if str(rec.outcome) == "failed"}
                    started_keys = tuple(
                        sorted(
                            {
                                str(rec.issue_key)
                                for rec in executed
                                if str(rec.outcome) == "started" and str(rec.issue_key) not in failed_keys
                            }
                        )
                    )
                    history = (*combined, *executed)
                except Exception as exc:
                    logger.warning("[SYS] stage=audit_reconcile remediation=FAILED reason=%s", type(exc).__name__)
                    remediation_failed = True
                    history = ()
                    settled = ()
                    started_keys = ()

    def _held(key: str) -> bool:
        if remediation is None or remediation_failed or _is_remediation_excluded(key):
            return False
        try:
            return bool(_should_hold(key, first_seen=_first_seen_at(key), now=now, history=history))
        except Exception as exc:
            logger.warning("[SYS] stage=audit_reconcile remediation=FAILED reason=%s", type(exc).__name__)
            return False

    held_set = {key for key in opened if _held(key)}
    held = tuple(sorted(held_set))
    released = tuple(
        sorted(
            key for key in new_keys
            if key in prev_keys
            and isinstance(prev_open.get(key), dict)
            and "last_notified" not in prev_open[key]
            and key not in opened
            and not _held(key)
            and not _is_remediation_excluded(key)
        )
    )
    notify_opened = tuple(sorted((set(opened) - held_set) | set(released)))
    silent_resolved = tuple(
        sorted(
            key for key in resolved
            if isinstance(prev_open.get(key), dict) and "last_notified" not in prev_open[key]
        )
    )
    announced_resolved = tuple(sorted(set(resolved) - set(silent_resolved)))
    auto_notes: list[str] = []
    if silent_resolved and remediation_enabled and not remediation_failed:
        by_key_unit: dict[str, str] = {}
        for rec in (*history, *settled):
            key = str(getattr(rec, "issue_key", ""))
            if key in silent_resolved:
                by_key_unit[key] = str(getattr(rec, "unit", "unknown"))
        auto_notes.extend(f"auto_remediated={key}:{by_key_unit.get(key, 'unknown')}" for key in silent_resolved)

    prev_severity = str((heartbeat or {}).get("severity", ""))
    prev_subject = (heartbeat or {}).get("subject")

    def _refresh_heartbeat(subject: str, severity: str, outbox_pending: int | None, extra_notes: Sequence[str] = ()) -> None:
        prev_hb = heartbeat or {}
        day_kind = str(prev_hb.get("day_kind", "trading"))
        finished_raw = prev_hb.get("finished_at")
        try:
            finished_at = datetime.fromisoformat(str(finished_raw)) if finished_raw else now
        except ValueError:
            finished_at = now
        if finished_at.tzinfo is None:
            finished_at = finished_at.replace(tzinfo=SEOUL)
        if outbox_pending is None:
            try:
                undelivered = int(prev_hb.get("undelivered_alerts", 0))
            except (TypeError, ValueError):
                undelivered = 0
        else:
            undelivered = int(outbox_pending)
        open_issues = [
            {"key": key, "transient": new_open[key]["transient"], "text": new_open[key]["text"]}
            for key in sorted(new_keys)
        ]
        prev_notes = prev_hb.get("info_notes", [])
        info_notes = list(prev_notes) if isinstance(prev_notes, list) else []
        if extra_notes:
            seen = set(info_notes)
            for note in extra_notes:
                if note not in seen:
                    info_notes.append(note)
                    seen.add(note)
            auto = [note for note in info_notes if note.startswith("auto_remediated=")]
            others = [note for note in info_notes if not note.startswith("auto_remediated=")]
            info_notes = [*others, *auto[-_AUTO_REMEDIATED_NOTES_LIMIT:]]
        write_audit_heartbeat(
            snapshot_date,
            day_kind=day_kind,
            subject=subject,
            undelivered_alerts=undelivered,
            finished_at=finished_at,
            path=heartbeat_path,
            severity=severity,
            open_issues=open_issues,
            provisional_reasons=provisional,
            audit_kind="reconcile",
            reconciled_at=now,
            info_notes=info_notes,
        )

    def _persist_state(notified_keys: set[str], reminder: bool, held_keys: set[str] | None = None) -> None:
        entries: dict[str, dict[str, Any]] = {}
        stamp = now.isoformat()
        held_now = held_keys or set()
        for key in sorted(new_keys):
            prev = prev_open.get(key)
            first_seen = str(prev.get("first_seen", stamp)) if isinstance(prev, dict) else stamp
            if key in notified_keys or reminder:
                last_notified = stamp
            elif key in held_now:
                entry: dict[str, Any] = {
                    "transient": bool(new_open[key]["transient"]),
                    "text": str(new_open[key]["text"]),
                    "first_seen": first_seen,
                }
                entries[key] = entry
                continue
            elif isinstance(prev, dict) and "last_notified" in prev:
                last_notified = str(prev["last_notified"])
            else:
                last_notified = stamp
            entries[key] = {
                "transient": bool(new_open[key]["transient"]),
                "text": str(new_open[key]["text"]),
                "first_seen": first_seen,
                "last_notified": last_notified,
            }
        write_audit_alert_state(
            snapshot_date=snapshot_date, updated_at=now, open_entries=entries, path=state_path,
            pending_resolutions=pending_resolutions,
        )

    def _current_outbox() -> int | None:
        try:
            return int(outbox_fn())
        except Exception:
            return None

    still_held = {
        key for key in new_keys
        if key in prev_keys and isinstance(prev_open.get(key), dict) and "last_notified" not in prev_open[key] and _held(key)
    }
    persist_held = set(held_set) | still_held

    if not notify_opened and not announced_resolved:
        if not dry_run and (state is None or pending_resolutions):
            _persist_state(set(), reminder=False, held_keys=persist_held)
        if not new_keys:
            if dry_run:
                return ReconcileResult(action=ACTION_NOOP if not silent_resolved else ACTION_RESOLVED, resolved=resolved, held=held, remediated=started_keys)
            if silent_resolved:
                _persist_state(set(), reminder=False)
                if heartbeat is not None:
                    _refresh_heartbeat(
                        _ops_resolution_subject(now, list(resolved)) if _all_ops_keys(resolved) else _resolution_subject(snapshot_date, list(resolved)),
                        "OK", _current_outbox(), auto_notes,
                    )
                return ReconcileResult(action=ACTION_RESOLVED, resolved=resolved, notified=False, held=held, remediated=started_keys)
            if heartbeat is not None:
                severity = prev_severity if prev_severity == "HOLIDAY_SKIP" else "OK"
                subject = (
                    str(prev_subject) if severity == "HOLIDAY_SKIP" and "경고" not in str(prev_subject)
                    else _pick_resolution_subject(snapshot_date, now, list(resolved))
                )
                _refresh_heartbeat(subject, severity, _current_outbox(), auto_notes)
            return ReconcileResult(action=ACTION_NOOP, held=held, remediated=started_keys)
        overdue = False
        for key in sorted(new_keys):
            entry = prev_open.get(key, {})
            last = entry.get("last_notified") if isinstance(entry, dict) else None
            if now - _parse_time(last, now) > reminder_after:
                overdue = True
                break
        if opened and not overdue:
            if dry_run:
                return ReconcileResult(action=ACTION_OPENED, opened=opened, still_open=still_open, notified=False, held=held, remediated=started_keys)
            _persist_state(set(), reminder=False, held_keys=persist_held)
            if heartbeat is not None:
                _refresh_heartbeat(_pick_warning_subject(snapshot_date, now, list(opened)), "WARNING", _current_outbox(), auto_notes)
            return ReconcileResult(action=ACTION_OPENED, opened=opened, still_open=still_open, notified=False, held=held, remediated=started_keys)
        if not overdue:
            if not dry_run and heartbeat is not None:
                _refresh_heartbeat(_pick_warning_subject(snapshot_date, now, sorted(new_keys)), "WARNING", _current_outbox(), auto_notes)
                _persist_state(set(), reminder=False, held_keys=persist_held)
            elif dry_run:
                return ReconcileResult(action=ACTION_NOOP, still_open=still_open, held=held, remediated=started_keys)
            return ReconcileResult(action=ACTION_NOOP, still_open=still_open, held=held, remediated=started_keys)
        if _all_ops_keys(sorted(new_keys)):
            remind_label = now.astimezone(SEOUL).date().isoformat() if now.tzinfo is not None else str(now.date())
            subject = f"[kca] 🚨 {remind_label} 운영감시 경고(재알림): {', '.join(sorted(new_keys))}"
        else:
            subject = f"[kca] 🚨 {snapshot_date} 일일점검 경고(재알림): {', '.join(sorted(new_keys))}"
        lines = [
            "==================================================",
            f"🔔 K-Closing Alpha 점검 재알림 ({snapshot_date})",
            "==================================================",
            *[f"• {new_open[key]['text']}" for key in sorted(new_keys)],
            "• 조치 안내: or-vps 서버 상태 점검 요망",
        ]
        body = "\n".join(lines)
        if dry_run:
            return ReconcileResult(action=ACTION_REMINDED, still_open=still_open, notified=False, held=held, remediated=started_keys)
        results = dispatch_fn(subject, body)
        if not any(results.values()):
            return ReconcileResult(action=ACTION_REMINDED, still_open=still_open, notified=False, held=held, remediated=started_keys)
        _persist_state(set(), reminder=True)
        _refresh_heartbeat(subject, "WARNING", _current_outbox(), auto_notes)
        return ReconcileResult(action=ACTION_REMINDED, still_open=still_open, notified=True, held=held, remediated=started_keys)

    def _display_resolved(keys: Sequence[str]) -> list[str]:
        labels = list(keys)
        if draining and "offsite_backup:deferred" in labels:
            labels = ["offsite_backup:deferred -> draining" if key == "offsite_backup:deferred" else key for key in labels]
        return labels

    if announced_resolved and not new_keys:
        display = _display_resolved(list(announced_resolved))
        subject = _ops_resolution_subject(now, display) if _all_ops_keys(announced_resolved) else _resolution_subject(snapshot_date, display)
        lines = [
            "==================================================",
            f"✅ K-Closing Alpha 점검 정상화 ({snapshot_date})",
            "==================================================",
            f"• 해소: {', '.join(display)}",
            "• 상태: 🟢 전 항목 정상 (재확인 완료)",
        ]
        body = "\n".join(lines)
        if dry_run:
            return ReconcileResult(action=ACTION_RESOLVED, resolved=resolved, notified=False, held=held, remediated=started_keys)
        results = dispatch_fn(subject, body)
        if not any(results.values()):
            return ReconcileResult(action=ACTION_RESOLVED, resolved=resolved, still_open=(), notified=False, held=held, remediated=started_keys)
        _persist_state(set(), reminder=False)
        _refresh_heartbeat(subject, "OK", _current_outbox(), auto_notes)
        return ReconcileResult(action=ACTION_RESOLVED, resolved=resolved, notified=True, held=held, remediated=started_keys)

    if notify_opened and not announced_resolved:
        subject = _pick_warning_subject(snapshot_date, now, list(notify_opened))
        lines = [
            "==================================================",
            f"🚨 K-Closing Alpha 장애/누락 알림 ({snapshot_date})",
            "==================================================",
            *[f"• {new_open[key]['text']}" for key in notify_opened],
            "• 조치 안내: or-vps 서버 상태 점검 요망",
        ]
        body = "\n".join(lines)
        if dry_run:
            return ReconcileResult(action=ACTION_OPENED, opened=opened, still_open=still_open, notified=False, held=held, remediated=started_keys)
        results = dispatch_fn(subject, body)
        if not any(results.values()):
            return ReconcileResult(action=ACTION_OPENED, opened=opened, resolved=resolved, still_open=still_open, notified=False, held=held, remediated=started_keys)
        _persist_state(set(notify_opened), reminder=False, held_keys=persist_held)
        _refresh_heartbeat(subject, "WARNING", _current_outbox(), auto_notes)
        return ReconcileResult(action=ACTION_OPENED, opened=opened, still_open=still_open, notified=True, held=held, remediated=started_keys)

    display_changed = _display_resolved(list(announced_resolved))
    changed_union = [*announced_resolved, *notify_opened, *sorted(new_keys)]
    if _all_ops_keys(changed_union):
        changed_label = now.astimezone(SEOUL).date().isoformat() if now.tzinfo is not None else str(now.date())
        subject = (
            f"[kca] 🚨 {changed_label} 운영감시 경고: 정상화 {', '.join(display_changed)}"
            + (f" / 신규 {', '.join(notify_opened)}" if notify_opened else "")
            + f" / 잔여 {', '.join(sorted(new_keys))}"
        )
    else:
        subject = (
            f"[kca] 🚨 {snapshot_date} 일일점검 경고: 정상화 {', '.join(display_changed)}"
            + (f" / 신규 {', '.join(notify_opened)}" if notify_opened else "")
            + f" / 잔여 {', '.join(sorted(new_keys))}"
        )
    lines = [
        "==================================================",
        f"🚨 K-Closing Alpha 점검 변동 알림 ({snapshot_date})",
        "==================================================",
        f"• 해소: {', '.join(display_changed)}",
        *[f"• 신규: {new_open[key]['text']}" for key in notify_opened],
        f"• 잔여: {', '.join(sorted(new_keys))}",
        "• 조치 안내: or-vps 서버 상태 점검 요망",
    ]
    body = "\n".join(lines)
    if dry_run:
        return ReconcileResult(
            action=ACTION_CHANGED, opened=opened, resolved=resolved, still_open=still_open, notified=False, held=held, remediated=started_keys
        )
    results = dispatch_fn(subject, body)
    if not any(results.values()):
        return ReconcileResult(
            action=ACTION_CHANGED, opened=opened, resolved=resolved, still_open=still_open, notified=False, held=held, remediated=started_keys
        )
    _persist_state(set(notify_opened), reminder=False, held_keys=persist_held)
    _refresh_heartbeat(subject, "WARNING", _current_outbox(), auto_notes)
    return ReconcileResult(action=ACTION_CHANGED, opened=opened, resolved=resolved, still_open=still_open, notified=True, held=held, remediated=started_keys)


def _unit_busy(unit: str) -> bool:
    """True when the unit is active/activating, or when its state cannot be read (never restart blind)."""
    try:
        result = subprocess.run(  # noqa: S603 - argv list, unit validated by the remediation planner
            ["systemctl", "--user", "show", "--property=ActiveState", "--value", unit],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=SYSTEMCTL_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    if result.returncode != 0:
        return True
    return result.stdout.strip() in ("active", "activating", "reloading", "deactivating", "")


def main() -> int:  # pragma: no cover - CLI entry; logic covered via run_audit_reconcile scenarios
    """CLI entry for the systemd reconcile unit."""
    parser = argparse.ArgumentParser(description="Re-measure self-healing audit conditions")
    parser.add_argument("--dry-run", action="store_true", help="Print the decision without writing state or heartbeat")
    args = parser.parse_args()
    now = datetime.now(SEOUL)
    tick_start = time.monotonic()
    try:
        from src.tools.auto_remediation import (
            execute_remediation,
            plan_remediation,
            read_remediation_ledger,
            settle_remediation,
        )
        from src.tools.ops_sentinel import measure_ops

        remediation = Remediation(
            plan_fn=lambda keys, at, history: tuple(
                rec for rec, _rule in plan_remediation(
                    keys, now=at, history=history, unit_busy=_unit_busy
                )
            ),
            execute_fn=lambda planned: execute_remediation(planned),
            settle_fn=lambda keys, at, history: settle_remediation(keys, now=at, history=history),
            history_fn=lambda: read_remediation_ledger(),
        )
        result = run_audit_reconcile(
            now=now,
            failed_units_fn=list_failed_kca_units,
            stale_tokens_fn=lambda snapshot: list_stale_kis_tokens(snapshot, allow_newer=True),
            backup_issues_fn=_default_backup_issues,
            outbox_fn=_default_outbox_count,
            dispatch_fn=dispatch_digest,
            dry_run=args.dry_run,
            ops_issues_fn=measure_ops,
            remediation=remediation,
        )
    except OSError as exc:
        logger.error("[SYS] stage=audit_reconcile status=PERSIST_FAILED reason=%s", exc)
        return 1
    elapsed = time.monotonic() - tick_start
    logger.info("[SYS] stage=audit_reconcile status=TICK elapsed_s=%.1f", elapsed)
    print(json.dumps({  # noqa: T201 - CLI decision output for --dry-run/operators
        "action": result.action,
        "opened": list(result.opened),
        "resolved": list(result.resolved),
        "still_open": list(result.still_open),
        "held": list(result.held),
        "remediated": list(result.remediated),
        "notified": result.notified,
        "dry_run": args.dry_run,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    raise SystemExit(main())
