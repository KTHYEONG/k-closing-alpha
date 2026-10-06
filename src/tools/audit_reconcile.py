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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.data.capture_contracts import SEOUL
from src.tools.alerts import alert_outbox_dir, dispatch_digest
from src.tools.daily_audit import (
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
class ReconcileResult:
    """Outcome of one reconcile run."""

    action: str
    opened: tuple[str, ...] = ()
    resolved: tuple[str, ...] = ()
    still_open: tuple[str, ...] = ()
    notified: bool = False


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

    return measured, provisional, draining


def _warning_subject(snapshot_date: str, keys: Sequence[str]) -> str:
    return f"[kca] 🚨 {snapshot_date} 일일점검 경고: {', '.join(keys)}"


def _resolution_subject(snapshot_date: str, resolved: Sequence[str]) -> str:
    return f"[kca] ✅ {snapshot_date} 점검 정상화: {', '.join(resolved)}"


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
) -> ReconcileResult:
    """Re-measure the self-healing audit conditions and publish the state change, never the full audit.

    Why: the full audit is a once-per-weekday snapshot; between audits a warning can heal (backup finished, unit
    restarted) or appear (backup interrupted). Without a re-measurement the heartbeat that the dashboard shows stays
    frozen at the last audit's verdict.

    Returns:
        ReconcileResult(action, opened, resolved, still_open, notified): action is NOOP, RESOLVED, OPENED, CHANGED or
        REMINDED.

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

    prev_severity = str((heartbeat or {}).get("severity", ""))
    prev_subject = (heartbeat or {}).get("subject")

    def _refresh_heartbeat(subject: str, severity: str, outbox_pending: int | None) -> None:
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

    def _persist_state(notified_keys: set[str], reminder: bool) -> None:
        entries: dict[str, dict[str, Any]] = {}
        stamp = now.isoformat()
        for key in sorted(new_keys):
            prev = prev_open.get(key)
            first_seen = str(prev.get("first_seen", stamp)) if isinstance(prev, dict) else stamp
            if key in notified_keys or reminder:
                last_notified = stamp
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

    if not opened and not resolved:
        if not dry_run and (state is None or pending_resolutions):
            _persist_state(set(), reminder=False)
        if not new_keys:
            if dry_run:
                return ReconcileResult(action=ACTION_NOOP)
            if heartbeat is not None:
                severity = prev_severity if prev_severity == "HOLIDAY_SKIP" else "OK"
                subject = (
                    str(prev_subject) if severity == "HOLIDAY_SKIP" and "경고" not in str(prev_subject)
                    else _resolution_subject(snapshot_date, ())
                )
                _refresh_heartbeat(subject, severity, _current_outbox())
            return ReconcileResult(action=ACTION_NOOP)
        overdue = False
        for key in sorted(new_keys):
            entry = prev_open.get(key, {})
            last = entry.get("last_notified") if isinstance(entry, dict) else None
            if now - _parse_time(last, now) > reminder_after:
                overdue = True
                break
        if not overdue:
            if not dry_run and heartbeat is not None:
                _refresh_heartbeat(_warning_subject(snapshot_date, sorted(new_keys)), "WARNING", _current_outbox())
            return ReconcileResult(action=ACTION_NOOP, still_open=still_open)
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
            return ReconcileResult(action=ACTION_REMINDED, still_open=still_open, notified=False)
        results = dispatch_fn(subject, body)
        if not any(results.values()):
            return ReconcileResult(action=ACTION_REMINDED, still_open=still_open, notified=False)
        _persist_state(set(), reminder=True)
        _refresh_heartbeat(subject, "WARNING", _current_outbox())
        return ReconcileResult(action=ACTION_REMINDED, still_open=still_open, notified=True)

    def _display_resolved(keys: Sequence[str]) -> list[str]:
        labels = list(keys)
        if draining and "offsite_backup:deferred" in labels:
            labels = ["offsite_backup:deferred -> draining" if key == "offsite_backup:deferred" else key for key in labels]
        return labels

    if resolved and not new_keys:
        display = _display_resolved(list(resolved))
        subject = _resolution_subject(snapshot_date, display)
        lines = [
            "==================================================",
            f"✅ K-Closing Alpha 점검 정상화 ({snapshot_date})",
            "==================================================",
            f"• 해소: {', '.join(display)}",
            "• 상태: 🟢 전 항목 정상 (재확인 완료)",
        ]
        body = "\n".join(lines)
        if dry_run:
            return ReconcileResult(action=ACTION_RESOLVED, resolved=resolved, notified=False)
        results = dispatch_fn(subject, body)
        if not any(results.values()):
            return ReconcileResult(action=ACTION_RESOLVED, resolved=resolved, still_open=(), notified=False)
        _persist_state(set(), reminder=False)
        _refresh_heartbeat(subject, "OK", _current_outbox())
        return ReconcileResult(action=ACTION_RESOLVED, resolved=resolved, notified=True)

    if opened and not resolved:
        subject = _warning_subject(snapshot_date, list(opened))
        lines = [
            "==================================================",
            f"🚨 K-Closing Alpha 장애/누락 알림 ({snapshot_date})",
            "==================================================",
            *[f"• {new_open[key]['text']}" for key in opened],
            "• 조치 안내: or-vps 서버 상태 점검 요망",
        ]
        body = "\n".join(lines)
        if dry_run:
            return ReconcileResult(action=ACTION_OPENED, opened=opened, still_open=still_open, notified=False)
        results = dispatch_fn(subject, body)
        if not any(results.values()):
            return ReconcileResult(action=ACTION_OPENED, opened=opened, resolved=resolved, still_open=still_open, notified=False)
        _persist_state(set(opened), reminder=False)
        _refresh_heartbeat(subject, "WARNING", _current_outbox())
        return ReconcileResult(action=ACTION_OPENED, opened=opened, still_open=still_open, notified=True)

    display_changed = _display_resolved(list(resolved))
    subject = (
        f"[kca] 🚨 {snapshot_date} 일일점검 경고: 정상화 {', '.join(display_changed)}"
        + (f" / 신규 {', '.join(opened)}" if opened else "")
        + f" / 잔여 {', '.join(sorted(new_keys))}"
    )
    lines = [
        "==================================================",
        f"🚨 K-Closing Alpha 점검 변동 알림 ({snapshot_date})",
        "==================================================",
        f"• 해소: {', '.join(display_changed)}",
        *[f"• 신규: {new_open[key]['text']}" for key in opened],
        f"• 잔여: {', '.join(sorted(new_keys))}",
        "• 조치 안내: or-vps 서버 상태 점검 요망",
    ]
    body = "\n".join(lines)
    if dry_run:
        return ReconcileResult(
            action=ACTION_CHANGED, opened=opened, resolved=resolved, still_open=still_open, notified=False
        )
    results = dispatch_fn(subject, body)
    if not any(results.values()):
        return ReconcileResult(
            action=ACTION_CHANGED, opened=opened, resolved=resolved, still_open=still_open, notified=False
        )
    _persist_state(set(opened), reminder=False)
    _refresh_heartbeat(subject, "WARNING", _current_outbox())
    return ReconcileResult(action=ACTION_CHANGED, opened=opened, resolved=resolved, still_open=still_open, notified=True)


def main() -> int:  # pragma: no cover - CLI entry; logic covered via run_audit_reconcile scenarios
    """CLI entry for the systemd reconcile unit."""
    parser = argparse.ArgumentParser(description="Re-measure self-healing audit conditions")
    parser.add_argument("--dry-run", action="store_true", help="Print the decision without writing state or heartbeat")
    args = parser.parse_args()
    now = datetime.now(SEOUL)
    try:
        result = run_audit_reconcile(
            now=now,
            failed_units_fn=list_failed_kca_units,
            stale_tokens_fn=lambda snapshot: list_stale_kis_tokens(snapshot, allow_newer=True),
            backup_issues_fn=_default_backup_issues,
            outbox_fn=_default_outbox_count,
            dispatch_fn=dispatch_digest,
            dry_run=args.dry_run,
        )
    except OSError as exc:
        logger.error("[SYS] stage=audit_reconcile status=PERSIST_FAILED reason=%s", exc)
        return 1
    print(json.dumps({  # noqa: T201 - CLI decision output for --dry-run/operators
        "action": result.action,
        "opened": list(result.opened),
        "resolved": list(result.resolved),
        "still_open": list(result.still_open),
        "notified": result.notified,
        "dry_run": args.dry_run,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    raise SystemExit(main())
