"""Cross-path incident pager: one mail per CRITICAL incident per KST day, plus one resolution mail.

Every entry path (systemd OnFailure, daily audit, reconcile tick, run outcomes) pages through this
ledger, keyed by `IncidentKind`, so one incident never mails twice on one day no matter how many
paths observe it. Sources (issue keys / run ids) attached to a kind decide when it resolves.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from src import settings
from src.data.io_utils import atomic_write_text, store_write_lock
from src.tools.alert_policy import (
    ALERT_POLICY_TZ,
    DEFAULT_ALERT_POLICY,
    AlertPolicyConfig,
    IncidentKind,
    SourceScope,
    classify_issue_key,
    format_page,
    format_resolved,
    incident_scope,
    issue_page_due,
)

logger = logging.getLogger(__name__)

INCIDENT_LEDGER_RELPATH: str = "logs/alerts/incidents.json"
SILENT_RECORD_RELDIR: str = "logs/alerts/silent"
INCIDENT_LEDGER_LOCK_TIMEOUT_SECONDS: float = 10.0

SendFn = Callable[[str, str], Mapping[str, bool]]


class PageDecision(StrEnum):
    """Outcome of one page request."""

    SENT = "SENT"
    DEDUPED = "DEDUPED"
    REMINDER_WAIT = "REMINDER_WAIT"


@dataclass(frozen=True)
class _Ledger:
    sources: dict[str, dict[str, str]]
    pages: dict[str, dict[str, str]]


def incident_ledger_path() -> Path:
    """Shared incident ledger: DATA_DIR/logs/alerts/incidents.json."""
    return Path(settings.DATA_DIR) / INCIDENT_LEDGER_RELPATH


def record_silent(source: str, *, reason: str, now: datetime, status: str = "SILENT", root: Path | None = None) -> None:
    """Log and append one non-paging alert record to DATA_DIR/logs/alerts/silent/<KST date>.jsonl.

    `reason` must already be credential-redacted. A write failure only logs.
    """
    logger.info("[SYS] stage=alert status=%s source=%s reason=%s", status, source, reason)
    box = Path(root) if root is not None else Path(settings.DATA_DIR) / SILENT_RECORD_RELDIR
    record = {"ts": now.isoformat(timespec="seconds"), "status": status, "source": source, "reason": reason}
    try:
        box.mkdir(parents=True, exist_ok=True)
        with (box / f"{now.astimezone(ALERT_POLICY_TZ).date().isoformat()}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("[SYS] stage=alert status=SILENT_RECORD_FAILED reason=%s", type(exc).__name__)


def _default_send(subject: str, body: str) -> Mapping[str, bool]:
    from src.tools.alerts import dispatch_page

    return dispatch_page(subject, body)


def _kst_day(at: datetime) -> str:
    return at.astimezone(ALERT_POLICY_TZ).date().isoformat()


def _str_map(raw: object) -> dict[str, dict[str, str]]:
    if not isinstance(raw, dict):
        return {}
    return {
        str(key): {str(k): str(v) for k, v in value.items()}
        for key, value in raw.items()
        if isinstance(value, dict)
    }


class IncidentPager:
    """Ledger-backed pager shared by every alert entry path.

    The ledger is read-modify-written under a kernel lock; mail is sent after the lock is released
    and the page is recorded first, so a crash between record and send loses at most one mail
    (dispatch itself falls back to the outbox). An unreadable or locked ledger fails toward paging.
    """

    def __init__(
        self,
        *,
        ledger_path: Path | None = None,
        send_fn: SendFn | None = None,
        cfg: AlertPolicyConfig = DEFAULT_ALERT_POLICY,
        lock_timeout_seconds: float = INCIDENT_LEDGER_LOCK_TIMEOUT_SECONDS,
    ) -> None:
        self._path = Path(ledger_path) if ledger_path is not None else incident_ledger_path()
        self._send = send_fn if send_fn is not None else _default_send
        self._cfg = cfg
        self._lock_timeout = lock_timeout_seconds

    def _load(self) -> _Ledger:
        try:
            raw: Any = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return _Ledger({}, {})
        if not isinstance(raw, dict):
            raise ValueError("incident ledger is not a mapping")
        return _Ledger(_str_map(raw.get("sources")), _str_map(raw.get("pages")))

    def _save(self, ledger: _Ledger) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"sources": ledger.sources, "pages": ledger.pages}
        atomic_write_text(self._path, json.dumps(payload, ensure_ascii=False, sort_keys=True), mode=None)

    @staticmethod
    def _rollover(ledger: _Ledger, today: str) -> None:
        for source, entry in list(ledger.sources.items()):
            if entry.get("scope") == SourceScope.DAY.value and entry.get("day", "") < today:
                del ledger.sources[source]
        live_kinds = {entry.get("kind") for entry in ledger.sources.values()}
        for kind in list(ledger.pages):
            if kind not in live_kinds:
                del ledger.pages[kind]

    def _mutate(self, now: datetime, change: Callable[[_Ledger, str], Any]) -> Any:
        with store_write_lock(self._path, purpose="incident-ledger", timeout_seconds=self._lock_timeout):
            ledger = self._load()
            today = _kst_day(now)
            self._rollover(ledger, today)
            result = change(ledger, today)
            self._save(ledger)
            return result

    def first_seen(self, source: str) -> datetime | None:
        """First sighting of `source` in the ledger, or None when absent or unreadable."""
        try:
            entry = self._load().sources.get(source)
        except (OSError, ValueError):
            return None
        if entry is None:
            return None
        try:
            return datetime.fromisoformat(entry["first_seen"])
        except (KeyError, ValueError):
            return None

    def observe(self, kind: IncidentKind, source: str, *, scope: SourceScope, now: datetime) -> datetime:
        """Attach `source` to `kind` without paging; returns its first sighting (now when unrecorded)."""

        def _change(ledger: _Ledger, today: str) -> datetime:
            entry = ledger.sources.setdefault(
                source, {"kind": kind.value, "scope": scope.value, "first_seen": now.isoformat(), "day": today}
            )
            entry["day"] = today
            return datetime.fromisoformat(entry["first_seen"])

        try:
            seen: datetime = self._mutate(now, _change)
        except (OSError, ValueError) as exc:
            logger.warning("[SYS] stage=alert status=LEDGER_FAILED op=observe source=%s reason=%s", source, type(exc).__name__)
            return now
        return seen

    def page(
        self, kind: IncidentKind, source: str, *, scope: SourceScope, now: datetime, auto_action: str | None = None
    ) -> PageDecision:
        """Page `kind` unless it already paged today, or it paged on an earlier day and the reminder hour has not come."""

        def _change(ledger: _Ledger, today: str) -> PageDecision:
            entry = ledger.sources.setdefault(
                source, {"kind": kind.value, "scope": scope.value, "first_seen": now.isoformat(), "day": today}
            )
            entry["day"] = today
            previous = ledger.pages.get(kind.value)
            if previous is not None and previous.get("day") == today:
                return PageDecision.DEDUPED
            if previous is not None and now.astimezone(ALERT_POLICY_TZ).time() < self._cfg.reminder_earliest:
                return PageDecision.REMINDER_WAIT
            ledger.pages[kind.value] = {"day": today, "at": now.isoformat()}
            return PageDecision.SENT

        try:
            decision: PageDecision = self._mutate(now, _change)
        except (OSError, ValueError) as exc:
            logger.warning("[SYS] stage=alert status=LEDGER_FAILED op=page kind=%s reason=%s", kind.value, type(exc).__name__)
            decision = PageDecision.SENT
        if decision is PageDecision.SENT:
            subject, body = format_page(kind, auto_action=auto_action)
            results = self._send(subject, body)
            logger.warning(
                "[SYS] stage=alert status=PAGED kind=%s source=%s delivered=%s", kind.value, source, any(results.values())
            )
        else:
            logger.info("[SYS] stage=alert status=%s kind=%s source=%s", decision.value, kind.value, source)
        return decision

    def resolve(self, sources: Collection[str], *, now: datetime) -> tuple[IncidentKind, ...]:
        """Detach `sources`; each kind left with no source and previously paged gets one resolution mail."""
        wanted = set(sources)
        if not wanted:
            return ()

        def _change(ledger: _Ledger, today: str) -> tuple[IncidentKind, ...]:
            touched = {ledger.sources.pop(src)["kind"] for src in wanted if src in ledger.sources}
            live = {entry.get("kind") for entry in ledger.sources.values()}
            done = sorted(kind for kind in touched if kind not in live and ledger.pages.pop(kind, None) is not None)
            return tuple(IncidentKind(kind) for kind in done)

        try:
            if not self._path.exists():
                return ()
            resolved: tuple[IncidentKind, ...] = self._mutate(now, _change)
        except (OSError, ValueError) as exc:
            logger.warning("[SYS] stage=alert status=LEDGER_FAILED op=resolve reason=%s", type(exc).__name__)
            return ()
        for kind in resolved:
            subject, body = format_resolved(kind, at=now)
            self._send(subject, body)
            logger.info("[SYS] stage=alert status=RESOLVED kind=%s", kind.value)
        return resolved

    def resolve_absent(
        self, open_keys: Collection[str], *, measured_prefixes: tuple[str, ...], now: datetime
    ) -> tuple[IncidentKind, ...]:
        """Resolve OPEN-scope sources under `measured_prefixes` that the caller just measured as absent."""
        try:
            ledger = self._load()
        except (OSError, ValueError) as exc:
            logger.warning("[SYS] stage=alert status=LEDGER_FAILED op=resolve reason=%s", type(exc).__name__)
            return ()
        present = set(open_keys)
        gone = [
            source
            for source, entry in ledger.sources.items()
            if entry.get("scope") == SourceScope.OPEN.value and source.startswith(measured_prefixes) and source not in present
        ]
        return self.resolve(gone, now=now)


IssuePendingFn = Callable[[str, datetime, datetime], bool]
"""(issue_key, first_seen, now) -> True while auto-remediation may still clear the key."""


def page_issue_keys(
    pager: IncidentPager,
    open_keys: Collection[str],
    *,
    now: datetime,
    pending: IssuePendingFn | None = None,
    auto_action: Callable[[str], str | None] | None = None,
    cfg: AlertPolicyConfig = DEFAULT_ALERT_POLICY,
) -> dict[str, str]:
    """Page every measured CRITICAL, due, non-held audit issue key.

    SILENT keys are left to the heartbeat/digest; keys of event kinds (failed order/decision units)
    are owned by the OnFailure and run-outcome paths and skipped here.

    Returns:
        Mapping of each paged-kind key to its status: "NOT_DUE", "HELD", or a PageDecision value.
    """
    statuses: dict[str, str] = {}
    for key in sorted(set(open_keys)):
        kind = classify_issue_key(key)
        if kind is None or incident_scope(kind) is not SourceScope.OPEN:
            continue
        scope = SourceScope.OPEN
        first_seen = pager.observe(kind, key, scope=scope, now=now)
        if not issue_page_due(kind, first_seen=first_seen, now=now, cfg=cfg):
            statuses[key] = "NOT_DUE"
        elif pending is not None and pending(key, first_seen, now):
            statuses[key] = "HELD"
        else:
            action = auto_action(key) if auto_action is not None else None
            statuses[key] = pager.page(kind, key, scope=scope, now=now, auto_action=action).value
    if statuses:
        logger.info("[SYS] stage=alert op=page_issue_keys statuses=%s", statuses)
    return statuses
