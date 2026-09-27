"""External liveness probe run on the VPS by the GitHub watchdog workflow."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from src import settings
from src.data.capture_contracts import SEOUL
from src.tools.alerts import alert_outbox_dir
from src.tools.daily_audit import AUDIT_HEARTBEAT_RELPATH, SYSTEMCTL_TIMEOUT_SEC, list_failed_kca_units
from src.utils.cli_logging import CLI_LOG_FORMAT_TIMESTAMPED, configure_cli_logging

logger = logging.getLogger(__name__)

WATCHDOG_HEARTBEAT_MAX_LAG_WEEKDAYS: int = 1


@dataclass(frozen=True)
class WatchdogVerdict:
    """Outcome of one external liveness probe.

    Attributes:
        problems: Machine-readable problem tags; empty means healthy.
        expected_snapshot_date: Weekday whose audit heartbeat must exist.
    """

    problems: tuple[str, ...]
    expected_snapshot_date: str


def expected_audit_date(now: datetime) -> str:
    """Most recent Mon-Fri KST date strictly before now's KST date.

    The probe runs the next morning, so the previous weekday's 20:15 audit
    (bounded by 02's timeouts to finish before ~23:40) must have landed.
    """
    day = now.astimezone(SEOUL).date() - timedelta(days=WATCHDOG_HEARTBEAT_MAX_LAG_WEEKDAYS)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return day.isoformat()


def evaluate_watchdog(
    now: datetime,
    *,
    heartbeat_path: Path,
    outbox_dir: Path,
    failed_units: Sequence[str],
    inactive_timers: Sequence[str],
) -> WatchdogVerdict:
    """Classify host liveness from persisted evidence only.

    Args:
        now: Timezone-aware probe time.
        heartbeat_path: Audit heartbeat JSON.
        outbox_dir: Alert outbox directory (undelivered alerts).
        failed_units: `systemctl --user list-units --failed kca-*` names.
        inactive_timers: kca timers declared in deploy/systemd that are not active.

    Returns:
        WatchdogVerdict with problems among: heartbeat_missing,
        heartbeat_unreadable, heartbeat_stale:<date>, alert_outbox:<n>,
        failed_units:<comma list>, timers_inactive:<comma list>.
    """
    expected = expected_audit_date(now)
    problems: list[str] = []
    try:
        payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append("heartbeat_missing")
        payload = None
    except (OSError, ValueError):
        problems.append("heartbeat_unreadable")
        payload = None
    if payload is not None:
        snapshot = payload.get("snapshot_date") if isinstance(payload, dict) else None
        if not isinstance(snapshot, str):
            problems.append("heartbeat_unreadable")
        elif snapshot < expected:
            problems.append(f"heartbeat_stale:{snapshot}")
    pending = len(list(outbox_dir.glob("*.json"))) if outbox_dir.is_dir() else 0
    if pending > 0:
        problems.append(f"alert_outbox:{pending}")
    if failed_units:
        problems.append(f"failed_units:{','.join(sorted(failed_units))}")
    if inactive_timers:
        problems.append(f"timers_inactive:{','.join(sorted(inactive_timers))}")
    return WatchdogVerdict(problems=tuple(problems), expected_snapshot_date=expected)


def _host_heartbeat_path() -> Path:
    return Path(settings.DATA_DIR) / AUDIT_HEARTBEAT_RELPATH


def _inactive_timers(unit_dir: Path | None = None) -> list[str]:
    """kca timers declared in the host checkout that are not active."""
    directory = unit_dir if unit_dir is not None else Path(__file__).resolve().parents[2] / "deploy" / "systemd"
    declared = sorted(path.name for path in directory.glob("kca-*.timer"))
    if not declared:
        return ["none_declared"]
    inactive: list[str] = []
    for name in declared:
        try:
            active = (
                subprocess.run(  # noqa: S603 - unit names come from the repo's own timer files
                    ["/usr/bin/systemctl", "--user", "is-active", name],
                    capture_output=True,
                    text=True,
                    timeout=SYSTEMCTL_TIMEOUT_SEC,
                    check=False,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired):
            active = False
        if not active:
            inactive.append(name)
    return inactive


def main(argv: Sequence[str] | None = None) -> int:
    """CLI run on the VPS by the GitHub watchdog over SSH.

    Prints one `[SYS] stage=watchdog status=OK|PROBLEM problems=...` line and
    returns 0 when healthy, 1 otherwise (the workflow step fails → GitHub mail).
    """
    parser = argparse.ArgumentParser(description="External liveness probe for the VPS")
    parser.parse_args(argv)
    verdict = evaluate_watchdog(
        datetime.now(SEOUL),
        heartbeat_path=_host_heartbeat_path(),
        outbox_dir=alert_outbox_dir(),
        failed_units=list_failed_kca_units(),
        inactive_timers=_inactive_timers(),
    )
    status = "OK" if not verdict.problems else "PROBLEM"
    problems = ",".join(verdict.problems) if verdict.problems else "none"
    logger.info("[SYS] stage=watchdog status=%s problems=%s", status, problems)
    return 0 if not verdict.problems else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging(CLI_LOG_FORMAT_TIMESTAMPED)
    raise SystemExit(main())
