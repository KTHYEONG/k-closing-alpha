"""Alert transport (webhook + Gmail app-password mail) and the systemd OnFailure= entry point.

Which failures reach a human is decided by `src.tools.alert_policy`; this module only delivers,
redacts, and queues undelivered mail. A channel with empty credentials is skipped silently.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import smtplib
import subprocess
import time
import uuid
from collections.abc import Callable, Collection, Mapping
from datetime import UTC, datetime
from email.message import EmailMessage
from enum import StrEnum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from src import settings
from src.api.kis.key_pool import load_kis_env
from src.config.secret_fields import configured_secret_values
from src.data.io_utils import atomic_write_text
from src.tools.alert_policy import FAILED_UNIT_PREFIX, classify_unit, incident_scope, unit_base_name
from src.tools.incident_pager import IncidentPager, record_silent
from src.utils.cli_logging import configure_cli_logging
from src.utils.redaction import redact_secrets

logger = logging.getLogger(__name__)

ALERT_JOURNAL_TAIL_LINES: int = 40
ALERT_LINE_MAX_CHARS: int = 300
ALERT_COMMAND_TIMEOUT_SEC: float = 10.0
_REDELIVERY_SUBJECT_PREFIX: str = "[지연전송] "
# KST day boundary for digest files and the OnFailure clock.
ALERT_DAY_TZ: ZoneInfo = ZoneInfo("Asia/Seoul")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class ChannelOutcome(StrEnum):
    """Delivery result of one alert channel."""

    NOT_CONFIGURED = "NOT_CONFIGURED"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"


def deliver_with_retry(
    send: Callable[[], bool],
    *,
    channel: str,
    attempts: int,
    backoff_seconds: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> ChannelOutcome:
    """Deliver through one channel with bounded exponential backoff.

    A send callable returning False means the channel is not configured and is
    never retried. Transport/SMTP failures are retried up to `attempts` total
    tries with waits backoff_seconds * 2**k between them.

    Args:
        send: Channel sender; True on delivery, False when not configured.
        channel: Channel label for logs ("webhook" or "email").
        attempts: Total tries (>= 1).
        backoff_seconds: Base wait before the second try.
        sleep_fn: Injectable sleeper.

    Returns:
        NOT_CONFIGURED, DELIVERED or FAILED.
    """
    for attempt in range(1, attempts + 1):
        try:
            if send():
                return ChannelOutcome.DELIVERED
            return ChannelOutcome.NOT_CONFIGURED
        except (requests.RequestException, smtplib.SMTPException, OSError) as exc:
            reason = type(exc).__name__
            if attempt >= attempts:
                logger.warning(
                    "[SYS] stage=alert_delivery channel=%s attempt=%d/%d status=FAILED reason=%s",
                    channel, attempt, attempts, reason,
                )
            else:
                logger.warning(
                    "[SYS] stage=alert_delivery channel=%s attempt=%d/%d status=RETRY reason=%s",
                    channel, attempt, attempts, reason,
                )
                sleep_fn(backoff_seconds * (2.0 ** (attempt - 1)))
    return ChannelOutcome.FAILED


def _deliver_channel(send: Callable[[], bool], *, channel: str) -> bool:
    attempts = int(settings.ALERT_RETRY_ATTEMPTS)
    backoff_seconds = float(settings.ALERT_RETRY_BACKOFF_SECONDS)
    return deliver_with_retry(send, channel=channel, attempts=attempts, backoff_seconds=backoff_seconds) is ChannelOutcome.DELIVERED


def alert_outbox_dir() -> Path:
    """Directory of undelivered alerts: DATA_DIR/logs/alerts/outbox."""
    return Path(settings.DATA_DIR) / "logs" / "alerts" / "outbox"


def enqueue_undelivered(subject: str, body: str, *, kind: str, outbox: Path | None = None) -> Path:
    """Persist an alert that no configured channel delivered.

    One JSON file per alert (`<utc_ts>_<uuid8>.json`, written via temp file +
    os.replace) so concurrent writers from containers and host never interleave
    and a partially written alert is never read.

    Args:
        subject: Alert subject.
        body: Alert body.
        kind: "failure" or "digest".
        outbox: Override directory (tests).

    Returns:
        Path of the stored alert file.
    """
    box = Path(outbox) if outbox is not None else alert_outbox_dir()
    box.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    name = f"{now.strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex[:8]}.json"
    payload = {"subject": subject, "body": body, "kind": kind, "enqueued_at": now.isoformat()}
    target = box / name
    atomic_write_text(target, json.dumps(payload, ensure_ascii=False), mode=None)
    return target


def _deliver_now(subject: str, body: str) -> dict[str, bool]:
    """Send a digest-shaped alert through both channels without enqueueing."""
    return {
        "webhook": _deliver_channel(
            lambda: post_webhook_text(settings.ALERT_WEBHOOK_URL, f"{subject}\n{body}"),
            channel="webhook",
        ),
        "email": _deliver_channel(
            lambda: send_email(
                gmail_user=settings.ALERT_GMAIL_USER,
                gmail_app_password=settings.ALERT_GMAIL_APP_PASSWORD,
                to_addr=settings.ALERT_GMAIL_TO,
                subject=subject,
                body=body,
            ),
            channel="email",
        ),
    }


def _persist_undelivered(subject: str, body: str, *, kind: str) -> None:
    try:
        stored = enqueue_undelivered(subject, body, kind=kind)
    except OSError as exc:
        logger.warning("[SYS] stage=alert_delivery status=OUTBOX_FAILED reason=%s", type(exc).__name__)
        stored = alert_outbox_dir()
    logger.error("[SYS] stage=alert_delivery status=UNDELIVERED subject=%s outbox=%s", subject, stored)


def alert_credential_env_file() -> Path:
    """Credential env file whose values are exact-masked in alert egress (BASE_DIR/.env)."""
    return Path(settings.BASE_DIR) / ".env"


def _alert_secret_values(env: Mapping[str, str] | None = None) -> frozenset[str]:
    """Resolve configured credential values for alert redaction.

    Reads the live settings singleton plus the credential env (``load_kis_env`` of
    ``alert_credential_env_file()``, which also merges ``os.environ`` — on the VPS the
    alert unit's EnvironmentFile places every pool key there). A read failure degrades
    to settings-only values and is logged by exception type name only (OD-3).
    """
    if env is None:
        try:
            env = load_kis_env(alert_credential_env_file())
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            logger.warning("[SYS] stage=alert_redaction status=DEGRADED reason=%s", type(exc).__name__)
            env = {}
    cfg: Any = settings
    return configured_secret_values(cfg, env)


def redact_for_egress(text: str) -> str:
    """Mask configured credential values and credential-shaped pairs in operator-facing text.

    Callers that truncate or persist free text outside the alert dispatchers (event logs,
    run-outcome reasons) must mask first, so a cut can never leave an unmatched secret prefix.
    """
    return redact_secrets(text, _alert_secret_values())


def drain_alert_outbox(
    *, max_items: int | None = None, outbox: Path | None = None, send_fn: Callable[[str, str], dict[str, bool]] | None = None
) -> tuple[int, int]:
    """Re-deliver stored alerts oldest-first and delete each one delivered.

    Stops at the first alert that still fails (preserves order, avoids hammering
    a down channel).

    Returns:
        (delivered_count, remaining_count).
    """
    box = Path(outbox) if outbox is not None else alert_outbox_dir()
    if not box.is_dir():
        return (0, 0)
    secrets = _alert_secret_values()
    files = sorted(box.glob("*.json"))
    if max_items is not None:
        files = files[:max_items]
    sender = send_fn if send_fn is not None else _deliver_now
    delivered = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("[SYS] stage=alert_delivery status=CORRUPT_OUTBOX path=%s", path.name)
            break
        if not isinstance(payload, dict) or not isinstance(payload.get("subject"), str) or not isinstance(payload.get("body"), str):
            logger.warning("[SYS] stage=alert_delivery status=CORRUPT_OUTBOX path=%s", path.name)
            break
        masked_subject = redact_secrets(payload["subject"], secrets)
        masked_body = redact_secrets(payload["body"], secrets)
        redelivered_subject = f"{_REDELIVERY_SUBJECT_PREFIX}{masked_subject}"
        redelivered_body = f"{masked_body}\noriginal_enqueued_at={payload.get('enqueued_at', '')}"
        results = sender(redelivered_subject, redelivered_body)
        if not any(results.values()):
            break
        path.unlink()
        delivered += 1
    remaining = len(list(box.glob("*.json")))
    return (delivered, remaining)


def post_webhook_text(webhook_url: str, text: str) -> bool:
    """Slack/Discord 호환 웹훅으로 임의 텍스트를 보낸다.

    Args:
        webhook_url: 웹훅 URL. 빈 문자열이면 미설정으로 간주해 스킵한다.
        text: 보낼 본문.

    Returns:
        전송을 시도했으면 True, URL이 비어 스킵했으면 False.

    Raises:
        requests.RequestException: 전송 자체가 실패한 경우 그대로 전파한다.
    """
    if not webhook_url:
        return False
    resp = requests.post(webhook_url, json={"text": text}, timeout=10)
    resp.raise_for_status()
    return True


def send_email(*, gmail_user: str, gmail_app_password: str, to_addr: str, subject: str, body: str) -> bool:
    """Gmail 앱비밀번호 SMTP_SSL로 임의 제목/본문 메일을 보낸다.

    Args:
        gmail_user: 발신 Gmail 계정.
        gmail_app_password: Gmail 앱 비밀번호.
        to_addr: 수신 이메일 주소.
        subject: 메일 제목.
        body: 메일 본문.

    Returns:
        세 자격증명이 모두 있으면 전송 후 True, 하나라도 비어있으면 False.

    Raises:
        smtplib.SMTPException: 전송 자체가 실패한 경우 그대로 전파한다.
    """
    if not gmail_user or not gmail_app_password or not to_addr:
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = to_addr
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
        smtp.login(gmail_user, gmail_app_password)
        smtp.send_message(msg)
    return True


def dispatch_page(subject: str, body: str) -> dict[str, bool]:
    """Send one operator page (CRITICAL incident or its resolution) through webhook and email.

    Credentials are masked before any channel or the outbox sees the text; when no channel
    delivers, the page is queued in the outbox for redelivery.

    Returns:
        {"webhook": bool, "email": bool}; both False means the page was queued.
    """
    secrets = _alert_secret_values()
    subject = redact_secrets(subject, secrets)
    body = redact_secrets(body, secrets)
    results = _deliver_now(subject, body)
    if any(results.values()):
        logger.info(
            "[SYS] stage=alert_delivery status=SENT kind=page webhook=%s email=%s subject_key=%s",
            results.get("webhook"),
            results.get("email"),
            subject[:80],
        )
    else:
        _persist_undelivered(subject, body, kind="page")
    return results


def digest_record_dir() -> Path:
    """Directory of SILENT digests the ops check reads: DATA_DIR/logs/alerts/digest."""
    return alert_outbox_dir().parent / "digest"


def record_digest(subject: str, body: str, *, now: datetime | None = None, root: Path | None = None) -> dict[str, bool]:
    """Record a non-paging digest to the daily JSONL digest file instead of mailing it.

    Matches the (subject, body) -> results sink contract of the audit/reconcile digest path: it returns
    {"recorded": True} on success so callers that gate state on delivery treat the digest as handled,
    and {"recorded": False} when the file cannot be written.
    """
    at = now if now is not None else datetime.now(ALERT_DAY_TZ)
    box = Path(root) if root is not None else digest_record_dir()
    secrets = _alert_secret_values()
    record = {
        "ts": at.isoformat(timespec="seconds"),
        "subject": redact_secrets(subject, secrets),
        "body": redact_secrets(body, secrets),
    }
    target = box / f"{at.astimezone(ALERT_DAY_TZ).date().isoformat()}.jsonl"
    try:
        box.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("[SYS] stage=alert status=DIGEST_RECORD_FAILED reason=%s", type(exc).__name__)
        return {"recorded": False}
    logger.info("[SYS] stage=alert status=SILENT kind=digest subject_key=%s", record["subject"][:80])
    return {"recorded": True}


def sanitize_journal_tail(text: str, *, secret_values: Collection[str] = ()) -> str:
    """Strip ANSI escapes and carriage-return segments, mask credentials, drop blanks, cap line length.

    Masking runs per line before the length cap: truncating first could split a long
    secret (KIS app secrets are ~180 chars) and leave an unmasked prefix that exact
    matching no longer recognizes.

    Args:
        text: Raw journal output.
        secret_values: Configured credential values to exact-mask (see redact_secrets).

    Returns:
        Sanitized newline-joined lines, each at most ALERT_LINE_MAX_CHARS plus the ellipsis.
    """
    lines = []
    for raw in text.split("\n"):
        line = _ANSI_ESCAPE_RE.sub("", raw).split("\r")[-1].rstrip()
        if not line:
            continue
        line = redact_secrets(line, secret_values)
        if len(line) > ALERT_LINE_MAX_CHARS:
            line = line[:ALERT_LINE_MAX_CHARS] + "…"
        lines.append(line)
    return "\n".join(lines)


def parse_systemctl_show(stdout: str) -> dict[str, str]:
    """Parse key-value properties from systemctl show output.

    Args:
        stdout: Raw output from systemctl show command containing 'Key=Value' lines.

    Returns:
        Mapping of property keys to property values, with whitespace stripped.
        Empty lines or lines without '=' are ignored.
    """
    props: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        props[key.strip()] = val.strip()
    return props


def extract_failure_summary(journal_text: str, unit_status: dict[str, str]) -> dict[str, str]:
    """Extract key error reason, source location, and sanitized context from journal text.

    Scans journal logs to extract the fundamental cause of failure, prioritizing:
    1. Python Exception message (e.g. 'FileNotFoundError: ...') and its source location in project code.
    2. Structured log error lines containing '[ERROR]' or 'status=ERROR'.
    3. Critical/fatal log lines.
    4. Fallback to systemctl Result and exit code when no application log exists (e.g. OOM or SIGKILL).

    Filters out high-volume diagnostic noise such as replacement lists, progress bars, and repeated borders.

    Args:
        journal_text: Sanitized journal log lines from the failing invocation.
        unit_status: Parsed status properties from systemctl show (Result, ExecMainStatus, etc.).

    Returns:
        Dictionary containing:
        - 'reason': 1-line summarized root cause string.
        - 'location': File and line number string if identified from traceback, else empty string.
        - 'context': Tail of meaningful error context lines (up to 8 lines).
    """
    lines = [line.strip() for line in journal_text.splitlines() if line.strip()]

    error_reason = ""
    error_location = ""

    exc_re = re.compile(r"^([A-Z][a-zA-Z0-9_.]*(?:Error|Exception|Interrupt|Exit|Fault)): (.*)$")
    file_re = re.compile(r'^File "([^"]+)", line (\d+), in (.*)$')

    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        m = exc_re.match(line)
        if m and not error_reason:
            error_reason = line
            for j in range(i - 1, -1, -1):
                fm = file_re.match(lines[j])
                if fm:
                    fpath, lineno, func = fm.group(1), fm.group(2), fm.group(3)
                    if not error_location or ("src/" in fpath or "deploy/" in fpath):
                        error_location = f"{fpath}:{lineno} in {func}"
                        if "src/" in fpath:
                            break
            break

    if not error_reason:
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if "[ERROR]" in line or "status=ERROR" in line:
                error_reason = line
                break

    if not error_reason:
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if "[CRITICAL]" in line or "[FATAL]" in line:
                error_reason = line
                break

    result = unit_status.get("Result", "unknown")
    exit_status = unit_status.get("ExecMainStatus", "unknown")
    if not error_reason:
        error_reason = f"Process exited with {result} (status {exit_status})"

    clean_lines = []
    for line in lines[-20:]:
        if "stage=intraday_replace" in line and "replaced=" in line:
            continue
        if re.match(r"^[━─=─-]{10,}$", line):
            continue
        if "<frozen " in line:
            continue
        clean_lines.append(line)

    context = "\n".join(clean_lines[-8:]) if clean_lines else "(상세 로그 없음)"
    return {
        "reason": error_reason,
        "location": error_location,
        "context": context,
    }


def collect_unit_diagnostics(
    unit: str,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[dict[str, str], str]:
    """Collect isolated invocation journal and system status for a failing unit.

    Attempts to query the unit's InvocationID to fetch logs strictly from the current
    failed run, preventing logs from previous successful runs from polluting the alert.
    Falls back to unit tail if InvocationID is absent or fails.

    Args:
        unit: Name of the systemd unit.
        run: Injectable command runner for subprocess invocation.

    Returns:
        Tuple of (parsed_unit_status, sanitized_journal_text).
    """
    unit_status: dict[str, str] = {}
    try:
        show = run(
            [
                "systemctl",
                "--user",
                "show",
                unit,
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
                "-p",
                "ExecMainStartTimestamp",
                "-p",
                "ExecMainExitTimestamp",
                "-p",
                "InvocationID",
            ],
            capture_output=True,
            text=True,
            timeout=ALERT_COMMAND_TIMEOUT_SEC,
            check=True,
        )
        unit_status = parse_systemctl_show(show.stdout)
    except (OSError, subprocess.SubprocessError) as exc:
        unit_status = {"Result": f"unavailable: {type(exc).__name__}", "ExecMainStatus": "unavailable"}

    inv_id = unit_status.get("InvocationID", "")
    journal_text = ""
    secrets = _alert_secret_values()
    if inv_id:
        try:
            journal = run(
                ["journalctl", "--user", f"_SYSTEMD_INVOCATION_ID={inv_id}", "-o", "cat", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=ALERT_COMMAND_TIMEOUT_SEC,
                check=True,
            )
            journal_text = sanitize_journal_tail(journal.stdout, secret_values=secrets)
        except (OSError, subprocess.SubprocessError):
            journal_text = ""

    if not journal_text:
        try:
            journal = run(
                ["journalctl", "--user", "-u", unit, "-n", str(ALERT_JOURNAL_TAIL_LINES), "-o", "cat", "--all", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=ALERT_COMMAND_TIMEOUT_SEC,
                check=True,
            )
            journal_text = sanitize_journal_tail(journal.stdout, secret_values=secrets)
        except (OSError, subprocess.SubprocessError) as exc:
            journal_text = f"journal unavailable: {type(exc).__name__}"

    return unit_status, journal_text


def handle_unit_failure(
    unit: str,
    *,
    now: datetime,
    pager: IncidentPager,
    pending: Callable[[str, datetime, datetime], bool],
    auto_action: Callable[[str], str | None],
    diagnostics: Callable[[str], tuple[dict[str, str], str]] = collect_unit_diagnostics,
) -> str:
    """Route one systemd unit failure through the paging policy.

    SILENT units are only logged and recorded. A CRITICAL unit whose auto-remediation is still
    pending is recorded as DEFERRED (the reconcile tick pages once remediation is exhausted and the
    unit is still failed); any other CRITICAL unit pages through the shared incident ledger.

    Args:
        pending: (issue_key, first_seen, now) -> True while auto-remediation may still clear the key.
        auto_action: issue_key -> `자동 조치` line, or None for the incident default.

    Returns:
        "SILENT", "DEFERRED", or the PageDecision value.
    """
    kind = classify_unit(unit)
    source = f"{FAILED_UNIT_PREFIX}{unit_base_name(unit)}.service"
    unit_status, journal_text = diagnostics(unit)
    reason = extract_failure_summary(journal_text, unit_status)["reason"]
    if kind is None:
        record_silent(source, reason=reason, now=now)
        return "SILENT"
    scope = incident_scope(kind)
    first_seen = pager.observe(kind, source, scope=scope, now=now)
    if pending(source, first_seen, now):
        record_silent(source, reason=reason, now=now, status="DEFERRED")
        return "DEFERRED"
    logger.warning("[SYS] stage=alert status=CRITICAL unit=%s kind=%s reason=%s", unit, kind.value, reason)
    return pager.page(kind, source, scope=scope, now=now, auto_action=auto_action(source)).value


def main(argv: list[str] | None = None) -> None:
    """systemd OnFailure= entry point: drain the outbox, then route `--unit` through the paging policy.

    A drain failure never blocks routing the new failure.
    """
    from src.tools.auto_remediation import load_paging_view

    parser = argparse.ArgumentParser(description="Route a systemd unit failure through the paging policy (OnFailure= entrypoint)")
    parser.add_argument("--unit", required=True, help="failing unit name (systemd %%i specifier)")
    args = parser.parse_args(argv)
    try:
        drain_alert_outbox(max_items=settings.ALERT_OUTBOX_MAX_DRAIN)
    except Exception as exc:
        logger.warning("[SYS] stage=alert_delivery status=DRAIN_FAILED reason=%s", type(exc).__name__)
    now = datetime.now(ALERT_DAY_TZ)
    view = load_paging_view(now)
    status = handle_unit_failure(
        args.unit,
        now=now,
        pager=IncidentPager(),
        pending=view.pending,
        auto_action=view.auto_action,
        diagnostics=collect_unit_diagnostics,
    )
    logger.info("[SYS] stage=alert unit=%s status=%s", args.unit, status)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    configure_cli_logging()
    main()
