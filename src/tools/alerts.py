"""경량 실패 알림: 웹훅(Slack/Discord 호환) + Gmail 앱비밀번호 이메일.

유료 인프라 없이 systemd OnFailure= 훅에서 직접 호출 가능하도록 설계한다.
자격증명이 비어있으면 각 채널은 예외 없이 조용히 스킵된다(선택적 기능).
"""

from __future__ import annotations

import argparse
import logging
import re
import smtplib
import subprocess
from collections.abc import Callable
from email.message import EmailMessage

import requests

from src import settings

logger = logging.getLogger(__name__)

ALERT_JOURNAL_TAIL_LINES: int = 40
ALERT_LINE_MAX_CHARS: int = 300
ALERT_COMMAND_TIMEOUT_SEC: float = 10.0
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


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


def dispatch_digest(subject: str, body: str) -> dict[str, bool]:
    """일일 요약을 웹훅과 이메일 양쪽에 보내고 채널별 성공여부를 반환한다.

    채널 실패는 dispatch_failure_alert와 같은 방식으로 좁게 잡아 격리한다.

    Args:
        subject: 요약 제목.
        body: 요약 본문.

    Returns:
        {"webhook": bool, "email": bool}.
    """
    results = {"webhook": False, "email": False}
    try:
        results["webhook"] = post_webhook_text(settings.ALERT_WEBHOOK_URL, f"{subject}\n{body}")
    except (requests.RequestException, OSError) as exc:
        logger.warning("[SYS] digest webhook dispatch failed reason=%s", type(exc).__name__)
    try:
        results["email"] = send_email(
            gmail_user=settings.ALERT_GMAIL_USER,
            gmail_app_password=settings.ALERT_GMAIL_APP_PASSWORD,
            to_addr=settings.ALERT_GMAIL_TO,
            subject=subject,
            body=body,
        )
    except (smtplib.SMTPException, OSError) as exc:
        logger.warning("[SYS] digest email dispatch failed reason=%s", type(exc).__name__)
    return results


def post_webhook_alert(webhook_url: str, *, unit: str, detail: str = "", subject: str | None = None) -> bool:
    """Slack/Discord 호환 웹훅으로 실패 알림을 보낸다.

    Args:
        webhook_url: 웹훅 URL. 빈 문자열이면 미설정으로 간주해 스킵한다.
        unit: 실패한 systemd 유닛 이름.
        detail: 부가 설명(선택).
        subject: 알림 제목 override(선택).

    Returns:
        전송을 시도했으면 True, webhook_url 이 비어 스킵했으면 False.

    Raises:
        requests.RequestException: 전송 자체가 실패한 경우 그대로 전파한다.
    """
    title = subject or f"[KCA][실패] systemd unit failed: {unit}"
    text = f"{title}\n\n{detail}" if detail else title
    return post_webhook_text(webhook_url, text)


def send_email_alert(
    *,
    gmail_user: str,
    gmail_app_password: str,
    to_addr: str,
    unit: str,
    detail: str = "",
    subject: str | None = None,
) -> bool:
    """Gmail 앱비밀번호 SMTP_SSL 로 실패 알림을 보낸다.

    Args:
        gmail_user: 발신 Gmail 계정.
        gmail_app_password: Gmail 앱 비밀번호.
        to_addr: 수신 이메일 주소.
        unit: 실패한 systemd 유닛 이름.
        detail: 부가 설명(선택).
        subject: 메일 제목 override(선택).

    Returns:
        세 자격증명(gmail_user/gmail_app_password/to_addr)이 모두 있으면 전송
        시도 후 True, 하나라도 비어있으면 스킵한 True 없이 False.

    Raises:
        smtplib.SMTPException: 전송 자체가 실패한 경우 그대로 전파한다.
    """
    return send_email(
        gmail_user=gmail_user,
        gmail_app_password=gmail_app_password,
        to_addr=to_addr,
        subject=subject or f"[KCA][실패] systemd unit failed: {unit}",
        body=detail or f"unit={unit} failed with no further detail",
    )


def dispatch_failure_alert(unit: str, *, detail: str = "", subject: str | None = None) -> dict[str, bool]:
    """웹훅과 이메일 채널 모두 시도하고 채널별 성공여부를 반환한다.

    각 채널의 전송 실패(네트워크/SMTP 오류)는 이 함수 안에서만 좁게 잡아
    한 채널의 실패가 다른 채널 시도를 막지 않게 한다. 두 값 모두 False 인
    상태(미설정 또는 두 채널 모두 실패)도 정상 반환이며 예외가 아니다.

    Args:
        unit: 실패한 systemd 유닛 이름(OnFailure= 의 %i, 또는 호출부가
            구성한 논리적 식별자).
        detail: 부가 설명(선택).
        subject: 제목 override(선택).

    Returns:
        {"webhook": bool, "email": bool}.
    """
    results = {"webhook": False, "email": False}
    try:
        if subject is not None:
            results["webhook"] = post_webhook_alert(settings.ALERT_WEBHOOK_URL, unit=unit, detail=detail, subject=subject)
        else:
            results["webhook"] = post_webhook_alert(settings.ALERT_WEBHOOK_URL, unit=unit, detail=detail)
    except (requests.RequestException, OSError) as exc:
        logger.warning("[SYS] alert webhook dispatch failed unit=%s reason=%s", unit, type(exc).__name__)
    try:
        if subject is not None:
            results["email"] = send_email_alert(
                gmail_user=settings.ALERT_GMAIL_USER,
                gmail_app_password=settings.ALERT_GMAIL_APP_PASSWORD,
                to_addr=settings.ALERT_GMAIL_TO,
                unit=unit,
                detail=detail,
                subject=subject,
            )
        else:
            results["email"] = send_email_alert(
                gmail_user=settings.ALERT_GMAIL_USER,
                gmail_app_password=settings.ALERT_GMAIL_APP_PASSWORD,
                to_addr=settings.ALERT_GMAIL_TO,
                unit=unit,
                detail=detail,
            )
    except (smtplib.SMTPException, OSError) as exc:
        logger.warning("[SYS] alert email dispatch failed unit=%s reason=%s", unit, type(exc).__name__)
    return results


def sanitize_journal_tail(text: str) -> str:
    """Strip ANSI escapes and carriage-return segments, drop blanks, cap line length."""
    lines = []
    for raw in text.split("\n"):
        line = _ANSI_ESCAPE_RE.sub("", raw).split("\r")[-1].rstrip()
        if not line:
            continue
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


def format_failure_alert(
    unit: str,
    unit_status: dict[str, str],
    diagnostics: dict[str, str],
) -> tuple[str, str]:
    """Construct human-readable subject and structured body for a unit failure alert.

    Produces a clean 5-line summary card matching the KCA alert design convention
    (daily_audit and run_outcome), followed by the sanitized error context and
    a copy-pasteable server journal inspection command.

    Args:
        unit: Name of the failing systemd unit.
        unit_status: Parsed systemctl properties (Result, ExecMainStatus, timestamps).
        diagnostics: Output from extract_failure_summary ('reason', 'location', 'context').

    Returns:
        Tuple of (subject, body):
        - subject: '[kca] 🚨 유닛 실행 실패: {unit}'
        - body: Structured card with unit, exit status, timestamp, core reason, location, and journal command.
    """
    exit_status = unit_status.get("ExecMainStatus", "unknown")
    result = unit_status.get("Result", "unknown")
    start_ts = unit_status.get("ExecMainStartTimestamp", "unknown")
    exit_ts = unit_status.get("ExecMainExitTimestamp", "unknown")

    subject = f"[kca] 🚨 유닛 실행 실패: {unit}"

    reason = diagnostics.get("reason", "알 수 없는 원인")
    location = diagnostics.get("location", "")
    context = diagnostics.get("context", "")

    summary_lines = [
        "==================================================",
        f"🚨 KCA 시스템 유닛 장애 알림 ({unit})",
        "==================================================",
        f"• 실패 유닛: {unit}",
        f"• 종료 상태: {result} (exit code: {exit_status})",
        f"• 발생 시각: {exit_ts} (시작: {start_ts})",
        f"• 핵심 원인: {reason}",
    ]
    if location:
        summary_lines.append(f"• 발생 위치: {location}")
    summary_lines.append(f"• 저널 확인: journalctl --user -u {unit} -n 50 --no-pager")
    summary_lines.append("")
    summary_lines.append("[핵심 에러 로그]")
    summary_lines.append(context)

    return subject, "\n".join(summary_lines)


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
    if inv_id:
        try:
            journal = run(
                ["journalctl", "--user", f"_SYSTEMD_INVOCATION_ID={inv_id}", "-o", "cat", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=ALERT_COMMAND_TIMEOUT_SEC,
                check=True,
            )
            journal_text = sanitize_journal_tail(journal.stdout)
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
            journal_text = sanitize_journal_tail(journal.stdout)
        except (OSError, subprocess.SubprocessError) as exc:
            journal_text = f"journal unavailable: {type(exc).__name__}"

    return unit_status, journal_text


def main(argv: list[str] | None = None) -> None:
    """systemd OnFailure= 진입점: --unit 을 파싱해 얼러트를 발송한다."""
    parser = argparse.ArgumentParser(description="Dispatch a failure alert for a systemd unit (OnFailure= entrypoint)")
    parser.add_argument("--unit", required=True, help="failing unit name (systemd %%i specifier)")
    parser.add_argument("--detail", default="", help="optional extra detail text")
    args = parser.parse_args(argv)
    if args.detail:
        results = dispatch_failure_alert(args.unit, detail=args.detail)
    else:
        unit_status, journal_text = collect_unit_diagnostics(args.unit)
        diag = extract_failure_summary(journal_text, unit_status)
        subject, body = format_failure_alert(args.unit, unit_status, diag)
        results = dispatch_failure_alert(args.unit, detail=body, subject=subject)
    logger.info("[SYS] alert dispatch unit=%s webhook=%s email=%s", args.unit, results["webhook"], results["email"])


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
