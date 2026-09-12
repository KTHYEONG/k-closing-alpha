"""경량 실패 알림: 웹훅(Slack/Discord 호환) + Gmail 앱비밀번호 이메일.

유료 인프라 없이 systemd OnFailure= 훅에서 직접 호출 가능하도록 설계한다.
자격증명이 비어있으면 각 채널은 예외 없이 조용히 스킵된다(선택적 기능).
"""

from __future__ import annotations

import argparse
import logging
import smtplib
from email.message import EmailMessage

import requests

from src import settings

logger = logging.getLogger(__name__)


def post_webhook_alert(webhook_url: str, *, unit: str, detail: str = "") -> bool:
    """Slack/Discord 호환 웹훅으로 실패 알림을 보낸다.

    Args:
        webhook_url: 웹훅 URL. 빈 문자열이면 미설정으로 간주해 스킵한다.
        unit: 실패한 systemd 유닛 이름.
        detail: 부가 설명(선택).

    Returns:
        전송을 시도했으면 True, webhook_url 이 비어 스킵했으면 False.

    Raises:
        requests.RequestException: 전송 자체가 실패한 경우 그대로 전파한다.
    """
    if not webhook_url:
        return False
    text = f"[KCA] systemd unit failed: {unit}"
    if detail:
        text += f"\n{detail}"
    resp = requests.post(webhook_url, json={"text": text}, timeout=10)
    resp.raise_for_status()
    return True


def send_email_alert(*, gmail_user: str, gmail_app_password: str, to_addr: str, unit: str, detail: str = "") -> bool:
    """Gmail 앱비밀번호 SMTP_SSL 로 실패 알림을 보낸다.

    Args:
        gmail_user: 발신 Gmail 계정.
        gmail_app_password: Gmail 앱 비밀번호.
        to_addr: 수신 이메일 주소.
        unit: 실패한 systemd 유닛 이름.
        detail: 부가 설명(선택).

    Returns:
        세 자격증명(gmail_user/gmail_app_password/to_addr)이 모두 있으면 전송
        시도 후 True, 하나라도 비어있으면 스킵한 True 없이 False.

    Raises:
        smtplib.SMTPException: 전송 자체가 실패한 경우 그대로 전파한다.
    """
    if not gmail_user or not gmail_app_password or not to_addr:
        return False
    msg = EmailMessage()
    msg["Subject"] = f"[KCA] systemd unit failed: {unit}"
    msg["From"] = gmail_user
    msg["To"] = to_addr
    msg.set_content(detail or f"unit={unit} failed with no further detail")
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
        smtp.login(gmail_user, gmail_app_password)
        smtp.send_message(msg)
    return True


def dispatch_failure_alert(unit: str, *, detail: str = "") -> dict[str, bool]:
    """웹훅과 이메일 채널 모두 시도하고 채널별 성공여부를 반환한다.

    각 채널의 전송 실패(네트워크/SMTP 오류)는 이 함수 안에서만 좁게 잡아
    한 채널의 실패가 다른 채널 시도를 막지 않게 한다. 두 값 모두 False 인
    상태(미설정 또는 두 채널 모두 실패)도 정상 반환이며 예외가 아니다.

    Args:
        unit: 실패한 systemd 유닛 이름(OnFailure= 의 %i, 또는 호출부가
            구성한 논리적 식별자).
        detail: 부가 설명(선택).

    Returns:
        {"webhook": bool, "email": bool}.
    """
    results = {"webhook": False, "email": False}
    try:
        results["webhook"] = post_webhook_alert(settings.ALERT_WEBHOOK_URL, unit=unit, detail=detail)
    except (requests.RequestException, OSError) as exc:
        logger.warning("[SYS] alert webhook dispatch failed unit=%s reason=%s", unit, type(exc).__name__)
    try:
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


def main(argv: list[str] | None = None) -> None:
    """systemd OnFailure= 진입점: --unit 을 파싱해 얼러트를 발송한다."""
    parser = argparse.ArgumentParser(description="Dispatch a failure alert for a systemd unit (OnFailure= entrypoint)")
    parser.add_argument("--unit", required=True, help="failing unit name (systemd %%i specifier)")
    parser.add_argument("--detail", default="", help="optional extra detail text")
    args = parser.parse_args(argv)
    results = dispatch_failure_alert(args.unit, detail=args.detail)
    logger.info("[SYS] alert dispatch unit=%s webhook=%s email=%s", args.unit, results["webhook"], results["email"])


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
