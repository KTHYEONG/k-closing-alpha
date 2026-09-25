"""실패 알림 설정 도메인 (AlertSettings).

systemd OnFailure= 훅과 daily_audit 결손 감사가 공유하는 웹훅/이메일 알림
자격증명을 담당한다. 모든 필드는 기본값이 빈 문자열이며, 미설정 시 각
알림 채널은 예외 없이 조용히 스킵된다(선택적 기능; 필수 인프라 아님).
"""

from __future__ import annotations

from pydantic import Field

from src.config._env import EnvSettings


class AlertSettings(EnvSettings):
    """실패 알림 웹훅/이메일 자격증명."""

    ALERT_WEBHOOK_URL: str = Field(default="")
    ALERT_GMAIL_USER: str = Field(default="")
    ALERT_GMAIL_APP_PASSWORD: str = Field(default="")
    ALERT_GMAIL_TO: str = Field(default="")
    ALERT_RETRY_ATTEMPTS: int = Field(default=3, ge=1)
    ALERT_RETRY_BACKOFF_SECONDS: float = Field(default=5.0, gt=0.0)
    ALERT_OUTBOX_MAX_DRAIN: int = Field(default=20, ge=1)
