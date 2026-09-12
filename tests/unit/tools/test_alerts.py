def test_post_webhook_alert_sends_json_and_skips_when_url_empty(monkeypatch) -> None:
    import requests

    from src.tools.alerts import post_webhook_alert

    captured: dict = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(requests, "post", _fake_post)

    # When: URL 설정됨
    sent = post_webhook_alert("https://hooks.example.com/x", unit="kca-collect.service", detail="boom")

    # Then
    assert sent is True
    assert captured["url"] == "https://hooks.example.com/x"
    assert "kca-collect.service" in captured["json"]["text"]
    assert "boom" in captured["json"]["text"]

    # When / Then: URL 비어있으면 네트워크 호출 없이 스킵
    assert post_webhook_alert("", unit="kca-collect.service") is False


def test_send_email_alert_skips_on_missing_credentials_and_sends_via_smtp_ssl(monkeypatch) -> None:
    import smtplib

    from src.tools import alerts

    # Given / When / Then: 자격증명 미완성이면 SMTP 시도 없이 False
    assert alerts.send_email_alert(gmail_user="", gmail_app_password="x", to_addr="a@b.com", unit="u") is False
    assert alerts.send_email_alert(gmail_user="a@b.com", gmail_app_password="", to_addr="a@b.com", unit="u") is False
    assert alerts.send_email_alert(gmail_user="a@b.com", gmail_app_password="x", to_addr="", unit="u") is False

    # Given: 자격증명 완비 + SMTP_SSL 가짜 구현
    sent: dict = {}

    class _FakeSMTP:
        def __init__(self, host, port, timeout):
            sent["host"] = host
            sent["port"] = port

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, user, password):
            sent["user"] = user

        def send_message(self, msg):
            sent["subject"] = msg["Subject"]
            sent["to"] = msg["To"]

    monkeypatch.setattr(smtplib, "SMTP_SSL", _FakeSMTP)

    # When
    ok = alerts.send_email_alert(
        gmail_user="bot@example.com", gmail_app_password="app-pw", to_addr="ops@example.com",
        unit="kca-predict.service", detail="lightgbm crashed",
    )

    # Then
    assert ok is True
    assert sent["host"] == "smtp.gmail.com"
    assert sent["user"] == "bot@example.com"
    assert sent["to"] == "ops@example.com"
    assert "kca-predict.service" in sent["subject"]


def test_dispatch_failure_alert_isolates_channel_failures(monkeypatch) -> None:
    from src.tools import alerts

    monkeypatch.setattr(alerts.settings, "ALERT_WEBHOOK_URL", "https://hooks.example.com/x", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_USER", "bot@example.com", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_APP_PASSWORD", "pw", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_TO", "ops@example.com", raising=False)

    def _boom_webhook(url, *, unit, detail=""):
        raise alerts.requests.RequestException("network down")

    # Given: 웹훅 채널은 예외, 이메일 채널은 성공
    monkeypatch.setattr(alerts, "post_webhook_alert", _boom_webhook)
    monkeypatch.setattr(alerts, "send_email_alert", lambda **kw: True)

    # When / Then: 실패가 웹훅에만 격리되고 이메일은 정상 시도/성공
    assert alerts.dispatch_failure_alert("kca-collect.service", detail="boom") == {"webhook": False, "email": True}

    def _boom_email(**kw):
        raise alerts.smtplib.SMTPException("auth failed")

    # Given: 반대로 웹훅은 성공, 이메일 채널이 예외
    monkeypatch.setattr(alerts, "post_webhook_alert", lambda url, *, unit, detail="": True)
    monkeypatch.setattr(alerts, "send_email_alert", _boom_email)

    # When / Then: 실패가 이메일에만 격리되고 웹훅은 정상 시도/성공
    assert alerts.dispatch_failure_alert("kca-predict.service", detail="boom") == {"webhook": True, "email": False}


def test_alerts_main_parses_unit_and_dispatches(monkeypatch) -> None:
    from src.tools import alerts

    captured: dict = {}

    def _fake_dispatch(unit, *, detail=""):
        captured["unit"] = unit
        captured["detail"] = detail
        return {"webhook": True, "email": False}

    monkeypatch.setattr(alerts, "dispatch_failure_alert", _fake_dispatch)

    # When
    alerts.main(["--unit", "kca-finalize-close.service", "--detail", "exit code 1"])

    # Then
    assert captured == {"unit": "kca-finalize-close.service", "detail": "exit code 1"}
