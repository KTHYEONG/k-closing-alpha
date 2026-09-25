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
    monkeypatch.setattr(alerts.settings, "ALERT_RETRY_ATTEMPTS", 2, raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_RETRY_BACKOFF_SECONDS", 0.01, raising=False)

    def _boom_webhook(url, *, unit, detail="", subject=None):
        raise alerts.requests.RequestException("network down")

    # Given: 웹훅 채널은 예외, 이메일 채널은 성공
    monkeypatch.setattr(alerts, "post_webhook_alert", _boom_webhook)
    monkeypatch.setattr(alerts, "send_email_alert", lambda **kw: True)

    # When / Then: 실패가 웹훅에만 격리되고 이메일은 정상 시도/성공
    assert alerts.dispatch_failure_alert("kca-collect.service", detail="boom") == {"webhook": False, "email": True}

    def _boom_email(**kw):
        raise alerts.smtplib.SMTPException("auth failed")

    # Given: 반대로 웹훅은 성공, 이메일 채널이 예외
    monkeypatch.setattr(alerts, "post_webhook_alert", lambda url, *, unit, detail="", subject=None: True)
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
    monkeypatch.setattr(alerts, "drain_alert_outbox", lambda **kw: (0, 0))

    # When
    alerts.main(["--unit", "kca-finalize-close.service", "--detail", "exit code 1"])

    # Then
    assert captured == {"unit": "kca-finalize-close.service", "detail": "exit code 1"}


def test_dispatch_digest_sends_subject_and_body_and_isolates_channel_failures(monkeypatch) -> None:
    from src.tools import alerts

    monkeypatch.setattr(alerts.settings, "ALERT_WEBHOOK_URL", "https://hooks.example.com/x", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_USER", "bot@example.com", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_APP_PASSWORD", "pw", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_TO", "ops@example.com", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_RETRY_ATTEMPTS", 2, raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_RETRY_BACKOFF_SECONDS", 0.01, raising=False)
    sent: dict = {}

    class _FakeSMTP:
        def __init__(self, host, port, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def login(self, user, password):
            pass

        def send_message(self, msg):
            sent["subject"] = msg["Subject"]
            sent["body"] = msg.get_content()

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

    posted: dict = {}

    def _post(url, json, timeout):
        posted["text"] = json["text"]
        return _FakeResponse()

    monkeypatch.setattr(alerts.smtplib, "SMTP_SSL", _FakeSMTP)
    monkeypatch.setattr(alerts.requests, "post", _post)

    # When: 두 채널 모두 정상
    results = alerts.dispatch_digest("[KCA] 2026-09-14 일일점검 OK", "archive=OK")

    # Then
    assert results == {"webhook": True, "email": True}
    assert sent["subject"] == "[KCA] 2026-09-14 일일점검 OK"
    assert "archive=OK" in sent["body"]
    assert posted["text"].startswith("[KCA] 2026-09-14 일일점검 OK")

    # Given: 웹훅 장애
    def _down(url, json, timeout):
        raise alerts.requests.ConnectionError("down")

    monkeypatch.setattr(alerts.requests, "post", _down)

    # Then: 이메일은 여전히 발송된다
    assert alerts.dispatch_digest("s", "b") == {"webhook": False, "email": True}

    # Given: 반대로 SMTP 장애
    monkeypatch.setattr(alerts.requests, "post", _post)

    class _BrokenSMTP(_FakeSMTP):
        def login(self, user, password):
            raise alerts.smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(alerts.smtplib, "SMTP_SSL", _BrokenSMTP)

    # Then: 웹훅은 여전히 발송된다
    assert alerts.dispatch_digest("s", "b") == {"webhook": True, "email": False}


def test_sanitize_journal_tail_strips_ansi_carriage_returns_and_caps_length() -> None:
    from src.tools import alerts

    raw = (
        "\x1b[92m✅ 데이터 수집 완료\x1b[0m\n"
        "⏳ 10%\r⏳ 50%\r⏳ 100%\n"
        "\n"
        + "x" * 500
        + "\nValueError: real-time collection coverage 0.8843 below 0.99\n"
    )

    out = alerts.sanitize_journal_tail(raw)

    assert out.split("\n") == [
        "✅ 데이터 수집 완료",
        "⏳ 100%",
        "x" * alerts.ALERT_LINE_MAX_CHARS + "…",
        "ValueError: real-time collection coverage 0.8843 below 0.99",
    ]


def test_collect_unit_diagnostics_isolates_invocation_id_and_survives_command_failure() -> None:
    import subprocess

    from src.tools import alerts

    calls = []

    def _run_with_inv(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[0] == "systemctl":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="Result=exit-code\nExecMainStatus=1\nInvocationID=13f7ced75516425e9282aaf0973ce250\n",
                stderr="",
            )
        return subprocess.CompletedProcess(cmd, 0, stdout="\x1b[91mTraceback\x1b[0m\nValueError: boom\n", stderr="")

    # When: InvocationID 가 존재하는 경우
    status, journal = alerts.collect_unit_diagnostics("kca-collect.service", run=_run_with_inv)

    # Then: _SYSTEMD_INVOCATION_ID= 가 사용되고 단위 상태가 정확히 파싱됨
    assert status["Result"] == "exit-code"
    assert status["ExecMainStatus"] == "1"
    assert status["InvocationID"] == "13f7ced75516425e9282aaf0973ce250"
    assert "Traceback\nValueError: boom" in journal
    assert calls[0][0][:4] == ["systemctl", "--user", "show", "kca-collect.service"]
    assert calls[1][0][:3] == ["journalctl", "--user", "_SYSTEMD_INVOCATION_ID=13f7ced75516425e9282aaf0973ce250"]

    # Given: InvocationID 가 없는 경우 fallback 으로 -u unit -n 40 사용
    calls.clear()

    def _run_without_inv(cmd, **kwargs):
        calls.append((cmd, kwargs))
        if cmd[0] == "systemctl":
            return subprocess.CompletedProcess(cmd, 0, stdout="Result=exit-code\nExecMainStatus=1\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="fallback log\n", stderr="")

    status2, journal2 = alerts.collect_unit_diagnostics("kca-collect.service", run=_run_without_inv)
    assert calls[1][0][:4] == ["journalctl", "--user", "-u", "kca-collect.service"]
    assert "fallback log" in journal2

    # Given: 명령 실행 실패
    def _broken(cmd, **kwargs):
        if cmd[0] == "systemctl":
            raise FileNotFoundError("systemctl")
        raise subprocess.CalledProcessError(1, cmd)

    status_fail, journal_fail = alerts.collect_unit_diagnostics("kca-collect.service", run=_broken)
    assert "unavailable: FileNotFoundError" in status_fail["Result"]
    assert "journal unavailable: CalledProcessError" in journal_fail


def test_extract_failure_summary_identifies_python_traceback_and_location() -> None:
    from src.tools import alerts

    raw_journal = (
        "Finished kca-auction-close.service - KCA optional auction close capture.\n"
        "Starting kca-auction-close.service - KCA optional auction close capture...\n"
        "Traceback (most recent call last):\n"
        '  File "<frozen runpy>", line 198, in _run_module_as_main\n'
        '  File "/app/src/daily/auction_capture.py", line 96, in _resolve_roster\n'
        "    cohort = store.read_cohort(snapshot_date, available_by=now)\n"
        '  File "/app/src/data/capture_store.py", line 441, in read_cohort\n'
        "    raise FileNotFoundError(f\"no qualifying cohort: {snapshot_date!r}\")\n"
        "FileNotFoundError: no qualifying cohort: '2026-09-24'\n"
        '  File "/usr/local/lib/python3.11/asyncio/runners.py", line 190, in run\n'
    )
    status = {"Result": "exit-code", "ExecMainStatus": "1"}

    diag = alerts.extract_failure_summary(raw_journal, status)

    assert diag["reason"] == "FileNotFoundError: no qualifying cohort: '2026-09-24'"
    # stdlib/frozen 러너 대신 프로젝트 코드(src/...) 위치 우선 식별
    assert diag["location"] == "/app/src/data/capture_store.py:441 in read_cohort"
    assert "FileNotFoundError" in diag["context"]


def test_extract_failure_summary_identifies_structured_error_log() -> None:
    from src.tools import alerts

    raw_journal = (
        "2026-09-24 15:40:17,015 [INFO] 🚀 [Intraday 아카이브 시작] 대상일: 2026-09-24\n"
        "2026-09-24 15:40:17,461 [ERROR] [DATA] stage=intraday_archive status=ERROR reason=no qualifying cohort: '2026-09-24'\n"
        "kca-archive-intraday-regular.service: Main process exited, code=exited, status=1/FAILURE\n"
    )
    status = {"Result": "exit-code", "ExecMainStatus": "1"}

    diag = alerts.extract_failure_summary(raw_journal, status)

    assert "[ERROR]" in diag["reason"]
    assert "reason=no qualifying cohort: '2026-09-24'" in diag["reason"]


def test_extract_failure_summary_falls_back_to_systemctl_result_when_log_empty() -> None:
    from src.tools import alerts

    status = {"Result": "oom-kill", "ExecMainStatus": "137"}
    diag = alerts.extract_failure_summary("", status)

    assert diag["reason"] == "Process exited with oom-kill (status 137)"
    assert diag["location"] == ""
    assert diag["context"] == "(상세 로그 없음)"


def test_extract_failure_summary_prunes_noise_and_caps_context() -> None:
    from src.tools import alerts

    noisy_journal = (
        "2026-09-23 16:33:19,504 [INFO] [DATA] stage=intraday_replace date=2026-09-23 session=regular replaced=['420770', '424760']\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        '<frozen runpy>: line 88\n'
        "2026-09-24 15:40:17,461 [ERROR] [DATA] stage=intraday_archive status=ERROR reason=failed\n"
    )
    status = {"Result": "exit-code", "ExecMainStatus": "1"}

    diag = alerts.extract_failure_summary(noisy_journal, status)

    assert "replaced=" not in diag["context"]
    assert "━━━━" not in diag["context"]
    assert "<frozen " not in diag["context"]
    assert "[ERROR]" in diag["context"]


def test_format_failure_alert_builds_unified_card_and_subject() -> None:
    from src.tools import alerts

    status = {
        "Result": "exit-code",
        "ExecMainStatus": "1",
        "ExecMainStartTimestamp": "Thu 2026-09-24 15:40:13 KST",
        "ExecMainExitTimestamp": "Thu 2026-09-24 15:40:17 KST",
    }
    diag = {
        "reason": "FileNotFoundError: no qualifying cohort: '2026-09-24'",
        "location": "src/daily/auction_capture.py:96 in _resolve_roster",
        "context": "FileNotFoundError: no qualifying cohort: '2026-09-24'",
    }

    subject, body = alerts.format_failure_alert("kca-auction-close.service", status, diag)

    assert subject == "[kca] 🚨 유닛 실행 실패: kca-auction-close.service"
    assert "==================================================" in body
    assert "🚨 KCA 시스템 유닛 장애 알림 (kca-auction-close.service)" in body
    assert "• 실패 유닛: kca-auction-close.service" in body
    assert "• 종료 상태: exit-code (exit code: 1)" in body
    assert "• 발생 시각: Thu 2026-09-24 15:40:17 KST (시작: Thu 2026-09-24 15:40:13 KST)" in body
    assert "• 핵심 원인: FileNotFoundError: no qualifying cohort: '2026-09-24'" in body
    assert "• 발생 위치: src/daily/auction_capture.py:96 in _resolve_roster" in body
    assert "• 저널 확인: journalctl --user -u kca-auction-close.service -n 50 --no-pager" in body
    assert "[핵심 에러 로그]" in body


def test_alerts_main_attaches_formatted_failure_alert_when_detail_absent(monkeypatch) -> None:
    from src.tools import alerts

    captured = {}

    def _fake_diag(unit):
        return (
            {"Result": "exit-code", "ExecMainStatus": "1", "ExecMainStartTimestamp": "T1", "ExecMainExitTimestamp": "T2"},
            "ValueError: test boom",
        )

    def _fake_dispatch(unit, *, detail="", subject=None):
        captured["unit"] = unit
        captured["detail"] = detail
        captured["subject"] = subject
        return {"webhook": False, "email": True}

    monkeypatch.setattr(alerts, "collect_unit_diagnostics", _fake_diag)
    monkeypatch.setattr(alerts, "dispatch_failure_alert", _fake_dispatch)
    monkeypatch.setattr(alerts, "drain_alert_outbox", lambda **kw: (0, 0))

    # When: systemd OnFailure 기본 호출(--detail 없음)
    alerts.main(["--unit", "kca-collect.service"])

    # Then
    assert captured["unit"] == "kca-collect.service"
    assert captured["subject"] == "[kca] 🚨 유닛 실행 실패: kca-collect.service"
    assert "🚨 KCA 시스템 유닛 장애 알림" in captured["detail"]
    assert "ValueError: test boom" in captured["detail"]

    # When: 명시적 detail
    captured.clear()
    alerts.main(["--unit", "kca-collect.service", "--detail", "manual note"])

    # Then
    assert captured["unit"] == "kca-collect.service"
    assert captured["detail"] == "manual note"


def test_parse_systemctl_show_handles_empty_and_malformed_lines() -> None:
    from src.tools import alerts

    raw = "\n  \nKey1=Val1\nInvalidLineWithoutEquals\nKey2 = Val2 \n\n"
    res = alerts.parse_systemctl_show(raw)
    assert res == {"Key1": "Val1", "Key2": "Val2"}


def test_extract_failure_summary_identifies_critical_or_fatal_log() -> None:
    from src.tools import alerts

    raw_journal = "2026-09-24 12:00:00 [CRITICAL] kernel out of resources\n"
    diag = alerts.extract_failure_summary(raw_journal, {"Result": "exit-code", "ExecMainStatus": "1"})
    assert "[CRITICAL] kernel out of resources" in diag["reason"]

    raw_journal2 = "2026-09-24 12:00:00 [FATAL] aborting process\n"
    diag2 = alerts.extract_failure_summary(raw_journal2, {"Result": "exit-code", "ExecMainStatus": "1"})
    assert "[FATAL] aborting process" in diag2["reason"]


def test_dispatch_failure_alert_propagates_custom_subject(monkeypatch) -> None:
    from src.tools import alerts

    passed = {}

    def _fake_post(url, *, unit, detail="", subject=None):
        passed["webhook_subject"] = subject
        return True

    def _fake_email(*, gmail_user, gmail_app_password, to_addr, unit, detail="", subject=None):
        passed["email_subject"] = subject
        return True

    monkeypatch.setattr(alerts, "post_webhook_alert", _fake_post)
    monkeypatch.setattr(alerts, "send_email_alert", _fake_email)

    alerts.dispatch_failure_alert("kca-predict.service", detail="d", subject="[kca] 🚨 유닛 실행 실패: kca-predict.service")
    assert passed["webhook_subject"] == "[kca] 🚨 유닛 실행 실패: kca-predict.service"
    assert passed["email_subject"] == "[kca] 🚨 유닛 실행 실패: kca-predict.service"


def test_collect_unit_diagnostics_falls_back_when_invocation_id_fails() -> None:
    import subprocess

    from src.tools import alerts

    def _run(cmd, **kwargs):
        if cmd[0] == "systemctl":
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="Result=exit-code\nExecMainStatus=1\nInvocationID=fail_id\n",
                stderr="",
            )
        if "_SYSTEMD_INVOCATION_ID=fail_id" in cmd[2]:
            raise subprocess.CalledProcessError(1, cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="recovered fallback log\n", stderr="")

    status, journal = alerts.collect_unit_diagnostics("kca-collect.service", run=_run)
    assert status["InvocationID"] == "fail_id"
    assert "recovered fallback log" in journal



def _fast_retry(monkeypatch) -> None:
    from src.tools import alerts

    monkeypatch.setattr(alerts.settings, "ALERT_RETRY_ATTEMPTS", 3, raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_RETRY_BACKOFF_SECONDS", 5.0, raising=False)


def _empty_alert_creds(monkeypatch) -> None:
    from src.tools import alerts

    monkeypatch.setattr(alerts.settings, "ALERT_WEBHOOK_URL", "", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_USER", "", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_APP_PASSWORD", "", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_TO", "", raising=False)


def test_deliver_with_retry_delivers_after_transient_failure() -> None:
    import smtplib

    from src.tools import alerts

    calls: list = []
    sleeps: list = []

    def _send():
        calls.append(1)
        if len(calls) == 1:
            raise smtplib.SMTPException("temporary")
        return True

    outcome = alerts.deliver_with_retry(
        _send, channel="email", attempts=3, backoff_seconds=5.0, sleep_fn=sleeps.append
    )

    assert outcome is alerts.ChannelOutcome.DELIVERED
    assert len(calls) == 2
    assert sleeps == [5.0]
    assert not alerts.alert_outbox_dir().exists()


def test_deliver_with_retry_skips_retry_when_unconfigured() -> None:
    from src.tools import alerts

    calls: list = []
    sleeps: list = []

    outcome = alerts.deliver_with_retry(
        lambda: calls.append(1) or False, channel="webhook", attempts=3, backoff_seconds=5.0, sleep_fn=sleeps.append
    )

    assert outcome is alerts.ChannelOutcome.NOT_CONFIGURED
    assert len(calls) == 1
    assert sleeps == []


def test_deliver_with_retry_fails_after_exhausting_attempts(caplog) -> None:
    import logging
    import smtplib

    from src.tools import alerts

    sleeps: list = []

    def _always_fail():
        raise smtplib.SMTPException("down")

    with caplog.at_level(logging.WARNING, logger=alerts.logger.name):
        outcome = alerts.deliver_with_retry(
            _always_fail, channel="email", attempts=3, backoff_seconds=5.0, sleep_fn=sleeps.append
        )

    assert outcome is alerts.ChannelOutcome.FAILED
    assert sleeps == [5.0, 10.0]
    assert any("attempt=1/3 status=RETRY" in rec.message for rec in caplog.records)
    assert any("attempt=3/3 status=FAILED" in rec.message for rec in caplog.records)


def test_dispatch_failure_alert_persists_total_failure(monkeypatch, caplog) -> None:
    import json
    import logging
    import smtplib

    from src.tools import alerts

    _fast_retry(monkeypatch)
    monkeypatch.setattr(alerts.settings, "ALERT_WEBHOOK_URL", "", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_USER", "bot@example.com", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_APP_PASSWORD", "pw", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_TO", "ops@example.com", raising=False)
    real_retry = alerts.deliver_with_retry
    sleeps: list = []
    monkeypatch.setattr(alerts, "deliver_with_retry", lambda send, **kw: real_retry(send, sleep_fn=sleeps.append, **kw))

    def _boom(**kw):
        raise smtplib.SMTPException("auth failed")

    monkeypatch.setattr(alerts, "send_email", _boom)

    with caplog.at_level(logging.ERROR, logger=alerts.logger.name):
        results = alerts.dispatch_failure_alert("kca-collect.service", detail="boom")

    assert results == {"webhook": False, "email": False}
    assert sleeps == [5.0, 10.0]
    files = sorted(alerts.alert_outbox_dir().glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["kind"] == "failure"
    assert "kca-collect.service" in payload["subject"]
    assert "boom" in payload["body"]
    assert any("status=UNDELIVERED" in rec.message for rec in caplog.records)


def test_dispatch_digest_persists_when_nothing_configured(monkeypatch) -> None:
    import json

    from src.tools import alerts

    _empty_alert_creds(monkeypatch)
    _fast_retry(monkeypatch)
    calls = {"webhook": 0, "email": 0}

    def _no_webhook(url, text):
        calls["webhook"] += 1
        return False

    def _no_email(**kw):
        calls["email"] += 1
        return False

    monkeypatch.setattr(alerts, "post_webhook_text", _no_webhook)
    monkeypatch.setattr(alerts, "send_email", _no_email)

    results = alerts.dispatch_digest("s", "b")

    assert results == {"webhook": False, "email": False}
    assert calls == {"webhook": 1, "email": 1}
    files = sorted(alerts.alert_outbox_dir().glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert (payload["subject"], payload["body"], payload["kind"]) == ("s", "b", "digest")


def _write_outbox_file(box, name, subject, body="x-body") -> None:
    import json

    (box).mkdir(parents=True, exist_ok=True)
    (box / name).write_text(
        json.dumps({"subject": subject, "body": body, "kind": "digest", "enqueued_at": "2020-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )


def test_drain_alert_outbox_delivers_oldest_first_and_stops(tmp_path) -> None:
    from src.tools import alerts

    box = tmp_path / "outbox"
    _write_outbox_file(box, "20200101T000000000000_aaaaaaaa.json", "first")
    _write_outbox_file(box, "20200101T000001000000_bbbbbbbb.json", "second")
    _write_outbox_file(box, "20200101T000002000000_cccccccc.json", "third")
    sent: list = []

    def _send(subject, body):
        sent.append(subject)
        return {"webhook": subject != "[지연전송] second", "email": False}

    delivered, remaining = alerts.drain_alert_outbox(outbox=box, send_fn=_send)

    assert (delivered, remaining) == (1, 2)
    assert sent == ["[지연전송] first", "[지연전송] second"]
    assert not (box / "20200101T000000000000_aaaaaaaa.json").exists()
    assert (box / "20200101T000001000000_bbbbbbbb.json").exists()
    assert (box / "20200101T000002000000_cccccccc.json").exists()


def test_drain_alert_outbox_respects_max_items(tmp_path) -> None:
    from src.tools import alerts

    box = tmp_path / "outbox"
    _write_outbox_file(box, "20200101T000000000000_aaaaaaaa.json", "first")
    _write_outbox_file(box, "20200101T000001000000_bbbbbbbb.json", "second")
    _write_outbox_file(box, "20200101T000002000000_cccccccc.json", "third")

    delivered, remaining = alerts.drain_alert_outbox(
        outbox=box, max_items=2, send_fn=lambda subject, body: {"webhook": True, "email": False}
    )

    assert (delivered, remaining) == (2, 1)


def test_drain_alert_outbox_marks_redelivery(tmp_path) -> None:
    from src.tools import alerts

    box = tmp_path / "outbox"
    _write_outbox_file(box, "20200101T000000000000_aaaaaaaa.json", "s", body="b")
    sent: dict = {}

    def _send(subject, body):
        sent["subject"] = subject
        sent["body"] = body
        return {"webhook": True, "email": False}

    assert alerts.drain_alert_outbox(outbox=box, send_fn=_send) == (1, 0)
    assert sent["subject"].startswith("[지연전송] ")
    assert "original_enqueued_at=2020-01-01T00:00:00+00:00" in sent["body"]


def test_drain_alert_outbox_empty_when_missing(tmp_path) -> None:
    from src.tools import alerts

    def _never(subject, body):
        raise AssertionError("missing outbox must not attempt delivery")

    assert alerts.drain_alert_outbox(outbox=tmp_path / "nope", send_fn=_never) == (0, 0)


def test_drain_alert_outbox_keeps_corrupt_file(tmp_path, caplog) -> None:
    import logging

    from src.tools import alerts

    box = tmp_path / "outbox"
    box.mkdir()
    (box / "20200101T000000000000_aaaaaaaa.json").write_text("not json{{{", encoding="utf-8")

    def _never(subject, body):
        raise AssertionError("corrupt file must stop the drain")

    with caplog.at_level(logging.WARNING, logger=alerts.logger.name):
        assert alerts.drain_alert_outbox(outbox=box, send_fn=_never) == (0, 1)
    assert any("CORRUPT_OUTBOX" in rec.message for rec in caplog.records)

    box2 = tmp_path / "outbox2"
    box2.mkdir()
    (box2 / "20200101T000000000000_aaaaaaaa.json").write_text('["not", "a", "dict"]', encoding="utf-8")
    assert alerts.drain_alert_outbox(outbox=box2, send_fn=_never) == (0, 1)


def test_enqueue_undelivered_is_atomic(monkeypatch, tmp_path) -> None:
    import pytest

    from src.tools import alerts

    box = tmp_path / "outbox"

    def _crash(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(alerts.os, "replace", _crash)

    with pytest.raises(OSError, match="disk gone"):
        alerts.enqueue_undelivered("s", "b", kind="digest", outbox=box)

    assert list(box.glob("*.json")) == []
    assert list(box.glob("*.tmp")) == []


def test_alert_logs_never_leak_credentials(monkeypatch, caplog) -> None:
    import logging
    import smtplib

    from src.tools import alerts

    webhook_url = "https://hooks.example.com/secret-token-abc"
    gmail_pw = "super-secret-app-password"
    monkeypatch.setattr(alerts.settings, "ALERT_WEBHOOK_URL", webhook_url, raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_USER", "bot@example.com", raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_APP_PASSWORD", gmail_pw, raising=False)
    monkeypatch.setattr(alerts.settings, "ALERT_GMAIL_TO", "ops@example.com", raising=False)
    _fast_retry(monkeypatch)

    def _boom_webhook(url, text):
        raise alerts.requests.RequestException(f"POST {url} connection refused")

    def _boom_email(**kw):
        raise smtplib.SMTPException(f"login failed for {kw.get('gmail_user')}")

    monkeypatch.setattr(alerts, "post_webhook_text", _boom_webhook)
    monkeypatch.setattr(alerts, "send_email", _boom_email)

    with caplog.at_level(logging.WARNING, logger=alerts.logger.name):
        results = alerts.dispatch_digest("subject-line", "body-line")

    assert results == {"webhook": False, "email": False}
    assert webhook_url not in caplog.text
    assert gmail_pw not in caplog.text


def test_persist_undelivered_survives_outbox_failure(monkeypatch, caplog) -> None:
    import logging

    from src.tools import alerts

    _empty_alert_creds(monkeypatch)
    _fast_retry(monkeypatch)

    def _boom(*args, **kwargs):
        raise OSError("read-only fs")

    monkeypatch.setattr(alerts, "enqueue_undelivered", _boom)

    with caplog.at_level(logging.WARNING, logger=alerts.logger.name):
        results = alerts.dispatch_digest("s", "b")

    assert results == {"webhook": False, "email": False}
    assert any("OUTBOX_FAILED" in rec.message for rec in caplog.records)
    assert any("UNDELIVERED" in rec.message for rec in caplog.records)


def test_alerts_main_drains_before_dispatch(monkeypatch) -> None:
    from src.tools import alerts

    order: list = []

    def _drain(**kw):
        order.append("drain")
        assert kw.get("max_items") == alerts.settings.ALERT_OUTBOX_MAX_DRAIN
        return (0, 0)

    def _dispatch(unit, *, detail="", subject=None):
        order.append("dispatch")
        return {"webhook": True, "email": False}

    monkeypatch.setattr(alerts, "drain_alert_outbox", _drain)
    monkeypatch.setattr(alerts, "dispatch_failure_alert", _dispatch)
    monkeypatch.setattr(alerts, "collect_unit_diagnostics", lambda unit: ({}, ""))

    alerts.main(["--unit", "krx-host-backup.service"])

    assert order == ["drain", "dispatch"]


def test_alerts_main_survives_drain_failure(monkeypatch, caplog) -> None:
    import logging

    from src.tools import alerts

    def _boom(**kw):
        raise OSError("outbox unreadable")

    captured: dict = {}

    def _dispatch(unit, *, detail="", subject=None):
        captured["unit"] = unit
        return {"webhook": False, "email": False}

    monkeypatch.setattr(alerts, "drain_alert_outbox", _boom)
    monkeypatch.setattr(alerts, "dispatch_failure_alert", _dispatch)
    monkeypatch.setattr(alerts, "collect_unit_diagnostics", lambda unit: ({}, ""))

    with caplog.at_level(logging.WARNING, logger=alerts.logger.name):
        alerts.main(["--unit", "kca-collect.service"])

    assert captured == {"unit": "kca-collect.service"}
    assert any("DRAIN_FAILED" in rec.message for rec in caplog.records)
