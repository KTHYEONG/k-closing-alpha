from src.tools.alerts import alert_credential_env_file as _real_alert_credential_env_file


def test_dispatch_page_sends_subject_and_body_and_isolates_channel_failures(monkeypatch) -> None:
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
    results = alerts.dispatch_page("[KCA] 2026-09-14 일일점검 OK", "archive=OK")

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
    assert alerts.dispatch_page("s", "b") == {"webhook": False, "email": True}

    # Given: 반대로 SMTP 장애
    monkeypatch.setattr(alerts.requests, "post", _post)

    class _BrokenSMTP(_FakeSMTP):
        def login(self, user, password):
            raise alerts.smtplib.SMTPAuthenticationError(535, b"bad credentials")

    monkeypatch.setattr(alerts.smtplib, "SMTP_SSL", _BrokenSMTP)

    # Then: 웹훅은 여전히 발송된다
    assert alerts.dispatch_page("s", "b") == {"webhook": True, "email": False}


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


def test_dispatch_page_persists_when_nothing_configured(monkeypatch) -> None:
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

    results = alerts.dispatch_page("s", "b")

    assert results == {"webhook": False, "email": False}
    assert calls == {"webhook": 1, "email": 1}
    files = sorted(alerts.alert_outbox_dir().glob("*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert (payload["subject"], payload["body"], payload["kind"]) == ("s", "b", "page")


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
    import os
    import pytest

    from src.tools import alerts

    box = tmp_path / "outbox"

    def _crash(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(os, "replace", _crash)

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
        results = alerts.dispatch_page("subject-line", "body-line")

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
        results = alerts.dispatch_page("s", "b")

    assert results == {"webhook": False, "email": False}
    assert any("OUTBOX_FAILED" in rec.message for rec in caplog.records)
    assert any("UNDELIVERED" in rec.message for rec in caplog.records)


def test_sanitize_journal_tail_masks_secret_before_truncation() -> None:
    from src.tools import alerts

    secret = "Q" * 90 + "W" * 90
    out = alerts.sanitize_journal_tail("x" * 250 + secret, secret_values={secret})
    assert all(secret[i : i + 8] not in out for i in range(len(secret) - 7))
    assert len(out) <= alerts.ALERT_LINE_MAX_CHARS + 1


def test_sanitize_journal_tail_masks_pattern_without_configured_secrets() -> None:
    from src.tools import alerts

    raw = (
        "requests.exceptions.HTTPError: 401 for url: "
        "https://opendart.fss.or.kr/api/list.json?crtfc_key=ABCDEF0123456789&page_no=1"
    )
    out = alerts.sanitize_journal_tail(raw)
    assert "ABCDEF0123456789" not in out
    assert "page_no=1" in out


def test_dispatch_page_masks_before_outbox(monkeypatch) -> None:
    from src.tools import alerts

    _empty_alert_creds(monkeypatch)
    _fast_retry(monkeypatch)

    alerts.dispatch_page("[KCA] digest", "context Authorization: Bearer tok.abc.def")

    files = sorted(alerts.alert_outbox_dir().glob("*.json"))
    assert len(files) == 1
    stored = files[0].read_text(encoding="utf-8")
    assert "Bearer ***" in stored
    assert "tok.abc.def" not in stored


def test_drain_alert_outbox_masks_legacy_entries(monkeypatch, tmp_path) -> None:
    import json

    from src.tools import alerts

    monkeypatch.setattr(alerts.settings, "KIS_APP_SECRET", "legacy-secret-0123456789", raising=False)
    box = tmp_path / "outbox"
    box.mkdir()
    legacy = box / "20260901T000000000000_deadbeef.json"
    legacy.write_text(
        json.dumps({"subject": "s legacy-secret-0123456789", "body": "b legacy-secret-0123456789", "kind": "digest"}),
        encoding="utf-8",
    )
    received: list[tuple[str, str]] = []

    def _send(subject: str, body: str) -> dict[str, bool]:
        received.append((subject, body))
        return {"webhook": True, "email": False}

    delivered, remaining = alerts.drain_alert_outbox(outbox=box, send_fn=_send)

    assert (delivered, remaining) == (1, 0)
    assert received
    assert all("legacy-secret-0123456789" not in part for pair in received for part in pair)
    assert not legacy.exists()


def test_alert_secret_values_read_credential_env_file_under_base_dir(monkeypatch, tmp_path) -> None:
    from src.tools import alerts

    monkeypatch.setattr(alerts.settings, "BASE_DIR", tmp_path, raising=False)
    monkeypatch.setattr(alerts, "alert_credential_env_file", _real_alert_credential_env_file)
    (tmp_path / ".env").write_text("KIS_DATA_3_APP_SECRET=file-pool-secret-0003\nKIS_DATA_SLOTS=3\n", encoding="utf-8")

    assert alerts.alert_credential_env_file() == tmp_path / ".env"
    assert "file-pool-secret-0003" in alerts._alert_secret_values()
    assert "file-pool-secret-0003" not in alerts.redact_for_egress("boom file-pool-secret-0003")


def test_digest_receipt_masks_before_truncation_and_omits_body(monkeypatch, caplog) -> None:
    import logging

    from src.tools import alerts

    secret = "receipt-secret-" + "z" * 100
    monkeypatch.setattr(alerts.settings, "KIS_APP_SECRET", secret, raising=False)
    delivered = []

    def deliver(subject, body):
        delivered.append((subject, body))
        return {"webhook": True, "email": False}

    monkeypatch.setattr(alerts, "_deliver_now", deliver)
    subject = "[kca] " + secret + "x" * 100
    with caplog.at_level(logging.INFO, logger=alerts.logger.name):
        result = alerts.dispatch_page(subject, "private-body-content")
    assert result == {"webhook": True, "email": False}
    receipts = [rec.message for rec in caplog.records if "status=SENT" in rec.message]
    assert receipts == [
        "[SYS] stage=alert_delivery status=SENT kind=page webhook=True email=False subject_key=" + delivered[0][0][:80]
    ]
    assert "***" in receipts[0]
    assert secret not in caplog.text and "z" * 8 not in caplog.text
    assert "private-body-content" not in caplog.text
    assert not list(alerts.alert_outbox_dir().glob("*.json"))


def test_digest_undelivered_receipt_keeps_outbox_path(monkeypatch, caplog) -> None:
    import json
    import logging

    from src.tools import alerts

    monkeypatch.setattr(alerts, "_deliver_now", lambda *args: {"webhook": False, "email": False})
    with caplog.at_level(logging.INFO, logger=alerts.logger.name):
        assert alerts.dispatch_page("receipt-title", "receipt-body") == {"webhook": False, "email": False}
    files = list(alerts.alert_outbox_dir().glob("*.json"))
    assert len(files) == 1
    assert json.loads(files[0].read_text())["kind"] == "page"
    assert any(rec.levelno == logging.ERROR and "status=UNDELIVERED" in rec.message for rec in caplog.records)
    assert not any("status=SENT" in rec.message for rec in caplog.records)
