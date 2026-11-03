"""OnFailure routing (path A), page transport and the silent digest record."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.tools import alerts
from src.tools.incident_pager import IncidentPager


def _kst(hh: int, mm: int = 0, day: int = 3) -> datetime:
    return datetime(2026, 11, day, hh, mm, tzinfo=ZoneInfo("Asia/Seoul"))


def _capture_pager(tmp_path: Path, sent: list[tuple[str, str]]) -> IncidentPager:
    def _send(subject: str, body: str) -> dict[str, bool]:
        sent.append((subject, body))
        return {"webhook": False, "email": True}

    return IncidentPager(ledger_path=tmp_path / "incidents.json", send_fn=_send)


def _no_remediation(key: str, first_seen: datetime, now: datetime) -> bool:
    return False


def _no_action(key: str) -> str | None:
    return None


def _diag(unit: str) -> tuple[dict[str, str], str]:
    return ({"Result": "exit-code", "ExecMainStatus": "1"}, "TimeoutError: krx read timed out")


def _silent_rows() -> list[dict[str, str]]:
    path = Path(alerts.settings.DATA_DIR) / "logs" / "alerts" / "silent" / "2026-11-03.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _never_deliver(subject: str, body: str) -> dict[str, bool]:
    raise AssertionError("no mail expected")


def test_post_webhook_text_and_send_email_skip_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    import smtplib

    assert alerts.post_webhook_text("", "x") is False
    assert alerts.send_email(gmail_user="", gmail_app_password="x", to_addr="a@b.com", subject="s", body="b") is False
    sent: dict[str, str] = {}

    class _FakeSMTP:
        def __init__(self, host: str, port: int, timeout: float) -> None:
            sent["host"] = host

        def __enter__(self) -> _FakeSMTP:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def login(self, user: str, password: str) -> None:
            sent["user"] = user

        def send_message(self, msg: object) -> None:
            sent["subject"] = str(msg["Subject"])  # type: ignore[index]

    monkeypatch.setattr(smtplib, "SMTP_SSL", _FakeSMTP)
    assert alerts.send_email(
        gmail_user="bot@example.com", gmail_app_password="pw", to_addr="ops@example.com", subject="[긴급][kca] x", body="b"
    )
    assert sent == {"host": "smtp.gmail.com", "user": "bot@example.com", "subject": "[긴급][kca] x"}


def test_dispatch_page_masks_channels_and_persists_undelivered_as_page(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(alerts.settings, "KIS_APP_SECRET", "kis-secret-0123456789", raising=False)
    delivered: list[tuple[str, str]] = []

    def _deliver(subject: str, body: str) -> dict[str, bool]:
        delivered.append((subject, body))
        return {"webhook": True, "email": True}

    monkeypatch.setattr(alerts, "_deliver_now", _deliver)
    assert alerts.dispatch_page("[긴급][kca] kis-secret-0123456789", "body appkey=RAWKEY12345") == {"webhook": True, "email": True}
    assert all("kis-secret-0123456789" not in s and "RAWKEY12345" not in b for s, b in delivered)

    monkeypatch.setenv("KIS_DATA_1_APP_SECRET", "pool-secret-value-001")
    monkeypatch.setattr(alerts, "_deliver_now", lambda s, b: {"webhook": False, "email": False})
    with caplog.at_level(logging.ERROR, logger=alerts.logger.name):
        alerts.dispatch_page("p pool-secret-value-001", "b pool-secret-value-001")
    files = sorted(alerts.alert_outbox_dir().glob("*.json"))
    assert len(files) == 1
    stored = files[0].read_text(encoding="utf-8")
    assert "pool-secret-value-001" not in stored
    assert json.loads(stored)["kind"] == "page"
    assert any("status=UNDELIVERED" in rec.getMessage() for rec in caplog.records)


def test_record_digest_writes_masked_daily_jsonl_without_mail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(alerts.settings, "KIS_APP_SECRET", "kis-secret-0123456789", raising=False)
    monkeypatch.setattr(alerts, "_deliver_now", _never_deliver)

    result = alerts.record_digest("[kca] 경고 kis-secret-0123456789", "line", now=_kst(21, 20), root=tmp_path)

    assert result == {"recorded": True}
    rows = [json.loads(line) for line in (tmp_path / "2026-11-03.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert "kis-secret-0123456789" not in rows[0]["subject"]
    blocked = tmp_path / "file"
    blocked.write_text("x", encoding="utf-8")
    assert alerts.record_digest("s", "b", now=_kst(21, 20), root=blocked) == {"recorded": False}
    assert alerts.record_digest("s", "b")["recorded"] is True


@pytest.mark.parametrize(
    "unit",
    [
        "kca-price-ingest.service",
        "kca-auction-close.service",
        "kca-paper-entry.service",
        "kca-retrain.service",
        "krx-host-backup.service",
        "vps-image-gc.service",
        "crypto-pilot-backup.service",
    ],
)
def test_handle_unit_failure_silent_units_never_mail(tmp_path: Path, unit: str) -> None:
    sent: list[tuple[str, str]] = []

    status = alerts.handle_unit_failure(
        unit, now=_kst(8, 31), pager=_capture_pager(tmp_path, sent), pending=_no_remediation, auto_action=_no_action, diagnostics=_diag
    )

    assert status == "SILENT"
    assert sent == []
    rows = _silent_rows()
    assert rows[0]["status"] == "SILENT"
    assert rows[0]["source"] == f"failed_unit:{alerts.unit_base_name(unit)}.service"
    assert "TimeoutError" in rows[0]["reason"]


def test_handle_unit_failure_critical_pages_once_per_day_with_three_line_body(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    pager = _capture_pager(tmp_path, sent)

    def _route(unit: str, now: datetime) -> str:
        return alerts.handle_unit_failure(
            unit, now=now, pager=pager, pending=_no_remediation, auto_action=_no_action, diagnostics=_diag
        )

    assert _route("kca-order-exit.service", _kst(8, 45)) == "SENT"
    assert _route("kca-order-exit.service", _kst(9, 5)) == "DEDUPED"
    assert _route("kse-alert@kse-evening.service.service", _kst(8, 50)) == "SENT"
    assert _route("krx-deferred-recreate.service", _kst(22, 30)) == "SENT"
    assert _route("kca-order-exit.service", _kst(8, 45, day=4)) == "SENT"

    assert [s for s, _ in sent] == [
        "[긴급][kca] 시가 매도 주문 실패",
        "[긴급][kca] kse 긴급 알림 전달 실패",
        "[긴급][kca] krx 수집기 재생성 실패",
        "[긴급][kca] 시가 매도 주문 실패",
    ]
    lines = sent[0][1].splitlines()
    assert [line.split(":", 1)[0] for line in lines] == ["무슨 일", "자동 조치", "지금 할 일"]
    assert "journalctl" not in sent[0][1]
    assert "TimeoutError" not in sent[0][1]


def test_handle_unit_failure_defers_while_remediation_pending(tmp_path: Path) -> None:
    sent: list[tuple[str, str]] = []
    pager = _capture_pager(tmp_path, sent)
    pending_calls: list[str] = []

    def _pending(key: str, first_seen: datetime, now: datetime) -> bool:
        pending_calls.append(key)
        return True

    status = alerts.handle_unit_failure(
        "kca-kis-token-warmup.service", now=_kst(7, 25), pager=pager, pending=_pending, auto_action=_no_action, diagnostics=_diag
    )

    assert status == "DEFERRED"
    assert sent == []
    assert pending_calls == ["failed_unit:kca-kis-token-warmup.service"]
    assert pager.first_seen("failed_unit:kca-kis-token-warmup.service") == _kst(7, 25)
    assert _silent_rows()[0]["status"] == "DEFERRED"

    status = alerts.handle_unit_failure(
        "kca-kis-token-warmup.service",
        now=_kst(8, 30),
        pager=pager,
        pending=_no_remediation,
        auto_action=lambda key: "자동 재실행 2회를 시도했지만 해결되지 않았습니다.",
        diagnostics=_diag,
    )
    assert status == "SENT"
    assert "자동 조치: 자동 재실행 2회를 시도했지만 해결되지 않았습니다." in sent[0][1]


def test_alerts_main_drains_then_routes_without_mail_for_silent_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []

    def _drain(**kw: object) -> tuple[int, int]:
        order.append("drain")
        assert kw.get("max_items") == alerts.settings.ALERT_OUTBOX_MAX_DRAIN
        return (0, 0)

    def _diagnostics(unit: str) -> tuple[dict[str, str], str]:
        order.append("diag")
        return ({}, "")

    monkeypatch.setattr(alerts, "drain_alert_outbox", _drain)
    monkeypatch.setattr(alerts, "collect_unit_diagnostics", _diagnostics)
    monkeypatch.setattr(alerts, "_deliver_now", _never_deliver)

    alerts.main(["--unit", "kca-price-ingest.service"])

    assert order == ["drain", "diag"]


def test_alerts_main_survives_drain_failure_and_pages_critical(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    def _boom(**kw: object) -> tuple[int, int]:
        raise OSError("outbox unreadable")

    sent: list[str] = []

    def _deliver(subject: str, body: str) -> dict[str, bool]:
        sent.append(subject)
        return {"webhook": False, "email": True}

    monkeypatch.setattr(alerts, "drain_alert_outbox", _boom)
    monkeypatch.setattr(alerts, "collect_unit_diagnostics", lambda unit: ({}, ""))
    monkeypatch.setattr(alerts, "_deliver_now", _deliver)

    with caplog.at_level(logging.WARNING, logger=alerts.logger.name):
        alerts.main(["--unit", "kca-predict.service"])

    assert sent == ["[긴급][kca] 오늘 매수 결정 생성 실패"]
    assert any("DRAIN_FAILED" in rec.message for rec in caplog.records)
