from __future__ import annotations


def test_audit_daily_completeness_flags_missing_steps(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.tools import daily_audit

    # Given: 아카이브에는 오늘 행이 있지만 결정 기록은 없는 상태
    monkeypatch.setattr(
        daily_audit,
        "fetch_archive_snapshot",
        lambda snapshot_date=None, **kw: pd.DataFrame({"종목코드": ["005930"]}),
    )
    monkeypatch.setattr(daily_audit.settings, "PAPER_DIR", tmp_path, raising=False)

    # When
    result = daily_audit.audit_daily_completeness("2026-09-10")

    # Then
    assert set(result) == {"archive", "minute_bars", "decision", "close_confirmed"}
    assert result["archive"] is True
    assert result["decision"] is False


def test_systemd_units_encode_persistence_and_timezone_policy() -> None:
    from pathlib import Path

    root = Path("deploy/systemd")

    # Given: 결정창에 묶인 타이머들은 지연 캐치업이 무의미하다
    # (kca-paper-entry는 고정 타이머가 아니라 finalize-close OnSuccess 체이닝이므로 제외)
    for name in ("kca-collect", "kca-predict", "kca-finalize-close"):
        text = (root / f"{name}.timer").read_text(encoding="utf-8")
        assert "Persistent=false" in text, f"{name} must not catch up outside the decision window"
        assert "Asia/Seoul" in text, f"{name} must pin KST explicitly"

    # And: 저녁 아카이브만 지연 캐치업이 유효하다
    evening = (root / "kca-archive-intraday.timer").read_text(encoding="utf-8")
    assert "Persistent=true" in evening
    assert "Asia/Seoul" in evening

    # And: 부팅 감사 유닛은 타이머가 아니라 부팅시 1회 서비스다
    audit = (root / "kca-daily-audit.service").read_text(encoding="utf-8")
    assert "Type=oneshot" in audit


def test_systemd_timers_align_with_decision_and_finalize_gates() -> None:
    import re
    from pathlib import Path

    from src.config.market_session import DECISION_WINDOW_END_HHMMSS, DECISION_WINDOW_START_HHMMSS

    def _hhmmss(unit: str) -> str:
        text = Path(f"deploy/systemd/{unit}.timer").read_text(encoding="utf-8")
        m = re.search(r"OnCalendar=.*?(\d{2}):(\d{2}):(\d{2})", text)
        assert m, f"no OnCalendar in {unit}"
        return "".join(m.groups())

    collect_hhmmss = _hhmmss("kca-collect")
    predict_hhmmss = _hhmmss("kca-predict")

    # collect는 반드시 결정창 안에서 발화해야 한다(그렇지 않으면 매일 RuntimeError)
    assert DECISION_WINDOW_START_HHMMSS <= collect_hhmmss <= DECISION_WINDOW_END_HHMMSS
    # predict는 collect 이후
    assert predict_hhmmss > collect_hhmmss

    # paper-entry는 고정 타이머가 아니라 finalize-close 성공에 이벤트로 체이닝된다
    assert not Path("deploy/systemd/kca-paper-entry.timer").exists()
    finalize_text = Path("deploy/systemd/kca-finalize-close.service").read_text(encoding="utf-8")
    assert "OnSuccess=kca-paper-entry.service" in finalize_text


def test_audit_or_skip_skips_non_trading_day(monkeypatch) -> None:
    from src.tools import daily_audit

    calls = {"n": 0}

    def _never(_date: str) -> dict[str, bool]:
        calls["n"] += 1
        return {"archive": True, "minute_bars": True, "decision": True, "close_confirmed": True}

    monkeypatch.setattr(daily_audit, "audit_daily_completeness", _never)

    # Given: 휴장일(토요일)
    monkeypatch.setattr(daily_audit, "is_krx_trading_day", lambda _d: False)

    # When / Then: 감사 자체를 수행하지 않는다(휴장일 MISSING 오탐 제거)
    assert daily_audit.audit_or_skip("2026-01-03") is None
    assert calls["n"] == 0

    # And: 거래일이면 기존 4키 결과를 그대로 반환한다
    monkeypatch.setattr(daily_audit, "is_krx_trading_day", lambda _d: True)
    result = daily_audit.audit_or_skip("2026-09-09")
    assert result is not None
    assert set(result) == {"archive", "minute_bars", "decision", "close_confirmed"}
    assert calls["n"] == 1


def test_daily_audit_audits_weekday_even_when_krx_calendar_is_unpublished(monkeypatch) -> None:
    from src.tools import daily_audit

    expected = {"archive": True, "minute_bars": True, "decision": False, "close_confirmed": False}
    monkeypatch.setattr(daily_audit, "audit_daily_completeness", lambda _d: dict(expected))
    # Given: KRX 지수 일별매매정보가 아직 미게시(1일 이상 지연) -> False 반환
    monkeypatch.setattr(daily_audit, "is_krx_trading_day", lambda _d: False)

    # When: 평일(2026-09-10, 목)
    result = daily_audit.audit_or_skip("2026-09-10")

    # Then: 침묵 스킵하지 않고 감사를 수행한다 (P0 미탐지 회귀 방지)
    assert result == expected

    # And: 주말(2026-09-12, 토)만 스킵
    assert daily_audit.audit_or_skip("2026-09-12") is None


def test_daily_audit_still_audits_when_calendar_lookup_fails(monkeypatch) -> None:
    from src.tools import daily_audit

    expected = {"archive": True, "minute_bars": False, "decision": False, "close_confirmed": False}
    monkeypatch.setattr(daily_audit, "audit_daily_completeness", lambda _d: dict(expected))

    def _boom(_date):
        raise RuntimeError("krx network down")

    # Given: KRX 달력 조회가 네트워크 장애로 실패
    monkeypatch.setattr(daily_audit, "is_krx_trading_day", _boom)

    # When / Then: 부가 정보 실패가 감사 자체를 막지 않는다 (장애 조기 발견 목적 유지)
    assert daily_audit.audit_or_skip("2026-09-10") == expected


def test_notify_if_missing_dispatches_alert_only_when_steps_missing(monkeypatch) -> None:
    from src.tools import daily_audit

    captured: dict = {}
    monkeypatch.setattr(
        "src.tools.alerts.dispatch_failure_alert",
        lambda unit, *, detail="": captured.update(unit=unit, detail=detail) or {"webhook": True, "email": True},
    )

    # When: 두 단계 누락
    missing = daily_audit._notify_if_missing(
        "2026-09-10",
        {"archive": True, "minute_bars": False, "decision": False, "close_confirmed": True},
    )

    # Then: 정렬된 누락 목록 + 얼러트 발송
    assert missing == ["decision", "minute_bars"]
    assert "2026-09-10" in captured["unit"]
    assert "decision" in captured["detail"]
    assert "minute_bars" in captured["detail"]

    # Given: 아무것도 누락 없음
    called = {"n": 0}
    monkeypatch.setattr(
        "src.tools.alerts.dispatch_failure_alert",
        lambda *a, **kw: called.__setitem__("n", called["n"] + 1) or {},
    )

    # When
    missing2 = daily_audit._notify_if_missing(
        "2026-09-10",
        {"archive": True, "minute_bars": True, "decision": True, "close_confirmed": True},
    )

    # Then: 얼러트 미발송
    assert missing2 == []
    assert called["n"] == 0
