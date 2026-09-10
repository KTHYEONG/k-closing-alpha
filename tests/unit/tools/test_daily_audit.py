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
