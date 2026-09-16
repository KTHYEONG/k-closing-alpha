from __future__ import annotations


def test_systemd_units_encode_persistence_and_timezone_policy() -> None:
    from pathlib import Path

    root = Path("deploy/systemd")

    # Given: 결정창에 묶인 타이머들은 지연 캐치업이 무의미하다
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

    # paper-entry는 finalize-close 종료(성공/실패 무관) ExecStopPost 체이닝이 1차 경로이고,
    # 체이닝이 조용히 끊기는 경우를 대비한 독립 백스톱 타이머(kca-paper-entry.timer)가 보증 경로다
    finalize_lines = Path("deploy/systemd/kca-finalize-close.service").read_text(encoding="utf-8").splitlines()
    assert "ExecStopPost=/usr/bin/systemctl --user start --no-block kca-paper-entry.service" in finalize_lines
    assert not any(line.startswith("OnSuccess=") for line in finalize_lines)


def test_audit_daily_completeness_reports_all_steps_from_topk_log_and_fills(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.tools import daily_audit

    parquet_dir = tmp_path / "parquet"
    paper_dir = tmp_path / "paper"
    parquet_dir.mkdir()
    paper_dir.mkdir()
    monkeypatch.setattr(daily_audit.settings, "PARQUET_DIR", parquet_dir, raising=False)
    monkeypatch.setattr(daily_audit.settings, "PAPER_DIR", paper_dir, raising=False)
    pd.DataFrame({"decision_date": ["2026-09-14"], "symbol": ["005930"]}).to_parquet(parquet_dir / "topk_decisions.parquet")
    pd.DataFrame(
        {"order_id": ["2026-09-14:005930:entry"], "symbol": ["005930"], "side": ["buy"], "qty": [10],
         "fill_price": [70_000], "decision_date": ["2026-09-14"]}
    ).to_parquet(paper_dir / "fills.parquet")
    price_history = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-09-11"])}).to_parquet(price_history)
    bars = tmp_path / "bars.parquet"
    bars.write_text("x")
    monkeypatch.setattr(daily_audit.settings, "PRICE_HISTORY_PARQUET_PATH", price_history, raising=False)
    monkeypatch.setattr(
        daily_audit,
        "fetch_archive_snapshot",
        lambda snapshot_date=None, **kw: pd.DataFrame({"종목코드": ["005930"], daily_audit.CLOSE_CONFIRMED_COL: [True]}),
    )
    monkeypatch.setattr(daily_audit, "resolve_previous_archive_date", lambda _d: "2026-09-11")
    monkeypatch.setattr(daily_audit, "intraday_partition_path", lambda *_a: bars)
    monkeypatch.setattr(daily_audit, "load_run_outcomes", lambda _d: {})

    # When
    result = daily_audit.audit_daily_completeness("2026-09-14")

    # Then
    assert set(result) == set(daily_audit.AUDIT_STEPS)
    assert all(result.values()), result


def test_audit_daily_completeness_accepts_explicit_no_decision_record(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.tools import daily_audit

    parquet_dir = tmp_path / "parquet"
    paper_dir = tmp_path / "paper"
    parquet_dir.mkdir()
    paper_dir.mkdir()
    monkeypatch.setattr(daily_audit.settings, "PARQUET_DIR", parquet_dir, raising=False)
    monkeypatch.setattr(daily_audit.settings, "PAPER_DIR", paper_dir, raising=False)
    pd.DataFrame({"decision_date": ["2026-09-14"], "symbol": [""], "reason": ["admitted_below_top_k"]}).to_parquet(
        paper_dir / "decisions.parquet"
    )
    monkeypatch.setattr(daily_audit, "fetch_archive_snapshot", lambda snapshot_date=None, **kw: pd.DataFrame())
    monkeypatch.setattr(daily_audit, "resolve_previous_archive_date", lambda _d: None)
    monkeypatch.setattr(daily_audit, "intraday_partition_path", lambda *_a: tmp_path / "missing.parquet")
    monkeypatch.setattr(daily_audit, "load_run_outcomes", lambda _d: {"predict": "OK"})

    # When
    result = daily_audit.audit_daily_completeness("2026-09-14")

    # Then
    assert result["decision"] is True
    assert result["paper_entry"] is True
    assert result["archive"] is False
    assert result["close_confirmed"] is False
    assert result["minute_bars"] is False
    # 직전 아카이브 영업일이 없으면 기대치가 없어 신선도는 판정 보류(True)
    assert result["price_history_fresh"] is True


def test_audit_daily_completeness_flags_stale_or_missing_price_history(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.tools import daily_audit

    parquet_dir = tmp_path / "parquet"
    paper_dir = tmp_path / "paper"
    parquet_dir.mkdir()
    paper_dir.mkdir()
    monkeypatch.setattr(daily_audit.settings, "PARQUET_DIR", parquet_dir, raising=False)
    monkeypatch.setattr(daily_audit.settings, "PAPER_DIR", paper_dir, raising=False)
    price_history = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-09-09"])}).to_parquet(price_history)
    monkeypatch.setattr(daily_audit.settings, "PRICE_HISTORY_PARQUET_PATH", price_history, raising=False)
    monkeypatch.setattr(daily_audit, "fetch_archive_snapshot", lambda snapshot_date=None, **kw: pd.DataFrame())
    monkeypatch.setattr(daily_audit, "resolve_previous_archive_date", lambda _d: "2026-09-11")
    monkeypatch.setattr(daily_audit, "intraday_partition_path", lambda *_a: tmp_path / "missing.parquet")
    monkeypatch.setattr(daily_audit, "load_run_outcomes", lambda _d: {})

    # When: 적재가 2영업일 밀림
    stale = daily_audit.audit_daily_completeness("2026-09-14")

    # Then
    assert stale["price_history_fresh"] is False
    assert stale["decision"] is False
    assert stale["paper_entry"] is False

    # Given: price_history 파일 자체가 없음
    monkeypatch.setattr(daily_audit.settings, "PRICE_HISTORY_PARQUET_PATH", tmp_path / "absent.parquet", raising=False)

    # Then
    assert daily_audit.audit_daily_completeness("2026-09-14")["price_history_fresh"] is False


def test_audit_daily_completeness_tolerates_empty_or_schema_drifted_ledgers(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.tools import daily_audit

    parquet_dir = tmp_path / "parquet"
    paper_dir = tmp_path / "paper"
    parquet_dir.mkdir()
    paper_dir.mkdir()
    monkeypatch.setattr(daily_audit.settings, "PARQUET_DIR", parquet_dir, raising=False)
    monkeypatch.setattr(daily_audit.settings, "PAPER_DIR", paper_dir, raising=False)
    # Given: 컬럼이 빠진 결정 로그, side 없는 체결 원장, 행이 없는 price_history
    pd.DataFrame({"symbol": ["005930"]}).to_parquet(parquet_dir / "topk_decisions.parquet")
    pd.DataFrame({"order_id": ["x"], "decision_date": ["2026-09-14"]}).to_parquet(paper_dir / "fills.parquet")
    price_history = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]")}).to_parquet(price_history)
    monkeypatch.setattr(daily_audit.settings, "PRICE_HISTORY_PARQUET_PATH", price_history, raising=False)
    monkeypatch.setattr(daily_audit, "fetch_archive_snapshot", lambda snapshot_date=None, **kw: pd.DataFrame())
    monkeypatch.setattr(daily_audit, "resolve_previous_archive_date", lambda _d: "2026-09-11")
    monkeypatch.setattr(daily_audit, "intraday_partition_path", lambda *_a: tmp_path / "missing.parquet")
    monkeypatch.setattr(daily_audit, "load_run_outcomes", lambda _d: {})

    # When
    result = daily_audit.audit_daily_completeness("2026-09-14")

    # Then: 예외 없이 전부 미수행으로 판정
    assert result["decision"] is False
    assert result["paper_entry"] is False
    assert result["price_history_fresh"] is False


def test_classify_day_weekend_holiday_trading_and_lookup_failure() -> None:
    from src.tools import daily_audit

    calls = {"n": 0}

    def _never(_date: str) -> bool:
        calls["n"] += 1
        return True

    # Given/When/Then: 주말은 오라클을 호출하지 않는다
    assert daily_audit.classify_day("2026-09-12", _never) == daily_audit.DAY_WEEKEND
    assert calls["n"] == 0

    # And: 평일 휴장일(추석)과 거래일
    assert daily_audit.classify_day("2026-09-24", lambda _d: False) == daily_audit.DAY_HOLIDAY
    assert daily_audit.classify_day("2026-09-14", lambda _d: True) == daily_audit.DAY_TRADING

    # And: 오라클 장애는 휴장일로 단정하지 않는다
    def _boom(_date: str) -> bool:
        raise RuntimeError("KIS trading-day oracle failed")

    assert daily_audit.classify_day("2026-09-14", _boom) == daily_audit.DAY_UNKNOWN


def test_list_failed_kca_units_parses_plain_output_and_surfaces_unavailable() -> None:
    import subprocess

    from src.tools import daily_audit

    seen: dict = {}

    def _fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout=(
                "kca-price-ingest.service loaded failed failed KCA price_history ingest\n"
                "kca-backup.service loaded failed failed KCA offsite backup\n"
            ),
            stderr="",
        )

    # When
    units = daily_audit.list_failed_kca_units(_fake_run)

    # Then
    assert units == ["kca-backup.service", "kca-price-ingest.service"]
    assert "--failed" in seen["cmd"] and "kca-*" in seen["cmd"]

    # Given: systemctl 실행 불가
    def _missing(cmd, **kwargs):
        raise FileNotFoundError("systemctl")

    # Then: 빈 목록으로 숨기지 않는다
    assert daily_audit.list_failed_kca_units(_missing) == ["<systemctl unavailable: FileNotFoundError>"]


def test_list_stale_kis_data_tokens_reports_missing_and_stale_slots(tmp_path) -> None:
    import json

    from src.api.kis.key_pool import token_cache_path
    from src.tools import daily_audit

    env = {
        "KIS_DATA_SLOTS": "1,2,3",
        "KIS_HOST_DATA_SLOTS": "1,2,3",
        "KIS_DATA_1_APP_KEY": "key1",
        "KIS_DATA_1_APP_SECRET": "sec1",
        "KIS_DATA_2_APP_KEY": "key2",
        "KIS_DATA_2_APP_SECRET": "sec2",
        "KIS_DATA_3_APP_KEY": "key3",
        "KIS_DATA_3_APP_SECRET": "sec3",
    }

    # Given: 슬롯1은 오늘자 토큰, 슬롯2는 어제자(구식), 슬롯3은 파일 자체 없음
    token_cache_path("key1", tmp_path).write_text(
        json.dumps({"issued_at": "2026-09-16T07:05:01+09:00"}), encoding="utf-8"
    )
    token_cache_path("key2", tmp_path).write_text(
        json.dumps({"issued_at": "2026-09-15T07:05:01+09:00"}), encoding="utf-8"
    )

    # When
    stale = daily_audit.list_stale_kis_data_tokens("2026-09-16", env=env, cache_dir=tmp_path)

    # Then
    assert stale == ["DATA_2", "DATA_3"]

    # And: 호스트 슬롯 설정 자체가 어긋나면 숨기지 않고 표식을 반환한다
    broken_env = {
        "KIS_DATA_SLOTS": "1,2",
        "KIS_HOST_DATA_SLOTS": "1,2",
        "KIS_DATA_1_APP_KEY": "key1",
        "KIS_DATA_1_APP_SECRET": "sec1",
    }
    result = daily_audit.list_stale_kis_data_tokens("2026-09-16", env=broken_env, cache_dir=tmp_path)
    assert len(result) == 1 and result[0].startswith("<kis host slot config invalid")


def test_build_digest_ok_warning_and_holiday_subjects() -> None:
    import pytest

    from src.tools import daily_audit

    all_ok = dict.fromkeys(daily_audit.AUDIT_STEPS, True)

    # When/Then: 정상
    subject, body = daily_audit.build_digest("2026-09-14", daily_audit.DAY_TRADING, all_ok, [], [])
    assert subject == "[KCA] 2026-09-14 일일점검 OK"
    assert "failed_units=none" in body
    assert "stale_kis_tokens=none" in body

    # And: 누락 + 실패유닛 + KIS 토큰 누락
    partial = dict(all_ok, paper_entry=False)
    subject, body = daily_audit.build_digest(
        "2026-09-14", daily_audit.DAY_TRADING, partial, ["kca-backup.service"], ["DATA_3"]
    )
    assert "경고" in subject and "paper_entry" in subject and "kca-backup.service" in subject and "DATA_3" in subject
    assert "paper_entry=MISSING" in body
    assert "stale_kis_tokens=DATA_3" in body

    # And: 휴장일
    subject, _ = daily_audit.build_digest("2026-09-24", daily_audit.DAY_HOLIDAY, None, [], [])
    assert subject == "[KCA] 2026-09-24 휴장일 SKIP"

    # And: 거래일인데 감사 결과가 없으면 거부
    with pytest.raises(ValueError, match="audit result required"):
        daily_audit.build_digest("2026-09-14", daily_audit.DAY_TRADING, None, [], [])


def test_run_daily_audit_sends_exactly_one_digest_per_weekday(monkeypatch) -> None:
    from src.tools import daily_audit

    audited: list[str] = []
    monkeypatch.setattr(
        daily_audit,
        "audit_daily_completeness",
        lambda d: audited.append(d) or dict.fromkeys(daily_audit.AUDIT_STEPS, True),
    )
    sent: list[tuple[str, str]] = []

    def _dispatch(subject: str, body: str) -> dict[str, bool]:
        sent.append((subject, body))
        return {"webhook": False, "email": True}

    # When/Then: 주말은 아무것도 보내지 않는다
    assert daily_audit.run_daily_audit(
        "2026-09-13",
        trading_day_fn=lambda _d: True,
        failed_units_fn=list,
        stale_tokens_fn=lambda _d: [],
        dispatch_fn=_dispatch,
    ) is None
    assert sent == [] and audited == []

    # And: 평일 휴장일은 감사 없이 휴장일 요약 1통
    subject = daily_audit.run_daily_audit(
        "2026-09-24",
        trading_day_fn=lambda _d: False,
        failed_units_fn=list,
        stale_tokens_fn=lambda _d: [],
        dispatch_fn=_dispatch,
    )
    assert subject == "[KCA] 2026-09-24 휴장일 SKIP"
    assert audited == [] and len(sent) == 1

    # And: 거래일은 감사 후 요약 1통
    subject = daily_audit.run_daily_audit(
        "2026-09-14",
        trading_day_fn=lambda _d: True,
        failed_units_fn=lambda: [],
        stale_tokens_fn=lambda _d: [],
        dispatch_fn=_dispatch,
    )
    assert subject == "[KCA] 2026-09-14 일일점검 OK"
    assert audited == ["2026-09-14"] and len(sent) == 2


def test_audit_decision_requires_topk_or_predict_ok_outcome(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.tools import daily_audit

    parquet_dir = tmp_path / "parquet"
    paper_dir = tmp_path / "paper"
    parquet_dir.mkdir()
    paper_dir.mkdir()
    monkeypatch.setattr(daily_audit.settings, "PARQUET_DIR", parquet_dir, raising=False)
    monkeypatch.setattr(daily_audit.settings, "PAPER_DIR", paper_dir, raising=False)
    pd.DataFrame({"decision_date": ["2026-09-14"], "symbol": [""], "reason": ["no_persisted_decision"]}).to_parquet(
        paper_dir / "decisions.parquet"
    )
    monkeypatch.setattr(daily_audit, "fetch_archive_snapshot", lambda snapshot_date=None, **kw: pd.DataFrame())
    monkeypatch.setattr(daily_audit, "resolve_previous_archive_date", lambda _d: None)
    monkeypatch.setattr(daily_audit, "intraday_partition_path", lambda *_a: tmp_path / "missing.parquet")
    asked: list[str] = []

    def _outcomes(run_date):
        asked.append(run_date)
        return {"predict": "NO_DECISION"}

    monkeypatch.setattr(daily_audit, "load_run_outcomes", _outcomes)

    # When: 시스템성 무결정 -> 페이퍼 무결정 기록만으로 OK 처리 금지
    degraded = daily_audit.audit_daily_completeness("2026-09-14")
    monkeypatch.setattr(daily_audit, "load_run_outcomes", lambda _d: {"predict": "OK"})
    normal = daily_audit.audit_daily_completeness("2026-09-14")

    # Then
    assert asked == ["2026-09-14"]
    assert degraded["decision"] is False
    assert degraded["paper_entry"] is True
    assert normal["decision"] is True


def test_audit_daily_completeness_flags_stale_open_position(monkeypatch, tmp_path) -> None:
    import pandas as pd

    from src.tools import daily_audit

    parquet_dir = tmp_path / "parquet"
    paper_dir = tmp_path / "paper"
    parquet_dir.mkdir()
    paper_dir.mkdir()
    monkeypatch.setattr(daily_audit.settings, "PARQUET_DIR", parquet_dir, raising=False)
    monkeypatch.setattr(daily_audit.settings, "PAPER_DIR", paper_dir, raising=False)
    pd.DataFrame({"decision_date": ["2026-09-14"], "symbol": ["005930"]}).to_parquet(parquet_dir / "topk_decisions.parquet")
    price_history = tmp_path / "price_history.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2026-09-11"])}).to_parquet(price_history)
    bars = tmp_path / "bars.parquet"
    bars.write_text("x")
    monkeypatch.setattr(daily_audit.settings, "PRICE_HISTORY_PARQUET_PATH", price_history, raising=False)
    monkeypatch.setattr(
        daily_audit,
        "fetch_archive_snapshot",
        lambda snapshot_date=None, **kw: pd.DataFrame({"종목코드": ["005930"], daily_audit.CLOSE_CONFIRMED_COL: [True]}),
    )
    monkeypatch.setattr(daily_audit, "resolve_previous_archive_date", lambda _d: "2026-09-11")
    monkeypatch.setattr(daily_audit, "intraday_partition_path", lambda *_a: bars)
    monkeypatch.setattr(daily_audit, "load_run_outcomes", lambda _d: {})

    def _write_fills(rows: list[dict]) -> None:
        pd.DataFrame(rows).to_parquet(paper_dir / "fills.parquet")

    today_buy = {"order_id": "2026-09-14:005930:entry", "symbol": "005930", "side": "buy", "qty": 10,
                 "fill_price": 70_000, "decision_date": "2026-09-14", "entry_order_id": None}
    old_buy = {"order_id": "2026-09-11:000660:entry", "symbol": "000660", "side": "buy", "qty": 5,
               "fill_price": 200_000, "decision_date": "2026-09-11", "entry_order_id": None}
    old_sell = {"order_id": "2026-09-11:000660:entry:exit:2026-09-14", "symbol": "000660", "side": "sell", "qty": 5,
                "fill_price": 210_000, "decision_date": "2026-09-14", "entry_order_id": "2026-09-11:000660:entry"}

    # When: 직전 결정일 로트가 오늘 청산되지 않고 남아 있다
    _write_fills([today_buy, old_buy])
    stale = daily_audit.audit_daily_completeness("2026-09-14")

    # Then
    assert "paper_exit" in daily_audit.AUDIT_STEPS
    assert stale["paper_exit"] is False
    assert stale["paper_entry"] is True

    # When: 같은 로트가 청산되었다
    _write_fills([today_buy, old_buy, old_sell])
    healthy = daily_audit.audit_daily_completeness("2026-09-14")

    # Then: 당일 진입 로트만 남아 있으면 정상
    assert healthy["paper_exit"] is True
