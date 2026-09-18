from __future__ import annotations


def _collection_profile(tmp_path, *, raw=True, auction=False, altdata=False):
    from src.config.collection import CollectionSettings

    kwargs: dict = {
        "COLLECTION_ROOT": tmp_path / "capture",
        "COLLECTION_RAW_ENABLED": raw,
        "COLLECTION_AUCTION_ENABLED": auction,
        "COLLECTION_ALTDATA_ENABLED": altdata,
        "COLLECTION_RESEARCH_SLOTS": ("1",) if auction else (),
        "_env_file": None,
    }
    return CollectionSettings(**kwargs)


def _session_clock(day):
    from datetime import date

    from src.data.capture_contracts import SessionClock

    return SessionClock.standard(date.fromisoformat(day))


def _audit_moment(day, clock="20:15:00"):
    from datetime import datetime

    return datetime.fromisoformat(f"{day}T{clock}+09:00")


def _capture_context(day, run_id, dataset, reason, session="regular"):
    from datetime import date

    from src.data.capture_contracts import CaptureContext

    return CaptureContext(
        trading_date=date.fromisoformat(day),
        run_id=run_id,
        dataset=dataset,
        vendor="kis",
        endpoint="test-endpoint",
        symbol=None,
        venue="KRX",
        session=session,
        capture_reason=reason,
        cohort_id=None,
        scheduled_at=None,
    )


def _publish_cohort_decision(store, day, eligible, *, run_id="run-decision", admitted=None):
    from datetime import date, datetime

    import pandas as pd

    from src.data.capture_contracts import build_cohort

    trading_day = date.fromisoformat(day)
    rejected = {} if "999999" in eligible else {"999999": "out_of_band"}
    cohort = build_cohort(
        trading_day, [*eligible, *rejected], list(eligible), rejected, eligibility_rule_version="v1"
    )
    flags = list(admitted) if admitted is not None else [True] * len(eligible)
    stamp = datetime.fromisoformat(f"{day}T15:19:00+09:00")
    frame = pd.DataFrame(
        {
            "symbol": list(eligible),
            "admitted": flags,
            "snapshot_timestamp": [stamp] * len(eligible),
            "feature_available_timestamp": [stamp] * len(eligible),
        }
    )
    store.publish_decision(
        frame,
        cohort=cohort,
        run_id=run_id,
        completed_at=datetime.fromisoformat(f"{day}T15:20:00+09:00"),
        entries=(),
    )
    return cohort


def _publish_chart_manifest(
    store,
    day,
    run_id,
    dataset,
    symbols,
    *,
    status=None,
    reason="exhausted:regular=10",
    first_time="09:00:00",
    last_time="15:30:00",
):
    from datetime import datetime

    from src.data.capture_contracts import CaptureManifest, CaptureStatus, CoverageEntry

    first = datetime.fromisoformat(f"{day}T{first_time}+09:00")
    last = datetime.fromisoformat(f"{day}T{last_time}+09:00")
    entries = [
        CoverageEntry(
            symbol=symbol,
            dataset=dataset,
            venue="KRX",
            session="regular",
            scheduled_at=None,
            status=status or CaptureStatus.COMPLETE,
            rows=10,
            first_event_time=first,
            last_event_time=last,
            reason=reason,
            raw_refs=(),
        )
        for symbol in symbols
    ]
    manifest_status = (
        CaptureStatus.COMPLETE
        if all(e.status == CaptureStatus.COMPLETE for e in entries)
        else CaptureStatus.PARTIAL
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=_capture_context(day, run_id, dataset, f"intraday-{dataset.value.lower()}"),
        cohort=None,
        completed_at=datetime.fromisoformat(f"{day}T19:00:00+09:00"),
        entries=tuple(entries),
        artifacts=(),
        status=manifest_status,
    )
    return store.publish_manifest(manifest)


def _publish_slow_manifest(store, day, run_id, *, status):
    from datetime import datetime

    from src.data.capture_contracts import CaptureDataset, CaptureManifest

    manifest = CaptureManifest(
        schema_version=1,
        context=_capture_context(day, run_id, CaptureDataset.SHORTING, "altdata-backfill"),
        cohort=None,
        completed_at=datetime.fromisoformat(f"{day}T21:40:00+09:00"),
        entries=(),
        artifacts=(),
        status=status,
    )
    return store.publish_manifest(manifest)


def _publish_auction_close(store, day, run_id, symbols, *, clock, interval=60):
    from datetime import datetime

    from src.daily.auction_capture import _close_rounds, _program_rounds

    from src.data.capture_contracts import CaptureDataset, CaptureManifest, CaptureStatus, CoverageEntry

    rounds = [(slot, CaptureDataset.ORDERBOOK) for slot in _close_rounds(clock, interval)]
    rounds.extend((slot, CaptureDataset.PROGRAM) for slot in _program_rounds(clock))
    entries = [
        CoverageEntry(
            symbol=symbol,
            dataset=dataset,
            venue="KRX",
            session="regular",
            scheduled_at=slot,
            status=CaptureStatus.COMPLETE,
            rows=1,
            first_event_time=slot,
            last_event_time=slot,
            reason="auction-close",
            raw_refs=(),
        )
        for symbol in symbols
        for slot, dataset in rounds
    ]
    manifest = CaptureManifest(
        schema_version=1,
        context=_capture_context(day, run_id, CaptureDataset.ORDERBOOK, "auction-close"),
        cohort=None,
        completed_at=datetime.fromisoformat(f"{day}T15:40:00+09:00"),
        entries=tuple(entries),
        artifacts=(),
        status=CaptureStatus.COMPLETE,
    )
    return store.publish_manifest(manifest)


def _publish_auction_open(store, day, run_id, symbols, *, clock, status=None):
    from datetime import datetime, timedelta

    from src.data.capture_contracts import CaptureDataset, CaptureManifest, CaptureStatus, CoverageEntry

    floor = clock.open_at + timedelta(seconds=30)
    entries = [
        CoverageEntry(
            symbol=symbol,
            dataset=CaptureDataset.PRICE,
            venue="KRX",
            session="regular",
            scheduled_at=floor,
            status=status or CaptureStatus.COMPLETE,
            rows=1,
            first_event_time=floor,
            last_event_time=floor,
            reason="auction-open",
            raw_refs=(),
        )
        for symbol in symbols
    ]
    manifest_status = (
        CaptureStatus.COMPLETE
        if all(e.status == CaptureStatus.COMPLETE for e in entries)
        else CaptureStatus.PARTIAL
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=_capture_context(day, run_id, CaptureDataset.PRICE, "auction-open"),
        cohort=None,
        completed_at=datetime.fromisoformat(f"{day}T09:05:00+09:00"),
        entries=tuple(entries),
        artifacts=(),
        status=manifest_status,
    )
    return store.publish_manifest(manifest)


def _audit(store, day, profile, clock, moment):
    from datetime import date

    from src.tools import daily_audit

    return daily_audit.audit_collection_manifests(
        date.fromisoformat(day),
        store=store,
        profile=profile,
        session_clock=clock,
        audit_at=moment,
    )


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


def test_list_stale_kis_tokens_reports_missing_pool_slots_and_declared_keys(tmp_path) -> None:
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
        "KIS_APP_KEY": "primary",
        "KIS_APP_SECRET": "psec",
        "KIS_HTS_ID": "phts",
    }

    # Given: 슬롯1은 오늘자 토큰, 슬롯2는 어제자(구식), 슬롯3과 선언키는 파일 자체 없음
    token_cache_path("key1", tmp_path).write_text(
        json.dumps({"issued_at": "2026-09-16T07:05:01+09:00"}), encoding="utf-8"
    )
    token_cache_path("key2", tmp_path).write_text(
        json.dumps({"issued_at": "2026-09-15T07:05:01+09:00"}), encoding="utf-8"
    )

    # When
    stale = daily_audit.list_stale_kis_tokens("2026-09-16", env=env, cache_dir=tmp_path)

    # Then: 풀 슬롯뿐 아니라 선언된 비풀 키(PRIMARY) 누락도 드러난다
    assert stale == ["DATA_2", "DATA_3", "PRIMARY"]

    # And: 키 선언 자체가 어긋나면 숨기지 않고 표식을 반환한다
    broken_env = {
        "KIS_DATA_SLOTS": "1,2",
        "KIS_HOST_DATA_SLOTS": "1,2",
        "KIS_DATA_1_APP_KEY": "key1",
        "KIS_DATA_1_APP_SECRET": "sec1",
    }
    result = daily_audit.list_stale_kis_tokens("2026-09-16", env=broken_env, cache_dir=tmp_path)
    assert len(result) == 1 and result[0].startswith("<kis host key config invalid")


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

    # And: 평일 휴장일(정상)은 요약 발송 스킵
    subject = daily_audit.run_daily_audit(
        "2026-09-24",
        trading_day_fn=lambda _d: False,
        failed_units_fn=list,
        stale_tokens_fn=lambda _d: [],
        dispatch_fn=_dispatch,
    )
    assert subject == "[KCA] 2026-09-24 휴장일 SKIP"
    assert audited == [] and len(sent) == 0

    # And: 거래일 정상 동작(OK)은 요약 발송 스킵
    subject = daily_audit.run_daily_audit(
        "2026-09-14",
        trading_day_fn=lambda _d: True,
        failed_units_fn=lambda: [],
        stale_tokens_fn=lambda _d: [],
        dispatch_fn=_dispatch,
    )
    assert subject == "[KCA] 2026-09-14 일일점검 OK"
    assert audited == ["2026-09-14"] and len(sent) == 0

    # And: 경고 발생 시에는 요약 발송
    subject = daily_audit.run_daily_audit(
        "2026-09-15",
        trading_day_fn=lambda _d: True,
        failed_units_fn=lambda: ["kca-predict.service"],
        stale_tokens_fn=lambda _d: [],
        dispatch_fn=_dispatch,
    )
    assert "경고" in subject
    assert len(sent) == 1
    assert sent[0][0] == subject



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


def test_audit_collection_uses_whole_candidate_denominator(tmp_path) -> None:
    """454 defines coverage."""
    from src.data.capture_contracts import CaptureDataset
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    eligible = [f"{i:06d}" for i in range(1, 455)]
    _publish_cohort_decision(store, day, eligible, admitted=[i <= 30 for i in range(1, 455)])
    _publish_chart_manifest(store, day, "run-bars", CaptureDataset.MINUTE_BARS, eligible[:30])

    # When
    issues = _audit(store, day, _collection_profile(tmp_path), _session_clock(day), _audit_moment(day))

    # Then: admitted 30이 아니라 eligible 454이 분모다
    assert "collection:charts:424:missing_entries" in issues
    assert "collection:ticks:454:missing_entries" in issues


def test_audit_collection_reports_partial_chart_as_incomplete(tmp_path) -> None:
    """Incompleteness is reported."""
    from src.data.capture_contracts import CaptureDataset, CaptureStatus
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, ["005930"])
    _publish_chart_manifest(
        store, day, "run-bars", CaptureDataset.MINUTE_BARS, ["005930"],
        status=CaptureStatus.PARTIAL, reason="incomplete:capped",
    )

    # When
    issues = _audit(store, day, _collection_profile(tmp_path), _session_clock(day), _audit_moment(day))

    # Then: 파일 존재가 아니라 터미널 상태로 판정한다
    assert "collection:charts:1:incomplete_entries" in issues
    assert not any("terminal_proof_missing" in issue for issue in issues)


def test_audit_collection_accepts_duplicate_tick_events(tmp_path) -> None:
    """No event deduplication assumption."""
    from src.data.capture_contracts import CaptureDataset, CaptureManifest, CaptureStatus
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, ["005930"])
    _publish_chart_manifest(store, day, "run-ticks", CaptureDataset.TRADE_TICKS, ["005930"])
    _publish_chart_manifest(store, day, "run-bars", CaptureDataset.MINUTE_BARS, ["005930"])
    pending = CaptureManifest(
        schema_version=1,
        context=_capture_context(day, "run-pending", CaptureDataset.MINUTE_BARS, "intraday-minute_bars"),
        cohort=None,
        completed_at=_audit_moment(day, "19:00:00"),
        entries=(),
        artifacts=(),
        status=CaptureStatus.PENDING,
    )
    store.publish_manifest(pending)

    # When: 동일 이벤트가 중복 적재돼도(행 수준 중복) 커버리지는 정상이다
    issues = _audit(store, day, _collection_profile(tmp_path), _session_clock(day), _audit_moment(day))

    # Then
    assert not any(issue.startswith("collection:ticks") for issue in issues)
    assert not any(issue.startswith("collection:charts") for issue in issues)


def test_audit_collection_exposes_wrong_session_event(tmp_path) -> None:
    """Venue/session violation is reported."""
    from src.data.capture_contracts import CaptureDataset
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, ["005930", "000660"])
    _publish_chart_manifest(store, day, "run-bars", CaptureDataset.MINUTE_BARS, ["005930", "000660"])
    _publish_chart_manifest(
        store, day, "run-ticks", CaptureDataset.TRADE_TICKS, ["005930"], last_time="20:00:00"
    )
    _publish_chart_manifest(
        store, day, "run-ticks-early", CaptureDataset.TRADE_TICKS, ["000660"], first_time="08:00:00"
    )

    # When
    issues = _audit(store, day, _collection_profile(tmp_path), _session_clock(day), _audit_moment(day))

    # Then: 정규 세션 파일을 벗어난 20:00 이벤트와 이른 08:00 이벤트가 드러난다
    assert "collection:ticks:2:session_violation" in issues


def test_audit_collection_requires_chart_terminal_proof(tmp_path) -> None:
    """Terminal proof is required."""
    from src.data.capture_contracts import CaptureDataset
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, ["005930"])
    _publish_chart_manifest(
        store, day, "run-bars", CaptureDataset.MINUTE_BARS, ["005930"], reason="capped:regular=5"
    )
    _publish_chart_manifest(store, day, "run-ticks", CaptureDataset.TRADE_TICKS, ["005930"])

    # When
    issues = _audit(store, day, _collection_profile(tmp_path), _session_clock(day), _audit_moment(day))

    # Then
    assert "collection:charts:1:terminal_proof_missing" in issues


def test_audit_collection_reports_tampered_decision_evidence(tmp_path) -> None:
    """Integrity failure is reported."""
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, ["005930"])

    # Given: 불변 결정 산출물이 사후 변경됐다
    inputs = list((tmp_path / "capture").rglob("input.parquet"))
    assert len(inputs) == 1
    with open(inputs[0], "ab") as handle:
        handle.write(b"\x00")

    # When
    issues = _audit(store, day, _collection_profile(tmp_path), _session_clock(day), _audit_moment(day))

    # Then
    assert "collection:manifest:1:unreadable_evidence" in issues


def test_audit_collection_keeps_future_slow_data_pending(tmp_path) -> None:
    """Future task is not failed."""
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, ["005930"])

    # When: 20:15 감사는 21:35 슬로우데이터를 실패로 단정하지 않는다
    issues = _audit(
        store, day, _collection_profile(tmp_path, altdata=True), _session_clock(day), _audit_moment(day)
    )

    # Then
    assert not any(issue.startswith("collection:slow_data") for issue in issues)


def test_audit_collection_reports_overdue_slow_data_run(tmp_path) -> None:
    """Missing run is reported."""
    from src.data.capture_store import CaptureStore

    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, "2026-09-17", ["005930"])

    # When: 다음날 감사가 전날 밤 due였던 실행의 부재를 본다
    issues = _audit(
        store,
        "2026-09-17",
        _collection_profile(tmp_path, altdata=True),
        _session_clock("2026-09-17"),
        _audit_moment("2026-09-18"),
    )

    # Then
    assert "collection:slow_data:1:missing_run" in issues


def test_audit_collection_reports_incomplete_slow_data_run(tmp_path) -> None:
    """Partial slow-data acquisition is reported."""
    from src.data.capture_contracts import CaptureStatus
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, ["005930"])
    _publish_slow_manifest(store, day, "run-slow", status=CaptureStatus.PARTIAL)

    # When
    issues = _audit(
        store,
        day,
        _collection_profile(tmp_path, altdata=True),
        _session_clock(day),
        _audit_moment("2026-09-18", "22:00:00"),
    )

    # Then
    assert "collection:slow_data:1:incomplete_run" in issues


def test_audit_collection_marks_disabled_jobs_explicitly(tmp_path) -> None:
    """Disabled is distinct from success."""
    from src.data.capture_contracts import CaptureDataset
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, ["005930"])
    _publish_chart_manifest(store, day, "run-bars", CaptureDataset.MINUTE_BARS, ["005930"])
    _publish_chart_manifest(store, day, "run-ticks", CaptureDataset.TRADE_TICKS, ["005930"])

    # When
    issues = _audit(store, day, _collection_profile(tmp_path), _session_clock(day), _audit_moment(day))

    # Then: 비활성 작업은 성공으로 보이지 않고 명시된다
    assert "collection:auction:0:disabled" in issues
    assert "collection:slow_data:0:disabled" in issues
    assert "COMPLETE" not in " ".join(issues)


def test_audit_collection_rejects_inconsistent_date_and_naive_cutoff(tmp_path) -> None:
    """Inconsistent date or naive audit cutoff."""
    from datetime import date, datetime

    import pytest

    from src.data.capture_store import CaptureStore
    from src.tools import daily_audit

    store = CaptureStore(tmp_path / "capture")
    profile = _collection_profile(tmp_path)
    clock = _session_clock("2026-09-18")

    with pytest.raises(ValueError, match="trading_date"):
        daily_audit.audit_collection_manifests(
            date.fromisoformat("2026-09-17"),
            store=store,
            profile=profile,
            session_clock=clock,
            audit_at=_audit_moment("2026-09-18"),
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        daily_audit.audit_collection_manifests(
            date.fromisoformat("2026-09-18"),
            store=store,
            profile=profile,
            session_clock=clock,
            audit_at=datetime.fromisoformat("2026-09-18T20:15:00"),
        )


def test_audit_collection_reports_legacy_mode_without_provenance(tmp_path) -> None:
    """Raw-disabled legacy mode has explicit provenance-unavailable issues."""
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")

    # When
    issues = _audit(
        store, day, _collection_profile(tmp_path, raw=False), _session_clock(day), _audit_moment(day)
    )

    # Then
    assert issues == ("collection:provenance:0:raw_disabled",)


def test_audit_collection_reports_missing_decision_input(tmp_path) -> None:
    """Partial decision publication without qualifying input is reported."""
    from datetime import date, datetime

    import pandas as pd

    from src.data.capture_contracts import (
        CaptureDataset,
        CaptureStatus,
        CoverageEntry,
        build_cohort,
    )
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    cohort = build_cohort(
        date.fromisoformat(day),
        ["005930", "000660", "999999"],
        ["005930", "000660"],
        {"999999": "out_of_band"},
        eligibility_rule_version="v1",
    )
    failed = CoverageEntry(
        symbol="005930",
        dataset=CaptureDataset.SCAN,
        venue="KRX",
        session="regular",
        scheduled_at=None,
        status=CaptureStatus.FAILED,
        rows=0,
        first_event_time=None,
        last_event_time=None,
        reason="vendor_failure",
        raw_refs=(),
    )
    stamp = datetime.fromisoformat(f"{day}T15:19:00+09:00")
    frame = pd.DataFrame(
        {
            "symbol": ["005930", "000660"],
            "admitted": [True, False],
            "snapshot_timestamp": [stamp, stamp],
            "feature_available_timestamp": [stamp, stamp],
        }
    )
    store.publish_decision(
        frame,
        cohort=cohort,
        run_id="run-broken",
        completed_at=datetime.fromisoformat(f"{day}T15:20:00+09:00"),
        entries=(failed,),
    )

    # When: 결정 매니페스트는 있으나 적격 입력이 복원되지 않는다
    issues = _audit(store, day, _collection_profile(tmp_path), _session_clock(day), _audit_moment(day))

    # Then
    assert "collection:decision:1:missing_decision_input" in issues


def test_audit_collection_reports_decision_without_input_artifact(tmp_path) -> None:
    """Decision manifest without restorable input is reported."""
    from datetime import date

    from src.data.capture_contracts import (
        CaptureDataset,
        CaptureManifest,
        CaptureStatus,
        build_cohort,
    )
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    cohort = build_cohort(
        date.fromisoformat(day),
        ["005930", "999999"],
        ["005930"],
        {"999999": "out_of_band"},
        eligibility_rule_version="v1",
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=_capture_context(day, "run-hollow", CaptureDataset.SCAN, "decision-input"),
        cohort=cohort,
        completed_at=_audit_moment(day, "15:20:00"),
        entries=(),
        artifacts=(),
        status=CaptureStatus.COMPLETE,
    )
    store.publish_manifest(manifest)

    # When
    issues = _audit(store, day, _collection_profile(tmp_path), _session_clock(day), _audit_moment(day))

    # Then
    assert "collection:decision:1:integrity_failure" in issues


def test_audit_collection_omits_secrets_from_issues(tmp_path) -> None:
    """Credentials and raw request headers are absent."""
    from src.data.capture_contracts import CaptureDataset, CaptureStatus
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, ["005930"])
    _publish_chart_manifest(
        store, day, "run-bars", CaptureDataset.MINUTE_BARS, ["005930"],
        status=CaptureStatus.FAILED, reason="auth:appkey=SECRET app_secret=XYZ",
    )
    _publish_chart_manifest(store, day, "run-ticks", CaptureDataset.TRADE_TICKS, ["005930"])

    # When
    issues = _audit(store, day, _collection_profile(tmp_path), _session_clock(day), _audit_moment(day))

    # Then: 실패는 보고하되 자격증명은 노출하지 않는다
    assert "collection:charts:1:incomplete_entries" in issues
    joined = " ".join(issues)
    assert "SECRET" not in joined and "appkey" not in joined and "app_secret" not in joined


def test_audit_collection_reconciles_enabled_auction_sweeps(tmp_path) -> None:
    """Elapsed sweeps may not disappear from coverage."""
    from src.data.capture_contracts import CaptureDataset
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    clock = _session_clock(day)
    profile = _collection_profile(tmp_path, auction=True)

    symbols = ["005930", "000660"]
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, symbols)
    _publish_chart_manifest(store, day, "run-bars", CaptureDataset.MINUTE_BARS, symbols)
    _publish_chart_manifest(store, day, "run-ticks", CaptureDataset.TRADE_TICKS, symbols)
    _publish_auction_close(store, day, "run-close", symbols, clock=clock)

    # When: open 스윕 매니페스트가 통째로 없다
    issues = _audit(store, day, profile, clock, _audit_moment(day))

    # Then
    assert "collection:auction_open:1:missing_manifest" in issues
    assert not any(issue.startswith("collection:auction_close") for issue in issues)


def test_audit_collection_reports_partial_auction_close_coverage(tmp_path) -> None:
    """Missing slots and failed entries stay visible."""
    from src.daily.auction_capture import _close_rounds, _program_rounds
    from src.data.capture_contracts import (
        CaptureDataset,
        CaptureManifest,
        CaptureStatus,
        CoverageEntry,
    )
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    clock = _session_clock(day)
    profile = _collection_profile(tmp_path, auction=True)
    store = CaptureStore(tmp_path / "capture")
    symbols = ["005930", "000660"]
    _publish_cohort_decision(store, day, symbols)
    _publish_chart_manifest(store, day, "run-bars", CaptureDataset.MINUTE_BARS, symbols)
    _publish_chart_manifest(store, day, "run-ticks", CaptureDataset.TRADE_TICKS, symbols)
    rounds = _close_rounds(clock, 60)
    program_rounds = _program_rounds(clock)
    prog_slot = program_rounds[0]
    entries = [
        CoverageEntry(
            symbol="005930",
            dataset=CaptureDataset.ORDERBOOK,
            venue="KRX",
            session="regular",
            scheduled_at=slot,
            status=CaptureStatus.COMPLETE,
            rows=1,
            first_event_time=slot,
            last_event_time=slot,
            reason="auction-close",
            raw_refs=(),
        )
        for slot in rounds
    ]
    entries.append(
        CoverageEntry(
            symbol="005930",
            dataset=CaptureDataset.PROGRAM,
            venue="KRX",
            session="regular",
            scheduled_at=prog_slot,
            status=CaptureStatus.FAILED,
            rows=0,
            first_event_time=prog_slot,
            last_event_time=prog_slot,
            reason="vendor_failure",
            raw_refs=(),
        )
    )
    manifest = CaptureManifest(
        schema_version=1,
        context=_capture_context(day, "run-close-partial", CaptureDataset.ORDERBOOK, "auction-close"),
        cohort=None,
        completed_at=_audit_moment(day, "15:40:00"),
        entries=tuple(entries),
        artifacts=(),
        status=CaptureStatus.PARTIAL,
    )
    store.publish_manifest(manifest)
    _publish_auction_open(store, day, "run-open", symbols, clock=clock)

    # When
    issues = _audit(store, day, profile, clock, _audit_moment(day))

    # Then: 22개 기대 슬롯 중 10개만 있고 실패 1건이 보인다
    assert "collection:auction_close:12:missing_entries" in issues
    assert "collection:auction_close:1:incomplete_entries" in issues


def test_audit_collection_reports_failed_auction_open_entries(tmp_path) -> None:
    """Auction failures stay visible."""
    from src.data.capture_contracts import CaptureDataset, CaptureStatus
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    clock = _session_clock(day)
    profile = _collection_profile(tmp_path, auction=True)
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, day, ["005930"])
    _publish_chart_manifest(store, day, "run-bars", CaptureDataset.MINUTE_BARS, ["005930"])
    _publish_chart_manifest(store, day, "run-ticks", CaptureDataset.TRADE_TICKS, ["005930"])
    _publish_auction_close(store, day, "run-close", ["005930"], clock=clock)
    _publish_auction_open(store, day, "run-open", ["005930"], clock=clock, status=CaptureStatus.FAILED)

    # When
    issues = _audit(store, day, profile, clock, _audit_moment(day))

    # Then
    assert "collection:auction_open:1:incomplete_entries" in issues


def test_audit_collection_passes_clean_when_everything_certified(tmp_path) -> None:
    """Complete acquisition reports no issues."""
    from src.data.capture_contracts import CaptureDataset, CaptureStatus
    from src.data.capture_store import CaptureStore

    day = "2026-09-18"
    clock = _session_clock(day)
    profile = _collection_profile(tmp_path, auction=True, altdata=True)
    store = CaptureStore(tmp_path / "capture")
    symbols = ["005930", "000660"]
    _publish_cohort_decision(store, day, symbols)
    _publish_chart_manifest(store, day, "run-bars", CaptureDataset.MINUTE_BARS, symbols)
    _publish_chart_manifest(store, day, "run-ticks", CaptureDataset.TRADE_TICKS, symbols)
    _publish_auction_close(store, day, "run-close", symbols, clock=clock)
    _publish_auction_open(store, day, "run-open", symbols, clock=clock)
    _publish_slow_manifest(store, day, "run-slow", status=CaptureStatus.COMPLETE)

    # When
    issues = _audit(store, day, profile, clock, _audit_moment(day, "22:00:00"))

    # Then
    assert issues == ()


def test_build_digest_includes_collection_issues_compatibly() -> None:
    """Prior result format remains valid."""
    import pytest

    from src.tools import daily_audit

    all_ok = dict.fromkeys(daily_audit.AUDIT_STEPS, True)

    # Given: 기존 위치 인자 호출
    subject, body = daily_audit.build_digest("2026-09-14", daily_audit.DAY_TRADING, all_ok, [], [])

    # Then: 기존 형식 그대로
    assert subject == "[KCA] 2026-09-14 일일점검 OK"
    assert "collection_issues=none" in body

    # When: 수집 이상이 함께 보고된다
    subject, body = daily_audit.build_digest(
        "2026-09-14",
        daily_audit.DAY_TRADING,
        all_ok,
        [],
        [],
        collection_issues=("collection:charts:2:missing_entries",),
    )

    # Then
    assert "경고" in subject and "collection:charts:2:missing_entries" in subject
    assert "collection_issues=collection:charts:2:missing_entries" in body

    # And: 지원하지 않는 일자 구분은 거부된다
    with pytest.raises(ValueError, match="unsupported day_kind"):
        daily_audit.build_digest("2026-09-14", "lunar", all_ok, [], [])


def test_run_daily_audit_keeps_unknown_calendar_visible(monkeypatch) -> None:
    """UNKNOWN remains visible."""
    from src.tools import daily_audit

    monkeypatch.setattr(
        daily_audit,
        "audit_daily_completeness",
        lambda d: dict.fromkeys(daily_audit.AUDIT_STEPS, True),
    )
    sent: list[tuple[str, str]] = []

    def _dispatch(subject: str, body: str) -> dict[str, bool]:
        sent.append((subject, body))
        return {"webhook": False, "email": True}

    def _boom(_date: str) -> bool:
        raise RuntimeError("KIS trading-day oracle failed")

    # When: 달력 조회가 실패한 평일
    subject = daily_audit.run_daily_audit(
        "2026-09-14",
        trading_day_fn=_boom,
        failed_units_fn=lambda: ["kca-backup.service"],
        stale_tokens_fn=lambda _d: [],
        dispatch_fn=_dispatch,
    )

    # Then: 휴장일 면제로 숨지 않고 UNKNOWN이 그대로 보인다
    assert subject is not None and "휴장일" not in subject
    assert len(sent) == 1
    assert "day=unknown" in sent[0][1]


def test_run_daily_audit_includes_collection_gaps(monkeypatch, tmp_path) -> None:
    """Manifest audit wires into the daily digest."""
    from src.data.capture_contracts import CaptureDataset
    from src.data.capture_store import CaptureStore
    from src.tools import daily_audit

    profile = _collection_profile(tmp_path)
    monkeypatch.setattr(daily_audit, "CollectionSettings", lambda *a, **k: profile)
    store = CaptureStore(tmp_path / "capture")
    _publish_cohort_decision(store, "2026-09-14", ["005930", "000660"])
    _publish_chart_manifest(store, "2026-09-14", "run-bars", CaptureDataset.MINUTE_BARS, ["005930"])
    _publish_chart_manifest(store, "2026-09-14", "run-ticks", CaptureDataset.TRADE_TICKS, ["005930", "000660"])
    monkeypatch.setattr(
        daily_audit,
        "audit_daily_completeness",
        lambda d: dict.fromkeys(daily_audit.AUDIT_STEPS, True),
    )
    sent: list[tuple[str, str]] = []

    def _dispatch(subject: str, body: str) -> dict[str, bool]:
        sent.append((subject, body))
        return {"webhook": False, "email": True}

    # When
    subject = daily_audit.run_daily_audit(
        "2026-09-14",
        trading_day_fn=lambda _d: True,
        failed_units_fn=list,
        stale_tokens_fn=lambda _d: [],
        dispatch_fn=_dispatch,
    )

    # Then
    assert subject is not None and "경고" in subject
    assert "collection:charts:1:missing_entries" in subject
    assert len(sent) == 1
    assert "collection:charts:1:missing_entries" in sent[0][1]


def test_run_daily_audit_survives_collection_prep_failure(monkeypatch) -> None:
    """Collection audit failure degrades to an explicit issue."""
    from src.tools import daily_audit

    def _boom(*args, **kwargs):
        raise ValueError("bad profile")

    monkeypatch.setattr(daily_audit, "CollectionSettings", _boom)
    monkeypatch.setattr(
        daily_audit,
        "audit_daily_completeness",
        lambda d: dict.fromkeys(daily_audit.AUDIT_STEPS, True),
    )
    sent: list[tuple[str, str]] = []

    def _dispatch(subject: str, body: str) -> dict[str, bool]:
        sent.append((subject, body))
        return {"webhook": False, "email": True}

    # When
    subject = daily_audit.run_daily_audit(
        "2026-09-14",
        trading_day_fn=lambda _d: True,
        failed_units_fn=list,
        stale_tokens_fn=lambda _d: [],
        dispatch_fn=_dispatch,
    )

    # Then: 기존 감사는 계속되고 수집 감시는 unavailable으로 명시된다
    assert subject is not None and "collection:audit:1:unavailable" in subject
    assert len(sent) == 1
