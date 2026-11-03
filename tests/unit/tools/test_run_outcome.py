from __future__ import annotations

"""Run outcome event log scenarios."""


def _pager(tmp_path, sent):
    from src.tools.incident_pager import IncidentPager

    def _send(subject, body):
        sent.append((subject, body))
        return {"webhook": False, "email": True}

    return IncidentPager(ledger_path=tmp_path / "incidents.json", send_fn=_send)


def test_run_events_path_partitions_by_month(monkeypatch, tmp_path) -> None:
    from src.tools import run_outcome

    monkeypatch.setattr(run_outcome.settings, "DATA_DIR", tmp_path, raising=False)

    # When
    path = run_outcome.run_events_path("2026-09-14")

    # Then
    assert path == tmp_path / "logs" / "events" / "2026-09" / "2026-09-14.jsonl"


def test_record_run_outcome_appends_jsonl_without_alert_for_ok(tmp_path) -> None:
    import json

    from src.tools import run_outcome

    path = tmp_path / "events" / "2026-09-14.jsonl"
    sent: list[tuple[str, str]] = []

    # When
    record = run_outcome.record_run_outcome(
        "predict",
        run_outcome.RUN_OUTCOME_OK,
        run_date="2026-09-14",
        reason="",
        metrics={"n_picks": 3, "coverage": 0.987654321},
        path=path,
        pager=_pager(tmp_path, sent),
    )

    # Then
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["job"] == "predict"
    assert stored["outcome"] == "OK"
    assert stored["run_date"] == "2026-09-14"
    assert stored["metrics"] == {"n_picks": 3, "coverage": 0.987654321}
    assert record["outcome"] == "OK"
    assert sent == []


def test_record_run_outcome_pages_only_critical_outcomes_once_per_day(tmp_path) -> None:
    from src.tools import run_outcome

    path = tmp_path / "2026-09-14.jsonl"
    sent: list[tuple[str, str]] = []
    pager = _pager(tmp_path, sent)

    # When
    run_outcome.record_run_outcome(
        "finalize_close", run_outcome.RUN_OUTCOME_DEGRADED, run_date="2026-09-14",
        reason="picks_unconfirmed", metrics={"unconfirmed_picks": ["005930"]}, path=path, pager=pager,
    )
    run_outcome.record_run_outcome(
        "predict", run_outcome.RUN_OUTCOME_NO_DECISION, run_date="2026-09-14",
        reason="ValueError: stale price_history", path=path, pager=pager,
    )
    run_outcome.record_run_outcome(
        "predict", run_outcome.RUN_OUTCOME_NO_DECISION, run_date="2026-09-14", reason="again", path=path, pager=pager,
    )
    run_outcome.record_run_outcome(
        "altdata_capture", run_outcome.RUN_OUTCOME_DEGRADED, run_date="2026-09-14", reason="partial", path=path, pager=pager,
    )
    run_outcome.record_run_outcome(
        "price_ingest", run_outcome.RUN_OUTCOME_DEGRADED, run_date="2026-09-14",
        reason="flow_coverage_below_min", path=path, pager=pager,
    )

    # Then
    assert [s for s, _ in sent] == ["[긴급][kca] 종가 확정 실패", "[긴급][kca] 오늘 매수 결정 생성 실패"]
    for _, body in sent:
        assert len(body.splitlines()) == 3
        assert str(path) not in body and "005930" not in body
    assert "005930" in path.read_text(encoding="utf-8")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 5


def test_record_run_outcome_ok_resolves_paged_job_with_one_line_mail(tmp_path) -> None:
    from src.tools import run_outcome

    path = tmp_path / "2026-09-14.jsonl"
    sent: list[tuple[str, str]] = []
    pager = _pager(tmp_path, sent)

    run_outcome.record_run_outcome(
        "price_ingest", run_outcome.RUN_OUTCOME_DEGRADED, run_date="2026-09-14", reason="stale_panel", path=path, pager=pager,
    )
    run_outcome.record_run_outcome("price_ingest", run_outcome.RUN_OUTCOME_OK, run_date="2026-09-14", path=path, pager=pager)
    run_outcome.record_run_outcome("price_ingest", run_outcome.RUN_OUTCOME_OK, run_date="2026-09-14", path=path, pager=pager)

    assert [s for s, _ in sent] == ["[긴급][kca] 가격 이력 미갱신(오늘 결정 입력 누락)", "[해결됨][kca] 가격 이력 미갱신(오늘 결정 입력 누락)"]
    assert len(sent[1][1].splitlines()) == 1


def test_record_run_outcome_routing_failure_never_raises(tmp_path, caplog) -> None:
    import logging

    from src.tools import run_outcome

    class _Broken:
        def page(self, *args, **kwargs):
            raise RuntimeError("ledger exploded")

    with caplog.at_level(logging.WARNING, logger=run_outcome.logger.name):
        record = run_outcome.record_run_outcome(
            "predict", run_outcome.RUN_OUTCOME_NO_DECISION, run_date="2026-09-14", reason="x",
            path=tmp_path / "e.jsonl", pager=_Broken(),  # type: ignore[arg-type]
        )

    assert record["outcome"] == "NO_DECISION"
    assert any("ROUTE_FAILED" in rec.getMessage() for rec in caplog.records)


def test_record_run_outcome_rejects_unknown_outcome(tmp_path) -> None:
    import pytest

    from src.tools import run_outcome

    path = tmp_path / "2026-09-14.jsonl"

    # When/Then
    with pytest.raises(ValueError, match="unknown run outcome"):
        run_outcome.record_run_outcome("predict", "SUCCESS", run_date="2026-09-14", path=path)
    assert not path.exists()


def test_record_run_outcome_survives_event_write_failure(tmp_path, caplog) -> None:
    import logging

    from src.tools import run_outcome

    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    path = blocker / "2026-09-14.jsonl"
    sent: list[str] = []

    # When
    with caplog.at_level(logging.WARNING, logger=run_outcome.logger.name):
        record = run_outcome.record_run_outcome(
            "finalize_close", run_outcome.RUN_OUTCOME_DEGRADED, run_date="2026-09-14",
            reason="zero_confirmed", path=path, pager=_pager(tmp_path, sent),
        )

    # Then
    assert record["outcome"] == "DEGRADED"
    assert [subject for subject, _ in sent] == ["[긴급][kca] 종가 확정 실패"]
    assert any("EVENT_WRITE_FAILED" in rec.getMessage() for rec in caplog.records)


def test_load_run_outcomes_last_wins_and_skips_corrupt_lines(tmp_path) -> None:
    import json

    from src.tools import run_outcome

    path = tmp_path / "2026-09-14.jsonl"
    rows = [
        {"job": "predict", "run_date": "2026-09-14", "outcome": "NO_DECISION"},
        {"job": "finalize_close", "run_date": "2026-09-14", "outcome": "DEGRADED"},
        {"job": "predict", "run_date": "2026-09-13", "outcome": "OK"},
        {"job": "predict", "run_date": "2026-09-14", "outcome": "OK"},
    ]
    text = "\n".join(json.dumps(r) for r in rows[:2]) + "\n{truncated\n" + "\n".join(json.dumps(r) for r in rows[2:]) + "\n"
    path.write_text(text, encoding="utf-8")

    # When
    outcomes = run_outcome.load_run_outcomes("2026-09-14", path=path)

    # Then
    assert outcomes == {"predict": "OK", "finalize_close": "DEGRADED"}
    assert run_outcome.load_run_outcomes("2026-09-14", path=tmp_path / "missing.jsonl") == {}




def test_record_run_outcome_masks_reason_before_truncation(tmp_path, monkeypatch, caplog) -> None:
    import json
    import logging

    from src.tools import alerts, run_outcome

    secret = "S" * 60 + "T" * 60
    monkeypatch.setattr(alerts.settings, "KIS_APP_SECRET", secret, raising=False)
    path = tmp_path / "events.jsonl"
    sent: list[tuple[str, str]] = []
    reason = "x" * (run_outcome.RUN_EVENT_REASON_MAX_CHARS - 20) + secret + " url?crtfc_key=RAWDARTKEY99"

    with caplog.at_level(logging.WARNING, logger=run_outcome.logger.name):
        record = run_outcome.record_run_outcome(
            "predict", run_outcome.RUN_OUTCOME_DEGRADED, run_date="2026-10-02", reason=reason,
            path=path, pager=_pager(tmp_path, sent),
        )

    stored = json.loads(path.read_text(encoding="utf-8").splitlines()[0])["reason"]
    for text in (record["reason"], stored, *(b for _, b in sent), *(rec.getMessage() for rec in caplog.records)):
        assert all(secret[i : i + 8] not in text for i in range(len(secret) - 7))
    assert len(stored) <= run_outcome.RUN_EVENT_REASON_MAX_CHARS


def test_skipped_is_recorded_without_alert_and_loads_back(tmp_path) -> None:
    import json

    from src.tools import run_outcome

    path = tmp_path / "2026-10-05.jsonl"
    sent: list[tuple[str, str]] = []

    record = run_outcome.record_run_outcome(
        "collect",
        run_outcome.RUN_OUTCOME_SKIPPED,
        run_date="2026-10-05",
        reason="non_trading_day",
        metrics={"session": "CLOSED"},
        path=path,
        pager=_pager(tmp_path, sent),
    )

    assert record["outcome"] == "SKIPPED"
    assert sent == []
    stored = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert stored["outcome"] == "SKIPPED"
    assert stored["reason"] == "non_trading_day"
    assert run_outcome.load_run_outcomes("2026-10-05", path=path) == {"collect": "SKIPPED"}


def test_skipped_requires_non_blank_reason(tmp_path) -> None:
    import pytest

    from src.tools import run_outcome

    path = tmp_path / "2026-10-05.jsonl"
    for bad in ("", "   "):
        with pytest.raises(ValueError, match="non-blank reason"):
            run_outcome.record_run_outcome(
                "predict", run_outcome.RUN_OUTCOME_SKIPPED, run_date="2026-10-05", reason=bad, path=path,
            )
    assert not path.exists()
