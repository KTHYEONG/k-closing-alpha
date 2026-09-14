from __future__ import annotations

"""Run outcome event log scenarios."""


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
        alert_fn=lambda subject, body: sent.append((subject, body)) or {"webhook": False, "email": False},
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


def test_record_run_outcome_alerts_on_degraded_and_no_decision(tmp_path) -> None:
    from src.tools import run_outcome

    path = tmp_path / "2026-09-14.jsonl"
    sent: list[tuple[str, str]] = []

    def _alert(subject, body):
        sent.append((subject, body))
        return {"webhook": True, "email": True}

    # When
    run_outcome.record_run_outcome(
        "finalize_close", run_outcome.RUN_OUTCOME_DEGRADED, run_date="2026-09-14",
        reason="picks_unconfirmed", metrics={"unconfirmed_picks": ["005930"]}, path=path, alert_fn=_alert,
    )
    run_outcome.record_run_outcome(
        "predict", run_outcome.RUN_OUTCOME_NO_DECISION, run_date="2026-09-14",
        reason="ValueError: stale price_history", path=path, alert_fn=_alert,
    )

    # Then
    assert [s for s, _ in sent] == [
        "[KCA] finalize_close DEGRADED 2026-09-14",
        "[KCA] predict NO_DECISION 2026-09-14",
    ]
    assert "reason=picks_unconfirmed" in sent[0][1]
    assert "005930" in sent[0][1]
    assert "reason=ValueError: stale price_history" in sent[1][1]
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_record_run_outcome_rejects_unknown_outcome(tmp_path) -> None:
    import pytest

    from src.tools import run_outcome

    path = tmp_path / "2026-09-14.jsonl"

    # When/Then
    with pytest.raises(ValueError, match="unknown run outcome"):
        run_outcome.record_run_outcome("predict", "SUCCESS", run_date="2026-09-14", path=path, alert_fn=lambda s, b: {})
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
            reason="zero_confirmed", path=path, alert_fn=lambda s, b: sent.append(s) or {},
        )

    # Then
    assert record["outcome"] == "DEGRADED"
    assert sent == ["[KCA] finalize_close DEGRADED 2026-09-14"]
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


