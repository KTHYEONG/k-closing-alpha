from __future__ import annotations


def test_unknown_key_is_blocking() -> None:
    from src.tools.issue_registry import IssueTier, classify_issue_tier

    assert classify_issue_tier("collection:charts:1:some_new_reason") is IssueTier.DATA_INTEGRITY
    assert classify_issue_tier("brand_new_key") is IssueTier.DATA_INTEGRITY


def test_only_tick_gaps_are_research_degraded() -> None:
    from src.tools.issue_registry import IssueTier, classify_issue_tier

    assert classify_issue_tier("intraday:regular_ticks:2:volume_gap") is IssueTier.RESEARCH_DEGRADED
    assert classify_issue_tier("intraday:regular_ticks:2:certified_gap") is IssueTier.RESEARCH_DEGRADED
    assert classify_issue_tier("intraday:krx_aftermarket_ticks:2:volume_mismatch") is IssueTier.RESEARCH_DEGRADED
    assert classify_issue_tier("intraday:nxt_aftermarket_ticks:1:volume_mismatch") is IssueTier.RESEARCH_DEGRADED
    assert classify_issue_tier("intraday:krx_aftermarket_ticks:1:missing_partition") is IssueTier.DATA_INTEGRITY
    assert classify_issue_tier("intraday:regular_ticks:2:missing_partition") is IssueTier.DATA_INTEGRITY
    assert classify_issue_tier("intraday:regular:3:missing_stamps") is IssueTier.DATA_INTEGRITY


def test_class_strips_count() -> None:
    from src.tools.issue_registry import issue_class

    assert issue_class("intraday:regular_ticks:2:volume_gap") == issue_class("intraday:regular_ticks:9:volume_gap")


def test_streak_extends_on_consecutive_audits() -> None:
    from src.tools.issue_registry import issue_class, update_advisory_streaks

    cls = issue_class("intraday:regular_ticks:2:volume_gap")
    m1, e1 = update_advisory_streaks({}, [cls], audit_date="2026-10-01", previous_audit_date=None)
    assert e1 == ()
    m2, e2 = update_advisory_streaks(m1, [cls], audit_date="2026-10-02", previous_audit_date="2026-10-01")
    assert e2 == ()
    m3, e3 = update_advisory_streaks(m2, [cls], audit_date="2026-10-03", previous_audit_date="2026-10-02")
    assert m3[cls].consecutive_audits == 3
    assert e3 == (cls,)


def test_streak_resets_across_gap() -> None:
    from src.tools.issue_registry import issue_class, update_advisory_streaks

    cls = issue_class("intraday:regular_ticks:2:volume_gap")
    m1, _ = update_advisory_streaks({}, [cls], audit_date="2026-10-01", previous_audit_date=None)
    m2, _ = update_advisory_streaks(m1, [], audit_date="2026-10-02", previous_audit_date="2026-10-01")
    assert m2 == {}
    m3, e3 = update_advisory_streaks(m2, [cls], audit_date="2026-10-03", previous_audit_date="2026-10-02")
    assert m3[cls].consecutive_audits == 1
    assert e3 == ()


def test_idempotent_same_date_update() -> None:
    from src.tools.issue_registry import issue_class, update_advisory_streaks

    cls = issue_class("intraday:regular_ticks:2:volume_gap")
    m1, _ = update_advisory_streaks({}, [cls], audit_date="2026-10-01", previous_audit_date=None)
    m2, e2 = update_advisory_streaks(m1, [cls], audit_date="2026-10-01", previous_audit_date=None)
    assert m2 == m1
    assert e2 == ()


def test_weekend_adjacency() -> None:
    from src.tools.issue_registry import issue_class, update_advisory_streaks

    cls = issue_class("intraday:regular_ticks:2:volume_gap")
    m1, _ = update_advisory_streaks({}, [cls], audit_date="2026-10-03", previous_audit_date=None)
    m2, _ = update_advisory_streaks(m1, [cls], audit_date="2026-10-06", previous_audit_date="2026-10-03")
    assert m2[cls].consecutive_audits == 2


def test_corrupt_streak_file(tmp_path) -> None:
    from src.tools.issue_registry import load_advisory_streaks

    bad = tmp_path / "streaks.json"
    bad.write_text("{garbage", encoding="utf-8")
    assert load_advisory_streaks(bad) == {}
    assert load_advisory_streaks(tmp_path / "absent.json") == {}


def test_trading_chain_tiers() -> None:
    from src.tools.issue_registry import IssueTier, classify_issue_tier

    assert classify_issue_tier("missing:archive") is IssueTier.TRADING_CHAIN
    assert classify_issue_tier("missing:decision") is IssueTier.TRADING_CHAIN
    assert classify_issue_tier("missing:intraday_complete") is IssueTier.DATA_INTEGRITY
    assert classify_issue_tier("failed_unit:kca-predict.service") is IssueTier.TRADING_CHAIN
    assert classify_issue_tier("failed_unit:collect") is IssueTier.TRADING_CHAIN
    assert classify_issue_tier("failed_unit:kca-backup.service") is IssueTier.DATA_INTEGRITY


def test_streak_persistence_round_trip(tmp_path) -> None:
    from src.tools.issue_registry import AdvisoryStreak, load_advisory_streaks, write_advisory_streaks

    path = tmp_path / "streaks.json"
    streaks = {"intraday:regular_ticks:volume_gap": AdvisoryStreak("2026-10-01", "2026-10-03", 3)}
    write_advisory_streaks(streaks, path)
    assert load_advisory_streaks(path) == streaks
    assert load_advisory_streaks(tmp_path / "empty.json") == {}
    payload = path.read_text(encoding="utf-8")
    assert "005930" not in payload and "symbols" not in payload


def test_invalid_streak_schema_is_empty(tmp_path) -> None:
    import json

    from src.tools.issue_registry import load_advisory_streaks

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert load_advisory_streaks(bad) == {}
    bad.write_text(json.dumps({"streaks": {"c": {"first_date": "d"}}}), encoding="utf-8")
    assert load_advisory_streaks(bad) == {}
    bad.write_text(
        json.dumps({"c": {"first_date": "2026-10-01", "last_date": "2026-10-01", "consecutive_audits": 0}}),
        encoding="utf-8",
    )
    assert load_advisory_streaks(bad) == {}
    bad.write_text(json.dumps({"schema_version": 1, "streaks": "nope"}), encoding="utf-8")
    assert load_advisory_streaks(bad) == {}
    bad.write_text(json.dumps({"c": 5}), encoding="utf-8")
    assert load_advisory_streaks(bad) == {}
